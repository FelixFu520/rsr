"""物理求解器验证任务的机器人：Franka FR3（Newton）、Franka Panda（PhysX）和 V7 长夹块 Franka。

- ``franka_newton``：``FrankaNewtonRobot`` 导入 FR3 URDF，由 Planner 的 Newton 运行时驱动。
- ``franka_physx``：``FrankaPhysxRobot`` 加载 Panda USD、PhysX 驱动与 Lula IK，也为 hybrid
  世界提供实测的手指/手掌碰撞位姿。
- ``franka_package_pads``：``PackagePadRobot`` 在 FR3 手指刚体上安装 V7 的长平行夹块。

重型依赖在初始化阶段加载，读取配置与列举后端不会启动 Kit。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, PositiveFloat, field_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.robot.base import BaseRobot
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.cameras import CameraConfig, create_cameras, validate_camera_list

# ================================================================================================
# PhysicSolver 案例的机器人定义，重型依赖在初始化阶段加载。
#
# franka_newton 后端使用 FrankaNewtonRobot：导入 FR3 URDF，由 Planner 的 Newton
# 运行时驱动；franka_physx 后端使用 FrankaPhysxRobot：加载 Panda USD、PhysX 驱动与
# Lula IK，也为 hybrid 世界提供实测的手指/手掌碰撞位姿。两种后端由配置显式选择。
# ================================================================================================


class _RobotCameraConfig(BaseConfig):
    """Named cameras mounted in the robot's local link frames."""

    cameras: list[CameraConfig] = Field(default_factory=list)

    @field_validator("cameras")
    @classmethod
    def _validate_cameras(cls, value: list[CameraConfig]) -> list[CameraConfig]:
        return validate_camera_list(value)


class FrankaNewtonRobotConfig(_RobotCameraConfig):
    """Joint coordinates are arm radians followed by two finger openings in metres."""

    urdf_path: str = "assets/PhysicSolver/Robot/franka_emika_panda/urdf/fr3_franka_hand.urdf"
    prim_path: str = "/World/robots/Franka"
    base_position: tuple[float, float, float] = (-0.38, 0.0, 0.732389)
    base_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    initial_joint_positions: tuple[float, ...] = (
        0.0,
        -0.569,
        0.0,
        -2.81,
        0.0,
        3.037,
        0.741,
        0.04,
        0.04,
    )
    arm_stiffness: float = Field(default=1200.0, gt=0)
    arm_damping: float = Field(default=80.0, gt=0)
    finger_stiffness: float = Field(default=1600.0, gt=0)
    finger_damping: float = Field(default=40.0, gt=0)
    finger_max_force: float = Field(
        default=60.0, gt=0, description="Authored N; Featherstone does not enforce effort limits"
    )
    finger_friction: float = Field(default=3.0, ge=0)
    control_mode: Literal["position", "velocity"] = "position"


class FrankaNewtonRobot(BaseRobot):
    """Scene mounts the base; the planner runtime adds this URDF to its Newton model."""

    config_cls = FrankaNewtonRobotConfig

    def __init__(self, config: FrankaNewtonRobotConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else self.config_cls.from_settings("robot")

    @property
    def pure_newton(self) -> bool:
        """This robot backend uses direct Newton dynamics."""
        return True

    def init_robot(self, sim_world: Any) -> None:
        """Create a visual root without starting or resetting physics."""
        from pxr import UsdGeom

        if not getattr(sim_world, "uses_direct_newton", False):
            raise ValueError(
                "franka_newton requires physics_engine=newton, newton_execution=direct"
            )
        self.sim_world = sim_world
        UsdGeom.Xform.Define(sim_world.stage, self.config.prim_path)

    def arrange(self) -> None:
        """Cache the table-mounted robot base pose."""
        self.poses["franka"] = (
            np.asarray(self.config.base_position),
            np.asarray(self.config.base_orientation),
        )

    def adjust(self) -> None:
        """Keep the explicitly configured base pose."""

    def initialize_cameras(self) -> None:
        """Mount cameras after Newton has created and positioned the visual link frames."""
        self.sensors.update(create_cameras(self.sim_world.stage, self.config.cameras))

    def add_newton_robot(self, builder: Any, *, scale: float = 100.0) -> None:
        """Import the bundled official Newton FR3 URDF with a fixed base."""
        import newton
        import warp as wp

        path = Path(resolve_usd_path(self.config.urdf_path))
        if not path.is_file():
            raise FileNotFoundError(f"Missing official Franka URDF: {path}")
        quat = self.config.base_orientation
        builder.add_urdf(
            str(path),
            xform=wp.transform(
                np.asarray(self.config.base_position) * scale,
                wp.quat(quat[1], quat[2], quat[3], quat[0]),
            ),
            floating=False,
            scale=scale,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        # Newton 1.6.1's URDF importer ignores <limit velocity>. Restore the
        # authored limits, converting only prismatic axes to model centimetres.
        joints = {joint.get("name"): joint for joint in ET.parse(path).getroot().findall("joint")}
        for index, label in enumerate(builder.joint_label):
            joint = joints.get(label.rsplit("/", 1)[-1])
            if joint is None or joint.get("type") not in {"revolute", "prismatic"}:
                continue
            limit = joint.find("limit")
            if limit is not None and limit.get("velocity") is not None:
                velocity = float(limit.get("velocity"))
                if joint.get("type") == "prismatic":
                    velocity *= scale
                builder.joint_velocity_limit[builder.joint_qd_start[index]] = velocity
        q = list(self.config.initial_joint_positions)
        q[-2:] = [value * scale for value in q[-2:]]
        builder.joint_q[:9] = q
        builder.joint_target_q[:9] = q
        if self.config.control_mode == "velocity":
            # Match the official cloth_franka controller: velocities drive Featherstone,
            # with the URDF's native joint settings and no added position servos.
            return
        builder.joint_target_ke[:7] = [self.config.arm_stiffness * scale**2] * 7
        builder.joint_target_kd[:7] = [self.config.arm_damping * scale**2] * 7
        builder.joint_target_ke[7:9] = [self.config.finger_stiffness] * 2
        builder.joint_target_kd[7:9] = [self.config.finger_damping] * 2
        builder.joint_target_mode[:9] = [int(newton.JointTargetMode.POSITION)] * 9
        # URDF import scales length/inertia, while these limits are authored
        # explicitly in the centimetre model's kg·cm/s² and kg·cm²/s² units.
        builder.joint_effort_limit[:7] = [value * scale**2 for value in [87.0] * 4 + [12.0] * 3]
        builder.joint_effort_limit[7:9] = [self.config.finger_max_force * scale] * 2


class FrankaPhysxRobotConfig(_RobotCameraConfig):
    """Configuration shared by the PhysX Panda and Newton FR3."""

    usd_path: str = "assets/PhysicSolver/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    urdf_path: str = "assets/PhysicSolver/Robot/franka_emika_panda/urdf/fr3_franka_hand.urdf"
    prim_path: str = "/World/robots/Franka"
    base_position: tuple[float, float, float] = (-0.38, 0.0, 0.732389)
    base_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    initial_joint_positions: tuple[float, ...] = (
        0.0,
        -0.569,
        0.0,
        -2.810,
        0.0,
        3.037,
        0.741,
        0.04,
        0.04,
    )
    arm_stiffness: float = Field(default=1200.0, gt=0.0)
    arm_damping: float = Field(default=80.0, gt=0.0)
    finger_stiffness: float = Field(default=1600.0, gt=0.0)
    finger_damping: float = Field(default=40.0, gt=0.0)
    finger_max_force: float = Field(
        default=60.0, gt=0.0, description="Force cap in N, enforced by PhysX/MuJoCo."
    )
    finger_friction: float = Field(default=3.0, ge=0.0)


class FrankaPhysxRobot(BaseRobot):
    """Load a Panda for PhysX, or leave Newton to instantiate the official FR3 URDF."""

    config_cls = FrankaPhysxRobotConfig

    def __init__(self, config: FrankaPhysxRobotConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("robot")
        self.articulation: Any = None
        self.kinematics: Any = None
        self._lula: Any = None
        self._finger_links: list[Any] = []
        self._collider_link_indices: list[int] = []
        self._recorded_action = np.asarray(self.config.initial_joint_positions, dtype=float)

    @property
    def pure_newton(self) -> bool:
        """Whether the world delegates all physics to Newton."""
        return False

    def init_robot(self, sim_world: Any) -> None:
        """Reference the official USD only when a PhysX robot is needed."""
        from pxr import UsdGeom

        if sim_world.physics_engine not in {"physx", "hybrid"}:
            raise ValueError("franka_physx requires PhysX or hybrid world")
        self.sim_world = sim_world
        if self.pure_newton:
            UsdGeom.Xform.Define(sim_world.stage, self.config.prim_path)
            return
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(resolve_usd_path(self.config.usd_path), self.config.prim_path)
        self.articulation = SingleArticulation(
            prim_path=self.config.prim_path,
            name="demo_franka",
            position=np.asarray(self.config.base_position),
            orientation=np.asarray(self.config.base_orientation),
        )
        self.sensors.update(create_cameras(sim_world.stage, self.config.cameras))

    def arrange(self) -> None:
        """Record the table-mounted base pose and configure actual finger materials."""
        from pxr import Usd, UsdPhysics, UsdShade

        self.poses["franka"] = (
            np.asarray(self.config.base_position),
            np.asarray(self.config.base_orientation),
        )
        if self.pure_newton:
            return
        stage = self.sim_world.stage
        material = UsdShade.Material.Define(stage, "/World/DemoFingerMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(self.config.finger_friction)
        physics_material.CreateDynamicFrictionAttr(self.config.finger_friction)
        physics_material.CreateRestitutionAttr(0.0)
        for prim in Usd.PrimRange(stage.GetPrimAtPath(self.config.prim_path)):
            if "finger" in str(prim.GetPath()).lower() and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, UsdShade.Tokens.weakerThanDescendants, "physics"
                )

    def initialize_runtime(self) -> None:
        """Initialize tensors after world reset and create the official Panda Lula IK."""
        if self.pure_newton:
            return
        from isaacsim.robot_motion.motion_generation import (
            ArticulationKinematicsSolver,
            LulaKinematicsSolver,
            interface_config_loader,
        )

        self.articulation.initialize()
        q = np.asarray(self.config.initial_joint_positions, dtype=np.float64)
        self.articulation.set_joint_positions(q)
        self.articulation.set_joint_velocities(np.zeros_like(q))
        self._configure_drives()
        self._lula = LulaKinematicsSolver(
            **interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        )
        self._lula.set_robot_base_pose(
            np.asarray(self.config.base_position), np.asarray(self.config.base_orientation)
        )
        self.kinematics = ArticulationKinematicsSolver(
            self.articulation, self._lula, "right_gripper"
        )
        stage = self.sim_world.stage
        self._finger_links = [
            stage.GetPrimAtPath(self.config.prim_path + "/" + name)
            for name in ("panda_leftfinger", "panda_rightfinger", "panda_hand")
        ]
        self._collider_link_indices = [
            self.articulation._articulation_view.get_link_index(name)
            for name in ("panda_leftfinger", "panda_rightfinger", "panda_hand")
        ]
        self.state["ready"] = True

    def ensure_runtime_handles(self) -> None:
        """Restore PhysX handles after GUI Stop/Play without resetting the robot pose."""
        if self.pure_newton:
            return
        if self.articulation is None:
            raise RuntimeError("Franka articulation has not been created")
        if not self.articulation.handles_initialized:
            # initialize() recreates views; pose reset belongs to initialize_runtime().
            self.articulation.initialize()
            if not self.articulation.handles_initialized:
                raise RuntimeError(
                    "Franka PhysX handles could not be restored after GUI inspection"
                )
            self._configure_drives()

    def _configure_drives(self) -> None:
        """Apply the configured drive gains and effort limits to the current PhysX view."""
        controller = self.articulation.get_articulation_controller()
        controller.set_gains(
            kps=np.array([self.config.arm_stiffness] * 7 + [self.config.finger_stiffness] * 2),
            kds=np.array([self.config.arm_damping] * 7 + [self.config.finger_damping] * 2),
        )
        controller.set_max_efforts(
            np.array([87.0] * 4 + [12.0] * 3 + [self.config.finger_max_force] * 2)
        )

    def command_tcp(self, position: np.ndarray, quaternion: np.ndarray, finger: float) -> bool:
        """Apply IK joint position targets and a force-limited finger opening in metres."""
        from isaacsim.core.utils.types import ArticulationAction

        action, solved = self.kinematics.compute_inverse_kinematics(position, quaternion)
        if solved:
            self.articulation.apply_action(action)
            indices = action.joint_indices
            indices = (
                np.arange(len(action.joint_positions)) if indices is None else np.asarray(indices)
            )
            self._recorded_action[indices] = np.asarray(action.joint_positions)

        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=np.asarray([finger, finger]), joint_indices=np.asarray([7, 8])
            )
        )
        self._recorded_action[-2:] = finger
        self.state["ik_solved"] = bool(solved)
        return bool(solved)

    def tcp_position(self) -> np.ndarray:
        """Read the actual Panda TCP, rather than the commanded target."""
        position, _ = self.kinematics.compute_end_effector_pose()
        return np.asarray(position, dtype=float)

    def finger_positions(self) -> np.ndarray:
        """Read the two actual finger prismatic positions from PhysX, in metres."""
        return np.asarray(self.articulation.get_joint_positions(), dtype=float)[7:9]

    def collider_transforms(self) -> np.ndarray:
        """Return finger and palm body transforms as xyz plus xyzw, in metres."""
        # Read PhysX tensors: render/Fabric updates can leave USD transforms
        # several steps behind the real link poses.
        values = self.articulation._articulation_view._physics_view.get_link_transforms()
        if hasattr(values, "numpy"):
            values = values.numpy()
        return np.asarray(values, dtype=np.float32)[0, self._collider_link_indices].copy()

    def recording_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Read actual PhysX joints and the position targets last sent to the controller."""
        values = self.articulation.get_joint_positions()
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=float).copy(), self._recorded_action.copy()


# ================================================================================================
# Franka articulation with the supplied V7 experiment's long parallel jaw geometry.
# ================================================================================================


class PackagePadRobotConfig(FrankaNewtonRobotConfig):
    """V7-sized simulation jaws, expressed in each mirrored finger's local frame."""

    backend: str = "franka_package_pads"
    pad_half_extents_m: tuple[PositiveFloat, PositiveFloat, PositiveFloat] = (0.03, 0.007, 0.042)
    pad_center_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.0, 0.007, 0.055)
    pad_density_kg_m3: FiniteFloat = Field(default=1000.0, gt=0)


class PackagePadRobot(FrankaNewtonRobot):
    """Attach the V7 14×60×84 mm boxes to actual Featherstone finger bodies."""

    config_cls = PackagePadRobotConfig

    def add_newton_robot(self, builder: Any, *, scale: float = 100.0) -> None:
        """Replace finger collision proxies while retaining articulation and finger visuals."""
        import newton
        import warp as wp

        super().add_newton_robot(builder, scale=scale)
        bodies = [
            i
            for i, label in enumerate(builder.body_label)
            if label.endswith(("fr3_leftfinger", "fr3_rightfinger"))
        ]
        if len(bodies) != 2:
            raise ValueError("V7 jaws require two independent Franka finger bodies")
        collision = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        for index, body in enumerate(builder.shape_body):
            if body in bodies:
                builder.shape_flags[index] &= ~collision
        self.package_pad_shapes = []
        extents = [value * scale for value in self.config.pad_half_extents_m]
        center = [value * scale for value in self.config.pad_center_m]
        for body in bodies:
            shape = builder.add_shape_box(
                body,
                xform=wp.transform(center, wp.quat_identity()),
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=self.config.pad_density_kg_m3 / scale**3,
                    mu=self.config.finger_friction,
                    margin=0.0002 * scale,
                ),
                label="V7_package_pad",
            )
            self.package_pad_shapes.append(shape)


# ================================================================================================
# Guided axial assembly arm with a continuously rotating tool spindle.
# ================================================================================================


class ScrewSpindleRobotConfig(_RobotCameraConfig):
    """A vertical assembly arm, axial spring feed, and unlimited rotary tool joint."""

    backend: str = "screw_spindle"
    prim_path: str = "/World/robots/ScrewSpindle"
    base_position: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.1, 0, 0.732389)
    axial_stiffness_n_m: float = Field(default=100.0, gt=0)
    axial_damping_n_s_m: float = Field(default=5.0, ge=0)
    axial_preload_target_m: float = Field(default=-0.04, lt=0, ge=-0.06)
    spindle_damping_nm_s_rad: float = Field(default=0.02, gt=0)
    maximum_torque_nm: float = Field(default=0.3, gt=0)
    carriage_mass_kg: float = Field(default=0.05, gt=0)


class ScrewSpindleRobot(BaseRobot):
    """Represent the bolt head held by a declared rigid tool chuck on a guided spindle."""

    config_cls = ScrewSpindleRobotConfig
    pure_newton = True
    recording_robot_type = "guided_continuous_screw_spindle"
    recording_joint_names = ["axial_displacement_m", "spindle_angle_rad"]

    def __init__(self, config: ScrewSpindleRobotConfig | None = None) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("robot")

    def add_newton_robot(
        self, builder: Any, *, mount_z_m: float, bolt_mesh: Any, shape_config: Any
    ) -> int:
        """Define the guided axial articulation and unlimited tool; return its spindle body."""
        import warp as wp

        mount = wp.transform((0, 0, mount_z_m), wp.quat_identity())
        carriage = builder.add_link(
            xform=mount,
            mass=self.config.carriage_mass_kg,
            inertia=wp.mat33(np.eye(3) * 1e-4),
            label="axial_carriage",
        )
        spindle = builder.add_link(xform=mount, label="continuous_spindle")
        builder.add_shape_mesh(
            spindle,
            mesh=bolt_mesh,
            xform=wp.transform((0, 0, 0), wp.quat(1, 0, 0, 0)),
            cfg=shape_config,
            label="chucked_bolt",
        )
        axial = builder.add_joint_prismatic(
            -1,
            carriage,
            parent_xform=mount,
            axis=wp.vec3(0, 0, 1),
            target_ke=self.config.axial_stiffness_n_m,
            target_kd=self.config.axial_damping_n_s_m,
            target_pos=self.config.axial_preload_target_m,
            limit_lower=-0.04,
            limit_upper=0.005,
            effort_limit=10,
            label="axial_spring_feed",
        )
        rotary = builder.add_joint_revolute(
            carriage,
            spindle,
            axis=wp.vec3(0, 0, 1),
            target_ke=0,
            target_kd=self.config.spindle_damping_nm_s_rad,
            target_vel=0,
            effort_limit=self.config.maximum_torque_nm,
            label="unlimited_rotary_spindle",
        )
        builder.add_articulation([axial, rotary])
        return spindle

    def init_robot(self, sim_world: Any) -> None:
        """Author the visible support, rail, carriage and spindle without creating physics."""
        from pxr import Gf, UsdGeom

        if not sim_world.uses_direct_newton:
            raise ValueError("Screw spindle requires direct Newton")
        self.sim_world = sim_world
        root = UsdGeom.Xform.Define(sim_world.stage, self.config.prim_path)
        root.AddTranslateOp().Set(Gf.Vec3d(*self.config.base_position))
        for name, center, scale in (
            ("column", (-0.12, 0, 0.14), (0.025, 0.035, 0.14)),
            ("arm", (-0.06, 0, 0.24), (0.085, 0.035, 0.025)),
            ("guide", (0, 0.03, 0.15), (0.02, 0.01, 0.09)),
        ):
            cube = UsdGeom.Cube.Define(sim_world.stage, f"{root.GetPath()}/{name}")
            cube.CreateSizeAttr(2)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3f(*scale))
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.14, 0.3, 0.5)])
        tool = UsdGeom.Xform.Define(sim_world.stage, f"{root.GetPath()}/tool")
        self.tool_transform = tool.AddTransformOp()
        cylinder = UsdGeom.Cylinder.Define(sim_world.stage, f"{tool.GetPath()}/chuck")
        cylinder.CreateRadiusAttr(0.018)
        cylinder.CreateHeightAttr(0.04)
        cylinder.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.025))
        cylinder.CreateDisplayColorAttr([Gf.Vec3f(0.13, 0.14, 0.16)])
        stripe = UsdGeom.Cube.Define(sim_world.stage, f"{tool.GetPath()}/rotation_marker")
        stripe.CreateSizeAttr(2)
        stripe.AddTranslateOp().Set(Gf.Vec3d(0.0175, 0, 0.025))
        stripe.AddScaleOp().Set(Gf.Vec3f(0.001, 0.003, 0.019))
        stripe.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.25, 0.03)])
        self.sensors.update(create_cameras(sim_world.stage, self.config.cameras))

    def arrange(self) -> None:
        """Keep the explicitly mounted assembly arm pose."""
        position = np.array(self.config.base_position)
        self.poses["base"] = (position, np.array([1, 0, 0, 0]))
        self.poses_realtime.update(self.poses)

    def adjust(self) -> None:
        """The fixture uses a deterministic alignment."""


class CartonGripperRobotConfig(PackagePadRobotConfig):
    """Franka arm with an explicitly simulated 260mm-stroke parallel gripper."""

    backend: str = "franka_carton_gripper"
    pad_half_extents_m: tuple[PositiveFloat, PositiveFloat, PositiveFloat] = (
        0.00875,
        0.0076,
        0.045,
    )
    pad_center_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0, 0.00758, 0.075)
    tcp_extension_m: FiniteFloat = Field(default=0.06, gt=0, le=0.08)
    maximum_finger_opening_m: float = Field(default=0.13, gt=0.105, le=0.15)


class CartonGripperRobot(PackagePadRobot):
    """Widen the visible finger carriage range; retain actual moving finger collisions."""

    config_cls = CartonGripperRobotConfig
    recording_robot_type = "franka_long_stroke_parallel_gripper"
    recording_joint_names = [f"fr3_joint{i}" for i in range(1, 8)] + [
        "left_finger_m",
        "right_finger_m",
    ]

    def add_newton_robot(self, builder: Any, *, scale: float = 1.0) -> None:
        """Declare the replacement gripper stroke in actual joint limits before finalization."""
        super().add_newton_robot(builder, scale=scale)
        for joint, label in enumerate(builder.joint_label):
            if "finger_joint" in label:
                axis = builder.joint_qd_start[joint]
                builder.joint_limit_upper[axis] = self.config.maximum_finger_opening_m * scale
