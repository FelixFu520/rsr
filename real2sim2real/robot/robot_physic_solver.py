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
from real2sim2real.simworld.base import BaseSimWorld
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
            apply_up_axis_from_stage=True,
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


# ================================================================================================
# UR5e + a flange-mounted electric driver, axial spring and continuous hex socket.
# The arm and spindle are one solved articulation; only actual body poses are published.
# ================================================================================================


class Ur5eScrewdriverRobotConfig(ScrewSpindleRobotConfig):
    """UR5e SI drive gains, task-local USD and compliant socket driver dimensions."""

    backend: str = "ur5e_screwdriver"
    prim_path: str = "/World/robots/UR5eScrewdriver"
    usd_path: str = "assets/PhysicSolver/Robot/ur5e/ur5e.usda"
    arm_base_offset_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (-0.45, 0.0, 0.0)
    tool_length_m: PositiveFloat = 0.30
    arm_stiffness_nm_rad: PositiveFloat = 12000.0
    arm_damping_nm_s_rad: PositiveFloat = 400.0
    arm_max_torque_nm: PositiveFloat = 150.0
    # Small downward loading move at startup; no pitch-based feed is prescribed.
    loading_stroke_m: float = Field(default=0.003, ge=0, le=0.005)
    loading_duration_s: PositiveFloat = 1.0


class Ur5eScrewdriverRobot(ScrewSpindleRobot):
    """A physically articulated UR5e holding a compliant electric socket driver."""

    config_cls = Ur5eScrewdriverRobotConfig
    recording_robot_type = "ur5e_compliant_electric_socket_driver"
    recording_joint_names = [
        "shoulder_pan_rad",
        "shoulder_lift_rad",
        "elbow_rad",
        "wrist_1_rad",
        "wrist_2_rad",
        "wrist_3_rad",
        "axial_displacement_m",
        "spindle_angle_rad",
    ]

    def init_robot(self, sim_world: Any) -> None:
        """Load task-local visual assets; defer solver allocation to episode start."""
        from pxr import Gf, UsdGeom

        if not sim_world.uses_direct_newton:
            raise ValueError("UR5e screwdriver requires direct Newton")
        self.sim_world = sim_world
        root = UsdGeom.Xform.Define(sim_world.stage, self.config.prim_path)
        root.AddTranslateOp().Set(Gf.Vec3d(*self.config.base_position))
        arm = UsdGeom.Xform.Define(sim_world.stage, f"{root.GetPath()}/arm")
        arm.GetPrim().GetReferences().AddReference(resolve_usd_path(self.config.usd_path))
        self._variants(arm.GetPrim())
        self._visual_ops = {}
        self.sensors.update(create_cameras(sim_world.stage, self.config.cameras))

    def arrange(self) -> None:
        """Cache the arm's actual table-mounted base, relative to the screw fixture."""
        position = np.array(self.config.base_position) + self.config.arm_base_offset_m
        self.poses["base"] = (position, np.array([1.0, 0.0, 0.0, 0.0]))
        self.poses_realtime.update(self.poses)

    @staticmethod
    def _variants(root: Any) -> None:
        for name, value in {
            "PhysX": "Enabled",
            "Newton": "Disabled",
            "MuJoCo": "Disabled",
            "Gripper": "none",
            "Sensor": "none",
        }.items():
            variant = root.GetVariantSets().GetVariantSet(name)
            if value in variant.GetVariantNames():
                variant.SetVariantSelection(value)

    def add_newton_robot(
        self, builder: Any, *, mount_z_m: float, bolt_mesh: Any, shape_config: Any
    ) -> int:
        """Import arm plus driver as one articulation, with real helical bolt collision."""
        import warp as wp
        from newton.usd import SchemaResolverPhysx
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        from real2sim2real.utils.battery_disassembly_utils import (
            ToolFrame,
            matrix_to_quat,
            solve_ik,
            transform,
        )
        from real2sim2real.utils.physic_solver_utils import screw_arm_chain

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/UR5e")
        root.GetPrim().GetReferences().AddReference(resolve_usd_path(self.config.usd_path))
        self._variants(root.GetPrim())
        # The no-gripper variant leaves a dangling fixed joint; it must not anchor the wrist.
        UsdPhysics.Joint(
            stage.GetPrimAtPath("/UR5e/joints/robot_gripper_joint")
        ).CreateJointEnabledAttr(False)
        chain, mount = screw_arm_chain(stage, "/UR5e", self.config.tool_length_m)
        self.chain, self.chain_base = chain, transform(translation=self.config.arm_base_offset_m)
        self.tool_mount = mount
        self.mount_z_m = mount_z_m
        seeds = [
            [0, -1.2, 1.7, -2, -1.57, 0],
            [0, -1.5, 2, -2, 1.57, 0],
            [0.4, -1, 1.5, -2, -1.5, 0],
        ]
        for seed in seeds:
            goal, pe, ae = solve_ik(
                chain,
                self.chain_base,
                ToolFrame(np.zeros(3)),
                [0, 0, mount_z_m],
                rotation=np.eye(3),
                seed=seed,
                iterations=600,
                position_tol=1e-8,
                angle_tol=1e-8,
            )
            if pe < 1e-6 and ae < 1e-6:
                break
        else:
            raise ValueError(f"UR5e screwdriver IK unreachable: position={pe}, angle={ae}")
        self.arm_goal, self.control_time_s = goal, 0.0
        root.ClearXformOpOrder()
        root.AddTranslateOp().Set(Gf.Vec3d(*self.config.arm_base_offset_m))
        wrist = UsdGeom.Xformable(stage.GetPrimAtPath("/UR5e/wrist_3_link"))
        zero_tip = np.asarray(wrist.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T @ mount

        def body(path: str, mass: float) -> None:
            item = UsdGeom.Xform.Define(stage, path)
            item.AddTransformOp().Set(Gf.Matrix4d(zero_tip.T.tolist()))
            UsdPhysics.RigidBodyAPI.Apply(item.GetPrim())
            api = UsdPhysics.MassAPI.Apply(item.GetPrim())
            api.CreateMassAttr(mass)
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4, 1e-4, 1e-4))

        body("/UR5e/driver_float", self.config.carriage_mass_kg)
        body("/UR5e/driver_spindle", 0.05)
        axial = UsdPhysics.PrismaticJoint.Define(stage, "/UR5e/joints/driver_axial")
        axial.CreateBody0Rel().SetTargets(["/UR5e/wrist_3_link"])
        axial.CreateBody1Rel().SetTargets(["/UR5e/driver_float"])
        axial.CreateLocalPos0Attr(Gf.Vec3f(*mount[:3, 3]))
        q = matrix_to_quat(mount[:3, :3])
        axial.CreateLocalRot0Attr(Gf.Quatf(float(q[0]), *q[1:].tolist()))
        axial.CreateAxisAttr("Z")
        axial.CreateLowerLimitAttr(-0.04)
        axial.CreateUpperLimitAttr(0.005)
        spin = UsdPhysics.RevoluteJoint.Define(stage, "/UR5e/joints/driver_spin")
        spin.CreateBody0Rel().SetTargets(["/UR5e/driver_float"])
        spin.CreateBody1Rel().SetTargets(["/UR5e/driver_spindle"])
        spin.CreateAxisAttr("Z")
        if getattr(self.config, "releasable_socket", False):
            for side in ("p", "n"):
                path = f"/UR5e/driver_socket_{side}"
                body(path, 0.025)
                joint = UsdPhysics.PrismaticJoint.Define(stage, f"/UR5e/joints/socket_{side}")
                joint.CreateBody0Rel().SetTargets(["/UR5e/driver_spindle"])
                joint.CreateBody1Rel().SetTargets([path])
                joint.CreateAxisAttr("X")
                if side == "n":
                    joint.CreateLocalRot0Attr(Gf.Quatf(0.0, 0.0, 0.0, 1.0))
                    joint.CreateLocalRot1Attr(Gf.Quatf(0.0, 0.0, 0.0, 1.0))
                joint.CreateLowerLimitAttr(0.0)
                joint.CreateUpperLimitAttr(0.03)
        parsed = builder.add_usd(
            stage,
            root_path="/UR5e",
            floating=False,
            enable_self_collisions=False,
            load_visual_shapes=False,
            schema_resolvers=[SchemaResolverPhysx()],
        )
        joint_map = parsed["path_joint_map"]
        self.arm_dofs = [
            builder.joint_q_start[joint_map[f"/UR5e/joints/{name}"]] for name in chain.names
        ]
        self.screw_dofs = [
            builder.joint_q_start[joint_map[f"/UR5e/joints/driver_{name}"]]
            for name in ("axial", "spin")
        ]
        if self.arm_dofs + self.screw_dofs != list(range(8)):
            raise ValueError(f"Unexpected UR5e joint layout: {self.arm_dofs}, {self.screw_dofs}")
        if getattr(self.config, "releasable_socket", False):
            self.socket_dofs = [
                builder.joint_q_start[joint_map[f"/UR5e/joints/socket_{s}"]] for s in ("p", "n")
            ]
            if self.socket_dofs != [9, 8]:
                raise ValueError(f"Unexpected socket joint order: {self.socket_dofs}")
            for dof in self.socket_dofs:
                builder.joint_q[dof] = self.config.socket_initial_open_m
                builder.joint_target_q[dof] = self.config.socket_initial_open_m
                builder.joint_target_ke[dof] = self.config.socket_stiffness_n_m
                builder.joint_target_kd[dof] = self.config.socket_damping_ns_m
                builder.joint_effort_limit[dof] = self.config.socket_max_force_n
                builder.joint_target_mode[dof] = 3
        for dof, value in zip(self.arm_dofs, goal, strict=True):
            builder.joint_q[dof] = float(value)
            builder.joint_target_q[dof] = float(value)
            builder.joint_target_ke[dof] = self.config.arm_stiffness_nm_rad
            builder.joint_target_kd[dof] = self.config.arm_damping_nm_s_rad
            builder.joint_effort_limit[dof] = self.config.arm_max_torque_nm
            builder.joint_armature[dof] = 0.05
            builder.joint_target_mode[dof] = 3  # POSITION_VELOCITY
        a, s = self.screw_dofs
        builder.joint_target_ke[a] = self.config.axial_stiffness_n_m
        builder.joint_target_kd[a] = self.config.axial_damping_n_s_m
        builder.joint_target_mode[a] = 3
        builder.joint_effort_limit[a] = 10.0
        builder.joint_target_ke[s] = 0.0
        builder.joint_target_kd[s] = self.config.spindle_damping_nm_s_rad
        builder.joint_target_mode[s] = 2  # VELOCITY
        builder.joint_effort_limit[s] = self.config.maximum_torque_nm
        spindle = parsed["path_body_map"]["/UR5e/driver_spindle"]
        self.spindle_body = spindle
        self.wrist_body = parsed["path_body_map"]["/UR5e/wrist_3_link"]
        if getattr(self.config, "releasable_socket", False):
            from dataclasses import replace

            from real2sim2real.utils.physic_solver_utils import screw_socket_parts

            cfg = replace(
                shape_config,
                kh=self.config.socket_hydroelastic_stiffness,
                mu=self.config.socket_friction,
                density=0,
                sdf_max_resolution=64,
            )
            self.socket_shapes = []
            self.socket_bodies = {
                side: parsed["path_body_map"][f"/UR5e/driver_socket_{side}"] for side in ("p", "n")
            }
            for part in screw_socket_parts(self.config.socket_inner_radius_m):
                parent = (
                    spindle
                    if part["side"] == "fixed"
                    else parsed["path_body_map"][f"/UR5e/driver_socket_{part['side']}"]
                )
                shape = builder.add_shape_box(
                    parent,
                    hx=part["half"][0],
                    hy=part["half"][1],
                    hz=part["half"][2],
                    xform=wp.transform(
                        part["center"], wp.quat_from_axis_angle(wp.vec3(0, 0, 1), part["yaw"])
                    ),
                    cfg=cfg,
                    label=f"contact_socket_{part['name']}",
                )
                self.socket_shapes.append(shape)
        else:
            builder.add_shape_mesh(
                spindle,
                mesh=bolt_mesh,
                xform=wp.transform((0, 0, 0), wp.quat(1, 0, 0, 0)),
                cfg=shape_config,
                label="socket_clamped_bolt",
            )
        self._author_driver_visuals()
        for path, index in parsed["path_body_map"].items():
            leaf = path.rsplit("/", 1)[-1]
            visual_path = (
                f"{self.config.prim_path}/{leaf}"
                if leaf.startswith("driver_")
                else f"{self.config.prim_path}/arm/{leaf}"
            )
            prim = self.sim_world.stage.GetPrimAtPath(visual_path)
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            self._visual_ops[index] = xf.AddTransformOp()
        # Keep the full dependency closure verifiable, including materials and textures.
        self.state["asset_layers"] = [
            layer.identifier for layer in stage.GetUsedLayers() if not layer.anonymous
        ]
        return spindle

    def _author_driver_visuals(self) -> None:
        from pxr import Gf, UsdGeom

        stage = self.sim_world.stage
        for body in ("driver_housing", "driver_float", "driver_spindle"):
            UsdGeom.Xform.Define(stage, f"{self.config.prim_path}/{body}")
        self._housing_op = UsdGeom.Xformable(
            stage.GetPrimAtPath(f"{self.config.prim_path}/driver_housing")
        ).AddTransformOp()
        # Body is bolted to flange; shaft and visible orange marker rotate with the socket.
        for parent, name, radius, height, z, color in (
            ("driver_housing", "motor", 0.032, 0.16, 0.19, (0.12, 0.22, 0.32)),
            ("driver_housing", "flange_adapter", 0.043, 0.025, 0.2875, (0.25, 0.28, 0.31)),
            ("driver_float", "nose", 0.022, 0.055, 0.0825, (0.5, 0.53, 0.57)),
            ("driver_spindle", "shaft", 0.008, 0.055, 0.030, (0.65, 0.68, 0.72)),
            ("driver_spindle", "hex_socket", 0.019, 0.025, 0.006, (0.23, 0.26, 0.30)),
        ):
            if name == "hex_socket" and getattr(self.config, "releasable_socket", False):
                continue
            mesh = UsdGeom.Cylinder.Define(stage, f"{self.config.prim_path}/{parent}/{name}")
            mesh.CreateRadiusAttr(radius)
            mesh.CreateHeightAttr(height)
            mesh.AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        stripe = UsdGeom.Cube.Define(
            stage, f"{self.config.prim_path}/driver_spindle/rotation_marker"
        )
        stripe.AddTranslateOp().Set(Gf.Vec3d(0.0185, 0, 0.006))
        stripe.AddScaleOp().Set(Gf.Vec3f(0.0015, 0.004, 0.0125))
        stripe.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.25, 0.02)])
        if getattr(self.config, "releasable_socket", False):
            from real2sim2real.utils.physic_solver_utils import screw_socket_parts

            for part in screw_socket_parts(self.config.socket_inner_radius_m):
                parent = (
                    "driver_spindle" if part["side"] == "fixed" else f"driver_socket_{part['side']}"
                )
                UsdGeom.Xform.Define(stage, f"{self.config.prim_path}/{parent}")
                mesh = UsdGeom.Cube.Define(
                    stage, f"{self.config.prim_path}/{parent}/socket_{part['name']}"
                )
                mesh.AddTranslateOp().Set(Gf.Vec3d(*part["center"]))
                mesh.AddRotateZOp().Set(float(np.rad2deg(part["yaw"])))
                mesh.AddScaleOp().Set(Gf.Vec3f(*part["half"]))
                mesh.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.35, 0.42)])

    def screw_control(
        self, measured: np.ndarray, speed: float, hold: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the flange by 3mm once, then hold it; the thread determines actual feed."""
        from real2sim2real.utils.battery_disassembly_utils import ToolFrame, solve_ik

        self.control_time_s += self.sim_world.config.physics_dt
        u = min(self.control_time_s / self.config.loading_duration_s, 1.0)
        dz = self.config.loading_stroke_m * u * u * (3 - 2 * u)
        goal, pe, ae = solve_ik(
            self.chain,
            self.chain_base,
            ToolFrame(np.zeros(3)),
            [0, 0, self.mount_z_m - dz],
            rotation=np.eye(3),
            seed=self.arm_goal,
            position_tol=1e-7,
            angle_tol=1e-7,
        )
        if pe > 1e-5 or ae > 1e-5:
            raise ValueError("UR5e loading stroke lost IK continuity")
        self.arm_goal = goal
        q = np.zeros(8)
        q[self.arm_dofs] = goal
        q[self.screw_dofs[0]] = (
            measured[self.screw_dofs[0]] if hold else self.config.axial_preload_target_m
        )
        qd = np.zeros(8)
        qd[self.screw_dofs[1]] = speed
        action = q.copy()
        action[self.screw_dofs[1]] = speed
        return q, qd, action

    def publish_newton(self, poses: np.ndarray) -> None:
        """Publish solved link transforms; source material bindings remain intact."""
        from pxr import Gf

        from real2sim2real.utils.battery_disassembly_utils import quat_to_matrix, transform

        wrist = poses[self.wrist_body]
        housing = transform(quat_to_matrix(wrist[[6, 3, 4, 5]]), wrist[:3]) @ self.tool_mount
        self._housing_op.Set(Gf.Matrix4d(housing.T.tolist()))

        for index, op in self._visual_ops.items():
            pose = poses[index]
            matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(float(pose[6]), *pose[3:6].astype(float)))
            matrix.SetTranslateOnly(Gf.Vec3d(*pose[:3].astype(float)))
            op.Set(matrix)
        pose = poses[self.spindle_body]
        self.poses_realtime["tool"] = (
            pose[:3].astype(float) + self.config.base_position,
            pose[[6, 3, 4, 5]].astype(float),
        )
        self.poses.setdefault("tool", self.poses_realtime["tool"])
        self.state["arm_target_rad"] = self.arm_goal.copy()


# ================================================================================================
# Dual UR5e assembly: the second arm closes two explicit fingers around the free M20 nut.
# ================================================================================================


class DualUr5eScrewRobotConfig(Ur5eScrewdriverRobotConfig):
    """Two physical UR5e arms, an electric driver and an explicit parallel nut gripper."""

    backend: str = "dual_ur5e_screwdriver"
    nut_arm_base_offset_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.45, 0.28, 0.0)
    gripper_length_m: PositiveFloat = 0.14
    finger_open_m: float = Field(default=0.028, gt=0.02, le=0.04)
    finger_closed_m: float = Field(default=0.013, gt=0.01, lt=0.015)
    finger_stiffness_n_m: PositiveFloat = 30000.0
    finger_damping_ns_m: PositiveFloat = 150.0
    finger_max_force_n: PositiveFloat = 100.0
    finger_friction: PositiveFloat = 1.5
    finger_hydroelastic_stiffness: PositiveFloat = 1e10
    releasable_socket: bool = False
    socket_inner_radius_m: float = Field(default=0.0145, gt=0.014, lt=0.015)
    socket_friction: PositiveFloat = 0.8
    socket_hydroelastic_stiffness: PositiveFloat = 1e10
    socket_open_m: float = Field(default=0.025, ge=0.02, le=0.028)
    socket_initial_open_m: float = Field(default=0.001, ge=0.001, le=0.005)
    socket_stiffness_n_m: PositiveFloat = 30000.0
    socket_damping_ns_m: PositiveFloat = 150.0
    socket_max_force_n: PositiveFloat = 100.0


class DualUr5eScrewRobot(Ur5eScrewdriverRobot):
    """One driver arm and one contact-only nut-holding arm in the same Newton model."""

    config_cls = DualUr5eScrewRobotConfig
    recording_robot_type = "dual_ur5e_nut_grip_and_electric_driver"

    @property
    def recording_joint_names(self) -> list[str]:
        """Include the physical bolt-clamp axes only for the detachable tool variant."""
        driver = [f"driver_{name}" for name in Ur5eScrewdriverRobot.recording_joint_names]
        if self.config.releasable_socket:
            driver += ["driver_socket_n_m", "driver_socket_p_m"]
        return (
            driver
            + [
                f"holder_{name}_rad"
                for name in (
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow",
                    "wrist_1",
                    "wrist_2",
                    "wrist_3",
                )
            ]
            + ["holder_finger_n_m", "holder_finger_p_m"]
        )

    def init_robot(self, sim_world: Any) -> None:
        """Reference a second complete arm; generate only its visible gripper geometry."""
        from pxr import UsdGeom

        super().init_robot(sim_world)
        arm = UsdGeom.Xform.Define(sim_world.stage, f"{self.config.prim_path}/nut_arm")
        arm.GetPrim().GetReferences().AddReference(resolve_usd_path(self.config.usd_path))
        self._variants(arm.GetPrim())
        self._holder_ops = {}

    def add_newton_robot(
        self, builder: Any, *, mount_z_m: float, bolt_mesh: Any, shape_config: Any
    ) -> int:
        """Import both driven arms, followed by two independent prismatic fingers."""
        import newton
        import warp as wp
        from newton.usd import SchemaResolverPhysx
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        from real2sim2real.utils.battery_disassembly_utils import matrix_to_quat, transform
        from real2sim2real.utils.physic_solver_utils import screw_arm_chain

        spindle = super().add_newton_robot(
            builder, mount_z_m=mount_z_m, bolt_mesh=bolt_mesh, shape_config=shape_config
        )
        self.bolt_shape = builder.shape_count - 1
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/NutArm")
        root.GetPrim().GetReferences().AddReference(resolve_usd_path(self.config.usd_path))
        self._variants(root.GetPrim())
        UsdPhysics.Joint(
            stage.GetPrimAtPath("/NutArm/joints/robot_gripper_joint")
        ).CreateJointEnabledAttr(False)
        self.holder_chain, mount = screw_arm_chain(stage, "/NutArm", self.config.gripper_length_m)
        self.holder_base = transform(translation=self.config.nut_arm_base_offset_m)
        # Local Y opens along world X (the hex flats); local Z points back toward +Y.
        self.holder_rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
        self.holder_goal = self._solve_pose(
            self.holder_chain, self.holder_base, [0, 0, self.pickup_z_m], self.holder_rotation
        )
        root.ClearXformOpOrder()
        root.AddTranslateOp().Set(Gf.Vec3d(*self.config.nut_arm_base_offset_m))
        wrist = UsdGeom.Xformable(stage.GetPrimAtPath("/NutArm/wrist_3_link"))
        zero_tip = np.asarray(wrist.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T @ mount
        for name, mass in (("holder_mount", 0.5), ("jaw_p", 0.08), ("jaw_n", 0.08)):
            body = UsdGeom.Xform.Define(stage, f"/NutArm/{name}")
            body.AddTransformOp().Set(Gf.Matrix4d(zero_tip.T.tolist()))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            api = UsdPhysics.MassAPI.Apply(body.GetPrim())
            api.CreateMassAttr(mass)
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.001, 0.001, 0.001))
        fixed = UsdPhysics.FixedJoint.Define(stage, "/NutArm/joints/holder_mount_joint")
        fixed.CreateBody0Rel().SetTargets(["/NutArm/wrist_3_link"])
        fixed.CreateBody1Rel().SetTargets(["/NutArm/holder_mount"])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(*mount[:3, 3]))
        q = matrix_to_quat(mount[:3, :3])
        fixed.CreateLocalRot0Attr(Gf.Quatf(float(q[0]), *q[1:].tolist()))
        for side in ("p", "n"):
            joint = UsdPhysics.PrismaticJoint.Define(stage, f"/NutArm/joints/finger_{side}")
            joint.CreateBody0Rel().SetTargets(["/NutArm/holder_mount"])
            joint.CreateBody1Rel().SetTargets([f"/NutArm/jaw_{side}"])
            joint.CreateAxisAttr("Y")
            # Both coordinates are positive openings; reflect the negative finger frame.
            if side == "n":
                joint.CreateLocalRot0Attr(Gf.Quatf(0.0, 0.0, 0.0, 1.0))
                joint.CreateLocalRot1Attr(Gf.Quatf(0.0, 0.0, 0.0, 1.0))
            joint.CreateLowerLimitAttr(0.009)
            joint.CreateUpperLimitAttr(0.035)
        parsed = builder.add_usd(
            stage,
            root_path="/NutArm",
            floating=False,
            enable_self_collisions=False,
            load_visual_shapes=False,
            schema_resolvers=[SchemaResolverPhysx()],
        )
        joint_map = parsed["path_joint_map"]
        self.holder_dofs = [
            builder.joint_q_start[joint_map[f"/NutArm/joints/{name}"]]
            for name in self.holder_chain.names
        ]
        self.finger_dofs = [
            builder.joint_q_start[joint_map[f"/NutArm/joints/finger_{side}"]] for side in ("p", "n")
        ]
        offset = 10 if self.config.releasable_socket else 8
        if self.holder_dofs != list(range(offset, offset + 6)) or self.finger_dofs != [
            offset + 7,
            offset + 6,
        ]:
            raise ValueError(f"Unexpected nut arm DOFs: {self.holder_dofs}, {self.finger_dofs}")
        for dof, value in zip(self.holder_dofs, self.holder_goal, strict=True):
            builder.joint_q[dof] = float(value)
            builder.joint_target_q[dof] = float(value)
            builder.joint_target_ke[dof] = self.config.arm_stiffness_nm_rad
            builder.joint_target_kd[dof] = self.config.arm_damping_nm_s_rad
            builder.joint_effort_limit[dof] = self.config.arm_max_torque_nm
            builder.joint_armature[dof] = 0.05
            builder.joint_target_mode[dof] = 3
        cfg = newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True,
            sdf_max_resolution=64,
            kh=self.config.finger_hydroelastic_stiffness,
            ke=1e7,
            kd=1e4,
            mu=self.config.finger_friction,
            margin=0,
            gap=0,
            density=1000,
        )
        self.finger_shapes, self.finger_bodies = [], []
        for side, dof in zip(("p", "n"), self.finger_dofs, strict=True):
            builder.joint_q[dof] = self.config.finger_open_m
            builder.joint_target_q[dof] = self.config.finger_open_m
            builder.joint_target_ke[dof] = self.config.finger_stiffness_n_m
            builder.joint_target_kd[dof] = self.config.finger_damping_ns_m
            builder.joint_effort_limit[dof] = self.config.finger_max_force_n
            builder.joint_target_mode[dof] = 3
            body = parsed["path_body_map"][f"/NutArm/jaw_{side}"]
            self.finger_bodies.append(body)
            sign = 1 if side == "p" else -1
            shape = builder.add_shape_box(
                body,
                hx=0.006,
                hy=0.004,
                hz=0.05,
                xform=wp.transform((0, sign * 0.004, 0.03), wp.quat_identity()),
                cfg=cfg,
                label=f"visible_nut_jaw_{side}",
            )
            self.finger_shapes.append(shape)
        self.holder_tcp_body = parsed["path_body_map"]["/NutArm/holder_mount"]
        for path, index in parsed["path_body_map"].items():
            leaf = path.rsplit("/", 1)[-1]
            visual_path = (
                f"{self.config.prim_path}/{leaf}"
                if leaf.startswith(("holder_", "jaw_"))
                else f"{self.config.prim_path}/nut_arm/{leaf}"
            )
            prim = UsdGeom.Xform.Define(self.sim_world.stage, visual_path)
            prim.ClearXformOpOrder()
            self._holder_ops[index] = prim.AddTransformOp()
        self._gripper_visuals()
        self.state["asset_layers"] = sorted(
            set(self.state["asset_layers"])
            | {layer.identifier for layer in stage.GetUsedLayers() if not layer.anonymous}
        )
        return spindle

    @staticmethod
    def _solve_pose(
        chain: Any,
        base: np.ndarray,
        point: Any,
        rotation: np.ndarray,
        seed: np.ndarray | None = None,
    ) -> np.ndarray:
        from real2sim2real.utils.battery_disassembly_utils import ToolFrame, solve_ik

        seeds = ([seed] if seed is not None else []) + [
            [2.6, -1.2, 1.7, -2, -1.57, 0],
            [0, -1.2, 1.7, -2, -1.57, 0],
            [0, -1.5, 2, -2, 1.57, 0],
            [-2.6, -1.2, 1.7, -2, -1.57, 0],
        ]
        for candidate in seeds:
            q, pe, ae = solve_ik(
                chain,
                base,
                ToolFrame(np.zeros(3)),
                point,
                rotation=rotation,
                seed=candidate,
                iterations=600,
                position_tol=1e-7,
                angle_tol=1e-7,
            )
            if pe < 1e-5 and ae < 1e-5:
                return q
        raise ValueError(f"Dual UR5e pose unreachable: {point}, residual={pe}, {ae}")

    def _gripper_visuals(self) -> None:
        from pxr import Gf, UsdGeom

        stage = self.sim_world.stage
        for side in ("p", "n"):
            sign = 1 if side == "p" else -1
            jaw = UsdGeom.Cube.Define(stage, f"{self.config.prim_path}/jaw_{side}/visible_finger")
            jaw.AddTranslateOp().Set(Gf.Vec3d(0, sign * 0.004, 0.03))
            jaw.AddScaleOp().Set(Gf.Vec3f(0.006, 0.004, 0.05))
            jaw.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.38, 0.06)])
        palm = UsdGeom.Cube.Define(stage, f"{self.config.prim_path}/holder_mount/palm")
        palm.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.11))
        palm.AddScaleOp().Set(Gf.Vec3f(0.025, 0.045, 0.03))
        palm.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.20, 0.28)])

    def dual_control(
        self,
        *,
        holder_z_m: float,
        driver_z_m: float,
        finger_m: float,
        preload: bool,
        speed: float,
        hold: bool,
        measured: np.ndarray,
        socket_opening_m: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Command both arms and the two fingers, without prescribing nut or thread motion."""
        self.arm_goal = self._solve_pose(
            self.chain, self.chain_base, [0, 0, driver_z_m], np.eye(3), self.arm_goal
        )
        self.holder_goal = self._solve_pose(
            self.holder_chain,
            self.holder_base,
            [0, 0, holder_z_m],
            self.holder_rotation,
            self.holder_goal,
        )
        q, qd = np.zeros(len(self.recording_joint_names)), np.zeros(len(self.recording_joint_names))
        q[self.arm_dofs], q[self.holder_dofs] = self.arm_goal, self.holder_goal
        q[self.finger_dofs] = finger_m
        if self.config.releasable_socket:
            q[self.socket_dofs] = socket_opening_m
        q[self.screw_dofs[0]] = (
            measured[self.screw_dofs[0]]
            if hold
            else self.config.axial_preload_target_m
            if preload
            else 0.0
        )
        qd[self.screw_dofs[1]] = speed
        action = q.copy()
        action[self.screw_dofs[1]] = speed
        return q, qd, action

    def publish_newton(self, poses: np.ndarray) -> None:
        """Publish both solved arms and fingers at the same physical sample."""
        from pxr import Gf

        super().publish_newton(poses)
        for index, op in self._holder_ops.items():
            pose = poses[index]
            matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(float(pose[6]), *pose[3:6].astype(float)))
            matrix.SetTranslateOnly(Gf.Vec3d(*pose[:3].astype(float)))
            op.Set(matrix)
        pose = poses[self.holder_tcp_body]
        self.poses_realtime["holder_tcp"] = (
            pose[:3].astype(float) + self.config.base_position,
            pose[[6, 3, 4, 5]].astype(float),
        )


# ================================================================================================
# Scripted kitchen knife; a visual tool without a hidden manipulator or contact-force model.
# ================================================================================================


class ScriptedKnifeConfig(BaseConfig):
    """Knife meshes supplied with the cutting example."""

    backend: str = "scripted_knife"
    mesh_path: str = "assets/PhysicSolver/Banana/knife.json"


class ScriptedKnifeRobot(BaseRobot):
    """Expose the prescribed knife pose through the ordinary robot lifecycle."""

    config_cls = ScriptedKnifeConfig

    def __init__(self, config: ScriptedKnifeConfig | None = None) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("robot")

    def init_robot(self, sim_world: BaseSimWorld) -> None:
        """Create the original blade, handle, bolster and rivets without physical contact."""
        import json

        from pxr import UsdGeom

        from real2sim2real.utils.physic_solver_utils import (
            CuttingMesh,
            cutting_author_mesh,
            cutting_material,
        )

        self.sim_world = sim_world
        path = Path(self.config.mesh_path).resolve()
        if not path.is_relative_to(Path("assets/PhysicSolver").resolve()):
            raise ValueError("Knife mesh must stay inside assets/PhysicSolver")
        root = UsdGeom.Xform.Define(sim_world.stage, "/World/Knife")
        self.transform = root.AddTransformOp()
        steel = cutting_material(sim_world.stage, "Steel", (0.52, 0.57, 0.62), 0.21, 1)
        handle = cutting_material(sim_world.stage, "Handle", (0.012, 0.016, 0.018), 0.26)
        for part in json.loads(path.read_text()):
            mesh = CuttingMesh(
                np.array(part["vertices"]),
                np.array(part["faces"]),
                np.zeros(len(part["faces"]), dtype=int),
            )
            cutting_author_mesh(
                sim_world.stage,
                "/World/Knife/" + part["name"],
                mesh,
                [steel if part["material"] == "steel" else handle],
            )

    def arrange(self) -> None:
        """Start with the knife raised above the first cut."""
        self.command(np.array([-0.096, -0.018, 0.15, 0.0]))
        self.poses.update(self.poses_realtime)

    def command(self, coordinates: np.ndarray) -> None:
        """Set x/y/z metres and yaw radians for this explicitly scripted tool."""
        from pxr import Gf

        x, y, z, yaw = coordinates
        matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(np.degrees(yaw))))
        matrix.SetTranslateOnly(Gf.Vec3d(float(x), float(y), float(z)))
        self.transform.Set(matrix)
        self.coordinates = np.asarray(coordinates, dtype=float)
        self.poses_realtime["knife"] = (
            self.coordinates[:3].copy(),
            np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
        )
