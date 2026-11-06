"""PhysicSolver 案例的机器人定义，重型依赖在初始化阶段加载。

franka_newton 后端使用 FrankaNewtonRobot：导入 FR3 URDF，由 Planner 的 Newton
运行时驱动；franka_physx 后端使用 FrankaPhysxRobot：加载 Panda USD、PhysX 驱动与
Lula IK，也为 hybrid 世界提供实测的手指/手掌碰撞位姿。两种后端由配置显式选择。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from real2sim2real.core.config import BaseConfig
from real2sim2real.robot.base import BaseRobot
from real2sim2real.utils.asset_utils import resolve_usd_path


class FrankaNewtonRobotConfig(BaseConfig):
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
        q = list(self.config.initial_joint_positions)
        q[-2:] = [value * scale for value in q[-2:]]
        builder.joint_q[:9] = q
        builder.joint_target_q[:9] = q
        builder.joint_target_ke[:7] = [self.config.arm_stiffness * scale**2] * 7
        builder.joint_target_kd[:7] = [self.config.arm_damping * scale**2] * 7
        builder.joint_target_ke[7:9] = [self.config.finger_stiffness] * 2
        builder.joint_target_kd[7:9] = [self.config.finger_damping] * 2
        builder.joint_target_mode[:9] = [int(newton.JointTargetMode.POSITION)] * 9
        # URDF import scales length/inertia, while these limits are authored
        # explicitly in the centimetre model's kg·cm/s² and kg·cm²/s² units.
        builder.joint_effort_limit[:7] = [value * scale**2 for value in [87.0] * 4 + [12.0] * 3]
        builder.joint_effort_limit[7:9] = [self.config.finger_max_force * scale] * 2


class FrankaPhysxRobotConfig(BaseConfig):
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
