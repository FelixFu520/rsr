"""Native Pro-M6 USD articulation imported into Newton, without joint teleportation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from real2sim2real.core.config import BaseConfig
from real2sim2real.robot.base import BaseRobot
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.cameras import CameraConfig, create_cameras

READY = {"left": [0.1, 0.1, 0.1, -1.5, 0, 0, -1.3], "right": [-0.1, 0.1, -0.1, -1.5, 0, 0, 1.3]}


class SortingRobotConfig(BaseConfig):
    """Preserve authored robot masses, joints, collision geometry and world placement."""

    backend: str = "pro_m6_newton"
    usd_path: str = "assets/Robot/pro_m6_696/pro_m6_696.usda"
    prim_path: str = "/World/robots/ProM6"
    control_mode: Literal["position"] = "position"
    arm_stiffness: float = Field(default=6000, gt=0)
    arm_damping: float = Field(default=300, gt=0)
    finger_stiffness: float = Field(default=40, gt=0)
    finger_damping: float = Field(default=1, gt=0)
    finger_friction: float = Field(default=1.4, ge=0)
    pad_width_m: Literal[0.03] = 0.03
    initial_gripper_rad: float = Field(default=0.0, ge=0, le=0.7, allow_inf_nan=False)
    cameras: list[CameraConfig] = Field(default_factory=list)


class SortingRobot(BaseRobot):
    """The planner owns Newton dynamics; this component owns the native visual asset."""

    config_cls = SortingRobotConfig
    pure_newton = True
    recording_robot_type = "pro_m6_696_newton"

    def __init__(self, config: SortingRobotConfig | None = None) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("robot")

    def init_robot(self, sim_world: Any) -> None:
        """Reference the robot without starting any simulation resource."""
        from pxr import UsdGeom

        if not sim_world.uses_direct_newton or sim_world.config.newton_solver != "mujoco":
            raise ValueError("Pro-M6 sorting requires direct Newton MuJoCo")
        self.sim_world = sim_world
        self.source_path = resolve_usd_path(self.config.usd_path)
        UsdGeom.Xform.Define(
            sim_world.stage, self.config.prim_path
        ).GetPrim().GetReferences().AddReference(self.source_path)

    def arrange(self) -> None:
        """Cache the authored world pose, retaining the original sorting station placement."""
        from real2sim2real.utils.asset_utils import read_prim_world_pose

        pose = read_prim_world_pose(self.sim_world.stage, self.config.prim_path)
        self.poses["pro_m6"] = pose
        self.poses_realtime["pro_m6"] = pose

    def initialize_cameras(self) -> None:
        """Mount robot cameras on the actual simulated link hierarchy."""
        self.sensors.update(create_cameras(self.sim_world.stage, self.config.cameras))

    def add_newton_robot(self, builder: Any, *, scale: float = 1.0) -> None:
        """Import native geometry and explicitly restore the source Robotiq mimic relations."""
        import newton
        from pxr import Usd

        if scale != 1:
            raise ValueError("Pro-M6 sorting uses SI units")
        self.source_stage = Usd.Stage.Open(self.source_path)
        self.import_map = builder.add_usd(
            self.source_stage,
            floating=False,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            load_visual_shapes=False,
        )
        self.source_root = str(self.source_stage.GetDefaultPrim().GetPath())
        self.arm_indices, self.gripper_indices = {}, {}
        self.mimic = {}
        imported_mimic_pairs = set(
            zip(builder.constraint_mimic_joint0, builder.constraint_mimic_joint1, strict=True)
        )
        for side, suffix in (("left", "L"), ("right", "R")):
            self.arm_indices[side] = [
                builder.joint_q_start[
                    next(
                        i
                        for i, label in enumerate(builder.joint_label)
                        if label.endswith(f"/Joint{j}_{suffix}")
                    )
                ]
                for j in range(1, 8)
            ]
            master_label = f"/robotiq_{side}/joints/finger_joint"
            master = next(
                i for i, label in enumerate(builder.joint_label) if label.endswith(master_label)
            )
            self.gripper_indices[side] = builder.joint_q_start[master]
            self.mimic[side] = []
            for i, label in enumerate(builder.joint_label):
                if f"/robotiq_{side}/joints/" not in label:
                    continue
                prim = self.source_stage.GetPrimAtPath(label)
                gearing = prim.GetAttribute("physxMimicJoint:rotX:gearing")
                # USD relation: follower + gearing * reference + offset = 0.
                ratio = -float(gearing.Get()) if gearing and gearing.HasAuthoredValue() else 1.0
                self.mimic[side].append((builder.joint_q_start[i], ratio))
                if (
                    gearing
                    and gearing.HasAuthoredValue()
                    and (i, master) not in imported_mimic_pairs
                ):
                    builder.add_constraint_mimic(
                        joint0=i,
                        joint1=master,
                        coef0=0.0,
                        coef1=ratio,
                        label=f"sorting_mimic_{side}_{i}",
                    )
            for index, value in zip(self.arm_indices[side], READY[side], strict=True):
                builder.joint_q[index] = value
        for pairs in self.mimic.values():
            for index, ratio in pairs:
                builder.joint_q[index] = self.config.initial_gripper_rad * ratio
        for i, label in enumerate(builder.joint_label):
            if sum(builder.joint_dof_dim[i]) == 0:
                continue
            dof, q = builder.joint_qd_start[i], builder.joint_q_start[i]
            arm = "/Physics/Joint" in label
            driven = arm or label.endswith("/joints/finger_joint")
            builder.joint_target_ke[dof] = (
                (self.config.arm_stiffness if arm else self.config.finger_stiffness)
                if driven
                else 0.0
            )
            builder.joint_target_kd[dof] = (
                (self.config.arm_damping if arm else self.config.finger_damping) if driven else 0.0
            )
            builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
            builder.joint_target_q[dof] = builder.joint_q[q]
        self.joint_names = [
            f"{side}_joint{j}_rad" for side in ("left", "right") for j in range(1, 8)
        ] + ["left_gripper_rad", "right_gripper_rad"]

    @property
    def recording_joint_names(self) -> list[str]:
        """Return the two arm chains and the two actuated gripper coordinates."""
        return self.joint_names

    def adjust(self) -> None:
        """Retain the authored station placement."""
