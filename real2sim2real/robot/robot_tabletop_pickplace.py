"""Franka Panda standing on the table (PhysX), driven by joint position targets.

Migrated from Real2Sim2Real ``FrankaRobot`` (``SingleManipulator`` + parallel gripper). The USD's
own drives are kept (arm 400 per degree / 80, 87 / 12 N·m caps; finger joint 1 400 N/m / 80,
7.2 N cap, finger joint 2 follows by a PhysX mimic joint). The base is placed on the measured
table top at ``base_offset_m`` (table frame) and, like the source ``snap_base_to_support``, its
footprint is shifted back onto the table when it would overhang the edge.

Sections: 1. configuration and pure helpers, 2. robot backend.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import LoggerMixin
from real2sim2real.robot.base import BaseRobot
from real2sim2real.utils.asset_utils import prim_aabb, resolve_usd_path
from real2sim2real.utils.cameras import validate_camera_list
from real2sim2real.utils.tabletop_pickplace_utils import (
    SensorCameraConfig,
    create_sensor_cameras,
    set_prim_pose,
    table_point,
    to_numpy,
    yaw_quat,
)

ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
LINK_CATEGORIES = {
    "panda_link0": "base",
    **{f"panda_link{i}": "arm" for i in range(1, 8)},
    "panda_hand": "hand",
    "panda_leftfinger": "finger_left",
    "panda_rightfinger": "finger_right",
}

# ================================================================================================
# 1. Configuration and pure helpers
# ================================================================================================


def footprint_shift(foot_lo: Any, foot_hi: Any, table_lo: Any, table_hi: Any) -> np.ndarray:
    """XY shift that moves a base footprint inside the table bounds (zero when it fits)."""
    foot_lo, foot_hi = np.asarray(foot_lo, float)[:2], np.asarray(foot_hi, float)[:2]
    table_lo, table_hi = np.asarray(table_lo, float)[:2], np.asarray(table_hi, float)[:2]
    if np.any(foot_hi - foot_lo > table_hi - table_lo):
        raise ValueError("Robot base footprint is larger than the table")
    return np.maximum(table_lo - foot_lo, 0.0) + np.minimum(table_hi - foot_hi, 0.0)


class FrankaTabletopRobotConfig(BaseConfig):
    """``[robot]`` of the tabletop tasks."""

    backend: str = "franka_tabletop"
    usd_path: str = "assets/TabletopPickplace/Robot/FrankaPanda/franka.usd"
    prim_path: str = "/World/robots/Franka"
    base_offset_m: tuple[FiniteFloat, FiniteFloat] = Field(
        default=(-0.35, 0.0), description="table-frame XY of the base from the table centre"
    )
    yaw_deg: FiniteFloat = 0.0
    keep_base_on_table: bool = Field(
        default=True, description="shift the base footprint inside the table (source snap)"
    )
    initial_joint_positions: tuple[FiniteFloat, ...] = Field(
        # The USD drive targets of the Isaac Franka asset (its ready pose), fingers open.
        default=(0.012, -0.570, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04),
        description="rad for the 7 arm joints, m for the two fingers",
    )
    gripper_open_m: FiniteFloat = Field(default=0.04, gt=0, le=0.04)
    gripper_closed_m: FiniteFloat = Field(
        default=0.0, ge=0, description="per-finger target; below the object half width"
    )
    cameras: list[SensorCameraConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_robot(self) -> Self:
        if len(self.initial_joint_positions) != 9:
            raise ValueError("initial_joint_positions needs 7 arm + 2 finger values")
        if self.gripper_closed_m >= self.gripper_open_m:
            raise ValueError("gripper_closed_m must be below gripper_open_m")
        validate_camera_list(list(self.cameras))
        return self


# ================================================================================================
# 2. Robot backend
# ================================================================================================


class FrankaTabletopRobot(BaseRobot, LoggerMixin):
    """Franka Panda with the Default gripper; tensor views after ``sim_world.reset``."""

    config_cls = FrankaTabletopRobotConfig
    recording_robot_type = "franka_panda"
    recording_joint_names = [*ARM_JOINTS, *FINGER_JOINTS]
    action_names = [*ARM_JOINTS, FINGER_JOINTS[0]]

    def __init__(self, config: FrankaTabletopRobotConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config: FrankaTabletopRobotConfig = config or self.config_cls.from_settings("robot")
        self.cameras = list(self.config.cameras)
        self.capture_prims: dict[str, str] = {}
        self.support: dict[str, Any] = {}
        self.base_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.articulation: Any = None
        self._links: Any = None
        self._arm_idx: list[int] = []
        self._finger_idx: list[int] = []
        self.targets = np.asarray(self.config.initial_joint_positions[:8], dtype=float)

    # --- lifecycle --------------------------------------------------------------------------------
    def set_support(
        self, table_top_z: float, table_bounds: Any, table_center: Any, table_yaw_deg: float
    ) -> None:
        """Table measured by the scene (called in ``before_place_robot_hook``)."""
        self.support = {
            "table_top_z": float(table_top_z),
            "table_bounds_m": np.asarray(table_bounds, float),
            "table_center": np.asarray(table_center, float),
            "table_yaw_deg": float(table_yaw_deg),
        }

    def init_robot(self, sim_world: Any) -> None:
        """Reference the Franka USD; cameras mount below their links."""
        from pxr import UsdGeom

        if sim_world.physics_engine != "physx":
            raise ValueError("franka_tabletop requires physics_engine='physx'")
        self.sim_world = sim_world
        stage = sim_world.stage
        prim = UsdGeom.Xform.Define(stage, self.config.prim_path).GetPrim()
        if not prim.GetReferences().AddReference(resolve_usd_path(self.config.usd_path)):
            raise RuntimeError(f"Reference failed: {self.config.usd_path}")
        for link in LINK_CATEGORIES:
            if not stage.GetPrimAtPath(f"{self.config.prim_path}/{link}"):
                raise RuntimeError(f"Franka USD lacks link {link}")
        self.capture_prims = create_sensor_cameras(stage, self.cameras)
        self.sensors.update(self.capture_prims)

    def arrange(self) -> None:
        """Put the base on the table top; shift its footprint inside the table if needed."""
        if not self.support:
            raise RuntimeError("Call set_support() with the measured table before arrange()")
        support = self.support
        xy = table_point(
            support["table_center"], support["table_yaw_deg"], self.config.base_offset_m
        )
        yaw = support["table_yaw_deg"] + self.config.yaw_deg
        position = np.array([*xy, support["table_top_z"]])
        quat = yaw_quat(yaw)
        self._set_root(position, quat)
        stage = self.sim_world.stage
        lo, hi = prim_aabb(stage, f"{self.config.prim_path}/panda_link0", gprims_only=True)
        position[2] += support["table_top_z"] - lo[2]
        shift = np.zeros(2)
        if self.config.keep_base_on_table:
            table_lo, table_hi = support["table_bounds_m"]
            shift = footprint_shift(lo, hi, table_lo, table_hi)
            position[:2] += shift
        self._set_root(position, quat)
        self.base_pose = (position, quat)
        self.poses["base"] = (position.copy(), quat.copy())
        self.state["base_shift_m"] = shift.tolist()
        self.log.info("机器人摆放", base=position.round(4).tolist(), shift=shift.round(4).tolist())

    def initialize_runtime(self) -> None:
        """Articulation / link views; write the initial joint state (after ``reset``)."""
        from isaacsim.core.experimental.prims import Articulation, RigidPrim

        self.articulation = Articulation(self.config.prim_path)
        names = list(self.articulation.dof_names)
        missing = [j for j in (*ARM_JOINTS, *FINGER_JOINTS) if j not in names]
        if missing:
            raise RuntimeError(f"Franka articulation lacks joints {missing}: {names}")
        self._arm_idx = [names.index(j) for j in ARM_JOINTS]
        self._finger_idx = [names.index(j) for j in FINGER_JOINTS]
        q = np.zeros(len(names))
        q[self._arm_idx + self._finger_idx] = self.config.initial_joint_positions
        self.articulation.set_dof_positions(q.reshape(1, -1))
        self.articulation.set_dof_velocities(np.zeros((1, len(names))))
        self.articulation.set_dof_position_targets(q.reshape(1, -1))
        self.articulation.set_dof_velocity_targets(np.zeros((1, len(names))))
        self.targets = np.asarray(self.config.initial_joint_positions[:8], dtype=float)
        root = self.config.prim_path
        links = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
        self._links = RigidPrim([f"{root}/{link}" for link in links])
        self.state["dof_names"] = names

    # --- control and sensing ----------------------------------------------------------------------
    def command(self, arm_q: Any, finger_m: float, arm_qd: Any | None = None) -> None:
        """Joint position targets (rad) with optional velocity feed-forward; one finger target."""
        arm_q = np.asarray(arm_q, float).reshape(7)
        qd = np.zeros(7) if arm_qd is None else np.asarray(arm_qd, float).reshape(7)
        indices = self._arm_idx + self._finger_idx[:1]
        self.articulation.set_dof_position_targets(
            np.r_[arm_q, float(finger_m)].reshape(1, 8), dof_indices=indices
        )
        self.articulation.set_dof_velocity_targets(
            np.r_[qd, 0.0].reshape(1, 8), dof_indices=indices
        )
        self.targets = np.r_[arm_q, float(finger_m)]

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Measured positions and velocities of the 7 arm joints + 2 fingers."""
        idx = self._arm_idx + self._finger_idx
        q = to_numpy(self.articulation.get_dof_positions())[0][idx]
        qd = to_numpy(self.articulation.get_dof_velocities())[0][idx]
        return q.astype(float), qd.astype(float)

    def link_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """World positions (3, 3) and wxyz orientations (3, 4) of hand, left and right finger."""
        p, q = self._links.get_world_poses()
        return to_numpy(p).astype(float), to_numpy(q).astype(float)

    def contact_bodies(self) -> list[str]:
        return [f"{self.config.prim_path}/{link}" for link in LINK_CATEGORIES]

    def contact_category(self, path: str) -> str | None:
        root = self.config.prim_path + "/"
        if not path.startswith(root):
            return None
        return LINK_CATEGORIES.get(path[len(root) :].split("/", 1)[0], "arm")

    def _set_root(self, position: np.ndarray, quat: np.ndarray) -> None:
        set_prim_pose(self.sim_world.stage, self.config.prim_path, position, quat)
