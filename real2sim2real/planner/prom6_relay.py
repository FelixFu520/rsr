# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pro-M6 双臂分拣中继:柔性包裹 入带 → 平台 → 出带.

刚体(臂/夹爪/工位)仍由 PhysX Articulation 驱动.袋装包裹走 Newton VBD
体积软体,靠夹爪摩擦接触承重,不再使用掩码 AutoDeformableAttachment.
松爪后出带以 kinematic box 的 -X 速度靠摩擦送走.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, auto
from pathlib import Path

import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import (
    Articulation,
    DeformablePrim,
    RigidPrim,
    XformPrim,
)
from isaacsim.core.simulation_manager import (
    PhysicsScene,
    SimulationManager,
)
from omni.physx.scripts import deformableUtils
from omni.physx.scripts.ifaces import (
    get_physx_attachment_private_interface,
)
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from real2sim2real.core.logger import get_logger
from real2sim2real.planner.newton_sorting import NewtonSortingWorld

_LOGGER = get_logger("ProM6Relay")


def _log(*parts: object) -> None:
    """把原脚本的 ``print`` 转成一条 structlog 记录.

    Args:
        *parts: 原 ``print`` 的位置参数,按空格拼成一行.

    Returns:
        None.
    """
    _LOGGER.info(" ".join(str(part) for part in parts))


# =============================================================================
# Scene, physics, grasp, and motion configuration
# =============================================================================

ROBOT_PATH = "/World/Pro_m6_696_dynamics_fixed"
CLOTHING_ROOT_PATH = "/World/ImportedBaggedClothing"
CLOTHING_SURFACE_PATH = "/World/ImportedBaggedClothing/Asset/visual/visual"
CLOTHING_VOLUME_PROXY_PATH = "/World/ImportedBaggedClothing/Asset/visual/visual_proxy"
# SUPPORT_PATH = "/World/platform/Empty/Plane/Plane_002"
SUPPORT_PATH = "/World/platform/Empty/Plane/Plane_002"
BELT_PATH = "/World/ConveyorBelt_A06/Belt"
LEFT_GRIPPER_PATH = f"{ROBOT_PATH}/robotiq_left"
RIGHT_GRIPPER_PATH = f"{ROBOT_PATH}/robotiq_right"

RUNTIME_ROOT_PATH = "/World/ProM6_LeftClothingDragRuntime"
ANCHOR_PATH = f"{RUNTIME_ROOT_PATH}/ClothingGraspFrame"
ATTACHMENT_SCOPE_PATH = f"{RUNTIME_ROOT_PATH}/Attachments"
ATTACHMENT_PATH = f"{ATTACHMENT_SCOPE_PATH}/Surface"

REPLAY_CACHE_SCHEMA_VERSION = 1
try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()
REPLAY_CACHE_DIRECTORY = Path(
    os.environ.get(
        "PRO_M6_REPLAY_CACHE_DIRECTORY",
        str(_SCRIPT_DIR / "pro_m6_replay_cache"),
    )
)
REPLAY_CACHE_FILE_PREFIX = "pro_m6_clothing_drag_"

PHYSICS_DEVICE = "cuda:0"
GPU_COLLISION_STACK_SIZE_MIB = 192
NOMINAL_PHYSICS_HZ = 60.0

GRASP_EDGE_INSET_M = 0.022
LEFT_GRASP_FROM_CENTER_FRAC = 0.20  # 左抓 X 从袋心往 -X 偏,咬袋体而不是 +X 薄边
LEFT_GRASP_Y_INSET_M = 0.045  # 从袋 +Y 边往里咬实体;贴边会推走,再往里常是空腔
LEFT_GRASP_BITE_M = 0.008  # 相对 AABB 顶再往下咬,合爪时袋会瘪
LEFT_MIN_FINGER_CONTACTS = 40  # 入带袋更鼓,接触峰值约 50;80 会过压把袋捅穿
LEFT_LIFT_MAX_WORLD_CONTACTS = 10  # 提起后袋身仍大量贴入带,说明只掐到边
LEFT_GRASP_DIP_STEP_M = 0.005
LEFT_GRASP_DIP_MAX_M = 0.020
RIGHT_GRASP_EDGE_INSET_M = 0.035  # 右爪从 -X 边再往里探,避开变形后的空边
RIGHT_GRASP_FROM_CENTER_FRAC = 0.20  # 从袋心往 -X 偏半宽的这一比例,咬实体而不是薄边
RIGHT_GRASP_BITE_M = 0.008  # 相对当前 AABB 顶再往下咬,合爪时袋会瘪
RIGHT_MIN_FINGER_CONTACTS = 80  # 右提之前 Newton 指垫接触数;不够就继续下压
RIGHT_GRASP_DIP_STEP_M = 0.005
RIGHT_GRASP_DIP_MAX_M = 0.020
GRIPPER_SURFACE_OFFSET_M = -0.014
GRASP_MASK_SIZE_M = np.array([0.080, 0.080, 0.030], dtype=np.float64)
PREGRASP_CLEARANCE_M = 0.080
LOW_LIFT_HEIGHT_M = 0.28
PLACEMENT_CLEARANCE_M = 0.002
LEFT_LOWER_DISTANCE_OFFSET_M = 0.025  # Meters: positive lowers farther; negative lowers less.
LEFT_RETRACT_HEIGHT_M = 0.28  # 松爪后先抬高再回 ready,避免关节插值把袋带走
LEFT_RETRACT_HOLD_S = 1.2  # 张开后降摩擦再停一拍,让袋从指垫脱落
LEFT_RETURN_CLEAR_X_OFFSET_M = 0.40  # 只往 +X 挪开;改 Y 会把袋拖出皮带 Y 范围
LEFT_PAD_RETRACT_M = 0.30  # 松爪时指垫沿 TCP -Z 瞬移抽离,避免 VBD 扫掠把袋拉走
PAD_RETRACT_HOLD_FRAMES = 6
LEFT_PAD_CLEAR_GAP_M = 0.04  # 指垫距袋 AABB 超过此值即视为已脱离,不再等接触归零

LEFT_READY_JOINTS = np.array(
    # [-0.05, 0.65, 0.65, -1.25, -0.90, 0.30, -0.35],
    [0.10, 0.10, 0.10, -1.50, 0.0, 0.0, -1.30],
    dtype=np.float64,
)
# Pro-m6 left/right mirror convention negates Joint1, Joint3, and Joint7.
RIGHT_READY_JOINTS = np.array(
    # [0.05, 0.65, -0.65, -1.25, -0.90, 0.30, 0.35],
    [-0.10, 0.10, -0.10, -1.50, 0.0, 0.0, 1.30],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TimingConfig:
    """User-editable durations and safety timeouts, all expressed in seconds."""

    settle_seconds: float = 1.0
    waiting_seconds: float = 1.0
    left_move_pregrasp_seconds: float = 2.5
    left_approach_seconds: float = 2.5
    left_close_seconds: float = 2.5
    left_grip_settle_seconds: float = 1.5
    left_lift_seconds: float = 0.5
    left_drag_seconds: float = 6.0
    left_lower_seconds: float = 2.5
    left_place_settle_seconds: float = 2.0
    left_open_seconds: float = 3.0
    left_return_ready_seconds: float = 1.0
    right_move_pregrasp_seconds: float = 2.5
    right_approach_seconds: float = 2.5
    right_close_seconds: float = 2.5
    right_grip_settle_seconds: float = 1.5
    right_lift_seconds: float = 0.5
    right_release_seconds: float = 3.0
    right_return_ready_seconds: float = 2.5
    left_grip_stable_seconds: float = 1.0 / 6.0
    right_grip_stable_seconds: float = 1.0 / 6.0
    center_stable_seconds: float = 0.5
    place_stable_seconds: float = 0.5
    belt_entry_stable_seconds: float = 0.5
    final_verify_stable_seconds: float = 0.5
    ik_search_timeout_seconds: float = 30.0
    center_correction_timeout_seconds: float = 12.0
    placement_settle_timeout_seconds: float = 8.0
    handoff_prepare_timeout_seconds: float = 5.0
    right_drag_timeout_seconds: float = 30.0
    final_verify_timeout_seconds: float = 20.0

    def validate(self) -> TimingConfig:
        timeout_fields = {
            "ik_search_timeout_seconds",
            "center_correction_timeout_seconds",
            "placement_settle_timeout_seconds",
            "handoff_prepare_timeout_seconds",
            "right_drag_timeout_seconds",
            "final_verify_timeout_seconds",
        }
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"TimingConfig.{name} must be finite, got {value!r}")
            if name in timeout_fields and value <= 0.0:
                raise ValueError(f"TimingConfig.{name} must be > 0, got {value!r}")
            if name not in timeout_fields and value < 0.0:
                raise ValueError(f"TimingConfig.{name} must be >= 0, got {value!r}")
        return self


# Edit this one object to tune the task timing from the Isaac Sim Script Editor.
TASK_TIMING = TimingConfig(
    settle_seconds=2.0,
    waiting_seconds=1.0,
    left_move_pregrasp_seconds=1.0,
    left_approach_seconds=1.0,
    left_close_seconds=1.0,
    left_grip_settle_seconds=1.0,
    left_lift_seconds=3.0,
    left_drag_seconds=3.0,
    left_lower_seconds=1.0,
    left_place_settle_seconds=1.0,
    left_open_seconds=1.0,
    left_return_ready_seconds=0.8,
    right_move_pregrasp_seconds=1.0,
    right_approach_seconds=1.0,
    right_close_seconds=1.0,
    right_grip_settle_seconds=1.0,
    right_lift_seconds=3.0,
    right_release_seconds=1.0,
    right_return_ready_seconds=0.8,
    left_grip_stable_seconds=1.0 / 6.0,
    right_grip_stable_seconds=1.0 / 6.0,
    center_stable_seconds=0.5,
    place_stable_seconds=0.5,
    belt_entry_stable_seconds=0.5,
    final_verify_stable_seconds=0.5,
    placement_settle_timeout_seconds=14.0,
    final_verify_timeout_seconds=20.0,
)

LEFT_MAX_JOINT_SPEED_RAD_S = 0.9
RIGHT_MAX_JOINT_SPEED_RAD_S = 0.9

RIGHT_LIFT_HEIGHT_M = 0.28  # Independent world-Z lift; finite >= 0.
RIGHT_DRAG_LOWER_DISTANCE_M = 0.02  # 短拖时略降,右臂 IK 比悬空 28cm 拖更稳
RIGHT_DRAG_SPEED_M_S = 0.140
RIGHT_MAX_DRAG_DISTANCE_M = 0.56
BELT_ENTRY_INSET_M = 0.010
BELT_TAIL_INSET_M = 0.050  # 袋 +X 尾须进入出带这么深才松爪

RELEASE_CONVEYOR_SPEED_MPS = 0.5
RELEASE_CONVEYOR_DIRECTION_XY = np.array([-1.0, 0.0], dtype=np.float64)
RELEASE_CONVEYOR_STOP_XY_REGION: dict[str, float] | None = None
RELEASE_CONVEYOR_STOP_ON_ARRIVAL = False
RELEASE_CONVEYOR_ZERO_XY_ON_EXIT = False

GRIPPER_OPEN_POSITION_RAD = 0.0
GRIPPER_CLOSED_POSITION_RAD = 0.8
GRIPPER_POSITION_TOLERANCE_RAD = 0.050

CENTER_CORRECTION_STEP_M = 0.001
CENTER_RELEASE_TOLERANCE_M = 0.050
MAX_CENTER_CORRECTION_M = 0.500

TCP_APPROACH_POSITION_TOLERANCE_M = 0.003
TCP_APPROACH_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
TCP_GRASP_POSITION_TOLERANCE_M = 0.004
TCP_GRASP_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
TCP_LIFT_POSITION_TOLERANCE_M = 0.025
TCP_LIFT_ORIENTATION_TOLERANCE_RAD = math.radians(20.0)
DRAG_ENDPOINT_TOLERANCE_M = 0.010
PROGRESS_LOG_INTERVAL_FRAMES = 60
MAX_DEFORMABLE_SPAN_M = 1.5

IK_RANDOM_SEED = 696
IK_RANDOM_SEED_COUNT = 40
IK_CANDIDATE_COUNT = IK_RANDOM_SEED_COUNT + 2
IK_MAX_ITERATIONS = 900
IK_DAMPING = 0.035
ONLINE_DLS_DAMPING = 0.06
IK_MAX_JOINT_STEP_RAD = 0.035
# Joint4 相对 Joint2 在世界 XY 上至少离开躯干这么远,才算肘外展.
IK_ELBOW_OUTBOARD_MIN_M = 0.12
IK_ELBOW_IN_PENALTY = 25.0
IK_JOINT1_WRAP_SCORE_WEIGHT = 0.4
IK_POSTURE_SCORE_WEIGHT = 0.08
_SHOULDER_JOINT_INDEX = 1  # Joint2:上臂根,用来量肘外展
_ELBOW_JOINT_INDEX = 3  # Joint4:肘
LEFT_ELBOW_OUTWARD_XY = np.array([1.0, 0.0], dtype=np.float64)  # 左臂外侧:世界 +X
RIGHT_ELBOW_OUTWARD_XY = np.array([-1.0, 0.0], dtype=np.float64)  # 右臂外侧:世界 -X
IK_INNER_POSITION_TOLERANCE_M = 2.0e-5
IK_INNER_ORIENTATION_TOLERANCE_RAD = 2.0e-4
IK_FINAL_POSITION_TOLERANCE_M = 0.002
IK_FINAL_ORIENTATION_TOLERANCE_RAD = math.radians(1.0)
IK_MODEL_POSITION_TOLERANCE_M = 0.001
IK_MODEL_ORIENTATION_TOLERANCE_RAD = math.radians(0.5)
IK_MODEL_JACOBIAN_TOLERANCE = 5.0e-3
IK_JOINT_LIMIT_MARGIN_RAD = 0.0
IK_PREFERRED_SEED = np.array(
    [-0.09, 1.367, 0.861, -2.069, -2.147, 0.482, -0.761],
    dtype=np.float64,
)
RIGHT_IK_PREFERRED_SEED = IK_PREFERRED_SEED * np.array(
    [-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0],
    dtype=np.float64,
)
# Joint3 更正,让左肘先偏到入带外侧,再让 42 种子搜索收敛.
LEFT_IK_ELBOW_OUT_SEED = np.array(
    [0.20, 1.15, 1.25, -1.75, -1.60, 0.40, -0.80],
    dtype=np.float64,
)
RIGHT_IK_ELBOW_OUT_SEED = LEFT_IK_ELBOW_OUT_SEED * np.array(
    [-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0],
    dtype=np.float64,
)

CALLBACK_GLOBAL_NAME = "_pro_m6_left_clothing_drag_callback_id"
STOP_CALLBACK_GLOBAL_NAME = "_pro_m6_left_clothing_drag_stop_callback_id"
CONTROLLER_GLOBAL_NAME = "_pro_m6_left_clothing_drag_controller"
OWN_CALLBACK_GLOBAL_NAMES = (
    CALLBACK_GLOBAL_NAME,
    STOP_CALLBACK_GLOBAL_NAME,
)


def apply_settings(
    *,
    robot_path: str = "",
    clothing_root_path: str = "",
    clothing_surface_path: str = "",
    clothing_volume_proxy_path: str = "",
    support_path: str = "",
    belt_path: str = "",
    left_gripper_path: str = "",
    right_gripper_path: str = "",
    runtime_root_path: str = "",
    replay_cache_dir: str = "",
    physics_device: str = "",
    physics_hz: float = 0.0,
) -> None:
    """用 ``[planner]`` 配置改写本模块的场景常量;空值保留原脚本默认.

    原脚本把 prim 路径写死在模块级,状态机各处直接引用.迁移保留这一写法,
    只在装配阶段由 ``SortingPlanner`` 调用本函数改写一次,避免逐处改签名.

    Args:
        robot_path: 机器人根 prim.
        clothing_root_path: 柔性包裹根 prim.
        clothing_surface_path: 柔性面片 Mesh(须为 Surface Deformable).
        clothing_volume_proxy_path: 体积代理 prim;必须处于 inactive.
        support_path: 平台支撑面 prim.
        belt_path: 出带皮带 prim.
        left_gripper_path: 左夹爪 prim;空则取 ``<robot>/robotiq_left``.
        right_gripper_path: 右夹爪 prim;空则取 ``<robot>/robotiq_right``.
        runtime_root_path: 运行期临时 prim 根(锚点 / 附着).
        replay_cache_dir: 成功回放 npz 目录.
        physics_device: PhysX 设备,如 ``cuda:0``.
        physics_hz: 物理频率;与 ``[sim_world].physics_dt`` 保持一致.

    Returns:
        None.
    """
    global ROBOT_PATH, CLOTHING_ROOT_PATH, CLOTHING_SURFACE_PATH
    global CLOTHING_VOLUME_PROXY_PATH, SUPPORT_PATH, BELT_PATH
    global LEFT_GRIPPER_PATH, RIGHT_GRIPPER_PATH
    global RUNTIME_ROOT_PATH, ANCHOR_PATH, ATTACHMENT_SCOPE_PATH, ATTACHMENT_PATH
    global REPLAY_CACHE_DIRECTORY, PHYSICS_DEVICE, NOMINAL_PHYSICS_HZ

    if robot_path:
        ROBOT_PATH = robot_path
    if clothing_root_path:
        CLOTHING_ROOT_PATH = clothing_root_path
    if clothing_surface_path:
        CLOTHING_SURFACE_PATH = clothing_surface_path
    if clothing_volume_proxy_path:
        CLOTHING_VOLUME_PROXY_PATH = clothing_volume_proxy_path
    if support_path:
        SUPPORT_PATH = support_path
    if belt_path:
        BELT_PATH = belt_path
    LEFT_GRIPPER_PATH = left_gripper_path or f"{ROBOT_PATH}/robotiq_left"
    RIGHT_GRIPPER_PATH = right_gripper_path or f"{ROBOT_PATH}/robotiq_right"
    if runtime_root_path:
        RUNTIME_ROOT_PATH = runtime_root_path
    ANCHOR_PATH = f"{RUNTIME_ROOT_PATH}/ClothingGraspFrame"
    ATTACHMENT_SCOPE_PATH = f"{RUNTIME_ROOT_PATH}/Attachments"
    ATTACHMENT_PATH = f"{ATTACHMENT_SCOPE_PATH}/Surface"
    if replay_cache_dir:
        REPLAY_CACHE_DIRECTORY = Path(replay_cache_dir)
    if physics_device:
        PHYSICS_DEVICE = physics_device
    if physics_hz > 0.0:
        NOMINAL_PHYSICS_HZ = float(physics_hz)


class Phase(Enum):
    SETTLE = auto()
    WAITING = auto()
    SEARCH_PREGRASP_IK = auto()
    MOVE_TO_PREGRASP = auto()
    VERTICAL_APPROACH = auto()
    CLOSE_GRIPPER = auto()
    GRIP_SETTLE = auto()
    DRAG = auto()
    CENTER_CORRECT = auto()
    LOWER = auto()
    PLACE_SETTLE = auto()
    LEFT_OPEN = auto()
    LEFT_RETURN_READY = auto()
    PREPARE_RIGHT_HANDOFF = auto()
    RIGHT_SEARCH_IK = auto()
    RIGHT_MOVE_TO_PREGRASP = auto()
    RIGHT_VERTICAL_APPROACH = auto()
    RIGHT_CLOSE = auto()
    RIGHT_GRIP_SETTLE = auto()
    RIGHT_DRAG_TO_BELT = auto()
    RIGHT_RELEASE = auto()
    RIGHT_RETURN_READY = auto()
    FINAL_VERIFY = auto()
    DONE = auto()
    FAILED = auto()
    LEFT_LIFT = auto()  # Append to preserve existing phase IDs in diagnostics/replay.
    RIGHT_LIFT = auto()


PHASE_LABELS = {
    Phase.SETTLE: "hold both arms ready while the deformable settles",
    Phase.WAITING: "visible delay before left-arm motion",
    Phase.SEARCH_PREGRASP_IK: "search 42 deterministic pregrasp IK branches",
    Phase.MOVE_TO_PREGRASP: "move to top-down pregrasp",
    Phase.VERTICAL_APPROACH: "vertical approach",
    Phase.CLOSE_GRIPPER: "close left gripper",
    Phase.GRIP_SETTLE: "verify grasp pose",
    Phase.LEFT_LIFT: "lift vertically by the configured height before horizontal dragging",
    Phase.DRAG: "horizontal smoothstep drag at the raised height",
    Phase.CENTER_CORRECT: "correct XY at the raised height, then calculate tabletop setdown",
    Phase.LOWER: "lower clothing to tabletop",
    Phase.PLACE_SETTLE: "verify center on the tabletop before handoff",
    Phase.LEFT_OPEN: "open the left gripper, retract off the parcel, then leave it on the platform",
    Phase.LEFT_RETURN_READY: "return the left arm to ready",
    Phase.PREPARE_RIGHT_HANDOFF: "recompute the right grasp from the live Newton AABB",
    Phase.RIGHT_SEARCH_IK: "search 42 deterministic right pregrasp IK branches",
    Phase.RIGHT_MOVE_TO_PREGRASP: "move the right arm to top-down pregrasp",
    Phase.RIGHT_VERTICAL_APPROACH: "right-arm vertical approach",
    Phase.RIGHT_CLOSE: "close the right gripper",
    Phase.RIGHT_GRIP_SETTLE: "verify the right grasp pose",
    Phase.RIGHT_LIFT: "lift the right TCP by RIGHT_LIFT_HEIGHT_M before dragging",
    Phase.RIGHT_DRAG_TO_BELT: (
        "drag along world -X while descending, then release after height and belt stability checks"
    ),
    Phase.RIGHT_RELEASE: "open the right gripper and start Newton belt transport",
    Phase.RIGHT_RETURN_READY: "return the right arm to ready",
    Phase.FINAL_VERIFY: "wait for the released package to exit the belt along -X",
    Phase.DONE: "done",
    Phase.FAILED: "failed",
}


def build_phase_timeouts(timing: TimingConfig) -> dict[Phase, float]:
    """Build duration-aware phase timeouts while preserving current margins."""
    timing.validate()
    return {
        Phase.SETTLE: timing.settle_seconds + 9.0,
        Phase.WAITING: timing.waiting_seconds + 5.0,
        Phase.MOVE_TO_PREGRASP: timing.left_move_pregrasp_seconds + 9.5,
        Phase.VERTICAL_APPROACH: timing.left_approach_seconds + 9.5,
        Phase.CLOSE_GRIPPER: timing.left_close_seconds + 5.0,
        Phase.GRIP_SETTLE: max(
            timing.left_grip_settle_seconds + 7.0,
            timing.left_grip_stable_seconds + 7.0,
        ),
        Phase.DRAG: timing.left_drag_seconds + 12.0,
        Phase.LEFT_LIFT: timing.left_lift_seconds + 22.0,
        Phase.CENTER_CORRECT: max(
            timing.center_correction_timeout_seconds,
            timing.center_stable_seconds + 5.0,
        ),
        Phase.LOWER: timing.left_lower_seconds + 10.0,
        Phase.PLACE_SETTLE: max(
            timing.placement_settle_timeout_seconds,
            timing.left_place_settle_seconds + 6.0,
            timing.place_stable_seconds + 6.0,
        ),
        Phase.LEFT_OPEN: timing.left_open_seconds + 18.0,
        Phase.LEFT_RETURN_READY: timing.left_return_ready_seconds + 16.0,
        Phase.PREPARE_RIGHT_HANDOFF: timing.handoff_prepare_timeout_seconds,
        Phase.RIGHT_MOVE_TO_PREGRASP: timing.right_move_pregrasp_seconds + 9.5,
        Phase.RIGHT_VERTICAL_APPROACH: timing.right_approach_seconds + 9.5,
        Phase.RIGHT_CLOSE: timing.right_close_seconds + 5.0,
        Phase.RIGHT_GRIP_SETTLE: max(
            timing.right_grip_settle_seconds + 7.0,
            timing.right_grip_stable_seconds + 7.0,
        ),
        Phase.RIGHT_LIFT: timing.right_lift_seconds + 16.0,
        Phase.RIGHT_DRAG_TO_BELT: max(
            timing.right_drag_timeout_seconds,
            timing.belt_entry_stable_seconds + 5.0,
        ),
        Phase.RIGHT_RELEASE: timing.right_release_seconds + 5.0,
        Phase.RIGHT_RETURN_READY: timing.right_return_ready_seconds + 8.0,
        Phase.FINAL_VERIFY: max(
            timing.final_verify_timeout_seconds,
            timing.final_verify_stable_seconds + 4.5,
        ),
    }


# =============================================================================
# Pure geometry and numerical helpers
# =============================================================================


def _to_numpy(value) -> np.ndarray:
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _array_string(value, precision: int = 4) -> str:
    return np.array2string(
        np.asarray(value),
        precision=precision,
        suppress_small=True,
    )


def _smoothstep(progress: float) -> float:
    value = float(np.clip(progress, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _timed_smoothstep(elapsed: float, duration: float) -> float:
    """Return completed progress for zero duration, else timed smoothstep."""
    duration = float(duration)
    if duration <= 0.0:
        return 1.0
    return _smoothstep(float(elapsed) / duration)


def _accumulate_stable_time(
    current: float,
    condition: bool,
    dt: float,
) -> float:
    """Accumulate consecutive stable seconds using the actual physics dt."""
    dt = float(dt)
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError(f"Stable-time dt must be finite and non-negative: {dt!r}")
    return float(current) + dt if condition else 0.0


def _stable_time_reached(current: float, required: float) -> bool:
    """Compare durations without adding a spurious frame from float rounding."""
    return float(current) + 1.0e-12 >= float(required)


def _clamp_vector_norm(vector: np.ndarray, maximum_norm: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= maximum_norm or norm < 1.0e-12:
        return value.copy()
    return value * (maximum_norm / norm)


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise RuntimeError(f"Invalid quaternion: {quaternion}")
    return quaternion / norm


def _quaternion_conjugate(value: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(value)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = _normalize_quaternion(left)
    rw, rx, ry, rz = _normalize_quaternion(right)
    return _normalize_quaternion(
        np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )
    )


def _quaternion_rotation_matrix(value: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(value)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _local_offset_from_world_point(
    frame_position: np.ndarray,
    frame_orientation: np.ndarray,
    world_point: np.ndarray,
) -> np.ndarray:
    """Express a world point as a translation in one rigid frame."""
    return _quaternion_rotation_matrix(_quaternion_conjugate(frame_orientation)) @ (
        np.asarray(world_point, dtype=np.float64).reshape(3)
        - np.asarray(frame_position, dtype=np.float64).reshape(3)
    )


def _matrix_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    return _normalize_quaternion(quaternion)


def _pose_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_rotation_matrix(orientation)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def _matrix_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return matrix[:3, 3].copy(), _matrix_quaternion(matrix[:3, :3])


def _pose_error(
    current_position: np.ndarray,
    current_orientation: np.ndarray,
    target_position: np.ndarray,
    target_orientation: np.ndarray,
) -> np.ndarray:
    delta = _quaternion_multiply(
        target_orientation,
        _quaternion_conjugate(current_orientation),
    )
    sign = 1.0 if delta[0] >= 0.0 else -1.0
    return np.concatenate(
        [
            np.asarray(target_position, dtype=np.float64).reshape(3)
            - np.asarray(current_position, dtype=np.float64).reshape(3),
            2.0 * sign * delta[1:],
        ]
    )


def _wrapped_joint_delta(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """最短角位移 ``wrap(target - current)``,落到 ``(-pi, pi]``."""
    delta = np.asarray(target, dtype=np.float64).reshape(7) - np.asarray(
        current, dtype=np.float64
    ).reshape(7)
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def _dls_joint_delta(
    jacobian: np.ndarray,
    error: np.ndarray,
    *,
    damping: float,
    maximum_step: float,
    nullspace_task: np.ndarray | None = None,
    nullspace_gain: float = 0.0,
) -> np.ndarray:
    """Return one finite damped-least-squares joint increment."""
    matrix = np.asarray(jacobian, dtype=np.float64).reshape(6, 7)
    value = np.asarray(error, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(value)):
        raise RuntimeError("DLS inputs must be finite")
    transpose = matrix.T
    damped = matrix @ transpose + float(damping) ** 2 * np.eye(6)
    delta = transpose @ np.linalg.solve(damped, value)
    gain = float(nullspace_gain)
    if nullspace_task is not None and gain > 0.0:
        task = np.asarray(nullspace_task, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(task)):
            raise RuntimeError("DLS nullspace task must be finite")
        pinv = transpose @ np.linalg.solve(damped, np.eye(6))
        delta = delta + gain * ((np.eye(7) - pinv @ matrix) @ task)
    return np.clip(delta, -float(maximum_step), float(maximum_step))


def _elbow_outboard_m(
    kinematics: UsdArmKinematics,
    positions: np.ndarray,
    outward_xy: np.ndarray,
) -> float:
    """肘相对肩在世界 XY 上沿 ``outward_xy`` 的外展距离;正=向外."""
    _, joints, _ = kinematics._forward_chain(positions)
    shoulder = np.asarray(joints[_SHOULDER_JOINT_INDEX][:2], dtype=np.float64)
    elbow = np.asarray(joints[_ELBOW_JOINT_INDEX][:2], dtype=np.float64)
    axis = np.asarray(outward_xy, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-9:
        return 0.0
    return float(np.dot(elbow - shoulder, axis / norm))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _gf_quaternion(value) -> np.ndarray:
    if value is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    imaginary = value.GetImaginary()
    return _normalize_quaternion(
        np.array(
            [
                float(value.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ],
            dtype=np.float64,
        )
    )


def _attribute_pose(position_attr, rotation_attr) -> np.ndarray:
    position_value = position_attr.Get()
    position = (
        np.zeros(3, dtype=np.float64)
        if position_value is None
        else np.asarray(position_value, dtype=np.float64).reshape(3)
    )
    return _pose_matrix(position, _gf_quaternion(rotation_attr.Get()))


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=np.float64).reshape(3)
    unit_axis /= max(float(np.linalg.norm(unit_axis)), 1.0e-12)
    x, y, z = unit_axis
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    one_minus_cosine = 1.0 - cosine
    rotation = np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def _rigid_world_matrix(path: str) -> np.ndarray:
    positions, orientations = RigidPrim(path).get_world_poses()
    position = _to_numpy(positions).reshape(-1, 3)[0].astype(np.float64)
    orientation = _to_numpy(orientations).reshape(-1, 4)[0].astype(np.float64)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation)):
        raise RuntimeError(f"Rigid-body world pose is non-finite: {path}")
    return _pose_matrix(position, orientation)


@dataclass(frozen=True)
class UsdRevoluteJointSpec:
    path: str
    body0_path: str
    body1_path: str
    body0_to_joint: np.ndarray
    body1_to_joint: np.ndarray
    axis: np.ndarray


class UsdArmKinematics:
    """Offline world-frame FK/Jacobian built from seven USD revolute joints."""

    def __init__(
        self,
        joint_specs: list[UsdRevoluteJointSpec],
        root_world: np.ndarray,
        end_adapter: np.ndarray,
        tcp_adapter: np.ndarray,
        joint_signs: np.ndarray | None = None,
    ) -> None:
        if len(joint_specs) != 7:
            raise RuntimeError(f"Expected seven USD revolute joints, got {len(joint_specs)}")
        self.joint_specs = tuple(joint_specs)
        self.root_world = np.asarray(root_world, dtype=np.float64).reshape(4, 4)
        self.end_adapter = np.asarray(end_adapter, dtype=np.float64).reshape(4, 4)
        self.tcp_adapter = np.asarray(tcp_adapter, dtype=np.float64).reshape(4, 4)
        self.joint_signs = (
            np.ones(7, dtype=np.float64)
            if joint_signs is None
            else np.asarray(joint_signs, dtype=np.float64).reshape(7)
        )
        if not np.all(np.isin(self.joint_signs, (-1.0, 1.0))):
            raise RuntimeError(f"USD joint coordinate signs must be +/-1: {self.joint_signs}")

    @staticmethod
    def _joint_axis(joint: UsdPhysics.RevoluteJoint) -> np.ndarray:
        token = str(joint.GetAxisAttr().Get() or "X").upper()
        axes = {
            "X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        if token not in axes:
            raise RuntimeError(f"Unsupported revolute-joint axis token: {token!r}")
        return axes[token]

    @classmethod
    def from_stage(
        cls,
        stage: Usd.Stage,
        joint_paths: list[str],
        current_positions: np.ndarray,
        gripper_base_path: str,
        tcp_adapter: np.ndarray,
        live_tcp_jacobian: np.ndarray,
    ) -> UsdArmKinematics:
        specs: list[UsdRevoluteJointSpec] = []
        for path in joint_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdPhysics.RevoluteJoint):
                raise RuntimeError(f"Arm DOF is not a USD RevoluteJoint: {path}")
            joint = UsdPhysics.RevoluteJoint(prim)
            body0_targets = joint.GetBody0Rel().GetTargets()
            body1_targets = joint.GetBody1Rel().GetTargets()
            if len(body0_targets) != 1 or len(body1_targets) != 1:
                raise RuntimeError(f"Joint {path} must have exactly one body0 and body1 target")
            specs.append(
                UsdRevoluteJointSpec(
                    path=str(path),
                    body0_path=body0_targets[0].pathString,
                    body1_path=body1_targets[0].pathString,
                    body0_to_joint=_attribute_pose(
                        joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()
                    ),
                    body1_to_joint=_attribute_pose(
                        joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()
                    ),
                    axis=cls._joint_axis(joint),
                )
            )
        for previous, current in zip(specs, specs[1:], strict=False):
            if previous.body1_path != current.body0_path:
                raise RuntimeError(
                    "Arm USD joints are not one continuous serial chain: "
                    f"{previous.path} body1={previous.body1_path}, "
                    f"{current.path} body0={current.body0_path}"
                )

        root_world = _rigid_world_matrix(specs[0].body0_path)
        live_tcp_jacobian = np.asarray(live_tcp_jacobian, dtype=np.float64).reshape(6, 7)
        world_body = root_world.copy()
        joint_signs: list[float] = []
        for index, (spec, angle) in enumerate(
            zip(specs, np.asarray(current_positions, dtype=np.float64).reshape(7), strict=False)
        ):
            world_joint = world_body @ spec.body0_to_joint
            nominal_axis = world_joint[:3, :3] @ spec.axis
            live_axis = live_tcp_jacobian[3:, index]
            alignment = float(np.dot(nominal_axis, live_axis))
            if abs(alignment) < 0.95:
                raise RuntimeError(
                    "USD joint axis does not align with the live PhysX "
                    f"Jacobian at {spec.path}: dot={alignment:.6f}"
                )
            sign = 1.0 if alignment >= 0.0 else -1.0
            joint_signs.append(sign)
            world_body = (
                world_joint
                @ _axis_rotation(spec.axis, sign * float(angle))
                @ np.linalg.inv(spec.body1_to_joint)
            )
        predicted_end = world_body
        measured_end = _rigid_world_matrix(specs[-1].body1_path)
        end_position, end_orientation = _matrix_pose(predicted_end)
        measured_position, measured_orientation = _matrix_pose(measured_end)
        end_error = _pose_error(
            end_position,
            end_orientation,
            measured_position,
            measured_orientation,
        )
        if (
            np.linalg.norm(end_error[:3]) > IK_MODEL_POSITION_TOLERANCE_M
            or np.linalg.norm(end_error[3:]) > IK_MODEL_ORIENTATION_TOLERANCE_RAD
        ):
            raise RuntimeError(
                "USD joint-chain FK does not reproduce the live Link7 pose: "
                f"position_error={np.linalg.norm(end_error[:3]):.6f} m, "
                f"orientation_error={math.degrees(np.linalg.norm(end_error[3:])):.4f} deg"
            )

        measured_gripper = _rigid_world_matrix(gripper_base_path)
        end_adapter = np.linalg.inv(predicted_end) @ measured_gripper
        return cls(
            specs,
            root_world,
            end_adapter,
            tcp_adapter,
            np.asarray(joint_signs, dtype=np.float64),
        )

    def _forward_chain(
        self,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        joint_positions = np.asarray(positions, dtype=np.float64).reshape(7)
        world_body = self.root_world.copy()
        world_joint_positions: list[np.ndarray] = []
        world_joint_axes: list[np.ndarray] = []
        for spec, sign, angle in zip(
            self.joint_specs, self.joint_signs, joint_positions, strict=False
        ):
            world_joint = world_body @ spec.body0_to_joint
            world_joint_positions.append(world_joint[:3, 3].copy())
            world_joint_axes.append(sign * (world_joint[:3, :3] @ spec.axis))
            world_body = (
                world_joint
                @ _axis_rotation(spec.axis, sign * float(angle))
                @ np.linalg.inv(spec.body1_to_joint)
            )
        return world_body, world_joint_positions, world_joint_axes

    def forward_and_jacobian(
        self,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        world_end, joint_positions, joint_axes = self._forward_chain(positions)
        world_gripper = world_end @ self.end_adapter
        world_tcp = world_gripper @ self.tcp_adapter
        tcp_position, tcp_orientation = _matrix_pose(world_tcp)
        gripper_position, _ = _matrix_pose(world_gripper)
        jacobian = np.zeros((6, 7), dtype=np.float64)
        for index, (joint_position, joint_axis) in enumerate(
            zip(joint_positions, joint_axes, strict=False)
        ):
            jacobian[:3, index] = np.cross(joint_axis, tcp_position - joint_position)
            jacobian[3:, index] = joint_axis
        return tcp_position, tcp_orientation, jacobian, gripper_position


def _build_ik_candidate_seeds(
    lower: np.ndarray,
    upper: np.ndarray,
    preferred_seed: np.ndarray = IK_PREFERRED_SEED,
    ready_seed: np.ndarray | None = None,
) -> list[np.ndarray]:
    lower = np.asarray(lower, dtype=np.float64).reshape(7)
    upper = np.asarray(upper, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise RuntimeError("42-seed search requires seven finite joint limits")
    if np.any(lower >= upper):
        raise RuntimeError(f"Invalid left-arm joint limits: {lower}, {upper}")
    generator = np.random.default_rng(IK_RANDOM_SEED)
    preferred = np.asarray(preferred_seed, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(preferred)):
        raise RuntimeError(f"IK preferred seed is non-finite: {preferred}")
    second = np.asarray(preferred if ready_seed is None else ready_seed, dtype=np.float64)
    second = second.reshape(7)
    if not np.all(np.isfinite(second)):
        raise RuntimeError(f"IK ready seed is non-finite: {second}")
    seeds = [preferred.copy(), np.clip(second, lower, upper)]
    seeds.extend(generator.uniform(lower, upper) for _ in range(IK_RANDOM_SEED_COUNT))
    if len(seeds) != IK_CANDIDATE_COUNT:
        raise RuntimeError(f"Expected {IK_CANDIDATE_COUNT} IK seeds, generated {len(seeds)}")
    return [np.asarray(seed, dtype=np.float64).copy() for seed in seeds]


def solve_pregrasp_ik_42_seeds(
    kinematics: UsdArmKinematics,
    target_position: np.ndarray,
    target_orientation: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_limit_margin: float,
    preferred_seed: np.ndarray = IK_PREFERRED_SEED,
    search_timeout_seconds: float = 30.0,
    ready_seed: np.ndarray | None = None,
    outward_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Select one safe deterministic pregrasp branch using exactly 42 seeds.

    冗余臂会有肘内折 / 肘外展两支.本搜索用零空间把解钉在偏好姿态附近,
    再按肘外展距离打分,避免选到贴躯干的内折支.
    """
    lower = np.asarray(lower, dtype=np.float64).reshape(7)
    upper = np.asarray(upper, dtype=np.float64).reshape(7)
    safe_lower = lower + float(joint_limit_margin)
    safe_upper = upper - float(joint_limit_margin)
    if np.any(safe_lower >= safe_upper):
        raise RuntimeError("ProM6 joint-limit margin leaves no safe interval for seeded IK")
    preferred_seed = np.asarray(preferred_seed, dtype=np.float64).reshape(7)
    search_timeout_seconds = float(search_timeout_seconds)
    if not math.isfinite(search_timeout_seconds) or search_timeout_seconds <= 0.0:
        raise RuntimeError(
            f"IK search timeout must be finite and positive: {search_timeout_seconds!r}"
        )
    elbow_axis = LEFT_ELBOW_OUTWARD_XY if outward_xy is None else outward_xy
    seeds = _build_ik_candidate_seeds(lower, upper, preferred_seed, ready_seed)
    best_positions: np.ndarray | None = None
    best_score = math.inf
    best_seed_index = -1
    best_position_error = math.inf
    best_orientation_error = math.inf
    best_elbow_outboard = math.nan
    valid_solution_count = 0
    search_started = time.perf_counter()

    for seed_index, candidate_seed in enumerate(seeds):
        positions = candidate_seed.copy()
        for _ in range(IK_MAX_ITERATIONS):
            if time.perf_counter() - search_started > search_timeout_seconds:
                raise RuntimeError(
                    f"42-seed pregrasp IK search exceeded {search_timeout_seconds:.1f} seconds"
                )
            position, orientation, jacobian, _ = kinematics.forward_and_jacobian(positions)
            error = _pose_error(
                position,
                orientation,
                target_position,
                target_orientation,
            )
            if (
                np.linalg.norm(error[:3]) < IK_INNER_POSITION_TOLERANCE_M
                and np.linalg.norm(error[3:]) < IK_INNER_ORIENTATION_TOLERANCE_RAD
            ):
                break
            delta = _dls_joint_delta(
                jacobian,
                error,
                damping=IK_DAMPING,
                maximum_step=IK_MAX_JOINT_STEP_RAD,
            )
            positions = np.clip(
                positions + delta,
                lower,
                upper,
            )

        final_position, final_orientation, _, _ = kinematics.forward_and_jacobian(positions)
        final_error = _pose_error(
            final_position,
            final_orientation,
            target_position,
            target_orientation,
        )
        position_error = float(np.linalg.norm(final_error[:3]))
        orientation_error = float(np.linalg.norm(final_error[3:]))
        inside_safe_limits = bool(
            np.all(positions >= safe_lower) and np.all(positions <= safe_upper)
        )
        meets_final_tolerance = (
            position_error <= IK_FINAL_POSITION_TOLERANCE_M
            and orientation_error <= IK_FINAL_ORIENTATION_TOLERANCE_RAD
        )
        if not inside_safe_limits or not meets_final_tolerance:
            continue
        elbow_outboard = _elbow_outboard_m(kinematics, positions, elbow_axis)
        if elbow_outboard < IK_ELBOW_OUTBOARD_MIN_M:
            continue
        valid_solution_count += 1
        normalized_margin = np.minimum(
            (positions - lower) / np.maximum(upper - lower, 1.0e-9),
            (upper - positions) / np.maximum(upper - lower, 1.0e-9),
        )
        limit_penalty = float(np.sum(np.maximum(0.12 - normalized_margin, 0.0) ** 2))
        posture_penalty = float(np.linalg.norm(_wrapped_joint_delta(preferred_seed, positions)))
        joint1_wrap = abs(float(_wrapped_joint_delta(preferred_seed, positions)[0]))
        elbow_penalty = IK_ELBOW_IN_PENALTY * max(0.0, IK_ELBOW_OUTBOARD_MIN_M - elbow_outboard)
        score = float(
            position_error
            + 0.20 * orientation_error
            + 0.02 * limit_penalty
            + IK_POSTURE_SCORE_WEIGHT * posture_penalty
            + IK_JOINT1_WRAP_SCORE_WEIGHT * joint1_wrap
            + elbow_penalty
        )
        if score < best_score:
            best_positions = positions.copy()
            best_score = score
            best_seed_index = seed_index
            best_position_error = position_error
            best_orientation_error = orientation_error
            best_elbow_outboard = elbow_outboard

    if best_positions is None:
        raise RuntimeError(
            "42-seed pregrasp IK search found no solution satisfying pose "
            "tolerances and the ProM6 joint-limit safety margin"
        )
    return best_positions, {
        "candidate_count": len(seeds),
        "valid_solution_count": valid_solution_count,
        "seed_index": best_seed_index,
        "score": best_score,
        "position_error": best_position_error,
        "orientation_error": best_orientation_error,
        "elbow_outboard_m": best_elbow_outboard,
    }


def get_world_bounds(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    """Return one finite world-axis-aligned bounding box."""
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=False,
    )
    world_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if world_range.IsEmpty():
        raise RuntimeError(f"Prim has no finite world bounds: {prim.GetPath()}")
    minimum = np.asarray(world_range.GetMin(), dtype=np.float64)
    maximum = np.asarray(world_range.GetMax(), dtype=np.float64)
    if (
        minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(maximum < minimum)
    ):
        raise RuntimeError(
            f"Invalid world bounds for {prim.GetPath()}: min={minimum}, max={maximum}"
        )
    return minimum, maximum


def calculate_grasp_geometry(
    clothing_bounds: tuple[np.ndarray, np.ndarray],
    support_bounds: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Calculate a bag-body grasp (near AABB center) and tabletop-center translation."""
    clothing_min, clothing_max = (
        np.asarray(value, dtype=np.float64).reshape(3) for value in clothing_bounds
    )
    support_min, support_max = (
        np.asarray(value, dtype=np.float64).reshape(3) for value in support_bounds
    )
    values = np.concatenate([clothing_min, clothing_max, support_min, support_max])
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Clothing/support bounds contain non-finite values")
    clothing_center = 0.5 * (clothing_min + clothing_max)
    support_center = 0.5 * (support_min + support_max)
    clothing_half_xy = 0.5 * (clothing_max[:2] - clothing_min[:2])
    support_half_xy = 0.5 * (support_max[:2] - support_min[:2])
    if np.any(clothing_half_xy <= 0.0) or np.any(support_half_xy <= 0.0):
        raise RuntimeError(
            "Clothing and tabletop must have positive XY extents: "
            f"clothing_half={clothing_half_xy}, "
            f"support_half={support_half_xy}"
        )
    if np.any(clothing_half_xy + 0.020 > support_half_xy):
        raise RuntimeError(
            "The clothing footprint does not fit on the tabletop center with "
            f"a 20 mm margin: clothing_half={clothing_half_xy}, "
            f"support_half={support_half_xy}"
        )

    inset = GRASP_EDGE_INSET_M
    half_x = max(float(clothing_half_xy[0]), 0.04)
    grasp_x = float(clothing_center[0]) - LEFT_GRASP_FROM_CENTER_FRAC * half_x
    grasp_y = float(clothing_max[1]) - LEFT_GRASP_Y_INSET_M
    grasp_x = min(max(grasp_x, float(clothing_min[0]) + inset), float(clothing_max[0]) - inset)
    grasp_y = min(max(grasp_y, float(clothing_min[1]) + inset), float(clothing_max[1]) - inset)
    attachment_center = np.array(
        [grasp_x, grasp_y, float(clothing_max[2]) - LEFT_GRASP_BITE_M],
        dtype=np.float64,
    )
    grasp_point = attachment_center + np.array(
        [0.0, 0.0, GRIPPER_SURFACE_OFFSET_M], dtype=np.float64
    )
    pregrasp_point = grasp_point + np.array([0.0, 0.0, PREGRASP_CLEARANCE_M], dtype=np.float64)
    center_delta_xy = support_center[:2] - clothing_center[:2]
    return {
        "clothing_center": clothing_center,
        "support_center": support_center,
        "attachment_center": attachment_center,
        "grasp_point": grasp_point,
        "pregrasp_point": pregrasp_point,
        "center_delta_xy": center_delta_xy,
    }


def _xy_center_error(center: np.ndarray, target: np.ndarray) -> float:
    """Return planar distance between two world-space centers."""
    center_xy = np.asarray(center, dtype=np.float64).reshape(3)[:2]
    target_xy = np.asarray(target, dtype=np.float64).reshape(3)[:2]
    return float(np.linalg.norm(center_xy - target_xy))


def _center_is_within_tolerance(
    center: np.ndarray,
    target: np.ndarray,
    tolerance: float = CENTER_RELEASE_TOLERANCE_M,
) -> bool:
    return _xy_center_error(center, target) <= float(tolerance)


def _calculate_frozen_drag_target(
    grasp_point: np.ndarray,
    node_center: np.ndarray,
    support_center: np.ndarray,
) -> np.ndarray:
    """Freeze the live center-to-center XY translation plus the low lift."""
    grasp = np.asarray(grasp_point, dtype=np.float64).reshape(3)
    center = np.asarray(node_center, dtype=np.float64).reshape(3)
    support = np.asarray(support_center, dtype=np.float64).reshape(3)
    return grasp + np.array(
        [
            support[0] - center[0],
            support[1] - center[1],
            LOW_LIFT_HEIGHT_M,
        ],
        dtype=np.float64,
    )


def calculate_shared_attachment_grasp(
    anchor_world_position: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate right grasp targets from the existing shared anchor."""
    attachment_center = np.asarray(anchor_world_position, dtype=np.float64).reshape(3).copy()
    if not np.all(np.isfinite(attachment_center)):
        raise RuntimeError(f"Shared attachment position is non-finite: {attachment_center}")
    grasp_point = attachment_center + np.array(
        [0.0, 0.0, GRIPPER_SURFACE_OFFSET_M], dtype=np.float64
    )
    pregrasp_point = grasp_point + np.array([0.0, 0.0, PREGRASP_CLEARANCE_M], dtype=np.float64)
    return {
        "attachment_center": attachment_center,
        "grasp_point": grasp_point,
        "pregrasp_point": pregrasp_point,
    }


def calculate_belt_tail_clearance(
    clothing_max_x: float,
    belt_bounds: tuple[np.ndarray, np.ndarray],
    inset: float = BELT_TAIL_INSET_M,
) -> float:
    """Return the world -X drag needed so the bag +X tail sits inset on the belt."""
    maximum = np.asarray(belt_bounds[1], dtype=np.float64).reshape(3)
    tail = float(clothing_max_x)
    inset = float(inset)
    if not np.all(np.isfinite(np.concatenate([maximum, [tail, inset]]))) or inset < 0.0:
        raise RuntimeError(
            "Invalid bag tail or belt bounds for clearance: "
            f"tail_x={tail}, belt_max={maximum}, inset={inset}"
        )
    return max(0.0, tail - (float(maximum[0]) - inset))


def aabb_tail_is_on_belt(
    clothing_max_xy: np.ndarray,
    belt_bounds: tuple[np.ndarray, np.ndarray],
    inset: float = BELT_TAIL_INSET_M,
) -> bool:
    """True when the bag +X tail is inset on the belt and Y is inside the footprint."""
    point = np.asarray(clothing_max_xy, dtype=np.float64).reshape(-1)
    minimum, maximum = (np.asarray(value, dtype=np.float64).reshape(3) for value in belt_bounds)
    inset = float(inset)
    if point.size < 2 or not np.all(np.isfinite(np.concatenate([point[:2], minimum, maximum]))):
        return False
    return bool(point[0] <= maximum[0] - inset and minimum[1] <= point[1] <= maximum[1])


def calculate_belt_entry_distance(
    clothing_center: np.ndarray,
    belt_bounds: tuple[np.ndarray, np.ndarray],
    inset: float = BELT_ENTRY_INSET_M,
) -> float:
    """Return the required world -X distance to enter the near belt edge."""
    center = np.asarray(clothing_center, dtype=np.float64).reshape(3)
    minimum, maximum = (np.asarray(value, dtype=np.float64).reshape(3) for value in belt_bounds)
    inset = float(inset)
    if (
        not np.all(np.isfinite(np.concatenate([center, minimum, maximum])))
        or np.any(maximum <= minimum)
        or inset < 0.0
        or maximum[0] - minimum[0] <= inset
    ):
        raise RuntimeError(
            "Invalid clothing center, belt bounds, or entry inset: "
            f"center={center}, min={minimum}, max={maximum}, inset={inset}"
        )
    return max(0.0, float(center[0] - (maximum[0] - inset)))


def center_is_in_belt_xy(
    clothing_center: np.ndarray,
    belt_bounds: tuple[np.ndarray, np.ndarray],
    entry_inset: float = BELT_ENTRY_INSET_M,
) -> bool:
    """Test the live center against the belt footprint and inset near edge."""
    center = np.asarray(clothing_center, dtype=np.float64).reshape(3)
    minimum, maximum = (np.asarray(value, dtype=np.float64).reshape(3) for value in belt_bounds)
    inset = float(entry_inset)
    if not np.all(np.isfinite(np.concatenate([center, minimum, maximum]))):
        return False
    return bool(
        minimum[0] <= center[0] <= maximum[0] - inset and minimum[1] <= center[1] <= maximum[1]
    )


def _as_nodal_matrix(values: object, label: str) -> np.ndarray:
    nodes = np.asarray(values, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[0] == 0 or nodes.shape[1] != 3:
        raise ValueError(f"{label} must have shape (num_nodes, 3), got {nodes.shape}")
    if not np.all(np.isfinite(nodes)):
        raise ValueError(f"{label} contains non-finite values")
    return nodes


def _single_nodal_matrix(value: object, label: str) -> np.ndarray:
    primary = value[0] if isinstance(value, tuple) else value
    nodes = _to_numpy(primary)
    if nodes.ndim != 3 or nodes.shape[0] != 1:
        raise RuntimeError(f"{label} must have shape (1, num_nodes, 3), got {nodes.shape}")
    try:
        return _as_nodal_matrix(nodes[0], label)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _normalized_planar_direction(direction_xy: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction_xy, dtype=np.float64).reshape(-1)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("direction_xy must contain two finite values")
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("direction_xy must be non-zero")
    return direction / norm


def _compose_nodal_planar_velocity(
    current_velocities: object,
    speed_mps: float,
    direction_xy: np.ndarray,
) -> np.ndarray:
    """Overwrite node XY velocity while preserving every node's Z velocity."""
    current = _as_nodal_matrix(current_velocities, "nodal_velocities")
    speed = float(speed_mps)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("speed_mps must be finite and non-negative")
    result = current.copy()
    result[:, :2] = speed * _normalized_planar_direction(direction_xy)
    return result


def _belt_xy_bounds(
    belt_bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    minimum, maximum = (np.asarray(value, dtype=np.float64).reshape(3) for value in belt_bounds)
    values = np.array(
        [minimum[0], minimum[1], maximum[0], maximum[1]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values[2:] <= values[:2]):
        raise ValueError(f"Invalid belt XY bounds: {values}")
    return values


def _point_in_xy_bounds(point_xy: object, bounds: object) -> bool:
    point = np.asarray(point_xy, dtype=np.float64).reshape(-1)
    region = np.asarray(bounds, dtype=np.float64).reshape(-1)
    if (
        point.shape != (2,)
        or region.shape != (4,)
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(region))
        or np.any(region[2:] <= region[:2])
    ):
        raise ValueError(f"Invalid point or XY bounds: point={point}, bounds={region}")
    return bool(region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3])


def center_exited_belt_negative_x(
    clothing_center: np.ndarray,
    belt_bounds: tuple[np.ndarray, np.ndarray],
) -> bool:
    """Return whether the center crossed the -X edge while retaining belt Y."""
    center = np.asarray(clothing_center, dtype=np.float64).reshape(3)
    bounds = _belt_xy_bounds(belt_bounds)
    if not np.all(np.isfinite(center)):
        return False
    return bool(center[0] < bounds[0] and bounds[1] <= center[1] <= bounds[3])


# =============================================================================
# Stage preparation and deformable attachment helpers
# =============================================================================


def require_timeline_stopped(stage: Usd.Stage) -> None:
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError(
            "Stop the Timeline before running this script.  The deformable "
            "attachment and GPU settings must exist before the first Play."
        )
    if stage is None:
        raise RuntimeError("No USD Stage is open")


def require_uninitialized_physics() -> None:
    """附着须在物理视图建立前写入;框架里 ``init_world`` 已 reset 过,故只告警.

    Returns:
        None.
    """
    if SimulationManager.get_physics_simulation_view() is not None:
        # Job 流水线在 init_world 就 reset 过一次;后续 reset+play 会重新解析 stage,
        # 因此这里不再像原脚本那样抛错,只提示附着是在已有物理视图之后写入的.
        _LOGGER.warning(
            "physics_view_already_exists",
            tip="附着将在下一次 reset/play 时重新烘焙",
        )


def configure_gpu_physics(stage: Usd.Stage) -> None:
    SimulationManager.set_physics_sim_device(PHYSICS_DEVICE)

    scene_paths = PhysicsScene.get_physics_scene_paths(stage)
    if not scene_paths:
        raise RuntimeError("The Stage has no USD PhysicsScene")

    requested_stack = GPU_COLLISION_STACK_SIZE_MIB * 1024 * 1024
    for scene_path in scene_paths:
        scene_prim = stage.GetPrimAtPath(scene_path)
        scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        scene_api.CreateEnableGPUDynamicsAttr().Set(True)
        scene_api.CreateBroadphaseTypeAttr().Set("GPU")
        scene_api.CreateGpuCollisionStackSizeAttr().Set(requested_stack)
        SimulationManager.set_physics_dt(
            1.0 / NOMINAL_PHYSICS_HZ,
            physics_scene=str(scene_path),
        )


def remove_owned_runtime_prims(stage: Usd.Stage) -> None:
    prim = stage.GetPrimAtPath(RUNTIME_ROOT_PATH)
    if prim.IsValid():
        stage.RemovePrim(RUNTIME_ROOT_PATH)


def set_xform_world_translation(
    xform: UsdGeom.Xform,
    position: np.ndarray,
) -> None:
    """Set one runtime Xform whose parent hierarchy has identity transforms."""
    value = np.asarray(position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"Anchor position is non-finite: {value}")
    ops = xform.GetOrderedXformOps()
    translate = next(
        (op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    if translate is None:
        translate = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate.Set(Gf.Vec3d(*(float(component) for component in value)))


def get_xform_world_translation(xform: UsdGeom.Xform) -> np.ndarray:
    """Read one finite world-space translation from a USD Xform."""
    if xform is None or not xform.GetPrim().IsValid():
        raise RuntimeError("Shared attachment anchor Xform is invalid")
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    position = np.asarray(matrix.ExtractTranslation(), dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(position)):
        raise RuntimeError(f"Shared attachment anchor position is non-finite: {position}")
    return position


def create_masked_surface_attachment(
    stage: Usd.Stage,
    attachment_center: np.ndarray,
    *,
    anchor_path: str = "",
    attachment_path: str = "",
    deformable: DeformablePrim | None = None,
) -> tuple[UsdGeom.Xform, DeformablePrim]:
    """Create the temporary world-space anchor and masked surface attachment.

    ``anchor_path`` / ``attachment_path`` 留空表示用模块级路径:默认参数在函数定义时
    就绑定,而 :func:`apply_settings` 是运行期改写全局,写成默认值会取到旧路径.
    """
    anchor_path = anchor_path or ANCHOR_PATH
    attachment_path = attachment_path or ATTACHMENT_PATH
    runtime_root = UsdGeom.Xform.Define(stage, RUNTIME_ROOT_PATH)
    runtime_root.GetPrim().SetMetadata("comment", "Owned by ProM6 clothing drag script")
    anchor = UsdGeom.Xform.Define(stage, anchor_path)
    set_xform_world_translation(anchor, attachment_center)
    # Keep the attachment target as a plain Xform, matching
    # screw_unscrew_demo.py and OmniPhysicsVtxXformAttachment semantics.  A
    # kinematic RigidBody has a separate runtime Physics pose; changing only
    # its USD xformOp can move the viewport Prim without moving the attachment
    # target seen by the GPU solver.
    UsdGeom.Scope.Define(stage, ATTACHMENT_SCOPE_PATH)

    if deformable is None:
        deformable = DeformablePrim(
            CLOTHING_SURFACE_PATH,
            deformable_type="surface",
        )
    if not deformable.valid:
        raise RuntimeError(f"Invalid surface DeformablePrim: {CLOTHING_SURFACE_PATH}")
    if deformable.deformable_type != "surface":
        raise RuntimeError(f"Expected a surface deformable, got {deformable.deformable_type!r}")
    if not deformable.simulation_mesh_paths:
        raise RuntimeError(f"Surface deformable has no simulation mesh: {CLOTHING_SURFACE_PATH}")

    attachment_sdf_path = Sdf.Path(attachment_path)
    if not deformableUtils.create_auto_deformable_attachment(
        stage,
        target_attachment_path=attachment_sdf_path,
        attachable0_path=Sdf.Path(CLOTHING_SURFACE_PATH),
        attachable1_path=Sdf.Path(anchor_path),
    ):
        raise RuntimeError(f"Failed to create deformable attachment: {attachment_path}")

    mask_path = attachment_sdf_path.AppendChild("Mask")
    mask = UsdGeom.Cube.Define(stage, mask_path)
    mask.CreateSizeAttr(1.0)
    mask.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*(float(value) for value in attachment_center))
    )
    mask.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(*(float(value) for value in GRASP_MASK_SIZE_M))
    )
    UsdGeom.Imageable(mask.GetPrim()).MakeInvisible()

    attachment_prim = stage.GetPrimAtPath(attachment_sdf_path)
    attachment_prim.GetRelationship("physxAutoDeformableAttachment:maskShapes").SetTargets(
        [mask_path]
    )
    prefix = "physxAutoDeformableAttachment:"
    attachment_prim.GetAttribute(prefix + "deformableVertexOverlapOffset").Set(0.002)
    attachment_interface = get_physx_attachment_private_interface()
    updated = attachment_interface.update_auto_deformable_attachment(attachment_path)
    if not updated:
        raise RuntimeError(f"Failed to cook/update masked attachment: {attachment_path}")
    return anchor, deformable


def attachment_vertex_count(stage: Usd.Stage, attachment_path: str) -> int:
    """Count generated low-level source vertices below one attachment."""
    root = stage.GetPrimAtPath(attachment_path)
    if not root.IsValid():
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        attr = prim.GetAttribute("omniphysics:vtxIndicesSrc0")
        if not attr.IsValid():
            continue
        value = attr.Get()
        if value is not None:
            count += len(value)
    return count


def disable_gripper_collisions(
    stage: Usd.Stage,
    gripper_path: str,
    side_label: str,
) -> list[tuple[str, bool, bool | None]]:
    """Disable authored gripper colliders and retain restoration data.

    Some Pro-m6 Robotiq variants contain only visual meshes plus empty
    ``collisions`` Xforms.  In that case there is no collision geometry that
    can strike the deformable, so an empty result is valid rather than a
    startup failure.
    """
    root = stage.GetPrimAtPath(gripper_path)
    if not root.IsValid():
        raise RuntimeError(f"{side_label} gripper Prim does not exist: {gripper_path}")
    saved: list[tuple[str, bool, bool | None]] = []
    for prim in Usd.PrimRange(root):
        collision_attr = prim.GetAttribute("physics:collisionEnabled")
        if not prim.HasAPI(UsdPhysics.CollisionAPI) and not collision_attr.IsValid():
            continue
        if prim.IsInstanceProxy():
            raise RuntimeError(
                f"{side_label} gripper collider is an instance proxy and cannot be "
                f"temporarily disabled safely: {prim.GetPath()}"
            )
        collision = UsdPhysics.CollisionAPI(prim)
        attr = collision.GetCollisionEnabledAttr()
        if not attr.IsValid():
            attr = collision.CreateCollisionEnabledAttr()
        had_authored = bool(attr and attr.HasAuthoredValueOpinion())
        old_value = attr.Get() if attr else None
        attr.Set(False)
        saved.append(
            (
                prim.GetPath().pathString,
                had_authored,
                None if old_value is None else bool(old_value),
            )
        )
    if not saved:
        _log(
            f"[ProM6ClothingDrag] {side_label} Robotiq has no authored collision "
            "geometry; collision override is unnecessary"
        )
    return saved


def restore_collision_attributes(
    stage: Usd.Stage,
    saved: list[tuple[str, bool, bool | None]],
) -> None:
    for path, had_authored, old_value in saved:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        attr = prim.GetAttribute("physics:collisionEnabled")
        if not attr.IsValid():
            continue
        if had_authored:
            attr.Set(bool(old_value))
        else:
            attr.Clear()


# =============================================================================
# Successful-operation state cache
# =============================================================================


class SuccessfulOperationRecorder:
    """Keep live tensor state in memory and commit only a verified success."""

    def __init__(
        self,
        stage: Usd.Stage,
        robot: Articulation,
        dof_names: tuple[str, ...],
        dof_paths: tuple[str, ...],
        *,
        nodal_position_fn: Callable[[], np.ndarray],
        nodal_velocity_fn: Callable[[], np.ndarray],
    ) -> None:
        if not dof_names or len(dof_names) != len(dof_paths):
            raise RuntimeError("Replay recorder requires matching non-empty DOF metadata")
        self.stage = stage
        self.robot = robot
        self.nodal_position_fn = nodal_position_fn
        self.nodal_velocity_fn = nodal_velocity_fn
        self.dof_names = tuple(str(value) for value in dof_names)
        self.dof_paths = tuple(str(value) for value in dof_paths)
        self.node_count: int | None = None
        self.committed_path: Path | None = None
        self._frames: dict[str, list[object]] = {
            "frame_dt": [],
            "phase_ids": [],
            "dof_positions": [],
            "dof_velocities": [],
            "dof_position_targets": [],
            "nodal_positions": [],
            "nodal_velocities": [],
            "attachment_active": [],
            "anchor_positions": [],
        }

    @property
    def frame_count(self) -> int:
        return len(self._frames["frame_dt"])

    @staticmethod
    def _finite_vector(value: object, size: int, label: str) -> np.ndarray:
        result = _to_numpy(value).reshape(-1).astype(np.float64)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise RuntimeError(
                f"Replay recorder {label} must contain {size} finite values; "
                f"got shape={result.shape}"
            )
        return result

    def capture(
        self,
        dt: float,
        phase: Phase,
        *,
        attachment_active: bool,
        anchor_position: np.ndarray | None,
    ) -> None:
        if self.committed_path is not None:
            raise RuntimeError("Cannot capture frames after replay cache commit")
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0 or dt > 0.1:
            raise RuntimeError(f"Replay recorder received invalid dt={dt!r}")

        dof_count = len(self.dof_paths)
        dof_positions = self._finite_vector(
            self.robot.get_dof_positions(), dof_count, "DOF positions"
        )
        dof_velocities = self._finite_vector(
            self.robot.get_dof_velocities(), dof_count, "DOF velocities"
        )
        dof_position_targets = self._finite_vector(
            self.robot.get_dof_position_targets(),
            dof_count,
            "DOF position targets",
        )
        nodal_positions = _as_nodal_matrix(self.nodal_position_fn(), "replay nodal positions")
        nodal_velocities = _as_nodal_matrix(self.nodal_velocity_fn(), "replay nodal velocities")
        if nodal_positions.shape != nodal_velocities.shape:
            raise RuntimeError(
                "Replay recorder nodal position/velocity shapes differ: "
                f"positions={nodal_positions.shape}, "
                f"velocities={nodal_velocities.shape}"
            )
        if self.node_count is None:
            self.node_count = int(nodal_positions.shape[0])
        elif nodal_positions.shape[0] != self.node_count:
            raise RuntimeError(
                "Replay recorder deformable node count changed: "
                f"expected={self.node_count}, actual={nodal_positions.shape[0]}"
            )

        active = bool(attachment_active)
        if active:
            if anchor_position is None:
                raise RuntimeError("Active attachment has no replay anchor position")
            anchor = np.asarray(anchor_position, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(anchor)):
                raise RuntimeError(f"Active attachment replay anchor is non-finite: {anchor}")
        else:
            anchor = np.full(3, np.nan, dtype=np.float64)

        cache_dtype = np.float32
        self._frames["frame_dt"].append(dt)
        self._frames["phase_ids"].append(int(phase.value))
        self._frames["dof_positions"].append(dof_positions.astype(cache_dtype))
        self._frames["dof_velocities"].append(dof_velocities.astype(cache_dtype))
        self._frames["dof_position_targets"].append(dof_position_targets.astype(cache_dtype))
        self._frames["nodal_positions"].append(nodal_positions.astype(cache_dtype))
        self._frames["nodal_velocities"].append(nodal_velocities.astype(cache_dtype))
        self._frames["attachment_active"].append(active)
        self._frames["anchor_positions"].append(anchor.astype(cache_dtype))

    def discard(self) -> None:
        if self.committed_path is not None:
            return
        for values in self._frames.values():
            values.clear()
        self.node_count = None

    def _metadata(self) -> dict[str, object]:
        if self.node_count is None or self.frame_count <= 0:
            raise RuntimeError("Cannot build metadata for an empty replay cache")
        root_layer = self.stage.GetRootLayer()
        stage_identifier = "" if root_layer is None else str(root_layer.identifier)
        frame_dt = np.asarray(self._frames["frame_dt"], dtype=np.float64)
        return {
            "schema_version": REPLAY_CACHE_SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "stage_identifier": stage_identifier,
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(self.stage)),
            "robot_path": ROBOT_PATH,
            "clothing_surface_path": CLOTHING_SURFACE_PATH,
            "anchor_path": ANCHOR_PATH,
            "dof_names": list(self.dof_names),
            "dof_paths": list(self.dof_paths),
            "frame_count": self.frame_count,
            "node_count": self.node_count,
            "dtype": "float32",
            "median_physics_dt": float(np.median(frame_dt)),
            "phase_names": {str(value.value): value.name for value in Phase},
        }

    def commit_success(self) -> Path:
        if self.committed_path is not None:
            return self.committed_path
        metadata = self._metadata()
        arrays = {
            "frame_dt": np.asarray(self._frames["frame_dt"], dtype=np.float32),
            "phase_ids": np.asarray(self._frames["phase_ids"], dtype=np.int16),
            "dof_positions": np.stack(self._frames["dof_positions"]).astype(np.float32, copy=False),
            "dof_velocities": np.stack(self._frames["dof_velocities"]).astype(
                np.float32, copy=False
            ),
            "dof_position_targets": np.stack(self._frames["dof_position_targets"]).astype(
                np.float32, copy=False
            ),
            "nodal_positions": np.stack(self._frames["nodal_positions"]).astype(
                np.float32, copy=False
            ),
            "nodal_velocities": np.stack(self._frames["nodal_velocities"]).astype(
                np.float32, copy=False
            ),
            "attachment_active": np.asarray(self._frames["attachment_active"], dtype=np.bool_),
            "anchor_positions": np.stack(self._frames["anchor_positions"]).astype(
                np.float32, copy=False
            ),
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
        }

        REPLAY_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        final_path = REPLAY_CACHE_DIRECTORY / (f"{REPLAY_CACHE_FILE_PREFIX}{timestamp}.npz")
        temporary_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
        if final_path.exists() or temporary_path.exists():
            raise RuntimeError(f"Replay cache output path already exists: {final_path}")
        try:
            with temporary_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, final_path)
        except Exception:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        self.committed_path = final_path
        _log(
            "[ProM6ClothingDrag][cache] Saved successful replay: "
            f"{final_path} (frames={self.frame_count}, nodes={self.node_count})"
        )
        return final_path


# =============================================================================
# Embedded deformable conveyor controller
# =============================================================================


class ConveyorState(Enum):
    WAITING_FOR_BELT = auto()
    TRANSPORTING = auto()
    STOPPED = auto()
    EXITED = auto()


class DeformableConveyorController:
    """松爪后给 Newton 出带代理 -X 速度,用 AABB 判定是否送出."""

    def __init__(
        self,
        newton_world: NewtonSortingWorld,
        belt_xy_bounds: np.ndarray,
        stop_xy_region: dict[str, float] | None,
        speed_mps: float,
        direction_xy: np.ndarray,
        *,
        stop_on_arrival: bool,
        zero_xy_on_exit: bool,
        log_interval_frames: int,
    ) -> None:
        self.newton_world = newton_world
        self.belt_xy_bounds = np.asarray(belt_xy_bounds, dtype=np.float64).reshape(-1)
        if (
            self.belt_xy_bounds.shape != (4,)
            or not np.all(np.isfinite(self.belt_xy_bounds))
            or np.any(self.belt_xy_bounds[2:] <= self.belt_xy_bounds[:2])
        ):
            raise ValueError(f"Invalid conveyor belt XY bounds: {self.belt_xy_bounds}")

        self.stop_xy_region = None if stop_xy_region is None else dict(stop_xy_region)
        self.stop_xy_bounds: np.ndarray | None = None
        if self.stop_xy_region is not None:
            expected = {"x_min", "x_max", "y_min", "y_max"}
            if set(self.stop_xy_region) != expected:
                raise ValueError("stop_xy_region must contain exactly x_min, x_max, y_min, y_max")
            self.stop_xy_bounds = np.array(
                [
                    self.stop_xy_region["x_min"],
                    self.stop_xy_region["y_min"],
                    self.stop_xy_region["x_max"],
                    self.stop_xy_region["y_max"],
                ],
                dtype=np.float64,
            )
            _point_in_xy_bounds(self.stop_xy_bounds[:2], self.stop_xy_bounds)

        self.speed_mps = float(speed_mps)
        self.direction_xy = _normalized_planar_direction(direction_xy)
        self.stop_on_arrival = bool(stop_on_arrival)
        self.zero_xy_on_exit = bool(zero_xy_on_exit)
        self.log_interval_frames = int(log_interval_frames)
        if not math.isfinite(self.speed_mps) or self.speed_mps < 0.0:
            raise ValueError("speed_mps must be finite and non-negative")
        if self.log_interval_frames <= 0:
            raise ValueError("log_interval_frames must be positive")
        if self.stop_on_arrival and self.stop_xy_bounds is None:
            raise ValueError("stop_xy_region is required when stop_on_arrival=True")

        self.state = ConveyorState.WAITING_FOR_BELT
        self.active = False
        self.cleaned = False
        self.was_in_belt = False
        self.frame = 0
        self.exit_center: np.ndarray | None = None
        self.last_command: np.ndarray | None = None

    def start(self) -> None:
        if self.cleaned:
            raise RuntimeError("Cannot restart a cleaned conveyor controller")
        if self.active:
            return
        self.active = True
        self.newton_world.set_belt_speed(self.speed_mps)
        _log(
            "[ProM6ClothingDrag][conveyor] Started: "
            f"speed={self.speed_mps:.3f} m/s, "
            f"direction_xy={_array_string(self.direction_xy)}, "
            f"stop_xy_region={self.stop_xy_region}, "
            f"stop_on_arrival={self.stop_on_arrival}, "
            f"zero_xy_on_exit={self.zero_xy_on_exit}"
        )

    def _set_state(self, state: ConveyorState) -> None:
        if state == self.state:
            return
        self.state = state
        _log(f"[ProM6ClothingDrag][conveyor] State {state.name}")

    def _read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = _as_nodal_matrix(self.newton_world.surface_positions(), "nodal_positions")
        velocities = _as_nodal_matrix(
            self.newton_world.particle_velocities()[: self.newton_world.surface_count],
            "nodal_velocities",
        )
        if positions.shape != velocities.shape:
            raise RuntimeError(
                "Release conveyor nodal position/velocity shapes differ: "
                f"positions={positions.shape}, velocities={velocities.shape}"
            )
        center = 0.5 * (np.min(positions, axis=0) + np.max(positions, axis=0))
        return positions, velocities, center

    def _write_command(self, command: np.ndarray) -> None:
        del command
        # 接触抓取不再写粒子速度;皮带运动学速度在 start/cleanup 里切换.
        self.last_command = None

    def step(self, dt: float) -> None:
        if self.cleaned or not self.active:
            return
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0 or dt > 0.1:
            raise RuntimeError(f"Invalid release conveyor dt={dt!r}")

        self.frame += 1
        positions, velocities, center = self._read_state()
        inside_belt = _point_in_xy_bounds(center[:2], self.belt_xy_bounds)
        inside_stop = bool(
            inside_belt
            and self.stop_on_arrival
            and self.stop_xy_bounds is not None
            and _point_in_xy_bounds(center[:2], self.stop_xy_bounds)
        )

        if inside_stop:
            command = velocities.copy()
            command[:, :2] = 0.0
            self._write_command(command)
            self.was_in_belt = True
            self._set_state(ConveyorState.STOPPED)
        elif inside_belt:
            command = _compose_nodal_planar_velocity(velocities, self.speed_mps, self.direction_xy)
            self._write_command(command)
            self.was_in_belt = True
            self._set_state(ConveyorState.TRANSPORTING)
        elif self.was_in_belt:
            if self.zero_xy_on_exit:
                command = velocities.copy()
                command[:, :2] = 0.0
                self._write_command(command)
            self.exit_center = center.copy()
            self._set_state(ConveyorState.EXITED)
            _log(
                f"[ProM6ClothingDrag][conveyor] Center exited belt: center={_array_string(center)}"
            )
            self.cleanup()
            return
        else:
            self._set_state(ConveyorState.WAITING_FOR_BELT)

        if self.frame % self.log_interval_frames == 0:
            span = np.max(positions, axis=0) - np.min(positions, axis=0)
            mean_xy = np.mean(velocities[:, :2], axis=0)
            _log(
                "[ProM6ClothingDrag][conveyor][status] "
                f"state={self.state.name}, center={_array_string(center)}, "
                f"span={_array_string(span)}, mean_xy={_array_string(mean_xy)}, "
                f"contacts={self.newton_world.contact_breakdown()}, "
                f"belt_speed={self.newton_world.belt_speed_mps:.3f}"
            )

    def cleanup(self) -> None:
        if self.cleaned:
            return
        self.cleaned = True
        self.active = False
        self.newton_world.set_belt_speed(0.0)
        _log("[ProM6ClothingDrag][conveyor] Controller cleaned")


# =============================================================================
# Dual-arm relay task controller
# =============================================================================


def validate_right_lift_height(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("RIGHT_LIFT_HEIGHT_M must be finite and nonnegative")
    return value


def validate_right_drag_lower_distance(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("RIGHT_DRAG_LOWER_DISTANCE_M must be finite and nonnegative")
    return value


class ProM6LeftClothingDragController:
    """Deterministic dual-arm relay modeled after the source demo."""

    def __init__(
        self,
        stage: Usd.Stage,
        timing: TimingConfig,
        articulation: Articulation | None = None,
    ) -> None:
        validate_right_lift_height(RIGHT_LIFT_HEIGHT_M)
        validate_right_drag_lower_distance(RIGHT_DRAG_LOWER_DISTANCE_M)
        if not math.isfinite(float(LEFT_LOWER_DISTANCE_OFFSET_M)):
            raise ValueError("LEFT_LOWER_DISTANCE_OFFSET_M must be finite")
        self.stage = stage
        self.timing = timing.validate()
        self.phase_timeouts = build_phase_timeouts(self.timing)
        self.timeline = omni.timeline.get_timeline_interface()
        # 机器人后端已建过同一 prim 的 view,复用它,避免一根关节体上挂两份句柄
        self.robot = articulation if articulation is not None else Articulation(ROBOT_PATH)
        self.anchor: UsdGeom.Xform | None = None
        self.deformable: DeformablePrim | None = None
        self.newton_world: NewtonSortingWorld | None = None
        self._newton_hold_frames = 0
        self.min_lift_height_m = 0.08
        self.left_lift_com_z: float | None = None
        self.right_lift_com_z: float | None = None
        self.replay_recorder: SuccessfulOperationRecorder | None = None
        self.release_conveyor: DeformableConveyorController | None = None
        self.saved_collisions: list[tuple[str, bool, bool | None]] = []

        self.callback_id: int | None = None
        self.stop_callback_id: int | None = None
        self.initialized = False
        self.cleaned = False
        self.failed = False
        self.phase = Phase.SETTLE
        self.phase_time = 0.0
        self.phase_frame = 0
        self.stable_time = 0.0

        self.left_arm_indices: list[int] = []
        self.right_arm_indices: list[int] = []
        self.left_arm_dof_paths: list[str] = []
        self.right_arm_dof_paths: list[str] = []
        self.left_arm_lower_limits: np.ndarray | None = None
        self.left_arm_upper_limits: np.ndarray | None = None
        self.right_arm_lower_limits: np.ndarray | None = None
        self.right_arm_upper_limits: np.ndarray | None = None
        self.left_base_link_path: str | None = None
        self.left_base_link_index: int | None = None
        self.left_base_jacobian_row: int | None = None
        self.left_tool_center_path: str | None = None
        self.right_base_link_path: str | None = None
        self.right_base_link_index: int | None = None
        self.right_base_jacobian_row: int | None = None
        self.right_tool_center_path: str | None = None
        self.left_gripper_index: int | None = None
        self.right_gripper_index: int | None = None
        self.left_gripper_lower_limit = -math.inf
        self.left_gripper_upper_limit = math.inf
        self.right_gripper_lower_limit = -math.inf
        self.right_gripper_upper_limit = math.inf
        self.last_joint_step = 0.0
        self.last_tcp_error = math.nan

        self.left_start_q: np.ndarray | None = None
        self.left_commanded_q: np.ndarray | None = None
        self.right_start_q: np.ndarray | None = None
        self.right_commanded_q: np.ndarray | None = None
        self.left_return_start_q: np.ndarray | None = None
        self.right_return_start_q: np.ndarray | None = None
        self.pregrasp_start_q: np.ndarray | None = None
        self.pregrasp_joint_target: np.ndarray | None = None
        self.right_pregrasp_start_q: np.ndarray | None = None
        self.right_pregrasp_joint_target: np.ndarray | None = None
        self.tcp_offset_local: np.ndarray | None = None
        self.tcp_adapter: np.ndarray | None = None
        self.usd_kinematics: UsdArmKinematics | None = None
        self.right_tcp_offset_local: np.ndarray | None = None
        self.right_tcp_adapter: np.ndarray | None = None
        self.right_usd_kinematics: UsdArmKinematics | None = None

        self.grasp_orientation = np.array(
            [0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0],
            dtype=np.float64,
        )
        self.geometry: dict[str, np.ndarray] | None = None
        self.clothing_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.support_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.grasp_point: np.ndarray | None = None
        self.pregrasp_point: np.ndarray | None = None
        self.attachment_center: np.ndarray | None = None
        self.right_handoff_anchor_position: np.ndarray | None = None
        self.right_handoff_tcp_position: np.ndarray | None = None
        self.right_grasp_point: np.ndarray | None = None
        self.right_pregrasp_point: np.ndarray | None = None
        self.right_lift_start: np.ndarray | None = None
        self.right_lift_target: np.ndarray | None = None
        self.right_drag_start: np.ndarray | None = None
        self.right_drag_target: np.ndarray | None = None
        self.right_nominal_drag_distance = 0.0
        self.right_actual_drag_distance = 0.0
        self.support_center: np.ndarray | None = None
        self.belt_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.drag_start: np.ndarray | None = None
        self.left_lift_start: np.ndarray | None = None
        self.left_lift_target: np.ndarray | None = None
        self.drag_target: np.ndarray | None = None
        self.correction_target: np.ndarray | None = None
        self.correction_origin: np.ndarray | None = None
        self.lower_start: np.ndarray | None = None
        self.setdown_target: np.ndarray | None = None
        self.attachment_active = False
        self.initial_node_center: np.ndarray | None = None
        self.initial_node_height: float | None = None
        self.final_center_error: float | None = None
        self._cached_nodes: np.ndarray | None = None
        self._right_grasp_z_refreshed = False
        self._right_grasp_dip_m = 0.0
        self._left_grasp_dip_m = 0.0
        self.left_retract_target: np.ndarray | None = None
        self._left_retract_waypoints: list[np.ndarray] = []
        self._left_retract_i = 0
        self._left_return_clear_target: np.ndarray | None = None
        self._left_return_cartesian_done = False
        self._left_return_joint_elapsed = 0.0
        self._left_retract_hold_s = 0.0
        self._left_close_xy_refreshed = False
        self._right_close_elapsed = 0.0

    @staticmethod
    def _single_exact_index(values: tuple[str, ...], expected: str, label: str) -> int:
        matches = [index for index, value in enumerate(values) if value == expected]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {label} {expected!r}; matches={matches}")
        return matches[0]

    @staticmethod
    def _single_suffix_index(values: tuple[str, ...], suffix: str, label: str) -> int:
        matches = [index for index, value in enumerate(values) if value.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {label} ending in {suffix!r}; matches={matches}")
        return matches[0]

    @staticmethod
    def _jacobian_row(link_index: int, link_count: int, row_count: int) -> int:
        if row_count == link_count:
            row = link_index
        elif row_count == link_count - 1:
            row = link_index - 1
        else:
            raise RuntimeError(f"Cannot map Jacobian rows={row_count} to links={link_count}")
        if not 0 <= row < row_count:
            raise RuntimeError(f"Link index {link_index} maps to invalid row {row}")
        return row

    @staticmethod
    def _get_limits(articulation: Articulation) -> tuple[np.ndarray, np.ndarray]:
        limits = articulation.get_dof_limits()
        if isinstance(limits, tuple) and len(limits) == 2:
            return (
                _to_numpy(limits[0]).reshape(-1).astype(np.float64),
                _to_numpy(limits[1]).reshape(-1).astype(np.float64),
            )
        array = _to_numpy(limits)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or array.shape[1] != 2:
            raise RuntimeError(f"Unexpected DOF limit shape: {array.shape}")
        return array[:, 0].astype(np.float64), array[:, 1].astype(np.float64)

    @staticmethod
    def _world_pose(prim: RigidPrim | XformPrim) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations = prim.get_world_poses()
        position = _to_numpy(positions).reshape(-1, 3)[0].astype(np.float64)
        orientation = _normalize_quaternion(
            _to_numpy(orientations).reshape(-1, 4)[0].astype(np.float64)
        )
        if not np.all(np.isfinite(position)):
            raise RuntimeError("World pose contains a non-finite position")
        return position, orientation

    def prepare_before_play(self) -> None:
        robot_prim = self.stage.GetPrimAtPath(ROBOT_PATH)
        clothing_root = self.stage.GetPrimAtPath(CLOTHING_ROOT_PATH)
        clothing_surface = self.stage.GetPrimAtPath(CLOTHING_SURFACE_PATH)
        support = self.stage.GetPrimAtPath(SUPPORT_PATH)
        belt = self.stage.GetPrimAtPath(BELT_PATH)
        for label, prim, path in (
            ("robot", robot_prim, ROBOT_PATH),
            ("clothing root", clothing_root, CLOTHING_ROOT_PATH),
            ("clothing surface", clothing_surface, CLOTHING_SURFACE_PATH),
            ("support", support, SUPPORT_PATH),
            ("belt", belt, BELT_PATH),
        ):
            if not prim.IsValid():
                raise RuntimeError(f"Missing {label} Prim: {path}")
        if not clothing_surface.IsA(UsdGeom.Mesh) or not clothing_surface.IsActive():
            raise RuntimeError(f"Clothing surface must be one active Mesh: {CLOTHING_SURFACE_PATH}")
        from real2sim2real.planner.newton_sorting import (
            restore_cached_rest_mesh,
            strip_clothing_physx,
        )

        restore_cached_rest_mesh(self.stage, CLOTHING_SURFACE_PATH)
        strip_clothing_physx(
            self.stage,
            clothing_root_path=CLOTHING_ROOT_PATH,
            clothing_surface_path=CLOTHING_SURFACE_PATH,
            volume_proxy_path=CLOTHING_VOLUME_PROXY_PATH,
        )

        self.clothing_bounds = get_world_bounds(clothing_root)
        self.support_bounds = get_world_bounds(support)
        self.belt_bounds = get_world_bounds(belt)
        self.geometry = calculate_grasp_geometry(self.clothing_bounds, self.support_bounds)
        self.grasp_point = self.geometry["grasp_point"].copy()
        self.pregrasp_point = self.geometry["pregrasp_point"].copy()
        self.attachment_center = self.geometry["attachment_center"].copy()
        self.support_center = self.geometry["support_center"].copy()

        remove_owned_runtime_prims(self.stage)
        try:
            self.saved_collisions.extend(
                disable_gripper_collisions(self.stage, LEFT_GRIPPER_PATH, "Left")
            )
            self.saved_collisions.extend(
                disable_gripper_collisions(self.stage, RIGHT_GRIPPER_PATH, "Right")
            )
            self.attachment_active = False
        except Exception:
            restore_collision_attributes(self.stage, self.saved_collisions)
            self.saved_collisions.clear()
            remove_owned_runtime_prims(self.stage)
            raise

        _log("[ProM6ClothingDrag] Prepared before first Play")
        _log(f"  robot: {ROBOT_PATH}")
        _log(f"  surface: {CLOTHING_SURFACE_PATH}")
        _log(f"  support: {SUPPORT_PATH}")
        _log(f"  clothing bounds: min={self.clothing_bounds[0]}, max={self.clothing_bounds[1]}")
        _log(f"  support bounds: min={self.support_bounds[0]}, max={self.support_bounds[1]}")
        _log(f"  belt: {BELT_PATH}")
        _log(f"  belt bounds: min={self.belt_bounds[0]}, max={self.belt_bounds[1]}")
        _log(f"  grasp point: {_array_string(self.grasp_point)}")
        _log(f"  grasp patch (legacy attachment_center): {_array_string(self.attachment_center)}")
        _log(f"  platform center: {_array_string(self.support_center)}")
        _log(f"  center tolerance: {CENTER_RELEASE_TOLERANCE_M:.3f} m")
        _log(f"  disabled gripper colliders: {len(self.saved_collisions)}")
        _log("  timing configuration (seconds):")
        for name in self.timing.__dataclass_fields__:
            _log(f"    {name}={float(getattr(self.timing, name)):.6g}")

    def _sync_newton_jaws(self) -> None:
        if self.newton_world is None or self.left_gripper_index is None:
            return
        from real2sim2real.planner.newton_sorting import jaw_separation_m

        q = self._current_dof_positions()
        left_q = float(q[self.left_gripper_index])
        right_index = self.right_gripper_index
        right_q = float(q[right_index]) if right_index is not None else 0.0
        left_closed = float(
            np.clip(
                GRIPPER_CLOSED_POSITION_RAD,
                self.left_gripper_lower_limit,
                self.left_gripper_upper_limit,
            )
        )
        right_closed = left_closed
        if right_index is not None:
            right_closed = float(
                np.clip(
                    GRIPPER_CLOSED_POSITION_RAD,
                    self.right_gripper_lower_limit,
                    self.right_gripper_upper_limit,
                )
            )
        self.newton_world.set_jaw_separations(
            jaw_separation_m(left_q, closed_rad=left_closed),
            jaw_separation_m(right_q, closed_rad=right_closed),
        )
        self.newton_world.set_finger_grip_scale(
            left_q / max(left_closed, 1.0e-6),
            right_q / max(right_closed, 1.0e-6),
        )

    def _teleport_pad_retract(self, retract_m: float) -> None:
        """松爪时瞬移指垫并冻结 VBD 几帧,避免扫掠把袋拉走."""
        if self.newton_world is None:
            return
        self.newton_world.set_pad_retract(retract_m)
        self._newton_hold_frames = max(self._newton_hold_frames, PAD_RETRACT_HOLD_FRAMES)
        _log(
            "[ProM6ClothingDrag] Pad retract teleport: "
            f"retract={retract_m:.3f} m, hold_frames={self._newton_hold_frames}, "
            f"{self.newton_world.finger_debug()}"
        )

    def bind_newton(self, world: NewtonSortingWorld, *, min_lift_height_m: float) -> None:
        """Play 前绑上 Newton 软袋世界;接触抓取在物理回调里步进."""
        self.newton_world = world
        self.min_lift_height_m = float(min_lift_height_m)
        _log(
            "[ProM6ClothingDrag] Newton contact world bound: "
            f"particles={world.model.particle_count}, "
            f"min_lift={self.min_lift_height_m:.3f} m"
        )

    def _current_dof_positions(self) -> np.ndarray:
        return _to_numpy(self.robot.get_dof_positions()).reshape(-1).astype(np.float64)

    def _build_arm_kinematics(
        self,
        *,
        side: str,
        arm_indices: list[int],
        arm_dof_paths: list[str],
        start_q: np.ndarray,
        base_link_path: str,
        base_link_index: int,
        base_jacobian_row: int,
        tool_center_path: str,
    ) -> tuple[UsdArmKinematics, np.ndarray, np.ndarray]:
        base_position, base_orientation = self._world_pose(RigidPrim(base_link_path))
        tool_position, tool_orientation = self._world_pose(XformPrim(tool_center_path))
        tcp_offset_local = _local_offset_from_world_point(
            base_position,
            base_orientation,
            tool_position,
        )
        tcp_adapter = np.linalg.inv(_pose_matrix(base_position, base_orientation)) @ _pose_matrix(
            tool_position, tool_orientation
        )
        live_jacobian = self._live_tcp_jacobian(
            base_link_index,
            base_jacobian_row,
            arm_indices,
            base_position,
            base_orientation,
            tool_position,
        )
        model = UsdArmKinematics.from_stage(
            self.stage,
            arm_dof_paths,
            start_q,
            base_link_path,
            tcp_adapter,
            live_jacobian,
        )
        predicted_position, predicted_orientation, predicted_jacobian, _ = (
            model.forward_and_jacobian(start_q)
        )
        model_error = _pose_error(
            predicted_position,
            predicted_orientation,
            tool_position,
            tool_orientation,
        )
        jacobian_error = float(np.max(np.abs(predicted_jacobian - live_jacobian)))
        _log(
            f"[ProM6ClothingDrag][{side} IK model] validation: "
            f"position={1000.0 * np.linalg.norm(model_error[:3]):.3f} mm, "
            f"orientation={math.degrees(np.linalg.norm(model_error[3:])):.3f} deg, "
            f"jacobian_max_abs={jacobian_error:.6f}, "
            f"tcp_offset_local={_array_string(tcp_offset_local, 6)}"
        )
        if np.linalg.norm(model_error[:3]) > IK_MODEL_POSITION_TOLERANCE_M:
            raise RuntimeError(f"{side} USD FK position validation failed")
        if np.linalg.norm(model_error[3:]) > IK_MODEL_ORIENTATION_TOLERANCE_RAD:
            raise RuntimeError(f"{side} USD FK orientation validation failed")
        if jacobian_error > IK_MODEL_JACOBIAN_TOLERANCE:
            raise RuntimeError(f"{side} USD Jacobian validation failed")
        return model, tcp_offset_local, tcp_adapter

    def _initialize_after_play(self) -> None:
        dof_names = tuple(str(value) for value in self.robot.dof_names)
        try:
            dof_paths = tuple(str(value) for value in self.robot.dof_paths[0])
            link_paths = tuple(str(value) for value in self.robot.link_paths[0])
            jacobian_shape = tuple(int(value) for value in self.robot.jacobian_matrix_shape)
        except (AttributeError, IndexError, TypeError) as error:
            raise RuntimeError("Articulation structure is unavailable after Play") from error
        if len(jacobian_shape) != 3 or jacobian_shape[1] != 6:
            raise RuntimeError(f"Unexpected Jacobian metadata: {jacobian_shape}")

        self.left_arm_indices = [
            self._single_exact_index(dof_names, f"Joint{number}_L", "left arm DOF")
            for number in range(1, 8)
        ]
        self.right_arm_indices = [
            self._single_exact_index(dof_names, f"Joint{number}_R", "right arm DOF")
            for number in range(1, 8)
        ]
        self.left_gripper_index = self._single_suffix_index(
            dof_paths, "/robotiq_left/joints/finger_joint", "left gripper DOF"
        )
        self.right_gripper_index = self._single_suffix_index(
            dof_paths, "/robotiq_right/joints/finger_joint", "right gripper DOF"
        )
        self.left_base_link_index = self._single_suffix_index(
            link_paths,
            "/robotiq_left/robotiq_arg2f_base_link",
            "left Robotiq base link",
        )
        self.left_base_link_path = link_paths[self.left_base_link_index]
        self.left_tool_center_path = f"{self.left_base_link_path}/ToolCenter_L"
        if not self.stage.GetPrimAtPath(self.left_tool_center_path).IsValid():
            raise RuntimeError(f"Missing left ToolCenter: {self.left_tool_center_path}")
        self.left_base_jacobian_row = self._jacobian_row(
            self.left_base_link_index, len(link_paths), jacobian_shape[0]
        )
        self.left_arm_dof_paths = [dof_paths[index] for index in self.left_arm_indices]
        self.right_base_link_index = self._single_suffix_index(
            link_paths,
            "/robotiq_right/robotiq_arg2f_base_link",
            "right Robotiq base link",
        )
        self.right_base_link_path = link_paths[self.right_base_link_index]
        self.right_tool_center_path = f"{self.right_base_link_path}/ToolCenter_R"
        if not self.stage.GetPrimAtPath(self.right_tool_center_path).IsValid():
            raise RuntimeError(f"Missing right ToolCenter: {self.right_tool_center_path}")
        self.right_base_jacobian_row = self._jacobian_row(
            self.right_base_link_index, len(link_paths), jacobian_shape[0]
        )
        self.right_arm_dof_paths = [dof_paths[index] for index in self.right_arm_indices]

        lower, upper = self._get_limits(self.robot)
        if lower.size != len(dof_names) or upper.size != len(dof_names):
            raise RuntimeError("DOF limit count does not match the articulation")
        self.left_arm_lower_limits = lower[self.left_arm_indices].copy()
        self.left_arm_upper_limits = upper[self.left_arm_indices].copy()
        self.right_arm_lower_limits = lower[self.right_arm_indices].copy()
        self.right_arm_upper_limits = upper[self.right_arm_indices].copy()
        self.left_gripper_lower_limit = float(lower[self.left_gripper_index])
        self.left_gripper_upper_limit = float(upper[self.left_gripper_index])
        self.right_gripper_lower_limit = float(lower[self.right_gripper_index])
        self.right_gripper_upper_limit = float(upper[self.right_gripper_index])
        _log(
            "[ProM6ClothingDrag] Gripper limits: "
            f"left=[{self.left_gripper_lower_limit:.4f}, {self.left_gripper_upper_limit:.4f}], "
            f"right=[{self.right_gripper_lower_limit:.4f}, {self.right_gripper_upper_limit:.4f}]"
        )
        for label, target, target_lower, target_upper in (
            (
                "left ready",
                LEFT_READY_JOINTS,
                self.left_arm_lower_limits,
                self.left_arm_upper_limits,
            ),
            (
                "right ready",
                RIGHT_READY_JOINTS,
                self.right_arm_lower_limits,
                self.right_arm_upper_limits,
            ),
        ):
            if np.any(target < target_lower) or np.any(target > target_upper):
                raise RuntimeError(
                    f"Configured {label} posture violates live joint limits: "
                    f"target={target}, lower={target_lower}, upper={target_upper}"
                )

        current = self._current_dof_positions()
        self.left_start_q = current[self.left_arm_indices].copy()
        self.left_commanded_q = self.left_start_q.copy()
        self.right_start_q = current[self.right_arm_indices].copy()
        self.right_commanded_q = self.right_start_q.copy()

        self.usd_kinematics, self.tcp_offset_local, self.tcp_adapter = self._build_arm_kinematics(
            side="left",
            arm_indices=self.left_arm_indices,
            arm_dof_paths=self.left_arm_dof_paths,
            start_q=self.left_start_q,
            base_link_path=self.left_base_link_path,
            base_link_index=self.left_base_link_index,
            base_jacobian_row=self.left_base_jacobian_row,
            tool_center_path=self.left_tool_center_path,
        )
        (
            self.right_usd_kinematics,
            self.right_tcp_offset_local,
            self.right_tcp_adapter,
        ) = self._build_arm_kinematics(
            side="right",
            arm_indices=self.right_arm_indices,
            arm_dof_paths=self.right_arm_dof_paths,
            start_q=self.right_start_q,
            base_link_path=self.right_base_link_path,
            base_link_index=self.right_base_link_index,
            base_jacobian_row=self.right_base_jacobian_row,
            tool_center_path=self.right_tool_center_path,
        )

        # The initial jump is intentional: calibrate the USD model against the
        # measured pre-command pose first, then start the task with both arms in
        # the configured safe ready configuration.
        self._write_left_q(LEFT_READY_JOINTS)
        self._write_right_ready()
        self.left_start_q = LEFT_READY_JOINTS.copy()
        self.right_start_q = RIGHT_READY_JOINTS.copy()
        self.right_commanded_q = RIGHT_READY_JOINTS.copy()
        _log(
            "[ProM6ClothingDrag] Immediate dual-arm ready initialized: "
            f"left_indices={self.left_arm_indices}, "
            f"left_q={_array_string(LEFT_READY_JOINTS)}, "
            f"right_indices={self.right_arm_indices}, "
            f"right_q={_array_string(RIGHT_READY_JOINTS)}"
        )

        self._cached_nodes = None
        if self.newton_world is None:
            raise RuntimeError("Newton sorting world was not bound before Play")
        self.newton_world.set_tcp_pose_fn("left", self._current_tcp)
        self.newton_world.set_tcp_pose_fn("right", self._current_right_tcp)
        self.newton_world.reset_kinematic_history()
        self._newton_hold_frames = 6
        self._deformable_nodes()
        self.replay_recorder = SuccessfulOperationRecorder(
            self.stage,
            self.robot,
            dof_names,
            dof_paths,
            nodal_position_fn=self.newton_world.surface_positions,
            nodal_velocity_fn=self._newton_surface_velocities,
        )
        _log(
            "[ProM6ClothingDrag][cache] Recording full state in memory; "
            f"successful runs will be archived in {REPLAY_CACHE_DIRECTORY}"
        )
        self.initialized = True
        self._announce_phase()

    def _live_tcp_jacobian(
        self,
        base_link_index: int,
        base_jacobian_row: int,
        arm_indices: list[int],
        base_position: np.ndarray,
        base_orientation: np.ndarray,
        tcp_position: np.ndarray,
    ) -> np.ndarray:
        result = self.robot.get_link_coms(link_indices=[base_link_index])
        raw_positions = result[0] if isinstance(result, tuple) else result
        com_local = _to_numpy(raw_positions).reshape(3).astype(np.float64)
        base_com_world = base_position + (_quaternion_rotation_matrix(base_orientation) @ com_local)
        jacobians = _to_numpy(self.robot.get_jacobian_matrices()).astype(np.float64)
        if jacobians.ndim == 4:
            if jacobians.shape[0] != 1:
                raise RuntimeError(f"Expected one articulation, got {jacobians.shape}")
            jacobians = jacobians[0]
        if jacobians.ndim != 3 or jacobians.shape[1] != 6:
            raise RuntimeError(f"Unexpected live Jacobian shape: {jacobians.shape}")
        base_jacobian = jacobians[base_jacobian_row]
        tcp_jacobian = base_jacobian[:, arm_indices].copy()
        offset_world = tcp_position - base_com_world
        tcp_jacobian[:3] -= _skew(offset_world) @ tcp_jacobian[3:6]
        return tcp_jacobian

    def _announce_phase(self) -> None:
        _log(f"[ProM6ClothingDrag] Phase {self.phase.name}: {PHASE_LABELS[self.phase]}")

    def _set_phase(self, phase: Phase) -> None:
        self.phase = phase
        self.phase_time = 0.0
        self.phase_frame = 0
        self.stable_time = 0.0
        self._right_grasp_z_refreshed = False
        self._right_grasp_dip_m = 0.0
        self._left_grasp_dip_m = 0.0
        self.left_retract_target = None
        self._left_retract_waypoints = []
        self._left_retract_i = 0
        self._left_retract_hold_s = 0.0
        self._left_return_clear_target = None
        self._left_return_cartesian_done = False
        self._left_return_joint_elapsed = 0.0
        self._left_close_xy_refreshed = False
        self._right_close_elapsed = 0.0
        self._announce_phase()

    def _current_tcp(self) -> tuple[np.ndarray, np.ndarray]:
        if self.usd_kinematics is None or self.left_commanded_q is None:
            raise RuntimeError("Authoritative left-arm FK is unavailable")
        position, orientation, _, _ = self.usd_kinematics.forward_and_jacobian(
            self.left_commanded_q
        )
        return position, orientation

    def _current_right_tcp(self) -> tuple[np.ndarray, np.ndarray]:
        if self.right_usd_kinematics is None or self.right_commanded_q is None:
            raise RuntimeError("Authoritative right-arm FK is unavailable")
        position, orientation, _, _ = self.right_usd_kinematics.forward_and_jacobian(
            self.right_commanded_q
        )
        return position, orientation

    def _write_left_q(self, target: np.ndarray) -> None:
        if self.left_arm_lower_limits is None or self.left_arm_upper_limits is None:
            raise RuntimeError("Left-arm limits are unavailable")
        value = np.clip(
            np.asarray(target, dtype=np.float64).reshape(7),
            self.left_arm_lower_limits,
            self.left_arm_upper_limits,
        )
        if self.left_commanded_q is not None:
            self.last_joint_step = float(np.linalg.norm(value - self.left_commanded_q))
        self.robot.set_dof_positions(value[None, :], dof_indices=self.left_arm_indices)
        self.robot.set_dof_velocities(
            np.zeros((1, 7), dtype=np.float64),
            dof_indices=self.left_arm_indices,
        )
        self.robot.set_dof_position_targets(value[None, :], dof_indices=self.left_arm_indices)
        self.left_commanded_q = value.copy()

    def _write_right_ready(self) -> None:
        self._write_right_q(RIGHT_READY_JOINTS)

    def _write_right_q(self, target: np.ndarray) -> None:
        if (
            self.right_arm_lower_limits is None
            or self.right_arm_upper_limits is None
            or len(self.right_arm_indices) != 7
        ):
            raise RuntimeError("Right-arm ready hold is unavailable")
        value = np.clip(
            np.asarray(target, dtype=np.float64).reshape(7),
            self.right_arm_lower_limits,
            self.right_arm_upper_limits,
        )
        if self.right_commanded_q is not None:
            self.last_joint_step = float(np.linalg.norm(value - self.right_commanded_q))
        self.robot.set_dof_positions(value[None, :], dof_indices=self.right_arm_indices)
        self.robot.set_dof_velocities(
            np.zeros((1, 7), dtype=np.float64),
            dof_indices=self.right_arm_indices,
        )
        self.robot.set_dof_position_targets(value[None, :], dof_indices=self.right_arm_indices)
        self.right_commanded_q = value.copy()

    def _slew_joint_target(self, desired: np.ndarray, dt: float) -> float:
        if self.left_commanded_q is None:
            raise RuntimeError("Left-arm command state is unavailable")
        max_step = LEFT_MAX_JOINT_SPEED_RAD_S * dt
        next_q = self.left_commanded_q + np.clip(
            np.asarray(desired, dtype=np.float64).reshape(7) - self.left_commanded_q,
            -max_step,
            max_step,
        )
        step_norm = float(np.linalg.norm(next_q - self.left_commanded_q))
        self._write_left_q(next_q)
        return step_norm

    def _slew_right_joint_target(self, desired: np.ndarray, dt: float) -> float:
        if self.right_commanded_q is None:
            raise RuntimeError("Right-arm command state is unavailable")
        max_step = RIGHT_MAX_JOINT_SPEED_RAD_S * dt
        next_q = self.right_commanded_q + np.clip(
            np.asarray(desired, dtype=np.float64).reshape(7) - self.right_commanded_q,
            -max_step,
            max_step,
        )
        step_norm = float(np.linalg.norm(next_q - self.right_commanded_q))
        self._write_right_q(next_q)
        return step_norm

    def _command_pose(
        self,
        target_position: np.ndarray,
        dt: float,
    ) -> tuple[float, float, float]:
        if self.usd_kinematics is None or self.left_commanded_q is None:
            raise RuntimeError("Online IK is unavailable")
        current_position, current_orientation, jacobian, _ = (
            self.usd_kinematics.forward_and_jacobian(self.left_commanded_q)
        )
        error = _pose_error(
            current_position,
            current_orientation,
            target_position,
            self.grasp_orientation,
        )
        delta = _dls_joint_delta(
            jacobian,
            error,
            damping=ONLINE_DLS_DAMPING,
            maximum_step=IK_MAX_JOINT_STEP_RAD,
        )
        max_step = LEFT_MAX_JOINT_SPEED_RAD_S * dt
        delta = np.clip(delta, -max_step, max_step)
        next_q = self.left_commanded_q + delta
        self._write_left_q(next_q)
        self.last_tcp_error = float(np.linalg.norm(error[:3]))
        return (
            self.last_tcp_error,
            float(np.linalg.norm(error[3:])),
            float(np.linalg.norm(delta)),
        )

    def _command_right_pose(
        self,
        target_position: np.ndarray,
        dt: float,
    ) -> tuple[float, float, float]:
        if self.right_usd_kinematics is None or self.right_commanded_q is None:
            raise RuntimeError("Right-arm online IK is unavailable")
        current_position, current_orientation, jacobian, _ = (
            self.right_usd_kinematics.forward_and_jacobian(self.right_commanded_q)
        )
        error = _pose_error(
            current_position,
            current_orientation,
            target_position,
            self.grasp_orientation,
        )
        delta = _dls_joint_delta(
            jacobian,
            error,
            damping=ONLINE_DLS_DAMPING,
            maximum_step=IK_MAX_JOINT_STEP_RAD,
        )
        max_step = RIGHT_MAX_JOINT_SPEED_RAD_S * dt
        delta = np.clip(delta, -max_step, max_step)
        self._write_right_q(self.right_commanded_q + delta)
        self.last_tcp_error = float(np.linalg.norm(error[:3]))
        return (
            self.last_tcp_error,
            float(np.linalg.norm(error[3:])),
            float(np.linalg.norm(delta)),
        )

    def _set_gripper_target(self, position: float, *, snap: bool = False) -> None:
        if self.left_gripper_index is None:
            raise RuntimeError("Left gripper DOF is unavailable")
        target = float(
            np.clip(
                position,
                self.left_gripper_lower_limit,
                self.left_gripper_upper_limit,
            )
        )
        value = np.array([[target]], dtype=np.float64)
        if snap:
            self.robot.set_dof_positions(value, dof_indices=[self.left_gripper_index])
            self.robot.set_dof_velocities(
                np.zeros((1, 1), dtype=np.float64),
                dof_indices=[self.left_gripper_index],
            )
        self.robot.set_dof_position_targets(value, dof_indices=[self.left_gripper_index])

    def _set_right_gripper_target(self, position: float, *, snap: bool = False) -> None:
        if self.right_gripper_index is None:
            raise RuntimeError("Right gripper DOF is unavailable")
        target = float(
            np.clip(
                position,
                self.right_gripper_lower_limit,
                self.right_gripper_upper_limit,
            )
        )
        value = np.array([[target]], dtype=np.float64)
        if snap:
            self.robot.set_dof_positions(value, dof_indices=[self.right_gripper_index])
            self.robot.set_dof_velocities(
                np.zeros((1, 1), dtype=np.float64),
                dof_indices=[self.right_gripper_index],
            )
        self.robot.set_dof_position_targets(value, dof_indices=[self.right_gripper_index])

    def _deformable_nodes(self) -> np.ndarray:
        if self._cached_nodes is not None:
            return self._cached_nodes
        if self.newton_world is None:
            raise RuntimeError("Newton sorting world is unavailable")
        positions = np.asarray(self.newton_world.surface_positions(), dtype=np.float64)
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise RuntimeError(f"Invalid Newton surface positions: {positions.shape}")
        span = np.max(positions, axis=0) - np.min(positions, axis=0)
        if np.any(span > MAX_DEFORMABLE_SPAN_M):
            raise RuntimeError(f"Deformable surface is unstable: span={span}")
        self._cached_nodes = positions
        return positions

    def _newton_surface_velocities(self) -> np.ndarray:
        if self.newton_world is None:
            raise RuntimeError("Newton sorting world is unavailable")
        return np.asarray(
            self.newton_world.particle_velocities()[: self.newton_world.surface_count],
            dtype=np.float64,
        )

    def _parcel_com(self) -> np.ndarray:
        if self.newton_world is None:
            raise RuntimeError("Newton sorting world is unavailable")
        return np.asarray(self.newton_world.com(), dtype=np.float64).reshape(3)

    def _require_lifted(self, baseline_z: float | None, label: str) -> None:
        if baseline_z is None:
            raise RuntimeError(f"{label} lift has no COM baseline")
        com = self._parcel_com()
        lifted = float(com[2] - baseline_z)
        contacts = self.newton_world.contact_breakdown() if self.newton_world is not None else {}
        _log(f"[ProM6ClothingDrag] {label} lift COM delta_z={lifted:.4f} m contacts={contacts}")
        if lifted < self.min_lift_height_m:
            raise RuntimeError(
                f"{label} contact grasp failed to lift the parcel: "
                f"delta_z={lifted:.4f} m < {self.min_lift_height_m:.4f} m"
            )
        if label == "left":
            world_n = int(contacts.get("world", 0) or 0)
            finger_n = int(contacts.get("finger", 0) or 0)
            if world_n > LEFT_LIFT_MAX_WORLD_CONTACTS:
                raise RuntimeError(
                    "left contact grasp lifted a flap while the parcel stayed on "
                    f"the infeed: world={world_n}, finger={finger_n}, "
                    f"delta_z={lifted:.4f} m"
                )

    def _apply_live_grasp_geometry(self, minimum: np.ndarray, maximum: np.ndarray) -> None:
        """落稳后按 Newton AABB 重算左抓点,TCP 压进当前袋面而不是 USD 静止高度."""
        if self.support_bounds is None:
            raise RuntimeError("Support bounds are unavailable after settle")
        self.clothing_bounds = (
            np.asarray(minimum, dtype=np.float64).reshape(3).copy(),
            np.asarray(maximum, dtype=np.float64).reshape(3).copy(),
        )
        self.geometry = calculate_grasp_geometry(self.clothing_bounds, self.support_bounds)
        self.grasp_point = self.geometry["grasp_point"].copy()
        self.pregrasp_point = self.geometry["pregrasp_point"].copy()
        self.attachment_center = self.geometry["attachment_center"].copy()
        self.support_center = self.geometry["support_center"].copy()
        _log(
            "[ProM6ClothingDrag] Live Newton grasp: "
            f"grasp={_array_string(self.grasp_point)}, "
            f"pregrasp={_array_string(self.pregrasp_point)}, "
            f"patch={_array_string(self.attachment_center)}"
        )

    def _node_aabb(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nodes = self._deformable_nodes()
        minimum = np.min(nodes, axis=0)
        maximum = np.max(nodes, axis=0)
        return minimum, maximum, 0.5 * (minimum + maximum)

    def _capture_replay_frame(self, dt: float) -> None:
        if self.replay_recorder is None:
            raise RuntimeError("Successful-operation replay recorder is unavailable")
        anchor_position = None
        if self.attachment_active:
            if self.anchor is None:
                raise RuntimeError("Active attachment has no anchor during recording")
            anchor_position = get_xform_world_translation(self.anchor)
        self.replay_recorder.capture(
            dt,
            self.phase,
            attachment_active=self.attachment_active,
            anchor_position=anchor_position,
        )

    def _log_progress(self) -> None:
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES != 0:
            return
        _, _, center = self._node_aabb()
        center_error = _xy_center_error(center, self.support_center)
        tcp_error = (
            "n/a" if not math.isfinite(self.last_tcp_error) else f"{self.last_tcp_error:.5f} m"
        )
        _log(
            "[ProM6ClothingDrag][status] "
            f"phase={self.phase.name}, tcp_error={tcp_error}, "
            f"joint_step={self.last_joint_step:.5f} rad, "
            f"center_error={center_error:.4f} m"
        )

    def _update_anchor_from_actual_tcp(self) -> None:
        if not self.attachment_active:
            return
        if self.anchor is None or self.grasp_point is None or self.attachment_center is None:
            raise RuntimeError("Attachment anchor geometry is incomplete")
        tcp_position, _ = self._current_tcp()
        set_xform_world_translation(
            self.anchor,
            self.attachment_center + (tcp_position - self.grasp_point),
        )

    def _update_shared_anchor_from_right_tcp(self) -> None:
        if not self.attachment_active:
            return
        if (
            self.anchor is None
            or self.right_handoff_anchor_position is None
            or self.right_handoff_tcp_position is None
        ):
            raise RuntimeError("Shared attachment handoff geometry is incomplete")
        tcp_position, _ = self._current_right_tcp()
        set_xform_world_translation(
            self.anchor,
            self.right_handoff_anchor_position + (tcp_position - self.right_handoff_tcp_position),
        )

    def _release_attachment(self) -> None:
        if self.attachment_active:
            prim = self.stage.GetPrimAtPath(ATTACHMENT_PATH)
            if prim.IsValid():
                self.stage.RemovePrim(ATTACHMENT_PATH)
            self.attachment_active = False
            _log("[ProM6ClothingDrag] Shared surface attachment removed")
        anchor_prim = self.stage.GetPrimAtPath(ANCHOR_PATH)
        if anchor_prim.IsValid():
            self.stage.RemovePrim(ANCHOR_PATH)
        self.anchor = None

    def _step_settle(self) -> None:
        self._write_left_q(self.left_commanded_q)
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        if self.phase_time >= self.timing.settle_seconds:
            minimum, maximum, center = self._node_aabb()
            self.initial_node_center = center.copy()
            self.initial_node_height = float(maximum[2] - minimum[2])
            _log(
                "[ProM6ClothingDrag] Settled nodes: "
                f"min={_array_string(minimum)}, max={_array_string(maximum)}, "
                f"center={_array_string(center)}, "
                f"com={_array_string(self._parcel_com())}"
            )
            self._apply_live_grasp_geometry(minimum, maximum)
            self._set_phase(Phase.WAITING)

    def _step_search_pregrasp_ik(self) -> None:
        if (
            self.usd_kinematics is None
            or self.pregrasp_point is None
            or self.left_arm_lower_limits is None
            or self.left_arm_upper_limits is None
        ):
            raise RuntimeError("Pregrasp IK inputs are unavailable")
        self._write_left_q(self.left_commanded_q)
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        selected, diagnostics = solve_pregrasp_ik_42_seeds(
            self.usd_kinematics,
            self.pregrasp_point,
            self.grasp_orientation,
            self.left_arm_lower_limits,
            self.left_arm_upper_limits,
            IK_JOINT_LIMIT_MARGIN_RAD,
            search_timeout_seconds=self.timing.ik_search_timeout_seconds,
            ready_seed=LEFT_IK_ELBOW_OUT_SEED,
            outward_xy=LEFT_ELBOW_OUTWARD_XY,
        )
        self.pregrasp_joint_target = selected.copy()
        self.pregrasp_start_q = self.left_commanded_q.copy()
        _log(
            "[ProM6ClothingDrag][IK search] selected branch: "
            f"seed={int(diagnostics['seed_index'])}, "
            f"valid={int(diagnostics['valid_solution_count'])}/"
            f"{int(diagnostics['candidate_count'])}, "
            f"position={1000.0 * float(diagnostics['position_error']):.3f} mm, "
            f"orientation={math.degrees(float(diagnostics['orientation_error'])):.3f} deg, "
            f"elbow_out={float(diagnostics['elbow_outboard_m']):.3f} m, "
            f"q={_array_string(selected, 5)}"
        )
        self._set_phase(Phase.MOVE_TO_PREGRASP)

    def _step_joint_segment(
        self,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        dt: float,
        next_phase: Phase,
    ) -> None:
        progress = _timed_smoothstep(self.phase_time, duration)
        desired = (1.0 - progress) * start + progress * target
        step_norm = self._slew_joint_target(desired, dt)
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        joint_error = float(np.max(np.abs(self.left_commanded_q - target)))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                f"[ProM6ClothingDrag] {self.phase.name}: progress={progress:.3f}, "
                f"joint_error={joint_error:.5f} rad, step_norm={step_norm:.5f}"
            )
        if self.phase_time >= duration and joint_error <= 1.0e-4:
            self._set_phase(next_phase)

    def _step_approach(self, dt: float) -> None:
        progress = _timed_smoothstep(self.phase_time, self.timing.left_approach_seconds)
        target = self.pregrasp_point - np.array(
            [0.0, 0.0, progress * PREGRASP_CLEARANCE_M], dtype=np.float64
        )
        position_error, orientation_error, step_norm = self._command_pose(target, dt)
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] approach: "
                f"progress={progress:.3f}, position_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"step_norm={step_norm:.5f}"
            )
        if (
            self.phase_time >= self.timing.left_approach_seconds
            and position_error <= TCP_APPROACH_POSITION_TOLERANCE_M
            and orientation_error <= TCP_APPROACH_ORIENTATION_TOLERANCE_RAD
        ):
            self._set_phase(Phase.CLOSE_GRIPPER)

    def _step_close_gripper(self, dt: float) -> None:
        if not self._left_close_xy_refreshed:
            minimum, maximum, _ = self._node_aabb()
            self._apply_live_grasp_geometry(minimum, maximum)
            self._left_close_xy_refreshed = True
            _log(
                "[ProM6ClothingDrag] Left grasp refreshed before close: "
                f"grasp={_array_string(self.grasp_point)}"
            )
        position_error, orientation_error, _ = self._command_pose(self.grasp_point, dt)
        fraction = _timed_smoothstep(self.phase_time, self.timing.left_close_seconds)
        target = GRIPPER_OPEN_POSITION_RAD + fraction * (
            GRIPPER_CLOSED_POSITION_RAD - GRIPPER_OPEN_POSITION_RAD
        )
        self._set_gripper_target(target, snap=True)
        if self.phase_time >= self.timing.left_close_seconds:
            if (
                position_error > TCP_GRASP_POSITION_TOLERANCE_M
                or orientation_error > TCP_GRASP_ORIENTATION_TOLERANCE_RAD
            ):
                raise RuntimeError(
                    "Grasp pose left tolerance during gripper closure: "
                    f"position={position_error:.5f}, orientation={orientation_error:.5f}"
                )
            self._set_phase(Phase.GRIP_SETTLE)

    def _step_grip_settle(self, dt: float) -> None:
        position_error, orientation_error, _ = self._command_pose(self.grasp_point, dt)
        self._set_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        gripper_error = abs(
            float(self._current_dof_positions()[self.left_gripper_index])
            - float(
                np.clip(
                    GRIPPER_CLOSED_POSITION_RAD,
                    self.left_gripper_lower_limit,
                    self.left_gripper_upper_limit,
                )
            )
        )
        pose_ok = (
            position_error <= TCP_GRASP_POSITION_TOLERANCE_M
            and orientation_error <= TCP_GRASP_ORIENTATION_TOLERANCE_RAD
        )
        finger_n = self._finger_contacts()
        if (
            pose_ok
            and finger_n < LEFT_MIN_FINGER_CONTACTS
            and self._left_grasp_dip_m < LEFT_GRASP_DIP_MAX_M
            and self.grasp_point is not None
        ):
            step = min(LEFT_GRASP_DIP_STEP_M, LEFT_GRASP_DIP_MAX_M - self._left_grasp_dip_m)
            self.grasp_point = np.asarray(self.grasp_point, dtype=np.float64).copy()
            self.grasp_point[2] -= step
            self._left_grasp_dip_m += step
            _log(
                "[ProM6ClothingDrag] Left grasp dip: "
                f"finger={finger_n}, dip={self._left_grasp_dip_m:.3f} m, "
                f"z={self.grasp_point[2]:.4f}"
            )
        holding = finger_n >= LEFT_MIN_FINGER_CONTACTS
        converged = pose_ok and gripper_error <= GRIPPER_POSITION_TOLERANCE_RAD and holding
        self.stable_time = _accumulate_stable_time(self.stable_time, converged, dt)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            debug = ""
            if finger_n < LEFT_MIN_FINGER_CONTACTS and self.newton_world is not None:
                debug = f", fingers={self.newton_world.finger_debug()}"
            _log(
                "[ProM6ClothingDrag] left grip settle: "
                f"tcp_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"gripper_error={gripper_error:.5f} rad, "
                f"finger={finger_n}, "
                f"stable={self.stable_time:.3f}/"
                f"{self.timing.left_grip_stable_seconds:.3f} s{debug}"
            )
        if (
            converged
            and self.phase_time >= self.timing.left_grip_settle_seconds
            and _stable_time_reached(self.stable_time, self.timing.left_grip_stable_seconds)
        ):
            tcp_position, _ = self._current_tcp()
            self.left_lift_start = tcp_position.copy()
            self.left_lift_target = tcp_position + np.array(
                [0.0, 0.0, LOW_LIFT_HEIGHT_M], dtype=np.float64
            )
            if self.newton_world is not None:
                q = self._current_dof_positions()
                _log(
                    "[ProM6ClothingDrag] Finger/bag before lift: "
                    f"left_finger_q={float(q[self.left_gripper_index]):.4f} rad, "
                    f"{self.newton_world.finger_debug()}"
                )
            _log(
                "[ProM6ClothingDrag] Separate vertical lift: "
                f"start={_array_string(self.left_lift_start)}, "
                f"target={_array_string(self.left_lift_target)}"
            )
            self.left_lift_com_z = float(self._parcel_com()[2])
            self._set_phase(Phase.LEFT_LIFT)
        elif self.phase_time >= max(
            self.timing.left_grip_settle_seconds + 5.0,
            self.timing.left_grip_stable_seconds + 5.0,
        ):
            raise RuntimeError(
                "Grasp did not settle: "
                f"position={position_error:.5f}, orientation={orientation_error:.5f}, "
                f"gripper={gripper_error:.5f}, "
                f"finger={finger_n}"
            )

    def _step_left_lift(self, dt: float) -> None:
        if self.left_lift_start is None or self.left_lift_target is None:
            raise RuntimeError("Left lift geometry is unavailable")
        progress = _timed_smoothstep(self.phase_time, self.timing.left_lift_seconds)
        target = (1.0 - progress) * self.left_lift_start + progress * self.left_lift_target
        self._command_pose(target, dt)
        self._set_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_anchor_from_actual_tcp()
        # Gate against the final lift target using the newly commanded FK pose.
        tcp_position, tcp_orientation = self._current_tcp()
        error = _pose_error(
            tcp_position, tcp_orientation, self.left_lift_target, self.grasp_orientation
        )
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] left lift: "
                f"progress={progress:.3f}, position_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg"
            )
        if (
            self.phase_time >= self.timing.left_lift_seconds
            and position_error <= TCP_LIFT_POSITION_TOLERANCE_M
            and orientation_error <= TCP_LIFT_ORIENTATION_TOLERANCE_RAD
        ):
            self._require_lifted(self.left_lift_com_z, "left")
            _, _, center = self._node_aabb()
            self.drag_start = tcp_position.copy()
            self.drag_target = self.drag_start.copy()
            self.drag_target[:2] += self.support_center[:2] - center[:2]
            _log(
                "[ProM6ClothingDrag] Frozen horizontal drag vector: "
                f"node_center={_array_string(center)}, "
                f"delta={_array_string(self.drag_target - self.drag_start)}, "
                f"target={_array_string(self.drag_target)}"
            )
            self._set_phase(Phase.DRAG)

    def _step_drag(self, dt: float) -> None:
        progress = _timed_smoothstep(self.phase_time, self.timing.left_drag_seconds)
        target = (1.0 - progress) * self.drag_start + progress * self.drag_target
        position_error, orientation_error, step_norm = self._command_pose(target, dt)
        self._set_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_anchor_from_actual_tcp()
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _, _, center = self._node_aabb()
            center_error = _xy_center_error(center, self.support_center)
            _log(
                "[ProM6ClothingDrag] drag: "
                f"progress={progress:.3f}, tcp_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"joint_step={step_norm:.5f}, center_error={center_error:.4f} m"
            )
        if (
            self.phase_time >= self.timing.left_drag_seconds
            and position_error <= DRAG_ENDPOINT_TOLERANCE_M
        ):
            _, _, center = self._node_aabb()
            center_error = _xy_center_error(center, self.support_center)
            if center_error > 0.25:
                raise RuntimeError(
                    "Left drag did not carry the parcel toward the platform: "
                    f"center_error={center_error:.4f} m"
                )
            self.correction_target = self.drag_target.copy()
            self.correction_origin = self.drag_target.copy()
            self._set_phase(Phase.CENTER_CORRECT)

    def _correct_center_xy(self, dt: float, target: np.ndarray) -> tuple[float, float]:
        _, _, center = self._node_aabb()
        error_xy = self.support_center[:2] - center[:2]
        center_error = float(np.linalg.norm(error_xy))
        if not _center_is_within_tolerance(center, self.support_center):
            target[:2] += _clamp_vector_norm(error_xy, CENTER_CORRECTION_STEP_M)
        position_error, _, _ = self._command_pose(target, dt)
        self._set_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_anchor_from_actual_tcp()
        return center_error, position_error

    def _step_center_correct(self, dt: float) -> None:
        center_error, tcp_error = self._correct_center_xy(dt, self.correction_target)
        correction_distance = float(
            np.linalg.norm(self.correction_target[:2] - self.correction_origin[:2])
        )
        if correction_distance > MAX_CENTER_CORRECTION_M:
            raise RuntimeError(f"Center correction exceeded {MAX_CENTER_CORRECTION_M:.3f} m")
        within_tolerance = center_error <= CENTER_RELEASE_TOLERANCE_M
        self.stable_time = _accumulate_stable_time(self.stable_time, within_tolerance, dt)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] center correction: "
                f"center_error={center_error:.4f} m, tcp_error={tcp_error:.4f} m, "
                f"correction={correction_distance:.4f} m, "
                f"stable={self.stable_time:.3f}/"
                f"{self.timing.center_stable_seconds:.3f} s"
            )
        if within_tolerance and _stable_time_reached(
            self.stable_time, self.timing.center_stable_seconds
        ):
            _, _, center = self._node_aabb()
            lower_offset = float(LEFT_LOWER_DISTANCE_OFFSET_M)
            if not math.isfinite(lower_offset):
                raise ValueError("LEFT_LOWER_DISTANCE_OFFSET_M must be finite")
            desired_center_z = (
                self.support_bounds[1][2]
                + 0.5 * self.initial_node_height
                + PLACEMENT_CLEARANCE_M
                - lower_offset
            )
            self.lower_start = self.correction_target.copy()
            self.setdown_target = self.correction_target.copy()
            self.setdown_target[2] += desired_center_z - center[2]
            vertical_displacement = float(self.setdown_target[2] - self.lower_start[2])
            if abs(vertical_displacement) > 0.350:
                raise RuntimeError(
                    "Calculated setdown exceeds the 350 mm vertical safety "
                    f"bound: displacement={vertical_displacement:.4f} m"
                )
            _log(
                "[ProM6ClothingDrag] Setdown target: "
                f"current_center_z={center[2]:.4f}, desired_center_z={desired_center_z:.4f}, "
                f"lower_offset={lower_offset:.4f} m, "
                f"vertical_displacement={vertical_displacement:.4f} m, "
                f"tcp={_array_string(self.setdown_target)}"
            )
            self._set_phase(Phase.LOWER)
        elif self.phase_time >= self.phase_timeouts[Phase.CENTER_CORRECT]:
            raise RuntimeError(f"Center correction timed out: error={center_error:.4f} m")

    def _step_lower(self, dt: float) -> None:
        progress = _timed_smoothstep(self.phase_time, self.timing.left_lower_seconds)
        target = (1.0 - progress) * self.lower_start + progress * self.setdown_target
        position_error, orientation_error, _ = self._command_pose(target, dt)
        self._set_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_anchor_from_actual_tcp()
        if (
            self.phase_time >= self.timing.left_lower_seconds
            and position_error <= DRAG_ENDPOINT_TOLERANCE_M
        ):
            self._set_phase(Phase.PLACE_SETTLE)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] lower: "
                f"progress={progress:.3f}, position_error={position_error:.4f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg"
            )

    def _step_place_settle(self, dt: float) -> None:
        center_error, tcp_error = self._correct_center_xy(dt, self.setdown_target)
        correction_distance = float(
            np.linalg.norm(self.setdown_target[:2] - self.correction_origin[:2])
        )
        if correction_distance > MAX_CENTER_CORRECTION_M:
            raise RuntimeError(
                "Placement correction exceeded its safety bound: "
                f"distance={correction_distance:.4f} m"
            )
        within_tolerance = center_error <= CENTER_RELEASE_TOLERANCE_M
        self.stable_time = _accumulate_stable_time(self.stable_time, within_tolerance, dt)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] place settle: "
                f"center_error={center_error:.4f} m, tcp_error={tcp_error:.4f} m, "
                f"stable={self.stable_time:.3f}/"
                f"{self.timing.place_stable_seconds:.3f} s"
            )
        if (
            within_tolerance
            and self.phase_time >= self.timing.left_place_settle_seconds
            and _stable_time_reached(self.stable_time, self.timing.place_stable_seconds)
        ):
            self._teleport_pad_retract(LEFT_PAD_RETRACT_M)
            self._set_phase(Phase.LEFT_OPEN)
        elif self.phase_time >= self.phase_timeouts[Phase.PLACE_SETTLE]:
            raise RuntimeError(
                f"Clothing is not centered closely enough to release: error={center_error:.4f} m"
            )

    def _step_left_open(self, dt: float) -> None:
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD, snap=True)
        if self.newton_world is not None:
            self.newton_world.set_pad_retract(LEFT_PAD_RETRACT_M)
        if self.phase_time < self.timing.left_open_seconds:
            self._write_left_q(self.left_commanded_q)
            return
        if self.usd_kinematics is None or self.left_commanded_q is None:
            raise RuntimeError("Left retract IK is unavailable after release")
        tcp_position, _ = self._current_tcp()
        if self.left_retract_target is None:
            self.left_retract_target = np.array(
                [
                    float(tcp_position[0]),
                    float(tcp_position[1]),
                    float(tcp_position[2]) + LEFT_RETRACT_HEIGHT_M,
                ],
                dtype=np.float64,
            )
            _log(
                "[ProM6ClothingDrag] Left retract after release: "
                f"from={_array_string(tcp_position)}, "
                f"to={_array_string(self.left_retract_target)}, "
                f"finger={self._finger_contacts()}"
            )
        position_error, _, _ = self._command_pose(self.left_retract_target, dt)
        if position_error > TCP_LIFT_POSITION_TOLERANCE_M:
            return
        self._left_retract_hold_s += dt
        if self._left_retract_hold_s < LEFT_RETRACT_HOLD_S:
            self._write_left_q(self.left_commanded_q)
            if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
                _log(
                    "[ProM6ClothingDrag] Left retract hold: "
                    f"finger={self._finger_contacts()}, t={self._left_retract_hold_s:.2f} s"
                )
            return
        _log(
            "[ProM6ClothingDrag] Left retract reached height: "
            f"finger={self._finger_contacts()}, pads_clear={self._left_pads_clear_of_bag()}"
        )
        if self.newton_world is not None:
            _log(f"[ProM6ClothingDrag] Left retract pads: {self.newton_world.finger_debug()}")
        self.left_return_start_q = self.left_commanded_q.copy()
        self._set_phase(Phase.LEFT_RETURN_READY)

    def _step_left_return_ready(self, dt: float) -> None:
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD, snap=True)
        tcp_position, _ = self._current_tcp()
        if self._left_return_clear_target is None:
            self._left_return_clear_target = np.array(
                [
                    float(tcp_position[0]) + LEFT_RETURN_CLEAR_X_OFFSET_M,
                    float(tcp_position[1]),
                    float(tcp_position[2]),
                ],
                dtype=np.float64,
            )
            _log(
                "[ProM6ClothingDrag] Left return clear: "
                f"from={_array_string(tcp_position)}, "
                f"to={_array_string(self._left_return_clear_target)}"
            )
        if not self._left_return_cartesian_done:
            finger_n = self._finger_contacts()
            pads_clear = self._left_pads_clear_of_bag()
            if finger_n > 30 and not pads_clear:
                self._write_left_q(self.left_commanded_q)
                if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
                    _log(
                        "[ProM6ClothingDrag] Left return wait contacts: "
                        f"finger={finger_n}, pads_clear={pads_clear}"
                    )
                return
            if finger_n > 30 and pads_clear:
                _log(
                    "[ProM6ClothingDrag] Left return skip stale contacts: "
                    f"finger={finger_n}, pads_clear={pads_clear}"
                )
            position_error, _, _ = self._command_pose(self._left_return_clear_target, dt)
            if position_error <= TCP_LIFT_POSITION_TOLERANCE_M:
                self._left_return_cartesian_done = True
                self.left_return_start_q = self.left_commanded_q.copy()
                _log("[ProM6ClothingDrag] Left return clear reached, joint-slew to ready")
            return
        if self.left_return_start_q is None:
            raise RuntimeError("Left return start joints are unavailable after clear")
        self._left_return_joint_elapsed += dt
        progress = _timed_smoothstep(
            self._left_return_joint_elapsed, self.timing.left_return_ready_seconds
        )
        desired = (1.0 - progress) * self.left_return_start_q + progress * LEFT_READY_JOINTS
        step_norm = self._slew_joint_target(desired, dt)
        joint_error = float(np.max(np.abs(self.left_commanded_q - LEFT_READY_JOINTS)))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] left return: "
                f"progress={progress:.3f}, joint_error={joint_error:.5f} rad, "
                f"step_norm={step_norm:.5f}"
            )
        if (
            self._left_return_joint_elapsed >= self.timing.left_return_ready_seconds
            and joint_error <= 1.0e-4
        ):
            self._write_left_q(LEFT_READY_JOINTS)
            self._set_phase(Phase.PREPARE_RIGHT_HANDOFF)

    def _right_grasp_patch(self, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
        """右抓点:X 从袋心偏 -X;Y 收向平台中心/袋 +Y 边,避免过深 -Y 让右 IK 无解."""
        min_x = float(minimum[0])
        max_x = float(maximum[0])
        min_y = float(minimum[1])
        max_y = float(maximum[1])
        center_x = 0.5 * (min_x + max_x)
        half_x = 0.5 * (max_x - min_x)
        grasp_x = center_x - RIGHT_GRASP_FROM_CENTER_FRAC * max(half_x, 0.04)
        grasp_x = min(grasp_x, max_x - 0.03)
        grasp_x = max(grasp_x, min_x + RIGHT_GRASP_EDGE_INSET_M)
        inset_y = RIGHT_GRASP_EDGE_INSET_M
        y_lo = min_y + inset_y
        y_hi = max_y - inset_y
        if y_hi < y_lo:
            y_lo, y_hi = y_hi, y_lo
        preferred_y = y_hi
        if self.support_center is not None:
            preferred_y = float(self.support_center[1])
        grasp_y = float(np.clip(preferred_y, y_lo, y_hi))
        return np.array(
            [
                grasp_x,
                grasp_y,
                float(maximum[2]) - RIGHT_GRASP_BITE_M,
            ],
            dtype=np.float64,
        )

    def _refresh_right_grasp_from_live_nodes(self, *, xy: bool) -> None:
        """按当前 Newton AABB 更新右抓 TCP;合爪时只跟高度,避免袋瘪后擦顶滑脱."""
        minimum, maximum, _ = self._node_aabb()
        geometry = calculate_shared_attachment_grasp(self._right_grasp_patch(minimum, maximum))
        grasp = geometry["grasp_point"].copy()
        pregrasp = geometry["pregrasp_point"].copy()
        if xy or self.right_grasp_point is None:
            self.right_grasp_point = grasp
            self.right_pregrasp_point = pregrasp
        else:
            current = np.asarray(self.right_grasp_point, dtype=np.float64).reshape(3).copy()
            current[2] = float(grasp[2])
            self.right_grasp_point = current
            if self.right_pregrasp_point is not None:
                pre = np.asarray(self.right_pregrasp_point, dtype=np.float64).reshape(3).copy()
                pre[2] = float(pregrasp[2])
                self.right_pregrasp_point = pre
        _log(
            "[ProM6ClothingDrag] Right grasp refreshed: "
            f"xy={xy}, grasp={_array_string(self.right_grasp_point)}, "
            f"nodes_min={_array_string(minimum)}, nodes_max={_array_string(maximum)}"
        )

    def _step_prepare_right_handoff(self) -> None:
        self._teleport_pad_retract(0.0)
        self._write_left_q(LEFT_READY_JOINTS)
        self._write_right_ready()
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        minimum, maximum, center = self._node_aabb()
        geometry = calculate_shared_attachment_grasp(self._right_grasp_patch(minimum, maximum))
        self.right_handoff_anchor_position = geometry["attachment_center"].copy()
        self.right_grasp_point = geometry["grasp_point"].copy()
        self.right_pregrasp_point = geometry["pregrasp_point"].copy()
        self.belt_bounds = get_world_bounds(self.stage.GetPrimAtPath(BELT_PATH))
        if not self.belt_bounds[0][1] <= center[1] <= self.belt_bounds[1][1]:
            raise RuntimeError(
                "Clothing center Y is outside the belt footprint and a "
                "world -X drag cannot correct it: "
                f"center_y={center[1]:.4f}, "
                f"belt_y=[{self.belt_bounds[0][1]:.4f}, "
                f"{self.belt_bounds[1][1]:.4f}]"
            )
        _log(
            "[ProM6ClothingDrag] Right grasp from live Newton AABB: "
            f"grasp={_array_string(self.right_grasp_point)}, "
            f"pregrasp={_array_string(self.right_pregrasp_point)}, "
            f"nodes_min={_array_string(minimum)}, "
            f"nodes_max={_array_string(maximum)}"
        )
        self._set_phase(Phase.RIGHT_SEARCH_IK)

    def _right_pregrasp_candidates(self) -> list[np.ndarray]:
        """主抓点 + 袋上更靠 +Y 的备选,给右 IK 一次退路."""
        if self.right_pregrasp_point is None:
            raise RuntimeError("Right pregrasp point is unavailable")
        primary = np.asarray(self.right_pregrasp_point, dtype=np.float64).reshape(3)
        candidates = [primary.copy()]
        minimum, maximum, _ = self._node_aabb()
        inset = RIGHT_GRASP_EDGE_INSET_M
        y_hi = float(maximum[1]) - inset
        y_lo = float(minimum[1]) + inset
        if y_hi < y_lo:
            y_lo, y_hi = y_hi, y_lo
        for y in (y_hi, 0.5 * (y_lo + y_hi)):
            alt = primary.copy()
            alt[1] = float(np.clip(y, y_lo, y_hi))
            if min(float(np.linalg.norm(alt[:2] - item[:2])) for item in candidates) > 0.02:
                candidates.append(alt)
        return candidates

    def _step_right_search_ik(self) -> None:
        if (
            self.right_usd_kinematics is None
            or self.right_pregrasp_point is None
            or self.right_arm_lower_limits is None
            or self.right_arm_upper_limits is None
        ):
            raise RuntimeError("Right pregrasp IK inputs are unavailable")
        self._write_left_q(LEFT_READY_JOINTS)
        self._write_right_q(self.right_commanded_q)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        last_error = "Right pregrasp IK search found no solution"
        selected: np.ndarray | None = None
        diagnostics: dict[str, float | int] | None = None
        for index, pregrasp in enumerate(self._right_pregrasp_candidates()):
            try:
                selected, diagnostics = solve_pregrasp_ik_42_seeds(
                    self.right_usd_kinematics,
                    pregrasp,
                    self.grasp_orientation,
                    self.right_arm_lower_limits,
                    self.right_arm_upper_limits,
                    IK_JOINT_LIMIT_MARGIN_RAD,
                    RIGHT_IK_PREFERRED_SEED,
                    self.timing.ik_search_timeout_seconds,
                    ready_seed=RIGHT_IK_ELBOW_OUT_SEED,
                    outward_xy=RIGHT_ELBOW_OUTWARD_XY,
                )
            except RuntimeError as exc:
                last_error = str(exc)
                _log(
                    "[ProM6ClothingDrag][right IK search] candidate failed: "
                    f"index={index}, pregrasp={_array_string(pregrasp)}, error={last_error}"
                )
                continue
            if index > 0:
                self.right_pregrasp_point = pregrasp.copy()
                if self.right_grasp_point is not None:
                    grasp = np.asarray(self.right_grasp_point, dtype=np.float64).reshape(3).copy()
                    grasp[0] = float(pregrasp[0])
                    grasp[1] = float(pregrasp[1])
                    self.right_grasp_point = grasp
            _log(
                "[ProM6ClothingDrag][right IK search] selected branch: "
                f"candidate={index}, "
                f"seed={int(diagnostics['seed_index'])}, "
                f"valid={int(diagnostics['valid_solution_count'])}/"
                f"{int(diagnostics['candidate_count'])}, "
                f"position={1000.0 * float(diagnostics['position_error']):.3f} mm, "
                f"orientation={math.degrees(float(diagnostics['orientation_error'])):.3f} deg, "
                f"elbow_out={float(diagnostics['elbow_outboard_m']):.3f} m, "
                f"q={_array_string(selected, 5)}"
            )
            break
        if selected is None or diagnostics is None:
            raise RuntimeError(last_error)
        self.right_pregrasp_joint_target = selected.copy()
        self.right_pregrasp_start_q = self.right_commanded_q.copy()
        self._set_phase(Phase.RIGHT_MOVE_TO_PREGRASP)

    def _step_right_joint_segment(
        self,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        dt: float,
        next_phase: Phase,
    ) -> None:
        progress = _timed_smoothstep(self.phase_time, duration)
        desired = (1.0 - progress) * start + progress * target
        step_norm = self._slew_right_joint_target(desired, dt)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        joint_error = float(np.max(np.abs(self.right_commanded_q - target)))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                f"[ProM6ClothingDrag] {self.phase.name}: "
                f"progress={progress:.3f}, joint_error={joint_error:.5f} rad, "
                f"step_norm={step_norm:.5f}"
            )
        if self.phase_time >= duration and joint_error <= 1.0e-4:
            self._set_phase(next_phase)

    def _step_right_approach(self, dt: float) -> None:
        if not self._right_grasp_z_refreshed:
            self._refresh_right_grasp_from_live_nodes(xy=False)
            self._right_grasp_z_refreshed = True
        progress = _timed_smoothstep(self.phase_time, self.timing.right_approach_seconds)
        target = self.right_pregrasp_point - np.array(
            [0.0, 0.0, progress * PREGRASP_CLEARANCE_M],
            dtype=np.float64,
        )
        position_error, orientation_error, step_norm = self._command_right_pose(target, dt)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] right approach: "
                f"progress={progress:.3f}, position_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"step_norm={step_norm:.5f}"
            )
        if (
            self.phase_time >= self.timing.right_approach_seconds
            and position_error <= TCP_APPROACH_POSITION_TOLERANCE_M
            and orientation_error <= TCP_APPROACH_ORIENTATION_TOLERANCE_RAD
        ):
            self._set_phase(Phase.RIGHT_CLOSE)

    def _step_right_close(self, dt: float) -> None:
        if not self._right_grasp_z_refreshed:
            self._refresh_right_grasp_from_live_nodes(xy=True)
            self._right_grasp_z_refreshed = True
        position_error, orientation_error, _ = self._command_right_pose(self.right_grasp_point, dt)
        pose_ok = (
            position_error <= TCP_GRASP_POSITION_TOLERANCE_M
            and orientation_error <= TCP_GRASP_ORIENTATION_TOLERANCE_RAD
        )
        if not pose_ok:
            self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
            self._right_close_elapsed = 0.0
            return
        self._right_close_elapsed += dt
        fraction = _timed_smoothstep(self._right_close_elapsed, self.timing.right_close_seconds)
        target = GRIPPER_OPEN_POSITION_RAD + fraction * (
            GRIPPER_CLOSED_POSITION_RAD - GRIPPER_OPEN_POSITION_RAD
        )
        self._set_right_gripper_target(target, snap=True)
        if self._right_close_elapsed >= self.timing.right_close_seconds:
            self._set_phase(Phase.RIGHT_GRIP_SETTLE)

    def _finger_contacts(self) -> int:
        if self.newton_world is None:
            return 0
        return int(self.newton_world.contact_breakdown().get("finger", 0) or 0)

    def _left_pads_clear_of_bag(self, min_gap_m: float = LEFT_PAD_CLEAR_GAP_M) -> bool:
        """左指垫中心已离开袋 AABB 时视为脱离,避免陈旧接触把回 ready 卡死."""
        if self.newton_world is None:
            return False
        dists = [
            float(row.get("dist_aabb", 0.0) or 0.0)
            for row in self.newton_world.finger_debug()[1:]
            if row.get("side") == "left"
        ]
        return bool(dists) and min(dists) >= float(min_gap_m)

    def _right_finger_contacts(self) -> int:
        return self._finger_contacts()

    def _step_right_grip_settle(self, dt: float) -> None:
        if not self._right_grasp_z_refreshed:
            self._refresh_right_grasp_from_live_nodes(xy=False)
            self._right_grasp_z_refreshed = True
        finger_n = self._right_finger_contacts()
        if (
            finger_n < RIGHT_MIN_FINGER_CONTACTS
            and self._right_grasp_dip_m < RIGHT_GRASP_DIP_MAX_M
            and self.right_grasp_point is not None
        ):
            step = min(RIGHT_GRASP_DIP_STEP_M, RIGHT_GRASP_DIP_MAX_M - self._right_grasp_dip_m)
            self.right_grasp_point = np.asarray(self.right_grasp_point, dtype=np.float64).copy()
            self.right_grasp_point[2] -= step
            self._right_grasp_dip_m += step
            _log(
                "[ProM6ClothingDrag] Right grasp dip: "
                f"finger={finger_n}, dip={self._right_grasp_dip_m:.3f} m, "
                f"z={self.right_grasp_point[2]:.4f}"
            )
        position_error, orientation_error, _ = self._command_right_pose(self.right_grasp_point, dt)
        self._set_right_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        target = float(
            np.clip(
                GRIPPER_CLOSED_POSITION_RAD,
                self.right_gripper_lower_limit,
                self.right_gripper_upper_limit,
            )
        )
        gripper_error = abs(float(self._current_dof_positions()[self.right_gripper_index]) - target)
        holding = finger_n >= RIGHT_MIN_FINGER_CONTACTS
        converged = (
            position_error <= TCP_GRASP_POSITION_TOLERANCE_M
            and orientation_error <= TCP_GRASP_ORIENTATION_TOLERANCE_RAD
            and gripper_error <= GRIPPER_POSITION_TOLERANCE_RAD
            and holding
        )
        self.stable_time = _accumulate_stable_time(self.stable_time, converged, dt)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] right grip settle: "
                f"tcp_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"gripper_error={gripper_error:.5f} rad, "
                f"finger={finger_n}, "
                f"stable={self.stable_time:.3f}/"
                f"{self.timing.right_grip_stable_seconds:.3f} s"
            )
        if (
            converged
            and self.phase_time >= self.timing.right_grip_settle_seconds
            and _stable_time_reached(self.stable_time, self.timing.right_grip_stable_seconds)
        ):
            self.right_handoff_tcp_position, _ = self._current_right_tcp()
            self.right_handoff_anchor_position = self.right_grasp_point.copy()
            self.right_lift_start = self.right_handoff_tcp_position.copy()
            self.right_lift_target = self.right_lift_start + np.array(
                [0.0, 0.0, RIGHT_LIFT_HEIGHT_M], dtype=np.float64
            )
            self.right_actual_drag_distance = 0.0
            self.belt_bounds = get_world_bounds(self.stage.GetPrimAtPath(BELT_PATH))
            _, _, center = self._node_aabb()
            if not self.belt_bounds[0][1] <= center[1] <= self.belt_bounds[1][1]:
                raise RuntimeError(
                    "Clothing center Y is outside the belt footprint at "
                    f"right-arm takeover: center_y={center[1]:.4f}"
                )
            self.right_nominal_drag_distance = calculate_belt_entry_distance(
                center, self.belt_bounds
            )
            if self.right_nominal_drag_distance > RIGHT_MAX_DRAG_DISTANCE_M:
                raise RuntimeError(
                    "Required right drag exceeds the "
                    f"{RIGHT_MAX_DRAG_DISTANCE_M:.2f} m safety bound: "
                    f"distance={self.right_nominal_drag_distance:.4f} m"
                )
            if self.newton_world is not None:
                q = self._current_dof_positions()
                _log(
                    "[ProM6ClothingDrag] Finger/bag before right lift: "
                    f"right_finger_q={float(q[self.right_gripper_index]):.4f} rad, "
                    f"{self.newton_world.finger_debug()}"
                )
            _log(
                "[ProM6ClothingDrag] Right arm contact takeover: "
                f"grasp={_array_string(self.right_handoff_anchor_position)}, "
                f"tcp={_array_string(self.right_handoff_tcp_position)}, "
                f"center={_array_string(center)}, "
                f"belt_min={_array_string(self.belt_bounds[0])}, "
                f"belt_max={_array_string(self.belt_bounds[1])}, "
                f"entry_x={self.belt_bounds[1][0] - BELT_ENTRY_INSET_M:.4f}, "
                f"nominal_distance={self.right_nominal_drag_distance:.4f} m, "
                f"lift_target={_array_string(self.right_lift_target)}, "
                f"lower_distance={RIGHT_DRAG_LOWER_DISTANCE_M:.4f} m"
            )
            self.right_lift_com_z = float(self._parcel_com()[2])
            self._set_phase(Phase.RIGHT_LIFT)
        elif self.phase_time >= max(
            self.timing.right_grip_settle_seconds + 5.0,
            self.timing.right_grip_stable_seconds + 5.0,
        ):
            raise RuntimeError(
                "Right grasp did not settle: "
                f"position={position_error:.5f}, "
                f"orientation={orientation_error:.5f}, "
                f"gripper={gripper_error:.5f}, "
                f"finger={finger_n}"
            )

    def _step_right_lift(self, dt: float) -> None:
        if self.right_lift_start is None or self.right_lift_target is None:
            raise RuntimeError("Right lift geometry is unavailable")
        progress = _timed_smoothstep(self.phase_time, self.timing.right_lift_seconds)
        target = (1.0 - progress) * self.right_lift_start + progress * self.right_lift_target
        self._command_right_pose(target, dt)
        self._set_right_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_shared_anchor_from_right_tcp()
        # Gate against the final lift target using the newly commanded FK pose.
        tcp_position, tcp_orientation = self._current_right_tcp()
        error = _pose_error(
            tcp_position, tcp_orientation, self.right_lift_target, self.grasp_orientation
        )
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] right lift: "
                f"progress={progress:.3f}, position_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg"
            )
        if (
            self.phase_time >= self.timing.right_lift_seconds
            and position_error <= TCP_LIFT_POSITION_TOLERANCE_M
            and orientation_error <= TCP_LIFT_ORIENTATION_TOLERANCE_RAD
        ):
            self._require_lifted(self.right_lift_com_z, "right")
            # Preserve the original handoff references and frozen entry distance.
            self.right_drag_start = tcp_position.copy()
            self.right_drag_target = self.right_drag_start.copy()
            self.right_actual_drag_distance = 0.0
            self._set_phase(Phase.RIGHT_DRAG_TO_BELT)

    def _step_right_drag_to_belt(self, dt: float) -> None:
        commanded_distance = min(
            RIGHT_DRAG_SPEED_M_S * self.phase_time,
            RIGHT_MAX_DRAG_DISTANCE_M,
        )
        self.right_drag_target[:] = self.right_drag_start
        self.right_drag_target[0] -= commanded_distance
        lower_progress = float(
            np.clip(
                commanded_distance / self.right_nominal_drag_distance
                if self.right_nominal_drag_distance > 0.0
                else self.phase_time / 0.5,
                0.0,
                1.0,
            )
        )
        self.right_drag_target[2] -= RIGHT_DRAG_LOWER_DISTANCE_M * lower_progress
        position_error, orientation_error, _ = self._command_right_pose(self.right_drag_target, dt)
        self._set_right_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_shared_anchor_from_right_tcp()
        tcp_position, _ = self._current_right_tcp()
        self.right_actual_drag_distance = max(
            0.0, float(self.right_drag_start[0] - tcp_position[0])
        )
        fk_lower_distance = float(self.right_drag_start[2] - tcp_position[2])
        final_height_error = abs(fk_lower_distance - RIGHT_DRAG_LOWER_DISTANCE_M)
        lowering_complete = RIGHT_DRAG_LOWER_DISTANCE_M <= 1.0e-6 or (
            lower_progress >= 1.0 and final_height_error <= TCP_APPROACH_POSITION_TOLERANCE_M
        )
        minimum, maximum, center = self._node_aabb()
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            self.belt_bounds = get_world_bounds(self.stage.GetPrimAtPath(BELT_PATH))
        inside = center_is_in_belt_xy(center, self.belt_bounds)
        self.stable_time = _accumulate_stable_time(self.stable_time, inside, dt)
        entry_x = self.belt_bounds[1][0] - BELT_ENTRY_INSET_M
        remaining = max(0.0, float(center[0] - entry_x))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] right belt drag: "
                f"tcp_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"center={_array_string(center)}, "
                f"aabb_max_x={maximum[0]:.4f}, remaining={remaining:.4f} m, "
                f"dragged={self.right_actual_drag_distance:.4f} m, "
                f"lower_progress={lower_progress:.3f}, "
                f"target_lower={RIGHT_DRAG_LOWER_DISTANCE_M * lower_progress:.4f}/"
                f"{RIGHT_DRAG_LOWER_DISTANCE_M:.4f} m, "
                f"fk_lower={fk_lower_distance:.4f} m, "
                f"stable={self.stable_time:.3f}/"
                f"{self.timing.belt_entry_stable_seconds:.3f} s"
            )
        if (
            inside
            and lowering_complete
            and _stable_time_reached(self.stable_time, self.timing.belt_entry_stable_seconds)
        ):
            self._release_attachment()
            self._set_phase(Phase.RIGHT_RELEASE)
        elif (
            commanded_distance >= RIGHT_MAX_DRAG_DISTANCE_M
            and position_error <= DRAG_ENDPOINT_TOLERANCE_M
        ):
            raise RuntimeError(
                "Right drag reached the "
                f"{RIGHT_MAX_DRAG_DISTANCE_M:.2f} m safety bound before release "
                f"conditions were met: center={center}, tail_x={maximum[0]:.4f}, "
                f"lower_progress={lower_progress:.3f}, "
                f"final_height_error={final_height_error:.5f} m"
            )

    def _start_release_conveyor(self) -> None:
        if self.release_conveyor is not None:
            return
        if self.newton_world is None:
            raise RuntimeError("Cannot start conveyor without the Newton world")
        self.belt_bounds = get_world_bounds(self.stage.GetPrimAtPath(BELT_PATH))
        speed = float(self.newton_world.params.belt_speed_mps)
        self.release_conveyor = DeformableConveyorController(
            self.newton_world,
            _belt_xy_bounds(self.belt_bounds),
            RELEASE_CONVEYOR_STOP_XY_REGION,
            speed,
            RELEASE_CONVEYOR_DIRECTION_XY,
            stop_on_arrival=RELEASE_CONVEYOR_STOP_ON_ARRIVAL,
            zero_xy_on_exit=RELEASE_CONVEYOR_ZERO_XY_ON_EXIT,
            log_interval_frames=PROGRESS_LOG_INTERVAL_FRAMES,
        )
        self.release_conveyor.start()

    def _step_release_conveyor(self, dt: float) -> None:
        if self.release_conveyor is None:
            raise RuntimeError("Release conveyor was not started with the right gripper")
        self.release_conveyor.step(dt)

    def _step_right_release(self) -> None:
        self._write_left_q(LEFT_READY_JOINTS)
        self._write_right_q(self.right_commanded_q)
        self._start_release_conveyor()
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        if self.phase_frame == 1:
            self._teleport_pad_retract(LEFT_PAD_RETRACT_M)
        if self.phase_time >= self.timing.right_release_seconds:
            self.right_return_start_q = self.right_commanded_q.copy()
            self._set_phase(Phase.RIGHT_RETURN_READY)

    def _step_right_return_ready(self, dt: float) -> None:
        progress = _timed_smoothstep(self.phase_time, self.timing.right_return_ready_seconds)
        desired = (1.0 - progress) * self.right_return_start_q + progress * RIGHT_READY_JOINTS
        step_norm = self._slew_right_joint_target(desired, dt)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        joint_error = float(np.max(np.abs(self.right_commanded_q - RIGHT_READY_JOINTS)))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] right return: "
                f"progress={progress:.3f}, joint_error={joint_error:.5f} rad, "
                f"step_norm={step_norm:.5f}"
            )
        if self.phase_time >= self.timing.right_return_ready_seconds and joint_error <= 1.0e-4:
            self._write_right_ready()
            self._set_phase(Phase.FINAL_VERIFY)

    def _step_final_verify(self, dt: float) -> None:
        del dt
        self._write_left_q(LEFT_READY_JOINTS)
        self._write_right_ready()
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        self.belt_bounds = get_world_bounds(self.stage.GetPrimAtPath(BELT_PATH))
        _, _, center = self._node_aabb()
        if self.release_conveyor is None:
            raise RuntimeError("Release conveyor is unavailable during final verify")
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            contacts = (
                self.newton_world.contact_breakdown() if self.newton_world is not None else {}
            )
            _log(
                "[ProM6ClothingDrag] final verify: "
                f"center={_array_string(center)}, "
                f"conveyor_state={self.release_conveyor.state.name}, "
                f"contacts={contacts}"
            )
        if self.release_conveyor.state == ConveyorState.EXITED:
            self._report_result()
            if self.replay_recorder is None:
                raise RuntimeError("Successful operation has no replay recorder")
            self.replay_recorder.commit_success()
            self._set_phase(Phase.DONE)
            self.cleanup(hold_left=True)
            if self.timeline.is_playing():
                self.timeline.stop()

    def _report_result(self) -> None:
        _, _, final_center = self._node_aabb()
        _log(f"[ProM6ClothingDrag] Final node center: {_array_string(final_center)}")
        _log(
            "[ProM6ClothingDrag] Belt bounds: "
            f"min={_array_string(self.belt_bounds[0])}, "
            f"max={_array_string(self.belt_bounds[1])}"
        )
        if not center_exited_belt_negative_x(final_center, self.belt_bounds):
            raise RuntimeError(
                "Clothing did not leave through the belt -X edge while "
                f"remaining inside its Y span: center={final_center}"
            )
        _log(
            "[ProM6ClothingDrag][SUCCESS] Clothing exited "
            "ConveyorBelt_A06/Belt through its world -X edge"
        )

    def step(self, dt: float, context: object) -> None:
        del context
        if not self.initialized:
            self._initialize_after_play()
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0 or dt > 0.1:
            raise RuntimeError(f"Invalid physics callback dt={dt!r}")
        if self.initialized and self.newton_world is not None:
            if self._newton_hold_frames > 0:
                self._sync_newton_jaws()
                self.newton_world.reset_kinematic_history()
                self._newton_hold_frames -= 1
            else:
                self._sync_newton_jaws()
                self.newton_world.step(dt)
        phase_at_frame_start = self.phase
        right_task_phases = (
            Phase.PREPARE_RIGHT_HANDOFF,
            Phase.RIGHT_SEARCH_IK,
            Phase.RIGHT_MOVE_TO_PREGRASP,
            Phase.RIGHT_VERTICAL_APPROACH,
            Phase.RIGHT_CLOSE,
            Phase.RIGHT_GRIP_SETTLE,
            Phase.RIGHT_LIFT,
            Phase.RIGHT_DRAG_TO_BELT,
            Phase.RIGHT_RELEASE,
            Phase.RIGHT_RETURN_READY,
            Phase.FINAL_VERIFY,
        )
        if self.phase in right_task_phases:
            self._write_left_q(LEFT_READY_JOINTS)
        else:
            self._write_right_ready()
        self._cached_nodes = None
        self.phase_frame += 1
        self.phase_time += dt
        timeout = self.phase_timeouts.get(self.phase)
        if timeout is not None and self.phase_time > timeout:
            raise RuntimeError(f"Phase {self.phase.name} exceeded its {timeout:.1f} second timeout")
        self._deformable_nodes()
        self._capture_replay_frame(dt)

        if self.phase == Phase.SETTLE:
            self._step_settle()
        elif self.phase == Phase.WAITING:
            self._write_left_q(self.left_commanded_q)
            self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
            if self.phase_time >= self.timing.waiting_seconds:
                self._set_phase(Phase.SEARCH_PREGRASP_IK)
        elif self.phase == Phase.SEARCH_PREGRASP_IK:
            self._step_search_pregrasp_ik()
        elif self.phase == Phase.MOVE_TO_PREGRASP:
            self._step_joint_segment(
                self.pregrasp_start_q,
                self.pregrasp_joint_target,
                self.timing.left_move_pregrasp_seconds,
                dt,
                Phase.VERTICAL_APPROACH,
            )
        elif self.phase == Phase.VERTICAL_APPROACH:
            self._step_approach(dt)
        elif self.phase == Phase.CLOSE_GRIPPER:
            self._step_close_gripper(dt)
        elif self.phase == Phase.GRIP_SETTLE:
            self._step_grip_settle(dt)
        elif self.phase == Phase.LEFT_LIFT:
            self._step_left_lift(dt)
        elif self.phase == Phase.DRAG:
            self._step_drag(dt)
        elif self.phase == Phase.CENTER_CORRECT:
            self._step_center_correct(dt)
        elif self.phase == Phase.LOWER:
            self._step_lower(dt)
        elif self.phase == Phase.PLACE_SETTLE:
            self._step_place_settle(dt)
        elif self.phase == Phase.LEFT_OPEN:
            self._step_left_open(dt)
        elif self.phase == Phase.LEFT_RETURN_READY:
            self._step_left_return_ready(dt)
        elif self.phase == Phase.PREPARE_RIGHT_HANDOFF:
            self._step_prepare_right_handoff()
        elif self.phase == Phase.RIGHT_SEARCH_IK:
            self._step_right_search_ik()
        elif self.phase == Phase.RIGHT_MOVE_TO_PREGRASP:
            self._step_right_joint_segment(
                self.right_pregrasp_start_q,
                self.right_pregrasp_joint_target,
                self.timing.right_move_pregrasp_seconds,
                dt,
                Phase.RIGHT_VERTICAL_APPROACH,
            )
        elif self.phase == Phase.RIGHT_VERTICAL_APPROACH:
            self._step_right_approach(dt)
        elif self.phase == Phase.RIGHT_CLOSE:
            self._step_right_close(dt)
        elif self.phase == Phase.RIGHT_GRIP_SETTLE:
            self._step_right_grip_settle(dt)
        elif self.phase == Phase.RIGHT_LIFT:
            self._step_right_lift(dt)
        elif self.phase == Phase.RIGHT_DRAG_TO_BELT:
            self._step_right_drag_to_belt(dt)
        elif self.phase == Phase.RIGHT_RELEASE:
            self._step_right_release()
        elif self.phase == Phase.RIGHT_RETURN_READY:
            self._step_right_return_ready(dt)
        elif self.phase == Phase.FINAL_VERIFY:
            self._step_final_verify(dt)
        if not self.cleaned and phase_at_frame_start in (
            Phase.RIGHT_RELEASE,
            Phase.RIGHT_RETURN_READY,
            Phase.FINAL_VERIFY,
        ):
            self._step_release_conveyor(dt)
        if not self.cleaned:
            self._log_progress()

    def cleanup(self, *, hold_left: bool) -> None:
        if self.cleaned:
            return
        self.cleaned = True
        cleanup_errors: list[tuple[str, BaseException]] = []
        if hold_left and self.initialized and self.left_commanded_q is not None:
            try:
                self._write_left_q(self.left_commanded_q)
            except Exception as error:
                cleanup_errors.append(("hold left arm", error))
            try:
                self._write_right_ready()
            except Exception as error:
                cleanup_errors.append(("hold right arm ready", error))
        if self.release_conveyor is not None:
            try:
                self.release_conveyor.cleanup()
            except Exception as error:
                cleanup_errors.append(("clean release conveyor", error))
        if self.replay_recorder is not None:
            try:
                self.replay_recorder.discard()
            except Exception as error:
                cleanup_errors.append(("discard uncommitted replay frames", error))
        try:
            self._release_attachment()
        except Exception as error:
            cleanup_errors.append(("remove shared attachment", error))
        try:
            restore_collision_attributes(self.stage, self.saved_collisions)
        except Exception as error:
            cleanup_errors.append(("restore gripper collisions", error))
        finally:
            self.saved_collisions.clear()
        try:
            remove_owned_runtime_prims(self.stage)
        except Exception as error:
            cleanup_errors.append(("remove runtime Prims", error))
        try:
            self._deregister_callbacks()
        except Exception as error:
            cleanup_errors.append(("deregister callbacks", error))
        finally:
            globals()[CONTROLLER_GLOBAL_NAME] = None
        for action, error in cleanup_errors:
            _log(f"[ProM6ClothingDrag][WARNING] Cleanup could not {action}: {error}")
        _log("[ProM6ClothingDrag] Runtime Prims/collision overrides cleaned")

    def fail(self, error: BaseException) -> None:
        if self.failed:
            return
        self.failed = True
        self.phase = Phase.FAILED
        _log("[ProM6ClothingDrag][ERROR]", error)
        traceback.print_exc()
        try:
            self.cleanup(hold_left=True)
        finally:
            if self.timeline.is_playing():
                self.timeline.stop()

    def _deregister_callbacks(self) -> None:
        for attribute, global_name in (
            ("callback_id", CALLBACK_GLOBAL_NAME),
            ("stop_callback_id", STOP_CALLBACK_GLOBAL_NAME),
        ):
            callback_id = getattr(self, attribute)
            if callback_id is not None:
                try:
                    SimulationManager.deregister_callback(callback_id)
                except Exception:
                    pass
                setattr(self, attribute, None)
            globals()[global_name] = None
