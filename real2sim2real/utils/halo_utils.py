"""HALO G1 helpers: GEAR-SONIC contract and tracker, timing, cameras, contacts and acceptance.

Ported from the HALO project (crawl_v3 / crouch_v3 / sit_v1 scripts ``sonic_contract.py``,
``sonic_tracker.py``, ``sampling.py``, ``native_motor_control.py``, ``follow_camera.py``,
``physx_world.ContactMonitor``, ``camera_geometry.py`` and ``validate_physx.py``).
Only numpy is imported at module level; onnxruntime, scipy and pxr are imported inside the
functions that need them, so ``list-backends`` and the unit tests never start Kit.

Sections (dependency order):
1. GEAR-SONIC G1 contract: joint orders, motor gains, default pose, observation history
2. SONIC tracker: ONNX encoder/decoder at 50 Hz
3. Control timing and image sampling
4. Motor target conditioning
5. Camera poses: look-at, measured-base follow camera, pinhole projection
6. Contact accounting (PhysX contact reports → per-category impulses)
7. Native G1 collider geometry and joint-frame FK
8. Acceptance checks: common, desk passage, posture under the desk, seated finish
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# =====================================================================================
# 1. GEAR-SONIC G1 contract (NVlabs/GR00T-WholeBodyControl g1 deploy)
# =====================================================================================

MUJOCO_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]  # fmt: skip
ISAACLAB_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]  # fmt: skip
ISAACLAB_TO_MUJOCO = np.array([ISAACLAB_JOINT_NAMES.index(n) for n in MUJOCO_JOINT_NAMES])
MUJOCO_TO_ISAACLAB = np.array([MUJOCO_JOINT_NAMES.index(n) for n in ISAACLAB_JOINT_NAMES])

_NATURAL_FREQ = 10.0 * 2.0 * math.pi
_DAMPING_RATIO = 2.0
MOTOR_ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
MOTOR_EFFORT_LIMIT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
# (motor type, stiffness multiplier) per MuJoCo-order joint; ankles and waist pitch/roll ×2.
MOTOR_TABLE: list[tuple[str, float]] = [
    ("7520_22", 1.0), ("7520_22", 1.0), ("7520_14", 1.0), ("7520_22", 1.0), ("5020", 2.0),
    ("5020", 2.0), ("7520_22", 1.0), ("7520_22", 1.0), ("7520_14", 1.0), ("7520_22", 1.0),
    ("5020", 2.0), ("5020", 2.0), ("7520_14", 1.0), ("5020", 2.0), ("5020", 2.0),
    ("5020", 1.0), ("5020", 1.0), ("5020", 1.0), ("5020", 1.0), ("5020", 1.0), ("4010", 1.0),
    ("4010", 1.0), ("5020", 1.0), ("5020", 1.0), ("5020", 1.0), ("5020", 1.0), ("5020", 1.0),
    ("4010", 1.0), ("4010", 1.0),
]  # fmt: skip


def _stiffness(motor: str) -> float:
    return MOTOR_ARMATURE[motor] * _NATURAL_FREQ**2


KPS = np.array([_stiffness(m) * mult for m, mult in MOTOR_TABLE])  # N·m/rad
KDS = np.array(
    [2.0 * _DAMPING_RATIO * MOTOR_ARMATURE[m] * _NATURAL_FREQ * mult for m, mult in MOTOR_TABLE]
)  # N·m·s/rad
ACTION_SCALE = np.array([0.25 * MOTOR_EFFORT_LIMIT[m] / _stiffness(m) for m, _ in MOTOR_TABLE])
EFFORT_LIMIT = np.array([MOTOR_EFFORT_LIMIT[m] for m, _ in MOTOR_TABLE])  # N·m
ARMATURE = np.array([MOTOR_ARMATURE[m] for m, _ in MOTOR_TABLE])  # kg·m²
DEFAULT_ANGLES = np.array(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0] * 2
    + [0.0, 0.0, 0.0]
    + [0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0]
    + [0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0]
)  # rad, MuJoCo order
DEFAULT_ANGLES_ISAACLAB = DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]

# Encoder input layout; only the four G1 entries marked * are filled, the rest stay zero.
ENCODER_ORDER: list[tuple[str, int]] = [
    ("encoder_mode_4", 4),  # *
    ("motion_joint_positions_10frame_step5", 290),  # *
    ("motion_joint_velocities_10frame_step5", 290),  # *
    ("motion_root_z_position_10frame_step5", 10),
    ("motion_root_z_position", 1),
    ("motion_anchor_orientation", 6),
    ("motion_anchor_orientation_10frame_step5", 60),  # *
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
]
CONTROL_HZ = 50  # SONIC policy rate; one decoder action per 20 ms


def quat_wxyz_to_matrix(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Rotation matrix of one ``(w, x, y, z)`` quaternion or a ``(..., 4)`` stack."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = (q[..., i] for i in range(4))
    n = w * w + x * x + y * y + z * z
    s = np.where(n < 1e-12, 0.0, 2.0 / np.where(n < 1e-12, 1.0, n))
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - s * (y * y + z * z)
    m[..., 0, 1] = s * (x * y - w * z)
    m[..., 0, 2] = s * (x * z + w * y)
    m[..., 1, 0] = s * (x * y + w * z)
    m[..., 1, 1] = 1 - s * (x * x + z * z)
    m[..., 1, 2] = s * (y * z - w * x)
    m[..., 2, 0] = s * (x * z - w * y)
    m[..., 2, 1] = s * (y * z + w * x)
    m[..., 2, 2] = 1 - s * (x * x + y * y)
    return m


def matrix_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` with w ≥ 0 for a 3×3 rotation matrix."""
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(np.asarray(m, dtype=float)).as_quat()[[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def rotmat_6d(rm: np.ndarray) -> np.ndarray:
    """SONIC 6D rotation: first two columns flattened row by row (official C++ reference)."""
    return np.asarray(rm, dtype=float)[:, :2].reshape(-1)


class SonicHistory:
    """10-frame 50 Hz decoder history (oldest → newest)."""

    def __init__(self, frames: int = 10) -> None:
        self.ang_vel = np.zeros((frames, 3))
        self.joint_pos = np.zeros((frames, 29))
        self.joint_vel = np.zeros((frames, 29))
        self.last_action = np.zeros((frames, 29))
        self.gravity_dir = np.zeros((frames, 3))
        self.gravity_dir[:, 2] = -1.0

    def push(self, ang_vel, joint_pos, joint_vel, last_action, gravity_dir) -> None:
        for buf, value in (
            (self.ang_vel, ang_vel),
            (self.joint_pos, joint_pos),
            (self.joint_vel, joint_vel),
            (self.last_action, last_action),
            (self.gravity_dir, gravity_dir),
        ):
            buf[:-1] = buf[1:]
            buf[-1] = value

    def decoder_input(self, token: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                token,
                self.ang_vel.ravel(),
                self.joint_pos.ravel(),
                self.joint_vel.ravel(),
                self.last_action.ravel(),
                self.gravity_dir.ravel(),
            ]
        )


# =====================================================================================
# 2. SONIC tracker: encoder (reference window → token) + decoder (token + history → action)
# =====================================================================================


def resample_reference(qpos: np.ndarray, reference_fps: float, control_hz: int) -> np.ndarray:
    """Linear joints/position and slerped base orientation at the control rate.

    ``qpos`` rows are ``[x, y, z, qw, qx, qy, qz, 29 joints (MuJoCo order)]``.
    """
    from scipy.spatial.transform import Rotation, Slerp

    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2:
        raise ValueError("SONIC reference must be (frames, 36): base pose + 29 joints")
    old = np.arange(len(qpos)) / reference_fps
    new = np.arange(0, old[-1] + 1e-4, 1.0 / control_hz)
    ref = np.stack([np.interp(new, old, qpos[:, i]) for i in range(36)], axis=-1)
    rotations = Rotation.from_quat(qpos[:, 3:7][:, [1, 2, 3, 0]])
    ref[:, 3:7] = Slerp(old, rotations)(new).as_quat()[:, [3, 0, 1, 2]]
    return ref


class SonicTracker:
    """GEAR-SONIC whole-body tracking of a reference motion, one action per 20 ms.

    ``step`` returns MuJoCo-order joint position targets (rad). Call
    ``set_applied_target`` with the target actually sent to the motors so the next
    observation's last-action history matches what the robot executed.
    """

    def __init__(
        self, model_dir: Path, qpos: np.ndarray, *, reference_fps: float = 30, threads: int = 4
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - Isaac Sim 6.2 ships onnxruntime
            raise RuntimeError(
                "halo 需要 onnxruntime：./app/python.sh -m pip install onnxruntime"
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(
            str(model_dir / "model_encoder.onnx"), sess_options=options, providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(model_dir / "model_decoder.onnx"), sess_options=options, providers=providers
        )
        self.ref = resample_reference(qpos, reference_fps, CONTROL_HZ)
        self.jpos = self.ref[:, 7:][:, MUJOCO_TO_ISAACLAB]
        self.jvel = np.gradient(self.jpos, 1.0 / CONTROL_HZ, axis=0)
        self.history = SonicHistory()
        self.last = np.zeros(29)

    def __len__(self) -> int:
        return len(self.ref)

    def reset(self, qpos: np.ndarray, angvel: np.ndarray | None = None) -> None:
        """Fill the history with a still pose (the reference start pose in HALO)."""
        self.history = SonicHistory()
        joints = np.asarray(qpos[7:])[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES_ISAACLAB
        gravity = quat_wxyz_to_matrix(qpos[3:7]).T @ np.array([0.0, 0.0, -1.0])
        angvel = np.zeros(3) if angvel is None else angvel
        for _ in range(10):
            self.history.push(angvel, joints, np.zeros(29), np.zeros(29), gravity)
        self.last = np.zeros(29)

    def step(
        self, frame: int, qpos: np.ndarray, joint_vel: np.ndarray, angvel_world: np.ndarray
    ) -> np.ndarray:
        """Targets for control frame ``frame`` from the measured base pose and joint state."""
        rotation = quat_wxyz_to_matrix(qpos[3:7])
        self.history.push(
            rotation.T @ angvel_world,
            np.asarray(qpos[7:])[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES_ISAACLAB,
            np.asarray(joint_vel)[MUJOCO_TO_ISAACLAB],
            self.last,
            rotation.T @ np.array([0.0, 0.0, -1.0]),
        )
        ids = np.clip(int(frame) + np.arange(10) * 5, 0, len(self.ref) - 1)
        anchor = np.stack(
            [rotmat_6d(rotation.T @ quat_wxyz_to_matrix(q)) for q in self.ref[ids, 3:7]]
        )
        segments = {
            "encoder_mode_4": np.zeros(4),
            "motion_joint_positions_10frame_step5": self.jpos[ids].reshape(-1),
            "motion_joint_velocities_10frame_step5": self.jvel[ids].reshape(-1),
            "motion_anchor_orientation_10frame_step5": anchor.reshape(-1),
        }
        encoder_input = np.concatenate([segments.get(k, np.zeros(n)) for k, n in ENCODER_ORDER])
        token = self.encoder.run(None, {"obs_dict": encoder_input.astype(np.float32)[None]})[0]
        decoder_input = self.history.decoder_input(token.reshape(-1)).astype(np.float32)[None]
        self.last = self.decoder.run(None, {"obs_dict": decoder_input})[0].reshape(-1)
        if not np.isfinite(self.last).all():
            raise RuntimeError("SONIC produced a nonfinite action")
        return DEFAULT_ANGLES + self.last[ISAACLAB_TO_MUJOCO] * ACTION_SCALE

    def set_applied_target(self, target: np.ndarray) -> None:
        """Replace the decoder's last action by the conditioned target sent to the motors."""
        self.last = ((np.asarray(target) - DEFAULT_ANGLES) / ACTION_SCALE)[MUJOCO_TO_ISAACLAB]


# =====================================================================================
# 3. Control timing and image sampling
# =====================================================================================


def image_schedule(control_frames: int, control_hz: int, image_fps: int) -> np.ndarray:
    """Control-frame index nearest to each image time ``i / image_fps`` (no interpolation)."""
    if not 0 < image_fps <= control_hz:
        raise ValueError("image fps must be positive and not above the control rate")
    count = int(np.ceil(control_frames / control_hz * image_fps - 1e-9))
    desired = np.arange(count, dtype=np.float64) / image_fps
    indices = np.floor(desired * control_hz + 0.5).astype(np.int64)
    indices = indices[indices < control_frames]
    if np.any(np.diff(indices) <= 0):
        raise ValueError("image schedule must be strictly increasing")
    return indices


def phase_index(
    frame: int, phases: Sequence[dict], *, control_hz: int, reference_fps: float
) -> int:
    """Reference phase of control frame ``frame``; phases use inclusive reference frames."""
    t = frame / control_hz
    for i, phase in enumerate(phases):
        if phase["start"] / reference_fps <= t < (phase["end"] + 1) / reference_fps:
            return i
    return len(phases) - 1


# =====================================================================================
# 4. Motor target conditioning
# =====================================================================================

ARM_JOINT_PARTS = ("shoulder", "elbow", "wrist")


class MotorTargetFilter:
    """Causal first-order target smoothing with speed and acceleration limits (crawl arms)."""

    def __init__(self, initial: np.ndarray, settings: Any, joint_names: Sequence[str]) -> None:
        self.settings = settings
        self.target = np.asarray(initial, dtype=float).copy()
        self.velocity = np.zeros_like(self.target)
        self.arm = np.array([any(p in n for p in ARM_JOINT_PARTS) for n in joint_names])
        self.speed = np.where(self.arm, settings.arm_max_speed_rad_s, settings.leg_max_speed_rad_s)

    def step(self, desired: np.ndarray, dt: float) -> np.ndarray:
        desired = np.asarray(desired, dtype=float)
        cfg = self.settings
        if not cfg.enabled:
            self.target = desired.copy()
            return self.target.copy()
        alpha = 1 - np.exp(-dt / cfg.time_constant_s)
        requested = np.clip(alpha * (desired - self.target) / dt, -self.speed, self.speed)
        increment = cfg.max_acceleration_rad_s2 * dt
        self.velocity += np.clip(requested - self.velocity, -increment, increment)
        following = self.target + self.velocity * dt
        overshoot = (desired - self.target) * (desired - following) < 0
        following[overshoot] = desired[overshoot]
        self.velocity[overshoot] = 0
        self.target = following
        if cfg.arms_only:
            self.target[~self.arm] = desired[~self.arm]
            self.velocity[~self.arm] = 0
        return self.target.copy()


# =====================================================================================
# 5. Camera poses
# =====================================================================================


def rotation_z(yaw_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def look_at_quat_wxyz(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """USD camera orientation (−Z forward, +Y up) looking from ``eye`` to ``target``."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    back = (eye - target) / np.linalg.norm(eye - target)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    camera_up = np.cross(back, right)
    return matrix_to_quat_wxyz(np.stack([right, camera_up, back], axis=1))


class FollowCameraFilter:
    """Third-person camera following the measured pelvis; filtered position, no yaw jitter."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.rotation = rotation_z(settings.task_yaw_deg)
        self.filtered: np.ndarray | None = None

    def step(self, base: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(eye, target, quat_wxyz)`` for the measured base pose ``base[:3]``."""
        cfg = self.settings
        desired = np.asarray(base[:3], dtype=float).copy()
        desired[2] = cfg.target_height_base_gain * base[2] + cfg.target_height_offset_m
        if self.filtered is None:
            self.filtered = desired
        else:
            tau = np.array([cfg.position_time_constant_s] * 2 + [cfg.height_time_constant_s])
            self.filtered = self.filtered + (1 - np.exp(-dt / tau)) * (desired - self.filtered)
        target = self.filtered + self.rotation @ np.asarray(cfg.look_ahead_task_m, dtype=float)
        eye = target + self.rotation @ np.asarray(cfg.eye_offset_task_m, dtype=float)
        return eye.copy(), target.copy(), look_at_quat_wxyz(eye, target)


def project_points(
    points: np.ndarray,
    pose: np.ndarray,
    *,
    resolution: tuple[int, int],
    focal_length: float,
    horizontal_aperture: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole pixels ``(u, v)`` and depth of world points for a USD camera pose ``[p, q]``."""
    width, height = resolution
    vertical_aperture = horizontal_aperture * height / width
    rotation = quat_wxyz_to_matrix(pose[3:7])
    camera = (np.asarray(points) - pose[:3]) @ rotation
    depth = -camera[:, 2]
    u = width / 2 + width * focal_length / horizontal_aperture * camera[:, 0] / depth
    v = height / 2 - height * focal_length / vertical_aperture * camera[:, 1] / depth
    return u, v, depth


# =====================================================================================
# 6. Contact accounting
# =====================================================================================

CONTACT_CATEGORIES = ("ground", "table", "chair", "box", "self", "home_obstacle")
GROUND, TABLE, CHAIR, BOX, SELF, HOME = range(6)


@dataclass
class ContactPrefixes:
    """Prim-path prefixes deciding the category of the body touched by the robot."""

    robot: str
    table: str | None = None
    seat: str | None = None
    home: str | None = None

    def category(self, other: str) -> int:
        """HALO order: seat, table, robot (self), home obstacle, otherwise ground."""
        if self.seat and other.startswith(self.seat):
            return CHAIR
        if self.table and other.startswith(self.table):
            return TABLE
        if other.startswith(self.robot):
            return SELF
        if self.home and other.startswith(self.home):
            return HOME
        return GROUND


@dataclass
class ContactAccumulator:
    """Per-control-period impulses by category, foot support and penetration (N·s, m)."""

    prefixes: ContactPrefixes
    left_foot: str = "left_ankle_roll_link"
    right_foot: str = "right_ankle_roll_link"
    timestamp_s: float = -1.0
    frame_impulses: np.ndarray = field(default_factory=lambda: np.zeros(6))
    vertical_impulses: np.ndarray = field(default_factory=lambda: np.zeros(6))
    foot_impulses: np.ndarray = field(default_factory=lambda: np.zeros(3))
    min_separation: float = 0.0
    max_penetration: float = 0.0
    events: list[dict] = field(default_factory=list)
    actor_impulses: dict[str, float] = field(default_factory=dict)
    points: int = 0  # robot contact points reported in the current control period

    def add(self, path0: str, path1: str, impulse: Sequence[float], separation: float) -> None:
        """One contact point of ``path0``/``path1``; ignored unless it touches the robot."""
        robot = self.prefixes.robot
        in0, in1 = path0.startswith(robot), path1.startswith(robot)
        if not (in0 or in1):
            return
        other = path1 if in0 else path0
        self.points += 1
        category = self.prefixes.category(other)
        vector = np.asarray(impulse, dtype=float)
        magnitude = float(np.linalg.norm(vector))
        vertical = abs(float(vector[2]))
        self.frame_impulses[category] += magnitude
        self.vertical_impulses[category] += vertical
        self.min_separation = min(self.min_separation, float(separation))
        self.max_penetration = max(self.max_penetration, -float(separation))
        pair = f"{path0} | {path1}"
        self.actor_impulses[pair] = self.actor_impulses.get(pair, 0.0) + magnitude
        body = (path0 if in0 else path1).rsplit("/", 1)[-1]
        foot = 0 if body == self.left_foot else 1 if body == self.right_foot else 2
        if category == GROUND:
            self.foot_impulses[foot] += magnitude
        if (
            category in (TABLE, CHAIR, BOX, HOME)
            or separation < -0.005
            or (category == GROUND and foot == 2 and magnitude > 1e-4)
        ):
            self.events.append(
                {
                    "timestamp_s": self.timestamp_s,
                    "actors": [path0, path1],
                    "separation_m": float(separation),
                    "impulse_Ns": magnitude,
                    "vertical_impulse_Ns": vertical,
                    "category": category,
                }
            )

    def consume(self) -> np.ndarray:
        """Category impulses plus this period's penetration; resets both."""
        out = np.r_[self.frame_impulses, max(0.0, -self.min_separation)].astype(np.float32)
        self.frame_impulses[:] = 0
        self.min_separation = 0.0
        return out

    def end_period(self, timestamp_s: float) -> None:
        self.vertical_impulses[:] = 0
        self.foot_impulses[:] = 0
        self.points = 0
        self.timestamp_s = timestamp_s

    def clear(self) -> None:
        """Forget the warm-up: events, totals and per-period values."""
        self.consume()
        self.end_period(self.timestamp_s)
        self.events.clear()
        self.actor_impulses.clear()
        self.max_penetration = 0.0


# =====================================================================================
# 7. Native G1 collider geometry and joint-frame FK (read once from the robot USD)
# =====================================================================================

_UPPER_BODIES = ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link")
_UPPER_PARTS = (
    "shoulder", "elbow", "wrist", "hand", "palm", "head", "logo", "imu",
    "zero", "one", "two", "three", "four", "five", "six",
)  # fmt: skip


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by ``angle`` about the unit vector ``axis`` (Rodrigues)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


@dataclass
class RobotGeometry:
    """Collider hulls in body frames and the joint tree of the native G1 USD.

    ``body_names`` follow the USD traversal order of rigid bodies relative to the default
    prim; the same order is used for the recorded ``robot_body_poses``.
    """

    body_names: list[str]
    group_body: np.ndarray  # (G,) body index of each collider hull
    group_names: list[str]
    group_points: list[np.ndarray]  # hull vertices in the body frame (m)
    joints: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, int]]

    @property
    def collider_count(self) -> int:
        return len(self.group_names)

    @property
    def root_id(self) -> int:
        return self.body_names.index("pelvis")

    @property
    def upper_vertex_mask(self) -> np.ndarray:
        masks = []
        for body, points in zip(self.group_body, self.group_points, strict=True):
            name = self.body_names[int(body)]
            upper = name in _UPPER_BODIES or any(part in name for part in _UPPER_PARTS)
            masks.append(np.full(len(points), upper))
        return np.concatenate(masks)

    @classmethod
    def from_usd(cls, path: str | Path) -> RobotGeometry:
        """Read collider hulls and joints from the robot asset (needs pxr and scipy)."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from scipy.spatial import ConvexHull, QhullError

        stage = Usd.Stage.Open(str(path))
        prefix = str(stage.GetDefaultPrim().GetPath()) + "/"
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        names = [str(p.GetPath())[len(prefix) :] for p in bodies]
        ids = {name: i for i, name in enumerate(names)}
        cache = UsdGeom.XformCache()
        group_body, group_names, group_points = [], [], []
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            body = prim.GetParent()
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body:
                raise ValueError(f"Collider without rigid body: {prim.GetPath()}")
            relative = (
                cache.GetLocalToWorldTransform(prim)
                * cache.GetLocalToWorldTransform(body).GetInverse()
            )
            if prim.IsA(UsdGeom.Mesh):
                local = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            else:
                if prim.IsA(UsdGeom.Sphere):
                    extents = np.full(3, UsdGeom.Sphere(prim).GetRadiusAttr().Get())
                elif prim.IsA(UsdGeom.Cylinder):
                    shape = UsdGeom.Cylinder(prim)
                    extents = np.full(3, shape.GetRadiusAttr().Get())
                    extents["XYZ".index(str(shape.GetAxisAttr().Get()))] = (
                        shape.GetHeightAttr().Get() / 2
                    )
                elif prim.IsA(UsdGeom.Cube):
                    extents = np.full(3, UsdGeom.Cube(prim).GetSizeAttr().Get() / 2)
                else:
                    raise ValueError(f"Unsupported collider: {prim.GetPath()}")
                corners = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
                local = np.array(corners, dtype=float) * extents
            points = np.array([relative.Transform(Gf.Vec3d(*v)) for v in local])
            try:
                points = points[ConvexHull(points).vertices]
            except QhullError:
                pass
            group_body.append(ids[str(body.GetPath())[len(prefix) :]])
            group_names.append(str(prim.GetPath())[len(prefix) :])
            group_points.append(points)
        joints = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            b0, b1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not b0 or not b1 or not joint.GetJointEnabledAttr().Get():
                continue
            parent, child = (str(paths[0])[len(prefix) :] for paths in (b0, b1))
            frames = []
            for pos, rot in (
                (joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                (joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
            ):
                m = np.eye(4)
                m[:3, :3] = quat_wxyz_to_matrix([rot.GetReal(), *rot.GetImaginary()])
                m[:3, 3] = pos
                frames.append(m)
            axis = np.zeros(3)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                axis["XYZ".index(str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()))] = 1
            qi = (
                MUJOCO_JOINT_NAMES.index(prim.GetName())
                if prim.GetName() in MUJOCO_JOINT_NAMES
                else -1
            )
            joints.append((ids[parent], ids[child], frames[0], np.linalg.inv(frames[1]), axis, qi))
        known, ordered = {ids["pelvis"]}, []
        while joints:
            ready = [j for j in joints if j[0] in known]
            if not ready:
                raise ValueError("G1 joint graph is not connected to pelvis")
            for item in ready:
                ordered.append(item)
                known.add(item[1])
                joints.remove(item)
        if len(known) != len(names):
            raise ValueError("Not every G1 body is reachable from pelvis")
        return cls(names, np.asarray(group_body), group_names, group_points, ordered)

    def save(self, path: Path) -> None:
        counts = np.array([len(p) for p in self.group_points])
        np.savez_compressed(
            path,
            body_names=np.array(self.body_names),
            group_body=self.group_body,
            group_names=np.array(self.group_names),
            group_counts=counts,
            group_points=np.concatenate(self.group_points),
            joint_index=np.array([[p, c, q] for p, c, _, _, _, q in self.joints]),
            joint_a=np.stack([j[2] for j in self.joints]),
            joint_b_inv=np.stack([j[3] for j in self.joints]),
            joint_axis=np.stack([j[4] for j in self.joints]),
        )

    @classmethod
    def load(cls, path: Path) -> RobotGeometry:
        data = np.load(path)
        points = np.split(data["group_points"], np.cumsum(data["group_counts"])[:-1])
        joints = [
            (int(p), int(c), a, b, axis, int(q))
            for (p, c, q), a, b, axis in zip(
                data["joint_index"],
                data["joint_a"],
                data["joint_b_inv"],
                data["joint_axis"],
                strict=True,
            )
        ]
        return cls(
            [str(n) for n in data["body_names"]],
            data["group_body"],
            [str(n) for n in data["group_names"]],
            points,
            joints,
        )

    def body_matrices(self, qpos: np.ndarray) -> np.ndarray:
        """FK from the measured base pose and 29 body joints; fingers stay at zero."""
        matrices = np.zeros((len(self.body_names), 4, 4))
        root = np.eye(4)
        root[:3, :3] = quat_wxyz_to_matrix(qpos[3:7])
        root[:3, 3] = qpos[:3]
        matrices[self.root_id] = root
        for parent, child, a, b_inv, axis, qi in self.joints:
            motion = np.eye(4)
            if qi >= 0:
                motion[:3, :3] = _axis_rotation(axis, float(qpos[7 + qi]))
            matrices[child] = matrices[parent] @ a @ motion @ b_inv
        return matrices

    def world_points(
        self, body_poses: np.ndarray, groups: Sequence[int] | None = None
    ) -> np.ndarray:
        """Collider hull vertices in the world for measured body poses ``(B, 7)``."""
        rotation = quat_wxyz_to_matrix(body_poses[:, 3:7])
        selected = range(len(self.group_points)) if groups is None else groups
        return np.concatenate(
            [
                self.group_points[g] @ rotation[self.group_body[g]].T
                + body_poses[self.group_body[g], :3]
                for g in selected
            ]
        )


# =====================================================================================
# 8. Acceptance checks on the full 50 Hz control recording (HALO validate_physx.py)
# =====================================================================================


@dataclass
class DeskSpec:
    """Desk placement and the dimensions the passage checks use (m, deg)."""

    xy: tuple[float, float]
    yaw_deg: float
    base_z: float
    depth_m: float  # along the walking direction (desk local x)
    width_m: float
    clearance_z_m: float  # underside of the crossbeam
    column_spacing_m: float  # column centre distance; inner gap = spacing - 0.092 m

    def to_local(self, points: np.ndarray) -> np.ndarray:
        origin = np.array([*self.xy, self.base_z])
        return (np.asarray(points) - origin) @ rotation_z(self.yaw_deg)


def _pitch_deg(poses: np.ndarray) -> np.ndarray:
    """Forward torso pitch from ``[p, q]`` poses (positive = bent forward)."""
    rotation = quat_wxyz_to_matrix(poses[:, 3:7])
    return np.degrees(np.arcsin(np.clip(-rotation[:, 2, 0], -1, 1)))


def _first_sustained(mask: np.ndarray, count: int = 10) -> int | None:
    hits = np.flatnonzero(
        np.convolve(np.asarray(mask, dtype=int), np.ones(count, dtype=int), "valid") == count
    )
    return int(hits[0]) if len(hits) else None


def common_checks(
    rec: dict[str, np.ndarray],
    *,
    expected_frames: int,
    joint_limits_rad: np.ndarray,
    max_penetration_m: float,
    penetration_limit_m: float = 0.005,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Trace completeness, finite values, timing, quaternions, penetration and joint limits."""
    n = len(rec["qpos"])
    lo, hi = joint_limits_rad[:, 0], joint_limits_rad[:, 1]

    def excess(q: np.ndarray) -> float:
        return float(np.maximum(np.maximum(lo - q, q - hi), 0).max())

    metrics = {
        "control_frames": n,
        "max_contact_penetration_m": float(max_penetration_m),
        "max_actual_joint_limit_excess_rad": excess(rec["qpos"][:, 7:]),
        "max_target_joint_limit_excess_rad": excess(rec["action"]),
        "home_obstacle_contact_impulse_Ns": float(rec["contacts"][:, HOME].sum()),
    }
    checks = {
        "complete_control_trace": n == expected_frames and n > 100,
        "finite_measurements": all(np.isfinite(v).all() for v in rec.values()),
        "uniform_50Hz_control_time": bool(
            np.allclose(rec["timestamp"], np.arange(n) / CONTROL_HZ, atol=3e-6)
        ),
        "unit_base_quaternions": bool(
            np.max(np.abs(np.linalg.norm(rec["qpos"][:, 3:7], axis=1) - 1)) < 1e-3
        ),
        "contact_penetration_below_5mm": max_penetration_m <= penetration_limit_m,
        "actual_joint_limits_within_0_01rad": metrics["max_actual_joint_limit_excess_rad"] <= 0.01,
        "motor_targets_within_limits": metrics["max_target_joint_limit_excess_rad"] <= 1e-5,
    }
    return checks, metrics


def framing_report(
    rec: dict[str, np.ndarray],
    geometry: RobotGeometry,
    *,
    resolution: tuple[int, int],
    focal_length: float,
    horizontal_aperture: float,
    landmark_names: Sequence[str],
    extra_box: np.ndarray | None = None,
) -> dict[str, Any]:
    """Every collider hull inside the third-person image (≥ 2 % margin) and FK landmark error.

    ``extra_box`` (``[[min], [max]]`` world AABB, e.g. the target sofa) must also be in frame.
    """
    width, height = resolution
    ids = [geometry.body_names.index(name) for name in landmark_names]
    margins, min_depth, fk_error = [], math.inf, 0.0
    camera = dict(
        resolution=resolution, focal_length=focal_length, horizontal_aperture=horizontal_aperture
    )
    for i in range(len(rec["qpos"])):
        points = geometry.world_points(rec["robot_body_poses"][i])
        u, v, depth = project_points(points, rec["rear_pose"][i], **camera)
        margins.append(float(min(u.min(), width - u.max(), v.min(), height - v.max())))
        min_depth = min(min_depth, float(depth.min()))
        fk = geometry.body_matrices(rec["qpos"][i])
        fk_error = max(fk_error, float(np.abs(fk[ids, :3, 3] - rec["landmark_positions"][i]).max()))
    report = {
        "robot_collision_shapes": geometry.collider_count,
        "checked_frames": len(margins),
        "minimum_robot_frame_margin_px": min(margins),
        "worst_frame": int(np.argmin(margins)),
        "minimum_robot_depth_m": min_depth,
        "FK_landmark_max_error_m": fk_error,
    }
    passed = min(margins) >= width * 0.02 and min_depth > 0.03 and fk_error < 0.002
    if extra_box is not None:
        lo, hi = np.asarray(extra_box, dtype=float)
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        u, v, _ = project_points(corners, rec["rear_pose"][0], **camera)
        margin = float(min(u.min(), width - u.max(), v.min(), height - v.max()))
        report["minimum_target_frame_margin_px"] = margin
        passed = passed and margin >= width * 0.02
    report["passed"] = bool(passed)
    return report


def desk_passage_report(
    rec: dict[str, np.ndarray],
    geometry: RobotGeometry,
    desk: DeskSpec,
    *,
    mode: str,
    transition_time_s: float,
    centering: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lateral corridor margin, exit/rise timing and (crawl) trunk centring under the desk.

    ``mode`` is ``"crawl"`` (rise = pelvis > 0.42 m and pitch < 45° for 0.2 s) or ``"crouch"``
    (rise = pitch < 30°). ``transition_time_s`` is the reference time of the rise phase.
    """
    if mode not in {"crawl", "crouch"}:
        raise ValueError(f"Unknown desk mode: {mode}")
    n = len(rec["qpos"])
    half_depth = desk.depth_m / 2
    gap = desk.column_spacing_m - 0.092
    upper = geometry.upper_vertex_mask
    trailing_all, trailing_upper, corridor, widths, corridor_frames = [], [], [], [], []
    for i in range(n):
        local = desk.to_local(geometry.world_points(rec["robot_body_poses"][i]))
        trailing_all.append(float(local[:, 0].min()))
        trailing_upper.append(float(local[upper, 0].min()))
        inside = (
            (np.abs(local[:, 0]) <= half_depth)
            & (local[:, 2] >= -0.005)
            & (local[:, 2] <= desk.clearance_z_m)
        )
        if np.any(inside):
            lateral = local[inside, 1]
            corridor.append(float(min(lateral.min() + gap / 2, gap / 2 - lateral.max())))
            widths.append(float(np.ptp(lateral)))
            corridor_frames.append(i)
    passage = {
        "inner_column_clear_width_m": gap,
        "minimum_lateral_corridor_margin_m": min(corridor) if corridor else None,
        "maximum_body_width_within_table_footprint_m": max(widths) if widths else None,
        "checked_frames": len(corridor_frames),
        "worst_frame": corridor_frames[int(np.argmin(corridor))] if corridor else None,
        "passed": bool(corridor) and min(corridor) > 0.01,
    }
    upper_exit = _first_sustained(np.array(trailing_upper) > half_depth + 0.005)
    body_exit = _first_sustained(np.array(trailing_all) > half_depth + 0.005)
    pitch = _pitch_deg(rec["torso_pose"])
    times = np.arange(n) / CONTROL_HZ
    final_forward = float(desk.to_local(rec["qpos"][-1, :3])[0] - half_depth)
    final_pitch = float(np.median(pitch[-50:]))
    if mode == "crawl":
        onset = _first_sustained(
            (times >= transition_time_s - 0.9) & (pitch < 45) & (rec["qpos"][:, 2] > 0.42)
        )
    else:
        onset = _first_sustained((times >= transition_time_s - 0.9) & (pitch < 30))

    def at(index: int | None) -> float | None:
        return float(times[index]) if index is not None else None

    timing = {
        "upper_body_clear_time_s": at(upper_exit),
        "whole_body_clear_time_s": at(body_exit),
        "measured_rise_onset_s": at(onset),
        "reference_transition_s": transition_time_s,
        "rise_delay_after_upper_body_exit_s": at(onset) - at(upper_exit)
        if onset is not None and upper_exit is not None
        else None,
        "rise_delay_after_whole_body_exit_s": at(onset) - at(body_exit)
        if onset is not None and body_exit is not None
        else None,
        "upper_body_exit_margin_at_rise_onset_m": trailing_upper[onset] - half_depth
        if onset is not None
        else None,
        "final_base_distance_beyond_table_m": final_forward,
        "final_torso_pitch_median_deg": final_pitch,
    }
    if mode == "crawl":
        timing["passed"] = bool(
            onset is not None
            and body_exit is not None
            and -0.25 <= times[onset] - times[body_exit] <= 1.2
            and -0.5 <= transition_time_s - times[body_exit] <= 1.0
            and 0 < final_forward < 1.0
            and final_pitch < 10
        )
    else:
        delay = timing["rise_delay_after_upper_body_exit_s"]
        timing["passed"] = bool(
            delay is not None
            and -0.02 <= delay <= 1.2
            and trailing_upper[onset] >= half_depth
            and body_exit is not None
            and 0 < final_forward < 0.85
            and final_pitch < 10
        )
    report = {"table_passage": passage, "passage_timing": timing}
    if centering is not None:
        report["passage_centering"] = centering_report(
            rec, geometry, desk, transition_time_s=transition_time_s, **centering
        )
    return report


def centering_report(
    rec: dict[str, np.ndarray],
    geometry: RobotGeometry,
    desk: DeskSpec,
    *,
    transition_time_s: float,
    neck_body: str,
    neck_local_point_m: Sequence[float],
    maximum_median_bias_m: float,
    maximum_p95_offset_m: float,
    maximum_instant_offset_m: float,
) -> dict[str, Any]:
    """Lateral offset of the trunk centre (pelvis ↔ neck midpoint) from the desk centreline."""
    poses = rec["robot_body_poses"][:, geometry.body_names.index(neck_body)]
    neck = poses[:, :3] + np.einsum(
        "fij,j->fi", quat_wxyz_to_matrix(poses[:, 3:7]), np.asarray(neck_local_point_m)
    )
    local = desk.to_local((rec["qpos"][:, :3] + neck) / 2)
    mask = (np.abs(local[:, 0]) <= desk.depth_m / 2) & (rec["timestamp"] < transition_time_s)
    offset = local[mask, 1]
    if not len(offset):
        return {"checked_control_frames": 0, "passed": False}
    median, p95 = float(np.median(offset)), float(np.percentile(np.abs(offset), 95))
    maximum = float(np.abs(offset).max())
    return {
        "definition": "Midpoint of measured pelvis and neck point, lateral offset in desk frame",
        "checked_control_frames": int(mask.sum()),
        "median_signed_offset_m": median,
        "mean_signed_offset_m": float(np.mean(offset)),
        "p95_absolute_offset_m": p95,
        "maximum_absolute_offset_m": maximum,
        "limits": {
            "maximum_median_bias_m": maximum_median_bias_m,
            "maximum_p95_offset_m": maximum_p95_offset_m,
            "maximum_instant_offset_m": maximum_instant_offset_m,
        },
        "passed": bool(
            mask.sum() >= 50
            and abs(median) <= maximum_median_bias_m
            and p95 <= maximum_p95_offset_m
            and maximum <= maximum_instant_offset_m
        ),
    }


def under_desk_checks(
    rec: dict[str, np.ndarray],
    geometry: RobotGeometry,
    desk: DeskSpec,
    *,
    mode: str,
    under_phase: int,
    head_collider: str,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Traversal, head clearance, posture and support while under the desk, then the finish.

    crawl: hands/knees must carry load (non-foot ground impulse > 5 N·s), torso pitch ≥ 30°.
    crouch: feet only (< 0.02 N·s), knees > 0.10 m, wrists > 0.20 m, pitch 25–70°,
    at least two unloaded swing bouts per foot.
    """
    bent = mode == "crouch"
    mask = rec["phase_index"] == under_phase
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {"under_desk_control_frames": int(mask.sum())}
    if not mask.any():
        return {"under_desk_phase_recorded": False}, metrics
    half_depth, half_width = desk.depth_m / 2, desk.width_m / 2
    q = desk.to_local(rec["qpos"][mask, :3])
    h = desk.to_local(rec["head_pose"][mask, :3])
    checks["traverses_both_table_edges"] = bool(q[0, 0] < -half_depth and q[-1, 0] > half_depth)
    under = (np.abs(h[:, 0]) < half_depth) & (np.abs(h[:, 1]) < half_width)
    metrics["head_origin_max_z_under_table_m"] = float(h[under, 2].max()) if under.any() else None
    checks["head_passes_under_table"] = bool(
        under.sum() >= 50 and float(h[under, 2].max()) < desk.clearance_z_m
    ) if under.any() else False  # fmt: skip
    group = geometry.group_names.index(head_collider)
    head_body = int(geometry.group_body[group])
    vertices = geometry.group_points[group]
    poses = rec["robot_body_poses"][:, head_body]
    world = np.einsum("fij,vj->fvi", quat_wxyz_to_matrix(poses[:, 3:7]), vertices)
    world = world + poses[:, None, :3]
    metrics["standing_head_crown_height_m"] = float(np.median(world[:50, :, 2].max(1)))
    metrics["table_underside_height_m"] = desk.clearance_z_m
    checks["standing_robot_requires_lowering"] = (
        metrics["standing_head_crown_height_m"] > desk.clearance_z_m + 0.10
    )
    local = desk.to_local(world[mask].reshape(-1, 3)).reshape(-1, len(vertices), 3)
    inside = (
        (local[:, :, 0].min(1) > -half_depth)
        & (local[:, :, 0].max(1) < half_depth)
        & (local[:, :, 1].min(1) > -half_width)
        & (local[:, :, 1].max(1) < half_width)
    )
    clearance = (
        float(desk.clearance_z_m - world[mask][inside, :, 2].max()) if inside.any() else -1.0
    )
    metrics["minimum_head_collider_clearance_m"] = clearance
    checks["head_collider_passes_below_table"] = bool(inside.sum() >= 20 and clearance >= 0)
    pitch = _pitch_deg(rec["torso_pose"][mask])
    metrics["torso_forward_pitch_median_deg"] = float(np.median(pitch))
    checks["forward_bent_torso"] = bool(
        np.median(pitch) >= 25 and np.percentile(pitch, 95) < 70 if bent else np.median(pitch) >= 30
    )
    landmarks = rec["landmark_positions"][mask]
    metrics["minimum_knee_origin_height_m"] = float(landmarks[:, 2:4, 2].min())
    metrics["minimum_wrist_origin_height_m"] = float(landmarks[:, 4:6, 2].min())
    metrics["nonfoot_ground_impulse_Ns"] = float(rec["foot_support_impulses"][:, 2].sum())
    feet = rec["foot_support_impulses"][mask]
    bouts = []
    for side in (0, 1):
        swing = (feet[:, side] < 0.1) & (feet[:, 1 - side] > 0.5)
        edges = np.diff(np.r_[False, swing, False].astype(int))
        bouts.append(int(np.sum(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1) >= 4)))
    metrics["left_right_unloaded_foot_swing_bouts"] = bouts
    if bent:
        checks["knees_clear_of_ground"] = metrics["minimum_knee_origin_height_m"] > 0.10
        checks["hands_clear_of_ground"] = metrics["minimum_wrist_origin_height_m"] > 0.20
        checks["feet_only_ground_support"] = metrics["nonfoot_ground_impulse_Ns"] < 0.02
        checks["alternating_biped_steps"] = min(bouts) >= 2
    else:
        checks["crawl_uses_nonfoot_ground_support"] = metrics["nonfoot_ground_impulse_Ns"] > 5
    metrics["table_contact_impulse_Ns"] = float(rec["contacts"][:, TABLE].sum())
    checks["no_table_contact"] = metrics["table_contact_impulse_Ns"] < 0.001
    tail = rec["qpos"][-50:]
    up = 1 - 2 * (rec["qpos"][:, 4] ** 2 + rec["qpos"][:, 5] ** 2)
    metrics["last_second_position_range_m"] = np.ptp(tail[:, :3], axis=0).tolist()
    checks["ends_upright"] = bool(np.min(up[-50:]) > 0.8)
    checks["stops_after_rising"] = float(np.max(np.ptp(tail[:, :3], axis=0))) < 0.05
    checks["finishes_standing"] = float(tail[:, 2].min()) > 0.65
    return checks, metrics


def seated_checks(
    rec: dict[str, np.ndarray],
    *,
    cushion_center_xyz_m: Sequence[float],
    max_horizontal_center_error_m: float,
    max_lateral_center_error_m: float,
    robot_mass_kg: float,
    facing_axis_xy: Sequence[float] = (-1.0, 0.0),
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Seated on the target cushion, carried by the sofa, upright, facing ``facing_axis_xy``."""
    tail = rec["qpos"][-100:]
    center = np.asarray(cushion_center_xyz_m, dtype=float)
    metrics: dict[str, Any] = {
        "final_mean_pelvis_xyz_m": tail[:, :3].mean(0).tolist(),
        "seated_center_error_m": float(np.linalg.norm(tail[:, :2].mean(0) - center[:2])),
        "seated_lateral_error_m": float(abs(tail[:, 1].mean() - center[1])),
        "last_2s_position_range_m": np.ptp(tail[:, :3], axis=0).tolist(),
    }
    weight = robot_mass_kg * 9.81
    support = rec["contact_vertical_impulses"][-100:, :4].mean(0) * CONTROL_HZ
    metrics["last_2s_sofa_support_weight_fraction"] = float(support[CHAIR] / weight)
    metrics["last_2s_vertical_weight_balance_error"] = float(abs(support.sum() / weight - 1))
    torso = quat_wxyz_to_matrix(rec["torso_pose"][-100:, 3:7])
    facing = np.asarray(facing_axis_xy, dtype=float) / np.linalg.norm(facing_axis_xy)
    metrics["minimum_torso_up_cosine_last_2s"] = float(torso[:, 2, 2].min())
    metrics["minimum_torso_facing_cosine_last_2s"] = float(
        (torso[:, 0, 0] * facing[0] + torso[:, 1, 0] * facing[1]).min()
    )
    metrics["nonfoot_ground_impulse_Ns"] = float(rec["foot_support_impulses"][:, 2].sum())
    metrics["other_home_obstacle_impulse_Ns"] = float(rec["contacts"][:, HOME].sum())
    checks = {
        "centered_sit": metrics["seated_center_error_m"] <= max_horizontal_center_error_m
        and metrics["seated_lateral_error_m"] <= max_lateral_center_error_m,
        "stable_seated_finish": float(np.ptp(tail[:, :3], axis=0).max()) < 0.025
        and center[2] + 0.08 < float(tail[:, 2].mean()) < center[2] + 0.25,
        "sustained_sofa_support": support[CHAIR] / weight > 0.35
        and int(np.sum(rec["contacts"][-100:, CHAIR] > 0.001)) >= 95,
        "vertical_weight_balance": metrics["last_2s_vertical_weight_balance_error"] < 0.1,
        "upright_facing_target": metrics["minimum_torso_up_cosine_last_2s"] > 0.95
        and metrics["minimum_torso_facing_cosine_last_2s"] > 0.95,
        "no_fall_or_other_home_collision": metrics["nonfoot_ground_impulse_Ns"] < 0.05
        and metrics["other_home_obstacle_impulse_Ns"] < 0.05,
    }
    return {k: bool(v) for k, v in checks.items()}, metrics


def camera_motion_checks(
    rec: dict[str, np.ndarray], *, mode: str, follow: FollowCameraFilter | None = None
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Fixed camera never moves; the follow camera equals the filter replayed on measured bases."""
    metrics = {
        "third_person_position_range_m": float(np.max(np.ptp(rec["rear_pose"][:, :3], axis=0))),
        "third_person_orientation_range": float(np.max(np.ptp(rec["rear_pose"][:, 3:], axis=0))),
    }
    if mode == "fixed":
        ok = metrics["third_person_position_range_m"] < 1e-6
        ok = ok and metrics["third_person_orientation_range"] < 1e-6
        return {"third_person_camera_fixed_world": bool(ok)}, metrics
    if follow is None:
        raise ValueError("follow camera checks need the follow filter settings")
    planned = [follow.step(q, 1.0 / CONTROL_HZ) for q in rec["qpos"]]
    eyes = np.array([p[0] for p in planned])
    targets = np.array([p[1] for p in planned])
    metrics["follow_camera_max_eye_error_m"] = float(np.max(np.abs(eyes - rec["rear_pose"][:, :3])))
    metrics["follow_camera_max_target_error_m"] = float(
        np.max(np.abs(targets - rec["rear_target"]))
    )
    ok = (
        metrics["follow_camera_max_eye_error_m"] < 1e-5
        and metrics["follow_camera_max_target_error_m"] < 1e-5
        and metrics["third_person_position_range_m"] > 0.5
    )
    return {"third_person_camera_follows_measured_base": bool(ok)}, metrics
