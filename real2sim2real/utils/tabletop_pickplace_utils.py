"""Helpers of the tabletop pick-and-place task (Kit / pxr / cuMotion imports stay lazy).

Sections:

1. Phases, layout sampling and pick/place hand targets (pure numpy).
2. Sensor cameras referenced from a camera USD (housing + optical prim).
3. PhysX contact reports grouped by body category.
4. Physical acceptance shared by the Job and ``scripts/tabletop_pickplace/validate_*.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from real2sim2real.utils.battery_disassembly_utils import quat_to_matrix, tilt_deg
from real2sim2real.utils.cameras import CameraConfig, create_cameras

# ================================================================================================
# 1. Phases, layout sampling and pick/place hand targets
# ================================================================================================

PHASES = (
    "start",
    "approach_pick",
    "descend",
    "close",
    "lift",
    "transfer",
    "lower",
    "open",
    "retreat",
    "rest",
    "done",
)
PHASE_INDEX = {name: index for index, name in enumerate(PHASES)}
MOTION_PHASES = ("approach_pick", "descend", "lift", "transfer", "lower", "retreat")
# Phases during which the target must stay in the closed gripper.
HOLD_PHASES = ("lift", "transfer", "lower")


def to_numpy(value: Any) -> np.ndarray:
    """Copy a warp / torch / numpy tensor to numpy."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def yaw_quat(yaw_deg: float) -> np.ndarray:
    """wxyz quaternion of a rotation about world +Z."""
    half = math.radians(yaw_deg) / 2
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def set_prim_pose(stage: Any, path: str, position: Any, quat: Any) -> None:
    """Author a prim's pose as one ``xformOp:transform:rsr`` op (referenced assets keep their
    own, possibly float-precision, translate/orient/scale attributes out of the op order)."""
    from pxr import Gf, UsdGeom

    matrix = np.eye(4)
    matrix[:3, :3] = quat_to_matrix(quat)
    matrix[:3, 3] = np.asarray(position, float)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xform.ClearXformOpOrder()
    op = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "rsr")
    op.Set(Gf.Matrix4d(matrix.T.tolist()))


def table_point(center: Any, yaw_deg: float, offset_xy: Any) -> np.ndarray:
    """World XY of an offset given in the table frame (table centre, rotated by ``yaw_deg``)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = (float(v) for v in offset_xy)
    return np.asarray(center, dtype=float)[:2] + np.array([c * dx - s * dy, s * dx + c * dy])


def footprint_gap(center_a: Any, half_a: Any, center_b: Any, half_b: Any) -> float:
    """Distance in metres between two axis-aligned XY rectangles (negative when overlapping)."""
    delta = np.abs(np.asarray(center_a, float)[:2] - np.asarray(center_b, float)[:2])
    gap = delta - np.asarray(half_a, float)[:2] - np.asarray(half_b, float)[:2]
    if np.all(gap <= 0):
        return float(np.max(gap))
    return float(np.linalg.norm(np.clip(gap, 0.0, None)))


def jitter_offset(
    rng: np.random.Generator,
    radius_m: float,
    accept: Callable[[np.ndarray], bool] | None = None,
    *,
    max_draws: int = 1000,
) -> tuple[np.ndarray, int]:
    """Uniform XY jitter in a square of half side ``radius_m`` (source ``jitter_world_xy``).

    ``accept`` rejects layouts that violate clearances; the draw count is returned so the
    layout records whether the source's random sequence was preserved (1 draw per episode).
    """
    for draws in range(1, max_draws + 1):
        offset = np.zeros(2)
        if radius_m > 0:
            offset = np.array([rng.uniform(-radius_m, radius_m), rng.uniform(-radius_m, radius_m)])
        if accept is None or accept(offset):
            return offset, draws
    raise RuntimeError(f"No layout satisfied the clearance constraints in {max_draws} draws")


def world_to_base(base_pos: Any, base_quat: Any, point: Any) -> np.ndarray:
    """Express a world point in the robot base frame."""
    rotation = quat_to_matrix(base_quat)
    return rotation.T @ (np.asarray(point, float) - np.asarray(base_pos, float))


def hand_targets(
    pick: Any, place: Any, *, hand_above_m: float, clearance_m: float
) -> dict[str, np.ndarray]:
    """Hand (``panda_hand``) positions of every motion phase, in the frame of ``pick``/``place``.

    ``pick`` is the object centre; ``place`` the object centre at release. The hand stays
    ``hand_above_m`` above the object centre while holding it, ``clearance_m`` higher above
    pick / place during approach, lift, transfer and retreat (source cuRobo ``_make_segments``).
    """
    pick, place = np.asarray(pick, float), np.asarray(place, float)
    up = np.array([0.0, 0.0, 1.0])
    pick_hand, place_hand = pick + hand_above_m * up, place + hand_above_m * up
    return {
        "approach_pick": pick_hand + clearance_m * up,
        "descend": pick_hand,
        "lift": pick_hand + clearance_m * up,
        "transfer": place_hand + clearance_m * up,
        "lower": place_hand,
        "retreat": place_hand + clearance_m * up,
    }


def sample_trajectory(
    evaluate: Callable[[float], Any], start: float, span: float, dt: float, time_scale: float = 1.0
) -> np.ndarray:
    """Sample a continuous trajectory at the physics rate; ``time_scale > 1`` slows it down."""
    if span < 0 or dt <= 0 or time_scale <= 0:
        raise ValueError("Trajectory span must be non-negative; dt and time_scale positive")
    steps = max(1, math.ceil(span * time_scale / dt - 1e-9))
    times = np.minimum(np.arange(steps + 1) * dt / time_scale, span)
    return np.asarray([np.asarray(evaluate(start + t), float).reshape(-1) for t in times])


def relative_position(hand_pos: Any, hand_quat: Any, point: Any) -> np.ndarray:
    """Points expressed in the hand frame; one row per (hand pose, point) pair."""
    hand_pos = np.atleast_2d(np.asarray(hand_pos, float))
    hand_quat = np.atleast_2d(np.asarray(hand_quat, float))
    point = np.atleast_2d(np.asarray(point, float))
    rows = zip(hand_pos, hand_quat, point, strict=True)
    return np.asarray([quat_to_matrix(q).T @ (p - h) for h, q, p in rows])


# ================================================================================================
# 2. Sensor cameras: reference a camera USD (housing + optical Camera prim) at a mount pose
# ================================================================================================


class SensorCameraConfig(CameraConfig):
    """Camera entry of ``[[scene.cameras]]`` / ``[[robot.cameras]]``.

    With ``usd_path`` the sensor USD is referenced at ``prim_path`` and ``pose`` places its
    housing; intrinsics (including fisheye projections) come from the optical prim
    ``optical_prim`` inside the USD and the lens fields of :class:`CameraConfig` are unused.
    Without ``usd_path`` a pinhole camera is created from the lens fields.
    """

    usd_path: str = ""
    optical_prim: str = Field(default="", description="Camera prim path relative to prim_path")
    hide_housing: bool = True

    @model_validator(mode="after")
    def _validate_sensor(self) -> Self:
        if bool(self.usd_path) != bool(self.optical_prim):
            raise ValueError("Sensor cameras need both usd_path and optical_prim")
        if self.optical_prim and (
            self.optical_prim.startswith("/")
            or not all(part.isidentifier() for part in self.optical_prim.split("/"))
        ):
            raise ValueError("optical_prim must be a relative prim path")
        return self

    @property
    def capture_prim(self) -> str:
        """USD Camera prim rendered for this camera."""
        return f"{self.prim_path}/{self.optical_prim}" if self.optical_prim else self.prim_path


def create_sensor_cameras(stage: Any, cameras: list[SensorCameraConfig]) -> dict[str, str]:
    """Create every camera below its mount frame; returns name -> rendered Camera prim path."""
    from pxr import Gf, Usd, UsdGeom

    from real2sim2real.utils.asset_utils import resolve_usd_path

    plain = [c for c in cameras if not c.usd_path]
    create_cameras(stage, plain)
    out = {c.name: c.capture_prim for c in plain}
    cache = UsdGeom.XformCache()
    for config in (c for c in cameras if c.usd_path):
        reference = stage.GetPrimAtPath(config.pose_relative_to)
        if not reference or not reference.IsA(UsdGeom.Xformable):
            raise ValueError(f"Camera reference frame does not exist: {config.pose_relative_to}")
        parent_path = config.prim_path.rsplit("/", 1)[0]
        if not stage.GetPrimAtPath(parent_path):
            UsdGeom.Xform.Define(stage, parent_path)
        housing = UsdGeom.Xform.Define(stage, config.prim_path)
        housing.GetPrim().GetReferences().AddReference(resolve_usd_path(config.usd_path))
        pose = Gf.Matrix4d(1.0)
        q = config.pose.orientation_wxyz
        pose.SetRotate(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
        pose.SetTranslateOnly(Gf.Vec3d(*config.pose.position))
        reference_world = cache.GetLocalToWorldTransform(reference)
        parent_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path))
        housing.ClearXformOpOrder()
        housing.AddTransformOp().Set(pose * reference_world * parent_world.GetInverse())
        optical = stage.GetPrimAtPath(config.capture_prim)
        if not optical or not optical.IsA(UsdGeom.Camera):
            raise ValueError(f"Sensor USD has no Camera prim at {config.capture_prim}")
        UsdGeom.Camera(optical).GetClippingRangeAttr().Set(Gf.Vec2f(*config.clipping_range))
        if config.hide_housing:
            for prim in Usd.PrimRange(housing.GetPrim()):
                if prim.IsA(UsdGeom.Gprim):
                    UsdGeom.Imageable(prim).MakeInvisible()
        out[config.name] = config.capture_prim
    return out


# ================================================================================================
# 3. PhysX contact reports grouped by body category
# ================================================================================================


class ContactMonitor:
    """Accumulate PhysX contact impulses per category pair and phase.

    ``bodies`` are rigid-body prims that receive ``PhysxContactReportAPI`` (threshold 0);
    ``classify`` maps any reported actor path to a category (unknown -> ``environment``).
    Must be created before the physics views are initialized.
    """

    def __init__(self, stage: Any, bodies: list[str], classify: Callable[[str], str | None]):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools, PhysxSchema

        for path in bodies:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError(f"Contact report body does not exist: {path}")
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr(0.0)
        self.classify = classify
        self.to_path = PhysicsSchemaTools.intToSdfPath
        self.cache: dict[int, str] = {}
        self.time = 0.0
        self.phase = "pre_roll"
        self.reset()
        interface = get_physx_simulation_interface()
        self.subscription = interface.subscribe_contact_report_events(self._callback)

    def reset(self) -> None:
        """Start a new episode."""
        self.impulses: dict[str, float] = {}
        self.by_phase: dict[str, dict[str, float]] = {}
        self.first_time: dict[str, float] = {}
        self.last_time: dict[str, float] = {}

    def _category(self, actor: int) -> str:
        if actor not in self.cache:
            self.cache[actor] = self.classify(str(self.to_path(actor))) or "environment"
        return self.cache[actor]

    def _callback(self, headers: Any, data: Any) -> None:
        for header in headers:
            a, b = sorted((self._category(header.actor0), self._category(header.actor1)))
            start = header.contact_data_offset
            impulse = sum(
                float(np.linalg.norm(np.asarray(data[j].impulse, dtype=float)))
                for j in range(start, start + header.num_contact_data)
            )
            self.add(f"{a}|{b}", impulse)

    def add(self, key: str, impulse: float) -> None:
        """Record one contact impulse (N·s) at the current time and phase."""
        self.impulses[key] = self.impulses.get(key, 0.0) + impulse
        phase = self.by_phase.setdefault(self.phase, {})
        phase[key] = phase.get(key, 0.0) + impulse
        if impulse > 0:
            self.first_time.setdefault(key, self.time)
            self.last_time[key] = self.time

    def summary(self) -> dict[str, Any]:
        """JSON-ready totals."""
        return {
            "impulses_ns": dict(sorted(self.impulses.items())),
            "by_phase_ns": {k: dict(sorted(v.items())) for k, v in self.by_phase.items()},
            "first_time_s": dict(sorted(self.first_time.items())),
            "last_time_s": dict(sorted(self.last_time.items())),
        }

    def close(self) -> None:
        self.subscription = None


# ================================================================================================
# 4. Physical acceptance (state log + layout + contacts), shared with the validation script
# ================================================================================================


class PickPlaceAcceptanceConfig(BaseModel):
    """Limits of the physical checks; lengths in metres, impulses in N·s."""

    model_config = ConfigDict(extra="forbid")
    min_lift_m: FiniteFloat = Field(default=0.08, gt=0, description="object rise while held")
    max_hold_slip_m: FiniteFloat = Field(default=0.01, gt=0, description="drift in hand frame")
    max_place_error_m: FiniteFloat = Field(default=0.02, gt=0, description="final XY error")
    max_support_gap_m: FiniteFloat = Field(default=0.003, gt=0, description="bottom vs table top")
    max_tilt_deg: FiniteFloat = Field(default=5.0, gt=0)
    rest_window_s: FiniteFloat = Field(default=0.5, gt=0)
    max_rest_motion_m: FiniteFloat = Field(default=0.002, gt=0)
    max_release_follow_m: FiniteFloat = Field(default=0.003, gt=0, description="rise on retreat")
    min_open_fraction: FiniteFloat = Field(default=0.9, gt=0, le=1, description="final opening")
    max_distractor_motion_m: FiniteFloat = Field(
        default=0.01, gt=0, description="after the arm starts moving"
    )
    max_arm_contact_impulse_ns: FiniteFloat = Field(default=1e-3, ge=0)
    max_stray_contact_impulse_ns: FiniteFloat = Field(default=0.01, ge=0)


# Contact categories: robot links and scene bodies.
FINGERS = ("finger_left", "finger_right")
ARM = ("arm", "hand")


def _pairs(contacts: dict[str, float], a: str, others: tuple[str, ...]) -> float:
    total = 0.0
    for key, value in contacts.items():
        x, y = key.split("|")
        if (x == a and y in others) or (y == a and x in others):
            total += value
    return total


def evaluate_pickplace(
    log: dict[str, np.ndarray],
    *,
    layout: dict[str, Any],
    contacts: dict[str, float],
    acceptance: PickPlaceAcceptanceConfig,
    sequence_completed: bool,
) -> dict[str, Any]:
    """Physical checks of one episode from the full-rate state log.

    ``log`` keys: ``time``, ``phase_index``, ``object_pos``, ``object_quat``, ``hand_pos``,
    ``hand_quat``, ``finger_q`` (N, 2), ``distractor_pos`` (N, K, 3). ``layout`` keys:
    ``table_top_z``, ``object_half_extents_m``, ``place_target_xy``, ``gripper_open_m``.
    """
    acc = acceptance
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}
    phase = np.asarray(log["phase_index"]).astype(int)
    pos, quat = np.asarray(log["object_pos"], float), np.asarray(log["object_quat"], float)
    hand_pos, hand_quat = np.asarray(log["hand_pos"], float), np.asarray(log["hand_quat"], float)
    finite = (bool(np.isfinite(np.asarray(v, float)).all()) for v in log.values())
    checks["finite_states"] = all(finite)
    checks["sequence_completed"] = bool(sequence_completed)

    def mask(*names: str) -> np.ndarray:
        return np.isin(phase, [PHASE_INDEX[n] for n in names])

    held = mask(*HOLD_PHASES)
    lift = float(pos[held, 2].max() - pos[0, 2]) if held.any() else 0.0
    metrics["object_lift_m"] = lift
    checks["object_lifted"] = lift >= acc.min_lift_m
    if held.any():
        rel = relative_position(hand_pos[held], hand_quat[held], pos[held])
        slip = float(np.linalg.norm(rel - rel[0], axis=1).max())
        metrics["grasp_offset_in_hand_m"] = rel[0].round(4).tolist()
    else:
        slip = math.inf
    metrics["hold_slip_m"] = slip
    checks["object_held_without_slip"] = slip <= acc.max_hold_slip_m

    final_pos, final_quat = pos[-1], quat[-1]
    error = float(np.linalg.norm(final_pos[:2] - np.asarray(layout["place_target_xy"], float)))
    metrics["place_error_m"] = error
    checks["placed_at_target"] = error <= acc.max_place_error_m
    half_z = float(layout["object_half_extents_m"][2])
    gap = float(final_pos[2] - half_z - float(layout["table_top_z"]))
    metrics["support_gap_m"] = gap
    checks["resting_on_table"] = abs(gap) <= acc.max_support_gap_m
    tilt = tilt_deg(final_quat)
    metrics["final_tilt_deg"] = tilt
    checks["object_upright"] = tilt <= acc.max_tilt_deg
    times = np.asarray(log["time"], float)
    window = times >= times[-1] - acc.rest_window_s
    motion = float(np.linalg.norm(pos[window] - pos[-1], axis=1).max())
    metrics["rest_motion_m"] = motion
    checks["object_at_rest"] = motion <= acc.max_rest_motion_m

    retreat = mask("retreat")
    opened = np.nonzero(mask("open"))[0]
    if retreat.any() and len(opened):
        follow = float(pos[retreat, 2].max() - pos[opened[-1], 2])
    else:
        follow = math.inf
    metrics["release_follow_m"] = follow
    opening = float(np.asarray(log["finger_q"], float)[-1].min())
    metrics["final_finger_opening_m"] = opening
    checks["object_released"] = (
        follow <= acc.max_release_follow_m
        and opening >= acc.min_open_fraction * float(layout["gripper_open_m"])
    )
    # Distractors settle (a round apple rolls) during the start hold, before the arm moves;
    # the check covers the motion from the first robot motion phase on.
    distractors = np.asarray(log["distractor_pos"], float)
    moving = np.nonzero(phase >= PHASE_INDEX["approach_pick"])[0]
    first = int(moving[0]) if len(moving) else len(phase) - 1
    if distractors.size:
        moved = float(np.linalg.norm(distractors[first:] - distractors[first], axis=2).max())
        settle = float(np.linalg.norm(distractors[first] - distractors[0], axis=1).max())
    else:
        moved = settle = 0.0
    metrics["distractor_motion_m"] = moved
    metrics["distractor_settle_m"] = settle
    checks["distractors_undisturbed"] = moved <= acc.max_distractor_motion_m

    robot = (*ARM, *FINGERS, "base")
    others = ("target", "distractor", "table", "environment", *robot)
    arm = sum(_pairs(contacts, part, others) for part in ARM)
    metrics["arm_contact_impulse_ns"] = arm
    checks["arm_links_collision_free"] = arm <= acc.max_arm_contact_impulse_ns
    stray = sum(_pairs(contacts, f, ("distractor", "table", "environment")) for f in FINGERS)
    metrics["finger_stray_impulse_ns"] = stray
    checks["fingers_touch_only_target"] = stray <= acc.max_stray_contact_impulse_ns
    target_stray = _pairs(contacts, "target", ("distractor", "environment", *ARM, "base"))
    metrics["target_stray_impulse_ns"] = target_stray
    checks["target_touches_only_table_and_fingers"] = (
        target_stray <= acc.max_stray_contact_impulse_ns
    )
    grip = {f: _pairs(contacts, f, ("target",)) for f in FINGERS}
    metrics["finger_target_impulse_ns"] = grip
    checks["both_fingers_contacted_target"] = all(v > 0 for v in grip.values())
    checks = {k: bool(v) for k, v in checks.items()}
    return {
        "success": all(checks.values()),
        "checks": checks,
        "failed_checks": [k for k, ok in checks.items() if not ok],
        "metrics": metrics,
        "contacts_ns": {k: round(v, 6) for k, v in sorted(contacts.items())},
    }
