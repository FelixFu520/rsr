"""Pure numpy helpers of the dual-KR210 battery disassembly task (no Kit / pxr imports).

Sections:

1. Rotations and camera orientation.
2. KR210 kinematics from the URDF (FK, geometric Jacobian, damped least-squares IK).
3. Joint paths: Cartesian lines through IK, densification, smoothstep timing with velocity.
4. Collision proxy layouts in the Battery.usd asset frame (boxes only).
5. Physical laws used by the controller (magnetic bit holding force).
6. Acceptance helpers shared by the Job and ``scripts/battery_disassembly/validate_*.py``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ================================================================================================
# 1. Rotations and camera orientation (quaternions are scalar-first wxyz)
# ================================================================================================


def quat_to_matrix(q: Any) -> np.ndarray:
    """Rotation matrix of a (not necessarily unit) wxyz quaternion."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(rotation: Any) -> np.ndarray:
    """wxyz quaternion (w >= 0) of a rotation matrix."""
    m = np.asarray(rotation, dtype=float)
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def yaw_matrix(yaw_deg: float) -> np.ndarray:
    """Rotation about +Z."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def zyx_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll); positive pitch tilts +X downward."""
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def transform(rotation: Any = None, translation: Any = (0.0, 0.0, 0.0)) -> np.ndarray:
    """4x4 homogeneous transform (column vectors)."""
    out = np.eye(4)
    if rotation is not None:
        out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


# USD camera axes (+X right, +Y up, -Z forward) expressed in a "housing" frame that looks along
# +X with +Z up (RealSense D455 convention used by the source configs).
_CAMERA_IN_HOUSING = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def housing_camera_quat(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """USD camera orientation of a +X-looking housing posed with yaw/pitch/roll (degrees)."""
    return matrix_to_quat(zyx_matrix(yaw_deg, pitch_deg, roll_deg) @ _CAMERA_IN_HOUSING)


def look_at_camera_quat(eye: Any, target: Any, up: Any = (0.0, 0.0, 1.0)) -> np.ndarray:
    """USD camera orientation (-Z toward ``target``, +Y toward ``up``)."""
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=float))
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("Camera up vector is parallel to the viewing direction")
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return matrix_to_quat(np.column_stack([right, true_up, -forward]))


# ================================================================================================
# 2. KR210 kinematics from the URDF
# ================================================================================================


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    return zyx_matrix(math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


@dataclass
class SerialChain:
    """Revolute chain base_link → tip parsed from a URDF (SI units, radians)."""

    origins: list[np.ndarray]  # fixed 4x4 transform before each revolute joint
    axes: list[np.ndarray]  # joint axis in the joint frame
    lower: np.ndarray
    upper: np.ndarray
    tip: np.ndarray  # fixed transform after the last revolute joint up to the tip link
    names: list[str] = field(default_factory=list)

    @classmethod
    def from_urdf(cls, path: str | Path, tip: str = "tool0") -> SerialChain:
        root = ET.parse(str(path)).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain = []
        child = tip
        while child in by_child:
            joint = by_child[child]
            chain.append(joint)
            child = joint.find("parent").get("link")
        chain.reverse()
        origins, axes, lower, upper, names = [], [], [], [], []
        pending = np.eye(4)
        for joint in chain:
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            pending = pending @ transform(_rpy_matrix(rpy), xyz)
            if joint.get("type") in {"revolute", "continuous"}:
                origins.append(pending)
                pending = np.eye(4)
                axes.append(np.fromstring(joint.find("axis").get("xyz"), sep=" "))
                limit = joint.find("limit")
                lower.append(float(limit.get("lower")))
                upper.append(float(limit.get("upper")))
                names.append(joint.get("name"))
        return cls(origins, axes, np.asarray(lower), np.asarray(upper), pending, names)

    @property
    def dof(self) -> int:
        return len(self.axes)

    def fk(
        self, q: Any, base: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tip pose (4x4) plus world origins / axes of every joint."""
        frame = np.eye(4) if base is None else np.asarray(base, dtype=float).copy()
        points, axes = [], []
        for origin, axis, angle in zip(self.origins, self.axes, np.asarray(q, float), strict=True):
            frame = frame @ origin
            points.append(frame[:3, 3].copy())
            axes.append(frame[:3, :3] @ axis)
            frame = frame @ transform(_axis_angle(axis, float(angle)))
        return frame @ self.tip, np.asarray(points), np.asarray(axes)

    def jacobian(self, q: Any, base: np.ndarray, point: np.ndarray) -> np.ndarray:
        """6xN geometric Jacobian (linear; angular) of a point rigidly attached to the tip."""
        _, origins, axes = self.fk(q, base)
        jac = np.zeros((6, self.dof))
        for i in range(self.dof):
            jac[:3, i] = np.cross(axes[i], point - origins[i])
            jac[3:, i] = axes[i]
        return jac


@dataclass(frozen=True)
class ToolFrame:
    """TCP relative to the chain tip (``tool0``)."""

    offset: np.ndarray  # TCP position in the tip frame
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # TCP axes in the tip frame


def tcp_pose(chain: SerialChain, base: np.ndarray, tool: ToolFrame, q: Any) -> np.ndarray:
    """World TCP pose for joint angles ``q``."""
    tip, _, _ = chain.fk(q, base)
    return tip @ transform(tool.rotation, tool.offset)


def _wrap_near(q: np.ndarray, seed: np.ndarray, chain: SerialChain) -> np.ndarray:
    """Move periodic wrist joints by 2*pi toward ``seed`` when the limits allow it."""
    q = q.copy()
    for i in range(chain.dof):
        if chain.upper[i] - chain.lower[i] <= 2 * math.pi:
            continue
        while q[i] - seed[i] > math.pi and q[i] - 2 * math.pi >= chain.lower[i]:
            q[i] -= 2 * math.pi
        while seed[i] - q[i] > math.pi and q[i] + 2 * math.pi <= chain.upper[i]:
            q[i] += 2 * math.pi
    return q


def solve_ik(
    chain: SerialChain,
    base: np.ndarray,
    tool: ToolFrame,
    position: Any,
    *,
    rotation: Any | None = None,
    axis: Any | None = None,
    axis_index: int = 0,
    seed: Any,
    iterations: int = 200,
    position_tol: float = 2e-4,
    angle_tol: float = math.radians(0.1),
) -> tuple[np.ndarray, float, float]:
    """Damped least-squares IK; returns (q, position error m, orientation error rad).

    ``rotation`` fixes the full TCP orientation. ``axis`` only aligns TCP axis ``axis_index``
    with a world direction and leaves rotation about it free (screwdriver: 5-DOF task).
    """
    if (rotation is None) == (axis is None):
        raise ValueError("Give exactly one of rotation or axis")
    target = np.asarray(position, dtype=float)
    q = np.clip(np.asarray(seed, dtype=float).copy(), chain.lower, chain.upper)
    pos_err = ang_err = math.inf
    for _ in range(iterations):
        pose = tcp_pose(chain, base, tool, q)
        error_p = target - pose[:3, 3]
        if rotation is not None:
            goal = np.asarray(rotation, dtype=float)
            error_w = 0.5 * sum(np.cross(pose[:3, i], goal[:, i]) for i in range(3))
        else:
            direction = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
            error_w = np.cross(pose[:3, axis_index], direction)
        pos_err, ang_err = float(np.linalg.norm(error_p)), float(np.linalg.norm(error_w))
        if rotation is None:
            ang_err = math.acos(np.clip(np.dot(pose[:3, axis_index], direction), -1.0, 1.0))
        if pos_err < position_tol and ang_err < angle_tol:
            break
        jac = chain.jacobian(q, base, pose[:3, 3])
        error = np.concatenate([error_p, error_w])
        damping = 1e-4 + 1e-2 * min(1.0, pos_err)
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), error)
        q = np.clip(q + np.clip(step, -0.2, 0.2), chain.lower, chain.upper)
    return _wrap_near(q, np.asarray(seed, dtype=float), chain), pos_err, ang_err


# ================================================================================================
# 3. Joint paths and timing
# ================================================================================================


def line_points(start: Any, end: Any, spacing: float = 0.005) -> np.ndarray:
    """Points on a straight line, at most ``spacing`` apart (both ends included)."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    count = max(2, int(math.ceil(np.linalg.norm(end - start) / spacing)) + 1)
    return np.linspace(start, end, count)


def densify(path: Any, max_step: float = 0.01) -> np.ndarray:
    """Insert linear joint samples so no joint moves more than ``max_step`` rad per sample."""
    path = np.asarray(path, dtype=float)
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:], strict=True):
        count = max(1, int(math.ceil(np.max(np.abs(b - a)) / max_step)))
        out.extend(a + (b - a) * (i / count) for i in range(1, count + 1))
    return np.asarray(out)


def joint_blend(start: Any, end: Any, max_step: float = 0.01) -> np.ndarray:
    """Straight joint-space path between two configurations."""
    return densify(np.stack([np.asarray(start, float), np.asarray(end, float)]), max_step)


def solve_line(
    chain: SerialChain,
    base: np.ndarray,
    tool: ToolFrame,
    points: np.ndarray,
    *,
    seed: Any,
    rotation: Any | None = None,
    axis: Any | None = None,
    label: str = "",
    max_joint_step: float = 0.05,
    position_tol: float = 5e-4,
    angle_tol: float = math.radians(0.3),
) -> np.ndarray:
    """IK along Cartesian points seeded sequentially; raises when the path breaks."""
    q = np.asarray(seed, dtype=float)
    path = []
    for i, point in enumerate(points):
        solved, pe, ae = solve_ik(
            chain, base, tool, point, rotation=rotation, axis=axis, seed=q, iterations=400
        )
        if pe > position_tol or ae > angle_tol:
            raise RuntimeError(
                f"IK failed on {label}[{i}]: {pe * 1e3:.2f} mm, {math.degrees(ae):.2f} deg"
            )
        if path and np.max(np.abs(solved - path[-1])) > max_joint_step:
            raise RuntimeError(
                f"IK path {label} jumps {np.max(np.abs(solved - path[-1])):.3f} rad at sample {i}"
            )
        path.append(solved)
        q = solved
    return np.asarray(path)


def smoothstep(x: float) -> tuple[float, float]:
    """Cubic smoothstep value and derivative on [0, 1]."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x), 6 * x * (1 - x)


@dataclass
class TimedPath:
    """Joint path traversed by arclength with smoothstep timing (zero end velocities)."""

    path: np.ndarray
    duration: float
    cumulative: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.path = np.asarray(self.path, dtype=float)
        steps = np.linalg.norm(np.diff(self.path, axis=0), axis=1) if len(self.path) > 1 else []
        self.cumulative = np.concatenate([[0.0], np.cumsum(steps)])
        if self.duration <= 0:
            raise ValueError("Path duration must be positive")

    @property
    def length(self) -> float:
        return float(self.cumulative[-1])

    def at_fraction(self, fraction: float) -> np.ndarray:
        """Configuration at a fraction of the arclength."""
        if self.length < 1e-12:
            return self.path[-1].copy()
        distance = min(max(fraction, 0.0), 1.0) * self.length
        i = int(
            np.clip(
                np.searchsorted(self.cumulative, distance, side="right") - 1, 0, len(self.path) - 2
            )
        )
        span = max(self.cumulative[i + 1] - self.cumulative[i], 1e-12)
        a = (distance - self.cumulative[i]) / span
        return (1 - a) * self.path[i] + a * self.path[i + 1]

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity targets at time ``t`` (s)."""
        s, ds = smoothstep(t / self.duration)
        q = self.at_fraction(s)
        if self.length < 1e-12 or t <= 0 or t >= self.duration:
            return q, np.zeros_like(q)
        eps = 1e-4
        tangent = (self.at_fraction(min(1.0, s + eps)) - self.at_fraction(max(0.0, s - eps))) / (
            min(1.0, s + eps) - max(0.0, s - eps)
        )
        return q, tangent * ds / self.duration


# ================================================================================================
# 4. Collision proxy layouts (asset frame of Battery.usd, metres)
# ================================================================================================


@dataclass(frozen=True)
class Box:
    """Axis-aligned box given by min / max corners."""

    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (np.asarray(self.lo) + np.asarray(self.hi))

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.hi) - np.asarray(self.lo)

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))


def flange_boxes(
    x_half: float,
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    holes_xy: list[tuple[float, float]],
    hole_half: float,
) -> list[Box]:
    """Rectangular plate split into boxes leaving square holes; holes come in ±x pairs per row."""
    rows = sorted({round(y, 6) for _, y in holes_xy})
    z0, z1 = z_range
    boxes, y = [], y_range[0]
    for row in rows:
        if row - hole_half > y:
            boxes.append(
                Box(f"flange_{len(boxes)}", (-x_half, y, z0), (x_half, row - hole_half, z1))
            )
        xs = sorted(x for x, yy in holes_xy if abs(yy - row) < 1e-6)
        edges = [-x_half] + [v for x in xs for v in (x - hole_half, x + hole_half)] + [x_half]
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            if b - a > 1e-4:
                boxes.append(
                    Box(f"flange_{len(boxes)}", (a, row - hole_half, z0), (b, row + hole_half, z1))
                )
        y = row + hole_half
    if y_range[1] > y:
        boxes.append(Box(f"flange_{len(boxes)}", (-x_half, y, z0), (x_half, y_range[1], z1)))
    return boxes


def cover_boxes(spec: Any, holes_xy: list[tuple[float, float]]) -> list[Box]:
    """Hollow cover: flange plate with screw holes, side walls, stepped top and raised section."""
    f0, f1 = spec.flange_z_m
    x, w = spec.body_x_half_m, spec.wall_m
    y0, y1 = spec.body_y_m
    r0, r1 = spec.raised_y_m
    top, raised = spec.body_top_z_m, spec.raised_top_z_m
    boxes = flange_boxes(
        spec.flange_x_half_m, spec.flange_y_m, (f0, f1), holes_xy, spec.hole_half_m
    )
    boxes += [
        Box("wall_xn_low", (-x, y0, f1), (-x + w, r0, top)),
        Box("wall_xp_low", (x - w, y0, f1), (x, r0, top)),
        Box("wall_xn_high", (-x, r0, f1), (-x + w, y1, raised)),
        Box("wall_xp_high", (x - w, r0, f1), (x, y1, raised)),
        Box("wall_y0", (-x + w, y0, f1), (x - w, y0 + w, top)),
        Box("wall_y1", (-x + w, y1 - w, f1), (x - w, y1, raised)),
        Box("wall_step", (-x + w, r0, top - w), (x - w, r0 + w, raised)),
        Box("top_main", (-x + w, y0 + w, top - w), (x - w, r0, top)),
        Box("top_raised", (-x + w, r0 + w, raised - w), (x - w, y1 - w, raised)),
    ]
    return boxes


def basket_boxes(lo: Any, hi: Any, wall: float, floor: float) -> list[Box]:
    """Open box: floor and four walls inside the basket bounds."""
    (x0, y0, z0), (x1, y1, z1) = np.asarray(lo, float), np.asarray(hi, float)
    return [
        Box("floor", (x0, y0, z0), (x1, y1, z0 + floor)),
        Box("wall_x0", (x0, y0, z0), (x0 + wall, y1, z1)),
        Box("wall_x1", (x1 - wall, y0, z0), (x1, y1, z1)),
        Box("wall_y0", (x0 + wall, y0, z0), (x1 - wall, y0 + wall, z1)),
        Box("wall_y1", (x0 + wall, y1 - wall, z0), (x1 - wall, y1, z1)),
    ]


def boxes_centroid(boxes: list[Box]) -> np.ndarray:
    """Volume-weighted centroid (uniform density)."""
    volumes = np.array([b.volume for b in boxes])
    return (np.stack([b.center for b in boxes]) * volumes[:, None]).sum(axis=0) / volumes.sum()


def screw_head_boxes(spec: Any) -> list[Box]:
    """Cross-recess quadrants on top of the head base (screw frame: head underside at z=0)."""
    w, a = 0.5 * spec.recess_width_m, spec.recess_half_m
    z0 = spec.head_base_m
    z1 = z0 + spec.recess_depth_m
    return [
        Box(
            f"quad_{i}",
            (min(sx * w, sx * a), min(sy * w, sy * a), z0),
            (max(sx * w, sx * a), max(sy * w, sy * a), z1),
        )
        for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    ]


# ================================================================================================
# 5. Physical laws
# ================================================================================================


def magnet_force(
    tip: Any,
    seat: Any,
    *,
    max_force_n: float,
    range_m: float,
    contact_direction: Any = (0.0, 0.0, 1.0),
    contact_gap_m: float = 0.0005,
) -> np.ndarray:
    """Attraction of the magnetic bit on a steel screw (N), linear fall-off to ``range_m``.

    ``seat`` is the screw point that touches the bit tip (recess floor). Within
    ``contact_gap_m`` the pull acts along ``contact_direction`` (from the screw toward the tool).
    """
    delta = np.asarray(tip, dtype=float) - np.asarray(seat, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance >= range_m:
        return np.zeros(3)
    magnitude = max_force_n * (1.0 - distance / range_m)
    if distance <= contact_gap_m:
        direction = np.asarray(contact_direction, dtype=float)
        return magnitude * direction / np.linalg.norm(direction)
    return magnitude * delta / distance


# ================================================================================================
# 6. Acceptance helpers
# ================================================================================================


def unwrap_angle(previous_unwrapped: float, previous_raw: float, raw: float) -> float:
    """Accumulate a wrapped joint angle (rad) into a continuous one."""
    delta = (raw - previous_raw + math.pi) % (2 * math.pi) - math.pi
    return previous_unwrapped + delta


def inside_box(point: Any, lo: Any, hi: Any, margin: float = 0.0) -> bool:
    """Point inside an axis-aligned box shrunk by ``margin``."""
    p = np.asarray(point, dtype=float)
    return bool(np.all(p >= np.asarray(lo) + margin) and np.all(p <= np.asarray(hi) - margin))


def thread_report(rows: list[dict[str, Any]], pitch_m: float) -> dict[str, float]:
    """Turns, rise and lead of one screw from per-step rows with ``turn_rad`` / ``slide_m``."""
    if len(rows) < 2:
        return dict(turns=0.0, rise_m=0.0, lead_m=0.0, lead_error_fraction=1.0)
    turns = (rows[-1]["turn_rad"] - rows[0]["turn_rad"]) / (2 * math.pi)
    rise = rows[-1]["slide_m"] - rows[0]["slide_m"]
    lead = rise / turns if abs(turns) > 1e-9 else 0.0
    return dict(
        turns=float(turns),
        rise_m=float(rise),
        lead_m=float(lead),
        lead_error_fraction=float(abs(lead - pitch_m) / pitch_m),
    )


def tilt_deg(quat: Any) -> float:
    """Angle between a body's +Z axis and world +Z."""
    return math.degrees(math.acos(np.clip(quat_to_matrix(quat)[2, 2], -1.0, 1.0)))
