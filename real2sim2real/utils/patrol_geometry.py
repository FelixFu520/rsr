"""Geometry for patrol inspection: occupancy grid, A*, camera projection and photo checks.

All coordinates are world metres, Z-up. The occupancy grid is filled from PhysX scene queries
against the stage's real collision geometry (``scan_physx_occupancy``); everything else here is
pure numpy. Camera matrices follow the USD/Gf row-vector convention ``p_world = p_local @ M``
with the camera looking along local -Z and +Y up.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class OccupancyGrid:
    """Top-down grid; row 0 is the lowest Y, column 0 the lowest X.

    ``occupied``: the cell's column overlaps collision geometry inside the robot's height band.
    ``no_floor``: no collision surface below the cell (outside the building).
    """

    origin: np.ndarray  # world XY of the lower-left corner of cell (0, 0)
    resolution: float
    occupied: np.ndarray
    no_floor: np.ndarray
    _inflated: dict[float, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @property
    def blocked(self) -> np.ndarray:
        return self.occupied | self.no_floor

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape

    def cell(self, xy: np.ndarray) -> tuple[int, int]:
        """(row, col) of a world point; may lie outside the grid."""
        i, j = np.floor((np.asarray(xy, float)[:2] - self.origin) / self.resolution).astype(int)
        return int(j), int(i)

    def center(self, row: int, col: int) -> np.ndarray:
        return self.origin + (np.array([col, row]) + 0.5) * self.resolution

    def inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.shape[0] and 0 <= col < self.shape[1]

    def inflate(self, radius: float) -> np.ndarray:
        """Cells that contain any point closer than ``radius`` to a blocked cell (cached).

        The disk is grown by half a cell diagonal, so every point of a free cell, not only
        its centre, keeps ``radius`` clearance.
        """
        key = round(radius, 4)
        if key not in self._inflated:
            blocked = self.blocked
            out = blocked.copy()
            rows, cols = blocked.shape
            reach = radius / self.resolution + math.sqrt(0.5)
            cells = int(math.ceil(reach))
            for dy in range(-cells, cells + 1):
                for dx in range(-cells, cells + 1):
                    # Cell centres closer than ``radius`` to any point of a blocked cell.
                    if max(abs(dx) - 0.5, 0) ** 2 + max(abs(dy) - 0.5, 0) ** 2 > reach**2:
                        continue
                    src = blocked[max(0, -dy) : rows - max(0, dy), max(0, -dx) : cols - max(0, dx)]
                    out[max(0, dy) : rows - max(0, -dy), max(0, dx) : cols - max(0, -dx)] |= src
            self._inflated[key] = out
        return self._inflated[key]

    def free(self, xy: np.ndarray, radius: float) -> bool:
        row, col = self.cell(xy)
        return self.inside(row, col) and not self.inflate(radius)[row, col]

    def clearance(self, xy: np.ndarray, limit: float = 3.0) -> float:
        """Distance (m) from ``xy`` to the nearest blocked cell; 0 inside or off the grid."""
        row, col = self.cell(xy)
        if not self.inside(row, col) or self.blocked[row, col]:
            return 0.0
        n = int(math.ceil(limit / self.resolution))
        r0, c0 = max(0, row - n), max(0, col - n)
        rows, cols = np.nonzero(self.blocked[r0 : row + n + 1, c0 : col + n + 1])
        if rows.size == 0:
            return limit
        centres = self.origin + (np.c_[cols + c0, rows + r0] + 0.5) * self.resolution
        gap = np.maximum(np.abs(centres - np.asarray(xy, float)[:2]) - self.resolution / 2, 0.0)
        return float(min(limit, np.hypot(gap[:, 0], gap[:, 1]).min()))

    def segment_free(self, a: np.ndarray, b: np.ndarray, radius: float) -> bool:
        inflated = self.inflate(radius)
        for point in densify(np.vstack([a, b])[:, :2], self.resolution / 2):
            row, col = self.cell(point)
            if not self.inside(row, col) or inflated[row, col]:
                return False
        return True

    def path_report(self, points_xy: np.ndarray, radius: float) -> dict:
        """Smallest clearance along a polyline and every sample closer than ``radius``."""
        samples = densify(np.asarray(points_xy, float)[:, :2], self.resolution / 2)
        values = np.array([self.clearance(p) for p in samples])
        bad = samples[values < radius]
        index = int(values.argmin())
        return {
            "min_clearance_m": round(float(values[index]), 3),
            "at_xy": samples[index].round(3).tolist(),
            "violations": len(bad),
            "first_violation_xy": bad[0].round(3).tolist() if len(bad) else None,
        }

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        radius: float,
        margin: float = 0.3,
        soft_cost: float = 1.3,
    ) -> np.ndarray | None:
        """A* on the grid inflated by ``radius``; cells within ``radius + margin`` cost more.

        ``soft_cost`` keeps paths centred in open areas without forcing long detours around
        aisles that are narrower than ``radius + margin`` but still passable.

        Returns a line-of-sight smoothed polyline from ``start`` to ``goal`` or None.
        """
        hard = self.inflate(radius)
        soft = self.inflate(radius + margin) & ~hard
        s, g = self.cell(start), self.cell(goal)
        if not (self.inside(*s) and self.inside(*g)) or hard[s] or hard[g]:
            return None
        moves = [(dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
        moves = [move for move in moves if move[2] > 0]
        best = {s: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        queue = [(math.dist(s, g), s)]
        while queue:
            _, node = heapq.heappop(queue)
            if node == g:
                break
            for dr, dc, length in moves:
                nxt = (node[0] + dr, node[1] + dc)
                if not self.inside(*nxt) or hard[nxt]:
                    continue
                cost = best[node] + length * (soft_cost if soft[nxt] else 1.0)
                if cost < best.get(nxt, math.inf):
                    best[nxt], parent[nxt] = cost, node
                    heapq.heappush(queue, (cost + math.dist(nxt, g), nxt))
        if g not in best:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        raw = [np.asarray(start, float)[:2]]
        raw += [self.center(*c) for c in reversed(cells[1:-1])]
        raw.append(np.asarray(goal, float)[:2])
        return self._shortcut(np.asarray(raw), radius + margin, radius)

    def _shortcut(self, path: np.ndarray, preferred: float, required: float) -> np.ndarray:
        """Greedy line-of-sight smoothing.

        A shortcut must keep ``preferred`` clearance; where the A* path itself is already
        narrower than that (a tight doorway), ``required`` clearance is enough.
        """
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            narrow = not self.segment_free(path[i], path[i + 1], preferred)
            radius = required if narrow else preferred
            j = len(path) - 1
            while j > i + 1 and not self.segment_free(path[i], path[j], radius):
                j -= 1
            out.append(path[j])
            i = j
        return np.asarray(out)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, origin=self.origin, resolution=self.resolution,
            occupied=self.occupied, no_floor=self.no_floor,
        )  # fmt: skip


def scan_physx_occupancy(
    bounds_xy: tuple[tuple[float, float], tuple[float, float]],
    resolution: float,
    z_band: tuple[float, float],
    *,
    floor_search_m: float = 0.5,
) -> OccupancyGrid:
    """Fill an occupancy grid from PhysX scene queries; the physics view must be initialized.

    A cell is occupied when a box covering the cell and ``z_band`` overlaps any collider.
    A cell has no floor when a ray cast down from just below the band hits nothing.
    Only colliders already in the PhysX scene are seen, so scan before adding the robot.
    """
    import carb
    from omni.physx import get_physx_scene_query_interface

    query = get_physx_scene_query_interface()
    (x0, y0), (x1, y1) = bounds_xy
    origin = np.array([x0, y0], float)
    cols = math.ceil((x1 - x0) / resolution)
    rows = math.ceil((y1 - y0) / resolution)
    z_low, z_high = z_band
    half = carb.Float3(resolution / 2, resolution / 2, (z_high - z_low) / 2)
    z_mid = (z_low + z_high) / 2
    identity = carb.Float4(0.0, 0.0, 0.0, 1.0)
    down = carb.Float3(0.0, 0.0, -1.0)
    occupied = np.zeros((rows, cols), bool)
    no_floor = np.zeros((rows, cols), bool)
    for row in range(rows):
        y = y0 + (row + 0.5) * resolution
        for col in range(cols):
            x = x0 + (col + 0.5) * resolution
            hits = query.overlap_box(half, carb.Float3(x, y, z_mid), identity, lambda _: True, True)
            occupied[row, col] = hits > 0
            hit = query.raycast_closest(carb.Float3(x, y, z_low - 0.01), down, floor_search_m)
            no_floor[row, col] = not hit["hit"]
    return OccupancyGrid(origin, resolution, occupied, no_floor)


def densify(points: np.ndarray, spacing: float) -> np.ndarray:
    """Sample a polyline every ``spacing`` metres, keeping all vertices."""
    points = np.asarray(points, dtype=float)
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:], strict=True):
        n = max(1, math.ceil(float(np.linalg.norm(b - a)) / spacing))
        out.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    return np.asarray(out)


def polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


@dataclass(frozen=True)
class Projection:
    """A world point in camera space and image pixels."""

    in_view: bool
    depth: float
    u: float
    v: float
    off_axis_rad: float


def project_point(
    world_point: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    focal_length: float,
    horizontal_aperture: float,
    resolution: tuple[int, int],
    near: float,
    far: float,
    margin: float = 1.0,
) -> Projection:
    """Project a world point with a pinhole USD camera; ``margin`` < 1 shrinks the FOV."""
    width, height = resolution
    local = np.append(np.asarray(world_point, float), 1.0) @ np.linalg.inv(camera_matrix)
    x, y, z = local[:3]
    depth = -z
    tan_h = horizontal_aperture / 2 / focal_length
    tan_v = tan_h * height / width
    if depth <= 1e-9:
        return Projection(False, float(depth), math.nan, math.nan, math.pi)
    nx, ny = x / depth / tan_h, y / depth / tan_v
    u = (nx + 1) / 2 * width
    v = (1 - ny) / 2 * height
    in_view = near <= depth <= far and abs(nx) <= margin and abs(ny) <= margin
    off_axis = math.atan2(math.hypot(x, y), depth)
    return Projection(bool(in_view), float(depth), float(u), float(v), float(off_axis))


def fire_pixel_mask(rgb: np.ndarray) -> np.ndarray:
    """Bright warm pixels typical of rendered flames (orange/yellow, red dominant)."""
    rgb = np.asarray(rgb, dtype=np.int32)[..., :3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r >= 200) & (r > g + 25) & (g > b + 15) & (b <= 170)


def fire_pixel_score(rgb: np.ndarray, u: float, v: float, window: int) -> dict:
    """Flame-coloured pixels in a ``window`` square centred on the projected fire."""
    mask = fire_pixel_mask(rgb)
    height, width = mask.shape
    half = window // 2
    u0, v0 = round(u), round(v)
    x0, x1 = max(0, u0 - half), min(width, u0 + half)
    y0, y1 = max(0, v0 - half), min(height, v0 + half)
    area = mask[y0:y1, x0:x1]
    return {
        "window_px": [x0, y0, x1, y1],
        "window_fraction": float(area.mean()) if area.size else 0.0,
        "window_pixels": int(area.sum()),
        "image_fraction": float(mask.mean()),
    }


def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def pose_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Row-vector 4x4 matrix (USD convention) from position and quaternion."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rot.T  # row-vector convention stores the transpose
    matrix[3, :3] = np.asarray(position, float)
    return matrix


def save_patrol_map(
    path: Path,
    grid: OccupancyGrid,
    *,
    fire_xy: np.ndarray,
    paths: dict[str, np.ndarray],
    points: dict[str, np.ndarray] | None = None,
    scale: int = 2,
) -> Path:
    """Top-down PNG of the occupancy grid with named polylines and points, plus JSON."""
    from PIL import Image, ImageDraw

    rgb = np.full((*grid.shape, 3), 250, np.uint8)
    rgb[grid.no_floor] = (255, 215, 215)
    rgb[grid.occupied] = (60, 60, 60)
    image = Image.fromarray(rgb[::-1]).resize(
        (grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST
    )
    draw = ImageDraw.Draw(image)
    top = grid.origin[1] + grid.shape[0] * grid.resolution

    def px(xy: np.ndarray) -> tuple[float, float]:
        return (
            (xy[0] - grid.origin[0]) / grid.resolution * scale,
            (top - xy[1]) / grid.resolution * scale,
        )

    x_lo, y_lo = grid.origin
    x_hi = x_lo + grid.shape[1] * grid.resolution
    for gx in range(math.ceil(x_lo / 2) * 2, int(x_hi) + 1, 2):
        draw.line([px((gx, y_lo)), px((gx, top))], fill=(205, 220, 255))
        draw.text(px((gx + 0.1, top - 0.2)), str(gx), fill=(0, 0, 180))
    for gy in range(math.ceil(y_lo / 2) * 2, int(top) + 1, 2):
        draw.line([px((x_lo, gy)), px((x_hi, gy))], fill=(205, 220, 255))
        draw.text(px((x_lo + 0.1, gy + 0.45)), str(gy), fill=(0, 0, 180))
    colours = [(30, 110, 230), (20, 160, 60), (200, 120, 0), (160, 40, 200)]
    for index, (name, poly) in enumerate(paths.items()):
        poly = np.asarray(poly, float)
        if len(poly) >= 2:
            colour = colours[index % len(colours)]
            draw.line([px(p) for p in poly[:, :2]], fill=colour, width=3)
            for p in poly[:, :2]:
                x, y = px(p)
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=colour)
            draw.text(px(poly[0]), name, fill=colour)
    for name, point in (points or {}).items():
        x, y = px(np.asarray(point, float))
        draw.rectangle([x - 5, y - 5, x + 5, y + 5], outline=(0, 0, 0), width=2)
        draw.text((x + 7, y - 7), name, fill=(0, 0, 0))
    fx, fy = px(np.asarray(fire_xy, float))
    draw.ellipse([fx - 8, fy - 8, fx + 8, fy + 8], fill=(230, 50, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "origin_xy": grid.origin.tolist(),
                "resolution_m": grid.resolution,
                "shape_rows_cols": list(grid.shape),
                "fire_xy": np.asarray(fire_xy, float)[:2].tolist(),
                "paths": {k: np.asarray(v).round(3).tolist() for k, v in paths.items()},
                "points": {k: np.asarray(v).round(3).tolist() for k, v in (points or {}).items()},
            },
            indent=1,
        )
        + "\n"
    )
    return path
