"""Independent NumPy ray-parity and point-to-triangle checks for the closed parcel bag."""

from __future__ import annotations

import numpy as np


def _inside(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    # An oblique ray avoids the axis-aligned grid edges in the authored parcel.
    direction = np.array([1.0, 0.137, 0.213])
    direction /= np.linalg.norm(direction)
    side = np.cross(direction, [0, 0, 1])
    side /= np.linalg.norm(side)
    basis = np.stack([direction, side, np.cross(direction, side)], axis=1)
    tri, probes = triangles @ basis, points @ basis
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v, w = b - a, c - a
    determinant = v[:, 1] * w[:, 2] - v[:, 2] * w[:, 1]
    valid = np.abs(determinant) > 1e-15
    determinant = np.where(valid, determinant, 1.0)
    result = []
    for chunk in np.array_split(probes, max(1, (len(probes) + 127) // 128)):
        delta = chunk[:, None] - a
        u = (delta[:, :, 1] * w[:, 2] - delta[:, :, 2] * w[:, 1]) / determinant
        t = (v[:, 1] * delta[:, :, 2] - v[:, 2] * delta[:, :, 1]) / determinant
        intersection_x = a[:, 0] + u * v[:, 0] + t * w[:, 0]
        hits = valid & (u >= 0) & (t >= 0) & (u + t <= 1) & (intersection_x > chunk[:, None, 0])
        result.append(np.count_nonzero(hits, axis=1) % 2 == 1)
    return np.concatenate(result)


def _surface_distance(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    aa, bb, cc = (ab * ab).sum(1), (ab * ac).sum(1), (ac * ac).sum(1)
    det = np.maximum(aa * cc - bb * bb, 1e-30)
    result = []
    for chunk in np.array_split(points, max(1, (len(points) + 63) // 64)):
        ap = chunk[:, None] - a
        d, e = (ap * ab).sum(2), (ap * ac).sum(2)
        u, v = (cc * d - bb * e) / det, (aa * e - bb * d) / det
        projection = ap - u[:, :, None] * ab - v[:, :, None] * ac
        distance = np.where(
            (u >= 0) & (v >= 0) & (u + v <= 1), (projection * projection).sum(2), np.inf
        )
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end - start
            offset = chunk[:, None] - start
            t = np.clip((offset * edge).sum(2) / np.maximum((edge * edge).sum(1), 1e-30), 0, 1)
            residual = offset - t[:, :, None] * edge
            distance = np.minimum(distance, (residual * residual).sum(2))
        result.append(np.sqrt(distance.min(1)))
    return np.concatenate(result)


def check_package_containment(
    positions: np.ndarray,
    triangles: np.ndarray,
    garment_count: int,
    *,
    times: np.ndarray,
    tolerance_m: float = 0.001,
    checkpoint_times_s: list[float] | None = None,
) -> dict:
    """Check all garment vertices and face centres at ten task checkpoints.

    Ray parity uses the closed, consistently triangulated bag. Outside probes
    within tolerance count as contact. This is sampled geometry evidence, not
    a continuous-time proof against self-intersection or tunnelling.
    """
    if not len(times):
        return dict(
            passed=False,
            tolerance_m=tolerance_m,
            checkpoints=[],
            error="No physical particle samples",
        )
    bag = triangles[np.all(triangles >= garment_count, axis=1)] - garment_count
    garment = triangles[np.all(triangles < garment_count, axis=1)]
    samples = checkpoint_times_s or [0, 2, 3.5, 5, 7, 8.5, 9.5, 10.5, 11.5, 13]
    indices = np.unique([np.abs(times - time).argmin() for time in samples])
    rows = []
    for index in indices:
        points = positions[index]
        probes = np.concatenate([points[:garment_count], points[garment].mean(axis=1)])
        surface = points[garment_count:][bag]
        inside = _inside(probes, surface)
        distances = _surface_distance(probes[~inside], surface) if (~inside).any() else np.zeros(1)
        rows.append(
            dict(
                time_s=float(times[index]),
                probes=len(probes),
                outside_tolerance=int(np.count_nonzero(distances > tolerance_m)),
                maximum_outside_distance_m=float(distances.max()),
            )
        )
    return dict(
        passed=all(row["outside_tolerance"] == 0 for row in rows),
        tolerance_m=tolerance_m,
        checkpoints=rows,
        method="oblique_ray_parity_and_exact_point_triangle_distance",
    )
