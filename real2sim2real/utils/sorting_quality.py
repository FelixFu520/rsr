"""Independent V7 shape and bag/garment-gap checks on saved physical vertices."""

from __future__ import annotations

import numpy as np

from real2sim2real.utils.package_geometry import _surface_distance


def unit_box_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Return a closed box with unit half-extents for auditing analytic box colliders."""
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    triangles = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        int,
    )
    return vertices, triangles


def aligned_shape_rms(rest: np.ndarray, positions: np.ndarray, masses: np.ndarray) -> float:
    """Mass-weighted shape change after removing rigid translation and rotation."""
    weights = masses / masses.sum()
    reference = rest - np.average(rest, axis=0, weights=weights)
    current = positions - np.average(positions, axis=0, weights=weights)
    u, _, vt = np.linalg.svd((reference * weights[:, None]).T @ current)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return float(np.sqrt(np.sum(weights * np.sum((reference @ rotation - current) ** 2, axis=1))))


def audit_held_shape(
    positions: np.ndarray,
    rest: np.ndarray,
    triangles: np.ndarray,
    masses: np.ndarray,
    garment_count: int,
    transfer_groups: dict[str, list[int] | np.ndarray],
    *,
    maximum_shape_rms_m: float = 0.037413,
    maximum_bag_gap_m: float = 0.01,
    extra_samples: dict[str, list[int]] | None = None,
    quality_reference: dict | None = None,
) -> dict:
    """Check three saved physical frames per transfer using V7's unchanged limits.

    All bag vertices and face centres contribute to each gap quantile. Sampling
    establishes evidence at those frames, not continuous-time nonpenetration.
    Refined assets additionally retain their original material-point population
    and garment mass weights; both populations must pass the same limits.
    """
    garment = triangles[np.all(triangles < garment_count, axis=1)]
    bag = triangles[np.all(triangles >= garment_count, axis=1)] - garment_count
    rows = []
    for side, indices in transfer_groups.items():
        if not len(indices):
            continue
        selected = np.asarray(indices)[np.linspace(0, len(indices) - 1, 3).astype(int)]
        if extra_samples is not None:
            selected = np.r_[selected, np.asarray(extra_samples.get(side, []), dtype=int)]
        for index in np.unique(selected):
            if not 0 <= index < len(positions):
                raise ValueError("Shape checkpoint is outside the recorded physics samples")
            vertices = positions[index]
            probes = np.concatenate(
                [vertices[garment_count:], vertices[garment_count:][bag].mean(1)]
            )
            distances = _surface_distance(probes, vertices[garment])
            rms = aligned_shape_rms(
                rest[:garment_count], vertices[:garment_count], masses[:garment_count]
            )
            row = dict(
                side=side,
                sample_index=int(index),
                bag_gap_q95_m=float(np.quantile(distances, 0.95)),
                aligned_garment_shape_rms_m=rms,
            )
            if quality_reference is not None:
                reference_masses = np.asarray(quality_reference["garment_masses"], dtype=float)
                reference_count = len(reference_masses)
                bag_count = int(quality_reference["bag_vertex_count"])
                reference_bag = np.asarray(quality_reference["bag_triangles"], dtype=int)
                if (
                    not 0 < reference_count <= garment_count
                    or not np.isfinite(reference_masses).all()
                    or np.any(reference_masses <= 0)
                    or not 0 < bag_count <= len(vertices) - garment_count
                    or np.any(reference_bag < 0)
                    or np.any(reference_bag >= bag_count)
                ):
                    raise ValueError("Invalid original-sampling reference for refined mesh")
                if "bag_face_probe_indices" in quality_reference:
                    support = np.asarray(quality_reference["bag_face_probe_indices"], dtype=int)
                    weights = np.asarray(quality_reference["bag_face_probe_weights"], dtype=float)
                    if (
                        support.shape != reference_bag.shape
                        or weights.shape != support.shape
                        or np.any(support < 0)
                        or np.any(support >= len(vertices) - garment_count)
                        or not np.isfinite(weights).all()
                        or np.any(weights < 0)
                        or not np.allclose(weights.sum(1), 1)
                    ):
                        raise ValueError("Invalid refined material-probe coordinates")
                    centres = (vertices[garment_count:][support] * weights[:, :, None]).sum(1)
                else:
                    centres = vertices[garment_count:][reference_bag].mean(1)
                reference_probes = np.concatenate(
                    [
                        vertices[garment_count : garment_count + bag_count],
                        centres,
                    ]
                )
                reference_distance = _surface_distance(reference_probes, vertices[garment])
                row.update(
                    reference_bag_gap_q95_m=float(np.quantile(reference_distance, 0.95)),
                    reference_aligned_garment_shape_rms_m=aligned_shape_rms(
                        rest[:reference_count], vertices[:reference_count], reference_masses
                    ),
                )
            rows.append(row)
    complete = all(len(indices) > 0 for indices in transfer_groups.values()) and bool(
        transfer_groups
    )
    checks = dict(
        held_shape_rms_bounded=complete
        and all(row["aligned_garment_shape_rms_m"] < maximum_shape_rms_m for row in rows),
        held_bag_gap_bounded=complete
        and all(row["bag_gap_q95_m"] < maximum_bag_gap_m for row in rows),
    )
    if quality_reference is not None:
        checks["held_original_sampling_bounded"] = complete and all(
            row["reference_bag_gap_q95_m"] < maximum_bag_gap_m
            and row["reference_aligned_garment_shape_rms_m"] < maximum_shape_rms_m
            for row in rows
        )
    return dict(
        checks=checks,
        maximum_shape_rms_m=maximum_shape_rms_m,
        maximum_bag_gap_m=maximum_bag_gap_m,
        samples=rows,
        scope=(
            "Three physics frames per transfer plus recorded online shape checkpoints; "
            "all bag vertices and face centres."
            if extra_samples is not None
            else "Three physics frames per transfer; all bag vertices and face centres."
        ),
    )


def convex_planes(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Build verified outward halfspaces for an actual convex collision mesh."""
    faces = vertices[triangles]
    normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    lengths = np.linalg.norm(normal, axis=1)
    valid = lengths > 1e-12
    normal, faces = normal[valid] / lengths[valid, None], faces[valid]
    flip = np.sum(normal * (vertices.mean(0) - faces[:, 0]), axis=1) > 0
    normal[flip] *= -1
    planes = np.c_[normal, -np.sum(normal * faces[:, 0], axis=1)]
    if not len(planes) or (vertices @ planes[:, :3].T + planes[:, 3]).max() > 1e-5:
        raise ValueError("Finger collision geometry is not a valid convex mesh")
    return np.unique(np.round(planes, 8), axis=0)


def audit_jaw_penetration(
    positions: np.ndarray,
    triangles: np.ndarray,
    body_poses: np.ndarray,
    jaws: list[dict],
    *,
    tolerance_m: float = 0.0015,
) -> dict:
    """Check every saved vertex and face centre against all native convex finger colliders."""
    colliders = [(jaw, convex_planes(jaw["vertices"], jaw["triangles"])) for jaw in jaws]
    maximum = 0.0
    witness = None
    for frame, (vertices, poses) in enumerate(zip(positions, body_poses, strict=True)):
        probes = np.concatenate([vertices, vertices[triangles].mean(1)])
        for collider, (jaw, planes) in enumerate(colliders):
            pose = poses[jaw["body_id"]]
            delta = probes - pose[:3]
            cross = 2 * np.cross(pose[3:6], delta)
            local = delta - pose[6] * cross + np.cross(pose[3:6], cross)
            mesh = jaw["vertices"]
            nearby = np.all((local >= mesh.min(0)) & (local <= mesh.max(0)), axis=1)
            if nearby.any():
                signed = (local[nearby] @ planes[:, :3].T + planes[:, 3]).max(1)
                index = int(np.argmin(signed))
                depth = float(-signed[index])
                if depth > maximum:
                    maximum = depth
                    probe = int(np.flatnonzero(nearby)[index])
                    witness = dict(
                        frame_index=frame,
                        collider_index=collider,
                        body_id=int(jaw["body_id"]),
                        probe_index=probe,
                        probe_kind="vertex" if probe < len(vertices) else "face_centre",
                        local_point_m=local[probe].tolist(),
                    )
    return dict(
        passed=bool(len(positions) and jaws and maximum <= tolerance_m),
        maximum_penetration_m=maximum,
        tolerance_m=tolerance_m,
        checked_physics_frames=len(positions),
        finger_colliders=len(jaws),
        worst_penetration=witness,
        method="Exact convex halfspaces; all saved vertices and face centres per physics frame.",
    )
