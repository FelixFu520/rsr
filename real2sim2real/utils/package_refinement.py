"""Conforming local subdivision with conserved surface density and original anchors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RefinedSurface:
    """Refined triangles and their barycentric provenance on the input surface."""

    vertices: np.ndarray
    triangles: np.ndarray
    masses: np.ndarray
    parents: np.ndarray
    barycentric: np.ndarray


def map_material_probes(
    refined: RefinedSurface,
    source_triangles: np.ndarray,
    support: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep material sample locations on the refined finite-element surface."""
    lookup = {tuple(face): i for i, face in enumerate(source_triangles)}
    children: dict[int, list[int]] = {}
    for child, parent in enumerate(refined.parents):
        children.setdefault(int(parent), []).append(child)
    indices, mapped = [], []
    for face, barycentric in zip(support, weights, strict=True):
        parent = lookup.get(tuple(face))
        if parent is None:
            raise ValueError("Material probe support is not a source triangle")
        for child in children[parent]:
            local = np.linalg.solve(refined.barycentric[child].T, barycentric)
            if np.all(local >= -1e-10) and np.all(local <= 1 + 1e-10):
                local = np.maximum(local, 0)
                indices.append(refined.triangles[child])
                mapped.append(local / local.sum())
                break
        else:
            raise ValueError("Material probe is outside its refined parent triangle")
    return np.asarray(indices, dtype=int), np.asarray(mapped, dtype=float)


def lumped_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return one third of each incident triangle's area per vertex."""
    face = vertices[triangles]
    area = np.linalg.norm(np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0]), axis=1) / 2
    result = np.zeros(len(vertices))
    np.add.at(result, triangles.ravel(), np.repeat(area / 3, 3))
    return result


def refine_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    masses: np.ndarray,
    *,
    center_xy: tuple[float, float] = (0.0, 0.0),
    half_size_xy: tuple[float, float] = (0.085, 0.055),
) -> RefinedSurface:
    """Split local triangle edges and conform all neighbours without opening cracks.

    Original vertices keep their indices, preserving discrete loft anchors.
    This operation requires uniform areal density, which the source V7 meshes
    satisfy. New masses use that same density and the unchanged total mass.
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    masses = np.asarray(masses, dtype=float)
    area = lumped_areas(vertices, triangles)
    if (
        not np.isfinite(vertices).all()
        or not np.isfinite(masses).all()
        or np.any(area <= 0)
        or np.any(masses <= 0)
    ):
        raise ValueError("Refinement requires finite dynamic vertices and positive surface areas")
    density = masses.sum() / area.sum()
    if not np.allclose(masses, density * area, rtol=1e-4, atol=1e-12):
        raise ValueError("Local refinement requires the source's uniform areal density")
    half = np.asarray(half_size_xy, dtype=float)
    center = np.asarray(center_xy, dtype=float)
    if not np.isfinite([*half, *center]).all() or np.any(half <= 0):
        raise ValueError("Refinement region must have finite coordinates and positive half sizes")
    face_xy = vertices[triangles, :2]
    selected = np.all(face_xy.max(1) >= center - half, axis=1) & np.all(
        face_xy.min(1) <= center + half, axis=1
    )
    chosen = triangles[selected]
    edges = np.unique(
        np.sort(np.concatenate([chosen[:, :2], chosen[:, 1:], chosen[:, [2, 0]]]), axis=1),
        axis=0,
    )
    edge_ids = {tuple(edge): len(vertices) + i for i, edge in enumerate(edges)}
    points = np.concatenate([vertices, vertices[edges].mean(1)])
    faces, parents, barycentric = [], [], []
    for parent, triangle in enumerate(triangles):
        a, b, c = map(int, triangle)
        corners = [a, b, c]
        mids = [edge_ids.get(tuple(sorted(pair))) for pair in [(a, b), (b, c), (c, a)]]
        weights = {a: np.array([1.0, 0, 0]), b: np.array([0.0, 1, 0]), c: np.array([0.0, 0, 1])}
        for i, midpoint in enumerate(mids):
            if midpoint is not None:
                weights[midpoint] = (weights[corners[i]] + weights[corners[(i + 1) % 3]]) / 2
        count = sum(midpoint is not None for midpoint in mids)
        if count == 0:
            split = [(a, b, c)]
        elif count == 3:
            ab, bc, ca = mids
            split = [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        else:
            # Rotate cyclically to a canonical one-edge or two-edge pattern.
            while not (mids[0] is not None and ((mids[1] is not None) == (count == 2))):
                corners = corners[1:] + corners[:1]
                mids = mids[1:] + mids[:1]
            a, b, c = corners
            ab, bc, _ = mids
            split = (
                [(a, ab, c), (ab, b, c)] if count == 1 else [(ab, b, bc), (a, ab, bc), (a, bc, c)]
            )
        for face in split:
            faces.append(face)
            parents.append(parent)
            barycentric.append([weights[index] for index in face])
    faces = np.asarray(faces, dtype=int)
    # USD points are float32. Compute masses from the actual stored coordinates.
    points = points.astype(np.float32).astype(float)
    new_area = lumped_areas(points, faces)
    new_mass = new_area * (masses.sum() / new_area.sum())
    return RefinedSurface(points, faces, new_mass, np.asarray(parents), np.asarray(barycentric))
