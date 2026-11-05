"""Tetrahedral mesh validation and render-mesh embedding, independent of Isaac imports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def perimeter_stiffness_scales(
    points: np.ndarray,
    surface: np.ndarray,
    minimum_scale: float,
    softening_radii: tuple[float, float] = (0.2, 0.6),
) -> np.ndarray:
    """Blend a parcel's stiffness across its local XY plane, preserving its central core."""
    start, end = softening_radii
    if not (0.0 < minimum_scale <= 1.0 and 0.0 <= start < end <= 1.0):
        raise ValueError("Invalid parcel perimeter stiffness profile")
    points, surface = np.asarray(points), np.asarray(surface)
    low, high = surface[:, :2].min(axis=0), surface[:, :2].max(axis=0)
    if not np.isfinite(points).all() or not np.isfinite(surface).all() or np.any(high <= low):
        raise ValueError("Invalid parcel material coordinates")
    uv = (points[:, :2] - (low + high) / 2.0) / ((high - low) / 2.0)
    radius = np.sum(uv**4, axis=1) ** 0.25
    blend = np.clip((radius - start) / (end - start), 0.0, 1.0)
    return 1.0 - (1.0 - minimum_scale) * blend**2 * (3.0 - 2.0 * blend)


def surface_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute area-weighted normals from the current deformed triangle surface."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    points = vertices[triangles]
    face_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)


def signed_tet_volumes(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Return signed tetrahedral volumes in cubic metres."""
    points = np.asarray(vertices, dtype=np.float64)[np.asarray(tetrahedra, dtype=np.int64)]
    return np.linalg.det(points[:, 1:] - points[:, :1]) / 6.0


def orient_tetrahedra(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Orient cells positively; reject degenerate cells and unreferenced nodes."""
    vertices = np.asarray(vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64).reshape(-1, 4).copy()
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.all(np.isfinite(vertices))
        or not tetrahedra.size
        or tetrahedra.min() < 0
        or tetrahedra.max() >= len(vertices)
    ):
        raise ValueError("Invalid tetrahedral mesh")
    volumes = signed_tet_volumes(vertices, tetrahedra)
    if np.any(np.abs(volumes) < 1e-12):
        raise ValueError("Degenerate tetrahedral cell")
    reverse = volumes < 0
    tetrahedra[reverse, :2] = tetrahedra[reverse, 1::-1]
    if len(np.unique(tetrahedra)) != len(vertices):
        raise ValueError("Unreferenced simulation nodes would have zero mass")
    return tetrahedra.astype(np.int32)


def clean_tetrahedral_mesh(
    vertices: np.ndarray, tetrahedra: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weld duplicates on a 0.1 micrometre grid and require a manifold boundary."""
    vertices = np.asarray(vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64).reshape(-1, 4)
    _, first, inverse = np.unique(
        np.round(vertices, 7), axis=0, return_index=True, return_inverse=True
    )
    vertices, tetrahedra = vertices[first], inverse[tetrahedra]
    _, first = np.unique(np.sort(tetrahedra, axis=1), axis=0, return_index=True)
    tetrahedra = tetrahedra[np.sort(first)]
    used, inverse = np.unique(tetrahedra, return_inverse=True)
    vertices, tetrahedra = vertices[used], inverse.reshape(-1, 4)
    tetrahedra = orient_tetrahedra(vertices, tetrahedra)
    faces = np.sort(
        tetrahedra[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), axis=1
    )
    faces, counts = np.unique(faces, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Non-manifold volume: more than two cells share a face")
    boundary = faces[counts == 1]
    edges = np.sort(
        np.concatenate([boundary[:, [0, 1]], boundary[:, [1, 2]], boundary[:, [2, 0]]]), axis=1
    )
    if not len(edges) or np.any(np.unique(edges, axis=0, return_counts=True)[1] != 2):
        raise ValueError("Non-manifold tetrahedral boundary")
    return vertices, tetrahedra


@dataclass(frozen=True)
class TetEmbedding:
    """Bind each render vertex to four simulation nodes with barycentric weights."""

    indices: np.ndarray
    weights: np.ndarray

    def apply(self, nodes: np.ndarray) -> np.ndarray:
        """Interpolate positions or velocities without changing render topology."""
        return np.einsum("ni,nij->nj", self.weights, np.asarray(nodes)[self.indices])

    @classmethod
    def build(
        cls, render_vertices: np.ndarray, vertices: np.ndarray, tetrahedra: np.ndarray
    ) -> TetEmbedding:
        """Find containing cells; refuse to silently bind vertices outside the volume."""
        points = np.asarray(vertices, dtype=np.float64)[tetrahedra]
        low, high = points.min(axis=1) - 1e-7, points.max(axis=1) + 1e-7
        inverse = np.linalg.inv(points[:, 1:] - points[:, :1])
        cells, weights = [], []
        for point in np.asarray(render_vertices, dtype=np.float64):
            candidates = np.flatnonzero(np.all((point >= low) & (point <= high), axis=1))
            if not len(candidates):
                raise ValueError("Render vertex is outside the simulation volume")
            bary = np.einsum("ni,nij->nj", point - points[candidates, 0], inverse[candidates])
            bary = np.column_stack((1.0 - bary.sum(axis=1), bary))
            best = int(np.argmax(bary.min(axis=1)))
            if bary[best].min() < -1e-4:
                raise ValueError("No containing tetrahedron for render vertex")
            cells.append(tetrahedra[candidates[best]])
            weights.append(bary[best])
        embedding = cls(np.asarray(cells, dtype=np.int32), np.asarray(weights))
        if not np.allclose(embedding.apply(vertices), render_vertices, atol=1e-6):
            raise ValueError("Rest-pose render embedding mismatch")
        return embedding
