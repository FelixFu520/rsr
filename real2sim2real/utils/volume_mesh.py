"""Tetrahedral mesh validation and render-mesh embedding, independent of Isaac imports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
