"""用材质 UV 跟踪柔性网格表面区域，不依赖仿真运行时。"""

from __future__ import annotations

import numpy as np


def uv_patch_triangles(
    triangles: np.ndarray, face_uvs: np.ndarray, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    """选择 UV 中心落在 ``(u_min, v_min, u_max, v_max)`` 内的三角面。"""
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    uvs = np.asarray(face_uvs, dtype=float).reshape(-1, 3, 2)
    box = np.asarray(bounds, dtype=float).reshape(4)
    if len(faces) != len(uvs) or not np.all(np.isfinite(uvs)):
        raise ValueError("三角面与 UV 数量须一致，UV 须为有限数值")
    if not np.all(np.isfinite(box)) or np.any(box[:2] >= box[2:]):
        raise ValueError("UV 区域须为有限、严格递增的矩形")
    centers = uvs.mean(axis=1)
    selected = np.all((centers >= box[:2]) & (centers <= box[2:]), axis=1)
    patch = faces[selected].copy()
    if not len(patch):
        raise ValueError("标签 UV 区域中没有三角面，请检查资产与 label_uv_bounds")
    return patch


def surface_patch_normal(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """计算当前表面区域按面积加权的单位法线；折叠至退化时拒绝判定朝向。"""
    points = np.asarray(vertices, dtype=float).reshape(-1, 3)
    indices = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not len(indices) or np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError("表面区域须包含有效三角面索引")
    patch = points[indices]
    normals = np.cross(patch[:, 1] - patch[:, 0], patch[:, 2] - patch[:, 0])
    normal = normals.sum(axis=0)
    area = float(np.linalg.norm(normals, axis=1).sum())
    magnitude = float(np.linalg.norm(normal))
    if not np.all(np.isfinite(normal)) or area <= 1.0e-12 or magnitude <= area * 0.1:
        raise ValueError("表面区域退化或严重折叠，无法可靠判定朝向")
    return normal / magnitude
