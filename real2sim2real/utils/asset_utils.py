"""Small lazy USD helpers used by the cloth task."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def resolve_usd_path(path: str) -> str:
    """Resolve a local asset against cwd, then optional RSR_ASSETS_ROOT."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    root = os.environ.get("RSR_ASSETS_ROOT")
    if root:
        relative = Path(*candidate.parts[1:]) if candidate.parts[:1] == ("assets",) else candidate
        candidate = Path(root).expanduser() / relative
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(path)


def normalize_prim_path(prim_path: str) -> str:
    """把 prim 路径收成 ``/A/B``;空路径视为 ``/World``.

    去掉反斜杠与空段,保证 poses 字典的键可互相比对.

    Args:
        prim_path: 任意分隔的 prim 路径(可含反斜杠或多余 ``/``).

    Returns:
        规范化路径,如 ``/World`` 或 ``/A/B``.
    """
    parts = [p for p in prim_path.replace("\\", "/").split("/") if p]
    if not parts:
        return "/World"
    return "/" + "/".join(parts)


def prim_aabb(
    stage: Any,
    prim_path: str,
    *,
    gprims_only: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """计算 prim 世界 AABB;``gprims_only`` 时只合并 Mesh 等可渲染几何.

    instanceable 几何(Franka Performance 变体)须 ``TraverseInstanceProxies``,
    否则底座 AABB 为 None,贴合支撑面失败.

    Args:
        stage: 当前 USD stage;None 则返回 None.
        prim_path: 任意已在 stage 上的 prim.
        gprims_only: True 时跳过无几何的 Xform,避免空盒污染并集.

    Returns:
        世界 AABB ``(min_xyz, max_xyz)``;stage/prim 无效或无有效几何则为 ``None``.
    """
    if stage is None:
        return None
    prim = stage.GetPrimAtPath(normalize_prim_path(prim_path))
    if prim is None or not prim.IsValid():
        return None
    try:
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        total_min: np.ndarray | None = None
        total_max: np.ndarray | None = None
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if gprims_only and not child.IsA(UsdGeom.Gprim):
                continue
            if child.IsA(UsdGeom.Imageable):
                vis = UsdGeom.Imageable(child).ComputeVisibility()
                if vis == UsdGeom.Tokens.invisible:
                    continue
            box = cache.ComputeWorldBound(child)
            rng = box.ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            mn, mx = rng.GetMin(), rng.GetMax()
            cur_min = np.array([mn[0], mn[1], mn[2]], dtype=float)
            cur_max = np.array([mx[0], mx[1], mx[2]], dtype=float)
            if float(np.linalg.norm(cur_max - cur_min)) < 1e-8:
                continue
            if total_min is None or total_max is None:
                total_min, total_max = cur_min, cur_max
            else:
                total_min = np.minimum(total_min, cur_min)
                total_max = np.maximum(total_max, cur_max)
        if total_min is None or total_max is None:
            return None
        return total_min, total_max
    except Exception:  # noqa: BLE001
        return None


def read_prim_world_pose(stage: object, prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """从 USD stage 读取 prim 当前世界位姿(含物理驱动后的实时变换);无效则 ``None``.

    与摆放阶段缓存的静态位姿不同,本函数每次都重新计算,
    适合在物理仿真步进过程中轮询被抓物体等动态 prim。

    Args:
        stage: 当前 USD stage;None 时直接返回 None(stub).
        prim_path: 目标 prim 路径,会先 ``normalize_prim_path``.

    Returns:
        世界 ``(position, orientation_wxyz)``;stage/prim 无效或非 Xformable
        则为 ``None``.
    """
    if stage is None:
        return None  # 未绑定 stage(stub)时无法查
    # Kit 运行时对象:用字符串路径取 prim,不存在则后面判 Invalid
    prim = stage.GetPrimAtPath(normalize_prim_path(prim_path))  # type: ignore[attr-defined]
    if prim is None or not prim.IsValid():
        return None  # 路径尚未 spawn 或已被删
    try:
        from pxr import Usd, UsdGeom  # noqa: PLC0415  # 仅 Isaac 进程有 pxr

        xf = UsdGeom.Xformable(prim)  # 非 Xformable 会在下面 except
        mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())  # 含父链
        trans = mat.ExtractTranslation()  # 世界平移
        quat = mat.ExtractRotationQuat()  # pxr 四元数
        imag = quat.GetImaginary()  # (x, y, z)
        pos = np.array([trans[0], trans[1], trans[2]], dtype=float)  # pxr → numpy
        ori = np.array([quat.GetReal(), imag[0], imag[1], imag[2]], dtype=float)  # wxyz
        return pos, ori  # 与 poses 缓存同约定 (pos, wxyz)
    except Exception:  # noqa: BLE001  # 非 Xformable 等
        return None
