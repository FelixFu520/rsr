"""物理工具：碰撞体、刚体及摩擦等物理属性；当前提供静态碰撞补全与子树碰撞/刚体禁用。

关节与 articulation 操作放在 robot_utils，表面 UV 与法线计算放在 material_utils。

pxr 仅在函数内 import,避免 stub 通路被重依赖拖垮.
"""

from __future__ import annotations

import re
from typing import Any

from real2sim2real.utils.isaacsim_utils.openusd_utils import normalize_prim_path

_SKIP_COLLIDER_NAME_RE = re.compile(
    r"(sky|dome|atmosphere|skysphere|celestial)",
    re.IGNORECASE,
)
_MAX_STATIC_COLLIDER_SPAN_M = 250.0


def ensure_static_collider(
    stage: Any,
    root_path: str,
    *,
    max_span_m: float = _MAX_STATIC_COLLIDER_SPAN_M,
) -> int:
    """给纯视觉 Mesh 补 ``PhysicsCollisionAPI``(精确三角网格).

    桌子等家具常常只有可渲染 Mesh、没有任何物理碰撞体;
    不补碰撞的话物体会直接穿透桌面自由下落.工厂背景同样常缺碰撞,
    occupancy 的 omap 也只能扫到 PhysX 碰撞体.

    Args:
        stage: 当前 USD stage;None 则返回 0.
        root_path: 资产根 prim.
        max_span_m: XY 任一边超过则跳过,避免天空球/无限地面撑爆 PhysX.

    Returns:
        新补上 CollisionAPI 的 Mesh 数量.
    """
    if stage is None:
        return 0
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(normalize_prim_path(root_path))
    if root is None or not root.IsValid():
        return 0
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )
    n = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        name = prim.GetName()
        path = str(prim.GetPath())
        if _SKIP_COLLIDER_NAME_RE.search(name) or _SKIP_COLLIDER_NAME_RE.search(path):
            continue
        try:
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            mn, mx = rng.GetMin(), rng.GetMax()
            if (float(mx[0] - mn[0]) > max_span_m) or (float(mx[1] - mn[1]) > max_span_m):
                continue
        except Exception:  # noqa: BLE001
            continue
        try:
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr("none")
        except Exception:  # noqa: BLE001
            continue
        n += 1
    return n


def disable_subtree_collision(stage: Any, prim_path: str) -> None:
    """关掉子树碰撞/刚体,避免相机挂在 link 下时把 PhysX tensor view 弄失效.

    Args:
        stage: 当前 USD stage;None 则直接返回.
        prim_path: 子树根 prim.

    Returns:
        None.
    """
    if stage is None:
        return
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(normalize_prim_path(prim_path))
    if root is None or not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
