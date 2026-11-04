"""机器人工具：关节、articulation 及其关联刚体的控制与配置。

当前提供子树木偶模式切换；通用碰撞、刚体及摩擦属性工具放在 physics_utils。
pxr / PhysxSchema 仅在函数内 import，避免 stub 通路被重依赖拖垮。
"""

from __future__ import annotations

from typing import Any

from real2sim2real.utils.isaacsim_utils.openusd_utils import normalize_prim_path


def set_subtree_puppet(stage: Any, root_path: str, *, enabled: bool) -> tuple[int, int, int]:
    """拆 articulation / 关节并锁 kinematic,把子树当木偶挪根位姿.

    只标 kinematic 时 PhysX 仍会解析 ArticulationRootAPI 和关节;
    导航不需要关节动力学,关掉后只保留碰撞与 xform.

    Args:
        stage: 当前 USD stage;None 则返回 ``(0, 0, 0)``.
        root_path: 子树根 prim.
        enabled: True 拆 articulation / 关节并锁 kinematic;False 恢复.

    Returns:
        ``(刚体数, 关节数, 被拆/恢复的 articulation 根数)``.
    """
    if stage is None:
        return 0, 0, 0
    from pxr import Usd, UsdPhysics

    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None  # type: ignore[misc, assignment]

    root = stage.GetPrimAtPath(normalize_prim_path(root_path))
    if root is None or not root.IsValid():
        return 0, 0, 0
    n_bodies = 0
    n_joints = 0
    n_art = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if enabled:
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                n_art += 1
            else:
                UsdPhysics.ArticulationRootAPI.Apply(prim)
        if PhysxSchema is not None and prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI(prim).CreateArticulationEnabledAttr(not enabled)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(not enabled)
            n_joints += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(bool(enabled))
            n_bodies += 1
    return n_bodies, n_joints, n_art
