"""UsdPhysics 辅助:kinematic 刚体、静态碰撞、关闭子树碰撞.

pxr 仅在函数内 import,避免 stub 通路被重依赖拖垮.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from real2sim2real.utils.isaacsim_utils.openusd_utils import normalize_prim_path

_SKIP_COLLIDER_NAME_RE = re.compile(
    r"(sky|dome|atmosphere|skysphere|celestial)",
    re.IGNORECASE,
)
_MAX_STATIC_COLLIDER_SPAN_M = 250.0


def pump_frames(sim_world: Any, frames: int) -> None:
    """泵若干帧 Kit 主循环,让刚挂上的 USD 完成加载与同步.

    走 ``BaseSimWorld.update``(Isaac 泵 App,stub no-op).不跑物理.

    Args:
        sim_world: 已绑定的仿真世界;None 则什么都不做.
        frames: 要泵的帧数;<=0 则什么都不做.

    Returns:
        None.
    """
    if sim_world is None:
        return
    for _ in range(max(0, frames)):
        sim_world.update()


def set_kinematic(
    stage: Any,
    root_path: str,
    *,
    enabled: bool,
    apply_if_missing: bool = False,
) -> int:
    """把子树内带 ``PhysicsRigidBodyAPI`` 的 prim 设为 kinematic.

    Args:
        stage: 当前 USD stage;None 则返回 0.
        root_path: 根 prim;会遍历子树.
        enabled: True=锁成 kinematic;False=恢复动态刚体.
        apply_if_missing: True 时若根 prim 没有刚体 API 则补上(导航木块).

    Returns:
        被改写的刚体 prim 数.
    """
    if stage is None:
        return 0
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(normalize_prim_path(root_path))
    if root is None or not root.IsValid():
        return 0
    if apply_if_missing and not root.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(root)
    n = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(bool(enabled))
            n += 1
    return n


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


def apply_joint_drive_targets(stage: Any, root_path: str, targets: dict[str, float]) -> int:
    """按关节名写 USD drive 目标:转动副用度,移动副用米.

    Args:
        stage: 当前 USD stage;None 则返回 0.
        root_path: 搜索根 prim.
        targets: 关节名 → 目标(转动副为弧度,移动副为米).

    Returns:
        成功写入的关节数.
    """
    if stage is None or not targets:
        return 0
    import math

    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(normalize_prim_path(root_path))
    if root is None or not root.IsValid():
        return 0
    n_set = 0
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name not in targets:
            continue
        value = float(targets[name])
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive and drive.GetTargetPositionAttr():
                drive.GetTargetPositionAttr().Set(math.degrees(value))
                n_set += 1
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            drive = UsdPhysics.DriveAPI.Get(prim, "linear")
            if drive:
                if drive.GetTargetPositionAttr():
                    drive.GetTargetPositionAttr().Set(value)
                else:
                    drive.CreateTargetPositionAttr(value)
                n_set += 1
            pos_attr = prim.GetAttribute("state:linear:physics:position")
            if pos_attr and pos_attr.IsValid():
                pos_attr.Set(value)
    return n_set


def is_physics_tensor_ready(art: Any) -> bool:
    """articulation 的 PhysX tensor view 是否已建立.

    Args:
        art: Isaac Articulation;None 视为未就绪.

    Returns:
        tensor 有效则为 True.
    """
    if art is None:
        return False
    try:
        ready = getattr(art, "is_physics_tensor_entity_valid", None)
        return bool(ready()) if callable(ready) else False
    except Exception:  # noqa: BLE001
        return False


def _as_dof_vector(raw: Any) -> np.ndarray:
    """把 articulation 读出的 batch 张量收成 1D float 向量.

    Args:
        raw: ``get_*_positions`` / ``get_*_velocities`` 的返回值.

    Returns:
        ``(n_dof,)`` 数组.
    """
    try:
        arr = np.asarray(raw.numpy(), dtype=float)
    except AttributeError:
        arr = np.asarray(raw, dtype=float)
    if arr.ndim >= 2:
        return arr[0]
    return arr


def read_articulation_world_pose(art: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """从 PhysX 读 articulation 根世界位姿.

    Args:
        art: Isaac Articulation.

    Returns:
        ``(pos, quat_wxyz)``;未就绪或读失败则为 ``None``.
    """
    if not is_physics_tensor_ready(art):
        return None
    try:
        positions, orientations = art.get_world_poses()
        pos = _as_dof_vector(positions)
        quat = _as_dof_vector(orientations)
        return pos[:3].copy(), quat[:4].copy()
    except Exception:  # noqa: BLE001
        return None


def read_articulation_dof_vectors(art: Any) -> tuple[np.ndarray, np.ndarray]:
    """读关节位置 / 速度;优先 ``get_joint_*``,否则 ``get_dof_*``.

    Args:
        art: Isaac Articulation / SingleManipulator;None 则返回两个空向量.

    Returns:
        ``(q, qd)``;读失败则为空或与 q 同形的零.
    """
    empty = np.zeros(0, dtype=float)
    if art is None:
        return empty, empty
    q = empty
    for getter in ("get_joint_positions", "get_dof_positions"):
        fn = getattr(art, getter, None)
        if not callable(fn):
            continue
        try:
            q = _as_dof_vector(fn())
            break
        except Exception:  # noqa: BLE001
            continue
    qd = np.zeros_like(q)
    for getter in ("get_joint_velocities", "get_dof_velocities"):
        fn = getattr(art, getter, None)
        if not callable(fn):
            continue
        try:
            qd = _as_dof_vector(fn())
            break
        except Exception:  # noqa: BLE001
            continue
    return q, qd


def resolve_articulation_link_index(art: Any, prim_path: str) -> int | None:
    """按 prim 路径或末段名在 ``link_names`` / ``link_paths`` 里找下标.

    Args:
        art: locomotion 的 articulation.
        prim_path: 绑定 link.

    Returns:
        link 下标;找不到则为 ``None``.
    """
    if art is None:
        return None
    want = normalize_prim_path(prim_path)
    name = want.rsplit("/", 1)[-1]
    names = list(getattr(art, "link_names", None) or [])
    if name in names:
        return names.index(name)
    paths = getattr(art, "link_paths", None)
    if not paths:
        return None
    flat = paths[0] if paths and isinstance(paths[0], (list, tuple)) else paths
    for i, item in enumerate(flat):
        key = normalize_prim_path(str(item))
        if key == want or key.endswith("/" + name):
            return i
    return None


def read_articulation_link_pose(art: Any, link_index: int) -> tuple[np.ndarray, np.ndarray] | None:
    """从 PhysX tensor 读指定 link 的世界位姿.

    Args:
        art: locomotion 的 articulation.
        link_index: ``resolve_articulation_link_index`` 得到的下标.

    Returns:
        世界 ``(pos, quat_wxyz)``;未就绪则为 ``None``.
    """
    if not is_physics_tensor_ready(art):
        return None
    view = getattr(art, "_physics_articulation_view", None)
    if view is None:
        return None
    try:
        data = view.get_link_transforms()
        try:
            raw = np.asarray(data.numpy(), dtype=float)
        except AttributeError:
            raw = np.asarray(data, dtype=float)
        if raw.ndim == 3:
            pose7 = raw[0, link_index]
        elif raw.ndim == 2:
            pose7 = raw[link_index]
        else:
            return None
        pos = np.asarray(pose7[:3], dtype=float).copy()
        qx, qy, qz, qw = (float(v) for v in pose7[3:7])
        return pos, np.array([qw, qx, qy, qz], dtype=float)
    except Exception:  # noqa: BLE001
        return None


def link_pose_from_articulation(
    art: Any,
    prim_path: str,
    bind_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """从 articulation PhysX tensor 读 link 世界位姿.

    Args:
        art: 整机关节体.
        prim_path: 绑定 link.
        bind_index: link prim → 下标缓存;本函数会写入.

    Returns:
        世界 ``(pos, quat_wxyz)``;未就绪或找不到 link 则为 ``None``.
    """
    idx = bind_index.get(prim_path)
    if idx is None:
        idx = resolve_articulation_link_index(art, prim_path)
        if idx is None:
            return None
        bind_index[prim_path] = idx
    return read_articulation_link_pose(art, idx)
