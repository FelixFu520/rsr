"""改写 USD stage:补 Xform、挂引用、写位姿、加载 payload、泵 Kit 帧.

pxr 仅在函数内 import,避免 stub 通路被重依赖拖垮.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from real2sim2real.core.logger import get_logger
from real2sim2real.utils.isaacsim_utils.openusd_utils import (
    WORLD_MOUNT_ROOTS,
    get_stage,
    is_descendant,
    is_world_frame,
    normalize_prim_path,
    resolve_usd_path,
)
from real2sim2real.utils.pose import compose_pose, read_prim_world_pose

_LOG = get_logger(__name__)
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


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


def ensure_xform(stage: Any, prim_path: str) -> None:
    """沿 prim 路径逐级创建缺失的 Xform,保证引用有合法父节点.

    例如 ``/World/scene/Objects/cube_0`` 会确保 /World、/World/scene、/World/scene/Objects 都存在.

    Args:
        stage: 当前 USD stage;None 则直接返回.
        prim_path: 目标路径(含或不含最后一级都可以).

    Returns:
        None.
    """
    if stage is None:
        return
    from pxr import UsdGeom

    acc = ""
    for part in [p for p in normalize_prim_path(prim_path).split("/") if p]:
        acc += f"/{part}"
        prim = stage.GetPrimAtPath(acc)
        if prim is None or not prim.IsValid():
            UsdGeom.Xform.Define(stage, acc)


def ensure_world_mount_roots(stage: Any) -> None:
    """在 ``/World`` 下建 background / scene / robots 三个互为兄弟的空 Xform.

    背景、场景、机器人分别挂到对应子树,资产内部 scale 不会传到另外两棵树.
    只补缺失节点,不写 translate / orient / scale.

    Args:
        stage: 当前 USD stage;None 则直接返回.

    Returns:
        None.
    """
    for path in WORLD_MOUNT_ROOTS:
        ensure_xform(stage, path)


def add_usd_reference(*, usd_path: str, prim_path: str) -> None:
    """用 pxr ``AddReference`` 挂资产,绕开 Isaac ``add_reference_to_stage``.

    Isaac 那条 API 会走 Metrics Assembler:部分工厂文件标 ``metersPerUnit=0.01``,
    但几何数字已经是米,Assembler 会在引用根注入 ``scale=0.01``,整座厂房缩成厘米.

    Args:
        usd_path: 已 resolve 的绝对 USD 路径.
        prim_path: 已存在的实例根.

    Returns:
        None.

    Raises:
        RuntimeError: 当前 stage 或目标 prim 无效.
    """
    import omni.usd
    from pxr import Sdf

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path) if stage is not None else None
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"引用目标 prim 不存在: {prim_path}")
    prim.GetReferences().AddReference(Sdf.Reference(usd_path))


def mark_ignore_metrics_assembler(stage: Any, prim_path: str) -> None:
    """在引用根上标记忽略 Metrics Assembler,避免再次自动缩尺度.

    Args:
        stage: 当前 USD stage.
        prim_path: 引用根.

    Returns:
        None.
    """
    if stage is None:
        return
    prim = stage.GetPrimAtPath(normalize_prim_path(prim_path))
    if prim is None or not prim.IsValid():
        return
    for key in (
        "metricsAssembler:ignore",
        "disableMetricsAssembler",
        "omni:metricsassembler:ignore",
    ):
        try:
            prim.SetCustomDataByKey(key, True)
        except Exception:  # noqa: BLE001
            pass


def spawn_reference(
    stage: Any,
    prim_path: str,
    usd_path: str,
    add_reference_to_stage: Callable[..., Any],
) -> None:
    """空 Xform 作实例根,USD 挂到该 prim,避免改引用层的 xform 把 Kit 打崩.

    先 ``ensure_xform`` 再 ``add_reference_to_stage``:Isaac 要求目标 prim
    已存在或可创建,且不要直接改 payload 层的 transform.

    Args:
        stage: 当前 USD stage;给父节点补 Xform.
        prim_path: 实例根.
        usd_path: 已 resolve 的绝对 USD 路径.
        add_reference_to_stage: Isaac 挂引用 API.

    Returns:
        None.
    """
    ensure_xform(stage, prim_path)
    add_reference_to_stage(usd_path=usd_path, prim_path=normalize_prim_path(prim_path))


def spawn_subprim_reference(
    stage: Any,
    prim_path: str,
    usd_path: str,
    src_prim_path: str,
) -> None:
    """把 USD 里指定 prim 引用到 ``prim_path``,不改资产内部尺度.

    ``add_reference_to_stage`` 只会挂默认 prim;场景包默认 prim 是整份 ``/World``.
    机器人视觉网格是 instanceable 子树,必须按源 prim 路径引用,不能用碰撞方块代理 USD.

    Args:
        stage: 当前 USD stage;给路径补 Xform.
        prim_path: 本 stage 上的实例根.
        usd_path: 已 resolve 的绝对 USD 路径.
        src_prim_path: 被引用文件里的源 prim,如 ``/World/Pro_m6_696_dynamics_fixed``.

    Returns:
        None.

    Raises:
        RuntimeError: stage 未就绪,或 AddReference 失败.
    """
    if stage is None:
        raise RuntimeError(f"Stage 未就绪,无法引用 {src_prim_path}")
    dest = normalize_prim_path(prim_path)
    src = normalize_prim_path(src_prim_path)
    ensure_xform(stage, dest)
    prim = stage.GetPrimAtPath(dest)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"无法创建引用根: {dest}")
    if not prim.GetReferences().AddReference(usd_path, src):
        raise RuntimeError(f"引用子 prim 失败: {usd_path} {src} -> {dest}")


def apply_pose(
    stage: Any,
    prim_path: str,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
) -> None:
    """把世界位姿写入 prim 的 translate / orient xform;prim 缺失则跳过.

    复用已有 op,避免 AddTranslateOp 叠多层把资产内部局部变换冲掉.
    orient 优先 Quatd,部分资产只接受 Quatf.

    Args:
        stage: 当前 USD stage;None 则直接返回.
        prim_path: 要写 xform 的 prim.
        position: 世界 xyz(米).
        orientation_wxyz: 世界朝向 ``(w,x,y,z)``.

    Returns:
        None. prim 缺失或 stage 未就绪时直接返回,不抛错.

    Raises:
        RuntimeError: xform 写入失败(例如 op 类型不兼容).
    """
    if stage is None:
        return
    prim = stage.GetPrimAtPath(normalize_prim_path(prim_path))
    if prim is None or not prim.IsValid():
        return
    from pxr import Gf, UsdGeom

    pos = [float(x) for x in position]
    w, x, y, z = (float(v) for v in orientation_wxyz)
    try:
        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        orient_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
                orient_op = op
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if orient_op is None:
            orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(*pos))
        try:
            orient_op.Set(Gf.Quatd(w, x, y, z))
        except Exception:  # noqa: BLE001
            orient_op.Set(Gf.Quatf(w, x, y, z))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"设置位姿失败: {prim_path}") from exc


def parent_pose(
    relative_to: str,
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    stage: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """解析 ``pose_relative_to`` 的父坐标系世界位姿;未知则退回世界原点.

    先查 ``poses`` 缓存,再读 stage(机械臂动态 link 如 ``panda_hand``).

    Args:
        relative_to: ``/World`` 或已摆放 prim 路径.
        poses: prim → (世界 pos, wxyz) 缓存;从 stage 读到时会回写.
        stage: 可选 USD stage;缓存未命中时 ``read_prim_world_pose``.

    Returns:
        父坐标系世界 ``(pos, quat_wxyz)``;未知则零平移与单位四元数.
    """
    if is_world_frame(relative_to):
        return np.zeros(3, dtype=float), _IDENTITY_QUAT.copy()
    key = normalize_prim_path(relative_to)
    cached = poses.get(key)
    if cached is not None:
        return cached
    if stage is not None:
        from_stage = read_prim_world_pose(stage, key)
        if from_stage is not None:
            poses[key] = from_stage
            return from_stage
    return np.zeros(3, dtype=float), _IDENTITY_QUAT.copy()


def place_asset(
    stage: Any,
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    prim_path: str,
    pose: Any,
    pose_relative_to: str,
    *,
    local_if_descendant: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """相对父坐标系合成世界位姿,写回 prim 并缓存到 ``poses``.

    Args:
        stage: 当前 USD stage.
        poses: prim → (世界 pos, wxyz) 缓存;本函数会写入目标 prim.
        prim_path: 要摆的资产根.
        pose: ``PoseConfig``,提供 ``as_arrays()`` → (rel_pos, rel_quat).
        pose_relative_to: 父坐标系 prim;``/World`` 表示世界系.
        local_if_descendant: True 且 prim 是父系子孙时写本地位姿(腕部相机).

    Returns:
        合成后的世界 ``(pos, quat_wxyz)``;同时写入 ``poses`` 缓存.
    """
    parent_pos, parent_quat = parent_pose(pose_relative_to, poses, stage=stage)
    rel_pos, rel_quat = pose.as_arrays()
    world_pos, world_quat = compose_pose(parent_pos, parent_quat, rel_pos, rel_quat)
    if local_if_descendant and is_descendant(prim_path, pose_relative_to):
        apply_pose(stage, prim_path, rel_pos, rel_quat)
    else:
        apply_pose(stage, prim_path, world_pos, world_quat)
    poses[normalize_prim_path(prim_path)] = (world_pos, world_quat)
    return world_pos, world_quat


def ensure_prim_loaded(stage: Any, prim_path: str, *, select_mesh_variant: bool = False) -> None:
    """加载 payload;可选选中 ``Mesh`` variant,让实例网格进入 PrimRange.

    Args:
        stage: 当前 USD stage;None 则直接返回.
        prim_path: 刚挂上的引用根.
        select_mesh_variant: True 时优先 Performance / Detailed(Franka).

    Returns:
        None.
    """
    if stage is None:
        return
    prim = stage.GetPrimAtPath(normalize_prim_path(prim_path))
    if prim is None or not prim.IsValid():
        return
    try:
        prim.Load()
    except Exception:  # noqa: BLE001
        pass
    if not select_mesh_variant:
        return
    try:
        vsets = prim.GetVariantSets()
        if not vsets.HasVariantSet("Mesh"):
            return
        vset = vsets.GetVariantSet("Mesh")
        names = list(vset.GetVariantNames())
        for preferred in ("Performance", "Detailed"):
            if preferred in names:
                vset.SetVariantSelection(preferred)
                return
        if names and not vset.GetVariantSelection():
            vset.SetVariantSelection(names[0])
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("select_mesh_variant_failed", error=str(exc))


def open_robot_usd(robot: Any, sim_world: Any, *, mesh_variant: bool = False) -> Any:
    """绑定 sim world,挂机器人 USD,返回 stage.

    Args:
        robot: 须有 ``config.usd_path`` / ``config.prim_path``.
        sim_world: 已 init_world 的世界.
        mesh_variant: True 时选中 Mesh variant(桌面臂贴合需要).

    Returns:
        当前 USD stage.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage

    from real2sim2real.utils.misc import emit_log

    robot.sim_world = sim_world
    stage = get_stage(sim_world)
    assert stage is not None, "Stage is not initialized"
    usd = resolve_usd_path(getattr(robot.config, "usd_path", "") or "")
    spawn_reference(stage, robot.config.prim_path, usd, add_reference_to_stage)
    ensure_prim_loaded(stage, robot.config.prim_path, select_mesh_variant=mesh_variant)
    emit_log(robot, "info", "open_robot_usd", prim_path=robot.config.prim_path, usd=usd)
    pump_frames(sim_world, 2)
    return stage
