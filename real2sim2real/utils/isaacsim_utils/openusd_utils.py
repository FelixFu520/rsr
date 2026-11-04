"""OpenUSD 舞台辅助:路径解析、挂引用、写位姿、加载 payload、查 AABB / link 位姿.

``normalize_prim_path`` / ``resolve_usd_path`` 不依赖 Kit;
pxr / isaacsim 仅在函数内 import,避免 stub 通路被重依赖拖垮.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from real2sim2real.core.logger import get_logger
from real2sim2real.core.runtime_env import assets_root
from real2sim2real.utils.pose import compose_pose

_LOG = get_logger(__name__)
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

# 这些写法都当作世界系,compose_pose 时父位姿为单位变换
_WORLD_ALIASES = frozenset({"/world", "world", "/", ""})
# /World 下三个互为兄弟的挂载根:各自引用的 USD 尺度不会传到另外两棵子树
WORLD_BACKGROUND_PRIM = "/World/background"
WORLD_SCENE_PRIM = "/World/scene"
WORLD_ROBOTS_PRIM = "/World/robots"
WORLD_MOUNT_ROOTS: tuple[str, ...] = (
    WORLD_BACKGROUND_PRIM,
    WORLD_SCENE_PRIM,
    WORLD_ROBOTS_PRIM,
)


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


def is_world_frame(prim_path: str) -> bool:
    """是否表示世界坐标系(``/World`` / ``world`` / ``/``).

    ``/World`` 规范化后是 ``/World``,lower 后命中 ``/world``.

    Args:
        prim_path: 待判断的 prim 路径或别名.

    Returns:
        ``True`` 表示世界系;否则 ``False``.
    """
    return normalize_prim_path(prim_path).lower() in _WORLD_ALIASES


def resolve_usd_path(usd_path: str) -> str:
    """绝对路径优先,否则相对 ``RSR_ASSETS_ROOT``.

    配置里仍可写 ``assets/...``;该前缀会剥掉后再拼到资产根上.

    Args:
        usd_path: 配置里的 USD 路径;空字符串视为非法.

    Returns:
        已存在文件的绝对路径.

    Raises:
        FileNotFoundError: 空路径或候选位置都找不到文件.
    """
    raw = (usd_path or "").strip()
    if not raw:
        raise FileNotFoundError("usd_path 为空")
    path = Path(raw).expanduser()
    if path.is_file():
        return str(path.resolve())
    rel = Path(raw)
    if rel.parts[:1] == ("assets",):
        rel = Path(*rel.parts[1:])
    root = assets_root()
    candidates = [root / rel, root / raw]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    # 绝对路径写了另一种 Nucleus 根时,按 Isaac/... 后缀在配置的资产根下重找
    raw_norm = str(path).replace("\\", "/")
    marker = "/Isaac/"
    idx = raw_norm.find(marker)
    if idx >= 0:
        suffix = raw_norm[idx + 1 :]  # Isaac/...
        candidate = root / suffix
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"找不到 USD: {usd_path}")


def get_stage(sim_world: Any) -> Any | None:
    """从已绑定的 sim world 取当前 USD stage;未就绪则返回 None.

    Args:
        sim_world: 已绑定的仿真世界;None 或尚无 ``stage`` 属性视为未就绪.

    Returns:
        当前 ``Usd.Stage``;``sim_world`` 为 None 或未绑定时为 ``None``.
    """
    if sim_world is None:
        return None
    return getattr(sim_world, "stage", None)


def is_descendant(prim_path: str, pose_relative_to: str) -> bool:
    """``prim_path`` 是否已是 ``pose_relative_to`` 在 USD 树里的真实子孙.

    腕部相机挂在 ``.../panda_hand/wrist_cam`` 下时应写本地位姿,让父级
    每帧物理变换自动合成世界系;兄弟节点则必须写世界位姿.

    Args:
        prim_path: 待判断的子 prim.
        pose_relative_to: 假定的父坐标系 prim;世界系别名视为非子孙.

    Returns:
        ``True`` 当且仅当 ``prim_path`` 是父路径的真子孙(不含自身).
    """
    if is_world_frame(pose_relative_to):
        return False
    child = normalize_prim_path(prim_path)
    parent = normalize_prim_path(pose_relative_to)
    return child != parent and child.startswith(parent + "/")


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


def prim_aabb_z_min(stage: Any, prim_path: str) -> float | None:
    """返回 prim 世界 AABB 的 z 下界;算不出则 None.

    Args:
        stage: 当前 USD stage.
        prim_path: 任意已在 stage 上的 prim.

    Returns:
        世界 AABB 的 z 最小值(米);算不出则为 ``None``.
    """
    bounds = prim_aabb(stage, prim_path)
    return None if bounds is None else float(bounds[0][2])


def prim_aabb_z_max(stage: Any, prim_path: str) -> float | None:
    """返回 prim 世界 AABB 的 z 上界;算不出则 None.

    桌子用这个当桌面高度;几何原点常在桌腿,不能当支撑面.

    Args:
        stage: 当前 USD stage.
        prim_path: 任意已在 stage 上的 prim.

    Returns:
        世界 AABB 的 z 最大值(米);算不出则为 ``None``.
    """
    bounds = prim_aabb(stage, prim_path)
    return None if bounds is None else float(bounds[1][2])


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
        return None  # 读失败不当致命错误,Job 可跳过该 prim


def read_link_world_pose(stage: Any, prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """读 articulation link 的世界位姿;优先 Fabric,读不到再退回 USD.

    Fabric 是运行时场景缓存;PhysX / 渲染读它,USD 是编辑用的源.
    走路时 PhysX 只把 link 变换写进 Fabric,USD ``ComputeLocalToWorld``
    仍是站姿,用 USD 去跟相机会停在出生点.``fabric=True`` 才是真机位.

    Args:
        stage: 当前 USD stage;Fabric 失败时的退路.
        prim_path: link prim,如 ``/World/robots/G1/head_link``.

    Returns:
        世界 ``(pos, quat_wxyz)``;都读不到则为 ``None``.
    """
    path = normalize_prim_path(prim_path)
    try:
        from isaacsim.core.utils.xforms import get_world_pose

        pos, quat = get_world_pose(path, fabric=True)
        pos_a = np.asarray(pos, dtype=float).reshape(-1)
        quat_a = np.asarray(quat, dtype=float).reshape(-1)
        if pos_a.size >= 3 and quat_a.size >= 4:
            return pos_a[:3].copy(), quat_a[:4].copy()
    except Exception:  # noqa: BLE001
        pass
    return read_prim_world_pose(stage, path)


def find_prim_path(
    stage: Any,
    root_path: str,
    *,
    suffixes: tuple[str, ...] = (),
    names: set[str] | frozenset[str] | None = None,
    ignore_case: bool = True,
) -> str | None:
    """在 ``root`` 下找 prim:先试 ``root/suffix``,再按名字扫子树.

    Args:
        stage: 当前 USD stage;None 则返回 None.
        root_path: 搜索根.
        suffixes: 相对根的候选路径,如 ``panda_link0`` / ``Geometry/base_link``.
        names: 子树里按 ``GetName()`` 匹配的名字集合;None 则从 suffixes 末段推断.
        ignore_case: 按名扫描时是否忽略大小写.

    Returns:
        命中的 prim 路径;都找不到则为 ``None``.
    """
    if stage is None:
        return None
    root = normalize_prim_path(root_path)
    for suffix in suffixes:
        path = f"{root}/{suffix.lstrip('/')}"
        prim = stage.GetPrimAtPath(path)
        if prim is not None and prim.IsValid():
            return path
    want = names if names is not None else {s.rsplit("/", 1)[-1] for s in suffixes if s}
    if ignore_case:
        want = {n.lower() for n in want}
    root_prim = stage.GetPrimAtPath(root)
    if root_prim is None or not root_prim.IsValid() or not want:
        return None
    from pxr import Usd

    for prim in Usd.PrimRange(root_prim):
        name = prim.GetName()
        if (name.lower() if ignore_case else name) in want:
            return str(prim.GetPath())
    return None


def snap_prim_to_floor_z(
    stage: Any,
    prim_path: str,
    world_pos: np.ndarray,
    world_quat: np.ndarray,
    floor_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """把 prim 当前 AABB 底面落到 ``floor_z``;量不到包围盒则只改 z.

    Args:
        stage: 当前 USD stage.
        prim_path: 要贴地的根 prim.
        world_pos: 当前世界位置.
        world_quat: 当前世界朝向 ``wxyz``.
        floor_z: 地面世界 z.

    Returns:
        贴地后的 ``(世界 pos, wxyz)``;朝向不变.
    """
    snapped = np.asarray(world_pos, dtype=float).copy()
    apply_pose(stage, prim_path, snapped, world_quat)
    bounds = prim_aabb(stage, prim_path)
    if bounds is None:
        snapped[2] = float(floor_z)
        return snapped, world_quat
    snapped = snapped.copy()
    snapped[2] += float(floor_z) - float(bounds[0][2])
    return snapped, world_quat


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

    from real2sim2real.utils.isaacsim_utils.physics_utils import pump_frames
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
