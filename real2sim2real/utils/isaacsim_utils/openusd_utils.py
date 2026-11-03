"""OpenUSD 路径与查询:prim 规范化、解析 USD 文件、读 stage / AABB / link 位姿.

``normalize_prim_path`` / ``resolve_usd_path`` 不依赖 Kit;
pxr / isaacsim 仅在函数内 import,避免 stub 通路被重依赖拖垮.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from real2sim2real.core.runtime_env import assets_root
from real2sim2real.utils.pose import read_prim_world_pose

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
# Nucleus 资产根:standalone 常在 ~/isaacsim/Assets/isaacsim,本机缓存则在 ~/Assets/isaacsim
ISAAC_ASSET_ROOTS = (
    Path.home() / "isaacsim/Assets/isaacsim",
    Path.home() / "Assets/isaacsim",
    Path("/home/fufa/isaacsim/Assets/isaacsim"),
    Path("/home/fufa/Assets/isaacsim"),
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
    candidates = [assets_root() / rel, assets_root() / raw]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    # 绝对路径写了另一种 Nucleus 根时,按 Isaac/... 后缀在常见根下重找
    raw_norm = str(path).replace("\\", "/")
    marker = "/Isaac/"
    idx = raw_norm.find(marker)
    if idx >= 0:
        suffix = raw_norm[idx + 1 :]  # Isaac/...
        for root in ISAAC_ASSET_ROOTS:
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
    from real2sim2real.utils.isaacsim_utils.modify_stage_utils import apply_pose

    snapped = np.asarray(world_pos, dtype=float).copy()
    apply_pose(stage, prim_path, snapped, world_quat)
    bounds = prim_aabb(stage, prim_path)
    if bounds is None:
        snapped[2] = float(floor_z)
        return snapped, world_quat
    snapped = snapped.copy()
    snapped[2] += float(floor_z) - float(bounds[0][2])
    return snapped, world_quat
