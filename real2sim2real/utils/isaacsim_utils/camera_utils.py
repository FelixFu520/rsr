"""相机工具：Camera/壳体挂载、内参与裁剪、RTX 传感器绑定、筛选及数据读取。

统一负责 RGB、深度、点云和分割等相机 annotator 数据；通用 USD 操作放在 openusd_utils。

pxr / isaacsim 仅在函数内 import,避免 stub 通路被重依赖拖垮.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from real2sim2real.core.logger import get_logger
from real2sim2real.utils.isaacsim_utils.openusd_utils import (
    ensure_xform,
    normalize_prim_path,
    resolve_usd_path,
    spawn_reference,
)
from real2sim2real.utils.isaacsim_utils.physics_utils import disable_subtree_collision

_LOG = get_logger(__name__)


def spawn_camera(
    stage: Any,
    prim_path: str,
    usd_path: str,
    add_reference_to_stage: Callable[..., Any],
) -> None:
    """有壳体 USD 时只挂引用(资产内已含 Camera prim);否则建裸 Camera.

    Args:
        stage: 当前 USD stage.
        prim_path: 实例根路径.
        usd_path: 壳体 USD;空字符串表示 Define 一个裸 Camera.
        add_reference_to_stage: Isaac 挂引用函数.

    Returns:
        None.
    """
    usd = (usd_path or "").strip()
    if usd:
        spawn_reference(stage, prim_path, resolve_usd_path(usd), add_reference_to_stage)
        return
    if stage is None:
        return
    from pxr import UsdGeom

    path = normalize_prim_path(prim_path)
    parent = path.rsplit("/", 1)[0]
    if parent and parent != "/":
        ensure_xform(stage, parent)
    UsdGeom.Camera.Define(stage, path)


def _camera_prim_priority(prim_name: str) -> int:
    """多 Camera 壳体里选哪一路绑 RGB:Color/RGB 优先,Depth 垫底.

    RealSense D455 的 ``rsd455.usd`` 把 ``Camera_Pseudo_Depth`` 写在 Color
    前面;按 PrimRange 取第一个会绑到深度相机,采集画面不是 RGB。

    Args:
        prim_name: Camera prim 名,如 ``Camera_OmniVision_OV9782_Color``.

    Returns:
        越小越优先:Color/RGB=0,其它=1,Depth/Pseudo=2.
    """
    name = prim_name.lower()
    if "color" in name or "rgb" in name:
        return 0
    if "depth" in name or "pseudo" in name:
        return 2
    return 1


def camera_prim_path(stage: Any, root_path: str) -> str:
    """实例根下用于 RGB 采集的 Camera prim;根本身是 Camera 则直接返回.

    壳体 USD 的 Camera 常在子路径,RtxCamera 必须绑到真正的 Camera prim.
    同一壳体有多路 Camera 时优先 Color/RGB,避免绑到 Depth。

    Args:
        stage: 当前 USD stage.
        root_path: 配置里的相机实例根.

    Returns:
        选中的 ``Camera`` prim 路径;根本身是 Camera 则返回根路径.

    Raises:
        RuntimeError: stage 未就绪、根 prim 不存在、或子树没有 Camera.
    """
    path = normalize_prim_path(root_path)
    if stage is None:
        raise RuntimeError(f"Stage 未就绪,无法解析相机: {path}")
    from pxr import Usd

    root = stage.GetPrimAtPath(path)
    if root is None or not root.IsValid():
        raise RuntimeError(f"相机 prim 不存在: {path}")
    cameras = [prim for prim in Usd.PrimRange(root) if prim.GetTypeName() == "Camera"]
    if not cameras:
        raise RuntimeError(f"未找到 Camera prim: {path}")
    cameras.sort(key=lambda prim: _camera_prim_priority(prim.GetName()))
    return str(cameras[0].GetPath())


def apply_camera_clipping(
    stage: Any,
    cam_prim: str,
    clipping_range: list[float] | None,
) -> None:
    """把 ``clippingRange`` 写到 Camera prim;未配置则保留 USD 原值.

    Args:
        stage: 当前 USD stage;None 则直接返回.
        cam_prim: 真正的 Camera prim 路径(不是壳体根).
        clipping_range: ``[near, far]``;None 表示不动 USD.

    Returns:
        None.
    """
    if clipping_range is None or stage is None:
        return
    prim = stage.GetPrimAtPath(cam_prim)
    if prim is None or not prim.IsValid():
        _LOG.warning("设置相机裁剪范围失败", prim_path=cam_prim)
        return
    from pxr import Gf, UsdGeom

    near, far = clipping_range
    UsdGeom.Camera(prim).GetClippingRangeAttr().Set(Gf.Vec2f(float(near), float(far)))
    _LOG.info("设置相机裁剪范围成功", prim_path=cam_prim, near=near, far=far)


def hide_camera_visuals(stage: Any, root_path: str) -> int:
    """把相机实例下的可渲染几何设为不可见,避免壳体挡住镜头;Camera prim 本身保留.

    壳体 USD(如 SG5)自带外壳 Mesh,不藏的话镜头会拍到自己的黑块.
    视口里 Camera gizmo 仍在,只是 Mesh 不进渲染.

    Args:
        stage: 当前 USD stage;None 时直接返回 0.
        root_path: 相机实例根 prim(可能是壳体 Xform,Camera 在子树).

    Returns:
        被设为 invisible 的 Gprim 数量.
    """
    if stage is None:
        return 0
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(normalize_prim_path(root_path))
    if root is None or not root.IsValid():
        return 0
    hidden = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "Camera":
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        hidden += 1
    return hidden


def apply_camera_intrinsics(stage: Any, cam: Any) -> None:
    """给裸 Camera(``usd_path`` 为空)写焦距/光圈;壳体 USD 自带内参则不动.

    Args:
        stage: 当前 USD stage;None 则直接返回.
        cam: 相机配置(``usd_path`` / ``prim_path`` / 焦距光圈字段).

    Returns:
        None. 壳体 USD 已自带内参、未配置焦距光圈、或 prim 无效时直接返回.
    """
    if (getattr(cam, "usd_path", None) or "").strip():
        return
    if (
        getattr(cam, "focal_length", None) is None
        and getattr(cam, "horizontal_aperture", None) is None
        and getattr(cam, "vertical_aperture", None) is None
    ):
        return
    if stage is None:
        return
    try:
        prim_path = camera_prim_path(stage, cam.prim_path)
    except RuntimeError:
        prim_path = normalize_prim_path(cam.prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid() or prim.GetTypeName() != "Camera":
        return
    from pxr import UsdGeom

    usd_cam = UsdGeom.Camera(prim)
    if cam.focal_length is not None:
        usd_cam.GetFocalLengthAttr().Set(float(cam.focal_length))
    if cam.horizontal_aperture is not None:
        usd_cam.GetHorizontalApertureAttr().Set(float(cam.horizontal_aperture))
    if cam.vertical_aperture is not None:
        usd_cam.GetVerticalApertureAttr().Set(float(cam.vertical_aperture))


def bind_camera_sensor(
    stage: Any,
    cam: Any,
    *,
    disable_collision: bool = False,
    camera_prim: str | None = None,
    sensor_name: str | None = None,
    apply_visuals: bool = True,
) -> Any:
    """对已摆放的 Camera prim 包装 CameraSensor,按配置挂 annotators.

    须在 pose 写好之后调用;Play/reset 之后才能稳定出 RGB.

    Args:
        stage: 当前 USD stage.
        cam: 相机配置(``prim_path`` / ``resolution`` / ``annotators`` 等).
        disable_collision: True 时关掉相机子树碰撞(人形机挂在 link 下).
        camera_prim: 显式 Camera prim;空则从壳体里挑 Color/RGB.
        sensor_name: 日志/落盘名;空则用 ``cam.name``.
        apply_visuals: False 时不再藏壳体/关碰撞(组绑时只做一次).

    Returns:
        已绑定 annotators 的 ``CameraSensor``.
    """
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

    enable_extension("isaacsim.sensors.experimental.rtx")
    name = sensor_name or cam.name
    cam_prim = camera_prim or camera_prim_path(stage, cam.prim_path)
    if stage is not None:
        prim = stage.GetPrimAtPath(cam_prim)
        if prim is not None and prim.IsValid() and not prim.HasAPI("OmniSensorAPI"):
            prim.ApplyAPI("OmniSensorAPI")
    apply_camera_clipping(stage, cam_prim, cam.clipping_range)
    apply_camera_intrinsics(stage, cam)
    if apply_visuals and cam.hide_visuals:
        n_hidden = hide_camera_visuals(stage, cam.prim_path)
        _LOG.info(
            "隐藏相机可视化成功",
            name=name,
            prim_path=cam.prim_path,
            n_hidden=n_hidden,
        )
    if apply_visuals and disable_collision:
        disable_subtree_collision(stage, cam.prim_path)
    width, height = cam.resolution
    rtx = RtxCamera(
        cam_prim,
        tick_rate=cam.tick_rate,
        reset_xform_op_properties=False,
    )
    sensor = CameraSensor(
        rtx,
        resolution=(height, width),
        annotators=list(cam.annotators),
    )
    _LOG.info(
        "绑定相机传感器成功",
        name=name,
        camera_prim=cam_prim,
        resolution=cam.resolution,
        annotators=cam.annotators,
    )
    return sensor


# 读帧时的短名 → CameraSensor annotator;配置里仍写官方名
_ANNOTATOR_ALIASES = {
    "image": "rgb",
    "color": "rgb",
    "depth": "distance_to_image_plane",
}


def resolve_annotator(annotator: str) -> str:
    """把读帧短名收成 CameraSensor annotator.

    ``depth`` → ``distance_to_image_plane``;其余原样返回(小写去空白).

    Args:
        annotator: ``rgb`` / ``depth`` / ``distance_to_image_plane`` 等.

    Returns:
        Isaac ``get_data`` 用的 annotator 名.

    Raises:
        ValueError: 空字符串.
    """
    key = (annotator or "").strip().lower()
    if not key:
        raise ValueError("annotator 不能为空")
    return _ANNOTATOR_ALIASES.get(key, key)


def read_sensor_data(
    sensor: Any,
    annotator: str = "rgb",
    *,
    name: str = "",
) -> np.ndarray | None:
    """从已绑定的 ``CameraSensor`` 取一帧;失败则 ``None``.

    Args:
        sensor: ``CameraSensor``;None 则直接返回 None.
        annotator: ``rgb`` / ``depth`` / ``distance_to_image_plane`` 等;
            须已出现在该相机 ``annotators`` 里.
        name: 日志用相机名.

    Returns:
        numpy 数组;RGB 多为 ``uint8`` ``HxWx3``,深度多为浮点 ``HxW``.
        sensor 为空、取数失败或无数据则为 ``None``.
    """
    if sensor is None:
        return None
    try:
        key = resolve_annotator(annotator)
    except ValueError as exc:
        _LOG.warning("get_image_failed", name=name, annotator=annotator, error=str(exc))
        return None
    try:
        data, _info = sensor.get_data(key)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("get_image_failed", name=name, annotator=key, error=str(exc))
        return None
    if data is None:
        return None
    try:
        return data.numpy()
    except AttributeError:
        return np.asarray(data)


def normalize_camera_key(value: str) -> str:
    """收成可比较的 prim/名:去空白、反斜杠,路径压成 ``/A/B``.

    与 ``normalize_prim_path`` 不同:空输入仍为空,无前导 ``/`` 的当名称保留.

    Args:
        value: 配置或运行时路径/名称.

    Returns:
        规范化字符串;空输入仍为空.
    """
    raw = (value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    if raw.startswith("/"):
        parts = [p for p in raw.split("/") if p]
        return "/" + "/".join(parts) if parts else "/"
    return raw


def camera_matches_prims(aliases: list[str], filters: list[str]) -> bool:
    """``aliases`` 是否命中任一 prim/名过滤项.

    Args:
        aliases: 传感器名、Camera prim、壳体根、配置挂载 prim.
        filters: 配置里的 prim/名;空则全部通过.

    Returns:
        应保留该相机则为 ``True``.
    """
    if not filters:
        return True
    keys = [normalize_camera_key(a) for a in aliases if a and normalize_camera_key(a)]
    if not keys:
        return False
    for raw in filters:
        filt = normalize_camera_key(raw)
        if not filt:
            continue
        for key in keys:
            if key == filt:
                return True
            if key.startswith(filt.rstrip("/") + "/"):
                return True
            if "/" not in filt and (key == filt or key.startswith(filt + "_")):
                return True
            if key.rsplit("/", 1)[-1] == filt:
                return True
    return False


def iter_bound_cameras(
    *owners: Any,
) -> list[tuple[str, str, list[str]]]:
    """收集已绑定传感器:``(name, camera_prim, aliases)``.

    Args:
        owners: scene / robot,可缺.

    Returns:
        能读到 Camera prim 的条目;读失败的跳过.
    """
    entries: list[tuple[str, str, list[str]]] = []
    for owner in owners:
        if owner is None:
            continue
        cfgs = list(getattr(getattr(owner, "config", None), "cameras", None) or [])
        sensors = getattr(owner, "sensors", None) or {}
        for name, sensor in sensors.items():
            try:
                paths = sensor.camera.paths
                if not paths:
                    continue
                camera_prim = str(paths[0])
            except Exception:  # noqa: BLE001
                continue
            aliases = [name, camera_prim]
            for cfg in cfgs:
                cfg_name = str(getattr(cfg, "name", "") or "")
                if cfg_name and (name == cfg_name or name.startswith(cfg_name + "_")):
                    aliases.append(cfg_name)
                    prim_path = str(getattr(cfg, "prim_path", "") or "")
                    authored = str(getattr(cfg, "authored_prim_path", "") or "")
                    aliases.append(prim_path)
                    aliases.append(authored)
            entries.append((name, camera_prim, aliases))
    return entries


def filter_camera_targets(
    entries: list[tuple[str, str, list[str]]],
    filters: list[str],
) -> dict[str, str]:
    """按 prim 过滤后得到 ``name -> camera_prim``.

    ``[job].viewport_camera_prims`` / ``[job.recorder].record_camera_prims``
    为空则不过滤.过滤项可以是:

    - 壳体或实例根(``/World/Cameras/head_cam``),匹配其下全部 Camera
    - 真正的 Camera prim(``.../RSD455/Camera_OmniVision_OV9782_Color``)
    - 传感器名(``head_cam`` 会带上 ``head_cam_color`` 等组内流)

    Args:
        entries: ``iter_bound_cameras`` 的结果.
        filters: 视口或落盘 prim 列表;空则全收.

    Returns:
        命中条目的有序字典.
    """
    return {name: prim for name, prim, aliases in entries if camera_matches_prims(aliases, filters)}


CAMERA_ANNOTATORS = frozenset(
    {
        "rgb",  # 彩色图,采集主通路
        "rgba",
        "distance_to_image_plane",  # 针孔深度
        "distance_to_camera",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation",
        "instance_id_segmentation",
        "bounding_box_2d_tight",
        "bounding_box_2d_loose",
        "bounding_box_3d",
        "pointcloud",
    }
)
