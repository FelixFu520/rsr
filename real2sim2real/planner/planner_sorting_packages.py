# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""分拣包裹后端：配置与实现。"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import get_logger
from real2sim2real.planner.base import BasePlanner
from real2sim2real.utils import (
    sample_grasp_offsets,
    surface_patch_normal,
    uv_patch_triangles,
)


def _load_runtime_dependencies() -> None:
    """世界初始化后才导入 Kit 与 USD 运行时。"""
    global Articulation, DeformablePrim, Gf, PhysicsScene, PhysxSchema, RigidPrim, Sdf
    global SimulationManager, Usd, UsdGeom, UsdPhysics, XformPrim, deformableUtils
    global get_physx_attachment_private_interface, omni
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import (
        Articulation,
        DeformablePrim,
        RigidPrim,
        XformPrim,
    )
    from isaacsim.core.simulation_manager import (
        PhysicsScene,
        SimulationManager,
    )
    from omni.physx.scripts import deformableUtils
    from omni.physx.scripts.ifaces import (
        get_physx_attachment_private_interface,
    )
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


class SortingPlannerConfig(BaseConfig):
    """``[planner]``:Pro-M6 双臂分拣中继(柔性包裹 入带 → 平台 → 出带).

    状态机在 本模块.刚体仍走 PhysX;袋装包裹走
    Newton VBD 接触抓取.本配置暴露场景 prim、Newton 材料与提起验收.
    """

    timeout_s: float = Field(
        default=1800.0,
        description="单集分拣超时(s,墙钟);RTX 渲染 + GPU 柔性体下实时率远低于 1x",
    )
    robot_prim_path: str = Field(
        default="/World/robots/Pro_m6_696_dynamics_fixed", description="机器人根 prim"
    )
    clothing_root_prim_path: str = Field(
        default="/World/scene/ImportedBaggedClothing", description="柔性包裹根 prim"
    )
    clothing_surface_prim_path: str = Field(
        default="/World/scene/ImportedBaggedClothing/Asset/visual/visual",
        description="袋装包裹可视 Mesh;Play 前会去掉 PhysX 柔性 API,改由 Newton 写回顶点",
    )
    clothing_volume_proxy_prim_path: str = Field(
        default="/World/scene/ImportedBaggedClothing/Asset/visual/visual_proxy",
        description="体积代理 prim;存在时必须 inactive,避免与 Newton 软体抢显示",
    )
    support_prim_path: str = Field(
        default="/World/scene/platform/Empty/Plane/Plane_002", description="平台支撑面 prim"
    )
    belt_prim_path: str = Field(
        default="/World/scene/ConveyorBelt_A06/Belt", description="出带皮带 prim,判定包裹是否送出"
    )
    infeed_prim_path: str = Field(
        default="/World/scene/ConveyorBelt_A06_01",
        description="入带 prim;Newton 里做成静态碰撞盒,空则跳过",
    )
    left_gripper_prim_path: str = Field(
        default="", description="左夹爪 prim;空则 <root>/robotiq_left"
    )
    right_gripper_prim_path: str = Field(
        default="", description="右夹爪 prim;空则 <root>/robotiq_right"
    )
    left_finger_prim_paths: list[str] = Field(
        default_factory=list,
        description="左指垫 Mesh prim;空则在左夹爪下按 inner_finger 自动发现",
    )
    right_finger_prim_paths: list[str] = Field(
        default_factory=list,
        description="右指垫 Mesh prim;空则在右夹爪下按 inner_finger 自动发现",
    )
    runtime_root_prim_path: str = Field(
        default="/World/ProM6_SortingRuntime",
        description="运行期临时 prim 根,收尾时整棵删掉",
    )
    youngs_modulus: float = Field(default=10000.0, gt=0.0, description="Newton 包裹杨氏模量 Pa")
    poisson_ratio: float = Field(default=0.3, gt=0.0, lt=0.5, description="Newton 包裹泊松比")
    density: float = Field(default=70.0, gt=0.0, description="Newton 包裹等效密度 kg/m^3")
    particle_radius: float = Field(default=0.0015, gt=0.0, description="Newton 粒子接触半径 m")
    soft_contact_mu: float = Field(default=0.5, ge=0.0, description="Newton 软接触摩擦")
    shape_mu: float = Field(default=1.5, ge=0.0, description="Newton 刚体代理摩擦")
    newton_substeps: int = Field(default=10, ge=1, description="每物理帧 Newton 子步数")
    vbd_iterations: int = Field(default=10, ge=1, description="VBD 每子步迭代次数")
    min_lift_height_m: float = Field(
        default=0.08,
        ge=0.0,
        description="左右提起结束时包裹 COM 相对提起前至少升高这么多,否则 FAILED",
    )
    keep_label_up: bool = Field(
        default=False,
        description="左臂从主体中部抓取；平台松爪后须通过标签朝上检查，要求入带时标签已朝上",
    )
    label_uv_bounds: tuple[float, float, float, float] = Field(
        default=(0.14, 0.50, 0.32, 0.615),
        description="标签材质 UV 矩形 [u_min, v_min, u_max, v_max]；默认对应 bagged_clothing_000",
    )
    label_up_min_cos: float = Field(
        default=0.9, ge=0.0, le=1.0, description="平台交接时标签区域法线与世界 +Z 夹角的最小余弦"
    )
    left_lift_height_m: float = Field(
        default=0.28,
        gt=0.0,
        le=0.35,
        description="左臂起始提起高度；稳握后仍接触入带时最多补抬 0.12 m，总高度不超过 0.35 m",
    )
    left_setdown_offset_m: float = Field(
        default=0.025,
        ge=-0.05,
        le=0.05,
        description="平台下放额外向下偏移；标签朝上采集用 0，避免过度挤压翻折",
    )

    grasp_jitter_fraction_xy: tuple[float, float] = Field(
        default=(0.25, 0.3),
        description="抓点相对包裹半宽/半长的随机范围；[0, 0] 关闭，最多 0.5，保留主体抓取",
    )
    grasp_random_candidates: int = Field(default=12, ge=1, le=64, description="每侧抓点采样数")
    grasp_random_seed: int = Field(default=0, ge=0, description="抓点种子；0 每轮随机")
    grasp_min_distance_m: float = Field(
        default=0.01,
        ge=0.0,
        le=0.05,
        allow_inf_nan=False,
        description="随机抓点离中心、上一成功集抓点的最小 XY 距离(m)",
    )

    @field_validator("grasp_jitter_fraction_xy")
    @classmethod
    def _grasp_jitter_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        if any(not math.isfinite(v) or not 0.0 <= v <= 0.5 for v in value):
            raise ValueError("grasp_jitter_fraction_xy 必须是 [0, 0.5] 内的有限数")
        return value

    @field_validator("label_uv_bounds")
    @classmethod
    def _label_uv_rectangle(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        bounds = np.asarray(value)
        if not np.all(np.isfinite(bounds)) or np.any(bounds[:2] >= bounds[2:]):
            raise ValueError("label_uv_bounds 须为有限、严格递增的 UV 矩形")
        return value

    belt_speed_mps: float = Field(
        default=0.5, ge=0.0, description="松爪后出带运动学速度 m/s,沿世界 -X"
    )
    physics_device: str = Field(default="cuda:0", description="PhysX 设备;柔性体必须 GPU")
    physics_hz: float = Field(default=60.0, gt=0.0, description="物理频率,与 physics_dt 对齐")
    replay_cache_dir: str = Field(
        default="outputs/sorting_pro_m6/replay_cache",
        description="成功一次后落盘的 npz 回放目录",
    )
    drive_mode: Literal["callback", "manual"] = Field(
        default="callback",
        description=(
            "callback:注册 PHYSICS_POST_STEP 回调推进状态机(与原脚本一致);"
            "manual:由 Job 循环每步显式调用 step(dt)"
        ),
    )
    timing_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="覆盖 prom6_TimingConfig 的字段(秒),如 left_drag_seconds",
    )


_NEWTON_LOGGER = get_logger("NewtonSorting")

_FINGER_NAME_HINTS = (
    "inner_finger",
    "finger_pad",
    "pad_link",
    "outer_finger",
    "inner_knuckle",
)

_MIN_HALF_EXTENT_M = 0.004

_SLAB_MIN_HALF_Z_M = 0.010

_MAX_SLAB_HALF_XY_M = 6.0

_MAX_FINGER_HALF_M = 0.08

_MAX_FINGERS_PER_GRIPPER = 2

_MAX_KINEMATIC_SPEED_M_S = 2.5

_MAX_BAG_SPAN_M = 1.5

_DEFAULT_PAD_HALF_M = np.array([0.022, 0.010, 0.028], dtype=np.float64)

_JAW_OPEN_SEP_M = 0.040

_JAW_CLOSED_SEP_M = 0.011

_JAW_CLOSED_RAD = 0.8

_JAW_AXIS = 1  # ToolCenter 局部 Y → 开合;局部 Z → 压入袋面

_PAD_APPROACH_OFFSET_M = 0.024  # 沿 TCP +Z 探到袋底,形成托举面

_SUPPORT_SHAPE_MU = 0.12  # 平台/入带低摩擦,才能揭起袋角

_FINGER_SHAPE_MU = 3.0  # 指垫高摩擦

_BELT_SHAPE_MU = 1.2  # 出带中等摩擦,既能送走又不把袋尾粘死

_BELT_SUPPORT_GAP_M = 0.0  # 平台 -X 与出带 +X 对齐

_BELT_PLUS_X_EXTEND_M = 0.0  # 不加长;加长后袋尾压在出带上,右爪揭不起

RIGHT_PLATFORM_PAD_CLEARANCE_M = 0.003  # 右指垫最低点与平台顶面至少留 3 mm

RIGHT_PLATFORM_PENETRATION_TOLERANCE_M = 0.002  # 抬升前允许的软体数值穿透上限


@dataclass(frozen=True)
class NewtonSortingParams:
    """Newton VBD 材料、接触与步进;默认对齐 grasp_deformable 包裹 demo."""

    youngs_modulus: float = 10000.0
    poisson_ratio: float = 0.3
    density: float = 70.0
    particle_radius: float = 0.0015
    soft_contact_margin: float = 0.01
    soft_contact_ke: float = 2.0e6
    soft_contact_kd: float = 1.0e-7
    soft_contact_mu: float = 0.5
    shape_mu: float = 1.5
    k_damp: float = 1.0e-6
    physics_hz: float = 60.0
    sim_substeps: int = 10
    vbd_iterations: int = 10
    device: str = "cuda:0"
    belt_speed_mps: float = 0.5
    min_lift_height_m: float = 0.08


@dataclass
class _KinematicBody:
    """一个随 USD prim 更新的 Newton kinematic box."""

    body_id: int
    prim_path: str
    half_extents: np.ndarray
    local_center: np.ndarray
    prev_pos: np.ndarray | None = None
    prev_quat: np.ndarray | None = None
    kind: str = "finger"
    side: str = "left"
    jaw_sign: float = 0.0


def _lame(youngs: float, poisson: float) -> tuple[float, float]:
    mu = youngs / (2.0 * (1.0 + poisson))
    lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return float(mu), float(lam)


def jaw_separation_m(finger_joint_rad: float, closed_rad: float = _JAW_CLOSED_RAD) -> float:
    """finger_joint 0→张开,closed_rad→合拢;closed_rad 须是实际下发的合爪目标(含关节上限)."""
    closed = max(float(closed_rad), 1.0e-6)
    t = float(np.clip(float(finger_joint_rad) / closed, 0.0, 1.0))
    return (1.0 - t) * _JAW_OPEN_SEP_M + t * _JAW_CLOSED_SEP_M


def _tcp_frame_pad_half(measured: np.ndarray) -> np.ndarray:
    """把指尖 Mesh AABB 半长映射到 TCP:X 侧向、Y 开合、Z 压入;加厚加宽以便夹住袋角."""
    extents = np.sort(np.abs(np.asarray(measured, dtype=np.float64).reshape(3)))
    return np.array(
        [
            float(np.clip(max(extents[1], 0.018), 0.016, 0.028)),
            float(np.clip(max(extents[0], 0.010), 0.008, 0.012)),
            float(np.clip(max(extents[2], 0.024), 0.020, 0.035)),
        ],
        dtype=np.float64,
    )


def _jaw_local_center(jaw_sign: float, sep_m: float) -> np.ndarray:
    offset = np.zeros(3, dtype=np.float64)
    offset[_JAW_AXIS] = float(jaw_sign) * float(sep_m)
    offset[2] = _PAD_APPROACH_OFFSET_M
    return offset


def _point_aabb_distance(point: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> float:
    nearest = np.minimum(np.maximum(point, minimum), maximum)
    return float(np.linalg.norm(np.asarray(point, dtype=np.float64) - nearest))


def _newton_matrix_pose(matrix: Gf.Matrix4d) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray(matrix.ExtractTranslation(), dtype=np.float64).reshape(3)
    quat = matrix.ExtractRotationQuat()
    imag = quat.GetImaginary()
    xyzw = np.array([imag[0], imag[1], imag[2], quat.GetReal()], dtype=np.float64)
    norm = float(np.linalg.norm(xyzw))
    if not np.all(np.isfinite(xyzw)) or norm < 1e-12:
        xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        xyzw /= norm
    return translation, xyzw


def _rotate_wxyz(wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=np.float64).reshape(4)
    qvec = np.array([x, y, z], dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def _isaac_world_pose(
    prim_path: str, views: dict[str, object] | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """从 Isaac view/Fabric 读世界位姿;PhysX 驱动的关节 USD XformCache 往往是旧的.

    Returns:
        (position, xyzw) 或 None(view 尚未可用,回退 USD).
    """
    try:
        from isaacsim.core.experimental.prims import XformPrim
    except ImportError:
        return None
    view = None if views is None else views.get(prim_path)
    try:
        if view is None:
            view = XformPrim(prim_path)
            if views is not None:
                views[prim_path] = view
        positions, orientations = view.get_world_poses()
        if hasattr(positions, "numpy"):
            positions = positions.numpy()
        if hasattr(orientations, "numpy"):
            orientations = orientations.numpy()
        position = np.asarray(positions, dtype=np.float64).reshape(-1, 3)[0]
        wxyz = np.asarray(orientations, dtype=np.float64).reshape(-1, 4)[0]
    except Exception:  # noqa: BLE001  # Play 前 view 可能还没绑上
        return None
    norm = float(np.linalg.norm(wxyz))
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(wxyz)) or norm < 1e-12:
        return None
    wxyz = wxyz / norm
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64)
    return position, xyzw


def _prim_world_matrix(stage: Usd.Stage, path: str, cache: UsdGeom.XformCache) -> Gf.Matrix4d:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Newton 代理缺少 prim: {path}")
    return cache.GetLocalToWorldTransform(prim)


def _mesh_points_local(prim: Usd.Prim) -> np.ndarray:
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        raise RuntimeError(f"Mesh 没有顶点: {prim.GetPath()}")
    return np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)


def _set_mesh_local_points(mesh: UsdGeom.Mesh, local: np.ndarray) -> None:
    mesh.GetPointsAttr().Set([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in local])


_REST_MESH_LOCAL: dict[str, np.ndarray] = {}


def restore_cached_rest_mesh(stage: Usd.Stage, mesh_path: str) -> bool:
    """把袋表面写回首次四面体化时的静止局部顶点.

    上一集 Newton 会把 Mesh 写成变形后的形状;再做星形 tet 会失败.
    首次调用若尚无缓存,则把当前顶点记为静止形状,不改 Mesh.

    Args:
        stage: 当前 USD stage.
        mesh_path: 袋装包裹可视 Mesh 路径.

    Returns:
        True 表示写回了已有缓存;False 表示刚建立缓存.

    Raises:
        RuntimeError: prim 不是活动 Mesh.
    """
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"需要活动 Mesh: {mesh_path}")
    mesh = UsdGeom.Mesh(prim)
    cached = _REST_MESH_LOCAL.get(mesh_path)
    if cached is None:
        _REST_MESH_LOCAL[mesh_path] = _mesh_points_local(prim).copy()
        return False
    current = _mesh_points_local(prim)
    if current.shape != cached.shape or not np.allclose(current, cached, atol=1e-5):
        _set_mesh_local_points(mesh, cached)
        _NEWTON_LOGGER.info("已恢复袋表面静止网格", prim=mesh_path, n=int(cached.shape[0]))
    return True


def world_mesh_vertices(
    stage: Usd.Stage, mesh_path: str
) -> tuple[UsdGeom.Mesh, np.ndarray, Gf.Matrix4d]:
    """读取可视 Mesh 的世界坐标顶点,供四面体化."""
    restore_cached_rest_mesh(stage, mesh_path)
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"需要活动 Mesh: {mesh_path}")
    mesh = UsdGeom.Mesh(prim)
    cache = UsdGeom.XformCache()
    world = cache.GetLocalToWorldTransform(prim)
    local = _mesh_points_local(prim)
    vertices = np.array([world.Transform(Gf.Vec3d(*row)) for row in local], dtype=np.float64)
    return mesh, vertices, world


def star_tetmesh_from_surface(vertices: np.ndarray, face_indices: np.ndarray) -> tuple[object, int]:
    """封闭三角表面 → 中心扇形四面体.顶点须已是世界坐标,不做 recenter."""
    import newton

    verts = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(face_indices, dtype=np.int32).reshape(-1, 3)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 4:
        raise RuntimeError(f"无效表面顶点: {verts.shape}")
    if faces.shape[0] < 4:
        raise RuntimeError(f"表面三角太少: {faces.shape}")
    center = verts.mean(axis=0)
    relative = verts[faces] - center
    volumes = np.linalg.det(relative) / 6.0
    if not np.all(volumes > 1e-12):
        raise RuntimeError("袋表面相对中心不是正体积星形,无法做简化四面体化")
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
        axis=1,
    )
    counts = np.unique(edges, axis=0, return_counts=True)[1]
    if not np.all(counts == 2):
        raise RuntimeError("袋表面不是闭合流形(边共享次数不为 2)")
    tet_index = np.column_stack([np.full(len(faces), len(verts), dtype=np.int32), faces])
    tet = newton.TetMesh(np.vstack([verts, center.astype(np.float32)]), tet_index.flatten())
    return tet, int(len(verts))


def require_no_attachment_schemas(stage: Usd.Stage) -> None:
    """Stage 上若还有 Attachment schema,说明仍在走伪抓取."""
    leftover: list[str] = []
    for prim in stage.Traverse():
        type_name = prim.GetTypeName() or ""
        schemas = prim.GetAppliedSchemas()
        if "Attachment" in type_name or any("Attachment" in name for name in schemas):
            leftover.append(str(prim.GetPath()))
    if leftover:
        raise RuntimeError(f"仍存在柔性附着 prim,拒绝接触抓取: {leftover[:8]}")


def strip_clothing_physx(
    stage: Usd.Stage,
    *,
    clothing_root_path: str,
    clothing_surface_path: str,
    volume_proxy_path: str,
) -> None:
    """Play 前去掉袋子上的 PhysX 柔性/碰撞,只留可视 Mesh 给 Newton 写回."""
    volume = stage.GetPrimAtPath(volume_proxy_path)
    if volume.IsValid() and volume.IsActive():
        volume.SetActive(False)
        _NEWTON_LOGGER.info("已关掉体积代理", prim=volume_proxy_path)

    root = stage.GetPrimAtPath(clothing_root_path)
    if not root.IsValid():
        raise RuntimeError(f"缺少包裹根 prim: {clothing_root_path}")
    surface = stage.GetPrimAtPath(clothing_surface_path)
    if not surface.IsValid() or not surface.IsA(UsdGeom.Mesh) or not surface.IsActive():
        raise RuntimeError(f"包裹表面须是活动 Mesh: {clothing_surface_path}")

    removed: list[str] = []
    for prim in Usd.PrimRange(root):
        for schema in list(prim.GetAppliedSchemas()):
            if "Attachment" in schema:
                raise RuntimeError(f"包裹上已有附着 schema {schema}: {prim.GetPath()}")
            if "Deformable" not in schema:
                continue
            try:
                prim.RemoveAPI(schema)
                removed.append(f"{prim.GetPath()}:{schema}")
            except Exception as exc:  # noqa: BLE001  # 部分 schema 名要走 Registry
                api_type = Usd.SchemaRegistry.GetTypeFromSchemaTypeName(schema)
                if not api_type:
                    raise RuntimeError(
                        f"无法移除 deformable schema {schema} @ {prim.GetPath()}"
                    ) from exc
                prim.RemoveAPI(api_type)
                removed.append(f"{prim.GetPath()}:{schema}")
        collision_attr = prim.GetAttribute("physics:collisionEnabled")
        if prim.HasAPI(UsdPhysics.CollisionAPI) or collision_attr.IsValid():
            collision = UsdPhysics.CollisionAPI.Apply(prim)
            collision.CreateCollisionEnabledAttr().Set(False)

    _NEWTON_LOGGER.info(
        "已去掉袋子 PhysX 柔性", removed=len(removed), surface=clothing_surface_path
    )
    require_no_attachment_schemas(stage)


def discover_tool_centers(stage: Usd.Stage, robot_path: str) -> tuple[str, str]:
    """找随 Articulation 动的 ToolCenter;优先 Geometry 子树,避开根上的静态视觉夹爪."""
    left_candidates: list[str] = []
    right_candidates: list[str] = []
    prefix = robot_path.rstrip("/")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(prefix):
            continue
        name = prim.GetName()
        if name == "ToolCenter_L":
            left_candidates.append(path)
        elif name == "ToolCenter_R":
            right_candidates.append(path)

    def pick(candidates: list[str], label: str) -> str:
        if not candidates:
            raise RuntimeError(f"找不到 {label}")
        ranked = sorted(candidates, key=lambda p: (0 if "/Geometry/" in p else 1, len(p)))
        chosen = ranked[0]
        if len(candidates) > 1:
            _NEWTON_LOGGER.info("ToolCenter 候选", label=label, chosen=chosen, all=candidates)
        return chosen

    return pick(left_candidates, "ToolCenter_L"), pick(right_candidates, "ToolCenter_R")


def resolve_infeed_prim_path(stage: Usd.Stage, infeed_prim_path: str) -> str:
    """入带优先用 ``.../Belt`` 表面,避免整段机架 AABB 把袋子埋进盒子里."""
    if not infeed_prim_path:
        return ""
    belt_child = infeed_prim_path.rstrip("/") + "/Belt"
    if stage.GetPrimAtPath(belt_child).IsValid():
        return belt_child
    if stage.GetPrimAtPath(infeed_prim_path).IsValid():
        return infeed_prim_path
    return ""


def _prim_local_aabb(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    """指垫/指尖局部 AABB:扫子孙 Mesh(含 instance proxy),变回该 prim 局部坐标."""
    xcache = UsdGeom.XformCache()
    parent_world = xcache.GetLocalToWorldTransform(prim)
    inverse = parent_world.GetInverse()
    points: list[list[float]] = []
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not child.IsActive() or not child.IsA(UsdGeom.Mesh):
            continue
        try:
            local = _mesh_points_local(child)
        except RuntimeError:
            continue
        child_world = xcache.GetLocalToWorldTransform(child)
        to_parent = inverse * child_world
        for row in local:
            mapped = to_parent.Transform(Gf.Vec3d(float(row[0]), float(row[1]), float(row[2])))
            points.append([mapped[0], mapped[1], mapped[2]])
    if len(points) < 2:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        rng = cache.ComputeUntransformedBound(prim).GetRange()
        if rng.IsEmpty():
            return np.zeros(3, dtype=np.float64), _DEFAULT_PAD_HALF_M.copy()
        minimum = np.asarray(rng.GetMin(), dtype=np.float64).reshape(3)
        maximum = np.asarray(rng.GetMax(), dtype=np.float64).reshape(3)
        points = [minimum.tolist(), maximum.tolist()]
    center, half = _aabb_half_from_points(np.asarray(points, dtype=np.float64))
    if float(np.max(half)) > _MAX_FINGER_HALF_M:
        _NEWTON_LOGGER.warning(
            "指垫包围盒过大,已钳制", prim=str(prim.GetPath()), half=half.tolist()
        )
        half = np.minimum(half, _MAX_FINGER_HALF_M)
    if float(np.max(half)) < 0.006:
        half = np.maximum(half, _DEFAULT_PAD_HALF_M)
    return center, half


def _pick_opposing_fingers(paths: list[str]) -> list[str]:
    """每个夹爪取相对的两指:指垫(或 inner_finger) + outer_finger."""

    def leaf(path: str) -> str:
        return path.rsplit("/", 1)[-1].lower()

    pads = [
        path for path in paths if leaf(path).endswith("finger_pad") or leaf(path).endswith("pad")
    ]
    outers = [path for path in paths if leaf(path).endswith("outer_finger")]
    inners = [path for path in paths if leaf(path).endswith("inner_finger")]
    chosen: list[str] = []
    # 整段 inner/outer finger 比空的 pad Xform 更能包住指面网格
    if inners:
        chosen.append(inners[0])
    elif pads:
        chosen.append(pads[0])
    if outers:
        chosen.append(outers[0])
    elif pads:
        for path in pads:
            if path not in chosen:
                chosen.append(path)
                break
    return chosen[:_MAX_FINGERS_PER_GRIPPER]


def discover_finger_meshes(stage: Usd.Stage, gripper_path: str) -> list[str]:
    """在夹爪子树里找会随 finger_joint 动的一对指尖 prim.

    ToolCenter 在基座上,合爪时不夹紧,不能当接触代理.
    """
    root = stage.GetPrimAtPath(gripper_path)
    if not root.IsValid():
        raise RuntimeError(f"夹爪 prim 不存在: {gripper_path}")
    candidates: list[str] = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or not prim.IsA(UsdGeom.Xformable):
            continue
        path = str(prim.GetPath())
        lowered = path.lower()
        if "joint" in lowered or "toolcenter" in lowered or "visuals" in lowered:
            continue
        leaf = path.rsplit("/", 1)[-1].lower()
        if "knuckle" in leaf:
            continue
        if any(hint in lowered for hint in _FINGER_NAME_HINTS) or "finger" in lowered:
            candidates.append(path)
    chosen = _pick_opposing_fingers(candidates)
    if not chosen:
        raise RuntimeError(
            f"夹爪下没有合用的指垫/指尖 prim(不能用 ToolCenter 代替): {gripper_path}"
        )
    sizes = []
    for path in chosen:
        center, half = _prim_local_aabb(stage.GetPrimAtPath(path))
        sizes.append({"prim": path, "center": center.tolist(), "half": half.tolist()})
    _NEWTON_LOGGER.info("指垫代理", gripper=gripper_path, prims=chosen, sizes=sizes)
    return chosen


def _aabb_half_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    half = np.maximum(0.5 * (maximum - minimum), _MIN_HALF_EXTENT_M)
    center = 0.5 * (minimum + maximum)
    return center, half


def _static_slab_from_prim(stage: Usd.Stage, path: str) -> tuple[np.ndarray, np.ndarray]:
    """用世界 AABB 的顶面做薄板;避免整段机架体积把袋子埋进碰撞盒."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"缺少支撑/皮带 prim: {path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    world_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if world_range.IsEmpty():
        raise RuntimeError(f"prim 没有世界包围盒: {path}")
    minimum = np.asarray(world_range.GetMin(), dtype=np.float64).reshape(3)
    maximum = np.asarray(world_range.GetMax(), dtype=np.float64).reshape(3)
    half_xy = 0.5 * (maximum[:2] - minimum[:2])
    if np.any(half_xy > _MAX_SLAB_HALF_XY_M):
        raise RuntimeError(f"支撑/皮带 XY 过大,可能扫进了整厂: {path} half_xy={half_xy}")
    half_z = _SLAB_MIN_HALF_Z_M
    center = np.array(
        [
            0.5 * (minimum[0] + maximum[0]),
            0.5 * (minimum[1] + maximum[1]),
            float(maximum[2]) - half_z,
        ],
        dtype=np.float64,
    )
    half = np.array(
        [
            max(float(half_xy[0]), _MIN_HALF_EXTENT_M),
            max(float(half_xy[1]), _MIN_HALF_EXTENT_M),
            half_z,
        ],
        dtype=np.float64,
    )
    return center, half


def _extend_belt_plus_x(
    belt_center: np.ndarray,
    belt_half: np.ndarray,
    extra_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """把出带 +X 边缘向平台加长,短拖后袋尾仍落在传送带上."""
    extra = float(extra_m)
    if extra <= 0.0:
        return belt_center, belt_half
    center = belt_center.copy()
    half = belt_half.copy()
    min_x = float(center[0] - half[0])
    max_x = float(center[0] + half[0] + extra)
    center[0] = 0.5 * (min_x + max_x)
    half[0] = 0.5 * (max_x - min_x)
    return center, half


def _clip_support_away_from_belt(
    support_center: np.ndarray,
    support_half: np.ndarray,
    belt_center: np.ndarray,
    belt_half: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """缩短平台 -X,让静态薄板不再盖住出带接触面."""
    support_min_x = float(support_center[0] - support_half[0])
    support_max_x = float(support_center[0] + support_half[0])
    belt_max_x = float(belt_center[0] + belt_half[0])
    new_min = belt_max_x + _BELT_SUPPORT_GAP_M
    if support_min_x >= new_min:
        return support_center, support_half
    if new_min >= support_max_x - 0.10:
        raise RuntimeError(
            "平台与出带重叠过多,无法裁出支撑面: "
            f"support_x=[{support_min_x:.4f}, {support_max_x:.4f}], "
            f"belt_max_x={belt_max_x:.4f}"
        )
    center = support_center.copy()
    half = support_half.copy()
    half[0] = 0.5 * (support_max_x - new_min)
    center[0] = 0.5 * (new_min + support_max_x)
    return center, half


def _align_slab_top(
    source_center: np.ndarray,
    source_half: np.ndarray,
    target_center: np.ndarray,
    target_half: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把 target 顶面齐到 source,避免平台到出带有台阶."""
    top = float(source_center[2] + source_half[2])
    center = target_center.copy()
    half = target_half.copy()
    center[2] = top - float(half[2])
    return center, half


class NewtonSortingWorld:
    """一台 Newton 模型:软袋 + 平台/入带静态盒 + 出带/指垫 kinematic."""

    def __init__(
        self,
        stage: Usd.Stage,
        params: NewtonSortingParams,
        *,
        clothing_surface_path: str,
        support_prim_path: str,
        belt_prim_path: str,
        infeed_prim_path: str = "",
        finger_prim_paths: list[str] | None = None,
        fallback_finger_xforms: list[str] | None = None,
        tcp_jaw_prims: list[tuple[str, str]] | None = None,
        label_uv_bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        import newton
        import warp as wp
        from newton.solvers import SolverVBD

        self.stage = stage
        self._xform_views: dict[str, object] = {}
        self._tcp_pose_fns: dict[str, object] = {}
        self._use_isaac_pose = False
        self.params = params
        self.clothing_surface_path = clothing_surface_path
        self.belt_prim_path = belt_prim_path
        self.support_prim_path = support_prim_path
        wp.set_device(params.device)

        infeed_prim_path = resolve_infeed_prim_path(stage, infeed_prim_path)

        mesh, world_vertices, self._mesh_world = world_mesh_vertices(stage, clothing_surface_path)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        if counts.size == 0 or not np.all(counts == 3):
            raise RuntimeError(f"包裹表面必须是三角网格: {clothing_surface_path}")
        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        self.label_triangles = np.empty((0, 3), dtype=np.int64)
        if label_uv_bounds is not None:
            uv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st")
            if not uv or not uv.HasValue():
                raise RuntimeError("标签朝向检查需要包裹网格的 st UV")
            values = np.asarray(uv.ComputeFlattened(), dtype=float)
            if uv.GetInterpolation() == "vertex":
                values = values[faces]
            elif uv.GetInterpolation() != "faceVarying":
                raise RuntimeError("标签朝向检查只支持 vertex / faceVarying UV")
            self.label_triangles = uv_patch_triangles(faces, values, label_uv_bounds)
            if mesh.GetOrientationAttr().Get() == "leftHanded":
                self.label_triangles = self.label_triangles[:, [0, 2, 1]]
        tet, self.surface_count = star_tetmesh_from_surface(world_vertices, faces)
        span = world_vertices.max(axis=0) - world_vertices.min(axis=0)
        if np.any(span > 1.5) or np.any(span < 0.02):
            raise RuntimeError(
                f"包裹世界尺寸异常,拒绝四面体化: span={span}, center={world_vertices.mean(axis=0)}"
            )
        self._mesh = mesh
        self._world_to_local = self._mesh_world.GetInverse()

        k_mu, k_lambda = _lame(params.youngs_modulus, params.poisson_ratio)
        builder = newton.ModelBuilder(gravity=-9.81)
        builder.default_shape_cfg.mu = float(params.shape_mu)

        self._high_friction_shape_ids: list[int] = []
        support_center, support_half = _static_slab_from_prim(stage, support_prim_path)
        self.support_top_z = float(support_center[2] + support_half[2])
        belt_center, belt_half = _static_slab_from_prim(stage, belt_prim_path)
        belt_center, belt_half = _extend_belt_plus_x(belt_center, belt_half, _BELT_PLUS_X_EXTEND_M)
        # 不裁平台 -X:与出带重叠共面,避免立边钩住袋尾把软袋拉长.
        belt_center, belt_half = _align_slab_top(
            support_center, support_half, belt_center, belt_half
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(*support_center), wp.quat_identity()),
            hx=float(support_half[0]),
            hy=float(support_half[1]),
            hz=float(support_half[2]),
        )
        if infeed_prim_path and stage.GetPrimAtPath(infeed_prim_path).IsValid():
            in_center, in_half = _static_slab_from_prim(stage, infeed_prim_path)
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(*in_center), wp.quat_identity()),
                hx=float(in_half[0]),
                hy=float(in_half[1]),
                hz=float(in_half[2]),
            )

        self._belt_rest_pos = belt_center.copy()
        self._belt_rest_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        belt_id = builder.add_body(
            xform=wp.transform(wp.vec3(*belt_center), wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
            label="outfeed_belt",
        )
        belt_shape = builder.add_shape_box(
            belt_id,
            hx=float(belt_half[0]),
            hy=float(belt_half[1]),
            hz=float(belt_half[2]),
        )
        self._high_friction_shape_ids.append(int(belt_shape))
        self.belt_body_id = belt_id
        self._belt_shape_id = int(belt_shape)
        self.belt_speed_mps = 0.0

        self._kinematics: list[_KinematicBody] = []
        self._finger_grip_scale = {"left": 1.0, "right": 1.0}
        self._pad_retract_m = 0.0
        tcp_jaws = list(tcp_jaw_prims or [])
        if tcp_jaws:
            pad_half = _DEFAULT_PAD_HALF_M.copy()
            for mesh_path in finger_prim_paths or []:
                prim = stage.GetPrimAtPath(mesh_path)
                if not prim.IsValid():
                    continue
                try:
                    _, half = _prim_local_aabb(prim)
                except RuntimeError:
                    continue
                if float(np.max(half)) >= 0.006:
                    pad_half = _tcp_frame_pad_half(half)
                    break
            for path, side in tcp_jaws:
                if not stage.GetPrimAtPath(path).IsValid():
                    raise RuntimeError(f"TCP 指垫 prim 无效: {path}")
                for jaw_sign in (1.0, -1.0):
                    local_center = _jaw_local_center(jaw_sign, _JAW_OPEN_SEP_M)
                    body_id = builder.add_body(
                        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                        mass=0.0,
                        is_kinematic=True,
                        label=f"{side}_jaw_{int(jaw_sign)}",
                    )
                    jaw_shape = builder.add_shape_box(
                        body_id,
                        hx=float(pad_half[0]),
                        hy=float(pad_half[1]),
                        hz=float(pad_half[2]),
                    )
                    self._high_friction_shape_ids.append(int(jaw_shape))
                    self._kinematics.append(
                        _KinematicBody(
                            body_id=body_id,
                            prim_path=path,
                            half_extents=pad_half.copy(),
                            local_center=local_center,
                            side=side,
                            jaw_sign=jaw_sign,
                        )
                    )
        else:
            finger_paths = list(finger_prim_paths or [])
            if not finger_paths and fallback_finger_xforms:
                for path in fallback_finger_xforms:
                    if stage.GetPrimAtPath(path).IsValid():
                        finger_paths.append(path)
            if not finger_paths:
                raise RuntimeError("Newton 指垫代理为空:请配置 tcp_jaw_prims 或 finger prim")
            for path in finger_paths:
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise RuntimeError(f"指垫 prim 无效: {path}")
                try:
                    local_center, half = _prim_local_aabb(prim)
                except RuntimeError:
                    if prim.IsA(UsdGeom.Mesh):
                        local_pts = _mesh_points_local(prim)
                        local_center, half = _aabb_half_from_points(local_pts)
                    else:
                        local_center = np.zeros(3, dtype=np.float64)
                        half = _DEFAULT_PAD_HALF_M.copy()
                body_id = builder.add_body(
                    xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                    mass=0.0,
                    is_kinematic=True,
                    label=path.rsplit("/", 1)[-1],
                )
                finger_shape = builder.add_shape_box(
                    body_id,
                    hx=float(half[0]),
                    hy=float(half[1]),
                    hz=float(half[2]),
                )
                self._high_friction_shape_ids.append(int(finger_shape))
                self._kinematics.append(
                    _KinematicBody(
                        body_id=body_id,
                        prim_path=path,
                        half_extents=half,
                        local_center=local_center,
                    )
                )
        if not self._kinematics:
            raise RuntimeError("Newton 指垫代理为空")

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tet,
            density=float(params.density),
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=float(params.k_damp),
            particle_radius=float(params.particle_radius),
        )
        builder.color()
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = float(params.soft_contact_ke)
        self.model.soft_contact_kd = float(params.soft_contact_kd)
        self.model.soft_contact_mu = float(params.soft_contact_mu)
        self.model.shape_material_ke.fill_(float(params.soft_contact_ke))
        self.model.shape_material_kd.fill_(float(params.soft_contact_kd))
        mu = np.full(int(self.model.shape_count), _SUPPORT_SHAPE_MU, dtype=np.float32)
        for shape_id in self._high_friction_shape_ids:
            if 0 <= shape_id < mu.size:
                mu[shape_id] = float(max(params.shape_mu, _FINGER_SHAPE_MU))
        self._mu_template = mu
        self._assign_shape_mu()

        pinned = int(np.count_nonzero(self.model.particle_inv_mass.numpy() == 0))
        if pinned:
            raise RuntimeError(f"Newton 软体含 {pinned} 个固定粒子,接触抓取不允许 pin")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=float(params.soft_contact_margin),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.soft_solver = SolverVBD(
            self.model,
            iterations=int(params.vbd_iterations),
            integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=float(params.particle_radius) * 2.0,
            particle_self_contact_margin=float(params.soft_contact_margin),
            particle_enable_self_contact=False,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
        )
        self.frame_dt = 1.0 / float(params.physics_hz)
        self.sim_dt = self.frame_dt / float(params.sim_substeps)
        self._wp = wp
        self._sync_kinematics(self.frame_dt, first=True, update_history=True)
        self.write_surface_to_usd()
        self._assert_bag_sane()
        cache = UsdGeom.XformCache()
        finger_world = [self._body_transform(item, cache)[0].tolist() for item in self._kinematics]
        _NEWTON_LOGGER.info(
            "Newton 分拣世界已建",
            particles=int(self.model.particle_count),
            surface=self.surface_count,
            fingers=len(self._kinematics),
            bodies=int(self.model.body_count),
            bag_span=span.tolist(),
            support_half=support_half.tolist(),
            belt_half=belt_half.tolist(),
            finger_half=[item.half_extents.tolist() for item in self._kinematics],
            finger_local_center=[item.local_center.tolist() for item in self._kinematics],
            finger_world=finger_world,
            high_friction_shapes=self._high_friction_shape_ids,
            support_mu=_SUPPORT_SHAPE_MU,
        )

    def set_belt_speed(self, speed_mps: float) -> None:
        """出带 treadmill 速度,沿世界 -X;0 表示静止.静止时摩擦与平台相同,才能揭起袋尾."""
        self.belt_speed_mps = float(speed_mps)
        self._assign_shape_mu()

    def set_finger_grip_scale(self, left_t: float, right_t: float) -> None:
        """合爪比例 0→张开(低摩擦脱落) 1→夹紧(高摩擦承重)."""
        self._finger_grip_scale = {
            "left": float(np.clip(left_t, 0.0, 1.0)),
            "right": float(np.clip(right_t, 0.0, 1.0)),
        }
        self._assign_shape_mu()

    def _assign_shape_mu(self) -> None:
        mu = np.array(self._mu_template, dtype=np.float32, copy=True)
        if 0 <= int(self._belt_shape_id) < mu.size:
            if abs(self.belt_speed_mps) > 1.0e-8:
                mu[int(self._belt_shape_id)] = float(max(self.params.shape_mu, _BELT_SHAPE_MU))
            else:
                mu[int(self._belt_shape_id)] = _SUPPORT_SHAPE_MU
        body_of = np.asarray(self.model.shape_body.numpy(), dtype=np.int32).reshape(-1)
        for item in self._kinematics:
            scale = 0.02 + 0.98 * self._finger_grip_scale.get(item.side, 1.0)
            for shape_id, body_id in enumerate(body_of.tolist()):
                if int(body_id) == int(item.body_id) and 0 <= shape_id < mu.size:
                    mu[shape_id] = np.float32(float(self._mu_template[shape_id]) * scale)
        self.model.shape_material_mu.assign(mu)

    def set_pad_retract(self, retract_m: float) -> None:
        """松爪时把指垫沿 TCP -Z 抽回,退出袋体再让软袋落台.

        同时清掉有限差分历史:否则 ``step`` 会把 30cm 抽离插值成扫掠,把袋拉走.
        """
        self._pad_retract_m = float(max(0.0, retract_m))
        for item in self._kinematics:
            if item.jaw_sign == 0.0:
                continue
            center = np.asarray(item.local_center, dtype=np.float64).copy()
            center[2] = _PAD_APPROACH_OFFSET_M - self._pad_retract_m
            item.local_center = center
            item.prev_pos = None
            item.prev_quat = None

    def set_jaw_separations(self, left_m: float, right_m: float) -> None:
        """按左右 finger_joint 更新 TCP 指垫局部偏移,让合爪真正夹紧."""
        seps = {"left": float(left_m), "right": float(right_m)}
        for item in self._kinematics:
            if item.jaw_sign == 0.0:
                continue
            center = _jaw_local_center(item.jaw_sign, seps.get(item.side, _JAW_OPEN_SEP_M))
            center = center.copy()
            center[2] -= self._pad_retract_m
            item.local_center = center

    def set_tcp_pose_fn(self, side: str, fn: object) -> None:
        """用规划器 FK 的 TCP 驱动指垫;USD 视觉夹爪不在 PhysX Fabric 里."""
        self._tcp_pose_fns[side] = fn

    def top_down_tcp_floor_z(self, side: str, clearance_m: float = 0.0) -> float:
        """返回顶抓姿态下不让该侧指垫侵入平台所需的最低 TCP 世界 Z。"""
        pads = [item for item in self._kinematics if item.side == side and item.jaw_sign != 0.0]
        if not pads:
            raise RuntimeError(f"{side} 指垫代理不存在，无法计算平台安全高度")
        clearance = float(clearance_m)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError("指垫平台间隙必须是非负有限值")
        # 固定顶抓四元数把 TCP 局部 +Z 映射到世界 -Z；因此
        # local_center.z + half.z 是 TCP 到指垫最低点的距离。
        tcp_to_lowest = max(float(item.local_center[2] + item.half_extents[2]) for item in pads)
        return float(self.support_top_z + tcp_to_lowest + clearance)

    def _body_transform(self, item: _KinematicBody, cache: UsdGeom.XformCache):
        fn = self._tcp_pose_fns.get(item.side)
        if fn is not None and item.jaw_sign != 0.0:
            position, wxyz = fn()  # type: ignore[operator]
            wxyz = np.asarray(wxyz, dtype=np.float64).reshape(4)
            xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64)
            offset = _rotate_wxyz(wxyz, item.local_center)
            return position + offset, xyzw
        prim_path = item.prim_path
        local_center = item.local_center
        if self._use_isaac_pose:
            live = _isaac_world_pose(prim_path, self._xform_views)
            if live is not None:
                position, xyzw = live
                offset = _rotate_wxyz(
                    np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64),
                    local_center,
                )
                return position + offset, xyzw
            if prim_path not in self._xform_views:
                _NEWTON_LOGGER.warning("Isaac 指垫位姿不可用,回退 USD", prim=prim_path)
        world = _prim_world_matrix(self.stage, prim_path, cache)
        center_world = world.Transform(Gf.Vec3d(*local_center))
        pos = np.array([center_world[0], center_world[1], center_world[2]], dtype=np.float64)
        _, quat = _newton_matrix_pose(world)
        return pos, quat

    def reset_kinematic_history(self) -> None:
        """机器人跳到 ready 后清零有限差分速度,避免指垫瞬移打飞袋子."""
        self._xform_views.clear()
        self._use_isaac_pose = True
        for item in self._kinematics:
            item.prev_pos = None
            item.prev_quat = None
        self._sync_kinematics(self.frame_dt, first=True, update_history=True)

    def _empty_body_state(self, sim_dt: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (q_prev, q_curr, qd).

        SolverVBD 在 ``integrate_with_external_rigid_solver=True`` 时用
        ``(body_q_out - body_q_in) / dt`` 当接触点速度,忽略 ``body_qd``.
        出带碰撞盒钉在 rest,把 ``q_prev`` 沿 +X 错开 ``speed * sim_dt``,
        形成世界 -X 的 treadmill.
        """
        nbody = int(self.model.body_count)
        q_prev = np.zeros((nbody, 7), dtype=np.float32)
        q_curr = np.zeros((nbody, 7), dtype=np.float32)
        qd = np.zeros((nbody, 6), dtype=np.float32)
        if nbody:
            q_prev[:, 6] = 1.0
            q_curr[:, 6] = 1.0
        rest = np.array([*self._belt_rest_pos, *self._belt_rest_quat], dtype=np.float32)
        q_curr[self.belt_body_id] = rest
        q_prev[self.belt_body_id] = rest
        speed = abs(float(self.belt_speed_mps))
        dt = float(sim_dt)
        if speed > 0.0 and dt > 1.0e-8:
            q_prev[self.belt_body_id, 0] = np.float32(self._belt_rest_pos[0] + speed * dt)
            qd[self.belt_body_id, 0] = np.float32(-speed)
        return q_prev, q_curr, qd

    def _assign_bodies(self, q: np.ndarray, qd: np.ndarray) -> None:
        q_wp = self._wp.array(q, dtype=self._wp.transform)
        qd_wp = self._wp.array(qd, dtype=self._wp.spatial_vector)
        self.state_0.body_q.assign(q_wp)
        self.state_0.body_qd.assign(qd_wp)
        self.state_1.body_q.assign(q_wp)
        self.state_1.body_qd.assign(qd_wp)

    @staticmethod
    def _clamped_velocity(pos: np.ndarray, prev: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 1e-8:
            return np.zeros(3, dtype=np.float64)
        vel = (pos - prev) / dt
        speed = float(np.linalg.norm(vel))
        if speed > _MAX_KINEMATIC_SPEED_M_S:
            vel = vel * (_MAX_KINEMATIC_SPEED_M_S / speed)
        return vel

    def _sync_kinematics(self, dt: float, *, first: bool, update_history: bool) -> None:
        cache = UsdGeom.XformCache()
        _q_prev, q, qd = self._empty_body_state(sim_dt=0.0)
        for item in self._kinematics:
            pos, quat = self._body_transform(item, cache)
            q[item.body_id] = np.array([*pos, *quat], dtype=np.float32)
            if not first and item.prev_pos is not None:
                vel = self._clamped_velocity(pos, item.prev_pos, dt)
                qd[item.body_id, :3] = vel.astype(np.float32)
            if update_history:
                item.prev_pos = pos.copy()
                item.prev_quat = quat.copy()
        self._assign_bodies(q, qd)

    def step(self, dt: float) -> None:
        """同步 kinematic 代理,跑若干 VBD 子步,写回可视网格."""
        frame_dt = float(dt) if dt > 1e-8 else self.frame_dt
        cache = UsdGeom.XformCache()
        samples: list[
            tuple[_KinematicBody, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for item in self._kinematics:
            pos, quat = self._body_transform(item, cache)
            prev_pos = item.prev_pos if item.prev_pos is not None else pos
            prev_quat = item.prev_quat if item.prev_quat is not None else quat
            jump = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - prev_pos))
            if jump > 0.08:
                prev_pos = pos.copy()
                prev_quat = quat.copy()
                vel = np.zeros(3, dtype=np.float64)
            else:
                vel = self._clamped_velocity(pos, prev_pos, frame_dt)
            samples.append((item, prev_pos, pos, prev_quat, quat, vel))

        self.soft_solver.rebuild_bvh(self.state_0)
        substeps = max(1, int(round(frame_dt / self.sim_dt)))
        sim_dt = frame_dt / substeps
        for index in range(substeps):
            alpha_in = index / substeps
            alpha_out = (index + 1) / substeps
            q_in, q_out, qd = self._empty_body_state(sim_dt=sim_dt)
            for item, prev_pos, pos, prev_quat, quat, vel in samples:
                pos_in = (1.0 - alpha_in) * prev_pos + alpha_in * pos
                pos_out = (1.0 - alpha_out) * prev_pos + alpha_out * pos
                quat_in = prev_quat if alpha_in < 1.0 else quat
                q_in[item.body_id] = np.array([*pos_in, *quat_in], dtype=np.float32)
                q_out[item.body_id] = np.array([*pos_out, *quat], dtype=np.float32)
                qd[item.body_id, :3] = vel.astype(np.float32)
            self.state_0.body_q.assign(self._wp.array(q_in, dtype=self._wp.transform))
            self.state_0.body_qd.assign(self._wp.array(qd, dtype=self._wp.spatial_vector))
            self.state_1.body_q.assign(self._wp.array(q_out, dtype=self._wp.transform))
            self.state_1.body_qd.assign(self._wp.array(qd, dtype=self._wp.spatial_vector))
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.soft_solver.step(self.state_0, self.state_1, self.control, self.contacts, sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.state_0.body_q.assign(self._wp.array(q_out, dtype=self._wp.transform))
            self.state_0.body_qd.assign(self._wp.array(qd, dtype=self._wp.spatial_vector))
        for item, _, pos, _, quat, _ in samples:
            item.prev_pos = pos.copy()
            item.prev_quat = quat.copy()
        self._assert_bag_sane()
        self.write_surface_to_usd()

    def _assert_bag_sane(self) -> None:
        nodes = self.particle_positions()
        if nodes.size == 0 or not np.all(np.isfinite(nodes)):
            raise RuntimeError("Newton 粒子位置无效")
        span = nodes.max(axis=0) - nodes.min(axis=0)
        center = nodes.mean(axis=0)
        if np.any(span > _MAX_BAG_SPAN_M) or float(center[2]) < 0.5:
            raise RuntimeError(f"Newton 袋子不稳定: span={span}, center={center}")

    def particle_positions(self) -> np.ndarray:
        """全部仿真节点世界坐标,含内部中心."""
        return np.asarray(self.state_0.particle_q.numpy(), dtype=np.float64).reshape(-1, 3)

    def particle_velocities(self) -> np.ndarray:
        return np.asarray(self.state_0.particle_qd.numpy(), dtype=np.float64).reshape(-1, 3)

    def surface_positions(self) -> np.ndarray:
        return self.particle_positions()[: self.surface_count]

    def com(self) -> np.ndarray:
        nodes = self.particle_positions()
        return nodes.mean(axis=0)

    def aabb(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nodes = self.surface_positions()
        minimum = nodes.min(axis=0)
        maximum = nodes.max(axis=0)
        return minimum, maximum, 0.5 * (minimum + maximum)

    def soft_contact_count(self) -> int:
        """当前粒子-形状接触数,抓取调参用."""
        count = self.contacts.soft_contact_count
        if hasattr(count, "numpy"):
            count = count.numpy()
        return int(np.asarray(count).reshape(-1)[0])

    def contact_breakdown(self) -> dict[str, int]:
        """把软接触按世界支撑 / 出带 / 指垫拆开,判断是夹住了还是只坐在皮带上."""
        total = self.soft_contact_count()
        counts = {"total": total, "world": 0, "belt": 0, "finger": 0}
        if total <= 0:
            return counts
        shapes = np.asarray(self.contacts.soft_contact_shape.numpy(), dtype=np.int32).reshape(-1)[
            :total
        ]
        body_of = np.asarray(self.model.shape_body.numpy(), dtype=np.int32).reshape(-1)
        finger_ids = {item.body_id for item in self._kinematics}
        for shape_id in shapes.tolist():
            if shape_id < 0 or shape_id >= len(body_of):
                continue
            body_id = int(body_of[shape_id])
            if body_id < 0:
                counts["world"] += 1
            elif body_id == int(self.belt_body_id):
                counts["belt"] += 1
            elif body_id in finger_ids:
                counts["finger"] += 1
        return counts

    def finger_debug(self) -> list[dict[str, object]]:
        """抓取验收用:指垫世界位姿相对袋子 AABB."""
        cache = UsdGeom.XformCache()
        bag_min, bag_max, bag_center = self.aabb()
        rows: list[dict[str, object]] = [
            {
                "bag_min": bag_min.tolist(),
                "bag_max": bag_max.tolist(),
                "bag_center": bag_center.tolist(),
                "contacts": self.contact_breakdown(),
            }
        ]
        for item in self._kinematics:
            pos, _ = self._body_transform(item, cache)
            rows.append(
                {
                    "prim": item.prim_path.rsplit("/", 1)[-1],
                    "side": item.side,
                    "pos": np.round(pos, 4).tolist(),
                    "half": np.round(item.half_extents, 4).tolist(),
                    "local": np.round(item.local_center, 4).tolist(),
                    "dist_aabb": round(_point_aabb_distance(pos, bag_min, bag_max), 4),
                    "dist_center": round(float(np.linalg.norm(pos - bag_center)), 4),
                }
            )
        return rows

    def write_surface_to_usd(self) -> None:
        """把 Newton 表面节点变回 Mesh 局部坐标,保留 UV/材质."""
        nodes = self.surface_positions()
        local = [
            self._world_to_local.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            for p in nodes
        ]
        self._mesh.GetPointsAttr().Set([Gf.Vec3f(p[0], p[1], p[2]) for p in local])


_LOGGER = get_logger("ProM6Relay")


def _log(*parts: object) -> None:
    """把原脚本的 ``print`` 转成一条 structlog 记录.

    Args:
        *parts: 原 ``print`` 的位置参数,按空格拼成一行.

    Returns:
        None.
    """
    _LOGGER.info(" ".join(str(part) for part in parts))


ROBOT_PATH = "/World/Pro_m6_696_dynamics_fixed"

CLOTHING_ROOT_PATH = "/World/ImportedBaggedClothing"

CLOTHING_SURFACE_PATH = "/World/ImportedBaggedClothing/Asset/visual/visual"

CLOTHING_VOLUME_PROXY_PATH = "/World/ImportedBaggedClothing/Asset/visual/visual_proxy"

SUPPORT_PATH = "/World/platform/Empty/Plane/Plane_002"

BELT_PATH = "/World/ConveyorBelt_A06/Belt"

LEFT_GRIPPER_PATH = f"{ROBOT_PATH}/robotiq_left"

RIGHT_GRIPPER_PATH = f"{ROBOT_PATH}/robotiq_right"

RUNTIME_ROOT_PATH = "/World/ProM6_LeftClothingDragRuntime"

ANCHOR_PATH = f"{RUNTIME_ROOT_PATH}/ClothingGraspFrame"

ATTACHMENT_SCOPE_PATH = f"{RUNTIME_ROOT_PATH}/Attachments"

ATTACHMENT_PATH = f"{ATTACHMENT_SCOPE_PATH}/Surface"

REPLAY_CACHE_SCHEMA_VERSION = 1

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

REPLAY_CACHE_DIRECTORY = Path(
    os.environ.get(
        "PRO_M6_REPLAY_CACHE_DIRECTORY",
        str(_SCRIPT_DIR / "pro_m6_replay_cache"),
    )
)

REPLAY_CACHE_FILE_PREFIX = "pro_m6_clothing_drag_"

PHYSICS_DEVICE = "cuda:0"

GPU_COLLISION_STACK_SIZE_MIB = 192

NOMINAL_PHYSICS_HZ = 60.0

GRASP_EDGE_INSET_M = 0.022

LEFT_GRASP_FROM_CENTER_FRAC = 0.20  # 左抓 X 从袋心往 -X 偏,咬袋体而不是 +X 薄边

LEFT_GRASP_Y_INSET_M = 0.045  # 从袋 +Y 边往里咬实体;贴边会推走,再往里常是空腔

LEFT_GRASP_BITE_M = 0.008  # 相对 AABB 顶再往下咬,合爪时袋会瘪

LEFT_MIN_FINGER_CONTACTS = 40  # 入带袋更鼓,接触峰值约 50;80 会过压把袋捅穿

LEFT_LIFT_MAX_WORLD_CONTACTS = 10  # 提起后袋身仍大量贴入带,说明只掐到边

LEFT_LIFT_CLEARANCE_HOLD_S = 0.25

LEFT_LIFT_CLEARANCE_STEP_M = 0.02

LEFT_LIFT_MAX_EXTRA_HEIGHT_M = 0.12  # 偏心抓取时给悬垂袋尾留足离带高度;逐步检查 IK

LEFT_LIFT_MAX_HEIGHT_M = 0.35

LEFT_GRASP_DIP_STEP_M = 0.005

LEFT_GRASP_DIP_MAX_M = 0.020

RIGHT_GRASP_EDGE_INSET_M = 0.035  # 右爪从 -X 边再往里探,避开变形后的空边

RIGHT_GRASP_BITE_M = 0.008  # 相对当前 AABB 顶再往下咬,合爪时袋会瘪

RIGHT_MIN_FINGER_CONTACTS = 80  # 右提之前 Newton 指垫接触数;不够就继续下压

RIGHT_GRASP_DIP_STEP_M = 0.005

RIGHT_GRASP_DIP_MAX_M = 0.020

GRIPPER_SURFACE_OFFSET_M = -0.014

GRASP_MASK_SIZE_M = np.array([0.080, 0.080, 0.030], dtype=np.float64)

PREGRASP_CLEARANCE_M = 0.080

LOW_LIFT_HEIGHT_M = 0.28

PLACEMENT_CLEARANCE_M = 0.002

LEFT_LOWER_DISTANCE_OFFSET_M = 0.025  # Meters: positive lowers farther; negative lowers less.

LEFT_RETRACT_HEIGHT_M = 0.28  # 松爪后先抬高再回 ready,避免关节插值把袋带走

LEFT_RETRACT_HOLD_S = 1.2  # 张开后降摩擦再停一拍,让袋从指垫脱落

LEFT_RETURN_CLEAR_X_OFFSET_M = 0.40  # 只往 +X 挪开;改 Y 会把袋拖出皮带 Y 范围

LEFT_PAD_RETRACT_M = 0.30  # 松爪时指垫沿 TCP -Z 瞬移抽离,避免 VBD 扫掠把袋拉走

PAD_RETRACT_HOLD_FRAMES = 6

LEFT_PAD_CLEAR_GAP_M = 0.04  # 指垫距袋 AABB 超过此值即视为已脱离,不再等接触归零

LEFT_READY_JOINTS = np.array(
    # [-0.05, 0.65, 0.65, -1.25, -0.90, 0.30, -0.35],
    [0.10, 0.10, 0.10, -1.50, 0.0, 0.0, -1.30],
    dtype=np.float64,
)

RIGHT_READY_JOINTS = np.array(
    # [0.05, 0.65, -0.65, -1.25, -0.90, 0.30, 0.35],
    [-0.10, 0.10, -0.10, -1.50, 0.0, 0.0, 1.30],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TimingConfig:
    """User-editable durations and safety timeouts, all expressed in seconds."""

    settle_seconds: float = 1.0
    waiting_seconds: float = 1.0
    left_move_pregrasp_seconds: float = 2.5
    left_approach_seconds: float = 2.5
    left_close_seconds: float = 2.5
    left_grip_settle_seconds: float = 1.5
    left_lift_seconds: float = 0.5
    left_drag_seconds: float = 6.0
    left_lower_seconds: float = 2.5
    left_place_settle_seconds: float = 2.0
    left_open_seconds: float = 3.0
    left_return_ready_seconds: float = 1.0
    right_move_pregrasp_seconds: float = 2.5
    right_approach_seconds: float = 2.5
    right_close_seconds: float = 2.5
    right_grip_settle_seconds: float = 1.5
    right_lift_seconds: float = 0.5
    right_release_seconds: float = 3.0
    right_return_ready_seconds: float = 2.5
    left_grip_stable_seconds: float = 1.0 / 6.0
    right_grip_stable_seconds: float = 1.0 / 6.0
    center_stable_seconds: float = 0.5
    place_stable_seconds: float = 0.5
    belt_entry_stable_seconds: float = 0.5
    final_verify_stable_seconds: float = 0.5
    ik_search_timeout_seconds: float = 30.0
    center_correction_timeout_seconds: float = 12.0
    placement_settle_timeout_seconds: float = 8.0
    handoff_prepare_timeout_seconds: float = 5.0
    right_drag_timeout_seconds: float = 30.0
    final_verify_timeout_seconds: float = 20.0

    def validate(self) -> TimingConfig:
        timeout_fields = {
            "ik_search_timeout_seconds",
            "center_correction_timeout_seconds",
            "placement_settle_timeout_seconds",
            "handoff_prepare_timeout_seconds",
            "right_drag_timeout_seconds",
            "final_verify_timeout_seconds",
        }
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"TimingConfig.{name} must be finite, got {value!r}")
            if name in timeout_fields and value <= 0.0:
                raise ValueError(f"TimingConfig.{name} must be > 0, got {value!r}")
            if name not in timeout_fields and value < 0.0:
                raise ValueError(f"TimingConfig.{name} must be >= 0, got {value!r}")
        return self


TASK_TIMING = TimingConfig(
    settle_seconds=2.0,
    waiting_seconds=1.0,
    left_move_pregrasp_seconds=1.0,
    left_approach_seconds=1.0,
    left_close_seconds=1.0,
    left_grip_settle_seconds=1.0,
    left_lift_seconds=3.0,
    left_drag_seconds=3.0,
    left_lower_seconds=1.0,
    left_place_settle_seconds=1.0,
    left_open_seconds=1.0,
    left_return_ready_seconds=0.8,
    right_move_pregrasp_seconds=1.0,
    right_approach_seconds=1.0,
    right_close_seconds=1.0,
    right_grip_settle_seconds=1.0,
    right_lift_seconds=3.0,
    right_release_seconds=1.0,
    right_return_ready_seconds=0.8,
    left_grip_stable_seconds=1.0 / 6.0,
    right_grip_stable_seconds=1.0 / 6.0,
    center_stable_seconds=0.5,
    place_stable_seconds=0.5,
    belt_entry_stable_seconds=0.5,
    final_verify_stable_seconds=0.5,
    placement_settle_timeout_seconds=14.0,
    final_verify_timeout_seconds=20.0,
)

LEFT_MAX_JOINT_SPEED_RAD_S = 0.9

RIGHT_MAX_JOINT_SPEED_RAD_S = 0.9

RIGHT_LIFT_HEIGHT_M = 0.28  # Independent world-Z lift; finite >= 0.

RIGHT_DRAG_LOWER_DISTANCE_M = 0.02  # 短拖时略降,右臂 IK 比悬空 28cm 拖更稳

RIGHT_DRAG_SPEED_M_S = 0.140

RIGHT_MAX_DRAG_DISTANCE_M = 0.56

BELT_ENTRY_INSET_M = 0.010

BELT_TAIL_INSET_M = 0.050  # 袋 +X 尾须进入出带这么深才松爪

RELEASE_CONVEYOR_SPEED_MPS = 0.5

RELEASE_CONVEYOR_DIRECTION_XY = np.array([-1.0, 0.0], dtype=np.float64)

RELEASE_CONVEYOR_STOP_XY_REGION: dict[str, float] | None = None

RELEASE_CONVEYOR_STOP_ON_ARRIVAL = False

RELEASE_CONVEYOR_ZERO_XY_ON_EXIT = False

GRIPPER_OPEN_POSITION_RAD = 0.0

GRIPPER_CLOSED_POSITION_RAD = 0.8

GRIPPER_POSITION_TOLERANCE_RAD = 0.050

CENTER_CORRECTION_STEP_M = 0.001

CENTER_RELEASE_TOLERANCE_M = 0.050

MAX_CENTER_CORRECTION_M = 0.500

TCP_APPROACH_POSITION_TOLERANCE_M = 0.003

TCP_APPROACH_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)

TCP_GRASP_POSITION_TOLERANCE_M = 0.004

TCP_GRASP_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)

TCP_LIFT_POSITION_TOLERANCE_M = 0.025

LEFT_RETURN_STALL_HOLD_S = 0.5

TCP_LIFT_ORIENTATION_TOLERANCE_RAD = math.radians(20.0)

DRAG_ENDPOINT_TOLERANCE_M = 0.010

PROGRESS_LOG_INTERVAL_FRAMES = 60

MAX_DEFORMABLE_SPAN_M = 1.5

IK_RANDOM_SEED = 696

IK_RANDOM_SEED_COUNT = 40

IK_CANDIDATE_COUNT = IK_RANDOM_SEED_COUNT + 2

IK_MAX_ITERATIONS = 900

IK_DAMPING = 0.035

ONLINE_DLS_DAMPING = 0.06

IK_MAX_JOINT_STEP_RAD = 0.035

IK_ELBOW_OUTBOARD_MIN_M = 0.12

IK_ELBOW_IN_PENALTY = 25.0

IK_JOINT1_WRAP_SCORE_WEIGHT = 0.4

IK_POSTURE_SCORE_WEIGHT = 0.08

_SHOULDER_JOINT_INDEX = 1  # Joint2:上臂根,用来量肘外展

_ELBOW_JOINT_INDEX = 3  # Joint4:肘

LEFT_ELBOW_OUTWARD_XY = np.array([1.0, 0.0], dtype=np.float64)  # 左臂外侧:世界 +X

RIGHT_ELBOW_OUTWARD_XY = np.array([-1.0, 0.0], dtype=np.float64)  # 右臂外侧:世界 -X

IK_INNER_POSITION_TOLERANCE_M = 2.0e-5

IK_INNER_ORIENTATION_TOLERANCE_RAD = 2.0e-4

IK_FINAL_POSITION_TOLERANCE_M = 0.002

IK_FINAL_ORIENTATION_TOLERANCE_RAD = math.radians(1.0)

IK_MODEL_POSITION_TOLERANCE_M = 0.001

IK_MODEL_ORIENTATION_TOLERANCE_RAD = math.radians(0.5)

IK_MODEL_JACOBIAN_TOLERANCE = 5.0e-3

IK_JOINT_LIMIT_MARGIN_RAD = 0.0

IK_PREFERRED_SEED = np.array(
    [-0.09, 1.367, 0.861, -2.069, -2.147, 0.482, -0.761],
    dtype=np.float64,
)

RIGHT_IK_PREFERRED_SEED = IK_PREFERRED_SEED * np.array(
    [-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0],
    dtype=np.float64,
)

LEFT_IK_ELBOW_OUT_SEED = np.array(
    [0.20, 1.15, 1.25, -1.75, -1.60, 0.40, -0.80],
    dtype=np.float64,
)

RIGHT_IK_ELBOW_OUT_SEED = LEFT_IK_ELBOW_OUT_SEED * np.array(
    [-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0],
    dtype=np.float64,
)

CALLBACK_GLOBAL_NAME = "_pro_m6_left_clothing_drag_callback_id"

STOP_CALLBACK_GLOBAL_NAME = "_pro_m6_left_clothing_drag_stop_callback_id"

CONTROLLER_GLOBAL_NAME = "_pro_m6_left_clothing_drag_controller"

OWN_CALLBACK_GLOBAL_NAMES = (
    CALLBACK_GLOBAL_NAME,
    STOP_CALLBACK_GLOBAL_NAME,
)


def apply_settings(
    *,
    robot_path: str = "",
    clothing_root_path: str = "",
    clothing_surface_path: str = "",
    clothing_volume_proxy_path: str = "",
    support_path: str = "",
    belt_path: str = "",
    left_gripper_path: str = "",
    right_gripper_path: str = "",
    runtime_root_path: str = "",
    replay_cache_dir: str = "",
    physics_device: str = "",
    physics_hz: float = 0.0,
) -> None:
    """用 ``[planner]`` 配置改写本模块的场景常量;空值保留原脚本默认.

    原脚本把 prim 路径写死在模块级,状态机各处直接引用.迁移保留这一写法,
    只在装配阶段由 ``SortingPlanner`` 调用本函数改写一次,避免逐处改签名.

    Args:
        robot_path: 机器人根 prim.
        clothing_root_path: 柔性包裹根 prim.
        clothing_surface_path: 柔性面片 Mesh(须为 Surface Deformable).
        clothing_volume_proxy_path: 体积代理 prim;必须处于 inactive.
        support_path: 平台支撑面 prim.
        belt_path: 出带皮带 prim.
        left_gripper_path: 左夹爪 prim;空则取 ``<robot>/robotiq_left``.
        right_gripper_path: 右夹爪 prim;空则取 ``<robot>/robotiq_right``.
        runtime_root_path: 运行期临时 prim 根(锚点 / 附着).
        replay_cache_dir: 成功回放 npz 目录.
        physics_device: PhysX 设备,如 ``cuda:0``.
        physics_hz: 物理频率;与 ``[sim_world].physics_dt`` 保持一致.

    Returns:
        None.
    """
    global ROBOT_PATH, CLOTHING_ROOT_PATH, CLOTHING_SURFACE_PATH
    global CLOTHING_VOLUME_PROXY_PATH, SUPPORT_PATH, BELT_PATH
    global LEFT_GRIPPER_PATH, RIGHT_GRIPPER_PATH
    global RUNTIME_ROOT_PATH, ANCHOR_PATH, ATTACHMENT_SCOPE_PATH, ATTACHMENT_PATH
    global REPLAY_CACHE_DIRECTORY, PHYSICS_DEVICE, NOMINAL_PHYSICS_HZ

    if robot_path:
        ROBOT_PATH = robot_path
    if clothing_root_path:
        CLOTHING_ROOT_PATH = clothing_root_path
    if clothing_surface_path:
        CLOTHING_SURFACE_PATH = clothing_surface_path
    if clothing_volume_proxy_path:
        CLOTHING_VOLUME_PROXY_PATH = clothing_volume_proxy_path
    if support_path:
        SUPPORT_PATH = support_path
    if belt_path:
        BELT_PATH = belt_path
    LEFT_GRIPPER_PATH = left_gripper_path or f"{ROBOT_PATH}/robotiq_left"
    RIGHT_GRIPPER_PATH = right_gripper_path or f"{ROBOT_PATH}/robotiq_right"
    if runtime_root_path:
        RUNTIME_ROOT_PATH = runtime_root_path
    ANCHOR_PATH = f"{RUNTIME_ROOT_PATH}/ClothingGraspFrame"
    ATTACHMENT_SCOPE_PATH = f"{RUNTIME_ROOT_PATH}/Attachments"
    ATTACHMENT_PATH = f"{ATTACHMENT_SCOPE_PATH}/Surface"
    if replay_cache_dir:
        REPLAY_CACHE_DIRECTORY = Path(replay_cache_dir)
    if physics_device:
        PHYSICS_DEVICE = physics_device
    if physics_hz > 0.0:
        NOMINAL_PHYSICS_HZ = float(physics_hz)


class Phase(Enum):
    SETTLE = auto()
    WAITING = auto()
    SEARCH_PREGRASP_IK = auto()
    MOVE_TO_PREGRASP = auto()
    VERTICAL_APPROACH = auto()
    CLOSE_GRIPPER = auto()
    GRIP_SETTLE = auto()
    DRAG = auto()
    CENTER_CORRECT = auto()
    LOWER = auto()
    PLACE_SETTLE = auto()
    LEFT_OPEN = auto()
    LEFT_RETURN_READY = auto()
    PREPARE_RIGHT_HANDOFF = auto()
    RIGHT_SEARCH_IK = auto()
    RIGHT_MOVE_TO_PREGRASP = auto()
    RIGHT_VERTICAL_APPROACH = auto()
    RIGHT_CLOSE = auto()
    RIGHT_GRIP_SETTLE = auto()
    RIGHT_DRAG_TO_BELT = auto()
    RIGHT_RELEASE = auto()
    RIGHT_RETURN_READY = auto()
    FINAL_VERIFY = auto()
    DONE = auto()
    FAILED = auto()
    LEFT_LIFT = auto()  # Append to preserve existing phase IDs in diagnostics/replay.
    RIGHT_LIFT = auto()


PHASE_LABELS = {
    Phase.SETTLE: "hold both arms ready while the deformable settles",
    Phase.WAITING: "visible delay before left-arm motion",
    Phase.SEARCH_PREGRASP_IK: "search 42 deterministic pregrasp IK branches",
    Phase.MOVE_TO_PREGRASP: "move to top-down pregrasp",
    Phase.VERTICAL_APPROACH: "vertical approach",
    Phase.CLOSE_GRIPPER: "close left gripper",
    Phase.GRIP_SETTLE: "verify grasp pose",
    Phase.LEFT_LIFT: "lift vertically by the configured height before horizontal dragging",
    Phase.DRAG: "horizontal smoothstep drag at the raised height",
    Phase.CENTER_CORRECT: "correct XY at the raised height, then calculate tabletop setdown",
    Phase.LOWER: "lower clothing to tabletop",
    Phase.PLACE_SETTLE: "verify center on the tabletop before handoff",
    Phase.LEFT_OPEN: "open the left gripper, retract off the parcel, then leave it on the platform",
    Phase.LEFT_RETURN_READY: "return the left arm to ready",
    Phase.PREPARE_RIGHT_HANDOFF: "recompute the right grasp from the live Newton AABB",
    Phase.RIGHT_SEARCH_IK: "search 42 deterministic right pregrasp IK branches",
    Phase.RIGHT_MOVE_TO_PREGRASP: "move the right arm to top-down pregrasp",
    Phase.RIGHT_VERTICAL_APPROACH: "right-arm vertical approach",
    Phase.RIGHT_CLOSE: "close the right gripper",
    Phase.RIGHT_GRIP_SETTLE: "verify the right grasp pose",
    Phase.RIGHT_LIFT: "lift the right TCP by RIGHT_LIFT_HEIGHT_M before dragging",
    Phase.RIGHT_DRAG_TO_BELT: (
        "drag along world -X while descending, then release after height and belt stability checks"
    ),
    Phase.RIGHT_RELEASE: "open the right gripper and start Newton belt transport",
    Phase.RIGHT_RETURN_READY: "return the right arm to ready",
    Phase.FINAL_VERIFY: "wait for the released package to exit the belt along -X",
    Phase.DONE: "done",
    Phase.FAILED: "failed",
}


def build_phase_timeouts(timing: TimingConfig) -> dict[Phase, float]:
    """Build duration-aware phase timeouts while preserving current margins."""
    timing.validate()
    return {
        Phase.SETTLE: timing.settle_seconds + 9.0,
        Phase.WAITING: timing.waiting_seconds + 5.0,
        Phase.MOVE_TO_PREGRASP: timing.left_move_pregrasp_seconds + 9.5,
        Phase.VERTICAL_APPROACH: timing.left_approach_seconds + 9.5,
        Phase.CLOSE_GRIPPER: timing.left_close_seconds + 5.0,
        Phase.GRIP_SETTLE: max(
            timing.left_grip_settle_seconds + 7.0,
            timing.left_grip_stable_seconds + 7.0,
        ),
        Phase.DRAG: timing.left_drag_seconds + 12.0,
        Phase.LEFT_LIFT: timing.left_lift_seconds + 22.0,
        Phase.CENTER_CORRECT: max(
            timing.center_correction_timeout_seconds,
            timing.center_stable_seconds + 5.0,
        ),
        Phase.LOWER: timing.left_lower_seconds + 10.0,
        Phase.PLACE_SETTLE: max(
            timing.placement_settle_timeout_seconds,
            timing.left_place_settle_seconds + 6.0,
            timing.place_stable_seconds + 6.0,
        ),
        Phase.LEFT_OPEN: timing.left_open_seconds + 18.0,
        Phase.LEFT_RETURN_READY: timing.left_return_ready_seconds + 16.0,
        Phase.PREPARE_RIGHT_HANDOFF: timing.handoff_prepare_timeout_seconds,
        Phase.RIGHT_MOVE_TO_PREGRASP: timing.right_move_pregrasp_seconds + 9.5,
        Phase.RIGHT_VERTICAL_APPROACH: timing.right_approach_seconds + 9.5,
        Phase.RIGHT_CLOSE: timing.right_close_seconds + 5.0,
        Phase.RIGHT_GRIP_SETTLE: max(
            timing.right_grip_settle_seconds + 7.0,
            timing.right_grip_stable_seconds + 7.0,
        ),
        Phase.RIGHT_LIFT: timing.right_lift_seconds + 16.0,
        Phase.RIGHT_DRAG_TO_BELT: max(
            timing.right_drag_timeout_seconds,
            timing.belt_entry_stable_seconds + 5.0,
        ),
        Phase.RIGHT_RELEASE: timing.right_release_seconds + 5.0,
        Phase.RIGHT_RETURN_READY: timing.right_return_ready_seconds + 8.0,
        Phase.FINAL_VERIFY: max(
            timing.final_verify_timeout_seconds,
            timing.final_verify_stable_seconds + 4.5,
        ),
    }


def _to_numpy(value) -> np.ndarray:
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _array_string(value, precision: int = 4) -> str:
    return np.array2string(
        np.asarray(value),
        precision=precision,
        suppress_small=True,
    )


def _smoothstep(progress: float) -> float:
    value = float(np.clip(progress, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _timed_smoothstep(elapsed: float, duration: float) -> float:
    """Return completed progress for zero duration, else timed smoothstep."""
    duration = float(duration)
    if duration <= 0.0:
        return 1.0
    return _smoothstep(float(elapsed) / duration)


def _accumulate_stable_time(
    current: float,
    condition: bool,
    dt: float,
) -> float:
    """Accumulate consecutive stable seconds using the actual physics dt."""
    dt = float(dt)
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError(f"Stable-time dt must be finite and non-negative: {dt!r}")
    return float(current) + dt if condition else 0.0


def _stable_time_reached(current: float, required: float) -> bool:
    """Compare durations without adding a spurious frame from float rounding."""
    return float(current) + 1.0e-12 >= float(required)


def _clamp_vector_norm(vector: np.ndarray, maximum_norm: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= maximum_norm or norm < 1.0e-12:
        return value.copy()
    return value * (maximum_norm / norm)


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise RuntimeError(f"Invalid quaternion: {quaternion}")
    return quaternion / norm


def _quaternion_conjugate(value: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(value)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = _normalize_quaternion(left)
    rw, rx, ry, rz = _normalize_quaternion(right)
    return _normalize_quaternion(
        np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )
    )


def _quaternion_rotation_matrix(value: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(value)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _local_offset_from_world_point(
    frame_position: np.ndarray,
    frame_orientation: np.ndarray,
    world_point: np.ndarray,
) -> np.ndarray:
    """Express a world point as a translation in one rigid frame."""
    return _quaternion_rotation_matrix(_quaternion_conjugate(frame_orientation)) @ (
        np.asarray(world_point, dtype=np.float64).reshape(3)
        - np.asarray(frame_position, dtype=np.float64).reshape(3)
    )


def _matrix_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    return _normalize_quaternion(quaternion)


def _pose_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_rotation_matrix(orientation)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def _matrix_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return matrix[:3, 3].copy(), _matrix_quaternion(matrix[:3, :3])


def _pose_error(
    current_position: np.ndarray,
    current_orientation: np.ndarray,
    target_position: np.ndarray,
    target_orientation: np.ndarray,
) -> np.ndarray:
    delta = _quaternion_multiply(
        target_orientation,
        _quaternion_conjugate(current_orientation),
    )
    sign = 1.0 if delta[0] >= 0.0 else -1.0
    return np.concatenate(
        [
            np.asarray(target_position, dtype=np.float64).reshape(3)
            - np.asarray(current_position, dtype=np.float64).reshape(3),
            2.0 * sign * delta[1:],
        ]
    )


def _wrapped_joint_delta(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """最短角位移 ``wrap(target - current)``,落到 ``(-pi, pi]``."""
    delta = np.asarray(target, dtype=np.float64).reshape(7) - np.asarray(
        current, dtype=np.float64
    ).reshape(7)
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def _dls_joint_delta(
    jacobian: np.ndarray,
    error: np.ndarray,
    *,
    damping: float,
    maximum_step: float,
    nullspace_task: np.ndarray | None = None,
    nullspace_gain: float = 0.0,
) -> np.ndarray:
    """Return one finite damped-least-squares joint increment."""
    matrix = np.asarray(jacobian, dtype=np.float64).reshape(6, 7)
    value = np.asarray(error, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(value)):
        raise RuntimeError("DLS inputs must be finite")
    transpose = matrix.T
    damped = matrix @ transpose + float(damping) ** 2 * np.eye(6)
    delta = transpose @ np.linalg.solve(damped, value)
    gain = float(nullspace_gain)
    if nullspace_task is not None and gain > 0.0:
        task = np.asarray(nullspace_task, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(task)):
            raise RuntimeError("DLS nullspace task must be finite")
        pinv = transpose @ np.linalg.solve(damped, np.eye(6))
        delta = delta + gain * ((np.eye(7) - pinv @ matrix) @ task)
    return np.clip(delta, -float(maximum_step), float(maximum_step))


def _elbow_outboard_m(
    kinematics: UsdArmKinematics,
    positions: np.ndarray,
    outward_xy: np.ndarray,
) -> float:
    """肘相对肩在世界 XY 上沿 ``outward_xy`` 的外展距离;正=向外."""
    _, joints, _ = kinematics._forward_chain(positions)
    shoulder = np.asarray(joints[_SHOULDER_JOINT_INDEX][:2], dtype=np.float64)
    elbow = np.asarray(joints[_ELBOW_JOINT_INDEX][:2], dtype=np.float64)
    axis = np.asarray(outward_xy, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-9:
        return 0.0
    return float(np.dot(elbow - shoulder, axis / norm))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _gf_quaternion(value) -> np.ndarray:
    if value is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    imaginary = value.GetImaginary()
    return _normalize_quaternion(
        np.array(
            [
                float(value.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ],
            dtype=np.float64,
        )
    )


def _attribute_pose(position_attr, rotation_attr) -> np.ndarray:
    position_value = position_attr.Get()
    position = (
        np.zeros(3, dtype=np.float64)
        if position_value is None
        else np.asarray(position_value, dtype=np.float64).reshape(3)
    )
    return _pose_matrix(position, _gf_quaternion(rotation_attr.Get()))


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=np.float64).reshape(3)
    unit_axis /= max(float(np.linalg.norm(unit_axis)), 1.0e-12)
    x, y, z = unit_axis
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    one_minus_cosine = 1.0 - cosine
    rotation = np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def _rigid_world_matrix(path: str) -> np.ndarray:
    positions, orientations = RigidPrim(path).get_world_poses()
    position = _to_numpy(positions).reshape(-1, 3)[0].astype(np.float64)
    orientation = _to_numpy(orientations).reshape(-1, 4)[0].astype(np.float64)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation)):
        raise RuntimeError(f"Rigid-body world pose is non-finite: {path}")
    return _pose_matrix(position, orientation)


@dataclass(frozen=True)
class UsdRevoluteJointSpec:
    path: str
    body0_path: str
    body1_path: str
    body0_to_joint: np.ndarray
    body1_to_joint: np.ndarray
    axis: np.ndarray


class UsdArmKinematics:
    """Offline world-frame FK/Jacobian built from seven USD revolute joints."""

    def __init__(
        self,
        joint_specs: list[UsdRevoluteJointSpec],
        root_world: np.ndarray,
        end_adapter: np.ndarray,
        tcp_adapter: np.ndarray,
        joint_signs: np.ndarray | None = None,
    ) -> None:
        if len(joint_specs) != 7:
            raise RuntimeError(f"Expected seven USD revolute joints, got {len(joint_specs)}")
        self.joint_specs = tuple(joint_specs)
        self.root_world = np.asarray(root_world, dtype=np.float64).reshape(4, 4)
        self.end_adapter = np.asarray(end_adapter, dtype=np.float64).reshape(4, 4)
        self.tcp_adapter = np.asarray(tcp_adapter, dtype=np.float64).reshape(4, 4)
        self.joint_signs = (
            np.ones(7, dtype=np.float64)
            if joint_signs is None
            else np.asarray(joint_signs, dtype=np.float64).reshape(7)
        )
        if not np.all(np.isin(self.joint_signs, (-1.0, 1.0))):
            raise RuntimeError(f"USD joint coordinate signs must be +/-1: {self.joint_signs}")

    @staticmethod
    def _joint_axis(joint: UsdPhysics.RevoluteJoint) -> np.ndarray:
        token = str(joint.GetAxisAttr().Get() or "X").upper()
        axes = {
            "X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        if token not in axes:
            raise RuntimeError(f"Unsupported revolute-joint axis token: {token!r}")
        return axes[token]

    @classmethod
    def from_stage(
        cls,
        stage: Usd.Stage,
        joint_paths: list[str],
        current_positions: np.ndarray,
        gripper_base_path: str,
        tcp_adapter: np.ndarray,
        live_tcp_jacobian: np.ndarray,
    ) -> UsdArmKinematics:
        specs: list[UsdRevoluteJointSpec] = []
        for path in joint_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdPhysics.RevoluteJoint):
                raise RuntimeError(f"Arm DOF is not a USD RevoluteJoint: {path}")
            joint = UsdPhysics.RevoluteJoint(prim)
            body0_targets = joint.GetBody0Rel().GetTargets()
            body1_targets = joint.GetBody1Rel().GetTargets()
            if len(body0_targets) != 1 or len(body1_targets) != 1:
                raise RuntimeError(f"Joint {path} must have exactly one body0 and body1 target")
            specs.append(
                UsdRevoluteJointSpec(
                    path=str(path),
                    body0_path=body0_targets[0].pathString,
                    body1_path=body1_targets[0].pathString,
                    body0_to_joint=_attribute_pose(
                        joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()
                    ),
                    body1_to_joint=_attribute_pose(
                        joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()
                    ),
                    axis=cls._joint_axis(joint),
                )
            )
        for previous, current in zip(specs, specs[1:], strict=False):
            if previous.body1_path != current.body0_path:
                raise RuntimeError(
                    "Arm USD joints are not one continuous serial chain: "
                    f"{previous.path} body1={previous.body1_path}, "
                    f"{current.path} body0={current.body0_path}"
                )

        root_world = _rigid_world_matrix(specs[0].body0_path)
        live_tcp_jacobian = np.asarray(live_tcp_jacobian, dtype=np.float64).reshape(6, 7)
        world_body = root_world.copy()
        joint_signs: list[float] = []
        for index, (spec, angle) in enumerate(
            zip(specs, np.asarray(current_positions, dtype=np.float64).reshape(7), strict=False)
        ):
            world_joint = world_body @ spec.body0_to_joint
            nominal_axis = world_joint[:3, :3] @ spec.axis
            live_axis = live_tcp_jacobian[3:, index]
            alignment = float(np.dot(nominal_axis, live_axis))
            if abs(alignment) < 0.95:
                raise RuntimeError(
                    "USD joint axis does not align with the live PhysX "
                    f"Jacobian at {spec.path}: dot={alignment:.6f}"
                )
            sign = 1.0 if alignment >= 0.0 else -1.0
            joint_signs.append(sign)
            world_body = (
                world_joint
                @ _axis_rotation(spec.axis, sign * float(angle))
                @ np.linalg.inv(spec.body1_to_joint)
            )
        predicted_end = world_body
        measured_end = _rigid_world_matrix(specs[-1].body1_path)
        end_position, end_orientation = _matrix_pose(predicted_end)
        measured_position, measured_orientation = _matrix_pose(measured_end)
        end_error = _pose_error(
            end_position,
            end_orientation,
            measured_position,
            measured_orientation,
        )
        if (
            np.linalg.norm(end_error[:3]) > IK_MODEL_POSITION_TOLERANCE_M
            or np.linalg.norm(end_error[3:]) > IK_MODEL_ORIENTATION_TOLERANCE_RAD
        ):
            raise RuntimeError(
                "USD joint-chain FK does not reproduce the live Link7 pose: "
                f"position_error={np.linalg.norm(end_error[:3]):.6f} m, "
                f"orientation_error={math.degrees(np.linalg.norm(end_error[3:])):.4f} deg"
            )

        measured_gripper = _rigid_world_matrix(gripper_base_path)
        end_adapter = np.linalg.inv(predicted_end) @ measured_gripper
        return cls(
            specs,
            root_world,
            end_adapter,
            tcp_adapter,
            np.asarray(joint_signs, dtype=np.float64),
        )

    def _forward_chain(
        self,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        joint_positions = np.asarray(positions, dtype=np.float64).reshape(7)
        world_body = self.root_world.copy()
        world_joint_positions: list[np.ndarray] = []
        world_joint_axes: list[np.ndarray] = []
        for spec, sign, angle in zip(
            self.joint_specs, self.joint_signs, joint_positions, strict=False
        ):
            world_joint = world_body @ spec.body0_to_joint
            world_joint_positions.append(world_joint[:3, 3].copy())
            world_joint_axes.append(sign * (world_joint[:3, :3] @ spec.axis))
            world_body = (
                world_joint
                @ _axis_rotation(spec.axis, sign * float(angle))
                @ np.linalg.inv(spec.body1_to_joint)
            )
        return world_body, world_joint_positions, world_joint_axes

    def forward_and_jacobian(
        self,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        world_end, joint_positions, joint_axes = self._forward_chain(positions)
        world_gripper = world_end @ self.end_adapter
        world_tcp = world_gripper @ self.tcp_adapter
        tcp_position, tcp_orientation = _matrix_pose(world_tcp)
        gripper_position, _ = _matrix_pose(world_gripper)
        jacobian = np.zeros((6, 7), dtype=np.float64)
        for index, (joint_position, joint_axis) in enumerate(
            zip(joint_positions, joint_axes, strict=False)
        ):
            jacobian[:3, index] = np.cross(joint_axis, tcp_position - joint_position)
            jacobian[3:, index] = joint_axis
        return tcp_position, tcp_orientation, jacobian, gripper_position


def _build_ik_candidate_seeds(
    lower: np.ndarray,
    upper: np.ndarray,
    preferred_seed: np.ndarray = IK_PREFERRED_SEED,
    ready_seed: np.ndarray | None = None,
) -> list[np.ndarray]:
    lower = np.asarray(lower, dtype=np.float64).reshape(7)
    upper = np.asarray(upper, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise RuntimeError("42-seed search requires seven finite joint limits")
    if np.any(lower >= upper):
        raise RuntimeError(f"Invalid left-arm joint limits: {lower}, {upper}")
    generator = np.random.default_rng(IK_RANDOM_SEED)
    preferred = np.asarray(preferred_seed, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(preferred)):
        raise RuntimeError(f"IK preferred seed is non-finite: {preferred}")
    second = np.asarray(preferred if ready_seed is None else ready_seed, dtype=np.float64)
    second = second.reshape(7)
    if not np.all(np.isfinite(second)):
        raise RuntimeError(f"IK ready seed is non-finite: {second}")
    seeds = [preferred.copy(), np.clip(second, lower, upper)]
    seeds.extend(generator.uniform(lower, upper) for _ in range(IK_RANDOM_SEED_COUNT))
    if len(seeds) != IK_CANDIDATE_COUNT:
        raise RuntimeError(f"Expected {IK_CANDIDATE_COUNT} IK seeds, generated {len(seeds)}")
    return [np.asarray(seed, dtype=np.float64).copy() for seed in seeds]


def solve_pregrasp_ik_42_seeds(
    kinematics: UsdArmKinematics,
    target_position: np.ndarray,
    target_orientation: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_limit_margin: float,
    preferred_seed: np.ndarray = IK_PREFERRED_SEED,
    search_timeout_seconds: float = 30.0,
    ready_seed: np.ndarray | None = None,
    outward_xy: np.ndarray | None = None,
    waypoint_positions: tuple[np.ndarray, ...] = (),
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Select one safe deterministic pregrasp branch using exactly 42 seeds.

    冗余臂会有肘内折 / 肘外展两支.本搜索用零空间把解钉在偏好姿态附近,
    再按肘外展距离打分,避免选到贴躯干的内折支.
    """
    lower = np.asarray(lower, dtype=np.float64).reshape(7)
    upper = np.asarray(upper, dtype=np.float64).reshape(7)
    safe_lower = lower + float(joint_limit_margin)
    safe_upper = upper - float(joint_limit_margin)
    if np.any(safe_lower >= safe_upper):
        raise RuntimeError("ProM6 joint-limit margin leaves no safe interval for seeded IK")
    preferred_seed = np.asarray(preferred_seed, dtype=np.float64).reshape(7)
    search_timeout_seconds = float(search_timeout_seconds)
    if not math.isfinite(search_timeout_seconds) or search_timeout_seconds <= 0.0:
        raise RuntimeError(
            f"IK search timeout must be finite and positive: {search_timeout_seconds!r}"
        )
    elbow_axis = LEFT_ELBOW_OUTWARD_XY if outward_xy is None else outward_xy
    seeds = _build_ik_candidate_seeds(lower, upper, preferred_seed, ready_seed)
    best_positions: np.ndarray | None = None
    best_score = math.inf
    best_seed_index = -1
    best_position_error = math.inf
    best_orientation_error = math.inf
    best_elbow_outboard = math.nan
    valid_solution_count = 0
    search_started = time.perf_counter()

    for seed_index, candidate_seed in enumerate(seeds):
        positions = candidate_seed.copy()
        for _ in range(IK_MAX_ITERATIONS):
            if time.perf_counter() - search_started > search_timeout_seconds:
                raise RuntimeError(
                    f"42-seed pregrasp IK search exceeded {search_timeout_seconds:.1f} seconds"
                )
            position, orientation, jacobian, _ = kinematics.forward_and_jacobian(positions)
            error = _pose_error(
                position,
                orientation,
                target_position,
                target_orientation,
            )
            if (
                np.linalg.norm(error[:3]) < IK_INNER_POSITION_TOLERANCE_M
                and np.linalg.norm(error[3:]) < IK_INNER_ORIENTATION_TOLERANCE_RAD
            ):
                break
            delta = _dls_joint_delta(
                jacobian,
                error,
                damping=IK_DAMPING,
                maximum_step=IK_MAX_JOINT_STEP_RAD,
            )
            positions = np.clip(
                positions + delta,
                lower,
                upper,
            )

        final_position, final_orientation, _, _ = kinematics.forward_and_jacobian(positions)
        final_error = _pose_error(
            final_position,
            final_orientation,
            target_position,
            target_orientation,
        )
        position_error = float(np.linalg.norm(final_error[:3]))
        orientation_error = float(np.linalg.norm(final_error[3:]))
        inside_safe_limits = bool(
            np.all(positions >= safe_lower) and np.all(positions <= safe_upper)
        )
        meets_final_tolerance = (
            position_error <= IK_FINAL_POSITION_TOLERANCE_M
            and orientation_error <= IK_FINAL_ORIENTATION_TOLERANCE_RAD
        )
        if not inside_safe_limits or not meets_final_tolerance:
            continue
        elbow_outboard = _elbow_outboard_m(kinematics, positions, elbow_axis)
        if elbow_outboard < IK_ELBOW_OUTBOARD_MIN_M:
            continue
        if not _ik_waypoints_reachable(
            kinematics, positions, waypoint_positions, target_orientation, safe_lower, safe_upper
        ):
            continue
        valid_solution_count += 1
        normalized_margin = np.minimum(
            (positions - lower) / np.maximum(upper - lower, 1.0e-9),
            (upper - positions) / np.maximum(upper - lower, 1.0e-9),
        )
        limit_penalty = float(np.sum(np.maximum(0.12 - normalized_margin, 0.0) ** 2))
        posture_penalty = float(np.linalg.norm(_wrapped_joint_delta(preferred_seed, positions)))
        joint1_wrap = abs(float(_wrapped_joint_delta(preferred_seed, positions)[0]))
        elbow_penalty = IK_ELBOW_IN_PENALTY * max(0.0, IK_ELBOW_OUTBOARD_MIN_M - elbow_outboard)
        score = float(
            position_error
            + 0.20 * orientation_error
            + 0.02 * limit_penalty
            + IK_POSTURE_SCORE_WEIGHT * posture_penalty
            + IK_JOINT1_WRAP_SCORE_WEIGHT * joint1_wrap
            + elbow_penalty
        )
        if score < best_score:
            best_positions = positions.copy()
            best_score = score
            best_seed_index = seed_index
            best_position_error = position_error
            best_orientation_error = orientation_error
            best_elbow_outboard = elbow_outboard

    if best_positions is None:
        raise RuntimeError(
            "42-seed pregrasp IK search found no solution satisfying pose "
            "tolerances and the ProM6 joint-limit safety margin"
        )
    return best_positions, {
        "candidate_count": len(seeds),
        "valid_solution_count": valid_solution_count,
        "seed_index": best_seed_index,
        "score": best_score,
        "position_error": best_position_error,
        "orientation_error": best_orientation_error,
        "elbow_outboard_m": best_elbow_outboard,
    }


def _ik_waypoints_reachable(
    kinematics: UsdArmKinematics,
    start: np.ndarray,
    waypoints: tuple[np.ndarray, ...],
    orientation: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    """Check the approach and lift with the same bounded DLS used during execution."""
    positions = start.copy()
    for target in waypoints:
        for _ in range(IK_MAX_ITERATIONS):
            point, quaternion, jacobian, _ = kinematics.forward_and_jacobian(positions)
            error = _pose_error(point, quaternion, target, orientation)
            if (
                np.linalg.norm(error[:3]) <= IK_FINAL_POSITION_TOLERANCE_M
                and np.linalg.norm(error[3:]) <= IK_FINAL_ORIENTATION_TOLERANCE_RAD
            ):
                break
            positions = np.clip(
                positions
                + _dls_joint_delta(
                    jacobian, error, damping=IK_DAMPING, maximum_step=IK_MAX_JOINT_STEP_RAD
                ),
                lower,
                upper,
            )
        else:
            return False
    return True


def get_world_bounds(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    """Return one finite world-axis-aligned bounding box."""
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=False,
    )
    world_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if world_range.IsEmpty():
        raise RuntimeError(f"Prim has no finite world bounds: {prim.GetPath()}")
    minimum = np.asarray(world_range.GetMin(), dtype=np.float64)
    maximum = np.asarray(world_range.GetMax(), dtype=np.float64)
    if (
        minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(maximum < minimum)
    ):
        raise RuntimeError(
            f"Invalid world bounds for {prim.GetPath()}: min={minimum}, max={maximum}"
        )
    return minimum, maximum


def calculate_grasp_geometry(
    clothing_bounds: tuple[np.ndarray, np.ndarray],
    support_bounds: tuple[np.ndarray, np.ndarray],
    *,
    centered: bool = False,
) -> dict[str, np.ndarray]:
    """Calculate a bag-body grasp (near AABB center) and tabletop-center translation."""
    clothing_min, clothing_max = (
        np.asarray(value, dtype=np.float64).reshape(3) for value in clothing_bounds
    )
    support_min, support_max = (
        np.asarray(value, dtype=np.float64).reshape(3) for value in support_bounds
    )
    values = np.concatenate([clothing_min, clothing_max, support_min, support_max])
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Clothing/support bounds contain non-finite values")
    clothing_center = 0.5 * (clothing_min + clothing_max)
    support_center = 0.5 * (support_min + support_max)
    clothing_half_xy = 0.5 * (clothing_max[:2] - clothing_min[:2])
    support_half_xy = 0.5 * (support_max[:2] - support_min[:2])
    if np.any(clothing_half_xy <= 0.0) or np.any(support_half_xy <= 0.0):
        raise RuntimeError(
            "Clothing and tabletop must have positive XY extents: "
            f"clothing_half={clothing_half_xy}, "
            f"support_half={support_half_xy}"
        )
    if np.any(clothing_half_xy + 0.020 > support_half_xy):
        raise RuntimeError(
            "The clothing footprint does not fit on the tabletop center with "
            f"a 20 mm margin: clothing_half={clothing_half_xy}, "
            f"support_half={support_half_xy}"
        )

    inset = GRASP_EDGE_INSET_M
    half_x = max(float(clothing_half_xy[0]), 0.04)
    grasp_x = float(clothing_center[0]) - LEFT_GRASP_FROM_CENTER_FRAC * half_x
    grasp_y = float(clothing_max[1]) - LEFT_GRASP_Y_INSET_M
    if centered:
        grasp_x, grasp_y = (float(v) for v in clothing_center[:2])
    grasp_x = min(max(grasp_x, float(clothing_min[0]) + inset), float(clothing_max[0]) - inset)
    grasp_y = min(max(grasp_y, float(clothing_min[1]) + inset), float(clothing_max[1]) - inset)
    attachment_center = np.array(
        [grasp_x, grasp_y, float(clothing_max[2]) - LEFT_GRASP_BITE_M],
        dtype=np.float64,
    )
    grasp_point = attachment_center + np.array(
        [0.0, 0.0, GRIPPER_SURFACE_OFFSET_M], dtype=np.float64
    )
    pregrasp_point = grasp_point + np.array([0.0, 0.0, PREGRASP_CLEARANCE_M], dtype=np.float64)
    center_delta_xy = support_center[:2] - clothing_center[:2]
    return {
        "clothing_center": clothing_center,
        "support_center": support_center,
        "attachment_center": attachment_center,
        "grasp_point": grasp_point,
        "pregrasp_point": pregrasp_point,
        "center_delta_xy": center_delta_xy,
    }


def _xy_center_error(center: np.ndarray, target: np.ndarray) -> float:
    """Return planar distance between two world-space centers."""
    center_xy = np.asarray(center, dtype=np.float64).reshape(3)[:2]
    target_xy = np.asarray(target, dtype=np.float64).reshape(3)[:2]
    return float(np.linalg.norm(center_xy - target_xy))


def _center_out_grasp_offsets(half_extents: np.ndarray) -> list[np.ndarray]:
    """按距包裹中心的实际 XY 距离排列抓点,先中心再逐步向机器人侧搜索."""
    half = np.asarray(half_extents, dtype=np.float64).reshape(3)
    fractions = (
        (0.0, 0.0),
        (0.0, 0.125),
        (0.0, 0.25),
        (0.0, 0.375),
        (0.0, 0.5),
        (-0.2, 0.0),
        (-0.2, 0.25),
        (-0.2, 0.5),
        (-0.2, 0.65),
    )
    offsets = [np.array([x * half[0], y * half[1], 0.0]) for x, y in fractions]
    return sorted(offsets, key=lambda offset: float(np.linalg.norm(offset[:2])))


def _center_is_within_tolerance(
    center: np.ndarray,
    target: np.ndarray,
    tolerance: float = CENTER_RELEASE_TOLERANCE_M,
) -> bool:
    return _xy_center_error(center, target) <= float(tolerance)


def _calculate_frozen_drag_target(
    grasp_point: np.ndarray,
    node_center: np.ndarray,
    support_center: np.ndarray,
) -> np.ndarray:
    """Freeze the live center-to-center XY translation plus the low lift."""
    grasp = np.asarray(grasp_point, dtype=np.float64).reshape(3)
    center = np.asarray(node_center, dtype=np.float64).reshape(3)
    support = np.asarray(support_center, dtype=np.float64).reshape(3)
    return grasp + np.array(
        [
            support[0] - center[0],
            support[1] - center[1],
            LOW_LIFT_HEIGHT_M,
        ],
        dtype=np.float64,
    )


def calculate_shared_attachment_grasp(
    anchor_world_position: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate right grasp targets from the existing shared anchor."""
    attachment_center = np.asarray(anchor_world_position, dtype=np.float64).reshape(3).copy()
    if not np.all(np.isfinite(attachment_center)):
        raise RuntimeError(f"Shared attachment position is non-finite: {attachment_center}")
    grasp_point = attachment_center + np.array(
        [0.0, 0.0, GRIPPER_SURFACE_OFFSET_M], dtype=np.float64
    )
    pregrasp_point = grasp_point + np.array([0.0, 0.0, PREGRASP_CLEARANCE_M], dtype=np.float64)
    return {
        "attachment_center": attachment_center,
        "grasp_point": grasp_point,
        "pregrasp_point": pregrasp_point,
    }


def calculate_belt_tail_clearance(
    clothing_max_x: float,
    belt_bounds: tuple[np.ndarray, np.ndarray],
    inset: float = BELT_TAIL_INSET_M,
) -> float:
    """Return the world -X drag needed so the bag +X tail sits inset on the belt."""
    maximum = np.asarray(belt_bounds[1], dtype=np.float64).reshape(3)
    tail = float(clothing_max_x)
    inset = float(inset)
    if not np.all(np.isfinite(np.concatenate([maximum, [tail, inset]]))) or inset < 0.0:
        raise RuntimeError(
            "Invalid bag tail or belt bounds for clearance: "
            f"tail_x={tail}, belt_max={maximum}, inset={inset}"
        )
    return max(0.0, tail - (float(maximum[0]) - inset))


def aabb_tail_is_on_belt(
    clothing_max_xy: np.ndarray,
    belt_bounds: tuple[np.ndarray, np.ndarray],
    inset: float = BELT_TAIL_INSET_M,
) -> bool:
    """True when the bag +X tail is inset on the belt and Y is inside the footprint."""
    point = np.asarray(clothing_max_xy, dtype=np.float64).reshape(-1)
    minimum, maximum = (np.asarray(value, dtype=np.float64).reshape(3) for value in belt_bounds)
    inset = float(inset)
    if point.size < 2 or not np.all(np.isfinite(np.concatenate([point[:2], minimum, maximum]))):
        return False
    return bool(point[0] <= maximum[0] - inset and minimum[1] <= point[1] <= maximum[1])


def calculate_belt_entry_distance(
    clothing_center: np.ndarray,
    belt_bounds: tuple[np.ndarray, np.ndarray],
    inset: float = BELT_ENTRY_INSET_M,
) -> float:
    """Return the required world -X distance to enter the near belt edge."""
    center = np.asarray(clothing_center, dtype=np.float64).reshape(3)
    minimum, maximum = (np.asarray(value, dtype=np.float64).reshape(3) for value in belt_bounds)
    inset = float(inset)
    if (
        not np.all(np.isfinite(np.concatenate([center, minimum, maximum])))
        or np.any(maximum <= minimum)
        or inset < 0.0
        or maximum[0] - minimum[0] <= inset
    ):
        raise RuntimeError(
            "Invalid clothing center, belt bounds, or entry inset: "
            f"center={center}, min={minimum}, max={maximum}, inset={inset}"
        )
    return max(0.0, float(center[0] - (maximum[0] - inset)))


def center_is_in_belt_xy(
    clothing_center: np.ndarray,
    belt_bounds: tuple[np.ndarray, np.ndarray],
    entry_inset: float = BELT_ENTRY_INSET_M,
) -> bool:
    """Test the live center against the belt footprint and inset near edge."""
    center = np.asarray(clothing_center, dtype=np.float64).reshape(3)
    minimum, maximum = (np.asarray(value, dtype=np.float64).reshape(3) for value in belt_bounds)
    inset = float(entry_inset)
    if not np.all(np.isfinite(np.concatenate([center, minimum, maximum]))):
        return False
    return bool(
        minimum[0] <= center[0] <= maximum[0] - inset and minimum[1] <= center[1] <= maximum[1]
    )


def _as_nodal_matrix(values: object, label: str) -> np.ndarray:
    nodes = np.asarray(values, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[0] == 0 or nodes.shape[1] != 3:
        raise ValueError(f"{label} must have shape (num_nodes, 3), got {nodes.shape}")
    if not np.all(np.isfinite(nodes)):
        raise ValueError(f"{label} contains non-finite values")
    return nodes


def _single_nodal_matrix(value: object, label: str) -> np.ndarray:
    primary = value[0] if isinstance(value, tuple) else value
    nodes = _to_numpy(primary)
    if nodes.ndim != 3 or nodes.shape[0] != 1:
        raise RuntimeError(f"{label} must have shape (1, num_nodes, 3), got {nodes.shape}")
    try:
        return _as_nodal_matrix(nodes[0], label)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _normalized_planar_direction(direction_xy: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction_xy, dtype=np.float64).reshape(-1)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("direction_xy must contain two finite values")
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("direction_xy must be non-zero")
    return direction / norm


def _compose_nodal_planar_velocity(
    current_velocities: object,
    speed_mps: float,
    direction_xy: np.ndarray,
) -> np.ndarray:
    """Overwrite node XY velocity while preserving every node's Z velocity."""
    current = _as_nodal_matrix(current_velocities, "nodal_velocities")
    speed = float(speed_mps)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("speed_mps must be finite and non-negative")
    result = current.copy()
    result[:, :2] = speed * _normalized_planar_direction(direction_xy)
    return result


def _belt_xy_bounds(
    belt_bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    minimum, maximum = (np.asarray(value, dtype=np.float64).reshape(3) for value in belt_bounds)
    values = np.array(
        [minimum[0], minimum[1], maximum[0], maximum[1]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values[2:] <= values[:2]):
        raise ValueError(f"Invalid belt XY bounds: {values}")
    return values


def _point_in_xy_bounds(point_xy: object, bounds: object) -> bool:
    point = np.asarray(point_xy, dtype=np.float64).reshape(-1)
    region = np.asarray(bounds, dtype=np.float64).reshape(-1)
    if (
        point.shape != (2,)
        or region.shape != (4,)
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(region))
        or np.any(region[2:] <= region[:2])
    ):
        raise ValueError(f"Invalid point or XY bounds: point={point}, bounds={region}")
    return bool(region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3])


def center_exited_belt_negative_x(
    clothing_center: np.ndarray,
    belt_bounds: tuple[np.ndarray, np.ndarray],
) -> bool:
    """Return whether the center crossed the -X edge while retaining belt Y."""
    center = np.asarray(clothing_center, dtype=np.float64).reshape(3)
    bounds = _belt_xy_bounds(belt_bounds)
    if not np.all(np.isfinite(center)):
        return False
    return bool(center[0] < bounds[0] and bounds[1] <= center[1] <= bounds[3])


def require_timeline_stopped(stage: Usd.Stage) -> None:
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError(
            "Stop the Timeline before running this script.  The deformable "
            "attachment and GPU settings must exist before the first Play."
        )
    if stage is None:
        raise RuntimeError("No USD Stage is open")


def require_uninitialized_physics() -> None:
    """附着须在物理视图建立前写入;框架里 ``init_world`` 已 reset 过,故只告警.

    Returns:
        None.
    """
    if SimulationManager.get_physics_simulation_view() is not None:
        # Job 流水线在 init_world 就 reset 过一次;后续 reset+play 会重新解析 stage,
        # 因此这里不再像原脚本那样抛错,只提示附着是在已有物理视图之后写入的.
        _LOGGER.warning(
            "physics_view_already_exists",
            tip="附着将在下一次 reset/play 时重新烘焙",
        )


def configure_gpu_physics(stage: Usd.Stage) -> None:
    SimulationManager.set_physics_sim_device(PHYSICS_DEVICE)

    scene_paths = PhysicsScene.get_physics_scene_paths(stage)
    if not scene_paths:
        raise RuntimeError("The Stage has no USD PhysicsScene")

    requested_stack = GPU_COLLISION_STACK_SIZE_MIB * 1024 * 1024
    for scene_path in scene_paths:
        scene_prim = stage.GetPrimAtPath(scene_path)
        scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        scene_api.CreateEnableGPUDynamicsAttr().Set(True)
        scene_api.CreateBroadphaseTypeAttr().Set("GPU")
        scene_api.CreateGpuCollisionStackSizeAttr().Set(requested_stack)
        SimulationManager.set_physics_dt(
            1.0 / NOMINAL_PHYSICS_HZ,
            physics_scene=str(scene_path),
        )


def remove_owned_runtime_prims(stage: Usd.Stage) -> None:
    prim = stage.GetPrimAtPath(RUNTIME_ROOT_PATH)
    if prim.IsValid():
        stage.RemovePrim(RUNTIME_ROOT_PATH)


def set_xform_world_translation(
    xform: UsdGeom.Xform,
    position: np.ndarray,
) -> None:
    """Set one runtime Xform whose parent hierarchy has identity transforms."""
    value = np.asarray(position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"Anchor position is non-finite: {value}")
    ops = xform.GetOrderedXformOps()
    translate = next(
        (op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    if translate is None:
        translate = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate.Set(Gf.Vec3d(*(float(component) for component in value)))


def get_xform_world_translation(xform: UsdGeom.Xform) -> np.ndarray:
    """Read one finite world-space translation from a USD Xform."""
    if xform is None or not xform.GetPrim().IsValid():
        raise RuntimeError("Shared attachment anchor Xform is invalid")
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    position = np.asarray(matrix.ExtractTranslation(), dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(position)):
        raise RuntimeError(f"Shared attachment anchor position is non-finite: {position}")
    return position


def create_masked_surface_attachment(
    stage: Usd.Stage,
    attachment_center: np.ndarray,
    *,
    anchor_path: str = "",
    attachment_path: str = "",
    deformable: DeformablePrim | None = None,
) -> tuple[UsdGeom.Xform, DeformablePrim]:
    """Create the temporary world-space anchor and masked surface attachment.

    ``anchor_path`` / ``attachment_path`` 留空表示用模块级路径:默认参数在函数定义时
    就绑定,而 :func:`apply_settings` 是运行期改写全局,写成默认值会取到旧路径.
    """
    anchor_path = anchor_path or ANCHOR_PATH
    attachment_path = attachment_path or ATTACHMENT_PATH
    runtime_root = UsdGeom.Xform.Define(stage, RUNTIME_ROOT_PATH)
    runtime_root.GetPrim().SetMetadata("comment", "Owned by ProM6 clothing drag script")
    anchor = UsdGeom.Xform.Define(stage, anchor_path)
    set_xform_world_translation(anchor, attachment_center)
    # Keep the attachment target as a plain Xform, matching
    # screw_unscrew_demo.py and OmniPhysicsVtxXformAttachment semantics.  A
    # kinematic RigidBody has a separate runtime Physics pose; changing only
    # its USD xformOp can move the viewport Prim without moving the attachment
    # target seen by the GPU solver.
    UsdGeom.Scope.Define(stage, ATTACHMENT_SCOPE_PATH)

    if deformable is None:
        deformable = DeformablePrim(
            CLOTHING_SURFACE_PATH,
            deformable_type="surface",
        )
    if not deformable.valid:
        raise RuntimeError(f"Invalid surface DeformablePrim: {CLOTHING_SURFACE_PATH}")
    if deformable.deformable_type != "surface":
        raise RuntimeError(f"Expected a surface deformable, got {deformable.deformable_type!r}")
    if not deformable.simulation_mesh_paths:
        raise RuntimeError(f"Surface deformable has no simulation mesh: {CLOTHING_SURFACE_PATH}")

    attachment_sdf_path = Sdf.Path(attachment_path)
    if not deformableUtils.create_auto_deformable_attachment(
        stage,
        target_attachment_path=attachment_sdf_path,
        attachable0_path=Sdf.Path(CLOTHING_SURFACE_PATH),
        attachable1_path=Sdf.Path(anchor_path),
    ):
        raise RuntimeError(f"Failed to create deformable attachment: {attachment_path}")

    mask_path = attachment_sdf_path.AppendChild("Mask")
    mask = UsdGeom.Cube.Define(stage, mask_path)
    mask.CreateSizeAttr(1.0)
    mask.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*(float(value) for value in attachment_center))
    )
    mask.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(*(float(value) for value in GRASP_MASK_SIZE_M))
    )
    UsdGeom.Imageable(mask.GetPrim()).MakeInvisible()

    attachment_prim = stage.GetPrimAtPath(attachment_sdf_path)
    attachment_prim.GetRelationship("physxAutoDeformableAttachment:maskShapes").SetTargets(
        [mask_path]
    )
    prefix = "physxAutoDeformableAttachment:"
    attachment_prim.GetAttribute(prefix + "deformableVertexOverlapOffset").Set(0.002)
    attachment_interface = get_physx_attachment_private_interface()
    updated = attachment_interface.update_auto_deformable_attachment(attachment_path)
    if not updated:
        raise RuntimeError(f"Failed to cook/update masked attachment: {attachment_path}")
    return anchor, deformable


def attachment_vertex_count(stage: Usd.Stage, attachment_path: str) -> int:
    """Count generated low-level source vertices below one attachment."""
    root = stage.GetPrimAtPath(attachment_path)
    if not root.IsValid():
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        attr = prim.GetAttribute("omniphysics:vtxIndicesSrc0")
        if not attr.IsValid():
            continue
        value = attr.Get()
        if value is not None:
            count += len(value)
    return count


def disable_gripper_collisions(
    stage: Usd.Stage,
    gripper_path: str,
    side_label: str,
) -> list[tuple[str, bool, bool | None]]:
    """Disable authored gripper colliders and retain restoration data.

    Some Pro-m6 Robotiq variants contain only visual meshes plus empty
    ``collisions`` Xforms.  In that case there is no collision geometry that
    can strike the deformable, so an empty result is valid rather than a
    startup failure.
    """
    root = stage.GetPrimAtPath(gripper_path)
    if not root.IsValid():
        raise RuntimeError(f"{side_label} gripper Prim does not exist: {gripper_path}")
    saved: list[tuple[str, bool, bool | None]] = []
    for prim in Usd.PrimRange(root):
        collision_attr = prim.GetAttribute("physics:collisionEnabled")
        if not prim.HasAPI(UsdPhysics.CollisionAPI) and not collision_attr.IsValid():
            continue
        if prim.IsInstanceProxy():
            raise RuntimeError(
                f"{side_label} gripper collider is an instance proxy and cannot be "
                f"temporarily disabled safely: {prim.GetPath()}"
            )
        collision = UsdPhysics.CollisionAPI(prim)
        attr = collision.GetCollisionEnabledAttr()
        if not attr.IsValid():
            attr = collision.CreateCollisionEnabledAttr()
        had_authored = bool(attr and attr.HasAuthoredValueOpinion())
        old_value = attr.Get() if attr else None
        attr.Set(False)
        saved.append(
            (
                prim.GetPath().pathString,
                had_authored,
                None if old_value is None else bool(old_value),
            )
        )
    if not saved:
        _log(
            f"[ProM6ClothingDrag] {side_label} Robotiq has no authored collision "
            "geometry; collision override is unnecessary"
        )
    return saved


def restore_collision_attributes(
    stage: Usd.Stage,
    saved: list[tuple[str, bool, bool | None]],
) -> None:
    for path, had_authored, old_value in saved:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        attr = prim.GetAttribute("physics:collisionEnabled")
        if not attr.IsValid():
            continue
        if had_authored:
            attr.Set(bool(old_value))
        else:
            attr.Clear()


class SuccessfulOperationRecorder:
    """Keep live tensor state in memory and commit only a verified success."""

    def __init__(
        self,
        stage: Usd.Stage,
        robot: Articulation,
        dof_names: tuple[str, ...],
        dof_paths: tuple[str, ...],
        *,
        nodal_position_fn: Callable[[], np.ndarray],
        nodal_velocity_fn: Callable[[], np.ndarray],
    ) -> None:
        if not dof_names or len(dof_names) != len(dof_paths):
            raise RuntimeError("Replay recorder requires matching non-empty DOF metadata")
        self.stage = stage
        self.robot = robot
        self.nodal_position_fn = nodal_position_fn
        self.nodal_velocity_fn = nodal_velocity_fn
        self.dof_names = tuple(str(value) for value in dof_names)
        self.dof_paths = tuple(str(value) for value in dof_paths)
        self.node_count: int | None = None
        self.committed_path: Path | None = None
        self._frames: dict[str, list[object]] = {
            "frame_dt": [],
            "phase_ids": [],
            "dof_positions": [],
            "dof_velocities": [],
            "dof_position_targets": [],
            "nodal_positions": [],
            "nodal_velocities": [],
            "attachment_active": [],
            "anchor_positions": [],
        }

    @property
    def frame_count(self) -> int:
        return len(self._frames["frame_dt"])

    @staticmethod
    def _finite_vector(value: object, size: int, label: str) -> np.ndarray:
        result = _to_numpy(value).reshape(-1).astype(np.float64)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise RuntimeError(
                f"Replay recorder {label} must contain {size} finite values; "
                f"got shape={result.shape}"
            )
        return result

    def capture(
        self,
        dt: float,
        phase: Phase,
        *,
        attachment_active: bool,
        anchor_position: np.ndarray | None,
    ) -> None:
        if self.committed_path is not None:
            raise RuntimeError("Cannot capture frames after replay cache commit")
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0 or dt > 0.1:
            raise RuntimeError(f"Replay recorder received invalid dt={dt!r}")

        dof_count = len(self.dof_paths)
        dof_positions = self._finite_vector(
            self.robot.get_dof_positions(), dof_count, "DOF positions"
        )
        dof_velocities = self._finite_vector(
            self.robot.get_dof_velocities(), dof_count, "DOF velocities"
        )
        dof_position_targets = self._finite_vector(
            self.robot.get_dof_position_targets(),
            dof_count,
            "DOF position targets",
        )
        nodal_positions = _as_nodal_matrix(self.nodal_position_fn(), "replay nodal positions")
        nodal_velocities = _as_nodal_matrix(self.nodal_velocity_fn(), "replay nodal velocities")
        if nodal_positions.shape != nodal_velocities.shape:
            raise RuntimeError(
                "Replay recorder nodal position/velocity shapes differ: "
                f"positions={nodal_positions.shape}, "
                f"velocities={nodal_velocities.shape}"
            )
        if self.node_count is None:
            self.node_count = int(nodal_positions.shape[0])
        elif nodal_positions.shape[0] != self.node_count:
            raise RuntimeError(
                "Replay recorder deformable node count changed: "
                f"expected={self.node_count}, actual={nodal_positions.shape[0]}"
            )

        active = bool(attachment_active)
        if active:
            if anchor_position is None:
                raise RuntimeError("Active attachment has no replay anchor position")
            anchor = np.asarray(anchor_position, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(anchor)):
                raise RuntimeError(f"Active attachment replay anchor is non-finite: {anchor}")
        else:
            anchor = np.full(3, np.nan, dtype=np.float64)

        cache_dtype = np.float32
        self._frames["frame_dt"].append(dt)
        self._frames["phase_ids"].append(int(phase.value))
        self._frames["dof_positions"].append(dof_positions.astype(cache_dtype))
        self._frames["dof_velocities"].append(dof_velocities.astype(cache_dtype))
        self._frames["dof_position_targets"].append(dof_position_targets.astype(cache_dtype))
        self._frames["nodal_positions"].append(nodal_positions.astype(cache_dtype))
        self._frames["nodal_velocities"].append(nodal_velocities.astype(cache_dtype))
        self._frames["attachment_active"].append(active)
        self._frames["anchor_positions"].append(anchor.astype(cache_dtype))

    def discard(self) -> None:
        if self.committed_path is not None:
            return
        for values in self._frames.values():
            values.clear()
        self.node_count = None

    def _metadata(self) -> dict[str, object]:
        if self.node_count is None or self.frame_count <= 0:
            raise RuntimeError("Cannot build metadata for an empty replay cache")
        root_layer = self.stage.GetRootLayer()
        stage_identifier = "" if root_layer is None else str(root_layer.identifier)
        frame_dt = np.asarray(self._frames["frame_dt"], dtype=np.float64)
        return {
            "schema_version": REPLAY_CACHE_SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "stage_identifier": stage_identifier,
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(self.stage)),
            "robot_path": ROBOT_PATH,
            "clothing_surface_path": CLOTHING_SURFACE_PATH,
            "anchor_path": ANCHOR_PATH,
            "dof_names": list(self.dof_names),
            "dof_paths": list(self.dof_paths),
            "frame_count": self.frame_count,
            "node_count": self.node_count,
            "dtype": "float32",
            "median_physics_dt": float(np.median(frame_dt)),
            "phase_names": {str(value.value): value.name for value in Phase},
        }

    def commit_success(self) -> Path:
        if self.committed_path is not None:
            return self.committed_path
        metadata = self._metadata()
        arrays = {
            "frame_dt": np.asarray(self._frames["frame_dt"], dtype=np.float32),
            "phase_ids": np.asarray(self._frames["phase_ids"], dtype=np.int16),
            "dof_positions": np.stack(self._frames["dof_positions"]).astype(np.float32, copy=False),
            "dof_velocities": np.stack(self._frames["dof_velocities"]).astype(
                np.float32, copy=False
            ),
            "dof_position_targets": np.stack(self._frames["dof_position_targets"]).astype(
                np.float32, copy=False
            ),
            "nodal_positions": np.stack(self._frames["nodal_positions"]).astype(
                np.float32, copy=False
            ),
            "nodal_velocities": np.stack(self._frames["nodal_velocities"]).astype(
                np.float32, copy=False
            ),
            "attachment_active": np.asarray(self._frames["attachment_active"], dtype=np.bool_),
            "anchor_positions": np.stack(self._frames["anchor_positions"]).astype(
                np.float32, copy=False
            ),
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
        }

        REPLAY_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        final_path = REPLAY_CACHE_DIRECTORY / (f"{REPLAY_CACHE_FILE_PREFIX}{timestamp}.npz")
        temporary_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
        if final_path.exists() or temporary_path.exists():
            raise RuntimeError(f"Replay cache output path already exists: {final_path}")
        try:
            with temporary_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, final_path)
        except Exception:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        self.committed_path = final_path
        _log(
            "[ProM6ClothingDrag][cache] Saved successful replay: "
            f"{final_path} (frames={self.frame_count}, nodes={self.node_count})"
        )
        return final_path


class ConveyorState(Enum):
    WAITING_FOR_BELT = auto()
    TRANSPORTING = auto()
    STOPPED = auto()
    EXITED = auto()


class DeformableConveyorController:
    """松爪后给 Newton 出带代理 -X 速度,用 AABB 判定是否送出."""

    def __init__(
        self,
        newton_world: NewtonSortingWorld,
        belt_xy_bounds: np.ndarray,
        stop_xy_region: dict[str, float] | None,
        speed_mps: float,
        direction_xy: np.ndarray,
        *,
        stop_on_arrival: bool,
        zero_xy_on_exit: bool,
        log_interval_frames: int,
    ) -> None:
        self.newton_world = newton_world
        self.belt_xy_bounds = np.asarray(belt_xy_bounds, dtype=np.float64).reshape(-1)
        if (
            self.belt_xy_bounds.shape != (4,)
            or not np.all(np.isfinite(self.belt_xy_bounds))
            or np.any(self.belt_xy_bounds[2:] <= self.belt_xy_bounds[:2])
        ):
            raise ValueError(f"Invalid conveyor belt XY bounds: {self.belt_xy_bounds}")

        self.stop_xy_region = None if stop_xy_region is None else dict(stop_xy_region)
        self.stop_xy_bounds: np.ndarray | None = None
        if self.stop_xy_region is not None:
            expected = {"x_min", "x_max", "y_min", "y_max"}
            if set(self.stop_xy_region) != expected:
                raise ValueError("stop_xy_region must contain exactly x_min, x_max, y_min, y_max")
            self.stop_xy_bounds = np.array(
                [
                    self.stop_xy_region["x_min"],
                    self.stop_xy_region["y_min"],
                    self.stop_xy_region["x_max"],
                    self.stop_xy_region["y_max"],
                ],
                dtype=np.float64,
            )
            _point_in_xy_bounds(self.stop_xy_bounds[:2], self.stop_xy_bounds)

        self.speed_mps = float(speed_mps)
        self.direction_xy = _normalized_planar_direction(direction_xy)
        self.stop_on_arrival = bool(stop_on_arrival)
        self.zero_xy_on_exit = bool(zero_xy_on_exit)
        self.log_interval_frames = int(log_interval_frames)
        if not math.isfinite(self.speed_mps) or self.speed_mps < 0.0:
            raise ValueError("speed_mps must be finite and non-negative")
        if self.log_interval_frames <= 0:
            raise ValueError("log_interval_frames must be positive")
        if self.stop_on_arrival and self.stop_xy_bounds is None:
            raise ValueError("stop_xy_region is required when stop_on_arrival=True")

        self.state = ConveyorState.WAITING_FOR_BELT
        self.active = False
        self.cleaned = False
        self.was_in_belt = False
        self.frame = 0
        self.exit_center: np.ndarray | None = None
        self.last_command: np.ndarray | None = None

    def start(self) -> None:
        if self.cleaned:
            raise RuntimeError("Cannot restart a cleaned conveyor controller")
        if self.active:
            return
        self.active = True
        self.newton_world.set_belt_speed(self.speed_mps)
        _log(
            "[ProM6ClothingDrag][conveyor] Started: "
            f"speed={self.speed_mps:.3f} m/s, "
            f"direction_xy={_array_string(self.direction_xy)}, "
            f"stop_xy_region={self.stop_xy_region}, "
            f"stop_on_arrival={self.stop_on_arrival}, "
            f"zero_xy_on_exit={self.zero_xy_on_exit}"
        )

    def _set_state(self, state: ConveyorState) -> None:
        if state == self.state:
            return
        self.state = state
        _log(f"[ProM6ClothingDrag][conveyor] State {state.name}")

    def _read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = _as_nodal_matrix(self.newton_world.surface_positions(), "nodal_positions")
        velocities = _as_nodal_matrix(
            self.newton_world.particle_velocities()[: self.newton_world.surface_count],
            "nodal_velocities",
        )
        if positions.shape != velocities.shape:
            raise RuntimeError(
                "Release conveyor nodal position/velocity shapes differ: "
                f"positions={positions.shape}, velocities={velocities.shape}"
            )
        center = 0.5 * (np.min(positions, axis=0) + np.max(positions, axis=0))
        return positions, velocities, center

    def _write_command(self, command: np.ndarray) -> None:
        del command
        # 接触抓取不再写粒子速度;皮带运动学速度在 start/cleanup 里切换.
        self.last_command = None

    def step(self, dt: float) -> None:
        if self.cleaned or not self.active:
            return
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0 or dt > 0.1:
            raise RuntimeError(f"Invalid release conveyor dt={dt!r}")

        self.frame += 1
        positions, velocities, center = self._read_state()
        inside_belt = _point_in_xy_bounds(center[:2], self.belt_xy_bounds)
        inside_stop = bool(
            inside_belt
            and self.stop_on_arrival
            and self.stop_xy_bounds is not None
            and _point_in_xy_bounds(center[:2], self.stop_xy_bounds)
        )

        if inside_stop:
            command = velocities.copy()
            command[:, :2] = 0.0
            self._write_command(command)
            self.was_in_belt = True
            self._set_state(ConveyorState.STOPPED)
        elif inside_belt:
            command = _compose_nodal_planar_velocity(velocities, self.speed_mps, self.direction_xy)
            self._write_command(command)
            self.was_in_belt = True
            self._set_state(ConveyorState.TRANSPORTING)
        elif self.was_in_belt:
            if self.zero_xy_on_exit:
                command = velocities.copy()
                command[:, :2] = 0.0
                self._write_command(command)
            self.exit_center = center.copy()
            self._set_state(ConveyorState.EXITED)
            _log(
                f"[ProM6ClothingDrag][conveyor] Center exited belt: center={_array_string(center)}"
            )
            self.cleanup()
            return
        else:
            self._set_state(ConveyorState.WAITING_FOR_BELT)

        if self.frame % self.log_interval_frames == 0:
            span = np.max(positions, axis=0) - np.min(positions, axis=0)
            mean_xy = np.mean(velocities[:, :2], axis=0)
            _log(
                "[ProM6ClothingDrag][conveyor][status] "
                f"state={self.state.name}, center={_array_string(center)}, "
                f"span={_array_string(span)}, mean_xy={_array_string(mean_xy)}, "
                f"contacts={self.newton_world.contact_breakdown()}, "
                f"belt_speed={self.newton_world.belt_speed_mps:.3f}"
            )

    def cleanup(self) -> None:
        if self.cleaned:
            return
        self.cleaned = True
        self.active = False
        self.newton_world.set_belt_speed(0.0)
        _log("[ProM6ClothingDrag][conveyor] Controller cleaned")


def validate_right_lift_height(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("RIGHT_LIFT_HEIGHT_M must be finite and nonnegative")
    return value


def validate_right_drag_lower_distance(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("RIGHT_DRAG_LOWER_DISTANCE_M must be finite and nonnegative")
    return value


class ProM6LeftClothingDragController:
    """Dual-arm relay with seeded, reachability-constrained grasp sampling."""

    def __init__(
        self,
        stage: Usd.Stage,
        timing: TimingConfig,
        articulation: Articulation | None = None,
        keep_label_up: bool = False,
        label_up_min_cos: float = 0.9,
        left_lift_height_m: float = LOW_LIFT_HEIGHT_M,
        left_setdown_offset_m: float = LEFT_LOWER_DISTANCE_OFFSET_M,
    ) -> None:
        validate_right_lift_height(RIGHT_LIFT_HEIGHT_M)
        validate_right_drag_lower_distance(RIGHT_DRAG_LOWER_DISTANCE_M)
        if not math.isfinite(float(LEFT_LOWER_DISTANCE_OFFSET_M)):
            raise ValueError("LEFT_LOWER_DISTANCE_OFFSET_M must be finite")
        self.stage = stage
        self.keep_label_up = keep_label_up
        self.label_up_min_cos = label_up_min_cos
        self.left_lift_height_m = left_lift_height_m
        self.left_setdown_offset_m = left_setdown_offset_m
        self.label_up_cos_at_handoff: float | None = None
        self.timing = timing.validate()
        self.phase_timeouts = build_phase_timeouts(self.timing)
        self.timeline = omni.timeline.get_timeline_interface()
        # 机器人后端已建过同一 prim 的 view,复用它,避免一根关节体上挂两份句柄
        self.robot = articulation if articulation is not None else Articulation(ROBOT_PATH)
        self.anchor: UsdGeom.Xform | None = None
        self.deformable: DeformablePrim | None = None
        self.newton_world: NewtonSortingWorld | None = None
        self._newton_hold_frames = 0
        self.min_lift_height_m = 0.08
        self.left_lift_com_z: float | None = None
        self.right_lift_com_z: float | None = None
        self.replay_recorder: SuccessfulOperationRecorder | None = None
        self.release_conveyor: DeformableConveyorController | None = None
        self.saved_collisions: list[tuple[str, bool, bool | None]] = []

        self.callback_id: int | None = None
        self.stop_callback_id: int | None = None
        self.initialized = False
        self.cleaned = False
        self.failed = False
        self.failure_reason: str | None = None
        self.phase = Phase.SETTLE
        self.phase_time = 0.0
        self.phase_frame = 0
        self.stable_time = 0.0

        self.left_arm_indices: list[int] = []
        self.right_arm_indices: list[int] = []
        self.left_arm_dof_paths: list[str] = []
        self.right_arm_dof_paths: list[str] = []
        self.left_arm_lower_limits: np.ndarray | None = None
        self.left_arm_upper_limits: np.ndarray | None = None
        self.right_arm_lower_limits: np.ndarray | None = None
        self.right_arm_upper_limits: np.ndarray | None = None
        self.left_base_link_path: str | None = None
        self.left_base_link_index: int | None = None
        self.left_base_jacobian_row: int | None = None
        self.left_tool_center_path: str | None = None
        self.right_base_link_path: str | None = None
        self.right_base_link_index: int | None = None
        self.right_base_jacobian_row: int | None = None
        self.right_tool_center_path: str | None = None
        self.left_gripper_index: int | None = None
        self.right_gripper_index: int | None = None
        self.left_gripper_lower_limit = -math.inf
        self.left_gripper_upper_limit = math.inf
        self.right_gripper_lower_limit = -math.inf
        self.right_gripper_upper_limit = math.inf
        self.last_joint_step = 0.0
        self.last_tcp_error = math.nan

        self.left_start_q: np.ndarray | None = None
        self.left_commanded_q: np.ndarray | None = None
        self.right_start_q: np.ndarray | None = None
        self.right_commanded_q: np.ndarray | None = None
        self.left_return_start_q: np.ndarray | None = None
        self.right_return_start_q: np.ndarray | None = None
        self.pregrasp_start_q: np.ndarray | None = None
        self.pregrasp_joint_target: np.ndarray | None = None
        self.right_pregrasp_start_q: np.ndarray | None = None
        self.right_pregrasp_joint_target: np.ndarray | None = None
        self.tcp_offset_local: np.ndarray | None = None
        self.tcp_adapter: np.ndarray | None = None
        self.usd_kinematics: UsdArmKinematics | None = None
        self.right_tcp_offset_local: np.ndarray | None = None
        self.right_tcp_adapter: np.ndarray | None = None
        self.right_usd_kinematics: UsdArmKinematics | None = None

        self.grasp_orientation = np.array(
            [0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0],
            dtype=np.float64,
        )
        self.geometry: dict[str, np.ndarray] | None = None
        self.clothing_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.support_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.grasp_point: np.ndarray | None = None
        self._left_grasp_offset = np.zeros(3, dtype=np.float64)
        self._right_grasp_offset = np.zeros(3, dtype=np.float64)
        self.grasp_jitter_fraction_xy = (0.0, 0.0)
        self.grasp_random_candidates = 12
        self.grasp_min_distance_m = 0.01
        self.grasp_rngs: dict[str, np.random.Generator] = {}
        self.previous_grasp_offsets: dict[str, np.ndarray] = {}
        self.left_grasp_position: np.ndarray | None = None
        self.right_grasp_position: np.ndarray | None = None
        self.pregrasp_point: np.ndarray | None = None
        self.attachment_center: np.ndarray | None = None
        self.right_handoff_anchor_position: np.ndarray | None = None
        self.right_handoff_tcp_position: np.ndarray | None = None
        self.right_grasp_point: np.ndarray | None = None
        self.right_pregrasp_point: np.ndarray | None = None
        self.right_lift_start: np.ndarray | None = None
        self.right_lift_target: np.ndarray | None = None
        self.right_drag_start: np.ndarray | None = None
        self.right_drag_target: np.ndarray | None = None
        self.right_nominal_drag_distance = 0.0
        self.right_actual_drag_distance = 0.0
        self.support_center: np.ndarray | None = None
        self.belt_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.drag_start: np.ndarray | None = None
        self.left_lift_start: np.ndarray | None = None
        self.left_lift_target: np.ndarray | None = None
        self._left_lift_extra_height_m = 0.0
        self._left_lift_segment_started_at = 0.0
        self._left_lift_segment_duration = self.timing.left_lift_seconds
        self._left_lift_endpoint_hold_s = 0.0
        self.drag_target: np.ndarray | None = None
        self.correction_target: np.ndarray | None = None
        self.correction_origin: np.ndarray | None = None
        self.lower_start: np.ndarray | None = None
        self.setdown_target: np.ndarray | None = None
        self.attachment_active = False
        self.initial_node_center: np.ndarray | None = None
        self.initial_node_height: float | None = None
        self.final_center_error: float | None = None
        self._cached_nodes: np.ndarray | None = None
        self._right_grasp_z_refreshed = False
        self._right_grasp_dip_m = 0.0
        self._left_grasp_dip_m = 0.0
        self.left_retract_target: np.ndarray | None = None
        self._left_retract_waypoints: list[np.ndarray] = []
        self._left_retract_i = 0
        self._left_return_clear_target: np.ndarray | None = None
        self._left_return_cartesian_done = False
        self._left_return_joint_elapsed = 0.0
        self._left_return_stall_s = 0.0
        self._left_retract_hold_s = 0.0
        self._left_close_xy_refreshed = False
        self._left_close_elapsed = 0.0
        self._right_close_elapsed = 0.0

    @staticmethod
    def _single_exact_index(values: tuple[str, ...], expected: str, label: str) -> int:
        matches = [index for index, value in enumerate(values) if value == expected]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {label} {expected!r}; matches={matches}")
        return matches[0]

    @staticmethod
    def _single_suffix_index(values: tuple[str, ...], suffix: str, label: str) -> int:
        matches = [index for index, value in enumerate(values) if value.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {label} ending in {suffix!r}; matches={matches}")
        return matches[0]

    @staticmethod
    def _jacobian_row(link_index: int, link_count: int, row_count: int) -> int:
        if row_count == link_count:
            row = link_index
        elif row_count == link_count - 1:
            row = link_index - 1
        else:
            raise RuntimeError(f"Cannot map Jacobian rows={row_count} to links={link_count}")
        if not 0 <= row < row_count:
            raise RuntimeError(f"Link index {link_index} maps to invalid row {row}")
        return row

    @staticmethod
    def _get_limits(articulation: Articulation) -> tuple[np.ndarray, np.ndarray]:
        limits = articulation.get_dof_limits()
        if isinstance(limits, tuple) and len(limits) == 2:
            return (
                _to_numpy(limits[0]).reshape(-1).astype(np.float64),
                _to_numpy(limits[1]).reshape(-1).astype(np.float64),
            )
        array = _to_numpy(limits)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or array.shape[1] != 2:
            raise RuntimeError(f"Unexpected DOF limit shape: {array.shape}")
        return array[:, 0].astype(np.float64), array[:, 1].astype(np.float64)

    @staticmethod
    def _world_pose(prim: RigidPrim | XformPrim) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations = prim.get_world_poses()
        position = _to_numpy(positions).reshape(-1, 3)[0].astype(np.float64)
        orientation = _normalize_quaternion(
            _to_numpy(orientations).reshape(-1, 4)[0].astype(np.float64)
        )
        if not np.all(np.isfinite(position)):
            raise RuntimeError("World pose contains a non-finite position")
        return position, orientation

    def prepare_before_play(self) -> None:
        robot_prim = self.stage.GetPrimAtPath(ROBOT_PATH)
        clothing_root = self.stage.GetPrimAtPath(CLOTHING_ROOT_PATH)
        clothing_surface = self.stage.GetPrimAtPath(CLOTHING_SURFACE_PATH)
        support = self.stage.GetPrimAtPath(SUPPORT_PATH)
        belt = self.stage.GetPrimAtPath(BELT_PATH)
        for label, prim, path in (
            ("robot", robot_prim, ROBOT_PATH),
            ("clothing root", clothing_root, CLOTHING_ROOT_PATH),
            ("clothing surface", clothing_surface, CLOTHING_SURFACE_PATH),
            ("support", support, SUPPORT_PATH),
            ("belt", belt, BELT_PATH),
        ):
            if not prim.IsValid():
                raise RuntimeError(f"Missing {label} Prim: {path}")
        if not clothing_surface.IsA(UsdGeom.Mesh) or not clothing_surface.IsActive():
            raise RuntimeError(f"Clothing surface must be one active Mesh: {CLOTHING_SURFACE_PATH}")

        restore_cached_rest_mesh(self.stage, CLOTHING_SURFACE_PATH)
        strip_clothing_physx(
            self.stage,
            clothing_root_path=CLOTHING_ROOT_PATH,
            clothing_surface_path=CLOTHING_SURFACE_PATH,
            volume_proxy_path=CLOTHING_VOLUME_PROXY_PATH,
        )

        self.clothing_bounds = get_world_bounds(clothing_root)
        self.support_bounds = get_world_bounds(support)
        self.belt_bounds = get_world_bounds(belt)
        self.geometry = calculate_grasp_geometry(
            self.clothing_bounds, self.support_bounds, centered=True
        )
        self.grasp_point = self.geometry["grasp_point"].copy()
        self.pregrasp_point = self.geometry["pregrasp_point"].copy()
        self.attachment_center = self.geometry["attachment_center"].copy()
        self.support_center = self.geometry["support_center"].copy()

        remove_owned_runtime_prims(self.stage)
        try:
            self.saved_collisions.extend(
                disable_gripper_collisions(self.stage, LEFT_GRIPPER_PATH, "Left")
            )
            self.saved_collisions.extend(
                disable_gripper_collisions(self.stage, RIGHT_GRIPPER_PATH, "Right")
            )
            self.attachment_active = False
        except Exception:
            restore_collision_attributes(self.stage, self.saved_collisions)
            self.saved_collisions.clear()
            remove_owned_runtime_prims(self.stage)
            raise

        _log("[ProM6ClothingDrag] Prepared before first Play")
        _log(f"  robot: {ROBOT_PATH}")
        _log(f"  surface: {CLOTHING_SURFACE_PATH}")
        _log(f"  support: {SUPPORT_PATH}")
        _log(f"  clothing bounds: min={self.clothing_bounds[0]}, max={self.clothing_bounds[1]}")
        _log(f"  support bounds: min={self.support_bounds[0]}, max={self.support_bounds[1]}")
        _log(f"  belt: {BELT_PATH}")
        _log(f"  belt bounds: min={self.belt_bounds[0]}, max={self.belt_bounds[1]}")
        _log(f"  grasp point: {_array_string(self.grasp_point)}")
        _log(f"  grasp patch (legacy attachment_center): {_array_string(self.attachment_center)}")
        _log(f"  platform center: {_array_string(self.support_center)}")
        _log(f"  center tolerance: {CENTER_RELEASE_TOLERANCE_M:.3f} m")
        _log(f"  disabled gripper colliders: {len(self.saved_collisions)}")
        _log("  timing configuration (seconds):")
        for name in self.timing.__dataclass_fields__:
            _log(f"    {name}={float(getattr(self.timing, name)):.6g}")

    def _sync_newton_jaws(self) -> None:
        if self.newton_world is None or self.left_gripper_index is None:
            return

        q = self._current_dof_positions()
        left_q = float(q[self.left_gripper_index])
        right_index = self.right_gripper_index
        right_q = float(q[right_index]) if right_index is not None else 0.0
        left_closed = float(
            np.clip(
                GRIPPER_CLOSED_POSITION_RAD,
                self.left_gripper_lower_limit,
                self.left_gripper_upper_limit,
            )
        )
        right_closed = left_closed
        if right_index is not None:
            right_closed = float(
                np.clip(
                    GRIPPER_CLOSED_POSITION_RAD,
                    self.right_gripper_lower_limit,
                    self.right_gripper_upper_limit,
                )
            )
        self.newton_world.set_jaw_separations(
            jaw_separation_m(left_q, closed_rad=left_closed),
            jaw_separation_m(right_q, closed_rad=right_closed),
        )
        self.newton_world.set_finger_grip_scale(
            left_q / max(left_closed, 1.0e-6),
            right_q / max(right_closed, 1.0e-6),
        )

    def _teleport_pad_retract(self, retract_m: float) -> None:
        """松爪时瞬移指垫并冻结 VBD 几帧,避免扫掠把袋拉走."""
        if self.newton_world is None:
            return
        self.newton_world.set_pad_retract(retract_m)
        self._newton_hold_frames = max(self._newton_hold_frames, PAD_RETRACT_HOLD_FRAMES)
        _log(
            "[ProM6ClothingDrag] Pad retract teleport: "
            f"retract={retract_m:.3f} m, hold_frames={self._newton_hold_frames}, "
            f"{self.newton_world.finger_debug()}"
        )

    def bind_newton(self, world: NewtonSortingWorld, *, min_lift_height_m: float) -> None:
        """Play 前绑上 Newton 软袋世界;接触抓取在物理回调里步进."""
        self.newton_world = world
        self.min_lift_height_m = float(min_lift_height_m)
        _log(
            "[ProM6ClothingDrag] Newton contact world bound: "
            f"particles={world.model.particle_count}, "
            f"min_lift={self.min_lift_height_m:.3f} m"
        )

    def _current_dof_positions(self) -> np.ndarray:
        return _to_numpy(self.robot.get_dof_positions()).reshape(-1).astype(np.float64)

    def _build_arm_kinematics(
        self,
        *,
        side: str,
        arm_indices: list[int],
        arm_dof_paths: list[str],
        start_q: np.ndarray,
        base_link_path: str,
        base_link_index: int,
        base_jacobian_row: int,
        tool_center_path: str,
    ) -> tuple[UsdArmKinematics, np.ndarray, np.ndarray]:
        base_position, base_orientation = self._world_pose(RigidPrim(base_link_path))
        tool_position, tool_orientation = self._world_pose(XformPrim(tool_center_path))
        tcp_offset_local = _local_offset_from_world_point(
            base_position,
            base_orientation,
            tool_position,
        )
        tcp_adapter = np.linalg.inv(_pose_matrix(base_position, base_orientation)) @ _pose_matrix(
            tool_position, tool_orientation
        )
        live_jacobian = self._live_tcp_jacobian(
            base_link_index,
            base_jacobian_row,
            arm_indices,
            base_position,
            base_orientation,
            tool_position,
        )
        model = UsdArmKinematics.from_stage(
            self.stage,
            arm_dof_paths,
            start_q,
            base_link_path,
            tcp_adapter,
            live_jacobian,
        )
        predicted_position, predicted_orientation, predicted_jacobian, _ = (
            model.forward_and_jacobian(start_q)
        )
        model_error = _pose_error(
            predicted_position,
            predicted_orientation,
            tool_position,
            tool_orientation,
        )
        jacobian_error = float(np.max(np.abs(predicted_jacobian - live_jacobian)))
        _log(
            f"[ProM6ClothingDrag][{side} IK model] validation: "
            f"position={1000.0 * np.linalg.norm(model_error[:3]):.3f} mm, "
            f"orientation={math.degrees(np.linalg.norm(model_error[3:])):.3f} deg, "
            f"jacobian_max_abs={jacobian_error:.6f}, "
            f"tcp_offset_local={_array_string(tcp_offset_local, 6)}"
        )
        if np.linalg.norm(model_error[:3]) > IK_MODEL_POSITION_TOLERANCE_M:
            raise RuntimeError(f"{side} USD FK position validation failed")
        if np.linalg.norm(model_error[3:]) > IK_MODEL_ORIENTATION_TOLERANCE_RAD:
            raise RuntimeError(f"{side} USD FK orientation validation failed")
        if jacobian_error > IK_MODEL_JACOBIAN_TOLERANCE:
            raise RuntimeError(f"{side} USD Jacobian validation failed")
        return model, tcp_offset_local, tcp_adapter

    def _initialize_after_play(self) -> None:
        dof_names = tuple(str(value) for value in self.robot.dof_names)
        try:
            dof_paths = tuple(str(value) for value in self.robot.dof_paths[0])
            link_paths = tuple(str(value) for value in self.robot.link_paths[0])
            jacobian_shape = tuple(int(value) for value in self.robot.jacobian_matrix_shape)
        except (AttributeError, IndexError, TypeError) as error:
            raise RuntimeError("Articulation structure is unavailable after Play") from error
        if len(jacobian_shape) != 3 or jacobian_shape[1] != 6:
            raise RuntimeError(f"Unexpected Jacobian metadata: {jacobian_shape}")

        self.left_arm_indices = [
            self._single_exact_index(dof_names, f"Joint{number}_L", "left arm DOF")
            for number in range(1, 8)
        ]
        self.right_arm_indices = [
            self._single_exact_index(dof_names, f"Joint{number}_R", "right arm DOF")
            for number in range(1, 8)
        ]
        self.left_gripper_index = self._single_suffix_index(
            dof_paths, "/robotiq_left/joints/finger_joint", "left gripper DOF"
        )
        self.right_gripper_index = self._single_suffix_index(
            dof_paths, "/robotiq_right/joints/finger_joint", "right gripper DOF"
        )
        self.left_base_link_index = self._single_suffix_index(
            link_paths,
            "/robotiq_left/robotiq_arg2f_base_link",
            "left Robotiq base link",
        )
        self.left_base_link_path = link_paths[self.left_base_link_index]
        self.left_tool_center_path = f"{self.left_base_link_path}/ToolCenter_L"
        if not self.stage.GetPrimAtPath(self.left_tool_center_path).IsValid():
            raise RuntimeError(f"Missing left ToolCenter: {self.left_tool_center_path}")
        self.left_base_jacobian_row = self._jacobian_row(
            self.left_base_link_index, len(link_paths), jacobian_shape[0]
        )
        self.left_arm_dof_paths = [dof_paths[index] for index in self.left_arm_indices]
        self.right_base_link_index = self._single_suffix_index(
            link_paths,
            "/robotiq_right/robotiq_arg2f_base_link",
            "right Robotiq base link",
        )
        self.right_base_link_path = link_paths[self.right_base_link_index]
        self.right_tool_center_path = f"{self.right_base_link_path}/ToolCenter_R"
        if not self.stage.GetPrimAtPath(self.right_tool_center_path).IsValid():
            raise RuntimeError(f"Missing right ToolCenter: {self.right_tool_center_path}")
        self.right_base_jacobian_row = self._jacobian_row(
            self.right_base_link_index, len(link_paths), jacobian_shape[0]
        )
        self.right_arm_dof_paths = [dof_paths[index] for index in self.right_arm_indices]

        lower, upper = self._get_limits(self.robot)
        if lower.size != len(dof_names) or upper.size != len(dof_names):
            raise RuntimeError("DOF limit count does not match the articulation")
        self.left_arm_lower_limits = lower[self.left_arm_indices].copy()
        self.left_arm_upper_limits = upper[self.left_arm_indices].copy()
        self.right_arm_lower_limits = lower[self.right_arm_indices].copy()
        self.right_arm_upper_limits = upper[self.right_arm_indices].copy()
        self.left_gripper_lower_limit = float(lower[self.left_gripper_index])
        self.left_gripper_upper_limit = float(upper[self.left_gripper_index])
        self.right_gripper_lower_limit = float(lower[self.right_gripper_index])
        self.right_gripper_upper_limit = float(upper[self.right_gripper_index])
        _log(
            "[ProM6ClothingDrag] Gripper limits: "
            f"left=[{self.left_gripper_lower_limit:.4f}, {self.left_gripper_upper_limit:.4f}], "
            f"right=[{self.right_gripper_lower_limit:.4f}, {self.right_gripper_upper_limit:.4f}]"
        )
        for label, target, target_lower, target_upper in (
            (
                "left ready",
                LEFT_READY_JOINTS,
                self.left_arm_lower_limits,
                self.left_arm_upper_limits,
            ),
            (
                "right ready",
                RIGHT_READY_JOINTS,
                self.right_arm_lower_limits,
                self.right_arm_upper_limits,
            ),
        ):
            if np.any(target < target_lower) or np.any(target > target_upper):
                raise RuntimeError(
                    f"Configured {label} posture violates live joint limits: "
                    f"target={target}, lower={target_lower}, upper={target_upper}"
                )

        current = self._current_dof_positions()
        self.left_start_q = current[self.left_arm_indices].copy()
        self.left_commanded_q = self.left_start_q.copy()
        self.right_start_q = current[self.right_arm_indices].copy()
        self.right_commanded_q = self.right_start_q.copy()

        self.usd_kinematics, self.tcp_offset_local, self.tcp_adapter = self._build_arm_kinematics(
            side="left",
            arm_indices=self.left_arm_indices,
            arm_dof_paths=self.left_arm_dof_paths,
            start_q=self.left_start_q,
            base_link_path=self.left_base_link_path,
            base_link_index=self.left_base_link_index,
            base_jacobian_row=self.left_base_jacobian_row,
            tool_center_path=self.left_tool_center_path,
        )
        (
            self.right_usd_kinematics,
            self.right_tcp_offset_local,
            self.right_tcp_adapter,
        ) = self._build_arm_kinematics(
            side="right",
            arm_indices=self.right_arm_indices,
            arm_dof_paths=self.right_arm_dof_paths,
            start_q=self.right_start_q,
            base_link_path=self.right_base_link_path,
            base_link_index=self.right_base_link_index,
            base_jacobian_row=self.right_base_jacobian_row,
            tool_center_path=self.right_tool_center_path,
        )

        # The initial jump is intentional: calibrate the USD model against the
        # measured pre-command pose first, then start the task with both arms in
        # the configured safe ready configuration.
        self._write_left_q(LEFT_READY_JOINTS)
        self._write_right_ready()
        self.left_start_q = LEFT_READY_JOINTS.copy()
        self.right_start_q = RIGHT_READY_JOINTS.copy()
        self.right_commanded_q = RIGHT_READY_JOINTS.copy()
        _log(
            "[ProM6ClothingDrag] Immediate dual-arm ready initialized: "
            f"left_indices={self.left_arm_indices}, "
            f"left_q={_array_string(LEFT_READY_JOINTS)}, "
            f"right_indices={self.right_arm_indices}, "
            f"right_q={_array_string(RIGHT_READY_JOINTS)}"
        )

        self._cached_nodes = None
        if self.newton_world is None:
            raise RuntimeError("Newton sorting world was not bound before Play")
        self.newton_world.set_tcp_pose_fn("left", self._current_tcp)
        self.newton_world.set_tcp_pose_fn("right", self._current_right_tcp)
        self.newton_world.reset_kinematic_history()
        self._newton_hold_frames = 6
        self._deformable_nodes()
        self.replay_recorder = SuccessfulOperationRecorder(
            self.stage,
            self.robot,
            dof_names,
            dof_paths,
            nodal_position_fn=self.newton_world.surface_positions,
            nodal_velocity_fn=self._newton_surface_velocities,
        )
        _log(
            "[ProM6ClothingDrag][cache] Recording full state in memory; "
            f"successful runs will be archived in {REPLAY_CACHE_DIRECTORY}"
        )
        self.initialized = True
        self._announce_phase()

    def _live_tcp_jacobian(
        self,
        base_link_index: int,
        base_jacobian_row: int,
        arm_indices: list[int],
        base_position: np.ndarray,
        base_orientation: np.ndarray,
        tcp_position: np.ndarray,
    ) -> np.ndarray:
        result = self.robot.get_link_coms(link_indices=[base_link_index])
        raw_positions = result[0] if isinstance(result, tuple) else result
        com_local = _to_numpy(raw_positions).reshape(3).astype(np.float64)
        base_com_world = base_position + (_quaternion_rotation_matrix(base_orientation) @ com_local)
        jacobians = _to_numpy(self.robot.get_jacobian_matrices()).astype(np.float64)
        if jacobians.ndim == 4:
            if jacobians.shape[0] != 1:
                raise RuntimeError(f"Expected one articulation, got {jacobians.shape}")
            jacobians = jacobians[0]
        if jacobians.ndim != 3 or jacobians.shape[1] != 6:
            raise RuntimeError(f"Unexpected live Jacobian shape: {jacobians.shape}")
        base_jacobian = jacobians[base_jacobian_row]
        tcp_jacobian = base_jacobian[:, arm_indices].copy()
        offset_world = tcp_position - base_com_world
        tcp_jacobian[:3] -= _skew(offset_world) @ tcp_jacobian[3:6]
        return tcp_jacobian

    def _announce_phase(self) -> None:
        _log(f"[ProM6ClothingDrag] Phase {self.phase.name}: {PHASE_LABELS[self.phase]}")

    def _set_phase(self, phase: Phase) -> None:
        self.phase = phase
        self.phase_time = 0.0
        self.phase_frame = 0
        self.stable_time = 0.0
        self._right_grasp_z_refreshed = False
        self._right_grasp_dip_m = 0.0
        self._left_grasp_dip_m = 0.0
        self.left_retract_target = None
        self._left_retract_waypoints = []
        self._left_retract_i = 0
        self._left_retract_hold_s = 0.0
        self._left_return_clear_target = None
        self._left_return_cartesian_done = False
        self._left_return_joint_elapsed = 0.0
        self._left_return_stall_s = 0.0
        self._left_close_xy_refreshed = False
        self._left_close_elapsed = 0.0
        self._right_close_elapsed = 0.0
        self._announce_phase()

    def _current_tcp(self) -> tuple[np.ndarray, np.ndarray]:
        if self.usd_kinematics is None or self.left_commanded_q is None:
            raise RuntimeError("Authoritative left-arm FK is unavailable")
        position, orientation, _, _ = self.usd_kinematics.forward_and_jacobian(
            self.left_commanded_q
        )
        return position, orientation

    def _current_right_tcp(self) -> tuple[np.ndarray, np.ndarray]:
        if self.right_usd_kinematics is None or self.right_commanded_q is None:
            raise RuntimeError("Authoritative right-arm FK is unavailable")
        position, orientation, _, _ = self.right_usd_kinematics.forward_and_jacobian(
            self.right_commanded_q
        )
        return position, orientation

    def _write_left_q(self, target: np.ndarray) -> None:
        if self.left_arm_lower_limits is None or self.left_arm_upper_limits is None:
            raise RuntimeError("Left-arm limits are unavailable")
        value = np.clip(
            np.asarray(target, dtype=np.float64).reshape(7),
            self.left_arm_lower_limits,
            self.left_arm_upper_limits,
        )
        if self.left_commanded_q is not None:
            self.last_joint_step = float(np.linalg.norm(value - self.left_commanded_q))
        self.robot.set_dof_positions(value[None, :], dof_indices=self.left_arm_indices)
        self.robot.set_dof_velocities(
            np.zeros((1, 7), dtype=np.float64),
            dof_indices=self.left_arm_indices,
        )
        self.robot.set_dof_position_targets(value[None, :], dof_indices=self.left_arm_indices)
        self.left_commanded_q = value.copy()

    def _write_right_ready(self) -> None:
        self._write_right_q(RIGHT_READY_JOINTS)

    def _write_right_q(self, target: np.ndarray) -> None:
        if (
            self.right_arm_lower_limits is None
            or self.right_arm_upper_limits is None
            or len(self.right_arm_indices) != 7
        ):
            raise RuntimeError("Right-arm ready hold is unavailable")
        value = np.clip(
            np.asarray(target, dtype=np.float64).reshape(7),
            self.right_arm_lower_limits,
            self.right_arm_upper_limits,
        )
        if self.right_commanded_q is not None:
            self.last_joint_step = float(np.linalg.norm(value - self.right_commanded_q))
        self.robot.set_dof_positions(value[None, :], dof_indices=self.right_arm_indices)
        self.robot.set_dof_velocities(
            np.zeros((1, 7), dtype=np.float64),
            dof_indices=self.right_arm_indices,
        )
        self.robot.set_dof_position_targets(value[None, :], dof_indices=self.right_arm_indices)
        self.right_commanded_q = value.copy()

    def _slew_joint_target(self, desired: np.ndarray, dt: float) -> float:
        if self.left_commanded_q is None:
            raise RuntimeError("Left-arm command state is unavailable")
        max_step = LEFT_MAX_JOINT_SPEED_RAD_S * dt
        next_q = self.left_commanded_q + np.clip(
            np.asarray(desired, dtype=np.float64).reshape(7) - self.left_commanded_q,
            -max_step,
            max_step,
        )
        step_norm = float(np.linalg.norm(next_q - self.left_commanded_q))
        self._write_left_q(next_q)
        return step_norm

    def _slew_right_joint_target(self, desired: np.ndarray, dt: float) -> float:
        if self.right_commanded_q is None:
            raise RuntimeError("Right-arm command state is unavailable")
        max_step = RIGHT_MAX_JOINT_SPEED_RAD_S * dt
        next_q = self.right_commanded_q + np.clip(
            np.asarray(desired, dtype=np.float64).reshape(7) - self.right_commanded_q,
            -max_step,
            max_step,
        )
        step_norm = float(np.linalg.norm(next_q - self.right_commanded_q))
        self._write_right_q(next_q)
        return step_norm

    def _command_pose(
        self,
        target_position: np.ndarray,
        dt: float,
    ) -> tuple[float, float, float]:
        if self.usd_kinematics is None or self.left_commanded_q is None:
            raise RuntimeError("Online IK is unavailable")
        current_position, current_orientation, jacobian, _ = (
            self.usd_kinematics.forward_and_jacobian(self.left_commanded_q)
        )
        error = _pose_error(
            current_position,
            current_orientation,
            target_position,
            self.grasp_orientation,
        )
        delta = _dls_joint_delta(
            jacobian,
            error,
            damping=ONLINE_DLS_DAMPING,
            maximum_step=IK_MAX_JOINT_STEP_RAD,
        )
        max_step = LEFT_MAX_JOINT_SPEED_RAD_S * dt
        delta = np.clip(delta, -max_step, max_step)
        next_q = self.left_commanded_q + delta
        self._write_left_q(next_q)
        self.last_tcp_error = float(np.linalg.norm(error[:3]))
        return (
            self.last_tcp_error,
            float(np.linalg.norm(error[3:])),
            float(np.linalg.norm(delta)),
        )

    def _command_right_pose(
        self,
        target_position: np.ndarray,
        dt: float,
    ) -> tuple[float, float, float]:
        if self.right_usd_kinematics is None or self.right_commanded_q is None:
            raise RuntimeError("Right-arm online IK is unavailable")
        current_position, current_orientation, jacobian, _ = (
            self.right_usd_kinematics.forward_and_jacobian(self.right_commanded_q)
        )
        error = _pose_error(
            current_position,
            current_orientation,
            target_position,
            self.grasp_orientation,
        )
        delta = _dls_joint_delta(
            jacobian,
            error,
            damping=ONLINE_DLS_DAMPING,
            maximum_step=IK_MAX_JOINT_STEP_RAD,
        )
        max_step = RIGHT_MAX_JOINT_SPEED_RAD_S * dt
        delta = np.clip(delta, -max_step, max_step)
        self._write_right_q(self.right_commanded_q + delta)
        self.last_tcp_error = float(np.linalg.norm(error[:3]))
        return (
            self.last_tcp_error,
            float(np.linalg.norm(error[3:])),
            float(np.linalg.norm(delta)),
        )

    def _set_gripper_target(self, position: float, *, snap: bool = False) -> None:
        if self.left_gripper_index is None:
            raise RuntimeError("Left gripper DOF is unavailable")
        target = float(
            np.clip(
                position,
                self.left_gripper_lower_limit,
                self.left_gripper_upper_limit,
            )
        )
        value = np.array([[target]], dtype=np.float64)
        if snap:
            self.robot.set_dof_positions(value, dof_indices=[self.left_gripper_index])
            self.robot.set_dof_velocities(
                np.zeros((1, 1), dtype=np.float64),
                dof_indices=[self.left_gripper_index],
            )
        self.robot.set_dof_position_targets(value, dof_indices=[self.left_gripper_index])

    def _set_right_gripper_target(self, position: float, *, snap: bool = False) -> None:
        if self.right_gripper_index is None:
            raise RuntimeError("Right gripper DOF is unavailable")
        target = float(
            np.clip(
                position,
                self.right_gripper_lower_limit,
                self.right_gripper_upper_limit,
            )
        )
        value = np.array([[target]], dtype=np.float64)
        if snap:
            self.robot.set_dof_positions(value, dof_indices=[self.right_gripper_index])
            self.robot.set_dof_velocities(
                np.zeros((1, 1), dtype=np.float64),
                dof_indices=[self.right_gripper_index],
            )
        self.robot.set_dof_position_targets(value, dof_indices=[self.right_gripper_index])

    def _deformable_nodes(self) -> np.ndarray:
        if self._cached_nodes is not None:
            return self._cached_nodes
        if self.newton_world is None:
            raise RuntimeError("Newton sorting world is unavailable")
        positions = np.asarray(self.newton_world.surface_positions(), dtype=np.float64)
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise RuntimeError(f"Invalid Newton surface positions: {positions.shape}")
        span = np.max(positions, axis=0) - np.min(positions, axis=0)
        if np.any(span > MAX_DEFORMABLE_SPAN_M):
            raise RuntimeError(f"Deformable surface is unstable: span={span}")
        self._cached_nodes = positions
        return positions

    def _newton_surface_velocities(self) -> np.ndarray:
        if self.newton_world is None:
            raise RuntimeError("Newton sorting world is unavailable")
        return np.asarray(
            self.newton_world.particle_velocities()[: self.newton_world.surface_count],
            dtype=np.float64,
        )

    def _parcel_com(self) -> np.ndarray:
        if self.newton_world is None:
            raise RuntimeError("Newton sorting world is unavailable")
        return np.asarray(self.newton_world.com(), dtype=np.float64).reshape(3)

    def _require_lifted(self, baseline_z: float | None, label: str) -> None:
        if baseline_z is None:
            raise RuntimeError(f"{label} lift has no COM baseline")
        com = self._parcel_com()
        lifted = float(com[2] - baseline_z)
        contacts = self.newton_world.contact_breakdown() if self.newton_world is not None else {}
        _log(f"[ProM6ClothingDrag] {label} lift COM delta_z={lifted:.4f} m contacts={contacts}")
        if lifted < self.min_lift_height_m:
            raise RuntimeError(
                f"{label} contact grasp failed to lift the parcel: "
                f"delta_z={lifted:.4f} m < {self.min_lift_height_m:.4f} m"
            )
        if label == "left":
            world_n = int(contacts.get("world", 0) or 0)
            finger_n = int(contacts.get("finger", 0) or 0)
            if world_n > LEFT_LIFT_MAX_WORLD_CONTACTS:
                raise RuntimeError(
                    "left contact grasp lifted a flap while the parcel stayed on "
                    f"the infeed: world={world_n}, finger={finger_n}, "
                    f"delta_z={lifted:.4f} m"
                )

    def _apply_live_grasp_geometry(
        self, minimum: np.ndarray, maximum: np.ndarray, *, preserve_height: bool = False
    ) -> None:
        """按 Newton AABB 重算左抓点;闭爪时可保留已经验证可达的高度."""
        if self.support_bounds is None:
            raise RuntimeError("Support bounds are unavailable after settle")
        self.clothing_bounds = (
            np.asarray(minimum, dtype=np.float64).reshape(3).copy(),
            np.asarray(maximum, dtype=np.float64).reshape(3).copy(),
        )
        self.geometry = calculate_grasp_geometry(
            self.clothing_bounds, self.support_bounds, centered=True
        )
        for key in ("grasp_point", "pregrasp_point", "attachment_center"):
            self.geometry[key] += self._left_grasp_offset
            if preserve_height:
                self.geometry[key][2] = getattr(self, key)[2]
        self.grasp_point = self.geometry["grasp_point"].copy()
        self.pregrasp_point = self.geometry["pregrasp_point"].copy()
        self.attachment_center = self.geometry["attachment_center"].copy()
        self.support_center = self.geometry["support_center"].copy()
        _log(
            "[ProM6ClothingDrag] Live Newton grasp: "
            f"grasp={_array_string(self.grasp_point)}, "
            f"pregrasp={_array_string(self.pregrasp_point)}, "
            f"patch={_array_string(self.attachment_center)}"
        )

    def _node_aabb(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nodes = self._deformable_nodes()
        minimum = np.min(nodes, axis=0)
        maximum = np.max(nodes, axis=0)
        return minimum, maximum, 0.5 * (minimum + maximum)

    def _capture_replay_frame(self, dt: float) -> None:
        if self.replay_recorder is None:
            raise RuntimeError("Successful-operation replay recorder is unavailable")
        anchor_position = None
        if self.attachment_active:
            if self.anchor is None:
                raise RuntimeError("Active attachment has no anchor during recording")
            anchor_position = get_xform_world_translation(self.anchor)
        self.replay_recorder.capture(
            dt,
            self.phase,
            attachment_active=self.attachment_active,
            anchor_position=anchor_position,
        )

    def _log_progress(self) -> None:
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES != 0:
            return
        _, _, center = self._node_aabb()
        center_error = _xy_center_error(center, self.support_center)
        tcp_error = (
            "n/a" if not math.isfinite(self.last_tcp_error) else f"{self.last_tcp_error:.5f} m"
        )
        _log(
            "[ProM6ClothingDrag][status] "
            f"phase={self.phase.name}, tcp_error={tcp_error}, "
            f"joint_step={self.last_joint_step:.5f} rad, "
            f"center_error={center_error:.4f} m"
        )

    def _update_anchor_from_actual_tcp(self) -> None:
        if not self.attachment_active:
            return
        if self.anchor is None or self.grasp_point is None or self.attachment_center is None:
            raise RuntimeError("Attachment anchor geometry is incomplete")
        tcp_position, _ = self._current_tcp()
        set_xform_world_translation(
            self.anchor,
            self.attachment_center + (tcp_position - self.grasp_point),
        )

    def _update_shared_anchor_from_right_tcp(self) -> None:
        if not self.attachment_active:
            return
        if (
            self.anchor is None
            or self.right_handoff_anchor_position is None
            or self.right_handoff_tcp_position is None
        ):
            raise RuntimeError("Shared attachment handoff geometry is incomplete")
        tcp_position, _ = self._current_right_tcp()
        set_xform_world_translation(
            self.anchor,
            self.right_handoff_anchor_position + (tcp_position - self.right_handoff_tcp_position),
        )

    def _release_attachment(self) -> None:
        if self.attachment_active:
            prim = self.stage.GetPrimAtPath(ATTACHMENT_PATH)
            if prim.IsValid():
                self.stage.RemovePrim(ATTACHMENT_PATH)
            self.attachment_active = False
            _log("[ProM6ClothingDrag] Shared surface attachment removed")
        anchor_prim = self.stage.GetPrimAtPath(ANCHOR_PATH)
        if anchor_prim.IsValid():
            self.stage.RemovePrim(ANCHOR_PATH)
        self.anchor = None

    def _step_settle(self) -> None:
        self._write_left_q(self.left_commanded_q)
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        if self.phase_time >= self.timing.settle_seconds:
            minimum, maximum, center = self._node_aabb()
            self.initial_node_center = center.copy()
            self.initial_node_height = float(maximum[2] - minimum[2])
            _log(
                "[ProM6ClothingDrag] Settled nodes: "
                f"min={_array_string(minimum)}, max={_array_string(maximum)}, "
                f"center={_array_string(center)}, "
                f"com={_array_string(self._parcel_com())}"
            )
            self._apply_live_grasp_geometry(minimum, maximum)
            self._set_phase(Phase.WAITING)

    def _grasp_offsets(
        self, minimum: np.ndarray, maximum: np.ndarray, side: str
    ) -> list[np.ndarray]:
        """在近中心区域随机采样；关闭随机化时沿用中心优先候选."""
        fractions = getattr(self, "grasp_jitter_fraction_xy", (0.0, 0.0))
        if not any(fractions):
            return _center_out_grasp_offsets(0.5 * (maximum - minimum))
        candidates = sample_grasp_offsets(
            minimum,
            maximum,
            fractions,
            self.grasp_random_candidates,
            self.grasp_rngs[side],
            min_distance_m=self.grasp_min_distance_m,
            previous_offset=self.previous_grasp_offsets.get(side),
        )
        if not candidates:
            raise RuntimeError(f"{side} 无法在主体范围内产生足够差异的抓点，请调整扰动范围")
        return candidates

    def _screen_random_grasp(
        self,
        kinematics: UsdArmKinematics,
        pregrasp: np.ndarray,
        waypoints: tuple[np.ndarray, ...],
        lower: np.ndarray,
        upper: np.ndarray,
        preferred: np.ndarray,
        ready: np.ndarray,
    ) -> bool:
        """先用有限种子筛掉不可达随机点；通过后仍执行完整 42 种子验收."""
        if not any(getattr(self, "grasp_jitter_fraction_xy", (0.0, 0.0))):
            return True
        seeds = _build_ik_candidate_seeds(lower, upper, preferred, ready)[:6]
        return any(
            _ik_waypoints_reachable(
                kinematics,
                seed,
                (pregrasp, *waypoints),
                self.grasp_orientation,
                lower + IK_JOINT_LIMIT_MARGIN_RAD,
                upper - IK_JOINT_LIMIT_MARGIN_RAD,
            )
            for seed in seeds
        )

    def _step_search_pregrasp_ik(self) -> None:
        if (
            self.usd_kinematics is None
            or self.pregrasp_point is None
            or self.left_arm_lower_limits is None
            or self.left_arm_upper_limits is None
            or self.grasp_point is None
            or self.clothing_bounds is None
        ):
            raise RuntimeError("Pregrasp IK inputs are unavailable")
        self._write_left_q(self.left_commanded_q)
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        # A jittered bag center can be beyond the arm's reach. Search the bag body,
        # and verify descent/lift as well as the elevated pregrasp pose.
        offsets = self._grasp_offsets(*self.clothing_bounds, "left")
        initial_grasp = self.grasp_point.copy()
        initial_pregrasp = self.pregrasp_point.copy()
        last_error: RuntimeError | None = None
        for offset in offsets:
            grasp = initial_grasp + offset
            pregrasp = initial_pregrasp + offset
            waypoints = (grasp, grasp + np.array([0.0, 0.0, self.left_lift_height_m]))
            if not self._screen_random_grasp(
                self.usd_kinematics,
                pregrasp,
                waypoints,
                self.left_arm_lower_limits,
                self.left_arm_upper_limits,
                IK_PREFERRED_SEED,
                LEFT_IK_ELBOW_OUT_SEED,
            ):
                _log(f"[ProM6ClothingDrag][IK search] Random grasp unreachable: {grasp}")
                continue
            try:
                selected, diagnostics = solve_pregrasp_ik_42_seeds(
                    self.usd_kinematics,
                    pregrasp,
                    self.grasp_orientation,
                    self.left_arm_lower_limits,
                    self.left_arm_upper_limits,
                    IK_JOINT_LIMIT_MARGIN_RAD,
                    search_timeout_seconds=self.timing.ik_search_timeout_seconds,
                    ready_seed=LEFT_IK_ELBOW_OUT_SEED,
                    outward_xy=LEFT_ELBOW_OUTWARD_XY,
                    waypoint_positions=waypoints,
                )
            except RuntimeError as exc:
                last_error = exc
                _log(f"[ProM6ClothingDrag][IK search] Body grasp rejected: {grasp}: {exc}")
                continue
            self._left_grasp_offset = offset.copy()
            self._apply_live_grasp_geometry(*self.clothing_bounds)
            _log(f"[ProM6ClothingDrag][IK search] Reachable body offset: {offset}")
            break
        else:
            raise RuntimeError("No reachable left body grasp and lift path") from last_error
        self.pregrasp_joint_target = selected.copy()
        self.pregrasp_start_q = self.left_commanded_q.copy()
        _log(
            "[ProM6ClothingDrag][IK search] selected branch: "
            f"seed={int(diagnostics['seed_index'])}, "
            f"valid={int(diagnostics['valid_solution_count'])}/"
            f"{int(diagnostics['candidate_count'])}, "
            f"position={1000.0 * float(diagnostics['position_error']):.3f} mm, "
            f"orientation={math.degrees(float(diagnostics['orientation_error'])):.3f} deg, "
            f"elbow_out={float(diagnostics['elbow_outboard_m']):.3f} m, "
            f"q={_array_string(selected, 5)}"
        )
        self._set_phase(Phase.MOVE_TO_PREGRASP)

    def _step_joint_segment(
        self,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        dt: float,
        next_phase: Phase,
    ) -> None:
        progress = _timed_smoothstep(self.phase_time, duration)
        desired = (1.0 - progress) * start + progress * target
        step_norm = self._slew_joint_target(desired, dt)
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        joint_error = float(np.max(np.abs(self.left_commanded_q - target)))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                f"[ProM6ClothingDrag] {self.phase.name}: progress={progress:.3f}, "
                f"joint_error={joint_error:.5f} rad, step_norm={step_norm:.5f}"
            )
        if self.phase_time >= duration and joint_error <= 1.0e-4:
            self._set_phase(next_phase)

    def _step_approach(self, dt: float) -> None:
        progress = _timed_smoothstep(self.phase_time, self.timing.left_approach_seconds)
        target = self.pregrasp_point - np.array(
            [0.0, 0.0, progress * PREGRASP_CLEARANCE_M], dtype=np.float64
        )
        position_error, orientation_error, step_norm = self._command_pose(target, dt)
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] approach: "
                f"progress={progress:.3f}, position_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"step_norm={step_norm:.5f}"
            )
        if (
            self.phase_time >= self.timing.left_approach_seconds
            and position_error <= TCP_APPROACH_POSITION_TOLERANCE_M
            and orientation_error <= TCP_APPROACH_ORIENTATION_TOLERANCE_RAD
        ):
            self._set_phase(Phase.CLOSE_GRIPPER)

    def _step_close_gripper(self, dt: float) -> None:
        if not self._left_close_xy_refreshed:
            minimum, maximum, _ = self._node_aabb()
            # The descending pads compress the bag. Its new AABB top is not a
            # new approach height: chasing it can move below the reachable path
            # selected by IK. Track XY drift at the already validated height;
            # GRIP_SETTLE handles any further contact-seeking descent.
            self._apply_live_grasp_geometry(minimum, maximum, preserve_height=True)
            self._left_close_xy_refreshed = True
            _log(
                "[ProM6ClothingDrag] Left grasp refreshed before close: "
                f"grasp={_array_string(self.grasp_point)}"
            )
        position_error, orientation_error, _ = self._command_pose(self.grasp_point, dt)
        # Refreshing the deforming parcel can move the target after approach.
        # As for the right hand, converge with open jaws before timing closure;
        # the phase timeout still bounds an unreachable refreshed target.
        pose_ok = (
            position_error <= TCP_GRASP_POSITION_TOLERANCE_M
            and orientation_error <= TCP_GRASP_ORIENTATION_TOLERANCE_RAD
        )
        if not pose_ok:
            self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
            self._left_close_elapsed = 0.0
            return
        self._left_close_elapsed += dt
        fraction = _timed_smoothstep(self._left_close_elapsed, self.timing.left_close_seconds)
        target = GRIPPER_OPEN_POSITION_RAD + fraction * (
            GRIPPER_CLOSED_POSITION_RAD - GRIPPER_OPEN_POSITION_RAD
        )
        self._set_gripper_target(target, snap=True)
        if self._left_close_elapsed >= self.timing.left_close_seconds:
            self._set_phase(Phase.GRIP_SETTLE)

    def _step_grip_settle(self, dt: float) -> None:
        position_error, orientation_error, _ = self._command_pose(self.grasp_point, dt)
        self._set_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        gripper_error = abs(
            float(self._current_dof_positions()[self.left_gripper_index])
            - float(
                np.clip(
                    GRIPPER_CLOSED_POSITION_RAD,
                    self.left_gripper_lower_limit,
                    self.left_gripper_upper_limit,
                )
            )
        )
        pose_ok = (
            position_error <= TCP_GRASP_POSITION_TOLERANCE_M
            and orientation_error <= TCP_GRASP_ORIENTATION_TOLERANCE_RAD
        )
        finger_n = self._finger_contacts()
        if (
            pose_ok
            and finger_n < LEFT_MIN_FINGER_CONTACTS
            and self._left_grasp_dip_m < LEFT_GRASP_DIP_MAX_M
            and self.grasp_point is not None
        ):
            step = min(LEFT_GRASP_DIP_STEP_M, LEFT_GRASP_DIP_MAX_M - self._left_grasp_dip_m)
            self.grasp_point = np.asarray(self.grasp_point, dtype=np.float64).copy()
            self.grasp_point[2] -= step
            self._left_grasp_dip_m += step
            _log(
                "[ProM6ClothingDrag] Left grasp dip: "
                f"finger={finger_n}, dip={self._left_grasp_dip_m:.3f} m, "
                f"z={self.grasp_point[2]:.4f}"
            )
        holding = finger_n >= LEFT_MIN_FINGER_CONTACTS
        converged = pose_ok and gripper_error <= GRIPPER_POSITION_TOLERANCE_RAD and holding
        self.stable_time = _accumulate_stable_time(self.stable_time, converged, dt)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            debug = ""
            if finger_n < LEFT_MIN_FINGER_CONTACTS and self.newton_world is not None:
                debug = f", fingers={self.newton_world.finger_debug()}"
            _log(
                "[ProM6ClothingDrag] left grip settle: "
                f"tcp_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"gripper_error={gripper_error:.5f} rad, "
                f"finger={finger_n}, "
                f"stable={self.stable_time:.3f}/"
                f"{self.timing.left_grip_stable_seconds:.3f} s{debug}"
            )
        if (
            converged
            and self.phase_time >= self.timing.left_grip_settle_seconds
            and _stable_time_reached(self.stable_time, self.timing.left_grip_stable_seconds)
        ):
            tcp_position, _ = self._current_tcp()
            self.left_lift_start = tcp_position.copy()
            self.left_lift_target = tcp_position + np.array(
                [0.0, 0.0, self.left_lift_height_m], dtype=np.float64
            )
            self._left_lift_extra_height_m = 0.0
            self._left_lift_segment_started_at = 0.0
            self._left_lift_segment_duration = self.timing.left_lift_seconds
            self._left_lift_endpoint_hold_s = 0.0
            if self.newton_world is not None:
                q = self._current_dof_positions()
                _log(
                    "[ProM6ClothingDrag] Finger/bag before lift: "
                    f"left_finger_q={float(q[self.left_gripper_index]):.4f} rad, "
                    f"{self.newton_world.finger_debug()}"
                )
            _log(
                "[ProM6ClothingDrag] Separate vertical lift: "
                f"start={_array_string(self.left_lift_start)}, "
                f"target={_array_string(self.left_lift_target)}"
            )
            self.left_grasp_position = self.grasp_point.copy()
            self.left_lift_com_z = float(self._parcel_com()[2])
            self._set_phase(Phase.LEFT_LIFT)
        elif self.phase_time >= max(
            self.timing.left_grip_settle_seconds + 5.0,
            self.timing.left_grip_stable_seconds + 5.0,
        ):
            raise RuntimeError(
                "Grasp did not settle: "
                f"position={position_error:.5f}, orientation={orientation_error:.5f}, "
                f"gripper={gripper_error:.5f}, "
                f"finger={finger_n}"
            )

    def _step_left_lift(self, dt: float) -> None:
        if self.left_lift_start is None or self.left_lift_target is None:
            raise RuntimeError("Left lift geometry is unavailable")
        segment_time = self.phase_time - self._left_lift_segment_started_at
        progress = _timed_smoothstep(segment_time, self._left_lift_segment_duration)
        target = (1.0 - progress) * self.left_lift_start + progress * self.left_lift_target
        self._command_pose(target, dt)
        self._set_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_anchor_from_actual_tcp()
        # Gate against the final lift target using the newly commanded FK pose.
        tcp_position, tcp_orientation = self._current_tcp()
        error = _pose_error(
            tcp_position, tcp_orientation, self.left_lift_target, self.grasp_orientation
        )
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] left lift: "
                f"progress={progress:.3f}, position_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg"
            )
        reached = (
            segment_time >= self._left_lift_segment_duration
            and position_error <= TCP_LIFT_POSITION_TOLERANCE_M
            and orientation_error <= TCP_LIFT_ORIENTATION_TOLERANCE_RAD
        )
        self._left_lift_endpoint_hold_s = _accumulate_stable_time(
            self._left_lift_endpoint_hold_s, reached, dt
        )
        if reached and _stable_time_reached(
            self._left_lift_endpoint_hold_s, LEFT_LIFT_CLEARANCE_HOLD_S
        ):
            if self._extend_left_lift_for_clearance(tcp_position):
                return
            self._require_lifted(self.left_lift_com_z, "left")
            _, _, center = self._node_aabb()
            self.drag_start = tcp_position.copy()
            self.drag_target = self.drag_start.copy()
            self.drag_target[:2] += self.support_center[:2] - center[:2]
            _log(
                "[ProM6ClothingDrag] Frozen horizontal drag vector: "
                f"node_center={_array_string(center)}, "
                f"delta={_array_string(self.drag_target - self.drag_start)}, "
                f"target={_array_string(self.drag_target)}"
            )
            self._set_phase(Phase.DRAG)

    def _extend_left_lift_for_clearance(self, tcp_position: np.ndarray) -> bool:
        """Raise a held parcel in bounded increments until height and clearance pass."""
        if self.newton_world is None or self.left_lift_com_z is None:
            return False
        contacts = self.newton_world.contact_breakdown()
        world_n = int(contacts.get("world", 0) or 0)
        lifted = float(self._parcel_com()[2] - self.left_lift_com_z)
        if world_n <= LEFT_LIFT_MAX_WORLD_CONTACTS and lifted >= self.min_lift_height_m:
            return False
        finger_n = int(contacts.get("finger", 0) or 0)
        # A nearly lifted parcel may still need one clearance step to reach the COM gate.
        # Keep rejecting a slipped/stuck grasp; final acceptance still checks the full height.
        if (
            lifted < max(0.0, self.min_lift_height_m - LEFT_LIFT_CLEARANCE_STEP_M)
            or finger_n < LEFT_MIN_FINGER_CONTACTS
        ):
            return False
        remaining = min(
            LEFT_LIFT_MAX_EXTRA_HEIGHT_M - self._left_lift_extra_height_m,
            LEFT_LIFT_MAX_HEIGHT_M - self.left_lift_height_m - self._left_lift_extra_height_m,
        )
        if remaining <= 1.0e-6:
            return False  # Never bypass the contact check when the recovery budget is exhausted.
        increment = min(LEFT_LIFT_CLEARANCE_STEP_M, remaining)
        target = self.left_lift_target.copy()
        target[2] += increment
        if not _ik_waypoints_reachable(
            self.usd_kinematics,
            self.left_commanded_q,
            (target,),
            self.grasp_orientation,
            self.left_arm_lower_limits + IK_JOINT_LIMIT_MARGIN_RAD,
            self.left_arm_upper_limits - IK_JOINT_LIMIT_MARGIN_RAD,
        ):
            raise RuntimeError(
                "Left parcel still contacts the infeed; clearance lift is unreachable"
            )
        self.left_lift_start = tcp_position.copy()
        self.left_lift_target = target
        self._left_lift_extra_height_m += increment
        self._left_lift_segment_started_at = self.phase_time
        self._left_lift_segment_duration = max(
            0.4, self.timing.left_lift_seconds * increment / self.left_lift_height_m
        )
        self._left_lift_endpoint_hold_s = 0.0
        _log(
            "[ProM6ClothingDrag] Left lift recovery for height/clearance: "
            f"world={world_n}, finger={finger_n}, delta_z={lifted:.4f} m, "
            f"extra_height={self._left_lift_extra_height_m:.3f} m, "
            f"target={_array_string(target)}"
        )
        return True

    def _step_drag(self, dt: float) -> None:
        progress = _timed_smoothstep(self.phase_time, self.timing.left_drag_seconds)
        target = (1.0 - progress) * self.drag_start + progress * self.drag_target
        position_error, orientation_error, step_norm = self._command_pose(target, dt)
        self._set_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_anchor_from_actual_tcp()
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _, _, center = self._node_aabb()
            center_error = _xy_center_error(center, self.support_center)
            _log(
                "[ProM6ClothingDrag] drag: "
                f"progress={progress:.3f}, tcp_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"joint_step={step_norm:.5f}, center_error={center_error:.4f} m"
            )
        if (
            self.phase_time >= self.timing.left_drag_seconds
            and position_error <= DRAG_ENDPOINT_TOLERANCE_M
        ):
            _, _, center = self._node_aabb()
            center_error = _xy_center_error(center, self.support_center)
            if center_error > 0.25:
                raise RuntimeError(
                    "Left drag did not carry the parcel toward the platform: "
                    f"center_error={center_error:.4f} m"
                )
            self.correction_target = self.drag_target.copy()
            self.correction_origin = self.drag_target.copy()
            self._set_phase(Phase.CENTER_CORRECT)

    def _correct_center_xy(self, dt: float, target: np.ndarray) -> tuple[float, float]:
        _, _, center = self._node_aabb()
        error_xy = self.support_center[:2] - center[:2]
        center_error = float(np.linalg.norm(error_xy))
        if not _center_is_within_tolerance(center, self.support_center):
            target[:2] += _clamp_vector_norm(error_xy, CENTER_CORRECTION_STEP_M)
        position_error, _, _ = self._command_pose(target, dt)
        self._set_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_anchor_from_actual_tcp()
        return center_error, position_error

    def _step_center_correct(self, dt: float) -> None:
        center_error, tcp_error = self._correct_center_xy(dt, self.correction_target)
        correction_distance = float(
            np.linalg.norm(self.correction_target[:2] - self.correction_origin[:2])
        )
        if correction_distance > MAX_CENTER_CORRECTION_M:
            raise RuntimeError(f"Center correction exceeded {MAX_CENTER_CORRECTION_M:.3f} m")
        within_tolerance = center_error <= CENTER_RELEASE_TOLERANCE_M
        self.stable_time = _accumulate_stable_time(self.stable_time, within_tolerance, dt)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] center correction: "
                f"center_error={center_error:.4f} m, tcp_error={tcp_error:.4f} m, "
                f"correction={correction_distance:.4f} m, "
                f"stable={self.stable_time:.3f}/"
                f"{self.timing.center_stable_seconds:.3f} s"
            )
        if within_tolerance and _stable_time_reached(
            self.stable_time, self.timing.center_stable_seconds
        ):
            _, _, center = self._node_aabb()
            lower_offset = float(self.left_setdown_offset_m)
            if not math.isfinite(lower_offset):
                raise ValueError("LEFT_LOWER_DISTANCE_OFFSET_M must be finite")
            desired_center_z = (
                self.support_bounds[1][2]
                + 0.5 * self.initial_node_height
                + PLACEMENT_CLEARANCE_M
                - lower_offset
            )
            self.lower_start = self.correction_target.copy()
            self.setdown_target = self.correction_target.copy()
            self.setdown_target[2] += desired_center_z - center[2]
            vertical_displacement = float(self.setdown_target[2] - self.lower_start[2])
            if abs(vertical_displacement) > 0.350:
                raise RuntimeError(
                    "Calculated setdown exceeds the 350 mm vertical safety "
                    f"bound: displacement={vertical_displacement:.4f} m"
                )
            _log(
                "[ProM6ClothingDrag] Setdown target: "
                f"current_center_z={center[2]:.4f}, desired_center_z={desired_center_z:.4f}, "
                f"lower_offset={lower_offset:.4f} m, "
                f"vertical_displacement={vertical_displacement:.4f} m, "
                f"tcp={_array_string(self.setdown_target)}"
            )
            self._set_phase(Phase.LOWER)
        elif self.phase_time >= self.phase_timeouts[Phase.CENTER_CORRECT]:
            raise RuntimeError(f"Center correction timed out: error={center_error:.4f} m")

    def _step_lower(self, dt: float) -> None:
        progress = _timed_smoothstep(self.phase_time, self.timing.left_lower_seconds)
        target = (1.0 - progress) * self.lower_start + progress * self.setdown_target
        position_error, orientation_error, _ = self._command_pose(target, dt)
        self._set_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_anchor_from_actual_tcp()
        if (
            self.phase_time >= self.timing.left_lower_seconds
            and position_error <= DRAG_ENDPOINT_TOLERANCE_M
        ):
            self._set_phase(Phase.PLACE_SETTLE)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] lower: "
                f"progress={progress:.3f}, position_error={position_error:.4f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg"
            )

    def _step_place_settle(self, dt: float) -> None:
        center_error, tcp_error = self._correct_center_xy(dt, self.setdown_target)
        correction_distance = float(
            np.linalg.norm(self.setdown_target[:2] - self.correction_origin[:2])
        )
        if correction_distance > MAX_CENTER_CORRECTION_M:
            raise RuntimeError(
                "Placement correction exceeded its safety bound: "
                f"distance={correction_distance:.4f} m"
            )
        within_tolerance = center_error <= CENTER_RELEASE_TOLERANCE_M
        self.stable_time = _accumulate_stable_time(self.stable_time, within_tolerance, dt)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] place settle: "
                f"center_error={center_error:.4f} m, tcp_error={tcp_error:.4f} m, "
                f"stable={self.stable_time:.3f}/"
                f"{self.timing.place_stable_seconds:.3f} s"
            )
        if (
            within_tolerance
            and self.phase_time >= self.timing.left_place_settle_seconds
            and _stable_time_reached(self.stable_time, self.timing.place_stable_seconds)
        ):
            self._teleport_pad_retract(LEFT_PAD_RETRACT_M)
            self._set_phase(Phase.LEFT_OPEN)
        elif self.phase_time >= self.phase_timeouts[Phase.PLACE_SETTLE]:
            raise RuntimeError(
                f"Clothing is not centered closely enough to release: error={center_error:.4f} m"
            )

    def _step_left_open(self, dt: float) -> None:
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD, snap=True)
        if self.newton_world is not None:
            self.newton_world.set_pad_retract(LEFT_PAD_RETRACT_M)
        if self.phase_time < self.timing.left_open_seconds:
            self._write_left_q(self.left_commanded_q)
            return
        if self.usd_kinematics is None or self.left_commanded_q is None:
            raise RuntimeError("Left retract IK is unavailable after release")
        tcp_position, _ = self._current_tcp()
        if self.left_retract_target is None:
            self.left_retract_target = np.array(
                [
                    float(tcp_position[0]),
                    float(tcp_position[1]),
                    float(tcp_position[2]) + LEFT_RETRACT_HEIGHT_M,
                ],
                dtype=np.float64,
            )
            _log(
                "[ProM6ClothingDrag] Left retract after release: "
                f"from={_array_string(tcp_position)}, "
                f"to={_array_string(self.left_retract_target)}, "
                f"finger={self._finger_contacts()}"
            )
        position_error, _, _ = self._command_pose(self.left_retract_target, dt)
        if position_error > TCP_LIFT_POSITION_TOLERANCE_M:
            return
        self._left_retract_hold_s += dt
        if self._left_retract_hold_s < LEFT_RETRACT_HOLD_S:
            self._write_left_q(self.left_commanded_q)
            if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
                _log(
                    "[ProM6ClothingDrag] Left retract hold: "
                    f"finger={self._finger_contacts()}, t={self._left_retract_hold_s:.2f} s"
                )
            return
        _log(
            "[ProM6ClothingDrag] Left retract reached height: "
            f"finger={self._finger_contacts()}, pads_clear={self._left_pads_clear_of_bag()}"
        )
        if self.newton_world is not None:
            _log(f"[ProM6ClothingDrag] Left retract pads: {self.newton_world.finger_debug()}")
        self.left_return_start_q = self.left_commanded_q.copy()
        self._set_phase(Phase.LEFT_RETURN_READY)

    def _step_left_return_ready(self, dt: float) -> None:
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD, snap=True)
        tcp_position, _ = self._current_tcp()
        if self._left_return_clear_target is None:
            self._left_return_clear_target = np.array(
                [
                    float(tcp_position[0]) + LEFT_RETURN_CLEAR_X_OFFSET_M,
                    float(tcp_position[1]),
                    float(tcp_position[2]),
                ],
                dtype=np.float64,
            )
            _log(
                "[ProM6ClothingDrag] Left return clear: "
                f"from={_array_string(tcp_position)}, "
                f"to={_array_string(self._left_return_clear_target)}"
            )
        if not self._left_return_cartesian_done:
            finger_n = self._finger_contacts()
            pads_clear = self._left_pads_clear_of_bag()
            if finger_n > 30 and not pads_clear:
                self._write_left_q(self.left_commanded_q)
                if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
                    _log(
                        "[ProM6ClothingDrag] Left return wait contacts: "
                        f"finger={finger_n}, pads_clear={pads_clear}"
                    )
                return
            if finger_n > 30 and pads_clear:
                _log(
                    "[ProM6ClothingDrag] Left return skip stale contacts: "
                    f"finger={finger_n}, pads_clear={pads_clear}"
                )
            previous_q = self.left_commanded_q.copy()
            position_error, _, _ = self._command_pose(self._left_return_clear_target, dt)
            current_tcp, _ = self._current_tcp()
            # _write_left_q clips IK commands at the joint limits. Inspect the actual
            # command change, rather than the unclipped delta returned by _command_pose.
            stalled = (
                np.linalg.norm(self.left_commanded_q - previous_q) <= 1.0e-3 * dt
                and np.linalg.norm(current_tcp - tcp_position) <= 1.0e-4 * dt
            )
            self._left_return_stall_s = self._left_return_stall_s + dt if stalled else 0.0
            residual = self._left_return_clear_target - current_tcp
            # This is an empty-hand clearance waypoint, not a grasp/placement goal.
            # Permit joint return only near that waypoint, at the retracted height,
            # with no bag contact. Keep all manipulation and success checks unchanged.
            clear_at_limit = (
                self._left_return_stall_s >= LEFT_RETURN_STALL_HOLD_S
                and finger_n == 0
                and pads_clear
                and abs(float(residual[0])) <= 2.0 * TCP_LIFT_POSITION_TOLERANCE_M
                and abs(float(residual[1])) <= TCP_LIFT_POSITION_TOLERANCE_M
                and float(residual[2]) <= TCP_LIFT_POSITION_TOLERANCE_M
            )
            if position_error <= TCP_LIFT_POSITION_TOLERANCE_M or clear_at_limit:
                self._left_return_cartesian_done = True
                self.left_return_start_q = self.left_commanded_q.copy()
                if clear_at_limit and position_error > TCP_LIFT_POSITION_TOLERANCE_M:
                    _log(
                        "[ProM6ClothingDrag] Left return IK stalled near clearance waypoint; "
                        f"pads clear, no contact, residual={_array_string(residual)}, "
                        "joint-slew to ready"
                    )
                else:
                    _log("[ProM6ClothingDrag] Left return clear reached, joint-slew to ready")
            return
        if self.left_return_start_q is None:
            raise RuntimeError("Left return start joints are unavailable after clear")
        self._left_return_joint_elapsed += dt
        progress = _timed_smoothstep(
            self._left_return_joint_elapsed, self.timing.left_return_ready_seconds
        )
        desired = (1.0 - progress) * self.left_return_start_q + progress * LEFT_READY_JOINTS
        step_norm = self._slew_joint_target(desired, dt)
        joint_error = float(np.max(np.abs(self.left_commanded_q - LEFT_READY_JOINTS)))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] left return: "
                f"progress={progress:.3f}, joint_error={joint_error:.5f} rad, "
                f"step_norm={step_norm:.5f}"
            )
        if (
            self._left_return_joint_elapsed >= self.timing.left_return_ready_seconds
            and joint_error <= 1.0e-4
        ):
            self._write_left_q(LEFT_READY_JOINTS)
            self._set_phase(Phase.PREPARE_RIGHT_HANDOFF)

    def _right_grasp_patch(self, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
        """右抓点优先取包裹中心,保留 IK 选出的偏移并跟踪包裹漂移."""
        min_x = float(minimum[0])
        max_x = float(maximum[0])
        min_y = float(minimum[1])
        max_y = float(maximum[1])
        center_x = 0.5 * (min_x + max_x)
        grasp_x = center_x + self._right_grasp_offset[0]
        grasp_x = min(grasp_x, max_x - 0.03)
        grasp_x = max(grasp_x, min_x + RIGHT_GRASP_EDGE_INSET_M)
        inset_y = RIGHT_GRASP_EDGE_INSET_M
        y_lo = min_y + inset_y
        y_hi = max_y - inset_y
        if y_hi < y_lo:
            y_lo, y_hi = y_hi, y_lo
        preferred_y = 0.5 * (min_y + max_y) + self._right_grasp_offset[1]
        grasp_y = float(np.clip(preferred_y, y_lo, y_hi))
        return np.array(
            [
                grasp_x,
                grasp_y,
                float(maximum[2]) - RIGHT_GRASP_BITE_M,
            ],
            dtype=np.float64,
        )

    def _right_grasp_geometry(
        self, minimum: np.ndarray, maximum: np.ndarray
    ) -> dict[str, np.ndarray]:
        """计算右抓位姿，并把指垫最低点限制在平台顶面之上。"""
        if self.newton_world is None:
            raise RuntimeError("右抓平台安全高度需要 Newton 世界")
        geometry = calculate_shared_attachment_grasp(self._right_grasp_patch(minimum, maximum))
        floor_z = self.newton_world.top_down_tcp_floor_z("right", RIGHT_PLATFORM_PAD_CLEARANCE_M)
        delta_z = max(0.0, floor_z - float(geometry["grasp_point"][2]))
        if delta_z > 0.0:
            for key in ("attachment_center", "grasp_point", "pregrasp_point"):
                geometry[key] = geometry[key].copy()
                geometry[key][2] += delta_z
            _log(
                "[ProM6ClothingDrag] Right grasp raised for platform clearance: "
                f"delta_z={delta_z:.4f} m, tcp_floor_z={floor_z:.4f}"
            )
        return geometry

    def _assert_no_right_platform_penetration(self) -> None:
        """右臂抬升前拒绝已经落到平台碰撞面下方的软体节点。"""
        if self.newton_world is None:
            raise RuntimeError("右抓穿模检查需要 Newton 世界")
        positions = np.asarray(self.newton_world.surface_positions(), dtype=np.float64)
        limit_z = float(self.newton_world.support_top_z) - RIGHT_PLATFORM_PENETRATION_TOLERANCE_M
        below = positions[:, 2] < limit_z
        minimum_z = float(np.min(positions[:, 2]))
        if np.any(below):
            raise RuntimeError(
                "Right grasp pushed the parcel through the platform: "
                f"minimum_z={minimum_z:.5f}, platform_top_z="
                f"{self.newton_world.support_top_z:.5f}, "
                f"deep_nodes={int(np.count_nonzero(below))}"
            )
        _log(
            "[ProM6ClothingDrag] Right grasp platform clearance check: "
            f"minimum_z={minimum_z:.5f}, platform_top_z="
            f"{self.newton_world.support_top_z:.5f}"
        )

    def _refresh_right_grasp_from_live_nodes(self, *, xy: bool) -> None:
        """按当前 Newton AABB 更新右抓 TCP;合爪时只跟高度,避免袋瘪后擦顶滑脱."""
        minimum, maximum, _ = self._node_aabb()
        geometry = self._right_grasp_geometry(minimum, maximum)
        grasp = geometry["grasp_point"].copy()
        pregrasp = geometry["pregrasp_point"].copy()
        if xy or self.right_grasp_point is None:
            self.right_grasp_point = grasp
            self.right_pregrasp_point = pregrasp
        else:
            current = np.asarray(self.right_grasp_point, dtype=np.float64).reshape(3).copy()
            current[2] = float(grasp[2])
            self.right_grasp_point = current
            if self.right_pregrasp_point is not None:
                pre = np.asarray(self.right_pregrasp_point, dtype=np.float64).reshape(3).copy()
                pre[2] = float(pregrasp[2])
                self.right_pregrasp_point = pre
        _log(
            "[ProM6ClothingDrag] Right grasp refreshed: "
            f"xy={xy}, grasp={_array_string(self.right_grasp_point)}, "
            f"nodes_min={_array_string(minimum)}, nodes_max={_array_string(maximum)}"
        )

    def _check_label_up(self) -> None:
        """松爪并撤离后检查真实标签表面，翻折时拒绝进入右臂接力。"""
        if not self.keep_label_up:
            return
        if self.newton_world is None:
            raise RuntimeError("标签朝向检查需要 Newton 包裹表面")
        try:
            normal = surface_patch_normal(
                self.newton_world.surface_positions(), self.newton_world.label_triangles
            )
        except ValueError as exc:
            raise RuntimeError(f"标签朝向不可判定: {exc}") from exc
        self.label_up_cos_at_handoff = float(normal[2])
        _log(
            "[ProM6ClothingDrag] Platform label-up check: "
            f"normal={_array_string(normal)}, min_cos={self.label_up_min_cos:.3f}"
        )
        if self.label_up_cos_at_handoff < self.label_up_min_cos:
            raise RuntimeError(
                "Package label is not facing up on the platform: "
                f"normal_z={self.label_up_cos_at_handoff:.4f} < {self.label_up_min_cos:.4f}"
            )

    def _step_prepare_right_handoff(self) -> None:
        self._check_label_up()
        self._teleport_pad_retract(0.0)
        self._write_left_q(LEFT_READY_JOINTS)
        self._write_right_ready()
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        minimum, maximum, center = self._node_aabb()
        self._right_grasp_offset = np.zeros(3, dtype=np.float64)
        geometry = self._right_grasp_geometry(minimum, maximum)
        self.right_handoff_anchor_position = geometry["attachment_center"].copy()
        self.right_grasp_point = geometry["grasp_point"].copy()
        self.right_pregrasp_point = geometry["pregrasp_point"].copy()
        self.belt_bounds = get_world_bounds(self.stage.GetPrimAtPath(BELT_PATH))
        if not self.belt_bounds[0][1] <= center[1] <= self.belt_bounds[1][1]:
            raise RuntimeError(
                "Clothing center Y is outside the belt footprint and a "
                "world -X drag cannot correct it: "
                f"center_y={center[1]:.4f}, "
                f"belt_y=[{self.belt_bounds[0][1]:.4f}, "
                f"{self.belt_bounds[1][1]:.4f}]"
            )
        _log(
            "[ProM6ClothingDrag] Right grasp from live Newton AABB: "
            f"grasp={_array_string(self.right_grasp_point)}, "
            f"pregrasp={_array_string(self.right_pregrasp_point)}, "
            f"nodes_min={_array_string(minimum)}, "
            f"nodes_max={_array_string(maximum)}"
        )
        self._set_phase(Phase.RIGHT_SEARCH_IK)

    def _right_pregrasp_candidates(self) -> list[np.ndarray]:
        """从包裹中心逐步向外搜索,按实际 XY 距离优先选靠近中心的点."""
        if self.right_pregrasp_point is None:
            raise RuntimeError("Right pregrasp point is unavailable")
        primary = np.asarray(self.right_pregrasp_point, dtype=np.float64).reshape(3)
        minimum, maximum, _ = self._node_aabb()
        randomized = any(getattr(self, "grasp_jitter_fraction_xy", (0.0, 0.0)))
        offsets = self._grasp_offsets(minimum, maximum, "right")
        inset = RIGHT_GRASP_EDGE_INSET_M
        x_lo, x_hi = float(minimum[0]) + inset, float(maximum[0]) - inset
        y_hi = float(maximum[1]) - inset
        y_lo = float(minimum[1]) + inset
        if y_hi < y_lo:
            y_lo, y_hi = y_hi, y_lo
        candidates = [] if randomized else [primary.copy()]
        if not randomized:
            offsets.append(np.array([0.0, y_hi - primary[1], 0.0]))
        for offset in offsets:
            alt = primary + offset
            alt[0] = float(np.clip(alt[0], min(x_lo, x_hi), max(x_lo, x_hi)))
            alt[1] = float(np.clip(alt[1], y_lo, y_hi))
            if (
                not candidates
                or min(float(np.linalg.norm(alt[:2] - item[:2])) for item in candidates) > 1.0e-6
            ):
                candidates.append(alt)
        if randomized:
            return candidates
        return sorted(candidates, key=lambda point: float(np.linalg.norm(point[:2] - primary[:2])))

    def _step_right_search_ik(self) -> None:
        if (
            self.right_usd_kinematics is None
            or self.right_pregrasp_point is None
            or self.right_arm_lower_limits is None
            or self.right_arm_upper_limits is None
        ):
            raise RuntimeError("Right pregrasp IK inputs are unavailable")
        self._write_left_q(LEFT_READY_JOINTS)
        self._write_right_q(self.right_commanded_q)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        last_error = "Right pregrasp IK search found no solution"
        selected: np.ndarray | None = None
        diagnostics: dict[str, float | int] | None = None
        initial_pregrasp = self.right_pregrasp_point.copy()
        for index, pregrasp in enumerate(self._right_pregrasp_candidates()):
            waypoints = (
                pregrasp - np.array([0.0, 0.0, PREGRASP_CLEARANCE_M]),
                pregrasp + np.array([0.0, 0.0, RIGHT_LIFT_HEIGHT_M - PREGRASP_CLEARANCE_M]),
            )
            if not self._screen_random_grasp(
                self.right_usd_kinematics,
                pregrasp,
                waypoints,
                self.right_arm_lower_limits,
                self.right_arm_upper_limits,
                RIGHT_IK_PREFERRED_SEED,
                RIGHT_IK_ELBOW_OUT_SEED,
            ):
                _log(f"[ProM6ClothingDrag][right IK search] Random grasp unreachable: {pregrasp}")
                continue
            try:
                selected, diagnostics = solve_pregrasp_ik_42_seeds(
                    self.right_usd_kinematics,
                    pregrasp,
                    self.grasp_orientation,
                    self.right_arm_lower_limits,
                    self.right_arm_upper_limits,
                    IK_JOINT_LIMIT_MARGIN_RAD,
                    RIGHT_IK_PREFERRED_SEED,
                    self.timing.ik_search_timeout_seconds,
                    ready_seed=RIGHT_IK_ELBOW_OUT_SEED,
                    outward_xy=RIGHT_ELBOW_OUTWARD_XY,
                    waypoint_positions=waypoints,
                )
            except RuntimeError as exc:
                last_error = str(exc)
                _log(
                    "[ProM6ClothingDrag][right IK search] candidate failed: "
                    f"index={index}, pregrasp={_array_string(pregrasp)}, error={last_error}"
                )
                continue
            self._right_grasp_offset = pregrasp - initial_pregrasp
            self._right_grasp_offset[2] = 0.0
            self.right_pregrasp_point = pregrasp.copy()
            if self.right_grasp_point is not None:
                grasp = np.asarray(self.right_grasp_point, dtype=np.float64).reshape(3).copy()
                grasp[:2] = pregrasp[:2]
                self.right_grasp_point = grasp
            _log(
                "[ProM6ClothingDrag][right IK search] selected branch: "
                f"candidate={index}, "
                f"seed={int(diagnostics['seed_index'])}, "
                f"valid={int(diagnostics['valid_solution_count'])}/"
                f"{int(diagnostics['candidate_count'])}, "
                f"position={1000.0 * float(diagnostics['position_error']):.3f} mm, "
                f"orientation={math.degrees(float(diagnostics['orientation_error'])):.3f} deg, "
                f"elbow_out={float(diagnostics['elbow_outboard_m']):.3f} m, "
                f"q={_array_string(selected, 5)}"
            )
            break
        if selected is None or diagnostics is None:
            raise RuntimeError(last_error)
        self.right_pregrasp_joint_target = selected.copy()
        self.right_pregrasp_start_q = self.right_commanded_q.copy()
        self._set_phase(Phase.RIGHT_MOVE_TO_PREGRASP)

    def _step_right_joint_segment(
        self,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        dt: float,
        next_phase: Phase,
    ) -> None:
        progress = _timed_smoothstep(self.phase_time, duration)
        desired = (1.0 - progress) * start + progress * target
        step_norm = self._slew_right_joint_target(desired, dt)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        joint_error = float(np.max(np.abs(self.right_commanded_q - target)))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                f"[ProM6ClothingDrag] {self.phase.name}: "
                f"progress={progress:.3f}, joint_error={joint_error:.5f} rad, "
                f"step_norm={step_norm:.5f}"
            )
        if self.phase_time >= duration and joint_error <= 1.0e-4:
            self._set_phase(next_phase)

    def _step_right_approach(self, dt: float) -> None:
        if not self._right_grasp_z_refreshed:
            self._refresh_right_grasp_from_live_nodes(xy=False)
            self._right_grasp_z_refreshed = True
        progress = _timed_smoothstep(self.phase_time, self.timing.right_approach_seconds)
        target = self.right_pregrasp_point - np.array(
            [0.0, 0.0, progress * PREGRASP_CLEARANCE_M],
            dtype=np.float64,
        )
        position_error, orientation_error, step_norm = self._command_right_pose(target, dt)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] right approach: "
                f"progress={progress:.3f}, position_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"step_norm={step_norm:.5f}"
            )
        if (
            self.phase_time >= self.timing.right_approach_seconds
            and position_error <= TCP_APPROACH_POSITION_TOLERANCE_M
            and orientation_error <= TCP_APPROACH_ORIENTATION_TOLERANCE_RAD
        ):
            self._set_phase(Phase.RIGHT_CLOSE)

    def _step_right_close(self, dt: float) -> None:
        if not self._right_grasp_z_refreshed:
            self._refresh_right_grasp_from_live_nodes(xy=True)
            self._right_grasp_z_refreshed = True
        position_error, orientation_error, _ = self._command_right_pose(self.right_grasp_point, dt)
        pose_ok = (
            position_error <= TCP_GRASP_POSITION_TOLERANCE_M
            and orientation_error <= TCP_GRASP_ORIENTATION_TOLERANCE_RAD
        )
        if not pose_ok:
            self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
            self._right_close_elapsed = 0.0
            return
        self._right_close_elapsed += dt
        fraction = _timed_smoothstep(self._right_close_elapsed, self.timing.right_close_seconds)
        target = GRIPPER_OPEN_POSITION_RAD + fraction * (
            GRIPPER_CLOSED_POSITION_RAD - GRIPPER_OPEN_POSITION_RAD
        )
        self._set_right_gripper_target(target, snap=True)
        if self._right_close_elapsed >= self.timing.right_close_seconds:
            self._set_phase(Phase.RIGHT_GRIP_SETTLE)

    def _finger_contacts(self) -> int:
        if self.newton_world is None:
            return 0
        return int(self.newton_world.contact_breakdown().get("finger", 0) or 0)

    def _left_pads_clear_of_bag(self, min_gap_m: float = LEFT_PAD_CLEAR_GAP_M) -> bool:
        """左指垫中心已离开袋 AABB 时视为脱离,避免陈旧接触把回 ready 卡死."""
        if self.newton_world is None:
            return False
        dists = [
            float(row.get("dist_aabb", 0.0) or 0.0)
            for row in self.newton_world.finger_debug()[1:]
            if row.get("side") == "left"
        ]
        return bool(dists) and min(dists) >= float(min_gap_m)

    def _right_finger_contacts(self) -> int:
        return self._finger_contacts()

    def _step_right_grip_settle(self, dt: float) -> None:
        if not self._right_grasp_z_refreshed:
            self._refresh_right_grasp_from_live_nodes(xy=False)
            self._right_grasp_z_refreshed = True
        finger_n = self._right_finger_contacts()
        if (
            finger_n < RIGHT_MIN_FINGER_CONTACTS
            and self._right_grasp_dip_m < RIGHT_GRASP_DIP_MAX_M
            and self.right_grasp_point is not None
        ):
            step = min(RIGHT_GRASP_DIP_STEP_M, RIGHT_GRASP_DIP_MAX_M - self._right_grasp_dip_m)
            self.right_grasp_point = np.asarray(self.right_grasp_point, dtype=np.float64).copy()
            if self.newton_world is None:
                raise RuntimeError("右抓下探限制需要 Newton 世界")
            floor_z = self.newton_world.top_down_tcp_floor_z(
                "right", RIGHT_PLATFORM_PAD_CLEARANCE_M
            )
            requested_z = float(self.right_grasp_point[2] - step)
            safe_z = max(requested_z, floor_z)
            actual_step = float(self.right_grasp_point[2] - safe_z)
            self.right_grasp_point[2] = safe_z
            if actual_step > 1.0e-6:
                self._right_grasp_dip_m += actual_step
                _log(
                    "[ProM6ClothingDrag] Right grasp dip: "
                    f"finger={finger_n}, dip={self._right_grasp_dip_m:.3f} m, "
                    f"z={self.right_grasp_point[2]:.4f}"
                )
            else:
                self._right_grasp_dip_m = RIGHT_GRASP_DIP_MAX_M
                _log(
                    "[ProM6ClothingDrag] Right grasp dip stopped at platform clearance: "
                    f"finger={finger_n}, tcp_floor_z={floor_z:.4f}"
                )
        position_error, orientation_error, _ = self._command_right_pose(self.right_grasp_point, dt)
        self._set_right_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        target = float(
            np.clip(
                GRIPPER_CLOSED_POSITION_RAD,
                self.right_gripper_lower_limit,
                self.right_gripper_upper_limit,
            )
        )
        gripper_error = abs(float(self._current_dof_positions()[self.right_gripper_index]) - target)
        holding = finger_n >= RIGHT_MIN_FINGER_CONTACTS
        converged = (
            position_error <= TCP_GRASP_POSITION_TOLERANCE_M
            and orientation_error <= TCP_GRASP_ORIENTATION_TOLERANCE_RAD
            and gripper_error <= GRIPPER_POSITION_TOLERANCE_RAD
            and holding
        )
        self.stable_time = _accumulate_stable_time(self.stable_time, converged, dt)
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] right grip settle: "
                f"tcp_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"gripper_error={gripper_error:.5f} rad, "
                f"finger={finger_n}, "
                f"stable={self.stable_time:.3f}/"
                f"{self.timing.right_grip_stable_seconds:.3f} s"
            )
        if (
            converged
            and self.phase_time >= self.timing.right_grip_settle_seconds
            and _stable_time_reached(self.stable_time, self.timing.right_grip_stable_seconds)
        ):
            self._assert_no_right_platform_penetration()
            self.right_handoff_tcp_position, _ = self._current_right_tcp()
            self.right_handoff_anchor_position = self.right_grasp_point.copy()
            self.right_grasp_position = self.right_grasp_point.copy()
            self.right_lift_start = self.right_handoff_tcp_position.copy()
            self.right_lift_target = self.right_lift_start + np.array(
                [0.0, 0.0, RIGHT_LIFT_HEIGHT_M], dtype=np.float64
            )
            self.right_actual_drag_distance = 0.0
            self.belt_bounds = get_world_bounds(self.stage.GetPrimAtPath(BELT_PATH))
            _, _, center = self._node_aabb()
            if not self.belt_bounds[0][1] <= center[1] <= self.belt_bounds[1][1]:
                raise RuntimeError(
                    "Clothing center Y is outside the belt footprint at "
                    f"right-arm takeover: center_y={center[1]:.4f}"
                )
            self.right_nominal_drag_distance = calculate_belt_entry_distance(
                center, self.belt_bounds
            )
            if self.right_nominal_drag_distance > RIGHT_MAX_DRAG_DISTANCE_M:
                raise RuntimeError(
                    "Required right drag exceeds the "
                    f"{RIGHT_MAX_DRAG_DISTANCE_M:.2f} m safety bound: "
                    f"distance={self.right_nominal_drag_distance:.4f} m"
                )
            if self.newton_world is not None:
                q = self._current_dof_positions()
                _log(
                    "[ProM6ClothingDrag] Finger/bag before right lift: "
                    f"right_finger_q={float(q[self.right_gripper_index]):.4f} rad, "
                    f"{self.newton_world.finger_debug()}"
                )
            _log(
                "[ProM6ClothingDrag] Right arm contact takeover: "
                f"grasp={_array_string(self.right_handoff_anchor_position)}, "
                f"tcp={_array_string(self.right_handoff_tcp_position)}, "
                f"center={_array_string(center)}, "
                f"belt_min={_array_string(self.belt_bounds[0])}, "
                f"belt_max={_array_string(self.belt_bounds[1])}, "
                f"entry_x={self.belt_bounds[1][0] - BELT_ENTRY_INSET_M:.4f}, "
                f"nominal_distance={self.right_nominal_drag_distance:.4f} m, "
                f"lift_target={_array_string(self.right_lift_target)}, "
                f"lower_distance={RIGHT_DRAG_LOWER_DISTANCE_M:.4f} m"
            )
            self.right_lift_com_z = float(self._parcel_com()[2])
            self._set_phase(Phase.RIGHT_LIFT)
        elif self.phase_time >= max(
            self.timing.right_grip_settle_seconds + 5.0,
            self.timing.right_grip_stable_seconds + 5.0,
        ):
            raise RuntimeError(
                "Right grasp did not settle: "
                f"position={position_error:.5f}, "
                f"orientation={orientation_error:.5f}, "
                f"gripper={gripper_error:.5f}, "
                f"finger={finger_n}"
            )

    def _step_right_lift(self, dt: float) -> None:
        if self.right_lift_start is None or self.right_lift_target is None:
            raise RuntimeError("Right lift geometry is unavailable")
        progress = _timed_smoothstep(self.phase_time, self.timing.right_lift_seconds)
        target = (1.0 - progress) * self.right_lift_start + progress * self.right_lift_target
        self._command_right_pose(target, dt)
        self._set_right_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_shared_anchor_from_right_tcp()
        # Gate against the final lift target using the newly commanded FK pose.
        tcp_position, tcp_orientation = self._current_right_tcp()
        error = _pose_error(
            tcp_position, tcp_orientation, self.right_lift_target, self.grasp_orientation
        )
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] right lift: "
                f"progress={progress:.3f}, position_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg"
            )
        if (
            self.phase_time >= self.timing.right_lift_seconds
            and position_error <= TCP_LIFT_POSITION_TOLERANCE_M
            and orientation_error <= TCP_LIFT_ORIENTATION_TOLERANCE_RAD
        ):
            self._require_lifted(self.right_lift_com_z, "right")
            # Preserve the original handoff references and frozen entry distance.
            self.right_drag_start = tcp_position.copy()
            self.right_drag_target = self.right_drag_start.copy()
            self.right_actual_drag_distance = 0.0
            self._set_phase(Phase.RIGHT_DRAG_TO_BELT)

    def _step_right_drag_to_belt(self, dt: float) -> None:
        commanded_distance = min(
            RIGHT_DRAG_SPEED_M_S * self.phase_time,
            RIGHT_MAX_DRAG_DISTANCE_M,
        )
        self.right_drag_target[:] = self.right_drag_start
        self.right_drag_target[0] -= commanded_distance
        lower_progress = float(
            np.clip(
                commanded_distance / self.right_nominal_drag_distance
                if self.right_nominal_drag_distance > 0.0
                else self.phase_time / 0.5,
                0.0,
                1.0,
            )
        )
        self.right_drag_target[2] -= RIGHT_DRAG_LOWER_DISTANCE_M * lower_progress
        position_error, orientation_error, _ = self._command_right_pose(self.right_drag_target, dt)
        self._set_right_gripper_target(GRIPPER_CLOSED_POSITION_RAD, snap=True)
        self._update_shared_anchor_from_right_tcp()
        tcp_position, _ = self._current_right_tcp()
        self.right_actual_drag_distance = max(
            0.0, float(self.right_drag_start[0] - tcp_position[0])
        )
        fk_lower_distance = float(self.right_drag_start[2] - tcp_position[2])
        final_height_error = abs(fk_lower_distance - RIGHT_DRAG_LOWER_DISTANCE_M)
        lowering_complete = RIGHT_DRAG_LOWER_DISTANCE_M <= 1.0e-6 or (
            lower_progress >= 1.0 and final_height_error <= TCP_APPROACH_POSITION_TOLERANCE_M
        )
        minimum, maximum, center = self._node_aabb()
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            self.belt_bounds = get_world_bounds(self.stage.GetPrimAtPath(BELT_PATH))
        inside = center_is_in_belt_xy(center, self.belt_bounds)
        self.stable_time = _accumulate_stable_time(self.stable_time, inside, dt)
        entry_x = self.belt_bounds[1][0] - BELT_ENTRY_INSET_M
        remaining = max(0.0, float(center[0] - entry_x))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] right belt drag: "
                f"tcp_error={position_error:.5f} m, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg, "
                f"center={_array_string(center)}, "
                f"aabb_max_x={maximum[0]:.4f}, remaining={remaining:.4f} m, "
                f"dragged={self.right_actual_drag_distance:.4f} m, "
                f"lower_progress={lower_progress:.3f}, "
                f"target_lower={RIGHT_DRAG_LOWER_DISTANCE_M * lower_progress:.4f}/"
                f"{RIGHT_DRAG_LOWER_DISTANCE_M:.4f} m, "
                f"fk_lower={fk_lower_distance:.4f} m, "
                f"stable={self.stable_time:.3f}/"
                f"{self.timing.belt_entry_stable_seconds:.3f} s"
            )
        if (
            inside
            and lowering_complete
            and _stable_time_reached(self.stable_time, self.timing.belt_entry_stable_seconds)
        ):
            self._release_attachment()
            self._set_phase(Phase.RIGHT_RELEASE)
        elif (
            commanded_distance >= RIGHT_MAX_DRAG_DISTANCE_M
            and position_error <= DRAG_ENDPOINT_TOLERANCE_M
        ):
            raise RuntimeError(
                "Right drag reached the "
                f"{RIGHT_MAX_DRAG_DISTANCE_M:.2f} m safety bound before release "
                f"conditions were met: center={center}, tail_x={maximum[0]:.4f}, "
                f"lower_progress={lower_progress:.3f}, "
                f"final_height_error={final_height_error:.5f} m"
            )

    def _start_release_conveyor(self) -> None:
        if self.release_conveyor is not None:
            return
        if self.newton_world is None:
            raise RuntimeError("Cannot start conveyor without the Newton world")
        self.belt_bounds = get_world_bounds(self.stage.GetPrimAtPath(BELT_PATH))
        speed = float(self.newton_world.params.belt_speed_mps)
        self.release_conveyor = DeformableConveyorController(
            self.newton_world,
            _belt_xy_bounds(self.belt_bounds),
            RELEASE_CONVEYOR_STOP_XY_REGION,
            speed,
            RELEASE_CONVEYOR_DIRECTION_XY,
            stop_on_arrival=RELEASE_CONVEYOR_STOP_ON_ARRIVAL,
            zero_xy_on_exit=RELEASE_CONVEYOR_ZERO_XY_ON_EXIT,
            log_interval_frames=PROGRESS_LOG_INTERVAL_FRAMES,
        )
        self.release_conveyor.start()

    def _step_release_conveyor(self, dt: float) -> None:
        if self.release_conveyor is None:
            raise RuntimeError("Release conveyor was not started with the right gripper")
        self.release_conveyor.step(dt)

    def _step_right_release(self) -> None:
        self._write_left_q(LEFT_READY_JOINTS)
        self._write_right_q(self.right_commanded_q)
        self._start_release_conveyor()
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        if self.phase_frame == 1:
            self._teleport_pad_retract(LEFT_PAD_RETRACT_M)
        if self.phase_time >= self.timing.right_release_seconds:
            self.right_return_start_q = self.right_commanded_q.copy()
            self._set_phase(Phase.RIGHT_RETURN_READY)

    def _step_right_return_ready(self, dt: float) -> None:
        progress = _timed_smoothstep(self.phase_time, self.timing.right_return_ready_seconds)
        desired = (1.0 - progress) * self.right_return_start_q + progress * RIGHT_READY_JOINTS
        step_norm = self._slew_right_joint_target(desired, dt)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        joint_error = float(np.max(np.abs(self.right_commanded_q - RIGHT_READY_JOINTS)))
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            _log(
                "[ProM6ClothingDrag] right return: "
                f"progress={progress:.3f}, joint_error={joint_error:.5f} rad, "
                f"step_norm={step_norm:.5f}"
            )
        if self.phase_time >= self.timing.right_return_ready_seconds and joint_error <= 1.0e-4:
            self._write_right_ready()
            self._set_phase(Phase.FINAL_VERIFY)

    def _step_final_verify(self, dt: float) -> None:
        del dt
        self._write_left_q(LEFT_READY_JOINTS)
        self._write_right_ready()
        self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        self._set_right_gripper_target(GRIPPER_OPEN_POSITION_RAD)
        self.belt_bounds = get_world_bounds(self.stage.GetPrimAtPath(BELT_PATH))
        _, _, center = self._node_aabb()
        if self.release_conveyor is None:
            raise RuntimeError("Release conveyor is unavailable during final verify")
        if self.phase_frame % PROGRESS_LOG_INTERVAL_FRAMES == 0:
            contacts = (
                self.newton_world.contact_breakdown() if self.newton_world is not None else {}
            )
            _log(
                "[ProM6ClothingDrag] final verify: "
                f"center={_array_string(center)}, "
                f"conveyor_state={self.release_conveyor.state.name}, "
                f"contacts={contacts}"
            )
        if self.release_conveyor.state == ConveyorState.EXITED:
            self._report_result()
            if self.replay_recorder is None:
                raise RuntimeError("Successful operation has no replay recorder")
            self.replay_recorder.commit_success()
            self._set_phase(Phase.DONE)
            self.cleanup(hold_left=True)
            if self.timeline.is_playing():
                self.timeline.stop()

    def _report_result(self) -> None:
        _, _, final_center = self._node_aabb()
        _log(f"[ProM6ClothingDrag] Final node center: {_array_string(final_center)}")
        _log(
            "[ProM6ClothingDrag] Belt bounds: "
            f"min={_array_string(self.belt_bounds[0])}, "
            f"max={_array_string(self.belt_bounds[1])}"
        )
        if not center_exited_belt_negative_x(final_center, self.belt_bounds):
            raise RuntimeError(
                "Clothing did not leave through the belt -X edge while "
                f"remaining inside its Y span: center={final_center}"
            )
        _log(
            "[ProM6ClothingDrag][SUCCESS] Clothing exited "
            "ConveyorBelt_A06/Belt through its world -X edge"
        )

    def step(self, dt: float, context: object) -> None:
        del context
        if not self.initialized:
            self._initialize_after_play()
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0 or dt > 0.1:
            raise RuntimeError(f"Invalid physics callback dt={dt!r}")
        if self.initialized and self.newton_world is not None:
            if self._newton_hold_frames > 0:
                self._sync_newton_jaws()
                self.newton_world.reset_kinematic_history()
                self._newton_hold_frames -= 1
            else:
                self._sync_newton_jaws()
                self.newton_world.step(dt)
        phase_at_frame_start = self.phase
        right_task_phases = (
            Phase.PREPARE_RIGHT_HANDOFF,
            Phase.RIGHT_SEARCH_IK,
            Phase.RIGHT_MOVE_TO_PREGRASP,
            Phase.RIGHT_VERTICAL_APPROACH,
            Phase.RIGHT_CLOSE,
            Phase.RIGHT_GRIP_SETTLE,
            Phase.RIGHT_LIFT,
            Phase.RIGHT_DRAG_TO_BELT,
            Phase.RIGHT_RELEASE,
            Phase.RIGHT_RETURN_READY,
            Phase.FINAL_VERIFY,
        )
        if self.phase in right_task_phases:
            self._write_left_q(LEFT_READY_JOINTS)
        else:
            self._write_right_ready()
        self._cached_nodes = None
        self.phase_frame += 1
        self.phase_time += dt
        timeout = self.phase_timeouts.get(self.phase)
        if timeout is not None and self.phase_time > timeout:
            raise RuntimeError(f"Phase {self.phase.name} exceeded its {timeout:.1f} second timeout")
        self._deformable_nodes()
        self._capture_replay_frame(dt)

        if self.phase == Phase.SETTLE:
            self._step_settle()
        elif self.phase == Phase.WAITING:
            self._write_left_q(self.left_commanded_q)
            self._set_gripper_target(GRIPPER_OPEN_POSITION_RAD)
            if self.phase_time >= self.timing.waiting_seconds:
                self._set_phase(Phase.SEARCH_PREGRASP_IK)
        elif self.phase == Phase.SEARCH_PREGRASP_IK:
            self._step_search_pregrasp_ik()
        elif self.phase == Phase.MOVE_TO_PREGRASP:
            self._step_joint_segment(
                self.pregrasp_start_q,
                self.pregrasp_joint_target,
                self.timing.left_move_pregrasp_seconds,
                dt,
                Phase.VERTICAL_APPROACH,
            )
        elif self.phase == Phase.VERTICAL_APPROACH:
            self._step_approach(dt)
        elif self.phase == Phase.CLOSE_GRIPPER:
            self._step_close_gripper(dt)
        elif self.phase == Phase.GRIP_SETTLE:
            self._step_grip_settle(dt)
        elif self.phase == Phase.LEFT_LIFT:
            self._step_left_lift(dt)
        elif self.phase == Phase.DRAG:
            self._step_drag(dt)
        elif self.phase == Phase.CENTER_CORRECT:
            self._step_center_correct(dt)
        elif self.phase == Phase.LOWER:
            self._step_lower(dt)
        elif self.phase == Phase.PLACE_SETTLE:
            self._step_place_settle(dt)
        elif self.phase == Phase.LEFT_OPEN:
            self._step_left_open(dt)
        elif self.phase == Phase.LEFT_RETURN_READY:
            self._step_left_return_ready(dt)
        elif self.phase == Phase.PREPARE_RIGHT_HANDOFF:
            self._step_prepare_right_handoff()
        elif self.phase == Phase.RIGHT_SEARCH_IK:
            self._step_right_search_ik()
        elif self.phase == Phase.RIGHT_MOVE_TO_PREGRASP:
            self._step_right_joint_segment(
                self.right_pregrasp_start_q,
                self.right_pregrasp_joint_target,
                self.timing.right_move_pregrasp_seconds,
                dt,
                Phase.RIGHT_VERTICAL_APPROACH,
            )
        elif self.phase == Phase.RIGHT_VERTICAL_APPROACH:
            self._step_right_approach(dt)
        elif self.phase == Phase.RIGHT_CLOSE:
            self._step_right_close(dt)
        elif self.phase == Phase.RIGHT_GRIP_SETTLE:
            self._step_right_grip_settle(dt)
        elif self.phase == Phase.RIGHT_LIFT:
            self._step_right_lift(dt)
        elif self.phase == Phase.RIGHT_DRAG_TO_BELT:
            self._step_right_drag_to_belt(dt)
        elif self.phase == Phase.RIGHT_RELEASE:
            self._step_right_release()
        elif self.phase == Phase.RIGHT_RETURN_READY:
            self._step_right_return_ready(dt)
        elif self.phase == Phase.FINAL_VERIFY:
            self._step_final_verify(dt)
        if not self.cleaned and phase_at_frame_start in (
            Phase.RIGHT_RELEASE,
            Phase.RIGHT_RETURN_READY,
            Phase.FINAL_VERIFY,
        ):
            self._step_release_conveyor(dt)
        if not self.cleaned:
            self._log_progress()

    def cleanup(self, *, hold_left: bool) -> None:
        if self.cleaned:
            return
        self.cleaned = True
        cleanup_errors: list[tuple[str, BaseException]] = []
        if hold_left and self.initialized and self.left_commanded_q is not None:
            try:
                self._write_left_q(self.left_commanded_q)
            except Exception as error:
                cleanup_errors.append(("hold left arm", error))
            try:
                self._write_right_ready()
            except Exception as error:
                cleanup_errors.append(("hold right arm ready", error))
        if self.release_conveyor is not None:
            try:
                self.release_conveyor.cleanup()
            except Exception as error:
                cleanup_errors.append(("clean release conveyor", error))
        if self.replay_recorder is not None:
            try:
                self.replay_recorder.discard()
            except Exception as error:
                cleanup_errors.append(("discard uncommitted replay frames", error))
        try:
            self._release_attachment()
        except Exception as error:
            cleanup_errors.append(("remove shared attachment", error))
        try:
            restore_collision_attributes(self.stage, self.saved_collisions)
        except Exception as error:
            cleanup_errors.append(("restore gripper collisions", error))
        finally:
            self.saved_collisions.clear()
        try:
            remove_owned_runtime_prims(self.stage)
        except Exception as error:
            cleanup_errors.append(("remove runtime Prims", error))
        try:
            self._deregister_callbacks()
        except Exception as error:
            cleanup_errors.append(("deregister callbacks", error))
        finally:
            globals()[CONTROLLER_GLOBAL_NAME] = None
        for action, error in cleanup_errors:
            _log(f"[ProM6ClothingDrag][WARNING] Cleanup could not {action}: {error}")
        _log("[ProM6ClothingDrag] Runtime Prims/collision overrides cleaned")

    def fail(self, error: BaseException) -> None:
        if self.failed:
            return
        self.failed = True
        self.failure_reason = f"{type(error).__name__}: {error}"
        self.phase = Phase.FAILED
        _LOGGER.error("分拣控制器失败", error=self.failure_reason)
        traceback.print_exc()
        try:
            self.cleanup(hold_left=True)
        finally:
            if self.timeline.is_playing():
                self.timeline.stop()

    def _deregister_callbacks(self) -> None:
        for attribute, global_name in (
            ("callback_id", CALLBACK_GLOBAL_NAME),
            ("stop_callback_id", STOP_CALLBACK_GLOBAL_NAME),
        ):
            callback_id = getattr(self, attribute)
            if callback_id is not None:
                try:
                    SimulationManager.deregister_callback(callback_id)
                except Exception:
                    pass
                setattr(self, attribute, None)
            globals()[global_name] = None


class SortingPlanner(BasePlanner):
    """Pro-M6 双臂分拣:入带抓包裹 → 平台 → 出带,Newton 接触承重.

    ``place_robot``::

        init_planner(robot)                    # 配 GPU 物理 + 去 PhysX 柔性 + 建 Newton 软袋

    每集 ``run_episode``::

        begin_episode(ep, attempt)             # 注册物理回调,状态机开始跑
        loop: on_physics_step(robot)           # 读阶段/动作,判断本集是否结束
        reset()                                # 重试前清回调,再重新装配
    """

    config_cls = SortingPlannerConfig

    def __init__(self, config: SortingPlannerConfig | Any | None = None, **kwargs: Any) -> None:
        """写入 ``[planner]`` 配置;控制器留到 ``init_planner`` 再建.

        Args:
            config: 已构造配置;None 则从 ``[planner]`` 加载.
            **kwargs: 工厂多余字段,丢弃.

        Returns:
            None.
        """
        super().__init__()
        self.attached = False
        self.place_pose: Any | None = None
        del kwargs
        self.config: SortingPlannerConfig = config or self.config_cls.from_settings("planner")
        self.state["timeout_s"] = float(self.config.timeout_s)

        self._controller: Any | None = None  # 分拣控制器实例
        self._robot: Any | None = None  # 已 arrange 的 ProM6Robot
        self._callback_ids: list[int] = []  # 已注册的 SimulationManager 回调
        self._physics_dt = 1.0 / float(self.config.physics_hz)  # manual 模式下的步长
        self._grasp_seed = self.config.grasp_random_seed or int(np.random.SeedSequence().entropy)
        self._previous_grasp_offsets: dict[str, np.ndarray] = {}
        self._episode_grasp_seed: int | None = None

    # ============= 规划流水线 <==========

    def init_planner(self, robot: Any) -> None:
        """``place_robot`` 末尾:配 GPU 物理,去掉袋子 PhysX 柔性,建 Newton 接触世界.

        Args:
            robot: 已 ``arrange`` 的 Pro-M6 机器人.

        Returns:
            None.
        """
        _load_runtime_dependencies()

        self._robot = robot
        sim_world = robot.sim_world
        assert sim_world is not None, "robot 尚未绑定 sim_world"
        stage = sim_world.stage
        assert stage is not None, "stage 未初始化"

        cfg = self.config
        apply_settings(
            robot_path=cfg.robot_prim_path,
            clothing_root_path=cfg.clothing_root_prim_path,
            clothing_surface_path=cfg.clothing_surface_prim_path,
            clothing_volume_proxy_path=cfg.clothing_volume_proxy_prim_path,
            support_path=cfg.support_prim_path,
            belt_path=cfg.belt_prim_path,
            left_gripper_path=cfg.left_gripper_prim_path,
            right_gripper_path=cfg.right_gripper_prim_path,
            runtime_root_path=cfg.runtime_root_prim_path,
            replay_cache_dir=cfg.replay_cache_dir,
            physics_device=cfg.physics_device,
            physics_hz=cfg.physics_hz,
        )

        sim_world.stop()
        require_timeline_stopped(stage)
        require_uninitialized_physics()
        remove_owned_runtime_prims(stage)
        configure_gpu_physics(stage)

        timing = self._build_timing()
        self._controller = ProM6LeftClothingDragController(
            stage,
            timing,
            articulation=getattr(robot, "articulation", None),
            keep_label_up=cfg.keep_label_up,
            label_up_min_cos=cfg.label_up_min_cos,
            left_lift_height_m=cfg.left_lift_height_m,
            left_setdown_offset_m=cfg.left_setdown_offset_m,
        )
        self._controller.prepare_before_play()

        finger_paths = list(cfg.left_finger_prim_paths) + list(cfg.right_finger_prim_paths)
        if not finger_paths:
            try:
                finger_paths = discover_finger_meshes(
                    stage, LEFT_GRIPPER_PATH
                ) + discover_finger_meshes(stage, RIGHT_GRIPPER_PATH)
            except RuntimeError:
                finger_paths = []
        left_tcp, right_tcp = discover_tool_centers(stage, ROBOT_PATH)
        tcp_jaws = [(left_tcp, "left"), (right_tcp, "right")]
        infeed_path = resolve_infeed_prim_path(stage, cfg.infeed_prim_path)
        params = NewtonSortingParams(
            youngs_modulus=float(cfg.youngs_modulus),
            poisson_ratio=float(cfg.poisson_ratio),
            density=float(cfg.density),
            particle_radius=float(cfg.particle_radius),
            soft_contact_mu=float(cfg.soft_contact_mu),
            shape_mu=float(cfg.shape_mu),
            physics_hz=float(cfg.physics_hz),
            sim_substeps=int(cfg.newton_substeps),
            vbd_iterations=int(cfg.vbd_iterations),
            device=str(cfg.physics_device),
            belt_speed_mps=float(cfg.belt_speed_mps),
            min_lift_height_m=float(cfg.min_lift_height_m),
        )
        newton_world = NewtonSortingWorld(
            stage,
            params,
            clothing_surface_path=cfg.clothing_surface_prim_path,
            support_prim_path=cfg.support_prim_path,
            belt_prim_path=cfg.belt_prim_path,
            infeed_prim_path=infeed_path,
            finger_prim_paths=finger_paths,
            tcp_jaw_prims=tcp_jaws,
            label_uv_bounds=cfg.label_uv_bounds if cfg.keep_label_up else None,
        )
        self._controller.bind_newton(newton_world, min_lift_height_m=float(cfg.min_lift_height_m))
        self.state["ready"] = True
        self._physics_dt = 1.0 / float(cfg.physics_hz)
        self.log.info(
            "分拣规划器已就绪",
            drive_mode=cfg.drive_mode,
            grasp_point=self._controller.grasp_point.tolist(),
            support_center=self._controller.support_center.tolist(),
            newton_particles=int(newton_world.model.particle_count),
            infeed=infeed_path,
            fingers=len(finger_paths),
            left_tcp=left_tcp,
            right_tcp=right_tcp,
        )

    def begin_episode(self, episode_index: int, *, attempt: int = 1) -> None:
        """每集 Play 之后调用:注册物理回调,状态机从 SETTLE 开始跑.

        Args:
            episode_index: 0-based 集号,仅用于日志.
            attempt: 1-based 尝试序号,仅用于日志.

        Returns:
            None.
        """
        if self._controller is None:
            self.log.warning("尚未 init_planner,跳过 begin_episode")
            return
        sequence = np.random.SeedSequence([self._grasp_seed, episode_index, attempt])
        self._episode_grasp_seed = int(sequence.generate_state(1)[0])
        controller = self._controller
        controller.grasp_jitter_fraction_xy = self.config.grasp_jitter_fraction_xy
        controller.grasp_random_candidates = self.config.grasp_random_candidates
        controller.grasp_min_distance_m = self.config.grasp_min_distance_m
        controller.grasp_rngs = {
            side: np.random.default_rng(child)
            for side, child in zip(("left", "right"), sequence.spawn(2), strict=True)
        }
        controller.previous_grasp_offsets = {
            side: offset.copy() for side, offset in self._previous_grasp_offsets.items()
        }
        if self.config.drive_mode == "callback":
            self._register_callbacks()
        self.log.info(
            "分拣本集开始",
            episode=episode_index,
            attempt=attempt,
            drive_mode=self.config.drive_mode,
            grasp_seed=self._episode_grasp_seed,
        )

    def remember_successful_grasps(self) -> None:
        """保留上一成功集的相对抓点,下一集采样时避开其邻域."""
        if self._controller is not None and self.phase == "done":
            self._previous_grasp_offsets = {
                side: getattr(self._controller, f"_{side}_grasp_offset").copy()
                for side in ("left", "right")
            }

    def on_physics_step(self, robot: Any) -> bool:
        """每步:manual 模式推进状态机,并把阶段/动作同步给 Job.

        Args:
            robot: 当前机器人;关节由控制器直接写,这里不用.

        Returns:
            本集是否已结束(DONE / FAILED).
        """
        del robot  # 关节由控制器经 Articulation 直接下发
        controller = self._controller
        if controller is None:
            return True
        if self.config.drive_mode == "manual" and not controller.cleaned:
            try:
                controller.step(self._physics_dt, None)
            except Exception as exc:  # noqa: BLE001  # 与原脚本一致:失败即收尾
                controller.fail(exc)
        self.state["last_action"] = self._read_commanded_action(controller)
        return self.phase in ("done", "failed")

    # ============= 辅助函数 <==========

    @property
    def phase(self) -> str:
        """控制器当前阶段;Job 用它判定 success / failed;控制器未建时取 ``state``.

        Returns:
            ``done`` / ``failed`` / 小写阶段名;控制器未建时为占位阶段.
        """
        controller = self._controller
        if controller is None:
            return str(self.state["phase"])
        return str(controller.phase.name).lower()

    def _build_timing(self) -> Any:
        """按 ``timing_overrides`` 覆盖默认时长表.

        Returns:
            校验过的 ``TimingConfig``.

        Raises:
            ValueError: 覆盖了不存在的时长字段.
        """
        overrides = dict(self.config.timing_overrides)
        if not overrides:
            return TASK_TIMING
        known = set(TimingConfig.__dataclass_fields__)
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"timing_overrides 含未知字段 {unknown};可选: {sorted(known)}")
        return replace(TASK_TIMING, **{k: float(v) for k, v in overrides.items()})

    def _register_callbacks(self) -> None:
        """注册 ``PHYSICS_POST_STEP`` 与 ``SIMULATION_STOPPED`` 回调.

        与原脚本一致:物理步后推进状态机,时间轴停止时收尾.

        Returns:
            None.
        """
        from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

        self._deregister_callbacks()
        controller = self._controller
        assert controller is not None

        def physics_callback(dt: float, context: object) -> None:
            try:
                controller.step(dt, context)
            except Exception as exc:  # noqa: BLE001  # 单帧异常转为 FAILED,不炸进程
                controller.fail(exc)

        def stopped_callback(*_args: Any, **_kwargs: Any) -> None:
            if controller.cleaned:
                return
            controller.cleanup(hold_left=True)

        self._callback_ids = [
            SimulationManager.register_callback(
                physics_callback, event=SimulationEvent.PHYSICS_POST_STEP
            ),
            SimulationManager.register_callback(
                stopped_callback, event=SimulationEvent.SIMULATION_STOPPED
            ),
        ]
        controller.callback_id = self._callback_ids[0]
        controller.stop_callback_id = self._callback_ids[1]

    def _deregister_callbacks(self) -> None:
        """注销本规划器注册过的回调;重复调用安全.

        Returns:
            None.
        """
        if not self._callback_ids:
            return
        from isaacsim.core.simulation_manager import SimulationManager

        for callback_id in self._callback_ids:
            try:
                SimulationManager.deregister_callback(callback_id)
            except Exception:  # noqa: BLE001  # 已随时间轴停止注销过
                pass
        self._callback_ids = []

    @staticmethod
    def _read_commanded_action(controller: Any) -> np.ndarray | None:
        """本步下发的双臂关节目标(左 7 + 右 7),供 recorder 落盘.

        Args:
            controller: 分拣控制器.

        Returns:
            长度 14 的关节目标;状态机尚未初始化则为 ``None``.
        """
        left = getattr(controller, "left_commanded_q", None)
        right = getattr(controller, "right_commanded_q", None)
        if left is None and right is None:
            return None
        parts = [np.asarray(q, dtype=float).reshape(-1) for q in (left, right) if q is not None]
        return np.concatenate(parts)

    def reset(self) -> None:
        """重试前:收掉旧控制器与回调,再按同一配置重新装配.

        Returns:
            None.
        """
        self.state["phase"] = "idle"
        self.state["last_action"] = None
        self.attached = False
        robot = self._robot
        self.cleanup()
        if robot is not None:
            self.init_planner(robot)

    def cleanup(self) -> None:
        """Job ``close`` / 重试时调用:清临时 prim、碰撞覆盖与回调.

        Returns:
            None.
        """
        controller = self._controller
        if controller is not None:
            try:
                controller.cleanup(hold_left=False)
            except Exception as exc:  # noqa: BLE001  # 收尾失败不阻断 Job 退出
                self.log.warning("清理分拣控制器失败", error=str(exc))
        self._deregister_callbacks()
        self._controller = None
        self.state["ready"] = False

    def get_meta(self) -> dict[str, Any]:
        """并入 ``last_stats`` 的规划器快照.

        Returns:
            阶段、右臂拖拽距离与回放帧数.
        """
        controller = self._controller
        if controller is None:
            return {"planner_phase": self.phase}
        recorder = getattr(controller, "replay_recorder", None)
        return {
            "planner_phase": self.phase,
            "failure_reason": getattr(controller, "failure_reason", None),
            "grasp_seed": self._episode_grasp_seed,
            "grasp_run_seed": self._grasp_seed,
            "grasp_jitter_fraction_xy": list(self.config.grasp_jitter_fraction_xy),
            **{
                f"{side}_grasp_position": (
                    value.tolist()
                    if (value := getattr(controller, f"{side}_grasp_position", None)) is not None
                    else None
                )
                for side in ("left", "right")
            },
            **{
                f"{side}_grasp_offset_xy": getattr(controller, f"_{side}_grasp_offset")[:2].tolist()
                for side in ("left", "right")
            },
            "right_drag_distance_m": float(
                getattr(controller, "right_actual_drag_distance", 0.0) or 0.0
            ),
            "replay_frames": int(getattr(recorder, "frame_count", 0) or 0),
            "label_up_cos_at_handoff": controller.label_up_cos_at_handoff,
        }

    @property
    def timeout_s(self) -> float:
        """Job 墙钟超时(秒)."""
        return float(self.state["timeout_s"])

    @property
    def ready(self) -> bool:
        """求解器是否已加载成功."""
        return bool(self.state["ready"])

    @property
    def last_action(self) -> np.ndarray | None:
        """最近一次下发的动作副本;尚未规划时为 None."""
        action = self.state["last_action"]
        return None if action is None else np.asarray(action).copy()

    def pick_and_place(
        self,
        pick_pos: np.ndarray,
        place_pos: np.ndarray,
        start_joint_angles: np.ndarray,
    ) -> bool:
        """编排抓放并规划第一段轨迹;默认不支持,返回 False.

        Args:
            pick_pos: 抓取点世界坐标.
            place_pos: 放置点世界坐标.
            start_joint_angles: 规划起点关节角.

        Returns:
            True 表示已编排并规划成功;默认不支持.
        """
        del pick_pos, place_pos, start_joint_angles
        return False

    @property
    def log(self) -> Any:
        """返回绑定组件名的日志器。"""
        return get_logger(f"{type(self).__module__}.{type(self).__qualname__}").bind(
            component=type(self).__qualname__
        )
