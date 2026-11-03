"""分拣用 Newton 侧车:体积软袋 + 运动学刚体代理,不导入机器人 URDF.

刚体(Pro-M6 / 平台 / 皮带)仍由 PhysX 驱动.本模块只求解袋装包裹,每步把
指垫/皮带位姿同步成 kinematic box,再把表面顶点写回 USD Mesh.
须在 ``isaacsim.pip.newton`` 启用之后构造 :class:`NewtonSortingWorld`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from real2sim2real.core.logger import get_logger

_LOGGER = get_logger("NewtonSorting")

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


def _matrix_pose(matrix: Gf.Matrix4d) -> tuple[np.ndarray, np.ndarray]:
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


# 首次四面体化时的袋表面局部顶点;多集必须写回,变形网格不能再做星形 tet.
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
        _LOGGER.info("已恢复袋表面静止网格", prim=mesh_path, n=int(cached.shape[0]))
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
        _LOGGER.info("已关掉体积代理", prim=volume_proxy_path)

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

    _LOGGER.info("已去掉袋子 PhysX 柔性", removed=len(removed), surface=clothing_surface_path)
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
            _LOGGER.info("ToolCenter 候选", label=label, chosen=chosen, all=candidates)
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
        _LOGGER.warning("指垫包围盒过大,已钳制", prim=str(prim.GetPath()), half=half.tolist())
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
    _LOGGER.info("指垫代理", gripper=gripper_path, prims=chosen, sizes=sizes)
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
        _LOGGER.info(
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
                _LOGGER.warning("Isaac 指垫位姿不可用,回退 USD", prim=prim_path)
        world = _prim_world_matrix(self.stage, prim_path, cache)
        center_world = world.Transform(Gf.Vec3d(*local_center))
        pos = np.array([center_world[0], center_world[1], center_world[2]], dtype=np.float64)
        _, quat = _matrix_pose(world)
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
