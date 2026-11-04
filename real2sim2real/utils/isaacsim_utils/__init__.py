"""Isaac Sim / USD 工具，按职责组织并在包入口统一导出常用符号。

- material_utils：材质 UV、网格表面区域和法线计算。
- physics_utils：碰撞、刚体及摩擦等物理属性工具。
- robot_utils：关节、articulation 及关联刚体控制。
- camera_utils：相机配置、传感器与 RGB / 深度 / 点云等数据。
- openusd_utils：USD 路径、舞台、引用、位姿、加载同步与包围盒。

pxr / isaacsim 仅在函数内 import，纯 NumPy 工具可在未启动 Kit 时使用。
"""

from __future__ import annotations

from real2sim2real.utils.isaacsim_utils.camera_utils import (
    apply_camera_clipping,
    apply_camera_intrinsics,
    bind_camera_sensor,
    camera_matches_prims,
    camera_prim_path,
    filter_camera_targets,
    hide_camera_visuals,
    iter_bound_cameras,
    normalize_camera_key,
    read_sensor_data,
    resolve_annotator,
    spawn_camera,
)
from real2sim2real.utils.isaacsim_utils.material_utils import (
    surface_patch_normal,
    uv_patch_triangles,
)
from real2sim2real.utils.isaacsim_utils.openusd_utils import (
    WORLD_BACKGROUND_PRIM,
    WORLD_MOUNT_ROOTS,
    WORLD_ROBOTS_PRIM,
    WORLD_SCENE_PRIM,
    add_usd_reference,
    apply_pose,
    ensure_prim_loaded,
    ensure_world_mount_roots,
    ensure_xform,
    get_stage,
    is_descendant,
    is_world_frame,
    mark_ignore_metrics_assembler,
    normalize_prim_path,
    parent_pose,
    place_asset,
    prim_aabb,
    pump_frames,
    read_prim_world_pose,
    resolve_usd_path,
    spawn_reference,
    spawn_subprim_reference,
)
from real2sim2real.utils.isaacsim_utils.physics_utils import (
    disable_subtree_collision,
    ensure_static_collider,
)
from real2sim2real.utils.isaacsim_utils.robot_utils import set_subtree_puppet

__all__ = [
    "add_usd_reference",
    "apply_camera_clipping",
    "apply_camera_intrinsics",
    "apply_pose",
    "bind_camera_sensor",
    "camera_matches_prims",
    "camera_prim_path",
    "disable_subtree_collision",
    "ensure_prim_loaded",
    "ensure_static_collider",
    "ensure_world_mount_roots",
    "ensure_xform",
    "filter_camera_targets",
    "get_stage",
    "hide_camera_visuals",
    "is_descendant",
    "is_world_frame",
    "iter_bound_cameras",
    "mark_ignore_metrics_assembler",
    "normalize_camera_key",
    "normalize_prim_path",
    "parent_pose",
    "place_asset",
    "prim_aabb",
    "pump_frames",
    "read_prim_world_pose",
    "read_sensor_data",
    "resolve_annotator",
    "resolve_usd_path",
    "set_subtree_puppet",
    "spawn_camera",
    "spawn_reference",
    "spawn_subprim_reference",
    "surface_patch_normal",
    "uv_patch_triangles",
    "WORLD_BACKGROUND_PRIM",
    "WORLD_MOUNT_ROOTS",
    "WORLD_ROBOTS_PRIM",
    "WORLD_SCENE_PRIM",
]
