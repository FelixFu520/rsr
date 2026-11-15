"""Tabletop pick-and-place scene: a table, a target object and distractors in a home interior.

Migrated from Real2Sim2Real ``TabletopScene``
(``configs/tabletop-pickplace_franka_curobo_home000.toml``).
The home background is loaded by ``IsaacsimWorld`` (``background_usd_path``); this scene adds:

- the table at ``table.position_m`` / ``yaw_deg``; its visual meshes get static colliders
  (triangle meshes, as in the source ``ensure_static_collider``);
- objects at ``offset_m`` from the table centre (table frame) with their AABB bottom
  ``clearance_m`` above the measured table top; optional runtime mass overrides;
- fixed cameras (``SensorCameraConfig``; the source used the Sensing SG5 fisheye USD).

Sections: 1. configuration, 2. scene backend (lifecycle, layout, runtime views, contacts).
"""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import LoggerMixin
from real2sim2real.scene.base import BaseScene
from real2sim2real.utils.asset_utils import prim_aabb, resolve_usd_path
from real2sim2real.utils.cameras import validate_camera_list
from real2sim2real.utils.tabletop_pickplace_utils import (
    SensorCameraConfig,
    create_sensor_cameras,
    footprint_gap,
    quat_to_matrix,
    set_prim_pose,
    table_point,
    to_numpy,
    yaw_quat,
)

Vector3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]

# ================================================================================================
# 1. Configuration
# ================================================================================================


def _below(path: str, root: str) -> bool:
    return path.startswith(root + "/") and all(p.isidentifier() for p in path.split("/")[1:])


class TableConfig(BaseModel):
    """Table asset; ``position_m`` is the asset origin in the world (source ``[scene.table]``)."""

    model_config = ConfigDict(extra="forbid")
    usd_path: str
    prim_path: str = "/World/scene/Table"
    position_m: Vector3 = (0.45, -3.75, 0.415)
    yaw_deg: FiniteFloat = 0.0
    collision_approximation: Literal["none", "convexHull", "convexDecomposition"] = Field(
        default="none", description="static collider of the visual meshes; none = triangle mesh"
    )


class TabletopObjectConfig(BaseModel):
    """One object on the table (source ``[[scene.objects]]``)."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    usd_path: str
    prim_path: str
    body_prim: str = Field(default="", description="rigid-body prim relative to prim_path")
    role: Literal["target", "distractor"]
    offset_m: tuple[FiniteFloat, FiniteFloat] = Field(description="table-frame XY from centre")
    yaw_deg: FiniteFloat = 0.0
    clearance_m: FiniteFloat = Field(default=0.001, ge=0, description="bottom above table top")
    mass_kg: FiniteFloat | None = Field(default=None, gt=0, description="None keeps the asset")

    @property
    def body_path(self) -> str:
        return f"{self.prim_path}/{self.body_prim}" if self.body_prim else self.prim_path


class TabletopHomeSceneConfig(BaseConfig):
    """``[scene]`` of the tabletop tasks."""

    backend: str = "tabletop_home"
    prim_path: str = "/World/scene"
    table: TableConfig
    objects: list[TabletopObjectConfig]
    cameras: list[SensorCameraConfig] = Field(default_factory=list)
    min_object_gap_m: FiniteFloat = Field(
        default=0.02, ge=0, description="XY footprint clearance of a jittered target"
    )
    table_edge_margin_m: FiniteFloat = Field(default=0.05, ge=0)

    @model_validator(mode="after")
    def _validate_scene(self) -> Self:
        if [o.role for o in self.objects].count("target") != 1:
            raise ValueError("The tabletop scene needs exactly one target object")
        names = [o.name for o in self.objects]
        paths = [self.table.prim_path, *(o.prim_path for o in self.objects)]
        if len(set(names)) != len(names) or len(set(paths)) != len(paths):
            raise ValueError("Object names and prim paths must be unique")
        if not all(_below(p, self.prim_path) for p in paths):
            raise ValueError(f"Table and objects must live below {self.prim_path}")
        validate_camera_list(list(self.cameras))
        return self

    @property
    def target(self) -> TabletopObjectConfig:
        return next(o for o in self.objects if o.role == "target")

    @property
    def distractors(self) -> list[TabletopObjectConfig]:
        return [o for o in self.objects if o.role == "distractor"]


# ================================================================================================
# 2. Scene backend
# ================================================================================================


class TabletopHomeScene(BaseScene, LoggerMixin):
    """Table + objects + fixed cameras; per-episode target offsets; PhysX views of the objects.

    Lifecycle: ``init_scene`` references the assets, ``arrange`` places them on the measured
    table top and fills ``layout``; before each episode the Job calls ``set_target_offset``
    (stage edit while the timeline is stopped), then ``initialize_runtime`` after ``reset``.
    """

    config_cls = TabletopHomeSceneConfig

    def __init__(self, config: TabletopHomeSceneConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config: TabletopHomeSceneConfig = config or self.config_cls.from_settings("scene")
        self.cameras = list(self.config.cameras)
        self.capture_prims: dict[str, str] = {}
        self.layout: dict[str, Any] = {}
        self._nominal: dict[str, np.ndarray] = {}
        self._views: dict[str, Any] = {}

    # --- lifecycle --------------------------------------------------------------------------------
    def init_scene(self, sim_world: Any) -> None:
        """Reference table, objects and cameras below ``prim_path``."""
        from pxr import UsdGeom

        if sim_world.physics_engine != "physx":
            raise ValueError("tabletop_home uses PhysX colliders; set physics_engine='physx'")
        self.sim_world = sim_world
        stage = sim_world.stage
        UsdGeom.Xform.Define(stage, self.config.prim_path)
        for path, usd in [
            (self.config.table.prim_path, self.config.table.usd_path),
            *((o.prim_path, o.usd_path) for o in self.config.objects),
        ]:
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            if not prim.GetReferences().AddReference(resolve_usd_path(usd)):
                raise RuntimeError(f"Reference failed: {usd}")
        for obj in self.config.objects:
            if not stage.GetPrimAtPath(obj.body_path):
                raise RuntimeError(f"{obj.name}: rigid body prim missing: {obj.body_path}")
        self.capture_prims = create_sensor_cameras(stage, self.cameras)
        self.sensors.update(self.capture_prims)

    def arrange(self) -> None:
        """Place the table, add its colliders, put every object on the measured table top."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = self.sim_world.stage
        table = self.config.table
        self._set_pose(table.prim_path, np.asarray(table.position_m), yaw_quat(table.yaw_deg))
        bounds = prim_aabb(stage, table.prim_path, gprims_only=True)
        if bounds is None:
            raise RuntimeError("Table has no measurable geometry")
        colliders = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(table.prim_path)):
            if prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                    table.collision_approximation
                )
                colliders += 1
        table_top = float(bounds[1][2])
        self.poses["table"] = (np.asarray(table.position_m, float), yaw_quat(table.yaw_deg))
        self.layout = {
            "table_top_z": table_top,
            "table_center_xy": ((bounds[0][:2] + bounds[1][:2]) / 2).tolist(),
            "table_bounds_m": [bounds[0].tolist(), bounds[1].tolist()],
            "table_colliders": colliders,
            "objects": {},
        }
        for obj in self.config.objects:
            if obj.mass_kg is not None:
                UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(obj.body_path)).CreateMassAttr(
                    obj.mass_kg
                )
            xy = table_point(table.position_m, table.yaw_deg, obj.offset_m)
            quat = yaw_quat(obj.yaw_deg)
            self._set_pose(obj.prim_path, np.array([*xy, table_top]), quat)
            lo, hi = prim_aabb(stage, obj.prim_path, gprims_only=True)
            position = np.array([*xy, table_top + (table_top - lo[2]) + obj.clearance_m])
            self._set_pose(obj.prim_path, position, quat)
            lo, hi = prim_aabb(stage, obj.prim_path, gprims_only=True)
            body = self._body_world(obj)
            self._nominal[obj.name] = position
            self.poses[obj.name] = (position, quat)
            self.layout["objects"][obj.name] = {
                "role": obj.role,
                "prim_origin_m": position.tolist(),
                "body_origin_m": body.tolist(),
                "aabb_center_m": ((lo + hi) / 2).tolist(),
                "half_extents_m": ((hi - lo) / 2).tolist(),
                "mass_kg": obj.mass_kg,
            }
        self.log.info("桌面场景就绪", table_top_z=round(table_top, 4), colliders=colliders)

    # --- per-episode layout -----------------------------------------------------------------------
    def accept_offset(self, offset_xy: Any) -> bool:
        """True when the jittered target keeps ``min_object_gap_m`` and stays on the table."""
        target = self.layout["objects"][self.config.target.name]
        center = np.asarray(target["aabb_center_m"][:2]) + np.asarray(offset_xy, float)
        half = np.asarray(target["half_extents_m"][:2])
        lo, hi = (np.asarray(b[:2]) for b in self.layout["table_bounds_m"])
        margin = self.config.table_edge_margin_m
        if np.any(center - half < lo + margin) or np.any(center + half > hi - margin):
            return False
        for obj in self.config.distractors:
            other = self.layout["objects"][obj.name]
            gap = footprint_gap(center, half, other["aabb_center_m"], other["half_extents_m"])
            if gap < self.config.min_object_gap_m:
                return False
        return True

    def set_target_offset(self, offset_xy: Any) -> np.ndarray:
        """Move the target to its nominal pose + ``offset_xy`` (only while stopped)."""
        if self.sim_world.is_playing():
            raise RuntimeError("Target layout can only change while the timeline is stopped")
        target = self.config.target
        position = self._nominal[target.name] + np.array([*np.asarray(offset_xy, float), 0.0])
        self._set_pose(target.prim_path, position, yaw_quat(target.yaw_deg))
        self.poses[target.name] = (position, yaw_quat(target.yaw_deg))
        return position

    def place_target_xy(self, offset_xy: Any) -> np.ndarray:
        """World XY of a table-frame offset (place targets)."""
        table = self.config.table
        return table_point(table.position_m, table.yaw_deg, offset_xy)

    # --- runtime ----------------------------------------------------------------------------------
    def initialize_runtime(self) -> None:
        """Physics views of the target and distractors (after ``sim_world.reset``)."""
        from isaacsim.core.experimental.prims import RigidPrim

        self._views = {"target": RigidPrim(self.config.target.body_path)}
        if self.config.distractors:
            self._views["distractors"] = RigidPrim([o.body_path for o in self.config.distractors])
        # AABB centre in the body frame at the start pose (USD still holds the arranged pose).
        lo, hi = prim_aabb(self.sim_world.stage, self.config.target.prim_path, gprims_only=True)
        position, quat = self.target_pose()
        self.state["target_center_offset_m"] = quat_to_matrix(quat).T @ ((lo + hi) / 2 - position)

    def target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and wxyz orientation of the target rigid body."""
        p, q = self._views["target"].get_world_poses()
        return to_numpy(p)[0].astype(float), to_numpy(q)[0].astype(float)

    def distractor_positions(self) -> np.ndarray:
        """(K, 3) world positions of the distractor bodies."""
        if "distractors" not in self._views:
            return np.zeros((0, 3))
        p, _ = self._views["distractors"].get_world_poses()
        return to_numpy(p).astype(float).reshape(-1, 3)

    def target_center(self) -> np.ndarray:
        """World position of the target AABB centre (grasp point), following the body."""
        position, quat = self.target_pose()
        return position + quat_to_matrix(quat) @ self.state["target_center_offset_m"]

    def distractor_boxes(self, measured: bool = False) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """(name, AABB centre, half extents) of each distractor.

        ``measured`` follows the PhysX body (the arranged body-to-centre offset is kept), so
        planning obstacles match objects that rolled while settling.
        """
        positions = self.distractor_positions() if measured else None
        boxes = []
        for k, obj in enumerate(self.config.distractors):
            entry = self.layout["objects"][obj.name]
            center = np.asarray(entry["aabb_center_m"], float)
            if positions is not None:
                center = positions[k] + center - np.asarray(entry["body_origin_m"], float)
            boxes.append((obj.name, center, np.asarray(entry["half_extents_m"], float)))
        return boxes

    def contact_bodies(self) -> list[str]:
        return [o.body_path for o in self.config.objects]

    def contact_category(self, path: str) -> str | None:
        """Category of a PhysX actor path for contact accounting."""
        table = self.config.table.prim_path
        if path == table or path.startswith(table + "/"):
            return "table"
        for obj in self.config.objects:
            if path == obj.prim_path or path.startswith(obj.prim_path + "/"):
                return obj.role
        return None

    # --- helpers ----------------------------------------------------------------------------------
    def _set_pose(self, path: str, position: np.ndarray, quat: np.ndarray) -> None:
        set_prim_pose(self.sim_world.stage, path, position, quat)

    def _body_world(self, obj: TabletopObjectConfig) -> np.ndarray:
        from pxr import UsdGeom

        prim = self.sim_world.stage.GetPrimAtPath(obj.body_path)
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        return np.asarray(matrix.ExtractTranslation(), dtype=float)
