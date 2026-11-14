"""HALO task scene: home_010 living room with static colliders, an optional desk and target sofa.

The living room is loaded by the world as a background reference (``sim_world``
``background_usd_path``). On the runtime stage only, this scene reproduces HALO
``physx_world.build``:

- every home mesh becomes a static triangle-mesh collider (4/0.5 mm offsets, μs 1.0, μd 0.8);
  flat floor meshes (< 2 cm thick, top within 3 cm of z=0) and decals are disabled and an
  invisible box ``FloorCollider`` carries the robot instead; the one home rigid body is removed;
- an optional Blender desk (crawl / crouch) referenced under ``/World/scene/Desk``;
- an optional target seat (sit) that must stay the original, unmoved home sofa;
- the third-person RGB camera, fixed in the world or following the measured pelvis (crawl).

Sections: configuration, scene backend.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.scene.base import BaseScene
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.cameras import CameraConfig, CameraPose, create_cameras
from real2sim2real.utils.halo_utils import (
    ContactPrefixes,
    DeskSpec,
    FollowCameraFilter,
    look_at_quat_wxyz,
)

Vector3 = tuple[float, float, float]


# =====================================================================================
# Configuration
# =====================================================================================


class HaloDeskConfig(BaseModel):
    """Desk asset, placement (m, deg) and the dimensions used by the passage checks."""

    model_config = ConfigDict(extra="forbid")
    usd_path: str
    asset_prim: str = "/Desk"
    xy: tuple[float, float]
    yaw_deg: float
    base_z_m: float = 0.0
    scale: Vector3 = (1.0, 1.0, 1.0)
    depth_m: float = Field(gt=0, description="along the walking direction")
    width_m: float = Field(gt=0)
    top_height_m: float = Field(gt=0)
    clearance_z_m: float = Field(gt=0, description="underside of the crossbeam")
    column_spacing_m: float = Field(gt=0, description="column centre distance")


class HaloSeatConfig(BaseModel):
    """Target seat inside the home asset; it must not be moved, scaled or edited."""

    model_config = ConfigDict(extra="forbid")
    prim: str = Field(description="path below the background prim, e.g. sofa_main/sofa_254_002")
    contact_group: str = Field(description="home subtree counted as seat contact, e.g. sofa_main")
    cushion_center_xyz_m: Vector3
    expected_bounds_m: tuple[Vector3, Vector3] = Field(
        description="world AABB of the original seat after the background offset"
    )


class HaloFollowCameraConfig(BaseModel):
    """Measured-pelvis follow camera (HALO crawl_v3), filtered at the 50 Hz control rate."""

    model_config = ConfigDict(extra="forbid")
    task_yaw_deg: float = 90.0
    eye_offset_task_m: Vector3 = (2.0, -1.45, 0.52)
    look_ahead_task_m: Vector3 = (0.2, 0.0, 0.0)
    target_height_base_gain: float = 0.56
    target_height_offset_m: float = 0.21
    position_time_constant_s: float = Field(default=0.28, gt=0)
    height_time_constant_s: float = Field(default=0.38, gt=0)


class HaloThirdPersonCameraConfig(BaseModel):
    """Third-person RGB camera; lens in mm (D455 colour sensor: 1.93 / 3.896 mm).

    Width/height are scalars so a debug config can inherit and override them (TOML arrays
    are appended, not replaced, when configs are merged).
    """

    model_config = ConfigDict(extra="forbid")
    name: str = "rear"
    prim_path: str = "/World/Cameras/rear"
    width: int = Field(default=640, gt=0, multiple_of=2)
    height: int = Field(default=360, gt=0, multiple_of=2)
    focal_length: float = Field(default=1.93, gt=0)
    horizontal_aperture: float = Field(default=3.896, gt=0)
    clipping_range: tuple[float, float] = (0.02, 100.0)
    mode: Literal["fixed", "follow"] = "fixed"
    eye_m: Vector3 = Field(description="fixed pose, or the pose before the first follow update")
    target_m: Vector3
    follow: HaloFollowCameraConfig | None = None

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        if self.mode == "follow" and self.follow is None:
            raise ValueError("third_person.mode='follow' needs [scene.third_person.follow]")
        return self

    def camera_config(self) -> CameraConfig:
        return CameraConfig(
            name=self.name,
            prim_path=self.prim_path,
            pose_relative_to="/World",
            pose=CameraPose(
                position=self.eye_m,
                orientation_wxyz=tuple(look_at_quat_wxyz(self.eye_m, self.target_m)),
            ),
            resolution=(self.width, self.height),
            clipping_range=self.clipping_range,
            focal_length=self.focal_length,
            horizontal_aperture=self.horizontal_aperture,
        )


class HaloHomeSceneConfig(BaseConfig):
    """Static home colliders, floor proxy, optional desk / seat and the third-person camera."""

    backend: str = "halo_home"
    floor_proxy_center_m: Vector3 = (2.9, -0.7, -0.1)
    floor_proxy_size_m: Vector3 = (3.4, 7.8, 0.2)
    floor_mesh_max_thickness_m: float = Field(default=0.02, gt=0)
    floor_mesh_max_top_m: float = Field(default=0.03, gt=0)
    contact_offset_m: float = Field(default=0.004, gt=0)
    rest_offset_m: float = Field(default=0.0005, ge=0)
    static_friction: float = Field(default=1.0, ge=0)
    dynamic_friction: float = Field(default=0.8, ge=0)
    restitution: float = Field(default=0.0, ge=0, le=1)
    desk: HaloDeskConfig | None = None
    seat: HaloSeatConfig | None = None
    third_person: HaloThirdPersonCameraConfig


# =====================================================================================
# Scene backend
# =====================================================================================

DESK_PATH = "/World/scene/Desk"
FLOOR_PATH = "/World/scene/FloorCollider"
MATERIAL_PATH = "/World/scene/PhysicsMaterial"


class HaloHomeScene(BaseScene):
    """home_010 living room prepared for free-base G1 contact in PhysX."""

    config_cls = HaloHomeSceneConfig

    def __init__(self, config: HaloHomeSceneConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("scene")
        self.background_path = ""
        self.cameras = [self.config.third_person.camera_config()]
        self.follow: FollowCameraFilter | None = None
        self._rear_op: Any = None

    @property
    def desk(self) -> DeskSpec | None:
        desk = self.config.desk
        if desk is None:
            return None
        return DeskSpec(
            xy=desk.xy,
            yaw_deg=desk.yaw_deg,
            base_z=desk.base_z_m,
            depth_m=desk.depth_m,
            width_m=desk.width_m,
            clearance_z_m=desk.clearance_z_m,
            column_spacing_m=desk.column_spacing_m,
        )

    def contact_prefixes(self, robot_prefix: str) -> ContactPrefixes:
        """Desk → table, seat subtree → chair, other home prims → obstacle, rest → ground."""
        seat = self.config.seat
        return ContactPrefixes(
            robot=robot_prefix,
            table=DESK_PATH + "/" if self.config.desk else None,
            seat=f"{self.background_path}/{seat.contact_group}/" if seat else None,
            home=self.background_path + "/",
        )

    def init_scene(self, sim_world: Any) -> None:
        """Author colliders on the home, add the floor proxy, desk and third-person camera."""
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        if sim_world.physics_engine != "physx":
            raise ValueError("halo_home requires sim_world.physics_engine='physx'")
        self.sim_world = sim_world
        stage = sim_world.stage
        self.background_path = sim_world.config.background_prim_path
        background = stage.GetPrimAtPath(self.background_path)
        if not sim_world.config.background_usd_path or not background:
            raise ValueError("halo_home needs sim_world.background_usd_path (home_010.usdc)")
        cfg = self.config
        material = UsdShade.Material.Define(stage, MATERIAL_PATH)
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(cfg.static_friction)
        api.CreateDynamicFrictionAttr(cfg.dynamic_friction)
        api.CreateRestitutionAttr(cfg.restitution)
        self._material = material
        report = {"home": self._static_colliders(background, skip_floor=True)}
        floor = UsdGeom.Cube.Define(stage, FLOOR_PATH)
        floor.CreateSizeAttr(1.0)
        floor.AddTranslateOp().Set(Gf.Vec3d(*cfg.floor_proxy_center_m))
        floor.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*cfg.floor_proxy_size_m))
        floor.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        self._collider(floor.GetPrim(), triangle=False)
        if cfg.desk is not None:
            report["desk"] = self._add_desk(stage)
        if cfg.seat is not None:
            report["seat"] = self._check_seat(stage)
        self.sensors.update(create_cameras(stage, self.cameras))
        rear = stage.GetPrimAtPath(cfg.third_person.prim_path)
        self._rear_op = UsdGeom.Xformable(rear).GetOrderedXformOps()[0]
        if cfg.third_person.mode == "follow":
            self.follow = FollowCameraFilter(cfg.third_person.follow)
        self.state["scene_report"] = report

    def _collider(self, prim: Any, *, triangle: bool) -> None:
        from pxr import PhysxSchema, UsdPhysics, UsdShade

        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        if triangle:
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(self.config.contact_offset_m)
        collision.CreateRestOffsetAttr(self.config.rest_offset_m)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            self._material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    def _static_colliders(self, root: Any, *, skip_floor: bool) -> dict[str, Any]:
        """Triangle-mesh colliders on every mesh; flat floor meshes and decals disabled."""
        from pxr import Usd, UsdGeom, UsdPhysics

        for prim in list(Usd.PrimRange(root)):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        enabled, disabled, removed_bodies = [], [], []
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                removed_bodies.append(path)
            if not prim.IsA(UsdGeom.Mesh):
                continue
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            lo, hi = box.GetMin(), box.GetMax()
            # Ground geometry is found by its height, not its name: floor_lamp is an obstacle.
            floor = (
                skip_floor
                and hi[2] - lo[2] < self.config.floor_mesh_max_thickness_m
                and abs(hi[2]) < self.config.floor_mesh_max_top_m
            )
            if "/Decals/" in path or floor:
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                disabled.append(path)
                continue
            self._collider(prim, triangle=True)
            enabled.append(path)
        return {
            "static_colliders": len(enabled),
            "floor_lamp_colliders": sum("/floor_lamp/" in p for p in enabled),
            "disabled_meshes": [p.rsplit("/", 1)[-1] for p in disabled],
            "removed_rigid_bodies": removed_bodies,
        }

    def _add_desk(self, stage: Any) -> dict[str, Any]:
        from pxr import Gf, UsdGeom

        desk = self.config.desk
        parent = UsdGeom.Xform.Define(stage, DESK_PATH)
        half = math.radians(desk.yaw_deg) / 2
        pose = Gf.Matrix4d(1.0)
        pose.SetRotate(Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half))))
        pose.SetTranslateOnly(Gf.Vec3d(desk.xy[0], desk.xy[1], desk.base_z_m))
        parent.AddTransformOp().Set(pose)
        parent.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "layout").Set(Gf.Vec3d(*desk.scale))
        asset = stage.DefinePrim(f"{DESK_PATH}/Asset", "Xform")
        asset.GetReferences().AddReference(resolve_usd_path(desk.usd_path), desk.asset_prim)
        if not asset.GetChildren():
            raise RuntimeError(f"Desk asset has no prims: {desk.usd_path}{desk.asset_prim}")
        report = self._static_colliders(asset, skip_floor=False)
        box = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(asset)
        aligned = box.ComputeAlignedBox()
        report["world_bounds_m"] = [list(aligned.GetMin()), list(aligned.GetMax())]
        return report

    def _check_seat(self, stage: Any) -> dict[str, Any]:
        """The target seat must be the original visible, collidable and unedited home mesh."""
        from pxr import Usd, UsdGeom, UsdPhysics

        seat = self.config.seat
        prim = stage.GetPrimAtPath(f"{self.background_path}/{seat.prim}")
        if not prim:
            raise RuntimeError(f"Target seat not found: {seat.prim}")
        source = Usd.Stage.Open(resolve_usd_path(self.sim_world.config.background_usd_path))
        original = source.GetPrimAtPath(f"{source.GetDefaultPrim().GetPath()}/{seat.prim}")
        attributes = ("points", "faceVertexCounts", "faceVertexIndices")
        unchanged = bool(original) and all(
            np.array_equal(
                np.asarray(prim.GetAttribute(a).Get()), np.asarray(original.GetAttribute(a).Get())
            )
            for a in attributes
        )
        aligned = (
            UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedBox()
        )
        bounds = np.array([list(aligned.GetMin()), list(aligned.GetMax())])
        expected = np.asarray(seat.expected_bounds_m)
        return {
            "prim": str(prim.GetPath()),
            "visible": UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible,
            "collidable": bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),
            "mesh_unchanged": unchanged,
            "world_bounds_m": bounds.tolist(),
            "bounds_error_m": float(np.max(np.abs(bounds - expected))),
        }

    def arrange(self) -> None:
        """Record the desk pose and seat bounds."""
        if self.config.desk is not None:
            desk = self.config.desk
            half = math.radians(desk.yaw_deg) / 2
            self.poses["desk"] = (
                np.array([*desk.xy, desk.base_z_m]),
                np.array([math.cos(half), 0.0, 0.0, math.sin(half)]),
            )
        if self.config.seat is not None:
            self.state["seat_bounds_m"] = self.state["scene_report"]["seat"]["world_bounds_m"]

    def adjust(self) -> None:
        """Keep the configured placement."""

    def update_third_person(self, base: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Move the follow camera for the measured base; returns ``(eye, look target)``."""
        cam = self.config.third_person
        if self.follow is None:
            return np.asarray(cam.eye_m, dtype=float), np.asarray(cam.target_m, dtype=float)
        from pxr import Gf

        eye, target, quat = self.follow.step(base, dt)
        pose = Gf.Matrix4d(1.0)
        pose.SetRotate(Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:]))))
        pose.SetTranslateOnly(Gf.Vec3d(*map(float, eye)))
        self._rear_op.Set(pose)
        return eye, target
