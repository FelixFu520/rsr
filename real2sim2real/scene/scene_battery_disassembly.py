"""Battery disassembly workcell (PhysX): workstation, battery pack, threaded screws, basket, pallet.

Sections:

1. Configuration (asset-frame proxy dimensions of Battery.usd, screw thread, pallet, lights).
2. ``BatteryWorkcellScene``: references Battery.usd below ``/World/scene`` and authors the
   physics on the runtime stage only (the asset file is never edited):

   - the asset's own triangle / convex-decomposition colliders of the pack, cover and basket are
     disabled and replaced by box proxies (hollow cover with screw holes, case block, open
     basket) so contacts stay well conditioned on the CPU solver;
   - each original screw mesh is re-instanced as a free rigid body (head, cross recess, shank
     colliders). Its thread is a small articulation anchored to the case:
     ``anchor -(prismatic slide)- slider -(revolute turn)- carrier`` with a PhysX mimic joint that
     ties ``slide = pitch * turn / 360°`` (right-hand thread, counter-clockwise loosens). A fixed
     joint holds the screw on the carrier and is disabled once the slide has used up the thread
     length, i.e. the screw comes out of the thread. Torque reaches the screw only through bit /
     recess contacts; viscous damping on ``turn`` stands in for thread friction;
   - the 12 kg cover is a dynamic body resting on the case and clamped by the screw heads.

The Job drives the lifecycle; the planner reads the states exposed here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import LoggerMixin
from real2sim2real.scene.base import BaseScene
from real2sim2real.utils.asset_utils import prim_aabb, resolve_usd_path
from real2sim2real.utils.battery_disassembly_utils import (
    Box,
    basket_boxes,
    boxes_centroid,
    cover_boxes,
    screw_head_boxes,
)
from real2sim2real.utils.cameras import CameraConfig, create_cameras, validate_camera_list

Pair = tuple[FiniteFloat, FiniteFloat]
Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]

# ================================================================================================
# 1. Configuration
# ================================================================================================


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MaterialConfig(_Strict):
    """PhysX material (friction coefficients are unitless)."""

    static_friction: FiniteFloat = Field(default=0.5, ge=0)
    dynamic_friction: FiniteFloat = Field(default=0.4, ge=0)
    restitution: FiniteFloat = Field(default=0.0, ge=0, le=1)


class CoverProxyConfig(_Strict):
    """Hollow cover proxy in the Battery.usd asset frame (before ``battery_yaw_deg``)."""

    mass_kg: FiniteFloat = Field(default=12.0, gt=0)
    flange_z_m: Pair = (0.8147, 0.8190)
    flange_x_half_m: FiniteFloat = 0.4221
    flange_y_m: Pair = (-0.6519, 0.6419)
    body_x_half_m: FiniteFloat = 0.4015
    body_y_m: Pair = (-0.6324, 0.6224)
    body_top_z_m: FiniteFloat = 0.899
    raised_y_m: Pair = (0.198, 0.6224)
    raised_top_z_m: FiniteFloat = 0.960
    wall_m: FiniteFloat = Field(default=0.004, gt=0)
    hole_half_m: FiniteFloat = Field(default=0.0065, gt=0, description="square hole half size")
    material: MaterialConfig = Field(
        default_factory=lambda: MaterialConfig(static_friction=0.6, dynamic_friction=0.5)
    )


class ScrewConfig(_Strict):
    """M10 screw: thread, mass and collider dimensions (screw frame: head underside at z=0)."""

    pitch_m: FiniteFloat = Field(default=0.0015, gt=0)
    thread_length_m: FiniteFloat = Field(default=0.024, gt=0)
    mass_kg: FiniteFloat = Field(default=0.03, gt=0)
    head_height_m: FiniteFloat = Field(default=0.0031, gt=0, description="visual head height")
    head_radius_m: FiniteFloat = Field(default=0.0101, gt=0)
    head_base_m: FiniteFloat = Field(default=0.0015, gt=0, description="solid head below recess")
    recess_depth_m: FiniteFloat = Field(default=0.003, gt=0)
    recess_width_m: FiniteFloat = Field(default=0.0025, gt=0)
    recess_half_m: FiniteFloat = Field(default=0.0071, gt=0)
    shank_radius_m: FiniteFloat = Field(default=0.0051, gt=0)
    shank_length_m: FiniteFloat = Field(default=0.032, gt=0)
    thread_damping_nms_rad: FiniteFloat = Field(default=0.002, ge=0)
    release_torque_nm: FiniteFloat = Field(
        default=0.5,
        gt=0,
        description="hold joint break torque: reached only when the turn stops at the thread end "
        "and the bit keeps turning (normal unscrewing transmits ~0.05 N m)",
    )
    release_force_n: FiniteFloat = Field(default=500.0, gt=0, description="hold joint break force")
    contact_offset_m: FiniteFloat = Field(default=0.0008, gt=0)
    torsional_patch_radius_m: FiniteFloat = Field(
        default=0.003, ge=0, description="torsional friction about the contact normal"
    )
    angular_damping: FiniteFloat = Field(default=0.2, ge=0)
    material: MaterialConfig = Field(
        default_factory=lambda: MaterialConfig(static_friction=0.3, dynamic_friction=0.25)
    )

    @property
    def head_top_m(self) -> float:
        """Proxy head top above the head underside."""
        return self.head_base_m + self.recess_depth_m


class PalletConfig(_Strict):
    """Static pallet receiving the cover; position relative to ``/World/scene``."""

    position_m: Vec3 = (0.0, -1.75, 0.06)
    size_m: Vec3 = (1.5, 1.0, 0.12)
    color: Vec3 = (0.22, 0.16, 0.12)
    material: MaterialConfig = Field(
        default_factory=lambda: MaterialConfig(static_friction=0.55, dynamic_friction=0.4)
    )


class LightsConfig(_Strict):
    """Workcell lights of the source project (dome + key + fill); ``enabled=false`` skips them."""

    enabled: bool = True
    dome_intensity: FiniteFloat = 650.0
    key_intensity: FiniteFloat = 2800.0
    fill_intensity: FiniteFloat = 1100.0


class BatteryWorkcellSceneConfig(BaseConfig):
    """Workcell asset, pose and physics proxies."""

    backend: str = "battery_workcell"
    usd_path: str = "assets/BatteryDisassembly/Scene/Battery/Battery.usd"
    prim_path: str = "/World/scene"
    position_m: Vec3 = (0.0, 0.0, 0.0)
    yaw_deg: FiniteFloat = 0.0
    battery_yaw_deg: FiniteFloat = Field(
        default=-90.0, description="absolute Z rotation of Pure_Battery/Battery (source configs)"
    )
    battery_prim: str = "Pure_Battery/Battery"
    cover_prim: str = "Pure_Battery/Battery/BatteryCover"
    cover_mesh_prim: str = "Pure_Battery/Battery/BatteryCover/Cover"
    screw_prims: list[str] = Field(
        default_factory=lambda: [
            f"Pure_Battery/Battery/BatteryCover/screw_0{i}" for i in range(1, 5)
        ],
        min_length=1,
    )
    basket_prim: str = "Basket/Basket1"
    workstation_prim: str = "Workstation/Workstation"
    case_box_m: tuple[Vec3, Vec3] = ((-0.425, -0.629, 0.7504), (0.425, 0.629, 0.8147))
    basket_wall_m: FiniteFloat = Field(default=0.008, gt=0)
    basket_floor_m: FiniteFloat = Field(default=0.015, gt=0)
    cover: CoverProxyConfig = Field(default_factory=CoverProxyConfig)
    screw: ScrewConfig = Field(default_factory=ScrewConfig)
    pallet: PalletConfig = Field(default_factory=PalletConfig)
    lights: LightsConfig = Field(default_factory=LightsConfig)
    proxy_contact_offset_m: FiniteFloat = Field(default=0.002, gt=0)
    show_proxies: bool = Field(default=False, description="render collision proxies (debug)")
    cameras: list[CameraConfig] = Field(default_factory=list)


# ================================================================================================
# 2. Scene
# ================================================================================================


def _matrix(prim: Any) -> np.ndarray:
    """World matrix of a prim in column-vector convention."""
    from pxr import UsdGeom

    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=float).T


class BatteryWorkcellScene(BaseScene, LoggerMixin):
    """Battery pack workcell with physically threaded screws and a liftable cover."""

    config_cls = BatteryWorkcellSceneConfig

    def __init__(self, config: BatteryWorkcellSceneConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config: BatteryWorkcellSceneConfig = config or self.config_cls.from_settings("scene")
        self.cameras = validate_camera_list(list(self.config.cameras))
        self.screw_paths: list[str] = []
        self.thread_paths: list[str] = []
        self.hold_joints: list[str] = []
        self.screw_frames: list[np.ndarray] = []  # world 4x4 of each screw frame at start
        self.layout: dict[str, Any] = {}
        self._views: dict[str, Any] = {}
        self.engaged = [True] * len(self.config.screw_prims)
        self.disengaged_step: list[int | None] = [None] * len(self.config.screw_prims)

    # --- paths -------------------------------------------------------------------------------
    def path(self, relative: str) -> str:
        return f"{self.config.prim_path}/{relative}"

    @property
    def cover_path(self) -> str:
        return self.path(self.config.cover_prim)

    @property
    def case_proxy_path(self) -> str:
        return self.path(f"{self.config.battery_prim}/rsr_case_proxy")

    # --- lifecycle -----------------------------------------------------------------------------
    def init_scene(self, sim_world: Any) -> None:
        """Reference the workcell, author proxies, rotate the pack, create screws and threads."""
        from pxr import Gf, UsdGeom

        if sim_world.physics_engine != "physx":
            raise ValueError("battery_workcell requires sim_world.physics_engine='physx'")
        self.sim_world = sim_world
        stage = sim_world.stage
        cfg = self.config
        root = UsdGeom.Xform.Define(stage, cfg.prim_path)
        if not root.GetPrim().GetReferences().AddReference(resolve_usd_path(cfg.usd_path)):
            raise RuntimeError(f"Workcell reference failed: {cfg.usd_path}")
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        q = np.r_[
            math.cos(math.radians(cfg.yaw_deg) / 2), 0, 0, math.sin(math.radians(cfg.yaw_deg) / 2)
        ]
        xform.AddTranslateOp().Set(Gf.Vec3d(*cfg.position_m))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
        required = [
            cfg.battery_prim, cfg.cover_prim, cfg.cover_mesh_prim, cfg.basket_prim,
            cfg.workstation_prim, *cfg.screw_prims,
        ]  # fmt: skip
        missing = [p for p in required if not stage.GetPrimAtPath(self.path(p)).IsValid()]
        if missing:
            raise RuntimeError(f"Workcell USD lacks prims: {missing}")
        self._materials()
        self._disable_asset_colliders()
        self._author_cover()
        self._author_case()
        self._author_basket()
        self._rotate_battery()
        self._author_screws()
        self._author_pallet()
        if cfg.lights.enabled:
            self._author_lights()
        self.sensors.update(create_cameras(stage, self.cameras))
        self.log.info(
            "电池工位已装配", screws=len(self.screw_paths), proxies=self.layout["proxy_count"]
        )

    def arrange(self) -> None:
        """Record the world layout the planner works with."""
        stage = self.sim_world.stage
        cfg = self.config
        scene = _matrix(stage.GetPrimAtPath(cfg.prim_path))
        heads = []
        for frame in self.screw_frames:
            heads.append(frame[:3, 3] + frame[:3, 2] * cfg.screw.head_top_m)
        basket = self.layout["basket_asset"]
        lo = scene[:3, :3] @ np.asarray(basket[0]) + scene[:3, 3]
        hi = scene[:3, :3] @ np.asarray(basket[1]) + scene[:3, 3]
        wall, floor = cfg.basket_wall_m, cfg.basket_floor_m
        pallet = self.layout["pallet_world"]
        self.layout.update(
            scene_matrix=scene,
            screw_heads=np.asarray(heads),
            basket_inner_lo=np.minimum(lo, hi) + np.array([wall, wall, floor]),
            basket_inner_hi=np.maximum(lo, hi) - np.array([wall, wall, 0.0]),
            pallet_top_z=float(pallet[2] + 0.5 * cfg.pallet.size_m[2]),
        )
        self.poses["scene"] = (scene[:3, 3].copy(), np.r_[1.0, 0, 0, 0])
        self.log.info(
            "工位布局",
            screw_heads=np.round(heads, 4).tolist(),
            cover_com=np.round(self.layout["cover_com_world"], 4).tolist(),
            basket=[np.round(self.layout["basket_inner_lo"], 3).tolist(),
                    np.round(self.layout["basket_inner_hi"], 3).tolist()],
            pallet_top=round(self.layout["pallet_top_z"], 4),
        )  # fmt: skip

    def adjust(self) -> None:
        """Layout is fixed by configuration."""

    # --- authoring helpers -----------------------------------------------------------------------
    def _materials(self) -> None:
        from pxr import UsdPhysics, UsdShade

        stage = self.sim_world.stage
        self._mat = {}
        for name, spec in {
            "cover": self.config.cover.material,
            "screw": self.config.screw.material,
            "pallet": self.config.pallet.material,
            "case": MaterialConfig(static_friction=0.6, dynamic_friction=0.5),
        }.items():
            material = UsdShade.Material.Define(stage, self.path(f"rsr_materials/{name}"))
            api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(spec.static_friction)
            api.CreateDynamicFrictionAttr(spec.dynamic_friction)
            api.CreateRestitutionAttr(spec.restitution)
            self._mat[name] = material

    def _disable_asset_colliders(self) -> None:
        """Pack / cover / basket colliders of the asset are replaced by proxies."""
        from pxr import Usd, UsdPhysics

        stage = self.sim_world.stage
        disabled = []
        for relative in ("Pure_Battery", "Basket"):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(self.path(relative))):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                    disabled.append(str(prim.GetPath()))
        basket = stage.GetPrimAtPath(self.path(self.config.basket_prim))
        if basket.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(basket).CreateRigidBodyEnabledAttr(False)
        self.layout["disabled_asset_colliders"] = disabled

    def _box(self, parent: str, box: Box, material: str, *, asset_frame: bool = True) -> str:
        """Box collider under ``parent``; dimensions in the asset frame (or parent frame)."""
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        stage = self.sim_world.stage
        parent_prim = stage.GetPrimAtPath(parent)
        if asset_frame:
            world = _matrix(stage.GetPrimAtPath(self.config.prim_path))
            local = np.linalg.inv(_matrix(parent_prim)) @ world
        else:
            local = np.eye(4)
        matrix = local @ np.diag([*box.size, 1.0])
        matrix[:3, 3] = (local @ np.r_[box.center, 1.0])[:3]
        path = f"{parent}/rsr_proxy_{box.name}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        UsdGeom.Xformable(cube).AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Matrix4d(matrix.T.tolist())
        )
        if not self.config.show_proxies:
            UsdGeom.Imageable(cube).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physx = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        physx.CreateContactOffsetAttr(self.config.proxy_contact_offset_m)
        physx.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            self._mat[material], UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        self.layout["proxy_count"] = self.layout.get("proxy_count", 0) + 1
        return path

    def _asset_screw_xy(self) -> list[tuple[float, float]]:
        """Screw axes in the asset frame from the original screw meshes (before rotation)."""
        stage = self.sim_world.stage
        inv_scene = np.linalg.inv(_matrix(stage.GetPrimAtPath(self.config.prim_path)))
        out = []
        for relative in self.config.screw_prims:
            lo, hi = prim_aabb(stage, self.path(relative), gprims_only=True)
            center = inv_scene @ np.r_[0.5 * (lo + hi), 1.0]
            out.append((float(center[0]), float(center[1])))
        return out

    def _author_cover(self) -> None:
        from pxr import PhysxSchema, UsdPhysics

        stage = self.sim_world.stage
        cfg = self.config.cover
        prim = stage.GetPrimAtPath(self.cover_path)
        boxes = cover_boxes(cfg, self._asset_screw_xy())
        for box in boxes:
            self._box(self.cover_path, box, "cover")
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cfg.mass_kg)
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body.CreateSolverPositionIterationCountAttr(32)
        body.CreateSolverVelocityIterationCountAttr(4)
        body.CreateMaxDepenetrationVelocityAttr(0.5)
        body.CreateSleepThresholdAttr(0.0)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        self.layout["cover_boxes_asset"] = boxes
        self.layout["cover_com_asset"] = boxes_centroid(boxes)
        self.layout["cover_world_before"] = _matrix(prim)

    def _author_case(self) -> None:
        lo, hi = self.config.case_box_m
        parent = self.path(self.config.battery_prim)
        path = self._box(parent, Box("case", lo, hi), "case")
        # the proxy is the only child we add there; keep its path stable for filtering
        self.layout["case_proxy"] = path

    def _author_basket(self) -> None:
        stage = self.sim_world.stage
        lo, hi = prim_aabb(stage, self.path(self.config.basket_prim), gprims_only=True)
        inv_scene = np.linalg.inv(_matrix(stage.GetPrimAtPath(self.config.prim_path)))
        lo_a, hi_a = (inv_scene @ np.r_[lo, 1.0])[:3], (inv_scene @ np.r_[hi, 1.0])[:3]
        parent = self.path("Basket")
        for box in basket_boxes(lo_a, hi_a, self.config.basket_wall_m, self.config.basket_floor_m):
            self._box(parent, box, "case")
        self.layout["basket_asset"] = (lo_a, hi_a)

    def _rotate_battery(self) -> None:
        from pxr import Gf, UsdGeom

        prim = self.sim_world.stage.GetPrimAtPath(self.path(self.config.battery_prim))
        xform = UsdGeom.Xformable(prim)
        op = next(
            (
                o
                for o in xform.GetOrderedXformOps()
                if o.GetOpType() == UsdGeom.XformOp.TypeRotateZYX
            ),
            None,
        )
        if op is None:
            op = xform.AddRotateZYXOp(UsdGeom.XformOp.PrecisionDouble)
        value = op.Get() or Gf.Vec3d(0, 0, 0)
        op.Set(type(value)(value[0], value[1], self.config.battery_yaw_deg))
        stage = self.sim_world.stage
        before = self.layout["cover_world_before"]
        after = _matrix(stage.GetPrimAtPath(self.cover_path))
        scene = _matrix(stage.GetPrimAtPath(self.config.prim_path))
        com_before = scene @ np.r_[self.layout["cover_com_asset"], 1.0]
        self.layout["cover_world_start"] = after
        self.layout["cover_com_world"] = (after @ np.linalg.inv(before) @ com_before)[:3]
        # proxy boxes of the cover in its body frame (for grasp planning and checks)
        to_body = np.linalg.inv(before) @ scene
        self.layout["cover_boxes_body"] = [
            ((to_body @ np.r_[b.lo, 1.0])[:3], (to_body @ np.r_[b.hi, 1.0])[:3])
            for b in self.layout["cover_boxes_asset"]
        ]

    def _author_screws(self) -> None:
        """Free screw bodies plus thread articulations at the rotated screw locations."""
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        stage = self.sim_world.stage
        cfg = self.config.screw
        asset = resolve_usd_path(self.config.usd_path)
        UsdGeom.Scope.Define(stage, self.path("rsr_screws"))
        UsdGeom.Scope.Define(stage, self.path("rsr_threads"))
        for i, relative in enumerate(self.config.screw_prims):
            source = stage.GetPrimAtPath(self.path(relative))
            lo, hi = prim_aabb(stage, str(source.GetPath()), gprims_only=True)
            origin = np.array(
                [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), hi[2] - cfg.head_height_m]
            )
            frame = np.eye(4)
            frame[:3, 3] = origin
            self.screw_frames.append(frame)
            # bindings must be read before the source is deactivated
            bindings = {}
            for prim in Usd.PrimRange(source):
                if prim.IsA(UsdGeom.Gprim):
                    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                    if material:
                        bindings[str(prim.GetPath())[len(str(source.GetPath())) :]] = material
            source_world = _matrix(source)
            body_path = self.path(f"rsr_screws/S{i + 1}")
            body = UsdGeom.Xform.Define(stage, body_path)
            body.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*origin))
            body.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0))
            UsdGeom.Xformable(body).SetResetXformStack(True)
            visual = stage.DefinePrim(f"{body_path}/visual")
            source_layer_path = Sdf.Path("/root").AppendPath(Sdf.Path(relative))
            visual.GetReferences().AddReference(asset, source_layer_path)
            vx = UsdGeom.Xformable(visual)
            vx.ClearXformOpOrder()
            local = np.linalg.inv(frame) @ source_world
            vx.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d(local.T.tolist()))
            for suffix, material in bindings.items():
                target = stage.GetPrimAtPath(f"{body_path}/visual{suffix}")
                if target.IsValid():
                    UsdShade.MaterialBindingAPI.Apply(target).Bind(
                        material, UsdShade.Tokens.strongerThanDescendants
                    )
            for prim in Usd.PrimRange(visual):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            source.SetActive(False)
            self._screw_colliders(body_path)
            prim = body.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cfg.mass_kg)
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateSolverPositionIterationCountAttr(16)
            rb.CreateSolverVelocityIterationCountAttr(2)
            rb.CreateMaxDepenetrationVelocityAttr(0.3)
            rb.CreateSleepThresholdAttr(0.0)
            rb.CreateEnableCCDAttr(True)
            rb.CreateAngularDampingAttr(cfg.angular_damping)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(
                [Sdf.Path(self.layout["case_proxy"])]
            )
            self.screw_paths.append(body_path)
            self._author_thread(i, origin, body_path)

    def _screw_colliders(self, body_path: str) -> None:
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        stage = self.sim_world.stage
        cfg = self.config.screw

        def finish(prim: Any) -> None:
            if not self.config.show_proxies:
                UsdGeom.Imageable(prim).MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(prim)
            physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx.CreateContactOffsetAttr(cfg.contact_offset_m)
            physx.CreateRestOffsetAttr(0.0)
            physx.CreateTorsionalPatchRadiusAttr(cfg.torsional_patch_radius_m)
            physx.CreateMinTorsionalPatchRadiusAttr(0.5 * cfg.torsional_patch_radius_m)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                self._mat["screw"], UsdShade.Tokens.weakerThanDescendants, "physics"
            )

        for name, radius, z0, z1 in (
            ("head", cfg.head_radius_m, 0.0, cfg.head_base_m),
            ("shank", cfg.shank_radius_m, -cfg.shank_length_m, 0.0),
        ):
            cylinder = UsdGeom.Cylinder.Define(stage, f"{body_path}/rsr_proxy_{name}")
            cylinder.CreateRadiusAttr(radius)
            cylinder.CreateHeightAttr(z1 - z0)
            cylinder.CreateAxisAttr("Z")
            cylinder.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5 * (z0 + z1)))
            finish(cylinder.GetPrim())
        for box in screw_head_boxes(cfg):
            cube = UsdGeom.Cube.Define(stage, f"{body_path}/rsr_proxy_{box.name}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*box.center.tolist()))
            cube.AddScaleOp().Set(Gf.Vec3f(*box.size.tolist()))
            finish(cube.GetPrim())

    def _author_thread(self, index: int, origin: np.ndarray, screw_path: str) -> None:
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        stage = self.sim_world.stage
        cfg = self.config.screw
        root = self.path(f"rsr_threads/T{index + 1}")
        UsdGeom.Xform.Define(stage, root)

        def body(name: str) -> Any:
            xf = UsdGeom.Xform.Define(stage, f"{root}/{name}")
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*origin))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0))
            UsdGeom.Xformable(xf).SetResetXformStack(True)
            prim = xf.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(0.005)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-7, 1e-7, 1e-7))
            return prim

        anchor, slider, carrier = body("anchor"), body("slider"), body("carrier")
        UsdPhysics.ArticulationRootAPI.Apply(anchor)
        art = PhysxSchema.PhysxArticulationAPI.Apply(anchor)
        art.CreateEnabledSelfCollisionsAttr(False)
        art.CreateSolverPositionIterationCountAttr(16)
        art.CreateSolverVelocityIterationCountAttr(2)
        art.CreateSleepThresholdAttr(0.0)

        def joint(kind: Any, name: str, body0: Any, body1: Any, pos0: Any) -> Any:
            j = kind.Define(stage, f"{root}/{name}")
            if body0 is not None:
                j.CreateBody0Rel().SetTargets([body0.GetPath()])
            j.CreateBody1Rel().SetTargets([body1.GetPath()])
            j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0))
            if kind is not UsdPhysics.FixedJoint:
                j.CreateAxisAttr("Z")
            return j

        joint(UsdPhysics.FixedJoint, "root_joint", None, anchor, origin.tolist())
        slide = joint(UsdPhysics.PrismaticJoint, "slide", anchor, slider, (0, 0, 0))
        slide.CreateLowerLimitAttr(-0.001)
        slide.CreateUpperLimitAttr(cfg.thread_length_m + 0.0005)
        turn = joint(UsdPhysics.RevoluteJoint, "turn", slider, carrier, (0, 0, 0))
        # the thread ends after thread_length / pitch turns (counter-clockwise = +turn loosens);
        # a hard angular stop lets the bit torque reach the hold joint, which then breaks
        turn.CreateLowerLimitAttr(-10.0)
        turn.CreateUpperLimitAttr(360.0 * cfg.thread_length_m / cfg.pitch_m)
        drive = UsdPhysics.DriveAPI.Apply(turn.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(cfg.thread_damping_nms_rad * math.pi / 180.0)
        drive.CreateMaxForceAttr(1.0e6)
        mimic = PhysxSchema.PhysxMimicJointAPI.Apply(slide.GetPrim(), UsdPhysics.Tokens.rotZ)
        mimic.CreateReferenceJointRel().SetTargets([turn.GetPath()])
        mimic.CreateReferenceJointAxisAttr(UsdPhysics.Tokens.rotZ)
        # slide + gearing * turn(deg) = 0  →  slide = pitch * turn / 360
        mimic.CreateGearingAttr(-cfg.pitch_m / 360.0)
        mimic.CreateOffsetAttr(0.0)
        hold = joint(
            UsdPhysics.FixedJoint, "hold", carrier, stage.GetPrimAtPath(screw_path), (0, 0, 0)
        )
        hold.CreateExcludeFromArticulationAttr(True)
        # the thread runs out at the slide limit; the solver breaks the hold there (no USD edits)
        hold.CreateBreakTorqueAttr(cfg.release_torque_nm)
        hold.CreateBreakForceAttr(cfg.release_force_n)
        self.thread_paths.append(f"{root}/anchor")
        self.hold_joints.append(f"{root}/hold")

    def _author_pallet(self) -> None:
        from pxr import Gf, UsdGeom

        stage = self.sim_world.stage
        cfg = self.config.pallet
        scene = _matrix(stage.GetPrimAtPath(self.config.prim_path))
        world = (scene @ np.r_[cfg.position_m, 1.0])[:3]
        path = self.path("rsr_pallet")
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*cfg.position_m))
        box = Box("body", tuple(-0.5 * np.asarray(cfg.size_m)), tuple(0.5 * np.asarray(cfg.size_m)))
        body = self._box(path, box, "pallet", asset_frame=False)
        prim = stage.GetPrimAtPath(body)
        UsdGeom.Imageable(prim).MakeVisible()
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        self.layout["pallet_world"] = world
        self.layout["pallet_path"] = body

    def _author_lights(self) -> None:
        from pxr import Gf, UsdLux

        stage = self.sim_world.stage
        cfg = self.config.lights
        dome = UsdLux.DomeLight.Define(stage, "/World/rsr_lights/Environment")
        dome.CreateIntensityAttr(cfg.dome_intensity)
        dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
        for name, intensity, angle, color, rotation in (
            ("Key", cfg.key_intensity, 2.5, (1.0, 0.92, 0.82), (45.0, -25.0, 35.0)),
            ("Fill", cfg.fill_intensity, 4.0, (0.72, 0.82, 1.0), (60.0, 30.0, -120.0)),
        ):
            light = UsdLux.DistantLight.Define(stage, f"/World/rsr_lights/{name}")
            light.CreateIntensityAttr(intensity)
            light.CreateAngleAttr(angle)
            light.CreateColorAttr(Gf.Vec3f(*color))
            light.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))

    # --- runtime (after world reset + play) -------------------------------------------------------
    def initialize_runtime(self) -> None:
        """Tensor views of screws, cover and thread articulations."""
        from isaacsim.core.experimental.prims import Articulation, RigidPrim

        self._views = {
            "screws": RigidPrim(self.screw_paths),
            "cover": RigidPrim(self.cover_path),
            "threads": Articulation(self.thread_paths),
            "carriers": RigidPrim([p.replace("/anchor", "/carrier") for p in self.thread_paths]),
        }
        names = list(self._views["threads"].dof_names)
        self._slide, self._turn = names.index("slide"), names.index("turn")
        self.engaged = [True] * len(self.screw_paths)
        self.disengaged_step = [None] * len(self.screw_paths)
        self.state["thread_dofs"] = names

    def thread_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Axial slide (m) and turn angle (rad) of each thread."""
        q = _numpy(self._views["threads"].get_dof_positions())
        return q[:, self._slide].copy(), q[:, self._turn].copy()

    def screw_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """World positions (N,3) and wxyz orientations (N,4) of the screw frames."""
        p, q = self._views["screws"].get_world_poses()
        return _numpy(p), _numpy(q)

    def screw_velocities(self) -> np.ndarray:
        lin, _ = self._views["screws"].get_velocities()
        return _numpy(lin)

    def cover_pose(self) -> np.ndarray:
        """Cover body ``[x, y, z, qw, qx, qy, qz]``."""
        p, q = self._views["cover"].get_world_poses()
        return np.r_[_numpy(p)[0], _numpy(q)[0]]

    def apply_screw_forces(self, forces: np.ndarray) -> None:
        """External forces (N,3) at the screw body origins (world frame)."""
        if np.any(forces):
            self._views["screws"].apply_forces(np.asarray(forces, dtype=np.float32))

    def update_threads(self, step: int) -> list[int]:
        """Track which screws came out: slide at the thread end and screw separated from carrier.

        PhysX breaks the hold joint itself (break torque); this only reads states.
        """
        slide, _ = self.thread_states()
        positions, _ = self.screw_poses()
        carriers = self.carrier_positions()
        released = []
        for i, value in enumerate(slide):
            apart = float(np.linalg.norm(positions[i] - carriers[i]))
            if (
                self.engaged[i]
                and value >= self.config.screw.thread_length_m - 2e-4
                and apart > 5e-4
            ):
                self.engaged[i] = False
                self.disengaged_step[i] = step
                released.append(i)
                self.log.info("螺纹脱开", screw=i + 1, slide_m=round(float(value), 5), step=step)
        return released

    def carrier_positions(self) -> np.ndarray:
        p, _ = self._views["carriers"].get_world_poses()
        return _numpy(p)

    def contact_owner(self, path: str) -> str | None:
        """Classify a contact actor of this scene."""
        if path.startswith(self.cover_path):
            return "cover"
        for i, screw in enumerate(self.screw_paths):
            if path.startswith(screw):
                return f"screw_{i + 1}"
        if path.startswith(self.layout.get("pallet_path", "~")):
            return "pallet"
        if path.startswith(self.path("Basket")):
            return "basket"
        if path.startswith(self.config.prim_path):
            return "workcell"
        return None


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=float)
    return array.reshape(1, -1) if array.ndim == 1 else array
