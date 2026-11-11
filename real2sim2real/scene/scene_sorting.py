"""Sorting station with the V7 bag and garment, retaining native station geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field, FiniteFloat, field_validator

from real2sim2real.scene.scene_package import PackageScene, PackageSceneConfig
from real2sim2real.utils.asset_utils import prim_aabb
from real2sim2real.utils.cameras import create_cameras


class SortingSceneConfig(PackageSceneConfig):
    """Station supports are measured from their USD visuals in world metres."""

    backend: str = "sorting_package"
    cloth_usd_path: str = "assets/PhysicSolver/Package/v7_sorting/parcel_two_mesh.usd"
    station_usd_path: str = (
        "assets/Scene/sorting_station/composed/scene_parcel_5cm_loose_edges.usdc"
    )
    infeed_path: str = "ConveyorBelt_A06_01/Belt"
    platform_path: str = "platform/Empty/Plane/Plane_002"
    outfeed_path: str = "ConveyorBelt_A06/Belt"
    target_xy: list[float] = Field(default_factory=lambda: [0.55, -1.0])
    garment_mass_kg: float = Field(default=0.2, gt=0, allow_inf_nan=False)
    bag_mass_kg: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    initial_clearance_m: float = Field(default=0.002, ge=0)
    support_depth_m: float = Field(default=0.04, gt=0)
    outfeed_target_xy: tuple[float, float] = (-0.55, -0.876)
    platform_target_xy: tuple[FiniteFloat, FiniteFloat] | None = None
    label_uv_bounds: tuple[float, float, float, float] = (0.20, 0.44, 0.66, 0.65)
    clear_legacy_parcels: bool = False
    infeed_travel_m: FiniteFloat = Field(default=0.0, ge=0)
    floor_z_m: FiniteFloat = 0.0
    rigid_usd_path: str = ""
    rigid_scale: FiniteFloat = Field(default=0.5, gt=0)
    rigid_mass_kg: FiniteFloat = Field(default=0.03125, gt=0)
    belt_markers: bool = False

    @field_validator("label_uv_bounds")
    @classmethod
    def _label_rectangle(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if not (0 <= value[0] < value[2] <= 1 and 0 <= value[1] < value[3] <= 1):
            raise ValueError("Label bounds must be a nonempty UV rectangle within [0, 1]")
        return value


class SortingScene(PackageScene):
    """Three measured support surfaces and two freely simulated parcel meshes."""

    config_cls = SortingSceneConfig

    def init_scene(self, sim_world: Any) -> None:
        """Reference the station, disable its historical robot and single-mesh parcel."""
        from pxr import Gf, UsdGeom, UsdLux

        self.sim_world = sim_world
        root = UsdGeom.Xform.Define(sim_world.stage, self.config.prim_path)
        root.GetPrim().GetReferences().AddReference(
            self._asset(self.config.station_usd_path), "/World"
        )
        for name in ("Pro_m6_696_dynamics_fixed", "ImportedBaggedClothing", "PhysicsScene"):
            prim = sim_world.stage.GetPrimAtPath(f"{self.config.prim_path}/{name}")
            if prim:
                prim.SetActive(False)
        if self.config.clear_legacy_parcels:
            for name in ("box01", "box02", "ImportedBaggedClothing02"):
                prim = sim_world.stage.GetPrimAtPath(f"{self.config.prim_path}/{name}")
                if prim:
                    prim.SetActive(False)
        light = UsdLux.DomeLight.Define(sim_world.stage, f"{self.config.prim_path}/sorting_light")
        light.CreateIntensityAttr(self.config.fill_light_intensity)
        key = UsdLux.DistantLight.Define(sim_world.stage, f"{self.config.prim_path}/sorting_key")
        key.CreateIntensityAttr(self.config.key_light_intensity)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35, -25, -30))

    def arrange(self) -> None:
        """Place the unchanged V7 asset on the measured infeed top."""
        self.supports = {}
        for name in ("infeed", "platform", "outfeed"):
            relative = getattr(self.config, f"{name}_path")
            path = f"{self.config.prim_path}/{relative}"
            bounds = prim_aabb(self.sim_world.stage, path, gprims_only=True)
            if bounds is None:
                raise ValueError(f"Missing sorting support geometry: {path}")
            low, high = (point.copy() for point in bounds)
            low[2] = high[2] - self.config.support_depth_m
            self.supports[name] = (low, high)
        self.table_bounds = self.supports["infeed"]
        self.table_top_z = float(self.table_bounds[1][2] + self.config.initial_clearance_m)
        self._arrange_cloth()
        self._cache_label_triangles()
        if self.config.rigid_usd_path:
            self._arrange_rigid()
        if self.config.belt_markers:
            self._create_belt_markers()
        self.sensors.update(create_cameras(self.sim_world.stage, self.config.cameras))
        self.refresh_poses()

    def _create_belt_markers(self) -> None:
        """Add thin painted seams to make the commanded belt surface motion visible."""
        from pxr import Gf, UsdGeom

        self.belt_marker_ops = {}
        for name, axis in (("infeed", 1), ("outfeed", 0)):
            low, high = self.supports[name]
            count = max(2, int(np.ceil((high[axis] - low[axis]) / 0.25)))
            ops = []
            for index in range(count):
                cube = UsdGeom.Cube.Define(
                    self.sim_world.stage, f"{self.config.prim_path}/belt_marks/{name}_{index}"
                )
                cube.CreateSizeAttr(1.0)
                cube.CreateDisplayColorAttr([Gf.Vec3f(0.07, 0.075, 0.08)])
                ops.append(cube.AddTranslateOp())
                scale = high - low
                scale[axis], scale[2] = 0.003, 0.0001
                cube.AddScaleOp().Set(Gf.Vec3f(*scale))
            self.belt_marker_ops[name] = (axis, low, high, ops)
        self.publish_belt_markers(np.zeros(2))

    def publish_belt_markers(self, offsets: np.ndarray) -> None:
        """Wrap painted belt seams, leaving all collision geometry unchanged."""
        from pxr import Gf

        for slot, (name, (axis, low, high, ops)) in enumerate(self.belt_marker_ops.items()):
            length = high[axis] - low[axis]
            direction = 1 if name == "infeed" else -1
            for index, op in enumerate(ops):
                position = (low + high) / 2
                position[2] = high[2] + 0.00015
                position[axis] = (
                    low[axis] + (length * index / len(ops) + direction * offsets[slot]) % length
                )
                op.Set(Gf.Vec3d(*position))

    def _arrange_rigid(self) -> None:
        """Keep the V10 carton mesh/materials and explicitly scale its SI geometry."""
        from pxr import Gf, Usd, UsdGeom

        source = Usd.Stage.Open(self._asset(self.config.rigid_usd_path))
        meshes = [UsdGeom.Mesh(p) for p in source.Traverse() if p.IsA(UsdGeom.Mesh)]
        if len(meshes) != 1 or UsdGeom.GetStageMetersPerUnit(source) != 1:
            raise ValueError("The rigid stream asset must contain one mesh in metres")
        mesh = meshes[0]
        points = np.asarray(mesh.GetPointsAttr().Get(), float)
        transform = np.asarray(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(0)).T
        points = points @ transform[:3, :3].T + transform[:3, 3]
        center = (points.min(0) + points.max(0)) / 2
        self.rigid_vertices = (points - center) * self.config.rigid_scale
        if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
            raise ValueError("The rigid stream asset must be triangulated")
        self.rigid_triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int).reshape(-1, 3)
        self.rigid_dimensions = np.ptp(self.rigid_vertices, axis=0)
        if self.rigid_dimensions[0] > 0.080:
            raise ValueError("Rigid parcel width exceeds the original Pro-M6 gripper opening")
        path = f"{self.config.prim_path}/stream_carton"
        root = UsdGeom.Xform.Define(self.sim_world.stage, path)
        self.rigid_visual_op = root.AddTransformOp()
        self.rigid_visual = UsdGeom.Imageable(root)
        asset = UsdGeom.Xform.Define(self.sim_world.stage, f"{path}/asset")
        asset.GetPrim().GetReferences().AddReference(
            self._asset(self.config.rigid_usd_path), "/World"
        )
        asset.AddTranslateOp().Set(Gf.Vec3d(*(-center * self.config.rigid_scale)))
        asset.AddScaleOp().Set(Gf.Vec3f(self.config.rigid_scale))
        self.rigid_visual.MakeInvisible()

    def publish_rigid_pose(self, pose: np.ndarray, *, visible: bool) -> None:
        """Display the solved carton pose, with visibility controlled only by spawn time."""
        from pxr import Gf

        matrix = Gf.Matrix4d(1)
        matrix.SetRotate(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6]))))
        matrix.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
        self.rigid_visual_op.Set(matrix)
        if visible:
            self.rigid_visual.MakeVisible()
        else:
            self.rigid_visual.MakeInvisible()

    def _arrange_cloth(self) -> None:
        """Apply declared masses without changing either surface or its relative mass weights."""
        target = self.config.target_xy.copy()
        try:
            travel = getattr(self.config, "infeed_travel_m", 0.0)
            self.config.target_xy = [target[0], target[1] - travel]
            super()._arrange_cloth()
        finally:
            self.config.target_xy = target
        if travel > 0:
            low, high = self.supports["infeed"]
            if not np.all(
                (self.cloth_vertices[:, :2] >= low[:2]) & (self.cloth_vertices[:, :2] <= high[:2])
            ):
                raise ValueError("Spawned parcel must fit entirely on the infeed belt")
        for part in self.package_parts:
            name = part["name"].lower()
            authored = float(part["masses"].sum())
            target = getattr(self.config, f"{name}_mass_kg")
            scale = target / authored if target is not None else 1.0
            setattr(self, f"authored_{name}_mass_kg", authored)
            setattr(self, f"{name}_mass_scale", scale)
            if target is not None:
                part["masses"] = part["masses"] * scale
                part["mesh"].GetPrim().GetAttribute("physics:masses").Set(part["masses"].tolist())

    def _cache_label_triangles(self) -> None:
        """Use the visible V7 photograph's label UV rectangle as an orientation marker."""
        from pxr import UsdGeom

        bag = next(part for part in self.package_parts if part["name"] == "Bag")
        mesh, triangles = bag["mesh"], bag["triangles"]
        primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        uv = np.asarray(primvar.ComputeFlattened())
        if primvar.GetInterpolation() == UsdGeom.Tokens.faceVarying:
            uv = uv.reshape(-1, 3, 2).mean(1)
        elif primvar.GetInterpolation() == UsdGeom.Tokens.vertex:
            uv = uv[triangles].mean(1)
        else:
            raise ValueError("V7 label requires vertex or face-varying UV coordinates")
        subset = UsdGeom.Subset.Get(self.sim_world.stage, f"{mesh.GetPath()}/FrontPhotograph")
        front = np.asarray(subset.GetIndicesAttr().Get(), dtype=int)
        u0, v0, u1, v1 = self.config.label_uv_bounds
        selected = front[np.all((uv[front] >= [u0, v0]) & (uv[front] <= [u1, v1]), axis=1)]
        if not len(selected):
            raise ValueError("Label UV rectangle does not contain any front photograph triangles")
        self.label_triangles = triangles[selected] + bag["start"]

    def label_up_cos(self, vertices: np.ndarray) -> float:
        """Read the printed label patch's area-weighted normal from actual solver particles."""
        tri = vertices[self.label_triangles]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(0)
        return float(normal[2] / max(np.linalg.norm(normal), 1e-12))

    def support_penetration(self, vertices: np.ndarray) -> float:
        """Measure depth inside the finite support solids, including their side/end faces."""
        maximum = 0.0
        for low, high in self.supports.values():
            inside = np.all((vertices >= low) & (vertices <= high), axis=1)
            if inside.any():
                depth = np.minimum(vertices[inside] - low, high - vertices[inside]).min(axis=1)
                maximum = max(maximum, float(depth.max()))
        return maximum
