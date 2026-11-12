"""物理求解器验证任务的场景：共用桌面、衣服、刚体方块、官方 Newton 衬衫与 V7 包裹。

- ``pickplace_cloth``：``PickPlaceClothScene`` 发布交给 Newton 模拟的自由衣服网格。
- ``pickplace_rigid``：``PickPlaceRigidScene`` 在相同桌面上添加 PhysX 刚体方块。
- ``newton_cloth_franka``：``OfficialClothScene`` 把官方 cloth_franka 布局整体平移到项目桌面。
- ``pickplace_package``：``PackageScene`` 加载 V7 袋膜 + 衣物双网格，保留质量、UV 与材质；
  分拣任务的 ``SortingScene`` 继承它。

世界初始化与物理步进由 ``IsaacsimWorld`` 管理。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import get_logger
from real2sim2real.scene.base import BaseScene
from real2sim2real.simworld.base import BaseSimWorld
from real2sim2real.utils.asset_utils import prim_aabb, read_prim_world_pose, resolve_usd_path
from real2sim2real.utils.cameras import CameraConfig, create_cameras, validate_camera_list

# ================================================================================================
# PhysicSolver 案例的共用桌面布局、衣服场景和刚体场景。
#
# pickplace_cloth 后端使用 PickPlaceClothScene，发布交给 Newton 模拟的自由衣服网格；
# pickplace_rigid 后端使用 PickPlaceRigidScene，在相同桌面上添加 PhysX 刚体方块。
# 两者复用桌子、灯光和摆放逻辑；世界初始化与物理步进仍由 IsaacsimWorld 管理。
# ================================================================================================


class PickPlaceClothSceneConfig(BaseConfig):
    """Assets and metre-scale layout for the cloth task."""

    object_kind: Literal["cloth"] = "cloth"
    table_usd_path: str = "assets/PhysicSolver/Table/Aligned.usd"
    cloth_usd_path: str = "assets/PhysicSolver/Cloth/unisex_shirt.usd"
    prim_path: str = "/World/scene"
    table_scale: list[float] = Field(default_factory=lambda: [2.0, 1.2, 1.0])
    table_xy: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    target_xy: list[float] = Field(default_factory=lambda: [0.14, 0.0])
    robot_base_xy: list[float] = Field(default_factory=lambda: [-0.38, 0.0])
    cloth_initial_clearance: float = Field(default=0.015, ge=0.005)
    cloth_thickness_scale: float = Field(default=0.03, gt=0.0, le=1.0)
    grasp_vertex: int = Field(
        default=1922, ge=0, description="Validated side-seam vertex of the bundled shirt."
    )
    friction: float = Field(default=1.2, ge=0.0)
    fill_light_intensity: float = Field(default=650.0, ge=0)
    key_light_intensity: float = Field(default=2200.0, ge=0)
    cameras: list[CameraConfig] = Field(default_factory=list)

    @field_validator("cameras")
    @classmethod
    def _validate_cameras(cls, value: list[CameraConfig]) -> list[CameraConfig]:
        return validate_camera_list(value)

    @field_validator("table_scale")
    @classmethod
    def _positive_scale(cls, value: list[float]) -> list[float]:
        if len(value) != 3 or any(component <= 0 for component in value):
            raise ValueError("table_scale must contain three positive scale factors")
        return value

    @field_validator("table_xy", "target_xy", "robot_base_xy")
    @classmethod
    def _xy(cls, value: list[float]) -> list[float]:
        if len(value) != 2:
            raise ValueError("XY positions must contain two coordinates in meters")
        return value


class PickPlaceClothScene(BaseScene):
    """Place a fixed table and a Newton cloth mesh.

    Newton vertices and table bounds are always world-space meters. Cloth points
    are authored in an identity-transform mesh so the solver can publish them
    without baking a second unit conversion into the visual geometry.
    """

    config_cls = PickPlaceClothSceneConfig

    def __init__(self, config: PickPlaceClothSceneConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        del kwargs
        self.config = config or self.config_cls.from_settings("scene")
        self.table_top_z = 0.0
        self.table_bounds = (np.zeros(3), np.zeros(3))
        self.cloth_vertices = np.empty((0, 3), dtype=np.float64)
        self.triangle_indices = np.empty(0, dtype=np.int32)
        self.cloth_mesh: Any | None = None
        self.cloth_grasp_vertex: int | None = None
        self._target_initial_position = np.zeros(3)

    @property
    def table_prim_path(self) -> str:
        """Return the authored table root."""
        return f"{self.config.prim_path.rstrip('/')}/table"

    @property
    def target_prim_path(self) -> str:
        """Return the cloth root."""
        return f"{self.config.prim_path.rstrip('/')}/{self.config.object_kind}"

    def init_scene(self, sim_world: BaseSimWorld) -> None:
        """Reference the copied table and the requested grasp asset."""
        from pxr import Gf, UsdGeom, UsdLux

        self.sim_world = sim_world
        if sim_world.stage is None:
            raise RuntimeError("Cloth scene requires an initialized USD stage")
        stage = sim_world.stage
        dome = UsdLux.DomeLight.Define(stage, "/World/scene/lighting/fill")
        dome.CreateIntensityAttr(self.config.fill_light_intensity)
        key = UsdLux.DistantLight.Define(stage, "/World/scene/lighting/key")
        key.CreateIntensityAttr(self.config.key_light_intensity)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35, -25, -30))
        UsdGeom.Xform.Define(stage, self.config.prim_path)
        table = UsdGeom.Xform.Define(stage, self.table_prim_path)
        table.GetPrim().GetReferences().AddReference(self._asset(self.config.table_usd_path))
        self.sensors.update(create_cameras(stage, self.config.cameras))

    def arrange(self) -> None:
        """Place the table on the floor and the target on its measured top."""
        from pxr import Gf, UsdGeom

        stage = self.sim_world.stage
        table = UsdGeom.Xformable(stage.GetPrimAtPath(self.table_prim_path))
        table.ClearXformOpOrder()
        translate = table.AddTranslateOp(opSuffix="demo")
        table.AddScaleOp(opSuffix="demo").Set(Gf.Vec3f(*self.config.table_scale))
        bounds = prim_aabb(stage, self.table_prim_path, gprims_only=True)
        if bounds is None:
            raise RuntimeError("Could not measure the copied table asset")
        translate.Set(Gf.Vec3d(*self.config.table_xy, -float(bounds[0][2])))
        self.table_bounds = prim_aabb(stage, self.table_prim_path, gprims_only=True)
        self.table_top_z = float(self.table_bounds[1][2])
        self._configure_table_physics()
        self._arrange_target()
        self.refresh_poses()
        self._target_initial_position = self.target_position.copy()
        get_logger(type(self).__name__).info(
            "Cloth table and target ready",
            object_kind=self.config.object_kind,
            table_top_z=self.table_top_z,
            target_position=self.target_position.tolist(),
            cloth_vertices=len(self.cloth_vertices),
        )

    def adjust(self) -> None:
        """The explicit demo layout needs no interactive adjustment."""

    def _arrange_target(self) -> None:
        self._arrange_cloth()

    def _configure_table_physics(self) -> None:
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = self.sim_world.stage
        root = stage.GetPrimAtPath(self.table_prim_path)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
        self._bind_friction(root)

    def _arrange_cloth(self) -> None:
        from pxr import Gf, Usd, UsdGeom, Vt

        source = Usd.Stage.Open(self._asset(self.config.cloth_usd_path))
        if source is None:
            raise RuntimeError("Could not open the copied shirt USD")
        mesh_prim = source.GetPrimAtPath("/root/shirt")
        if not mesh_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError("The Newton shirt asset must contain /root/shirt")
        source_mesh = UsdGeom.Mesh(mesh_prim)
        points = self._initial_cloth_points(source, mesh_prim)
        counts = np.asarray(source_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        indices = np.asarray(source_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if not np.all(counts == 3):
            raise ValueError("The demo Newton shirt asset must contain triangular faces")
        self.cloth_vertices = points
        self.triangle_indices = indices
        self.cloth_mesh = UsdGeom.Mesh.Define(self.sim_world.stage, self.target_prim_path)
        self.cloth_mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts.tolist()))
        self.cloth_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices.tolist()))
        self.cloth_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self.cloth_mesh.CreateDoubleSidedAttr(True)
        self.cloth_mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.12, 0.42, 0.75)]))
        if self.config.grasp_vertex >= len(points):
            raise ValueError("grasp_vertex exceeds the copied shirt vertex count")
        self.cloth_grasp_vertex = self.config.grasp_vertex
        self.publish_cloth_vertices(points)

    def _initial_cloth_points(self, source: Any, mesh_prim: Any) -> np.ndarray:
        from pxr import Gf, UsdGeom

        points = np.asarray(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(), dtype=np.float64)
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim)
        points = np.asarray([transform.Transform(Gf.Vec3d(*p)) for p in points])
        points *= UsdGeom.GetStageMetersPerUnit(source)
        points -= 0.5 * (points.min(axis=0) + points.max(axis=0))
        points[:, 2] *= self.config.cloth_thickness_scale
        points[:, :2] += self.config.target_xy
        points[:, 2] += self.table_top_z + self.config.cloth_initial_clearance
        return points

    def _bind_friction(self, prim: Any) -> None:
        from pxr import UsdPhysics, UsdShade

        material = UsdShade.Material.Define(
            self.sim_world.stage, f"{self.config.prim_path}/friction_material"
        )
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(self.config.friction)
        api.CreateDynamicFrictionAttr(self.config.friction)
        api.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")

    def publish_cloth_vertices(self, world_vertices: np.ndarray) -> None:
        """Publish Newton world positions (meters) to the render mesh and cache."""
        from pxr import Gf, Vt

        points = np.asarray(world_vertices, dtype=np.float64)
        if points.shape != self.cloth_vertices.shape or not np.isfinite(points).all():
            raise ValueError("Newton cloth positions must have the original finite (N, 3) shape")
        self.cloth_vertices = points.copy()
        self.cloth_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        self.cloth_mesh.CreateExtentAttr(
            Vt.Vec3fArray([Gf.Vec3f(*points.min(axis=0)), Gf.Vec3f(*points.max(axis=0))])
        )
        pose = self._runtime_target_pose()
        if pose is not None:
            self._cache_pose(self.target_prim_path, pose)

    @property
    def target_position(self) -> np.ndarray:
        """Return the selected cloth grasp vertex in metres."""
        pose = self._runtime_target_pose()
        if pose is not None:
            return pose[0]
        if self.sim_world is not None:
            bounds = prim_aabb(self.sim_world.stage, self.target_prim_path, gprims_only=True)
            if bounds is not None:
                return 0.5 * (bounds[0] + bounds[1])
        return np.zeros(3)

    @property
    def target_lift(self) -> float:
        """Return target height gain since arrangement in meters."""
        return float(self.target_position[2] - self._target_initial_position[2])

    @property
    def robot_base_position(self) -> np.ndarray:
        """Return the robot mounting position on the measured table top."""
        return np.array([*self.config.robot_base_xy, self.table_top_z])

    def refresh_poses(self) -> None:
        """Refresh live physical poses while preserving the initial arrangement.

        The cloth pose follows its selected grasp vertex, matching
        ``target_position``. Its quaternion is identity: a deforming surface
        has no single rigid orientation.
        """
        table_pose = read_prim_world_pose(self.sim_world.stage, self.table_prim_path)
        if table_pose is not None:
            self._cache_pose(self.table_prim_path, table_pose)
        target_pose = self._runtime_target_pose()
        if target_pose is None:
            target_pose = read_prim_world_pose(self.sim_world.stage, self.target_prim_path)
            if target_pose is not None:
                target_pose = (self.target_position, target_pose[1])
        if target_pose is not None:
            self._cache_pose(self.target_prim_path, target_pose)

    def _runtime_target_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.config.object_kind == "cloth" and self.cloth_grasp_vertex is not None:
            position = self.cloth_vertices[self.cloth_grasp_vertex].copy()
            return (position, np.array([1.0, 0.0, 0.0, 0.0]))
        return None

    def _cache_pose(self, path: str, pose: tuple[np.ndarray, np.ndarray]) -> None:
        position, orientation = pose
        self.poses_realtime[path] = (position.copy(), orientation.copy())
        self.poses.setdefault(path, (position.copy(), orientation.copy()))

    @staticmethod
    def _asset(path: str) -> str:
        resolved = resolve_usd_path(path)
        if not Path(resolved).is_file():
            raise FileNotFoundError(
                f"Missing cloth asset: {resolved}. "
                "Run ./app/python.sh scripts/physic_solver/prepare_physic_solver_assets.py first."
            )
        return resolved


class PickPlaceRigidSceneConfig(PickPlaceClothSceneConfig):
    """Use a PhysX rigid target on the same copied table."""

    object_kind: Literal["rigid"] = "rigid"
    cube_usd_path: str = "assets/PhysicSolver/Cube/Aligned.usd"
    cube_mass: float = Field(default=0.1, gt=0)
    object_clearance: float = Field(default=0.004, ge=0.001)


class PickPlaceRigidScene(PickPlaceClothScene):
    """Share table layout and lighting; read target state from the physical rigid-body view."""

    config_cls = PickPlaceRigidSceneConfig

    def __init__(self, config: PickPlaceRigidSceneConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        self.target_rigid_prim = None

    def init_scene(self, sim_world: Any) -> None:
        """Load the common table and reference the rigid cube."""
        from pxr import UsdGeom

        super().init_scene(sim_world)
        cube = UsdGeom.Xform.Define(sim_world.stage, self.target_prim_path)
        cube.GetPrim().GetReferences().AddReference(self._asset(self.config.cube_usd_path))

    def initialize_runtime(self) -> None:
        """Bind the cube tensor view after the world initializes PhysX."""
        self.target_rigid_prim.initialize()

    def _runtime_target_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.target_rigid_prim is None:
            return None
        position, orientation = self.target_rigid_prim.get_world_pose()
        result = []
        for value in (position, orientation):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            result.append(np.asarray(value, dtype=float).copy())
        return result[0], result[1]

    def _arrange_target(self) -> None:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = self.sim_world.stage
        root = stage.GetPrimAtPath(self.target_prim_path)
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        op = xform.AddTranslateOp(opSuffix="demo")
        bounds = prim_aabb(stage, self.target_prim_path, gprims_only=True)
        if bounds is None:
            raise RuntimeError("Could not measure the copied cube asset")
        z = self.table_top_z - float(bounds[0][2]) + self.config.object_clearance
        op.Set(Gf.Vec3d(*self.config.target_xy, z))
        UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(self.config.cube_mass)
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        self._bind_friction(root)
        from isaacsim.core.prims import SingleRigidPrim

        self.target_rigid_prim = SingleRigidPrim(
            prim_path=self.target_prim_path, name="pickplace_cube", reset_xform_properties=False
        )


# ================================================================================================
# Official Newton shirt layout, translated onto the project's table without flattening.
# ================================================================================================


class OfficialClothSceneConfig(PickPlaceClothSceneConfig):
    """Use the independently copied Newton asset and the official 80 cm workspace."""

    backend: str = "newton_cloth_franka"
    cloth_usd_path: str = "assets/PhysicSolver/Cloth/newton_official/unisex_shirt.usd"
    table_scale: list[float] = [4 / 3, 0.8, 1.0]
    robot_base_xy: list[float] = [-0.5, 0.0]
    cloth_thickness_scale: float = 1.0
    cloth_base_color: tuple[float, float, float] = Field(default=(0.16, 0.20, 0.65))
    cloth_roughness: float = Field(default=0.92, ge=0, le=1)
    cloth_specular: float = Field(default=0.02, ge=0, le=1)

    @field_validator("cloth_base_color")
    @classmethod
    def _color_range(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(not 0 <= component <= 1 for component in value):
            raise ValueError("cloth_base_color components must be between 0 and 1")
        return value


class OfficialClothScene(PickPlaceClothScene):
    """Preserve official rest geometry, rotation, and placement relative to the tabletop."""

    config_cls = OfficialClothSceneConfig

    @property
    def official_frame_translation(self) -> np.ndarray:
        """Translate official world metres (table centre 0,-0.5,0.2) to this table."""
        return np.array([*self.config.table_xy, self.table_top_z]) - [0.0, -0.5, 0.2]

    @property
    def robot_base_position(self) -> np.ndarray:
        """Match the official robot/table height difference and horizontal placement."""
        return np.array([-0.5, -0.5, 0.0]) + self.official_frame_translation

    def _initial_cloth_points(self, source: Any, mesh_prim: Any) -> np.ndarray:
        from pxr import Gf, UsdGeom

        points = np.asarray(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(), dtype=float)
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim)
        points = np.asarray([transform.Transform(Gf.Vec3d(*point)) for point in points])
        points *= UsdGeom.GetStageMetersPerUnit(source)
        # Official cloth_franka: rotate pi about Z; translate (0,70,30) centimetres.
        points[:, :2] *= -1
        points += np.array([0.0, 0.7, 0.3]) + self.official_frame_translation
        return points

    def arrange(self) -> None:
        """Place the authored shirt and add a visible support below the robot base."""
        from pxr import Gf, UsdGeom

        super().arrange()
        self._bind_cloth_material()
        height = float(self.robot_base_position[2])
        if height > 0:
            pedestal = UsdGeom.Cube.Define(
                self.sim_world.stage, f"{self.config.prim_path}/robot_pedestal"
            )
            pedestal.CreateSizeAttr(1.0)
            pedestal.AddTranslateOp().Set(Gf.Vec3d(*self.robot_base_position[:2], height / 2))
            pedestal.AddScaleOp().Set(Gf.Vec3f(0.18, 0.18, height))
            pedestal.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.35, 0.35)])

    def _bind_cloth_material(self) -> None:
        """Bind a matte dielectric; displayColor alone uses renderer fallback shading."""
        from pxr import Gf, Sdf, UsdShade

        stage = self.sim_world.stage
        path = f"{self.config.prim_path}/cloth_material"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        color = Gf.Vec3f(*self.config.cloth_base_color)
        specular = Gf.Vec3f(self.config.cloth_specular)
        for name, kind, value in (
            ("diffuseColor", Sdf.ValueTypeNames.Color3f, color),
            ("roughness", Sdf.ValueTypeNames.Float, self.config.cloth_roughness),
            ("metallic", Sdf.ValueTypeNames.Float, 0.0),
            ("clearcoat", Sdf.ValueTypeNames.Float, 0.0),
            ("useSpecularWorkflow", Sdf.ValueTypeNames.Int, 1),
            ("specularColor", Sdf.ValueTypeNames.Color3f, specular),
        ):
            shader.CreateInput(name, kind).Set(value)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(self.cloth_mesh.GetPrim()).Bind(material)
        self.cloth_mesh.CreateDisplayColorAttr([color])

    def publish_cloth_vertices(self, world_vertices: np.ndarray) -> None:
        """Update physical vertices and smooth shading normals on the same surface."""
        from pxr import UsdGeom, Vt

        super().publish_cloth_vertices(world_vertices)
        triangles = self.triangle_indices.reshape(-1, 3)
        points = np.asarray(world_vertices)
        face_normals = np.cross(
            points[triangles[:, 1]] - points[triangles[:, 0]],
            points[triangles[:, 2]] - points[triangles[:, 0]],
        )
        normals = np.zeros_like(points)
        for corner in range(3):
            np.add.at(normals, triangles[:, corner], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        self.cloth_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        self.cloth_mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))


# ================================================================================================
# V7 two-surface parcel, preserving authored masses, UVs and material bindings.
# ================================================================================================


class PackageSceneConfig(PickPlaceClothSceneConfig):
    """Load the independent V7 copy without flattening either physical mesh."""

    backend: str = "pickplace_package"
    cloth_usd_path: str = "assets/PhysicSolver/Package/v7/parcel_two_mesh.usd"
    target_xy: list[float] = [0.10, 0.0]
    robot_base_xy: list[float] = [-0.42, 0.0]


class PackageScene(PickPlaceClothScene):
    """Publish both freely simulated surfaces using the original asset's materials."""

    config_cls = PackageSceneConfig

    def _arrange_cloth(self) -> None:
        from pxr import Gf, Usd, UsdGeom

        path = self._asset(self.config.cloth_usd_path)
        source = Usd.Stage.Open(path)
        if UsdGeom.GetStageMetersPerUnit(source) != 1 or UsdGeom.GetStageUpAxis(source) != "Z":
            raise ValueError("Package asset must use metres and Z up")
        root = source.GetPrimAtPath("/World/Parcel")
        reference = root.GetAttribute("parcel:qualityReference")
        self.quality_reference = (
            json.loads(reference.Get())
            if reference and reference.HasAuthoredValueOpinion()
            else None
        )
        self.package_parameters = json.loads(root.GetAttribute("parcel:runtimeParameters").Get())
        self.loft_springs = np.asarray(root.GetAttribute("parcel:loftSpringIndices").Get(), int)
        weights = root.GetAttribute("parcel:loftSpringStiffnessWeights")
        self.loft_weights = (
            np.asarray(weights.Get(), float)
            if weights and weights.HasAuthoredValueOpinion()
            else np.ones(len(self.loft_springs))
        )
        self.package_offset = np.array([*self.config.target_xy, self.table_top_z])
        target = UsdGeom.Xform.Define(self.sim_world.stage, self.target_prim_path)
        # Looks is a sibling of Parcel. Reference their common ancestor so USD
        # remaps material bindings and GeomSubsets into the same composition arc.
        asset_path = f"{self.target_prim_path}/asset"
        asset = UsdGeom.Xform.Define(self.sim_world.stage, asset_path)
        asset.GetPrim().GetReferences().AddReference(path, "/World")
        # The copied asset's standalone PhysicsScene is inactive in the host world.
        self.sim_world.stage.GetPrimAtPath(f"{asset_path}/PhysicsScene").SetActive(False)
        target.ClearXformOpOrder()
        target.AddTranslateOp().Set(Gf.Vec3d(*self.package_offset))
        self.package_parts: list[dict[str, Any]] = []
        points, triangles, offset = [], [], 0
        for name in ("Garment", "Bag"):
            relative = f"{name}Body/{name}"
            original = UsdGeom.Mesh(source.GetPrimAtPath(f"/World/Parcel/{relative}"))
            vertices = np.asarray(original.GetPointsAttr().Get(), dtype=float)
            counts = np.asarray(original.GetFaceVertexCountsAttr().Get())
            indices = np.asarray(original.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            masses = np.asarray(original.GetPrim().GetAttribute("physics:masses").Get(), float)
            if not np.all(counts == 3) or len(masses) != len(vertices) or np.any(masses <= 0):
                raise ValueError(f"Invalid dynamic triangle surface: {name}")
            mesh = UsdGeom.Mesh.Get(self.sim_world.stage, f"{asset_path}/Parcel/{relative}")
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            self.package_parts.append(
                dict(
                    name=name,
                    mesh=mesh,
                    start=offset,
                    end=offset + len(vertices),
                    triangles=indices.reshape(-1, 3),
                    masses=masses,
                )
            )
            points.append(vertices + self.package_offset)
            triangles.append(indices + offset)
            offset += len(vertices)
        self.cloth_vertices = np.concatenate(points)
        self.triangle_indices = np.concatenate(triangles)
        self.garment_count = self.package_parts[0]["end"]
        if np.any(self.loft_springs < 0) or np.any(self.loft_springs >= self.garment_count):
            raise ValueError("Loft springs must connect garment vertices only")
        if self.package_parameters.get("bag_prestretch", 0) != 0:
            raise ValueError("This V7 importer requires an unprestretched bag")
        self.cloth_grasp_vertex = int(np.argmax(self.cloth_vertices[:, 2]))
        self.publish_cloth_vertices(self.cloth_vertices)

    def publish_cloth_vertices(self, world_vertices: np.ndarray) -> None:
        """Update actual local mesh points and normals, retaining UVs and subsets."""
        from pxr import UsdGeom, Vt

        points = np.asarray(world_vertices)
        if points.shape != self.cloth_vertices.shape or not np.isfinite(points).all():
            raise ValueError("Package positions must have the original finite (N, 3) shape")
        self.cloth_vertices = points.copy()
        for part in self.package_parts:
            local = points[part["start"] : part["end"]] - self.package_offset
            mesh, tri = part["mesh"], part["triangles"]
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
            mesh.CreateExtentAttr(
                Vt.Vec3fArray.FromNumpy(
                    np.asarray([local.min(axis=0), local.max(axis=0)], dtype=np.float32)
                )
            )
            normals = np.zeros_like(local)
            face = np.cross(
                local[tri[:, 1]] - local[tri[:, 0]], local[tri[:, 2]] - local[tri[:, 0]]
            )
            for corner in range(3):
                np.add.at(normals, tri[:, corner], face)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        self.refresh_poses()
