"""Independent tabletop assets for the Franka rigid and cloth grasp demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import get_logger
from real2sim2real.scene.base import BaseScene
from real2sim2real.simworld.base import BaseSimWorld
from real2sim2real.utils import prim_aabb, read_prim_world_pose, resolve_usd_path


class FrankaDemoSceneConfig(BaseConfig):
    """Assets and meter-scale layout for ``[scene].backend = 'franka_demo'``."""

    object_kind: Literal["rigid", "cloth"] = "rigid"
    table_usd_path: str = "assets/Demo/Table/Aligned.usd"
    cube_usd_path: str = "assets/Demo/Cube/Aligned.usd"
    cloth_usd_path: str = "assets/Demo/Cloth/unisex_shirt.usd"
    prim_path: str = "/World/scene"
    table_scale: list[float] = Field(default_factory=lambda: [2.0, 1.2, 1.0])
    table_xy: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    target_xy: list[float] = Field(default_factory=lambda: [0.14, 0.0])
    robot_base_xy: list[float] = Field(default_factory=lambda: [-0.38, 0.0])
    cube_mass: float = Field(default=0.1, gt=0.0)
    object_clearance: float = Field(default=0.004, ge=0.001)
    cloth_initial_clearance: float = Field(default=0.015, ge=0.005)
    cloth_thickness_scale: float = Field(default=0.03, gt=0.0, le=1.0)
    grasp_vertex: int = Field(
        default=1922, ge=0, description="Validated side-seam vertex of the bundled shirt."
    )
    friction: float = Field(default=1.2, ge=0.0)

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


class FrankaDemoScene(BaseScene):
    """Place a fixed table and either a PhysX cube or a Newton cloth mesh.

    Newton vertices and table bounds are always world-space meters. Cloth points
    are authored in an identity-transform mesh so the solver can publish them
    without baking a second unit conversion into the visual geometry.
    """

    config_cls = FrankaDemoSceneConfig

    def __init__(self, config: FrankaDemoSceneConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        del kwargs
        self.config = config or self.config_cls.from_settings("scene")
        self.table_top_z = 0.0
        self.table_bounds = (np.zeros(3), np.zeros(3))
        self.cloth_vertices = np.empty((0, 3), dtype=np.float64)
        self.triangle_indices = np.empty(0, dtype=np.int32)
        self.cloth_mesh: Any | None = None
        self.cloth_grasp_vertex: int | None = None
        self.target_rigid_prim: Any | None = None
        self._target_initial_position = np.zeros(3)

    @property
    def table_prim_path(self) -> str:
        """Return the authored table root."""
        return f"{self.config.prim_path.rstrip('/')}/table"

    @property
    def target_prim_path(self) -> str:
        """Return the cube or cloth root."""
        return f"{self.config.prim_path.rstrip('/')}/{self.config.object_kind}"

    def init_scene(self, sim_world: BaseSimWorld) -> None:
        """Reference the copied table and the requested grasp asset."""
        from pxr import UsdGeom

        self.sim_world = sim_world
        if sim_world.stage is None:
            raise RuntimeError("Franka demo requires an initialized USD stage")
        stage = sim_world.stage
        UsdGeom.Xform.Define(stage, self.config.prim_path)
        table = UsdGeom.Xform.Define(stage, self.table_prim_path)
        table.GetPrim().GetReferences().AddReference(self._asset(self.config.table_usd_path))
        if self.config.object_kind == "rigid":
            cube = UsdGeom.Xform.Define(stage, self.target_prim_path)
            cube.GetPrim().GetReferences().AddReference(self._asset(self.config.cube_usd_path))

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
        if self.config.object_kind == "rigid":
            self._arrange_cube()
        else:
            self._arrange_cloth()
        self.refresh_poses()
        self._target_initial_position = self.target_position.copy()
        get_logger(type(self).__name__).info(
            "Franka demo table and target ready",
            object_kind=self.config.object_kind,
            table_top_z=self.table_top_z,
            target_position=self.target_position.tolist(),
            cloth_vertices=len(self.cloth_vertices),
        )

    def adjust(self) -> None:
        """The explicit demo layout needs no interactive adjustment."""

    def _configure_table_physics(self) -> None:
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = self.sim_world.stage
        root = stage.GetPrimAtPath(self.table_prim_path)
        # The table visual has two parts. Mesh collisions preserve the open leg
        # space and avoid a convex hull that fills the whole volume below it.
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
        self._bind_friction(root)

    def _arrange_cube(self) -> None:
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
        if getattr(self.sim_world, "world", None) is not None:
            from isaacsim.core.prims import SingleRigidPrim

            # Registered views initialize during the Job's reset after all
            # components are loaded. Read the tensor view during simulation;
            # Fabric can leave USD transform attributes at the initial pose.
            self.target_rigid_prim = self.sim_world.world.scene.add(
                SingleRigidPrim(
                    prim_path=self.target_prim_path,
                    name="franka_demo_cube",
                    reset_xform_properties=False,
                )
            )

    def _arrange_cloth(self) -> None:
        from pxr import Gf, Usd, UsdGeom, Vt

        source = Usd.Stage.Open(self._asset(self.config.cloth_usd_path))
        if source is None:
            raise RuntimeError("Could not open the copied shirt USD")
        mesh_prim = source.GetPrimAtPath("/root/shirt")
        if not mesh_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError("The Newton shirt asset must contain /root/shirt")
        source_mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(source_mesh.GetPointsAttr().Get(), dtype=np.float64)
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim)
        points = np.asarray([transform.Transform(Gf.Vec3d(*p)) for p in points])
        points *= UsdGeom.GetStageMetersPerUnit(source)
        # This particular Newton shirt is designed in the XY plane even though
        # its source stage says Y-up. Preserve its sewing topology and compress
        # the authored Z displacement to leave separated front/back shells.
        points -= 0.5 * (points.min(axis=0) + points.max(axis=0))
        points[:, 2] *= self.config.cloth_thickness_scale
        points[:, :2] += self.config.target_xy
        points[:, 2] += self.table_top_z + self.config.cloth_initial_clearance
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
        # Use a solid seam: inserting a finger through the sleeve opening can
        # keep the garment hooked even after the gripper has fully opened.
        self.cloth_grasp_vertex = self.config.grasp_vertex
        self.publish_cloth_vertices(points)

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
        """Return the current cube center or selected cloth vertex in meters."""
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
        has no single rigid orientation. The cube reads its PhysX tensor view.
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
            return position, np.array([1.0, 0.0, 0.0, 0.0])
        if self.target_rigid_prim is None:
            return None
        position, orientation = self.target_rigid_prim.get_world_pose()
        arrays = []
        for value in (position, orientation):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            arrays.append(np.asarray(value, dtype=np.float64).copy())
        return arrays[0], arrays[1]

    def _cache_pose(self, path: str, pose: tuple[np.ndarray, np.ndarray]) -> None:
        position, orientation = pose
        self.poses_realtime[path] = (position.copy(), orientation.copy())
        self.poses.setdefault(path, (position.copy(), orientation.copy()))

    @staticmethod
    def _asset(path: str) -> str:
        resolved = resolve_usd_path(path)
        if not Path(resolved).is_file():
            raise FileNotFoundError(
                f"Missing demo asset: {resolved}. Run scripts/prepare_franka_demo_assets.py "
                "with Isaac Sim python.sh first."
            )
        return resolved
