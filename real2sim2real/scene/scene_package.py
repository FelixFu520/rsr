"""V7 two-surface parcel, preserving authored masses, UVs and material bindings."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from real2sim2real.scene.scene_physic_solver import PickPlaceClothScene, PickPlaceClothSceneConfig


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
