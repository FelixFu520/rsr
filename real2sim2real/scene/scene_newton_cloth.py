"""Official Newton shirt layout, translated onto the project's table without flattening."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field, field_validator

from real2sim2real.scene.scene_physic_solver import PickPlaceClothScene, PickPlaceClothSceneConfig


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
