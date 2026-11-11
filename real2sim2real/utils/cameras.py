"""Named USD cameras shared by scene and robot backends; Kit imports stay lazy."""

from __future__ import annotations

import math
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

PositiveFloat = Annotated[FiniteFloat, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class CameraPose(BaseModel):
    """Camera pose in metres and a scalar-first quaternion, using USD camera axes (-Z/+Y)."""

    model_config = ConfigDict(extra="forbid")
    position: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.0, 0.0, 0.0)
    orientation_wxyz: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )

    @model_validator(mode="after")
    def _normalize_orientation(self) -> Self:
        norm = math.hypot(*self.orientation_wxyz)
        if norm < 1e-12:
            raise ValueError("Camera orientation must be a nonzero quaternion")
        self.orientation_wxyz = tuple(value / norm for value in self.orientation_wxyz)
        return self


class CameraConfig(BaseModel):
    """One entry in [[scene.cameras]] or [[robot.cameras]], mounted below its reference prim."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    prim_path: str
    pose_relative_to: str
    pose: CameraPose = Field(default_factory=CameraPose)
    resolution: tuple[PositiveInt, PositiveInt] = (640, 480)
    clipping_range: tuple[PositiveFloat, PositiveFloat] = (0.01, 80.0)
    focal_length: PositiveFloat = 18.0
    horizontal_aperture: PositiveFloat = 20.955

    @model_validator(mode="after")
    def _validate_camera(self) -> Self:
        for path in (self.prim_path, self.pose_relative_to):
            if not (path == "/World" or path.startswith("/World/")) or not all(
                part.isidentifier() for part in path.split("/")[1:]
            ):
                raise ValueError("Camera paths must be absolute prim paths below /World")
        if not self.prim_path.startswith(self.pose_relative_to + "/"):
            raise ValueError("Camera prim_path must be below pose_relative_to to follow its motion")
        if self.clipping_range[0] >= self.clipping_range[1]:
            raise ValueError("Camera clipping_range must satisfy 0 < near < far")
        if any(size % 2 for size in self.resolution):
            raise ValueError("H.264 camera dimensions must be even")
        return self


def validate_camera_list(cameras: list[CameraConfig]) -> list[CameraConfig]:
    """Reject duplicate camera names and paths before starting Kit."""
    if len({camera.name for camera in cameras}) != len(cameras):
        raise ValueError("Camera names must be unique")
    if len({camera.prim_path for camera in cameras}) != len(cameras):
        raise ValueError("Camera prim paths must be unique")
    return cameras


def create_cameras(stage: Any, cameras: list[CameraConfig]) -> dict[str, Any]:
    """Create optical prims under existing reference frames, preserving local mount poses."""
    from pxr import Gf, UsdGeom

    created = {}
    for config in cameras:
        reference = stage.GetPrimAtPath(config.pose_relative_to)
        if not reference or not reference.IsA(UsdGeom.Xformable):
            raise ValueError(f"Camera reference frame does not exist: {config.pose_relative_to}")
        parent_path = config.prim_path.rsplit("/", 1)[0]
        if not stage.GetPrimAtPath(parent_path):
            UsdGeom.Xform.Define(stage, parent_path)
        camera = UsdGeom.Camera.Define(stage, config.prim_path)
        pose = Gf.Matrix4d(1.0)
        q = config.pose.orientation_wxyz
        pose.SetRotate(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
        pose.SetTranslateOnly(Gf.Vec3d(*config.pose.position))
        cache = UsdGeom.XformCache()
        reference_world = cache.GetLocalToWorldTransform(reference)
        parent_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path))
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(pose * reference_world * parent_world.GetInverse())
        camera.CreateClippingRangeAttr(Gf.Vec2f(*config.clipping_range))
        # USD lens dimensions use tenths of stage units. This project uses metres,
        # so one authored unit is 100 mm. Both lengths scale together to preserve FOV.
        camera.CreateFocalLengthAttr(config.focal_length / 100.0)
        camera.CreateHorizontalApertureAttr(config.horizontal_aperture / 100.0)
        width, height = config.resolution
        camera.CreateVerticalApertureAttr(config.horizontal_aperture * height / width / 100.0)
        created[config.name] = camera
    return created
