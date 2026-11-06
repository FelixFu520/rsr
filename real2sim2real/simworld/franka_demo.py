"""Independently registered Isaac Sim world for the tabletop Franka demos."""

from __future__ import annotations

from pydantic import Field

from real2sim2real.simworld.isaacsim import IsaacsimSimWorld, IsaacsimSimWorldConfig


class FrankaDemoWorldConfig(IsaacsimSimWorldConfig):
    """Demo environment and camera; physics selection stays in the Isaac world."""

    background_usd_path: str = (
        "assets/Demo/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
    )
    background_skip_metrics_assembler: bool = True
    background_offset: tuple[float, float, float] = (-4.0, 0.0, 0.0)
    wait_init_time: int = 2
    physics_dt: float = Field(default=1.0 / 120.0, gt=0)
    rendering_dt: float = Field(default=1.0 / 30.0, gt=0)
    camera_eye: tuple[float, float, float] = (1.8, 1.65, 1.85)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.95)


class FrankaDemoWorld(IsaacsimSimWorld):
    """Warehouse background with a lit, close view of the robot and table."""

    config_cls = FrankaDemoWorldConfig

    def init_world(self) -> None:
        """Initialize the selected physics runtime, lighting, and inspection camera."""
        super().init_world()
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Gf, UsdGeom, UsdLux

        self.position_background()
        dome = UsdLux.DomeLight.Define(self.stage, "/World/DemoLighting/Fill")
        dome.CreateIntensityAttr(650.0)
        key = UsdLux.DistantLight.Define(self.stage, "/World/DemoLighting/Key")
        key.CreateIntensityAttr(2200.0)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35, -25, -30))
        set_camera_view(eye=self.config.camera_eye, target=self.config.camera_target)

    def position_background(self) -> None:
        """Center the workcell in the warehouse's open aisle, away from its shelves."""
        from pxr import Gf, UsdGeom

        if not self.config.background_usd_path or self.config.stage_usd_path:
            return
        prim = self.stage.GetPrimAtPath(self.config.background_prim_path)
        if not prim.IsValid():
            raise RuntimeError("The demo warehouse background was not loaded")
        xform = UsdGeom.Xformable(prim)
        name = "xformOp:translate:demo_aisle"
        existing = next((op for op in xform.GetOrderedXformOps() if op.GetOpName() == name), None)
        op = existing or xform.AddTranslateOp(opSuffix="demo_aisle")
        op.Set(Gf.Vec3d(*self.config.background_offset))
