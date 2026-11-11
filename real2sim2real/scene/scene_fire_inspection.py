"""Fire warehouse scene: NVIDIA Flow fire on a cardboard box plus fixed cameras.

The warehouse and the Flow fire are one USD stage (``sim_world.stage_usd_path``); this scene adds
no geometry. It verifies the fire prims, enables Flow rendering, reads the fire position from the
live stage, creates the configured fixed cameras, answers line-of-sight queries against PhysX
collision geometry and drives the Flow clock.

Flow clock: Flow simulates on Kit render updates using the timeline's time delta, while
``IsaacsimWorld.step`` advances PhysX without touching the timeline. ``start_fire_clock`` stops
the timeline from auto-advancing, and ``sync_fire_time`` sets it to the physics time after every
step. The fire therefore evolves with simulation time and stays frozen while several cameras
render the same physics instant. Committing a timeline time change would also run the PhysX
stage-update node (one extra, uncounted physics step), so that node stays disabled while the
clock is held; ``IsaacsimWorld.step`` drives PhysX directly and does not need it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, field_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import get_logger
from real2sim2real.scene.base import BaseScene
from real2sim2real.simworld.base import BaseSimWorld
from real2sim2real.utils.asset_utils import prim_aabb, resolve_usd_path
from real2sim2real.utils.cameras import CameraConfig, create_cameras, validate_camera_list

_FLOW_SETTINGS = {
    "/rtx/flow/enabled": True,
    "/rtx/flow/pathTracingEnabled": True,
    "/rtx/flow/rayTracedReflectionsEnabled": True,
    "/rtx/flow/rayTracedTranslucencyEnabled": True,
    "/rtx/post/histogram/enabled": False,
    "/rtx/post/tonemap/op": 6,
}


class FireWarehouseSceneConfig(BaseConfig):
    """Fire metadata and fixed cameras for the fire warehouse."""

    backend: str = "fire_warehouse"
    fire_manifest_path: str = "assets/FireInspection/warehouse/fire_manifest.json"
    fire_fx_prim: str = "/FireFX"
    flow_extension: str = "omni.flowusd"
    fire_sight_margin_m: float = Field(
        default=0.15, ge=0, description="Ray hits this close to the fire point do not occlude it."
    )
    light_scale: float = Field(
        default=1.0,
        gt=0,
        description="Multiplier on the intensity of every light in the warehouse stage "
        "(lamps, fire glow, fill light); applied on the live stage only, assets unchanged.",
    )
    flow_max_steps_per_update: int = Field(
        default=8,
        ge=1,
        description="Live-stage override of flowSimulate.maxStepsPerSimulate (asset: 2). Flow "
        "steps at 60 Hz; one render per recorded frame must cover the whole interval.",
    )
    cameras: list[CameraConfig] = Field(default_factory=list)

    @field_validator("cameras")
    @classmethod
    def _validate_cameras(cls, value: list[CameraConfig]) -> list[CameraConfig]:
        return validate_camera_list(value)


class FireWarehouseScene(BaseScene):
    """Expose the fire point, its visibility and the Flow clock."""

    config_cls = FireWarehouseSceneConfig

    def __init__(self, config: FireWarehouseSceneConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("scene")
        self.log = get_logger(__name__)
        self.fire_position = np.zeros(3)
        self.fire_target_prim = ""
        self._timeline: Any = None
        self._fire_time_offset = 0.0
        self.flow_steps_per_second = 60.0

    def init_scene(self, sim_world: BaseSimWorld) -> None:
        """Check the composed fire stage, enable Flow rendering and create cameras."""
        import carb
        import omni.kit.app

        self.sim_world = sim_world
        stage = sim_world.stage
        manifest = json.loads(Path(resolve_usd_path(self.config.fire_manifest_path)).read_text())
        self.fire_target_prim = manifest["target_prim"]
        for path in (self.fire_target_prim, self.config.fire_fx_prim):
            if not stage.GetPrimAtPath(path):
                raise RuntimeError(f"Fire stage is missing {path}; check sim_world.stage_usd_path")
        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled(self.config.flow_extension):
            raise RuntimeError(
                f"{self.config.flow_extension} is not enabled; add it to sim_world.extensions"
            )
        settings = carb.settings.get_settings()
        for key, value in _FLOW_SETTINGS.items():
            settings.set(key, value)
        self.state["lights_scaled"] = self._scale_lights(stage)
        simulate = stage.GetPrimAtPath(f"{self.config.fire_fx_prim}/flowSimulate")
        self.flow_steps_per_second = float(simulate.GetAttribute("stepsPerSecond").Get())
        simulate.GetAttribute("maxStepsPerSimulate").Set(self.config.flow_max_steps_per_update)
        self.fire_position = self._fire_point(stage, manifest)
        self.sensors.update(create_cameras(stage, self.config.cameras))
        self.state["fire"] = {
            "target_prim": self.fire_target_prim,
            "position": self.fire_position.round(4).tolist(),
            "flow_extension": self.config.flow_extension,
        }
        self.log.info("Fire scene ready", **self.state["fire"])

    def _scale_lights(self, stage: Any) -> int:
        """Brighten the warehouse; runs before the robot (and its beacon) is added."""
        from pxr import Usd, UsdLux

        if self.config.light_scale == 1.0:
            return 0
        count = 0
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            if prim.HasAPI(UsdLux.LightAPI):
                intensity = UsdLux.LightAPI(prim).GetIntensityAttr()
                intensity.Set(float(intensity.Get()) * self.config.light_scale)
                count += 1
        return count

    def _fire_point(self, stage: Any, manifest: dict) -> np.ndarray:
        """Use the authored /FireFX origin; require it on the target box's upper rim."""
        from pxr import Usd, UsdGeom

        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(self.config.fire_fx_prim))
        origin = np.asarray(
            matrix.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        )
        box = prim_aabb(stage, self.fire_target_prim, gprims_only=True)
        if box is None:
            raise RuntimeError(f"Fire target has no geometry: {self.fire_target_prim}")
        lo, hi = box
        if not np.all((lo[:2] - 0.05 <= origin[:2]) & (origin[:2] <= hi[:2] + 0.05)) or not (
            hi[2] - 0.1 <= origin[2] <= hi[2] + 0.1
        ):
            raise RuntimeError(f"Fire origin {origin} is not on {self.fire_target_prim} {box}")
        expected = np.asarray(manifest["ignition_world"], dtype=float)
        if np.linalg.norm(origin - expected) > 0.02:
            raise RuntimeError(f"Fire origin {origin} differs from manifest {expected}")
        return origin

    def arrange(self) -> None:
        """Record the fire pose; the warehouse itself is static."""
        self.poses["fire"] = (self.fire_position.copy(), np.array([1.0, 0.0, 0.0, 0.0]))

    def adjust(self) -> None:
        """The warehouse is static; nothing to adjust."""

    # --- runtime -------------------------------------------------------------------------
    def fire_occluder(self, eye: np.ndarray, ignore_prefixes: tuple[str, ...] = ()) -> str | None:
        """First collision prim between ``eye`` and the fire point, or None if it is visible.

        Uses PhysX scene queries, so the physics view must be initialized. Hits on the burning
        box, within ``fire_sight_margin_m`` of the fire point, or under ``ignore_prefixes``
        (the robot itself) do not count.
        """
        import carb
        from omni.physx import get_physx_scene_query_interface

        eye = np.asarray(eye, float)
        delta = self.fire_position - eye
        distance = float(np.linalg.norm(delta))
        if distance <= self.config.fire_sight_margin_m:
            return None
        reach = distance - self.config.fire_sight_margin_m
        ignored = (self.fire_target_prim, *ignore_prefixes)
        blockers: list[tuple[float, str]] = []

        def report(hit: Any) -> bool:
            path = str(hit.collision)
            if not any(path == p or path.startswith(p + "/") for p in ignored):
                blockers.append((float(hit.distance), path))
            return True

        get_physx_scene_query_interface().raycast_all(
            carb.Float3(*eye), carb.Float3(*(delta / distance)), reach, report, True
        )
        return min(blockers)[1] if blockers else None

    def start_fire_clock(self, offset_s: float = 0.0) -> None:
        """Hand the timeline (and so Flow) to ``sync_fire_time``; call after world.play()."""
        import omni.timeline
        from omni.stageupdate import get_stage_update_interface

        self._stage_update = get_stage_update_interface()
        self._physics_nodes = [
            node["index"]
            for node in self._stage_update.get_stage_update_nodes()
            if node["name"] == "Physics"
        ]
        if len(self._physics_nodes) != 1:
            raise RuntimeError("Expected one 'Physics' stage update node to hold during Flow")
        # Toggling the node per step makes omni.physx log "Stage already attached!" each time,
        # so it is switched off once here and back on in stop_fire_clock.
        self._stage_update.set_stage_update_node_enabled(self._physics_nodes[0], False)
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline.set_end_time(1e7)  # the stage ends at 15 s; looping would reset Flow
        self._timeline.set_looping(False)
        self._timeline.set_auto_update(False)
        self._fire_time_offset = offset_s
        self.sync_fire_time(0.0)

    def sync_fire_time(self, simulation_s: float) -> None:
        """Set Flow's clock to the physics time; Flow steps on the next render update."""
        self._timeline.set_current_time(self._fire_time_offset + simulation_s)
        self._timeline.commit()

    def stop_fire_clock(self) -> None:
        """Give the timeline and the PhysX stage-update node back (GUI inspection after a run)."""
        if self._timeline is None:
            return
        self._stage_update.set_stage_update_node_enabled(self._physics_nodes[0], True)
        self._timeline.set_auto_update(True)
        self._timeline.commit()

    @property
    def fire_time_s(self) -> float:
        return float(self._timeline.get_current_time()) if self._timeline is not None else 0.0

    @property
    def max_render_interval_s(self) -> float:
        """Longest gap between render updates that Flow can still simulate in full."""
        return self.config.flow_max_steps_per_update / self.flow_steps_per_second
