"""Real Newton SDK diagnostics without Kit or a renderer, using Isaac's installed libraries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from real2sim2real.simworld.base import BaseSimWorld
from real2sim2real.simworld.isaacsim import IsaacsimWorldConfig


class NewtonDiagnosticWorldConfig(IsaacsimWorldConfig):
    """Direct Newton only; the batch launcher supplies the installed Isaac SDK environment."""

    backend: str = "newton_diagnostic"
    physics_engine: Literal["newton"] = "newton"
    newton_execution: Literal["direct"] = "direct"


class NewtonDiagnosticWorld(BaseSimWorld):
    """Own a real USD stage and direct Newton callbacks, with no SimulationApp or RTX state."""

    config_cls = NewtonDiagnosticWorldConfig
    physics_engine = "newton"
    uses_direct_newton = True

    def __init__(self, config: NewtonDiagnosticWorldConfig | None = None) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("sim_world")
        self.physics_steps = 0
        self._playing = False
        self._initialized = False
        self._stepper: Callable | None = None
        self._reset: Callable | None = None
        self._close: Callable | None = None

    def init_world(self) -> None:
        """Open metre-scale USD and the declared environment; defer models to the planners."""
        try:
            from pxr import Gf, Usd, UsdGeom
        except ImportError as exc:
            raise RuntimeError(
                "Use check_package_matrix.py to load Isaac's Newton/USD SDK paths"
            ) from exc
        from real2sim2real.utils.asset_utils import resolve_usd_path

        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1)
        UsdGeom.SetStageUpAxis(self.stage, "Z")
        root = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(root.GetPrim())
        if self.config.stage_usd_path:
            self.stage = Usd.Stage.Open(resolve_usd_path(self.config.stage_usd_path))
        elif self.config.background_usd_path:
            background = UsdGeom.Xform.Define(self.stage, self.config.background_prim_path)
            background.AddTranslateOp().Set(Gf.Vec3d(*self.config.background_offset))
            asset = UsdGeom.Xform.Define(self.stage, f"{self.config.background_prim_path}/asset")
            asset.GetPrim().GetReferences().AddReference(
                resolve_usd_path(self.config.background_usd_path)
            )
        if (
            UsdGeom.GetStageMetersPerUnit(self.stage) != 1
            or UsdGeom.GetStageUpAxis(self.stage) != "Z"
        ):
            raise ValueError("Newton diagnostics require metres and Z up")

    def register_newton_stepper(
        self, stepper: Callable, *, reset: Callable | None = None, close: Callable | None = None
    ) -> None:
        """Install exactly one physical stepping callback, as in the direct Isaac world."""
        self._stepper, self._reset, self._close = stepper, reset, close

    def reset(self) -> None:
        """Build the real Newton models once before the episode starts."""
        if self._stepper is None:
            raise RuntimeError("Direct Newton diagnostics require a registered physical stepper")
        if self._reset:
            self._reset()
        self.physics_steps = 0
        self._initialized = True

    def play(self) -> None:
        """Enable manual stepping without starting a timeline or rendering process."""
        if not self._initialized:
            self.reset()
        self._playing = True

    def step(self, *, render: bool = False) -> None:
        """Advance the actual registered Newton models by one physical frame."""
        if render:
            raise RuntimeError("This world has no renderer; render the accepted saved trajectory")
        if self._playing:
            self._stepper(self.config.physics_dt)
            self.physics_steps += 1

    def is_running(self) -> bool:
        """Report whether the physical diagnostic stage is open."""
        return self.stage is not None

    def destroy_world(self) -> None:
        """Release this diagnostic world's callbacks and stage."""
        if self._close:
            self._close()
        self._playing = False
        self.stage = None
        self._stepper = self._reset = self._close = None
