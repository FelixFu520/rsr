"""Finite grasp demonstrations using the normal, unchanged BaseJob lifecycle."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from real2sim2real.core.config import BaseConfig
from real2sim2real.job.base import BaseJob


class FrankaDemoJobConfig(BaseConfig):
    """Runtime limits and review artifacts for one grasp, hold, and release."""

    inspect_gui: bool = False
    max_steps: int = Field(default=12000, ge=1)
    timeout_s: float = Field(default=1200.0, gt=0)
    render_stride: int = Field(default=4, ge=1)
    keep_open: bool = False
    output_dir: str = "outputs/franka_demo"
    capture_frames: bool = True
    record_video: bool = False
    video_width: int = Field(default=1280, ge=64, le=4096)
    video_height: int = Field(default=720, ge=64, le=4096)
    fail_on_unsuccessful: bool = True


class FrankaDemoJob(BaseJob):
    """Assemble independently registered demo components and record measured success."""

    config_cls = FrankaDemoJobConfig

    def __init__(self, config: FrankaDemoJobConfig | None = None) -> None:
        super().__init__(config=config)
        self._validate_components()

    def _validate_components(self) -> None:
        for name in ("sim_world", "scene", "robot", "planner"):
            if getattr(self, name).config.backend != "franka_demo":
                raise ValueError(f"Franka demo requires [{name}].backend = 'franka_demo'")
        engine = self.sim_world.config.physics_engine
        kind = self.scene.config.object_kind
        if (kind == "rigid" and engine != "physx") or (
            kind == "cloth" and engine not in {"newton", "hybrid"}
        ):
            raise ValueError("Rigid demo requires physx; cloth requires newton or hybrid")
        if engine == "newton" and self.sim_world.config.newton_execution != "direct":
            raise ValueError("Cloth demo requires newton_execution='direct' for its shared model")
        if engine == "hybrid" and self.sim_world.config.newton_solver != "vbd":
            raise ValueError("This demo's hybrid cloth runtime requires newton_solver='vbd'")
        if self.planner.config.cloth_solver == "xpbd" and not (
            kind == "cloth" and engine == "newton" and self.sim_world.config.newton_solver == "xpbd"
        ):
            raise ValueError(
                "cloth_solver='xpbd' requires a pure direct Newton world with newton_solver='xpbd'"
            )
        if self.config.keep_open and self.sim_world.config.headless:
            raise ValueError("keep_open requires headless=false")

    def before_place_robot_hook(self) -> None:
        """Mount the fixed robot base on the measured tabletop before robot creation."""
        self.robot.config.base_position = tuple(self.scene.robot_base_position)

    def before_start_simulation_hook(self) -> None:
        """Initialize articulation handles, then install the selected Newton stepper."""
        self.sim_world.reset()
        self.sim_world.play()
        self.robot.initialize_runtime()
        self.planner.configure_scene(self.scene)
        self.planner.begin_episode()

    def simulate(self) -> None:
        """Execute a finite trial and persist metrics even when grasp validation fails."""
        self.sim_world.play()  # GUI inspection stops the timeline after the startup hook.
        self.robot.ensure_runtime_handles()
        started = time.monotonic()
        output = Path(self.config.output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        configuration = {
            name: getattr(self, name).config.model_dump(mode="json")
            for name in ("sim_world", "scene", "robot", "planner")
        }
        configuration["job"] = self.config.model_dump(mode="json")
        (output / "configuration.json").write_text(
            json.dumps(configuration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        captured: set[str] = set()
        completed = False
        error: str | None = None
        steps = 0
        video = None
        video_error: str | None = None
        video_metadata: dict[str, Any] | None = None
        try:
            if self.config.record_video:
                from real2sim2real.job.franka_demo_video import FrankaDemoVideo

                video = FrankaDemoVideo(
                    self.sim_world,
                    output,
                    stride=self.config.render_stride,
                    width=self.config.video_width,
                    height=self.config.video_height,
                )
                video.capture(step=0, phase=self.planner.state.get("phase", "idle"))
            for steps in range(1, self.config.max_steps + 1):
                if not self.sim_world.is_running():
                    error = "Simulation window closed"
                    break
                if time.monotonic() - started > self.config.timeout_s:
                    error = "Demo exceeded wall-clock timeout"
                    break
                completed = self.planner.on_physics_step(self.robot)
                self.sim_world.step(render=steps % self.config.render_stride == 0)
                phase = self.planner.state.get("phase", "idle")
                if video is not None and steps % self.config.render_stride == 0:
                    video.capture(step=steps, phase=phase)
                if phase != self.state.get("phase"):
                    self.log.info(
                        "Franka demo phase",
                        phase=phase,
                        step=steps,
                        measurement=self.planner.rows[-1] if self.planner.rows else {},
                    )
                    self.state["phase"] = phase
                if self.config.capture_frames and phase in {"hold", "done"}:
                    if phase not in captured:
                        self._capture_frame(output / f"{phase}.png")
                        captured.add(phase)
                if completed:
                    break
            if not completed and error is None:
                error = "Demo exceeded max_steps"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if video is not None:
                try:
                    video_metadata = video.finalize()
                except Exception as exc:
                    video_error = f"{type(exc).__name__}: {exc}"
                    error = error or video_error
                    self.log.error("Franka demo video failed", error=video_error)
            result: dict[str, Any] = self.planner.result()
            result.update(
                completed=completed,
                physics_engine=self.sim_world.config.physics_engine,
                physx_solver=(
                    self.sim_world.config.physx_solver
                    if self.sim_world.config.physics_engine in {"physx", "hybrid"}
                    else None
                ),
                newton_execution=(
                    self.sim_world.config.newton_execution
                    if self.sim_world.config.physics_engine == "newton"
                    else None
                ),
                newton_solver=(
                    self.sim_world.config.newton_solver
                    if self.sim_world.config.physics_engine != "physx"
                    else None
                ),
                object_kind=self.scene.config.object_kind,
                steps=steps,
                elapsed_s=time.monotonic() - started,
                error=error,
                video=video_metadata,
            )
            result["success"] = bool(result.get("success") and completed and error is None)
            self.state["result"] = result
            with (output / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
                for row in self.planner.rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            (output / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            self.log.info("Franka demo result", **result)
        if video_error is not None:
            raise RuntimeError(video_error)
        if self.config.fail_on_unsuccessful and not self.state["result"]["success"]:
            raise RuntimeError(f"Grasp validation failed; inspect {output / 'result.json'}")
        if self.config.keep_open:
            self.sim_world.pause()
            while self.sim_world.is_running():
                self.sim_world.app.update()

    def _capture_frame(self, path: Path) -> None:
        """Capture a review frame while physics is paused; no extra simulation steps."""
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            self.log.warning("No viewport available for demo capture")
            return
        self.sim_world.pause()
        try:
            future = capture_viewport_to_file(viewport, str(path))
            task = asyncio.ensure_future(future.wait_for_result())
            for _ in range(120):
                self.sim_world.app.update()
                if task.done():
                    task.result()
                    return
            task.cancel()
            self.log.warning("Demo capture timed out", path=str(path))
        finally:
            self.sim_world.play()
