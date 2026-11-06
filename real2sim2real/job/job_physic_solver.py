"""PhysicSolver 求解器组合的衣服/刚体抓放任务与 LeRobot v3 采集。

PhysicSolverJob 复用 BaseJob 固定生命周期，装配统一世界、场景、机器人和 Planner，
通过任务钩子绑定物理步进与初始化，再在 simulate 中执行抓放、验收和同步录制。
配置继续使用 job.backend="pickplace_cloth"，现有 PhysicSolver TOML 无需修改。
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.job.base import BaseJob
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder, capture_viewport_frame


class PhysicSolverRecorderConfig(BaseModel):
    """RGB and joint data cadence; timestamps start at zero for each exported episode."""

    model_config = ConfigDict(extra="forbid")
    format: Literal["lerobot_v3"] = "lerobot_v3"
    fps: int = Field(default=15, gt=0)
    width: int = Field(default=640, ge=64)
    height: int = Field(default=480, ge=64)
    task: str = "Pick up the shirt, move it to the designated place on the table, and release it."
    save_failed_episodes: bool = False

    @model_validator(mode="after")
    def _validate_video(self) -> PhysicSolverRecorderConfig:
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 image dimensions must be even")
        return self


class PhysicSolverJobConfig(BaseConfig):
    """Finite task execution and accepted dataset destination."""

    output_dir: str = "outputs/PhysicSolver/featherstone_vbd_pickplace_cloth"
    max_steps: int = Field(default=6000, gt=0)
    timeout_s: float = Field(default=1800, gt=0)
    inspect_gui: bool = False
    recorder: PhysicSolverRecorderConfig = Field(default_factory=PhysicSolverRecorderConfig)


class PhysicSolverJob(BaseJob):
    """Keep task loops and recording in the Job; reuse the common IsaacsimWorld lifecycle."""

    config_cls = PhysicSolverJobConfig

    def __init__(self, config: PhysicSolverJobConfig | None = None) -> None:
        super().__init__(config=config)
        kind = self.scene.config.object_kind
        engine = self.sim_world.physics_engine
        if kind == "rigid" and engine != "physx":
            raise ValueError("The rigid case requires PhysX")
        if kind == "cloth" and not (
            getattr(self.sim_world, "uses_direct_newton", False) or engine == "hybrid"
        ):
            raise ValueError("Cloth requires Newton direct or hybrid execution")
        if engine == "hybrid" and self.planner.config.cloth_solver != "vbd":
            raise ValueError("Hybrid cloth requires VBD")
        if self.planner.config.cloth_solver == "xpbd" and not (
            kind == "cloth" and engine == "newton" and self.sim_world.config.newton_solver == "xpbd"
        ):
            raise ValueError("XPBD cloth requires the unified Newton XPBD model")
        cadence = 1 / (self.sim_world.config.physics_dt * self.config.recorder.fps)
        if not math.isclose(cadence, round(cadence), abs_tol=1e-8, rel_tol=0):
            raise ValueError("Recorder fps must divide the physics frequency exactly")
        self.record_stride = round(cadence)
        if (Path(self.config.output_dir) / "lerobot_v3").exists():
            raise FileExistsError("Accepted dataset already exists; select another job.output_dir")

    def before_place_robot_hook(self) -> None:
        """Mount the robot on the measured table before its asset is initialized."""
        self.robot.config.base_position = tuple(self.scene.robot_base_position)

    def before_start_simulation_hook(self) -> None:
        """Build and register the Newton model before the unified world checks readiness."""
        self.planner.configure_scene(self.scene)
        if self.scene.config.object_kind == "cloth":
            self.sim_world.register_newton_stepper(self._step_newton, reset=self._reset_components)
        self.sim_world.reset()
        if self.scene.config.object_kind == "rigid":
            self._reset_components()

    def _step_newton(self, dt: float) -> None:
        self.planner.runtime.step(dt)

    def _reset_components(self) -> None:
        for component in (self.robot, self.scene):
            initialize = getattr(component, "initialize_runtime", None)
            if initialize is not None:
                initialize()
        self.planner.begin_episode(register_stepper=False)

    def simulate(self) -> None:
        """Capture matched observations/actions and publish physically accepted data."""
        output = Path(self.config.output_dir).expanduser().resolve()
        run = output / "diagnostics" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True)
        configs = {
            name: getattr(self, name).config.model_dump(mode="json")
            for name in ("sim_world", "scene", "robot", "planner")
        }
        configs["job"] = self.config.model_dump(mode="json")
        (run / "configuration.json").write_text(json.dumps(configs, indent=2) + "\n")
        settings = self.config.recorder
        recorder = LeRobotV3Recorder(
            output / "lerobot_v3",
            fps=settings.fps,
            width=settings.width,
            height=settings.height,
            task=settings.task,
            robot_type="franka_fr3_newton" if self.robot.pure_newton else "franka_panda_physx",
            joint_names=None
            if self.robot.pure_newton
            else ([f"panda_joint{i}" for i in range(1, 8)] + ["left_finger_m", "right_finger_m"]),
        )
        for _ in range(8):
            self.sim_world.update()
        self.sim_world.play()
        started = time.monotonic()
        completed = False
        failure: str | None = None
        result = {}
        try:
            for step in range(self.config.max_steps):
                if not self.sim_world.is_running():
                    raise RuntimeError("Isaac Sim application closed")
                if time.monotonic() - started > self.config.timeout_s:
                    raise TimeoutError("PhysicSolver task exceeded timeout_s")
                completed = self.planner.on_physics_step(self.robot)
                phase = self.planner.state["phase"]
                if phase != self.state.get("phase"):
                    self.log.info("PhysicSolver task phase", phase=phase, step=step)
                    self.state["phase"] = phase
                if completed:
                    break
                if step % self.record_stride == 0:
                    capture_viewport_frame(
                        self.sim_world,
                        recorder.frame_path(),
                        width=settings.width,
                        height=settings.height,
                    )
                    if self.planner.runtime is not None and self.robot.pure_newton:
                        state, action = self.planner.runtime.recording_coordinates()
                    else:
                        state, action = self.robot.recording_coordinates()
                    recorder.add_frame(state, action)
                self.sim_world.step(render=(step + 1) % self.record_stride == 0)
            result = self.planner.result()
            result["completed"] = completed
            if not completed or (not result["success"] and not settings.save_failed_episodes):
                raise RuntimeError(
                    f"PhysicSolver physical acceptance failed: {result.get('checks')}"
                )
            dataset = recorder.finalize(episode_success=bool(result["success"]))
            result["dataset"] = str(dataset)
            result["frames"] = len(recorder.rows)
            result["dataset_use"] = "demonstration" if result["success"] else "diagnostic"
            (dataset / "meta/task_outcome.json").write_text(
                json.dumps({"physical_result": result, "configuration": configs}, indent=2) + "\n"
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            recorder.discard(run)
            raise
        finally:
            if not result:
                result = self.planner.result()
            result.update(
                error=failure,
                elapsed_s=time.monotonic() - started,
                success=bool(result.get("success") and completed and failure is None),
                physics_engine=self.sim_world.physics_engine,
                newton_execution=self.sim_world.config.newton_execution,
                rigid_solver=(
                    self.sim_world.config.newton_solver
                    if self.sim_world.physics_engine == "newton"
                    else self.sim_world.config.physx_solver.lower()
                ),
                contact_solver=(
                    self.planner.config.cloth_solver
                    if self.scene.config.object_kind == "cloth"
                    else self.sim_world.config.physx_solver.lower()
                ),
                cloth_solver=(
                    self.planner.config.cloth_solver
                    if self.scene.config.object_kind == "cloth"
                    else None
                ),
                object_kind=self.scene.config.object_kind,
            )
            self.state["result"] = result
            (run / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            (output / "result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            with (run / "trajectory.jsonl").open("w") as stream:
                for row in self.planner.rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.log.info("PhysicSolver task result", **result)
