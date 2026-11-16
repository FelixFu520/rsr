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
from real2sim2real.utils.cameras import validate_camera_list
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

    output_dir: str = "outputs/physic_solver/featherstone_vbd_cloth01_baseline"
    max_steps: int = Field(default=6000, gt=0)
    timeout_s: float = Field(
        default=1800, gt=0, description="Wall-clock limit for the loop, including RGB capture."
    )
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
        self.recording_cameras = validate_camera_list(
            [
                *self.scene.config.cameras,
                *self.robot.config.cameras,
            ]
        )
        if (Path(self.config.output_dir) / "lerobot_v3").exists():
            raise FileExistsError("Accepted dataset already exists; select another job.output_dir")

    def before_place_robot_hook(self) -> None:
        """Mount the robot on the measured table before its asset is initialized."""
        self.robot.config.base_position = tuple(self.scene.robot_base_position)
        prepare = getattr(self.planner, "prepare_initial_joints", None)
        if prepare is not None:
            prepare(self.robot, self.scene)

    def before_start_simulation_hook(self) -> None:
        """Build and register the Newton model before the unified world checks readiness."""
        self.planner.configure_scene(self.scene)
        if self.scene.config.object_kind == "cloth":
            self.sim_world.register_newton_stepper(self._step_newton, reset=self._reset_components)
        self.sim_world.reset()
        if self.scene.config.object_kind == "rigid":
            self._reset_components()

    def setup_gui_hook(self) -> None:
        """Show the first configured camera; all named cameras remain available in the viewport."""
        if self.recording_cameras:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = self.recording_cameras[0].prim_path

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
            robot_type=getattr(self.robot, "recording_robot_type", None)
            or ("franka_fr3_newton" if self.robot.pure_newton else "franka_panda_physx"),
            joint_names=getattr(self.robot, "recording_joint_names", None)
            if self.robot.pure_newton
            else ([f"panda_joint{i}" for i in range(1, 8)] + ["left_finger_m", "right_finger_m"]),
            cameras={cam.name: cam.resolution for cam in self.recording_cameras} or None,
            action_names=getattr(self.planner, "action_names", None),
        )
        for _ in range(8):
            self.sim_world.update()
        self.sim_world.play()
        started = time.monotonic()
        next_progress = started
        timings = dict(control=0.0, capture=0.0, recording=0.0, physics=0.0, export=0.0)
        expected_duration = getattr(self.planner, "expected_duration_s", None)
        self.log.info(
            "PhysicSolver recording started",
            renderer=self.sim_world.config.renderer,
            samples_per_pixel_per_frame=self.sim_world.config.samples_per_pixel_per_frame,
            rt_subframes_per_camera=self.sim_world.config.rt_subframes,
            cameras=recorder.cameras,
            fps=settings.fps,
            expected_simulation_s=expected_duration,
            wall_timeout_s=self.config.timeout_s,
        )
        completed = False
        failure: str | None = None
        result = {}
        try:
            for step in range(self.config.max_steps):
                if not self.sim_world.is_running():
                    raise RuntimeError("Isaac Sim application closed")
                elapsed = time.monotonic() - started
                if elapsed > self.config.timeout_s:
                    simulation_s = self.sim_world.physics_steps * self.sim_world.config.physics_dt
                    raise TimeoutError(
                        f"PhysicSolver task exceeded timeout_s={self.config.timeout_s:g}: "
                        f"wall={elapsed:.1f}s, simulation={simulation_s:.3f}s, "
                        f"frames={len(recorder.rows)}, capture={timings['capture']:.1f}s, "
                        f"physics={timings['physics']:.1f}s"
                    )
                operation_started = time.monotonic()
                completed = self.planner.on_physics_step(self.robot)
                timings["control"] += time.monotonic() - operation_started
                phase = self.planner.state["phase"]
                if phase != self.state.get("phase"):
                    self.log.info("PhysicSolver task phase", phase=phase, step=step)
                    self.state["phase"] = phase
                if completed:
                    break
                if step % self.record_stride == 0:
                    operation_started = time.monotonic()
                    self._capture_cameras(recorder)
                    timings["capture"] += time.monotonic() - operation_started
                    operation_started = time.monotonic()
                    if self.planner.runtime is not None and self.robot.pure_newton:
                        state, action = self.planner.runtime.recording_coordinates()
                    else:
                        state, action = self.robot.recording_coordinates()
                    recorder.add_frame(state, action)
                    timings["recording"] += time.monotonic() - operation_started
                operation_started = time.monotonic()
                # Every recorded view renders explicitly in _capture_cameras. A second
                # render after stepping only draws an uncaptured pose and resets PT history.
                self.sim_world.step(render=False)
                timings["physics"] += time.monotonic() - operation_started
                now = time.monotonic()
                if now >= next_progress:
                    simulation_s = self.sim_world.physics_steps * self.sim_world.config.physics_dt
                    remaining_s = (
                        (now - started) * max(0.0, expected_duration - simulation_s) / simulation_s
                        if expected_duration is not None and simulation_s >= 1.0
                        else None
                    )
                    self.log.info(
                        "PhysicSolver recording progress",
                        simulation_s=round(simulation_s, 3),
                        expected_simulation_s=expected_duration,
                        frames=len(recorder.rows),
                        wall_s=round(now - started, 1),
                        estimated_remaining_s=round(remaining_s, 1)
                        if remaining_s is not None
                        else None,
                        timings_s={name: round(value, 2) for name, value in timings.items()},
                        physical_progress=self.planner.state.get("progress"),
                    )
                    next_progress = now + 30.0
            result = self.planner.result()
            result["completed"] = completed
            if not completed or (not result["success"] and not settings.save_failed_episodes):
                raise RuntimeError(
                    f"PhysicSolver physical acceptance failed: {result.get('checks')}"
                )
            operation_started = time.monotonic()
            dataset = recorder.finalize(episode_success=bool(result["success"]))
            timings["export"] += time.monotonic() - operation_started
            result["timings_s"] = timings.copy()
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
                timings_s=timings.copy(),
                captured_frames=len(recorder.rows),
                physics_steps=self.sim_world.physics_steps,
                simulation_time_s=self.sim_world.physics_steps * self.sim_world.config.physics_dt,
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
            write_diagnostics = getattr(self.planner, "write_diagnostics", None)
            if write_diagnostics is not None:
                write_diagnostics(run)
            (run / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            (output / "result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            with (run / "trajectory.jsonl").open("w") as stream:
                for row in self.planner.rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.log.info("PhysicSolver task result", **result)

    def _capture_cameras(self, recorder: LeRobotV3Recorder) -> None:
        """Capture every view at the same physics instant and restore the GUI camera."""
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No active viewport for RGB capture")
        previous_camera, previous_resolution = viewport.camera_path, viewport.resolution
        physics_step = self.sim_world.physics_steps
        try:
            if not self.recording_cameras:
                capture_viewport_frame(
                    self.sim_world,
                    recorder.frame_path(),
                    width=recorder.width,
                    height=recorder.height,
                )
            for camera in self.recording_cameras:
                capture_viewport_frame(
                    self.sim_world,
                    recorder.frame_path(camera.name),
                    width=camera.resolution[0],
                    height=camera.resolution[1],
                    camera_prim=camera.prim_path,
                )
                if self.sim_world.physics_steps != physics_step:
                    raise RuntimeError("Multi-camera capture advanced physics between views")
            if self.sim_world.physics_steps != physics_step:
                raise RuntimeError("Multi-camera capture advanced physics")
        finally:
            viewport.camera_path, viewport.resolution = previous_camera, previous_resolution


# ================================================================================================
# Screw assembly shares the BaseJob lifecycle; recorder/state dimensions are tool-specific.
# ================================================================================================


class ScrewJobConfig(PhysicSolverJobConfig):
    """Screw assembly output and optional physics-only diagnostics."""

    backend: str = "screw_assembly"
    output_dir: str = "outputs/physic_solver/mujoco_hydroelastic_screw01_rotate"
    record_rgb: bool = True


class ScrewJob(BaseJob):
    """Assemble the fixed nut and guided spindle, execute physical threading, and record."""

    config_cls = ScrewJobConfig

    def __init__(self, config: ScrewJobConfig | None = None) -> None:
        super().__init__(config=config)
        if not self.sim_world.uses_direct_newton or self.sim_world.config.newton_solver != "mujoco":
            raise ValueError("Screw assembly requires direct Newton MuJoCo")
        combination = (
            self.scene.config.backend,
            self.robot.config.backend,
            self.planner.config.backend,
        )
        if combination not in {
            ("screw_assembly", "screw_spindle", "screw_assembly"),
            ("screw_assembly", "ur5e_screwdriver", "ur5e_screwdriver"),
            ("dual_screw_assembly", "dual_ur5e_screwdriver", "dual_ur5e_screwdriver"),
            ("pickplace_carton", "franka_carton_gripper", "pickplace_carton"),
        }:
            raise ValueError("Assembly requires its declared scene and robot combination")
        ratio = 1 / (self.config.recorder.fps * self.sim_world.config.physics_dt)
        if not math.isclose(ratio, round(ratio), abs_tol=1e-8):
            raise ValueError("Recorder fps must divide physics frequency")
        if (
            self.sim_world.config.newton_use_cuda_graph
            and not getattr(self.planner.config, "mujoco_cpu", False)
            and self.sim_world.config.newton_substeps % 2
        ):
            raise ValueError("Assembly CUDA graph requires an even number of substeps")
        self.record_stride = round(ratio)
        self.recording_cameras = validate_camera_list(
            [*self.scene.config.cameras, *self.robot.config.cameras]
        )
        output = Path(self.config.output_dir)
        if (output / "result.json").exists() or (output / "lerobot_v3").exists():
            raise FileExistsError(f"Select a fresh screw output directory: {output}")

    def before_place_robot_hook(self) -> None:
        """Mount the spindle exactly above the measured table fixture."""
        self.robot.config.base_position = tuple(self.scene.screw_origin)

    def before_start_simulation_hook(self) -> None:
        """Build the model and register its one direct-Newton step callback."""
        self.planner.begin_episode(self.scene)
        self.sim_world.register_newton_stepper(self.planner.runtime.step)
        self.sim_world.reset()

    def simulate(self) -> None:
        """Collect measured joint states and contact diagnostics; export accepted demonstrations."""
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        configs = {
            name: getattr(self, name).config.model_dump(mode="json")
            for name in ("sim_world", "scene", "robot", "planner")
        }
        configs["job"] = self.config.model_dump(mode="json")
        (output / "configuration.json").write_text(json.dumps(configs, indent=2) + "\n")
        recorder = None
        if self.config.record_rgb:
            recorder = LeRobotV3Recorder(
                output / "lerobot_v3",
                fps=self.config.recorder.fps,
                width=self.config.recorder.width,
                height=self.config.recorder.height,
                task=getattr(
                    self.planner,
                    "recording_task",
                    "Use the continuous rotary spindle to thread a bolt through the fixed nut.",
                ),
                robot_type=self.robot.recording_robot_type,
                joint_names=self.robot.recording_joint_names,
                action_names=self.planner.action_names,
                cameras={cam.name: cam.resolution for cam in self.recording_cameras},
            )
        self.sim_world.play()
        started, next_report = time.monotonic(), 0.0
        result = {}
        error = None
        try:
            for step in range(self.config.max_steps):
                if time.monotonic() - started > self.config.timeout_s:
                    raise TimeoutError("Screw assembly exceeded its wall-clock limit")
                if self.planner.on_physics_step(self.robot):
                    break
                if recorder is not None and step % self.record_stride == 0:
                    PhysicSolverJob._capture_cameras(self, recorder)
                    recorder.add_frame(*self.planner.runtime.recording_coordinates())
                self.sim_world.step(render=False)
                elapsed = time.monotonic() - started
                if elapsed >= next_report:
                    self.log.info(
                        "PhysicSolver assembly progress",
                        simulation_s=self.planner.elapsed,
                        phase=self.planner.state["phase"],
                        wall_s=elapsed,
                        measurement=self.planner.rows[-1],
                    )
                    next_report = elapsed + 30
            result = self.planner.result()
            result["completed"] = bool(self.planner.state["done"])
            if recorder is not None:
                if result["success"] or self.config.recorder.save_failed_episodes:
                    dataset = recorder.finalize(episode_success=result["success"])
                    result.update(
                        dataset=str(dataset),
                        frames=len(recorder.rows),
                        dataset_use="demonstration" if result["success"] else "diagnostic",
                    )
                    (dataset / "meta/task_outcome.json").write_text(
                        json.dumps({"physical_result": result, "configuration": configs}, indent=2)
                        + "\n"
                    )
                else:
                    recorder.discard(output / "diagnostics")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if recorder is not None:
                recorder.discard(output / "diagnostics")
            raise
        finally:
            if not result:
                result = self.planner.result()
            result.update(
                success=bool(result.get("success") and error is None),
                error=error,
                elapsed_s=time.monotonic() - started,
                physics_steps=self.sim_world.physics_steps,
            )
            self.state["result"] = result
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            (output / "trajectory.json").write_text(json.dumps(self.planner.rows) + "\n")
            self.log.info("PhysicSolver assembly result", **result)


class CartonJobConfig(ScrewJobConfig):
    """Rigid V10 carton demonstration using the same finite assembly recording loop."""

    backend: str = "pickplace_carton"
    output_dir: str = "outputs/physic_solver/mujoco_mujoco_package07_center"


class CartonJob(ScrewJob):
    """Reuse the fixed lifecycle and recorder for the long-stroke carton grasp subtask."""

    config_cls = CartonJobConfig

    def before_place_robot_hook(self) -> None:
        """Mount the Franka on the measured table instead of the screw-fixture origin."""
        self.robot.config.base_position = tuple(self.scene.robot_base_position)
        self.planner.prepare_initial_joints(self.robot, self.scene)


# ================================================================================================
# Banana mesh-cutting demonstration: measured PhysX poses, LeRobot data and independent USD replay.
# ================================================================================================


class BananaJobConfig(PhysicSolverJobConfig):
    """Finite cutting capture; failed runs belong in the debug tree."""

    backend: str = "banana_cutting"
    output_dir: str = "outputs/debug/physic_solver/kinematic_tgs_banana01_five_cuts_debug"
    record_rgb: bool = True
    render_warmup_updates: int = Field(default=48, ge=0)


class BananaJob(BaseJob):
    """Reuse the common world lifecycle and keep cutting-specific recording in this Job."""

    config_cls = BananaJobConfig

    def __init__(self, config: BananaJobConfig | None = None) -> None:
        super().__init__(config=config)
        if self.sim_world.physics_engine != "physx":
            raise ValueError("Banana fragments require PhysX")
        if (self.scene.config.backend, self.robot.config.backend, self.planner.config.backend) != (
            "banana_cutting",
            "scripted_knife",
            "banana_cutting",
        ):
            raise ValueError("Banana cutting requires its declared scene, knife and planner")
        ratio = 1 / (self.config.recorder.fps * self.sim_world.config.physics_dt)
        if not math.isclose(ratio, round(ratio), abs_tol=1e-8):
            raise ValueError("Recorder fps must divide the physics frequency")
        total = self.planner.config.duration_s / self.sim_world.config.physics_dt
        if not math.isclose(total, round(total), abs_tol=1e-8) or self.config.max_steps < round(
            total
        ):
            raise ValueError("duration_s must contain whole physics steps within max_steps")
        frames = self.planner.config.duration_s * self.config.recorder.fps
        if not math.isclose(frames, round(frames), abs_tol=1e-8):
            raise ValueError("duration_s must contain whole recording frames")
        self.record_stride = round(ratio)
        self.recording_cameras = validate_camera_list(self.scene.config.cameras)
        if self.config.record_rgb and not self.recording_cameras:
            raise ValueError("RGB capture requires at least one named scene camera")
        output = Path(self.config.output_dir)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Choose a fresh banana output directory: {output}")

    def before_start_simulation_hook(self) -> None:
        """Require USD pose publication so the trace contains actual PhysX transforms."""
        import carb
        from isaacsim.core.simulation_manager import SimulationManager

        carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
        carb.settings.get_settings().set_bool("/physics/updateVelocitiesToUsd", True)
        if self.sim_world.config.renderer == "MinimalRendering":
            carb.settings.get_settings().set_bool("/rtx/indirectDiffuse/enabled", False)
        SimulationManager.enable_fabric(False)
        self.planner.begin_episode(self.scene)
        self.sim_world.reset()

    def setup_gui_hook(self) -> None:
        """Warm the loaded materials at the recording camera before physical time starts."""
        PhysicSolverJob.setup_gui_hook(self)
        if self.config.record_rgb:
            for _ in range(self.config.render_warmup_updates):
                self.sim_world.update()

    def simulate(self) -> None:
        """Sample every physical step, capture RGB at exact cadence, and retain failure evidence."""
        import numpy as np

        output = Path(self.config.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        configs = {
            name: getattr(self, name).config.model_dump(mode="json")
            for name in ("sim_world", "scene", "robot", "planner")
        }
        configs["job"] = self.config.model_dump(mode="json")
        (output / "configuration.json").write_text(json.dumps(configs, indent=2) + "\n")
        recorder = None
        result, error = {}, None
        started = time.monotonic()
        try:
            if self.config.record_rgb:
                settings = self.config.recorder
                recorder = LeRobotV3Recorder(
                    output / "lerobot_v3",
                    fps=settings.fps,
                    width=settings.width,
                    height=settings.height,
                    task=settings.task,
                    robot_type="scripted_knife",
                    joint_names=self.planner.action_names,
                    action_names=self.planner.action_names,
                    cameras={c.name: c.resolution for c in self.recording_cameras},
                )
            self.sim_world.play()
            next_report = 0.0
            while not self.planner.on_physics_step(self.robot):
                if time.monotonic() - started > self.config.timeout_s:
                    raise TimeoutError("Banana capture exceeded timeout_s")
                self.planner.observe()
                if recorder is not None and self.sim_world.physics_steps % self.record_stride == 0:
                    PhysicSolverJob._capture_cameras(self, recorder)
                    recorder.add_frame(self.robot.coordinates, self.robot.coordinates)
                self.sim_world.step(render=False)
                if time.monotonic() - started >= next_report:
                    self.log.info(
                        "Banana cutting progress",
                        physics_steps=self.sim_world.physics_steps,
                        cuts=len(self.scene.events),
                        wall_s=time.monotonic() - started,
                    )
                    next_report = time.monotonic() - started + 30
            self.planner.observe()
            result = self.planner.result()
            self._export_replay(output)
            if recorder is not None:
                if result["success"] or self.config.recorder.save_failed_episodes:
                    dataset = recorder.finalize(episode_success=result["success"])
                    result.update(
                        frames=len(recorder.rows),
                        dataset_use="simulation_demo" if result["success"] else "diagnostic",
                    )
                    (dataset / "meta/task_outcome.json").write_text(
                        json.dumps({"physical_result": result, "configuration": configs}, indent=2)
                        + "\n"
                    )
                else:
                    recorder.discard(output / "diagnostics")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if recorder is not None and recorder.staging.exists():
                recorder.discard(output / "diagnostics")
            raise
        finally:
            if not result:
                result = self.planner.result()
            arrays = {}
            for index, mesh in enumerate(self.scene.meshes.values()):
                arrays[f"v{index}"] = mesh.vertices
                arrays[f"f{index}"] = mesh.faces
                arrays[f"m{index}"] = mesh.materials
            np.savez_compressed(output / "meshes.npz", **arrays)
            (output / "mesh_paths.json").write_text(json.dumps(list(self.scene.meshes)) + "\n")
            (output / "cut_events.json").write_text(json.dumps(self.scene.events, indent=2) + "\n")
            (output / "trajectory.json").write_text(json.dumps(self.planner.rows) + "\n")
            result.update(
                success=bool(result.get("success") and error is None),
                error=error,
                elapsed_s=time.monotonic() - started,
                physics_steps=self.sim_world.physics_steps,
                simulation_time_s=self.sim_world.physics_steps * self.sim_world.config.physics_dt,
                physics_engine="physx",
                contact_solver=self.sim_world.config.physx_solver,
                capture_advances_physics=False,
                asset_dependencies=self.scene.asset_dependencies,
            )
            self.state["result"] = result
            (output / "result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            self.log.info(
                "Banana cutting result", success=result["success"], cuts=result["cuts"], error=error
            )
        if not result["success"]:
            raise RuntimeError(f"Banana acceptance failed: {result['failed_checks']}")

    def _export_replay(self, output: Path) -> None:
        """Bake visibility and actual poses into a replay with all rigid integration disabled."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        fps = self.config.recorder.fps
        layer = self.sim_world.stage.Flatten()
        replay = Usd.Stage.Open(layer)
        replay.SetTimeCodesPerSecond(fps)
        replay.SetStartTimeCode(0)
        replay.SetEndTimeCode(self.planner.config.duration_s * fps - 1)
        for prim in replay.Traverse():
            body = UsdPhysics.RigidBodyAPI(prim)
            if body:
                body.CreateRigidBodyEnabledAttr(False)
        for path in self.scene.meshes:
            vis = UsdGeom.Imageable(replay.GetPrimAtPath(path)).CreateVisibilityAttr()
            vis.Clear()
            vis.Set("inherited" if path.endswith("Banana_0") else "invisible", 0)
        for event in self.scene.events:
            t = event["time_s"] * fps
            UsdGeom.Imageable(replay.GetPrimAtPath(event["old_path"])).GetVisibilityAttr().Set(
                "invisible", t
            )
            for name in ("fragment_path", "remaining_path"):
                UsdGeom.Imageable(replay.GetPrimAtPath(event[name])).GetVisibilityAttr().Set(
                    "inherited", t
                )
        knife = replay.GetPrimAtPath("/World/Knife").GetAttribute("xformOp:transform")
        transforms = {}
        for path in self.scene.fragments:
            xform = UsdGeom.Xformable(replay.GetPrimAtPath(path))
            xform.ClearXformOpOrder()
            transforms[path] = xform.AddTransformOp().GetAttr()
        for row in self.planner.rows:
            t = row["time_s"] * fps
            x, y, z, yaw = row["knife"]
            matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)))
            matrix.SetTranslateOnly(Gf.Vec3d(x, y, z))
            knife.Set(matrix, t)
            for piece in row["pieces"]:
                transforms[piece["path"]].Set(Gf.Matrix4d(piece["transform"]), t)
        layer.Export(str(output / "banana_cutting_replay.usdc"))
