"""Tabletop pick-and-place Job (PhysX): Franka picks the cube and places it on the table.

Migrated from Real2Sim2Real ``TabletopJob`` (``configs/tabletop-pickplace_franka_curobo_home000``):
same home background, table, cube + apple, overhead + wrist SG5 cameras, two episodes with a
uniform +-0.12 m XY jitter of the cube (seed 42). The grasp is pure PhysX friction; there is no
attachment. Unlike the source (success = "sequence finished after closing"), every episode
must pass physical checks before it is exported as a LeRobot v3 demonstration.

Lifecycle hooks:

- ``__init__``: PhysX only, integer physics Hz, fps divides it, refuse existing outputs.
- ``before_place_robot_hook``: hand the measured table to the robot (base on the table top).
- ``after_place_robot_hook``: load cuMotion and check the start configuration.
- ``before_start_simulation_hook``: contact reports (before any physics view exists).
- ``simulate``: per episode jitter -> reset/play -> one planner call and one physics step per
  ``physics_dt`` -> RGB at ``fps`` -> acceptance -> export ``episode_XXX/lerobot_v3``.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from real2sim2real.core.config import BaseConfig
from real2sim2real.job.base import BaseJob
from real2sim2real.utils.cameras import validate_camera_list
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder, capture_viewport_frame
from real2sim2real.utils.tabletop_pickplace_utils import (
    ContactMonitor,
    PickPlaceAcceptanceConfig,
    evaluate_pickplace,
    jitter_offset,
)

EXTRA_FEATURES = {
    "observation.object_pose": {
        "dtype": "float32", "shape": [7], "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]
    },
    "observation.ee_pose": {
        "dtype": "float32", "shape": [7], "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]
    },
    "observation.phase_index": {"dtype": "int64", "shape": [1], "names": None},
    "next.done": {"dtype": "bool", "shape": [1], "names": None},
    "next.success": {"dtype": "bool", "shape": [1], "names": None},
}  # fmt: skip


class TabletopRecorderConfig(BaseModel):
    """LeRobot v3 export; images at ``fps`` (must divide the physics rate)."""

    model_config = ConfigDict(extra="forbid")
    fps: int = Field(default=30, gt=0)
    record_rgb: bool = Field(default=True, description="false: physics only, no dataset")
    save_failed_episodes: bool = Field(default=False, description="export failures (diagnostic)")
    warmup_captures: int = Field(default=3, ge=0, description="unrecorded renders per camera")
    capture_mode: Literal["viewport", "parallel_cameras"] = Field(
        default="viewport",
        description="viewport: switch the active viewport per camera; parallel_cameras: one fixed "
        "render product per camera (battery_disassembly capture; plain USD cameras only)",
    )


class TabletopPickPlaceJobConfig(BaseConfig):
    """``[job]`` of the tabletop pick-and-place task."""

    backend: str = "tabletop_pickplace"
    task: str = "pick the cube and place it on the table"
    output_dir: str = "outputs/tabletop_pickplace/franka_physx_home000"
    overwrite_dataset: bool = Field(default=False, description="debug output only")
    failed_dir: str = Field(
        default="outputs/debug/tabletop_pickplace",
        description="a failed run without save_failed_episodes is moved here",
    )
    num_episodes: int = Field(default=2, gt=0)
    object_jitter_xy_m: FiniteFloat = Field(default=0.12, ge=0, description="half side of square")
    object_jitter_seed: int = Field(default=42, ge=0, description="0: unseeded")
    timeout_s: FiniteFloat = Field(default=7200.0, gt=0, description="wall clock, whole run")
    max_physics_steps: int | None = Field(default=None, gt=0, description="smoke tests")
    inspect_gui: bool = False
    viewport_camera: str = Field(default="", description="GUI viewport camera prim (optional)")
    recorder: TabletopRecorderConfig = Field(default_factory=TabletopRecorderConfig)
    acceptance: PickPlaceAcceptanceConfig = Field(default_factory=PickPlaceAcceptanceConfig)


class TabletopPickPlaceJob(BaseJob):
    """``num_episodes`` physically simulated pick-and-place episodes in one Kit process."""

    config_cls = TabletopPickPlaceJobConfig

    def __init__(self, config: TabletopPickPlaceJobConfig | None = None) -> None:
        super().__init__(config=config)
        world = self.sim_world.config
        if world.physics_engine != "physx":
            raise ValueError("tabletop tasks use PhysX; set sim_world.physics_engine='physx'")
        stride = (1.0 / world.physics_dt) / self.config.recorder.fps
        if not math.isclose(stride, round(stride), abs_tol=1e-8, rel_tol=0):
            raise ValueError("recorder.fps must divide the physics rate")
        self.image_stride = round(stride)
        self.cameras = validate_camera_list([*self.scene.cameras, *self.robot.cameras])
        if self.config.recorder.record_rgb and not self.cameras:
            raise ValueError("record_rgb needs at least one camera")
        if self.config.recorder.capture_mode == "parallel_cameras" and any(
            getattr(c, "usd_path", "") for c in self.cameras
        ):
            # Measured 2026-11-14: render products on the SG5 sensor Camera prim never advance.
            raise ValueError("parallel_cameras needs plain USD cameras (no sensor usd_path)")
        self.output = Path(self.config.output_dir).expanduser().resolve()
        if (
            self.config.overwrite_dataset
            and Path("outputs/debug").resolve() not in self.output.parents
        ):
            raise ValueError("overwrite_dataset is only allowed below outputs/debug/")
        if self.output.exists() and any(self.output.iterdir()):
            if not self.config.overwrite_dataset:
                raise FileExistsError(
                    f"Output exists: {self.output}; choose another job.output_dir"
                )
            shutil.rmtree(self.output)
        self.contacts: ContactMonitor | None = None
        self.episodes: list[dict[str, Any]] = []
        self._camera_capture: Any = None

    # --- hooks ------------------------------------------------------------------------------------
    def before_place_robot_hook(self) -> None:
        """The base stands on the table measured by the scene."""
        layout, table = self.scene.layout, self.scene.config.table
        self.robot.set_support(
            layout["table_top_z"], layout["table_bounds_m"], table.position_m, table.yaw_deg
        )

    def after_place_robot_hook(self) -> None:
        self.planner.build(self.scene, self.robot, self.sim_world.config.physics_dt)

    def before_start_simulation_hook(self) -> None:
        def classify(path: str) -> str | None:
            return self.robot.contact_category(path) or self.scene.contact_category(path)

        bodies = [*self.robot.contact_bodies(), *self.scene.contact_bodies()]
        self.contacts = ContactMonitor(self.sim_world.stage, bodies, classify)

    def setup_gui_hook(self) -> None:
        if not self.sim_world.config.headless and self.config.viewport_camera:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = self.config.viewport_camera

    def close(self) -> None:
        try:
            if self._camera_capture is not None:
                self._camera_capture.close()
                self._camera_capture = None
        finally:
            try:
                if self.contacts is not None:
                    self.contacts.close()
                    self.contacts = None
            finally:
                super().close()

    # --- run --------------------------------------------------------------------------------------
    def simulate(self) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_dir = self.output / "diagnostics" / stamp
        self.run_dir.mkdir(parents=True)
        self.configuration = {
            name: getattr(self, name).config.model_dump(mode="json")
            for name in ("sim_world", "scene", "robot", "planner")
        }
        self.configuration["job"] = self.config.model_dump(mode="json")
        (self.run_dir / "configuration.json").write_text(
            json.dumps(self.configuration, indent=2, ensure_ascii=False)
        )
        self.started = time.monotonic()
        failure: str | None = None
        self.planned: list[dict[str, Any]] = []
        try:
            self.planned = self.episode_plans()
            for index, episode in enumerate(self.planned):
                self.episodes.append(self._run_episode(index, episode))
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._write_run_result(failure)

    def episode_plans(self) -> list[dict[str, Any]]:
        """Per-episode layout: target XY offset (source jitter, rejected when too close)."""
        seed = self.config.object_jitter_seed
        rng = np.random.default_rng(None if seed == 0 else seed)
        plans = []
        for _ in range(self.config.num_episodes):
            offset, draws = jitter_offset(
                rng, self.config.object_jitter_xy_m, self.scene.accept_offset
            )
            plans.append({"target_offset_m": offset.tolist(), "jitter_draws": draws})
        return plans

    def begin_controller(self, index: int, episode: dict[str, Any]) -> None:
        """Start the planner of one episode (the replay task overrides this)."""
        del index
        place_xy = self.scene.place_target_xy(self.planner.config.place_offset_m)
        episode["place_target_xy"] = place_xy.tolist()
        self.planner.begin_episode(self.scene, self.robot, place_xy)

    def _run_episode(self, index: int, episode: dict[str, Any]) -> dict[str, Any]:
        world = self.sim_world
        episode_dir = self.output / f"episode_{index:03d}"
        diag = self.run_dir / f"episode_{index:03d}"
        diag.mkdir(parents=True)
        world.stop()
        start = self.scene.set_target_offset(episode["target_offset_m"])
        world.reset()
        world.play()
        self.scene.initialize_runtime()
        self.robot.initialize_runtime()
        self.contacts.reset()
        self.begin_controller(index, episode)
        controller = self.planner
        target_layout = self.scene.layout["objects"][self.scene.config.target.name]
        layout = {
            **episode,
            "episode": index,
            "seed": self.config.object_jitter_seed,
            "target_start_prim_m": start.tolist(),
            "target_start_center_m": self.scene.target_center().tolist(),
            "table_top_z": self.scene.layout["table_top_z"],
            "object_half_extents_m": target_layout["half_extents_m"],
            "gripper_open_m": self.robot.config.gripper_open_m,
            "base_pose": [v.tolist() for v in self.robot.base_pose],
            "scene": self.scene.layout,
        }
        recorder = self._recorder(episode_dir) if self.config.recorder.record_rgb else None
        if recorder is not None:
            self._warm_up_cameras(diag)
        timings = dict(control=0.0, capture=0.0, physics=0.0, export=0.0)
        log: dict[str, list] = {}
        result: dict[str, Any] = {}
        failure: str | None = None
        step = 0
        dt = world.config.physics_dt
        try:
            next_progress = time.monotonic() + 30.0
            while True:
                if not world.is_running():
                    raise RuntimeError("Isaac Sim application closed")
                if time.monotonic() - self.started > self.config.timeout_s:
                    raise TimeoutError(f"Run exceeded timeout_s={self.config.timeout_s}")
                t0 = time.monotonic()
                self.contacts.time, self.contacts.phase = step * dt, controller.state["phase"]
                done = controller.on_physics_step(self.robot)
                row = self._log_row(log, step, controller)
                timings["control"] += time.monotonic() - t0
                if recorder is not None and step % self.image_stride == 0:
                    t0 = time.monotonic()
                    self._capture(recorder)
                    recorder.add_frame(row["q"], row["action"], self._extras(row, controller))
                    timings["capture"] += time.monotonic() - t0
                limit = self.config.max_physics_steps
                if done or (limit is not None and step >= limit):
                    break
                t0 = time.monotonic()
                world.step(render=False)
                timings["physics"] += time.monotonic() - t0
                step += 1
                if time.monotonic() >= next_progress:
                    self.log.info(
                        "桌面抓放进度",
                        episode=index,
                        t=round(step * dt, 2),
                        phase=controller.state["phase"],
                        images=len(recorder.rows) if recorder else 0,
                        **{k: round(v, 1) for k, v in timings.items()},
                    )
                    next_progress = time.monotonic() + 30.0
            arrays = {key: np.asarray(value, dtype=np.float64) for key, value in log.items()}
            np.savez_compressed(diag / "state_log.npz", **arrays)
            contacts = self.contacts.summary()
            (diag / "contacts.json").write_text(json.dumps(contacts, indent=2))
            (diag / "layout.json").write_text(json.dumps(layout, indent=2))
            result = self.evaluate(arrays, layout, contacts, controller)
            (diag / "acceptance.json").write_text(json.dumps(result, indent=2, default=str))
            if recorder is not None:
                if not result["success"] and not self.config.recorder.save_failed_episodes:
                    raise RuntimeError(f"Episode {index} failed: {result['failed_checks']}")
                recorder.update_extra(len(recorder.rows) - 1, "next.success", result["success"])
                t0 = time.monotonic()
                dataset = recorder.finalize(episode_success=result["success"])
                timings["export"] = time.monotonic() - t0
                result["dataset"] = str(dataset)
                result["dataset_use"] = "demonstration" if result["success"] else "diagnostic"
                physics = dataset / "meta" / "physics" / "episode_000000"
                physics.mkdir(parents=True)
                for name in ("state_log.npz", "contacts.json", "layout.json", "acceptance.json"):
                    shutil.copy2(diag / name, physics / name)
                (physics / "configuration.json").write_text(
                    json.dumps(self.configuration, indent=2, ensure_ascii=False)
                )
                (dataset / "meta" / "task_outcome.json").write_text(
                    json.dumps(
                        {"result": result, "layout": layout, "configuration": self.configuration},
                        indent=2,
                        default=str,
                    )
                )
            elif not result["success"] and not self.config.recorder.save_failed_episodes:
                raise RuntimeError(f"Episode {index} failed: {result['failed_checks']}")
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if recorder is not None and recorder.staging.exists():
                recorder.discard(diag)
            if log and not (diag / "state_log.npz").exists():
                partial = {k: np.asarray(v) for k, v in log.items()}
                np.savez_compressed(diag / "state_log.npz", **partial)
            raise
        finally:
            result = self._write_episode_result(
                episode_dir, diag, index, result, failure, recorder, timings, step, controller
            )
        return result

    # --- per-step helpers -------------------------------------------------------------------------
    def _log_row(self, log: dict[str, list], step: int, controller: Any) -> dict[str, Any]:
        q, qd = self.robot.joint_state()
        link_pos, link_quat = self.robot.link_poses()
        _, object_quat = self.scene.target_pose()
        action = controller.state["last_action"]
        row = dict(
            step=step,
            time=step * self.sim_world.config.physics_dt,
            phase_index=controller.state["phase_index"],
            q=q,
            qd=qd,
            action=np.asarray(action, float),
            object_pos=self.scene.target_center(),
            object_quat=object_quat,
            hand_pos=link_pos[0],
            hand_quat=link_quat[0],
            finger_pos=link_pos[1:].reshape(-1),
            finger_q=q[7:9],
            distractor_pos=self.scene.distractor_positions(),
        )
        for key, value in row.items():
            log.setdefault(key, []).append(value)
        return row

    def _extras(self, row: dict[str, Any], controller: Any) -> dict[str, Any]:
        return {
            "observation.object_pose": np.r_[row["object_pos"], row["object_quat"]],
            "observation.ee_pose": np.r_[row["hand_pos"], row["hand_quat"]],
            "observation.phase_index": row["phase_index"],
            "next.done": bool(controller.state.get("done")),
            "next.success": False,
        }

    def _recorder(self, episode_dir: Path) -> LeRobotV3Recorder:
        first = self.cameras[0].resolution
        return LeRobotV3Recorder(
            episode_dir / "lerobot_v3",
            fps=self.config.recorder.fps,
            width=first[0],
            height=first[1],
            task=self.config.task,
            robot_type=self.robot.recording_robot_type,
            joint_names=list(self.robot.recording_joint_names),
            action_names=list(self.robot.action_names),
            cameras={c.name: c.resolution for c in self.cameras},
            extra_features=EXTRA_FEATURES,
        )

    def _capture_prim(self, name: str) -> str:
        return {**self.scene.capture_prims, **self.robot.capture_prims}[name]

    def _warm_up_cameras(self, diag: Path) -> None:
        """Render every camera before frame 0 (the first RTX frame is darker)."""
        before = self.sim_world.physics_steps
        warm = diag / "camera_warmup"
        warm.mkdir()
        if self.config.recorder.capture_mode == "parallel_cameras" and self._camera_capture is None:
            from real2sim2real.utils.battery_disassembly_utils import BatteryCameraCapture

            products = [
                SimpleNamespace(
                    name=c.name, prim_path=self._capture_prim(c.name), resolution=c.resolution
                )
                for c in self.cameras
            ]
            self._camera_capture = BatteryCameraCapture(self.sim_world, products)
        paths = {c.name: warm / f"{c.name}.png" for c in self.cameras}
        for _ in range(self.config.recorder.warmup_captures):
            self._capture_to(paths)
        if self.sim_world.physics_steps != before:
            raise RuntimeError("Camera warm-up advanced physics")

    def _capture(self, recorder: LeRobotV3Recorder) -> None:
        self._capture_to({c.name: recorder.frame_path(c.name) for c in self.cameras})

    def _capture_to(self, paths: dict[str, Path]) -> None:
        before = self.sim_world.physics_steps
        if self._camera_capture is not None:
            self._camera_capture.capture(paths)
        else:
            for camera in self.cameras:
                capture_viewport_frame(
                    self.sim_world,
                    paths[camera.name],
                    width=camera.resolution[0],
                    height=camera.resolution[1],
                    camera_prim=self._capture_prim(camera.name),
                )
        if self.sim_world.physics_steps != before:
            raise RuntimeError("Camera capture advanced physics")

    # --- acceptance and results -------------------------------------------------------------------
    def evaluate(
        self, log: dict[str, np.ndarray], layout: dict[str, Any], contacts: dict, controller: Any
    ) -> dict[str, Any]:
        """Physical checks of one episode (the replay task adds replay fidelity)."""
        return evaluate_pickplace(
            log,
            layout=layout,
            contacts=contacts["impulses_ns"],
            acceptance=self.config.acceptance,
            sequence_completed=controller.result()["sequence_completed"],
        )

    def _write_episode_result(
        self,
        episode_dir: Path,
        diag: Path,
        index: int,
        result: dict,
        failure: str | None,
        recorder: Any,
        timings: dict,
        steps: int,
        controller: Any,
    ) -> dict[str, Any]:
        result = dict(result)
        result.update(
            episode=index,
            error=failure,
            success=bool(result.get("success") and failure is None),
            completed=bool(result.get("checks")) and failure is None,
            task=self.config.task,
            images=len(recorder.rows) if recorder is not None else 0,
            record_rgb=self.config.recorder.record_rgb,
            fps=self.config.recorder.fps,
            physics_steps=steps,
            physics_dt=self.sim_world.config.physics_dt if self.sim_world else None,
            simulation_time_s=steps * self.sim_world.config.physics_dt if self.sim_world else None,
            timings_s={k: round(v, 2) for k, v in timings.items()},
            physics_engine="physx",
            planner=controller.result(),
        )
        text = json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n"
        (diag / "result.json").write_text(text)
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "result.json").write_text(text)
        failed = result.get("failed_checks")
        self.log.info("桌面抓放结果", episode=index, success=result["success"], failed=failed)
        return result

    def _write_run_result(self, failure: str | None) -> None:
        summary = {
            "success": failure is None
            and len(self.episodes) == len(self.planned)
            and all(e["success"] for e in self.episodes),
            "error": failure,
            "task": self.config.task,
            "episodes": [
                {k: e.get(k) for k in ("episode", "success", "failed_checks", "dataset", "metrics")}
                for e in self.episodes
            ],
            "num_episodes": len(self.planned),
            "elapsed_s": round(time.monotonic() - self.started, 1),
            "diagnostics": str(self.run_dir),
        }
        text = json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n"
        (self.run_dir / "run_result.json").write_text(text)
        (self.output / "result.json").write_text(text)
        if not summary["success"] and not self.config.recorder.save_failed_episodes:
            self._move_failed_run()

    def _move_failed_run(self) -> None:
        """Keep the formal output directory for accepted runs only."""
        failed_root = Path(self.config.failed_dir).expanduser().resolve()
        if failed_root == self.output or failed_root in self.output.parents:
            return
        destination = failed_root / f"{self.output.name}_failed_{self.run_dir.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.output), str(destination))
        self.log.error("运行未通过验收，输出已移到 debug", destination=str(destination))
