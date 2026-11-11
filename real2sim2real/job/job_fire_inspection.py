"""Fire inspection Job: one-line command → patrol route → find the fire → photograph it.

The Job keeps the fixed BaseJob lifecycle:

- ``__init__``: parse the operator's sentence (keywords; no LLM needed for this one mission).
- ``build_occ``: scan an occupancy grid from PhysX collision geometry before the robot exists.
- ``after_arrange_scene_hook``: plan the patrol route on the grid and save ``map/``.
- ``before_start_simulation_hook``: rebuild the physics view with the robot, stand it up.
- ``simulate``: Flow warm-up and check, then the patrol loop with synchronized capture,
  the inspection photo, acceptance checks and LeRobot v3 export.

Per physics step: capture the observation (on recording steps), let the planner choose the
command, record (observation, command), step PhysX, then move the Flow clock to the new physics
time. Every camera of a frame therefore shows the same physics instant and the same fire state.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from real2sim2real.core.config import BaseConfig
from real2sim2real.job.base import BaseJob
from real2sim2real.utils.cameras import CameraConfig, validate_camera_list
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder, capture_viewport_frame
from real2sim2real.utils.patrol_geometry import (
    fire_pixel_score,
    project_point,
    save_patrol_map,
    scan_physx_occupancy,
)

_MISSION_WORDS = (
    "巡检", "巡逻", "巡查", "巡航", "巡视", "检查", "查看", "排查", "搜索", "寻找", "找",
    "inspect", "patrol", "search", "find", "check",
)  # fmt: skip
_FIRE_WORDS = ("火", "烟", "fire", "smoke", "flame")
_AREA_WORDS = ("仓库", "库房", "货架", "warehouse")


def parse_inspection_command(text: str) -> dict:
    """Keyword parse of the operator sentence; raise when it is not an inspection request.

    Only one mission exists (patrol the loaded warehouse, find the fire, photograph it), so a
    sentence qualifies when it contains a patrol/search verb, or mentions fire.
    """
    normalized = text.strip().lower()
    if not normalized:
        raise ValueError("巡检指令为空；示例：巡检仓库")
    verb = next((w for w in _MISSION_WORDS if w in normalized), None)
    fire = next((w for w in _FIRE_WORDS if w in normalized), None)
    if verb is None and fire is None:
        raise ValueError(f"无法识别巡检指令: {text!r}；示例：巡检仓库、巡逻仓库找到着火点并拍照")
    area = next((w for w in _AREA_WORDS if w in normalized), None)
    return {
        "text": text.strip(),
        "mission": "patrol_find_fire_photo",
        "area": "warehouse",
        "matched": {"verb": verb, "fire": fire, "area": area},
        "parser": "keywords",
    }


class FireInspectionRecorderConfig(BaseModel):
    """LeRobot v3 cadence; the task string is the operator's command."""

    model_config = ConfigDict(extra="forbid")
    format: Literal["lerobot_v3"] = "lerobot_v3"
    fps: int = Field(default=15, gt=0)
    save_failed_episodes: bool = False


class FirePhotoConfig(BaseModel):
    """Photo resolution and the flame-pixel acceptance check around the projected fire."""

    model_config = ConfigDict(extra="forbid")
    resolution: tuple[int, int] = (1280, 960)
    window_px: int = Field(default=160, ge=8)
    min_window_fraction: float = Field(default=0.01, ge=0, le=1)
    min_window_pixels: int = Field(default=50, ge=0)


class FireInspectionJobConfig(BaseConfig):
    """Command, output, loop limits and acceptance settings."""

    backend: str = "fire_inspection"
    command: str = ""
    default_command: str = "巡检仓库"
    prompt_command: bool = True
    output_dir: str = "outputs/xunjian"
    overwrite_dataset: bool = Field(
        default=False, description="Delete an existing lerobot_v3 first (debug output only)"
    )
    max_steps: int = Field(default=30000, gt=0)
    timeout_s: float = Field(default=7200.0, gt=0)
    inspect_gui: bool = False
    fire_warmup_s: float = Field(default=4.0, ge=0, description="Flow needs ~2–4 s to ignite")
    fire_check_camera: str = Field(
        default="", description="Scene camera checked for flames after warm-up; empty skips."
    )
    fire_check_min_pixels: int = Field(
        default=20,
        ge=1,
        description="Flame-coloured pixels near the fire in fire_check_camera; Flow off gives ~0.",
    )
    keep_open: bool = False
    photo: FirePhotoConfig = Field(default_factory=FirePhotoConfig)
    recorder: FireInspectionRecorderConfig = Field(default_factory=FireInspectionRecorderConfig)


class FireInspectionJob(BaseJob):
    """Patrol the warehouse with the X30, find the fire and photograph it."""

    config_cls = FireInspectionJobConfig

    def __init__(self, config: FireInspectionJobConfig | None = None) -> None:
        super().__init__(config=config)
        if self.sim_world.physics_engine != "physx":
            raise ValueError("fire_inspection uses PhysX; set sim_world.physics_engine='physx'")
        physics_dt = self.sim_world.config.physics_dt
        cadence = 1 / (physics_dt * self.config.recorder.fps)
        if not math.isclose(cadence, round(cadence), abs_tol=1e-8, rel_tol=0):
            raise ValueError("Recorder fps must divide the physics frequency exactly")
        self.record_stride = round(cadence)
        self.recording_cameras = validate_camera_list(
            [*self.scene.config.cameras, *self.robot.config.cameras]
        )
        detect = self.robot.camera_config(self.robot.config.camera_for_detection)
        pw, ph = self.config.photo.resolution
        if (
            pw % 2
            or ph % 2
            or not math.isclose(pw / ph, detect.resolution[0] / detect.resolution[1])
        ):
            raise ValueError("photo.resolution must be even and match the detection camera aspect")
        self.output = Path(self.config.output_dir).expanduser().resolve()
        dataset = self.output / "lerobot_v3"
        if dataset.exists():
            if not self.config.overwrite_dataset:
                raise FileExistsError(
                    f"Dataset already exists in {self.output}; choose job.output_dir"
                )
            shutil.rmtree(dataset)
        if not self.robot.config.beacon.enabled:
            raise ValueError("The fire alarm needs robot.beacon.enabled = true")
        self.command = parse_inspection_command(self._read_command())
        self.log.info("巡检任务", **self.command)
        self.grid = None

    def _read_command(self) -> str:
        if self.config.command:
            return self.config.command
        if self.config.prompt_command and sys.stdin.isatty():
            text = input(f"请输入巡检指令 [{self.config.default_command}]: ").strip()
            return text or self.config.default_command
        return self.config.default_command

    # --- lifecycle hooks ------------------------------------------------------------------
    def build_occ(self) -> None:
        """Scan the warehouse collision geometry into an occupancy grid (robot not yet added)."""
        planner = self.planner.config
        started = time.monotonic()
        self.sim_world.reset()
        self.grid = scan_physx_occupancy(
            planner.map_bounds_xy, planner.map_resolution_m, planner.map_z_band_m
        )
        self.sim_world.stop()
        # Drop the robot-less simulation view; otherwise the X30 tensor view created after the
        # next reset binds to it and every state read fails.
        self.sim_world.world.invalidate_physics()
        (self.output / "map").mkdir(parents=True, exist_ok=True)
        self.grid.save(self.output / "map" / "occupancy.npz")
        self.state["occupancy"] = {
            "bounds_xy": planner.map_bounds_xy,
            "resolution_m": planner.map_resolution_m,
            "z_band_m": planner.map_z_band_m,
            "occupied_cells": int(self.grid.occupied.sum()),
            "no_floor_cells": int(self.grid.no_floor.sum()),
            "scan_s": round(time.monotonic() - started, 2),
        }
        self.log.info("占用图扫描完成", **self.state["occupancy"])

    def after_arrange_scene_hook(self) -> None:
        """Plan the patrol route on the occupancy grid and save a top-down map."""
        report = self.planner.set_map(self.grid, self.scene.fire_position)
        self.state["route"] = report
        paths = {"patrol": self.planner.patrol_path} if report["ok"] else {}
        points = {f"P{i}": p for i, p in enumerate(self.planner.config.route_xy)}
        if self.planner.config.standoff_xy is not None:
            points["photo"] = self.planner.config.standoff_xy
        map_path = save_patrol_map(
            self.output / "map" / "patrol_map.png",
            self.grid,
            fire_xy=self.scene.fire_position,
            paths=paths,
            points=points,
        )
        report["map"] = str(map_path)
        if not report["ok"]:
            raise RuntimeError(
                f"Patrol route cannot be planned: {report['legs'][-1]}; see {map_path}"
            )
        self.log.info(
            "巡航路线规划完成",
            length_m=report["length_m"],
            vertices=report["vertices"],
            map=str(map_path),
        )

    def before_start_simulation_hook(self) -> None:
        """Build PhysX views with the robot, stand it up and hand the runtime to the planner.

        The CPU tensor API only reads/writes articulation state while the timeline plays, so
        the world starts playing here; no physics step runs until ``simulate``.
        """
        self.sim_world.reset()
        self.sim_world.play()
        self.robot.initialize_runtime()
        self.planner.reset()
        robot_prefix = (self.robot.config.prim_path,)
        self.planner.set_runtime(
            occluder=lambda eye: self.scene.fire_occluder(eye, robot_prefix),
            dt=self.sim_world.config.physics_dt,
        )
        render_interval = self.record_stride * self.sim_world.config.physics_dt
        if render_interval > self.scene.max_render_interval_s + 1e-9:
            raise ValueError(
                f"Frames are {render_interval:.3f} s apart but Flow simulates at most "
                f"{self.scene.max_render_interval_s:.3f} s per render; raise "
                "scene.flow_max_steps_per_update or job.recorder.fps"
            )

    def setup_gui_hook(self) -> None:
        if not self.sim_world.config.headless:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = self.recording_cameras[-1].prim_path

    # --- main loop ------------------------------------------------------------------------
    @property
    def simulation_time_s(self) -> float:
        return self.sim_world.physics_steps * self.sim_world.config.physics_dt

    def _physics_step(self) -> None:
        """One PhysX step, then move Flow's clock to the new physics time."""
        self.sim_world.step(render=False)
        error = self.robot.after_physics_step()
        if self.planner.rows:
            self.planner.rows[-1]["leg_tracking_error_rad"] = round(error, 4)
        self.scene.sync_fire_time(self.simulation_time_s)

    def simulate(self) -> None:
        run = self.output / "diagnostics" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True)
        configs = {
            name: getattr(self, name).config.model_dump(mode="json")
            for name in ("sim_world", "scene", "robot", "planner")
        }
        configs["job"] = self.config.model_dump(mode="json")
        configs["command"] = self.command
        (run / "configuration.json").write_text(json.dumps(configs, indent=2, ensure_ascii=False))
        first = self.recording_cameras[0].resolution
        recorder = LeRobotV3Recorder(
            self.output / "lerobot_v3",
            fps=self.config.recorder.fps,
            width=first[0],
            height=first[1],
            task=self.command["text"],
            robot_type=self.robot.recording_robot_type,
            joint_names=self.robot.recording_joint_names,
            action_names=self.planner.action_names,
            cameras={camera.name: camera.resolution for camera in self.recording_cameras},
        )
        started = time.monotonic()
        timings = dict(warmup=0.0, control=0.0, physics=0.0, capture=0.0, photo=0.0, export=0.0)
        result: dict = {}
        failure: str | None = None
        completed = False
        photo_pending = False
        try:
            self.scene.start_fire_clock()
            t0 = time.monotonic()
            self._warm_up_fire(run)
            timings["warmup"] = time.monotonic() - t0
            self.state["patrol_start_step"] = self.sim_world.physics_steps
            next_progress = started
            for step in range(self.config.max_steps):
                if not self.sim_world.is_running():
                    raise RuntimeError("Isaac Sim application closed")
                if time.monotonic() - started > self.config.timeout_s:
                    raise TimeoutError(
                        f"Fire inspection exceeded timeout_s={self.config.timeout_s}"
                    )
                record = step % self.record_stride == 0
                if record:  # observation at the current physics instant
                    t0 = time.monotonic()
                    self._capture_cameras(recorder)
                    observation = self.robot.recording_state()
                    timings["capture"] += time.monotonic() - t0
                t0 = time.monotonic()
                completed = self.planner.on_physics_step(self.robot)
                self.robot.set_beacon(self.planner.state["beacon_on"])
                timings["control"] += time.monotonic() - t0
                if self.planner.state["photo_request"]:
                    t0 = time.monotonic()
                    self.planner.photo_taken(self._take_photo())
                    timings["photo"] += time.monotonic() - t0
                    photo_pending = True
                if record:  # the command chosen from that observation
                    action = np.asarray(self.planner.state["last_action"], dtype=float).copy()
                    action[3] = float(photo_pending)
                    photo_pending = False
                    recorder.add_frame(observation, action)
                phase = self.planner.state["phase"]
                if phase != self.state.get("phase"):
                    self.log.info("巡检阶段", step=step, **self.planner.state["progress"])
                    self.state["phase"] = phase
                if completed:
                    break
                t0 = time.monotonic()
                self._physics_step()
                timings["physics"] += time.monotonic() - t0
                if time.monotonic() >= next_progress:
                    self.log.info(
                        "巡检进度",
                        simulation_s=round(self.simulation_time_s, 2),
                        frames=len(recorder.rows),
                        wall_s=round(time.monotonic() - started, 1),
                        **self.planner.state["progress"],
                    )
                    next_progress = time.monotonic() + 30.0
            result = self._evaluate(completed)
            if not result["success"] and not self.config.recorder.save_failed_episodes:
                raise RuntimeError(f"Fire inspection failed: {result['failed_checks']}")
            t0 = time.monotonic()
            dataset = recorder.finalize(episode_success=result["success"])
            timings["export"] = time.monotonic() - t0
            result["dataset"] = str(dataset)
            result["dataset_use"] = "demonstration" if result["success"] else "diagnostic"
            outcome = {"result": result, "configuration": configs}
            (dataset / "meta/task_outcome.json").write_text(
                json.dumps(outcome, indent=2, ensure_ascii=False, default=str)
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            recorder.discard(run)
            raise
        finally:
            self._write_result(run, result, completed, failure, recorder, started, timings)

    def _write_result(
        self,
        run: Path,
        result: dict,
        completed: bool,
        failure: str | None,
        recorder: LeRobotV3Recorder,
        started: float,
        timings: dict,
    ) -> None:
        if not result:
            result.update(self._evaluate(completed))
        result.update(
            error=failure,
            success=bool(result.get("success") and failure is None),
            command=self.command,
            frames=len(recorder.rows),
            fps=self.config.recorder.fps,
            physics_steps=self.sim_world.physics_steps,
            patrol_start_step=self.state.get("patrol_start_step"),
            simulation_time_s=round(self.simulation_time_s, 4),
            fire_time_s=round(self.scene.fire_time_s, 4),
            elapsed_s=round(time.monotonic() - started, 1),
            timings_s={k: round(v, 2) for k, v in timings.items()},
            occupancy=self.state.get("occupancy"),
            route=self.state.get("route"),
            physics_engine=self.sim_world.physics_engine,
            locomotion=self.robot.locomotion,
            fire_detection="ground_truth: camera FOV + PhysX ray to the fire point",
        )
        text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        (run / "result.json").write_text(text + "\n")
        (self.output / "result.json").write_text(text + "\n")
        with (run / "trajectory.jsonl").open("w") as stream:
            for row in self.planner.rows:
                stream.write(json.dumps(row) + "\n")
        if self.grid is not None and self.planner.approach_path is not None:
            save_patrol_map(
                self.output / "map" / "patrol_map_result.png",
                self.grid,
                fire_xy=self.scene.fire_position,
                paths={
                    "patrol": self.planner.patrol_path,
                    "driven": np.asarray([row["xy"] for row in self.planner.rows[::24]]),
                    "approach": self.planner.approach_path,
                },
                points={"detected": self.planner.detection["xy"]},
            )
        self.log.info("巡检结果", success=result["success"], failed=result["failed_checks"])
        if self.config.keep_open and not self.sim_world.config.headless:
            self.scene.stop_fire_clock()
            self.sim_world.wait_for_gui_edit(reason="inspection finished")

    def _warm_up_fire(self, run: Path) -> None:
        """Stand still while Flow ignites, rendering once per recording interval."""
        steps = round(self.config.fire_warmup_s / self.sim_world.config.physics_dt)
        xy, yaw = self.robot.base_xy_yaw()
        for step in range(steps):
            self.robot.set_base(xy, yaw, np.zeros(3))
            self._physics_step()
            if step % self.record_stride == self.record_stride - 1:
                self.sim_world.update()  # Flow simulates on render updates
        if not self.config.fire_check_camera:
            return
        cameras = {c.name: c for c in self.scene.config.cameras}
        if self.config.fire_check_camera not in cameras:
            raise ValueError("job.fire_check_camera must name a [[scene.cameras]] entry")
        camera = cameras[self.config.fire_check_camera]
        path = run / "warmup_fire_check.png"
        capture_viewport_frame(
            self.sim_world,
            path,
            width=camera.resolution[0],
            height=camera.resolution[1],
            camera_prim=camera.prim_path,
        )
        from PIL import Image
        from pxr import UsdGeom

        prim = self.sim_world.stage.GetPrimAtPath(camera.prim_path)
        matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=float)
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        score = self._score(rgb, camera, matrix, camera.resolution)
        # This check only proves Flow is running; the photo keeps the stricter fraction test.
        score["pass"] = bool(
            score["fire_in_view"] and score["window_pixels"] >= self.config.fire_check_min_pixels
        )
        score["min_pixels"] = self.config.fire_check_min_pixels
        self.state["warmup_fire_check"] = {
            "image": str(path), "fire_time_s": round(self.scene.fire_time_s, 3), **score
        }  # fmt: skip
        self.log.info("Flow 火焰预检", **self.state["warmup_fire_check"])
        if not score["pass"]:
            raise RuntimeError(f"Flow fire not visible after warm-up: {score}; see {path}")

    def _score(
        self,
        rgb: np.ndarray,
        camera: CameraConfig,
        matrix: np.ndarray,
        resolution: tuple[int, int],
    ) -> dict:
        """Flame-pixel score around the projected fire; window scales with image width."""
        projection = project_point(
            self.scene.fire_position,
            matrix,
            focal_length=camera.focal_length,
            horizontal_aperture=camera.horizontal_aperture,
            resolution=tuple(resolution),
            near=camera.clipping_range[0],
            far=camera.clipping_range[1],
        )
        settings = self.config.photo
        scale = resolution[0] / settings.resolution[0]
        window = max(8, round(settings.window_px * scale))
        if projection.in_view:
            score = fire_pixel_score(rgb, projection.u, projection.v, window)
        else:
            score = {"window_px": None, "window_fraction": 0.0, "window_pixels": 0}
        score.update(
            fire_pixel=[round(projection.u, 1), round(projection.v, 1)],
            fire_in_view=projection.in_view,
            fire_depth_m=round(projection.depth, 3),
        )
        min_pixels = settings.min_window_pixels * scale**2
        score["pass"] = bool(
            projection.in_view
            and score["window_fraction"] >= settings.min_window_fraction
            and score["window_pixels"] >= min_pixels
        )
        return score

    def _take_photo(self) -> dict:
        """Capture the inspection photo from the detection camera and check it for flames."""
        from PIL import Image

        name = self.robot.config.camera_for_detection
        camera = self.robot.camera_config(name)
        width, height = self.config.photo.resolution
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.output / "photos" / f"fire_{stamp}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        capture_viewport_frame(
            self.sim_world, path, width=width, height=height, camera_prim=camera.prim_path
        )
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        score = self._score(rgb, camera, self.robot.camera_matrix(name), (width, height))
        xy, yaw = self.robot.base_xy_yaw()
        distance = float(np.linalg.norm(self.scene.fire_position[:2] - xy))
        info = {
            "image": str(path),
            "camera": name,
            "command": self.command["text"],
            "fire_target_prim": self.scene.fire_target_prim,
            "fire_world_xyz": self.scene.fire_position.round(4).tolist(),
            "robot_xy": xy.round(3).tolist(),
            "robot_yaw_rad": round(yaw, 4),
            "distance_to_fire_m": round(distance, 3),
            "beacon_on": bool(self.robot.state.get("beacon_on_now", False)),
            "simulation_time_s": round(self.simulation_time_s, 4),
            "fire_time_s": round(self.scene.fire_time_s, 4),
            **score,
        }
        path.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
        self.log.info("已拍摄着火点照片", **info)
        return info

    def _capture_cameras(self, recorder: LeRobotV3Recorder) -> None:
        """Capture every view at the same physics instant and restore the GUI camera."""
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No active viewport for RGB capture")
        previous = viewport.camera_path, viewport.resolution
        step, fire_time = self.sim_world.physics_steps, self.scene.fire_time_s
        try:
            for camera in self.recording_cameras:
                capture_viewport_frame(
                    self.sim_world,
                    recorder.frame_path(camera.name),
                    width=camera.resolution[0],
                    height=camera.resolution[1],
                    camera_prim=camera.prim_path,
                )
            if self.sim_world.physics_steps != step or self.scene.fire_time_s != fire_time:
                raise RuntimeError("Multi-camera capture advanced physics or the Flow clock")
        finally:
            if not self.sim_world.config.headless:  # headless: keep the capture resolution
                viewport.camera_path, viewport.resolution = previous

    def _leg_contact_segments(self, threshold: float = 0.05) -> list[dict]:
        """Stretches where a leg deviated from its target (a foot touched a collider)."""
        segments: list[dict] = []
        for row in self.planner.rows:
            error = row.get("leg_tracking_error_rad", 0.0)
            if error <= threshold:
                continue
            last = segments[-1] if segments else None
            if last is not None and row["step"] - last["to_step"] <= 3:
                last.update(to_step=row["step"], to_xy=row["xy"])
                last["max_error_rad"] = max(last["max_error_rad"], error)
            else:
                segments.append(
                    {
                        "from_step": row["step"],
                        "to_step": row["step"],
                        "from_xy": row["xy"],
                        "to_xy": row["xy"],
                        "max_error_rad": error,
                    }
                )
        return segments

    def _evaluate(self, completed: bool) -> dict:
        result = self.planner.result()
        photo = result.get("photo") or {}
        gait = self.robot.config.gait
        tracking = float(self.robot.state.get("max_tracking_error_rad", math.inf))
        route = self.state.get("route") or {}
        approach = (result.get("detection") or {}).get("approach_clearance") or {}
        alarm = result.get("alarm") or {}
        cfg, dt = self.planner.config, self.sim_world.config.physics_dt
        radius = self.planner.config.robot_radius_m
        checks = {
            "episode_completed": bool(completed),
            "route_clear": bool(route.get("ok"))
            and all(leg.get("violations", 1) == 0 for leg in route.get("legs", [])),
            "fire_found": bool(result.get("fire_found")),
            "hold_completed": alarm.get("held_s", 0.0) >= cfg.detect_hold_s - dt,
            "alarm_raised": alarm.get("end_step") is not None
            and self.robot.state["beacon_switches"]
            >= 2 * math.floor((cfg.detect_hold_s + cfg.alarm_s) * cfg.alarm_flash_hz),
            "beacon_off_for_photo": photo.get("beacon_on") is False,
            "approach_clear": approach.get("min_clearance_m", 0.0) >= radius - 1e-6,
            "reached_photo_phase": result.get("phase") == "done",
            "photo_taken": bool(photo.get("image")),
            "photo_shows_fire": bool(photo.get("pass")),
            # PhysX must integrate exactly the commanded twist each step (float32 at ~60 m
            # from the origin resolves ~5 µm); a larger gap means contacts or extra steps.
            "base_follows_command": result["max_base_drift_m"] < 1e-3,
            "legs_track_gait": bool(self.robot.state.get("ready"))
            and tracking < gait.max_tracking_error_rad,
        }
        result.update(
            checks=checks,
            failed_checks=[name for name, ok in checks.items() if not ok],
            success=all(checks.values()),
            max_leg_tracking_error_rad=round(tracking, 4),
            leg_contact_segments=self._leg_contact_segments(),
            beacon_switches=self.robot.state["beacon_switches"],
            leg_fk_error_m=self.robot.state.get("leg_fk_error_m"),
            warmup_fire_check=self.state.get("warmup_fire_check"),
        )
        return result
