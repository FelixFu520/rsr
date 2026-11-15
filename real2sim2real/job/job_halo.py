"""HALO task Job: G1 crawls / crouch-walks under a desk or walks to the sofa and sits, in PhysX.

Migrated from the HALO project (crawl_v3, crouch_v3, sit_v1). The fixed BaseJob lifecycle is
kept; task logic lives in the hooks and ``simulate``:

- ``__init__``: cross-component checks (PhysX, 50 Hz control on 500 Hz physics, episode needs a
  desk or a seat), refuse an existing dataset.
- ``before_place_robot_hook``: the robot starts in the reference's first pose.
- ``before_start_simulation_hook``: physics view, start state, SONIC reset, contact reports,
  locked exposure.
- ``simulate``: 1 s unrecorded warm-up, then one SONIC action per 20 ms (10 PhysX steps).
  Images are rendered only at the control frame nearest to each image time ``i / fps``
  (≤ 10 ms quantization, measured time recorded); rendering never advances physics.
  Afterwards the full 50 Hz recording is checked (HALO validate_physx.py) and exported as
  LeRobot v3 with the same extra columns as HALO.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from real2sim2real.core.config import BaseConfig
from real2sim2real.job.base import BaseJob
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.cameras import validate_camera_list
from real2sim2real.utils.halo_utils import (
    CHAIR,
    CONTACT_CATEGORIES,
    CONTROL_HZ,
    ContactAccumulator,
    FollowCameraFilter,
    RobotGeometry,
    camera_motion_checks,
    common_checks,
    desk_passage_report,
    framing_report,
    image_schedule,
    quat_wxyz_to_matrix,
    seated_checks,
    under_desk_checks,
)
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder, capture_viewport_frame

POSE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]


def halo_extra_features(landmarks: list[str]) -> dict[str, dict[str, Any]]:
    """Extra LeRobot columns, identical to the HALO export."""
    impulses = [f"{name}_impulse_Ns" for name in CONTACT_CATEGORIES]
    return {
        "observation.contacts": {
            "dtype": "float32", "shape": [7], "names": [*impulses, "max_penetration_m"]
        },
        "observation.camera_head_pose": {"dtype": "float32", "shape": [7], "names": POSE_NAMES},
        "observation.camera_rear_pose": {"dtype": "float32", "shape": [7], "names": POSE_NAMES},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        "next.success": {"dtype": "bool", "shape": [1], "names": None},
        "observation.contact_vertical_impulses": {
            "dtype": "float32", "shape": [6], "names": [f"{n}_Ns" for n in CONTACT_CATEGORIES]
        },
        "observation.phase_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.body_landmarks": {
            "dtype": "float32",
            "shape": [3 * len(landmarks)],
            "names": [f"{n}.{a}" for n in landmarks for a in "xyz"],
        },
        "observation.foot_support_impulses": {
            "dtype": "float32", "shape": [3],
            "names": ["left_foot_Ns", "right_foot_Ns", "nonfoot_ground_Ns"],
        },
        "observation.torso_pose": {"dtype": "float32", "shape": [7], "names": POSE_NAMES},
        "observation.measured_timestamp": {"dtype": "float32", "shape": [1], "names": ["seconds"]},
        "observation.control_frame_index": {
            "dtype": "int64", "shape": [1], "names": ["50Hz_control_step"]
        },
    }  # fmt: skip


class HaloRecorderConfig(BaseModel):
    """LeRobot v3 image rate; rows are the control frames nearest to ``i / fps``."""

    model_config = ConfigDict(extra="forbid")
    format: Literal["lerobot_v3"] = "lerobot_v3"
    fps: int = Field(default=15, gt=0, le=CONTROL_HZ)
    save_failed_episodes: bool = False


class HaloCenteringConfig(BaseModel):
    """Trunk centring under the desk (crawl_v3)."""

    model_config = ConfigDict(extra="forbid")
    neck_body: str = "torso_link"
    neck_local_point_m: tuple[float, float, float] = (0.004, 0.0, 0.315)
    maximum_median_bias_m: float = 0.015
    maximum_p95_offset_m: float = 0.04
    maximum_instant_offset_m: float = 0.05


class HaloSeatAcceptanceConfig(BaseModel):
    """Seated finish limits (sit_v1)."""

    model_config = ConfigDict(extra="forbid")
    max_horizontal_center_error_m: float = 0.10
    max_lateral_center_error_m: float = 0.055
    facing_axis_xy: tuple[float, float] = (-1.0, 0.0)  # television direction (world −X)


class HaloAcceptanceConfig(BaseModel):
    """Which HALO checks apply and their limits."""

    model_config = ConfigDict(extra="forbid")
    episode: Literal["crawl", "crouch", "sit"]
    under_desk_phase: str = Field(default="", description="reference phase under the desk")
    head_collider: str = "torso_link/collisions/head_link/mesh"
    max_penetration_m: float = Field(default=0.005, gt=0)
    min_static_home_colliders: int = Field(default=100, ge=0)
    centering: HaloCenteringConfig | None = None
    seat: HaloSeatAcceptanceConfig | None = None


class HaloJobConfig(BaseConfig):
    """Task text, output, limits, recorder and acceptance."""

    backend: str = "halo"
    task: str = ""
    output_dir: str = "outputs/halo/crawl"
    overwrite_dataset: bool = Field(
        default=False, description="Delete an existing lerobot_v3 first (debug output only)"
    )
    timeout_s: float = Field(default=14400.0, gt=0)
    inspect_gui: bool = False
    max_control_frames: int | None = Field(
        default=None, gt=0, description="Stop early for smoke tests; the episode then fails"
    )
    lock_exposure: bool = True
    stall_dump_s: float = Field(
        default=120.0,
        ge=0,
        description="Write all Python thread stacks to diagnostics/<run>/stall_tracebacks.txt "
        "when one control frame takes longer than this (0 disables); diagnostics only",
    )
    startup_app_updates: int = Field(
        default=1,
        ge=0,
        description="Kit updates with the timeline playing before the start state is written. "
        "HALO ran one (its physics steps move the robot left at the asset origin); the start "
        "state then overwrites pose, joints and velocities. Reproduces HALO trajectories.",
    )
    recorder: HaloRecorderConfig = Field(default_factory=HaloRecorderConfig)
    acceptance: HaloAcceptanceConfig


class _PhysxContactFeed:
    """Forward PhysX contact reports of the robot to a ``ContactAccumulator``."""

    def __init__(self, accumulator: ContactAccumulator) -> None:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        self.accumulator = accumulator
        self._to_path = PhysicsSchemaTools.intToSdfPath
        self._robot = accumulator.prefixes.robot
        interface = get_physx_simulation_interface()
        self.subscription = interface.subscribe_contact_report_events(self._callback)

    def _callback(self, headers: Any, data: Any) -> None:
        for header in headers:
            p0, p1 = str(self._to_path(header.actor0)), str(self._to_path(header.actor1))
            if not (p0.startswith(self._robot) or p1.startswith(self._robot)):
                continue
            start = header.contact_data_offset
            for j in range(start, start + header.num_contact_data):
                self.accumulator.add(p0, p1, data[j].impulse, data[j].separation)

    def close(self) -> None:
        self.subscription = None


def _world_pose(stage: Any, path: str) -> np.ndarray:
    """``[x, y, z, qw, qx, qy, qz]`` of a prim from the current USD transforms."""
    from pxr import UsdGeom

    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    rotation = matrix.ExtractRotationQuat().GetNormalized()
    return np.r_[
        list(matrix.ExtractTranslation()), rotation.GetReal(), list(rotation.GetImaginary())
    ]


class HaloJob(BaseJob):
    """Run one HALO G1 episode and export it as LeRobot v3 after the physical checks pass."""

    config_cls = HaloJobConfig

    def __init__(self, config: HaloJobConfig | None = None) -> None:
        super().__init__(config=config)
        world = self.sim_world.config
        if self.sim_world.physics_engine != "physx":
            raise ValueError("halo uses PhysX; set sim_world.physics_engine='physx'")
        substeps = 1 / (world.physics_dt * CONTROL_HZ)
        if not math.isclose(substeps, round(substeps), abs_tol=1e-8, rel_tol=0):
            raise ValueError("physics_dt must divide the 20 ms SONIC control period")
        self.substeps = round(substeps)
        accept = self.config.acceptance
        scene, planner = self.scene.config, self.planner
        if accept.episode in {"crawl", "crouch"}:
            if scene.desk is None or accept.under_desk_phase not in planner.phase_names:
                raise ValueError("crawl/crouch need scene.desk and a reference under_desk_phase")
            if planner.phase_names.index(accept.under_desk_phase) + 1 >= len(planner.phases):
                raise ValueError("under_desk_phase must be followed by the rise phase")
        if accept.episode == "crawl" and accept.centering is None:
            raise ValueError("crawl needs job.acceptance.centering")
        if accept.episode == "sit" and (scene.seat is None or accept.seat is None):
            raise ValueError("sit needs scene.seat and job.acceptance.seat")
        if planner.config.seated_hold.enabled:
            if scene.seat is None:
                raise ValueError("planner.seated_hold needs scene.seat")
            planner.seat_center_xy = np.asarray(scene.seat.cushion_center_xyz_m[:2], dtype=float)
        self.cameras = validate_camera_list([*self.robot.cameras, *self.scene.cameras])
        self.output = Path(self.config.output_dir).expanduser().resolve()
        dataset = self.output / "lerobot_v3"
        if dataset.exists():
            if not self.config.overwrite_dataset:
                raise FileExistsError(
                    f"Dataset already exists in {self.output}; choose job.output_dir"
                )
            shutil.rmtree(dataset)
        self.contacts: ContactAccumulator | None = None
        self._feed: _PhysxContactFeed | None = None
        self._stall_file: Any = None

    # --- lifecycle hooks ------------------------------------------------------------------
    def before_place_robot_hook(self) -> None:
        """The G1 starts in the first reference pose."""
        self.robot.set_initial_qpos(self.planner.initial_qpos)

    def before_start_simulation_hook(self) -> None:
        """Build the physics view, write the start state, subscribe to contacts, fix exposure."""
        self.sim_world.reset()
        self.sim_world.play()
        for _ in range(self.config.startup_app_updates):
            self.sim_world.app.update()  # physics may step here; not counted in physics_steps
        self.robot.initialize_runtime()
        self.planner.reset()
        prefixes = self.scene.contact_prefixes(self.robot.prefix)
        self.contacts = ContactAccumulator(prefixes)
        self._feed = _PhysxContactFeed(self.contacts)
        if self.config.lock_exposure:
            from pxr import Sdf

            self.sim_world._settings.set_bool("/rtx/post/histogram/enabled", False)
            for camera in self.cameras:
                prim = self.sim_world.stage.GetPrimAtPath(camera.prim_path)
                prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
                prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(
                    False
                )

    def setup_gui_hook(self) -> None:
        if not self.sim_world.config.headless:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = self.scene.config.third_person.prim_path

    def close(self) -> None:
        if self._feed is not None:
            self._feed.close()
            self._feed = None
        super().close()

    # --- main loop ------------------------------------------------------------------------
    def _physics_period(self) -> None:
        for _ in range(self.substeps):
            self.sim_world.step(render=False)

    def simulate(self) -> None:
        run = self.output / "diagnostics" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True)
        self._arm_stall_dump(run)
        configs = {
            name: getattr(self, name).config.model_dump(mode="json")
            for name in ("sim_world", "scene", "robot", "planner")
        }
        configs["job"] = self.config.model_dump(mode="json")
        (run / "configuration.json").write_text(json.dumps(configs, indent=2, ensure_ascii=False))
        geometry = RobotGeometry.from_usd(resolve_usd_path(self.robot.config.usd_path))
        if geometry.body_names != self.robot.body_names:
            raise RuntimeError("Collider geometry body order differs from the robot")
        geometry.save(run / "robot_geometry.npz")
        cameras = {camera.name: camera.resolution for camera in self.cameras}
        first = self.cameras[0].resolution
        recorder = LeRobotV3Recorder(
            self.output / "lerobot_v3",
            fps=self.config.recorder.fps,
            width=first[0],
            height=first[1],
            task=self.config.task,
            robot_type=self.robot.recording_robot_type,
            joint_names=list(self.robot.recording_joint_names),
            action_names=list(self.planner.action_names),
            cameras=cameras,
            extra_features=halo_extra_features(list(self.robot.config.landmark_bodies)),
        )
        started = time.monotonic()
        timings = dict(warmup=0.0, control=0.0, measure=0.0, capture=0.0, physics=0.0, export=0.0)
        records: dict[str, list] = {}
        result: dict[str, Any] = {}
        failure: str | None = None
        try:
            t0 = time.monotonic()
            for _ in range(self.planner.config.warmup_control_steps):
                self.planner.warmup_step(self.robot)
                self._physics_period()
            self.contacts.clear()
            self.sim_world.stage.GetRootLayer().Export(str(run / "physical_scene.usda"))
            timings["warmup"] = time.monotonic() - t0
            self._write_calibration(run)
            self.planner.begin_episode()
            frames = self.planner.control_frames
            if self.config.max_control_frames is not None:
                frames = min(frames, self.config.max_control_frames)
            schedule = image_schedule(
                self.planner.control_frames, CONTROL_HZ, self.config.recorder.fps
            )
            schedule = schedule[schedule < frames]
            self.state["image_schedule"] = schedule
            images = set(schedule.tolist())
            last_image = int(schedule[-1])
            next_progress = time.monotonic()
            for frame in range(frames):
                self._arm_stall_dump(run)
                if not self.sim_world.is_running():
                    raise RuntimeError("Isaac Sim application closed")
                if time.monotonic() - started > self.config.timeout_s:
                    raise TimeoutError(f"HALO episode exceeded timeout_s={self.config.timeout_s}")
                t0 = time.monotonic()
                self.planner.seat_vertical_impulse_ns = float(
                    self.contacts.vertical_impulses[CHAIR]
                )
                self.planner.on_physics_step(self.robot)
                timings["control"] += time.monotonic() - t0
                t0 = time.monotonic()
                row = self._measure(frame)
                for key, value in row.items():
                    records.setdefault(key, []).append(value)
                timings["measure"] += time.monotonic() - t0
                if frame in images:
                    t0 = time.monotonic()
                    self._capture(recorder)
                    recorder.add_frame(
                        self._state_vector(row),
                        row["action"],
                        self._extras(row, frame == last_image),
                    )
                    timings["capture"] += time.monotonic() - t0
                self.contacts.end_period(frame / CONTROL_HZ)
                if frame == frames - 1:
                    break
                t0 = time.monotonic()
                self._physics_period()
                self.state["last_physics_period_s"] = time.monotonic() - t0
                timings["physics"] += self.state["last_physics_period_s"]
                if time.monotonic() >= next_progress:
                    self.log.info(
                        "HALO 进度",
                        frame=frame,
                        frames=frames,
                        phase=self.planner.state["phase"],
                        images=len(recorder.rows),
                        base=np.round(row["qpos"][:3], 3).tolist(),
                        penetration_m=round(self.contacts.max_penetration, 5),
                        contact_points=row["contact_points"],
                        physics_period_s=round(row["physics_period_s"], 3),
                        wall_s=round(time.monotonic() - started, 1),
                    )
                    next_progress = time.monotonic() + 30.0
            recording = self._save_recording(run, records)
            result = self._evaluate(recording, geometry, len(recorder.rows))
            if not result["success"] and not self.config.recorder.save_failed_episodes:
                raise RuntimeError(f"HALO episode failed: {result['failed_checks']}")
            recorder.update_extra(len(recorder.rows) - 1, "next.success", result["success"])
            t0 = time.monotonic()
            dataset = recorder.finalize(episode_success=result["success"])
            timings["export"] = time.monotonic() - t0
            result["dataset"] = str(dataset)
            result["dataset_use"] = "demonstration" if result["success"] else "diagnostic"
            self._archive_physics(run, dataset, result, configs)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            recorder.discard(run)
            if records and not (run / "control_recording.npz").exists():
                self._save_recording(run, records)
            raise
        finally:
            self._disarm_stall_dump()
            self._write_result(run, result, failure, recorder, started, timings)

    def _arm_stall_dump(self, run: Path) -> None:
        """Re-arm the faulthandler watchdog; a stalled frame dumps every thread's stack."""
        if not self.config.stall_dump_s:
            return
        import faulthandler

        if self._stall_file is None:
            self._stall_file = (run / "stall_tracebacks.txt").open("a")
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(
            self.config.stall_dump_s, repeat=True, file=self._stall_file
        )

    def _disarm_stall_dump(self) -> None:
        if self._stall_file is not None:
            import faulthandler

            faulthandler.cancel_dump_traceback_later()
            self._stall_file.close()
            path = Path(self._stall_file.name)
            if path.stat().st_size == 0:
                path.unlink()  # no frame stalled
            self._stall_file = None

    # --- per-frame helpers -----------------------------------------------------------------
    def _measure(self, frame: int) -> dict[str, Any]:
        """Measured state of this control frame, camera poses and last period's contacts."""
        qpos, joint_vel, base_vel = self.planner.observation
        eye, look = self.scene.update_third_person(qpos, 1.0 / CONTROL_HZ)
        measured = self.robot.measure()
        stage = self.sim_world.stage
        head_camera = self.robot.cameras[0]
        head = _world_pose(stage, head_camera.prim_path)
        rear = _world_pose(stage, self.scene.config.third_person.prim_path)
        torso = measured["torso_pose"]
        mount = torso[:3] + quat_wxyz_to_matrix(torso[3:7]) @ np.asarray(head_camera.pose.position)
        head_error = float(np.max(np.abs(head[:3] - mount)))
        rear_error = float(np.max(np.abs(rear[:3] - eye)))
        self.state["max_head_mount_error_m"] = max(
            self.state.get("max_head_mount_error_m", 0.0), head_error
        )
        self.state["max_rear_pose_error_m"] = max(
            self.state.get("max_rear_pose_error_m", 0.0), rear_error
        )
        if head_error > 1e-3 or rear_error > 1e-4:
            raise RuntimeError(
                f"Camera transforms not synchronized: head {head_error:.2e} m, "
                f"rear {rear_error:.2e} m"
            )
        return {
            "qpos": qpos,
            "joint_vel": joint_vel,
            "base_vel": base_vel,
            "action": np.asarray(self.planner.state["last_action"], dtype=float),
            "contacts": self.contacts.consume(),
            "contact_vertical_impulses": self.contacts.vertical_impulses.copy(),
            "foot_support_impulses": self.contacts.foot_impulses.copy(),
            "head_pose": head,
            "rear_pose": rear,
            "rear_target": look,
            "phase_index": self.planner.state["phase_index"],
            "contact_points": self.contacts.points,
            "physics_period_s": self.state.get("last_physics_period_s", 0.0),
            **measured,
            "timestamp": frame / CONTROL_HZ,
            "control_frame": frame,
        }

    @staticmethod
    def _state_vector(row: dict[str, Any]) -> np.ndarray:
        """71-D state: 29 joint positions, 29 joint velocities, base pose (7) and twist (6)."""
        return np.r_[row["qpos"][7:], row["joint_vel"], row["qpos"][:7], row["base_vel"]]

    @staticmethod
    def _extras(row: dict[str, Any], done: bool) -> dict[str, Any]:
        return {
            "observation.contacts": row["contacts"],
            "observation.camera_head_pose": row["head_pose"],
            "observation.camera_rear_pose": row["rear_pose"],
            "next.done": done,
            "next.success": False,  # set on the last row once the episode passed
            "observation.contact_vertical_impulses": row["contact_vertical_impulses"],
            "observation.phase_index": row["phase_index"],
            "observation.body_landmarks": np.asarray(row["landmark_positions"]).reshape(-1),
            "observation.foot_support_impulses": row["foot_support_impulses"],
            "observation.torso_pose": row["torso_pose"],
            "observation.measured_timestamp": row["timestamp"],
            "observation.control_frame_index": row["control_frame"],
        }

    def _capture(self, recorder: LeRobotV3Recorder) -> None:
        """Both views of the same physics instant; capture must not step physics."""
        step = self.sim_world.physics_steps
        for camera in self.cameras:
            capture_viewport_frame(
                self.sim_world,
                recorder.frame_path(camera.name),
                width=camera.resolution[0],
                height=camera.resolution[1],
                camera_prim=camera.prim_path,
            )
        if self.sim_world.physics_steps != step:
            raise RuntimeError("Camera capture advanced physics")

    def _write_calibration(self, run: Path) -> None:
        third = self.scene.config.third_person
        cameras = {}
        for camera in self.cameras:
            width, height = camera.resolution
            fx = width * camera.focal_length / camera.horizontal_aperture
            cameras[camera.name] = {
                "prim_path": camera.prim_path,
                "width": width,
                "height": height,
                "K": [[fx, 0.0, width / 2], [0.0, fx, height / 2], [0.0, 0.0, 1.0]],
                "distortion_coefficients": [0.0] * 5,
                "focal_length_mm": camera.focal_length,
                "horizontal_aperture_mm": camera.horizontal_aperture,
                "vertical_aperture_mm": camera.horizontal_aperture * height / width,
                "clipping_range_m": list(camera.clipping_range),
                "mount": {
                    "parent": camera.pose_relative_to,
                    "translation_m": list(camera.pose.position),
                    "orientation_wxyz": list(camera.pose.orientation_wxyz),
                },
            }
        cameras[third.name]["mount"] = (
            {"type": "fixed_world", "eye_m": list(third.eye_m), "target_m": list(third.target_m)}
            if third.mode == "fixed"
            else {"type": "measured_base_follow", **third.follow.model_dump(mode="json")}
        )
        calibration = {
            "world_frame": "Z up, metres",
            "camera_pose_convention": "USD camera: +X right, +Y up, -Z forward; quaternion wxyz",
            "opencv_camera_from_usd_camera_rotation": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            "per_frame_extrinsics": "control_recording.npz head_pose / rear_pose",
            "cameras": cameras,
        }
        (run / "camera_calibration.json").write_text(json.dumps(calibration, indent=2))

    def _save_recording(self, run: Path, records: dict[str, list]) -> dict[str, np.ndarray]:
        """Full 50 Hz control recording (float32) and the image ↔ control frame mapping."""
        saved = {
            key: np.asarray(
                values,
                dtype=np.int32
                if key in {"phase_index", "control_frame", "contact_points"}
                else np.float32,
            )
            for key, values in records.items()
        }
        path = run / "control_recording.npz"
        np.savez_compressed(path, **saved)
        schedule = self.state.get("image_schedule", np.zeros(0, dtype=int))
        schedule = schedule[schedule < len(saved.get("qpos", []))]
        fps = self.config.recorder.fps
        with (run / "frame_mapping.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["image_index", "control_frame_index", "video_timestamp_s", "measured_timestamp_s"]
            )
            for i, index in enumerate(schedule):
                writer.writerow([i, int(index), f"{i / fps:.6f}", f"{index / CONTROL_HZ:.6f}"])
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    # --- acceptance and outputs ----------------------------------------------------------------
    def _evaluate(
        self, rec: dict[str, np.ndarray], geometry: RobotGeometry, image_count: int
    ) -> dict[str, Any]:
        """HALO physical, camera and task checks on the archived 50 Hz recording."""
        return evaluate_halo(
            rec,
            geometry,
            job=self.config,
            scene=self.scene.config,
            scene_report=self.scene.state["scene_report"],
            phases=self.planner.phases,
            reference_fps=self.planner.config.reference_fps,
            control_frames=self.planner.control_frames,
            joint_limits=self.robot.joint_limits,
            robot_mass_kg=self.robot.state["mass_kg"],
            landmarks=list(self.robot.config.landmark_bodies),
            image_count=image_count,
            runtime={
                "max_penetration_m": self.contacts.max_penetration,
                "drive_report": self.robot.drive_report(),
                "root_pose_writes_during_capture": self.robot.state[
                    "root_pose_writes_during_capture"
                ],
                "max_head_mount_error_m": self.state.get("max_head_mount_error_m"),
                "max_rear_pose_error_m": self.state.get("max_rear_pose_error_m"),
                "seated_hold_enabled": self.planner.config.seated_hold.enabled,
                "seated_hold_start_s": self.planner.hold_start_s,
                "joint_limits_rad": np.asarray(self.robot.joint_limits).tolist(),
                "robot_mass_kg": self.robot.state["mass_kg"],
            },
        )

    def _archive_physics(self, run: Path, dataset: Path, result: dict, configs: dict) -> None:
        """Keep the full control recording and evidence next to the dataset (HALO layout)."""
        (run / "acceptance.json").write_text(json.dumps(result, indent=2, default=str))
        target = dataset / "meta" / "physics" / "episode_000000"
        target.mkdir(parents=True)
        for name in (
            "control_recording.npz",
            "frame_mapping.csv",
            "robot_geometry.npz",
            "camera_calibration.json",
            "configuration.json",
            "acceptance.json",
        ):
            shutil.copy2(run / name, target / name)
        outcome = {"result": result, "configuration": configs}
        (dataset / "meta" / "task_outcome.json").write_text(
            json.dumps(outcome, indent=2, ensure_ascii=False, default=str)
        )

    def _write_result(
        self,
        run: Path,
        result: dict,
        failure: str | None,
        recorder: LeRobotV3Recorder,
        started: float,
        timings: dict,
    ) -> None:
        result.update(
            error=failure,
            success=bool(result.get("success") and failure is None),
            completed=bool(result) and failure is None,
            episode=self.config.acceptance.episode,
            task=self.config.task,
            images=len(recorder.rows),
            fps=self.config.recorder.fps,
            physics_steps=self.sim_world.physics_steps if self.sim_world else None,
            physics_dt=self.sim_world.config.physics_dt if self.sim_world else None,
            control_hz=CONTROL_HZ,
            elapsed_s=round(time.monotonic() - started, 1),
            timings_s={k: round(v, 2) for k, v in timings.items()},
            physics_engine="physx",
            controller="GEAR-SONIC encoder/decoder (onnxruntime CPU)",
            planner=self.planner.result(),
        )
        text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        (run / "result.json").write_text(text + "\n")
        (self.output / "result.json").write_text(text + "\n")
        with (run / "trajectory.jsonl").open("w") as stream:
            for row in self.planner.rows:
                stream.write(json.dumps(row) + "\n")
        if self.contacts is not None:
            (run / "contacts.json").write_text(json.dumps(self.contacts.events))
            (run / "actor_impulses.json").write_text(
                json.dumps(self.contacts.actor_impulses, indent=2)
            )
        self.log.info("HALO 结果", success=result["success"], failed=result.get("failed_checks"))


def evaluate_halo(
    rec: dict[str, np.ndarray],
    geometry: RobotGeometry,
    *,
    job: HaloJobConfig,
    scene: Any,
    scene_report: dict,
    phases: list[dict],
    reference_fps: float,
    control_frames: int,
    joint_limits: np.ndarray,
    robot_mass_kg: float,
    landmarks: list[str],
    image_count: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """All HALO checks for one episode; also used by ``scripts/halo/validate_halo.py``."""
    from real2sim2real.utils.halo_utils import DeskSpec

    accept = job.acceptance
    checks, metrics = common_checks(
        rec,
        expected_frames=control_frames,
        joint_limits_rad=np.asarray(joint_limits),
        max_penetration_m=runtime["max_penetration_m"],
        penetration_limit_m=accept.max_penetration_m,
    )
    home = scene_report["home"]
    checks["home_static_colliders"] = (
        home["static_colliders"] >= accept.min_static_home_colliders
        and home["floor_lamp_colliders"] > 0
    )
    checks["flat_home_floor_replaced_by_proxy"] = "floor_living_room" in home["disabled_meshes"]
    drives = runtime["drive_report"]
    checks["force_limited_body_motors"] = bool(
        drives["force_limited_body_motors"] and drives["body_motor_drives"] == 29
    )
    checks["all_native_motors_recorded"] = (
        rec["all_native_joint_pos"].shape[1] == drives["native_joints"] == 43
    )
    checks["free_base_no_state_writes"] = runtime["root_pose_writes_during_capture"] == 0
    checks["head_camera_rigid_on_torso"] = (runtime["max_head_mount_error_m"] or 0.0) < 1e-3
    expected_images = len(image_schedule(control_frames, CONTROL_HZ, job.recorder.fps))
    checks["all_images_captured"] = image_count == expected_images
    schedule = image_schedule(control_frames, CONTROL_HZ, job.recorder.fps)
    quantization = np.abs(np.arange(len(schedule)) / job.recorder.fps - schedule / CONTROL_HZ)
    metrics["maximum_capture_time_quantization_s"] = float(quantization.max())
    checks["capture_time_quantization_below_10ms"] = (
        metrics["maximum_capture_time_quantization_s"] <= 0.010002
    )
    third = scene.third_person
    seat_bounds = (
        np.asarray(scene_report["seat"]["world_bounds_m"]) if accept.episode == "sit" else None
    )
    framing = framing_report(
        rec,
        geometry,
        resolution=(third.width, third.height),
        focal_length=third.focal_length,
        horizontal_aperture=third.horizontal_aperture,
        landmark_names=landmarks,
        extra_box=seat_bounds,
    )
    checks["full_robot_in_third_person_frame"] = framing["passed"]
    follow = FollowCameraFilter(third.follow) if third.mode == "follow" else None
    camera_checks, camera_metrics = camera_motion_checks(rec, mode=third.mode, follow=follow)
    checks.update(camera_checks)
    metrics.update(camera_metrics)
    report: dict[str, Any] = {"camera_framing": framing}
    if accept.episode in {"crawl", "crouch"}:
        desk = DeskSpec(
            xy=scene.desk.xy,
            yaw_deg=scene.desk.yaw_deg,
            base_z=scene.desk.base_z_m,
            depth_m=scene.desk.depth_m,
            width_m=scene.desk.width_m,
            clearance_z_m=scene.desk.clearance_z_m,
            column_spacing_m=scene.desk.column_spacing_m,
        )
        names = [p["name"] for p in phases]
        under = names.index(accept.under_desk_phase)
        transition = phases[under + 1]["start"] / reference_fps
        centering = accept.centering.model_dump() if accept.episode == "crawl" else None
        passage = desk_passage_report(
            rec,
            geometry,
            desk,
            mode=accept.episode,
            transition_time_s=transition,
            centering=centering,
        )
        report["desk_passage"] = passage
        checks["narrow_table_passage_clear"] = passage["table_passage"]["passed"]
        checks["prompt_safe_rise_just_beyond_table"] = passage["passage_timing"]["passed"]
        if centering is not None:
            checks["trunk_passes_centre_between_legs"] = passage["passage_centering"]["passed"]
        desk_checks, desk_metrics = under_desk_checks(
            rec,
            geometry,
            desk,
            mode=accept.episode,
            under_phase=under,
            head_collider=accept.head_collider,
        )
        checks.update(desk_checks)
        metrics.update(desk_metrics)
        checks["robot_avoids_home_obstacles"] = metrics["home_obstacle_contact_impulse_Ns"] < 0.05
    else:
        seat = scene_report["seat"]
        checks["original_seat_visible_collidable_unchanged"] = bool(
            seat["visible"]
            and seat["collidable"]
            and seat["mesh_unchanged"]
            and seat["bounds_error_m"] < 1e-4
        )
        seat_checks, seat_metrics = seated_checks(
            rec,
            cushion_center_xyz_m=scene.seat.cushion_center_xyz_m,
            max_horizontal_center_error_m=accept.seat.max_horizontal_center_error_m,
            max_lateral_center_error_m=accept.seat.max_lateral_center_error_m,
            robot_mass_kg=robot_mass_kg,
            facing_axis_xy=accept.seat.facing_axis_xy,
        )
        checks.update(seat_checks)
        metrics.update(seat_metrics)
        if runtime["seated_hold_enabled"]:
            checks["seated_hold_triggered_by_contact"] = runtime["seated_hold_start_s"] is not None
            metrics["seated_hold_start_s"] = runtime["seated_hold_start_s"]
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "success": all(checks.values()),
        "checks": checks,
        "failed_checks": [key for key, ok in checks.items() if not ok],
        "metrics": metrics,
        "reports": report,
        "scene_report": scene_report,
        "runtime": runtime,
        "scope": "Measured PhysX states with configured rigid colliders and contact reports; "
        "penetration from PhysX separation. No claim of real-robot validation.",
    }
