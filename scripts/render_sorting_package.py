"""Render synchronized LeRobot v3 observations from a real MuJoCo + VBD trajectory.

Replay changes visual transforms only. Physical acceptance, measured joints and
actions come exclusively from the saved solver trajectory; physics is not rerun.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from real2sim2real.core.config import apply_settings_file
from real2sim2real.job.job_sorting import SortingJob, SortingJobConfig
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.cameras import CameraConfig, CameraPose
from real2sim2real.utils.kit_process import run_kit_worker
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder, capture_viewport_frame


class ReplayRenderProfile(BaseModel):
    """Presentation-only settings; never accepts dynamics or robot modifications."""

    model_config = ConfigDict(extra="forbid")
    renderer: Literal["RayTracedLighting", "PathTracing"] = "PathTracing"
    resolution: tuple[Annotated[int, Field(ge=64)], Annotated[int, Field(ge=64)]] = (
        1024,
        768,
    )
    samples_per_pixel_per_frame: int = Field(default=128, ge=1)
    rt_subframes: int = Field(default=8, ge=2)
    max_bounces: int = Field(default=6, ge=1)
    denoiser: bool = True
    cached_path_tracing: bool = False
    many_light_sampling: bool = False

    @model_validator(mode="after")
    def _even_resolution(self) -> ReplayRenderProfile:
        if any(value % 2 for value in self.resolution):
            raise ValueError("H.264 render dimensions must be even")
        return self


def override_render_profile(configuration: dict, profile: ReplayRenderProfile) -> dict:
    """Change only render settings and image dimensions, preserving physical provenance."""
    updated = copy.deepcopy(configuration)
    updated["sim_world"].update(
        renderer=profile.renderer,
        width=profile.resolution[0],
        height=profile.resolution[1],
        samples_per_pixel_per_frame=profile.samples_per_pixel_per_frame,
        rt_subframes=profile.rt_subframes,
        pathtracing_max_bounces=profile.max_bounces,
        pathtracing_denoiser=profile.denoiser,
        pathtracing_temporal_denoising=False,
        pathtracing_cached=profile.cached_path_tracing,
        pathtracing_many_light_sampling=profile.many_light_sampling,
        synchronous_capture=True,
        dlss=False,
        auto_exposure=False,
    )
    for section in ("scene", "robot"):
        for camera in updated[section]["cameras"]:
            camera["resolution"] = list(profile.resolution)
    return updated


def replay_indices(times: np.ndarray, fps: int) -> np.ndarray:
    """Choose exact physics samples at each video timestamp, without interpolation."""
    if not len(times) or fps <= 0 or not np.isfinite(times).all():
        raise ValueError("Replay needs finite physical timestamps and a positive FPS")
    if times[0] != 0 or not np.all(np.diff(times) > 0):
        raise ValueError("Physical timestamps must increase strictly from zero")
    count = max(1, int(np.ceil((times[-1] - 1e-8) * fps)))
    targets = np.arange(count) / fps
    indices = np.abs(times[:, None] - targets).argmin(0)
    if not np.allclose(times[indices], targets, atol=1e-7, rtol=0):
        raise ValueError("Video FPS must coincide with saved physics samples")
    return indices


def override_camera_poses(configuration: dict, overrides: dict) -> dict:
    """Copy a physical snapshot, changing only explicitly named optical poses."""
    if not isinstance(overrides, dict):
        raise ValueError("Camera pose overrides must be an object keyed by camera name")
    updated = copy.deepcopy(configuration)
    cameras = {
        camera["name"]: camera
        for section in ("scene", "robot")
        for camera in updated[section]["cameras"]
    }
    unknown = overrides.keys() - cameras.keys()
    if unknown:
        raise ValueError(f"Unknown replay cameras: {sorted(unknown)}")
    for name, pose in overrides.items():
        cameras[name]["pose"] = CameraPose.model_validate(pose).model_dump(mode="json")
    return updated


def valid_cached_frame(path: Path, resolution: tuple[int, int]) -> bool:
    """Accept only complete PNGs with the requested image dimensions."""
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != tuple(resolution):
                return False
            image.verify()
        return True
    except (OSError, SyntaxError, ValueError):
        return False


def prepare_frame_cache(
    cache: Path, output: Path, configuration: dict, fps: int, indices: np.ndarray
) -> Path:
    """Bind resumable RGB frames to exact source files, optics and rendering settings."""
    sources = {}
    for name in ("configuration.json", "result.json", "particles.npz", "stream.npz"):
        path = cache / name
        if path.is_file():
            with path.open("rb") as source:
                sources[name] = hashlib.file_digest(source, "sha256").hexdigest()
    manifest = dict(
        version=3,
        visual_geometry_contract="native_visible_rubber_pads_v1",
        capture_sync_contract="flush_scene_before_pt_counter_v1",
        sources=sources,
        configuration=configuration,
        fps=fps,
        frames=len(indices),
        indices_sha256=hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
    )
    # Normalize tuples and JSON lists before comparing persisted settings.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    root = output / ".render_cache"
    path = root / "manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError("Render cache source or settings changed; choose a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if (root / "frames").exists():
            raise ValueError("Render cache has frames without a provenance manifest")
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    for section in ("scene", "robot"):
        for camera in configuration[section]["cameras"]:
            (root / "frames" / camera["name"]).mkdir(parents=True, exist_ok=True)
    return root / "frames"


class SortingReplayJob(SortingJob):
    """Use the standard Job lifecycle and native visuals, keeping the world paused."""

    def __init__(
        self,
        cache: Path,
        output: Path,
        fps: int,
        camera_poses: Path | None = None,
        render_profile: Path | None = None,
        capture_camera: str | None = None,
        capture_chunk_frames: int = 600,
    ) -> None:
        self.cache, self.output = cache, output
        self.diagnostic_source = False
        self.capture_camera = capture_camera
        if capture_chunk_frames < 1:
            raise ValueError("Capture chunk size must be positive")
        self.capture_chunk_frames = capture_chunk_frames
        self.configuration = json.loads((cache / "configuration.json").read_text())
        self.physical_result = json.loads((cache / "result.json").read_text())
        if (
            self.physical_result["rigid_solver"] != "mujoco"
            or self.physical_result["contact_solver"] != "vbd"
            or self.configuration["planner"]["backend"] not in {"sorting_package", "sorting_stream"}
        ):
            raise ValueError("Replay requires a native Pro-M6 MuJoCo + VBD sorting cache")
        asset = Path(resolve_usd_path(self.configuration["scene"]["cloth_usd_path"]))
        if hashlib.sha256(asset.read_bytes()).hexdigest() != self.physical_result["asset_sha256"]:
            raise ValueError("Parcel asset changed after the physical solve")
        with np.load(cache / "particles.npz") as archive:
            self.saved = {key: archive[key] for key in archive.files}
        self.is_stream = (cache / "stream.npz").is_file()
        if self.is_stream:
            with np.load(cache / "stream.npz") as archive:
                self.saved.update({key: archive[key] for key in archive.files})
        self.indices = replay_indices(self.saved["times"], fps)
        self.sync_audit_frames: set[int] = set()
        if self.is_stream:
            rows = json.loads((cache / "stream_trajectory.json").read_text())
            if len(self.indices) > 340:
                self.sync_audit_frames.add(340)
            for parcel in ("soft", "rigid"):
                for phase in ("left_transfer", "right_transfer"):
                    own = [row for row in rows if row["parcel"] == parcel and row["phase"] == phase]
                    if own:
                        self.sync_audit_frames.add(round(own[len(own) // 2]["time_s"] * fps))
        # JSON is a supported Dynaconf settings format. Preserve the normal
        # default/project/secrets/environment priority when selecting components.
        apply_settings_file(cache / "configuration.json")
        config = SortingJobConfig(output_dir=str(output))
        config.recorder.fps = fps
        config.recorder.task = (
            "Relay the garment parcel from infeed through the platform to the outfeed belt "
            "using both Pro-M6 arms, releasing at each destination."
        )
        if self.is_stream:
            config.recorder.task = (
                "Relay a soft garment parcel then a rigid carton from the remote infeed belt "
                "through the platform using both native Pro-M6 arms; drop each onto the "
                "continuously running outfeed belt."
            )
        super().__init__(config=config)
        for section in ("sim_world", "scene", "robot", "planner"):
            component = getattr(self, section)
            actual = component.config.model_dump(mode="json")
            expected = component.config_cls(**self.configuration[section]).model_dump(mode="json")
            if actual != expected:
                raise ValueError(
                    f"Replay {section} differs from its physical snapshot; check RSR overrides"
                )
        self.camera_pose_overrides = {}
        if camera_poses is not None:
            requested = json.loads(camera_poses.read_text())
            self.configuration = override_camera_poses(self.configuration, requested)
            for section in ("scene", "robot"):
                component = getattr(self, section)
                for camera, saved in zip(
                    component.config.cameras, self.configuration[section]["cameras"], strict=True
                ):
                    camera.pose = CameraPose.model_validate(saved["pose"])
                    if camera.name in requested:
                        self.camera_pose_overrides[camera.name] = saved["pose"]
        self.render_profile = (
            ReplayRenderProfile.model_validate_json(render_profile.read_text())
            if render_profile is not None
            else None
        )
        if self.render_profile is not None:
            self.configuration = override_render_profile(self.configuration, self.render_profile)
            for section in ("sim_world", "scene", "robot"):
                component = getattr(self, section)
                component.config = component.config_cls(**self.configuration[section])
            self.recording_cameras = [*self.scene.config.cameras, *self.robot.config.cameras]
            self.config.recorder.width, self.config.recorder.height = self.render_profile.resolution
        if capture_camera is not None and capture_camera not in {
            camera.name for camera in self.recording_cameras
        }:
            raise ValueError(f"Unknown capture camera: {capture_camera}")

    def _prepare_offline_render(self, first_index: int, camera: CameraConfig) -> None:
        """Load materials/lights before externally controlling progressive accumulation."""
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if self.sim_world.config.synchronous_capture:
            self.sim_world._settings.set_bool(
                "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges", False
            )
        viewport.camera_path = camera.prim_path
        viewport.resolution = camera.resolution
        self._publish_sample(first_index)
        for _ in range(16):
            self.sim_world.update()

    def _sync_audit_path(self, camera: str, frame: int) -> Path:
        return self.output / "qa/render_sync" / camera / f"frame_{frame:06d}.json"

    def _valid_frame_audit(self, camera: str, frame: int, image: Path) -> bool:
        if frame not in getattr(self, "sync_audit_frames", ()):
            return True
        try:
            report = json.loads(self._sync_audit_path(camera, frame).read_text())
            return bool(
                report["identical"]
                and report["captured_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
            )
        except (OSError, ValueError, KeyError):
            return False

    def _audit_captured_frame(self, camera: CameraConfig, frame: int, image: Path) -> None:
        """Compare the actual exported frame with a fresh accumulation in the same renderer."""
        destination = self._sync_audit_path(camera.name, frame)
        destination.parent.mkdir(parents=True, exist_ok=True)
        reference = destination.with_suffix(".png")
        capture_viewport_frame(
            self.sim_world,
            reference,
            width=camera.resolution[0],
            height=camera.resolution[1],
            camera_prim=camera.prim_path,
            camera_sequence=False,
        )
        with Image.open(image) as captured, Image.open(reference) as static:
            delta = np.abs(
                np.asarray(captured.convert("RGB"), dtype=np.int16)
                - np.asarray(static.convert("RGB"), dtype=np.int16)
            )
        report = dict(
            camera=camera.name,
            frame=frame,
            source_index=int(self.indices[frame]),
            identical=bool(not delta.any()),
            mean_absolute_pixel_error=float(delta.mean()),
            maximum_pixel_error=int(delta.max()),
            captured_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        )
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if not report["identical"]:
            raise ValueError(f"Rendered sequence/static pose check failed: {destination}")
        self.log.info("Exported RGB matches refreshed static frame", **report)

    def simulate(self) -> None:
        """Publish cached physical poses, capture RGB, and preserve the source outcome."""
        runtime, saved = self.planner.runtime, self.saved
        if saved["body_poses"].shape[1:] != runtime.state_0.body_q.shape + (7,):
            raise ValueError("Cached native robot body topology differs from the renderer")
        np.testing.assert_array_equal(saved["masses"], runtime._masses)
        if "shape_material_mu" in saved:
            np.testing.assert_array_equal(
                saved["shape_material_mu"], runtime.model.shape_material_mu.numpy()
            )
            np.testing.assert_allclose(saved["soft_contact_mu"], runtime.model.soft_contact_mu)
        if json.loads(str(saved.get("quality_reference_json", "null"))) != getattr(
            self.scene, "quality_reference", None
        ):
            raise ValueError("Cached original-sampling reference differs from the parcel asset")
        for i, jaw in enumerate(runtime.jaw_colliders):
            if jaw["body_id"] != int(saved["jaw_body_ids"][i]):
                raise ValueError("Cached finger body mapping differs from the renderer")
            np.testing.assert_allclose(jaw["vertices"], saved[f"jaw_vertices_{i}"], atol=1e-8)
            np.testing.assert_array_equal(jaw["triangles"], saved[f"jaw_triangles_{i}"])
        if "fixed_robot_shape_ids" in saved:
            np.testing.assert_array_equal(
                saved["fixed_robot_shape_ids"],
                [collider["shape_id"] for collider in runtime.fixed_robot_colliders],
            )
            for i, collider in enumerate(runtime.fixed_robot_colliders):
                np.testing.assert_allclose(
                    collider["vertices"], saved[f"fixed_robot_vertices_{i}"], atol=1e-8
                )
                np.testing.assert_array_equal(
                    collider["triangles"], saved[f"fixed_robot_triangles_{i}"]
                )
            for name in ("particle_radius", "shape_margin"):
                np.testing.assert_array_equal(saved[name], getattr(runtime.model, name).numpy())
        self.frame_cache = prepare_frame_cache(
            self.cache, self.output, self.configuration, self.config.recorder.fps, self.indices
        )
        if self.capture_camera is not None:
            before = self.sim_world.physics_steps
            self._capture_samples(None, before)
            camera = next(c for c in self.recording_cameras if c.name == self.capture_camera)
            completed_frames = sum(
                valid_cached_frame(
                    self.frame_cache / camera.name / f"frame_{frame:06d}.png", camera.resolution
                )
                for frame in range(len(self.indices))
            )
            destination = self.output / "capture_workers" / f"{self.capture_camera}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(
                    dict(
                        success=True,
                        render_run_id=self.worker_id,
                        camera=self.capture_camera,
                        frames=len(self.indices),
                        completed_frames=completed_frames,
                        complete=completed_frames == len(self.indices),
                        physics_steps_advanced=self.sim_world.physics_steps - before,
                        maximum_pad_visual_error_m=runtime.visual_alignment_maximum_m,
                    ),
                    indent=2,
                )
                + "\n"
            )
            return
        # Revalidate every recorded view timestamp even when PNGs were resumed
        # from worker caches. The test uses the actual USD hierarchy and physics
        # poses, not just their shared root or a TCP position.
        for index in self.indices:
            runtime._publish_robot_visuals(saved["body_poses"][index])
        visual_geometry_audit = dict(
            passed=True,
            frames=len(self.indices),
            pads=len(runtime._pad_visual_checks),
            maximum_corner_error_m=runtime.visual_alignment_maximum_m,
            tolerance_m=2e-6,
            method=(
                "All eight visible USD pad corners versus solved collider corners "
                "at every video timestamp"
            ),
        )
        run = self.output / "diagnostics" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True)
        recorder = LeRobotV3Recorder(
            self.output / "lerobot_v3",
            fps=self.config.recorder.fps,
            width=self.config.recorder.width,
            height=self.config.recorder.height,
            task=self.config.recorder.task,
            robot_type=self.robot.recording_robot_type,
            joint_names=self.robot.recording_joint_names,
            action_names=self.planner.action_names,
            cameras={camera.name: camera.resolution for camera in self.recording_cameras},
        )
        before = self.sim_world.physics_steps
        try:
            first = int(self.indices[0])
            self._publish_sample(first)
            for _ in range(8):
                self.sim_world.update()
            self._capture_samples(recorder, before)
            result = dict(self.physical_result)
            if self.diagnostic_source:
                result.update(
                    success=False,
                    workflow_success=False,
                    strict_quality_success=False,
                    accepted_demonstration=False,
                    diagnostic_replay=True,
                    diagnostic_reason="Explicit replay of a source that fails current acceptance",
                )
            dataset = recorder.finalize(episode_success=bool(result["success"]))
            result.update(
                dataset=str(dataset),
                frames=len(self.indices),
                dataset_use="demonstration" if result["success"] else "diagnostic",
                observation_generation="rendered_from_recorded_physics",
                source_physics=str(self.cache),
                render_run_id=getattr(self, "worker_id", None),
                source_trajectory_sha256=hashlib.sha256(
                    (self.cache / "particles.npz").read_bytes()
                ).hexdigest(),
                source_configuration_sha256=hashlib.sha256(
                    (self.cache / "configuration.json").read_bytes()
                ).hexdigest(),
                observation_camera_pose_overrides=self.camera_pose_overrides,
                observation_render_profile=(
                    self.render_profile.model_dump(mode="json") if self.render_profile else None
                ),
                rendering_physics_steps_advanced=self.sim_world.physics_steps - before,
                observation_capture_order="camera",
                observation_capture_sync="flush_scene_before_pt_counter_v1",
                gripper_visual_geometry_audit=visual_geometry_audit,
                gripper_render_sync_audit=dict(
                    passed=True,
                    frames=sorted(self.sync_audit_frames),
                    cameras=[camera.name for camera in self.recording_cameras],
                    comparisons=len(self.sync_audit_frames) * len(self.recording_cameras),
                    method="Exported PNG versus fresh accumulation in the same renderer",
                ),
            )
            shutil.copy2(self.cache / "particles.npz", run / "particles.npz")
            if self.is_stream:
                shutil.copy2(self.cache / "stream.npz", run / "stream.npz")
                result["source_stream_sha256"] = hashlib.sha256(
                    (self.cache / "stream.npz").read_bytes()
                ).hexdigest()
            (run / "configuration.json").write_text(json.dumps(self.configuration, indent=2) + "\n")
            rows = json.loads(
                (
                    self.cache / ("stream_trajectory.json" if self.is_stream else "trajectory.json")
                ).read_text()
            )
            (run / "trajectory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            for destination in (run / "result.json", self.output / "result.json"):
                destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            (dataset / "meta/task_outcome.json").write_text(
                json.dumps(
                    {"physical_result": result, "configuration": self.configuration}, indent=2
                )
                + "\n"
            )
        except Exception:
            recorder.discard(run)
            raise

    def _capture_samples(self, recorder: LeRobotV3Recorder | None, before: int) -> None:
        """Render every camera against the same saved physical samples, without physics steps."""
        controlled = (
            self.sim_world.config.synchronous_capture
            and self.sim_world.config.renderer == "PathTracing"
        )
        control = "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges"
        settings = self.sim_world._settings if controlled else None
        previous_control = bool(settings.get(control)) if controlled else False
        started = time.monotonic()
        frame_cache = getattr(self, "frame_cache", None)
        frame_root = frame_cache if frame_cache is not None else recorder.frame_dir
        captured = 0
        try:
            for camera_index, camera in enumerate(self.recording_cameras):
                if getattr(self, "capture_camera", None) not in (None, camera.name):
                    continue
                paths = [
                    frame_root / camera.name / f"frame_{frame:06d}.png"
                    for frame in range(len(self.indices))
                ]
                cached = [
                    frame_cache is not None
                    and valid_cached_frame(path, camera.resolution)
                    and self._valid_frame_audit(camera.name, frame, path)
                    for frame, path in enumerate(paths)
                ]
                missing = [frame for frame, valid in enumerate(cached) if not valid]
                if missing:
                    self._prepare_offline_render(int(self.indices[missing[0]]), camera)
                    if controlled:
                        settings.set_bool(control, True)
                self.log.info("Sorting camera sequence started", camera=camera.name)
                for frame, index in enumerate(self.indices):
                    if not cached[frame]:
                        self._publish_sample(int(index))
                        capture_viewport_frame(
                            self.sim_world,
                            paths[frame],
                            width=camera.resolution[0],
                            height=camera.resolution[1],
                            camera_prim=camera.prim_path,
                            camera_sequence=frame != missing[0],
                        )
                        if frame in getattr(self, "sync_audit_frames", ()):
                            self._audit_captured_frame(camera, frame, paths[frame])
                        if frame_cache is not None and not valid_cached_frame(
                            paths[frame], camera.resolution
                        ):
                            raise RuntimeError(f"Incomplete captured image: {paths[frame]}")
                        captured += 1
                    chunk_done = recorder is None and captured >= getattr(
                        self, "capture_chunk_frames", 600
                    )
                    if self.sim_world.physics_steps != before:
                        raise RuntimeError("Rendering replay advanced physics")
                    if frame % 15 == 0 or frame == len(self.indices) - 1 or chunk_done:
                        progress = dict(
                            capture_order="camera",
                            camera=camera.name,
                            camera_index=camera_index,
                            cameras=len(self.recording_cameras),
                            frame=frame,
                            frames=len(self.indices),
                            source_index=int(index),
                            source_timestamp_s=float(self.saved["times"][index]),
                            elapsed_s=time.monotonic() - started,
                            physics_steps_advanced=0,
                            reused_frames=sum(cached),
                        )
                        (self.output / "render_progress.json").write_text(
                            json.dumps(progress, indent=2) + "\n"
                        )
                        self.log.info("Sorting camera sequence progress", **progress)
                    if chunk_done:
                        return
            # Validate all matching images before accepting shared state/action rows.
            if recorder is not None:
                if frame_cache is not None:
                    for camera in self.recording_cameras:
                        for frame in range(len(self.indices)):
                            name = f"frame_{frame:06d}.png"
                            source = frame_cache / camera.name / name
                            if not valid_cached_frame(source, camera.resolution):
                                raise RuntimeError(f"Incomplete camera sequence: {source}")
                            target = recorder.frame_dir / camera.name / name
                            try:
                                os.link(source, target)
                            except OSError:
                                shutil.copy2(source, target)
                for index in self.indices:
                    recorder.add_frame(self.saved["state"][index], self.saved["action"][index])
        finally:
            if controlled:
                settings.set_bool(control, previous_control)

    def _publish_sample(self, index: int) -> None:
        options = {}
        if self.is_stream:
            options["carton_visible"] = bool(self.saved["carton_visible"][index])
            if "belt_offsets_m" in self.saved:
                options["belt_offsets"] = self.saved["belt_offsets_m"][index]
        self.planner.runtime.publish_recorded_visuals(
            self.saved["positions"][index], self.saved["body_poses"][index], **options
        )


def run_camera_workers(cache: Path, output: Path, arguments: list[str]) -> int:
    """Capture bounded batches in fresh Kit processes, then encode the complete dataset."""
    configuration = json.loads((cache / "configuration.json").read_text())
    for section in ("scene", "robot"):
        for camera in configuration[section]["cameras"]:
            name = camera["name"]
            checkpoint = output / "capture_workers" / f"{name}.json"
            previous_frames = 0
            while True:
                status = run_kit_worker(
                    __file__,
                    [*arguments, "--capture-camera", name],
                    checkpoint,
                    marker_field="render_run_id",
                    require_success=True,
                )
                if status:
                    return status
                captured = json.loads(checkpoint.read_text())
                if captured["complete"]:
                    break
                if captured["completed_frames"] <= previous_frames:
                    raise RuntimeError(f"Camera capture made no progress: {name}")
                previous_frames = captured["completed_frames"]
    return run_kit_worker(__file__, arguments, output / "result.json", marker_field="render_run_id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--camera-poses", type=Path, help="JSON camera-name to optical-pose mapping"
    )
    parser.add_argument("--allow-failure", action="store_true")
    parser.add_argument(
        "--render-profile", type=Path, help="Strict presentation-only JSON settings"
    )
    parser.add_argument("--worker-id", help=argparse.SUPPRESS)
    parser.add_argument("--capture-camera", help=argparse.SUPPRESS)
    parser.add_argument("--capture-chunk-frames", type=int, default=600, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_id is None:
        if args.capture_camera is None:
            raise SystemExit(run_camera_workers(args.cache, args.output, sys.argv[1:]))
        raise SystemExit(
            run_kit_worker(
                __file__,
                sys.argv[1:],
                args.output / "capture_workers" / f"{args.capture_camera}.json",
                marker_field="render_run_id",
            )
        )
    result = json.loads((args.cache / "result.json").read_text())
    accepted = (
        result["completed"]
        and result["success"]
        and result.get("accepted_demonstration", True)
        and not result.get("diagnostic_restart", False)
    )
    if (args.cache / "stream.npz").is_file():
        accepted = accepted and all(
            result["checks"].get(name, False)
            for name in (
                "soft_entire_stream_stretch_bounded",
                "soft_entire_stream_triangles_valid",
                "soft_entire_stream_floor_clear",
            )
        )
    if not accepted and not args.allow_failure:
        parser.error(
            "Failed, incomplete, or restored source; --allow-failure permits diagnostic rendering"
        )
    job = SortingReplayJob(
        args.cache.resolve(),
        args.output.resolve(),
        args.fps,
        args.camera_poses,
        args.render_profile,
        args.capture_camera,
        args.capture_chunk_frames,
    )
    job.diagnostic_source = not accepted
    job.worker_id = args.worker_id
    job.run()
