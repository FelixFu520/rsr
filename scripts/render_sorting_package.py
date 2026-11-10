"""Render synchronized LeRobot v3 observations from a real MuJoCo + VBD trajectory.

Replay changes visual transforms only. Physical acceptance, measured joints and
actions come exclusively from the saved solver trajectory; physics is not rerun.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from real2sim2real.core.config import apply_settings_file
from real2sim2real.job.job_sorting import SortingJob, SortingJobConfig
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.cameras import CameraPose
from real2sim2real.utils.kit_process import run_kit_worker
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder


class ReplayRenderProfile(BaseModel):
    """Presentation-only settings; never accepts dynamics or robot modifications."""

    model_config = ConfigDict(extra="forbid")
    renderer: Literal["RayTracedLighting", "PathTracing"] = "PathTracing"
    resolution: tuple[Annotated[int, Field(ge=64)], Annotated[int, Field(ge=64)]] = (
        1920,
        1440,
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


class SortingReplayJob(SortingJob):
    """Use the standard Job lifecycle and native visuals, keeping the world paused."""

    def __init__(
        self,
        cache: Path,
        output: Path,
        fps: int,
        camera_poses: Path | None = None,
        render_profile: Path | None = None,
    ) -> None:
        self.cache, self.output = cache, output
        self.diagnostic_source = False
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

    def _prepare_offline_render(self, first_index: int) -> None:
        """Load materials/lights before externally controlling progressive accumulation."""
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        camera = self.recording_cameras[0]
        if self.sim_world.config.synchronous_capture:
            self.sim_world._settings.set_bool(
                "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges", False
            )
        viewport.camera_path = camera.prim_path
        viewport.resolution = camera.resolution
        self._publish_sample(first_index)
        for _ in range(16):
            self.sim_world.update()

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
            if self.render_profile is not None:
                self._prepare_offline_render(first)
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
                observation_capture_order="frame",
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

    def _capture_samples(self, recorder: LeRobotV3Recorder, before: int) -> None:
        """Capture every view and commit one row before reading the next source sample."""
        for frame, index in enumerate(self.indices):
            self._publish_sample(int(index))
            self._capture_cameras(recorder)
            if self.sim_world.physics_steps != before:
                raise RuntimeError("Rendering replay advanced physics")
            recorder.add_frame(self.saved["state"][index], self.saved["action"][index])
            if frame % 30 == 0:
                self.log.info("Sorting replay", frame=frame, total_frames=len(self.indices))

    def _publish_sample(self, index: int) -> None:
        options = {}
        if self.is_stream:
            options["carton_visible"] = bool(self.saved["carton_visible"][index])
            if "belt_offsets_m" in self.saved:
                options["belt_offsets"] = self.saved["belt_offsets_m"][index]
        self.planner.runtime.publish_recorded_visuals(
            self.saved["positions"][index], self.saved["body_poses"][index], **options
        )


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
    args = parser.parse_args()
    if args.worker_id is None:
        raise SystemExit(
            run_kit_worker(
                __file__,
                sys.argv[1:],
                args.output / "result.json",
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
    )
    job.diagnostic_source = not accepted
    job.worker_id = args.worker_id
    job.run()
