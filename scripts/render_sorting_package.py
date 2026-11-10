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

import numpy as np

from real2sim2real.core.config import apply_settings_file
from real2sim2real.job.job_sorting import SortingJob, SortingJobConfig
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.cameras import CameraPose
from real2sim2real.utils.kit_process import run_kit_worker
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder


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
        self, cache: Path, output: Path, fps: int, camera_poses: Path | None = None
    ) -> None:
        self.cache, self.output = cache, output
        self.configuration = json.loads((cache / "configuration.json").read_text())
        self.physical_result = json.loads((cache / "result.json").read_text())
        if (
            self.physical_result["rigid_solver"] != "mujoco"
            or self.physical_result["contact_solver"] != "vbd"
            or self.configuration["planner"]["backend"] != "sorting_package"
        ):
            raise ValueError("Replay requires a native Pro-M6 MuJoCo + VBD sorting cache")
        asset = Path(resolve_usd_path(self.configuration["scene"]["cloth_usd_path"]))
        if hashlib.sha256(asset.read_bytes()).hexdigest() != self.physical_result["asset_sha256"]:
            raise ValueError("Parcel asset changed after the physical solve")
        with np.load(cache / "particles.npz") as archive:
            self.saved = {key: archive[key] for key in archive.files}
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
            runtime.publish_recorded_visuals(saved["positions"][first], saved["body_poses"][first])
            for _ in range(8):
                self.sim_world.update()
            for frame, index in enumerate(self.indices):
                runtime.publish_recorded_visuals(
                    saved["positions"][index], saved["body_poses"][index]
                )
                self._capture_cameras(recorder)
                recorder.add_frame(saved["state"][index], saved["action"][index])
                if self.sim_world.physics_steps != before:
                    raise RuntimeError("Rendering replay advanced physics")
                if frame % 30 == 0:
                    self.log.info("Sorting replay", frame=frame, total_frames=len(self.indices))
            result = dict(self.physical_result)
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
            )
            shutil.copy2(self.cache / "particles.npz", run / "particles.npz")
            (run / "configuration.json").write_text(json.dumps(self.configuration, indent=2) + "\n")
            rows = json.loads((self.cache / "trajectory.json").read_text())
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--camera-poses", type=Path, help="JSON camera-name to optical-pose mapping"
    )
    parser.add_argument("--allow-failure", action="store_true")
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
    if not (result["completed"] and result["success"]) and not args.allow_failure:
        parser.error(
            "Physical trial failed or is incomplete; failed trials require --allow-failure"
        )
    job = SortingReplayJob(args.cache.resolve(), args.output.resolve(), args.fps, args.camera_poses)
    job.worker_id = args.worker_id
    job.run()
