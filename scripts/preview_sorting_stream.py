"""Render the initial station and mounted cameras; this is visual QA, not a physics run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from real2sim2real.core.config import apply_settings_file
from real2sim2real.job.job_sorting import SortingJob, SortingJobConfig
from real2sim2real.utils.cameras import CameraPose
from real2sim2real.utils.kit_process import run_kit_worker
from real2sim2real.utils.lerobot_v3 import capture_viewport_frame
from scripts.render_sorting_package import override_camera_poses


class PreviewJob(SortingJob):
    """Create the regular scene and capture its optical transforms without physics steps."""

    def simulate(self) -> None:
        from pxr import UsdGeom

        from real2sim2real.utils.asset_utils import prim_aabb

        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        alignment_error = None
        if self.show_carton:
            UsdGeom.Imageable(
                self.sim_world.stage.GetPrimAtPath(self.scene.target_prim_path)
            ).MakeInvisible()
            self.planner.runtime.spawn_carton()
            rendered = prim_aabb(
                self.sim_world.stage,
                f"{self.scene.config.prim_path}/stream_carton",
                gprims_only=True,
            )
            solved = self.planner.runtime.carton_vertices()
            expected = np.array([solved.min(0), solved.max(0)])
            alignment_error = float(np.abs(np.asarray(rendered) - expected).max())
            if alignment_error > 1e-6:
                raise ValueError("Rigid carton visual differs from its solved mesh pose")
        cameras = {}
        for camera in self.recording_cameras:
            capture_viewport_frame(
                self.sim_world,
                output / f"{camera.name}.png",
                width=camera.resolution[0],
                height=camera.resolution[1],
                camera_prim=camera.prim_path,
            )
            matrix = np.asarray(
                UsdGeom.XformCache().GetLocalToWorldTransform(
                    self.sim_world.stage.GetPrimAtPath(camera.prim_path)
                )
            )
            cameras[camera.name] = dict(
                position_m=matrix[3, :3].tolist(),
                forward=(-matrix[2, :3]).tolist(),
                world_from_camera=matrix.T.tolist(),
            )
        (output / "result.json").write_text(
            json.dumps(
                dict(
                    success=True,
                    scope="initial_visual_setup_only",
                    cameras=cameras,
                    physics_steps=self.sim_world.physics_steps,
                    carton_visible=self.show_carton,
                    rigid_visual_alignment_error_m=alignment_error,
                    preview_run_id=self.worker_id,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/PhysicSolver_mujoco_vbd_sorting_stream.toml")
    parser.add_argument("--output", type=Path, default=Path("outputs/debug/sorting_stream_preview"))
    parser.add_argument("--worker-id", help=argparse.SUPPRESS)
    parser.add_argument("--carton", action="store_true")
    parser.add_argument("--camera-poses", type=Path)
    args = parser.parse_args()
    if args.worker_id is None:
        raise SystemExit(
            run_kit_worker(
                __file__,
                sys.argv[1:],
                args.output / "result.json",
                marker_field="preview_run_id",
                require_success=True,
            )
        )
    apply_settings_file(args.config)
    config = SortingJobConfig.from_settings("job")
    config.output_dir = str(args.output)
    job = PreviewJob(config=config)
    job.worker_id = args.worker_id
    job.show_carton = args.carton
    if args.camera_poses:
        sections = {
            name: getattr(job, name).config.model_dump(mode="json") for name in ("scene", "robot")
        }
        updated = override_camera_poses(sections, json.loads(args.camera_poses.read_text()))
        for name in sections:
            for camera, saved in zip(
                getattr(job, name).config.cameras, updated[name]["cameras"], strict=True
            ):
                camera.pose = CameraPose.model_validate(saved["pose"])
    job.run()
