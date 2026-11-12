"""Compare moving-sequence RGB frames with independently refreshed static replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from real2sim2real.utils.cameras import CameraConfig
from real2sim2real.utils.kit_process import run_kit_worker
from real2sim2real.utils.lerobot_v3 import capture_viewport_frame
from scripts.sorting.render_sorting_package import (
    CAPTURE_SYNC_CONTRACT,
    SortingReplayJob,
    render_sync_audit_frames,
)


class RenderSyncAudit(SortingReplayJob):
    """Check actual rendered pixels, including the reported head-camera frame 340."""

    def simulate(self) -> None:
        rows = json.loads((self.cache / "stream_trajectory.json").read_text())
        selected = set(self.frames) or set(
            render_sync_audit_frames(rows, self.config.recorder.fps, len(self.indices))
        )
        selected = {frame for frame in selected if 2 <= frame < len(self.indices)}
        control = "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges"
        comparisons = []
        before = self.sim_world.physics_steps
        for camera in self.recording_cameras:
            directory = self.output / camera.name
            directory.mkdir(parents=True, exist_ok=True)

            def capture(path: Path, sequence: bool, camera: CameraConfig = camera) -> None:
                capture_viewport_frame(
                    self.sim_world,
                    path,
                    width=camera.resolution[0],
                    height=camera.resolution[1],
                    camera_prim=camera.prim_path,
                    camera_sequence=sequence,
                )

            for frame in sorted(selected):
                moving = directory / f"sequence_{frame:06d}.png"
                self._prepare_offline_render(int(self.indices[frame - 2]), camera)
                self.sim_world._settings.set_bool(control, True)
                for previous in range(frame - 2, frame + 1):
                    self._publish_sample(int(self.indices[previous]))
                    capture(moving, previous != frame - 2)
                static = directory / f"static_{frame:06d}.png"
                self._prepare_offline_render(int(self.indices[frame]), camera)
                capture(static, False)
                with Image.open(moving) as image:
                    actual = np.asarray(image.convert("RGB"), dtype=np.int16)
                with Image.open(static) as image:
                    expected = np.asarray(image.convert("RGB"), dtype=np.int16)
                delta = np.abs(actual - expected)
                comparison = dict(
                    camera=camera.name,
                    frame=frame,
                    timestamp_s=float(self.saved["times"][self.indices[frame]]),
                    sequence=str(moving),
                    static=str(static),
                    mean_absolute_pixel_error=float(delta.mean()),
                    maximum_pixel_error=int(delta.max()),
                    identical=bool(np.array_equal(actual, expected)),
                )
                comparisons.append(comparison)
                self.log.info("Sequence/static RGB comparison", **comparison)
        advanced = self.sim_world.physics_steps - before
        result = dict(
            success=advanced == 0 and all(row["identical"] for row in comparisons),
            render_run_id=self.worker_id,
            source_physics=str(self.cache),
            capture_sync_contract=CAPTURE_SYNC_CONTRACT,
            physics_steps_advanced=advanced,
            resolution=self.recording_cameras[0].resolution,
            comparisons=comparisons,
        )
        (self.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["success"]:
            raise ValueError("Sequence RGB differs from fully refreshed static replay")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render-profile", required=True, type=Path)
    parser.add_argument("--frames", default="", help="只检查这些帧（逗号分隔），默认全部审计帧")
    parser.add_argument("--worker-id")
    args = parser.parse_args()
    if args.worker_id is None:
        import sys

        return run_kit_worker(
            __file__,
            sys.argv[1:],
            args.output / "result.json",
            marker_field="render_run_id",
            require_success=True,
        )
    job = RenderSyncAudit(
        args.source.resolve(), args.output.resolve(), 15, render_profile=args.render_profile
    )
    job.worker_id = args.worker_id
    job.frames = [int(value) for value in args.frames.split(",") if value]
    job.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
