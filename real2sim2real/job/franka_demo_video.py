"""Viewport video capture with an explicit simulation-time frame cadence."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any


class FrankaDemoVideo:
    """Capture rendered poses without integrating physics, then encode an MP4."""

    def __init__(self, world: Any, output: Path, *, stride: int, width: int, height: int) -> None:
        from omni.kit.viewport.utility import get_active_viewport

        if width % 2 or height % 2:
            raise ValueError("Video width and height must be even for H.264")
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            raise RuntimeError("Recording requires ffmpeg on PATH")
        self.world = world
        self.output = output
        self.viewport = get_active_viewport()
        if self.viewport is None:
            raise RuntimeError("No active viewport available for video recording")
        self.viewport.resolution = (width, height)
        self.fps = Fraction(1 / (world.config.physics_dt * stride)).limit_denominator(100000)
        self.frames: list[dict[str, Any]] = []
        self.frame_dir = Path(tempfile.mkdtemp(prefix=".video_frames_", dir=output))
        for _ in range(8):
            world.update()

    def capture(self, *, step: int, phase: str) -> None:
        """Capture one pose and retain its exact physics-step timestamp."""
        from omni.kit.renderer_capture import acquire_renderer_capture_interface
        from omni.kit.viewport.utility import capture_viewport_to_file

        path = self.frame_dir / f"frame_{len(self.frames):06d}.png"
        physics_steps = self.world.physics_steps
        future = capture_viewport_to_file(self.viewport, str(path))
        task = asyncio.ensure_future(future.wait_for_result(completion_frames=0))
        for _ in range(120):
            self.world.update()
            if task.done():
                task.result()
                acquire_renderer_capture_interface().wait_async_capture()
                break
        else:
            task.cancel()
            raise TimeoutError(f"Viewport capture timed out: {path}")
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Viewport capture did not produce an image: {path}")
        if self.world.physics_steps != physics_steps:
            raise RuntimeError("Video capture unexpectedly advanced physics")
        self.frames.append(
            {
                "frame": len(self.frames),
                "step": step,
                "simulation_time_s": step * self.world.config.physics_dt,
                "phase": phase,
            }
        )

    def finalize(self) -> dict[str, Any]:
        """Encode continuous numbered frames and keep them if encoding fails."""
        if not self.frames:
            raise RuntimeError("No video frames were captured")
        video_path = self.output / "grasp.mp4"
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(self.fps),
            "-start_number",
            "0",
            "-i",
            str(self.frame_dir / "frame_%06d.png"),
            "-frames:v",
            str(len(self.frames)),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
        if result.returncode or not video_path.is_file() or not video_path.stat().st_size:
            raise RuntimeError(
                f"Video encoding failed; frames kept at {self.frame_dir}: {result.stderr}"
            )
        metadata = {
            "path": str(video_path),
            "fps": float(self.fps),
            "frame_count": len(self.frames),
            "duration_s": len(self.frames) / float(self.fps),
            "resolution": list(self.viewport.resolution),
            "capture_advances_physics": False,
        }
        (self.output / "video.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        with (self.output / "video_frames.jsonl").open("w", encoding="utf-8") as stream:
            for frame in self.frames:
                stream.write(json.dumps(frame) + "\n")
        shutil.rmtree(self.frame_dir)
        return metadata
