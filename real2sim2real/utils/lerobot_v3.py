"""Single-episode LeRobot v3.0 export with synchronized H.264 video and Parquet metadata.

Layout follows https://huggingface.co/docs/lerobot/lerobot-dataset-v3 .
A dataset is published atomically only after the task's physical acceptance checks pass.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)] + ["left_finger_m", "right_finger_m"]
VIDEO_KEY = "observation.images.overview"


class LeRobotV3Recorder:
    """Record one complete episode; existing datasets are never overwritten."""

    def __init__(
        self,
        root: Path,
        *,
        fps: int,
        width: int,
        height: int,
        task: str,
        robot_type: str = "franka_fr3_newton",
        joint_names: list[str] | None = None,
    ) -> None:
        if root.exists():
            raise FileExistsError(f"Dataset already exists: {root}")
        if fps <= 0 or width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("FPS must be positive and H.264 dimensions must be positive and even")
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            raise RuntimeError("LeRobot video recording requires ffmpeg")
        root.parent.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.staging = Path(tempfile.mkdtemp(prefix=".lerobot-v3-", dir=root.parent))
        self.frame_dir = self.staging / "frames"
        self.frame_dir.mkdir()
        self.fps, self.width, self.height, self.task = fps, width, height, task
        self.robot_type = robot_type
        self.joint_names = joint_names if joint_names is not None else JOINT_NAMES
        self.rows: list[dict[str, Any]] = []
        self.image_sum = np.zeros(3)
        self.image_squared_sum = np.zeros(3)
        self.image_min = np.ones(3)
        self.image_max = np.zeros(3)
        self.pixels = 0

    def frame_path(self) -> Path:
        """Return the next capture filename; add_frame must follow a successful capture."""
        return self.frame_dir / f"frame_{len(self.rows):06d}.png"

    def add_frame(self, state: np.ndarray, action: np.ndarray) -> None:
        """Store current joint positions and commanded targets at the same video frame time."""
        state, action = np.asarray(state, np.float32), np.asarray(action, np.float32)
        if state.shape != (9,) or action.shape != (9,) or not np.isfinite([state, action]).all():
            raise ValueError("State/action must each contain nine finite coordinates")
        with Image.open(self.frame_path()) as image:
            if image.size != (self.width, self.height):
                raise ValueError(f"Unexpected capture size: {image.size}")
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
        self.image_sum += rgb.sum(axis=(0, 1))
        self.image_squared_sum += np.square(rgb).sum(axis=(0, 1))
        self.image_min = np.minimum(self.image_min, rgb.min(axis=(0, 1)))
        self.image_max = np.maximum(self.image_max, rgb.max(axis=(0, 1)))
        self.pixels += self.width * self.height
        index = len(self.rows)
        self.rows.append(
            {
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "timestamp": index / self.fps,
                "frame_index": index,
                "episode_index": 0,
                "index": index,
                "task_index": 0,
            }
        )

    def finalize(self, *, episode_success: bool | None = None) -> Path:
        """Encode videos and publish a conforming v3 dataset after all files have been written."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if len(self.rows) < 2:
            raise ValueError("A dataset episode must contain at least two frames")
        video = self.staging / "videos" / VIDEO_KEY / "chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True)
        subprocess.run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(self.fps),
                "-i",
                str(self.frame_dir / "frame_%06d.png"),
                "-frames:v",
                str(len(self.rows)),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError("Video encoding produced no output")
        schema = pa.schema(
            [
                ("observation.state", pa.list_(pa.float32(), 9)),
                ("action", pa.list_(pa.float32(), 9)),
                ("timestamp", pa.float32()),
                *[
                    (name, pa.int64())
                    for name in ("frame_index", "episode_index", "index", "task_index")
                ],
            ]
        )
        data_path = self.staging / "data/chunk-000/file-000.parquet"
        data_path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(self.rows, schema=schema), data_path)
        stats = {}
        features = {}
        for name in self.rows[0]:
            is_vector = name in {"observation.state", "action"}
            array = np.asarray([row[name] for row in self.rows], dtype=float).reshape(
                len(self.rows), -1
            )
            stats[name] = self._stats(array)
            features[name] = {
                "dtype": "float32" if is_vector or name == "timestamp" else "int64",
                "shape": [9 if is_vector else 1],
                "names": self.joint_names if is_vector else None,
            }
        image_mean = self.image_sum / self.pixels
        image_std = np.sqrt(np.maximum(0, self.image_squared_sum / self.pixels - image_mean**2))
        stats[VIDEO_KEY] = {
            name: value.reshape(3, 1, 1).tolist()
            for name, value in {
                "min": self.image_min,
                "max": self.image_max,
                "mean": image_mean,
                "std": image_std,
            }.items()
        }
        stats[VIDEO_KEY]["count"] = [len(self.rows)]
        features[VIDEO_KEY] = {
            "dtype": "video",
            "shape": [self.height, self.width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": self.fps,
                "video.height": self.height,
                "video.width": self.width,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
        episode = {
            "episode_index": 0,
            "tasks": [self.task],
            "length": len(self.rows),
            "dataset_from_index": 0,
            "dataset_to_index": len(self.rows),
            "data/chunk_index": 0,
            "data/file_index": 0,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            f"videos/{VIDEO_KEY}/chunk_index": 0,
            f"videos/{VIDEO_KEY}/file_index": 0,
            f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
            f"videos/{VIDEO_KEY}/to_timestamp": len(self.rows) / self.fps,
        }
        if episode_success is not None:
            episode["success"] = episode_success
            episode["dataset_use"] = "demonstration" if episode_success else "diagnostic"
        for name, values in stats.items():
            episode.update({f"stats/{name}/{stat}": value for stat, value in values.items()})
        episode_path = self.staging / "meta/episodes/chunk-000/file-000.parquet"
        episode_path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([episode]), episode_path)
        # Official load_tasks uses a pandas index containing task descriptions.
        tasks = pa.table({"task": [self.task], "task_index": pa.array([0], type=pa.int64())})
        pandas_metadata = {
            "index_columns": ["task"],
            "column_indexes": [],
            "columns": [
                {
                    "name": name,
                    "field_name": name,
                    "pandas_type": dtype,
                    "numpy_type": numpy_type,
                    "metadata": None,
                }
                for name, dtype, numpy_type in [
                    ("task", "unicode", "object"),
                    ("task_index", "int64", "int64"),
                ]
            ],
            "creator": {"library": "pyarrow", "version": pa.__version__},
            "pandas_version": "2.2.3",
        }
        tasks = tasks.replace_schema_metadata({b"pandas": json.dumps(pandas_metadata).encode()})
        pq.write_table(tasks, self.staging / "meta/tasks.parquet")
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "fps": self.fps,
            "total_episodes": 1,
            "total_frames": len(self.rows),
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "splits": {"train": "0:1"},
            "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
        for name, value in [("info", info), ("stats", stats)]:
            (self.staging / "meta" / f"{name}.json").write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n"
            )
        shutil.rmtree(self.frame_dir)
        if self.root.exists():
            raise FileExistsError(self.root)
        self.staging.rename(self.root)
        return self.root

    def discard(self, diagnostics: Path) -> None:
        """Keep failed captures outside the accepted training dataset for inspection."""
        if self.staging.exists():
            self.staging.rename(diagnostics / "failed_recording")

    @staticmethod
    def _stats(array: np.ndarray) -> dict[str, Any]:
        return {
            "min": array.min(axis=0).tolist(),
            "max": array.max(axis=0).tolist(),
            "mean": array.mean(axis=0).tolist(),
            "std": array.std(axis=0).tolist(),
            "count": [len(array)],
        }


def capture_viewport_frame(world: Any, path: Path, *, width: int, height: int) -> None:
    """Capture the current USD pose without advancing physics; synchronous file completion."""
    import asyncio

    from omni.kit.renderer_capture import acquire_renderer_capture_interface
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport for RGB capture")
    viewport.resolution = (width, height)
    before = world.physics_steps
    future = capture_viewport_to_file(viewport, str(path))
    task = asyncio.ensure_future(future.wait_for_result(completion_frames=0))
    for _ in range(120):
        world.update()
        if task.done():
            task.result()
            acquire_renderer_capture_interface().wait_async_capture()
            break
    else:
        task.cancel()
        raise TimeoutError(f"RGB capture timed out: {path}")
    if not path.is_file() or world.physics_steps != before:
        raise RuntimeError("Capture failed or advanced physics")
