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
        action_names: list[str] | None = None,
        cameras: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        if root.exists():
            raise FileExistsError(f"Dataset already exists: {root}")
        if fps <= 0 or width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("FPS must be positive and H.264 dimensions must be positive and even")
        self.cameras = dict(cameras) if cameras is not None else {"overview": (width, height)}
        if not self.cameras or any(
            not name.isidentifier()
            or any(size <= 0 or size % 2 for size in resolution)
            or len(resolution) != 2
            for name, resolution in self.cameras.items()
        ):
            raise ValueError("Cameras require valid names and two positive even dimensions")
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            raise RuntimeError("LeRobot video recording requires ffmpeg")
        root.parent.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.staging = Path(tempfile.mkdtemp(prefix=".lerobot-v3-", dir=root.parent))
        self.frame_dir = self.staging / "frames"
        self.frame_dir.mkdir()
        for name in self.cameras:
            (self.frame_dir / name).mkdir()
        self.fps, self.width, self.height, self.task = fps, width, height, task
        self.robot_type = robot_type
        self.joint_names = joint_names if joint_names is not None else JOINT_NAMES
        self.action_names = action_names if action_names is not None else self.joint_names
        if not self.joint_names or not self.action_names:
            raise ValueError("State and action coordinate names must be nonempty")
        if len(set(self.joint_names)) != len(self.joint_names) or len(
            set(self.action_names)
        ) != len(self.action_names):
            raise ValueError("Coordinate names must be unique within each feature")
        self.rows: list[dict[str, Any]] = []
        self.image_stats = {
            name: dict(sum=np.zeros(3), squared_sum=np.zeros(3), min=np.ones(3), max=np.zeros(3))
            for name in self.cameras
        }

    def frame_path(self, camera: str = "overview") -> Path:
        """Return the next capture filename; add_frame must follow a successful capture."""
        if camera not in self.cameras:
            raise KeyError(f"Unknown recording camera: {camera}")
        return self.frame_dir / camera / f"frame_{len(self.rows):06d}.png"

    def add_frame(self, state: np.ndarray, action: np.ndarray) -> None:
        """Store current joint positions and commanded targets at the same video frame time."""
        state, action = np.asarray(state, np.float32), np.asarray(action, np.float32)
        if (
            state.shape != (len(self.joint_names),)
            or action.shape != (len(self.action_names),)
            or not np.isfinite(state).all()
            or not np.isfinite(action).all()
        ):
            raise ValueError("State/action must match their named finite coordinates")
        # Validate every view before committing any per-frame statistics or joint data.
        images = {}
        for name, resolution in self.cameras.items():
            with Image.open(self.frame_path(name)) as image:
                if image.size != resolution:
                    raise ValueError(f"Unexpected capture size for {name}: {image.size}")
                images[name] = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
        for name, rgb in images.items():
            stats = self.image_stats[name]
            stats["sum"] += rgb.sum(axis=(0, 1))
            stats["squared_sum"] += np.square(rgb).sum(axis=(0, 1))
            stats["min"] = np.minimum(stats["min"], rgb.min(axis=(0, 1)))
            stats["max"] = np.maximum(stats["max"], rgb.max(axis=(0, 1)))
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
        for name in self.cameras:
            self._encode_video(name)
        schema = pa.schema(
            [
                ("observation.state", pa.list_(pa.float32(), len(self.joint_names))),
                ("action", pa.list_(pa.float32(), len(self.action_names))),
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
                "shape": [array.shape[1] if is_vector else 1],
                "names": (self.action_names if name == "action" else self.joint_names)
                if is_vector
                else None,
            }
        for name, (width, height) in self.cameras.items():
            video_key = f"observation.images.{name}"
            image = self.image_stats[name]
            pixels = len(self.rows) * width * height
            mean = image["sum"] / pixels
            std = np.sqrt(np.maximum(0, image["squared_sum"] / pixels - mean**2))
            stats[video_key] = {
                key: value.reshape(3, 1, 1).tolist()
                for key, value in dict(
                    min=image["min"], max=image["max"], mean=mean, std=std
                ).items()
            }
            stats[video_key]["count"] = [len(self.rows)]
            features[video_key] = {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": self.fps,
                    "video.height": height,
                    "video.width": width,
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
        }
        for name in self.cameras:
            video_key = f"observation.images.{name}"
            episode.update(
                {
                    f"videos/{video_key}/chunk_index": 0,
                    f"videos/{video_key}/file_index": 0,
                    f"videos/{video_key}/from_timestamp": 0.0,
                    f"videos/{video_key}/to_timestamp": len(self.rows) / self.fps,
                }
            )
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

    def _encode_video(self, name: str) -> None:
        video = self.staging / "videos" / f"observation.images.{name}" / "chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
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
                str(self.frame_dir / name / "frame_%06d.png"),
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


def capture_viewport_frame(
    world: Any,
    path: Path,
    *,
    width: int,
    height: int,
    camera_prim: str | None = None,
    camera_sequence: bool = False,
) -> None:
    """Capture without physics steps; optionally reuse a prewarmed fixed-camera sequence."""
    import asyncio

    import omni.usd
    from omni.kit.renderer_capture import acquire_renderer_capture_interface
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport for RGB capture")
    changed_view = tuple(viewport.resolution) != (width, height) or (
        camera_prim is not None and str(viewport.camera_path) != camera_prim
    )
    if camera_prim is not None:
        from pxr import UsdGeom

        prim = world.stage.GetPrimAtPath(camera_prim)
        if not prim or not prim.IsA(UsdGeom.Camera):
            raise ValueError(f"Recording camera does not exist: {camera_prim}")
        if str(viewport.camera_path) != camera_prim:
            viewport.camera_path = camera_prim
        expected_view = np.asarray(
            UsdGeom.XformCache().GetLocalToWorldTransform(prim).GetInverse(), dtype=float
        )
    else:
        expected_view = None
    if tuple(viewport.resolution) != (width, height):
        viewport.resolution = (width, height)
    before = world.physics_steps
    path_tracing = getattr(world.config, "renderer", "") == "PathTracing"
    controlled = path_tracing and getattr(world.config, "synchronous_capture", False)
    history_control = "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges"
    settings = world._settings if controlled else None
    previous_control = bool(settings.get(history_control)) if controlled else False

    def wait_rendered_frames(count: int) -> None:
        # Kit updates can outpace Hydra/GPU completion. Count actual render events.
        rendered = asyncio.ensure_future(viewport.wait_for_rendered_frames(count))
        for _ in range(240):
            world.update()
            if rendered.done():
                rendered.result()
                return
        rendered.cancel()
        raise TimeoutError(f"Viewport render did not complete: {viewport.camera_path}")

    try:
        for _attempt in range(3):
            continuing_sequence = (
                controlled
                and camera_sequence
                and previous_control
                and not changed_view
                and not _attempt
            )
            # Flush USD camera/material updates before locking accumulation. Doing
            # this after locking can retain an empty or previous-camera frame.
            if controlled and not continuing_sequence:
                settings.set_bool(history_control, False)
            if changed_view or _attempt or (controlled and not continuing_sequence):
                omni.usd.get_context().reset_renderer_accumulation()
            subframes = max(2, world.config.rt_subframes)
            if controlled:
                # USD rigid transforms reach Hydra after deformable point updates.
                # Even with an unchanged camera, drain actual rendered frames
                # before resetting accumulation for the newly published sample.
                # Otherwise the first PT iteration freezes the previous robot
                # pose together with the current cloth mesh.
                wait_rendered_frames(2 if continuing_sequence else max(8, subframes))
                settings.set_bool(history_control, True)
                counter = int(settings.get("/rtx/externalFrameCounter") or 0) + 1
                settings.set_int("/rtx/externalFrameCounter", counter)
                samples = world.config.samples_per_pixel_per_frame
                iterations = (samples + min(32, samples) - 1) // min(32, samples)
                subframes = max(subframes, iterations + 1)
            elif changed_view and path_tracing:
                subframes = max(8, subframes)
            wait_rendered_frames(subframes)
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
            actual_view = np.asarray(future.view, dtype=float).reshape(4, 4)
            if tuple(future.resolution) == (width, height) and (
                expected_view is None or np.allclose(actual_view, expected_view, rtol=0, atol=1e-5)
            ):
                return
        raise RuntimeError(f"RGB capture did not match camera/resolution: {camera_prim}")
    finally:
        if controlled:
            settings.set_bool(history_control, previous_control)
