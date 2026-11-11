"""Serially collect the PhysicSolver cases and validate every LeRobot v3 result.

Verified existing datasets are reused. Failed physical trials are only exported when
job.recorder.save_failed_episodes=true, and remain explicitly marked as diagnostic.
The shared core/default_settings.toml file is never written by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from real2sim2real.core.logger import get_logger

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "featherstone_vbd_pickplace_cloth",
    "mujoco_vbd_pickplace_cloth",
    "xpbd_vbd_pickplace_cloth",
    "tgs_vbd_pickplace_cloth",
    "tgs_tgs_pickplace_rigid",
    "xpbd_xpbd_pickplace_cloth",
)


def validate_dataset(
    output: Path,
    *,
    allow_failure: bool = False,
    expected_cameras: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Check physical outcome, data/video time ranges, finite joints and video decodability."""
    result = json.loads((output / "result.json").read_text())
    root = output / "lerobot_v3"
    info = json.loads((root / "meta/info.json").read_text())
    episode = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
    if info["codebase_version"] != "v3.0" or not result["completed"]:
        raise ValueError("Incomplete task or unsupported dataset version")
    if not result["success"] and not allow_failure:
        raise ValueError("Physical acceptance failed; dataset cannot be treated as a demonstration")
    if result["success"] != all(result["checks"].values()):
        raise ValueError("Physical outcome disagrees with measured checks")
    if episode.get("success", result["success"]) != result["success"]:
        raise ValueError("Episode metadata mislabels the physical outcome")
    if not result["success"] and episode.get("dataset_use") != "diagnostic":
        raise ValueError("Failed trajectory is not marked as diagnostic")
    frames = info["total_frames"]
    data = pq.read_table(root / info["data_path"].format(chunk_index=0, file_index=0))
    if len(data) != frames or episode["length"] != frames or episode["dataset_to_index"] != frames:
        raise ValueError("Episode frame ranges do not match the data")
    np.testing.assert_array_equal(data["frame_index"].to_numpy(), np.arange(frames))
    np.testing.assert_allclose(
        data["timestamp"].to_numpy(), np.arange(frames) / info["fps"], atol=2e-6
    )
    for feature in ("observation.state", "action"):
        values = np.array(data[feature].to_pylist())
        metadata = info["features"][feature]
        size = metadata["shape"][0]
        if (
            values.shape != (frames, size)
            or len(metadata["names"]) != size
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid {feature}")
        if feature == "observation.state":
            arm_range = float(np.ptp(values[:, :7], axis=0).max())
            if result["success"] and arm_range < 0.1:
                raise ValueError("Successful grasp has stale or non-moving joint observations")
    video_features = {
        key: value for key, value in info["features"].items() if value["dtype"] == "video"
    }
    if not video_features:
        raise ValueError("No camera videos in dataset")
    if expected_cameras:
        for name, (width, height) in expected_cameras.items():
            if video_features.get(f"observation.images.{name}", {}).get("shape") != [
                height,
                width,
                3,
            ]:
                raise ValueError(f"Missing or incompatible camera: {name}; choose a new output_dir")
    videos = {}
    for video_key, feature in video_features.items():
        video = root / info["video_path"].format(video_key=video_key, chunk_index=0, file_index=0)
        stream = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_entries",
                    "stream=codec_name,nb_read_frames,r_frame_rate,width,height",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )["streams"][0]
        if int(stream["nb_read_frames"]) != frames:
            raise ValueError("Video/data frame count mismatch")
        numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
        if numerator / denominator != info["fps"]:
            raise ValueError("Video/data FPS mismatch")
        if (stream["width"], stream["height"]) != tuple(feature["shape"][1::-1]):
            raise ValueError(f"Video/metadata dimensions mismatch: {video_key}")
        if stream["codec_name"] != "h264":
            raise ValueError(f"Expected H.264 video: {video_key}")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"],
            check=True,
            capture_output=True,
        )
        if episode[f"videos/{video_key}/from_timestamp"] != 0.0:
            raise ValueError("Video must start at timestamp zero")
        if abs(episode[f"videos/{video_key}/to_timestamp"] - frames / info["fps"]) > 1e-6:
            raise ValueError("Video episode time range mismatch")
        videos[video_key] = str(video)
    report = {
        "validated": True,
        "grasp_success": result["success"],
        "frames": frames,
        "fps": info["fps"],
        "dataset": str(root),
        "video": videos.get("observation.images.overview", next(iter(videos.values()))),
        "videos": videos,
        "dataset_use": "demonstration" if result["success"] else "diagnostic",
        "place_error_m": result.get("place_error_m"),
        "minimum_hold_lift_m": result.get("minimum_hold_lift_m"),
        "checks": result["checks"],
        "maximum_arm_joint_range_rad": arm_range,
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    """Run each independent Kit process to completion before starting the next."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="Solver/task suffixes or PhysicSolver TOML paths")
    args = parser.parse_args()
    paths = []
    for value in args.cases or CASES:
        path = Path(value)
        if not path.is_file():
            name = value.removeprefix("PhysicSolver_").removesuffix(".toml")
            path = ROOT / "configs" / f"PhysicSolver_{name}.toml"
        if not path.is_file():
            parser.error(f"Missing configuration: {value}")
        paths.append(path.resolve())
    protected = ROOT / "real2sim2real/core/default_settings.toml"
    original = hashlib.sha256(protected.read_bytes()).hexdigest()
    root = ROOT / "outputs/PhysicSolver"
    logs = root / "capture_logs"
    logs.mkdir(parents=True, exist_ok=True)
    summary = {
        "serial": True,
        "cases": [],
        "collection_complete": False,
        "default_settings_sha256": original,
    }
    log = get_logger(__name__)
    for path in paths:
        config = tomllib.loads(path.read_text())
        output = ROOT / config["job"]["output_dir"]
        allow = config.get("job", {}).get("recorder", {}).get("save_failed_episodes", False)
        cameras = {
            camera["name"]: tuple(camera.get("resolution", [640, 480]))
            for section in ("scene", "robot")
            for camera in config.get(section, {}).get("cameras", [])
        }
        row = {"config": str(path), "output_dir": str(output), "validated": False}
        engine = config["sim_world"]["physics_engine"]
        kind = config["scene"]["object_kind"]
        row["robot_solver"] = (
            config["sim_world"]["newton_solver"]
            if engine == "newton"
            else config["sim_world"]["physx_solver"].lower()
        )
        row["contact_solver"] = (
            config["planner"]["cloth_solver"]
            if kind == "cloth"
            else config["sim_world"]["physx_solver"].lower()
        )
        row["task"] = (
            "sorting_package"
            if config["scene"]["backend"] == "sorting_package"
            else f"pickplace_{kind}"
        )
        started = time.monotonic()
        try:
            try:
                row.update(validate_dataset(output, allow_failure=allow, expected_cameras=cameras))
                row["reused_existing"] = True
            except (OSError, ValueError, KeyError, AssertionError):
                if (output / "lerobot_v3").exists():
                    raise ValueError(
                        "Existing dataset did not pass validation; choose a new output_dir"
                    ) from None
                log_path = logs / f"{path.stem}_{datetime.now(UTC):%Y%m%dT%H%M%S}.log"
                row["log"] = str(log_path)
                log.info("Collecting solver case", case=path.stem, log=str(log_path))
                environment = os.environ.copy()
                environment["RSR_JOB__OUTPUT_DIR"] = str(output)
                with log_path.open("w") as stream:
                    process = subprocess.run(
                        [
                            str(ROOT / "app/python.sh"),
                            "-m",
                            "real2sim2real.core.cli",
                            "collect",
                            "run",
                            "-c",
                            str(path),
                        ],
                        cwd=ROOT,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=config["job"].get("timeout_s", 1800) + 300,
                    )
                row["returncode"] = process.returncode
                if process.returncode:
                    raise RuntimeError(
                        f"Simulation exited with {process.returncode}; see {log_path}"
                    ) from None
                row.update(validate_dataset(output, allow_failure=allow, expected_cameras=cameras))
        except (
            OSError,
            ValueError,
            KeyError,
            AssertionError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as exc:
            row["error"] = str(exc)
            log.error("Solver collection failed", case=path.stem, error=str(exc))
        row["elapsed_s"] = time.monotonic() - started
        summary["cases"].append(row)
        summary["collection_complete"] = len(summary["cases"]) == len(paths) and all(
            x["validated"] for x in summary["cases"]
        )
        summary["all_grasps_successful"] = all(x.get("grasp_success") for x in summary["cases"])
        if hashlib.sha256(protected.read_bytes()).hexdigest() != original:
            raise RuntimeError("core/default_settings.toml changed during collection")
        summary_path = root / "collection_summary.json"
        temporary = summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(summary_path)
        log.info(
            "Solver case completed",
            case=path.stem,
            validated=row["validated"],
            success=row.get("grasp_success"),
        )
    return 0 if summary["collection_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
