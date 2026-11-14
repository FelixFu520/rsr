"""Independently check a HALO output directory and its LeRobot v3 dataset (no Kit).

Usage: ./app/python.sh scripts/halo/validate_halo.py outputs/halo/crawl [--compare <npz>]

Checks, from the published files only:
1. dataset layout: v3.0 info, parquet rows, 71-D state / 29-D action, every HALO extra column;
2. every data row equals the archived 50 Hz control recording at its control frame
   (no interpolation), nominal timestamps ``i / fps``, measured timestamps within 10 ms;
3. both videos: H.264, frame count, fps, resolution, full decode, no black frames;
4. the physical acceptance recomputed from ``meta/physics`` equals ``result.json``.
``--compare`` reports the deviation from a HALO source ``control_recording.npz``.
Writes ``validation.json`` and ``qa/keyframes.jpg`` (head row, rear row) next to ``result.json``;
exits non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from real2sim2real.job.job_halo import HaloJobConfig, evaluate_halo, halo_extra_features
from real2sim2real.scene.scene_halo import HaloHomeSceneConfig
from real2sim2real.utils.halo_utils import CONTROL_HZ, RobotGeometry, image_schedule

ROW_SOURCES = {
    "observation.contacts": "contacts",
    "observation.camera_head_pose": "head_pose",
    "observation.camera_rear_pose": "rear_pose",
    "observation.contact_vertical_impulses": "contact_vertical_impulses",
    "observation.phase_index": "phase_index",
    "observation.body_landmarks": "landmark_positions",
    "observation.foot_support_impulses": "foot_support_impulses",
    "observation.torso_pose": "torso_pose",
    "observation.measured_timestamp": "timestamp",
    "observation.control_frame_index": "control_frame",
}


def _probe(video: Path) -> dict:
    out = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries",
            "stream=nb_read_frames,width,height,codec_name,avg_frame_rate", "-of", "json",
            str(video),
        ],
        check=True, capture_output=True, text=True, timeout=1800,
    )  # fmt: skip
    return json.loads(out.stdout)["streams"][0]


def _black_frames(video: Path, size: tuple[int, int] = (160, 90)) -> list[int]:
    """Indices of decoded frames whose thumbnail mean and std are both below 1/255."""
    raw = subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-xerror", "-i", str(video),
            "-vf", f"scale={size[0]}:{size[1]}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True, capture_output=True, timeout=3600,
    ).stdout  # fmt: skip
    frames = np.frombuffer(raw, np.uint8).reshape(-1, size[1], size[0], 3).astype(float)
    flat = frames.reshape(len(frames), -1)
    return np.flatnonzero((flat.mean(1) < 1) & (flat.std(1) < 1)).tolist()


def _keyframes(videos: dict[str, Path], frames: int, target: Path, count: int = 6) -> None:
    """Contact sheet: one row per camera, ``count`` evenly spaced frames, 320 px wide each."""
    from PIL import Image

    picks = np.linspace(0, frames - 1, count).round().astype(int)
    rows = []
    for video in videos.values():
        select = "+".join(f"eq(n\\,{i})" for i in picks)
        raw = subprocess.run(
            [
                shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-i", str(video),
                "-vf", f"select='{select}',scale=320:180", "-vsync", "0",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ],
            check=True, capture_output=True, timeout=600,
        ).stdout  # fmt: skip
        tiles = np.frombuffer(raw, np.uint8).reshape(-1, 180, 320, 3)
        rows.append(np.concatenate(list(tiles), axis=1))
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(target, quality=90)


def validate(output: Path, compare: Path | None = None) -> dict:
    result = json.loads((output / "result.json").read_text())
    root = output / "lerobot_v3"
    physics = root / "meta/physics/episode_000000"
    info = json.loads((root / "meta/info.json").read_text())
    configs = json.loads((physics / "configuration.json").read_text())
    job = HaloJobConfig.model_validate(configs["job"])
    scene = HaloHomeSceneConfig.model_validate(configs["scene"])
    with np.load(physics / "control_recording.npz") as data:
        rec = {key: data[key] for key in data.files}
    geometry = RobotGeometry.load(physics / "robot_geometry.npz")
    table = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    checks: dict[str, bool] = {}
    details: dict = {}
    frames = info["total_frames"]
    fps = info["fps"]
    landmarks = configs["robot"]["landmark_bodies"]
    extras = halo_extra_features(landmarks)
    checks["dataset_v3"] = info["codebase_version"] == "v3.0" and info["total_episodes"] == 1
    checks["state_71_action_29"] = info["features"]["observation.state"]["shape"] == [71] and info[
        "features"
    ]["action"]["shape"] == [29]
    checks["halo_extra_columns"] = all(
        {k: info["features"].get(name, {}).get(k) for k in ("dtype", "shape", "names")} == spec
        for name, spec in extras.items()
    )
    checks["row_count"] = len(table["frame_index"]) == frames and fps == job.recorder.fps
    index = np.asarray(table["observation.control_frame_index"], dtype=int)
    planned = image_schedule(len(rec["qpos"]), CONTROL_HZ, fps)
    checks["rows_follow_image_schedule"] = np.array_equal(index, planned[: len(index)])
    checks["nominal_timestamps"] = bool(
        np.allclose(table["timestamp"], np.arange(frames) / fps, atol=2e-6)
    )
    measured = np.asarray(table["observation.measured_timestamp"])
    details["max_time_quantization_s"] = float(np.max(np.abs(measured - np.arange(frames) / fps)))
    checks["measured_time_within_10ms"] = details["max_time_quantization_s"] <= 0.010002
    state = np.asarray(table["observation.state"])
    expected = np.concatenate(
        [
            rec["qpos"][index, 7:],
            rec["joint_vel"][index],
            rec["qpos"][index, :7],
            rec["base_vel"][index],
        ],
        axis=1,
    )
    checks["state_rows_equal_control_recording"] = bool(
        np.array_equal(state.astype(np.float32), expected)
    )
    checks["action_rows_equal_control_recording"] = bool(
        np.array_equal(np.asarray(table["action"], dtype=np.float32), rec["action"][index])
    )
    extra_ok = []
    for name, key in ROW_SOURCES.items():
        column = np.asarray(table[name]).reshape(frames, -1)
        source = rec[key][index].reshape(frames, -1).astype(column.dtype)
        extra_ok.append(bool(np.allclose(column, source, atol=0, rtol=0)))
    checks["extra_rows_equal_control_recording"] = all(extra_ok)
    done = np.asarray(table["next.done"])
    success = np.asarray(table["next.success"])
    checks["done_flag_on_last_row"] = bool(done[-1] and not done[:-1].any())
    checks["success_flag_matches_result"] = bool(
        success[-1] == result["success"] and not success[:-1].any()
    )
    videos = {}
    for camera, (width, height) in {
        "head": (
            configs["robot"]["head_camera"]["width"],
            configs["robot"]["head_camera"]["height"],
        ),
        "rear": (scene.third_person.width, scene.third_person.height),
    }.items():
        video = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        stream = _probe(video)
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        black = _black_frames(video)
        videos[camera] = {**stream, "black_frames": black}
        checks[f"{camera}_video"] = (
            stream["codec_name"] == "h264"
            and int(stream["nb_read_frames"]) == frames
            and numerator / denominator == fps
            and (stream["width"], stream["height"]) == (width, height)
            and not black
        )
    details["videos"] = videos
    _keyframes(
        {c: root / f"videos/observation.images.{c}/chunk-000/file-000.mp4" for c in videos},
        frames,
        output / "qa" / "keyframes.jpg",
    )
    acceptance = json.loads((physics / "acceptance.json").read_text())
    recomputed = evaluate_halo(
        rec,
        geometry,
        job=job,
        scene=scene,
        scene_report=acceptance["scene_report"],
        phases=result["planner"]["phases"],
        reference_fps=configs["planner"]["reference_fps"],
        control_frames=result["planner"]["control_frames"],
        joint_limits=np.asarray(acceptance["runtime"]["joint_limits_rad"]),
        robot_mass_kg=acceptance["runtime"]["robot_mass_kg"],
        landmarks=landmarks,
        image_count=frames,
        runtime=acceptance["runtime"],
    )
    checks["acceptance_recomputed_equal"] = recomputed["checks"] == result["checks"]
    checks["physical_acceptance_passed"] = bool(recomputed["success"] and result["success"])
    details["failed_physical_checks"] = recomputed["failed_checks"]
    if compare is not None:
        with np.load(compare) as source:
            n = min(len(source["qpos"]), len(rec["qpos"]))
            base = np.linalg.norm(rec["qpos"][:n, :3] - source["qpos"][:n, :3], axis=1)
            joints = np.abs(rec["qpos"][:n, 7:] - source["qpos"][:n, 7:]).max(1)
            details["halo_source_comparison"] = {
                "source": str(compare),
                "frames": n,
                "source_frames": len(source["qpos"]),
                "base_position_diff_m": {"max": float(base.max()), "final": float(base[-1])},
                "joint_position_diff_rad": {
                    "max": float(joints.max()),
                    "median": float(np.median(joints)),
                },
            }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [k for k, ok in checks.items() if not ok],
        "frames": frames,
        "fps": fps,
        "dataset": str(root),
        "details": details,
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare", type=Path, help="HALO source control_recording.npz")
    args = parser.parse_args()
    report = validate(args.output, args.compare)
    print(json.dumps({k: report[k] for k in ("passed", "failed", "frames", "fps")}, indent=2))
    if "halo_source_comparison" in report["details"]:
        print(json.dumps(report["details"]["halo_source_comparison"], indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
