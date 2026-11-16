"""Independently validate banana topology, physical traces, LeRobot data and decoded video."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from real2sim2real.core.logger import get_logger
from real2sim2real.utils.physic_solver_utils import CuttingMesh, cutting_evaluate


def validate(output: Path) -> dict:
    """Recompute acceptance from saved meshes and every-step physical measurements, without Kit."""
    config = json.loads((output / "configuration.json").read_text())
    result = json.loads((output / "result.json").read_text())
    events = json.loads((output / "cut_events.json").read_text())
    rows = json.loads((output / "trajectory.json").read_text())
    paths = json.loads((output / "mesh_paths.json").read_text())
    with np.load(output / "meshes.npz", allow_pickle=False) as arrays:
        meshes = {
            path: CuttingMesh(arrays[f"v{i}"], arrays[f"f{i}"], arrays[f"m{i}"])
            for i, path in enumerate(paths)
        }
    planner, world = config["planner"], config["sim_world"]
    report = cutting_evaluate(
        events,
        rows,
        expected_cuts=len(planner["cut_positions_m"]),
        maximum_volume_relative_error=planner["maximum_volume_relative_error"],
        minimum_fragment_motion_m=planner["minimum_fragment_motion_m"],
        maximum_support_penetration_m=planner["maximum_support_penetration_m"],
        meshes=meshes,
        duration_s=planner["duration_s"],
        physics_dt=world["physics_dt"],
    )
    checks = report["checks"]
    checks["job_agrees"] = bool(result["success"]) and result["checks"] == report["checks"]
    checks["step_count"] = len(rows) == result["physics_steps"] + 1
    checks["replay_exists"] = (output / "banana_cutting_replay.usdc").is_file()
    checks["fragment_births_and_trace"] = all(
        {p["path"] for p in row["pieces"]}
        == {
            path
            for e in events
            if e["time_s"] <= row["time_s"] + 1e-9
            for path in e.get("released_paths", [e["fragment_path"]])
        }
        for row in rows
    )
    birth_checks = []
    for event in events:
        row = next(row for row in rows if abs(row["time_s"] - event["time_s"]) < 1e-8)
        for path in event.get("released_paths", [event["fragment_path"]]):
            piece = next(p for p in row["pieces"] if p["path"] == path)
            center = meshes[path].vertices.mean(axis=0)
            birth_checks.append(np.allclose(center, piece["position_m"], atol=1e-8, rtol=0))
    checks["exact_geometric_birth"] = bool(birth_checks) and all(birth_checks)
    asset_root = Path(config["scene"]["asset_dir"])
    manifest = json.loads((asset_root / "provenance.json").read_text())
    checks["asset_hashes"] = all(
        hashlib.sha256((asset_root / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        for entry in manifest["files"]
    )
    dependencies = result.get("asset_dependencies", {})
    checks["task_local_asset_closure"] = (
        bool(dependencies.get("files"))
        and not dependencies.get("unresolved", ["unknown"])
        and all(
            Path(path).resolve().is_relative_to(Path("assets/PhysicSolver").resolve())
            for path in dependencies.get("files", [])
        )
    )
    if config["job"]["record_rgb"]:
        try:
            report["recording"] = _validate_recording(output, config, rows)
            checks["recording"] = True
        except (ValueError, AssertionError, subprocess.CalledProcessError) as exc:
            checks["recording"] = False
            report["recording_error"] = str(exc)
    report["success"] = all(checks.values())
    report["failed_checks"] = [key for key, value in checks.items() if not value]
    report["source_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("configuration.json", "trajectory.json", "meshes.npz", "cut_events.json")
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _validate_recording(output: Path, config: dict, rows: list[dict]) -> dict:
    """Validate the four knife coordinates without assuming an articulated arm or grasp."""
    import pyarrow.parquet as pq

    root = output / "lerobot_v3"
    info = json.loads((root / "meta/info.json").read_text())
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    episode = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
    fps, frames = info["fps"], info["total_frames"]
    if (
        info["codebase_version"] != "v3.0"
        or info["robot_type"] != "scripted_knife"
        or frames != round(config["planner"]["duration_s"] * fps)
        or len(table) != frames
        or episode["length"] != frames
        or not episode["success"]
        or episode["dataset_use"] != "demonstration"
        or episode["dataset_from_index"] != 0
        or episode["dataset_to_index"] != frames
    ):
        raise ValueError("Incomplete or mislabeled knife dataset")
    np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(frames))
    np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(frames) / fps, atol=2e-6)
    indices = np.rint(np.arange(frames) / fps / config["sim_world"]["physics_dt"]).astype(int)
    expected = np.array([rows[i]["knife"] for i in indices], dtype=np.float32)
    for key in ("observation.state", "action"):
        values = np.array(table[key].to_pylist(), dtype=np.float32)
        if values.shape != (frames, 4) or not np.isfinite(values).all():
            raise ValueError(f"Invalid coordinates: {key}")
        if info["features"][key]["names"] != [
            "knife_x_m",
            "knife_y_m",
            "knife_z_m",
            "knife_yaw_rad",
        ]:
            raise ValueError(f"Incorrect coordinate metadata: {key}")
        np.testing.assert_array_equal(values, expected)
    videos = {}
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    for camera in config["scene"]["cameras"]:
        key = f"observation.images.{camera['name']}"
        path = root / info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
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
                    str(path),
                ]
            )
        )["streams"][0]
        numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
        if (
            stream["codec_name"] != "h264"
            or int(stream["nb_read_frames"]) != frames
            or numerator / denominator != fps
            or [stream["width"], stream["height"]] != camera["resolution"]
            or info["features"][key]["shape"] != [*camera["resolution"][::-1], 3]
            or episode[f"videos/{key}/from_timestamp"] != 0
            or abs(episode[f"videos/{key}/to_timestamp"] - frames / fps) > 1e-6
        ):
            raise ValueError(f"Video metadata mismatch: {key}")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
            check=True,
            capture_output=True,
        )
        sheet = Image.new("RGB", (960, 3 * 200), (24, 24, 24))
        means = []
        for i, time_s in enumerate((0, 2.3, 4.1, 5.9, 7.7, 9.5, 10.2, 11.5, 12.9)):
            data = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    str(time_s),
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ]
            )
            image = Image.open(io.BytesIO(data)).convert("RGB")
            pixels = np.asarray(image)
            means.append(float(pixels.mean()))
            if not 5 < pixels.mean() < 245 or pixels.std() < 10:
                raise ValueError(f"Blank or invalid RGB frame at {time_s}s")
            image.thumbnail((320, 180))
            x, y = (i % 3) * 320, (i // 3) * 200
            sheet.paste(image, (x, y))
            ImageDraw.Draw(sheet).text((x + 8, y + 182), f"{time_s:.1f}s", fill="white")
        sheet.save(qa / f"{camera['name']}_keyframes.jpg", quality=92)
        videos[camera["name"]] = str(path.relative_to(output))
    return dict(frames=frames, fps=fps, videos=videos, state_action_alignment=True)


def main() -> int:
    """Exit nonzero when any independent acceptance item fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = validate(args.output)
    get_logger(__name__).info(
        "Banana independent validation",
        success=report["success"],
        failed_checks=report["failed_checks"],
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
