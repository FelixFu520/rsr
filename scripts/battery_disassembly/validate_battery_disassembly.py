"""Independently check a battery disassembly output directory and its LeRobot v3 dataset (no Kit).

Usage: ./app/python.sh scripts/battery_disassembly/validate_battery_disassembly.py <output_dir>

Checks, from the published files only:

1. LeRobot v3 layout, H.264 videos of all three cameras (frame count, fps, size, full decode)
   via ``utils.lerobot_v3.validate_dataset``; no black frames in the sampled images;
2. every data row equals the archived physics state log at its physics step (``8 i`` at 240 Hz
   and 30 fps; ``16 i`` for 15 fps debug), including screw positions, thread slide and cover pose;
3. the physical acceptance recomputed from ``meta/physics/episode_000000`` equals ``result.json``.

Writes ``validation.json`` and ``qa/keyframes.jpg`` next to ``result.json``; exits non-zero when
a check fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from real2sim2real.job.job_battery_disassembly import (
    BatteryAcceptanceConfig,
    evaluate_battery,
    layout_from_json,
)
from real2sim2real.scene.scene_battery_disassembly import BatteryWorkcellSceneConfig
from real2sim2real.utils.lerobot_v3 import validate_dataset

CAMERAS = ("overview", "screwdriver", "gripper")


def _frames(video: Path, indices: list[int], width: int, height: int) -> list[np.ndarray]:
    """Decode selected frames (RGB uint8)."""
    select = "+".join(f"eq(n\\,{i})" for i in indices)
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='{select}'", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    )  # fmt: skip
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    if len(frames) != len(indices):
        raise ValueError(f"Decoded {len(frames)} of {len(indices)} frames from {video}")
    return list(frames)


def validate(output: Path) -> dict:
    result = json.loads((output / "result.json").read_text())
    physics = output / "lerobot_v3/meta/physics/episode_000000"
    configuration = json.loads((physics / "configuration.json").read_text())
    resolution = {
        c["name"]: tuple(c["resolution"])
        for c in [*configuration["scene"]["cameras"], *configuration["robot"]["cameras"]]
    }
    report = validate_dataset(output, expected_cameras=resolution)
    root = output / "lerobot_v3"
    info = json.loads((root / "meta/info.json").read_text())
    data = pq.read_table(root / info["data_path"].format(chunk_index=0, file_index=0))
    log = dict(np.load(physics / "state_log.npz"))
    world = configuration["sim_world"]
    job = configuration["job"]
    hz = round(1.0 / world["physics_dt"])
    stride = hz // info["fps"]
    log_stride = job["recorder"]["log_stride"]
    if stride % log_stride:
        raise ValueError("Image stride must be a multiple of the state-log stride")
    steps = log["step"].astype(int)
    frames = len(data)
    rows = [int(np.nonzero(steps == stride * i)[0][0]) for i in range(frames)]
    state = np.array(data["observation.state"].to_pylist())
    expected = np.column_stack([
        log["screw_q"][rows], log["grip_q"][rows], log["float"][rows], log["spindle"][rows],
        log["finger_p"][rows], log["finger_n"][rows],
    ])  # fmt: skip
    errors = {"observation.state": float(np.max(np.abs(state - expected)))}
    action = np.array(data["action"].to_pylist())
    errors["action"] = float(np.max(np.abs(action - log["action"][rows])))
    errors["observation.screw_positions"] = float(
        np.max(
            np.abs(
                np.array(data["observation.screw_positions"].to_pylist()) - log["screw_pos"][rows]
            )
        )
    )
    errors["observation.thread_slide_m"] = float(
        np.max(
            np.abs(
                np.array(data["observation.thread_slide_m"].to_pylist()) - log["thread_slide"][rows]
            )
        )
    )
    errors["observation.cover_pose"] = float(
        np.max(np.abs(np.array(data["observation.cover_pose"].to_pylist()) - log["cover"][rows]))
    )
    # float32 storage of float64 logs
    rows_match = all(value < 2e-4 for value in errors.values())
    layout = layout_from_json(json.loads((physics / "layout.json").read_text()))
    contacts = json.loads((physics / "contacts.json").read_text())
    recomputed = evaluate_battery(
        log,
        planner_result=result["planner"],
        scene_layout=layout,
        scene_config=BatteryWorkcellSceneConfig.model_validate(configuration["scene"]),
        acceptance=BatteryAcceptanceConfig.model_validate(job["acceptance"]),
        contacts=contacts["impulses_ns"],
        contact_first_time=contacts["first_time_s"],
        physics_dt=world["physics_dt"],
    )
    acceptance_match = recomputed["checks"] == result["checks"]
    # keyframes and black-frame check
    picks = sorted({0, frames // 4, frames // 2, 3 * frames // 4, frames - 1})
    tiles, dark = [], {}
    for name in CAMERAS:
        width, height = resolution[name]
        video = Path(report["videos"][f"observation.images.{name}"])
        images = _frames(video, picks, width, height)
        dark[name] = float(min(image.mean() for image in images))
        tiles.append(
            np.concatenate(
                [np.asarray(Image.fromarray(i).resize((320, 240))) for i in images], axis=1
            )
        )
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    Image.fromarray(np.concatenate(tiles, axis=0)).save(qa / "keyframes.jpg", quality=90)
    not_black = all(value > 10.0 for value in dark.values())
    summary = {
        **report,
        "rows_match_physics_log": rows_match,
        "row_errors": errors,
        "acceptance_recomputed_match": acceptance_match,
        "recomputed_failed_checks": recomputed["failed_checks"],
        "minimum_mean_brightness": dark,
        "no_black_frames": not_black,
        "keyframes": str(qa / "keyframes.jpg"),
        "keyframe_indices": picks,
        "passed": bool(report["validated"] and rows_match and acceptance_match and not_black),
    }
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summary = validate(parser.parse_args().output.resolve())
    keys = (
        "passed", "frames", "rows_match_physics_log", "acceptance_recomputed_match",
        "no_black_frames", "row_errors", "minimum_mean_brightness", "keyframes",
    )  # fmt: skip
    print(json.dumps({k: summary[k] for k in keys}, indent=2))
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
