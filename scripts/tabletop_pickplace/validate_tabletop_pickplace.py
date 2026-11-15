"""Independently check a tabletop pick-and-place output directory (no Kit).

Usage: ./app/python.sh scripts/tabletop_pickplace/validate_tabletop_pickplace.py <output_dir>

For every ``episode_XXX`` below ``<output_dir>``, from the published files only:

1. LeRobot v3 layout and H.264 videos of both cameras (frame count, fps, size, full decode) via
   ``utils.lerobot_v3.validate_dataset``; sampled frames are not black;
2. every data row equals the archived full-rate state log at its physics step
   (``4 i`` at 120 Hz / 30 fps, ``8 i`` at 15 fps): joints, action, object and hand poses, phase;
3. the physical acceptance recomputed from ``meta/physics/episode_000000`` equals ``result.json``.

Writes ``validation.json`` per episode, ``qa/keyframes.jpg`` per episode and a summary
``validation.json`` in ``<output_dir>``; exits non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from real2sim2real.utils.lerobot_v3 import validate_dataset
from real2sim2real.utils.tabletop_pickplace_utils import (
    PHASE_INDEX,
    PickPlaceAcceptanceConfig,
    evaluate_pickplace,
)

KEYFRAME_PHASES = ("start", "descend", "close", "transfer", "open", "rest")


def decode_frames(video: Path, indices: list[int], width: int, height: int) -> list[np.ndarray]:
    """Decode selected frames (RGB uint8) with ffmpeg."""
    select = "+".join(f"eq(n\\,{i})" for i in indices)
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='{select}'", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    )  # fmt: skip
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    if len(frames) != len(indices):
        raise ValueError(f"Decoded {len(frames)} of {len(indices)} frames from {video}")
    return list(frames)


def _compare(name: str, actual: np.ndarray, expected: np.ndarray, tol: float) -> float:
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: data shape {actual.shape} differs from {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{name}: non-finite values in data rows or physics log")
    error = float(np.max(np.abs(actual - expected)))
    if error > tol:
        raise ValueError(f"{name}: data rows differ from the physics log by {error:.3g}")
    return error


def validate_episode(episode: Path) -> dict[str, Any]:
    """Validate one ``episode_XXX`` directory; returns its report."""
    result = json.loads((episode / "result.json").read_text())
    root = episode / "lerobot_v3"
    physics = root / "meta/physics/episode_000000"
    configuration = json.loads((physics / "configuration.json").read_text())
    cameras = {
        c["name"]: tuple(c["resolution"])
        for c in [*configuration["scene"]["cameras"], *configuration["robot"]["cameras"]]
    }
    report = validate_dataset(episode, expected_cameras=cameras)
    info = json.loads((root / "meta/info.json").read_text())
    data = pq.read_table(root / info["data_path"].format(chunk_index=0, file_index=0))
    log = dict(np.load(physics / "state_log.npz"))
    stride = round(1.0 / configuration["sim_world"]["physics_dt"]) // info["fps"]
    steps = log["step"].astype(int)
    rows = [int(np.nonzero(steps == stride * i)[0][0]) for i in range(len(data))]

    def column(name: str) -> np.ndarray:
        return np.asarray(data[name].to_pylist(), dtype=float)

    errors = {
        "state": _compare("observation.state", column("observation.state"), log["q"][rows], 1e-5),
        "action": _compare("action", column("action"), log["action"][rows], 1e-5),
        "object_pose": _compare(
            "observation.object_pose",
            column("observation.object_pose"),
            np.column_stack([log["object_pos"][rows], log["object_quat"][rows]]),
            1e-5,
        ),
        "ee_pose": _compare(
            "observation.ee_pose",
            column("observation.ee_pose"),
            np.column_stack([log["hand_pos"][rows], log["hand_quat"][rows]]),
            1e-5,
        ),
        "phase": _compare(
            "observation.phase_index",
            column("observation.phase_index"),
            log["phase_index"][rows],
            0,
        ),
    }
    layout = json.loads((physics / "layout.json").read_text())
    contacts = json.loads((physics / "contacts.json").read_text())
    acceptance = PickPlaceAcceptanceConfig(**configuration["job"]["acceptance"])
    recomputed = evaluate_pickplace(
        log,
        layout=layout,
        contacts=contacts["impulses_ns"],
        acceptance=acceptance,
        sequence_completed=result["planner"]["sequence_completed"],
    )
    stored = {k: v for k, v in result["checks"].items() if k in recomputed["checks"]}
    if recomputed["checks"] != stored:
        raise ValueError(f"Recomputed checks differ: {recomputed['checks']} vs {stored}")
    if not recomputed["success"]:
        raise ValueError(f"Physical acceptance failed: {recomputed['failed_checks']}")
    report["keyframes"] = _keyframes(episode, root, info, data, cameras)
    report.update(
        episode=result["episode"],
        row_errors=errors,
        recomputed_checks=recomputed["checks"],
        metrics=recomputed["metrics"],
        visual_review="pending_visual_inspection",
    )
    (episode / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def _keyframes(
    episode: Path, root: Path, info: dict, data: Any, cameras: dict[str, tuple[int, int]]
) -> dict[str, Any]:
    """First frame of selected phases for every camera; rejects black frames."""
    phase = np.asarray(data["observation.phase_index"].to_pylist())
    present = [p for p in KEYFRAME_PHASES if (phase == PHASE_INDEX[p]).any()]
    indices = [int(np.argmax(phase == PHASE_INDEX[p])) for p in present]
    indices[-1] = len(phase) - 1
    rows, brightness = [], {}
    for name, (width, height) in cameras.items():
        key = f"observation.images.{name}"
        video = root / info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
        frames = decode_frames(video, indices, width, height)
        brightness[name] = [round(float(f.mean()), 1) for f in frames]
        if min(brightness[name]) < 5:
            raise ValueError(f"{name}: black frame among {indices}")
        thumbs = [np.asarray(Image.fromarray(f).resize((320, 240))) for f in frames]
        rows.append(np.concatenate(thumbs, axis=1))
    qa = episode / "qa"
    qa.mkdir(exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(qa / "keyframes.jpg", quality=90)
    return {"frames": indices, "mean_brightness": brightness, "image": str(qa / "keyframes.jpg")}


def validate(output: Path) -> dict[str, Any]:
    """Validate every episode below ``output`` and the run summary."""
    run = json.loads((output / "result.json").read_text())
    episodes = sorted(p for p in output.glob("episode_*") if (p / "lerobot_v3").is_dir())
    if not run["success"] or len(episodes) != run["num_episodes"]:
        raise ValueError(f"Run did not pass: {run.get('error')} ({len(episodes)} datasets)")
    reports = [validate_episode(p) for p in episodes]
    summary = {
        "validated": True,
        "episodes": [
            {
                "episode": r["episode"],
                "frames": r["frames"],
                "dataset": r["dataset"],
                "place_error_m": r["metrics"]["place_error_m"],
                "hold_slip_m": r["metrics"]["hold_slip_m"],
            }
            for r in reports
        ],
        "visual_review": "pending_visual_inspection",
    }
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        summary = validate(args.output)
    except Exception as exc:  # noqa: BLE001 - CLI reports the failure and exits non-zero
        print(f"validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
