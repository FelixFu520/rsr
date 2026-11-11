"""Measure temporal illumination stability on a stationary floor patch in the sorting overview."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from real2sim2real.core.logger import get_logger


def overview_luma(video: Path) -> np.ndarray:
    """Decode every frame at a fixed 4:3 analysis size, including frame zero."""
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            "scale=160:120",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    if not decoded or len(decoded) % (160 * 120):
        raise ValueError("Incomplete overview decode")
    frames = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, 120, 160)
    return frames[:, 12:65, 8:35].mean(axis=(1, 2))


def illumination_metrics(values: np.ndarray) -> dict:
    """Report adjacent-frame changes in a fixed background region without motion masking."""
    delta = np.abs(np.diff(values))
    return dict(
        frames=len(values),
        minimum_roi_luma=float(values.min()),
        maximum_roi_luma=float(values.max()),
        maximum_adjacent_roi_luma_jump=float(delta.max()),
        adjacent_roi_luma_jump_q95=float(np.quantile(delta, 0.95)),
        worst_transition=int(delta.argmax()),
    )


def audit_render(output: Path, reference_video: Path) -> dict:
    """Check a complete new capture against the original flickering overview."""
    video = output / "lerobot_v3/videos/observation.images.overview/chunk-000/file-000.mp4"
    info = json.loads((output / "lerobot_v3/meta/info.json").read_text())
    result = json.loads((output / "result.json").read_text())
    measured, reference = overview_luma(video), overview_luma(reference_video)
    new, old = illumination_metrics(measured), illumination_metrics(reference)
    checks = dict(
        complete_overview=new["frames"] == info["total_frames"],
        no_black_initial_or_later_floor_frames=new["minimum_roi_luma"] >= 20,
        bounded_adjacent_illumination=new["maximum_adjacent_roi_luma_jump"] <= 2.0,
        stable_typical_illumination=new["adjacent_roi_luma_jump_q95"] <= 0.5,
        substantially_less_flicker=(
            new["adjacent_roi_luma_jump_q95"] <= old["adjacent_roi_luma_jump_q95"] * 0.25
        ),
        path_traced=result["observation_render_profile"]["renderer"] == "PathTracing",
        no_physics_advanced=result["rendering_physics_steps_advanced"] == 0,
    )
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    report = dict(
        passed=all(checks.values()),
        checks=checks,
        failed_checks=[key for key, value in checks.items() if not value],
        reference=old,
        rendered=new,
        roi_analysis_xywh=[8, 12, 27, 53],
        analysis_resolution=[160, 120],
        luma_range=[0, 255],
        scope=(
            "All overview frames; a fixed stationary floor region. Does not certify every "
            "reflective surface or replace visual review."
        ),
    )
    (qa / "illumination_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(
        qa / "overview_floor_luma.csv",
        np.column_stack((np.arange(len(measured)) / info["fps"], measured)),
        delimiter=",",
        header="time_s,roi_luma_0_255",
        comments="",
    )
    get_logger(__name__).info("Sorting illumination audit", **report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--reference-video",
        type=Path,
        default=Path(
            "outputs/sorting_stream/lerobot_v3/videos/observation.images.overview/chunk-000/file-000.mp4"
        ),
    )
    arguments = parser.parse_args()
    raise SystemExit(
        0 if audit_render(arguments.output, arguments.reference_video)["passed"] else 1
    )
