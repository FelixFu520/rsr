"""Collect a banana case in debug staging; publish only independently verified, reviewed runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from real2sim2real.core.config import apply_settings_file
from real2sim2real.core.logger import get_logger
from real2sim2real.job.job_physic_solver import BananaJobConfig
from scripts.physic_solver.validate_banana_cutting import validate

ROOT = Path(__file__).resolve().parents[2]


def publish(staging: Path, destination: Path) -> None:
    """Publish a reviewed run atomically; preserve all pre-existing accepted results."""
    review = staging / "qa/visual_review.json"
    if not review.is_file() or not json.loads(review.read_text()).get("passed"):
        raise ValueError(f"Inspect the decoded keyframes and record the review in {review}")
    report = validate(staging)
    if not report["success"]:
        raise ValueError("Independent cutting validation failed")
    replay_check = staging / "replay_validation.json"
    if not replay_check.is_file() or not json.loads(replay_check.read_text()).get("success"):
        raise ValueError("Run validate_banana_replay.py before publishing")
    if destination.exists():
        raise FileExistsError(f"Accepted output already exists: {destination}")
    videos = report.get("recording", {}).get("videos", {})
    if "overview" in videos:
        shutil.copy2(staging / videos["overview"], staging / "banana_cutting.mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    validate(destination)
    get_logger(__name__).info("Banana case published", output=str(destination))


def main() -> int:
    """Run the configured fixed lifecycle in a child Kit process and retain its log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/physic_solver/kinematic_tgs_banana01_five_cuts.toml"),
    )
    parser.add_argument(
        "--publish", type=Path, help="Publish an already reviewed staging directory"
    )
    parser.add_argument(
        "--output", type=Path, help="Override destination (or debug capture directory)"
    )
    args = parser.parse_args()
    apply_settings_file(args.config.resolve())
    config = BananaJobConfig.from_settings("job")
    destination = (args.output or Path(config.output_dir)).resolve()
    if args.publish:
        publish(args.publish.resolve(), destination)
        return 0
    formal = destination.is_relative_to(ROOT / "outputs/physic_solver")
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    staging = (
        ROOT / "outputs/debug/physic_solver" / f"{args.config.stem}_{stamp}"
        if formal
        else destination
    )
    log_path = staging.with_suffix(".log")
    staging.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RSR_JOB__OUTPUT_DIR=str(staging))
    with log_path.open("w") as log:
        process = subprocess.run(
            [
                str(ROOT / "app/python.sh"),
                "-m",
                "real2sim2real.core.cli",
                "collect",
                "run",
                "-c",
                str(args.config.resolve()),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode or not (staging / "result.json").is_file():
        raise RuntimeError(f"Capture failed; evidence retained in {staging} and {log_path}")
    if not validate(staging)["success"]:
        raise RuntimeError(f"Independent validation failed: {staging / 'validation.json'}")
    shutil.move(str(log_path), str(staging / "capture.log"))
    get_logger(__name__).info(
        "Banana capture verified; inspect qa before publish",
        staging=str(staging),
        destination=str(destination),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
