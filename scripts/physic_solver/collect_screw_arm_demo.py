"""Capture the UR5e screw demo in staging, independently validate, then publish once."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from real2sim2real.core.config import BaseConfig, apply_settings_file
from real2sim2real.core.logger import get_logger
from real2sim2real.job.job_physic_solver import ScrewJobConfig
from real2sim2real.robot.robot_physic_solver import (
    DualUr5eScrewRobotConfig,
    Ur5eScrewdriverRobotConfig,
)
from scripts.battery_disassembly.collect_battery_disassembly import _prepare_budget_clock
from scripts.physic_solver.validate_screw_assembly import validate

ROOT = Path(__file__).resolve().parents[2]


def _demo_filter(configuration: dict) -> str:
    """Label the dual-arm operation stages without changing either recorded camera stream."""
    base = "[0:v][1:v]hstack=inputs=2"
    if configuration["robot"]["backend"] != "dual_ur5e_screwdriver":
        return base + "[v]"
    base += ",pad=iw:ih+52:0:52:color=0x101820"
    from real2sim2real.utils.physic_solver_utils import dual_screw_phases

    start = 0.0
    labels = dict(
        clamp="1  CLAMP NUT",
        lift="2  LIFT AND HOLD NUT",
        align="3  ALIGN ELECTRIC DRIVER",
        engage="4  ENGAGE THREAD",
        screw="5  SCREW IN BOLT",
        hold="6  STOP AND HOLD",
        release="7  OPEN NUT GRIPPER",
        released="8  NUT RELEASED - DRIVER STOPPED",
    )
    if configuration["planner"].get("release_target", "nut") == "bolt":
        labels.update(
            clamp="1  CLAMP NUT AND BOLT",
            open_bolt="7  OPEN BOLT CLAMP - KEEP NUT GRIPPED",
            release="8  WITHDRAW DRIVER - KEEP NUT GRIPPED",
            released="9  BOLT RELEASED - NUT STILL HELD",
        )
    for name, duration in dual_screw_phases(configuration["planner"]):
        label = labels[name]
        if name == "hold" and configuration["planner"].get("minimum_tip_protrusion_m", 0) > 0:
            label = "6  BOLT THROUGH NUT - STOP AND HOLD"
        end = start + duration
        base += (
            f",drawtext=font='DejaVu Sans':text='DUAL UR5e  |  {label}':"
            f"x=20:y=14:fontsize=24:fontcolor=white:enable='gte(t,{start})*lt(t,{end})'"
        )
        start = end
    return base + "[v]"


def publish(run: Path) -> Path:
    """Publish a validated capture after its keyframes have actually been reviewed."""
    run = run.resolve()
    status = run / "status.json"
    manifest = json.loads(status.read_text())
    staging, output = run / "capture", Path(manifest["output"])
    if output.exists():
        raise FileExistsError(output)
    review = staging / "qa/visual_review.json"
    if not review.exists() or not json.loads(review.read_text()).get("passed"):
        raise ValueError(f"Inspect both videos and keyframes, then record the review in {review}")
    if not validate(staging)["success"]:
        raise RuntimeError("Independent capture validation failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(output))
    for name in ("result.json", "validation.json"):
        path = output / name
        path.write_text(path.read_text().replace(str(staging), str(output)))
    manifest.update(state="published", validation="passed", visual_inspection="passed")
    (output / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    status.write_text(json.dumps(manifest, indent=2) + "\n")
    get_logger(__name__).info(
        "UR5e demo published", output=str(output), preview=str(output / "demo.mp4")
    )
    return output


def collect(config: Path, run: Path | None = None) -> Path:
    """Capture and validate one episode, retaining staging for actual visual review."""
    config = config.resolve()
    apply_settings_file(config)
    job = ScrewJobConfig.from_settings("job")
    robot_cls = (
        DualUr5eScrewRobotConfig
        if BaseConfig.peek_backend("robot") == "dual_ur5e_screwdriver"
        else Ur5eScrewdriverRobotConfig
    )
    robot = robot_cls.from_settings("robot")
    if robot.backend not in {"ur5e_screwdriver", "dual_ur5e_screwdriver"} or not job.record_rgb:
        raise ValueError("This collector requires the UR5e screwdriver RGB configuration")
    output = Path(job.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Select a fresh output; refusing to overwrite {output}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = (run or ROOT / "outputs/debug/physic_solver" / f"ur5e_capture_{stamp}").resolve()
    if not run.is_relative_to(ROOT / "outputs/debug/physic_solver"):
        raise ValueError("Run directory must be under outputs/debug/physic_solver")
    run.mkdir(parents=True, exist_ok=False)
    staging = run / "capture"
    environment = os.environ.copy()
    environment.update(RSR_JOB__output_dir=str(staging), PYTHONUNBUFFERED="1")
    clock = _prepare_budget_clock(run, environment)
    manifest = {
        "config": str(config),
        "output": str(output),
        "run": str(run),
        "render_budget_clock": clock,
        "state": "capturing",
    }
    status = run / "status.json"
    status.write_text(json.dumps(manifest, indent=2) + "\n")
    get_logger(__name__).info(
        "UR5e demo capturing", log=str(run / "capture.log"), output=str(output)
    )
    try:
        with (run / "capture.log").open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "real2sim2real.core.cli",
                    "collect",
                    "run",
                    "-c",
                    str(config),
                ],
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        report = validate(staging)
        if not report["success"]:
            raise RuntimeError(f"UR5e demo failed independent validation: {report['checks']}")
        info = json.loads((staging / "lerobot_v3/meta/info.json").read_text())
        for key, feature in info["features"].items():
            if feature["dtype"] == "video":
                name = key.removeprefix("observation.images.")
                relative = Path("lerobot_v3") / info["video_path"].format(
                    video_key=key, chunk_index=0, file_index=0
                )
                (staging / f"{name}.mp4").symlink_to(relative)
        # A portable synchronized side-by-side preview at the source frame rate.
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(staging / "overview.mp4"),
                "-i",
                str(staging / "threads.mp4"),
                "-filter_complex",
                _demo_filter(json.loads((staging / "configuration.json").read_text())),
                "-map",
                "[v]",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(staging / "demo.mp4"),
            ],
            check=True,
        )
        manifest.update(state="awaiting_visual_review", validation="passed")
        status.write_text(json.dumps(manifest, indent=2) + "\n")
        get_logger(__name__).info(
            "UR5e demo ready for visual review",
            staging=str(staging),
            preview=str(staging / "demo.mp4"),
        )
        return staging
    except Exception as exc:
        manifest.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        status.write_text(json.dumps(manifest, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/physic_solver/mujoco_hydroelastic_screw02_ur5e.toml"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--publish-run", type=Path, help="Publish an already captured and visually reviewed run"
    )
    args = parser.parse_args()
    if args.publish_run:
        publish(args.publish_run)
    else:
        collect(args.config, args.run_dir)
