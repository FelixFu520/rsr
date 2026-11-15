"""Collect tabletop episodes with the GPU clock workaround, validate, then publish.

Run with Isaac python.sh. In-progress and failed captures stay under outputs/debug.
Only independently validated episodes move to the configured formal output directory.
The Linux x86_64 clock workaround is shared with the battery task; it needs cc.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from real2sim2real.core.config import BaseConfig, apply_settings_file
from real2sim2real.core.logger import get_logger
from real2sim2real.job.job_tabletop_pickplace import TabletopPickPlaceJobConfig
from real2sim2real.job.job_tabletop_replay import TabletopReplayJobConfig
from real2sim2real.simworld.isaacsim import IsaacsimWorldConfig
from scripts.battery_disassembly.collect_battery_disassembly import (
    _prepare_budget_clock,
    _progress,
    _write_status,
)

ROOT = Path(__file__).resolve().parents[2]
LOG = get_logger(__name__)


def _rebase_receipts(staging: Path, output: Path) -> None:
    """Update published locations, retaining archived execution configurations unchanged."""

    def rebase(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: v if k == "configuration" else rebase(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rebase(v) for v in value]
        if isinstance(value, str) and (
            value == str(staging) or value.startswith(str(staging) + "/")
        ):
            return str(output) + value[len(str(staging)) :]
        return value

    files = [staging / "result.json", staging / "validation.json"]
    for pattern in (
        "episode_*/result.json",
        "episode_*/validation.json",
        "episode_*/lerobot_v3/meta/task_outcome.json",
        "diagnostics/*/run_result.json",
        "diagnostics/*/episode_*/result.json",
    ):
        files.extend(staging.glob(pattern))
    for path in files:
        path.write_text(json.dumps(rebase(json.loads(path.read_text())), indent=2) + "\n")


def collect(config: Path, run: Path | None = None, *, monotonic_clock: bool = True) -> Path:
    """Run either tabletop Job, independently validate all episodes, and publish once."""
    config = config.resolve()
    apply_settings_file(config)
    backend = BaseConfig.peek_backend("job")
    if backend not in ("tabletop_pickplace", "tabletop_replay"):
        raise ValueError("This collector requires a tabletop_pickplace or tabletop_replay Job")
    cls = TabletopReplayJobConfig if backend == "tabletop_replay" else TabletopPickPlaceJobConfig
    job = cls.from_settings("job")
    world = IsaacsimWorldConfig.from_settings("sim_world")
    output = Path(job.output_dir).expanduser().resolve()
    debug = (ROOT / "outputs/debug" / backend).resolve()
    if job.overwrite_dataset or job.recorder.save_failed_episodes or not job.recorder.record_rgb:
        raise ValueError("Use a formal RGB configuration without overwrite or failed episodes")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if output.is_relative_to((ROOT / "outputs/debug").resolve()):
        raise ValueError("The collector publishes formal results; use the CLI for debug runs")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = (run or debug / f"{config.stem}_{stamp}").resolve()
    if not run.is_relative_to(debug) or run == debug or output.is_relative_to(run):
        raise ValueError(f"run-dir must be a new directory below {debug}")
    run.mkdir(parents=True, exist_ok=False)
    staging = run / "episodes"
    log = run / "capture.log"
    status_path = run / "status.json"
    status = {
        "state": "starting",
        "config": str(config),
        "output": str(output),
        "staging": str(staging),
        "log": str(log),
        "parent_pid": os.getpid(),
        "started_utc": datetime.now(UTC).isoformat(),
    }
    environment = os.environ.copy()
    # Nested field spelling must match pydantic (dynaconf preserves nested key case).
    environment.update(
        RSR_JOB__output_dir=str(staging),
        RSR_JOB__overwrite_dataset="false",
        RSR_JOB__failed_dir=str(run / "failed"),
        RSR_JOB__RECORDER__save_failed_episodes="false",
        PYTHONUNBUFFERED="1",
    )
    worker = None
    started = time.monotonic()
    try:
        if monotonic_clock:
            status["render_budget_clock"] = _prepare_budget_clock(run, environment)
            status["render_budget_clock"]["artifact_dir"] = str(run / "capture")
        else:
            status["render_budget_clock"] = {"mode": "system"}
        _write_status(status_path, status)
        with log.open("w") as stream:
            worker = subprocess.Popen(
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
            )
            status.update(state="rendering", worker_pid=worker.pid)
            _write_status(status_path, status)
            while worker.poll() is None:
                time.sleep(5)
                status.update(
                    elapsed_s=round(time.monotonic() - started, 1), progress=_progress(log)
                )
                _write_status(status_path, status)
            if worker.returncode:
                raise RuntimeError(f"Kit worker exited with {worker.returncode}; inspect {log}")
        result = json.loads((staging / "result.json").read_text())
        if not result.get("success") or not result.get("episodes"):
            raise RuntimeError(f"Physical acceptance failed: {result.get('error')}")
        for episode in result["episodes"]:
            path = staging / f"episode_{episode['episode']:03d}" / "result.json"
            evidence = json.loads(path.read_text())
            if not evidence.get("success") or evidence.get("dataset_use") != "demonstration":
                raise RuntimeError(f"Episode did not produce a demonstration: {path}")
        status["state"] = "validating"
        _write_status(status_path, status)
        if backend == "tabletop_replay":
            from scripts.tabletop_replay.validate_tabletop_replay import validate
        else:
            from scripts.tabletop_pickplace.validate_tabletop_pickplace import validate
        report = validate(staging)
        if not report.get("validated"):
            raise RuntimeError("Independent validation failed")
        (staging / "capture_manifest.json").write_text(
            json.dumps(
                {
                    "config": str(config),
                    "original_staging": str(staging),
                    "output": str(output),
                    "render_budget_clock": status["render_budget_clock"],
                    "physics_passed": True,
                    "independent_validation_passed": True,
                    "visual_review": "pending_visual_inspection",
                    "world": world.model_dump(mode="json"),
                },
                indent=2,
            )
            + "\n"
        )
        _rebase_receipts(staging, output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Output appeared during capture: {output}")
        staging.rename(output)
        status.update(
            state="complete",
            finished_utc=datetime.now(UTC).isoformat(),
            elapsed_s=round(time.monotonic() - started, 1),
        )
        _write_status(status_path, status)
        LOG.info(
            "桌面采集已通过独立验收并发布", output=str(output), episodes=len(result["episodes"])
        )
        return output
    except BaseException as exc:
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGINT)
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.terminate()
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        _write_status(status_path, status)
        raise


def main(default_task: str = "tabletop_pickplace") -> None:
    """Command-line entry shared by both tabletop collectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / default_task / "franka_physx_home000.toml"
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--system-budget-clock",
        action="store_true",
        help="Disable the Linux x86_64 GPU clock workaround",
    )
    args = parser.parse_args()
    collect(args.config, args.run_dir, monotonic_clock=not args.system_budget_clock)


if __name__ == "__main__":
    main()
