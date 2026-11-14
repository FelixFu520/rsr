"""Collect, independently validate and publish a battery episode without overwriting old data.

Run with Isaac python.sh. Kit runs in a child process so fast shutdown cannot skip validation.
All in-progress and failed files stay under outputs/debug; only validated data is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from real2sim2real.core.config import apply_settings_file
from real2sim2real.core.logger import get_logger
from real2sim2real.job.job_battery_disassembly import BatteryDisassemblyJobConfig
from real2sim2real.simworld.isaacsim import IsaacsimWorldConfig
from scripts.battery_disassembly.validate_battery_disassembly import validate

ROOT = Path(__file__).resolve().parents[2]
LOG = get_logger(__name__)


def _write_status(path: Path, status: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _prepare_budget_clock(run: Path, environment: dict[str, str]) -> dict[str, Any]:
    """Build a process-local clock workaround before loading the renderer's native libraries."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("GPU budget clock workaround currently supports Linux x86_64 only")
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("GPU budget clock workaround needs a C compiler (cc)")
    native = run / "capture/native"
    native.mkdir(parents=True)
    source = native / "gpu_budget_clock.c"
    shutil.copyfile(Path(__file__).with_name(source.name), source)
    library = native / "libgpu_budget_clock.so"
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
         str(source), "-o", str(library), "-ldl", "-pthread"],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    previous = environment.get("LD_PRELOAD", "")
    environment["LD_PRELOAD"] = str(library) + (":" + previous if previous else "")
    report = {
        "mode": "monotonic_epoch", "scope": "libgpu.foundation.plugin.so system_clock calls",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "source": "native/gpu_budget_clock.c", "library": "native/libgpu_budget_clock.so",
        "execution_library": str(library),
    }
    (native / "clock.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _progress(log: Path) -> str:
    if not log.is_file():
        return ""
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 65536))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    return next((line for line in reversed(lines) if "拆解进度" in line or "[INFO]" in line), "")


def _video_links(output: Path) -> dict[str, str]:
    info = json.loads((output / "lerobot_v3/meta/info.json").read_text())
    links = {}
    for key, feature in info["features"].items():
        if feature["dtype"] != "video":
            continue
        name = key.removeprefix("observation.images.")
        video = (
            output
            / "lerobot_v3"
            / info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
        )
        link = output / f"{name}.mp4"
        link.symlink_to(video.relative_to(output))
        links[name] = link.name
    return links


def collect(config: Path, run: Path) -> Path:
    """Run one complete episode, validate videos and physics, then publish atomically."""
    config = apply_settings_file(config)
    job = BatteryDisassemblyJobConfig.from_settings("job")
    world = IsaacsimWorldConfig.from_settings("sim_world")
    output = Path(job.output_dir).expanduser().resolve()
    if job.backend != "battery_disassembly":
        raise ValueError("This collector requires the battery_disassembly Job")
    if job.overwrite_dataset or job.recorder.save_failed_episodes or not job.recorder.record_rgb:
        raise ValueError("Use a formal RGB configuration without overwrite or failed episodes")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    run = run.resolve()
    debug = (ROOT / "outputs/debug/battery_disassembly").resolve()
    if not run.is_relative_to(debug) or output.is_relative_to(run):
        raise ValueError("run-dir must be inside outputs/debug/battery_disassembly")
    run.mkdir(parents=True, exist_ok=False)
    staging = run / "capture"
    log = run / "capture.log"
    status_file = run / "status.json"
    status: dict[str, Any] = {
        "state": "starting", "started_utc": datetime.now(UTC).isoformat(),
        "config": str(config), "staging": str(staging), "output": str(output),
        "log": str(log), "renderer": world.renderer, "fps": job.recorder.fps,
        "resolution": [world.width, world.height], "parent_pid": os.getpid(),
        "multi_gpu": world.multi_gpu, "active_gpu": world.active_gpu,
        "renderer_multithreading": world.renderer_multithreading,
        "capture_mode": job.recorder.capture_mode,
        "visual_review": "Initial quality probe inspected; final keyframes pending agent review.",
    }  # fmt: skip
    _write_status(status_file, status)
    environment = os.environ.copy()
    environment["RSR_JOB__OUTPUT_DIR"] = str(staging)
    environment["RSR_JOB__OVERWRITE_DATASET"] = "false"
    environment["RSR_JOB__RECORDER__SAVE_FAILED_EPISODES"] = "false"
    environment["PYTHONUNBUFFERED"] = "1"
    worker = None
    started = time.monotonic()
    try:
        if job.monotonic_render_budget_clock:
            status["render_budget_clock"] = _prepare_budget_clock(run, environment)
            _write_status(status_file, status)
        with log.open("w") as stream:
            worker = subprocess.Popen(
                [sys.executable, "-m", "real2sim2real.core.cli",
                 "collect", "run", "-c", str(config)],
                cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
            )  # fmt: skip
            status.update(state="rendering", worker_pid=worker.pid)
            _write_status(status_file, status)
            while worker.poll() is None:
                time.sleep(30)
                status.update(
                    elapsed_s=round(time.monotonic() - started, 1), progress=_progress(log)
                )
                _write_status(status_file, status)
            if worker.returncode:
                raise RuntimeError(f"Kit worker exited with {worker.returncode}; inspect {log}")
        # Staging is unique to this run, so an old successful result cannot mask a Kit failure.
        result = json.loads((staging / "result.json").read_text())
        if not (result.get("success") and result.get("completed")):
            raise RuntimeError(f"Physical acceptance failed: {result.get('failed_checks')}")
        if result.get("dataset_use") != "demonstration":
            raise RuntimeError("Worker did not produce a demonstration dataset")
        info = json.loads((staging / "lerobot_v3/meta/info.json").read_text())
        dimensions = [world.height, world.width, 3]
        if info["fps"] != job.recorder.fps or any(
            feature["shape"] != dimensions
            for feature in info["features"].values()
            if feature["dtype"] == "video"
        ):
            raise RuntimeError("Captured fps/resolution differs from the requested profile")
        evidence = json.loads(
            (staging / "lerobot_v3/meta/physics/episode_000000/configuration.json").read_text()
        )
        if evidence["sim_world"] != world.model_dump(mode="json"):
            raise RuntimeError("Captured renderer/world settings differ from the requested profile")
        status["state"] = "validating"
        _write_status(status_file, status)
        report = validate(staging)
        if not report["passed"]:
            raise RuntimeError("Independent video/physics validation failed")
        links = _video_links(staging)
        manifest = {
            "config": str(config), "original_staging": str(staging), "output": str(output),
            "renderer": world.renderer, "fps": job.recorder.fps, "frames": result["images"],
            "resolution": [world.width, world.height], "videos": links,
            "multi_gpu": world.multi_gpu, "active_gpu": world.active_gpu,
            "renderer_multithreading": world.renderer_multithreading,
            "capture_mode": job.recorder.capture_mode,
            "render_budget_clock": status.get("render_budget_clock", {"mode": "system"}),
            "physics_passed": True, "independent_validation_passed": True,
            "visual_review": status["visual_review"],
        }  # fmt: skip
        (staging / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        result["dataset"] = str(output / "lerobot_v3")
        (staging / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        outcome_path = staging / "lerobot_v3/meta/task_outcome.json"
        outcome = json.loads(outcome_path.read_text())
        outcome["result"]["dataset"] = result["dataset"]
        outcome_path.write_text(json.dumps(outcome, indent=2) + "\n")
        # Keep archived execution configurations unchanged; they document the actual staging path.
        report["videos"] = {
            name: str(path).replace(str(staging), str(output), 1)
            for name, path in report["videos"].items()
        }
        report["keyframes"] = str(output / "qa/keyframes.jpg")
        (staging / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Output appeared during capture; refusing to overwrite {output}")
        staging.rename(output)
        status.update(
            state="complete", finished_utc=datetime.now(UTC).isoformat(),
            elapsed_s=round(time.monotonic() - started, 1), frames=result["images"],
            videos={name: str(output / path) for name, path in links.items()},
            validation=str(output / "validation.json"),
        )  # fmt: skip
        _write_status(status_file, status)
        LOG.info("电池采集已通过验收并发布", output=str(output), frames=result["images"])
        return output
    except BaseException as exc:
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGINT)
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.terminate()
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        _write_status(status_file, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/battery_disassembly/kr210_physx_c1003b1a1_1k.toml",
    )  # fmt: skip
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir or (
        ROOT
        / "outputs/debug/battery_disassembly"
        / f"capture_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    collect(args.config, run)


if __name__ == "__main__":
    main()
