#!/usr/bin/env python3
"""Record the six Franka cases serially with the installed Isaac Sim runtime.

Run from any directory with ``./app/python.sh scripts/record_franka_demos.py``.
Optional positional arguments select config paths or names such as
``franka_rigid_physx franka_cloth_hybrid``. Grasp acceptance and successful video
recording are reported separately, including the unified XPBD diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from real2sim2real.core.logger import get_logger

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    "franka_rigid_physx",
    "franka_cloth_newton",
    "franka_cloth_hybrid",
    "franka_cloth_newton_mujoco",
    "franka_cloth_newton_xpbd",
    "franka_cloth_newton_xpbd_unified",
)


def _config_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    candidates = [candidate, ROOT / candidate, ROOT / "configs" / candidate]
    if candidate.suffix != ".toml":
        candidates.append(ROOT / "configs" / f"{value}.toml")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise ValueError(f"Configuration not found: {value}")


def _output_dir(config: Path) -> Path:
    """Read the inherited output path without loading a simulation backend."""
    output: str | None = None
    seen: set[Path] = set()

    def visit(path: Path) -> None:
        nonlocal output
        path = path.expanduser().resolve()
        if path in seen:
            return
        seen.add(path)
        with path.open("rb") as stream:
            data = tomllib.load(stream)
        parents = data.get("inherit", data.get("dynaconf_include", []))
        if isinstance(parents, str):
            parents = [parents]
        for parent in parents or []:
            visit(path.parent / parent)
        if "output_dir" in data.get("job", {}):
            output = str(data["job"]["output_dir"])

    visit(config)
    if not output:
        raise ValueError(f"No [job].output_dir found in {config} or its parent configs")
    path = Path(output).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_json(path: Path, started_ns: int) -> dict[str, Any]:
    if path.stat().st_mtime_ns < started_ns - 1_000_000_000:
        raise ValueError(f"Stale artifact from an earlier recording: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _probe_video(path: Path) -> dict[str, Any] | None:
    """Decode video statistics when ffprobe is available; reject invalid streams."""
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    value = json.loads(process.stdout)
    streams = value.get("streams", [])
    if not streams:
        raise ValueError(f"No decodable video stream in {path}")
    stream = streams[0]
    result = {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "frame_count": int(stream.get("nb_read_frames", 0)),
        "frame_rate": stream.get("r_frame_rate"),
        "duration_s": float(value.get("format", {}).get("duration", 0)),
    }
    if any(result[key] <= 0 for key in ("width", "height", "frame_count", "duration_s")):
        raise ValueError(f"Incomplete video stream in {path}: {result}")
    return result


def _collect_artifacts(
    output: Path, video_dir: Path, name: str, started_ns: int, record: dict[str, Any]
) -> None:
    result = _read_json(output / "result.json", started_ns)
    metadata = _read_json(output / "video.json", started_ns)
    record.update(
        completed=bool(result.get("completed")),
        grasp_success=bool(result.get("success")),
        result_path=str(output / "result.json"),
        video_metadata=metadata,
    )
    if not record["completed"]:
        record["errors"].append("Simulation did not complete the grasp sequence")
    if int(metadata.get("frame_count", 0)) <= 0 or float(metadata.get("fps", 0)) <= 0:
        raise ValueError(f"Video metadata has no positive frame count or FPS: {output}")
    video = Path(metadata.get("path") or output / "grasp.mp4").expanduser()
    if not video.is_absolute():
        video = ROOT / video
    if not video.is_file() or video.stat().st_size == 0:
        raise ValueError(f"Missing or empty video: {video}")
    if video.stat().st_mtime_ns < started_ns - 1_000_000_000:
        raise ValueError(f"Stale video from an earlier recording: {video}")
    probe = _probe_video(video)
    record["ffprobe"] = probe
    if probe:
        if probe["frame_count"] != int(metadata["frame_count"]):
            record["errors"].append("Video frame count differs from video.json")
        duration = float(metadata.get("duration_s", 0))
        if abs(probe["duration_s"] - duration) > max(0.1, 1 / float(metadata["fps"])):
            record["errors"].append("Video duration differs from video.json")
    destination = video_dir / f"{name}.mp4"
    if video.resolve() != destination.resolve():
        shutil.copy2(video, destination)
    record["video"] = str(destination)
    record["recording_complete"] = not record["errors"]


def _record_case(
    config: Path, output: Path, video_dir: Path, *, skip_existing: bool = False
) -> dict[str, Any]:
    log = get_logger(__name__)
    log_path = ROOT / "outputs" / "recording_logs" / f"{config.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ROOT / "app" / "python.sh"),
        "-m",
        "real2sim2real.core.cli",
        "collect",
        "run",
        "-c",
        str(config),
    ]
    environment = os.environ.copy()
    environment.update(
        RSR_SIM_WORLD__HEADLESS="true",
        RSR_JOB__RECORD_VIDEO="true",
        RSR_JOB__FAIL_ON_UNSUCCESSFUL="false",
        RSR_JOB__TIMEOUT_S="7200",
        RSR_PLANNER__TIMEOUT_S="7200",
        RSR_JOB__OUTPUT_DIR=str(output),
    )
    started_ns = time.time_ns()
    record: dict[str, Any] = {
        "config": str(config),
        "output_dir": str(output),
        "log": str(log_path),
        "started_at": datetime.now(UTC).isoformat(),
        "expected_diagnostic": config.stem == "franka_cloth_newton_xpbd_unified",
        "recording_complete": False,
        "grasp_success": None,
        "errors": [],
    }
    if skip_existing:
        existing = {**record, "errors": [], "skipped_existing": True, "returncode": None}
        try:
            _collect_artifacts(output, video_dir, config.stem, 0, existing)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        else:
            if existing["recording_complete"]:
                existing["elapsed_s"] = (time.time_ns() - started_ns) / 1e9
                log.info("Using verified existing Franka recording", config=config.name)
                return existing
    log.info("Recording Franka case", config=config.name, log=str(log_path))
    try:
        # Each subprocess exits before another case starts; never launch Kit in parallel.
        with log_path.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
        record["returncode"] = process.returncode
        if process.returncode:
            record["errors"].append(f"Simulation exited with code {process.returncode}")
        _collect_artifacts(output, video_dir, config.stem, started_ns, record)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    record["elapsed_s"] = (time.time_ns() - started_ns) / 1e9
    log.info(
        "Franka recording finished",
        config=config.name,
        recording_complete=record["recording_complete"],
        grasp_success=record["grasp_success"],
        errors=record["errors"],
    )
    return record


def main() -> int:
    """Record requested configs serially and write a reviewable video index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", help="TOML paths or config names; default: all six")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Reuse completed recordings after video checks"
    )
    arguments = parser.parse_args()
    try:
        configs = [_config_path(value) for value in arguments.configs or DEFAULT_CONFIGS]
        if len({config.stem for config in configs}) != len(configs):
            raise ValueError("Config names must be unique for video and log filenames")
        outputs = [_output_dir(config) for config in configs]
        if len(set(outputs)) != len(outputs):
            raise ValueError("Each config must use a different [job].output_dir")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    video_dir = ROOT / "outputs" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    summary_path = video_dir / "recording_summary.json"
    summary: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "serial": True,
        "cases": [],
        "recording_complete": False,
    }
    for config, output in zip(configs, outputs, strict=True):
        summary["cases"].append(
            _record_case(config, output, video_dir, skip_existing=arguments.skip_existing)
        )
        summary["recording_complete"] = len(summary["cases"]) == len(configs) and all(
            case["recording_complete"] for case in summary["cases"]
        )
        summary["updated_at"] = datetime.now(UTC).isoformat()
        temporary = summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(summary_path)
    get_logger(__name__).info("Franka recording summary", path=str(summary_path))
    return 0 if summary["recording_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
