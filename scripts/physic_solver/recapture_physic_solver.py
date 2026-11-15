"""Fresh, resumable 640x480 collection using the canonical flat experiment filenames.

Run normally to collect; use --status to inspect and --approve CASE... only after actual visual QA.
Pure physics workers use separate visible GPUs; RTX recording is serialized on GPU0.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from real2sim2real.core.logger import get_logger
from real2sim2real.utils.lerobot_v3 import validate_dataset
from real2sim2real.utils.physic_solver_utils import (
    PHYSIC_SOLVER_CASE_GROUPS,
    PHYSIC_SOLVER_CASE_NAMES,
    PHYSIC_SOLVER_PACKAGE_CASES,
    physic_solver_case_name,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/physic_solver"
DEBUG = ROOT / "outputs/debug/physic_solver"
POINTER = DEBUG / "recapture/current_run.json"
LOG = get_logger(__name__)


def read(path: Path) -> dict:
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(0.05)
    raise RuntimeError("Unreachable JSON read")


def update(state: Path, name: str | None = None, **values) -> dict:
    """Merge one update under a process-wide file lock, including manual review commands."""
    with state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read(state)
        target = data if name is None else data["cases"][name]
        target.update(values)
        data["updated_utc"] = datetime.now(UTC).isoformat()
        temp = state.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        temp.replace(state)
    return data


def initialize() -> tuple[Path, Path]:
    if not POINTER.exists():
        if OUTPUT.exists() and any(OUTPUT.iterdir()):
            raise FileExistsError(
                "Formal outputs exist without a run pointer; preserve them and restore the "
                "matching recapture/current_run.json before resuming."
            )
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        run = DEBUG / "recapture" / run_id
        snapshots = run / "prepared_configs"
        snapshots.mkdir(parents=True)
        paths = sorted((ROOT / "configs/physic_solver").glob("*.toml"))
        assert {p.stem for p in paths} == set(PHYSIC_SOLVER_CASE_NAMES.values())
        for path in paths:
            shutil.copy2(path, snapshots / path.name)
        POINTER.parent.mkdir(parents=True, exist_ok=True)
        POINTER.write_text(
            json.dumps(
                dict(
                    run_id=run_id,
                    run_dir=str(run),
                    stage="prepared",
                    complete=False,
                    resolution=[640, 480],
                    config_count=len(paths),
                    expected_accepted_cases=27,
                    fresh_physics_required=True,
                ),
                indent=2,
            )
            + "\n"
        )
    current = read(POINTER)
    run = ROOT / current["run_dir"]
    state = run / "queue.json"
    if state.exists():
        return run, state
    cases = {}
    for group, keys in PHYSIC_SOLVER_CASE_GROUPS.items():
        for key in keys:
            name = physic_solver_case_name(key)
            path = ROOT / "configs/physic_solver" / f"{name}.toml"
            config = tomllib.loads(path.read_text())
            assert Path(config["job"]["output_dir"]).name == name
            cases[name] = dict(
                case=name,
                group=group,
                config=str(path),
                config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_success=group != "controls",
                sdk=name in PHYSIC_SOLVER_PACKAGE_CASES,
                backend=config["scene"]["backend"],
                status="pending",
                attempts=[],
                timeout_s=config["job"]["timeout_s"],
            )
    assert len(cases) == 29 and sum(c["expected_success"] for c in cases.values()) == 27
    state.write_text(
        json.dumps(
            {
                **current,
                "stage": "fresh_collection",
                "cases": cases,
                "resolution": [640, 480],
                "visual_review_required": True,
                "complete": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return run, state


def sheet(output: Path, dataset: dict) -> None:
    """Make a contact sheet from every actual camera, for a later explicit visual review."""
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    frames, fps = dataset["frames"], dataset["fps"]
    indices = [round((frames - 1) * f) for f in (0, 0.2, 0.4, 0.6, 0.8, 1)]
    videos = dataset["videos"]
    image = Image.new("RGB", (960, 528 * len(videos)), "#111b29")
    draw = ImageDraw.Draw(image)
    for camera_index, (camera, path) in enumerate(videos.items()):
        for i, frame in enumerate(indices):
            encoded = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-i",
                    path,
                    "-vf",
                    f"select=eq(n\\,{frame})",
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-threads",
                    "1",
                    "-",
                ]
            )
            x, y = i % 3 * 320, camera_index * 528 + i // 3 * 264
            image.paste(
                Image.open(io.BytesIO(encoded)).convert("RGB").resize((320, 240)), (x, y + 24)
            )
            draw.text((x + 5, y + 5), f"{camera.split('.')[-1]} t={frame / fps:.2f}s", fill="white")
    image.save(qa / "review_keyframes.jpg", quality=95)


def validate(row: dict, output: Path) -> dict:
    """Run the existing independent task audits and verify the newly captured resolution."""
    config = tomllib.loads(Path(row["config"]).read_text())
    cameras = {
        c["name"]: (640, 480)
        for domain in ("scene", "robot")
        for c in config[domain].get("cameras", [])
    }
    video_only = not row["expected_success"] and not (output / "lerobot_v3").exists()
    dataset = (
        validate_failed_frames(output, config, cameras)
        if video_only
        else validate_dataset(
            output, allow_failure=not row["expected_success"], expected_cameras=cameras
        )
    )
    if row["expected_success"]:
        assert dataset["grasp_success"]
        if row["sdk"]:
            from scripts.physic_solver.render_package_trajectory import validate_rendered_trajectory

            audit = validate_rendered_trajectory(Path(row["physics_source"]), output)
        elif row["backend"] == "pickplace_carton":
            from scripts.physic_solver.validate_carton_pickplace import validate as task_validate

            audit = task_validate(output)
        elif row["backend"] == "screw_assembly":
            from scripts.physic_solver.validate_screw_assembly import validate as task_validate

            audit = task_validate(output)
        elif row["backend"] == "pickplace_package":
            from scripts.physic_solver.validate_package_pickplace import audit as task_validate

            audit = task_validate(output)
        else:
            audit = dict(
                success=True, dataset=dataset, checks=dict(dataset_and_physical_result=True)
            )
        assert audit["success"] and all(audit.get("checks", {}).values())
    else:
        assert not dataset["grasp_success"], "Expected failed control unexpectedly passed; inspect"
        audit = dict(
            success=True,
            expected_physics_success=False,
            dataset=dataset,
            checks=dict(diagnostic_dataset_validated=True),
        )
    sheet(output, dataset)
    audit["visual_inspection"] = "pending"
    (output / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def validate_failed_frames(output: Path, config: dict, cameras: dict) -> dict:
    """Encode complete original failed captures without inventing missing state/action data."""
    result = read(output / "result.json")
    assert result["completed"] and not result["success"]
    assert result["checks"]["finite_state"]
    roots = list(output.glob("diagnostics/*/failed_recording/frames"))
    assert len(roots) == 1
    fps = config["job"]["recorder"]["fps"]
    stride = round(1 / (fps * config["sim_world"]["physics_dt"]))
    count = result["captured_frames"]
    assert count == len(range(0, result["physics_steps"], stride)) and count >= 2
    destination = output / "diagnostic_videos"
    destination.mkdir(exist_ok=True)
    videos, inventory = {}, {}
    for camera, resolution in cameras.items():
        frames = sorted((roots[0] / camera).glob("frame_*.png"))
        assert [p.name for p in frames] == [f"frame_{i:06d}.png" for i in range(count)]
        for frame in frames:
            with Image.open(frame) as picture:
                assert picture.size == resolution
        inventory[camera] = [
            dict(path=str(p.relative_to(output)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in frames
        ]
        video = destination / f"{camera}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(roots[0] / camera / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            check=True,
        )
        metadata = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_streams",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )["streams"][0]
        assert (metadata["width"], metadata["height"]) == resolution
        assert metadata["codec_name"] == "h264" and metadata["pix_fmt"] == "yuv420p"
        assert int(metadata["nb_frames"]) == count and metadata["avg_frame_rate"] == f"{fps}/1"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-threads",
                "1",
                "-i",
                str(video),
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        videos[f"observation.images.{camera}"] = str(video)
    (destination / "source_frames.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return dict(
        validated=True,
        grasp_success=False,
        frames=count,
        fps=fps,
        videos=videos,
        dataset_use="diagnostic_video_only",
        joint_state_action_exported=False,
        scope="Original complete camera frames retained after physical rejection; no LeRobot data.",
    )


def run_process(
    command: list[str], log_path: Path, gpu: int, output: Path | None, timeout_s: float
) -> int:
    environment = os.environ.copy()
    environment.pop("RSR_SETTINGS_FILE", None)
    environment.pop("RSR_JOB__OUTPUT_DIR", None)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if output:
        environment["RSR_JOB__OUTPUT_DIR"] = str(output)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_s + 300)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def failed(state: Path, row: dict, error: Exception, *, recording: bool = False) -> None:
    if row["group"] == "controls":
        update(
            state, row["case"], status="attention", last_error=f"{type(error).__name__}: {error}"
        )
        LOG.error(
            "Control outcome needs review; it will not be resampled to select a failure",
            case=row["case"],
            error=str(error),
        )
        return
    if recording and row["sdk"]:
        retries = row.get("capture_failures", 0) + 1
        update(
            state,
            row["case"],
            status="physics_passed" if retries < 3 else "attention",
            capture_failures=retries,
            last_error=f"{type(error).__name__}: {error}",
        )
    else:
        update(
            state,
            row["case"],
            status="pending" if len(row["attempts"]) < 3 else "attention",
            last_error=f"{type(error).__name__}: {error}",
        )
    LOG.error("Fresh collection attempt rejected", case=row["case"], error=str(error))


def new_attempt(run: Path, state: Path, row: dict, gpu: int) -> dict:
    count = len(row["attempts"]) + 1
    directory = run / "attempts" / row["case"] / f"{count:04d}"
    directory.mkdir(parents=True)
    row["attempts"].append(
        dict(
            directory=str(directory),
            started_utc=datetime.now(UTC).isoformat(),
            cuda_visible_devices=str(gpu),
            fresh_physics=True,
        )
    )
    update(state, row["case"], attempts=row["attempts"], attempt_dir=str(directory))
    row["attempt_dir"] = str(directory)
    return row


def physics(run: Path, state: Path, row: dict, gpu: int) -> None:
    row = new_attempt(run, state, row, gpu)
    attempt = Path(row["attempt_dir"])
    update(state, row["case"], active_log=str(attempt / "physics/worker.log"))
    try:
        rc = run_process(
            [
                str(ROOT / "app/python.sh"),
                "scripts/physic_solver/check_package_matrix.py",
                "--cases",
                row["case"],
                "--output",
                str(attempt / "physics"),
            ],
            attempt / "physics_parent.log",
            gpu,
            None,
            row["timeout_s"],
        )
        source = attempt / "physics" / row["case"]
        assert rc == 0 and read(source / "result.json")["success"]
        assert read(source / "physics_audit.json")["success"]
        update(state, row["case"], status="physics_passed", physics_source=str(source))
        LOG.info("Fresh full physics passed", case=row["case"], gpu=gpu)
    except Exception as exc:
        failed(state, row, exc)


def recording(run: Path, state: Path, row: dict, *, adopted: dict | None = None) -> None:
    if adopted:
        output = Path(adopted["dataset"])
        attempt = Path(adopted["attempt_dir"])
        if not adopted.get("orphaned_worker_exit_code_unavailable"):
            row["attempts"] = [
                dict(
                    directory=str(attempt),
                    started_utc=adopted["started_utc"],
                    cuda_visible_devices="0",
                    fresh_physics=True,
                )
            ]
        update(state, row["case"], attempts=row["attempts"], attempt_dir=str(attempt))
    else:
        if not row["sdk"]:
            row = new_attempt(run, state, row, 0)
        attempt = Path(row["attempt_dir"])
        output = (
            attempt / "recordings" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ") / row["case"]
        )
        output.parent.mkdir(parents=True)
    try:
        active_log = output / "worker.log" if row["sdk"] else output.parent / "worker.log"
        if adopted:
            active_log = Path(adopted.get("active_log", attempt / "worker.log"))
        update(state, row["case"], active_log=str(active_log), active_output=str(output))
        if not adopted:
            if row["sdk"]:
                command = [
                    str(ROOT / "app/python.sh"),
                    "scripts/physic_solver/render_package_trajectory.py",
                    row["physics_source"],
                    "--output",
                    str(output),
                ]
                destination = None
            else:
                command = [
                    str(ROOT / "app/python.sh"),
                    "-m",
                    "real2sim2real.core.cli",
                    "collect",
                    "run",
                    "-c",
                    row["config"],
                ]
                destination = output
            rc = run_process(
                command, output.parent / "worker.log", 0, destination, row["timeout_s"]
            )
            allowed = {0} if row["expected_success"] else {0, 1}
            assert rc in allowed, f"Worker exit{rc}; inspect {output.parent / 'worker.log'}"
        else:
            allowed = {0} if row["expected_success"] else {0, 1}
            assert adopted.get("returncode") in allowed or adopted.get(
                "orphaned_worker_exit_code_unavailable"
            )
        validate(row, output)
        note = dict(
            recapture_run=run.name,
            config=row["config"],
            config_sha256=row["config_sha256"],
            fresh_physics=True,
            resolution=[640, 480],
            attempts=row["attempts"],
            physics_source=row.get("physics_source"),
            quality_gates_relaxed=False,
            orphaned_worker_exit_code_unavailable=bool(
                adopted and adopted.get("orphaned_worker_exit_code_unavailable")
            ),
            scope=(
                "New physical episode and new camera capture; previous outputs archived separately."
            ),
        )
        (output / "run_note.json").write_text(json.dumps(note, indent=2) + "\n")
        update(
            state, row["case"], status="review_pending", staged_dataset=str(output), last_error=None
        )
        LOG.info(
            "New dataset numerically passed; visual review pending",
            case=row["case"],
            output=str(output),
        )
    except Exception as exc:
        failed(state, row, exc, recording=True)


def claim(state: Path, gpu: int) -> dict | None:
    with state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read(state)
        rows = list(data["cases"].values())
        choices = (
            (
                [r for r in rows if r["status"] == "physics_passed"]
                + [r for r in rows if r["status"] == "pending" and not r["sdk"]]
                + [r for r in rows if r["status"] == "pending" and r["sdk"]]
            )
            if gpu == 0
            else [r for r in rows if r["status"] == "pending" and r["sdk"]]
        )
        if not choices:
            return None
        row = choices[0]
        action = "physics_running" if row["sdk"] and row["status"] == "pending" else "recording"
        row.update(status=action, gpu=gpu)
        temporary = state.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(state)
        return row


def lane(run: Path, state: Path, gpu: int) -> None:
    for row in read(state)["cases"].values():
        if row.get("recovered_worker") and row.get("gpu") == gpu:
            adopt_worker(run, state, row)
    while True:
        row = claim(state, gpu)
        if row:
            if row["status"] == "physics_running":
                physics(run, state, row, gpu)
            else:
                recording(run, state, row)
            continue
        rows = read(state)["cases"].values()
        busy = any(
            r["status"] in {"pending", "physics_running", "physics_passed", "recording"}
            for r in rows
        )
        if not busy:
            return
        if gpu == 1 and not any(
            r["sdk"] and r["status"] in {"pending", "physics_running"}
            for r in read(state)["cases"].values()
        ):
            return
        time.sleep(5)


def matching_workers(row: dict) -> list[int]:
    """Find only this attempt's surviving children after loss of their controller."""
    attempt = row.get("attempt_dir")
    if not attempt:
        return []
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            args = (process / "cmdline").read_bytes().decode().split("\0")
            scripts = {Path(argument).name for argument in args}
            sdk = scripts & {"check_package_matrix.py", "render_package_trajectory.py"}
            if sdk and any(argument.startswith(attempt + "/") for argument in args):
                matches.append(int(process.name))
            elif "real2sim2real.core.cli" in args and row.get("config") in args:
                environment = (process / "environ").read_bytes().split(b"\0")
                expected = f"RSR_JOB__OUTPUT_DIR={row.get('active_output')}".encode()
                if expected in environment:
                    matches.append(int(process.name))
        except (OSError, UnicodeDecodeError):
            continue
    return matches


def adopt_worker(run: Path, state: Path, row: dict) -> None:
    """Wait for existing work, then audit its files without spawning a duplicate episode."""
    LOG.info("Adopting surviving collection worker", case=row["case"], pids=matching_workers(row))
    try:
        while matching_workers(row):
            time.sleep(5)
        if row["status"] == "physics_running":
            source = Path(row["attempt_dir"]) / "physics" / row["case"]
            assert read(source / "result.json")["success"]
            assert read(source / "physics_audit.json")["success"]
            update(state, row["case"], status="physics_passed", physics_source=str(source))
        else:
            recording(
                run,
                state,
                row,
                adopted=dict(
                    dataset=row["active_output"],
                    attempt_dir=row["attempt_dir"],
                    started_utc=row["attempts"][-1]["started_utc"],
                    active_log=row["active_log"],
                    orphaned_worker_exit_code_unavailable=True,
                ),
            )
    except Exception as exc:
        failed(state, row, exc, recording=row["status"] == "recording")
    finally:
        update(state, row["case"], recovered_worker=False)


def approve(state: Path, names: list[str], note: str) -> None:
    """Publish only episodes for which the operator has actually inspected the saved sheet."""
    for name in names:
        name = physic_solver_case_name(name)
        row = read(state)["cases"][name]
        assert row["status"] == "review_pending", (name, row["status"])
        staged = Path(row["staged_dataset"])
        image = staged / "qa/review_keyframes.jpg"
        review = dict(
            passed=True,
            reviewed_utc=datetime.now(UTC).isoformat(),
            observations=note,
            reviewed_images=["qa/review_keyframes.jpg"],
            image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            scope=(
                "Actual contact-sheet review plus complete video decode; "
                "not every frame manually reviewed."
            ),
        )
        (staged / "qa/visual_review.json").write_text(json.dumps(review, indent=2) + "\n")
        report = read(staged / "validation.json")
        assert report["success"]
        report["visual_inspection"] = "passed"
        destination = (OUTPUT if row["expected_success"] else DEBUG) / name
        assert not destination.exists(), destination
        # Dataset paths describe the published files; configuration and run_note retain staging
        # paths as the historical execution record.
        report_text = json.dumps(report, indent=2).replace(
            str(staged) + "/", str(destination) + "/"
        )
        (staged / "validation.json").write_text(report_text + "\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staged, destination)
        update(
            state,
            name,
            status="accepted" if row["expected_success"] else "diagnostic_verified",
            output=str(destination),
            visual_review=review,
        )
    data = read(state)
    accepted = [r for r in data["cases"].values() if r["status"] == "accepted"]
    complete = len(accepted) == 27 and all(
        r["status"] in {"accepted", "diagnostic_verified"} for r in data["cases"].values()
    )
    stage = "validated" if complete else "fresh_collection"
    update(state, complete=complete, accepted_count=len(accepted), stage=stage)
    (OUTPUT / "collection_summary.json").write_text(
        json.dumps(
            dict(
                run_id=data["run_id"],
                complete=complete,
                accepted_count=len(accepted),
                cases=list(read(state)["cases"].values()),
                resolution=[640, 480],
                fresh_physics=True,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    if POINTER.exists() and read(POINTER).get("run_id") == data["run_id"]:
        update(POINTER, complete=complete, accepted_count=len(accepted), stage=stage)


def recover(state: Path) -> None:
    """Resume a stopped controller without reusing incomplete physical trajectories."""
    data = read(state)
    previous = data.get("controller_pid")
    if previous and previous != os.getpid():
        path = Path(f"/proc/{previous}/cmdline")
        if path.exists() and b"recapture_physic_solver.py" in path.read_bytes():
            raise RuntimeError(f"The collection controller is already running: {previous}")
    for name, row in data["cases"].items():
        if row["status"] not in {"recording", "physics_running"}:
            continue
        workers = matching_workers(row)
        completed_capture = (
            row["status"] == "recording"
            and row.get("active_output")
            and (Path(row["active_output"]) / "result.json").is_file()
            and read(Path(row["active_output"]) / "result.json").get("completed")
        )
        if workers or completed_capture:
            update(
                state,
                name,
                recovered_worker=True,
                recovery_worker_pids=workers,
                recovery_note="Adopt surviving or completed worker; no duplicate simulation.",
            )
            continue
        source = Path(row.get("physics_source") or "")
        if row.get("attempt_dir") and not (source / "physics_audit.json").is_file():
            source = Path(row["attempt_dir"]) / "physics" / name
        valid = row["sdk"] and (source / "physics_audit.json").is_file()
        valid = bool(valid and read(source / "physics_audit.json").get("success"))
        update(
            state,
            name,
            status="physics_passed" if valid else "pending",
            physics_source=str(source) if valid else None,
            recovery_note="Resume after controller interruption; incomplete state not stitched.",
        )
    update(state, controller_pid=os.getpid())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--approve", nargs="+")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    run, state = initialize()
    if args.status:
        data = read(state)
        LOG.info(
            "Fresh capture queue",
            counts={
                s: sum(r["status"] == s for r in data["cases"].values())
                for s in sorted({r["status"] for r in data["cases"].values()})
            },
            run=str(run),
            active_cases=[
                r["case"]
                for r in data["cases"].values()
                if r["status"] in {"recording", "physics_running"}
            ],
            review_pending=[
                r["case"] for r in data["cases"].values() if r["status"] == "review_pending"
            ],
            attention=[
                dict(case=r["case"], error=r.get("last_error"))
                for r in data["cases"].values()
                if r["status"] == "attention"
            ],
        )
        return
    if args.approve:
        if not args.note:
            parser.error("--note must describe the actual visual inspection")
        approve(state, args.approve, args.note)
        return
    recover(state)
    # Preserve PathTracing's exclusive GPU phase before launching any second simulation worker.
    first = physic_solver_case_name("featherstone_vbd_pickplace_cloth")
    first_file = run / "first_case.json"
    if read(state)["cases"][first]["status"] == "pending" and first_file.exists():
        update(state, first, status="recording")
        external = read(first_file)
        while external["status"] == "running":
            proc = Path(f"/proc/{external['pid']}/cmdline")
            if not proc.exists() or first not in proc.read_bytes().decode():
                time.sleep(2)
                external = read(first_file)
                if external["status"] == "running":
                    raise RuntimeError(
                        "External first worker interrupted; preserve and restart explicitly"
                    )
                break
            time.sleep(10)
            external = read(first_file)
        recording(run, state, read(state)["cases"][first], adopted=external)
    while read(state)["cases"][first]["status"] == "pending":
        update(state, first, status="recording")
        recording(run, state, read(state)["cases"][first])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane, run, state, gpu) for gpu in (0, 1)]
        for future in futures:
            future.result()
    LOG.info("Capture workers finished; explicit visual reviews remain", state=str(state))


if __name__ == "__main__":
    main()
