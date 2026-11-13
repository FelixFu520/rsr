"""Serially collect the physics-solver cases and validate every LeRobot v3 result.

Each case runs in its own Kit process (``collect run -c configs/physic_solver/<case>.toml``).
Verified existing datasets are reused. A case whose run fails, or whose dataset is only a
diagnostic (physical acceptance failed, exported because ``save_failed_episodes=true``),
is moved from ``outputs/physic_solver/<case>`` to ``outputs/debug/physic_solver/<case>``,
so ``outputs/physic_solver`` only holds accepted demonstrations.
The shared core/default_settings.toml file is never written by this command.

    ./app/python.sh scripts/physic_solver/collect_physic_solver_cases.py
    ./app/python.sh scripts/physic_solver/collect_physic_solver_cases.py xpbd_vbd_pickplace_cloth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from real2sim2real.core.logger import get_logger
from real2sim2real.utils.lerobot_v3 import validate_dataset
from real2sim2real.utils.physic_solver_utils import (
    PHYSIC_SOLVER_CASE_GROUPS,
    physic_solver_case_name,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/physic_solver"
OUTPUTS = ROOT / "outputs/physic_solver"
DEBUG = ROOT / "outputs/debug/physic_solver"
CASES = tuple(
    physic_solver_case_name(key)
    for group in ("solver", "controls")
    for key in PHYSIC_SOLVER_CASE_GROUPS[group]
)


def _move_to_debug(output: Path) -> Path | None:
    """Move a failed or diagnostic case out of the accepted output tree."""
    if not output.exists() or not output.is_relative_to(OUTPUTS):
        return None
    target = DEBUG / output.relative_to(OUTPUTS)
    if target.exists():
        target = target.with_name(f"{target.name}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(output, target)
    return target


def main() -> int:
    """Run each independent Kit process to completion before starting the next."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="Case names or configs/physic_solver TOML paths")
    args = parser.parse_args()
    paths = []
    for value in args.cases or CASES:
        path = Path(value)
        if not path.is_file():
            name = value.removeprefix("PhysicSolver_").removesuffix(".toml")
            path = CONFIGS / f"{name}.toml"
        if not path.is_file():
            parser.error(f"Missing configuration: {value}")
        paths.append(path.resolve())
    protected = ROOT / "real2sim2real/core/default_settings.toml"
    original = hashlib.sha256(protected.read_bytes()).hexdigest()
    logs = OUTPUTS / "capture_logs"
    logs.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUTS / "collection_summary.json"
    # Running a subset updates those rows and keeps the other cases' latest results.
    previous = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    selected = {path.stem for path in paths}
    kept = [row for row in previous.get("cases", []) if row.get("case") not in selected]
    current: list[dict] = []
    summary = {
        "serial": True,
        "cases": kept,
        "this_run": sorted(selected),
        "collection_complete": False,
        "default_settings_sha256": original,
    }
    log = get_logger(__name__)
    for path in paths:
        config = tomllib.loads(path.read_text())
        output = ROOT / config["job"]["output_dir"]
        allow = config.get("job", {}).get("recorder", {}).get("save_failed_episodes", False)
        cameras = {
            camera["name"]: tuple(camera.get("resolution", [640, 480]))
            for section in ("scene", "robot")
            for camera in config.get(section, {}).get("cameras", [])
        }
        row = {"case": path.stem, "config": str(path), "output_dir": str(output)}
        row["validated"] = False
        engine = config["sim_world"]["physics_engine"]
        kind = config["scene"]["object_kind"]
        row["robot_solver"] = (
            config["sim_world"]["newton_solver"]
            if engine == "newton"
            else config["sim_world"]["physx_solver"].lower()
        )
        row["contact_solver"] = (
            config["planner"]["cloth_solver"]
            if kind == "cloth"
            else config["sim_world"]["physx_solver"].lower()
        )
        row["task"] = (
            "pickplace_package"
            if config["scene"]["backend"] == "pickplace_package"
            else f"pickplace_{kind}"
        )
        started = time.monotonic()
        try:
            try:
                row.update(validate_dataset(output, allow_failure=allow, expected_cameras=cameras))
                row["reused_existing"] = True
            except (OSError, ValueError, KeyError, AssertionError):
                if (output / "lerobot_v3").exists():
                    raise ValueError(
                        "Existing dataset did not pass validation; choose a new output_dir"
                    ) from None
                log_path = logs / f"{path.stem}_{datetime.now(UTC):%Y%m%dT%H%M%S}.log"
                row["log"] = str(log_path)
                log.info("Collecting solver case", case=path.stem, log=str(log_path))
                environment = os.environ.copy()
                environment["RSR_JOB__OUTPUT_DIR"] = str(output)
                with log_path.open("w") as stream:
                    process = subprocess.run(
                        [
                            str(ROOT / "app/python.sh"),
                            "-m",
                            "real2sim2real.core.cli",
                            "collect",
                            "run",
                            "-c",
                            str(path),
                        ],
                        cwd=ROOT,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=config["job"].get("timeout_s", 1800) + 300,
                    )
                row["returncode"] = process.returncode
                if process.returncode:
                    raise RuntimeError(
                        f"Simulation exited with {process.returncode}; see {log_path}"
                    ) from None
                row.update(validate_dataset(output, allow_failure=allow, expected_cameras=cameras))
        except (
            OSError,
            ValueError,
            KeyError,
            AssertionError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as exc:
            row["error"] = str(exc)
            log.error("Solver collection failed", case=path.stem, error=str(exc))
        if not (row["validated"] and row.get("grasp_success")):
            moved = _move_to_debug(output)
            if moved is not None:
                row["moved_to_debug"] = str(moved)
                log.warning("Case is not an accepted demonstration", case=path.stem, debug=moved)
        row["elapsed_s"] = time.monotonic() - started
        row["finished_at"] = datetime.now(UTC).isoformat()
        current.append(row)
        rows = sorted(
            [*kept, *current],
            key=lambda x: CASES.index(x["case"]) if x["case"] in CASES else len(CASES),
        )
        summary["cases"] = rows
        summary["collection_complete"] = len(current) == len(paths) and all(
            x["validated"] for x in current
        )
        summary["all_grasps_successful"] = all(x.get("grasp_success") for x in rows)
        summary["accepted_cases"] = [x["case"] for x in rows if x.get("grasp_success")]
        if hashlib.sha256(protected.read_bytes()).hexdigest() != original:
            raise RuntimeError("core/default_settings.toml changed during collection")
        temporary = summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(summary_path)
        log.info(
            "Solver case completed",
            case=path.stem,
            validated=row["validated"],
            success=row.get("grasp_success"),
        )
    return 0 if summary["collection_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
