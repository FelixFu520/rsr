"""Run the regular carton Job on real native CPU MuJoCo without starting Kit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from real2sim2real.utils.physic_solver_utils import isaac_newton_sdk_environment

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Use the fixed Job lifecycle and the SDK shipped with the current Isaac Python."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/physic_solver/mujoco_rigid_v10_box01.toml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = isaac_newton_sdk_environment()
    environment.update(
        {
            "RSR_SIM_WORLD__BACKEND": "newton_diagnostic",
            "RSR_SIM_WORLD__DEVICE": "cpu",
            "RSR_SIM_WORLD__NEWTON_USE_CUDA_GRAPH": "false",
            "RSR_PLANNER__MUJOCO_CPU": "true",
            "RSR_JOB__RECORD_RGB": "false",
            "RSR_JOB__OUTPUT_DIR": str(args.output.resolve()),
            "RSR_SCENE__CAMERAS": "@reset []",
            "RSR_ROBOT__CAMERAS": "@reset []",
        }
    )
    with (args.output / "worker.log").open("w") as stream:
        process = subprocess.run(
            [
                str(ROOT / "app/python.sh"),
                "-m",
                "real2sim2real.core.cli",
                "collect",
                "run",
                "-c",
                args.config,
            ],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    try:
        result = json.loads((args.output / "result.json").read_text())
    except (OSError, ValueError):
        return 1
    return 0 if process.returncode == 0 and result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
