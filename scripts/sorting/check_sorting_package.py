"""Run the full Pro-M6 sorting physics without RGB, using the production Job lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from real2sim2real.core.config import apply_settings_file
from real2sim2real.job.job_sorting import SortingJob, SortingJobConfig
from real2sim2real.utils.kit_process import run_kit_worker


class SortingPhysicsCheck(SortingJob):
    """Retain a complete physical trajectory for inexpensive contact/controller iterations."""

    def simulate(self) -> None:
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        configuration = {
            name: getattr(self, name).config.model_dump(mode="json")
            for name in ("sim_world", "scene", "robot", "planner")
        }
        (output / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
        self.sim_world.play()
        started, report = time.monotonic(), 0.0
        completed = False
        try:
            for _ in range(self.config.max_steps):
                completed = self.planner.on_physics_step(self.robot)
                if completed:
                    break
                self.sim_world.step(render=False)
                elapsed = time.monotonic() - started
                if elapsed > self.config.timeout_s:
                    raise TimeoutError("Sorting physics check exceeded timeout")
                if elapsed >= report:
                    self.log.info(
                        "Sorting physics check",
                        simulation_s=self.planner.elapsed,
                        phase=self.planner.state["phase"],
                        wall_s=elapsed,
                        progress=self.planner.state.get("progress"),
                    )
                    report = elapsed + 30
        finally:
            self.planner.write_diagnostics(output)
            result = self.planner.result()
            result.update(
                completed=bool(completed and self.planner.state.get("done")),
                rigid_solver="mujoco",
                contact_solver="vbd",
                elapsed_s=time.monotonic() - started,
                physics_run_id=getattr(self, "worker_id", None),
            )
            (output / "result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            (output / "trajectory.json").write_text(json.dumps(self.planner.rows) + "\n")
            self.log.info("Sorting physics result", **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sorting/mujoco_vbd_sorting.toml")
    parser.add_argument("--output", default="outputs/debug/sorting/physics")
    parser.add_argument("--max-steps", type=int, help="默认使用配置中的 job.max_steps")
    parser.add_argument("--worker-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_id is None:
        raise SystemExit(
            run_kit_worker(
                __file__,
                sys.argv[1:],
                Path(args.output) / "result.json",
                marker_field="physics_run_id",
                require_success=True,
            )
        )
    apply_settings_file(args.config)
    config = SortingJobConfig.from_settings("job")
    config.output_dir = args.output
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    job = SortingPhysicsCheck(config=config)
    job.worker_id = args.worker_id
    job.run()
    outcome = json.loads((Path(args.output) / "result.json").read_text())
    raise SystemExit(0 if outcome["success"] else 1)
