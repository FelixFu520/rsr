"""Run package physics without RGB capture, retaining the regular Job lifecycle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from real2sim2real.core.config import apply_settings_file
from real2sim2real.job.job_physic_solver import PhysicSolverJob, PhysicSolverJobConfig


class PackagePhysicsCheck(PhysicSolverJob):
    """Collect physical diagnostics for inexpensive grasp-geometry iterations."""

    def simulate(self) -> None:
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.sim_world.play()
        started, next_report = time.monotonic(), 0.0
        try:
            for _ in range(self.config.max_steps):
                if self.planner.on_physics_step(self.robot):
                    break
                self.sim_world.step(render=False)
                elapsed = time.monotonic() - started
                if elapsed > self.config.timeout_s:
                    raise TimeoutError("Package physics check exceeded wall-clock limit")
                if elapsed >= next_report:
                    self.log.info(
                        "Package physics check",
                        simulation_s=self.planner.elapsed,
                        wall_s=elapsed,
                        phase=self.planner.state["phase"],
                    )
                    next_report = elapsed + 30
        finally:
            result = self.planner.result()
            result.update(
                wall_s=time.monotonic() - started,
                substeps=self.sim_world.config.newton_substeps,
                iterations=self.sim_world.config.newton_iterations,
            )
            self.planner.write_diagnostics(output)
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            (output / "trajectory.json").write_text(json.dumps(self.planner.rows) + "\n")
            self.log.info("Package physics result", **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/physic_solver/featherstone_vbd_pickplace_package.toml"
    )
    parser.add_argument("--output", default="outputs/debug/physic_solver/package_physics_check")
    args = parser.parse_args()
    apply_settings_file(args.config)
    config = PhysicSolverJobConfig.from_settings("job")
    config.output_dir = args.output
    PackagePhysicsCheck(config=config).run()
