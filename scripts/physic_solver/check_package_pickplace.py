"""Run package physics without RGB capture, retaining the regular Job lifecycle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from real2sim2real.core.config import apply_settings_file
from real2sim2real.job.job_physic_solver import PhysicSolverJob, PhysicSolverJobConfig


class PackagePhysicsCheck(PhysicSolverJob):
    """Collect physical diagnostics for inexpensive grasp-geometry iterations."""

    def simulate(self) -> None:
        output = Path(self.config.output_dir)
        if (output / "result.json").exists():
            raise FileExistsError(f"Refusing to overwrite a physics attempt: {output}")
        output.mkdir(parents=True, exist_ok=True)
        configuration = {
            name: getattr(self, name).config.model_dump(mode="json")
            for name in ("sim_world", "scene", "robot", "planner")
        }
        configuration["job"] = self.config.model_dump(mode="json")
        configuration["recording_cameras"] = {
            name: getattr(self, name).config.model_dump(mode="json")["cameras"]
            for name in ("scene", "robot")
        }
        (output / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
        self.sim_world.play()
        started, next_report = time.monotonic(), 0.0
        full = {name: [] for name in ("positions", "bodies", "states", "actions", "times")}
        try:
            for _ in range(self.config.max_steps):
                if self.planner.on_physics_step(self.robot):
                    break
                runtime = self.planner.runtime
                state, action = runtime.recording_coordinates()
                full["positions"].append(runtime.vertices().astype("float32"))
                full["bodies"].append(runtime.state_0.body_q.numpy().copy())
                full["states"].append(state.astype("float32"))
                full["actions"].append(action.astype("float32"))
                full["times"].append(self.planner.elapsed - self.sim_world.config.physics_dt)
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
                        progress=self.planner.state.get("progress"),
                    )
                    next_report = elapsed + 30
        finally:
            try:
                result = self.planner.result()
            except Exception as exc:
                result = dict(
                    success=False,
                    checks=dict(result_evaluated=False),
                    evaluation_error=f"{type(exc).__name__}: {exc}",
                )
            result.update(
                wall_s=time.monotonic() - started,
                substeps=self.sim_world.config.newton_substeps,
                iterations=self.sim_world.config.newton_iterations,
            )
            self.planner.write_diagnostics(output)
            np.savez_compressed(
                output / "full_physics.npz",
                **{name: np.asarray(values) for name, values in full.items()},
            )
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            (output / "trajectory.json").write_text(json.dumps(self.planner.rows) + "\n")
            self.log.info("Package physics result", **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/physic_solver/featherstone_vbd_package01_center_control.toml"
    )
    parser.add_argument("--output", default="outputs/debug/physic_solver/package_physics_check")
    args = parser.parse_args()
    apply_settings_file(args.config)
    config = PhysicSolverJobConfig.from_settings("job")
    config.output_dir = args.output
    PackagePhysicsCheck(config=config).run()
