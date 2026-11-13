"""Batch independent Newton package models without Kit; retain every real trajectory.

./app/python.sh scripts/physic_solver/check_package_matrix.py --cases v9_mass100g shake_spring
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from real2sim2real.core.config import apply_settings_file
from real2sim2real.job.job_physic_solver import PhysicSolverJob, PhysicSolverJobConfig
from real2sim2real.planner.base import AutoPlanner
from real2sim2real.robot.base import AutoRobot
from real2sim2real.scene.base import AutoScene
from real2sim2real.simworld.isaacsim import IsaacsimWorldConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/physic_solver/package_cases"


class PackageBatchPhysicsJob(PhysicSolverJob):
    """Retain the fixed Job lifecycle and isolate each package in a separate Newton model."""

    def __init__(self, paths: list[Path], output: Path) -> None:
        apply_settings_file(paths[0])
        config = PhysicSolverJobConfig.from_settings("job")
        config.output_dir = str(output)
        config.timeout_s = max(config.timeout_s * len(paths), 7200)
        super().__init__(config=config)
        self.cases = []
        for index, path in enumerate(paths):
            apply_settings_file(path)
            scene = self.scene if index == 0 else AutoScene.from_config()
            robot = self.robot if index == 0 else AutoRobot.from_config()
            planner = self.planner if index == 0 else AutoPlanner.from_config()
            world_config = IsaacsimWorldConfig.from_settings("sim_world")
            if scene.config.backend != "pickplace_package":
                raise ValueError("Only independent two-surface package models belong in this batch")
            if world_config.physics_dt != self.sim_world.config.physics_dt:
                raise ValueError("Batch models must have the same frame rate")
            prefix = f"/World/cases/{path.stem}"
            cameras = {}
            for component, domain in ((scene, "scene"), (robot, "robot")):
                old = component.config.prim_path
                new = f"{prefix}/{domain}"
                cameras[domain] = []
                for camera in component.config.cameras:
                    copy = camera.model_copy(deep=True)
                    copy.prim_path = copy.prim_path.replace(old, new, 1)
                    copy.pose_relative_to = copy.pose_relative_to.replace(old, new, 1)
                    cameras[domain].append(copy.model_dump(mode="json"))
                component.config.prim_path = new
                component.config.cameras = []
            case = SimpleNamespace(
                name=path.stem,
                path=path,
                directory=output / path.stem,
                scene=scene,
                robot=robot,
                planner=planner,
                world_config=world_config,
                cameras=cameras,
                done=False,
                error=None,
                frames=[],
                body_frames=[],
                proxy_frames=[],
                motor_forces=[],
                external_forces=[],
                effective_controls=[],
                joints=[],
                actions=[],
                times=[],
            )
            case.directory.mkdir(parents=True)
            self.cases.append(case)

    def after_arrange_scene_hook(self) -> None:
        """Arrange every additional scene before placing any additional robot."""
        for index, case in enumerate(self.cases):
            if index:
                case.scene.init_scene(self.sim_world)
                case.scene.arrange()
            case.world = SimpleNamespace(
                config=case.world_config, stage=self.sim_world.stage, uses_direct_newton=True
            )
            case.scene.sim_world = case.world

    def before_place_robot_hook(self) -> None:
        """Mount all arms on their measured, isolated tables."""
        for case in self.cases:
            case.robot.config.base_position = tuple(case.scene.robot_base_position)
            case.planner.prepare_initial_joints(case.robot, case.scene)

    def after_place_robot_hook(self) -> None:
        """Finish robot/planner initialization for every extra model in the place stage."""
        for index, case in enumerate(self.cases):
            if index:
                case.robot.init_robot(case.world)
                case.robot.arrange()
                case.planner.init_planner(case.robot)
            case.robot.sim_world = case.world

    def before_start_simulation_hook(self) -> None:
        """Register the isolated worker's one real Newton model and reset callback."""
        self.sim_world.register_newton_stepper(self._step_batch, reset=self._reset_batches)
        self.sim_world.reset()

    def _reset_batches(self) -> None:
        for case in self.cases:
            configuration = {
                "sim_world": case.world_config.model_dump(mode="json"),
                **{
                    name: getattr(case, name).config.model_dump(mode="json")
                    for name in ("scene", "robot", "planner")
                },
                "job": self.config.model_dump(mode="json"),
                "recording_cameras": case.cameras,
                "source_config": str(case.path),
            }
            (case.directory / "configuration.json").write_text(
                json.dumps(configuration, indent=2) + "\n"
            )
            case.planner.configure_scene(case.scene)
            case.planner.begin_episode(register_stepper=False)

    def _step_batch(self, dt: float) -> None:
        import warp as wp

        active = [case for case in self.cases if not case.done and not case.error]
        # Prepare all native CPU trajectories before GPU soft work.
        for case in active:
            if case.planner.config.mujoco_cpu:
                case.planner.runtime.prepare_cpu_rigid_frame(dt)
        # A single ordered stream avoids unsupported overlap between independent VBD graphs.
        wp.synchronize_device()
        for case in active:
            case.planner.runtime.step(dt, publish=False)
        wp.synchronize_device()
        for case in active:
            case.planner.runtime._publish()

    def setup_gui_hook(self) -> None:
        """Physics-only batches do not create camera render products."""

    def _save(self, case: SimpleNamespace, elapsed: float) -> None:
        # Preserve downloaded evidence before any further CUDA reads after a device fault.
        (case.directory / "trajectory.json").write_text(json.dumps(case.planner.rows) + "\n")
        np.savez_compressed(
            case.directory / "full_physics.npz",
            positions=np.asarray(case.frames),
            bodies=np.asarray(case.body_frames),
            states=np.asarray(case.joints),
            actions=np.asarray(case.actions),
            times=np.asarray(case.times),
            **(
                {
                    "proxy_bodies": np.asarray(case.proxy_frames),
                    "actuator_forces": np.asarray(case.motor_forces),
                    "native_external_wrenches": np.asarray(case.external_forces),
                    "native_controls": np.asarray(case.effective_controls),
                }
                if case.proxy_frames
                else {}
            ),
        )
        try:
            result = case.planner.result()
        except Exception as exc:
            result = dict(
                success=False,
                checks=dict(result_evaluated=False),
                evaluation_error=f"{type(exc).__name__}: {exc}",
            )
        result.update(
            success=bool(result["success"] and case.error is None),
            completed=case.done,
            error=case.error,
            wall_s=elapsed,
            physics_steps=len(case.times),
            simulation_time_s=case.planner.elapsed,
            physics_engine="newton",
            rigid_solver=case.world_config.newton_solver,
            contact_solver="vbd",
            substeps=case.world_config.newton_substeps,
            iterations=case.world_config.newton_iterations,
            physics_source="independent_model_in_batch",
        )
        try:
            case.planner.write_diagnostics(case.directory)
        except Exception as exc:
            result.update(success=False, diagnostic_error=f"{type(exc).__name__}: {exc}")
        (case.directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        case.saved = True
        self.log.info(
            "Package batch case finished",
            case=case.name,
            success=result["success"],
            failed=[key for key, passed in result["checks"].items() if not passed],
        )

    def simulate(self) -> None:
        """Record continuous independent physical trajectories without rendering."""
        self.sim_world.play()
        started, next_report = time.monotonic(), 0.0
        try:
            self._run_frames(started, next_report)
        except BaseException as exc:
            for case in self.cases:
                if not case.done and case.error is None:
                    case.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            for case in self.cases:
                if not getattr(case, "saved", False):
                    self._save(case, time.monotonic() - started)

    @staticmethod
    def _check_grasp_pose_checkpoint(case: SimpleNamespace) -> None:
        """Audit the saved closed-grasp prefix before spending time on lift and transfer."""
        from scripts.physic_solver.audit_package_grasp_pose import audit_grasp_pose

        checkpoint = case.directory / "grasp_pose_checkpoint"
        checkpoint.mkdir()
        (checkpoint / "configuration.json").write_bytes(
            (case.directory / "configuration.json").read_bytes()
        )
        (checkpoint / "trajectory.json").write_text(json.dumps(case.planner.rows) + "\n")
        np.savez_compressed(checkpoint / "full_physics.npz", states=case.joints, times=case.times)
        report = audit_grasp_pose(checkpoint)
        case.pose_checked = True
        if not report["success"]:
            raise RuntimeError(
                "Early stop: measured grasp pose failed its unchanged position/orientation limits"
            )

    def _run_frames(self, started: float, next_report: float) -> None:
        for _ in range(self.config.max_steps):
            if time.monotonic() - started > self.config.timeout_s:
                raise TimeoutError("Package batch exceeded its wall-clock limit")
            for case in self.cases:
                if case.done or case.error:
                    continue
                try:
                    case.done = case.planner.on_physics_step(case.robot)
                    if case.planner.state["phase"] == "lift" and not getattr(
                        case, "pose_checked", False
                    ):
                        self._check_grasp_pose_checkpoint(case)
                    progress = case.planner.state.get("progress", {})
                    if not case.done:
                        runtime = case.planner.runtime
                        state, action = runtime.recording_coordinates()
                        case.frames.append(runtime.vertices().astype(np.float32))
                        case.body_frames.append(runtime.state_0.body_q.numpy().copy())
                        if runtime.coupled_solver is not None:
                            case.proxy_frames.append(
                                runtime.coupled_solver.entry_state("vbd").body_q.numpy().copy()
                            )
                            data = runtime.rigid_solver.mj_data
                            case.motor_forces.append(data.qfrc_actuator.copy())
                            case.external_forces.append(data.xfrc_applied.copy())
                            case.effective_controls.append(data.ctrl.copy())
                        case.joints.append(state.astype(np.float32))
                        case.actions.append(action.astype(np.float32))
                        case.times.append(runtime.sim_time)
                    if progress.get("maximum_edge_stretch", 0) >= 1.3:
                        raise RuntimeError(
                            "Early stop: maximum surface stretch already exceeds 1.3"
                        )
                    if (
                        progress.get("maximum_table_penetration_m", 0)
                        > case.planner.config.maximum_table_penetration_m
                    ):
                        raise RuntimeError(
                            "Early stop: table penetration already exceeds its limit"
                        )
                    if not case.done and case.planner.state["phase"] in ("transfer", "shake"):
                        row = case.planner.rows[-1]
                        lifts = (
                            np.array([row["garment_com_m"][2], row["bag_com_m"][2]])
                            - case.planner._baseline[:, 2]
                        )
                        if np.any(lifts < case.planner.config.minimum_lift_m):
                            raise RuntimeError(
                                "Early stop: sustained transfer/shake lift already below its limit"
                            )
                except Exception as exc:
                    case.error = f"{type(exc).__name__}: {exc}"
                if case.done or case.error:
                    self._save(case, time.monotonic() - started)
            if all(case.done or case.error for case in self.cases):
                return
            self.sim_world.step(render=False)
            elapsed = time.monotonic() - started
            if elapsed >= next_report:
                self.log.info(
                    "Package batch progress",
                    wall_s=elapsed,
                    cases={
                        case.name: dict(
                            time_s=case.planner.elapsed,
                            phase=case.planner.state["phase"],
                            progress=case.planner.state.get("progress"),
                            error=case.error,
                        )
                        for case in self.cases
                    },
                )
                next_report = elapsed + 30
        raise RuntimeError("Package batch did not finish within max_steps")


def main() -> int:
    """Run one real Newton SDK worker and independently audit each case after it exits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    names = args.cases or [path.stem for path in sorted(CONFIGS.glob("*.toml"))]
    paths = [CONFIGS / f"{name}.toml" for name in names]
    output = args.output or ROOT / "outputs/debug/physic_solver/package_batches" / datetime.now(
        UTC
    ).strftime("%Y%m%dT%H%M%S_%fZ")
    if args.worker:
        if len(paths) != 1:
            raise ValueError("Each diagnostic worker requires exactly one case")
        import signal

        def interrupted(signum, frame):
            del signum, frame
            raise KeyboardInterrupt("Diagnostic worker stopped")

        signal.signal(signal.SIGTERM, interrupted)
        PackageBatchPhysicsJob(paths, output).run()
        return 0
    output.mkdir(parents=True, exist_ok=False)
    from real2sim2real.utils.physic_solver_utils import isaac_newton_sdk_environment

    environment = isaac_newton_sdk_environment()
    environment["RSR_SIM_WORLD__BACKEND"] = "newton_diagnostic"
    environment.pop("RSR_SETTINGS_FILE", None)
    from real2sim2real.planner.planner_physic_solver import PackagePlannerConfig
    from scripts.physic_solver.run_package_matrix import audit_physics

    results = []
    with (output / "worker.log").open("w") as stream:
        for name in names:
            process = subprocess.run(
                [
                    str(ROOT / "app/python.sh"),
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--output",
                    str(output),
                    "--cases",
                    name,
                ],
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            directory = output / name
            row = dict(case=name, output=str(directory), returncode=process.returncode)
            try:
                config = json.loads((directory / "configuration.json").read_text())
                audit = audit_physics(
                    directory, PackagePlannerConfig.model_validate(config["planner"])
                )
                row["success"] = process.returncode == 0 and audit["success"]
            except (OSError, ValueError, KeyError, AssertionError) as exc:
                row.update(success=False, error=str(exc))
            results.append(row)
            complete = len(results) == len(names)
            (output / "batch_summary.json").write_text(
                json.dumps(
                    dict(
                        complete=complete,
                        success=complete and all(r["success"] for r in results),
                        execution="isolated_single_case_workers",
                        cases=results,
                    ),
                    indent=2,
                )
                + "\n"
            )
    return 0 if all(row["success"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
