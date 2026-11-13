"""Render an accepted continuous physical trajectory into dual-camera LeRobot v3.

No physics is integrated during rendering. Robot/particle visuals and measured state/action
come from the same original physical frame; the original source is retained and hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from real2sim2real.core.config import apply_settings_file
from real2sim2real.job.job_physic_solver import PhysicSolverJob, PhysicSolverJobConfig
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder

ROOT = Path(__file__).resolve().parents[2]


def validate_rendered_trajectory(source: Path, output: Path) -> dict:
    """Audit physical geometry, the immutable source and exported state/action alignment."""
    import pyarrow.parquet as pq

    from real2sim2real.planner.planner_physic_solver import PackagePlannerConfig
    from scripts.physic_solver.audit_package_grasp_pose import audit_grasp_pose
    from scripts.physic_solver.run_package_matrix import audit_physics
    from scripts.physic_solver.validate_package_pickplace import audit

    configuration = json.loads((source / "configuration.json").read_text())
    physics = audit_physics(source, PackagePlannerConfig.model_validate(configuration["planner"]))
    grasp_pose = audit_grasp_pose(source)
    dataset = audit(output)
    shutil.copy2(source / "grasp_pose_audit.json", output / "qa/grasp_pose_audit.json")
    result = json.loads((output / "result.json").read_text())
    data = np.load(source / "full_physics.npz")
    stride = round(
        1 / (configuration["sim_world"]["physics_dt"] * configuration["job"]["recorder"]["fps"])
    )
    table = pq.read_table(output / "lerobot_v3/data/chunk-000/file-000.parquet")
    for name, field in (("observation.state", "states"), ("action", "actions")):
        np.testing.assert_array_equal(
            np.array(table[name].to_pylist(), np.float32), data[field][::stride]
        )
    archived = max(output.glob("diagnostics/*/full_physics.npz"))
    expected = result["physics_source_sha256"]
    checks = dict(
        physical_source_passed=physics["success"],
        actual_grasp_pose_verified=grasp_pose["success"],
        dataset_passed=dataset["success"],
        no_capture_physics_steps=result["capture_physics_steps"] == 0,
        measured_joint_body_alignment=result["maximum_state_visual_fk_error_m"] <= 1e-5,
        source_sha256_unchanged=all(
            hashlib.sha256(p.read_bytes()).hexdigest() == expected
            for p in (source / "full_physics.npz", archived)
        ),
        recorded_state_action_alignment=True,
    )
    review = output / "qa/visual_review.json"
    report = dict(
        success=all(checks.values()),
        checks=checks,
        physics=physics,
        grasp_pose=grasp_pose,
        dataset=dataset,
        visual_inspection=(
            "passed"
            if review.exists() and json.loads(review.read_text()).get("passed")
            else "pending"
        ),
    )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class PackageTrajectoryRenderJob(PhysicSolverJob):
    """Reuse the regular lifecycle, rendering source poses without changing the physical model."""

    def __init__(self, source: Path, output: Path) -> None:
        self.source = source.resolve()
        original = json.loads((source / "result.json").read_text())
        if not original["success"]:
            raise ValueError("Only independently accepted physical trajectories may be rendered")
        from real2sim2real.planner.planner_physic_solver import PackagePlannerConfig
        from scripts.physic_solver.audit_package_grasp_pose import audit_grasp_pose
        from scripts.physic_solver.run_package_matrix import audit_physics

        payload = json.loads((source / "configuration.json").read_text())
        if not audit_physics(source, PackagePlannerConfig.model_validate(payload["planner"]))[
            "success"
        ]:
            raise ValueError("Independent source physics audit failed")
        if not audit_grasp_pose(source)["success"]:
            raise ValueError("Measured physical grasp does not match the requested pose")
        payload["sim_world"]["backend"] = "isaacsim"
        for domain in ("scene", "robot"):
            payload[domain]["cameras"] = payload["recording_cameras"][domain]
        payload["job"]["output_dir"] = str(output)
        payload["job"]["recorder"]["task"] = (
            f"Grasp parcel {source.name}, lift, hold, transfer, optionally shake, and release; "
            f"attachment_mode={payload['planner']['attachment_mode']}."
        )
        settings = output / "replay_settings.json"
        settings.write_text(json.dumps(payload, indent=2) + "\n")
        apply_settings_file(settings)
        config = PhysicSolverJobConfig.from_settings("job")
        super().__init__(config=config)
        self.source_result = original
        self.payload = payload

    def simulate(self) -> None:
        """Capture recorded source states with a fixed time axis and zero physical steps."""
        import warp as wp
        from pxr import Gf

        output = Path(self.config.output_dir)
        run = output / "diagnostics" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True)
        for name in ("particles.npz", "full_physics.npz", "result.json", "configuration.json"):
            shutil.copy2(
                self.source / name,
                run
                / (
                    "physics_source_result.json"
                    if name == "result.json"
                    else "physics_source_configuration.json"
                    if name == "configuration.json"
                    else name
                ),
            )
        rows = json.loads((self.source / "trajectory.json").read_text())
        (run / "trajectory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        configurations = {
            name: getattr(self, name).config.model_dump(mode="json")
            for name in ("sim_world", "scene", "robot", "planner")
        }
        configurations["job"] = self.config.model_dump(mode="json")
        (run / "configuration.json").write_text(json.dumps(configurations, indent=2) + "\n")
        source_path = self.source / "full_physics.npz"
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        data = np.load(source_path)
        positions, bodies, states, actions, times = [
            data[name] for name in ("positions", "bodies", "states", "actions", "times")
        ]
        np.testing.assert_allclose(
            times, np.arange(len(times)) * self.sim_world.config.physics_dt, atol=1e-8
        )
        result = {
            **self.source_result,
            "object": "two_mesh_garment_package",
            "package_variant": Path(self.scene.config.cloth_usd_path).parent.name,
            "checks": {**self.source_result["checks"], "rgb_capture_complete": False},
            "success": False,
            "capture_mode": "continuous_recorded_physics_replay",
            "physics_source": str(self.source),
            "physics_source_sha256": source_hash,
            "capture_physics_steps": 0,
        }
        recorder = LeRobotV3Recorder(
            output / "lerobot_v3",
            fps=self.config.recorder.fps,
            width=self.config.recorder.width,
            height=self.config.recorder.height,
            task=self.config.recorder.task,
            robot_type="franka_fr3_newton",
            joint_names=[f"fr3_joint{i}" for i in range(1, 8)]
            + ["left_finger_m", "right_finger_m"],
            action_names=self.planner.action_names,
            cameras={camera.name: camera.resolution for camera in self.recording_cameras},
        )
        scratch = self.planner.runtime.model.state()
        model = self.planner.runtime.model
        q_buffer = wp.empty_like(model.joint_q)
        maximum_fk_error = 0.0
        started = time.monotonic()
        next_report = 0.0
        self.sim_world.play()
        try:
            for index in range(0, len(times), self.record_stride):
                if time.monotonic() - started > self.config.timeout_s:
                    raise TimeoutError("Package trajectory rendering exceeded its wall-clock limit")
                self.scene.publish_cloth_vertices(positions[index])
                for op, pose in zip(
                    self.planner.runtime._visual_bodies, bodies[index], strict=False
                ):
                    matrix = Gf.Matrix4d().SetRotate(
                        Gf.Quatd(float(pose[6]), Gf.Vec3d(*pose[3:6].astype(float).tolist()))
                    )
                    matrix.SetTranslateOnly(Gf.Vec3d(*pose[:3].astype(float).tolist()))
                    op.Set(matrix)
                # Independent FK guards against a state/visual frame offset in the saved source.
                q_buffer.assign(states[index])
                self.planner.runtime.newton.eval_fk(model, q_buffer, model.joint_qd, scratch)
                error = np.max(
                    np.abs(
                        scratch.body_q.numpy()[: self.planner.runtime.robot_body_count, :3]
                        - bodies[index, : self.planner.runtime.robot_body_count, :3]
                    )
                )
                maximum_fk_error = max(maximum_fk_error, float(error))
                if maximum_fk_error > 1e-5:
                    raise ValueError(
                        f"Recorded robot state and body poses disagree: {maximum_fk_error}m"
                    )
                self._capture_cameras(recorder)
                recorder.add_frame(states[index], actions[index])
                if self.sim_world.physics_steps:
                    raise RuntimeError("Trajectory rendering advanced physical time")
                elapsed = time.monotonic() - started
                if elapsed >= next_report:
                    self.log.info(
                        "Package trajectory capture progress",
                        captured_frames=len(recorder.rows),
                        total_frames=len(range(0, len(times), self.record_stride)),
                        source_time_s=float(times[index]),
                        capture_physics_steps=self.sim_world.physics_steps,
                        wall_s=elapsed,
                    )
                    next_report = elapsed + 30
            dataset = recorder.finalize(episode_success=True)
            result.update(
                success=True,
                completed=True,
                dataset="lerobot_v3",
                frames=len(recorder.rows),
                captured_frames=len(recorder.rows),
                dataset_use="demonstration",
                error=None,
                capture_wall_s=time.monotonic() - started,
                maximum_state_visual_fk_error_m=maximum_fk_error,
            )
            result["checks"]["rgb_capture_complete"] = True
            (dataset / "meta/task_outcome.json").write_text(
                json.dumps(dict(physical_result=result, configuration=configurations), indent=2)
                + "\n"
            )
            import pyarrow.parquet as pq

            stored = pq.read_table(dataset / "data/chunk-000/file-000.parquet")
            np.testing.assert_array_equal(
                np.array(stored["observation.state"].to_pylist(), np.float32),
                states[:: self.record_stride],
            )
            np.testing.assert_array_equal(
                np.array(stored["action"].to_pylist(), np.float32), actions[:: self.record_stride]
            )
        except Exception as exc:
            result.update(success=False, error=f"{type(exc).__name__}: {exc}")
            result["checks"]["rgb_capture_complete"] = False
            if not (output / "lerobot_v3").exists():
                recorder.discard(run)
            raise
        finally:
            (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> int:
    """Validate the source and completed dataset in the parent process, beyond Kit shutdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        PackageTrajectoryRenderJob(args.source, args.output).run()
        return 0
    if args.validate_only:
        return 0 if validate_rendered_trajectory(args.source, args.output)["success"] else 1
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.pop("RSR_SETTINGS_FILE", None)
    # Independent physics workers can finish together; Kit renderers must remain serial.
    import fcntl

    lock_path = ROOT / "outputs/debug/physic_solver/.package_render.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock, (args.output / "worker.log").open("w") as stream:
        fcntl.flock(lock, fcntl.LOCK_EX)
        process = subprocess.run(
            [
                str(ROOT / "app/python.sh"),
                str(Path(__file__).resolve()),
                str(args.source.resolve()),
                "--output",
                str(args.output.resolve()),
                "--worker",
            ],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    from scripts.physic_solver.collect_physic_solver_cases import _move_to_debug

    try:
        if process.returncode:
            raise RuntimeError(f"Capture worker exited with {process.returncode}")
        report = validate_rendered_trajectory(args.source, args.output)
        if not report["success"]:
            raise ValueError("Captured trajectory failed independent validation")
    except (OSError, ValueError, KeyError, AssertionError, RuntimeError):
        _move_to_debug(args.output.resolve())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
