"""Screen a post-infeed grasp from a saved state; restored runs are diagnostic only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from real2sim2real.core.config import apply_settings_file
from real2sim2real.job.job_sorting import SortingJobConfig
from real2sim2real.planner.planner_sorting import SortingPlanner
from real2sim2real.utils.kit_process import run_kit_worker
from scripts.check_sorting_package import SortingPhysicsCheck


class SoftGraspCheck(SortingPhysicsCheck):
    """Restore measured particles/joints before descent to screen local grasp geometry."""

    def simulate(self) -> None:
        original, runtime = self.planner, self.planner.runtime
        with np.load(self.source / "particles.npz") as archive:
            saved = {key: archive[key] for key in archive.files}
        source_rows = json.loads((self.source / "trajectory.json").read_text())
        checkpoint = next(
            row["time_s"] for row in source_rows if row["phase"] == f"{self.side}_descend"
        )
        index = int(np.abs(saved["times"] - checkpoint).argmin())
        dt = self.sim_world.config.physics_dt
        q = saved["articulation_q"][index].copy()
        qd = np.zeros(runtime.model.joint_dof_count)
        qstarts = runtime.model.joint_q_start.numpy()
        vstarts = runtime.model.joint_qd_start.numpy()
        for joint, dimensions in enumerate(runtime.model.joint_dof_dim.numpy()):
            if sum(dimensions) == 1:
                qd[vstarts[joint]] = (
                    q[qstarts[joint]] - saved["articulation_q"][index - 1, qstarts[joint]]
                ) / dt
        position = saved["positions"][index] - runtime.physics_origin
        velocity = (saved["positions"][index] - saved["positions"][index - 1]) / dt
        for state in (runtime.state_0, runtime.state_1):
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            state.particle_q.assign(position)
            state.particle_qd.assign(velocity)
            runtime.newton.eval_fk(runtime.model, state.joint_q, state.joint_qd, state)
        runtime.rigid_solver.reset(runtime.state_0, flags=0)
        runtime.rigid_solver._update_mjc_data(
            runtime.rigid_solver.mj_data, runtime.model, runtime.state_0
        )
        runtime.rigid_solver._mujoco.mj_forward(
            runtime.rigid_solver.mj_model, runtime.rigid_solver.mj_data
        )
        runtime.target_q = q.copy()
        recorded = [
            *self.robot.arm_indices["left"],
            *self.robot.arm_indices["right"],
            self.robot.gripper_indices["left"],
            self.robot.gripper_indices["right"],
        ]
        runtime.target_q[recorded] = saved["action"][index]
        runtime.control.joint_target_q.assign(runtime.target_q)
        self.scene.config.infeed_travel_m = 0.0
        changes = {
            "grasp_offset_xy_m": (self.offset_x, self.offset_y),
            "grasp_pad_bottom_m": self.pad_bottom,
            "infeed_grasp_pad_bottom_m": self.pad_bottom if self.side == "left" else None,
        }
        if self.surface_indent is not None:
            changes["regrasp_surface_indent_m"] = self.surface_indent
        config = original.config.model_copy(update=changes)
        planner = SortingPlanner(config=config)
        planner.robot, planner.scene, planner.runtime = self.robot, self.scene, runtime
        planner._initialize_tracking()
        planner.rest, planner.rest_areas = original.rest, original.rest_areas
        planner._edge_indices, planner._edge_lengths = (
            original._edge_indices,
            original._edge_lengths,
        )
        approach = next(
            i
            for i, (side, phase, _) in enumerate(planner.schedule)
            if side == self.side and phase == "approach"
        )
        planner._enter(approach, planner._coms(runtime.vertices()))
        planner.elapsed = float(planner.ends[approach])
        planner._frame = round(planner.elapsed / dt)
        planner.state.update(ready=True, done=False)
        self.planner = planner
        try:
            super().simulate()
        finally:
            path = Path(self.config.output_dir) / "result.json"
            if path.exists():
                result = json.loads(path.read_text())
                tested = planner.legs[self.side]
                passed = bool(
                    planner.failure_reason is None
                    and tested["transfer_frames"] >= round(config.transfer_s / dt)
                    and tested["contact_frames"] / max(1, tested["transfer_frames"]) >= 0.95
                    and len(planner.online_held_shape_samples) >= 3
                )
                result.update(
                    accepted_demonstration=False,
                    diagnostic_restart=True,
                    source_cache=str(self.source),
                    source_time_s=float(saved["times"][index]),
                    tested_side=self.side,
                    probe_success=passed,
                    success=passed,
                )
                path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset-x", type=float, default=0.0)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument("--pad-bottom", type=float, default=0.015)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--surface-indent", type=float)
    parser.add_argument("--config", default="configs/PhysicSolver_mujoco_vbd_sorting_stream.toml")
    parser.add_argument("--worker-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_id is None:
        raise SystemExit(
            run_kit_worker(
                __file__,
                sys.argv[1:],
                args.output / "result.json",
                marker_field="physics_run_id",
                require_success=False,
            )
        )
    apply_settings_file(args.config)
    config = SortingJobConfig.from_settings("job")
    config.output_dir, config.max_steps = str(args.output), 780
    job = SoftGraspCheck(config=config)
    job.source, job.offset_x, job.offset_y = args.source, args.offset_x, args.offset_y
    job.pad_bottom, job.worker_id = args.pad_bottom, args.worker_id
    job.side, job.surface_indent = args.side, args.surface_indent
    job.run()
