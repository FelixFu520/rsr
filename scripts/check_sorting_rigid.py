"""Isolate the native rigid relay; its relocated soft parcel makes this diagnostic only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from real2sim2real.core.config import apply_settings_file
from real2sim2real.job.job_sorting import SortingJobConfig
from real2sim2real.planner.planner_sorting_stream import _RigidRelay, rigid_relay_config
from real2sim2real.utils.kit_process import run_kit_worker
from scripts.check_sorting_package import SortingPhysicsCheck
from scripts.validate_sorting_stream import soft_geometry_quality


class MeasuredRigidRelay(_RigidRelay):
    """Retain the previously deposited soft parcel throughout the rigid diagnostic."""

    def on_physics_step(self, robot: Any) -> bool:
        self.soft_samples.append(self.runtime.shared.vertices().astype(np.float32))
        return super().on_physics_step(robot)


class RigidCheck(SortingPhysicsCheck):
    """Keep the same rigid model and controller, excluding the earlier soft-relay history."""

    def simulate(self) -> None:
        runtime, config = self.planner.runtime, self.planner.config
        original = self.planner
        if self.soft_source:
            with np.load(self.soft_source / "particles.npz") as archive:
                saved = {key: archive[key] for key in archive.files}
            dt = self.sim_world.config.physics_dt
            q = saved["articulation_q"][-1]
            qd = np.zeros(runtime.model.joint_dof_count)
            qs = runtime.model.joint_q_start.numpy()
            vs = runtime.model.joint_qd_start.numpy()
            for joint, dimensions in enumerate(runtime.model.joint_dof_dim.numpy()):
                if sum(dimensions) == 1:
                    qd[vs[joint]] = (q[qs[joint]] - saved["articulation_q"][-2, qs[joint]]) / dt
            for state in (runtime.state_0, runtime.state_1):
                state.joint_q.assign(q)
                state.joint_qd.assign(qd)
                state.particle_q.assign(saved["positions"][-1] - runtime.physics_origin)
                state.particle_qd.assign((saved["positions"][-1] - saved["positions"][-2]) / dt)
                runtime.newton.eval_fk(runtime.model, state.joint_q, state.joint_qd, state)
            runtime.rigid_solver.reset(runtime.state_0, flags=0)
            runtime.rigid_solver._update_mjc_data(
                runtime.rigid_solver.mj_data, runtime.model, runtime.state_0
            )
            runtime.target_q = q.copy()
            indices = [
                *self.robot.arm_indices["left"],
                *self.robot.arm_indices["right"],
                self.robot.gripper_indices["left"],
                self.robot.gripper_indices["right"],
            ]
            runtime.target_q[indices] = saved["action"][-1]
            runtime.control.joint_target_q.assign(runtime.target_q)
            for side in ("left", "right"):
                runtime.iks[side].q = q[self.robot.arm_indices[side]].copy()
        else:
            # Explicit diagnostic isolation before time zero.
            points = runtime.state_0.particle_q.numpy()
            points += (
                np.array([10.0, 0.0, self.scene.config.floor_z_m + 0.001])
                - np.array([points[:, 0].mean(), points[:, 1].mean(), points[:, 2].min()])
                - np.array([0.0, 0.0, runtime.physics_origin[2]])
            )
            for state in (runtime.state_0, runtime.state_1):
                state.particle_q.assign(points)
                state.particle_qd.zero_()
            runtime.soft_asleep = True
        runtime.spawn_carton()
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        solver = runtime.rigid_solver
        solver._mujoco.mj_saveModel(solver.mj_model, str(output / "native_model.mjb"), None)
        np.savez(
            output / "native_state.npz",
            qpos=solver.mj_data.qpos,
            qvel=solver.mj_data.qvel,
            mocap_pos=solver.mj_data.mocap_pos,
            mocap_quat=solver.mj_data.mocap_quat,
            geom_shapes=runtime._mj_shapes,
            body_ids=runtime._mj_bodies,
            carton_body=runtime.carton_mj_body,
            carton_geom=runtime.carton_geom,
            belt_geoms=list(runtime.belt_geoms),
        )
        (output / "native_geometry.json").write_text(
            json.dumps(
                {
                    "disableflags": int(solver.mj_model.opt.disableflags),
                    "exclude": solver.mj_model.exclude_signature.tolist(),
                    "body_names": [
                        solver._mujoco.mj_id2name(
                            solver.mj_model, solver._mujoco.mjtObj.mjOBJ_BODY, i
                        )
                        for i in range(solver.mj_model.nbody)
                    ],
                    "geoms": [
                        dict(
                            id=i,
                            shape=int(shape),
                            position=solver.mj_data.geom_xpos[i].tolist(),
                            size=solver.mj_model.geom_size[i].tolist(),
                            contype=int(solver.mj_model.geom_contype[i]),
                            conaffinity=int(solver.mj_model.geom_conaffinity[i]),
                        )
                        for i, shape in enumerate(runtime._mj_shapes)
                        if shape >= runtime.robot_shape_count
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        rigid = MeasuredRigidRelay(config=rigid_relay_config(config))
        rigid.soft_samples = []
        rigid.initialize(self.robot, self.scene, runtime)
        self.planner = rigid
        super().simulate()
        path = Path(self.config.output_dir) / "result.json"
        result = json.loads(path.read_text())
        soft = np.asarray(rigid.soft_samples)
        quality = soft_geometry_quality(
            soft, original.rest, self.scene.triangle_indices.reshape(-1, 3)
        )
        quality["minimum_floor_z_m"] = float(soft[:, :, 2].min())
        result.update(
            accepted_demonstration=False,
            diagnostic_isolated_rigid=not bool(self.soft_source),
            diagnostic_restart=bool(self.soft_source),
            soft_source=str(self.soft_source) if self.soft_source else None,
            soft_entire_rigid_phase_quality=quality,
        )
        if self.soft_source:
            result["success"] = bool(
                result["success"]
                and quality["maximum_edge_stretch"] < config.maximum_edge_stretch
                and quality["minimum_triangle_area_ratio"] > config.minimum_triangle_area_ratio
                and quality["minimum_floor_z_m"]
                >= self.scene.config.floor_z_m - config.maximum_table_penetration_m
            )
        np.savez_compressed(output / "soft_history.npz", positions=soft)
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/PhysicSolver_mujoco_vbd_sorting_stream.toml")
    parser.add_argument("--output", type=Path, default=Path("outputs/debug/stream_rigid_probe"))
    parser.add_argument("--worker-id", help=argparse.SUPPRESS)
    parser.add_argument("--soft-source", type=Path)
    args = parser.parse_args()
    if args.worker_id is None:
        raise SystemExit(
            run_kit_worker(
                __file__,
                sys.argv[1:],
                args.output / "result.json",
                marker_field="physics_run_id",
                require_success=True,
            )
        )
    apply_settings_file(args.config)
    config = SortingJobConfig.from_settings("job")
    config.output_dir = str(args.output)
    job = RigidCheck(config=config)
    job.worker_id = args.worker_id
    job.soft_source = args.soft_source
    job.run()
