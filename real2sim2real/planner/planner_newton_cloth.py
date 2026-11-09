# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Adapt Newton 1.6.1 cloth_franka's keyframe/Jacobian controller to the RSR lifecycle.

Reference: newton/examples/cloth/example_cloth_franka.py (Apache-2.0).
Physics stays in NewtonClothRuntime; the Job owns stepping, recording and cleanup.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from real2sim2real.planner.newton_cloth_keyframes import OFFICIAL_KEYFRAMES
from real2sim2real.planner.planner_physic_solver import (
    NewtonClothRuntime,
    PickPlaceClothPlanner,
    PickPlaceClothPlannerConfig,
)


class OfficialClothPlannerConfig(PickPlaceClothPlannerConfig):
    """Official controller/material profile; the first seven keyframes perform one fold."""

    backend: str = "newton_cloth_franka"
    keyframe_limit: int = Field(default=7, ge=7, le=len(OFFICIAL_KEYFRAMES))
    final_settle_s: float = Field(default=2.0, gt=0)
    cloth_density_cm: float | None = Field(default=None, gt=0)
    contact_stiffness: float = Field(default=1e4, gt=0)
    contact_damping: float = Field(default=10.0, ge=0)
    shape_contact_stiffness: float = Field(default=5e4, gt=0)
    shape_contact_damping: float = Field(default=50.0, ge=0)
    shape_contact_friction: float = Field(default=1.5, ge=0)
    table_friction: float = Field(default=1.5, ge=0)
    friction_epsilon_cm_s: float = Field(default=0.01, gt=0)
    minimum_lift_m: float = Field(default=0.03, gt=0)
    minimum_release_drop_m: float = Field(default=0.01, ge=0)
    open_finger_m: float = Field(default=0.032, gt=0, le=0.04)
    closed_finger_m: float = Field(default=0.004, ge=0, le=0.04)
    full_surface_contacts: bool = False
    contact_sdf_resolution: int = Field(default=64, ge=16, le=256)
    keyframe_offset_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.0, 0.0, 0.0)
    finger_velocity_gain: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _finger_range(self) -> OfficialClothPlannerConfig:
        if self.closed_finger_m >= self.open_finger_m:
            raise ValueError("closed_finger_m must be smaller than open_finger_m")
        return self


def _soft_contact_geometry(
    particles: np.ndarray,
    radii: np.ndarray,
    body_q: np.ndarray,
    shape_body: np.ndarray,
    shape_margin: np.ndarray,
    shape_ids: np.ndarray,
    indices: np.ndarray,
    barycentric: np.ndarray,
    body_points: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bodies, raw signed distances and active masks, in Newton model units.

    Mirror VBD's particle/rigid contact geometry: collision candidates outside
    particle radius + shape margin exert no contact force and must not count.
    """
    bodies = shape_body[shape_ids]
    valid = indices >= 0
    safe = np.maximum(indices, 0)
    weights = np.where(valid, barycentric, 0.0)
    points = np.sum(particles[safe] * weights[:, :, None], axis=1)
    radius = np.max(np.where(valid, radii[safe], 0.0), axis=1)
    surface = body_points.copy()
    moving = bodies >= 0
    transforms = body_q[bodies[moving]]
    xyz, w = transforms[:, 3:6], transforms[:, 6:7]
    local = surface[moving]
    cross = 2 * np.cross(xyz, local)
    surface[moving] = local + w * cross + np.cross(xyz, cross) + transforms[:, :3]
    distance = np.sum(normals * (points - surface), axis=1)
    margin = shape_margin[shape_ids] if shape_margin.size else 0.0
    return bodies, distance, distance < radius + margin


@cache
def _control_kernels() -> tuple[Any, Any]:
    import warp as wp

    @wp.kernel
    def body_velocity(
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_com: wp.array[wp.vec3],
        body_id: int,
        offset: wp.vec3,
        output: wp.array[float],
    ):
        transform = body_q[body_id]
        radius = wp.transform_vector(transform, offset - body_com[body_id])
        velocity = body_qd[body_id]
        omega = wp.spatial_bottom(velocity)
        linear = wp.spatial_top(velocity) + wp.cross(omega, radius)
        output[0] = linear[0]
        output[1] = linear[1]
        output[2] = linear[2]
        output[3] = omega[0]
        output[4] = omega[1]
        output[5] = omega[2]

    @wp.kernel
    def end_effector_error(
        body_q: wp.array[wp.transform],
        body_id: int,
        offset: wp.transform,
        target: wp.transform,
        output: wp.array[wp.spatial_vector],
    ):
        actual = body_q[body_id] * offset
        position = wp.transform_get_translation(target) - wp.transform_get_translation(actual)
        rotation = wp.transform_get_rotation(target) * wp.quat_inverse(
            wp.transform_get_rotation(actual)
        )
        output[0] = wp.spatial_vector(
            position[0], position[1], position[2], rotation[0], rotation[1], rotation[2]
        )

    return body_velocity, end_effector_error


class OfficialClothRuntime(NewtonClothRuntime):
    """Use the official velocity controller and flat bending rest angles with shared stepping."""

    def __init__(self, robot: Any, scene: Any, config: Any) -> None:
        if not robot.pure_newton or robot.sim_world.config.newton_solver != "featherstone":
            raise ValueError("Official cloth_franka profile requires direct Newton Featherstone")
        if robot.config.control_mode != "velocity":
            raise ValueError("Official cloth_franka profile requires robot.control_mode=velocity")
        super().__init__(robot, scene, config)
        self.model.edge_rest_angle.zero_()
        self.model.shape_material_ke.fill_(config.shape_contact_stiffness)
        self.model.shape_material_kd.fill_(config.shape_contact_damping)
        self.model.shape_material_mu.assign(
            np.where(
                self.model.shape_body.numpy() < 0,
                config.table_friction,
                config.shape_contact_friction,
            )
        )

    def _configure_builder(self, builder: Any) -> None:
        if self.config.full_surface_contacts:
            builder.default_shape_cfg.configure_sdf(
                force_sdf=True, max_resolution=self.config.contact_sdf_resolution
            )

    def _create_collision_pipeline(self) -> Any:
        return self.newton.CollisionPipeline(
            self.model,
            soft_contact_gap=self.config.particle_radius_m * self.scale,
            enable_rigid_soft_full_surface_contact=self.config.full_surface_contacts,
        )

    def _publish(self) -> None:
        super()._publish()
        contacts, model = self.contacts, self.model
        count = min(self.last_contacts, contacts.soft_contact_shape.shape[0])
        if not count:
            self.last_active_contact_bodies = np.empty(0, dtype=int)
            self.last_contact_distances_m = dict.fromkeys(self.contact_body_ids)
            self.last_contact_particle_ids = {
                name: np.empty(0, dtype=int) for name in ("left", "right")
            }
            return
        shape_ids = contacts.soft_contact_shape[:count].numpy()
        valid = (shape_ids >= 0) & (shape_ids < model.shape_count)
        indices = contacts.soft_contact_indices[:count].numpy()[valid]
        bodies, distances, active = _soft_contact_geometry(
            self.state_0.particle_q.numpy(),
            model.particle_radius.numpy(),
            self.state_0.body_q.numpy(),
            model.shape_body.numpy(),
            model.shape_margin.numpy(),
            shape_ids[valid],
            indices,
            contacts.soft_contact_barycentric[:count].numpy()[valid],
            contacts.soft_contact_body_pos[:count].numpy()[valid],
            contacts.soft_contact_normal[:count].numpy()[valid],
        )
        self.last_contact_counts = {
            name: int(np.count_nonzero(active & (bodies == body)))
            for name, body in self.contact_body_ids.items()
        }
        self.last_active_contact_bodies = bodies[active].copy()
        self.last_contact_counts["table"] = int(np.count_nonzero(active & (bodies < 0)))
        self.last_contact_particle_ids = {}
        for name in ("left", "right"):
            ids = indices[active & (bodies == self.contact_body_ids[name])]
            self.last_contact_particle_ids[name] = np.unique(ids[ids >= 0])
        self.last_contact_distances_m = {
            name: float(distances[mask].min() / self.scale) if mask.any() else None
            for name, body in self.contact_body_ids.items()
            for mask in [bodies == body]
        }

    def _create_ik(self) -> None:
        super()._create_ik()
        # The reference targets a point 22 cm from the collapsed final arm link.
        self.tcp_offset_cm = 22.0
        wp = self.wp
        self._velocity_kernel, self._error_kernel = _control_kernels()
        self._jacobian_state = self.model.state(requires_grad=True)
        self._body_velocity = wp.empty(6, dtype=float, requires_grad=True)
        self._jacobian = wp.empty(6 * self.model.joint_dof_count, dtype=float)
        self._error = wp.empty(1, dtype=wp.spatial_vector)
        self._onehots = [wp.array(row, dtype=float) for row in np.eye(6, dtype=np.float32)]
        self._velocity_target = wp.zeros(self.model.joint_dof_count, dtype=float)
        self._posture = self.model.joint_q.numpy().copy()

    def track_keyframe(
        self, target_m: np.ndarray, quaternion_xyzw: np.ndarray, activation: float
    ) -> None:
        """Compute the reference pseudoinverse/nullspace velocity command once per physics frame."""
        wp = self.wp
        q, qd = self.state_0.joint_q, self.state_0.joint_qd
        q.requires_grad = qd.requires_grad = True
        offset = wp.vec3(0.0, 0.0, self.tcp_offset_cm)
        wp.launch(
            self._error_kernel,
            dim=1,
            inputs=[
                self.state_0.body_q,
                self.hand_body,
                wp.transform(offset, wp.quat_identity()),
                wp.transform(target_m * self.scale, quaternion_xyzw),
            ],
            outputs=[self._error],
            device=self.model.device,
        )
        tape = wp.Tape()
        with tape:
            self.newton.eval_fk(self.model, q, qd, self._jacobian_state)
            wp.launch(
                self._velocity_kernel,
                dim=1,
                inputs=[
                    self._jacobian_state.body_q,
                    self._jacobian_state.body_qd,
                    self.model.body_com,
                    self.hand_body,
                    offset,
                ],
                outputs=[self._body_velocity],
                device=self.model.device,
            )
        for row in range(6):
            tape.backward(grads={self._body_velocity: self._onehots[row]})
            wp.copy(
                self._jacobian[
                    row * self.model.joint_dof_count : (row + 1) * self.model.joint_dof_count
                ],
                qd.grad,
            )
            tape.zero()
        jacobian = self._jacobian.numpy().reshape(6, -1)
        inverse = np.linalg.pinv(jacobian)
        current = q.numpy()
        posture = self._posture.copy()
        posture[0] = current[0]
        velocity = inverse @ self._error.numpy()[0] + (
            np.eye(len(current)) - inverse @ jacobian
        ) @ (posture - current)
        # Reference activations 0.1/0.8 mean 4/32 mm per finger. Keep the
        # trajectory while allowing the physical opening to match cloth thickness.
        blend = np.clip((activation - 0.1) / 0.7, 0.0, 1.0)
        opening = self.config.closed_finger_m + blend * (
            self.config.open_finger_m - self.config.closed_finger_m
        )
        velocity[-2:] = self.config.finger_velocity_gain * (opening * self.scale - current[-2:])
        limits = self.model.joint_velocity_limit.numpy()[-2:]
        velocity[-2:] = np.clip(velocity[-2:], -limits, limits)
        if not np.isfinite(velocity).all():
            raise FloatingPointError("Official controller generated a non-finite velocity")
        self._velocity_target.assign(velocity)
        self.target_q = current + velocity * self.robot.sim_world.config.physics_dt

    def hold(self) -> None:
        """Stop joint motion after the final reference keyframe."""
        self._velocity_target.zero_()
        self.target_q = self.state_0.joint_q.numpy().copy()

    def recording_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Record actual positions and the commanded joint velocities in rad/s and m/s."""
        state, _ = super().recording_coordinates()
        velocity = self._velocity_target.numpy().astype(float).copy()
        velocity[-2:] /= self.scale
        return state, velocity

    def _apply_rigid_control(self) -> None:
        self.state_0.joint_qd.assign(self._velocity_target)


class OfficialClothPlanner(PickPlaceClothPlanner):
    """Execute the official folding sequence; report measured contact, lift and release."""

    config_cls = OfficialClothPlannerConfig
    runtime_cls = OfficialClothRuntime
    action_names = [f"fr3_joint{i}_velocity_rad_s" for i in range(1, 8)] + [
        "left_finger_velocity_m_s",
        "right_finger_velocity_m_s",
    ]

    @property
    def expected_duration_s(self) -> float:
        """Return the reference sequence and final settling duration in simulation seconds."""
        return sum(row[0] for row in OFFICIAL_KEYFRAMES[: self.config.keyframe_limit]) + (
            self.config.final_settle_s
        )

    def begin_episode(self, *, register_stepper: bool = True) -> None:
        """Build the shared runtime, then initialize the official reference schedule."""
        super().begin_episode(register_stepper=register_stepper)
        self._keyframes = np.asarray(OFFICIAL_KEYFRAMES[: self.config.keyframe_limit], dtype=float)
        self._ends = np.cumsum(self._keyframes[:, 0])
        self._reference_origin = self.scene.official_frame_translation
        self._initial_z: np.ndarray | None = None
        self._peak_lift = 0.0
        self._peak_contact_count = dict(left=0, right=0)
        self._bilateral_grasp = False
        self._release_initial_z: float | None = None
        self._release_particle_ids = np.empty(0, dtype=int)
        self._release_drop = 0.0
        self._previous_closed = False
        self._transfer_frames = 0
        self._transfer_contact_frames = 0

    def on_physics_step(self, robot: Any) -> bool:
        """Advance control time exactly once; physics integration belongs to sim_world.step()."""
        del robot
        index = int(np.searchsorted(self._ends, self.elapsed, side="right"))
        vertices = self.runtime.vertices()
        if self.elapsed >= self._ends[-1] + self.config.final_settle_s:
            self.runtime.hold()
            self.state.update(done=True, phase="done")
            return True
        if self._initial_z is None and self.elapsed >= self._ends[0]:
            self._initial_z = vertices[:, 2].copy()
        if self._initial_z is not None:
            # Robust lifted patch metric: average the twenty most lifted physical vertices.
            lifted = vertices[:, 2] - self._initial_z
            self._peak_lift = max(self._peak_lift, float(np.sort(lifted)[-20:].mean()))
        counts = self.runtime.last_contact_counts
        if self._ends[3] <= self.elapsed < self._ends[5]:
            self._transfer_frames += 1
            self._transfer_contact_frames += int(counts["left"] > 0 and counts["right"] > 0)
        for side in self._peak_contact_count:
            self._peak_contact_count[side] = max(self._peak_contact_count[side], counts[side])
        if index < len(self._keyframes):
            row = self._keyframes[index]
            if row[8] < 0.5 and counts["left"] > 0 and counts["right"] > 0:
                self._bilateral_grasp = True
            target = row[1:4] / 100.0 + self._reference_origin + self.config.keyframe_offset_m
            self.runtime.track_keyframe(target, row[4:8], row[8])
            phase = f"official_{index:02d}_{'close' if row[8] < 0.5 else 'open'}"
        else:
            self.runtime.hold()
            phase = "settle_after_release"
        closed = index < len(self._keyframes) and self._keyframes[index, 8] < 0.5
        if not closed and self._previous_closed:
            self._release_particle_ids = np.union1d(
                self.runtime.last_contact_particle_ids["left"],
                self.runtime.last_contact_particle_ids["right"],
            )
            if self._release_particle_ids.size:
                self._release_initial_z = float(vertices[self._release_particle_ids, 2].mean())
        if closed:
            self._release_initial_z = None
        elif self._release_initial_z is not None:
            # Follow the same patch that was held at opening, rather than selecting
            # different highest vertices from the rest of the shirt every frame.
            patch_z = float(vertices[self._release_particle_ids, 2].mean())
            self._release_drop = max(self._release_drop, self._release_initial_z - patch_z)
        self._previous_closed = bool(closed)
        self.state["phase"] = phase
        self.rows.append(
            {
                "phase": phase,
                "time_s": self.elapsed,
                "target_z_m": float(vertices[:, 2].max()),
                "contacts_left": counts["left"],
                "contacts_right": counts["right"],
            }
        )
        self.elapsed += self.robot.sim_world.config.physics_dt
        return False

    def result(self) -> dict[str, Any]:
        """Describe the measured demonstration without claiming a learned policy or full fold."""
        vertices = self.runtime.vertices()
        release_drop = self._release_drop
        transfer_contact_fraction = self._transfer_contact_frames / max(1, self._transfer_frames)
        checks = {
            "sequence_completed": bool(self.state.get("done")),
            "finite_cloth_state": bool(np.isfinite(vertices).all()),
            "bilateral_contact_observed": self._bilateral_grasp,
            "grasp_maintained_during_transfer": transfer_contact_fraction >= 0.95,
            "cloth_lifted": self._peak_lift >= self.config.minimum_lift_m,
            "cloth_dropped_after_release": release_drop >= self.config.minimum_release_drop_m,
            "no_pinned_vertices": self.runtime.pinned_vertex_count == 0,
            "no_attachments": True,
        }
        return {
            "success": all(checks.values()),
            "checks": checks,
            "strategy": "newton_cloth_franka_keyframes",
            "keyframes_executed": self.config.keyframe_limit,
            "full_official_sequence": self.config.keyframe_limit == len(OFFICIAL_KEYFRAMES),
            "maximum_patch_lift_m": self._peak_lift,
            "release_drop_m": release_drop,
            "release_measurement": "mean_height_of_last_grasped_contact_patch",
            "maximum_contacts": self._peak_contact_count,
            "transfer_bilateral_contact_fraction": transfer_contact_fraction,
            "contact_counts_kind": "inside_particle_radius_plus_shape_margin",
            "full_surface_contacts": self.config.full_surface_contacts,
            "closed_finger_m": self.config.closed_finger_m,
            "keyframe_offset_m": self.config.keyframe_offset_m,
            "particle_radius_m": self.config.particle_radius_m,
            "cloth_total_mass_kg": float(self.runtime.model.particle_mass.numpy().sum()),
            "physics_duration_s": self.elapsed,
            "attachments_used": False,
            "pinned_vertex_count": self.runtime.pinned_vertex_count,
            "coupling": "one_way_rigid_to_cloth",
        }
