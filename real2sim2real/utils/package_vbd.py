"""Translation-invariant frame for Newton's conservative self-contact bounds.

Subtract one common candidate displacement before DAT, restore it afterwards.
Relative trajectories (and therefore self-intersection) are unchanged by this
coordinate translation. No world-space motion or deformation is prescribed.
Applicable here to the freely simulated, unpinned bag/garment particles.
Adapted from the supplied package project's moving_frame_stable.py for Newton 1.6.1.
"""

from typing import Any

import newton
import numpy as np
import warp as wp


@wp.kernel
def partial_displacement(
    delta: wp.array[wp.vec3], weights: wp.array[float], partial: wp.array[wp.vec3]
):
    i = wp.tid()
    total = wp.vec3(0.0)
    for j in range(i, weights.shape[0], 64):
        total += weights[j] * delta[j]
    partial[i] = total


@wp.kernel
def sum_displacement(partial: wp.array[wp.vec3], mean: wp.array[wp.vec3]):
    total = wp.vec3(0.0)
    for j in range(64):
        total += partial[j]
    mean[0] = total


@wp.kernel
def shift_displacement(delta: wp.array[wp.vec3], mean: wp.array[wp.vec3], sign: float):
    i = wp.tid()
    delta[i] += sign * mean[0]


@wp.kernel
def apply_displacement(base: wp.array[wp.vec3], delta: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = base[i] + delta[i]


class MovingFrameVBD(newton.solvers.SolverVBD):
    """V7's moving DAT frame, with capacity checks over every physical substep."""

    def __init__(self, model: Any, moving_frame: bool = True, **kwargs: Any) -> None:
        self.moving_frame = moving_frame
        super().__init__(model, **kwargs)
        mass = model.particle_mass.numpy().astype(np.float64)
        if moving_frame and (np.any(mass <= 0) or np.any(model.particle_inv_mass.numpy() <= 0)):
            raise ValueError("Moving contact frame requires freely simulated, unpinned particles")
        self.motion_weights = wp.array(mass / mass.sum(), dtype=float, device=model.device)
        self.common_motion = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self.motion_partial = wp.empty(64, dtype=wp.vec3, device=model.device)
        self.maximum_contacts = wp.zeros(3, dtype=int, device=model.device)

    def step(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float) -> None:
        """Advance VBD and keep maxima so a transient overflow cannot pass validation."""
        super().step(state_in, state_out, control, contacts, dt)
        info = self.trimesh_collision_detector.collision_info
        for slot, counts in enumerate(
            (
                info.vertex_colliding_triangles_count,
                info.edge_colliding_edges_count,
                contacts.soft_contact_count,
            )
        ):
            wp.launch(
                audit_contact_capacity,
                64,
                inputs=[counts, self.maximum_contacts, slot],
                device=self.device,
            )

    def _penetration_free_truncation(self, particle_q_out: Any = None) -> None:
        if not self.moving_frame:
            return super()._penetration_free_truncation(particle_q_out)
        wp.launch(
            partial_displacement,
            64,
            inputs=[self.particle_displacements, self.motion_weights, self.motion_partial],
            device=self.device,
        )
        wp.launch(
            sum_displacement,
            1,
            inputs=[self.motion_partial, self.common_motion],
            device=self.device,
        )
        wp.launch(
            shift_displacement,
            self.model.particle_count,
            inputs=[self.particle_displacements, self.common_motion, -1.0],
            device=self.device,
        )
        super()._penetration_free_truncation(None)
        wp.launch(
            shift_displacement,
            self.model.particle_count,
            inputs=[self.particle_displacements, self.common_motion, 1.0],
            device=self.device,
        )
        if particle_q_out is not None:
            wp.launch(
                apply_displacement,
                self.model.particle_count,
                inputs=[
                    self.pos_prev_collision_detection,
                    self.particle_displacements,
                    particle_q_out,
                ],
                device=self.device,
            )


@wp.kernel
def audit_contact_capacity(counts: wp.array[int], maximum: wp.array[int], slot: int):
    value = int(0)  # noqa: UP018 - Warp requires a dynamic accumulator in this loop.
    for i in range(wp.tid(), counts.shape[0], 64):
        value = wp.max(value, counts[i])
    wp.atomic_max(maximum, slot, value)
