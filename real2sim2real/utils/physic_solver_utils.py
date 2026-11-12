"""物理求解器验证任务的辅助函数（分拣任务复用其中的 VBD 与包裹几何）。

- ``KinematicVBDStepper``：hybrid 世界中由外部给定刚体位姿驱动 VBD 的缓冲步进。
- ``MovingFrameVBD`` 等：平移不变的 VBD 自接触界限坐标系与接触容量审计。
- ``check_package_containment`` 等：闭合袋膜的 NumPy 射线奇偶与点到三角形距离检查。
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

# ================================================================================================
# Buffered Newton VBD stepping for externally prescribed kinematic colliders.
# ================================================================================================


class KinematicVBDStepper:
    """Replay the same VBD substeps in a CUDA graph, updating collider inputs each frame."""

    def __init__(
        self,
        model: Any,
        solver: Any,
        pipeline: Any,
        contacts: Any,
        state_0: Any,
        state_1: Any,
        control: Any,
        substeps: int,
        dt: float,
        *,
        use_graph: bool = True,
    ) -> None:
        import warp as wp

        self.wp = wp
        self.model, self.solver, self.pipeline, self.contacts = model, solver, pipeline, contacts
        self.state_0, self.state_1, self.control = state_0, state_1, control
        self.substeps, self.dt = substeps, dt
        self.use_graph = use_graph and model.device.is_cuda and substeps % 2 == 0
        self.graph = None
        self.warmed = False
        count = substeps * model.body_count
        self.q_in = wp.empty(count, dtype=wp.transform, device=model.device)
        self.q_out = wp.empty(count, dtype=wp.transform, device=model.device)
        self.velocity = wp.empty(count, dtype=wp.spatial_vector, device=model.device)

    def step(self, q_in: np.ndarray, q_out: np.ndarray, velocity: np.ndarray) -> None:
        """Advance one frame from explicitly supplied substep collider transforms/velocities."""
        self.q_in.assign(np.asarray(q_in, dtype=np.float32).reshape(-1, 7))
        self.q_out.assign(np.asarray(q_out, dtype=np.float32).reshape(-1, 7))
        self.velocity.assign(np.asarray(velocity, dtype=np.float32).reshape(-1, 6))
        if self.use_graph and self.warmed:
            if self.graph is None:
                with self.wp.ScopedCapture(device=self.model.device) as capture:
                    self._simulate()
                self.graph = capture.graph
            self.wp.capture_launch(self.graph)
        else:
            self._simulate()
            self.warmed = True

    def _simulate(self) -> None:
        self.solver.rebuild_bvh(self.state_0)
        count = self.model.body_count
        for index in range(self.substeps):
            offset = index * count
            self.wp.copy(self.state_0.body_q, self.q_in, src_offset=offset, count=count)
            self.wp.copy(self.state_1.body_q, self.q_out, src_offset=offset, count=count)
            self.wp.copy(self.state_0.body_qd, self.velocity, src_offset=offset, count=count)
            self.wp.copy(self.state_1.body_qd, self.velocity, src_offset=offset, count=count)
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.wp.copy(self.state_0.body_q, self.q_out, src_offset=offset, count=count)
            self.wp.copy(self.state_1.body_q, self.q_out, src_offset=offset, count=count)


# ================================================================================================
# Translation-invariant frame for Newton's conservative self-contact bounds.
#
# Subtract one common candidate displacement before DAT, restore it afterwards.
# Relative trajectories (and therefore self-intersection) are unchanged by this
# coordinate translation. No world-space motion or deformation is prescribed.
# Applicable here to the freely simulated, unpinned bag/garment particles.
# Adapted from the supplied package project's moving_frame_stable.py for Newton 1.6.1.
# Newton/Warp are only importable inside Kit, so the kernels and the SolverVBD subclass are
# defined on first use; the NumPy helpers in this module stay importable without Kit.
# ================================================================================================


@cache
def moving_frame_vbd_class() -> type:
    """Return ``MovingFrameVBD``, defining its Warp kernels on the first call."""
    import newton
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
    def apply_displacement(
        base: wp.array[wp.vec3], delta: wp.array[wp.vec3], out: wp.array[wp.vec3]
    ):
        i = wp.tid()
        out[i] = base[i] + delta[i]

    @wp.kernel
    def audit_contact_capacity(counts: wp.array[int], maximum: wp.array[int], slot: int):
        value = int(0)  # noqa: UP018 - Warp requires a dynamic accumulator in this loop.
        for i in range(wp.tid(), counts.shape[0], 64):
            value = wp.max(value, counts[i])
        wp.atomic_max(maximum, slot, value)

    class MovingFrameVBD(newton.solvers.SolverVBD):
        """V7's moving DAT frame, with capacity checks over every physical substep."""

        def __init__(self, model: Any, moving_frame: bool = True, **kwargs: Any) -> None:
            self.moving_frame = moving_frame
            super().__init__(model, **kwargs)
            mass = model.particle_mass.numpy().astype(np.float64)
            if moving_frame and (np.any(mass <= 0) or np.any(model.particle_inv_mass.numpy() <= 0)):
                raise ValueError(
                    "Moving contact frame requires freely simulated, unpinned particles"
                )
            self.motion_weights = wp.array(mass / mass.sum(), dtype=float, device=model.device)
            self.common_motion = wp.zeros(1, dtype=wp.vec3, device=model.device)
            self.motion_partial = wp.empty(64, dtype=wp.vec3, device=model.device)
            self.maximum_contacts = wp.zeros(3, dtype=int, device=model.device)

        def step(
            self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float
        ) -> None:
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

    return MovingFrameVBD


# ================================================================================================
# Independent NumPy ray-parity and point-to-triangle checks for the closed parcel bag.
# ================================================================================================


def _inside(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    # An oblique ray avoids the axis-aligned grid edges in the authored parcel.
    direction = np.array([1.0, 0.137, 0.213])
    direction /= np.linalg.norm(direction)
    side = np.cross(direction, [0, 0, 1])
    side /= np.linalg.norm(side)
    basis = np.stack([direction, side, np.cross(direction, side)], axis=1)
    tri, probes = triangles @ basis, points @ basis
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v, w = b - a, c - a
    determinant = v[:, 1] * w[:, 2] - v[:, 2] * w[:, 1]
    valid = np.abs(determinant) > 1e-15
    determinant = np.where(valid, determinant, 1.0)
    result = []
    for chunk in np.array_split(probes, max(1, (len(probes) + 127) // 128)):
        delta = chunk[:, None] - a
        u = (delta[:, :, 1] * w[:, 2] - delta[:, :, 2] * w[:, 1]) / determinant
        t = (v[:, 1] * delta[:, :, 2] - v[:, 2] * delta[:, :, 1]) / determinant
        intersection_x = a[:, 0] + u * v[:, 0] + t * w[:, 0]
        hits = valid & (u >= 0) & (t >= 0) & (u + t <= 1) & (intersection_x > chunk[:, None, 0])
        result.append(np.count_nonzero(hits, axis=1) % 2 == 1)
    return np.concatenate(result)


def _surface_distance_dense(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    aa, bb, cc = (ab * ab).sum(1), (ab * ac).sum(1), (ac * ac).sum(1)
    det = np.maximum(aa * cc - bb * bb, 1e-30)
    result = []
    for chunk in np.array_split(points, max(1, (len(points) + 63) // 64)):
        ap = chunk[:, None] - a
        d, e = (ap * ab).sum(2), (ap * ac).sum(2)
        u, v = (cc * d - bb * e) / det, (aa * e - bb * d) / det
        projection = ap - u[:, :, None] * ab - v[:, :, None] * ac
        distance = np.where(
            (u >= 0) & (v >= 0) & (u + v <= 1), (projection * projection).sum(2), np.inf
        )
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end - start
            offset = chunk[:, None] - start
            t = np.clip((offset * edge).sum(2) / np.maximum((edge * edge).sum(1), 1e-30), 0, 1)
            residual = offset - t[:, :, None] * edge
            distance = np.minimum(distance, (residual * residual).sum(2))
        result.append(np.sqrt(distance.min(1)))
    return np.concatenate(result)


def _surface_distance(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Exact triangle distances with conservative AABB pruning for large probe sets."""
    if len(points) * len(triangles) <= 32768 or len(triangles) <= 32:
        return _surface_distance_dense(points, triangles)
    lower, upper = triangles.min(1), triangles.max(1)
    centres = (lower + upper) / 2

    def build(indices: np.ndarray) -> tuple:
        low, high = lower[indices].min(0), upper[indices].max(0)
        if len(indices) <= 32:
            return low, high, indices, None, None
        axis = int(np.ptp(centres[indices], axis=0).argmax())
        middle = len(indices) // 2
        ordered = indices[np.argpartition(centres[indices, axis], middle)]
        return low, high, None, build(ordered[:middle]), build(ordered[middle:])

    def bound(node: tuple, indices: np.ndarray) -> np.ndarray:
        delta = np.maximum(np.maximum(node[0] - points[indices], points[indices] - node[1]), 0)
        return np.sum(delta * delta, axis=1)

    best = np.full(len(points), np.inf)

    def visit(node: tuple, indices: np.ndarray) -> None:
        indices = indices[bound(node, indices) <= best[indices] ** 2]
        if not len(indices):
            return
        if node[2] is not None:
            best[indices] = np.minimum(
                best[indices], _surface_distance_dense(points[indices], triangles[node[2]])
            )
            return
        left, right = node[3], node[4]
        first_left = bound(left, indices) <= bound(right, indices)
        left_points, right_points = indices[first_left], indices[~first_left]
        visit(left, left_points)
        visit(right, right_points)
        visit(right, left_points)
        visit(left, right_points)

    visit(build(np.arange(len(triangles))), np.arange(len(points)))
    return best


def check_package_containment(
    positions: np.ndarray,
    triangles: np.ndarray,
    garment_count: int,
    *,
    times: np.ndarray,
    tolerance_m: float = 0.001,
    checkpoint_times_s: list[float] | None = None,
) -> dict:
    """Check all garment vertices and face centres at ten task checkpoints.

    Ray parity uses the closed, consistently triangulated bag. Outside probes
    within tolerance count as contact. This is sampled geometry evidence, not
    a continuous-time proof against self-intersection or tunnelling.
    """
    if not len(times):
        return dict(
            passed=False,
            tolerance_m=tolerance_m,
            checkpoints=[],
            error="No physical particle samples",
        )
    bag = triangles[np.all(triangles >= garment_count, axis=1)] - garment_count
    garment = triangles[np.all(triangles < garment_count, axis=1)]
    samples = checkpoint_times_s or [0, 2, 3.5, 5, 7, 8.5, 9.5, 10.5, 11.5, 13]
    indices = np.unique([np.abs(times - time).argmin() for time in samples])
    rows = []
    for index in indices:
        points = positions[index]
        probes = np.concatenate([points[:garment_count], points[garment].mean(axis=1)])
        surface = points[garment_count:][bag]
        inside = _inside(probes, surface)
        distances = _surface_distance(probes[~inside], surface) if (~inside).any() else np.zeros(1)
        rows.append(
            dict(
                time_s=float(times[index]),
                probes=len(probes),
                outside_tolerance=int(np.count_nonzero(distances > tolerance_m)),
                maximum_outside_distance_m=float(distances.max()),
            )
        )
    return dict(
        passed=all(row["outside_tolerance"] == 0 for row in rows),
        tolerance_m=tolerance_m,
        checkpoints=rows,
        method="oblique_ray_parity_and_exact_point_triangle_distance",
    )
