"""物理求解器验证任务的辅助函数（分拣任务复用其中的 VBD 与包裹几何）。

- ``KinematicVBDStepper``：hybrid 世界中由外部给定刚体位姿驱动 VBD 的缓冲步进。
- ``MovingFrameVBD`` 等：平移不变的 VBD 自接触界限坐标系与接触容量审计。
- ``VBDSubstepGraph`` / ``proxy_pose_*``：CPU/GPU 双向耦合的局部图捕获与代理几何核对。
- ``check_package_containment`` 等：闭合袋膜的 NumPy 射线奇偶与点到三角形距离检查。
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

# Canonical filenames: robot solver / contact solver / object / experiment variant.
# Historical keys identify prior experiments only; configurations and outputs use the values.
PHYSIC_SOLVER_CASE_NAMES = {
    "featherstone_vbd_pickplace_cloth": "featherstone_vbd_cloth01_baseline",
    "mujoco_vbd_pickplace_cloth": "mujoco_vbd_cloth02_baseline",
    "xpbd_vbd_pickplace_cloth": "xpbd_vbd_cloth02_baseline",
    "tgs_vbd_pickplace_cloth": "tgs_vbd_cloth02_baseline",
    "tgs_tgs_pickplace_rigid": "tgs_tgs_rigid01_baseline",
    "mujoco_vbd_pickplace_package": "mujoco_vbd_package01_center_native",
    "featherstone_vbd_pickplace_package": "featherstone_vbd_package01_center_control",
    "xpbd_xpbd_pickplace_cloth": "xpbd_xpbd_cloth02_baseline_control",
    "v7_center": "mujoco_vbd_package01_center_pads",
    "v8_offset_x35": "mujoco_vbd_package01_pos_x35mm",
    "v8_offset_y55": "mujoco_vbd_package01_pos_y55mm",
    "v8_yaw45": "mujoco_vbd_package01_pos_x35mm_yaw45deg",
    "v8_yaw90": "mujoco_vbd_package01_pos_y175mm_yaw90deg_twoway",
    "v8_offset_tilt10": "mujoco_vbd_package01_pos_x35mm_yaw30deg_pitch10deg",
    "v9_mass100g": "mujoco_vbd_package01_mass100g_pos_x35mm",
    "v9_mass200g": "mujoco_vbd_package01_mass200g_pos_x35mm",
    "v9_mass350g": "mujoco_vbd_package01_mass350g_pos_x35mm",
    "v9_mass500g": "mujoco_vbd_package01_mass500g_pos_x35mm",
    "v10_station_clothing": "mujoco_vbd_package02_center",
    "v10_station_green": "mujoco_vbd_package03_center",
    "v10_pink_tshirt": "mujoco_vbd_package04_center",
    "v10_frosted_top": "mujoco_vbd_package05_center",
    "v10_black_hoodie": "mujoco_vbd_package06_center",
    "v10_box01": "mujoco_mujoco_package07_center",
    "v10_box02": "mujoco_mujoco_package08_center",
    "shake_none": "mujoco_vbd_package01_pos_x35mm_shake_noattach",
    "shake_spring": "mujoco_vbd_package01_center_shake_spring",
    "shake_spring_x35": "mujoco_vbd_package01_pos_x35mm_shake_spring",
    "mujoco_hydroelastic_screw": "mujoco_hydroelastic_screw01_rotate",
    "mujoco_hydroelastic_screw_debug": "mujoco_hydroelastic_screw01_rotate_debug",
}

PHYSIC_SOLVER_CASE_GROUPS = {
    "solver": (
        "tgs_tgs_pickplace_rigid",
        "featherstone_vbd_pickplace_cloth",
        "tgs_vbd_pickplace_cloth",
        "mujoco_vbd_pickplace_cloth",
        "xpbd_vbd_pickplace_cloth",
        "mujoco_vbd_pickplace_package",
    ),
    "positions": (
        "v7_center",
        "v8_offset_x35",
        "v8_offset_y55",
        "v8_yaw45",
        "v8_yaw90",
        "v8_offset_tilt10",
    ),
    "weights": ("v9_mass100g", "v9_mass200g", "v9_mass350g", "v9_mass500g"),
    "types": (
        "v10_station_clothing",
        "v10_station_green",
        "v10_pink_tshirt",
        "v10_frosted_top",
        "v10_black_hoodie",
    ),
    "shake_cartons": (
        "shake_none",
        "shake_spring",
        "shake_spring_x35",
        "v10_box01",
        "v10_box02",
    ),
    "controls": ("featherstone_vbd_pickplace_package", "xpbd_xpbd_pickplace_cloth"),
    "assembly": ("mujoco_hydroelastic_screw",),
}

PHYSIC_SOLVER_PACKAGE_CASES = tuple(
    PHYSIC_SOLVER_CASE_NAMES[key]
    for group in ("positions", "weights", "types", "shake_cartons")
    for key in PHYSIC_SOLVER_CASE_GROUPS[group]
    if key not in {"v10_box01", "v10_box02"}
)


def physic_solver_case_name(key: str) -> str:
    """Resolve an experiment identifier to its canonical flat config/output stem."""
    return PHYSIC_SOLVER_CASE_NAMES.get(key, key)


def proxy_pose_errors(source: np.ndarray, proxy: np.ndarray) -> tuple[float, float]:
    """Measure maximum proxy position/rotation error, accepting antipodal quaternions."""
    source, proxy = np.asarray(source, dtype=float), np.asarray(proxy, dtype=float)
    if source.ndim < 1 or source.shape != proxy.shape or source.shape[-1] != 7 or source.size == 0:
        raise ValueError("Source and proxy poses must have matching nonempty (..., 7) shapes")
    if not np.isfinite(source).all() or not np.isfinite(proxy).all():
        raise ValueError("Proxy poses must be finite")
    qs, qp = source[..., 3:], proxy[..., 3:]
    ns, np_ = np.linalg.norm(qs, axis=-1), np.linalg.norm(qp, axis=-1)
    if np.any(ns <= 0) or np.any(np_ <= 0):
        raise ValueError("Proxy quaternion norms must be positive")
    qs, qp = qs / ns[..., None], qp / np_[..., None]
    dot = np.sum(qs * qp, axis=-1)
    chord = np.linalg.norm(qs - np.where(dot[..., None] < 0, -qp, qp), axis=-1)
    angle = 4 * np.arcsin(np.clip(chord * 0.5, 0, 1))
    return float(np.linalg.norm(source[..., :3] - proxy[..., :3], axis=-1).max()), float(
        angle.max()
    )


@cache
def proxy_pose_maximum_kernel() -> Any:
    """Return the all-substep body proxy geometry audit without importing Warp at module load."""
    import warp as wp

    @wp.kernel
    def maximum(
        source: wp.array[wp.transform], proxy: wp.array[wp.transform], errors: wp.array[float]
    ):
        i = wp.tid()
        position = wp.length(
            wp.transform_get_translation(source[i]) - wp.transform_get_translation(proxy[i])
        )
        a, b = wp.transform_get_rotation(source[i]), wp.transform_get_rotation(proxy[i])
        qa = wp.normalize(wp.vec4(a[0], a[1], a[2], a[3]))
        qb = wp.normalize(wp.vec4(b[0], b[1], b[2], b[3]))
        if wp.dot(qa, qb) < 0.0:
            qb = -qb
        angle = 4.0 * wp.asin(wp.min(1.0, 0.5 * wp.length(qa - qb)))
        wp.atomic_max(errors, 0, position)
        wp.atomic_max(errors, 1, angle)

    return maximum


class VBDSubstepGraph:
    """Capture only the GPU VBD step when a coupled CPU solver must run between substeps."""

    def __init__(self, solver: Any) -> None:
        import warp as wp

        self.wp, self.model, self.original = wp, solver.model, solver.step
        self.graphs: dict[tuple, Any] = {}
        self.warmed: set[tuple] = set()

    def __call__(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float):
        key = (id(state_in), id(state_out), id(control), id(contacts), dt)
        if key not in self.warmed:
            self.original(state_in, state_out, control, contacts, dt)
            self.warmed.add(key)
            return
        if key not in self.graphs:
            with self.wp.ScopedCapture(device=self.model.device) as capture:
                self.original(state_in, state_out, control, contacts, dt)
            self.graphs[key] = capture.graph
        self.wp.capture_launch(self.graphs[key])


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
            assistance = getattr(self, "jaw_assistance", None)
            if assistance is not None:
                assistance.apply(state_in, state_out, dt)
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


# ================================================================================================
# Explicit, finite-compliance jaw assistance and smooth shake trajectories.
# ================================================================================================


def shake_displacement(
    elapsed_s: float, duration_s: float, frequency_hz: float, amplitudes: np.ndarray
) -> np.ndarray:
    """Return a sinusoid with zero displacement and velocity at both endpoints."""
    if duration_s <= 0 or elapsed_s <= 0 or elapsed_s >= duration_s:
        return np.zeros_like(amplitudes, dtype=float)
    envelope = np.sin(np.pi * elapsed_s / duration_s) ** 2
    return np.asarray(amplitudes) * envelope * np.sin(2 * np.pi * frequency_hz * elapsed_s)


def package_gripper_activation(elapsed_s: float, ramp_s: float, *, closed: bool) -> float:
    """Smooth an optional jaw command without changing the final physical finger opening."""
    if ramp_s <= 0:
        return 0.1 if closed else 0.8
    u = float(np.clip(elapsed_s / ramp_s, 0, 1))
    blend = u * u * (3 - 2 * u)
    return 0.8 - 0.7 * blend if closed else 0.1 + 0.7 * blend


def package_finger_reference(
    activation: float, open_m: float, closed_m: float, preload_n: float, stiffness_n_m: float
) -> float:
    """Express a bounded actuator's closing preload as a PD equilibrium offset, not a pose."""
    if preload_n < 0 or stiffness_n_m <= 0:
        raise ValueError("Closing preload must be nonnegative and finger stiffness positive")
    blend = float(np.clip((activation - 0.1) / 0.7, 0.0, 1.0))
    return closed_m + blend * (open_m - closed_m) - (1 - blend) * preload_n / stiffness_n_m


@cache
def _jaw_spring_kernel() -> Any:
    import warp as wp

    @wp.kernel
    def apply(
        points: wp.array[wp.vec3],
        velocities: wp.array[wp.vec3],
        old_bodies: wp.array[wp.transform],
        new_bodies: wp.array[wp.transform],
        ids: wp.array[int],
        bodies: wp.array[int],
        local: wp.array[wp.vec3],
        enabled: wp.array[float],
        stiffness: float,
        damping: float,
        maximum_force: float,
        dt: float,
        forces: wp.array[wp.vec3],
    ):
        slot = wp.tid()
        particle = ids[slot]
        if enabled[0] > 0.0 and particle >= 0:
            body = bodies[slot]
            old = wp.transform_point(old_bodies[body], local[slot])
            target = wp.transform_point(new_bodies[body], local[slot])
            force = stiffness * (target - points[particle])
            force += damping * ((target - old) / dt - velocities[particle])
            magnitude = wp.length(force)
            if magnitude > maximum_force:
                force *= maximum_force / magnitude
            forces[particle] += force

    return apply


class JawSpringAssistance:
    """Releasable local particle/body springs; never change positions, masses or flags."""

    def __init__(self, model: Any, config: Any) -> None:
        import warp as wp

        self.model, self.config, self.wp = model, config, wp
        capacity = 2 * config.attachment_vertices_per_group
        self.ids = wp.array(np.full(capacity, -1, np.int32), dtype=int, device=model.device)
        self.bodies = wp.zeros(capacity, dtype=int, device=model.device)
        self.local = wp.zeros(capacity, dtype=wp.vec3, device=model.device)
        self.enabled = wp.zeros(1, dtype=float, device=model.device)
        self.records: list[dict[str, Any]] = []
        self.active = False
        self.was_bound = False

    def bind(self, runtime: Any) -> None:
        """Select distinct local points only near the measured native finger pad boxes."""
        points = runtime.vertices()
        poses = runtime.state_0.body_q.numpy()
        shape_bodies = self.model.shape_body.numpy()
        shape_types = self.model.shape_type.numpy()
        transforms = self.model.shape_transform.numpy()
        extents = self.model.shape_scale.numpy()
        shape_flags = self.model.shape_flags.numpy()
        used: set[int] = set()
        selected, owners, local_points = [], [], []
        split = runtime.scene.garment_count
        for side in ("left", "right"):
            body = runtime.contact_body_ids[side]
            pose = poses[body]
            # quat xyzw to matrix; importing the planner here would introduce a cycle.
            x, y, z, w = pose[3:]
            rotation = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            local = (points - pose[:3]) @ rotation
            distance = np.full(len(points), np.inf)
            boxes = [
                int(shape)
                for shape in np.flatnonzero(shape_bodies == body)
                if int(shape_types[shape]) == int(runtime.newton.GeoType.BOX)
                and shape_flags[shape] & int(runtime.newton.ShapeFlags.COLLIDE_PARTICLES)
                and np.allclose(transforms[shape, 3:6], 0, atol=1e-5)
            ]
            if not boxes:
                raise ValueError(f"No active local finger-pad box for {side}")
            # The distal axis-aligned box is the native rubber tip or the declared V7 pad.
            for shape in [max(boxes, key=lambda i: transforms[i, 2])]:
                delta = np.abs(local - transforms[shape, :3]) - extents[shape]
                signed = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(1), 0)
                distance = np.minimum(distance, np.abs(signed))
            # Adhesion acts on the exterior film; contents remain supported by bag contacts.
            for material, start, end in (("bag", split, len(points)),):
                candidates = np.arange(start, end)
                candidates = candidates[
                    distance[candidates] <= self.config.attachment_capture_distance_m
                ]
                candidates = [
                    int(i)
                    for i in candidates[np.argsort(distance[candidates])]
                    if int(i) not in used
                ]
                chosen = candidates[: self.config.attachment_vertices_per_group]
                if len(chosen) < 3:
                    raise RuntimeError(
                        f"Insufficient local {material} points at {side} pad: {len(chosen)}"
                    )
                used.update(chosen)
                selected.extend(chosen)
                owners.extend([body] * len(chosen))
                local_points.extend(local[chosen])
                self.records.append(
                    dict(
                        side=side,
                        body_id=body,
                        capture_time_s=float(runtime.sim_time),
                        material=material,
                        ids=chosen,
                        local_points_m=local[chosen].tolist(),
                        distances_m=distance[chosen].tolist(),
                    )
                )
        for destination, values in (
            (self.ids, selected),
            (self.bodies, owners),
            (self.local, local_points),
        ):
            self.wp.copy(
                destination,
                self.wp.array(values, dtype=destination.dtype, device=self.model.device),
                count=len(selected),
            )
        self.enabled.fill_(1.0)
        self.active = self.was_bound = True
        self.capture_points = points.copy()
        self.capture_bodies = poses.copy()

    def release(self) -> None:
        """Disable spring forces; all dynamic particle state remains untouched."""
        self.enabled.zero_()
        self.active = False

    def apply(self, state_in: Any, state_out: Any, dt: float) -> None:
        """Apply finite spring forces using the actual solved finger motion each substep."""
        self.wp.launch(
            _jaw_spring_kernel(),
            len(self.ids),
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.body_q,
                state_out.body_q,
                self.ids,
                self.bodies,
                self.local,
                self.enabled,
                self.config.attachment_stiffness_n_m,
                self.config.attachment_damping_n_s_m,
                self.config.attachment_max_force_n,
                dt,
                state_in.particle_f,
            ],
            device=self.model.device,
        )


def package_grasp_quaternion(yaw_deg: float = 0, pitch_deg: float = 0) -> np.ndarray:
    """Compose world yaw/pitch with the existing downward parcel grasp, returning xyzw."""
    yaw, pitch = np.deg2rad([yaw_deg, pitch_deg]) / 2

    def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.r_[
            a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - np.dot(a[:3], b[:3])
        ]

    return multiply(
        multiply(
            np.array([0, 0, np.sin(yaw), np.cos(yaw)]),
            np.array([0, np.sin(pitch), 0, np.cos(pitch)]),
        ),
        np.array([2**-0.5, 2**-0.5, 0, 0]),
    )


def read_obj_triangles(path: str | Any) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and triangulate OBJ faces without an optional mesh dependency."""
    vertices, triangles = [], []
    with open(path) as stream:
        for line in stream:
            words = line.split()
            if not words:
                continue
            if words[0] == "v":
                vertices.append([float(value) for value in words[1:4]])
            elif words[0] == "f":
                ids = [int(value.split("/")[0]) for value in words[1:]]
                ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
                triangles.extend([[ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1)])
    points, faces = np.asarray(vertices, np.float32), np.asarray(triangles, np.int32)
    if points.ndim != 2 or faces.ndim != 2 or not np.isfinite(points).all():
        raise ValueError(f"Invalid triangle OBJ: {path}")
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError(f"OBJ face index outside vertex array: {path}")
    return points, faces


@cache
def contact_maximum_kernel() -> Any:
    """Return a capture-safe counter audit for every rigid contact substep."""
    import warp as wp

    @wp.kernel
    def audit(count: wp.array[int], maximum: wp.array[int]):
        wp.atomic_max(maximum, 0, count[0])

    return audit


def isaac_newton_sdk_environment() -> dict[str, str]:
    """Find Newton/Warp/USD shipped with the running Isaac Python, without starting Kit."""
    import os
    import sys
    from pathlib import Path

    isaac = Path(sys.executable).resolve().parents[3]
    usd = next((isaac / "extscache").glob("omni.usd.libs-*/pxr/Gf")).parent.parent.resolve()
    warp = next((isaac / "extscache").glob("omni.warp.core-1.17.0*"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(warp),
            str(isaac / "exts/isaacsim.pip.newton/pip_prebundle"),
            str(usd),
            environment.get("PYTHONPATH", ""),
        ]
    )
    environment["LD_LIBRARY_PATH"] = (
        str(usd / "bin") + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    )
    return environment


# ================================================================================================
# UR5e screwdriver: source USD joint frames -> reusable SI serial-chain kinematics.
# ================================================================================================


def screw_arm_chain(stage: Any, root: str, tool_length_m: float) -> tuple[Any, np.ndarray]:
    """Read the six UR5e joints and flange TCP without hard-coding DH dimensions."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from real2sim2real.utils.battery_disassembly_utils import (
        SerialChain,
        quat_to_matrix,
        transform,
    )

    def local_frame(joint: Any, side: int) -> np.ndarray:
        q = getattr(joint, f"GetLocalRot{side}Attr")().Get()
        p = getattr(joint, f"GetLocalPos{side}Attr")().Get()
        return transform(quat_to_matrix([q.GetReal(), *q.GetImaginary()]), p)

    names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    origins, axes, lower, upper = [], [], [], []
    pending = np.eye(4)
    for name in names:
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{root}/joints/{name}"))
        origins.append(pending @ local_frame(joint, 0))
        pending = np.linalg.inv(local_frame(joint, 1))
        axes.append(np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())])
        lower.append(np.deg2rad(joint.GetLowerLimitAttr().Get()))
        upper.append(np.deg2rad(joint.GetUpperLimitAttr().Get()))
    wrist = np.asarray(
        UsdGeom.Xformable(stage.GetPrimAtPath(f"{root}/wrist_3_link")).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
    ).T
    flange = np.asarray(
        UsdGeom.Xformable(
            stage.GetPrimAtPath(f"{root}/wrist_3_link/flange")
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ).T
    mount = (
        np.linalg.inv(wrist)
        @ flange
        @ transform(np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, tool_length_m])
    )
    return SerialChain(
        origins, axes, np.array(lower), np.array(upper), pending @ mount, names
    ), mount


# ================================================================================================
# Dual-arm screw phase scheduling and acceptance (no Kit imports).
# ================================================================================================


def screw_socket_parts(inner_radius_m: float) -> list[dict]:
    """Describe split-collet walls, shoulders and mount for collision and visible geometry."""
    parts = []
    for i in range(6):
        yaw = i * np.pi / 3
        radius = inner_radius_m + 0.003
        parts.append(
            dict(
                name=f"wall_{i}",
                half=[0.003, 0.007, 0.009],
                center=[radius * np.cos(yaw), radius * np.sin(yaw), -0.010],
                yaw=float(yaw),
                side="p" if np.cos(yaw) > 0 else "n",
            )
        )
        # A split retaining shoulder supports the cylindrical head without touching the shaft.
        parts.append(
            dict(
                name=f"shoulder_{i}",
                half=[0.004, 0.004, 0.0015],
                center=[0.0145 * np.cos(yaw), 0.0145 * np.sin(yaw), -0.022],
                yaw=float(yaw),
                side="p" if np.cos(yaw) > 0 else "n",
            )
        )
    parts.append(
        dict(
            name="backplate",
            half=[0.022, 0.022, 0.002],
            center=[0, 0, 0.002],
            yaw=0.0,
            side="fixed",
        )
    )
    parts.append(
        dict(name="rail", half=[0.05, 0.006, 0.003], center=[0, 0, 0.007], yaw=0.0, side="fixed")
    )
    for side, sign in (("p", 1), ("n", -1)):
        parts.append(
            dict(
                name=f"tab_{side}",
                half=[0.003, 0.004, 0.008],
                center=[sign * 0.023, 0, 0.004],
                yaw=0.0,
                side=side,
            )
        )
    return parts


def screw_socket_clearance(
    tool: np.ndarray, bolt: np.ndarray, inner_radius_m: float, openings_m: Any = (0.0, 0.0)
) -> float:
    """Measure the lowest socket corner above the bolt head plane in the bolt's frame."""
    from real2sim2real.utils.battery_disassembly_utils import quat_to_matrix

    tr, br = [quat_to_matrix(p[[6, 3, 4, 5]]) for p in (tool, bolt)]
    corners = []
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    for part in screw_socket_parts(inner_radius_m):
        c, s = np.cos(part["yaw"]), np.sin(part["yaw"])
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        center = np.array(part["center"], dtype=float)
        if part["side"] == "p":
            center[0] += openings_m[0]
        elif part["side"] == "n":
            center[0] -= openings_m[1]
        corners.extend(signs * part["half"] @ r.T + center)
    relative = (np.array(corners) @ tr.T + tool[:3] - bolt[:3]) @ br
    return float(relative[:, 2].min())


def dual_screw_phases(config: Any) -> list[tuple[str, float]]:
    """List active phases, preserving the original sequence when release is disabled."""
    phases = []
    for name in ("clamp", "lift", "align", "engage", "screw", "hold"):
        duration = float(
            config[f"{name}_s"] if isinstance(config, dict) else getattr(config, f"{name}_s")
        )
        phases.append((name, duration))
    opening = float(
        config.get("socket_open_s", 0)
        if isinstance(config, dict)
        else getattr(config, "socket_open_s", 0)
    )
    if opening > 0:
        phases.append(("open_bolt", opening))
    for name, key in (("release", "release_s"), ("released", "post_release_s")):
        duration = float(
            config.get(key, 0) if isinstance(config, dict) else getattr(config, key, 0)
        )
        if duration > 0:
            phases.append((name, duration))
    return phases


def dual_screw_phase(time_s: float, config: Any) -> tuple[str, float]:
    """Return ordered progress, clamped at the last active phase."""
    remaining = time_s
    phases = dual_screw_phases(config)
    for index, (name, duration) in enumerate(phases):
        if remaining < duration or index == len(phases) - 1:
            return name, float(np.clip(remaining / duration, 0.0, 1.0))
        remaining -= duration
    raise RuntimeError("Unreachable dual screw phase")


def screw_cylindrical_uv(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return face-varying UVs fixed to the bolt, unwrapping triangles across the seam."""
    corners = np.asarray(vertices)[triangles]
    u = np.mod(np.arctan2(corners[..., 1], corners[..., 0]) / (2 * np.pi), 1.0)
    seam = np.ptp(u, axis=1) > 0.5
    u = np.where(seam[:, None] & (u < 0.5), u + 1.0, u)
    z = np.asarray(vertices)[:, 2]
    v = (corners[..., 2] - z.min()) / (z.max() - z.min())
    return np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)


def evaluate_dual_screw_rows(rows: list[dict], config: dict, pitch_m: float) -> dict:
    """Measure threading relative to the free nut and actual contact-only pickup stability."""
    screw = [i for i, row in enumerate(rows) if row["phase"] == "screw"]
    hold = [i for i, row in enumerate(rows) if row["phase"] == "hold"]
    bolt_release = config.get("release_target", "nut") == "bolt"
    excluded = {"clamp"} if bolt_release else {"clamp", "release", "released"}
    carried = [i for i, row in enumerate(rows) if row["phase"] not in excluded]
    if not rows or not screw or not hold or not carried:
        return {"success": False, "checks": {"sequence_completed": False}}
    q = np.array([[r["axial_displacement_m"], r["spindle_angle_rad"]] for r in rows])
    threaded = np.array(screw + hold)
    start, end = q[screw[0]], q[hold[0]]
    turns = abs(end[1] - start[1]) / (2 * np.pi)
    advance = start[0] - end[0]
    expected = turns * pitch_m
    error = abs(advance - expected) / max(expected, 1e-12)
    drift = float(np.ptp(q[hold, 0]))
    residual = q[threaded, 0] - start[0] - (q[threaded, 1] - start[1]) * pitch_m / (2 * np.pi)
    unthreaded = float(np.max(np.abs(residual)))
    centers = np.array([r["nut_center_local_m"] for r in rows])
    lift = float(np.min(centers[threaded, 2]) - centers[0, 2])
    relative = np.array([rows[i]["nut_relative_holder_m"] for i in carried])
    slip = float(np.max(np.linalg.norm(relative - relative[0], axis=1)))
    yaw = np.unwrap(np.array([r["nut_yaw_rad"] for r in rows]))
    nut_rotation = float(np.max(np.abs(yaw[carried] - yaw[carried[0]])))
    bilateral = float(
        np.mean([rows[i]["jaw_p_contacts"] > 0 and rows[i]["jaw_n_contacts"] > 0 for i in carried])
    )
    thread_fraction = float(np.mean([rows[i]["contacts"] > 0 for i in screw]))
    arms = np.array([r["arm_q_rad"] for r in rows])
    targets = np.array([r["arm_target_rad"] for r in rows])
    arm_error = float(np.max(np.abs(arms - targets)))
    duration = sum(duration for _, duration in dual_screw_phases(config))
    checks = dict(
        sequence_completed=rows[-1]["time_s"] >= duration - 1e-8,
        finite_state=bool(np.isfinite(q).all() and np.isfinite(arms).all()),
        spindle_rotated=turns >= config["minimum_turns"],
        bolt_advanced=advance >= config["minimum_advance_m"],
        thread_lead_matches=error <= config["maximum_pitch_error_fraction"],
        helical_contact_observed=thread_fraction >= 0.95,
        retained_during_hold=drift <= config["maximum_hold_drift_m"],
        no_unthreaded_push_through=unthreaded <= config["maximum_unthreaded_advance_m"],
        nut_actually_lifted=lift >= config["minimum_nut_lift_m"],
        bilateral_finger_contact=bilateral >= config["minimum_bilateral_grip_fraction"],
        nut_held_without_slip=slip <= config["maximum_nut_grasp_slip_m"],
        nut_held_against_rotation=nut_rotation <= config["maximum_nut_yaw_rad"],
        nut_clear_of_support_while_threading=all(
            rows[i]["support_contacts"] == 0 for i in threaded
        ),
        both_arms_track=arm_error <= config["maximum_arm_tracking_error_rad"],
    )
    checks = {name: bool(value) for name, value in checks.items()}
    protrusion = {}
    if config.get("minimum_tip_protrusion_m", 0) > 0:
        minimum_tip = float(np.min([rows[i]["tip_protrusion_m"] for i in hold]))
        checks["bolt_protrudes_through_nut"] = bool(
            np.isfinite(minimum_tip) and minimum_tip >= config["minimum_tip_protrusion_m"]
        )
        protrusion["minimum_hold_tip_protrusion_m"] = minimum_tip
    if config.get("release_s", 0) > 0:
        released = [r for r in rows if r["phase"] == "released"]
        checks["released_observation_recorded"] = bool(released)
        if released:
            fraction = float(np.mean([r["contacts"] > 0 for r in released]))
            tip = float(np.min([r["tip_protrusion_m"] for r in released]))
            if bolt_release:
                clearance = float(np.min([r["socket_clearance_m"] for r in released]))
                settling = float(
                    np.max(np.abs([r["axial_displacement_m"] - end[0] for r in released]))
                )
                checks.update(
                    socket_contact_during_drive=np.mean(
                        [rows[i]["socket_contacts"] > 0 for i in screw]
                    )
                    >= 0.95,
                    socket_actually_detached=all(r["socket_contacts"] == 0 for r in released),
                    driver_retracted_clear=clearance >= config["minimum_tool_clearance_m"],
                    bolt_stays_in_held_nut=fraction >= 0.95 and tip > 0,
                    bolt_settling_after_release=settling <= config["maximum_bolt_settling_m"],
                    bolt_clamp_actually_open=np.min([r["socket_jaw_q_m"] for r in released])
                    >= config["minimum_socket_opening_m"],
                )
                protrusion.update(
                    minimum_socket_clearance_m=clearance, maximum_bolt_settling_m=settling
                )
            else:
                opening = float(np.min([r["joint_q"][14:16] for r in released]))
                checks.update(
                    gripper_actually_reopened=opening >= config["minimum_released_finger_m"],
                    nut_clear_of_both_fingers=all(
                        r["jaw_p_contacts"] == r["jaw_n_contacts"] == 0 for r in released
                    ),
                    nut_retained_on_bolt_after_release=fraction >= 0.95 and tip > 0,
                )
                protrusion["minimum_released_finger_m"] = opening
            if config.get("minimum_tip_protrusion_m", 0) > 0:
                checks["bolt_still_protrudes_after_release"] = (
                    tip >= config["minimum_tip_protrusion_m"]
                )
            protrusion.update(
                released_thread_contact_fraction=fraction,
                minimum_released_tip_protrusion_m=tip,
            )
    checks = {name: bool(value) for name, value in checks.items()}
    return dict(
        success=all(checks.values()),
        checks=checks,
        turns=float(turns),
        axial_advance_m=float(advance),
        expected_thread_advance_m=float(expected),
        pitch_error_fraction=float(error),
        hold_drift_m=drift,
        maximum_unthreaded_advance_m=unthreaded,
        minimum_nut_lift_m=lift,
        maximum_nut_grasp_slip_m=slip,
        maximum_nut_yaw_rad=nut_rotation,
        bilateral_grip_fraction=bilateral,
        thread_contact_fraction=thread_fraction,
        maximum_arm_tracking_error_rad=arm_error,
        **protrusion,
    )
