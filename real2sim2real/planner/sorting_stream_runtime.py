"""One shared MuJoCo/VBD world for a soft parcel followed by a dynamic V10 carton."""

from __future__ import annotations

from typing import Any

import numpy as np

from real2sim2real.planner.planner_physic_solver import _rotation_wxyz
from real2sim2real.planner.sorting_runtime import SortingRuntime


class SortingStreamRuntime(SortingRuntime):
    """Keep both parcels in the same model and insert the carton only once, at spawn."""

    def __init__(self, robot: Any, scene: Any, config: Any) -> None:
        self.carton_spawned = False
        self.soft_asleep = False
        self.belt_offsets = np.zeros(2)
        super().__init__(robot, scene, config)
        solver = self.rigid_solver
        native_dt = (
            robot.sim_world.config.physics_dt
            / config.newton_substeps
            / config.mujoco_substeps_per_vbd_step
        )
        if config.rigid_contact_time_constant_s < 2 * native_dt:
            raise ValueError("Rigid contact time constant must cover at least two native substeps")
        bodies = solver.mjc_body_to_newton.numpy()[0]
        shapes = solver.mjc_geom_to_newton_shape.numpy()[0]
        self.carton_mj_body = int(np.flatnonzero(bodies == self.carton_body)[0])
        self.carton_geom = int(np.flatnonzero(shapes == self.carton_shape)[0])
        self._carton_masks = (
            int(solver.mj_model.geom_contype[self.carton_geom]),
            int(solver.mj_model.geom_conaffinity[self.carton_geom]),
        )
        solver.mj_model.geom_contype[self.carton_geom] = 0
        solver.mj_model.geom_conaffinity[self.carton_geom] = 0
        solver.mj_model.body_gravcomp[self.carton_mj_body] = 1.0
        # Rigid contact uses its own critically damped native material. VBD's
        # soft-contact damping has a different meaning and is left unchanged.
        solver.mj_model.geom_solmix[:] = 0.0
        solver.mj_model.geom_solmix[self.carton_geom] = 1.0
        solver.mj_model.geom_solref[self.carton_geom] = [config.rigid_contact_time_constant_s, 1.0]
        solver.mj_model.geom_solimp[self.carton_geom, :2] = [0.95, 0.99]
        self._mj_shapes = shapes
        self._mj_bodies = bodies
        self.belt_geoms = {
            int(np.flatnonzero(shapes == self.infeed_shape)[0]): "infeed",
            int(np.flatnonzero(shapes == self.belt_shape)[0]): "outfeed",
        }
        # MuJoCo's fixed mocap bodies have no tangential surface velocity. Use
        # contact-normal-limited Coulomb belt forces in their tangent plane;
        # disable only their native sliding friction to avoid counting it twice.
        self._native_belt_mu = {}
        for geom in self.belt_geoms:
            self._native_belt_mu[geom] = self.config.table_friction
            solver.mj_model.geom_friction[geom, 0] = 0.0
        # MuJoCo combines native sliding friction with max(), whereas the VBD
        # path uses Newton's geometric mean. Let the counter-surface specify
        # the rigid contact coefficient explicitly.
        solver.mj_model.geom_friction[self.carton_geom, 0] = 0.0
        platform_geom = int(np.flatnonzero(shapes == self.platform_shape)[0])
        solver.mj_model.geom_friction[platform_geom, 0] = (
            self.config.table_friction
            if self.config.platform_friction is None
            else self.config.platform_friction
        )
        finger_bodies = {
            self.contact_body_ids[f"{side}_{i}"] for side in ("left", "right") for i in (0, 1)
        }
        shape_bodies = self.model.shape_body.numpy()
        for geom, shape in enumerate(shapes):
            if int(shape_bodies[shape]) in finger_bodies:
                solver.mj_model.geom_friction[geom, 0] = (
                    self.config.gripper_package_friction or self.config.shape_contact_friction
                )
        self.carton_contacts: dict[str, int] = {}
        self.carton_contact_shapes = np.empty(0, int)
        self.carton_contact_distances: dict[str, float | None] = {}
        self.maximum_carton_contact_penetration_m = 0.0
        self.maximum_belt_force_n = 0.0

    def _add_deformables(self, builder: Any, *, xpbd: bool) -> None:
        super()._add_deformables(builder, xpbd=xpbd)
        half = self.scene.rigid_dimensions / 2
        self.carton_body = builder.add_body(
            xform=self.wp.transform(
                (100.0, 0.0, half[2] + self.scene.config.floor_z_m), self.wp.quat_identity()
            ),
            label="stream_carton",
        )
        cfg = self.newton.ModelBuilder.ShapeConfig(
            density=float(self.scene.config.rigid_mass_kg / np.prod(2 * half)),
            mu=self.config.shape_contact_friction,
            margin=0.0002,
        )
        self.carton_shape = builder.add_shape_box(
            self.carton_body,
            hx=float(half[0]),
            hy=float(half[1]),
            hz=float(half[2]),
            cfg=cfg,
            label="stream_carton_collision",
        )

    def spawn_carton(self) -> None:
        """Initialize a new parcel at the remote infeed end; never reset it after insertion."""
        if self.carton_spawned:
            raise RuntimeError("A carton can only be spawned once per continuous run")
        solver = self.rigid_solver
        solver.mj_model.opt.solver = (
            solver._mujoco.mjtSolver.mjSOL_PGS
            if self.config.rigid_mujoco_solver == "pgs"
            else solver._mujoco.mjtSolver.mjSOL_NEWTON
        )
        if self.config.rigid_disable_warmstart:
            solver.mj_model.opt.disableflags |= int(solver._mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        solver.mj_model.opt.iterations = self.config.rigid_mujoco_iterations
        solver.mj_model.body_gravcomp[self.carton_mj_body] = 0.0
        solver.mj_model.geom_contype[self.carton_geom] = self._carton_masks[0]
        solver.mj_model.geom_conaffinity[self.carton_geom] = self._carton_masks[1]
        joint = int(solver.mj_model.body_jntadr[self.carton_mj_body])
        q = int(solver.mj_model.jnt_qposadr[joint])
        v = int(solver.mj_model.jnt_dofadr[joint])
        position = np.array(
            [
                self.scene.config.target_xy[0],
                self.scene.config.target_xy[1] - self.scene.config.infeed_travel_m,
                self.scene.supports["infeed"][1][2]
                + self.scene.rigid_dimensions[2] / 2
                + self.scene.config.initial_clearance_m,
            ]
        )
        solver.mj_data.qpos[q : q + 3] = position - self.physics_origin
        solver.mj_data.qpos[q + 3 : q + 7] = [1, 0, 0, 0]
        solver.mj_data.qvel[v : v + 6] = 0.0
        solver.mj_data.qacc_warmstart[:] = 0.0
        solver._mujoco.mj_forward(solver.mj_model, solver.mj_data)
        for state in (self.state_0, self.state_1):
            solver._update_newton_state(self.model, state, solver.mj_data, state_prev=state)
        self.carton_spawned = True
        self.infeed_speed = self.config.infeed_speed_mps
        self._publish()

    def carton_vertices(self, poses: np.ndarray | None = None) -> np.ndarray:
        """Return original mesh vertices transformed by the solved free rigid body."""
        pose = (self.body_poses() if poses is None else poses)[self.carton_body]
        return self.scene.rigid_vertices @ _rotation_wxyz(pose[[6, 3, 4, 5]]).T + pose[:3]

    def _before_native_substep(self, dt: float) -> None:
        if not self.carton_spawned:
            return
        solver, body = self.rigid_solver, self.carton_mj_body
        model, data = solver.mj_model, solver.mj_data
        data.xfrc_applied[body] = 0.0
        contacts = []
        total_normal = 0.0
        for index in range(data.ncon):
            contact = data.contact[index]
            g0, g1 = map(int, contact.geom)
            if self.carton_geom not in (g0, g1):
                continue
            other = g1 if g0 == self.carton_geom else g0
            self.maximum_carton_contact_penetration_m = max(
                getattr(self, "maximum_carton_contact_penetration_m", 0.0),
                max(0.0, -float(contact.dist)),
            )
            if other not in self.belt_geoms:
                continue
            force = np.zeros(6)
            solver._mujoco.mj_contactForce(model, data, index, force)
            normal = max(0.0, float(force[0]))
            if normal > 0:
                contacts.append((other, normal, contact.pos.copy()))
                total_normal += normal
        if not contacts:
            return
        velocity = np.zeros(6)
        solver._mujoco.mj_objectVelocity(
            model, data, solver._mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
        )
        for geom, normal, point in contacts:
            speed = (
                np.array([0.0, self.infeed_speed, 0.0])
                if self.belt_geoms[geom] == "infeed"
                else np.array([-self.belt_speed, 0.0, 0.0])
            )
            arm = point - data.xipos[body]
            slip = speed - (velocity[3:] + np.cross(velocity[:3], arm))
            slip[2] = 0.0
            # Apply one critically limited momentum correction across all belt
            # contacts, bounded by each contact's measured Coulomb capacity.
            force = model.body_mass[body] * slip / max(dt, 0.01) * normal / total_normal
            limit = self._native_belt_mu[geom] * normal
            force *= min(1.0, limit / max(np.linalg.norm(force), 1e-12))
            data.xfrc_applied[body, :3] += force
            data.xfrc_applied[body, 3:] += np.cross(arm, force)
        self.maximum_belt_force_n = max(
            self.maximum_belt_force_n, float(np.linalg.norm(data.xfrc_applied[body, :3]))
        )

    def step(self, frame_dt: float) -> None:
        """Advance both solvers, allowing a settled floor parcel to sleep until approached."""
        if self.carton_spawned and self.config.sleep_soft_when_settled:
            vertices = self.vertices()
            carton = self.carton_vertices()
            separation = np.maximum(
                np.maximum(vertices.min(0) - carton.max(0), carton.min(0) - vertices.max(0)),
                0.0,
            )
            floor = self.scene.config.floor_z_m
            settled = (
                vertices[:, 2].max() < floor + 0.10
                and vertices[:, 2].min() >= floor - 0.0015
                and np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max() < 0.02
            )
            asleep = settled and np.linalg.norm(separation) > 0.20
            if asleep and not self.soft_asleep:
                self.state_0.particle_qd.zero_()
                self.state_1.particle_qd.zero_()
            self.soft_asleep = bool(asleep)
        self.belt_offsets += np.array([self.infeed_speed, self.belt_speed]) * frame_dt
        super().step(frame_dt)

    def _advance_soft(self, dt: float) -> None:
        if self.soft_asleep:
            self.wp.copy(self.state_1.particle_q, self.state_0.particle_q)
            self.wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
        else:
            super()._advance_soft(dt)

    def _publish(self) -> None:
        super()._publish()
        if not hasattr(self, "carton_mj_body"):
            return
        self.scene.publish_rigid_pose(
            self.body_poses()[self.carton_body], visible=self.carton_spawned
        )
        if self.scene.config.belt_markers:
            self.scene.publish_belt_markers(self.belt_offsets)
        self.carton_contacts = dict.fromkeys(self.contact_body_ids, 0)
        self.carton_contacts.update(infeed=0, outfeed=0, platform=0)
        self.carton_contact_distances = dict.fromkeys(self.contact_body_ids)
        shapes = []
        data = self.rigid_solver.mj_data
        for contact in data.contact[: data.ncon]:
            g0, g1 = map(int, contact.geom)
            if self.carton_geom not in (g0, g1) or contact.dist > 0.0005:
                continue
            other = g1 if g0 == self.carton_geom else g0
            shape = int(self._mj_shapes[other])
            shapes.append(shape)
            body = int(self.model.shape_body.numpy()[shape])
            for name, target in (
                ("infeed", self.infeed_shape),
                ("outfeed", self.belt_shape),
                ("platform", self.platform_shape),
            ):
                self.carton_contacts[name] += int(shape == target)
            for name, target in self.contact_body_ids.items():
                if body == target:
                    self.carton_contacts[name] += 1
                    old = self.carton_contact_distances[name]
                    self.carton_contact_distances[name] = min(
                        float(contact.dist), old if old is not None else np.inf
                    )
            self.maximum_carton_contact_penetration_m = max(
                self.maximum_carton_contact_penetration_m, max(0.0, -float(contact.dist))
            )
        self.carton_contact_shapes = np.asarray(shapes, int)

    def publish_recorded_visuals(
        self,
        vertices: np.ndarray,
        body_poses: np.ndarray,
        *,
        carton_visible: bool = False,
        belt_offsets: np.ndarray | None = None,
    ) -> None:
        """Replay the same dynamic bodies and VBD particles without stepping physics."""
        super().publish_recorded_visuals(vertices, body_poses)
        self.scene.publish_rigid_pose(body_poses[self.carton_body], visible=carton_visible)
        if self.scene.config.belt_markers and belt_offsets is not None:
            self.scene.publish_belt_markers(belt_offsets)
