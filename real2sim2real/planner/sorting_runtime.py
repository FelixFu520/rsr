"""MuJoCo Pro-M6 articulation coupled to V7's two-mesh VBD contact model."""

from __future__ import annotations

from functools import cache
from types import SimpleNamespace
from typing import Any

import numpy as np

from real2sim2real.planner.planner_package import PackageRuntime
from real2sim2real.planner.planner_physic_solver import _rotation_wxyz
from real2sim2real.planner.sorting_kinematics import SortingArmIK
from real2sim2real.utils.sorting_quality import unit_box_mesh


@cache
def _belt_kernel() -> Any:
    import warp as wp

    @wp.kernel
    def belt_motion(
        previous: wp.array[wp.transform],
        current: wp.array[wp.transform],
        body: int,
        rest: wp.transform,
        speed: float,
        dt: float,
    ):
        p = wp.transform_get_translation(rest)
        q = wp.transform_get_rotation(rest)
        previous[body] = wp.transform(p + wp.vec3(speed * dt, 0.0, 0.0), q)
        current[body] = rest

    return belt_motion


class SortingRuntime(PackageRuntime):
    """Preserve native robot geometry and solve contacts without cloth attachments."""

    def __init__(self, robot: Any, scene: Any, config: Any) -> None:
        self.active_side = "left"
        self.belt_speed = 0.0
        self.physics_origin = np.asarray(config.physics_origin_m, dtype=float)
        super().__init__(robot, scene, config)
        self.belt_rest = self.model.body_q.numpy()[self.belt_body].copy()
        self._configure_contact_friction()
        self._belt_motion = _belt_kernel()
        self._masses = self.model.particle_mass.numpy()
        self.direct_use_graph = self.direct_use_graph and not config.mujoco_cpu
        self._soft_graphs = {}
        self._collision_views = {}
        # Return above the source/exit belts through the already screened grasp
        # workspace. A joint-space return to READY can sweep an open pad into
        # the platform even when both end poses are collision-free.
        self.parking_positions = {
            "left": np.r_[scene.config.target_xy, scene.supports["infeed"][1][2] + 0.25],
            "right": np.r_[
                scene.config.outfeed_target_xy,
                scene.supports["outfeed"][1][2] + config.place_height_m + 0.18,
            ],
        }
        self.jaw_colliders = []
        self.pad_colliders = []
        self.fixed_robot_colliders = []
        finger_bodies = {
            self.contact_body_ids[f"{side}_{i}"] for side in ("left", "right") for i in (0, 1)
        }
        for shape, body in enumerate(self.model.shape_body.numpy()):
            source = self.model.shape_source[shape]
            fixed_robot = shape < self.robot_shape_count and int(body) < 0
            if int(body) not in finger_bodies and not fixed_robot:
                continue
            is_pad = int(self.model.shape_type.numpy()[shape]) == int(self.newton.GeoType.BOX)
            if is_pad:
                points, triangles = unit_box_mesh()
            elif source is not None:
                points = np.asarray(source.vertices)
                triangles = np.asarray(source.indices, dtype=int).reshape(-1, 3)
            else:
                raise ValueError("Unsupported native finger collider; cannot certify geometry")
            transform = self.model.shape_transform.numpy()[shape]
            rotation = _rotation_wxyz(transform[[6, 3, 4, 5]])
            vertices = (points * self.model.shape_scale.numpy()[shape]) @ rotation.T
            if fixed_robot:
                self.fixed_robot_colliders.append(
                    dict(
                        shape_id=shape,
                        vertices=vertices + transform[:3] + self.physics_origin,
                        triangles=triangles,
                    )
                )
                continue
            self.jaw_colliders.append(
                dict(
                    body_id=int(body),
                    vertices=vertices + transform[:3],
                    triangles=triangles,
                )
            )
            if is_pad:
                self.pad_colliders.append(self.jaw_colliders[-1])
        self._pad_bottom_offsets = {}
        if config.grasp_pad_bottom_m is not None:
            self._calibrate_pad_bottom()

    def _configure_contact_friction(self) -> None:
        """Apply finger and support materials using Newton's geometric mixing."""
        shape_mu = self.model.shape_material_mu.numpy()
        bodies = self.model.shape_body.numpy()
        fingers = np.isin(
            bodies,
            [self.contact_body_ids[f"{side}_{i}"] for side in ("left", "right") for i in (0, 1)],
        )
        target = self.config.gripper_package_friction
        particle_mu = float(self.model.soft_contact_mu)
        if target is not None:
            if target > 0 and particle_mu <= 0:
                raise ValueError(
                    "Positive pair friction requires nonzero particle material friction"
                )
            shape_mu[fingers] = target**2 / particle_mu if target > 0 else 0.0
        shape_mu[self.belt_shape] = self.config.table_friction
        if self.config.platform_friction is not None:
            shape_mu[self.platform_shape] = self.config.platform_friction
        self.model.shape_material_mu.assign(shape_mu)
        supports = (bodies < 0) | (bodies == self.belt_body)
        self.contact_friction = dict(
            particle_material_mu=particle_mu,
            finger_material_mu=np.unique(shape_mu[fingers]).tolist(),
            effective_gripper_package_mu=np.unique(
                np.sqrt(particle_mu * shape_mu[fingers])
            ).tolist(),
            effective_support_package_mu=np.unique(
                np.sqrt(particle_mu * shape_mu[supports])
            ).tolist(),
            mixing="geometric_mean",
            platform_shape=self.platform_shape,
            platform_material_mu=float(shape_mu[self.platform_shape]),
            effective_platform_package_mu=float(
                np.sqrt(particle_mu * shape_mu[self.platform_shape])
            ),
        )

    def _calibrate_pad_bottom(self) -> None:
        """Evaluate native linkage FK in scratch buffers without advancing dynamics."""
        angles = np.linspace(0.0, 0.7, 141)
        offsets = {side: [] for side in self.hand_bodies}
        q = self.state_0.joint_q.numpy().copy()
        scratch = self.model.state()
        coordinates = self.wp.array(q, dtype=self.wp.float32, device=self.model.device)
        for angle in angles:
            for pairs in self.robot.mimic.values():
                for index, ratio in pairs:
                    q[index] = ratio * angle
            coordinates.assign(q)
            self.newton.eval_fk(self.model, coordinates, self.state_0.joint_qd, scratch)
            poses = scratch.body_q.numpy()
            for side, hand in self.hand_bodies.items():
                pose = poses[hand]
                rotation = _rotation_wxyz(pose[[6, 3, 4, 5]])
                tcp = pose[:3] + rotation @ self.iks[side].tip[:3, 3]
                tcp_rotation = rotation @ self.iks[side].tip[:3, :3]
                bodies = {self.contact_body_ids[f"{side}_{i}"] for i in (0, 1)}
                heights = []
                for pad in self.pad_colliders:
                    if pad["body_id"] not in bodies:
                        continue
                    pose = poses[pad["body_id"]]
                    world = pad["vertices"] @ _rotation_wxyz(pose[[6, 3, 4, 5]]).T
                    local = (world + pose[:3] - tcp) @ tcp_rotation
                    heights.append((local @ self.grasp_rotation.T)[:, 2].min())
                if not heights:
                    raise ValueError(f"No native pad geometry found for {side}")
                offsets[side].append(min(heights))
        self._pad_bottom_offsets = {
            side: (angles, np.asarray(values)) for side, values in offsets.items()
        }

    def pad_bottom_tcp_height(self, side: str, angle: float, bottom: float) -> float:
        """Return target TCP height for a specified native pad bottom plane."""
        angles, offsets = self._pad_bottom_offsets[side]
        if not angles[0] <= angle <= angles[-1]:
            raise ValueError("Pad calibration angle is outside the native linkage range")
        return float(bottom - np.interp(angle, angles, offsets))

    def pad_bottom_z(self, side: str, poses: np.ndarray) -> float:
        """Measure the lowest pad vertex from actual world-frame body poses."""
        bodies = {self.contact_body_ids[f"{side}_{i}"] for i in (0, 1)}
        return min(
            float(
                (
                    pad["vertices"] @ _rotation_wxyz(poses[pad["body_id"]][[6, 3, 4, 5]]).T
                    + poses[pad["body_id"]][:3]
                )[:, 2].min()
            )
            for pad in self.pad_colliders
            if pad["body_id"] in bodies
        )

    def _create_mujoco_solver(self) -> Any:
        from newton.solvers import SolverMuJoCo

        solver = SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            iterations=self.config.mujoco_iterations,
            use_mujoco_cpu=self.config.mujoco_cpu,
            update_data_interval=0 if self.config.mujoco_cpu else 1,
        )
        # Native MuJoCo joint equalities replace the USD's unapplied PhysX mimic
        # attributes. Keep their compliance explicit in the saved configuration.
        time_constant = max(
            self.config.mimic_time_constant_s,
            2
            * self.robot.sim_world.config.physics_dt
            / (self.config.newton_substeps * self.config.mujoco_substeps_per_vbd_step),
        )
        solver.mj_model.eq_solref[:, 0] = time_constant
        solver.mj_model.eq_solimp[:, 0] = self.config.mimic_impedance
        solver.mj_model.eq_solimp[:, 1] = 0.9999
        if not self.config.mujoco_cpu:
            solver.mjw_model.eq_solref.assign(solver.mj_model.eq_solref[None])
            solver.mjw_model.eq_solimp.assign(solver.mj_model.eq_solimp[None])
        else:
            # VBD does not feed joint state back into this one-way rigid model.
            # Keep MuJoCo's continuous double-precision state between substeps.
            solver.reset(self.state_0)
            mapping = {
                int(q): int(mj)
                for q, mj, dim in zip(
                    self.model.joint_q_start.numpy()[:-1],
                    solver.mj_q_start.numpy(),
                    self.model.joint_dof_dim.numpy(),
                    strict=True,
                )
                if sum(dim) > 0
            }
            pairs = [
                (mapping[index], mapping[self.robot.gripper_indices[side]], ratio)
                for side, joints in self.robot.mimic.items()
                for index, ratio in joints
            ]
            self._native_mimic_pairs = np.asarray(pairs)
            self.native_mimic_residuals: list[np.ndarray] = []
            self.maximum_native_mimic_error = 0.0
        return solver

    def _configure_builder(self, builder: Any) -> None:
        # Import the authored convex collision approximations first. Requesting SDFs
        # before USD approximation would cause Newton to reject hull conversion.
        pass

    def _finalize_model(self, builder: Any) -> Any:
        if np.any(self.physics_origin):
            # Newton's native merge transforms particles, fixed world shapes,
            # articulated roots and free joints together. A constant frame
            # translation adds no force and changes no relative geometry.
            local = self.newton.ModelBuilder()
            local.add_builder(
                builder,
                xform=self.wp.transform(-self.physics_origin, self.wp.quat_identity()),
            )
            builder = local
        return super()._finalize_model(builder)

    def vertices(self) -> np.ndarray:
        """Return world coordinates while the optional solver frame stays local."""
        return super().vertices() + self.physics_origin

    def body_poses(self) -> np.ndarray:
        """Return solved rigid poses in the same world frame as the recorded particles."""
        poses = self.state_0.body_q.numpy().astype(float)
        poses[:, :3] += self.physics_origin
        return poses

    def _add_deformables(self, builder: Any, *, xpbd: bool) -> None:
        wp = self.wp
        for shape, source in enumerate(builder.shape_source):
            if source is not None and hasattr(source, "vertices"):
                builder.shape_force_sdf[shape] = True
                builder.shape_sdf_max_resolution[shape] = self.config.contact_sdf_resolution
        for name in ("platform", "outfeed"):
            low, high = self.scene.supports[name]
            center, half = (low + high) / 2, (high - low) / 2
            body = -1
            transform = wp.transform(center, wp.quat_identity())
            if name == "outfeed":
                body = builder.add_body(xform=transform, is_kinematic=True, label="outfeed_belt")
                self.belt_body = body
                transform = wp.transform_identity()
            shape = builder.add_shape_box(body, xform=transform, hx=half[0], hy=half[1], hz=half[2])
            if name == "platform":
                self.platform_shape = shape
            if name == "outfeed":
                self.belt_shape = shape
        super()._add_deformables(builder, xpbd=xpbd)

    def _create_ik(self) -> None:
        self.iks = {side: SortingArmIK(self.robot, side) for side in ("left", "right")}
        self.hand_bodies = {
            side: self.model.body_label.index(ik.body_path) for side, ik in self.iks.items()
        }
        self.hand_body = self.hand_bodies["left"]

    def _contact_bodies(self) -> dict[str, int]:
        result = {}
        for side in ("left", "right"):
            for index, name in enumerate(("left_inner_finger", "right_inner_finger")):
                result[f"{side}_{index}"] = next(
                    i
                    for i, label in enumerate(self.model.body_label)
                    if label.endswith(f"/robotiq_{side}/{name}")
                )
        result.update(left=result["left_0"], right=result["left_1"], palm=self.hand_body)
        return result

    def _create_robot_visuals(self) -> None:
        from pxr import UsdGeom

        robot = self.robot
        self._body_ops = []
        for label in self.model.body_label[: self.robot_body_count]:
            path = label.replace(robot.source_root, robot.config.prim_path, 1)
            xf = UsdGeom.Xformable(robot.sim_world.stage.GetPrimAtPath(path))
            if not xf:
                raise ValueError(f"Missing robot visual body: {path}")
            xf.ClearXformOpOrder()
            xf.SetResetXformStack(True)
            self._body_ops.append(xf.AddTransformOp())
        self._gripper_ops = {}
        for side, ik in self.iks.items():
            path = f"{robot.source_root}/robotiq_{side}"
            source = robot.source_stage.GetPrimAtPath(path)
            from pxr import UsdGeom

            world = np.asarray(UsdGeom.Xformable(source).ComputeLocalToWorldTransform(0)).T
            end = robot.source_stage.GetPrimAtPath(ik.body_path)
            end_world = np.asarray(UsdGeom.Xformable(end).ComputeLocalToWorldTransform(0)).T
            xf = UsdGeom.Xformable(
                robot.sim_world.stage.GetPrimAtPath(
                    path.replace(robot.source_root, robot.config.prim_path, 1)
                )
            )
            xf.ClearXformOpOrder()
            xf.SetResetXformStack(True)
            self._gripper_ops[side] = (xf.AddTransformOp(), np.linalg.inv(end_world) @ world)
        self._publish_robot_visuals()
        robot.initialize_cameras()

    def _publish_robot_visuals(self, transforms: np.ndarray | None = None) -> None:
        from pxr import Gf

        if transforms is None:
            transforms = self.body_poses()
        matrices = []
        for pose in transforms[: self.robot_body_count]:
            matrix = np.eye(4)
            matrix[:3, :3] = _rotation_wxyz(pose[[6, 3, 4, 5]])
            matrix[:3, 3] = pose[:3]
            matrices.append(matrix)
        for op, matrix in zip(self._body_ops, matrices, strict=True):
            op.Set(Gf.Matrix4d(matrix.T))
        for side, (op, adapter) in self._gripper_ops.items():
            op.Set(Gf.Matrix4d((matrices[self.hand_bodies[side]] @ adapter).T))

    def publish_recorded_visuals(self, vertices: np.ndarray, body_poses: np.ndarray) -> None:
        """Publish cached observations without modifying either solver's state buffers."""
        self.scene.publish_cloth_vertices(vertices)
        self._publish_robot_visuals(body_poses)

    def tcp_position(self, side: str | None = None) -> np.ndarray:
        """Read the actual solved TCP, including MuJoCo tracking error."""
        side = side or self.active_side
        pose = self.state_0.body_q.numpy()[self.hand_bodies[side]]
        return (
            pose[:3]
            + self.physics_origin
            + _rotation_wxyz(pose[[6, 3, 4, 5]]) @ self.iks[side].tip[:3, 3]
        )

    def command_arm(self, side: str, target: np.ndarray, gripper_angle_rad: float) -> float:
        """Drive native joint targets; soft-body positions are never commanded."""
        self._activate_side(side)
        q, error = self.iks[side].solve(np.asarray(target), self.grasp_rotation)
        self.target_q[self.robot.arm_indices[side]] = q
        self.set_gripper_angle(side, gripper_angle_rad)
        self.control.joint_target_q.assign(self.target_q)
        return error

    def _activate_side(self, side: str) -> None:
        self.active_side = side
        self.contact_body_ids.update(
            left=self.contact_body_ids[f"{side}_0"],
            right=self.contact_body_ids[f"{side}_1"],
            palm=self.hand_bodies[side],
        )

    def prepare_approach(
        self, side: str, target: np.ndarray, waypoints: list[np.ndarray] | None = None
    ) -> float:
        """Find a reachable top-down posture before moving from the ready posture."""
        self.approach_start = self.target_q[self.robot.arm_indices[side]].copy()
        self.approach_goal, error = self.iks[side].solve_pregrasp(
            target, self.grasp_rotation, waypoints=waypoints
        )
        return error

    @property
    def grasp_rotation(self) -> np.ndarray:
        """Honor the configured physical TCP orientation in both IK stages."""
        return _rotation_wxyz(np.asarray(self.config.tcp_orientation_wxyz, dtype=float))

    def command_approach(self, side: str, blend: float) -> None:
        """Interpolate only arm joint targets; allow the articulation to solve actual motion."""
        self._activate_side(side)
        self.target_q[self.robot.arm_indices[side]] = (
            self.approach_start * (1 - blend) + self.approach_goal * blend
        )
        self.set_gripper(side, False)

    def set_gripper(self, side: str, closed: bool | float) -> None:
        """Set the master drive and matching passive mimic targets for metadata."""
        closure = float(np.clip(closed, 0.0, 1.0))
        angle = self.config.open_gripper_rad + closure * (
            self.config.closed_gripper_rad - self.config.open_gripper_rad
        )
        self.set_gripper_angle(side, angle)

    def set_gripper_angle(self, side: str, angle: float) -> None:
        """Command the actual master angle within the source USD's 0–0.7 rad limits."""
        if not 0 <= angle <= 0.7:
            raise ValueError("Pro-M6 gripper angle must remain within the authored joint limits")
        for index, ratio in self.robot.mimic[side]:
            self.target_q[index] = ratio * angle
        self.control.joint_target_q.assign(self.target_q)

    def recording_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return fourteen measured arm angles and two measured master gripper angles."""
        indices = [
            *self.robot.arm_indices["left"],
            *self.robot.arm_indices["right"],
            self.robot.gripper_indices["left"],
            self.robot.gripper_indices["right"],
        ]
        return self.state_0.joint_q.numpy()[indices], self.target_q[indices]

    def _advance_direct(self, frame_dt: float) -> None:
        dt = frame_dt / self.config.newton_substeps
        self.cloth_solver.rebuild_bvh(self.state_0)
        batch_control = self.config.mujoco_cpu and self.config.mujoco_control_per_frame
        if batch_control:
            # Every joint is a scalar hinge, targets are constant over this
            # physics frame and this coupling has no soft-to-rigid force path.
            # Newton's CPU bridge otherwise downloads identical control/zero
            # force buffers at every substep, synchronizing the CUDA soft solve.
            self.state_0.clear_forces()
            solver = self.rigid_solver
            solver._apply_mjc_control(self.model, self.state_0, self.control, solver.mj_data)
            rigid_substeps = self.config.mujoco_substeps_per_vbd_step
            solver.mj_model.opt.timestep = dt / rigid_substeps
            # Advance the independent rigid system first and upload its complete
            # solved substep trajectory together. Repeated small CPU→CUDA copies
            # otherwise wait for the preceding soft solve on the same stream.
            positions, velocities, forces, mimic_residuals = [], [], [], []
            pairs = self._native_mimic_pairs
            for _ in range(self.config.newton_substeps):
                for _ in range(rigid_substeps):
                    solver._mujoco.mj_step(solver.mj_model, solver.mj_data)
                    solver._step += 1
                    q = solver.mj_data.qpos
                    mimic_residuals.append(
                        q[pairs[:, 0].astype(int)] - q[pairs[:, 1].astype(int)] * pairs[:, 2]
                    )
                positions.append(solver.mj_data.qpos.copy())
                velocities.append(solver.mj_data.qvel.copy())
                forces.append(solver.mj_data.qfrc_actuator.copy())
            residuals = np.asarray(mimic_residuals)
            self.native_mimic_residuals.append(residuals)
            self.maximum_native_mimic_error = max(
                self.maximum_native_mimic_error, float(np.max(np.abs(residuals)))
            )
            arrays = [
                np.asarray(values, dtype=np.float32) for values in (positions, velocities, forces)
            ]
            if not hasattr(self, "_rigid_frame_buffers"):
                self._rigid_frame_buffers = [
                    self.wp.empty(values.shape, dtype=float, device=self.model.device)
                    for values in arrays
                ]
                q, qd, force = self._rigid_frame_buffers
                self._rigid_substeps = [
                    SimpleNamespace(
                        nworld=1,
                        qpos=q[i : i + 1],
                        qvel=qd[i : i + 1],
                        qfrc_actuator=force[i : i + 1],
                    )
                    for i in range(self.config.newton_substeps)
                ]
            for buffer, values in zip(self._rigid_frame_buffers, arrays, strict=True):
                buffer.assign(values)
        for substep in range(self.config.newton_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            count = self.model.particle_count
            self.model.particle_count = 0
            try:
                if batch_control:
                    # Keep all native MuJoCo integration, equalities and rigid
                    # contacts at the original substep rate (Newton 1.6.1 API).
                    solver._update_newton_state(
                        self.model,
                        self.state_1,
                        self._rigid_substeps[substep],
                        state_prev=self.state_0,
                    )
                else:
                    self.rigid_solver.step(self.state_0, self.state_1, self.control, None, dt)
            finally:
                self.model.particle_count = count
            # A treadmill changes only the support's tangential contact displacement.
            # Actual parcel displacement is produced entirely by VBD friction.
            key = (id(self.state_0), self.belt_speed)
            if self.config.mujoco_cpu and self.config.newton_use_cuda_graph and self.direct_warmed:
                if key not in self._soft_graphs:
                    with self.wp.ScopedCapture(device=self.model.device) as capture:
                        self._advance_soft(dt)
                    self._soft_graphs[key] = capture.graph
                self.wp.capture_launch(self._soft_graphs[key])
            else:
                self._advance_soft(dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _advance_soft(self, dt: float) -> None:
        self.wp.launch(
            self._belt_motion,
            1,
            inputs=[
                self.state_0.body_q,
                self.state_1.body_q,
                self.belt_body,
                self.wp.transform(self.belt_rest[:3], self.belt_rest[3:]),
                self.belt_speed,
                dt,
            ],
            device=self.model.device,
        )
        key = id(self.state_0)
        if key not in self._collision_views:
            # External VBD evaluates particle contacts against state_out's
            # already integrated rigid poses, while particles start in state_in.
            # Preserve the old rigid poses separately for frictional motion.
            view = self.newton.State()
            view.particle_q = self.state_0.particle_q
            view.particle_qd = self.state_0.particle_qd
            view.body_q = self.state_1.body_q
            view.body_qd = self.state_1.body_qd
            self._collision_views[key] = view
        self.pipeline.collide(self._collision_views[key], self.contacts)
        self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
