"""Newton cloth/contact runtime, using the installed official solver APIs.

The centimetre convention follows Newton's official ``example_cloth_franka``.
No garment vertices are pinned and no grasp attachments are created. VBD with
an external rigid solver is explicitly a one-way rigid-to-cloth coupling.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from real2sim2real.utils import resolve_usd_path
from real2sim2real.utils.newton_step import KinematicVBDStepper


def _rotation_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class _URDFHandIK:
    """Small geometric-Jacobian IK using the same official URDF as the solver.

    Keeping this CPU controller independent of Warp's IK kernels also avoids
    relying on an articulation-only IK model containing the cloth particles.
    """

    def __init__(self, robot: Any) -> None:
        root = ET.parse(resolve_usd_path(robot.config.urdf_path)).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, child = [], "fr3_hand"
        while child in by_child:
            joint = by_child[child]
            chain.append(joint)
            child = joint.find("parent").get("link")
        self.chain = list(reversed(chain))
        self.base = np.eye(4)
        self.base[:3, :3] = _rotation_wxyz(np.asarray(robot.config.base_orientation))
        self.base[:3, 3] = robot.config.base_position
        self.lower, self.upper = [], []
        for joint in self.chain:
            if joint.get("type") == "revolute":
                limit = joint.find("limit")
                self.lower.append(float(limit.get("lower")))
                self.upper.append(float(limit.get("upper")))
        self.q = np.asarray(robot.config.initial_joint_positions[:7], dtype=float)

    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transform = self.base.copy()
        origins, axes = [], []
        index = 0
        for joint in self.chain:
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            sr, sp, sy = np.sin([roll, pitch, yaw])
            cr, cp, cy = np.cos([roll, pitch, yaw])
            fixed = np.eye(4)
            fixed[:3, 3] = xyz
            fixed[:3, :3] = np.array(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr],
                ]
            )
            transform = transform @ fixed
            if joint.get("type") == "revolute":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                origins.append(transform[:3, 3].copy())
                axes.append(transform[:3, :3] @ axis)
                cross = np.array(
                    [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
                )
                rotation = np.eye(4)
                rotation[:3, :3] += np.sin(q[index]) * cross + (1 - np.cos(q[index])) * (
                    cross @ cross
                )
                transform = transform @ rotation
                index += 1
        tcp = transform[:3, 3] + transform[:3, :3] @ np.array([0.0, 0.0, 0.1034])
        jacobian = np.zeros((6, 7))
        for index, (origin, axis) in enumerate(zip(origins, axes, strict=True)):
            jacobian[:3, index] = np.cross(axis, tcp - origin)
            jacobian[3:, index] = axis
        pose = transform.copy()
        pose[:3, 3] = tcp
        return pose, jacobian

    def solve(self, position: np.ndarray, quaternion: np.ndarray, iterations: int) -> np.ndarray:
        rotation = _rotation_wxyz(quaternion)
        for _ in range(iterations):
            pose, jacobian = self._fk(self.q)
            angular = 0.5 * sum(
                (np.cross(pose[:3, i], rotation[:, i]) for i in range(3)), start=np.zeros(3)
            )
            error = np.concatenate((position - pose[:3, 3], angular))
            if np.linalg.norm(error) < 1e-5:
                break
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * 1e-4, error)
            self.q = np.clip(self.q + np.clip(step, -0.15, 0.15), self.lower, self.upper)
        return self.q.copy()


class FrankaClothRuntime:
    """Advance cloth plus a Newton articulation or live PhysX finger colliders."""

    def __init__(self, robot: Any, scene: Any, config: Any) -> None:
        import newton
        import warp as wp
        from newton.solvers import SolverFeatherstone, SolverMuJoCo, SolverVBD, SolverXPBD

        self.newton, self.wp = newton, wp
        self.robot, self.scene, self.config = robot, scene, config
        self.pure_newton = robot.pure_newton
        self.solver_name = str(robot.sim_world.config.newton_solver).lower()
        if not self.pure_newton and self.solver_name != "vbd":
            raise ValueError("The hybrid cloth driver requires newton_solver='vbd'")
        if self.solver_name == "semi_implicit":
            raise ValueError("The cloth grasp demo supports VBD, XPBD, MuJoCo, or Featherstone")
        xpbd = self.pure_newton and self.solver_name == "xpbd" and config.cloth_solver == "xpbd"
        if config.cloth_solver == "xpbd" and not xpbd:
            raise ValueError("cloth_solver='xpbd' requires pure Newton with newton_solver='xpbd'")
        self.scale = 100.0
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -981.0))
        if self.pure_newton and self.solver_name == "mujoco":
            SolverMuJoCo.register_custom_attributes(builder)
        self.proxy_body_ids: list[int] = []
        if self.pure_newton:
            robot.add_newton_robot(builder, scale=self.scale)
            self.robot_body_count = builder.body_count
            self.robot_shape_count = builder.shape_count
        else:
            # These are measured links of the running PhysX Panda. Each is
            # represented by its pad/palm collision box, never by a cloth pin.
            transforms = robot.collider_transforms()
            # Official Panda rubber tips occupy local +Y on the left and -Y
            # on the right (the USD mirrors its geometry, not its body frame).
            # Their inner faces lie at the measured prismatic joint opening.
            offsets = ([0.0, 0.00758, 0.04525], [0.0, -0.00758, 0.04525], [0.0, 0.0, 0.025])
            extents = ([0.00875, 0.0076, 0.00925], [0.00875, 0.0076, 0.00925], [0.03, 0.045, 0.025])
            for index, (transform, offset, extent) in enumerate(
                zip(transforms, offsets, extents, strict=True)
            ):
                body = builder.add_body(
                    xform=wp.transform(transform[:3] * self.scale, transform[3:]),
                    is_kinematic=True,
                    label=f"physx_finger_proxy_{index}",
                )
                self.proxy_body_ids.append(body)
                builder.add_shape_box(
                    body,
                    xform=wp.transform(np.asarray(offset) * self.scale, wp.quat_identity()),
                    hx=extent[0] * self.scale,
                    hy=extent[1] * self.scale,
                    hz=extent[2] * self.scale,
                )
            self.robot_body_count = 0
            self.robot_shape_count = 0
        low, high = np.asarray(scene.table_bounds, dtype=float)
        center, half = (low + high) * 0.5, (high - low) * 0.5
        builder.add_shape_box(
            -1,
            xform=wp.transform(center * self.scale, wp.quat_identity()),
            hx=half[0] * self.scale,
            hy=half[1] * self.scale,
            hz=half[2] * self.scale,
        )
        vertices_cm = np.asarray(scene.cloth_vertices) * self.scale
        triangles = np.asarray(scene.triangle_indices, dtype=np.int32).reshape(-1, 3)
        triangle_points = vertices_cm[triangles]
        area_cm2 = float(
            np.linalg.norm(
                np.cross(
                    triangle_points[:, 1] - triangle_points[:, 0],
                    triangle_points[:, 2] - triangle_points[:, 0],
                ),
                axis=1,
            ).sum()
            * 0.5
        )
        if area_cm2 <= 0:
            raise ValueError("The cloth surface has no positive triangle area")
        density = config.cloth_density_cm or config.cloth_mass_kg / area_cm2
        builder.add_cloth_mesh(
            vertices=[wp.vec3(*v) for v in vertices_cm],
            indices=triangles.reshape(-1),
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            scale=1.0,
            density=density,
            tri_ke=config.cloth_stretch_stiffness,
            tri_ka=config.cloth_stretch_stiffness,
            tri_kd=config.cloth_stretch_damping,
            edge_ke=config.cloth_bend_stiffness,
            edge_kd=config.cloth_bend_damping,
            # SolverXPBD projects edge springs; it does not solve the FEM
            # triangle stretch energies consumed by SolverVBD.
            add_springs=xpbd,
            spring_ke=config.xpbd_spring_stiffness,
            spring_kd=config.xpbd_spring_damping,
            particle_radius=config.particle_radius_m * self.scale,
        )
        if xpbd and config.xpbd_mass_lumping == "uniform":
            # The shirt's very small triangles create masses spanning >400x.
            # Uniform lumping is an explicit XPBD approximation that preserves
            # total mass while improving its unweighted spring Jacobi solve.
            total_mass = float(sum(builder.particle_mass))
            mass_per_vertex = total_mass / builder.particle_count
            builder.particle_mass[:] = [mass_per_vertex] * builder.particle_count
        builder.color()
        self.model = builder.finalize(device=robot.sim_world.config.device or "cuda:0")
        if xpbd and not config.xpbd_self_contact:
            # This kernel has no cloth-adjacency exclusion. Preserve the
            # particle radii for shape contacts while disabling its all-pairs
            # repulsion between neighboring garment vertices.
            self.model.particle_grid = None
            self.model.particle_max_radius = 0.0
        self.model.soft_contact_ke = config.contact_stiffness
        self.model.soft_contact_kd = config.contact_damping
        self.model.soft_contact_mu = config.cloth_friction
        bodies = self.model.shape_body.numpy()
        self.model.shape_material_mu.assign(
            np.where(bodies >= 0, robot.config.finger_friction, config.table_friction)
        )
        self.model.shape_material_ke.fill_(config.contact_stiffness)
        self.model.shape_material_kd.fill_(config.contact_damping)
        self.pinned_vertex_count = int(np.count_nonzero(self.model.particle_inv_mass.numpy() == 0))
        if self.pinned_vertex_count:
            raise RuntimeError("The cloth grasp demonstration requires zero pinned vertices")
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.rigid_previous_body_q = None
        self.rigid_previous_body_qd = None
        if self.pure_newton and self.solver_name == "xpbd" and not xpbd:
            # Native XPBD uses its input body buffers as Jacobi scratch space.
            # VBD's external-rigid path needs the untouched previous pose to
            # evaluate the fingers' displacement and contact friction.
            self.rigid_previous_body_q = wp.empty_like(self.state_0.body_q)
            self.rigid_previous_body_qd = wp.empty_like(self.state_0.body_qd)
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.pipeline = newton.CollisionPipeline(
            self.model, soft_contact_margin=config.particle_radius_m * self.scale
        )
        self.contacts = self.pipeline.contacts()
        self.unified_xpbd = xpbd
        if self.unified_xpbd:
            self.cloth_solver = SolverXPBD(self.model, iterations=config.solver_iterations)
            self.rigid_solver = None
        else:
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=config.solver_iterations,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=True,
                particle_self_contact_margin=0.2,
                particle_self_contact_gap=0.0,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.5,
                particle_vertex_contact_buffer_size=32,
                particle_edge_contact_buffer_size=32,
                particle_collision_detection_interval=-1,
                friction_epsilon=config.friction_epsilon_cm_s,
            )
            if self.pure_newton and self.solver_name == "mujoco":
                self.rigid_solver = SolverMuJoCo(
                    self.model, solver="newton", integrator="implicitfast", iterations=30
                )
            elif self.pure_newton and self.solver_name == "xpbd":
                self.rigid_solver = SolverXPBD(self.model, iterations=config.solver_iterations)
            elif self.pure_newton:
                self.rigid_solver = SolverFeatherstone(
                    self.model, update_mass_matrix_interval=config.newton_substeps
                )
            else:
                self.rigid_solver = None
        self.initial_q = self.model.joint_q.numpy().copy()
        self.target_q = self.initial_q.copy()
        self.tcp_target = np.asarray(scene.target_position).copy()
        self.finger_target = 0.04
        self.gravity_earth = wp.clone(self.model.gravity)
        self.gravity_zero = wp.zeros_like(self.model.gravity)
        self.sim_time = 0.0
        self._last_proxy = None
        self._visual_bodies: list[Any] = []
        if self.pure_newton:
            self._create_ik()
            self._create_robot_visuals()
        self.last_contacts = 0
        self.last_contact_counts = dict(left=0, right=0, palm=0, table=0)
        if self.pure_newton:
            labels = self.model.body_label
            self.contact_body_ids = {
                "left": next(i for i, name in enumerate(labels) if name.endswith("fr3_leftfinger")),
                "right": next(
                    i for i, name in enumerate(labels) if name.endswith("fr3_rightfinger")
                ),
                "palm": self.hand_body,
            }
        else:
            self.contact_body_ids = dict(
                zip(("left", "right", "palm"), self.proxy_body_ids, strict=True)
            )
        self.direct_graph: Any = None
        self.direct_warmed = False
        self.direct_use_graph = (
            self.pure_newton
            and config.newton_use_cuda_graph
            and self.model.device.is_cuda
            and config.newton_substeps % 2 == 0
        )
        if self.pure_newton and not self.unified_xpbd and self.solver_name != "xpbd":
            from real2sim2real.planner.franka_demo_newton_kernels import track_joint_velocity

            self.track_joint_velocity = track_joint_velocity
        self.kinematic_stepper = None
        if not self.pure_newton:
            self.kinematic_stepper = KinematicVBDStepper(
                self.model,
                self.cloth_solver,
                self.pipeline,
                self.contacts,
                self.state_0,
                self.state_1,
                self.control,
                config.newton_substeps,
                robot.sim_world.config.physics_dt / config.newton_substeps,
                use_graph=config.newton_use_cuda_graph,
            )

    def _create_ik(self) -> None:
        hand_bodies = [
            i for i, label in enumerate(self.model.body_label) if label.endswith("fr3_hand")
        ]
        if hand_bodies:
            self.hand_body, self.tcp_offset_cm = hand_bodies[0], 10.34
        else:
            # Fixed hand/flange/tool frames collapse into link7. Its TCP is
            # 0.107 + 0.1034 m along Z, as authored by the same official URDF.
            self.hand_body = next(
                i for i, label in enumerate(self.model.body_label) if label.endswith("fr3_link7")
            )
            self.tcp_offset_cm = 21.04
        self.ik = _URDFHandIK(self.robot)

    def command(self, position: np.ndarray, quaternion_wxyz: np.ndarray, finger: float) -> None:
        """Change only robot control targets; garment state remains unconstrained."""
        self.tcp_target = np.asarray(position, dtype=float)
        self.finger_target = float(finger)
        if not self.pure_newton:
            self.robot.command_tcp(position, quaternion_wxyz, finger)
            return
        self.target_q[:7] = self.ik.solve(
            self.tcp_target, quaternion_wxyz, self.config.ik_iterations
        )
        self.target_q[-2:] = finger * self.scale
        self.control.joint_target_q.assign(self.target_q)

    def step(self, frame_dt: float) -> None:
        """Advance all Newton substeps once and publish the resulting visual geometry."""
        substeps = self.config.newton_substeps
        proxy = None if self.pure_newton else self.robot.collider_transforms().copy()
        if proxy is not None:
            proxy[:, :3] *= self.scale
            previous = proxy if self._last_proxy is None else self._last_proxy
            q_inputs, q_outputs, velocities = [], [], []
            for index in range(substeps):
                alpha_0, alpha_1 = index / substeps, (index + 1) / substeps
                q_in = previous * (1 - alpha_0) + proxy * alpha_0
                q_out = previous * (1 - alpha_1) + proxy * alpha_1
                q_in[:, 3:] /= np.linalg.norm(q_in[:, 3:], axis=1, keepdims=True)
                q_out[:, 3:] /= np.linalg.norm(q_out[:, 3:], axis=1, keepdims=True)
                velocity = np.zeros((len(proxy), 6), dtype=np.float32)
                velocity[:, :3] = (proxy[:, :3] - previous[:, :3]) / frame_dt
                q_inputs.append(q_in)
                q_outputs.append(q_out)
                velocities.append(velocity)
            self.kinematic_stepper.step(
                np.asarray(q_inputs), np.asarray(q_outputs), np.asarray(velocities)
            )
            self.state_0 = self.kinematic_stepper.state_0
            self.state_1 = self.kinematic_stepper.state_1
            self._last_proxy = proxy
            self.sim_time += frame_dt
            self._publish()
            return
        if self.direct_use_graph and self.direct_warmed:
            if self.direct_graph is None:
                with self.wp.ScopedCapture(device=self.model.device) as capture:
                    self._advance_direct(frame_dt)
                self.direct_graph = capture.graph
            self.wp.capture_launch(self.direct_graph)
        else:
            self._advance_direct(frame_dt)
            self.direct_warmed = True
        self.sim_time += frame_dt
        self._publish()

    def _advance_direct(self, frame_dt: float) -> None:
        """Run a capture-compatible fixed number of shared-model substeps."""
        substeps = self.config.newton_substeps
        dt = frame_dt / substeps
        if hasattr(self.cloth_solver, "rebuild_bvh"):
            self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            if self.rigid_solver is not None:
                count = self.model.particle_count
                self.model.particle_count = 0
                try:
                    if self.solver_name != "mujoco":
                        # Gravity compensation for the mounted robot's driven
                        # trajectory; the freely simulated cloth keeps gravity.
                        self.model.gravity.assign(self.gravity_zero)
                    if self.solver_name not in {"mujoco", "xpbd"}:
                        self.wp.launch(
                            self.track_joint_velocity,
                            dim=self.model.joint_dof_count,
                            inputs=[self.control.joint_target_q, self.state_0.joint_q],
                            outputs=[self.state_0.joint_qd],
                            device=self.model.device,
                        )
                    if self.rigid_previous_body_q is not None:
                        self.wp.copy(self.rigid_previous_body_q, self.state_0.body_q)
                        self.wp.copy(self.rigid_previous_body_qd, self.state_0.body_qd)
                    self.rigid_solver.step(self.state_0, self.state_1, self.control, None, dt)
                    if self.rigid_previous_body_q is not None:
                        self.wp.copy(self.state_0.body_q, self.rigid_previous_body_q)
                        self.wp.copy(self.state_0.body_qd, self.rigid_previous_body_qd)
                finally:
                    self.model.particle_count = count
                    self.model.gravity.assign(self.gravity_earth)
                self.state_0.particle_f.zero_()
            self.pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _publish(self) -> None:
        """Publish and validate the latest actual state after one complete frame."""
        vertices = self.vertices()
        if not np.isfinite(vertices).all():
            raise FloatingPointError("Newton cloth generated non-finite vertex positions")
        self.scene.publish_cloth_vertices(vertices)
        if self.pure_newton:
            self._publish_robot_visuals()
        if hasattr(self.contacts, "soft_contact_count"):
            self.last_contacts = int(self.contacts.soft_contact_count.numpy()[0])
            shape_ids = self.contacts.soft_contact_shape.numpy()[: self.last_contacts]
            shape_ids = shape_ids[(shape_ids >= 0) & (shape_ids < self.model.shape_count)]
            bodies = self.model.shape_body.numpy()[shape_ids]
            self.last_contact_counts = {
                name: int(np.count_nonzero(bodies == body))
                for name, body in self.contact_body_ids.items()
            }
            self.last_contact_counts["table"] = int(np.count_nonzero(bodies < 0))

    def vertices(self) -> np.ndarray:
        """Return freely simulated garment positions in world metres."""
        return self.state_0.particle_q.numpy().astype(float) / self.scale

    def finger_gap_m(self) -> float:
        """Read actual finger separation, including XPBD's body-based states."""
        if not self.pure_newton:
            return float(np.sum(self.robot.finger_positions()))
        transforms = self.state_0.body_q.numpy()
        left, right = (self.contact_body_ids[name] for name in ("left", "right"))
        return float(np.linalg.norm(transforms[left, :3] - transforms[right, :3]) / self.scale)

    def tcp_position(self) -> np.ndarray:
        """Read actual simulated hand pose in world metres."""
        if not self.pure_newton:
            return self.robot.tcp_position()
        hand = self.state_0.body_q.numpy()[self.hand_body]
        point = self.wp.transform_point(
            self.wp.transform(hand[:3], hand[3:]), self.wp.vec3(0.0, 0.0, self.tcp_offset_cm)
        )
        return np.asarray(point, dtype=float) / self.scale

    def _create_robot_visuals(self) -> None:
        from pxr import Gf, UsdGeom, Vt

        model = self.model
        stage = self.robot.sim_world.stage
        for body in range(self.robot_body_count):
            xform = UsdGeom.Xform.Define(stage, f"{self.robot.config.prim_path}/body_{body}")
            self._visual_bodies.append(xform.AddTransformOp())
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_flags = model.shape_flags.numpy()
        transforms, scales = model.shape_transform.numpy(), model.shape_scale.numpy()
        visible = int(self.newton.ShapeFlags.VISIBLE)
        for index, body in enumerate(shape_body):
            if index >= self.robot_shape_count or not shape_flags[index] & visible:
                continue
            source = model.shape_source[index]
            if source is None or int(shape_type[index]) not in (
                int(self.newton.GeoType.MESH),
                int(self.newton.GeoType.CONVEX_MESH),
            ):
                continue
            if body < 0:
                # The URDF's fixed base collapses into the world. Its visual
                # transform already includes the table-mounted base pose.
                UsdGeom.Xform.Define(stage, f"{self.robot.config.prim_path}/fixed_base")
                path = f"{self.robot.config.prim_path}/fixed_base/shape_{index}"
            else:
                path = f"{self.robot.config.prim_path}/body_{body}/shape_{index}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(source.vertices, np.float32)))
            indices = np.asarray(source.indices, np.int32).reshape(-1)
            mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
            mesh.CreateFaceVertexIndicesAttr(indices.tolist())
            mesh.CreateDisplayColorAttr([Gf.Vec3f(0.88, 0.9, 0.92)])
            xf = UsdGeom.Xformable(mesh)
            tf = transforms[index]
            xf.AddTranslateOp().Set(Gf.Vec3d(*(tf[:3] / self.scale).tolist()))
            xf.AddOrientOp().Set(Gf.Quatf(float(tf[6]), *tf[3:6].tolist()))
            xf.AddScaleOp().Set(Gf.Vec3f(*(scales[index] / self.scale).tolist()))
        self._publish_robot_visuals()

    def _publish_robot_visuals(self) -> None:
        from pxr import Gf

        for op, transform in zip(self._visual_bodies, self.state_0.body_q.numpy(), strict=False):
            matrix = Gf.Matrix4d(1.0)
            matrix.SetRotate(Gf.Quatd(float(transform[6]), *transform[3:6].astype(float).tolist()))
            matrix.SetTranslateOnly(Gf.Vec3d(*(transform[:3] / self.scale).astype(float).tolist()))
            op.Set(matrix)
