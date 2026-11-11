"""Contact-driven Franka grasp of the V7 bag and its independent garment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from real2sim2real.planner.planner_newton_cloth import (
    OfficialClothPlanner,
    OfficialClothPlannerConfig,
    OfficialClothRuntime,
)
from real2sim2real.planner.planner_physic_solver import NewtonClothRuntime, PickPlaceClothPlanner


class PackagePlannerConfig(OfficialClothPlannerConfig):
    """Metre-based V7 materials are read from USD; these fields control the robot."""

    backend: str = "pickplace_package"
    full_surface_contacts: bool = True
    contact_stiffness: float = Field(default=20000.0, gt=0)
    contact_damping: float = Field(default=30.0, ge=0)
    shape_contact_stiffness: float = Field(default=20000.0, gt=0)
    shape_contact_damping: float = Field(default=30.0, ge=0)
    shape_contact_friction: float = Field(default=1.4, ge=0)
    table_friction: float = Field(default=0.5, ge=0)
    cloth_friction: float = Field(default=1.4, ge=0)
    open_finger_m: float = Field(default=0.04, gt=0, le=0.04)
    closed_finger_m: float = Field(default=0.00625, ge=0, le=0.04)
    finger_velocity_gain: float = Field(default=6.0, gt=0)
    grasp_height_m: float = Field(default=0.0704, gt=0)
    minimum_lift_m: float = Field(default=0.08, gt=0)
    maximum_table_penetration_m: float = Field(default=0.0015, ge=0)
    maximum_place_error_m: float = Field(default=0.08, gt=0)
    minimum_release_drop_m: float = Field(default=0.01, ge=0)
    lift_height_m: float = Field(default=0.18, gt=0)
    place_offset_m: tuple[float, float, float] = (0.0, -0.12, 0.0)
    place_height_m: float = Field(default=0.17, gt=0)
    self_contact_buffer: int = Field(default=512, ge=32)
    self_contact_gap_m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    friction_epsilon_m_s: float = Field(default=0.001, gt=0)
    arm_velocity_gain: float = Field(default=3.0, gt=0)
    approach_s: float = Field(default=2.0, gt=0)
    descend_s: float = Field(default=1.5, gt=0)
    close_s: float = Field(default=1.5, gt=0)
    lift_s: float = Field(default=2.0, gt=0)
    transfer_s: float = Field(default=1.5, gt=0)
    lower_s: float = Field(default=1.0, gt=0)
    release_s: float = Field(default=1.0, gt=0)
    retract_s: float = Field(default=1.0, gt=0)
    rest_s: float = Field(default=1.5, gt=0)
    align_place_to_com: bool = False


class PackageRuntime(OfficialClothRuntime):
    """Use Featherstone velocities or MuJoCo position drives with two-mesh VBD."""

    scale = 1.0
    color_bending = True

    def __init__(self, robot: Any, scene: Any, config: Any) -> None:
        solver = robot.sim_world.config.newton_solver
        mode = {"featherstone": "velocity", "mujoco": "position"}.get(solver)
        if not robot.pure_newton or mode is None:
            raise ValueError("Package requires direct Newton Featherstone or MuJoCo")
        if robot.config.control_mode != mode or config.cloth_solver != "vbd":
            raise ValueError(f"Package {solver} requires {mode} control and VBD")
        # Preserve authored bending rest angles; the shirt profile flattens them.
        NewtonClothRuntime.__init__(self, robot, scene, config)
        self.model.shape_material_ke.fill_(config.shape_contact_stiffness)
        self.model.shape_material_kd.fill_(config.shape_contact_damping)
        self.model.shape_material_mu.assign(
            np.where(
                self.model.shape_body.numpy() < 0,
                config.table_friction,
                config.shape_contact_friction,
            )
        )
        self.model.shape_margin.fill_(0.0002)

    def _create_ik(self) -> None:
        if self.robot.config.control_mode == "position":
            NewtonClothRuntime._create_ik(self)
            self.tcp_offset_cm = 0.2104  # Same TCP as the URDF geometric IK, in metres.
        else:
            super()._create_ik()
            self.tcp_offset_cm = 0.22  # In this runtime all Newton lengths are metres.

    def _create_robot_visuals(self) -> None:
        """Display any V7 jaw extensions using exactly their collision dimensions."""
        from pxr import Gf, UsdGeom

        super()._create_robot_visuals()
        model = self.model
        for shape in getattr(self.robot, "package_pad_shapes", []):
            body = int(model.shape_body.numpy()[shape])
            name = model.body_label[body].rsplit("/", 1)[-1]
            cube = UsdGeom.Cube.Define(
                self.robot.sim_world.stage, f"{self.robot.config.prim_path}/{name}/v7_pad"
            )
            cube.CreateSizeAttr(2.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.12, 0.12)])
            transform = model.shape_transform.numpy()[shape]
            cube.AddTranslateOp().Set(Gf.Vec3d(*(transform[:3] / self.scale).tolist()))
            cube.AddOrientOp().Set(Gf.Quatf(float(transform[6]), *transform[3:6].tolist()))
            cube.AddScaleOp().Set(
                Gf.Vec3f(*(model.shape_scale.numpy()[shape] / self.scale).tolist())
            )

    def track_keyframe(
        self, target_m: np.ndarray, quaternion_xyzw: np.ndarray, activation: float
    ) -> None:
        """Track the parcel path faster while respecting authored arm velocity limits."""
        if self.robot.config.control_mode == "position":
            blend = np.clip((activation - 0.1) / 0.7, 0.0, 1.0)
            opening = self.config.closed_finger_m + blend * (
                self.config.open_finger_m - self.config.closed_finger_m
            )
            self.command(target_m, quaternion_xyzw[[3, 0, 1, 2]], opening)
            return
        super().track_keyframe(target_m, quaternion_xyzw, activation)
        velocity = self._velocity_target.numpy()
        limits = self.model.joint_velocity_limit.numpy()[:7]
        velocity[:7] = np.clip(velocity[:7] * self.config.arm_velocity_gain, -limits, limits)
        self._velocity_target.assign(velocity)
        self.target_q = self.state_0.joint_q.numpy() + (
            velocity * self.robot.sim_world.config.physics_dt
        )

    def hold(self) -> None:
        """Hold the actual posture using the active controller's command type."""
        if self.robot.config.control_mode == "position":
            self.target_q = self.state_0.joint_q.numpy().copy()
            self.control.joint_target_q.assign(self.target_q)
        else:
            super().hold()

    def recording_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Record measured joints and either position or velocity commands."""
        if self.robot.config.control_mode == "position":
            return NewtonClothRuntime.recording_coordinates(self)
        return super().recording_coordinates()

    def _add_deformables(self, builder: Any, *, xpbd: bool) -> None:
        if xpbd:
            raise ValueError("V7 package materials require VBD")
        wp, scene = self.wp, self.scene
        p = scene.package_parameters
        for part in scene.package_parts:
            film = part["name"] == "Bag"
            start, end = part["start"], part["end"]
            stiffness = p["bag_stiffness"] if film else p["cloth_stiffness"]
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=scene.cloth_vertices[start:end].tolist(),
                indices=part["triangles"].reshape(-1).tolist(),
                density=1.0,
                tri_ke=stiffness,
                tri_ka=stiffness,
                tri_kd=p["bag_damping"] if film else p["cloth_damping"],
                edge_ke=0.00001 if film else p["bend_stiffness"],
                edge_kd=p["bag_bend_damping"] if film else p["cloth_bend_damping"],
                particle_radius=0.00035 if film else 0.0008,
            )
            builder.particle_mass[start:end] = part["masses"].tolist()
        for (i, j), weight in zip(scene.loft_springs, scene.loft_weights, strict=True):
            builder.add_spring(
                int(i), int(j), p["loft_stiffness"] * weight, p["loft_damping"] * weight, 0.0
            )

    def _create_cloth_solver(self) -> Any:
        from real2sim2real.utils.package_vbd import MovingFrameVBD

        return MovingFrameVBD(
            self.model,
            iterations=self.config.solver_iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.0016,
            particle_self_contact_gap=(
                self.scene.package_parameters["self_gap"]
                if self.config.self_contact_gap_m is None
                else self.config.self_contact_gap_m
            ),
            particle_topological_contact_filter_threshold=2,
            particle_vertex_contact_buffer_size=self.config.self_contact_buffer,
            particle_edge_contact_buffer_size=self.config.self_contact_buffer,
            rigid_body_particle_contact_buffer_size=16384,
            friction_epsilon=self.config.friction_epsilon_m_s,
        )

    def _create_collision_pipeline(self) -> Any:
        return self.newton.CollisionPipeline(
            self.model,
            soft_contact_gap=0.004,
            soft_contact_max=16384,
            enable_rigid_soft_full_surface_contact=True,
            deterministic=True,
        )


class PackagePlanner(OfficialClothPlanner):
    """Move the physical parcel with finger contacts and verify both surfaces lift."""

    config_cls = PackagePlannerConfig
    runtime_cls = PackageRuntime

    @property
    def expected_duration_s(self) -> float:
        """Return the duration of the complete approach, grasp and placement."""
        return sum(
            getattr(self.config, name)
            for name in (
                "approach_s",
                "descend_s",
                "close_s",
                "lift_s",
                "transfer_s",
                "lower_s",
                "release_s",
                "retract_s",
                "rest_s",
            )
        )

    @property
    def action_names(self) -> list[str]:
        """Describe commands in the actual controller's units."""
        if self.robot.config.control_mode == "position":
            return [f"fr3_joint{i}_target_rad" for i in range(1, 8)] + [
                "left_finger_target_m",
                "right_finger_target_m",
            ]
        return OfficialClothPlanner.action_names

    def begin_episode(self, *, register_stepper: bool = True) -> None:
        """Build both physical meshes and the robot, then initialize the task schedule."""
        PickPlaceClothPlanner.begin_episode(self, register_stepper=register_stepper)
        center = np.array([*self.scene.config.target_xy, self.scene.table_top_z])
        center += self.config.grasp_offset_m
        grasp = center + [0, 0, self.config.grasp_height_m]
        raised = grasp + [0, 0, self.config.lift_height_m]
        destination = raised + self.config.place_offset_m
        lower = center + self.config.place_offset_m + [0, 0, self.config.place_height_m]
        self._schedule = [
            ("approach", self.config.approach_s, center + [0, 0, 0.20], False),
            ("descend", self.config.descend_s, grasp, False),
            ("close", self.config.close_s, grasp, True),
            ("lift", self.config.lift_s, raised, True),
            ("transfer", self.config.transfer_s, destination, True),
            ("lower", self.config.lower_s, lower, True),
            ("release", self.config.release_s, lower, False),
            ("retract", self.config.retract_s, lower + [0, 0, 0.15], False),
            ("settle", self.config.rest_s, lower + [0, 0, 0.15], False),
        ]
        self._ends = np.cumsum([row[1] for row in self._schedule])
        self._start_tcp = self.runtime.tcp_position().copy()
        self._baseline: np.ndarray | None = None
        self._peak_lifts = np.zeros(2)
        self._bilateral = False
        self._transfer_count = self._transfer_contacts = 0
        self._release_z: float | None = None
        self._release_drop = self._penetration = self._stretch = 0.0
        self._particle_samples: list[np.ndarray] = []
        self._sample_times: list[float] = []
        self._placement_correction: np.ndarray | None = None

    def _align_placement(self, garment_com: np.ndarray) -> None:
        """Translate the remaining robot path so the hanging parcel COM reaches its goal."""
        goal = np.asarray(self.scene.config.target_xy) + np.asarray(self.config.place_offset_m[:2])
        destination = self._schedule[4][2]
        correction = np.zeros(3)
        correction[:2] = (
            goal - garment_com[:2] - (destination[:2] - self.runtime.tcp_position()[:2])
        )
        for index in range(4, len(self._schedule)):
            phase, duration, target, closed = self._schedule[index]
            self._schedule[index] = (phase, duration, target + correction, closed)
        self._placement_correction = correction

    def on_physics_step(self, robot: Any) -> bool:
        """Command only robot joints and measure the unconstrained bag and garment."""
        del robot
        index = int(np.searchsorted(self._ends, self.elapsed, side="right"))
        vertices = self.runtime.vertices()
        if len(self.rows) % 4 == 0:
            self._particle_samples.append(vertices.astype(np.float32))
            self._sample_times.append(self.elapsed)
        n = self.scene.garment_count
        masses = self.runtime.model.particle_mass.numpy()
        coms = np.asarray(
            [
                np.average(vertices[:n], axis=0, weights=masses[:n]),
                np.average(vertices[n:], axis=0, weights=masses[n:]),
            ]
        )
        if self._baseline is None and self.elapsed >= self._ends[0]:
            self._baseline = coms.copy()
        if self._baseline is not None:
            self._peak_lifts = np.maximum(self._peak_lifts, coms[:, 2] - self._baseline[:, 2])
        self._penetration = max(self._penetration, self.scene.table_top_z - vertices[:, 2].min())
        lengths = np.linalg.norm(
            vertices[self._edge_indices[:, 0]] - vertices[self._edge_indices[:, 1]], axis=1
        )
        self._stretch = max(self._stretch, float((lengths / self._edge_lengths).max()))
        if index >= len(self._schedule):
            self.runtime.hold()
            self.state.update(done=True, phase="done")
            return True
        if (
            self._schedule[index][0] == "transfer"
            and self.config.align_place_to_com
            and self._placement_correction is None
        ):
            self._align_placement(coms[0])
        phase, duration, target, closed = self._schedule[index]
        contacts = self.runtime.last_contact_counts
        self.state["progress"] = dict(
            com_lifts_m=self._peak_lifts.tolist(),
            finger_contacts=[contacts["left"], contacts["right"]],
            maximum_edge_stretch=float(self._stretch),
            maximum_table_penetration_m=float(self._penetration),
            tcp_m=self.runtime.tcp_position().tolist(),
        )
        bilateral = contacts["left"] > 0 and contacts["right"] > 0
        if closed and bilateral:
            self._bilateral = True
        if phase == "transfer":
            self._transfer_count += 1
            self._transfer_contacts += int(bilateral)
        if phase == "release" and self._release_z is None:
            self._release_z = float(coms[:, 2].mean())
        if self._release_z is not None:
            self._release_drop = max(self._release_drop, self._release_z - float(coms[:, 2].mean()))
        start_time = 0.0 if index == 0 else self._ends[index - 1]
        previous = self._start_tcp if index == 0 else self._schedule[index - 1][2]
        blend = np.clip((self.elapsed - start_time) / (duration * 0.6), 0, 1)
        blend = blend * blend * (3 - 2 * blend)
        command = previous * (1 - blend) + target * blend
        self.runtime.track_keyframe(
            command, np.array([2**-0.5, 2**-0.5, 0, 0]), 0.1 if closed else 0.8
        )
        self.state["phase"] = phase
        self.rows.append(
            dict(
                time_s=self.elapsed,
                phase=phase,
                garment_com_m=coms[0].tolist(),
                bag_com_m=coms[1].tolist(),
                tcp_m=self.runtime.tcp_position().tolist(),
                contacts_left=contacts["left"],
                contacts_right=contacts["right"],
                minimum_z_m=float(vertices[:, 2].min()),
            )
        )
        self.elapsed += self.robot.sim_world.config.physics_dt
        return False

    def write_diagnostics(self, directory: Path) -> None:
        """Save actual solver particles for independent geometry checks and replay."""
        np.savez_compressed(
            directory / "particles.npz",
            positions=np.asarray(self._particle_samples),
            times=np.asarray(self._sample_times),
            checkpoint_times_s=[0.0, *self._ends.tolist()],
            triangles=self.scene.triangle_indices.reshape(-1, 3),
            rest=self.runtime.model.particle_q.numpy(),
            masses=self.runtime.model.particle_mass.numpy(),
            garment_count=self.scene.garment_count,
        )

    def result(self) -> dict[str, Any]:
        """Reject partial lifts, lost transfers, penetration and misplaced releases."""
        from real2sim2real.utils.package_geometry import check_package_containment

        vertices = self.runtime.vertices()
        n = self.scene.garment_count
        masses = self.runtime.model.particle_mass.numpy()
        com = np.average(vertices[:n], axis=0, weights=masses[:n])
        goal = np.array(self.scene.config.target_xy) + np.array(self.config.place_offset_m[:2])
        place_error = float(np.linalg.norm(com[:2] - goal))
        fraction = self._transfer_contacts / max(1, self._transfer_count)
        capacity = self.runtime.cloth_solver.maximum_contacts.numpy().tolist()
        containment = check_package_containment(
            np.asarray(self._particle_samples),
            self.scene.triangle_indices.reshape(-1, 3),
            n,
            times=np.asarray(self._sample_times),
            checkpoint_times_s=[0.0, *self._ends.tolist()],
        )
        checks = dict(
            sequence_completed=bool(self.state.get("done")),
            finite_state=bool(np.isfinite(vertices).all()),
            bilateral_contact_observed=self._bilateral,
            garment_lifted=bool(self._peak_lifts[0] >= self.config.minimum_lift_m),
            bag_lifted=bool(self._peak_lifts[1] >= self.config.minimum_lift_m),
            transfer_contact_maintained=fraction >= 0.95,
            released_and_dropped=self._release_drop >= self.config.minimum_release_drop_m,
            placed_near_goal=place_error <= self.config.maximum_place_error_m,
            rests_on_table=float(com[2]) < self.scene.table_top_z + 0.07,
            table_penetration_bounded=self._penetration <= self.config.maximum_table_penetration_m,
            surface_stretch_bounded=self._stretch < 1.3,
            no_pinned_vertices=self.runtime.pinned_vertex_count == 0,
            garment_contained_at_checkpoints=containment["passed"],
            no_contact_buffer_overflow=(
                max(capacity[:2]) < self.config.self_contact_buffer
                and capacity[2] < min(16384, self.runtime.contacts.soft_contact_max)
            ),
        )
        checks = {name: bool(value) for name, value in checks.items()}
        return dict(
            success=all(checks.values()),
            checks=checks,
            object="V7_two_mesh_package",
            maximum_com_lift_m=self._peak_lifts.tolist(),
            release_drop_m=self._release_drop,
            place_error_m=place_error,
            transfer_bilateral_contact_fraction=fraction,
            maximum_table_penetration_m=float(self._penetration),
            maximum_edge_stretch=self._stretch,
            garment_mass_kg=float(masses[:n].sum()),
            bag_mass_kg=float(masses[n:].sum()),
            particles=len(vertices),
            garment_particles=n,
            bag_particles=len(vertices) - n,
            internal_loft_springs=len(self.scene.loft_springs),
            attachments_used=False,
            pinned_vertex_count=self.runtime.pinned_vertex_count,
            coupling="one_way_rigid_to_deformable",
            physics_duration_s=self.elapsed,
            maximum_substep_contacts=capacity,
            containment=containment,
            newton_version=self.runtime.newton.__version__,
            warp_version=self.runtime.wp.__version__,
            asset_sha256=hashlib.sha256(
                Path(self.scene._asset(self.scene.config.cloth_usd_path)).read_bytes()
            ).hexdigest(),
            robot_backend=self.robot.config.backend,
            gripper_geometry=(
                "V7_14x60x84mm_parallel_pads"
                if hasattr(self.robot, "package_pad_shapes")
                else "native_Franka_fingers"
            ),
            placement_correction_m=(
                self._placement_correction.tolist()
                if getattr(self, "_placement_correction", None) is not None
                else None
            ),
        )
