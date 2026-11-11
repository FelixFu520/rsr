"""Sequential soft/rigid relay, with a single continuously integrated physical world."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from real2sim2real.planner.planner_sorting import SortingPlanner, SortingPlannerConfig
from real2sim2real.planner.sorting_stream_runtime import SortingStreamRuntime


class SortingStreamPlannerConfig(SortingPlannerConfig):
    """Two parcels, inserted one at a time, sharing unchanged native robot hardware."""

    backend: str = "sorting_stream"
    rigid_closed_gripper_rad: FiniteFloat = Field(default=0.40, ge=0, le=0.7)
    rigid_hold_gripper_rad: FiniteFloat = Field(default=0.42, ge=0, le=0.7)
    rigid_pad_bottom_m: FiniteFloat = Field(default=0.015, ge=0, le=0.08)
    sleep_soft_when_settled: bool = True
    rigid_contact_time_constant_s: FiniteFloat = Field(default=0.0005, gt=0)
    rigid_mujoco_iterations: int = Field(default=100, gt=0)
    rigid_mujoco_solver: Literal["newton", "pgs"] = "pgs"
    rigid_disable_warmstart: bool = False
    separate_ground_drop_lanes: bool = False
    ground_lane_margin_m: FiniteFloat = Field(default=0.012, ge=0)
    lower_drop_tcp_orientation_wxyz: (
        tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] | None
    ) = None

    @model_validator(mode="after")
    def validate_lower_drop_orientation(self) -> SortingStreamPlannerConfig:
        """Require a unit quaternion for the optional lower-lane tool attitude."""
        if self.lower_drop_tcp_orientation_wxyz is not None and not np.isclose(
            np.linalg.norm(self.lower_drop_tcp_orientation_wxyz), 1.0, atol=1e-6
        ):
            raise ValueError("Lower drop TCP orientation must be a unit quaternion")
        return self


def rigid_relay_config(config: SortingStreamPlannerConfig) -> SortingStreamPlannerConfig:
    """Keep rigid grasps independent of the soft parcel's infeed depth and offsets."""
    return config.model_copy(
        update={
            "closed_gripper_rad": config.rigid_closed_gripper_rad,
            "hold_gripper_rad": config.rigid_hold_gripper_rad,
            "grasp_pad_bottom_m": config.rigid_pad_bottom_m,
            "infeed_grasp_pad_bottom_m": None,
            "grasp_offset_xy_m": (0.0, 0.0),
            "regrasp_offset_xy_m": (0.0, 0.0),
            "regrasp_surface_indent_m": None,
            "regrasp_surface_indent_fraction": None,
            "strain_limited_grip": False,
        }
    )


class _CartonView:
    """Expose the active rigid parcel to the shared motion controller, without altering physics."""

    def __init__(self, runtime: SortingStreamRuntime) -> None:
        object.__setattr__(self, "shared", runtime)

    def __getattr__(self, name: str) -> Any:
        aliases = {
            "vertices": "carton_vertices",
            "last_contact_counts": "carton_contacts",
            "last_active_contact_shapes": "carton_contact_shapes",
            "last_contact_distances_m": "carton_contact_distances",
        }
        return getattr(self.shared, aliases.get(name, name))

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.shared, name, value)


class _CartonScene:
    def __init__(self, scene: Any, outfeed_target: np.ndarray) -> None:
        self.shared = scene
        self.config = scene.config.model_copy(update={"outfeed_target_xy": outfeed_target.tolist()})
        self.triangle_indices = scene.rigid_triangles
        self.garment_count = len(scene.rigid_vertices)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.shared, name)

    def label_up_cos(self, vertices: np.ndarray) -> float:
        triangles = self.shared.rigid_triangles
        rest = self.shared.rigid_vertices[triangles]
        normals = np.cross(rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0])
        faces = vertices[triangles[normals[:, 2] > 1e-8]]
        normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]).sum(0)
        return float(normal[2] / max(np.linalg.norm(normal), 1e-12))


class _RigidRelay(SortingPlanner):
    """Use the same native-arm path with rigid contact measurements and rigid acceptance."""

    def initialize(self, robot: Any, scene: Any, runtime: SortingStreamRuntime) -> None:
        target = np.asarray(scene.config.outfeed_target_xy, float).copy()
        self.predicted_ground_lane_clearance_m = None
        if self.config.separate_ground_drop_lanes:
            soft = runtime.vertices()
            low, high = scene.supports["outfeed"]
            half = scene.rigid_dimensions[1] / 2
            candidates = [
                low[1] + half + self.config.ground_lane_margin_m,
                high[1] - half - self.config.ground_lane_margin_m,
            ]
            if candidates[0] > candidates[1]:
                raise ValueError("No full-width rigid drop lane fits on the outfeed belt")
            scores = [
                max(float(soft[:, 1].min()) - (y + half), (y - half) - float(soft[:, 1].max()))
                for y in candidates
            ]
            selected = int(np.argmax(scores))
            target[1] = candidates[selected]
            self.predicted_ground_lane_clearance_m = scores[selected]
            if selected == 0 and self.config.lower_drop_tcp_orientation_wxyz is not None:
                self.config = self.config.model_copy(
                    update={"tcp_orientation_wxyz": self.config.lower_drop_tcp_orientation_wxyz}
                )
        orientation_changed = not np.allclose(
            runtime.config.tcp_orientation_wxyz, self.config.tcp_orientation_wxyz
        )
        runtime.config.tcp_orientation_wxyz = self.config.tcp_orientation_wxyz
        if orientation_changed:
            runtime._calibrate_pad_bottom()
        self.outfeed_target = target
        self.robot, self.scene, self.runtime = (
            robot,
            _CartonScene(scene, target),
            _CartonView(runtime),
        )
        self.elapsed, self.rows = 0.0, []
        self.state.update(ready=True, done=False)
        self._initialize_tracking()
        triangles = scene.rigid_triangles
        edges = np.concatenate([triangles[:, :2], triangles[:, 1:], triangles[:, [2, 0]]])
        self._edge_indices = np.unique(np.sort(edges, axis=1), axis=0)
        self._edge_lengths = np.linalg.norm(
            self.rest[self._edge_indices[:, 0]] - self.rest[self._edge_indices[:, 1]], axis=1
        )

    def _coms(self, vertices: np.ndarray) -> np.ndarray:
        del vertices
        center = self.runtime.body_poses()[self.runtime.carton_body, :3]
        # Shared path code tracks two height channels for the soft parcel. Both
        # channels read the one measured rigid COM here; no duplicate body is built.
        return np.tile(center, (2, 1))

    def _measure_held_shape(self, vertices: np.ndarray, side: str) -> None:
        # Rigidity is measured from exact edge lengths every physics frame.
        pass

    def result(self) -> dict[str, Any]:
        checks = dict(
            sequence_completed=bool(self.state.get("done")),
            finite_state=bool(np.isfinite(self.samples).all()),
            rigid_geometry_preserved=abs(self.max_stretch - 1.0) < 1e-4,
            support_penetration_bounded=self.max_penetration
            <= self.config.maximum_table_penetration_m,
            finger_geometry_penetration_bounded=self.max_finger_geometry_penetration
            <= self.config.maximum_table_penetration_m,
            native_contact_penetration_bounded=self.runtime.maximum_carton_contact_penetration_m
            <= self.config.maximum_table_penetration_m,
            ik_targets_reachable=self.max_tcp_error < self.config.maximum_tcp_error_m,
            mimic_constraints_maintained=self.max_mimic_error < 0.03,
            conveyor_transport_observed=self.belt_displacement
            >= self.config.minimum_belt_displacement_m,
            parcel_exited_outfeed=self.belt_exited,
        )
        for side, leg in self.legs.items():
            checks.update(
                {
                    f"{side}_lifted": bool(np.all(leg["peak_lifts"] >= self.config.minimum_lift_m)),
                    f"{side}_bilateral_transfer": leg["transfer_frames"] > 0
                    and leg["contact_frames"] / max(leg["transfer_frames"], 1) >= 0.95,
                    f"{side}_released": bool(leg["released"]),
                    f"{side}_landed_on_destination": leg["footprint_supported"]
                    and leg["place_error"] <= self.config.maximum_place_error_m,
                }
            )
        checks = {key: bool(value) for key, value in checks.items()}
        return dict(
            success=all(checks.values()),
            checks=checks,
            failed_checks=[key for key, passed in checks.items() if not passed],
            failure_reason=self.failure_reason,
            belt_displacement_m=self.belt_displacement,
            maximum_contact_penetration_m=self.runtime.maximum_carton_contact_penetration_m,
            maximum_finger_geometry_penetration_m=self.max_finger_geometry_penetration,
            maximum_ik_target_error_m=self.max_tcp_error,
            actual_mass_kg=float(self.runtime.model.body_mass.numpy()[self.runtime.carton_body]),
            dimensions_m=self.scene.rigid_dimensions.tolist(),
            asset_sha256=hashlib.sha256(
                Path(self.scene._asset(self.scene.config.rigid_usd_path)).read_bytes()
            ).hexdigest(),
            rigid_scale=self.scene.config.rigid_scale,
            collision_geometry="measured_V10_carton_bounding_box",
            conveyor_model="contact_normal_limited_Coulomb_surface_velocity",
            native_solver_algorithm=self.config.rigid_mujoco_solver,
            native_warmstart_disabled=self.config.rigid_disable_warmstart,
            native_contact_time_constant_s=self.config.rigid_contact_time_constant_s,
            maximum_belt_force_n=self.runtime.maximum_belt_force_n,
            physics_duration_s=self.elapsed,
            attachments_used=False,
            outfeed_target_xy_m=self.outfeed_target.tolist(),
            predicted_ground_lane_clearance_m=self.predicted_ground_lane_clearance_m,
            tcp_orientation_wxyz=list(self.config.tcp_orientation_wxyz),
        )


class SortingStreamPlanner(SortingPlanner):
    """Run the successful soft relay, then spawn and relay one rigid parcel without reset."""

    config_cls = SortingStreamPlannerConfig
    runtime_cls = SortingStreamRuntime

    def begin_episode(self, *, register_stepper: bool = True) -> None:
        if not self.scene.config.rigid_usd_path or self.scene.config.infeed_travel_m <= 0:
            raise ValueError("The stream requires a V10 rigid asset and a positive infeed distance")
        if not self.config.continuous_outfeed:
            raise ValueError("The stream requires a continuously running outfeed belt")
        super().begin_episode(register_stepper=register_stepper)
        self.rigid: _RigidRelay | None = None
        self.soft_result: dict | None = None
        self.stream_samples: list[np.ndarray] = []
        self.stream_bodies: list[np.ndarray] = []
        self.stream_states: list[np.ndarray] = []
        self.stream_actions: list[np.ndarray] = []
        self.stream_rows: list[dict] = []
        self.carton_visibility: list[bool] = []
        self.belt_speeds: list[list[float]] = []
        self.belt_offsets: list[np.ndarray] = []
        self.soft_sleeping: list[bool] = []
        self.carton_spawn_time_s: float | None = None
        self.soft_native_frames = 0
        self.global_max_stretch = 1.0
        self.global_min_area = 1.0
        self.global_min_floor_z = float("inf")
        self.global_failure_reason = None

    @property
    def expected_duration_s(self) -> float:
        return 2 * super().expected_duration_s

    def on_physics_step(self, robot: Any) -> bool:
        time_s = len(self.stream_samples) * self.robot.sim_world.config.physics_dt
        vertices = self.runtime.vertices()
        self.stream_samples.append(vertices.astype(np.float32))
        lengths = np.linalg.norm(
            vertices[self._edge_indices[:, 0]] - vertices[self._edge_indices[:, 1]], axis=1
        )
        self.global_max_stretch = max(
            self.global_max_stretch, float((lengths / self._edge_lengths).max())
        )
        faces = vertices[self.scene.triangle_indices.reshape(-1, 3)]
        areas = np.linalg.norm(
            np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1
        )
        self.global_min_area = min(self.global_min_area, float((areas / self.rest_areas).min()))
        self.global_min_floor_z = min(self.global_min_floor_z, float(vertices[:, 2].min()))
        self.stream_bodies.append(self.runtime.body_poses().astype(np.float32))
        state, action = self.runtime.recording_coordinates()
        self.stream_states.append(state.copy())
        self.carton_visibility.append(self.runtime.carton_spawned)
        self.belt_offsets.append(self.runtime.belt_offsets.copy())
        self.soft_sleeping.append(self.runtime.soft_asleep)
        if self.rigid is None:
            completed = super().on_physics_step(robot)
            if self.rows:
                self.stream_rows.append(dict(self.rows[-1], parcel="soft", time_s=time_s))
            if completed and self.failure_reason is None and self.state.get("done"):
                self.soft_result = super().result()
                if self.soft_result["success"]:
                    self.soft_native_frames = len(self.runtime.native_mimic_residuals)
                    self.runtime.spawn_carton()
                    self.carton_spawn_time_s = time_s
                    config = rigid_relay_config(self.config)
                    self.rigid = _RigidRelay(config=config)
                    self.rigid.initialize(self.robot, self.scene, self.runtime)
                    self.state.update(done=False, phase="rigid_spawn")
                    completed = False
        else:
            completed = self.rigid.on_physics_step(robot)
            self.state.update(self.rigid.state)
            self.state["phase"] = "rigid_" + self.state.get("phase", "unknown")
            self.elapsed = time_s + (0 if completed else self.robot.sim_world.config.physics_dt)
            if self.rigid.rows:
                self.stream_rows.append(dict(self.rigid.rows[-1], parcel="rigid", time_s=time_s))
        _, action = self.runtime.recording_coordinates()
        self.stream_actions.append(action.copy())
        self.belt_speeds.append([self.runtime.infeed_speed, self.runtime.belt_speed])
        if (
            self.global_max_stretch >= self.config.maximum_edge_stretch
            or self.global_min_area <= self.config.minimum_triangle_area_ratio
            or self.global_min_floor_z
            < self.scene.config.floor_z_m - self.config.maximum_table_penetration_m
        ):
            self.global_failure_reason = (
                "Soft geometry or floor contact exceeded limits in the continuous stream"
            )
            self.state.update(done=False, phase="failed")
            completed = True
        return completed

    def write_diagnostics(self, directory: Path) -> None:
        native = self.runtime.native_mimic_residuals
        try:
            if self.soft_native_frames:
                self.runtime.native_mimic_residuals = native[: self.soft_native_frames]
            super().write_diagnostics(directory)
        finally:
            self.runtime.native_mimic_residuals = native
        np.savez_compressed(
            directory / "stream.npz",
            positions=np.asarray(self.stream_samples),
            body_poses=np.asarray(self.stream_bodies),
            times=np.arange(len(self.stream_samples)) * self.robot.sim_world.config.physics_dt,
            state=np.asarray(self.stream_states),
            action=np.asarray(self.stream_actions),
            carton_visible=np.asarray(self.carton_visibility),
            belt_speeds_mps=np.asarray(self.belt_speeds),
            belt_offsets_m=np.asarray(self.belt_offsets),
            soft_sleeping=np.asarray(self.soft_sleeping),
            native_mimic_residuals=(np.concatenate(native) if native else np.empty((0, 10))),
            carton_body=self.runtime.carton_body,
            carton_rest_vertices=self.scene.rigid_vertices,
            carton_triangles=self.scene.rigid_triangles,
        )
        (directory / "stream_trajectory.json").write_text(json.dumps(self.stream_rows) + "\n")
        if self.rigid is not None:
            (directory / "rigid_result.json").write_text(
                json.dumps(self.rigid.result(), indent=2) + "\n"
            )

    def result(self) -> dict[str, Any]:
        soft = self.soft_result if self.soft_result is not None else super().result()
        rigid = self.rigid.result() if self.rigid is not None else {"success": False}
        checks = dict(soft_parcel=soft["success"], rigid_parcel=rigid["success"])
        checks["outfeed_always_running"] = bool(
            self.belt_speeds
            and np.all(np.asarray(self.belt_speeds)[:, 1] == self.config.belt_speed_mps)
        )
        checks["stream_arrays_synchronized"] = (
            len(self.stream_samples)
            == len(self.stream_bodies)
            == len(self.stream_states)
            == len(self.stream_actions)
        )
        checks["finite_entire_stream"] = bool(
            np.isfinite(self.stream_samples).all() and np.isfinite(self.stream_bodies).all()
        )
        checks["soft_entire_stream_stretch_bounded"] = (
            self.global_max_stretch < self.config.maximum_edge_stretch
        )
        checks["soft_entire_stream_triangles_valid"] = (
            self.global_min_area > self.config.minimum_triangle_area_ratio
        )
        checks["soft_entire_stream_floor_clear"] = (
            self.global_min_floor_z
            >= self.scene.config.floor_z_m - self.config.maximum_table_penetration_m
        )
        capacity = self.runtime.cloth_solver.maximum_contacts.numpy()
        checks["no_contact_buffer_overflow"] = bool(
            max(capacity[:2]) < self.config.self_contact_buffer and capacity[2] < 16384
        )
        return dict(
            soft,
            success=all(checks.values()),
            workflow_success=all(checks.values()),
            strict_quality_success=bool(soft["strict_quality_success"] and all(checks.values())),
            checks=checks,
            failed_checks=[key for key, passed in checks.items() if not passed],
            parcels={"soft": soft, "rigid": rigid},
            sequence=["soft", "rigid"],
            carton_spawn_time_s=self.carton_spawn_time_s,
            continuous_world=True,
            physics_duration_s=self.elapsed,
            failure_reason=self.global_failure_reason
            or (self.rigid.failure_reason if self.rigid else self.failure_reason),
            maximum_edge_stretch=self.global_max_stretch,
            minimum_triangle_area_ratio=self.global_min_area,
            maximum_floor_penetration_m=max(
                0.0, self.scene.config.floor_z_m - self.global_min_floor_z
            ),
        )
