"""Two physically independent grasps: infeed → platform → outfeed, then belt transport."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from real2sim2real.core.logger import get_logger
from real2sim2real.planner.planner_package import PackagePlannerConfig
from real2sim2real.planner.planner_physic_solver import PickPlaceClothPlanner
from real2sim2real.planner.sorting_runtime import SortingRuntime
from real2sim2real.utils.package_geometry import check_package_containment
from real2sim2real.utils.sorting_quality import audit_held_shape, audit_jaw_penetration


def _motion_blend(progress: float, profile: Literal["cubic", "quintic"]) -> float:
    """Blend a bounded trajectory; quintic also has zero endpoint acceleration."""
    t = float(np.clip(progress, 0, 1))
    if profile == "quintic":
        return t**3 * (10 - 15 * t + 6 * t**2)
    return t * t * (3 - 2 * t)


@dataclass
class StrainGripController:
    """Relieve a position-controlled grip using measured deformation, with hysteresis."""

    target: float
    maximum_relief_rad: float
    rate_rad_s: float
    recovery_rate_rad_s: float | None = None
    hysteresis: float = 0.03
    relief_rad: float = 0.0
    peak_relief_rad: float = 0.0

    def command(self, nominal_rad: float, stretch: float, dt: float) -> float:
        """Relieve high strain and restore grip below the configured dead band."""
        if stretch > self.target:
            self.relief_rad += self.rate_rad_s * dt
        elif stretch < self.target - self.hysteresis:
            recovery = self.recovery_rate_rad_s
            self.relief_rad -= (0.1 * self.rate_rad_s if recovery is None else recovery) * dt
        self.relief_rad = float(np.clip(self.relief_rad, 0, self.maximum_relief_rad))
        self.peak_relief_rad = max(self.peak_relief_rad, self.relief_rad)
        return max(0.0, nominal_rad - self.relief_rad)


class SortingPlannerConfig(PackagePlannerConfig):
    """Reuse V7 material/contact settings, with separate relay and conveyor acceptance."""

    backend: str = "sorting_package"
    gripper_package_friction: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    tcp_orientation_wxyz: tuple[float, float, float, float] = (0, 2**-0.5, 2**-0.5, 0)
    mujoco_cpu: bool = True
    mujoco_control_per_frame: bool = True
    mujoco_iterations: int = Field(default=100, gt=0)
    mimic_time_constant_s: float = Field(default=0.0015, gt=0)
    mimic_impedance: float = Field(default=0.999, gt=0, lt=0.9999)
    open_gripper_rad: float = Field(default=0.0, ge=0, le=0.7)
    closed_gripper_rad: float = Field(default=0.64, gt=0, le=0.7)
    hold_gripper_rad: float = Field(default=0.675, gt=0, le=0.7)
    grasp_height_m: float = Field(default=0.048, ge=0, le=0.10)
    grasp_pad_bottom_m: float | None = Field(default=None, ge=0, le=0.08)
    grasp_offset_xy_m: tuple[FiniteFloat, FiniteFloat] = (0.0, 0.0)
    place_height_m: float = Field(default=0.20, gt=0)
    lift_height_m: float = Field(default=0.25, gt=0)
    belt_speed_mps: float = Field(default=0.35, gt=0)
    belt_transport_s: float = Field(default=6, gt=0)
    minimum_belt_displacement_m: float = Field(default=0.30, gt=0)
    maximum_edge_stretch: float = Field(default=1.3, gt=1)
    minimum_triangle_area_ratio: float = Field(default=0.03, gt=0, lt=1)
    maximum_shape_rms_m: float = Field(default=0.037413, gt=0)
    maximum_bag_gap_m: float = Field(default=0.01, gt=0)
    early_stop_on_shape_failure: bool = True
    maximum_tcp_error_m: float = Field(default=0.03, gt=0)
    return_s: float = Field(default=2.0, gt=0)
    transfer_profile: Literal["cubic", "quintic"] = "cubic"
    strain_limited_grip: bool = False
    grip_strain_target: float = Field(default=1.18, gt=1, lt=1.3)
    grip_maximum_relief_rad: float = Field(default=0.01, gt=0, le=0.05)
    grip_relief_rate_rad_s: float = Field(default=0.05, gt=0, le=0.5)
    grip_recovery_rate_rad_s: float = Field(default=0.005, gt=0, le=0.5)
    grip_strain_hysteresis: float = Field(default=0.03, gt=0, lt=0.3)
    hold_s: float = Field(default=0.0, ge=0)
    physics_origin_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.0, 0.0, 0.0)
    label_up_min_cos: float = Field(default=0.9, ge=0, le=1)

    @model_validator(mode="after")
    def _pair_friction(self) -> SortingPlannerConfig:
        if self.gripper_package_friction and self.cloth_friction <= 0:
            raise ValueError("Positive gripper_package_friction requires nonzero cloth_friction")
        return self


class SortingPlanner(PickPlaceClothPlanner):
    """Carry the unchanged two-mesh parcel through both arms without attachments."""

    config_cls = SortingPlannerConfig
    runtime_cls = SortingRuntime

    @property
    def expected_duration_s(self) -> float:
        """Return the finite full relay and conveyor duration."""
        return (
            2
            * sum(
                getattr(self.config, f"{phase}_s")
                for phase in (
                    "approach",
                    "descend",
                    "close",
                    "lift",
                    "hold",
                    "transfer",
                    "lower",
                    "release",
                    "retract",
                    "return",
                    "rest",
                )
            )
            + self.config.belt_transport_s
        )

    @property
    def action_names(self) -> list[str]:
        """Commands are native joint position targets in radians."""
        return [name.replace("_rad", "_target_rad") for name in self.robot.recording_joint_names]

    def begin_episode(self, *, register_stepper: bool = True) -> None:
        """Build one continuous physical model shared by both grasps and the conveyors."""
        if (
            self.robot.sim_world.config.newton_solver != "mujoco"
            or self.config.cloth_solver != "vbd"
        ):
            raise ValueError("Sorting requires MuJoCo robot + VBD parcel/contact")
        super().begin_episode(register_stepper=register_stepper)
        phases = (
            "approach",
            "descend",
            "close",
            "lift",
            "hold",
            "transfer",
            "lower",
            "release",
            "retract",
            "return",
            "rest",
        )
        self.schedule = [
            (side, phase, getattr(self.config, f"{phase}_s"))
            for side in ("left", "right")
            for phase in phases
            if getattr(self.config, f"{phase}_s") > 0
        ]
        self.schedule.append(("right", "conveyor", self.config.belt_transport_s))
        self.ends = np.cumsum([row[2] for row in self.schedule])
        self._index = -1
        self._frame = 0
        self.command_tcp: np.ndarray | None = None
        self.command_side: str | None = None
        self.samples, self.sample_times, self.joints, self.actions, self.hand_poses = (
            [],
            [],
            [],
            [],
            [],
        )
        self.checkpoint_times = [0.0]
        self.transfer_samples = []
        self.transfer_groups = {"left": [], "right": []}
        self.legs = {
            side: dict(
                peak_lifts=np.zeros(2),
                contact_frames=0,
                transfer_frames=0,
                hold_frames=0,
                hold_contact_frames=0,
                minimum_hold_lift=np.full(2, np.inf),
                minimum_transfer_lift=np.full(2, np.inf),
                released=False,
                release_drop=0.0,
                place_error=np.inf,
                minimum_height=np.inf,
                footprint_supported=False,
                label_up_cos=None,
            )
            for side in ("left", "right")
        }
        self.rest = self.runtime.vertices().copy()
        tri = self.rest[self.scene.triangle_indices.reshape(-1, 3)]
        self.rest_areas = np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
        self.max_stretch = 1.0
        self.min_area = 1.0
        self.max_penetration = self.max_tcp_error = self.max_mimic_error = 0.0
        self.max_contact_penetration = 0.0
        self.max_finger_geometry_penetration = 0.0
        self.online_held_shape_samples: list[dict[str, Any]] = []
        self._belt_start = None
        self.belt_displacement = 0.0
        self.failure_reason: str | None = None
        self.belt_exited = False
        self.articulation_samples: list[np.ndarray] = []
        self.body_samples: list[np.ndarray] = []
        self.grip_limiters = {
            side: StrainGripController(
                self.config.grip_strain_target,
                self.config.grip_maximum_relief_rad,
                self.config.grip_relief_rate_rad_s,
                self.config.grip_recovery_rate_rad_s,
                self.config.grip_strain_hysteresis,
            )
            for side in ("left", "right")
        }
        self.instantaneous_stretch = 1.0

    def _limited_gripper_angle(self, side: str, nominal_rad: float) -> float:
        if not self.config.strain_limited_grip:
            return nominal_rad
        return self.grip_limiters[side].command(
            nominal_rad, self.instantaneous_stretch, self.robot.sim_world.config.physics_dt
        )

    def _coms(self, vertices: np.ndarray) -> np.ndarray:
        n, mass = self.scene.garment_count, self.runtime._masses
        return np.array(
            [
                np.average(vertices[:n], axis=0, weights=mass[:n]),
                np.average(vertices[n:], axis=0, weights=mass[n:]),
            ]
        )

    def _grasp_tcp_height(self, side: str, angle: float) -> float:
        if self.config.grasp_pad_bottom_m is None:
            return self.config.grasp_height_m
        return self.runtime.pad_bottom_tcp_height(side, angle, self.config.grasp_pad_bottom_m)

    def _measure_held_shape(self, vertices: np.ndarray, side: str) -> None:
        """Retain strict measurements even when a diagnostic run continues past a failure."""
        held = audit_held_shape(
            vertices[None],
            self.rest,
            self.scene.triangle_indices.reshape(-1, 3),
            self.runtime._masses,
            self.scene.garment_count,
            {side: [0]},
            maximum_shape_rms_m=self.config.maximum_shape_rms_m,
            maximum_bag_gap_m=self.config.maximum_bag_gap_m,
            quality_reference=getattr(self.scene, "quality_reference", None),
        )
        self.online_held_shape_samples.append(
            dict(
                held["samples"][0],
                time_s=self.elapsed,
                sample_index=len(self.samples) - 1,
            )
        )
        get_logger(__name__).info(
            "Sorting held shape check",
            time_s=self.elapsed,
            measurements=held["samples"][0],
            checks=held["checks"],
        )
        if not all(held["checks"].values()) and self.config.early_stop_on_shape_failure:
            self.failure_reason = f"{side} held parcel exceeded V7 shape or bag-gap limits"

    def _enter(self, index: int, coms: np.ndarray) -> None:
        side, phase, _ = self.schedule[index]
        self._index = index
        self.checkpoint_times.append(self.elapsed)
        self.start_tcp = self.runtime.tcp_position(side)
        if phase != "approach" and self.command_side == side and self.command_tcp is not None:
            self.start_tcp = self.command_tcp.copy()
        if phase == "approach":
            self.pick = coms[0].copy()
            self.pick[:2] += self.config.grasp_offset_xy_m
            self.support_z = float(
                self.scene.supports["infeed" if side == "left" else "platform"][1][2]
            )
            self.legs[side]["baseline"] = coms[:, 2].copy()
            low, high = self.scene.supports["platform" if side == "left" else "outfeed"]
            self.destination_bounds = low, high
            self.goal = (low + high) / 2
            if side == "right":
                self.goal[:2] = self.scene.config.outfeed_target_xy
            self.goal[2] = high[2]
        previous_target = getattr(self, "target", self.start_tcp).copy()
        self.target = self.start_tcp.copy()
        if phase == "approach":
            self.target = np.r_[self.pick[:2], self.support_z + 0.20]
            nominal_goal = self.goal.copy()
            nominal_goal[:2] += self.config.grasp_offset_xy_m
            self.max_tcp_error = max(
                self.max_tcp_error,
                self.runtime.prepare_approach(
                    side,
                    self.target,
                    [
                        self.target,
                        np.r_[
                            self.pick[:2],
                            self.support_z
                            + self._grasp_tcp_height(side, self.config.open_gripper_rad),
                        ],
                        np.r_[
                            self.pick[:2],
                            self.support_z
                            + self._grasp_tcp_height(side, self.config.closed_gripper_rad),
                        ],
                        np.r_[self.pick[:2], self.support_z + self.config.lift_height_m],
                        np.r_[nominal_goal[:2], self.support_z + self.config.lift_height_m],
                        nominal_goal + [0, 0, self.config.place_height_m],
                        nominal_goal + [0, 0, self.config.place_height_m + 0.18],
                        self.runtime.parking_positions[side],
                    ],
                ),
            )
            self.command_tcp = self.target.copy()
            self.command_side = side
        elif phase in ("descend", "close"):
            angle = (
                self.config.open_gripper_rad
                if phase == "descend"
                else self.config.closed_gripper_rad
            )
            self.target = np.r_[self.pick[:2], self.support_z + self._grasp_tcp_height(side, angle)]
        elif phase == "lift":
            self.target[2] = self.support_z + self.config.lift_height_m
        elif phase == "hold":
            self.target = previous_target
        elif phase == "transfer":
            self.target[:2] += self.goal[:2] - coms[0, :2]
        elif phase == "lower":
            self.target[:2] += self.goal[:2] - coms[0, :2]
            self.target[2] = self.goal[2] + self.config.place_height_m
        elif phase == "release":
            self.legs[side]["release_z"] = coms[:, 2].copy()
            self.release_start_angle = float(
                self.runtime.target_q[self.robot.gripper_indices[side]]
            )
        elif phase == "retract":
            self.target[2] += 0.18
        elif phase == "return":
            self.target = self.runtime.parking_positions[side].copy()
        elif phase == "conveyor":
            self._belt_start = coms[:, :2].copy()
            self.runtime.belt_speed = self.config.belt_speed_mps

    def on_physics_step(self, robot: Any) -> bool:
        """Measure actual states, then change only robot/belt control commands."""
        del robot
        vertices = self.runtime.vertices()
        coms = self._coms(vertices)
        index = int(np.searchsorted(self.ends, self.elapsed, side="right"))
        q, action = self.runtime.recording_coordinates()
        self.samples.append(vertices.astype(np.float32))
        self.sample_times.append(self.elapsed)
        self.joints.append(q)
        self.actions.append(action)
        body_poses = self.runtime.body_poses()
        self.hand_poses.append(body_poses[list(self.runtime.hand_bodies.values())])
        self.max_penetration = max(self.max_penetration, self.scene.support_penetration(vertices))
        lengths = np.linalg.norm(
            vertices[self._edge_indices[:, 0]] - vertices[self._edge_indices[:, 1]], axis=1
        )
        self.instantaneous_stretch = float(np.max(lengths / self._edge_lengths))
        self.max_stretch = max(self.max_stretch, self.instantaneous_stretch)
        tri = vertices[self.scene.triangle_indices.reshape(-1, 3)]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        self.min_area = min(self.min_area, float(np.min(area / self.rest_areas)))
        full_q = self.runtime.state_0.joint_q.numpy()
        self.articulation_samples.append(full_q)
        self.body_samples.append(body_poses)
        jaw = audit_jaw_penetration(
            vertices[None],
            self.scene.triangle_indices.reshape(-1, 3),
            self.body_samples[-1][None],
            self.runtime.jaw_colliders,
            tolerance_m=self.config.maximum_table_penetration_m,
        )
        self.max_finger_geometry_penetration = max(
            self.max_finger_geometry_penetration, jaw["maximum_penetration_m"]
        )
        for side, pairs in self.robot.mimic.items():
            master = full_q[self.robot.gripper_indices[side]]
            self.max_mimic_error = max(
                self.max_mimic_error, max(abs(full_q[i] - ratio * master) for i, ratio in pairs)
            )
        if index >= len(self.schedule):
            self.runtime.belt_speed = 0.0
            self.state.update(done=True, phase="done")
            return True
        if self.max_mimic_error >= 0.03:
            self.failure_reason = "Native gripper mimic constraints exceeded 0.03 rad"
        if self.max_stretch >= self.config.maximum_edge_stretch:
            self.failure_reason = "Surface stretch exceeded the fixed V7 limit"
        if self.max_penetration > self.config.maximum_table_penetration_m:
            self.failure_reason = "Parcel penetrated a support beyond the fixed tolerance"
        if self.max_finger_geometry_penetration > self.config.maximum_table_penetration_m:
            self.failure_reason = "Parcel penetrated native finger geometry beyond tolerance"
        if index != self._index and self.schedule[index][1] == "transfer":
            previous_side = self.schedule[index][0]
            if not np.all(self.legs[previous_side]["peak_lifts"] >= self.config.minimum_lift_m):
                self.failure_reason = f"{previous_side} arm did not lift both physical meshes"
        if (
            index != self._index
            and self.schedule[index][1] in {"approach", "conveyor"}
            and index > 0
        ):
            previous_side = "left" if self.schedule[index][1] == "approach" else "right"
            previous = self.legs[previous_side]
            if not (
                previous["released"]
                and previous["place_error"] <= self.config.maximum_place_error_m
                and previous["minimum_height"] < 0.07
                and previous["footprint_supported"]
                and previous["label_up_cos"] is not None
                and previous["label_up_cos"] >= self.config.label_up_min_cos
            ):
                self.failure_reason = (
                    f"{previous_side} arm did not release onto the destination support"
                )
        if self.failure_reason is not None:
            self.runtime.belt_speed = 0.0
            self.state.update(done=False, phase="failed")
            return True
        if index != self._index:
            self._enter(index, coms)
        side, phase, duration = self.schedule[index]
        leg = self.legs[side]
        leg["peak_lifts"] = np.maximum(leg["peak_lifts"], coms[:, 2] - leg["baseline"])
        counts = self.runtime.last_contact_counts
        distances = getattr(self.runtime, "last_contact_distances_m", {})
        self.max_contact_penetration = max(
            self.max_contact_penetration,
            max(
                (
                    -value
                    for name, value in distances.items()
                    if name in {"left_0", "left_1", "right_0", "right_1"} and value is not None
                ),
                default=0.0,
            ),
        )
        bilateral = all(counts.get(f"{side}_{i}", 0) > 0 for i in (0, 1))
        active_bodies = getattr(self.runtime, "last_active_contact_bodies", np.empty(0, dtype=int))
        robot_contacts = int(
            np.count_nonzero((active_bodies >= 0) & (active_bodies < self.runtime.robot_body_count))
        )
        if (
            phase == "lift"
            and robot_contacts == 0
            and self.runtime.pad_bottom_z(side, body_poses) > float(vertices[:, 2].max()) + 0.01
        ):
            self.failure_reason = f"{side} arm lost grasp below the rising native pads"
            self.runtime.belt_speed = 0.0
            self.state.update(done=False, phase="failed")
            return True
        if phase == "hold":
            leg["hold_frames"] += 1
            leg["hold_contact_frames"] += int(bilateral)
            leg["minimum_hold_lift"] = np.minimum(
                leg["minimum_hold_lift"], coms[:, 2] - leg["baseline"]
            )
        if phase == "transfer":
            leg["transfer_frames"] += 1
            leg["contact_frames"] += int(bilateral)
            leg["minimum_transfer_lift"] = np.minimum(
                leg["minimum_transfer_lift"], coms[:, 2] - leg["baseline"]
            )
            self.transfer_samples.append(len(self.samples) - 1)
            self.transfer_groups[side].append(len(self.samples) - 1)
            expected_frames = np.ceil(duration / self.robot.sim_world.config.physics_dt) + 1
            missed = leg["transfer_frames"] - leg["contact_frames"]
            # These conditions already make final acceptance impossible. Stop
            # promptly while preserving synchronized failure evidence.
            if missed > 0.05 * expected_frames:
                self.failure_reason = f"{side} arm lost bilateral grip during transfer"
            elif np.any(leg["minimum_transfer_lift"] < self.config.minimum_lift_m):
                self.failure_reason = f"{side} arm lost sustained lift during transfer"
            frames = max(1, int(np.ceil(duration / self.robot.sim_world.config.physics_dt - 1e-8)))
            if leg["transfer_frames"] in {1, (frames - 1) // 2 + 1, frames}:
                self._measure_held_shape(vertices, side)
            if self.failure_reason is not None:
                self.runtime.belt_speed = 0.0
                self.state.update(done=False, phase="failed")
                return True
        if "release_z" in leg and phase in ("release", "retract", "rest"):
            leg["release_drop"] = max(
                leg["release_drop"], float(np.min(leg["release_z"] - coms[:, 2]))
            )
        if phase == "rest":
            leg["place_error"] = float(np.max(np.linalg.norm(coms[:, :2] - self.goal[:2], axis=1)))
            leg["minimum_height"] = float(np.max(coms[:, 2] - self.goal[2]))
            leg["released"] = robot_contacts == 0
            low, high = self.destination_bounds
            leg["footprint_supported"] = bool(
                np.all((vertices[:, :2] >= low[:2]) & (vertices[:, :2] <= high[:2]))
            )
            leg["label_up_cos"] = self.scene.label_up_cos(vertices)
        start = 0.0 if index == 0 else self.ends[index - 1]
        blend = _motion_blend(
            (self.elapsed - start) / duration,
            self.config.transfer_profile if phase == "transfer" else "cubic",
        )
        grip_blend = float(np.clip((self.elapsed - start) / (duration * 0.7), 0, 1))
        grip_blend = grip_blend * grip_blend * (3 - 2 * grip_blend)
        target = self.start_tcp * (1 - blend) + self.target * blend
        if phase == "hold":
            target = self.target
        if phase == "approach":
            self.runtime.command_approach(side, blend)
        elif phase in {"close", "release"}:
            opening = self.config.open_gripper_rad
            angle = (
                opening + grip_blend * (self.config.closed_gripper_rad - opening)
                if phase == "close"
                else opening + (1 - grip_blend) * (self.release_start_angle - opening)
            )
            if phase == "close" and self.config.grasp_pad_bottom_m is not None:
                target = self.start_tcp.copy()
                target[2] = self.support_z + self._grasp_tcp_height(side, angle)
                error = self.runtime.command_arm(side, target, angle)
                self.command_tcp = target.copy()
                self.command_side = side
                self.max_tcp_error = max(self.max_tcp_error, error)
            else:
                self.runtime.set_gripper_angle(side, angle)
        elif phase not in {"conveyor", "rest", "hold"}:
            angle = self.config.open_gripper_rad
            if phase == "lift":
                angle = self.config.closed_gripper_rad + blend * (
                    self.config.hold_gripper_rad - self.config.closed_gripper_rad
                )
            elif phase in {"transfer", "lower"}:
                angle = self.config.hold_gripper_rad
            if phase in {"lift", "transfer", "lower"}:
                angle = self._limited_gripper_angle(side, angle)
            error = self.runtime.command_arm(side, target, angle)
            self.command_tcp = target.copy()
            self.command_side = side
            self.max_tcp_error = max(self.max_tcp_error, error)
        elif phase == "hold":
            self.runtime.set_gripper_angle(
                side, self._limited_gripper_angle(side, self.config.hold_gripper_rad)
            )
        elif phase == "conveyor":
            self.belt_displacement = float(np.min(self._belt_start[:, 0] - coms[:, 0]))
            low, high = self.scene.supports["outfeed"]
            self.belt_exited = bool(
                vertices[:, 0].max() < low[0]
                and np.all((coms[:, 1] >= low[1]) & (coms[:, 1] <= high[1]))
            )
            if self.belt_exited:
                self.runtime.belt_speed = 0.0
                self.state.update(done=True, phase="done")
                return True
        state, action = self.runtime.recording_coordinates()
        self.actions[-1] = action
        self.rows.append(
            dict(
                time_s=self.elapsed,
                phase=f"{side}_{phase}",
                garment_com_m=coms[0].tolist(),
                bag_com_m=coms[1].tolist(),
                tcp_m=self.runtime.tcp_position(side).tolist(),
                pad_bottom_m=self.runtime.pad_bottom_z(side, body_poses) - self.support_z,
                target_tcp_m=(
                    self.command_tcp.tolist()
                    if self.command_tcp is not None and phase != "approach"
                    else None
                ),
                contacts=counts,
                robot_contacts=robot_contacts,
                grip_relief_rad=self.grip_limiters[side].relief_rad,
                state=state.tolist(),
                action=action.tolist(),
            )
        )
        self.state.update(
            phase=f"{side}_{phase}",
            progress=dict(
                maximum_edge_stretch=self.max_stretch,
                instantaneous_edge_stretch=self.instantaneous_stretch,
                grip_relief_rad=self.grip_limiters[side].relief_rad,
                minimum_triangle_area_ratio=self.min_area,
                belt_displacement_m=self.belt_displacement,
                contacts=counts,
                robot_contacts=robot_contacts,
                com_lift_m=(coms[:, 2] - leg["baseline"]).tolist(),
                gripper_angle_rad=float(full_q[self.robot.gripper_indices[side]]),
                maximum_mimic_error_rad=float(self.max_mimic_error),
                maximum_support_penetration_m=self.max_penetration,
                maximum_finger_geometry_penetration_m=self.max_finger_geometry_penetration,
            ),
        )
        self._frame += 1
        self.elapsed = self._frame * self.robot.sim_world.config.physics_dt
        return False

    def write_diagnostics(self, directory: Path) -> None:
        """Save every physics-frame particle sample and frame-aligned native joint data."""
        collision_data = {}
        for index, jaw in enumerate(self.runtime.jaw_colliders):
            collision_data[f"jaw_vertices_{index}"] = jaw["vertices"]
            collision_data[f"jaw_triangles_{index}"] = jaw["triangles"]
        np.savez_compressed(
            directory / "particles.npz",
            positions=np.asarray(self.samples),
            times=np.asarray(self.sample_times),
            triangles=self.scene.triangle_indices.reshape(-1, 3),
            rest=self.rest,
            masses=self.runtime._masses,
            garment_count=self.scene.garment_count,
            quality_reference_json=json.dumps(getattr(self.scene, "quality_reference", None)),
            checkpoint_times_s=sorted(set([*self.checkpoint_times, self.sample_times[-1]])),
            state=np.asarray(self.joints),
            action=np.asarray(self.actions),
            hand_poses=np.asarray(self.hand_poses),
            transfer_indices=np.asarray(self.transfer_samples),
            left_transfer_indices=np.asarray(self.transfer_groups["left"]),
            right_transfer_indices=np.asarray(self.transfer_groups["right"]),
            support_bounds=np.array(list(self.scene.supports.values())),
            label_triangles=self.scene.label_triangles,
            articulation_q=np.asarray(self.articulation_samples),
            body_poses=np.asarray(self.body_samples),
            jaw_body_ids=[jaw["body_id"] for jaw in self.runtime.jaw_colliders],
            shape_material_mu=self.runtime.model.shape_material_mu.numpy(),
            shape_body=self.runtime.model.shape_body.numpy(),
            soft_contact_mu=self.runtime.model.soft_contact_mu,
            belt_body=self.runtime.belt_body,
            mimic_pairs=[
                (index, self.robot.gripper_indices[side], ratio)
                for side, pairs in self.robot.mimic.items()
                for index, ratio in pairs
            ],
            **collision_data,
        )

    def result(self) -> dict[str, Any]:
        """Require both grasps, the platform release, the second release and belt motion."""
        capacity = self.runtime.cloth_solver.maximum_contacts.numpy()
        containment = check_package_containment(
            np.asarray(self.samples),
            self.scene.triangle_indices.reshape(-1, 3),
            self.scene.garment_count,
            times=np.asarray(self.sample_times),
            checkpoint_times_s=sorted(set([*self.checkpoint_times, self.sample_times[-1]])),
        )
        checks = dict(
            sequence_completed=bool(self.state.get("done")),
            finite_state=bool(np.isfinite(self.samples).all()),
            garment_mass_matches_config=abs(
                float(self.runtime._masses[: self.scene.garment_count].sum())
                - self.scene.config.garment_mass_kg
            )
            < 1e-6,
            no_pinned_vertices=self.runtime.pinned_vertex_count == 0,
            surface_stretch_bounded=self.max_stretch < self.config.maximum_edge_stretch,
            no_collapsed_triangles=self.min_area > self.config.minimum_triangle_area_ratio,
            support_penetration_bounded=self.max_penetration
            <= self.config.maximum_table_penetration_m,
            garment_contained_at_checkpoints=containment["passed"],
            no_contact_buffer_overflow=bool(
                max(capacity[:2]) < self.config.self_contact_buffer and capacity[2] < 16384
            ),
            ik_targets_reachable=self.max_tcp_error < self.config.maximum_tcp_error_m,
            mimic_constraints_maintained=self.max_mimic_error < 0.03,
            conveyor_transport_observed=self.belt_displacement
            >= self.config.minimum_belt_displacement_m,
            parcel_exited_outfeed=self.belt_exited,
            sampled_finger_contact_penetration_bounded=self.max_contact_penetration
            <= self.config.maximum_table_penetration_m,
        )
        shape = audit_held_shape(
            np.asarray(self.samples),
            self.rest,
            self.scene.triangle_indices.reshape(-1, 3),
            self.runtime._masses,
            self.scene.garment_count,
            self.transfer_groups,
            maximum_shape_rms_m=self.config.maximum_shape_rms_m,
            maximum_bag_gap_m=self.config.maximum_bag_gap_m,
            quality_reference=getattr(self.scene, "quality_reference", None),
            extra_samples={
                side: [
                    row["sample_index"]
                    for row in self.online_held_shape_samples
                    if row["side"] == side
                ]
                for side in self.transfer_groups
            },
        )
        checks.update(shape["checks"])
        jaw_audit = audit_jaw_penetration(
            np.asarray(self.samples),
            self.scene.triangle_indices.reshape(-1, 3),
            np.asarray(self.body_samples),
            self.runtime.jaw_colliders,
            tolerance_m=self.config.maximum_table_penetration_m,
        )
        checks["all_saved_vertices_and_face_centres_clear_of_fingers"] = jaw_audit["passed"]
        legs = {}
        for side, leg in self.legs.items():
            fraction = leg["contact_frames"] / max(1, leg["transfer_frames"])
            checks.update(
                {
                    f"{side}_both_meshes_lifted": bool(
                        np.all(leg["peak_lifts"] >= self.config.minimum_lift_m)
                    ),
                    f"{side}_sustained_transfer_lift": bool(
                        np.all(leg["minimum_transfer_lift"] >= self.config.minimum_lift_m)
                        and leg["transfer_frames"] > 0
                    ),
                    f"{side}_bilateral_transfer": fraction >= 0.95,
                    f"{side}_pretransfer_hold": self.config.hold_s == 0
                    or (
                        leg["hold_frames"] > 0
                        and leg["hold_contact_frames"] / leg["hold_frames"] >= 0.95
                        and np.all(leg["minimum_hold_lift"] >= self.config.minimum_lift_m)
                    ),
                    f"{side}_released": leg["released"]
                    and leg["release_drop"] >= self.config.minimum_release_drop_m,
                    f"{side}_placed_on_destination": leg["place_error"]
                    <= self.config.maximum_place_error_m
                    and leg["minimum_height"] < 0.07
                    and leg["footprint_supported"],
                    f"{side}_label_up": leg["label_up_cos"] is not None
                    and leg["label_up_cos"] >= self.config.label_up_min_cos,
                }
            )
            legs[side] = {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in leg.items()
            }
            legs[side]["bilateral_transfer_fraction"] = fraction
            for key, value in legs[side].items():
                if isinstance(value, list):
                    legs[side][key] = [v if np.isfinite(v) else None for v in value]
                elif isinstance(value, float) and not np.isfinite(value):
                    legs[side][key] = None
        checks = {key: bool(value) for key, value in checks.items()}
        return dict(
            success=all(checks.values()),
            checks=checks,
            failed_checks=[key for key, passed in checks.items() if not passed],
            legs=legs,
            object="V7_two_mesh_package",
            robot_backend=self.robot.config.backend,
            robot_asset_sha256=hashlib.sha256(
                Path(self.robot.source_path).read_bytes()
            ).hexdigest(),
            gripper_geometry="authored_ProM6_Robotiq_with_30x7.5x70mm_pads",
            pad_width_m=self.robot.config.pad_width_m,
            garment_mass_kg=float(self.runtime._masses[: self.scene.garment_count].sum()),
            bag_mass_kg=float(self.runtime._masses[self.scene.garment_count :].sum()),
            target_garment_mass_kg=self.scene.config.garment_mass_kg,
            authored_garment_mass_kg=self.scene.authored_garment_mass_kg,
            garment_mass_scale=self.scene.garment_mass_scale,
            contact_friction=self.runtime.contact_friction,
            coupling="one_way_rigid_to_deformable",
            attachments_used=False,
            strain_limited_grip=self.config.strain_limited_grip,
            peak_grip_relief_rad={
                side: limiter.peak_relief_rad for side, limiter in self.grip_limiters.items()
            },
            pinned_vertex_count=self.runtime.pinned_vertex_count,
            maximum_edge_stretch=self.max_stretch,
            minimum_triangle_area_ratio=self.min_area,
            maximum_support_penetration_m=self.max_penetration,
            maximum_sampled_finger_contact_penetration_m=self.max_contact_penetration,
            maximum_ik_target_error_m=self.max_tcp_error,
            maximum_mimic_error_rad=float(self.max_mimic_error),
            belt_displacement_m=self.belt_displacement,
            maximum_substep_contacts=capacity.tolist(),
            containment=containment,
            held_shape_audit=shape,
            online_held_shape_samples=self.online_held_shape_samples,
            finger_geometry_audit=jaw_audit,
            physics_duration_s=self.elapsed,
            failure_reason=self.failure_reason,
            newton_version=self.runtime.newton.__version__,
            warp_version=self.runtime.wp.__version__,
            asset_sha256=hashlib.sha256(
                Path(self.scene._asset(self.scene.config.cloth_usd_path)).read_bytes()
            ).hexdigest(),
        )
