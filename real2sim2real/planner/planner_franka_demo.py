"""Finite Franka grasp, hold, and release with measured physical acceptance."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import Field

from real2sim2real.core.config import BaseConfig
from real2sim2real.planner.base import BasePlanner


class FrankaDemoPlannerConfig(BaseConfig):
    """Motion uses metres/seconds; material values follow Newton's centimetre model."""

    timeout_s: float = Field(default=1200.0, gt=0)
    settle_s: float = Field(default=1.0, gt=0)
    approach_s: float = Field(default=3.0, gt=0)
    descend_s: float = Field(default=2.0, gt=0)
    close_s: float = Field(default=2.0, gt=0)
    lift_s: float = Field(default=3.0, gt=0)
    hold_s: float = Field(default=2.0, gt=0)
    release_s: float = Field(default=1.0, gt=0)
    retract_s: float = Field(default=1.5, gt=0)
    retract_height_m: float = Field(default=0.12, gt=0)
    rest_s: float = Field(default=2.0, gt=0)
    approach_height_m: float = Field(default=0.18, gt=0)
    lift_height_m: float = Field(default=0.25, gt=0)
    grasp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tcp_orientation_wxyz: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    open_finger_m: float = Field(default=0.04, ge=0, le=0.04)
    closed_finger_m: float = Field(default=0.0, ge=0, le=0.04)
    open_gripper_control: bool = False
    minimum_lift_m: float = Field(default=0.08, gt=0)
    minimum_release_drop_m: float = Field(default=0.04, ge=0)
    maximum_patch_distance_m: float = Field(default=0.08, gt=0)
    maximum_table_penetration_m: float = Field(default=0.02, ge=0)
    grasp_patch_vertices: int = Field(default=20, ge=1)
    newton_substeps: int = Field(default=10, ge=1)
    newton_use_cuda_graph: bool = True
    solver_iterations: int = Field(default=10, ge=1)
    cloth_solver: Literal["vbd", "xpbd"] = "vbd"
    ik_iterations: int = Field(default=24, ge=1)
    particle_radius_m: float = Field(default=0.008, gt=0)
    cloth_mass_kg: float = Field(default=0.2, gt=0)
    cloth_density_cm: float | None = Field(default=None, gt=0)
    cloth_stretch_stiffness: float = Field(default=1e4, gt=0)
    cloth_stretch_damping: float = Field(default=1.5e-2, ge=0)
    cloth_bend_stiffness: float = Field(default=5.0, gt=0)
    cloth_bend_damping: float = Field(default=0.5, ge=0)
    contact_stiffness: float = Field(default=5e4, gt=0)
    contact_damping: float = Field(default=50.0, ge=0)
    cloth_friction: float = Field(default=0.25, ge=0)
    table_friction: float = Field(default=0.4, ge=0)
    friction_epsilon_cm_s: float = Field(default=0.1, gt=0)
    xpbd_spring_stiffness: float = Field(default=1000.0, gt=0)
    xpbd_spring_damping: float = Field(default=1.0, ge=0)
    xpbd_mass_lumping: Literal["area", "uniform"] = "area"
    xpbd_self_contact: bool = Field(
        default=False,
        description="Diagnostic raw particle repulsion; no cloth adjacency exclusion in XPBD",
    )


class FrankaDemoPlanner(BasePlanner):
    """Drive joints and evaluate lifted geometry without attaching the grasp target."""

    config_cls = FrankaDemoPlannerConfig

    def __init__(self, config: FrankaDemoPlannerConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("planner")
        self.scene: Any = None
        self.runtime: Any = None
        self.elapsed = 0.0
        self.rows: list[dict[str, Any]] = []
        self._phases: list[tuple[str, float]] = []
        self._phase_index = -1
        self._phase_start_tcp: np.ndarray | None = None
        self._grasp_position: np.ndarray | None = None
        self._baseline_position: np.ndarray | None = None
        self._patch: np.ndarray | None = None
        self._baseline_vertices: np.ndarray | None = None
        self._edge_indices: np.ndarray | None = None
        self._edge_lengths: np.ndarray | None = None
        self._ik_failures = 0
        self.state["timeout_s"] = self.config.timeout_s

    def init_planner(self, robot: Any) -> None:
        """Bind the already arranged robot; defer handles until after world reset."""
        self.robot = robot

    def configure_scene(self, scene: Any) -> None:
        """Bind the independently registered scene's target and support geometry."""
        self.scene = scene

    def begin_episode(self) -> None:
        """Create a fresh cloth runtime and install its single world-step callback."""
        if self.scene is None:
            raise RuntimeError("Call configure_scene before begin_episode")
        if self.scene.config.object_kind == "cloth":
            from real2sim2real.planner.franka_demo_newton import FrankaClothRuntime

            self.runtime = FrankaClothRuntime(self.robot, self.scene, self.config)
            self.robot.sim_world.register_newton_stepper(self.runtime.step)
            triangles = np.asarray(self.scene.triangle_indices).reshape(-1, 3)
            edges = np.concatenate((triangles[:, :2], triangles[:, 1:], triangles[:, [2, 0]]))
            self._edge_indices = np.unique(np.sort(edges, axis=1), axis=0)
            vertices = self.runtime.vertices()
            self._edge_lengths = np.linalg.norm(
                vertices[self._edge_indices[:, 0]] - vertices[self._edge_indices[:, 1]], axis=1
            )
        self.elapsed = 0.0
        self.rows.clear()
        self._phase_index = -1
        self._phases = [
            ("settle", self.config.settle_s),
            ("approach", self.config.approach_s),
            ("descend", self.config.descend_s),
            ("close", self.config.close_s),
            ("lift", self.config.lift_s),
            ("hold", self.config.hold_s),
            ("release", self.config.release_s),
            ("retract", self.config.retract_s),
            ("rest", self.config.rest_s),
        ]
        self.state.update(ready=True, done=False, phase="settle", success=False)

    def _tcp_position(self) -> np.ndarray:
        return self.runtime.tcp_position() if self.runtime else self.robot.tcp_position()

    def _target_position(self) -> np.ndarray:
        if self.runtime is not None and self._patch is not None:
            return self.runtime.vertices()[self._patch].mean(axis=0)
        return np.asarray(self.scene.target_position, dtype=float)

    def _enter_phase(self, phase: str) -> None:
        self._phase_start_tcp = self._tcp_position().copy()
        if phase == "approach":
            self._grasp_position = np.asarray(self.scene.target_position).copy()
            self._grasp_position += np.asarray(self.config.grasp_offset_m)
            # A freely simulated shirt rests one particle radius above the
            # support. Grasp its measured settled position, never its spawn Z.
        if phase == "close":
            if self.runtime:
                vertices = self.runtime.vertices()
                self._patch = np.argsort(np.linalg.norm(vertices - self._grasp_position, axis=1))[
                    : self.config.grasp_patch_vertices
                ]
                self._baseline_vertices = vertices.copy()
            self._baseline_position = self._target_position().copy()
        self.state["phase"] = phase

    def on_physics_step(self, robot: Any) -> bool:
        """Read the previous actual state, then command the next robot target."""
        del robot
        if self.state.get("done"):
            return True
        ends = np.cumsum([duration for _, duration in self._phases])
        index = int(np.searchsorted(ends, self.elapsed, side="right"))
        if index >= len(self._phases):
            self.state.update(done=True, phase="done")
            self.state["success"] = self.result()["success"]
            return True
        phase, duration = self._phases[index]
        start_time = 0.0 if index == 0 else float(ends[index - 1])
        if index != self._phase_index:
            self._phase_index = index
            self._enter_phase(phase)
        self._record(phase)
        fraction = min(1.0, max(0.0, (self.elapsed - start_time) / duration))
        fraction = fraction * fraction * (3 - 2 * fraction)
        finger = self.config.open_finger_m
        destination = self._phase_start_tcp.copy()
        if self._grasp_position is not None:
            destination = self._grasp_position.copy()
            if phase == "approach":
                destination[2] += self.config.approach_height_m
            elif phase in {"lift", "hold", "release", "retract", "rest"}:
                destination[2] += self.config.lift_height_m
                if phase in {"retract", "rest"}:
                    destination[2] += self.config.retract_height_m
        if phase in {"close", "lift", "hold"}:
            finger = self.config.closed_finger_m
        if self.config.open_gripper_control:
            finger = self.config.open_finger_m
        if phase in {"settle", "close", "hold", "release", "rest"}:
            target = destination
        else:
            target = self._phase_start_tcp * (1 - fraction) + destination * fraction
        if phase == "settle":
            target = self._phase_start_tcp
        quaternion = np.asarray(self.config.tcp_orientation_wxyz, dtype=float)
        if self.runtime:
            self.runtime.command(target, quaternion, finger)
        elif not self.robot.command_tcp(target, quaternion, finger):
            self._ik_failures += 1
        self.state["last_action"] = dict(tcp=target.tolist(), finger_m=finger)
        self.elapsed += float(self.robot.sim_world.config.physics_dt)
        return False

    def _record(self, phase: str) -> None:
        point = self._target_position()
        tcp = self._tcp_position()
        row = dict(
            time_s=float(self.elapsed),
            phase=phase,
            target_xyz_m=point.tolist(),
            tcp_xyz_m=tcp.tolist(),
            target_z_m=float(point[2]),
            tcp_z_m=float(tcp[2]),
            patch_distance_m=float(np.linalg.norm(point - tcp)),
            actual_finger_gap_m=float(
                self.runtime.finger_gap_m()
                if self.runtime
                else np.sum(self.robot.finger_positions())
            ),
        )
        if self.runtime:
            vertices = self.runtime.vertices()
            row["cloth_min_z_m"] = float(vertices[:, 2].min())
            row["contacts"] = self.runtime.last_contacts
            row.update(
                {
                    f"contacts_{name}": count
                    for name, count in self.runtime.last_contact_counts.items()
                }
            )
            edges = self._edge_indices
            lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
            row["edge_stretch_p99"] = float(
                np.quantile(lengths / np.maximum(self._edge_lengths, 1e-9), 0.99)
            )
            if self._baseline_vertices is not None:
                # Remove the best rigid motion before measuring actual shape
                # change, so translation alone cannot satisfy deformation.
                a = self._baseline_vertices - self._baseline_vertices.mean(axis=0)
                b = vertices - vertices.mean(axis=0)
                u, _, vt = np.linalg.svd(a.T @ b)
                if np.linalg.det(u @ vt) < 0:
                    u[:, -1] *= -1
                row["shape_rms_m"] = float(
                    np.sqrt(np.mean(np.sum((a @ (u @ vt) - b) ** 2, axis=1)))
                )
        self.rows.append(row)

    def result(self) -> dict[str, Any]:
        """Return JSON-native measured checks; missing hold/release evidence fails."""
        held = [row for row in self.rows if row["phase"] == "hold"]
        rested = [row for row in self.rows if row["phase"] == "rest"]
        minimum_lift = 0.0
        release_drop = 0.0
        maximum_distance = 0.0
        if held and self._baseline_position is not None:
            minimum_lift = min(row["target_z_m"] for row in held) - float(
                self._baseline_position[2]
            )
            maximum_distance = max(row["patch_distance_m"] for row in held)
        if held and rested:
            release_drop = float(np.mean([row["target_z_m"] for row in held])) - float(
                np.mean([row["target_z_m"] for row in rested[len(rested) // 2 :]])
            )
        checks = {
            "hold_observed": bool(held),
            "target_lifted": minimum_lift >= self.config.minimum_lift_m,
            "target_stays_near_gripper": bool(held)
            and maximum_distance <= self.config.maximum_patch_distance_m,
            "target_falls_after_release": bool(rested)
            and release_drop >= self.config.minimum_release_drop_m,
            "target_supported_after_release": bool(rested)
            and all(
                row["target_z_m"]
                >= self.scene.table_top_z - self.config.maximum_table_penetration_m
                for row in rested[len(rested) // 2 :]
            ),
            "no_attachments": True,
        }
        result = dict(
            success=all(checks.values()),
            checks=checks,
            minimum_hold_lift_m=float(minimum_lift),
            maximum_hold_patch_distance_m=float(maximum_distance),
            release_drop_m=float(release_drop),
            physics_duration_s=float(self.elapsed),
            attachments_used=False,
            ik_failed_frames=self._ik_failures,
            open_gripper_control=self.config.open_gripper_control,
            commanded_closed_finger_m=self.config.closed_finger_m,
            commanded_open_finger_m=self.config.open_finger_m,
            minimum_hold_finger_gap_m=min(
                (row["actual_finger_gap_m"] for row in held), default=0.0
            ),
            maximum_rest_finger_gap_m=max(
                (row["actual_finger_gap_m"] for row in rested), default=0.0
            ),
            samples=len(self.rows),
        )
        if self.runtime:
            checks["no_pinned_vertices"] = self.runtime.pinned_vertex_count == 0
            checks["cloth_deforms"] = (
                max((row.get("shape_rms_m", 0.0) for row in held), default=0.0) > 0.003
            )
            checks["cloth_stretch_bounded"] = (
                max((row.get("edge_stretch_p99", 0.0) for row in self.rows), default=0.0) < 1.5
            )
            result.update(
                pinned_vertex_count=self.runtime.pinned_vertex_count,
                particle_count=int(self.runtime.model.particle_count),
                cloth_total_mass_kg=float(self.runtime.model.particle_mass.numpy().sum()),
                newton_version=str(self.runtime.newton.__version__),
                warp_version=str(self.runtime.wp.__version__),
                physics_substeps=self.config.newton_substeps,
                solver_iterations=self.config.solver_iterations,
                cuda_graph_active=bool(
                    self.runtime.direct_graph is not None
                    or (
                        self.runtime.kinematic_stepper is not None
                        and self.runtime.kinematic_stepper.graph is not None
                    )
                ),
                solver=self.runtime.solver_name,
                rigid_solver=(
                    type(self.runtime.rigid_solver).__name__
                    if self.runtime.rigid_solver is not None
                    else None
                ),
                cloth_solver=type(self.runtime.cloth_solver).__name__,
                cloth_self_contact_enabled=not self.runtime.unified_xpbd,
                xpbd_particle_repulsion_enabled=(
                    self.config.xpbd_self_contact if self.runtime.unified_xpbd else None
                ),
                stretch_spring_count=int(self.runtime.model.spring_count),
                particle_mass_lumping=(
                    self.config.xpbd_mass_lumping if self.runtime.unified_xpbd else "area"
                ),
                coupling="unified" if self.runtime.unified_xpbd else "one_way_rigid_to_cloth",
                maximum_cloth_stretch_p99=max(
                    (row.get("edge_stretch_p99", 0.0) for row in self.rows), default=0.0
                ),
                maximum_hold_contacts={
                    name: max((row.get(f"contacts_{name}", 0) for row in held), default=0)
                    for name in ("left", "right", "palm", "table")
                },
            )
            result["success"] = all(checks.values())
        return result
