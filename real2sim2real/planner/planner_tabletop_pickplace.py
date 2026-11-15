"""Pick-and-place planner: cuMotion trajectory optimization (Isaac Sim built-in) + phase machine.

Migrated from Real2Sim2Real ``TabletopCuroboPlanner``. The source planned eight segments with
cuRobo ``MotionPlanner.plan_pose`` (hand pointing down, fixed orientation in the base frame):
approach_pick -> grasp -> close -> lift -> approach_place -> place -> open -> retreat. Here every
motion segment is planned with the cuMotion ``TrajectoryOptimizer`` of the
``isaacsim.robot_motion.cumotion`` extension (no extra pip package): the vertical and transfer
segments use a linear path constraint, all segments keep the hand orientation. Trajectories are
sampled at the physics rate and sent as joint position targets with velocity feed-forward.

Differences from the source (see docs/tabletop_pickplace.md):

- the object position comes from PhysX after the start hold, the place height from the
  measured table top, and the hand-object offset measured after closing is kept for placing;
- distractor objects are cuMotion obstacles (the source planned without world obstacles);
- a final rest phase holds still so the Job can check that the object is at rest.

Sections: 1. configuration, 2. planner (phase machine, cuMotion calls).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pydantic import Field, FiniteFloat

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import LoggerMixin
from real2sim2real.planner.base import BasePlanner
from real2sim2real.utils.tabletop_pickplace_utils import (
    MOTION_PHASES,
    PHASE_INDEX,
    PHASES,
    hand_targets,
    quat_to_matrix,
    sample_trajectory,
    world_to_base,
)

# ================================================================================================
# 1. Configuration
# ================================================================================================


class CumotionPickPlacePlannerConfig(BaseConfig):
    """``[planner]``; times in seconds, lengths in metres (source values converted from steps)."""

    backend: str = "cumotion_pickplace"
    cumotion_robot: str = Field(default="franka", description="cuMotion supported robot config")
    tool_frame: str = "panda_hand"
    hand_orientation_base_wxyz: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] = Field(
        default=(0.0, 1.0, 0.0, 0.0), description="hand pointing down (source _DOWN_QUAT_WXYZ)"
    )
    hand_above_object_m: FiniteFloat = Field(default=0.103, gt=0, description="hand above centre")
    approach_clearance_m: FiniteFloat = Field(default=0.12, gt=0)
    place_offset_m: tuple[FiniteFloat, FiniteFloat] = Field(
        default=(0.15, -0.15), description="object centre XY at release, table frame"
    )
    place_clearance_m: FiniteFloat = Field(default=0.0, ge=0, description="bottom above table")
    linear_phases: list[str] = Field(
        default_factory=lambda: ["descend", "lift", "transfer", "lower", "retreat"]
    )
    path_deviation_m: FiniteFloat = Field(
        default=0.005, gt=0, description="max hand deviation from the straight line (checked)"
    )
    linear_speed_scale: FiniteFloat = Field(
        default=0.5, gt=0, le=1, description="fraction of the joint velocity/acc/jerk limits"
    )
    time_scale: FiniteFloat = Field(
        default=1.0, ge=1.0, description=">1 slows the free (approach) trajectories"
    )
    avoid_distractors: bool = True
    obstacle_margin_m: FiniteFloat = Field(default=0.01, ge=0)
    table_obstacle_depth_m: FiniteFloat | None = Field(
        default=0.01,
        gt=0,
        description="table slab obstacle whose top is this far below the table top; None: off",
    )
    table_obstacle_thickness_m: FiniteFloat = Field(default=0.2, gt=0)
    start_s: FiniteFloat = Field(default=0.25, ge=0, description="hold before approaching")
    settle_s: FiniteFloat = Field(default=0.067, ge=0, description="source settle_steps=4")
    settle_before_gripper_s: FiniteFloat = Field(default=0.267, ge=0, description="source 16")
    close_s: FiniteFloat = Field(default=0.6, gt=0, description="source close_hold_steps=36")
    open_s: FiniteFloat = Field(default=0.2, gt=0, description="source open_hold_steps=12")
    rest_s: FiniteFloat = Field(default=1.0, ge=0, description="final hold for rest checks")

    def validate_phases(self) -> None:
        unknown = set(self.linear_phases) - set(MOTION_PHASES)
        if unknown:
            raise ValueError(f"linear_phases must be motion phases {MOTION_PHASES}: {unknown}")


# ================================================================================================
# 2. Planner
# ================================================================================================


class CumotionPickPlacePlanner(BasePlanner, LoggerMixin):
    """One pick-and-place per episode; ``on_physics_step`` commands the robot every step."""

    config_cls = CumotionPickPlacePlannerConfig

    def __init__(self, config: CumotionPickPlacePlannerConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config: CumotionPickPlacePlannerConfig = config or self.config_cls.from_settings(
            "planner"
        )
        self.config.validate_phases()
        self.dt = 1 / 120
        self.rows: list[dict[str, Any]] = []
        self._cu: Any = None
        self._obstacles: dict[str, Any] = {}
        self._last_deviation: float | None = None
        self._reset_episode()

    # --- lifecycle --------------------------------------------------------------------------------
    def init_planner(self, robot: Any) -> None:
        """Keep the robot reference; cuMotion is loaded in :meth:`build`."""
        self.robot = robot

    def build(self, scene: Any, robot: Any, dt: float) -> None:
        """Load cuMotion, add distractor and table obstacles, check the start configuration."""
        import cumotion
        from isaacsim.robot_motion.cumotion import load_cumotion_supported_robot

        from real2sim2real.robot.robot_tabletop_pickplace import ARM_JOINTS

        self.dt = float(dt)
        self._cu = cumotion
        self._robot_cfg = load_cumotion_supported_robot(self.config.cumotion_robot)
        if list(self._robot_cfg.controlled_joint_names) != list(ARM_JOINTS):
            raise RuntimeError(f"cuMotion joints {self._robot_cfg.controlled_joint_names}")
        description = self._robot_cfg.robot_description
        if self.config.tool_frame not in description.tool_frame_names():
            raise ValueError(f"cuMotion robot has no tool frame {self.config.tool_frame}")
        self._world = cumotion.create_world()
        self._view = self._world.add_world_view()
        base_pos, base_quat = robot.base_pose
        self._base = (np.asarray(base_pos, float), np.asarray(base_quat, float))
        boxes = []
        if self.config.avoid_distractors:
            margin = self.config.obstacle_margin_m
            for name, center, half in scene.distractor_boxes():
                boxes.append((name, center, 2 * np.asarray(half, float) + 2 * margin))
        if self.config.table_obstacle_depth_m is not None:
            lo, hi = (np.asarray(b, float) for b in scene.layout["table_bounds_m"])
            thickness = self.config.table_obstacle_thickness_m
            top = scene.layout["table_top_z"] - self.config.table_obstacle_depth_m
            center = np.array([*(lo[:2] + hi[:2]) / 2, top - thickness / 2])
            boxes.append(("table", center, np.array([*(hi[:2] - lo[:2]), thickness])))
        obstacles = []
        self._obstacles: dict[str, Any] = {}
        for name, center, sides in boxes:
            obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
            obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, sides)
            self._obstacles[name] = self._world.add_obstacle(obstacle)
            base = self._set_obstacle(name, center)
            obstacles.append(
                {"name": name, "base_m": base.round(4).tolist(), "sides_m": sides.tolist()}
            )
        self._view.update()
        self._kinematics = self._robot_cfg.kinematics
        self._generator = cumotion.create_cspace_trajectory_generator(self._kinematics)
        n = self._kinematics.num_cspace_coords()
        scale = self.config.linear_speed_scale
        for setter, limit in (
            (self._generator.set_velocity_limits, self._kinematics.cspace_coord_velocity_limit),
            (
                self._generator.set_acceleration_limits,
                self._kinematics.cspace_coord_acceleration_limit,
            ),
            (self._generator.set_jerk_limits, self._kinematics.cspace_coord_jerk_limit),
        ):
            setter(np.array([limit(i) * scale for i in range(n)]))
        self._optimizer_config = cumotion.create_default_trajectory_optimizer_config(
            robot_description=description,
            tool_frame_name=self.config.tool_frame,
            world_view=self._view,
        )
        self._inspector = cumotion.create_robot_world_inspector(description, self._view)
        q0 = np.asarray(robot.config.initial_joint_positions[:7], float)
        self._ready_q = q0.copy()
        if self._in_collision(q0):
            raise RuntimeError("Initial Franka configuration is in collision for cuMotion")
        self.state.update(
            ready=True, obstacles=obstacles, cumotion_robot=self.config.cumotion_robot
        )
        self.log.info("cuMotion 规划器就绪", obstacles=[o["name"] for o in obstacles])

    def begin_episode(self, scene: Any, robot: Any, place_xy: Any) -> None:
        """Start a new episode from the robot's initial configuration."""
        self._reset_episode()
        self.scene, self.robot = scene, robot
        layout = scene.layout
        target = layout["objects"][scene.config.target.name]
        half_z = float(target["half_extents_m"][2])
        z = layout["table_top_z"] + half_z + self.config.place_clearance_m
        self.place_center = np.array([*np.asarray(place_xy, float), z])
        q0 = np.asarray(robot.config.initial_joint_positions, float)
        self._q_cmd = q0[:7].copy()
        self._finger_open = float(robot.config.gripper_open_m)
        self._finger_closed = float(robot.config.gripper_closed_m)
        self._finger = self._finger_open

    def _reset_episode(self) -> None:
        self.phase = "start"
        self._samples: tuple[np.ndarray, np.ndarray] | None = None
        self._i = 0
        self._targets: dict[str, np.ndarray] = {}
        self.plans: list[dict[str, Any]] = []
        self.failure: str | None = None
        self.grasp_offset_m: np.ndarray | None = None
        self.pick_center: np.ndarray | None = None
        self.place_center: np.ndarray | None = None
        self.step = 0
        self.rows = []
        self.state.update(phase="start", phase_index=0, last_action=None, done=False)

    # --- per physics step -------------------------------------------------------------------------
    def on_physics_step(self, robot: Any) -> bool:
        """Command one physics step; True once the sequence is done or failed."""
        if self.phase in ("done", "failed"):
            return True
        if self._samples is None and not self._enter(robot):
            return True
        positions, velocities = self._samples
        i = min(self._i, len(positions) - 1)
        robot.command(positions[i], self._finger, velocities[i])
        self.state["last_action"] = np.r_[positions[i], self._finger]
        self._q_cmd = positions[i].copy()
        self._i += 1
        self.step += 1
        if self._i >= len(positions):
            self._advance()
        return self.phase in ("done", "failed")

    def _advance(self) -> None:
        self.phase = PHASES[PHASE_INDEX[self.phase] + 1]
        self._samples, self._i = None, 0
        self.state.update(phase=self.phase, phase_index=PHASE_INDEX[self.phase])
        self.state["done"] = self.phase == "done"

    def _hold(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        steps = max(1, round(seconds / self.dt))
        return np.repeat(self._q_cmd[None], steps, 0), np.zeros((steps, 7))

    def _enter(self, robot: Any) -> bool:
        """Prepare the samples of the current phase; False when planning failed."""
        phase = self.phase
        self.rows.append({"phase": phase, "step": self.step})
        if phase == "done":
            return False
        if phase == "start":
            self._samples = self._hold(self.config.start_s)
            return True
        if phase in ("close", "open"):
            self._finger = self._finger_closed if phase == "close" else self._finger_open
            self._samples = self._hold(
                self.config.close_s if phase == "close" else self.config.open_s
            )
            return True
        if phase == "rest":
            self._samples = self._hold(self.config.rest_s)
            return True
        target = self._hand_target(phase, robot)
        plan = self._plan(phase, target)
        if plan is None:
            self.phase = "failed"
            self.state.update(phase="failed", done=True)
            return False
        positions, velocities = plan
        following = PHASES[PHASE_INDEX[phase] + 1]
        settle = (
            self.config.settle_before_gripper_s
            if following in ("close", "open")
            else self.config.settle_s
        )
        q_end = positions[-1]
        self._q_cmd = q_end
        hold_q, hold_qd = self._hold(settle)
        self._samples = (np.vstack([positions[1:], hold_q]), np.vstack([velocities[1:], hold_qd]))
        return True

    def _hand_target(self, phase: str, robot: Any) -> np.ndarray:
        """World hand target of a motion phase from measured object / hand poses."""
        cfg = self.config
        if phase == "approach_pick":
            # Distractors settled during the start hold; plan around their measured poses.
            if self._obstacles:
                for name, center, _ in self.scene.distractor_boxes(measured=True):
                    if name in self._obstacles:
                        self._set_obstacle(name, center)
                self._view.update()
            pick = self.scene.target_center()
            self.pick_center = pick
            self._targets = hand_targets(
                pick,
                self.place_center,
                hand_above_m=cfg.hand_above_object_m,
                clearance_m=cfg.approach_clearance_m,
            )
        if phase == "lift":
            # Keep the hand-object offset measured after closing for the place targets.
            hand = robot.link_poses()[0][0]
            self.grasp_offset_m = hand - self.scene.target_center()
            up = np.array([0.0, 0.0, cfg.approach_clearance_m])
            place_hand = self.place_center + self.grasp_offset_m
            self._targets.update(
                transfer=place_hand + up, lower=place_hand, retreat=place_hand + up
            )
        return self._targets[phase]

    def _plan(self, phase: str, target_world: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        target_base = world_to_base(*self._base, target_world)
        started = time.monotonic()
        self._last_deviation = None
        plan, status = self._solve(self._q_cmd, target_base, phase in self.config.linear_phases)
        record = {
            "phase": phase,
            "status": status,
            "target_world_m": np.round(target_world, 5).tolist(),
            "target_base_m": np.round(target_base, 5).tolist(),
            "planning_wall_s": round(time.monotonic() - started, 3),
        }
        if getattr(self, "_last_deviation", None) is not None:
            record["line_deviation_m"] = round(self._last_deviation, 5)
        if plan is not None:
            record["duration_s"] = round((len(plan[0]) - 1) * self.dt, 4)
            record["max_joint_speed_rad_s"] = float(np.abs(plan[1]).max())
        self.plans.append(record)
        if plan is None:
            self.failure = f"{phase}: {status}"
            self.log.error("cuMotion 规划失败", **record)
        else:
            self.log.info("cuMotion 规划", **record)
        return plan

    def _solve(
        self, q0: np.ndarray, target_base: np.ndarray, linear: bool
    ) -> tuple[tuple[np.ndarray, np.ndarray] | None, str]:
        """Joint trajectory to a hand position (base frame) at the fixed hand orientation."""
        if linear:
            return self._solve_linear(q0, target_base)
        return self._solve_free(q0, target_base)

    def _solve_free(
        self, q0: np.ndarray, target_base: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray] | None, str]:
        """IK seeded from ``q0`` (keeps the arm branch), then a collision-aware c-space plan."""
        cu = self._cu
        goal_q, status = self._ik(target_base, [q0, self._ready_q])
        if goal_q is None:
            return None, status
        optimizer_cls = cu.TrajectoryOptimizer
        goal = optimizer_cls.CSpaceTarget(
            goal_q,
            translation_path_constraint=optimizer_cls.CSpaceTarget.TranslationPathConstraint.none(),
            orientation_path_constraint=optimizer_cls.CSpaceTarget.OrientationPathConstraint.none(),
        )
        optimizer = cu.create_trajectory_optimizer(self._optimizer_config)
        self._view.update()
        result = optimizer.plan_to_cspace_target(np.asarray(q0, float), goal)
        status = result.status()
        if status != optimizer_cls.Results.Status.SUCCESS:
            return None, status.name
        trajectory = result.trajectory()
        domain = trajectory.domain()
        scale = self.config.time_scale
        positions = sample_trajectory(trajectory.eval, domain.lower, domain.span(), self.dt, scale)
        velocities = sample_trajectory(
            lambda t: trajectory.eval(t, 1), domain.lower, domain.span(), self.dt, scale
        )
        return (positions, velocities / scale), status.name

    def _solve_linear(
        self, q0: np.ndarray, target_base: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray] | None, str]:
        """Straight hand line: cuMotion task-space path -> c-space waypoints -> timed spline."""
        cu = self._cu
        rotation = cu.Rotation3(*self.config.hand_orientation_base_wxyz)
        start = self._kinematics.pose(np.asarray(q0, float), self.config.tool_frame)
        spec = cu.create_task_space_path_spec(start)
        if not spec.add_linear_path(cu.Pose3(translation=target_base, rotation=rotation)):
            return None, "INVALID_LINEAR_PATH"
        ik = cu.IkConfig()
        ik.cspace_seeds = [np.asarray(q0, float)]
        path = cu.convert_task_space_path_spec_to_cspace(
            spec, self._kinematics, self.config.tool_frame, cu.TaskSpacePathConversionConfig(), ik
        )
        if path is None or len(path.waypoints()) < 2:
            return None, "PATH_CONVERSION_FAILURE"
        trajectory = self._generator.generate_trajectory(list(path.waypoints()))
        if trajectory is None:
            return None, "TIME_PARAMETERIZATION_FAILURE"
        domain = trajectory.domain()
        positions = sample_trajectory(trajectory.eval, domain.lower, domain.span(), self.dt)
        velocities = sample_trajectory(
            lambda t: trajectory.eval(t, 1), domain.lower, domain.span(), self.dt
        )
        hand = np.array([self._kinematics.position(q, self.config.tool_frame) for q in positions])
        a, b = hand[0], np.asarray(target_base, float)
        axis = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        offset = hand - a
        deviation = float(np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1).max())
        self._last_deviation = deviation
        if deviation > self.config.path_deviation_m or float(np.linalg.norm(hand[-1] - b)) > 1e-3:
            return None, f"LINE_DEVIATION_{deviation * 1000:.1f}MM"
        if any(self._in_collision(q) for q in positions[:: max(1, len(positions) // 20)]):
            return None, "COLLISION_ON_LINEAR_PATH"
        return (positions, velocities), "SUCCESS"

    def _ik(self, target_base: np.ndarray, seeds: list[np.ndarray]) -> tuple[Any, str]:
        cu = self._cu
        rotation = cu.Rotation3(*self.config.hand_orientation_base_wxyz)
        config = cu.IkConfig()
        config.cspace_seeds = [np.asarray(q, float) for q in seeds]
        pose = cu.Pose3(translation=target_base, rotation=rotation)
        result = cu.solve_ik(self._kinematics, pose, self.config.tool_frame, config)
        if not result.success:
            return None, "INVERSE_KINEMATICS_FAILURE"
        q = np.asarray(result.cspace_position, float)
        if self._in_collision(q):
            return None, "IK_GOAL_IN_COLLISION"
        return q, "SUCCESS"

    def _in_collision(self, q: np.ndarray) -> bool:
        q = np.asarray(q, float)
        return bool(
            self._inspector.in_self_collision(q) or self._inspector.in_collision_with_obstacle(q)
        )

    def _set_obstacle(self, name: str, center_world: Any) -> np.ndarray:
        """Place a world-aligned box obstacle; returns its centre in the base frame."""
        pose = np.eye(4)
        pose[:3, :3] = quat_to_matrix(self._base[1]).T
        pose[:3, 3] = world_to_base(*self._base, center_world)
        self._world.set_pose(self._obstacles[name], self._cu.Pose3(pose))
        return pose[:3, 3]

    def result(self) -> dict[str, Any]:
        """Planner outcome of the current episode."""
        return {
            "sequence_completed": self.phase == "done",
            "phase": self.phase,
            "failure": self.failure,
            "plans": self.plans,
            "pick_center_m": None if self.pick_center is None else self.pick_center.tolist(),
            "place_center_m": None if self.place_center is None else self.place_center.tolist(),
            "grasp_offset_m": None if self.grasp_offset_m is None else self.grasp_offset_m.tolist(),
            "time_scale": self.config.time_scale,
            "phase_steps": self.rows,
        }
