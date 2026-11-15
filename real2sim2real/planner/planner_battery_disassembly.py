"""Unscrew-and-uncover controller of the dual KR210 battery disassembly (PhysX).

Sections:

1. Configuration (heights, speeds, screw / gripper phases).
2. Nominal plan: every Cartesian segment is solved with numpy IK before the episode starts, so
   unreachable layouts fail before simulation. Each arm gets an ordered list of phases.
3. ``BatteryUnscrewPlanner``: per physics step it reads joint sensors and sends drive targets.
   Closed-loop parts use robot-side sensors only:

   - seek: the bit turns slowly under spring preload until the float stroke shows that the
     blades dropped into the cross recess;
   - unscrew: the spindle turns the screw through contacts; the arm rises with the measured
     float compression (compliant follow); the phase ends when the measured axial rise of the
     bit reaches the thread length;
   - the magnetic bit attracts the screw head (force law in ``battery_disassembly_utils``) until
     the release phase over the basket; the reaction acts on the spindle;
   - gripper fingers close force-limited; a fully closed finger means the cover was missed.

The screwdriver arm works first; the gripper arm starts when the screwdriver arm is home.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import LoggerMixin
from real2sim2real.planner.base import BasePlanner
from real2sim2real.utils.battery_disassembly_utils import (
    TimedPath,
    joint_blend,
    line_points,
    magnet_force,
    solve_ik,
    solve_line,
    tcp_pose,
    yaw_matrix,
)

# ================================================================================================
# 1. Configuration
# ================================================================================================


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScrewdriverPlanConfig(_Strict):
    """Screwdriver arm motions (heights in the scene frame, metres)."""

    screw_order: list[int] = Field(default_factory=lambda: [1, 2, 3, 4], min_length=1)
    transit_z_m: FiniteFloat = 1.15
    seek_speed_deg_s: FiniteFloat = Field(default=60.0, gt=0)
    seek_timeout_s: FiniteFloat = Field(default=4.0, gt=0)
    engaged_float_m: FiniteFloat = Field(
        default=-0.0012, description="float stroke above which the blades sit in the recess"
    )
    engaged_hold_s: FiniteFloat = Field(default=0.1, ge=0)
    unscrew_speed_rps: FiniteFloat = Field(default=3.0, gt=0)
    unscrew_timeout_s: FiniteFloat = Field(default=12.0, gt=0)
    follow_gain_1_s: FiniteFloat = Field(default=20.0, gt=0)
    follow_max_speed_mps: FiniteFloat = Field(default=0.02, gt=0)
    extra_turns: FiniteFloat = Field(default=0.5, ge=0)
    drop_below_basket_top_m: FiniteFloat = Field(default=0.10, ge=0)
    drop_offsets_m: list[tuple[FiniteFloat, FiniteFloat]] = Field(
        default_factory=lambda: [(-0.10, -0.06), (0.10, -0.06), (-0.10, 0.06), (0.10, 0.06)]
    )
    release_wait_s: FiniteFloat = Field(default=0.8, gt=0)


class GripperPlanConfig(_Strict):
    """Gripper arm motions."""

    grasp_root_above_flange_m: FiniteFloat = Field(
        default=0.266, gt=0, description="gripper root above the cover flange top while grasping"
    )
    approach_height_m: FiniteFloat = Field(default=0.20, gt=0)
    lift_height_m: FiniteFloat = Field(default=0.20, gt=0)
    place_clearance_m: FiniteFloat = Field(default=0.005, ge=0)
    retract_height_m: FiniteFloat = Field(default=0.25, gt=0)
    close_wait_s: FiniteFloat = Field(default=1.5, gt=0)
    open_wait_s: FiniteFloat = Field(default=1.0, gt=0)
    missed_margin_m: FiniteFloat = Field(default=0.002, gt=0)


class BatteryUnscrewPlannerConfig(BaseConfig):
    """Speeds and phases of both arms."""

    backend: str = "battery_unscrew"
    transit_speed_mps: FiniteFloat = Field(default=0.30, gt=0)
    approach_speed_mps: FiniteFloat = Field(default=0.06, gt=0)
    contact_speed_mps: FiniteFloat = Field(default=0.02, gt=0, description="last centimetres")
    lift_speed_mps: FiniteFloat = Field(default=0.12, gt=0, description="screw on the magnet")
    contact_distance_m: FiniteFloat = Field(default=0.04, gt=0)
    joint_speed_rad_s: FiniteFloat = Field(default=0.45, gt=0)
    min_segment_s: FiniteFloat = Field(default=0.6, gt=0)
    settle_s: FiniteFloat = Field(default=2.0, ge=0)
    max_episode_s: FiniteFloat = Field(default=240.0, gt=0)
    screwdriver: ScrewdriverPlanConfig = Field(default_factory=ScrewdriverPlanConfig)
    gripper: GripperPlanConfig = Field(default_factory=GripperPlanConfig)


# ================================================================================================
# 2. Nominal plan
# ================================================================================================


@dataclass
class Phase:
    """One step of an arm program."""

    name: str
    kind: str  # path | seek | unscrew | release | close | open | wait
    path: TimedPath | None = None
    screw: int = -1
    duration: float = 0.0
    follow: np.ndarray | None = None  # unscrew: IK samples of the vertical rise
    follow_step_m: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


def _seeds() -> list[np.ndarray]:
    return [
        np.array([a1, a2, a3, 0.0, a5, 0.0])
        for a1 in (-1.6, -1.0, 0.0, 1.0, 1.6)
        for a2, a3, a5 in ((0.5, -0.5, 1.4), (0.3, 0.2, 1.0), (0.7, -0.9, 1.6))
    ]


class ArmPlanner:
    """IK helpers bound to one arm's base and TCP."""

    def __init__(
        self,
        chain: Any,
        base: np.ndarray,
        tool: Any,
        config: BatteryUnscrewPlannerConfig,
        *,
        rotation: np.ndarray | None = None,
        axis: np.ndarray | None = None,
    ) -> None:
        self.chain, self.base, self.tool, self.config = chain, base, tool, config
        self.rotation, self.axis = rotation, axis

    def tcp(self, q: np.ndarray) -> np.ndarray:
        return tcp_pose(self.chain, self.base, self.tool, q)[:3, 3]

    def solve(self, point: Any, seed: np.ndarray | None = None) -> np.ndarray:
        """Best IK over a seed set (wrist pointing down), nearest to ``seed`` when given."""
        best, best_cost = None, math.inf
        seeds = ([seed] if seed is not None else []) + _seeds()
        for s in seeds:
            q, pe, ae = solve_ik(
                self.chain, self.base, self.tool, point, rotation=self.rotation, axis=self.axis,
                seed=s, iterations=400,
            )  # fmt: skip
            if pe > 5e-4 or ae > math.radians(0.3):
                continue
            cost = float(np.linalg.norm(q - (seed if seed is not None else np.zeros(6))))
            if cost < best_cost:
                best, best_cost = q, cost
            if seed is not None and cost < 0.5:
                break
        if best is None:
            raise RuntimeError(f"No IK solution for TCP {np.round(point, 4).tolist()}")
        return best

    def line(
        self, q0: np.ndarray, end: Any, speed: float, label: str
    ) -> tuple[TimedPath, np.ndarray]:
        start = self.tcp(q0)
        points = line_points(start, end, spacing=0.005)
        path = solve_line(
            self.chain, self.base, self.tool, points, seed=q0, rotation=self.rotation,
            axis=self.axis, label=label,
        )  # fmt: skip
        distance = float(np.linalg.norm(np.asarray(end) - start))
        duration = max(self.config.min_segment_s, 1.5 * distance / speed)
        return TimedPath(path, duration), path[-1]

    def blend(self, q0: np.ndarray, q1: np.ndarray) -> tuple[TimedPath, np.ndarray]:
        path = joint_blend(q0, q1)
        span = float(np.max(np.abs(q1 - q0)))
        duration = max(self.config.min_segment_s, 1.5 * span / self.config.joint_speed_rad_s)
        return TimedPath(path, duration), np.asarray(q1, dtype=float)


def plan_screwdriver(arm: ArmPlanner, layout: dict[str, Any], screw_cfg: Any,
                     cfg: BatteryUnscrewPlannerConfig) -> list[Phase]:  # fmt: skip
    """Nominal phases of the screwdriver arm."""
    sd = cfg.screwdriver
    scene_z = float(layout["scene_matrix"][2, 3])
    transit = scene_z + sd.transit_z_m
    heads = layout["screw_heads"]
    lo, hi = layout["basket_inner_lo"], layout["basket_inner_hi"]
    basket_center = 0.5 * (lo + hi)
    basket_top = float(hi[2])
    drop_z = basket_top - sd.drop_below_basket_top_m
    if transit < basket_top + 0.06 or transit < float(np.max(heads[:, 2])) + 0.15:
        raise ValueError("transit_z_m must clear the basket and the screw heads")
    phases: list[Phase] = []
    q = np.zeros(6)
    for n, number in enumerate(sd.screw_order):
        i = number - 1
        head = heads[i]
        engage = np.array([head[0], head[1], head[2] - screw_cfg.recess_depth_m])
        above = np.array([head[0], head[1], transit])
        if n == 0:
            path, q = arm.blend(q, arm.solve(above))
        else:
            path, q = arm.line(q, above, cfg.transit_speed_mps, f"to_screw_{number}")
        phases.append(Phase(f"to_screw_{number}", "path", path, i))
        near = engage + np.array([0.0, 0.0, cfg.contact_distance_m])
        path, q = arm.line(q, near, cfg.transit_speed_mps, f"descend_{number}")
        phases.append(Phase(f"descend_{number}", "path", path, i))
        path, q = arm.line(q, engage, cfg.contact_speed_mps, f"approach_{number}")
        phases.append(Phase(f"approach_{number}", "path", path, i))
        phases.append(Phase(f"seek_{number}", "seek", screw=i, duration=sd.seek_timeout_s))
        rise = screw_cfg.thread_length_m + 0.01
        points = line_points(engage, engage + np.array([0, 0, rise]), spacing=0.001)
        follow = solve_line(arm.chain, arm.base, arm.tool, points, seed=q, axis=arm.axis,
                            label=f"unscrew_{number}")  # fmt: skip
        phases.append(Phase(f"unscrew_{number}", "unscrew", screw=i, duration=sd.unscrew_timeout_s,
                            follow=follow, follow_step_m=0.001))  # fmt: skip
        q = follow[int(round(screw_cfg.thread_length_m / 0.001))]
        path, q = arm.line(q, above, cfg.lift_speed_mps, f"lift_{number}")
        phases.append(Phase(f"lift_{number}", "path", path, i))
        dx, dy = sd.drop_offsets_m[n % len(sd.drop_offsets_m)]
        drop_xy = basket_center[:2] + np.array([dx, dy])
        over = np.array([*drop_xy, transit])
        path, q = arm.line(q, over, cfg.transit_speed_mps, f"to_basket_{number}")
        phases.append(Phase(f"to_basket_{number}", "path", path, i))
        path, q = arm.line(
            q, np.array([*drop_xy, drop_z]), cfg.transit_speed_mps, f"lower_{number}"
        )
        phases.append(Phase(f"lower_{number}", "path", path, i))
        phases.append(Phase(f"release_{number}", "release", screw=i, duration=sd.release_wait_s))
        path, q = arm.line(q, over, cfg.transit_speed_mps, f"raise_{number}")
        phases.append(Phase(f"raise_{number}", "path", path, i))
    path, q = arm.blend(q, np.zeros(6))
    phases.append(Phase("screw_home", "path", path))
    return phases


def grasp_targets(
    layout: dict[str, Any], cover_cfg: Any, grip: GripperPlanConfig
) -> dict[str, np.ndarray]:
    """Gripper root positions: above / at the grasp and at the pallet place pose."""
    scene = layout["scene_matrix"]
    com = np.asarray(layout["cover_com_world"], dtype=float)
    flange_top = (scene @ np.r_[0, 0, cover_cfg.flange_z_m[1], 1.0])[2]
    flange_bottom = (scene @ np.r_[0, 0, cover_cfg.flange_z_m[0], 1.0])[2]
    grasp = np.array([com[0], com[1], flange_top + grip.grasp_root_above_flange_m])
    pallet = np.asarray(layout["pallet_world"], dtype=float)
    drop = layout["pallet_top_z"] + grip.place_clearance_m - flange_bottom
    place = np.array([pallet[0], pallet[1], grasp[2] + drop])
    return {
        "above": grasp + np.array([0, 0, grip.approach_height_m]),
        "grasp": grasp,
        "lift": grasp + np.array([0, 0, grip.lift_height_m]),
        "over_place": np.array([place[0], place[1], grasp[2] + grip.lift_height_m]),
        "place": place,
        "retract": place + np.array([0, 0, grip.retract_height_m]),
    }


def plan_gripper(arm: ArmPlanner, layout: dict[str, Any], cover_cfg: Any,
                 cfg: BatteryUnscrewPlannerConfig) -> list[Phase]:  # fmt: skip
    """Nominal phases of the gripper arm."""
    g = cfg.gripper
    t = grasp_targets(layout, cover_cfg, g)
    phases = []
    path, q = arm.blend(np.zeros(6), arm.solve(t["above"]))
    phases.append(Phase("to_cover", "path", path))
    near = t["grasp"] + np.array([0.0, 0.0, cfg.contact_distance_m])
    path, q = arm.line(q, near, cfg.transit_speed_mps, "descend_cover")
    phases.append(Phase("descend_cover", "path", path))
    path, q = arm.line(q, t["grasp"], cfg.contact_speed_mps, "approach_cover")
    phases.append(Phase("approach_cover", "path", path))
    phases.append(Phase("close", "close", duration=g.close_wait_s))
    path, q = arm.line(q, t["lift"], cfg.approach_speed_mps, "lift_cover")
    phases.append(Phase("lift_cover", "path", path))
    path, q = arm.line(q, t["over_place"], cfg.transit_speed_mps, "transfer_cover")
    phases.append(Phase("transfer_cover", "path", path))
    near = t["place"] + np.array([0.0, 0.0, cfg.contact_distance_m])
    path, q = arm.line(q, near, cfg.transit_speed_mps, "lower_cover")
    phases.append(Phase("lower_cover", "path", path))
    path, q = arm.line(q, t["place"], cfg.contact_speed_mps, "set_cover")
    phases.append(Phase("set_cover", "path", path))
    phases.append(Phase("open", "open", duration=g.open_wait_s))
    path, q = arm.line(q, t["retract"], cfg.transit_speed_mps, "retract")
    phases.append(Phase("retract", "path", path))
    path, q = arm.blend(q, np.zeros(6))
    phases.append(Phase("grip_home", "path", path))
    return phases


# ================================================================================================
# 3. Planner
# ================================================================================================


class BatteryUnscrewPlanner(BasePlanner, LoggerMixin):
    """Runs the screwdriver program, then the gripper program, from joint sensors."""

    config_cls = BatteryUnscrewPlannerConfig

    def __init__(self, config: BatteryUnscrewPlannerConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config: BatteryUnscrewPlannerConfig = config or self.config_cls.from_settings(
            "planner"
        )
        self.rows: list[dict[str, Any]] = []
        self.programs: dict[str, list[Phase]] = {}

    @property
    def action_names(self) -> list[str]:
        arm = [f"{k}_arm.joint_a{i}_target_rad" for k in ("screw", "grip") for i in range(1, 7)]
        return arm + ["screwdriver.float_target_m", "screwdriver.spindle_velocity_target_rad_s",
                      "gripper.closing_target_m", "screwdriver.magnet_on"]  # fmt: skip

    def init_planner(self, robot: Any) -> None:
        """Bind the robot; the plan needs the arranged scene (``begin_episode``)."""
        self.robot = robot

    def build(self, scene: Any, robot: Any) -> None:
        """Solve the nominal programs (no Kit needed)."""
        self.scene, self.robot = scene, robot
        layout, cfg = scene.layout, self.config
        screw_arm = ArmPlanner(robot.chain, robot.bases["screw"], robot.tools["screw"], cfg,
                               axis=np.array([0.0, 0.0, -1.0]))  # fmt: skip
        best = None
        for yaw in (0.0, 180.0):
            grip_arm = ArmPlanner(robot.chain, robot.bases["grip"], robot.tools["grip"], cfg,
                                  rotation=yaw_matrix(yaw))  # fmt: skip
            try:
                program = plan_gripper(grip_arm, layout, scene.config.cover, cfg)
            except RuntimeError as exc:
                self.log.warning("夹爪朝向不可行", yaw=yaw, error=str(exc))
                continue
            span = sum(p.path.length for p in program if p.path is not None)
            if best is None or span < best[0]:
                best = (span, program, yaw)
        if best is None:
            raise RuntimeError("Gripper arm cannot reach the cover and the pallet")
        self.programs = {
            "screw": plan_screwdriver(screw_arm, layout, scene.config.screw, cfg),
            "grip": best[1],
        }
        self.arm_planners = {"screw": screw_arm}
        self.state["gripper_yaw_deg"] = best[2]
        self.state["ready"] = True
        self.log.info(
            "名义规划完成",
            screw_phases=len(self.programs["screw"]),
            grip_phases=len(self.programs["grip"]),
            gripper_yaw=best[2],
        )

    def begin_episode(self, scene: Any, robot: Any) -> None:
        """Reset the program counters (call after the runtime views exist)."""
        if not self.programs:
            self.build(scene, robot)
        self.active = {"screw": 0, "grip": 0}
        self.phase_t = {"screw": 0.0, "grip": 0.0}
        self.entered = {"screw": False, "grip": False}
        self.hold = {"screw": np.zeros(6), "grip": np.zeros(6)}
        self.blend: dict[str, TimedPath | None] = {"screw": None, "grip": None}
        self.elapsed = 0.0
        self.settle = 0.0
        self.magnet = False
        self.magnet_screw = -1
        self.float_target = self.robot.config.screwdriver.float_preload_extension_m
        self.spindle_velocity = 0.0
        self.closing = 0.0
        self.loosen_sign = 0.0
        self.seek_engaged_s = 0.0
        self.follow_s = 0.0
        self.max_rise = 0.0
        self.spindle_start = 0.0
        self.magnet_force = np.zeros(3)
        self.rows.clear()
        self.events: dict[str, Any] = {"failure": None}
        self.state.update(done=False, phase="start", phase_index=0, last_action=self._action())

    # --- per step ------------------------------------------------------------------------------
    def on_physics_step(self, robot: Any) -> bool:
        """Advance both programs by one physics step; True when the episode ended."""
        dt = robot.sim_world.config.physics_dt
        sensors = robot.read_state()
        if self.events["failure"] is not None:
            return True
        if self.elapsed > self.config.max_episode_s:
            return self._fail(f"episode exceeded {self.config.max_episode_s} s")
        screw_done = self.active["screw"] >= len(self.programs["screw"])
        self._run("screw", sensors, dt)
        if screw_done:
            self._run("grip", sensors, dt)
        else:
            robot.set_arm_targets("grip", self.hold["grip"], np.zeros(6))
        robot.set_screwdriver(self.float_target, self.spindle_velocity)
        robot.set_fingers(self.closing)
        self._apply_magnet(robot)
        self.elapsed += dt
        self.state["last_action"] = self._action()
        self.state["phase"] = self._phase_name()
        self.state["phase_index"] = self._phase_index()
        if self.active["screw"] >= len(self.programs["screw"]) and self.active["grip"] >= len(
            self.programs["grip"]
        ):
            self.settle += dt
            if self.settle >= self.config.settle_s:
                self.state["done"] = True
                return True
        return self.events["failure"] is not None

    def _run(self, key: str, sensors: dict[str, Any], dt: float) -> None:
        program = self.programs[key]
        index = self.active[key]
        if index >= len(program):
            self.robot.set_arm_targets(key, self.hold[key], np.zeros(6))
            return
        phase = program[index]
        if not self.entered[key]:
            self._enter(key, phase, sensors)
        t = self.phase_t[key]
        finished = False
        q_target, qd_target = self.hold[key], np.zeros(6)
        if phase.kind == "path":
            blend = self.blend[key]
            if blend is not None and t < blend.duration:
                q_target, qd_target = blend.sample(t)
            else:
                offset = blend.duration if blend is not None else 0.0
                q_target, qd_target = phase.path.sample(t - offset)
                finished = t - offset >= phase.path.duration
        elif phase.kind == "seek":
            finished = self._seek(phase, sensors, dt)
        elif phase.kind == "unscrew":
            q_target, qd_target, finished = self._unscrew(phase, sensors, dt)
        elif phase.kind in {"release", "close", "open", "wait"}:
            finished = t >= phase.duration
            if finished and phase.kind == "close":
                self._check_grasp(sensors)
        self.robot.set_arm_targets(key, q_target, qd_target)
        self.hold[key] = np.asarray(q_target, dtype=float).copy()
        self.phase_t[key] = t + dt
        if finished:
            self._exit(key, phase, sensors)
            self.active[key] = index + 1
            self.phase_t[key] = 0.0
            self.entered[key] = False

    def _enter(self, key: str, phase: Phase, sensors: dict[str, Any]) -> None:
        self.entered[key] = True
        self.blend[key] = None
        if phase.kind == "path":
            start = phase.path.path[0]
            if np.max(np.abs(start - self.hold[key])) > 1e-5:
                span = float(np.max(np.abs(start - self.hold[key])))
                if span > 0.05:
                    self._fail(f"{phase.name}: path start is {span:.3f} rad from the arm")
                    return
                self.blend[key] = TimedPath(
                    joint_blend(self.hold[key], start, 0.002), max(0.3, span / 0.05)
                )
        if phase.name.startswith("descend_") and key == "screw":
            self.magnet = True
            self.magnet_screw = phase.screw
            self.float_target = self.robot.config.screwdriver.float_preload_extension_m
        if phase.kind == "seek":
            tip, axis = self.robot.tip_pose()
            # right-hand thread: counter-clockwise seen from above loosens
            self.loosen_sign = 1.0 if axis[2] > 0 else -1.0
            self.spindle_velocity = self.loosen_sign * math.radians(
                self.config.screwdriver.seek_speed_deg_s
            )
            self.seek_engaged_s = 0.0
        if phase.kind == "unscrew":
            self.follow_s = 0.0
            self.max_rise = 0.0
            self.spindle_start = sensors["spindle"]
            self.spindle_velocity = (
                self.loosen_sign * 2 * math.pi * self.config.screwdriver.unscrew_speed_rps
            )
        if phase.kind == "release":
            self.magnet = False
        if phase.kind == "close":
            self.closing = self.robot.config.gripper.stroke_m
        if phase.kind == "open":
            self.closing = 0.0
        self._event(key, phase, "enter", sensors)

    def _exit(self, key: str, phase: Phase, sensors: dict[str, Any]) -> None:
        if phase.kind in {"seek", "unscrew"}:
            self.spindle_velocity = 0.0
        if phase.kind == "unscrew":
            # the free screw hangs on the magnet: hold the float at nominal instead of preloading
            self.float_target = 0.0
        self._event(key, phase, "exit", sensors)

    def _seek(self, phase: Phase, sensors: dict[str, Any], dt: float) -> bool:
        if sensors["float"] >= self.config.screwdriver.engaged_float_m:
            self.seek_engaged_s += dt
        else:
            self.seek_engaged_s = 0.0
        if self.seek_engaged_s >= self.config.screwdriver.engaged_hold_s:
            return True
        if self.phase_t["screw"] > phase.duration:
            self._fail(
                f"{phase.name}: bit did not enter the recess (float {sensors['float']:.4f} m)"
            )
        return False

    def _unscrew(
        self, phase: Phase, sensors: dict[str, Any], dt: float
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        sd = self.config.screwdriver
        compression = max(0.0, -sensors["float"])
        speed = min(sd.follow_max_speed_mps, sd.follow_gain_1_s * compression)
        limit = (len(phase.follow) - 1) * phase.follow_step_m
        self.follow_s = min(limit, self.follow_s + speed * dt)
        x = self.follow_s / phase.follow_step_m
        i = min(int(x), len(phase.follow) - 2)
        a = x - i
        q = (1 - a) * phase.follow[i] + a * phase.follow[i + 1]
        qd = (phase.follow[i + 1] - phase.follow[i]) / phase.follow_step_m * speed
        thread = self.scene.config.screw.thread_length_m
        # bit rise from the engaged height; after the thread lets go the bit drops on the spring
        self.max_rise = max(self.max_rise, self.follow_s - sensors["float"])
        rise = self.max_rise
        turns = abs(sensors["spindle"] - self.spindle_start) / (2 * math.pi)
        needed = thread / self.scene.config.screw.pitch_m + sd.extra_turns
        done = rise >= thread - 5e-4 and turns >= needed
        if self.phase_t["screw"] > phase.duration:
            self._fail(
                f"{phase.name}: screw did not come out "
                f"(rise {rise * 1e3:.1f} mm, {turns:.1f} turns)"
            )
        phase.info.update(rise_m=rise, turns=turns)
        return q, qd, done

    def _check_grasp(self, sensors: dict[str, Any]) -> None:
        stroke = self.robot.config.gripper.stroke_m
        closed = max(abs(sensors["finger_p"]), abs(sensors["finger_n"]))
        if closed > stroke - self.config.gripper.missed_margin_m:
            self._fail(f"gripper closed to {closed * 1e3:.1f} mm: cover missed")

    def _apply_magnet(self, robot: Any) -> None:
        self.magnet_force = np.zeros(3)
        if not self.magnet or self.magnet_screw < 0:
            return
        tip, axis = robot.tip_pose()
        positions, orientations = self.scene.screw_poses()
        cfg = self.scene.config.screw
        from real2sim2real.utils.battery_disassembly_utils import quat_to_matrix

        # the bit tip seats on the recess floor; attraction pulls that floor onto the tip
        seat = (
            positions[self.magnet_screw]
            + quat_to_matrix(orientations[self.magnet_screw])[:, 2] * cfg.head_base_m
        )
        sd = robot.config.screwdriver
        force = magnet_force(
            tip,
            seat,
            max_force_n=sd.magnet_max_force_n,
            range_m=sd.magnet_range_m,
            contact_direction=-axis,
        )
        forces = np.zeros((len(positions), 3))
        forces[self.magnet_screw] = force
        self.scene.apply_screw_forces(forces)
        robot.apply_spindle_force(-force)
        self.magnet_force = force

    # --- bookkeeping ---------------------------------------------------------------------------
    def _fail(self, reason: str) -> bool:
        if self.events["failure"] is None:
            self.events["failure"] = reason
            self.log.error("规划失败", reason=reason, t=round(self.elapsed, 3))
        return True

    def _event(self, key: str, phase: Phase, what: str, sensors: dict[str, Any]) -> None:
        row = dict(
            time_s=round(self.elapsed, 5), arm=key, phase=phase.name, event=what,
            float_m=sensors["float"], spindle_rad=sensors["spindle"],
            finger_p_m=sensors["finger_p"], finger_n_m=sensors["finger_n"],
            **{k: float(v) for k, v in phase.info.items() if isinstance(v, (int, float))},
        )  # fmt: skip
        self.rows.append(row)
        if what == "enter":
            self.log.info("阶段", arm=key, phase=phase.name, t=round(self.elapsed, 2))

    def _phase_name(self) -> str:
        for key in ("screw", "grip"):
            if self.active[key] < len(self.programs[key]):
                return self.programs[key][self.active[key]].name
        return "done"

    def _phase_index(self) -> int:
        return self.active["screw"] + (
            self.active["grip"] if self.active["screw"] >= len(self.programs["screw"]) else 0
        )

    def _action(self) -> np.ndarray:
        return np.r_[
            self.hold["screw"], self.hold["grip"], self.float_target, self.spindle_velocity,
            self.closing, float(self.magnet),
        ]  # fmt: skip

    def result(self) -> dict[str, Any]:
        """Program outcome (physical acceptance is evaluated by the Job)."""
        events = getattr(self, "events", {"failure": "not started"})
        unscrew = [
            r for r in self.rows if r["phase"].startswith("unscrew_") and r["event"] == "exit"
        ]
        return dict(
            sequence_completed=bool(self.state.get("done")) and events["failure"] is None,
            failure=events["failure"],
            elapsed_s=getattr(self, "elapsed", 0.0),
            gripper_yaw_deg=self.state.get("gripper_yaw_deg"),
            unscrew=[
                dict(phase=r["phase"], rise_m=r.get("rise_m"), turns=r.get("turns"))
                for r in unscrew
            ],
        )
