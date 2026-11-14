"""HALO task planner: GEAR-SONIC whole-body tracking of a fixed reference motion at 50 Hz.

The reference (``qpos.npy`` + ``spec.json``) is only a target for the SONIC encoder; the robot
state always comes from PhysX. Per control frame the planner reads the measured base pose and
joints, runs the encoder/decoder, conditions the targets and sends them to the force-limited
motors. Optional conditioning, all from HALO:

- ``motor_smoothing`` (crawl_v3): causal 15 ms smoothing of arm targets, 6 rad/s, 150 rad/s²;
- ``seated_hold`` (sit_v1): once the measured pelvis is low and centred on the cushion and the
  seat carries load, blend the targets to the measured joint pose over 0.6 s and steer the
  waist so the torso stays upright facing ``facing_yaw_deg``; the base is never written;
- ``seated_target_filter_alpha`` (sit_v1): low-pass the targets in the ``seated`` phase until
  the hold starts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from real2sim2real.core.config import BaseConfig
from real2sim2real.planner.base import BasePlanner
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.halo_utils import (
    CONTROL_HZ,
    MUJOCO_JOINT_NAMES,
    MotorTargetFilter,
    SonicTracker,
    phase_index,
    resample_reference,
)


class MotorSmoothingConfig(BaseModel):
    """Target velocity/acceleration limits before PhysX (HALO native_motor_control.py)."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    arms_only: bool = True
    time_constant_s: float = Field(default=0.015, gt=0)
    arm_max_speed_rad_s: float = Field(default=6.0, gt=0)
    leg_max_speed_rad_s: float = Field(default=12.0, gt=0)
    max_acceleration_rad_s2: float = Field(default=150.0, gt=0)


class SeatedHoldConfig(BaseModel):
    """Contact-triggered seated motor hold (HALO sit_v1)."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    phases: list[str] = Field(default_factory=lambda: ["sit_down", "seated"])
    trigger_min_pelvis_height_m: float = 0.49
    trigger_max_pelvis_height_m: float = 0.58
    trigger_max_center_error_m: float = Field(default=0.12, gt=0)
    min_seat_vertical_impulse_ns: float = Field(
        default=2.0, ge=0, description="seat vertical impulse in the last 20 ms control period"
    )
    blend_s: float = Field(default=0.6, gt=0)
    upright_torso: bool = True
    facing_yaw_deg: float = 180.0
    torso_recline_deg: float = -5.0
    torso_feedback_gain: float = 0.8


class SonicTrackingPlannerConfig(BaseConfig):
    """Reference motion, SONIC models and target conditioning."""

    backend: str = "sonic_tracking"
    reference_dir: str = "assets/Halo/reference/crawl"
    reference_fps: float = Field(default=30.0, gt=0)
    model_dir: str = "assets/Halo/models/sonic"
    onnx_threads: int = Field(default=4, ge=1)
    warmup_control_steps: int = Field(
        default=50, ge=0, description="unrecorded 50 Hz steps tracking the start pose"
    )
    joint_limit_margin_rad: float = Field(default=0.001, ge=0)
    motor_smoothing: MotorSmoothingConfig = Field(default_factory=MotorSmoothingConfig)
    seated_hold: SeatedHoldConfig = Field(default_factory=SeatedHoldConfig)
    seated_target_filter_alpha: float | None = Field(default=None, gt=0, le=1)


class SonicTrackingPlanner(BasePlanner):
    """One SONIC action per ``on_physics_step`` call (the Job calls it every 20 ms)."""

    config_cls = SonicTrackingPlannerConfig
    action_names = list(MUJOCO_JOINT_NAMES)

    def __init__(self, config: SonicTrackingPlannerConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("planner")
        reference = Path(resolve_usd_path(f"{self.config.reference_dir}/qpos.npy")).parent
        self.reference_qpos = np.load(reference / "qpos.npy")
        self.spec = json.loads((reference / "spec.json").read_text())
        if (
            abs(float(self.spec.get("fps", self.config.reference_fps)) - self.config.reference_fps)
            > 1e-9
        ):
            raise ValueError("planner.reference_fps differs from the reference spec.json fps")
        self.phases: list[dict] = self.spec["phases"]
        self.phase_names = [p["name"] for p in self.phases]
        hold = self.config.seated_hold
        if hold.enabled and not set(hold.phases) <= set(self.phase_names):
            raise ValueError(f"seated_hold.phases must be reference phases: {self.phase_names}")
        self.control_frames = len(
            resample_reference(self.reference_qpos, self.config.reference_fps, CONTROL_HZ)
        )
        self.seat_center_xy: np.ndarray | None = None
        self.seat_vertical_impulse_ns = 0.0
        self.tracker: SonicTracker | None = None
        self.frame = 0
        self.observation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.rows: list[dict] = []
        self._reset_conditioning()

    @property
    def initial_qpos(self) -> np.ndarray:
        """First reference row; the robot starts exactly in this pose."""
        return np.asarray(self.reference_qpos[0], dtype=float)

    def _reset_conditioning(self) -> None:
        self.filter: MotorTargetFilter | None = None
        self.last_target = self.initial_qpos[7:].copy()
        self.hold_pose: np.ndarray | None = None
        self.hold_start_s: float | None = None

    def init_planner(self, robot: Any) -> None:
        """Bind the robot and load the SONIC encoder/decoder (CPU onnxruntime)."""
        self.robot = robot
        model_dir = Path(resolve_usd_path(f"{self.config.model_dir}/model_encoder.onnx")).parent
        self.tracker = SonicTracker(
            model_dir,
            self.reference_qpos,
            reference_fps=self.config.reference_fps,
            threads=self.config.onnx_threads,
        )
        if len(self.tracker) != self.control_frames:
            raise RuntimeError("SONIC reference length changed after resampling")
        self.state["ready"] = True

    def reset(self) -> None:
        """History filled with the reference start pose; conditioning restarts."""
        self.tracker.reset(self.initial_qpos)
        self._reset_conditioning()
        self._warmup_target = self.initial_qpos[7:].copy()
        self.frame = 0
        self.rows.clear()

    def warmup_step(self, robot: Any) -> None:
        """Track reference frame 0 before recording; targets are only clipped to the limits."""
        qpos, joint_vel, base_vel = robot.read_state()
        target = self.tracker.step(0, qpos, joint_vel, base_vel[3:])
        lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
        self._warmup_target = np.clip(target, lo, hi)
        robot.set_body_targets(self._warmup_target)

    def begin_episode(self) -> None:
        """Start frame 0; the arm filter starts from the last warm-up target (as in HALO)."""
        self.filter = MotorTargetFilter(
            self._warmup_target, self.config.motor_smoothing, MUJOCO_JOINT_NAMES
        )
        self.frame = 0

    def on_physics_step(self, robot: Any) -> bool:
        """Compute and send the targets of the current control frame; True after the last one."""
        if self.filter is None:
            raise RuntimeError("Call begin_episode before the first control frame")
        frame = self.frame
        qpos, joint_vel, base_vel = robot.read_state()
        self.observation = (qpos, joint_vel, base_vel)
        target = self.tracker.step(frame, qpos, joint_vel, base_vel[3:])
        phase = phase_index(
            frame, self.phases, control_hz=CONTROL_HZ, reference_fps=self.config.reference_fps
        )
        name = self.phase_names[phase]
        hold = self.config.seated_hold
        blend = 0.0
        if hold.enabled and name in hold.phases:
            target, blend = self._seated_hold(frame, qpos, target, robot)
        alpha = self.config.seated_target_filter_alpha
        if name == "seated" and self.hold_pose is None and alpha:
            target = self.last_target + (target - self.last_target) * alpha
        margin = self.config.joint_limit_margin_rad
        lo, hi = robot.joint_limits[:, 0] + margin, robot.joint_limits[:, 1] - margin
        target = np.clip(self.filter.step(np.clip(target, lo, hi), 1.0 / CONTROL_HZ), lo, hi)
        robot.set_body_targets(target)
        self.tracker.set_applied_target(target)
        self.last_target = target.copy()
        self.state.update(phase=name, phase_index=phase, last_action=target)
        self.state["progress"] = {"frame": frame, "frames": self.control_frames, "phase": name}
        self.rows.append({"frame": frame, "phase": name, "hold_blend": round(float(blend), 4)})
        self.frame += 1
        return self.frame >= self.control_frames

    def _seated_hold(
        self, frame: int, qpos: np.ndarray, target: np.ndarray, robot: Any
    ) -> tuple[np.ndarray, float]:
        from scipy.spatial.transform import Rotation

        hold = self.config.seated_hold
        if self.seat_center_xy is None:
            raise RuntimeError("seated_hold needs the seat centre from the scene")
        now = frame / CONTROL_HZ
        if (
            self.hold_pose is None
            and hold.trigger_min_pelvis_height_m < qpos[2] < hold.trigger_max_pelvis_height_m
            and np.linalg.norm(qpos[:2] - self.seat_center_xy) < hold.trigger_max_center_error_m
            and self.seat_vertical_impulse_ns > hold.min_seat_vertical_impulse_ns
        ):
            self.hold_pose = qpos[7:].copy()
            self.hold_start_s = now
        if self.hold_pose is None:
            return target, 0.0
        blend = float(np.clip((now - self.hold_start_s) / hold.blend_s, 0, 1))
        blend = blend * blend * (3 - 2 * blend)
        target = (1 - blend) * target + blend * self.hold_pose
        if hold.upright_torso:
            torso = Rotation.from_quat(robot.torso_pose()[3:7][[1, 2, 3, 0]])
            goal = Rotation.from_euler(
                "z", hold.facing_yaw_deg, degrees=True
            ) * Rotation.from_euler("y", hold.torso_recline_deg, degrees=True)
            relative = Rotation.from_quat(qpos[3:7][[1, 2, 3, 0]]).inv() * goal
            angles = relative.as_euler("ZXY")
            error = (goal.inv() * torso).as_rotvec()
            for joint, angle, axis in (
                ("waist_yaw_joint", angles[0], 2),
                ("waist_roll_joint", angles[1], 0),
                ("waist_pitch_joint", angles[2], 1),
            ):
                j = MUJOCO_JOINT_NAMES.index(joint)
                upright = angle - hold.torso_feedback_gain * error[axis]
                target[j] = (1 - blend) * target[j] + blend * upright
        return target, blend

    def result(self) -> dict[str, Any]:
        return {
            "control_frames": self.control_frames,
            "executed_control_frames": self.frame,
            "phases": self.phases,
            "seated_hold_start_s": self.hold_start_s,
            "reference_source": self.spec.get("source_path"),
        }
