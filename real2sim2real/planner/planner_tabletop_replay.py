"""Replay planner: drive the Franka with the actions of recorded tabletop pick-and-place episodes.

Migrated from Real2Sim2Real ``ReplayPlanner`` (``configs/arm_franka_replay_pickplace``), which
replayed the LeRobot ``action`` (or ``observation.state``) of collected episodes without motion
planning and inferred the gripper from the finger opening. Here the recorded LeRobot v3
``action`` column (7 arm joint targets + finger target) of ``tabletop_pickplace`` datasets is
applied at the physics rate; no cuMotion call is made. The phase of each frame is taken from
the recorded ``observation.phase_index`` so the Job can reuse the pick-and-place checks.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import Field

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import LoggerMixin
from real2sim2real.planner.base import BasePlanner
from real2sim2real.utils.tabletop_pickplace_utils import PHASE_INDEX, PHASES
from real2sim2real.utils.tabletop_replay_utils import (
    SourceEpisode,
    load_source_episode,
    physics_stride,
    replay_command,
    source_episode_dirs,
)


class LeRobotReplayPlannerConfig(BaseConfig):
    """``[planner]`` of the replay task."""

    backend: str = "lerobot_replay"
    source_dir: str = Field(
        default="outputs/tabletop_pickplace/franka_physx_home000",
        description="tabletop_pickplace output with episode_XXX/lerobot_v3",
    )
    episodes: list[int] = Field(default_factory=list, description="empty: every source episode")
    interpolation: Literal["linear", "hold"] = "linear"
    require_demonstrations: bool = Field(
        default=True, description="refuse source episodes that failed acceptance"
    )


class LeRobotReplayPlanner(BasePlanner, LoggerMixin):
    """Applies one recorded episode per ``begin_episode``; ends on the last recorded frame."""

    config_cls = LeRobotReplayPlannerConfig

    def __init__(self, config: LeRobotReplayPlannerConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config: LeRobotReplayPlannerConfig = config or self.config_cls.from_settings("planner")
        self.sources: list[SourceEpisode] = []
        self.source: SourceEpisode | None = None
        self.rows: list[dict[str, Any]] = []
        self.stride = 1
        self.step = 0
        self.phase = "idle"

    def init_planner(self, robot: Any) -> None:
        self.robot = robot

    def build(self, scene: Any, robot: Any, dt: float) -> None:
        """Load the source episodes and check they match this robot and physics rate."""
        del scene
        self.sources = [
            load_source_episode(path)
            for path in source_episode_dirs(self.config.source_dir, self.config.episodes)
        ]
        for source in self.sources:
            if not math.isclose(source.physics_dt, dt, rel_tol=0, abs_tol=1e-12):
                raise ValueError(
                    f"{source.path}: recorded physics_dt={source.physics_dt} differs from {dt}"
                )
            if source.action_names != list(robot.action_names) or source.state_names != list(
                robot.recording_joint_names
            ):
                raise ValueError(f"{source.path}: joint names differ from the robot")
            if self.config.require_demonstrations and not source.success:
                raise ValueError(f"{source.path} is a diagnostic (failed) episode")
            physics_stride(dt, source.fps)
        self.dt = float(dt)
        self.state["ready"] = True
        self.log.info("回放数据就绪", episodes=[str(s.path) for s in self.sources])

    def begin_episode(self, source: SourceEpisode) -> None:
        self.source = source
        self.stride = physics_stride(self.dt, source.fps)
        self.step = 0
        self.rows = []
        self.phase = PHASES[int(source.phases[0])]
        self.state.update(
            phase=self.phase, phase_index=int(source.phases[0]), last_action=None, done=False
        )

    def on_physics_step(self, robot: Any) -> bool:
        """Command the recorded action of this step; True on the last recorded frame."""
        source = self.source
        target, velocity, frame = replay_command(
            source.actions, self.step, self.stride, source.fps, self.config.interpolation
        )
        robot.command(target[:7], float(target[7]), velocity)
        phase = PHASES[int(source.phases[frame])]
        if phase != self.phase or not self.rows:
            self.rows.append({"phase": phase, "step": self.step, "frame": frame})
        self.phase = phase
        last = self.step >= (len(source.actions) - 1) * self.stride
        if last:
            self.phase = "done"
        self.state.update(
            phase=self.phase,
            phase_index=PHASE_INDEX[self.phase],
            last_action=np.asarray(target, float),
            done=last,
        )
        self.step += 1
        return last

    def result(self) -> dict[str, Any]:
        source = self.source
        return {
            "sequence_completed": self.phase == "done",
            "phase": self.phase,
            "source_dataset": None if source is None else str(source.path),
            "source_frames": None if source is None else len(source.actions),
            "interpolation": self.config.interpolation,
            "phase_steps": self.rows,
        }
