"""Tabletop replay Job: replay recorded pick-and-place episodes in PhysX and re-check them.

Migrated from Real2Sim2Real ``TabletopJob`` + ``ReplayPlanner``
(``configs/arm_franka_replay_pickplace/home000.toml``): same scene, robot and cameras, the
planner only applies the recorded actions. Each replayed episode starts from the source
episode's layout (cube offset, table, robot base) read from ``meta/physics``; it passes when the
pick-and-place physical checks pass again and the replayed joints / final cube position stay
close to the recording. Accepted replays are exported as LeRobot v3 like the source task.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from real2sim2real.job.job_tabletop_pickplace import (
    TabletopPickPlaceJob,
    TabletopPickPlaceJobConfig,
)
from real2sim2real.utils.tabletop_replay_utils import ReplayAcceptanceConfig, replay_fidelity


class TabletopReplayJobConfig(TabletopPickPlaceJobConfig):
    """``[job]`` of the replay task; episodes come from ``planner.source_dir``."""

    backend: str = "tabletop_replay"
    output_dir: str = "outputs/tabletop_replay/franka_physx_home000"
    failed_dir: str = "outputs/debug/tabletop_replay"
    replay_acceptance: ReplayAcceptanceConfig = Field(default_factory=ReplayAcceptanceConfig)
    layout_tolerance_m: float = Field(default=1e-4, gt=0, description="source vs replay layout")


class TabletopReplayJob(TabletopPickPlaceJob):
    """One replay per source episode."""

    config_cls = TabletopReplayJobConfig

    def after_place_robot_hook(self) -> None:
        """Load the source episodes; the replay records at the source dataset fps."""
        super().after_place_robot_hook()
        fps = {source.fps for source in self.planner.sources}
        if fps != {self.config.recorder.fps}:
            raise ValueError(f"recorder.fps={self.config.recorder.fps}, source datasets: {fps}")

    def episode_plans(self) -> list[dict[str, Any]]:
        """Use the recorded layouts instead of new random offsets."""
        plans = []
        for index, source in enumerate(self.planner.sources):
            layout = source.layout
            plans.append(
                {
                    "target_offset_m": layout["target_offset_m"],
                    "jitter_draws": layout.get("jitter_draws"),
                    "source_index": index,
                    "source_dataset": str(source.path),
                    "source_episode": layout["episode"],
                }
            )
        return plans

    def begin_controller(self, index: int, episode: dict[str, Any]) -> None:
        source = self.planner.sources[episode["source_index"]]
        layout = source.layout
        self._check_layout(layout)
        episode["place_target_xy"] = layout["place_target_xy"]
        self.planner.begin_episode(source)

    def _check_layout(self, recorded: dict[str, Any]) -> None:
        """The replay must start from the recorded table, base and cube positions."""
        tolerance = self.config.layout_tolerance_m
        pairs = {
            "table_top_z": ([self.scene.layout["table_top_z"]], [recorded["table_top_z"]]),
            "base_position": (self.robot.base_pose[0], recorded["base_pose"][0]),
            "target_start_center": (self.scene.target_center(), recorded["target_start_center_m"]),
        }
        for name, (actual, expected) in pairs.items():
            error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
            if error > tolerance:
                raise RuntimeError(
                    f"Replay layout differs from the recording: {name} {error:.2e} m"
                )

    def evaluate(
        self, log: dict[str, np.ndarray], layout: dict[str, Any], contacts: dict, controller: Any
    ) -> dict[str, Any]:
        """Pick-and-place checks plus fidelity to the recorded states."""
        result = super().evaluate(log, layout, contacts, controller)
        source = controller.source
        frames = np.arange(len(source.states)) * controller.stride
        if frames[-1] >= len(log["q"]):
            raise RuntimeError("Replay ended before the last recorded frame")
        fidelity = replay_fidelity(
            np.asarray(log["q"])[frames],
            np.asarray(log["object_pos"])[-1],
            source,
            self.config.replay_acceptance,
        )
        result["checks"].update(fidelity["checks"])
        result["metrics"].update(fidelity["metrics"])
        result["checks"] = {k: bool(v) for k, v in result["checks"].items()}
        result["failed_checks"] = [k for k, ok in result["checks"].items() if not ok]
        result["success"] = all(result["checks"].values())
        return result
