"""Helpers of the tabletop replay task: read recorded LeRobot v3 episodes and compare replays.

Sections:

1. Source episodes written by ``tabletop_pickplace`` (``episode_XXX/lerobot_v3``).
2. Linear or zero-order-hold action schedule at the physics rate.
3. Replay fidelity checks (replayed states vs. the recorded states).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

# ================================================================================================
# 1. Source episodes
# ================================================================================================


@dataclass
class SourceEpisode:
    """One recorded episode: per-frame arrays at ``fps`` plus its physical layout."""

    path: Path
    fps: int
    physics_dt: float
    actions: np.ndarray  # (T, 8): 7 arm joint targets (rad) + finger target (m)
    states: np.ndarray  # (T, 9): measured joints
    phases: np.ndarray  # (T,): phase index
    object_pose: np.ndarray  # (T, 7): target centre xyz + wxyz
    layout: dict[str, Any]
    success: bool
    action_names: list[str]
    state_names: list[str]


def source_episode_dirs(source_dir: Path, episodes: list[int] | None = None) -> list[Path]:
    """``episode_XXX/lerobot_v3`` datasets below a tabletop_pickplace output directory."""
    root = Path(source_dir)
    found = sorted(info.parents[1] for info in root.glob("episode_*/lerobot_v3/meta/info.json"))
    if episodes:
        by_index = {int(p.parent.name.split("_")[-1]): p for p in found}
        missing = [i for i in episodes if i not in by_index]
        if missing:
            raise FileNotFoundError(f"Source episodes {missing} not found below {root}")
        found = [by_index[i] for i in episodes]
    if not found:
        raise FileNotFoundError(f"No episode_XXX/lerobot_v3 datasets below {root}")
    return found


def load_source_episode(dataset: Path) -> SourceEpisode:
    """Read data, info and the physical layout of one recorded LeRobot v3 episode."""
    import pyarrow.parquet as pq

    dataset = Path(dataset)
    info = json.loads((dataset / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError(f"Unsupported dataset version in {dataset}")
    table = pq.read_table(dataset / info["data_path"].format(chunk_index=0, file_index=0))
    outcome = json.loads((dataset / "meta/task_outcome.json").read_text())
    layout = json.loads((dataset / "meta/physics/episode_000000/layout.json").read_text())
    configuration = json.loads(
        (dataset / "meta/physics/episode_000000/configuration.json").read_text()
    )
    return SourceEpisode(
        path=dataset,
        fps=int(info["fps"]),
        physics_dt=float(configuration["sim_world"]["physics_dt"]),
        actions=np.asarray(table["action"].to_pylist(), dtype=float),
        states=np.asarray(table["observation.state"].to_pylist(), dtype=float),
        phases=table["observation.phase_index"].to_numpy().astype(int),
        object_pose=np.asarray(table["observation.object_pose"].to_pylist(), dtype=float),
        layout=layout,
        success=bool(outcome["result"]["success"]),
        action_names=list(info["features"]["action"]["names"]),
        state_names=list(info["features"]["observation.state"]["names"]),
    )


# ================================================================================================
# 2. Linear or zero-order-hold schedule
# ================================================================================================


def physics_stride(physics_dt: float, fps: int) -> int:
    """Physics steps per recorded frame; the dataset fps must divide the physics rate."""
    stride = (1.0 / physics_dt) / fps
    if not math.isclose(stride, round(stride), abs_tol=1e-8, rel_tol=0):
        raise ValueError(f"Dataset fps {fps} does not divide the physics rate {1 / physics_dt}")
    return round(stride)


def replay_command(
    actions: np.ndarray, step: int, stride: int, fps: int, interpolation: str = "linear"
) -> tuple[np.ndarray, np.ndarray, int]:
    """Joint target of a physics step from actions recorded every ``stride`` steps.

    ``hold`` keeps frame k for its ``stride`` steps (zero velocity target); ``linear``
    interpolates toward frame k+1 with the finite-difference arm velocity (rad/s) as
    feed-forward, which reproduces the timing of the original commands (offline replay: it uses
    the next recorded action, unlike a causal policy).
    Returns ``(target, arm_velocity, frame)``.
    """
    if interpolation not in ("hold", "linear"):
        raise ValueError(f"Unknown interpolation {interpolation}")
    frame = min(step // stride, len(actions) - 1)
    target = np.asarray(actions[frame], float).copy()
    velocity = np.zeros(7)
    if interpolation == "linear" and frame + 1 < len(actions):
        delta = np.asarray(actions[frame + 1], float) - target
        velocity = delta[:7] * fps
        target += delta * ((step - frame * stride) / stride)
    return target, velocity, frame


# ================================================================================================
# 3. Replay fidelity
# ================================================================================================


class ReplayAcceptanceConfig(BaseModel):
    """Replayed states compared with the recorded ones (same frame times)."""

    model_config = ConfigDict(extra="forbid")
    max_arm_rms_error_rad: FiniteFloat = Field(default=0.02, gt=0)
    max_arm_error_rad: FiniteFloat = Field(default=0.05, gt=0)
    max_finger_error_m: FiniteFloat = Field(default=0.005, gt=0)
    max_object_final_error_m: FiniteFloat = Field(default=0.01, gt=0)


def replay_fidelity(
    replay_states: np.ndarray,
    replay_object_final: np.ndarray,
    source: SourceEpisode,
    acceptance: ReplayAcceptanceConfig,
) -> dict[str, Any]:
    """Checks and metrics of a replay against its source episode."""
    states = np.asarray(replay_states, float)
    if states.shape != source.states.shape:
        raise ValueError(f"Replay has {states.shape} states, source {source.states.shape}")
    error = states - source.states
    arm = np.abs(error[:, :7])
    rms = float(np.sqrt(np.mean(error[:, :7] ** 2)))
    finger = float(np.abs(error[:, 7:9]).max())
    final = float(np.linalg.norm(np.asarray(replay_object_final)[:3] - source.object_pose[-1, :3]))
    metrics = {
        "arm_rms_error_rad": rms,
        "arm_max_error_rad": float(arm.max()),
        "finger_max_error_m": finger,
        "object_final_error_m": final,
        "source_success": source.success,
    }
    checks = {
        "source_was_demonstration": source.success,
        "arm_follows_recording": rms <= acceptance.max_arm_rms_error_rad
        and float(arm.max()) <= acceptance.max_arm_error_rad,
        "fingers_follow_recording": finger <= acceptance.max_finger_error_m,
        "object_ends_where_recorded": final <= acceptance.max_object_final_error_m,
    }
    return {"checks": checks, "metrics": metrics}
