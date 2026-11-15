"""Kit-free checks of the tabletop replay task: source datasets, schedule, fidelity, configs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from real2sim2real.core.config import apply_settings_file, reset_settings
from real2sim2real.core.registry import JOB_REGISTRY, ensure_job_backends
from real2sim2real.job.job_tabletop_pickplace import EXTRA_FEATURES
from real2sim2real.job.job_tabletop_replay import TabletopReplayJob, TabletopReplayJobConfig
from real2sim2real.planner.planner_tabletop_replay import (
    LeRobotReplayPlanner,
    LeRobotReplayPlannerConfig,
)
from real2sim2real.robot.robot_tabletop_pickplace import FrankaTabletopRobot
from real2sim2real.scene.scene_tabletop_pickplace import TabletopHomeSceneConfig
from real2sim2real.simworld.isaacsim import IsaacsimWorldConfig
from real2sim2real.utils import tabletop_replay_utils as ru
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder
from real2sim2real.utils.tabletop_pickplace_utils import PHASE_INDEX

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/tabletop_replay"


def _write_source(root: Path, frames: int = 6, success: bool = True) -> Path:
    """A tiny tabletop_pickplace-style episode (LeRobot v3 + meta/physics layout)."""
    episode = root / "episode_000"
    recorder = LeRobotV3Recorder(
        episode / "lerobot_v3", fps=30, width=16, height=12, task="pick", robot_type="franka_panda",
        joint_names=FrankaTabletopRobot.recording_joint_names,
        action_names=FrankaTabletopRobot.action_names, cameras={"overhead_cam": (16, 12)},
        extra_features=EXTRA_FEATURES,
    )  # fmt: skip
    for k in range(frames):
        Image.new("RGB", (16, 12), (k * 20, 0, 0)).save(recorder.frame_path("overhead_cam"))
        action = np.r_[np.full(7, 0.1 * k), 0.04 if k < 3 else 0.0]
        state = np.r_[np.full(7, 0.1 * k - 0.01), 0.04, 0.04]
        recorder.add_frame(state, action, {
            "observation.object_pose": [0.6, -3.66, 0.805 + 0.01 * k, 1, 0, 0, 0],
            "observation.ee_pose": [0.6, -3.66, 0.9, 0, 1, 0, 0],
            "observation.phase_index": PHASE_INDEX["start" if k < 2 else "lift"],
            "next.done": k == frames - 1, "next.success": False,
        })  # fmt: skip
    dataset = recorder.finalize(episode_success=success)
    physics = dataset / "meta/physics/episode_000000"
    physics.mkdir(parents=True)
    layout = {"episode": 0, "target_offset_m": [0.0657, -0.0147], "jitter_draws": 1,
              "place_target_xy": [0.6, -3.9], "table_top_z": 0.78,
              "base_pose": [[0.304, -3.75, 0.78], [1, 0, 0, 0]],
              "target_start_center_m": [0.6157, -3.6647, 0.806]}  # fmt: skip
    (physics / "layout.json").write_text(json.dumps(layout))
    (physics / "configuration.json").write_text(json.dumps({"sim_world": {"physics_dt": 1 / 120}}))
    (dataset / "meta/task_outcome.json").write_text(json.dumps({"result": {"success": success}}))
    return dataset


def test_load_source_episode_round_trip(tmp_path):
    dataset = _write_source(tmp_path)
    assert ru.source_episode_dirs(tmp_path) == [dataset]
    assert ru.source_episode_dirs(tmp_path, [0]) == [dataset]
    with pytest.raises(FileNotFoundError):
        ru.source_episode_dirs(tmp_path, [3])
    source = ru.load_source_episode(dataset)
    assert source.fps == 30 and source.success
    assert source.actions.shape == (6, 8) and source.states.shape == (6, 9)
    np.testing.assert_allclose(source.actions[2, :7], 0.2, atol=1e-6)
    assert list(source.phases) == [0, 0, 4, 4, 4, 4]
    assert source.layout["target_offset_m"] == [0.0657, -0.0147]
    assert source.action_names == FrankaTabletopRobot.action_names


def test_replay_command_hold_and_linear():
    actions = np.array([np.r_[np.zeros(7), 0.04], np.r_[np.full(7, 0.4), 0.0]])
    target, velocity, frame = ru.replay_command(actions, 2, 4, 30, "linear")
    assert frame == 0
    np.testing.assert_allclose(target, np.r_[np.full(7, 0.2), 0.02])
    np.testing.assert_allclose(velocity, np.full(7, 12.0))
    target, velocity, frame = ru.replay_command(actions, 2, 4, 30, "hold")
    np.testing.assert_allclose(target, actions[0])
    assert not velocity.any()
    target, velocity, frame = ru.replay_command(actions, 99, 4, 30, "linear")
    assert frame == 1 and not velocity.any()
    np.testing.assert_allclose(target, actions[1])
    assert ru.physics_stride(1 / 120, 30) == 4
    with pytest.raises(ValueError):
        ru.physics_stride(1 / 120, 25)


class FakeRobot:
    action_names = FrankaTabletopRobot.action_names
    recording_joint_names = FrankaTabletopRobot.recording_joint_names

    def __init__(self) -> None:
        self.commands = []

    def command(self, q, finger, qd=None) -> None:
        self.commands.append((np.asarray(q), finger, np.asarray(qd)))


def test_replay_planner_runs_until_the_last_recorded_frame(tmp_path):
    _write_source(tmp_path)
    planner = LeRobotReplayPlanner(LeRobotReplayPlannerConfig(source_dir=str(tmp_path)))
    robot = FakeRobot()
    planner.build(None, robot, 1 / 120)
    planner.begin_episode(planner.sources[0])
    steps = 0
    while not planner.on_physics_step(robot):
        steps += 1
    assert steps == 5 * 4  # last recorded frame is reached after (T-1) * stride steps
    assert planner.result()["sequence_completed"]
    assert [r["phase"] for r in planner.rows] == ["start", "lift"]
    np.testing.assert_allclose(robot.commands[-1][0], 0.5, atol=1e-6)
    assert robot.commands[-1][1] == pytest.approx(0.0)
    np.testing.assert_allclose(robot.commands[1][0], 0.025, atol=1e-6)  # linear between frames


def test_replay_planner_refuses_diagnostic_or_mismatched_sources(tmp_path):
    _write_source(tmp_path, success=False)
    planner = LeRobotReplayPlanner(LeRobotReplayPlannerConfig(source_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="diagnostic"):
        planner.build(None, FakeRobot(), 1 / 120)
    other = SimpleNamespace(action_names=["a"], recording_joint_names=["b"])
    with pytest.raises(ValueError, match="joint names"):
        planner.build(None, other, 1 / 120)


def test_replay_rejects_different_physics_rate_even_when_fps_divides_both(tmp_path):
    _write_source(tmp_path)
    planner = LeRobotReplayPlanner(LeRobotReplayPlannerConfig(source_dir=str(tmp_path)))
    # 30 fps divides both 60 Hz and 120 Hz, but their physical trajectories differ.
    with pytest.raises(ValueError, match="recorded physics_dt"):
        planner.build(None, FakeRobot(), 1 / 60)


def test_replay_fidelity(tmp_path):
    source = ru.load_source_episode(_write_source(tmp_path))
    acceptance = ru.ReplayAcceptanceConfig()
    good = ru.replay_fidelity(source.states + 0.005, source.object_pose[-1], source, acceptance)
    assert all(good["checks"].values())
    drift = source.states.copy()
    drift[:, :7] += 0.1
    bad = ru.replay_fidelity(drift, source.object_pose[-1] + [0.05, 0, 0, 0, 0, 0, 0], source,
                             acceptance)  # fmt: skip
    assert not bad["checks"]["arm_follows_recording"]
    assert not bad["checks"]["object_ends_where_recorded"]
    with pytest.raises(ValueError):
        ru.replay_fidelity(source.states[:3], source.object_pose[-1], source, acceptance)


@pytest.mark.parametrize("name", [p.name for p in sorted(CONFIGS.glob("*.toml"))])
def test_replay_configs_route(name):
    try:
        apply_settings_file(CONFIGS / name)
        world = IsaacsimWorldConfig.from_settings("sim_world")
        scene = TabletopHomeSceneConfig.from_settings("scene")
        job = TabletopReplayJobConfig.from_settings("job")
        planner = LeRobotReplayPlannerConfig.from_settings("planner")
        ensure_job_backends()
        assert JOB_REGISTRY.get(job.backend) is TabletopReplayJob
        assert world.physics_engine == "physx" and round(1 / world.physics_dt) == 120
        debug = name.endswith("_debug.toml")
        assert planner.source_dir == (
            "outputs/debug/tabletop_pickplace/franka_physx_home000_debug"
            if debug
            else "outputs/tabletop_pickplace/franka_physx_home000"
        )
        assert world.renderer == ("MinimalRendering" if debug else "PathTracing")
        assert job.recorder.fps == (15 if debug else 30)
        assert job.recorder.capture_mode == "parallel_cameras"
        assert all(c.resolution == ((640, 480) if debug else (1024, 768)) for c in scene.cameras)
        assert world.background_usd_path.startswith("assets/TabletopReplay/")
        assert all(o.usd_path.startswith("assets/TabletopReplay/") for o in scene.objects)
        assert job.output_dir.startswith(
            "outputs/debug/tabletop_replay/" if debug else "outputs/tabletop_replay/"
        )
    finally:
        reset_settings()
