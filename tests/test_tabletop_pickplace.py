"""Kit-free checks of the tabletop pick-and-place task: layout, targets, phase machine, checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from real2sim2real.core.config import apply_settings_file, reset_settings
from real2sim2real.core.registry import JOB_REGISTRY, ensure_job_backends
from real2sim2real.job.job_tabletop_pickplace import (
    TabletopPickPlaceJob,
    TabletopPickPlaceJobConfig,
)
from real2sim2real.planner.planner_tabletop_pickplace import (
    CumotionPickPlacePlanner,
    CumotionPickPlacePlannerConfig,
)
from real2sim2real.robot.robot_tabletop_pickplace import (
    FrankaTabletopRobotConfig,
    footprint_shift,
)
from real2sim2real.scene.scene_tabletop_pickplace import TabletopHomeScene, TabletopHomeSceneConfig
from real2sim2real.simworld.isaacsim import IsaacsimWorldConfig
from real2sim2real.utils import tabletop_pickplace_utils as tu

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/tabletop_pickplace"
TABLE = ((0.15, -4.25, 0.048), (0.75, -3.25, 0.7799))


def _scene() -> TabletopHomeScene:
    config = TabletopHomeSceneConfig(
        table={"usd_path": "table.usd"},
        objects=[
            {"name": "cube", "usd_path": "c.usd", "prim_path": "/World/scene/Objects/cube_0",
             "role": "target", "offset_m": (0.10, 0.10)},
            {"name": "apple", "usd_path": "a.usd", "prim_path": "/World/scene/Objects/apple_000",
             "role": "distractor", "offset_m": (0.0, 0.30), "body_prim": "body"},
        ],
    )  # fmt: skip
    scene = TabletopHomeScene(config)
    scene.layout = {
        "table_top_z": TABLE[1][2],
        "table_bounds_m": [list(TABLE[0]), list(TABLE[1])],
        "objects": {
            "cube": {"aabb_center_m": [0.55, -3.65, 0.805], "half_extents_m": [0.025] * 3},
            "apple": {"aabb_center_m": [0.45, -3.45, 0.827], "half_extents_m": [0.046, 0.0466,
                      0.0469]},
        },
    }  # fmt: skip
    return scene


def test_jitter_reproduces_the_source_layout_sequence():
    # Source run (2026-11-02): cube centres 0.0657/-0.0147 and 0.0861/+0.0474 m from nominal.
    scene = _scene()
    rng = np.random.default_rng(42)
    first, draws1 = tu.jitter_offset(rng, 0.12, scene.accept_offset)
    second, draws2 = tu.jitter_offset(rng, 0.12, scene.accept_offset)
    np.testing.assert_allclose(first, [0.0657, -0.0147], atol=1e-4)
    np.testing.assert_allclose(second, [0.0861, 0.0474], atol=1e-4)
    assert draws1 == draws2 == 1


def test_accept_offset_keeps_gap_to_distractor_and_table_edge():
    scene = _scene()
    assert scene.accept_offset([0.0, 0.0])
    assert not scene.accept_offset([-0.10, 0.15])  # cube footprint touching the apple
    assert not scene.accept_offset([0.20, 0.0])  # beyond the table edge margin
    assert tu.footprint_gap([0, 0], [0.1, 0.1], [0.3, 0], [0.1, 0.1]) == pytest.approx(0.1)
    assert tu.footprint_gap([0, 0], [0.1, 0.1], [0.15, 0], [0.1, 0.1]) == pytest.approx(-0.05)
    assert tu.footprint_gap([0, 0], [0.1, 0.1], [0.3, 0.3], [0.1, 0.1]) == pytest.approx(
        math.hypot(0.1, 0.1)
    )


def test_table_frame_points_and_base_footprint():
    np.testing.assert_allclose(tu.table_point([0.45, -3.75, 0.4], 0.0, [0.15, -0.15]),
                               [0.60, -3.90])  # fmt: skip
    np.testing.assert_allclose(tu.table_point([0, 0], 90.0, [1.0, 0.0]), [0.0, 1.0], atol=1e-12)
    # Base nominal at x=0.10 (table edge 0.15) -> moved inside, as the source snap did.
    shift = footprint_shift([-0.04, -3.86], [0.18, -3.64], TABLE[0], TABLE[1])
    np.testing.assert_allclose(shift, [0.19, 0.0])
    np.testing.assert_allclose(footprint_shift([0.3, -3.8], [0.4, -3.7], *TABLE), [0.0, 0.0])
    with pytest.raises(ValueError):
        footprint_shift([0, -5], [1, -2], *TABLE)


def test_hand_targets_follow_the_source_segments():
    pick, place = np.array([0.6, -3.66, 0.805]), np.array([0.6, -3.9, 0.805])
    targets = tu.hand_targets(pick, place, hand_above_m=0.103, clearance_m=0.12)
    np.testing.assert_allclose(targets["descend"], pick + [0, 0, 0.103])
    np.testing.assert_allclose(targets["approach_pick"], pick + [0, 0, 0.223])
    np.testing.assert_allclose(targets["lift"], targets["approach_pick"])
    np.testing.assert_allclose(targets["lower"], place + [0, 0, 0.103])
    np.testing.assert_allclose(targets["transfer"], targets["retreat"])
    base = tu.world_to_base([0.304, -3.75, 0.78], tu.yaw_quat(0.0), pick)
    np.testing.assert_allclose(base, [0.296, 0.09, 0.025], atol=1e-12)
    rotated = tu.world_to_base([0, 0, 0], tu.yaw_quat(90.0), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(rotated, [1.0, 0.0, 0.0], atol=1e-12)


def test_trajectory_sampling_and_time_scale():
    samples = tu.sample_trajectory(lambda t: [t, 2 * t], 1.0, 0.5, 0.1)
    np.testing.assert_allclose(samples[[0, -1]], [[1.0, 2.0], [1.5, 3.0]])
    assert len(samples) == 6
    slow = tu.sample_trajectory(lambda t: [t], 0.0, 0.5, 0.1, time_scale=2.0)
    assert len(slow) == 11 and slow[-1, 0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        tu.sample_trajectory(lambda t: [t], 0.0, -1.0, 0.1)


def test_sensor_camera_config():
    camera = tu.SensorCameraConfig(
        name="wrist_cam", prim_path="/World/robots/Franka/panda_hand/wrist_cam",
        pose_relative_to="/World/robots/Franka/panda_hand", usd_path="sg5.usd",
        optical_prim="Camera_SG5",
    )  # fmt: skip
    assert camera.capture_prim == "/World/robots/Franka/panda_hand/wrist_cam/Camera_SG5"
    with pytest.raises(ValueError):
        tu.SensorCameraConfig(name="a", prim_path="/World/a", pose_relative_to="/World",
                              usd_path="x.usd")  # fmt: skip


# --- phase machine with a fake cuMotion solver ---------------------------------------------------


class FakeScene:
    def __init__(self) -> None:
        self.center = np.array([0.6, -3.66, 0.805])
        self.config = SimpleNamespace(target=SimpleNamespace(name="cube"))
        self.layout = {"table_top_z": 0.78, "objects": {"cube": {"half_extents_m": [0.025] * 3}}}

    def target_center(self) -> np.ndarray:
        return self.center.copy()


class FakeRobot:
    def __init__(self) -> None:
        self.config = FrankaTabletopRobotConfig()
        self.base_pose = (np.array([0.304, -3.75, 0.78]), tu.yaw_quat(0.0))
        self.commands: list[tuple[np.ndarray, float, np.ndarray]] = []
        self.hand = np.array([0.6, -3.66, 0.91])

    def command(self, q, finger, qd=None) -> None:
        self.commands.append((np.asarray(q), float(finger), np.asarray(qd)))

    def link_poses(self):
        return np.array([self.hand, self.hand, self.hand]), np.tile(tu.yaw_quat(0.0), (3, 1))


def _fake_planner(**overrides) -> CumotionPickPlacePlanner:
    planner = CumotionPickPlacePlanner(CumotionPickPlacePlannerConfig(**overrides))
    planner.dt = 1 / 120
    planner._base = (np.array([0.304, -3.75, 0.78]), tu.yaw_quat(0.0))
    planner.solved = []

    def solve(q0, target_base, linear):
        planner.solved.append((np.asarray(target_base), linear))
        steps = 13
        positions = np.linspace(q0, q0 + 0.01, steps)
        return (positions, np.full((steps, 7), 1.2)), "SUCCESS"

    planner._solve = solve
    return planner


def test_phase_machine_runs_the_source_sequence_and_keeps_the_grasp_offset():
    planner, scene, robot = _fake_planner(), FakeScene(), FakeRobot()
    planner.begin_episode(scene, robot, [0.60, -3.90])
    np.testing.assert_allclose(planner.place_center, [0.60, -3.90, 0.805])
    phases = []
    for _ in range(5000):
        phase = planner.state["phase"]
        done = planner.on_physics_step(robot)
        phases.append(phase)
        if done:
            break
    assert planner.phase == "done" and planner.result()["sequence_completed"]
    order = list(dict.fromkeys(phases))
    assert order == list(tu.PHASES[:-1])
    assert [row["phase"] for row in planner.rows] == list(tu.PHASES[:-1])
    assert [linear for _, linear in planner.solved] == [False, True, True, True, True, True]
    # hand measured 0.105 m above the cube centre after closing -> place targets keep it
    np.testing.assert_allclose(planner.grasp_offset_m, [0, 0, 0.105], atol=1e-12)
    lower_base = planner.solved[4][0]
    expected = tu.world_to_base(*planner._base, np.array([0.60, -3.90, 0.805 + 0.105]))
    np.testing.assert_allclose(lower_base, expected, atol=1e-12)
    fingers = {p: {f for (_, f, _), q in zip(robot.commands, phases, strict=True) if q == p}
               for p in ("start", "close", "transfer", "open", "rest")}  # fmt: skip
    assert fingers["start"] == {0.04} and fingers["transfer"] == {0.0} and fingers["rest"] == {0.04}
    assert fingers["close"] == {0.0} and fingers["open"] == {0.04}
    durations = {p: phases.count(p) for p in ("start", "close", "open", "rest")}
    assert durations == {"start": 30, "close": 72, "open": 24, "rest": 120}
    # velocity feed-forward only while moving; holds command zero velocity
    holds = [qd for (_, _, qd), p in zip(robot.commands, phases, strict=True) if p == "close"]
    assert all(np.all(qd == 0) for qd in holds)


def test_planning_failure_stops_the_episode():
    planner, scene, robot = _fake_planner(), FakeScene(), FakeRobot()
    planner._solve = lambda q0, target, linear: (None, "INVERSE_KINEMATICS_FAILURE")
    planner.begin_episode(scene, robot, [0.60, -3.90])
    steps = 0
    while not planner.on_physics_step(robot):
        steps += 1
    assert planner.phase == "failed" and steps == 30
    assert planner.result()["failure"] == "approach_pick: INVERSE_KINEMATICS_FAILURE"
    with pytest.raises(ValueError):
        CumotionPickPlacePlannerConfig(linear_phases=["close"]).validate_phases()


# --- acceptance ----------------------------------------------------------------------------------


def _episode_log(drop_during_transfer: bool = False) -> tuple[dict, dict, dict]:
    phases, hand, obj = [], [], []
    start = np.array([0.6, -3.66, 0.805])
    place = np.array([0.6, -3.90, 0.805])
    offset = np.array([0.0, 0.0, 0.103])
    segments = [("start", 30), ("approach_pick", 60), ("descend", 40), ("close", 72),
                ("lift", 40), ("transfer", 60), ("lower", 40), ("open", 24),
                ("retreat", 40), ("rest", 120)]  # fmt: skip
    for name, steps in segments:
        for i in range(steps):
            s = (i + 1) / steps
            phases.append(tu.PHASE_INDEX[name])
            if name in ("start", "approach_pick", "descend", "close"):
                o = start
                h = start + offset + [0, 0, 0.12 * (1 - s) if name == "descend" else 0.12]
                if name == "close":
                    h = start + offset
            elif name == "lift":
                o = start + [0, 0, 0.12 * s]
                h = o + offset
            elif name == "transfer":
                o = start + [0, 0, 0.12] + (place - start) * s
                h = o + offset
                if drop_during_transfer and s > 0.5:
                    o = o - [0, 0, 0.03]
            elif name == "lower":
                o = place + [0, 0, 0.12 * (1 - s)]
                h = o + offset
            else:
                o = place
                h = place + offset + ([0, 0, 0.12 * s] if name == "retreat" else 0)
                if name == "rest":
                    h = place + offset + [0, 0, 0.12]
            hand.append(h)
            obj.append(o)
    n = len(phases)
    finger = np.full((n, 2), 0.04)
    closed = (np.asarray(phases) >= tu.PHASE_INDEX["close"]) & (
        np.asarray(phases) < tu.PHASE_INDEX["open"]
    )
    finger[closed] = 0.025
    log = {
        "time": np.arange(n) / 120,
        "phase_index": np.asarray(phases),
        "object_pos": np.asarray(obj, float),
        "object_quat": np.tile([1.0, 0, 0, 0], (n, 1)),
        "hand_pos": np.asarray(hand, float),
        "hand_quat": np.tile([0.0, 1, 0, 0], (n, 1)),
        "finger_q": finger,
        "distractor_pos": np.tile([[0.45, -3.45, 0.83]], (n, 1, 1)),
    }
    layout = {"table_top_z": 0.78, "object_half_extents_m": [0.025] * 3,
              "place_target_xy": [0.6, -3.9], "gripper_open_m": 0.04}  # fmt: skip
    contacts = {"finger_left|target": 2.0, "finger_right|target": 2.1, "table|target": 5.0,
                "distractor|table": 3.0, "base|table": 0.0}  # fmt: skip
    return log, layout, contacts


def test_evaluate_pickplace_passes_and_detects_failures():
    acceptance = tu.PickPlaceAcceptanceConfig()
    log, layout, contacts = _episode_log()
    ok = tu.evaluate_pickplace(log, layout=layout, contacts=contacts, acceptance=acceptance,
                               sequence_completed=True)  # fmt: skip
    assert ok["success"], ok["failed_checks"]
    assert ok["metrics"]["object_lift_m"] == pytest.approx(0.12)
    assert ok["metrics"]["place_error_m"] == pytest.approx(0.0, abs=1e-12)
    drop, _, _ = _episode_log(drop_during_transfer=True)
    bad = tu.evaluate_pickplace(drop, layout=layout, contacts=contacts, acceptance=acceptance,
                                sequence_completed=True)  # fmt: skip
    assert "object_held_without_slip" in bad["failed_checks"]
    touched = {**contacts, "arm|table": 0.5, "finger_left|table": 0.2}
    bad = tu.evaluate_pickplace(log, layout=layout, contacts=touched, acceptance=acceptance,
                                sequence_completed=True)  # fmt: skip
    assert {"arm_links_collision_free", "fingers_touch_only_target"} <= set(bad["failed_checks"])
    one_finger = {k: v for k, v in contacts.items() if k != "finger_right|target"}
    bad = tu.evaluate_pickplace(log, layout=layout, contacts=one_finger, acceptance=acceptance,
                                sequence_completed=False)  # fmt: skip
    assert {"both_fingers_contacted_target", "sequence_completed"} <= set(bad["failed_checks"])


def test_contact_monitor_accumulates_by_phase():
    monitor = tu.ContactMonitor.__new__(tu.ContactMonitor)
    monitor.time, monitor.phase = 0.0, "close"
    monitor.reset()
    monitor.add("finger_left|target", 0.5)
    monitor.time, monitor.phase = 1.0, "lift"
    monitor.add("finger_left|target", 0.25)
    summary = monitor.summary()
    assert summary["impulses_ns"] == {"finger_left|target": 0.75}
    assert summary["by_phase_ns"]["lift"] == {"finger_left|target": 0.25}
    assert summary["first_time_s"]["finger_left|target"] == 0.0
    assert summary["last_time_s"]["finger_left|target"] == 1.0


@pytest.mark.parametrize("corrupt", [np.nan, np.inf, -np.inf])
def test_independent_validation_rejects_nonfinite_object_pose(corrupt):
    from scripts.tabletop_pickplace.validate_tabletop_pickplace import _compare

    original = np.tile([0.6, -3.9, 0.805, 1, 0, 0, 0], (4, 1))
    damaged = original.copy()
    damaged[2, 0] = corrupt
    with pytest.raises(ValueError, match="non-finite"):
        _compare("observation.object_pose", damaged, original, 1e-5)


def test_independent_validation_rejects_broadcastable_pose_shape():
    from scripts.tabletop_pickplace.validate_tabletop_pickplace import _compare

    original = np.tile([0.6, -3.9, 0.805, 1, 0, 0, 0], (4, 1))
    with pytest.raises(ValueError, match="shape"):
        _compare("observation.object_pose", original[:1], original, 1e-5)


# --- configs -------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["tabletop_pickplace", "tabletop_replay"])
def test_existing_formal_output_is_never_deleted(tmp_path, monkeypatch, backend):
    from real2sim2real.job.job_tabletop_replay import TabletopReplayJob, TabletopReplayJobConfig

    monkeypatch.chdir(tmp_path)
    output = tmp_path / "outputs" / backend / "accepted"
    output.mkdir(parents=True)
    marker = output / "result.json"
    marker.write_text('{"success": true}')
    cls, config_cls = (
        (TabletopPickPlaceJob, TabletopPickPlaceJobConfig)
        if backend == "tabletop_pickplace"
        else (TabletopReplayJob, TabletopReplayJobConfig)
    )
    try:
        apply_settings_file(ROOT / "configs" / backend / "franka_physx_home000.toml")
        for overwrite in (False, True):
            with pytest.raises((ValueError, FileExistsError)):
                cls(config_cls(output_dir=str(output), overwrite_dataset=overwrite))
            assert marker.read_text() == '{"success": true}'
    finally:
        reset_settings()


def test_debug_output_can_be_replaced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "outputs/debug/tabletop_pickplace/probe"
    output.mkdir(parents=True)
    (output / "old.txt").write_text("old debug output")
    try:
        apply_settings_file(CONFIGS / "franka_physx_home000_debug.toml")
        job = TabletopPickPlaceJob(
            TabletopPickPlaceJobConfig(output_dir=str(output), overwrite_dataset=True)
        )
        assert not output.exists()
        job.close()
    finally:
        reset_settings()


@pytest.mark.parametrize("validation_passes", [True, False])
def test_collector_publishes_only_after_independent_validation(
    tmp_path, monkeypatch, validation_passes
):
    from scripts.tabletop_pickplace import collect_tabletop_pickplace as collector
    from scripts.tabletop_pickplace import validate_tabletop_pickplace as validator

    output = tmp_path / "outputs/tabletop_pickplace/formal"
    run = tmp_path / "outputs/debug/tabletop_pickplace/collector_test"
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    monkeypatch.setenv("RSR_JOB__output_dir", str(output))

    class Worker:
        pid = 123
        returncode = 0

        def __init__(self, args, **kwargs):
            staging = Path(kwargs["env"]["RSR_JOB__output_dir"])
            episode = staging / "episode_000"
            meta = episode / "lerobot_v3/meta"
            physics = meta / "physics/episode_000000"
            physics.mkdir(parents=True)
            result = {
                "success": True,
                "dataset_use": "demonstration",
                "episode": 0,
                "dataset": str(episode / "lerobot_v3"),
            }
            (staging / "result.json").write_text(
                json.dumps({"success": True, "episodes": [result], "num_episodes": 1})
            )
            (episode / "result.json").write_text(json.dumps(result))
            config = {"job": {"output_dir": str(staging)}}
            (meta / "task_outcome.json").write_text(
                json.dumps({"result": result, "configuration": config})
            )
            (physics / "configuration.json").write_text(json.dumps(config))

        def poll(self):
            return self.returncode

    def validate(staging):
        assert not output.exists(), "An unvalidated run must not be published"
        if not validation_passes:
            raise ValueError("corrupt video")
        (staging / "validation.json").write_text('{"validated": true}')
        (staging / "episode_000/validation.json").write_text('{"validated": true}')
        return {"validated": True}

    monkeypatch.setattr(collector.subprocess, "Popen", Worker)
    monkeypatch.setattr(validator, "validate", validate)
    try:
        if validation_passes:
            assert (
                collector.collect(CONFIGS / "franka_physx_home000.toml", run, monotonic_clock=False)
                == output
            )
            result = json.loads((output / "episode_000/result.json").read_text())
            assert result["dataset"] == str(output / "episode_000/lerobot_v3")
            meta = output / "episode_000/lerobot_v3/meta"
            actual = json.loads((meta / "physics/episode_000000/configuration.json").read_text())
            assert actual["job"]["output_dir"] == str(run / "episodes")
            outcome = json.loads((meta / "task_outcome.json").read_text())
            assert outcome["configuration"] == actual
            assert outcome["result"]["dataset"] == result["dataset"]
        else:
            with pytest.raises(ValueError, match="corrupt video"):
                collector.collect(CONFIGS / "franka_physx_home000.toml", run, monotonic_clock=False)
            assert not output.exists()
            assert (run / "episodes/episode_000/result.json").is_file()
    finally:
        reset_settings()


@pytest.mark.parametrize("name", [p.name for p in sorted(CONFIGS.glob("*.toml"))])
def test_configs_route_and_follow_capture_rules(name):
    try:
        apply_settings_file(CONFIGS / name)
        world = IsaacsimWorldConfig.from_settings("sim_world")
        scene = TabletopHomeSceneConfig.from_settings("scene")
        robot = FrankaTabletopRobotConfig.from_settings("robot")
        job = TabletopPickPlaceJobConfig.from_settings("job")
        planner = CumotionPickPlacePlannerConfig.from_settings("planner")
        ensure_job_backends()
        assert JOB_REGISTRY.get(job.backend) is TabletopPickPlaceJob
        assert world.physics_engine == "physx" and world.device == "cpu"
        assert round(1 / world.physics_dt) == 120 and (120 / job.recorder.fps).is_integer()
        assert "isaacsim.robot_motion.cumotion" in world.extensions
        assert planner.backend == "cumotion_pickplace" and planner.tool_frame == "panda_hand"
        cameras = [*scene.cameras, *robot.cameras]
        assert [c.name for c in cameras] == ["overhead_cam", "wrist_cam"]
        debug = name.endswith("_debug.toml")
        resolution = (640, 480) if debug else (1024, 768)
        assert all(c.resolution == resolution for c in cameras)
        assert (world.width, world.height) == resolution
        assert job.recorder.fps == (15 if debug else 30)
        assert world.renderer == ("MinimalRendering" if debug else "PathTracing")
        assert job.recorder.capture_mode == "parallel_cameras"
        assert all(not c.usd_path and c.focal_length == 4.26 for c in cameras)
        assert all(o.usd_path.startswith("assets/TabletopPickplace/") for o in scene.objects)
        assert world.background_usd_path.startswith("assets/TabletopPickplace/")
        assert [o.name for o in scene.objects] == ["cube", "apple"]
        assert scene.target.mass_kg == 0.1 and job.num_episodes == 2
        assert job.object_jitter_seed == 42 and job.object_jitter_xy_m == 0.12
        assert job.output_dir.startswith(
            "outputs/debug/tabletop_pickplace/" if debug else "outputs/tabletop_pickplace/"
        )
        assert job.overwrite_dataset is debug and job.recorder.save_failed_episodes is debug
    finally:
        reset_settings()
