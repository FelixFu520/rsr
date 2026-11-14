"""Kit-free checks of the battery disassembly task: kinematics, plans, proxies, configs, checks."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from real2sim2real.core.config import apply_settings_file, reset_settings
from real2sim2real.job.job_battery_disassembly import (
    BatteryAcceptanceConfig,
    BatteryDisassemblyJob,
    BatteryDisassemblyJobConfig,
    evaluate_battery,
)
from real2sim2real.planner.planner_battery_disassembly import (
    ArmPlanner,
    BatteryUnscrewPlannerConfig,
    grasp_targets,
    plan_gripper,
    plan_screwdriver,
)
from real2sim2real.robot.robot_battery_disassembly import GRIPPER_IN_TOOL0, Kr210DualRobotConfig
from real2sim2real.scene.scene_battery_disassembly import BatteryWorkcellSceneConfig
from real2sim2real.simworld.isaacsim import IsaacsimWorld, IsaacsimWorldConfig
from real2sim2real.utils import battery_disassembly_utils as bu
from scripts.battery_disassembly import collect_battery_disassembly as collector

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/BatteryDisassembly/Robot/KR210_L150/kr210.urdf"
CONFIGS = ROOT / "configs/battery_disassembly"
needs_urdf = pytest.mark.skipif(not URDF.is_file(), reason="run prepare_battery_disassembly_assets")

# layout measured on the composed c1003b1a1 stage (scene at (0.5103, 0.4987, 0))
SCENE = np.eye(4)
SCENE[:3, 3] = (0.5102575, 0.4987062, 0.0)
LAYOUT = {
    "scene_matrix": SCENE,
    "screw_heads": np.array(
        [
            [-0.0909, 0.9115, 0.8235],
            [-0.0909, 0.0859, 0.8235],
            [1.1057, 0.9115, 0.8235],
            [1.1057, 0.0859, 0.8235],
        ]
    ),  # fmt: skip
    "basket_inner_lo": np.array([0.098, 1.128, 0.765]),
    "basket_inner_hi": np.array([0.897, 1.559, 1.069]),
    "cover_com_world": np.array([0.5286, 0.4987, 0.8662]),
    "pallet_world": np.array([0.5103, -1.2513, 0.06]),
    "pallet_top_z": 0.12,
}


@needs_urdf
def test_urdf_chain_matches_kr210_usd_home_pose():
    chain = bu.SerialChain.from_urdf(URDF)
    assert chain.names == [f"joint_a{i}" for i in range(1, 7)]
    tip, origins, _ = chain.fk(np.zeros(6))
    # tool0 and joint_a2 world positions of the USD asset at q = 0
    np.testing.assert_allclose(tip[:3, 3], [2.0800014, 0.0, 1.9447919], atol=1e-6)
    np.testing.assert_allclose(origins[1], [0.35015, -0.0365, 0.75019], atol=1e-5)


@needs_urdf
def test_ik_round_trip_full_and_axis_targets():
    chain = bu.SerialChain.from_urdf(URDF)
    base = bu.transform(bu.yaw_matrix(90.0), (-1.54, 0.5, 0.0))
    tool = bu.ToolFrame(np.array([0.30, 0.0, 0.0]))
    rng = np.random.default_rng(3)
    q_true = np.array([-1.5, 0.6, -0.5, 0.1, 1.3, 0.2])
    target = bu.tcp_pose(chain, base, tool, q_true)
    q, pe, ae = bu.solve_ik(
        chain,
        base,
        tool,
        target[:3, 3],
        rotation=target[:3, :3],
        seed=q_true + rng.normal(0, 0.2, 6),
    )
    assert pe < 2e-4 and ae < math.radians(0.1)
    q, pe, ae = bu.solve_ik(chain, base, tool, (0.5, 0.9, 0.82), axis=(0, 0, -1), seed=q_true)
    pose = bu.tcp_pose(chain, base, tool, q)
    assert pe < 2e-4 and pose[2, 0] == pytest.approx(-1.0, abs=1e-5)


def test_timed_path_ends_at_rest_and_follows_arclength():
    path = bu.TimedPath(bu.joint_blend(np.zeros(6), np.r_[1.0, -0.5, 0, 0, 0, 0.2]), 2.0)
    q0, qd0 = path.sample(0.0)
    q1, qd1 = path.sample(2.0)
    np.testing.assert_allclose(q0, 0.0)
    np.testing.assert_allclose(q1, [1.0, -0.5, 0, 0, 0, 0.2])
    assert np.all(qd0 == 0) and np.all(qd1 == 0)
    q, qd = path.sample(1.0)
    np.testing.assert_allclose(q, 0.5 * q1, atol=1e-9)
    # smoothstep peak velocity 1.5 * distance / duration
    np.testing.assert_allclose(qd, 1.5 * q1 / 2.0, rtol=1e-3)


def test_flange_leaves_square_holes_at_the_screws():
    holes = [(-0.4128, -0.6012), (0.4128, -0.6012), (-0.4128, 0.6012), (0.4128, 0.6012)]
    boxes = bu.flange_boxes(0.4221, (-0.6519, 0.6419), (0.8147, 0.819), holes, 0.0065)

    def covered(x, y):
        return any(b.lo[0] <= x <= b.hi[0] and b.lo[1] <= y <= b.hi[1] for b in boxes)

    assert not any(covered(x, y) for x, y in holes)
    assert covered(0.0, 0.0) and covered(0.4128, -0.62) and covered(0.42, -0.6012)
    area = sum(b.size[0] * b.size[1] for b in boxes)
    assert area == pytest.approx(0.8442 * 1.2938 - 4 * 0.013**2, rel=1e-6)


def test_cover_and_basket_proxies():
    cover = BatteryWorkcellSceneConfig().cover
    boxes = bu.cover_boxes(cover, [(0.4128, 0.6012)])
    centroid = bu.boxes_centroid(boxes)
    assert abs(centroid[0]) < 1e-3 and 0.0 < centroid[1] < 0.05  # raised section is at +y
    assert max(b.hi[2] for b in boxes) == pytest.approx(cover.raised_top_z_m)
    basket = bu.basket_boxes((0, 0, 0), (0.8, 0.4, 0.3), 0.008, 0.015)
    assert len(basket) == 5 and basket[0].hi[2] == pytest.approx(0.015)


def test_screw_recess_quadrants_leave_a_cross():
    screw = BatteryWorkcellSceneConfig().screw
    quads = bu.screw_head_boxes(screw)
    assert len(quads) == 4
    for b in quads:
        assert min(abs(b.lo[0]), abs(b.hi[0])) == pytest.approx(0.5 * screw.recess_width_m)
        assert b.lo[2] == pytest.approx(screw.head_base_m)
    assert screw.head_top_m == pytest.approx(screw.head_base_m + screw.recess_depth_m)


def test_magnet_force_law():
    kw = dict(max_force_n=3.0, range_m=0.006, contact_direction=(0, 0, 1))
    assert not np.any(bu.magnet_force((0, 0, 0.01), (0, 0, 0), **kw))
    np.testing.assert_allclose(bu.magnet_force((0, 0, 0.003), (0, 0, 0), **kw), [0, 0, 1.5])
    np.testing.assert_allclose(bu.magnet_force((0, 0, 0.0), (0, 0, 0.0001), **kw), [0, 0, 2.95])


def test_camera_orientations():
    q = bu.housing_camera_quat(90.0, 0.0)
    forward = -bu.quat_to_matrix(q)[:, 2]
    np.testing.assert_allclose(forward, [0, 1, 0], atol=1e-9)
    q = bu.look_at_camera_quat((0, 0, 1), (1, 0, 1))
    m = bu.quat_to_matrix(q)
    np.testing.assert_allclose(-m[:, 2], [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(m[:, 1], [0, 0, 1], atol=1e-9)
    # gripper root frame is a proper rotation with its z axis along -tool0 X
    assert np.linalg.det(GRIPPER_IN_TOOL0) == pytest.approx(1.0)
    np.testing.assert_allclose(GRIPPER_IN_TOOL0[:, 2], [-1, 0, 0])


def test_unwrap_and_thread_report():
    raw = np.angle(np.exp(1j * np.linspace(0, 20 * math.pi, 400)))
    unwrapped, previous = 0.0, raw[0]
    for value in raw[1:]:
        unwrapped = bu.unwrap_angle(unwrapped, previous, value)
        previous = value
    assert unwrapped == pytest.approx(20 * math.pi, rel=1e-9)
    rows = [dict(turn_rad=0.0, slide_m=0.0), dict(turn_rad=32 * math.pi, slide_m=0.024)]
    report = bu.thread_report(rows, 0.0015)
    assert report["turns"] == pytest.approx(16.0) and report["lead_error_fraction"] < 1e-9


@needs_urdf
def test_nominal_programs_reach_every_target():
    robot = Kr210DualRobotConfig(urdf_path=str(URDF))
    scene = BatteryWorkcellSceneConfig()
    cfg = BatteryUnscrewPlannerConfig()
    chain = bu.SerialChain.from_urdf(URDF)
    bases = {
        key: SCENE @ bu.transform(bu.yaw_matrix(arm.yaw_deg), arm.position_m)
        for key, arm in (("screw", robot.screwdriver_arm), ("grip", robot.gripper_arm))
    }
    screw = ArmPlanner(chain, bases["screw"], bu.ToolFrame(np.array([0.3, 0, 0])), cfg,
                       axis=np.array([0, 0, -1.0]))  # fmt: skip
    phases = plan_screwdriver(screw, LAYOUT, scene.screw, cfg)
    assert [p.kind for p in phases].count("unscrew") == 4
    grip_tool = bu.ToolFrame(np.array([0.1244, 0, 0]), GRIPPER_IN_TOOL0)
    grip = ArmPlanner(chain, bases["grip"], grip_tool, cfg, rotation=bu.yaw_matrix(180.0))
    phases = plan_gripper(grip, LAYOUT, scene.cover, cfg)
    retract = next(p for p in phases if p.name == "retract")
    final = bu.tcp_pose(chain, bases["grip"], grip_tool, retract.path.path[-1])
    targets = grasp_targets(LAYOUT, scene.cover, cfg.gripper)
    np.testing.assert_allclose(final[:3, 3], targets["retract"], atol=5e-4)
    # cover bottom ends 5 mm above the pallet top at the release pose
    drop = targets["place"][2] - targets["grasp"][2]
    assert 0.8147 + drop == pytest.approx(0.12 + cfg.gripper.place_clearance_m, abs=1e-6)


def _record(success: bool) -> dict[str, np.ndarray]:
    n, steps = 4, 200
    t = np.arange(steps) * 0.0167
    slide = np.zeros((steps, n))
    turn = np.zeros((steps, n))
    engaged = np.ones((steps, n))
    spindle = np.zeros(steps)
    for i in range(n):
        a, b = 20 + 30 * i, 45 + 30 * i
        ramp = np.clip((np.arange(steps) - a) / (b - a), 0, 1)
        slide[:, i] = 0.024 * ramp
        turn[:, i] = 2 * math.pi * 16 * ramp
        spindle += -2 * math.pi * 16 * ramp
        engaged[b:, i] = 0.0
    basket = np.array([0.5, 1.3, 0.775])
    screw_pos = np.tile(
        np.r_[[basket + [0.05 * i, 0, 0] for i in range(n)]].reshape(-1), (steps, 1)
    )
    if not success:
        screw_pos[:, 0] = 2.0
    cover = np.tile([0.5, 0.5, 0.8147, 1, 0, 0, 0], (steps, 1)).astype(float)
    cover[100:, :3] = [0.51, -1.25, 0.1203]
    fingers = np.tile(
        np.r_[cover[0, :3] + [0, 0.4, 0.2], cover[0, :3] + [0, -0.4, 0.2]], (steps, 1)
    )
    fingers[100:] = np.r_[cover[100, :3] + [0, 0.4, 0.2], cover[100, :3] + [0, -0.4, 0.2]]
    action = np.zeros((steps, 16))
    action[150:180, 14] = 0.06
    return dict(
        time=t, thread_slide=slide, thread_turn=turn, engaged=engaged, spindle=spindle,
        screw_pos=screw_pos, screw_vel=np.zeros((steps, 3 * n)), cover=cover, fingers=fingers,
        phase_index=np.zeros(steps), action=action,
        finger_p=np.where(action[:, 14] > 0, -0.04, 0.0),
    )  # fmt: skip


def test_evaluate_battery_passes_and_detects_a_lost_screw():
    scene = BatteryWorkcellSceneConfig()
    layout = {**LAYOUT, "pallet_world": np.array([0.51, -1.25, 0.06])}
    kwargs = dict(
        scene_layout=layout, scene_config=scene, acceptance=BatteryAcceptanceConfig(),
        contacts={"screw_1|screw_tool": 1.0, "cover|grip_tool": 50.0, "basket|screw_1": 0.1},
        contact_first_time={}, physics_dt=1 / 240,
    )  # fmt: skip
    ok = evaluate_battery(_record(True), planner_result={"sequence_completed": True}, **kwargs)
    assert ok["success"], ok["failed_checks"]
    bad = evaluate_battery(_record(False), planner_result={"sequence_completed": True}, **kwargs)
    assert bad["failed_checks"] == ["screws_in_basket"]
    kwargs["contacts"] = {**kwargs["contacts"], "screw_arm|workcell": 0.5, "screw_2|workcell": 0.2}
    bad = evaluate_battery(_record(True), planner_result={"sequence_completed": True}, **kwargs)
    assert set(bad["failed_checks"]) == {
        "arm_links_collision_free",
        "screws_touch_only_cover_bit_basket",
    }


@pytest.mark.parametrize("name", [p.name for p in sorted(CONFIGS.glob("*.toml"))])
def test_configs_route_and_keep_capture(name):
    try:
        apply_settings_file(CONFIGS / name)
        world = IsaacsimWorldConfig.from_settings("sim_world")
        scene = BatteryWorkcellSceneConfig.from_settings("scene")
        robot = Kr210DualRobotConfig.from_settings("robot")
        job = BatteryDisassemblyJobConfig.from_settings("job")
        BatteryUnscrewPlannerConfig.from_settings("planner")
        assert world.physics_engine == "physx" and round(1 / world.physics_dt) == 240
        assert (240 / job.recorder.fps).is_integer()
        cameras = [*scene.cameras, *robot.cameras]
        assert [c.name for c in cameras] == ["overview", "screwdriver", "gripper"]
        debug = name.endswith("_debug.toml")
        resolution = (640, 480) if debug else ((1024, 768) if "_1k" in name else (2048, 1536))
        assert all(c.resolution == resolution for c in cameras)
        assert (world.width, world.height) == resolution
        assert job.recorder.fps == (15 if debug else 30)
        assert world.renderer == ("MinimalRendering" if debug else "PathTracing")
        if "_1k" in name:
            assert world.multi_gpu is True and world.active_gpu is None
            assert world.renderer_multithreading is False
            assert job.monotonic_render_budget_clock
            assert job.recorder.capture_mode == "parallel_cameras"
        else:
            assert world.multi_gpu is False and world.active_gpu == 0
            assert world.renderer_multithreading is None
            assert not job.monotonic_render_budget_clock
            assert job.recorder.capture_mode == "viewport"
        if debug:
            assert world.minimal_shading_mode == 2
        else:
            assert world.synchronous_capture and world.pathtracing_denoiser
            assert not world.pathtracing_cached and not world.pathtracing_temporal_denoising
        assert job.output_dir.startswith(
            "outputs/debug/" if debug else "outputs/battery_disassembly/"
        )
        assert job.overwrite_dataset is debug and job.recorder.save_failed_episodes is debug
    finally:
        reset_settings()


@pytest.mark.parametrize("enabled", [None, False, True])
def test_renderer_threading_override_preserves_gpu_selection(enabled):
    """Command recording can be changed without disabling multi-GPU or other backends' defaults."""
    world = IsaacsimWorld(IsaacsimWorldConfig(renderer_multithreading=enabled, multi_gpu=True))
    world._settings = Mock()
    world._settings.get.return_value = world.config.renderer
    world._configure_rendering()
    values = {call.args[0]: call.args[1] for call in world._settings.set_bool.call_args_list}
    if enabled is None:
        assert "/rtx/multiThreading/enabled" not in values
    else:
        assert values["/rtx/multiThreading/enabled"] is enabled
    assert not any("multiGpu" in key or "/mgpu/" in key for key in values)


def test_native_budget_clock_survives_realtime_reversal_and_preserves_other_callers(tmp_path):
    """A backwards wall clock must not make the GPU budget's elapsed time negative."""
    if platform.system() != "Linux" or platform.machine() != "x86_64" or not shutil.which("cc"):
        pytest.skip("native clock regression requires Linux x86_64 and cc")
    fake = tmp_path / "fake_clock.c"
    fake.write_text("""
#define _GNU_SOURCE
#include <stdint.h>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>
static int64_t fake_ns = 1794658305000000000LL;
void rewind_clock(void) { fake_ns -= 1000000000LL; }
int clock_gettime(clockid_t id, struct timespec *t) {
    if (id != CLOCK_REALTIME) return syscall(SYS_clock_gettime, id, t);
    t->tv_sec = fake_ns / 1000000000LL; t->tv_nsec = fake_ns % 1000000000LL;
    return 0;
}
""")
    gpu = tmp_path / "gpu.c"
    gpu.write_text("""
#include <stdint.h>
int64_t now(void) __asm__("_ZNSt6chrono3_V212system_clock3nowEv");
int64_t sample(void) { volatile int64_t value = now(); return value; }
""")
    fake_lib = tmp_path / "fake_clock.so"
    gpu_lib = tmp_path / "libgpu.foundation.plugin.so"
    for source, library in ((fake, fake_lib), (gpu, gpu_lib)):
        subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    environment = os.environ.copy()
    environment["LD_PRELOAD"] = str(fake_lib) + ":" + environment.get("LD_PRELOAD", "")
    script = """
import ctypes, json, sys
ctypes.CDLL("libstdc++.so.6", mode=ctypes.RTLD_GLOBAL)
lib = ctypes.CDLL(None)
wall = getattr(lib, "_ZNSt6chrono3_V212system_clock3nowEv")
wall.restype = ctypes.c_int64
gpu = ctypes.CDLL(sys.argv[1]); gpu.sample.restype = ctypes.c_int64
first, real = gpu.sample(), wall()
lib.rewind_clock()
print(json.dumps({"budget_delta": gpu.sample() - first, "other_delta": wall() - real}))
"""

    def sample():
        result = subprocess.run(
            [sys.executable, "-c", script, str(gpu_lib)],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    baseline = sample()
    assert baseline["budget_delta"] == baseline["other_delta"] == -1_000_000_000
    report = collector._prepare_budget_clock(tmp_path, environment)
    corrected = sample()
    assert corrected["budget_delta"] >= 0
    assert corrected["other_delta"] == -1_000_000_000
    assert report["mode"] == "monotonic_epoch"


def test_requested_native_clock_cannot_silently_run_without_bootstrap(monkeypatch):
    monkeypatch.setattr("ctypes.CDLL", lambda *args: SimpleNamespace())
    job = object.__new__(BatteryDisassemblyJob)
    job.config = BatteryDisassemblyJobConfig(monotonic_render_budget_clock=True)
    with pytest.raises(RuntimeError, match="collect_battery_disassembly.py"):
        job.before_init_world_hook()


@pytest.mark.parametrize(
    "failure", ["native_physics_step", "stale_camera", "empty_rgb", "reduced_pt_samples"]
)
def test_parallel_capture_rejects_unsynchronized_frames_and_restores_physics(
    tmp_path, monkeypatch, failure
):
    """A renderer must not silently advance PhysX or write a stale camera frame."""
    matrix = np.eye(4)
    usd = SimpleNamespace(
        XformCache=lambda: SimpleNamespace(
            GetLocalToWorldTransform=lambda prim: SimpleNamespace(GetInverse=lambda: matrix)
        )
    )
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(UsdGeom=usd))
    values = {"/app/player/playSimulations": True, "/rtx/pathtracing/totalSpp": 32}
    native_steps = [10]
    world = SimpleNamespace(
        physics_steps=10,
        _settings=SimpleNamespace(
            get=values.get, set=values.__setitem__, set_bool=values.__setitem__
        ),
        _manager=SimpleNamespace(get_num_physics_steps=lambda: native_steps[0]),
        config=SimpleNamespace(
            rt_subframes=4,
            renderer="PathTracing" if failure == "reduced_pt_samples" else "RayTracedLighting",
            samples_per_pixel_per_frame=64,
        ),
        update=lambda: None,
        is_playing=lambda: True,
        stage=SimpleNamespace(GetPrimAtPath=lambda path: path),
    )

    def render(**kwargs):
        if failure == "native_physics_step":
            native_steps[0] += 1

    capture = object.__new__(bu.BatteryCameraCapture)
    capture.world = world
    capture.rep = SimpleNamespace(orchestrator=SimpleNamespace(step=render))
    capture.cameras = [SimpleNamespace(name="overview", prim_path="/Camera", resolution=(8, 6))]
    actual = matrix.copy()
    actual[3, 0] = 1.0 if failure == "stale_camera" else 0.0
    rgb = np.empty(0) if failure == "empty_rgb" else np.zeros((6, 8, 4), dtype=np.uint8)
    capture.annotators = [
        (
            SimpleNamespace(get_data=lambda: rgb),
            SimpleNamespace(get_data=lambda: {"cameraViewTransform": actual}),
        )
    ]
    path = tmp_path / "frame.png"
    with pytest.raises(RuntimeError):
        capture.capture({"overview": path})
    assert values["/app/player/playSimulations"] is True
    assert not path.exists()


@pytest.mark.parametrize("physical_ok,video_ok", [(False, True), (True, False), (True, True)])
def test_collector_publishes_only_after_both_acceptance_gates(
    tmp_path, monkeypatch, physical_ok, video_ok
):
    """A zero Kit exit code cannot publish a failed task or an invalid video."""
    root = tmp_path / "repo"
    root.mkdir()
    output = root / "outputs/battery_disassembly/episode"
    run = root / "outputs/debug/battery_disassembly/run"
    config = root / "formal.toml"
    config.write_text(
        f'inherit = "{CONFIGS / "kr210_physx_c1003b1a1.toml"}"\n[job]\noutput_dir = "{output}"\n'
    )
    monkeypatch.setattr(collector, "ROOT", root)

    class FinishedKit:
        pid = 12345
        returncode = 0

        def __init__(self, *args, env, **kwargs):
            staging = Path(env["RSR_JOB__OUTPUT_DIR"])
            meta = staging / "lerobot_v3/meta"
            physics = meta / "physics/episode_000000"
            physics.mkdir(parents=True)
            result = dict(
                success=physical_ok, completed=True, dataset_use="demonstration", images=2
            )
            (staging / "result.json").write_text(json.dumps(result))
            (meta / "task_outcome.json").write_text(json.dumps({"result": result}))
            (meta / "info.json").write_text(json.dumps({
                "fps": 30, "features": {"observation.images.overview": {
                    "dtype": "video", "shape": [1536, 2048, 3],
                }}, "video_path": (
                    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
                ),
            }))  # fmt: skip
            (physics / "configuration.json").write_text(json.dumps({
                "sim_world": IsaacsimWorldConfig.from_settings("sim_world").model_dump(mode="json"),
            }))  # fmt: skip

        def poll(self):
            return self.returncode

    monkeypatch.setattr(collector.subprocess, "Popen", FinishedKit)
    validated = []

    def independent_validation(staging):
        assert not output.exists()
        validated.append(staging)
        video = staging / "lerobot_v3/videos/observation.images.overview/chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"fake video; decoding is the independent validator's responsibility")
        return {"passed": video_ok, "videos": {"overview": str(video)}}

    monkeypatch.setattr(collector, "validate", independent_validation)
    try:
        if physical_ok and video_ok:
            assert collector.collect(config, run) == output
            assert (output / "overview.mp4").is_file()
            assert not (run / "capture").exists()
            report = json.loads((output / "validation.json").read_text())
            assert all(str(output) in path for path in report["videos"].values())
        else:
            with pytest.raises(RuntimeError):
                collector.collect(config, run)
            assert not output.exists()
            assert (run / "capture/result.json").exists()
        assert bool(validated) is physical_ok
        status = json.loads((run / "status.json").read_text())
        assert status["state"] == ("complete" if physical_ok and video_ok else "failed")
    finally:
        reset_settings()
