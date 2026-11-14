"""Kit-free checks for the HALO task: SONIC contract, timing, cameras, contacts, FK, checks."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from real2sim2real.core.config import apply_settings_file, reset_settings
from real2sim2real.job.job_halo import HaloJobConfig, halo_extra_features
from real2sim2real.planner.planner_halo import MotorSmoothingConfig, SonicTrackingPlannerConfig
from real2sim2real.robot.robot_halo import STATE_NAMES, G1SonicRobotConfig
from real2sim2real.scene.scene_halo import HaloFollowCameraConfig, HaloHomeSceneConfig
from real2sim2real.simworld.isaacsim import IsaacsimWorldConfig
from real2sim2real.utils import halo_utils as hu
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder

CONFIGS = Path(__file__).resolve().parents[1] / "configs/halo"


def test_sonic_joint_orders_are_inverse_permutations():
    assert sorted(hu.MUJOCO_JOINT_NAMES) == sorted(hu.ISAACLAB_JOINT_NAMES)
    x = np.arange(29.0)
    np.testing.assert_array_equal(x[hu.MUJOCO_TO_ISAACLAB][hu.ISAACLAB_TO_MUJOCO], x)
    # 7520-22 hip motor: armature × (2π·10 Hz)²; ankle 5020 doubled; action scale 0.25·τ/k.
    assert hu.KPS[0] == pytest.approx(0.025101925 * (20 * math.pi) ** 2)
    assert hu.KPS[4] == pytest.approx(2 * 0.003609725 * (20 * math.pi) ** 2)
    assert hu.ACTION_SCALE[0] == pytest.approx(0.25 * 139.0 / (0.025101925 * (20 * math.pi) ** 2))
    assert len(STATE_NAMES) == 71 and len(set(STATE_NAMES)) == 71


def test_quaternion_matrix_roundtrip_and_6d_columns():
    q = np.array([0.59636781, 0.3799282, -0.3799282, -0.59636781])
    m = hu.quat_wxyz_to_matrix(q)
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(hu.matrix_to_quat_wxyz(m), q / np.linalg.norm(q), atol=1e-7)
    np.testing.assert_array_equal(hu.rotmat_6d(np.arange(9.0).reshape(3, 3)), [0, 1, 3, 4, 6, 7])
    # Head camera mount: optical axis (−Z) points forward and 25° down in the torso frame.
    forward = -m[:, 2]
    assert forward[0] == pytest.approx(math.cos(math.radians(25)), abs=1e-6)
    assert forward[2] == pytest.approx(-math.sin(math.radians(25)), abs=1e-6)


def test_reference_resampling_and_image_schedule_match_halo_counts():
    reference = np.zeros((600, 36))
    reference[:, 3] = 1.0
    reference[:, 0] = np.linspace(0, 2, 600)
    ref = hu.resample_reference(reference, 30, hu.CONTROL_HZ)
    assert len(ref) == 999  # crawl_v3: 999 control rows for 600 reference frames
    np.testing.assert_allclose(np.linalg.norm(ref[:, 3:7], axis=1), 1.0)
    for fps, count in ((30, 600), (15, 300)):
        schedule = hu.image_schedule(999, hu.CONTROL_HZ, fps)
        assert len(schedule) == count and np.all(np.diff(schedule) > 0)
        error = np.abs(np.arange(count) / fps - schedule / hu.CONTROL_HZ)
        assert error.max() <= 1 / (2 * hu.CONTROL_HZ) + 1e-12
    assert (
        len(hu.image_schedule(789, 50, 30)) == 474 and len(hu.image_schedule(1199, 50, 30)) == 720
    )


def test_phase_index_uses_inclusive_reference_frames():
    phases = [{"name": "a", "start": 0, "end": 29}, {"name": "b", "start": 30, "end": 59}]
    assert hu.phase_index(49, phases, control_hz=50, reference_fps=30) == 0  # 0.98 s
    assert hu.phase_index(50, phases, control_hz=50, reference_fps=30) == 1  # 1.00 s
    assert hu.phase_index(500, phases, control_hz=50, reference_fps=30) == 1


def test_motor_filter_limits_arm_speed_and_passes_legs():
    names = hu.MUJOCO_JOINT_NAMES
    settings = MotorSmoothingConfig(enabled=True)
    filt = hu.MotorTargetFilter(np.zeros(29), settings, names)
    step = filt.step(np.full(29, 1.0), 0.02)
    arm = np.array([any(p in n for p in hu.ARM_JOINT_PARTS) for n in names])
    np.testing.assert_array_equal(step[~arm], 1.0)  # legs and waist follow directly
    assert np.all(np.abs(step[arm]) <= settings.max_acceleration_rad_s2 * 0.02**2 + 1e-12)
    off = hu.MotorTargetFilter(np.zeros(29), MotorSmoothingConfig(enabled=False), names)
    np.testing.assert_array_equal(off.step(np.full(29, 0.3), 0.02), 0.3)


def test_look_at_and_follow_camera_point_at_the_target():
    eye, target = np.array([1.55, -1.4, 1.9]), np.array([3.0, -3.1, 0.65])
    rotation = hu.quat_wxyz_to_matrix(hu.look_at_quat_wxyz(eye, target))
    direction = (target - eye) / np.linalg.norm(target - eye)
    np.testing.assert_allclose(-rotation[:, 2], direction, atol=1e-9)
    assert rotation[2, 1] > 0  # image up has a positive world-Z component
    follow = hu.FollowCameraFilter(HaloFollowCameraConfig())
    base = np.array([2.68, -2.76, 0.775, 1, 0, 0, 0])
    first_eye, first_target, _ = follow.step(base, 0.02)
    np.testing.assert_allclose(first_target, [2.68, -2.76 + 0.2, 0.56 * 0.775 + 0.21], atol=1e-12)
    np.testing.assert_allclose(first_eye - first_target, [1.45, 2.0, 0.52], atol=1e-12)
    moved = base.copy()
    moved[1] += 1.0
    _, filtered, _ = follow.step(moved, 0.02)
    assert 0 < filtered[1] - first_target[1] < 0.1  # 0.28 s time constant, not a jump


def test_contact_accumulator_categories_feet_and_penetration():
    prefixes = hu.ContactPrefixes(
        robot="/World/robots/G1/",
        table="/World/scene/Desk/",
        seat="/World/background/sofa_main/",
        home="/World/background/",
    )
    acc = hu.ContactAccumulator(prefixes)
    acc.add("/World/robots/G1/left_ankle_roll_link", "/World/scene/FloorCollider", [0, 0, 3.0], 0.0)
    acc.add("/World/scene/FloorCollider", "/World/robots/G1/right_knee_link", [0, 4.0, 0], -0.002)
    acc.add("/World/robots/G1/pelvis", "/World/background/sofa_main/sofa_254_002", [0, 0, 5.0], 0)
    acc.add("/World/robots/G1/torso_link", "/World/scene/Desk/Asset/top", [1.0, 0, 0], 0)
    acc.add("/World/background/tv", "/World/background/wall", [9.0, 0, 0], -0.1)  # not robot
    np.testing.assert_allclose(acc.foot_impulses, [3.0, 0.0, 4.0])
    np.testing.assert_allclose(acc.vertical_impulses[[hu.GROUND, hu.CHAIR]], [3.0, 5.0])
    out = acc.consume()
    np.testing.assert_allclose(out[:6], [7.0, 1.0, 5.0, 0.0, 0.0, 0.0])
    assert out[6] == pytest.approx(0.002) and acc.max_penetration == pytest.approx(0.002)
    assert np.all(acc.consume()[:7] == 0)
    assert {e["category"] for e in acc.events} == {hu.GROUND, hu.CHAIR, hu.TABLE}
    acc.clear()
    assert not acc.events and acc.max_penetration == 0 and not acc.foot_impulses.any()


def _chain_geometry() -> hu.RobotGeometry:
    """pelvis → link1 revolute about Y (left_hip_pitch_joint), offset 0.5 m along X."""
    a = np.eye(4)
    a[0, 3] = 0.5
    box = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)])
    return hu.RobotGeometry(
        body_names=["pelvis", "link1"],
        group_body=np.array([0, 1]),
        group_names=["pelvis/collisions/box", "link1/collisions/box"],
        group_points=[box, box],
        joints=[(0, 1, a, np.eye(4), np.array([0.0, 1.0, 0.0]), 0)],
    )


def test_geometry_fk_world_points_and_npz_roundtrip(tmp_path):
    geometry = _chain_geometry()
    qpos = np.zeros(36)
    qpos[:3] = [1.0, 2.0, 3.0]
    qpos[3] = 1.0
    qpos[7] = math.pi / 2
    matrices = geometry.body_matrices(qpos)
    np.testing.assert_allclose(matrices[1][:3, 3], [1.5, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(matrices[1][:3, :3] @ [1, 0, 0], [0, 0, -1], atol=1e-12)
    poses = np.array([[1, 2, 3, 1, 0, 0, 0], [1.5, 2, 3, 1, 0, 0, 0]], dtype=float)
    points = geometry.world_points(poses)
    assert points.shape == (16, 3) and points[:, 0].max() == pytest.approx(1.6)
    geometry.save(tmp_path / "g.npz")
    loaded = hu.RobotGeometry.load(tmp_path / "g.npz")
    assert loaded.body_names == geometry.body_names and loaded.collider_count == 2
    np.testing.assert_allclose(loaded.body_matrices(qpos), matrices)
    assert loaded.upper_vertex_mask.sum() == 8  # only the pelvis box is upper body


def _seated_recording(frames: int = 120) -> dict[str, np.ndarray]:
    qpos = np.zeros((frames, 36))
    qpos[:, :3] = [3.69, -3.133, 0.527]
    qpos[:, 3] = 1.0
    torso = np.zeros((frames, 7))
    torso[:, 3] = 0.0
    torso[:, 6] = 1.0  # 180° yaw: torso x axis = world −X
    weight_impulse = 34.0 * 9.81 / 50
    vertical = np.zeros((frames, 6))
    vertical[:, hu.CHAIR] = 0.8 * weight_impulse
    vertical[:, hu.GROUND] = 0.2 * weight_impulse
    contacts = np.zeros((frames, 7))
    contacts[:, hu.CHAIR] = 0.8 * weight_impulse
    return {
        "qpos": qpos,
        "torso_pose": torso,
        "contact_vertical_impulses": vertical,
        "contacts": contacts,
        "foot_support_impulses": np.zeros((frames, 3)),
    }


def test_seated_checks_pass_centered_supported_and_facing_tv():
    rec = _seated_recording()
    kwargs = dict(
        cushion_center_xyz_m=(3.7, -3.13288, 0.37678),
        max_horizontal_center_error_m=0.10,
        max_lateral_center_error_m=0.055,
        robot_mass_kg=34.0,
    )
    checks, metrics = hu.seated_checks(rec, **kwargs)
    assert all(checks.values()), checks
    assert metrics["last_2s_sofa_support_weight_fraction"] == pytest.approx(0.8)
    rec["qpos"][:, 1] += 0.08  # 8 cm sideways: lateral limit 5.5 cm
    assert not hu.seated_checks(rec, **kwargs)[0]["centered_sit"]


def test_desk_frame_and_passage_corridor():
    desk = hu.DeskSpec(
        xy=(2.0, 0.0), yaw_deg=90.0, base_z=0.0, depth_m=0.6, width_m=1.1,
        clearance_z_m=0.785, column_spacing_m=0.98,
    )  # fmt: skip
    np.testing.assert_allclose(
        desk.to_local(np.array([2.0, 0.3, 0.1])), [0.3, 0.0, 0.1], atol=1e-12
    )
    geometry = _chain_geometry()
    frames = 80
    poses = np.zeros((frames, 2, 7))
    poses[:, :, 3] = 1.0
    poses[:, :, 0] = 2.0
    poses[:, 0, 1] = np.linspace(-1.0, 1.0, frames)  # walks along desk x (world +Y)
    poses[:, 1, 1] = poses[:, 0, 1]
    poses[:, :, 2] = 0.3
    rec = {
        "qpos": np.c_[poses[:, 0, :7], np.zeros((frames, 29))],
        "robot_body_poses": poses,
        "torso_pose": poses[:, 0],
    }
    report = hu.desk_passage_report(rec, geometry, desk, mode="crouch", transition_time_s=1.0)
    passage = report["table_passage"]
    assert passage["passed"]
    assert passage["minimum_lateral_corridor_margin_m"] == pytest.approx((0.98 - 0.092) / 2 - 0.1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_recorder_extra_features_scalar_and_vector_columns(tmp_path):
    extras = halo_extra_features(["a", "b"])
    root = tmp_path / "dataset"
    recorder = LeRobotV3Recorder(
        root, fps=15, width=64, height=36, task="halo", joint_names=["s0", "s1"],
        action_names=["a0"], cameras={"head": (64, 36)}, extra_features=extras,
    )  # fmt: skip
    for i in range(3):
        Image.fromarray(np.full((36, 64, 3), 40 * i, np.uint8)).save(recorder.frame_path("head"))
        values = {name: np.zeros(spec["shape"][0]) for name, spec in extras.items()}
        values.update({"next.done": i == 2, "next.success": False, "observation.phase_index": i})
        values["observation.control_frame_index"] = 3 * i
        recorder.add_frame(np.array([i, i]), np.array([i]), values)
    recorder.update_extra(2, "next.success", True)
    with pytest.raises(ValueError):
        recorder.add_frame(np.zeros(2), np.zeros(1), {"next.done": True})
    recorder.finalize(episode_success=True)
    info = json.loads((root / "meta/info.json").read_text())
    assert info["features"]["observation.body_landmarks"]["shape"] == [6]
    assert info["features"]["next.done"] == {"dtype": "bool", "shape": [1], "names": None}
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    assert table["next.success"].to_pylist() == [False, False, True]
    assert table["observation.control_frame_index"].to_pylist() == [0, 3, 6]
    assert str(table.schema.field("observation.phase_index").type) == "int64"
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats["observation.phase_index"]["max"] == [2.0]


def test_recorder_without_extras_keeps_the_original_columns(tmp_path):
    recorder = LeRobotV3Recorder(tmp_path / "d", fps=15, width=64, height=48, task="t")
    with pytest.raises(ValueError):
        recorder.add_frame(np.zeros(9), np.zeros(9), {"next.done": True})
    assert recorder.extra_features == {}


@pytest.mark.parametrize("task", ["crawl", "crouch", "sit"])
def test_debug_configs_inherit_formal_and_override_only_capture(monkeypatch, task):
    try:
        for name, size, fps in (
            (f"g1_physx_{task}.toml", 1024, 30),
            (f"g1_physx_{task}_debug.toml", 640, 15),
        ):
            apply_settings_file(CONFIGS / name)
            world = IsaacsimWorldConfig.from_settings("sim_world")
            scene = HaloHomeSceneConfig.from_settings("scene")
            robot = G1SonicRobotConfig.from_settings("robot")
            job = HaloJobConfig.from_settings("job")
            planner = SonicTrackingPlannerConfig.from_settings("planner")
            assert world.width == scene.third_person.width == robot.head_camera.width == size
            assert (
                world.height
                == scene.third_person.height
                == robot.head_camera.height
                == size * 9 // 16
            )
            assert world.renderer == ("RayTracedLighting" if fps == 15 else "PathTracing")
            assert job.recorder.fps == fps and job.acceptance.episode == task
            assert planner.reference_dir == f"assets/Halo/reference/{task}"
            assert ("debug" in job.output_dir) == name.endswith("_debug.toml")
            assert (scene.desk is None) == (task == "sit")
            assert planner.motor_smoothing.enabled == (task == "crawl")
            assert planner.seated_hold.enabled == (task == "sit")
            assert scene.third_person.mode == ("follow" if task == "crawl" else "fixed")
            assert robot.arm_damping_nms_rad == (4.0 if task == "crawl" else 3.0)
    finally:
        monkeypatch.delenv("RSR_SETTINGS_FILE", raising=False)
        reset_settings()


def test_job_config_rejects_unknown_episode():
    with pytest.raises(ValueError):
        HaloJobConfig.model_validate({"acceptance": {"episode": "jump"}})
