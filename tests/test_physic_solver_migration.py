"""Migrated case routing, smooth motion and physical screw-acceptance failure modes."""

import json
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from real2sim2real.core.config import reset_settings
from real2sim2real.job.base import AutoJob
from real2sim2real.planner.planner_physic_solver import (
    PackagePlanner,
    PackagePlannerConfig,
    ScrewPlanner,
    ScrewPlannerConfig,
)
from real2sim2real.utils.physic_solver_utils import (
    PHYSIC_SOLVER_CASE_NAMES,
    PHYSIC_SOLVER_PACKAGE_CASES,
    package_finger_reference,
    package_grasp_quaternion,
    package_gripper_activation,
    proxy_pose_errors,
    read_obj_triangles,
    shake_displacement,
)


@pytest.mark.parametrize(
    "path", [Path("configs/physic_solver") / f"{name}.toml" for name in PHYSIC_SOLVER_PACKAGE_CASES]
)
def test_migrated_cases_construct_without_kit_and_keep_independent_outputs(
    path, monkeypatch, tmp_path
):
    monkeypatch.setenv("RSR_JOB__OUTPUT_DIR", str(tmp_path / path.stem))
    monkeypatch.setenv("RSR_SETTINGS_FILE", str(path.resolve()))
    reset_settings()
    try:
        job = AutoJob.from_config()
        assert job.sim_world.app is None
        assert job.scene.config.backend == "pickplace_package"
        assert job.robot.config.backend in {"franka_package_pads", "franka_newton"}
        assert job.planner.config.attachment_mode == (
            "spring" if path.stem.endswith("_shake_spring") else "none"
        )
        if path.stem == PHYSIC_SOLVER_CASE_NAMES["v8_offset_x35"]:
            assert job.planner.config.grasp_offset_m == (0.035, 0, 0)
        if path.stem == PHYSIC_SOLVER_CASE_NAMES["v9_mass500g"]:
            assert job.scene.config.garment_mass_kg == 0.5
    finally:
        reset_settings()


def test_flat_configs_match_output_names_and_640_capture_contract():
    root = Path("configs/physic_solver")
    assert not any(path.is_dir() for path in root.iterdir())
    paths = list(root.glob("*.toml"))
    assert {p.stem for p in paths} == set(PHYSIC_SOLVER_CASE_NAMES.values())
    assert len(PHYSIC_SOLVER_PACKAGE_CASES) == 18
    for path in paths:
        config = tomllib.loads(path.read_text())
        assert "inherit" not in config
        assert Path(config["job"]["output_dir"]).name == path.stem
        assert (config["job"]["recorder"]["width"], config["job"]["recorder"]["height"]) == (
            640,
            480,
        )
        for domain in ("scene", "robot"):
            assert all(c["resolution"] == [640, 480] for c in config[domain].get("cameras", []))


def test_recapture_gpu_lanes_cannot_claim_the_same_physics_case(tmp_path):
    from scripts.physic_solver.recapture_physic_solver import claim

    state = tmp_path / "queue.json"
    state.write_text(
        json.dumps(
            dict(
                cases={
                    name: dict(case=name, sdk=True, status="pending")
                    for name in ("first", "second")
                }
            )
        )
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda gpu: claim(state, gpu), (0, 1)))
    assert {r["case"] for r in rows} == {"first", "second"}
    assert all(r["status"] == "physics_running" for r in rows)
    assert all(
        r["status"] == "physics_running" for r in json.loads(state.read_text())["cases"].values()
    )


def test_recapture_bootstrap_overrides_preparation_status_without_kit(tmp_path, monkeypatch):
    from scripts.physic_solver import recapture_physic_solver as collector

    run = tmp_path / "run"
    run.mkdir()
    pointer = tmp_path / "current.json"
    pointer.write_text(
        json.dumps(
            dict(
                run_id="test",
                run_dir=str(run),
                stage="prepared",
                complete=False,
                resolution=[640, 480],
            )
        )
    )
    monkeypatch.setattr(collector, "POINTER", pointer)
    _, state = collector.initialize()
    data = json.loads(state.read_text())
    assert data["stage"] == "fresh_collection"
    assert len(data["cases"]) == 29
    assert sum(row["expected_success"] for row in data["cases"].values()) == 27
    assert sum(row["sdk"] for row in data["cases"].values()) == 18


def test_recapture_resumes_kit_case_with_null_physics_source(tmp_path):
    from scripts.physic_solver.recapture_physic_solver import recover

    state = tmp_path / "queue.json"
    state.write_text(
        json.dumps(
            dict(
                cases={
                    "kit_case": dict(
                        sdk=False,
                        status="recording",
                        physics_source=None,
                        attempt_dir=str(tmp_path / "attempt"),
                    )
                }
            )
        )
    )
    recover(state)
    row = json.loads(state.read_text())["cases"]["kit_case"]
    assert row["status"] == "pending"
    assert row["physics_source"] is None


def test_recapture_first_run_initializes_without_ignored_output_state(tmp_path, monkeypatch):
    from scripts.physic_solver import recapture_physic_solver as collector

    monkeypatch.setattr(collector, "DEBUG", tmp_path / "debug")
    monkeypatch.setattr(collector, "OUTPUT", tmp_path / "formal")
    monkeypatch.setattr(collector, "POINTER", tmp_path / "debug/recapture/current_run.json")
    run, state = collector.initialize()
    assert len(list((run / "prepared_configs").glob("*.toml"))) == 30
    assert len(json.loads(state.read_text())["cases"]) == 29


def test_recapture_adopts_surviving_worker_without_claiming_duplicate(tmp_path, monkeypatch):
    from scripts.physic_solver import recapture_physic_solver as collector

    source = tmp_path / "attempt/physics/case"
    source.mkdir(parents=True)
    for name in ("result.json", "physics_audit.json"):
        (source / name).write_text(json.dumps(dict(success=True)))
    state = tmp_path / "queue.json"
    state.write_text(
        json.dumps(
            dict(
                cases={
                    "case": dict(
                        case="case",
                        sdk=True,
                        gpu=1,
                        status="physics_running",
                        attempt_dir=str(tmp_path / "attempt"),
                        attempts=[dict(fresh_physics=True)],
                    )
                }
            )
        )
    )
    monkeypatch.setattr(collector, "matching_workers", lambda row: [42])
    collector.recover(state)
    row = collector.read(state)["cases"]["case"]
    assert row["status"] == "physics_running" and row["recovered_worker"]
    assert collector.claim(state, 0) is None
    assert collector.claim(state, 1) is None
    monkeypatch.setattr(collector, "matching_workers", lambda row: [])
    collector.adopt_worker(tmp_path, state, row)
    row = collector.read(state)["cases"]["case"]
    assert row["status"] == "physics_passed" and not row["recovered_worker"]
    assert len(row["attempts"]) == 1 and row["physics_source"] == str(source)


def test_shake_starts_and_ends_at_rest_and_rotations_preserve_norm():
    amplitude = np.array([0.015, 0.01, 0.008, 12, 6])
    for t in (0, 2):
        np.testing.assert_array_equal(shake_displacement(t, 2, 1.5, amplitude), np.zeros(5))
        near = t + (1e-5 if t == 0 else -1e-5)
        assert np.linalg.norm(shake_displacement(near, 2, 1.5, amplitude)) / 1e-5 < 1e-5
    for yaw, pitch in ((0, 0), (45, 0), (90, 0), (30, 10), (12, 6)):
        assert np.linalg.norm(package_grasp_quaternion(yaw, pitch)) == pytest.approx(1)
    np.testing.assert_allclose(package_grasp_quaternion(), [2**-0.5, 2**-0.5, 0, 0])
    assert PackagePlannerConfig().attachment_mode == "none"
    assert PackagePlannerConfig().shake_s == 0
    assert PackagePlannerConfig().rigid_soft_coupling == "one_way"
    assert PackagePlannerConfig().proxy_mass_scale == 1


def test_proxy_coupling_requires_feedback_capable_rigid_execution():
    with pytest.raises(ValueError, match="native CPU MuJoCo"):
        PackagePlannerConfig(rigid_soft_coupling="proxy")
    with pytest.raises(ValueError, match="no jaw assistance"):
        PackagePlannerConfig(rigid_soft_coupling="proxy", mujoco_cpu=True, attachment_mode="spring")
    assert PackagePlannerConfig(rigid_soft_coupling="proxy", mujoco_cpu=True)
    for scale in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            PackagePlannerConfig(proxy_mass_scale=scale)
    with pytest.raises(ValueError, match="two-way"):
        PackagePlannerConfig(grip_preload_n=20)


def test_preload_changes_actuator_force_and_preserves_the_open_reference():
    kp, nominal, preload = 6000, 0.0, 20
    assert package_finger_reference(0.8, 0.04, nominal, preload, kp) == pytest.approx(0.04)
    reference = package_finger_reference(0.1, 0.04, nominal, preload, kp)
    actual, velocity, kd = 0.002, 0.001, 60
    baseline_force = kp * (nominal - actual) - kd * velocity
    biased_force = kp * (reference - actual) - kd * velocity
    assert biased_force == pytest.approx(baseline_force - preload)
    assert reference == pytest.approx(-0.0033333333333333335)
    assert package_finger_reference(0.1, 0.04, 0.005, 0, kp) == pytest.approx(0.005)


def test_release_compensation_changes_robot_path_without_moving_acceptance_goal():
    schedules = []
    for compensation in ((0.0, 0.0), (0.02, -0.2)):
        config = PackagePlannerConfig(
            align_place_to_com=True, release_compensation_xy_m=compensation
        )
        planner = PackagePlanner(config)
        planner.scene = SimpleNamespace(config=SimpleNamespace(target_xy=(0.1, 0.0)))
        planner.runtime = SimpleNamespace(tcp_position=lambda: np.array([0.1, 0.175, 1.2]))
        planner._schedule = [
            ("lift", 1.0, np.array([0.1, 0.175, 1.2]), True),
            ("transfer", 1.0, np.array([0.1, 0.055, 1.2]), True),
            ("release", 1.0, np.array([0.1, 0.055, 1.0]), False),
        ]
        planner._align_placement(np.array([0.1, 0.15, 1.0]))
        assert planner.scene.config.target_xy == (0.1, 0.0)
        assert config.place_offset_m == (0.0, -0.12, 0.0)
        schedules.append(np.array([row[2] for row in planner._schedule]))
    np.testing.assert_array_equal(schedules[1][0], schedules[0][0])
    np.testing.assert_allclose(schedules[1][1:] - schedules[0][1:], [[0.02, -0.2, 0]] * 2)
    with pytest.raises(ValueError, match="measured COM"):
        PackagePlannerConfig(release_compensation_xy_m=(0.0, -0.2))
    for invalid in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            PackagePlannerConfig(align_place_to_com=True, release_compensation_xy_m=(invalid, 0))


@pytest.mark.parametrize("position_error", [0.0, 0.0066])
def test_closed_grasp_checkpoint_runs_real_fk_and_rejects_position_error(tmp_path, position_error):
    import json

    from scripts.physic_solver.check_package_matrix import PackageBatchPhysicsJob

    urdf = tmp_path / "fixed_hand.urdf"
    urdf.write_text(
        '<robot name="test"><link name="base"/><link name="fr3_hand"/>'
        '<joint name="tool" type="fixed"><parent link="base"/>'
        '<child link="fr3_hand"/>'
        f'<origin xyz="{position_error} 0 0.1609" '
        'rpy="3.141592653589793 0 1.5707963267948966"/></joint></robot>'
    )
    configuration = dict(
        scene=dict(target_xy=[0.1, 0]),
        robot=dict(
            urdf_path=str(urdf),
            base_position=[0.1, 0.175, 0.7],
            base_orientation=[1, 0, 0, 0],
            initial_joint_positions=[0] * 9,
        ),
        planner=dict(
            grasp_yaw_deg=0,
            grasp_pitch_deg=0,
            grasp_offset_m=[0, 0.175, 0],
            grasp_height_m=0.0575,
        ),
    )
    (tmp_path / "configuration.json").write_text(json.dumps(configuration))
    case = SimpleNamespace(
        directory=tmp_path,
        planner=SimpleNamespace(
            rows=[dict(phase="close", time_s=0, tcp_m=[0.1 + position_error, 0.175, 0.7575])]
        ),
        joints=[np.zeros(9)],
        times=[0],
    )
    if position_error:
        with pytest.raises(RuntimeError, match="measured grasp pose"):
            PackageBatchPhysicsJob._check_grasp_pose_checkpoint(case)
    else:
        PackageBatchPhysicsJob._check_grasp_pose_checkpoint(case)
    report = json.loads((tmp_path / "grasp_pose_checkpoint/grasp_pose_audit.json").read_text())
    assert report["success"] == (position_error == 0)
    assert report["thresholds"]["position_error_m"] == 0.005
    assert report["fk_recorded_tcp_error_m"] < 1e-10


def test_proxy_pose_audit_handles_antipodes_and_detects_geometry_mismatch():
    source = np.array([[0, 0, 0, 0, 0, 0, 1.0], [1, 2, 3, 0, 0, 0, 1.0]])
    proxy = source.copy()
    proxy[:, 3:] *= -2  # Same rotation with an antipodal, non-unit representation.
    assert proxy_pose_errors(source, proxy) == pytest.approx((0, 0))
    proxy[1, 0] += 0.002
    proxy[1, 3:] = [0, 0, np.sin(np.deg2rad(2) / 2), np.cos(np.deg2rad(2) / 2)]
    assert proxy_pose_errors(source, proxy) == pytest.approx((0.002, np.deg2rad(2)))
    proxy[0, 3:] = 0
    with pytest.raises(ValueError, match="norms"):
        proxy_pose_errors(source, proxy)


def test_package_duration_includes_lift_hold_and_post_shake_hold(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "RSR_SETTINGS_FILE",
        str(Path("configs/physic_solver/mujoco_vbd_package01_center_shake_spring.toml").resolve()),
    )
    monkeypatch.setenv("RSR_JOB__OUTPUT_DIR", str(tmp_path))
    reset_settings()
    try:
        assert AutoJob.from_config().planner.expected_duration_s == pytest.approx(17.8)
    finally:
        reset_settings()


def test_obj_reader_triangulates_faces_and_supports_negative_indices(tmp_path):
    obj = tmp_path / "square.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4/1 -3/2 -2/3 -1/4\n")
    points, faces = read_obj_triangles(obj)
    assert points.shape == (4, 3)
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])


def test_optional_jaw_ramp_limits_target_speed_and_keeps_legacy_endpoints():
    from pydantic import ValidationError

    assert package_gripper_activation(0, 0, closed=True) == 0.1
    assert package_gripper_activation(0, 0, closed=False) == 0.8
    times = np.linspace(0, 0.6, 1001)
    activation = np.array([package_gripper_activation(t, 0.6, closed=True) for t in times])
    opening = 0.00625 + (activation - 0.1) / 0.7 * (0.04 - 0.00625)
    assert opening[0] == pytest.approx(0.04)
    assert opening[-1] == pytest.approx(0.00625)
    assert np.all(np.diff(opening) <= 1e-12)
    assert np.max(abs(np.diff(opening) / np.diff(times))) < 0.085
    with pytest.raises(ValidationError):
        PackagePlannerConfig(close_s=0.5, gripper_close_ramp_s=0.6)


def test_screw_rejects_empty_rotation_and_unthreaded_push_through():
    planner = ScrewPlanner(ScrewPlannerConfig())
    planner.scene = SimpleNamespace(config=SimpleNamespace(thread_pitch_m=0.0025))
    planner.runtime = SimpleNamespace(
        maximum_contacts=120, newton=SimpleNamespace(__version__="test")
    )
    planner.state["done"] = True
    planner.rows = [
        dict(phase="screw", axial_displacement_m=0, spindle_angle_rad=0, contacts=100),
        dict(phase="screw", axial_displacement_m=0, spindle_angle_rad=4 * np.pi, contacts=100),
        dict(phase="hold", axial_displacement_m=0, spindle_angle_rad=6 * np.pi, contacts=100),
        dict(phase="hold", axial_displacement_m=0, spindle_angle_rad=6 * np.pi, contacts=100),
    ]
    result = planner.result()
    assert not result["success"]
    assert result["checks"]["spindle_rotated"]
    assert not result["checks"]["bolt_advanced"]
    assert not result["checks"]["thread_lead_matches"]
    planner.rows = [
        dict(phase="engage", axial_displacement_m=0, spindle_angle_rad=0, contacts=100),
        dict(phase="screw", axial_displacement_m=-0.006, spindle_angle_rad=0, contacts=100),
        dict(phase="screw", axial_displacement_m=-0.011, spindle_angle_rad=4 * np.pi, contacts=100),
        dict(phase="hold", axial_displacement_m=-0.0135, spindle_angle_rad=6 * np.pi, contacts=100),
        dict(phase="hold", axial_displacement_m=-0.0135, spindle_angle_rad=6 * np.pi, contacts=100),
    ]
    result = planner.result()
    assert result["checks"]["thread_lead_matches"]
    assert result["checks"]["bolt_advanced"]
    assert not result["checks"]["no_unthreaded_push_through"]
    assert not result["success"]


def test_screw_backends_construct_lazily(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "RSR_SETTINGS_FILE",
        str(Path("configs/physic_solver/mujoco_hydroelastic_screw01_rotate.toml").resolve()),
    )
    monkeypatch.setenv("RSR_JOB__OUTPUT_DIR", str(tmp_path))
    reset_settings()
    try:
        job = AutoJob.from_config()
        assert job.sim_world.app is None
        assert job.planner.runtime is None
        assert job.robot.config.backend == "screw_spindle"
        assert job.planner.expected_duration_s == 9.0
    finally:
        reset_settings()


def test_assisted_springs_apply_finite_forces_and_release_without_state_overwrites():
    wp = pytest.importorskip("warp")
    from real2sim2real.utils.physic_solver_utils import JawSpringAssistance

    model = SimpleNamespace(device="cpu")
    settings = SimpleNamespace(
        attachment_vertices_per_group=1,
        attachment_stiffness_n_m=30.0,
        attachment_damping_n_s_m=0.015,
        attachment_max_force_n=0.5,
    )
    assistance = JawSpringAssistance(model, settings)
    assistance.ids.assign(np.array([0, -1], np.int32))
    assistance.enabled.fill_(1)
    old = SimpleNamespace(
        particle_q=wp.array([[0.0, 0, 0]], dtype=wp.vec3, device="cpu"),
        particle_qd=wp.zeros(1, dtype=wp.vec3, device="cpu"),
        particle_f=wp.zeros(1, dtype=wp.vec3, device="cpu"),
        body_q=wp.array([[0.0, 0, 0, 0, 0, 0, 1]], dtype=wp.transform, device="cpu"),
    )
    new = SimpleNamespace(
        body_q=wp.array([[0.01, 0, 0, 0, 0, 0, 1]], dtype=wp.transform, device="cpu")
    )
    initial = old.particle_q.numpy().copy()
    assistance.apply(old, new, 0.01)
    np.testing.assert_allclose(old.particle_f.numpy(), [[0.315, 0, 0]], atol=1e-6)
    np.testing.assert_array_equal(old.particle_q.numpy(), initial)
    np.testing.assert_array_equal(old.particle_qd.numpy(), [[0, 0, 0]])
    new.body_q.assign(np.array([[1.0, 0, 0, 0, 0, 0, 1]], np.float32))
    old.particle_f.zero_()
    assistance.apply(old, new, 0.01)
    np.testing.assert_allclose(old.particle_f.numpy(), [[0.5, 0, 0]], atol=1e-6)
    np.testing.assert_array_equal(old.particle_q.numpy(), initial)
    assistance.release()
    old.particle_f.zero_()
    assistance.apply(old, new, 0.01)
    np.testing.assert_array_equal(old.particle_f.numpy(), [[0, 0, 0]])


@pytest.mark.parametrize("name", ["box01", "box02"])
def test_original_cartons_use_a_declared_long_stroke_gripper(name, monkeypatch, tmp_path):
    monkeypatch.setenv(
        "RSR_SETTINGS_FILE",
        str(
            (
                Path("configs/physic_solver") / f"{PHYSIC_SOLVER_CASE_NAMES['v10_' + name]}.toml"
            ).resolve()
        ),
    )
    monkeypatch.setenv("RSR_JOB__OUTPUT_DIR", str(tmp_path))
    reset_settings()
    try:
        job = AutoJob.from_config()
        assert job.sim_world.app is None
        assert job.scene.config.backend == "pickplace_carton"
        assert job.robot.config.maximum_finger_opening_m == 0.13
        assert job.planner.config.open_finger_m == 0.125
        assert len(job.robot.recording_joint_names) == 9
    finally:
        reset_settings()


def test_long_jaw_ik_jacobian_matches_finite_differences():
    from real2sim2real.planner.planner_physic_solver import _LongJawIK
    from real2sim2real.robot.robot_physic_solver import CartonGripperRobot, CartonGripperRobotConfig

    config = CartonGripperRobotConfig()
    if not Path(config.urdf_path).is_file():
        pytest.skip("Prepare task assets before checking the real FR3 long-jaw Jacobian")
    ik = _LongJawIK(CartonGripperRobot(config))
    q = ik.q.copy()
    _, jacobian = ik._fk(q)
    measured = np.zeros((3, 7))
    for axis in range(7):
        delta = np.eye(7)[axis] * 1e-6
        measured[:, axis] = (ik._fk(q + delta)[0][:3, 3] - ik._fk(q - delta)[0][:3, 3]) / (2e-6)
    np.testing.assert_allclose(jacobian[:3], measured, atol=1e-8)


def test_attachment_audit_rejects_garment_ids_even_with_bag_labels(tmp_path):
    from scripts.physic_solver.run_package_matrix import _audit_assistance

    point = [0.03, 0.007, 0.055]
    np.savez(
        tmp_path / "full_physics.npz",
        positions=np.tile(point, (1, 12, 1)),
        bodies=np.tile([0, 0, 0, 0, 0, 0, 1], (1, 2, 1)),
        times=[0.0],
    )
    np.savez(tmp_path / "particles.npz", garment_count=4)
    configuration = dict(
        robot=dict(
            backend="franka_package_pads",
            pad_center_m=[0, 0.007, 0.055],
            pad_half_extents_m=[0.03, 0.007, 0.042],
        ),
        planner=dict(attachment_capture_distance_m=0.004, attachment_vertices_per_group=8),
    )
    groups = [
        dict(
            material="bag",
            ids=ids,
            capture_time_s=0,
            body_id=body,
            local_points_m=[point] * 3,
        )
        for body, ids in enumerate(([0, 1, 2], [8, 9, 10]))
    ]
    assert not _audit_assistance(tmp_path, dict(attachment_groups=groups), configuration)[
        "local_capture_verified"
    ]
    groups[0]["ids"] = [4, 5, 6]
    assert _audit_assistance(tmp_path, dict(attachment_groups=groups), configuration)[
        "local_capture_verified"
    ]
    groups[1]["body_id"] = 0
    assert not _audit_assistance(tmp_path, dict(attachment_groups=groups), configuration)[
        "local_capture_verified"
    ]


def test_newton_diagnostic_world_is_lazy_and_rejects_other_physics_modes():
    from pydantic import ValidationError

    from real2sim2real.simworld.base import AutoSimWorld
    from real2sim2real.simworld.simworld_physic_solver import NewtonDiagnosticWorldConfig

    world = AutoSimWorld.from_config(config=NewtonDiagnosticWorldConfig())
    assert world.app is None and world.stage is None
    assert world.uses_direct_newton
    with pytest.raises(ValidationError):
        NewtonDiagnosticWorldConfig(physics_engine="physx")
    with pytest.raises(ValidationError):
        NewtonDiagnosticWorldConfig(newton_execution="usd")
