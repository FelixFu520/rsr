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
    arm_cases = {
        "mujoco_hydroelastic_screw02_ur5e",
        "mujoco_hydroelastic_screw02_ur5e_debug",
        "mujoco_hydroelastic_screw03_dual_ur5e",
        "mujoco_hydroelastic_screw03_dual_ur5e_debug",
        "mujoco_hydroelastic_screw04_dual_through",
        "mujoco_hydroelastic_screw04_dual_through_debug",
        "mujoco_hydroelastic_screw05_dual_release",
        "mujoco_hydroelastic_screw05_dual_release_debug",
        "mujoco_hydroelastic_screw06_bolt_release",
        "mujoco_hydroelastic_screw06_bolt_release_debug",
    }
    banana_cases = {"kinematic_tgs_banana01_five_cuts", "kinematic_tgs_banana01_five_cuts_debug"}
    assert {p.stem for p in paths} == set(
        PHYSIC_SOLVER_CASE_NAMES.values()
    ) | arm_cases | banana_cases
    assert len(PHYSIC_SOLVER_PACKAGE_CASES) == 18
    for path in paths:
        if path.stem in arm_cases | banana_cases:
            continue  # New demos have their own formal/debug rendering contract below.
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


@pytest.mark.parametrize("debug", [False, True])
def test_ur5e_screw_driver_routes_lazily_and_keeps_thread_gates(debug, monkeypatch, tmp_path):
    name = "mujoco_hydroelastic_screw02_ur5e" + ("_debug" if debug else "")
    monkeypatch.setenv(
        "RSR_SETTINGS_FILE", str(Path(f"configs/physic_solver/{name}.toml").resolve())
    )
    monkeypatch.setenv("RSR_JOB__output_dir", str(tmp_path / name))
    reset_settings()
    try:
        job = AutoJob.from_config()
        assert job.sim_world.app is None
        assert job.robot.config.backend == job.planner.config.backend == "ur5e_screwdriver"
        assert len(job.robot.recording_joint_names) == len(job.planner.action_names) == 8
        assert job.planner.config.minimum_turns == 2.0
        assert job.planner.config.minimum_advance_m == 0.006
        assert job.planner.config.maximum_pitch_error_fraction == 0.25
        assert job.config.recorder.fps == (15 if debug else 30)
        assert job.sim_world.config.renderer == ("MinimalRendering" if debug else "PathTracing")
        assert {cam.resolution for cam in job.recording_cameras} == {
            (640, 480) if debug else (1024, 768)
        }
    finally:
        reset_settings()


def test_screw_arm_measures_absolute_bolt_advance_instead_of_float_joint():
    from real2sim2real.planner.planner_physic_solver import ScrewRuntime

    runtime = object.__new__(ScrewRuntime)
    runtime.screw_dofs = [6, 7]
    runtime.spindle_body = 0
    runtime.scene = SimpleNamespace(initial_bolt_z_m=0.077)
    runtime.robot = SimpleNamespace(screw_control=lambda: None)
    runtime.state_0 = SimpleNamespace(
        joint_q=SimpleNamespace(numpy=lambda: np.array([0.0] * 6 + [-0.004, -12.0])),
        body_q=SimpleNamespace(numpy=lambda: np.array([[0, 0, 0.067, 0, 0, 0, 1.0]])),
    )
    np.testing.assert_allclose(runtime.screw_coordinates(), [-0.010, -12.0])


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize(
    "stem,duration,tip",
    [
        ("screw03_dual_ur5e", 12.8, 0.0),
        ("screw04_dual_through", 19.8, 0.005),
        ("screw05_dual_release", 21.8, 0.005),
        ("screw06_bolt_release", 29.6, 0.005),
    ],
)
def test_dual_screw_routes_two_arms_without_loading_kit(
    debug, stem, duration, tip, monkeypatch, tmp_path
):
    name = f"mujoco_hydroelastic_{stem}" + ("_debug" if debug else "")
    monkeypatch.setenv(
        "RSR_SETTINGS_FILE", str(Path(f"configs/physic_solver/{name}.toml").resolve())
    )
    monkeypatch.setenv("RSR_JOB__output_dir", str(tmp_path / name))
    reset_settings()
    try:
        job = AutoJob.from_config()
        assert job.sim_world.app is None
        assert job.scene.config.backend == "dual_screw_assembly"
        assert job.robot.config.backend == job.planner.config.backend == "dual_ur5e_screwdriver"
        expected_dofs = 18 if job.robot.config.releasable_socket else 16
        assert (
            len(job.robot.recording_joint_names) == len(job.planner.action_names) == expected_dofs
        )
        assert job.planner.expected_duration_s == pytest.approx(duration)
        assert job.planner.config.minimum_tip_protrusion_m == tip
        assert job.scene.config.nut_support_height_m == 0.10
        assert job.planner.config.nut_lift_m == 0.16
        assert job.planner.config.minimum_advance_m == 0.006
    finally:
        reset_settings()


def test_dual_screw_requires_lift_bilateral_contact_and_low_slip():
    import copy

    from real2sim2real.planner.planner_physic_solver import DualScrewPlannerConfig
    from real2sim2real.utils.physic_solver_utils import dual_screw_phase, evaluate_dual_screw_rows

    cfg = DualScrewPlannerConfig().model_dump()
    cfg.update(engage_s=1.0, screw_s=7.0)
    samples = [
        (0.0, "clamp", 0.05, 0.0, 0.108),
        (0.8, "lift", 0.05, 0.0, 0.108),
        (2.3, "align", 0.05, 0.0, 0.268),
        (3.8, "engage", 0.05, 0.0, 0.268),
        (4.8, "screw", 0.05, 0.0, 0.268),
        (11.7, "screw", 0.04375, -5 * np.pi, 0.268),
        (11.8, "hold", 0.04375, -5 * np.pi, 0.268),
        (12.8, "hold", 0.04375, -5 * np.pi, 0.268),
    ]
    rows = [
        dict(
            time_s=t,
            phase=phase,
            axial_displacement_m=z,
            spindle_angle_rad=angle,
            contacts=5,
            jaw_p_contacts=3,
            jaw_n_contacts=3,
            support_contacts=0,
            nut_center_local_m=[0, 0, height],
            nut_relative_holder_m=[0, 0, 0.028],
            nut_yaw_rad=0.0,
            arm_q_rad=[0.0] * 12,
            arm_target_rad=[0.0] * 12,
        )
        for t, phase, z, angle, height in samples
    ]
    assert dual_screw_phase(0.4, cfg) == ("clamp", 0.5)
    assert dual_screw_phase(50.0, cfg) == ("hold", 1.0)
    result = evaluate_dual_screw_rows(rows, cfg, 0.0025)
    assert result["success"]
    assert result["minimum_nut_lift_m"] == pytest.approx(0.16)
    for field, value, failed_check in (
        ("nut_center_local_m", [0, 0, 0.108], "nut_actually_lifted"),
        ("jaw_n_contacts", 0, "bilateral_finger_contact"),
        ("support_contacts", 1, "nut_clear_of_support_while_threading"),
    ):
        candidate = copy.deepcopy(rows)
        for row in candidate:
            row[field] = value
        rejected = evaluate_dual_screw_rows(candidate, cfg, 0.0025)
        assert not rejected["success"] and not rejected["checks"][failed_check]
    candidate = copy.deepcopy(rows)
    candidate[-1]["nut_relative_holder_m"] = [0.01, 0, 0.028]
    assert not evaluate_dual_screw_rows(candidate, cfg, 0.0025)["checks"]["nut_held_without_slip"]
    cfg["minimum_tip_protrusion_m"] = 0.005
    for row in rows:
        row["tip_protrusion_m"] = 0.008
    assert evaluate_dual_screw_rows(rows, cfg, 0.0025)["success"]
    # Passing thread-lead and advance gates does not establish that the bolt exits the nut.
    rows[-1]["tip_protrusion_m"] = -0.001
    result = evaluate_dual_screw_rows(rows, cfg, 0.0025)
    assert result["checks"]["thread_lead_matches"]
    assert not result["checks"]["bolt_protrudes_through_nut"]
    rows[-1]["tip_protrusion_m"] = float("nan")
    assert not evaluate_dual_screw_rows(rows, cfg, 0.0025)["success"]
    rows[-1]["tip_protrusion_m"] = 0.008
    rows[-1]["time_s"] = 12.79
    cfg.update(release_s=1.0, post_release_s=1.0)
    for t, phase in [(12.8, "release"), (13.8, "released"), (14.8, "released")]:
        row = copy.deepcopy(rows[-1])
        row.update(
            time_s=t,
            phase=phase,
            joint_q=[0.0] * 14 + [0.028, 0.028],
            jaw_p_contacts=0,
            jaw_n_contacts=0,
            nut_relative_holder_m=[0.01, 0, 0.028],
        )
        rows.append(row)
    assert dual_screw_phase(14.8, cfg) == ("released", 1.0)
    assert evaluate_dual_screw_rows(rows, cfg, 0.0025)["success"]
    rows[-1]["joint_q"][14] = 0.012  # A target alone cannot establish that the jaw opened.
    assert not evaluate_dual_screw_rows(rows, cfg, 0.0025)["checks"]["gripper_actually_reopened"]
    rows[-1]["joint_q"][14] = 0.028
    rows[-1]["jaw_p_contacts"] = 1
    assert not evaluate_dual_screw_rows(rows, cfg, 0.0025)["checks"]["nut_clear_of_both_fingers"]
    cfg["release_target"] = "bolt"
    for row in rows:
        row.update(
            jaw_p_contacts=3,
            jaw_n_contacts=3,
            nut_relative_holder_m=[0, 0, 0.028],
            socket_contacts=0 if row["phase"] == "released" else 3,
            socket_clearance_m=0.06 if row["phase"] == "released" else -0.019,
            socket_jaw_q_m=[0.025, 0.025],
        )
    assert evaluate_dual_screw_rows(rows, cfg, 0.0025)["success"]
    rows[-1]["socket_contacts"] = 1
    assert not evaluate_dual_screw_rows(rows, cfg, 0.0025)["checks"]["socket_actually_detached"]
    rows[-1]["socket_contacts"] = 0
    rows[-1]["socket_jaw_q_m"] = [0.001, 0.025]
    assert not evaluate_dual_screw_rows(rows, cfg, 0.0025)["checks"]["bolt_clamp_actually_open"]
    rows[-1]["socket_jaw_q_m"] = [0.025, 0.025]
    rows[-1]["jaw_n_contacts"] = 0
    assert not evaluate_dual_screw_rows(rows, cfg, 0.0025)["checks"]["bilateral_finger_contact"]


def test_contact_socket_clearance_uses_all_corners_and_bolt_frame():
    from real2sim2real.utils.battery_disassembly_utils import quat_to_matrix
    from real2sim2real.utils.physic_solver_utils import screw_socket_clearance, screw_socket_parts

    assert len(screw_socket_parts(0.0148)) == 16
    # Opposite movable halves must retain a real seam, including the retaining shoulders.
    for part in screw_socket_parts(0.0145):
        if part["side"] == "fixed":
            continue
        extent = (
            abs(np.cos(part["yaw"])) * part["half"][0] + abs(np.sin(part["yaw"])) * part["half"][1]
        )
        assert abs(part["center"][0]) - extent > 0
    bolt = np.array([0, 0, 0, 0, 0, 0, 1.0])
    assert screw_socket_clearance(bolt, bolt, 0.0148) == pytest.approx(-0.0235)
    tool = bolt.copy()
    tool[2] = 0.1
    assert screw_socket_clearance(tool, bolt, 0.0148) == pytest.approx(0.0765)
    angle = 0.7
    q = np.array([np.sin(angle / 2), 0, 0, np.cos(angle / 2)])
    bolt = np.r_[[1, 2, 3], q]
    tool = bolt.copy()
    tool[:3] += quat_to_matrix(q[[3, 0, 1, 2]]) @ [0, 0, 0.1]
    assert screw_socket_clearance(tool, bolt, 0.0148) == pytest.approx(0.0765)


def test_screw_release_requires_observation_duration():
    from pydantic import ValidationError

    from real2sim2real.planner.planner_physic_solver import DualScrewPlannerConfig

    for fields in ({"release_s": 1}, {"post_release_s": 1}):
        with pytest.raises(ValidationError, match="both"):
            DualScrewPlannerConfig(**fields)


@pytest.mark.parametrize("latch", [False, True])
def test_screw_axial_stop_can_keep_first_position_target(latch):
    from types import SimpleNamespace

    from real2sim2real.planner.planner_physic_solver import DualScrewRuntime

    runtime = object.__new__(DualScrewRuntime)
    measured = np.zeros(23)

    def targets(**kwargs):
        q = np.zeros(16)
        q[6] = kwargs["measured"][6]
        return q, np.zeros(16), q.copy()

    runtime.robot = SimpleNamespace(screw_dofs=[6, 7], dual_control=targets)
    runtime.config = SimpleNamespace(latch_axial_hold=latch)
    runtime.state_0 = SimpleNamespace(joint_q=SimpleNamespace(numpy=lambda: measured))
    runtime.model = SimpleNamespace(joint_dof_count=22)
    captured = {}
    runtime.control = SimpleNamespace(
        joint_target_q=SimpleNamespace(assign=lambda q: captured.update(q=q.copy())),
        joint_target_qd=SimpleNamespace(assign=lambda qd: None),
    )
    for z in (-0.024, -0.027):
        measured[6] = z
        runtime.command_dual(
            holder_z_m=0.268, driver_z_m=0.342, finger_m=0.028, preload=True, speed=0.0, hold=True
        )
    assert captured["q"][6] == (-0.024 if latch else -0.027)
    assert runtime.target_q[6] == captured["q"][6]


def test_bolt_rotation_texture_uv_follows_mesh_and_unwraps_seam():
    from real2sim2real.utils.physic_solver_utils import screw_cylindrical_uv

    angles = np.deg2rad([-2.0, 2.0, 3.0])
    vertices = np.c_[np.cos(angles), np.sin(angles), [0.0, 1.0, 0.5]]
    uv = screw_cylindrical_uv(vertices, np.array([[0, 1, 2]]))
    assert uv.shape == (3, 2)
    assert np.ptp(uv[:, 0]) < 0.02  # Interpolate across the seam, not around the whole bolt.
    np.testing.assert_allclose(uv[:, 1], [0, 1, 0.5])
    rotated = vertices.copy()
    rotated[:, :2] = vertices[:, :2] @ np.array([[0, 1], [-1, 0]])
    turned = screw_cylindrical_uv(rotated, np.array([[0, 1, 2]]))
    np.testing.assert_allclose(np.mod(turned[:, 0] - uv[:, 0], 1), 0.25, atol=1e-7)
