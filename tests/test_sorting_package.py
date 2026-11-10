"""Pro-M6 task routing, arbitrary robot coordinate export and independent shape checks."""

import json
import shutil
import subprocess

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from real2sim2real.core.config import apply_settings_file, reset_settings
from real2sim2real.job.base import AutoJob
from real2sim2real.job.job_sorting import SortingJob
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder
from real2sim2real.utils.sorting_quality import aligned_shape_rms, audit_held_shape


def test_surface_distance_pruning_matches_all_triangles_including_degenerate_faces():
    from real2sim2real.utils.package_geometry import _surface_distance, _surface_distance_dense

    rng = np.random.default_rng(42)
    triangles = rng.normal(size=(256, 3, 3))
    triangles[:4, 1:] = triangles[:4, :1]
    points = rng.normal(size=(256, 3))
    np.testing.assert_allclose(
        _surface_distance(points, triangles),
        _surface_distance_dense(points, triangles),
        rtol=1e-12,
        atol=1e-12,
    )


def test_surface_distance_does_not_discard_a_large_face_with_a_remote_centroid():
    from real2sim2real.utils.package_geometry import _surface_distance, _surface_distance_dense

    rng = np.random.default_rng(7)
    small = rng.normal(size=(256, 3, 3)) * 0.01 + [0, 0, 0.1]
    large = np.array([[[-100, -100, 0], [100, -100, 0], [0, 100, 0]]])
    triangles = np.concatenate([small, large])
    points = np.c_[rng.uniform(-0.01, 0.01, (256, 2)), np.full(256, 0.005)]
    np.testing.assert_allclose(_surface_distance(points, triangles), 0.005, atol=1e-12)
    np.testing.assert_allclose(
        _surface_distance(points, triangles),
        _surface_distance_dense(points, triangles),
        atol=1e-12,
    )


def test_fixed_robot_root_contacts_count_as_robot_and_supports_do_not():
    from types import SimpleNamespace

    from real2sim2real.planner.planner_sorting import _robot_contact_count

    runtime = SimpleNamespace(
        last_active_contact_shapes=np.array([8, 25, 25, 44, 45, 46]),
        last_active_contact_bodies=np.array([5, -1, -1, -1, -1, 26]),
        robot_shape_count=44,
        robot_body_count=26,
    )
    assert _robot_contact_count(runtime) == 3


def test_air_drop_measures_touchdown_before_the_running_belt_moves_the_parcel():
    from types import SimpleNamespace

    from real2sim2real.planner.planner_sorting import SortingPlanner, SortingPlannerConfig

    planner = object.__new__(SortingPlanner)
    planner.config = SortingPlannerConfig(continuous_outfeed=True)
    planner.scene = SimpleNamespace(label_up_cos=lambda points: 1.0)
    planner.destination_bounds = np.array([[-2, -1, 0.9], [0, 1, 1.0]])
    planner.goal = np.array([-0.5, 0, 1.0])
    planner.legs = {"right": dict(released=False)}
    planner.elapsed = 1.0
    # A tilted falling parcel touches the belt while its COM is still high.
    points = np.array([[-0.5, -0.1, 1.0], [-0.4, 0.1, 1.24]])
    coms = np.array([[-0.47, 0, 1.12], [-0.47, 0, 1.12]])
    planner._measure_moving_landing(points, coms, robot_contacts=1, outfeed_contacts=2)
    assert "landing_time_s" not in planner.legs["right"]
    planner._measure_moving_landing(points, coms, robot_contacts=0, outfeed_contacts=2)
    assert planner.legs["right"]["place_error"] == pytest.approx(0.03)
    assert not planner.legs["right"]["released"]
    planner.elapsed = 1.3
    # It settles after travelling farther than the original 80 mm landing tolerance.
    points[:, 0] -= 0.2
    points[:, 2] = 1.02
    coms[:] = [-0.67, 0, 1.02]
    planner._measure_moving_landing(points, coms, robot_contacts=0, outfeed_contacts=2)
    assert planner.legs["right"]["released"]
    assert planner.legs["right"]["landing_time_s"] == 1.0
    assert planner.legs["right"]["release_verified_time_s"] == 1.3
    assert planner.legs["right"]["place_error"] == pytest.approx(0.03)


def test_sorting_metrics_preserve_nested_landing_coordinates_in_strict_json():
    from real2sim2real.planner.planner_sorting import _finite_metrics

    measured = dict(
        landing_xy_m=[[0.1, -0.8], [0.2, -0.9]],
        minimum_lift=np.array([np.inf, 0.2]),
        unavailable=float("nan"),
    )
    encoded = json.loads(json.dumps(_finite_metrics(measured), allow_nan=False))
    assert encoded["landing_xy_m"] == measured["landing_xy_m"]
    assert encoded["minimum_lift"] == [None, 0.2]
    assert encoded["unavailable"] is None


def test_fixed_root_clearance_includes_particle_radius_and_shape_margin():
    from real2sim2real.utils.sorting_quality import audit_fixed_robot_clearance, unit_box_mesh

    box, triangles = unit_box_mesh()
    collider = dict(shape_id=0, vertices=box, triangles=triangles)
    probes = np.array([[1.0005, 0, 0], [1.0005, 0.1, 0], [1.0005, 0, 0.1]])
    result = audit_fixed_robot_clearance(
        probes, np.array([[0, 1, 2]]), [collider], 0.0005, np.array([0.0001])
    )
    assert not result["passed"]
    probes[:, 0] += 0.001
    assert audit_fixed_robot_clearance(
        probes, np.array([[0, 1, 2]]), [collider], 0.0005, np.array([0.0001])
    )["passed"]


def test_diagnostic_cache_writes_native_substeps_from_runtime_frequency(tmp_path):
    from types import SimpleNamespace

    from real2sim2real.planner.planner_sorting import SortingPlanner, SortingPlannerConfig

    planner = object.__new__(SortingPlanner)
    planner.config = SortingPlannerConfig(mujoco_substeps_per_vbd_step=4)
    planner.robot = SimpleNamespace(
        sim_world=SimpleNamespace(config=SimpleNamespace(physics_dt=1 / 60)),
        mimic={"left": [(0, 1), (1, -1)], "right": [(2, 1), (3, -1)]},
        gripper_indices={"left": 0, "right": 2},
    )
    array = SimpleNamespace(numpy=lambda: np.array([0.5, 1.4]))
    residuals = np.arange(128 * 4).reshape(128, 4) * 1e-8
    planner.runtime = SimpleNamespace(
        config=SimpleNamespace(newton_substeps=32),
        jaw_colliders=[],
        _masses=np.ones(3),
        model=SimpleNamespace(
            shape_material_mu=array,
            shape_body=array,
            soft_contact_mu=1.4,
            particle_radius=array,
            shape_margin=array,
        ),
        belt_body=1,
        platform_shape=0,
        robot_shape_count=1,
        fixed_robot_colliders=[
            dict(shape_id=0, vertices=np.eye(3), triangles=np.array([[0, 1, 2]]))
        ],
        native_mimic_residuals=[residuals],
    )
    planner.scene = SimpleNamespace(
        triangle_indices=np.array([[0, 1, 2]]),
        garment_count=2,
        supports={"infeed": np.zeros((2, 3))},
        label_triangles=np.array([[0, 1, 2]]),
    )
    planner.rest = np.zeros((3, 3))
    planner.samples = [planner.rest, planner.rest + 0.01]
    planner.sample_times = [0.0, 1 / 60]
    planner.checkpoint_times = [0.0]
    planner.joints = planner.actions = np.zeros((2, 16))
    planner.hand_poses = planner.body_samples = np.zeros((2, 2, 7))
    planner.articulation_samples = np.zeros((2, 4))
    planner.transfer_samples = []
    planner.transfer_groups = {"left": [], "right": []}
    planner.write_diagnostics(tmp_path)
    with np.load(tmp_path / "particles.npz") as saved:
        np.testing.assert_array_equal(saved["native_mimic_residuals"], residuals)
        assert saved["native_mimic_dt_s"] == pytest.approx(1 / 7680)
        assert saved["platform_shape"] == 0
        assert saved["robot_shape_count"] == 1
        np.testing.assert_array_equal(saved["fixed_robot_shape_ids"], [0])
        np.testing.assert_array_equal(saved["fixed_robot_vertices_0"], np.eye(3))
        np.testing.assert_array_equal(saved["particle_radius"], array.numpy())
        np.testing.assert_array_equal(saved["times"], planner.sample_times)


def test_live_regrasp_uses_the_deformed_central_surface_and_preserves_floor_clearance():
    from real2sim2real.planner.planner_sorting import SortingPlannerConfig, _regrasp_pad_bottom

    # A layered parcel left arched after release needs a higher regrasp than its spawn plane.
    xy = np.array([[x, y] for x in (-0.02, 0, 0.02) for y in (-0.02, 0, 0.02)])
    garment = np.concatenate([np.c_[xy, np.full(len(xy), z)] for z in (1.82, 1.87, 1.90)])
    raised = _regrasp_pad_bottom(garment, np.zeros(2), 1.8, 0.03, 0.02)
    assert raised == pytest.approx(0.08)
    flattened = garment.copy()
    flattened[:, 2] -= 0.075
    assert _regrasp_pad_bottom(flattened, np.zeros(2), 1.8, 0.03, 0.02) == 0.03
    # Remote high folds must not raise a grasp elsewhere on the parcel.
    remote = np.array([[0.2, 0.2, 2.2]])
    assert _regrasp_pad_bottom(np.r_[garment, remote], np.zeros(2), 1.8, 0.03, 0.02) == raised
    with pytest.raises(ValueError, match="No live garment surface"):
        _regrasp_pad_bottom(remote, np.zeros(2), 1.8, 0.03, 0.02)
    with pytest.raises(ValueError, match="pad-bottom calibration"):
        SortingPlannerConfig(regrasp_surface_indent_m=0.02)
    with pytest.raises(ValueError, match="minimum surface indentation"):
        SortingPlannerConfig(regrasp_surface_indent_fraction=0.3)


@pytest.mark.parametrize("height, expected", [(0.114, 0.0798), (0.0714, 0.04998), (0.04, 0.02)])
def test_relative_regrasp_depth_adapts_to_handoff_height_with_minimum_indentation(height, expected):
    from real2sim2real.planner.planner_sorting import _regrasp_pad_bottom

    xy = np.array([[x, y] for x in (-0.02, 0, 0.02) for y in (-0.02, 0, 0.02)])
    garment = np.concatenate([np.c_[xy, np.full(len(xy), 1.8 + z)] for z in (0.002, height)])
    assert _regrasp_pad_bottom(garment, np.zeros(2), 1.8, 0.015, 0.02, 0.3) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("regrasp", [None, (0.0, 0.055)])
@pytest.mark.parametrize("platform_target", [None, (0.0, -0.94)])
def test_right_grasp_offset_preserves_left_pick_and_screens_offset_destination(
    regrasp, platform_target
):
    from types import SimpleNamespace

    from real2sim2real.planner.planner_sorting import SortingPlanner, SortingPlannerConfig

    calls = []

    def prepare(side, target, waypoints):
        calls.append((side, target.copy(), np.array(waypoints)))
        return 0.0

    planner = object.__new__(SortingPlanner)
    planner.config = SortingPlannerConfig(
        grasp_offset_xy_m=(0.01, 0.02), regrasp_offset_xy_m=regrasp
    )
    planner.runtime = SimpleNamespace(
        tcp_position=lambda side: np.array([0.4, -0.8, 2.2]),
        prepare_approach=prepare,
        parking_positions={"left": np.array([0.55, -0.9, 2.03]), "right": np.zeros(3)},
    )
    bounds = (np.array([-0.3, -1.1, 1.76]), np.array([0.3, -0.6, 1.8]))
    planner.scene = SimpleNamespace(
        supports={name: bounds for name in ("infeed", "platform", "outfeed")},
        config=SimpleNamespace(
            outfeed_target_xy=(-0.55, -0.876), platform_target_xy=platform_target
        ),
    )
    planner.schedule = [(side, "approach", 3.0) for side in ("left", "right")]
    planner.checkpoint_times, planner.elapsed = [], 0.0
    planner.command_side = planner.command_tcp = None
    planner.max_tcp_error = 0.0
    planner.legs = {"left": {}, "right": {}}
    coms = np.array([[0.1, -0.8, 1.84], [0.1, -0.8, 1.84]])
    planner._enter(0, coms)
    np.testing.assert_allclose(planner.pick[:2], [0.11, -0.78])
    np.testing.assert_allclose(
        planner.goal[:2], [0.0, -0.85] if platform_target is None else platform_target
    )
    planner._enter(1, coms)
    right_offset = np.array(regrasp if regrasp is not None else [0.01, 0.02])
    np.testing.assert_allclose(planner.pick[:2], coms[0, :2] + right_offset)
    np.testing.assert_allclose(calls[1][1][:2], planner.pick[:2])
    np.testing.assert_allclose(calls[1][2][4, :2], [-0.55, -0.876] + right_offset)
    with pytest.raises(ValueError):
        SortingPlannerConfig(regrasp_offset_xy_m=(float("nan"), 0))


@pytest.mark.parametrize("target", [None, 0.0, 2.0])
@pytest.mark.parametrize("platform", [None, 0.1])
@pytest.mark.parametrize("floor_stiffness", [None, 400000.0])
def test_contact_materials_change_only_declared_roles(target, platform, floor_stiffness):
    from types import SimpleNamespace

    from real2sim2real.planner.planner_sorting import SortingPlannerConfig
    from real2sim2real.planner.sorting_runtime import SortingRuntime

    class Array:
        def __init__(self, values):
            self.values = np.array(values)

        def numpy(self):
            return self.values.copy()

        def assign(self, values):
            self.values = values.copy()

    runtime = object.__new__(SortingRuntime)
    runtime.config = SortingPlannerConfig(
        gripper_package_friction=target,
        platform_friction=platform,
        floor_contact_stiffness=floor_stiffness,
    )
    runtime.contact_body_ids = dict(left_0=1, left_1=2, right_0=3, right_1=4)
    runtime.belt_body, runtime.belt_shape = 5, 6
    runtime.platform_shape = 0
    runtime.floor_shape = 7
    original = np.array([0.5, 1.4, 1.4, 1.4, 1.4, 1.4, 0.5, 0.5])
    runtime.model = SimpleNamespace(
        shape_body=Array([-1, 0, 1, 2, 3, 4, 5, -1]),
        shape_material_mu=Array(original),
        shape_material_ke=Array(np.full(8, 40000.0)),
        soft_contact_mu=1.4,
        soft_contact_ke=40000.0,
    )
    runtime._configure_contact_friction()
    actual = runtime.model.shape_material_mu.numpy()
    np.testing.assert_array_equal(actual[[1, 6, 7]], original[[1, 6, 7]])
    assert actual[0] == (0.5 if platform is None else platform)
    expected = 1.4 if target is None else target
    np.testing.assert_allclose(np.sqrt(1.4 * actual[2:6]), expected)
    assert runtime.model.soft_contact_mu == 1.4
    np.testing.assert_array_equal(runtime.model.shape_material_ke.numpy()[:7], 40000.0)
    assert runtime.model.shape_material_ke.numpy()[7] == (
        40000.0 if floor_stiffness is None else floor_stiffness
    )
    assert runtime.model.soft_contact_ke == 40000.0
    np.testing.assert_allclose(runtime.contact_friction["effective_gripper_package_mu"], [expected])
    np.testing.assert_allclose(
        runtime.contact_friction["effective_support_package_mu"],
        np.unique(np.sqrt(1.4 * np.array([0.5, actual[0]]))),
    )
    assert runtime.contact_friction["platform_material_mu"] == actual[0]
    for invalid in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SortingPlannerConfig(platform_friction=invalid)
    with pytest.raises(ValueError):
        SortingPlannerConfig(gripper_package_friction=2, cloth_friction=0)
    for invalid in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SortingPlannerConfig(gripper_package_friction=invalid)


def test_strain_grip_opens_with_bounded_relief_and_recovers_slowly():
    from real2sim2real.planner.planner_sorting import StrainGripController

    controller = StrainGripController(target=1.18, maximum_relief_rad=0.01, rate_rad_s=0.05)
    other_arm = StrainGripController(target=1.18, maximum_relief_rad=0.01, rate_rad_s=0.05)
    assert controller.command(0.66, 1.1, 0.02) == 0.66
    commands = [controller.command(0.66, 1.25, 0.02) for _ in range(15)]
    assert np.all(np.diff(commands) <= 0)
    assert commands[-1] == pytest.approx(0.65)
    assert controller.peak_relief_rad == pytest.approx(0.01)
    assert controller.command(0.66, 1.17, 0.02) == pytest.approx(0.65)
    for _ in range(10):
        recovered = controller.command(0.66, 1.1, 0.02)
    assert recovered == pytest.approx(0.651)
    assert other_arm.relief_rad == 0
    for _ in range(200):
        recovered = controller.command(0.66, 1.1, 0.02)
    assert recovered == 0.66
    restorative = StrainGripController(1.24, 0.01, 0.05, 0.025, 0.01)
    assert restorative.command(0.66, 1.28, 0.2) == pytest.approx(0.65)
    assert restorative.command(0.66, 1.235, 0.2) == pytest.approx(0.65)
    assert restorative.command(0.66, 1.21, 0.2) == pytest.approx(0.655)
    assert restorative.command(0.66, 1.28, 0.1) == pytest.approx(0.65)


def test_strain_relief_preserves_the_acquired_grip_during_lift_ramp():
    from real2sim2real.planner.planner_sorting import StrainGripController

    controller = StrainGripController(1.2, 0.01, 0.05)
    commands = [
        controller.command(nominal, 1.25, 1 / 60, minimum_rad=0.64)
        for nominal in np.linspace(0.64, 0.68, 120)
    ]
    assert commands[0] == 0.64
    assert min(commands) >= 0.64
    assert commands[-1] == pytest.approx(0.67)
    assert controller.peak_relief_rad == pytest.approx(0.01)


def test_sorting_mass_override_preserves_bag_geometry_and_original_asset(monkeypatch):
    from types import SimpleNamespace

    from real2sim2real.scene.scene_package import PackageScene
    from real2sim2real.scene.scene_sorting import SortingScene, SortingSceneConfig

    garment = np.array([0.02, 0.06, 0.12])
    bag = np.array([0.004, 0.006, 0.010542])
    published = []
    attribute = SimpleNamespace(Set=lambda value: published.extend(value))
    mesh = SimpleNamespace(GetPrim=lambda: SimpleNamespace(GetAttribute=lambda name: attribute))
    parts = [dict(name="Garment", masses=garment, mesh=mesh), dict(name="Bag", masses=bag)]
    geometry = np.arange(18).reshape(6, 3).copy()
    scene = object.__new__(SortingScene)
    scene.config = SortingSceneConfig(garment_mass_kg=0.1)
    scene.package_parts, scene.cloth_vertices = parts, geometry.copy()
    monkeypatch.setattr(PackageScene, "_arrange_cloth", lambda self: None)
    scene._arrange_cloth()
    np.testing.assert_allclose(parts[0]["masses"], garment * 0.5)
    assert parts[0]["masses"].sum() == pytest.approx(0.1)
    assert np.all(parts[0]["masses"] > 0)
    np.testing.assert_array_equal(parts[1]["masses"], bag)
    np.testing.assert_array_equal(scene.cloth_vertices, geometry)
    np.testing.assert_array_equal(garment, [0.02, 0.06, 0.12])
    np.testing.assert_allclose(published, parts[0]["masses"])
    assert scene.authored_garment_mass_kg == pytest.approx(0.2)
    assert scene.garment_mass_scale == pytest.approx(0.5)
    assert SortingSceneConfig().garment_mass_kg == 0.2
    for invalid in (0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SortingSceneConfig(garment_mass_kg=invalid)
        with pytest.raises(ValueError):
            SortingSceneConfig(bag_mass_kg=invalid)
    parts[0]["masses"] = garment.copy()
    parts[1].update(masses=bag.copy(), mesh=mesh)
    scene.config = SortingSceneConfig(garment_mass_kg=0.05, bag_mass_kg=float(bag.sum()) * 0.25)
    scene._arrange_cloth()
    np.testing.assert_allclose(parts[0]["masses"], garment * 0.25)
    np.testing.assert_allclose(parts[1]["masses"], bag * 0.25)
    np.testing.assert_array_equal(scene.cloth_vertices, geometry)
    np.testing.assert_array_equal(bag, [0.004, 0.006, 0.010542])
    assert scene.bag_mass_scale == pytest.approx(0.25)


@pytest.mark.skipif(
    not __import__("pathlib")
    .Path("assets/PhysicSolver/Package/v7_sorting/parcel_two_mesh.usd")
    .is_file(),
    reason="Local V7 source asset required",
)
def test_real_usd_mass_override_is_local_and_replayable():
    import os
    import sys
    from pathlib import Path

    usd = next(Path("app/extscache").glob("omni.usd.libs-*/pxr/Gf")).parent.parent.resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(usd) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["LD_LIBRARY_PATH"] = (
        str(usd / "bin") + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,pytest; "
            "runpy.run_path('tests/test_sorting_package.py')"
            "['_check_real_usd_mass'](pytest.MonkeyPatch())",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _check_real_usd_mass(monkeypatch):
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace

    from pxr import Usd

    from real2sim2real.scene.scene_sorting import SortingScene, SortingSceneConfig

    path = Path("assets/PhysicSolver/Package/v7_sorting/parcel_two_mesh.usd")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    scenes = []
    for target in (0.2, 0.1, 0.05):
        bag_target = (
            float(scenes[0].package_parts[1]["masses"].sum()) * 0.25 if target == 0.05 else None
        )
        scene = SortingScene(SortingSceneConfig(garment_mass_kg=target, bag_mass_kg=bag_target))
        scene.sim_world = SimpleNamespace(stage=Usd.Stage.CreateInMemory())
        scene.table_top_z = 1.78
        monkeypatch.setattr(scene, "refresh_poses", lambda: None)
        scene._arrange_cloth()
        scenes.append(scene)
    original = scenes[0]
    for lighter in scenes[1:]:
        np.testing.assert_array_equal(original.cloth_vertices, lighter.cloth_vertices)
        np.testing.assert_array_equal(original.triangle_indices, lighter.triangle_indices)
        np.testing.assert_array_equal(original.loft_springs, lighter.loft_springs)
        for before, after in zip(original.package_parts, lighter.package_parts, strict=True):
            target_mass = getattr(lighter.config, f"{after['name'].lower()}_mass_kg")
            scale = target_mass / before["masses"].sum() if target_mass is not None else 1
            np.testing.assert_allclose(after["masses"], before["masses"] * scale)
            np.testing.assert_allclose(
                after["mesh"].GetPrim().GetAttribute("physics:masses").Get(),
                after["masses"],
                rtol=1e-6,
            )
            assert np.all(after["masses"] > 0)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_quintic_transfer_has_bounded_motion_and_zero_endpoint_acceleration():
    from real2sim2real.planner.planner_sorting import _motion_blend

    values = np.array([_motion_blend(t, "quintic") for t in np.linspace(-0.1, 1.1, 121)])
    assert values[0] == 0 and values[-1] == 1
    assert np.all(np.diff(values) >= -1e-12)
    step = 1e-5
    for endpoint, direction in ((0, 1), (1, -1)):
        f0, f1, f2 = [_motion_blend(endpoint + direction * i * step, "quintic") for i in range(3)]
        assert abs((f1 - f0) / step) < 1e-5
        assert abs((f2 - 2 * f1 + f0) / step**2) < 0.002
    cubic_acceleration = (
        _motion_blend(2 * step, "cubic") - 2 * _motion_blend(step, "cubic")
    ) / step**2
    assert cubic_acceleration > 5.9


@pytest.mark.parametrize(
    "success,require_success,expected", [(True, True, 0), (False, True, 1), (False, False, 0)]
)
def test_kit_worker_checks_fresh_results_after_fast_exit(
    tmp_path, success, require_success, expected
):
    from real2sim2real.utils.kit_process import run_kit_worker

    worker = tmp_path / "worker.py"
    output = tmp_path / "result.json"
    worker.write_text(
        "import argparse,json,os\n"
        "p=argparse.ArgumentParser();p.add_argument('--worker-id');a=p.parse_args()\n"
        f"with open({str(output)!r},'w') as f: "
        f"json.dump(dict(run_id=a.worker_id,success={success!r}),f)\n"
        "os._exit(0)\n"
    )
    assert (
        run_kit_worker(
            str(worker), [], output, marker_field="run_id", require_success=require_success
        )
        == expected
    )
    worker.write_text("import os;os._exit(0)\n")
    with pytest.raises(RuntimeError, match="old result"):
        run_kit_worker(str(worker), [], output, marker_field="run_id")


def test_sorting_is_lazy_and_routes_native_components(monkeypatch, tmp_path):
    monkeypatch.setenv("RSR_JOB__OUTPUT_DIR", str(tmp_path))
    apply_settings_file("configs/PhysicSolver_mujoco_vbd_sorting_package.toml")
    try:
        job = AutoJob.from_config()
        assert isinstance(job, SortingJob)
        assert job.sim_world.app is None
        assert job.sim_world.config.newton_solver == "mujoco"
        assert job.planner.config.cloth_solver == "vbd"
        assert job.robot.config.backend == "pro_m6_newton"
        assert [camera.name for camera in job.recording_cameras] == [
            "overview",
            "head",
            "left_wrist",
            "right_wrist",
        ]
        assert job.record_stride == 4
        assert job.planner.config.maximum_edge_stretch == 1.3
        assert job.planner.config.grasp_height_m == pytest.approx(0.048)
    finally:
        monkeypatch.delenv("RSR_SETTINGS_FILE")
        reset_settings()


def test_cartesian_phases_preserve_commanded_pose_instead_of_accumulating_tracking_error():
    from types import SimpleNamespace

    from real2sim2real.planner.planner_sorting import SortingPlanner, SortingPlannerConfig

    planner = object.__new__(SortingPlanner)
    planner.config = SortingPlannerConfig()
    commanded = np.array([0.2, -0.9, 1.05])
    measured = commanded + [0.001, 0.003, -0.002]
    planner.runtime = SimpleNamespace(tcp_position=lambda side: measured.copy())
    planner.schedule = [("left", "lift", 3), ("right", "lift", 3)]
    planner.command_tcp, planner.command_side = commanded.copy(), "left"
    planner.checkpoint_times, planner.elapsed, planner.support_z = [], 0, 1.0
    coms = np.array([[0.2, -0.9, 1.02], [0.2, -0.9, 1.02]])
    planner._enter(0, coms)
    np.testing.assert_array_equal(planner.start_tcp, commanded)
    np.testing.assert_array_equal(planner.target[:2], commanded[:2])
    planner._enter(1, coms)
    np.testing.assert_array_equal(planner.start_tcp, measured)


def test_release_lowering_preserves_command_continuity_into_retraction():
    from types import SimpleNamespace

    from real2sim2real.planner.planner_sorting import SortingPlanner, SortingPlannerConfig

    planner = object.__new__(SortingPlanner)
    planner.config = SortingPlannerConfig(place_height_m=0.13, release_lowering_m=0.064)
    command = np.array([0.0, -0.9, 1.93])
    planner.runtime = SimpleNamespace(
        tcp_position=lambda side: command + [0.001, 0, -0.002],
        target_q=np.array([0.675]),
    )
    planner.robot = SimpleNamespace(gripper_indices={"left": 0})
    planner.legs = {"left": {}}
    planner.schedule = [("left", "release", 3), ("left", "retract", 1.5)]
    planner.command_tcp, planner.command_side = command.copy(), "left"
    planner.checkpoint_times, planner.elapsed = [], 0.0
    coms = np.array([[0.0, -0.9, 1.84], [0.0, -0.9, 1.84]])
    planner._enter(0, coms)
    np.testing.assert_array_equal(planner.start_tcp, command)
    np.testing.assert_allclose(planner.target, command - [0, 0, 0.064])
    planner.command_tcp = planner.target.copy()
    planner._enter(1, coms)
    np.testing.assert_allclose(planner.start_tcp, command - [0, 0, 0.064])
    np.testing.assert_allclose(planner.target, command + [0, 0, 0.116])
    for invalid in (-0.01, 0.13, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SortingPlannerConfig(place_height_m=0.13, release_lowering_m=invalid)


def test_grasp_offset_moves_the_pick_and_screened_tcp_path_but_keeps_the_com_destination():
    from types import SimpleNamespace

    from real2sim2real.planner.planner_sorting import SortingPlanner, SortingPlannerConfig

    planner = object.__new__(SortingPlanner)
    planner.config = SortingPlannerConfig(grasp_offset_xy_m=(0, 0.06))
    recorded = {}

    def prepare(side, target, waypoints):
        recorded.update(side=side, target=target.copy(), waypoints=waypoints)
        return 0.0

    planner.runtime = SimpleNamespace(
        tcp_position=lambda side: np.array([0.55, -0.9, 2.0]),
        prepare_approach=prepare,
        parking_positions={"left": np.array([0.55, -0.9, 2.0])},
    )
    planner.scene = SimpleNamespace(
        supports={
            "infeed": (np.array([0.3, -2.0, 1.74]), np.array([1.2, 0, 1.78])),
            "platform": (np.array([-0.35, -1.176, 1.76]), np.array([0.35, -0.576, 1.80])),
        },
        config=SimpleNamespace(platform_target_xy=None),
    )
    planner.schedule = [("left", "approach", 3), ("left", "transfer", 6)]
    planner.command_tcp, planner.command_side = None, None
    planner.checkpoint_times, planner.elapsed, planner.max_tcp_error = [], 0.0, 0.0
    planner.legs = {"left": {}}
    coms = np.array([[0.55, -0.9, 1.81], [0.55, -0.9, 1.81]])
    planner._enter(0, coms)
    np.testing.assert_allclose(planner.pick[:2], [0.55, -0.84])
    np.testing.assert_allclose(planner.goal[:2], [0.0, -0.876])
    np.testing.assert_allclose(recorded["waypoints"][4][:2], [0.0, -0.816])
    planner.command_tcp = np.array([0.55, -0.84, 2.11])
    planner._enter(1, coms)
    np.testing.assert_allclose(planner.target[:2], [0.0, -0.816])
    np.testing.assert_array_equal(coms[:, :2], [[0.55, -0.9], [0.55, -0.9]])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required for video export")
def test_sixteen_joint_export_keeps_named_coordinates_and_video_alignment(tmp_path):
    names = [f"joint_{i}_rad" for i in range(16)]
    recorder = LeRobotV3Recorder(
        tmp_path / "dataset",
        fps=15,
        width=64,
        height=48,
        task="relay",
        joint_names=names,
        action_names=[f"target_{name}" for name in names],
        robot_type="pro_m6_696_newton",
    )
    with pytest.raises(ValueError, match="named finite"):
        recorder.add_frame(np.zeros(9), np.zeros(9))
    for index in range(3):
        Image.new("RGB", (64, 48), (index * 80, 0, 0)).save(recorder.frame_path())
        recorder.add_frame(np.arange(16) + index, np.arange(16) + index + 0.1)
    root = recorder.finalize(episode_success=False)
    info = json.loads((root / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [16]
    assert info["features"]["action"]["names"] == [f"target_{name}" for name in names]
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    np.testing.assert_allclose(
        table["action"].to_pylist(), [np.arange(16) + i + 0.1 for i in range(3)], atol=1e-6
    )
    episode = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
    assert episode["dataset_use"] == "diagnostic" and not episode["success"]
    video = root / "videos/observation.images.overview/chunk-000/file-000.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )


def test_shape_metric_removes_rigid_motion_but_detects_deformation():
    rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    weights = np.array([1, 2, 3, 4], float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    moved = rest @ rotation + np.array([4, 5, 6])
    assert aligned_shape_rms(rest, moved, weights) < 1e-12
    moved[3] += np.array([0, 0, 0.5])
    assert aligned_shape_rms(rest, moved, weights) > 0.1


def test_absent_transfer_cannot_pass_shape_acceptance():
    report = audit_held_shape(
        np.empty((0, 0, 3)),
        np.empty((0, 3)),
        np.empty((0, 3), int),
        np.empty(0),
        0,
        {"left": [], "right": []},
    )
    assert not any(report["checks"].values())


@pytest.mark.parametrize("early_stop", [True, False])
def test_diagnostic_continuation_keeps_the_failed_shape_measurement(early_stop):
    from types import SimpleNamespace

    from real2sim2real.planner.planner_sorting import SortingPlanner, SortingPlannerConfig

    vertices = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.02], [0.1, 0, 0.02], [0, 0.1, 0.02]]
    )
    planner = object.__new__(SortingPlanner)
    planner.config = SortingPlannerConfig(early_stop_on_shape_failure=early_stop)
    planner.rest = vertices.copy()
    planner.scene = SimpleNamespace(
        garment_count=3, triangle_indices=np.array([[0, 1, 2], [3, 4, 5]])
    )
    planner.runtime = SimpleNamespace(_masses=np.ones(6))
    planner.samples, planner.elapsed = [vertices], 1.0
    planner.online_held_shape_samples, planner.failure_reason = [], None
    planner._measure_held_shape(vertices, "left")
    assert bool(planner.failure_reason) == early_stop
    measured = planner.online_held_shape_samples[0]
    assert measured["aligned_garment_shape_rms_m"] < 1e-10
    assert measured["bag_gap_q95_m"] == pytest.approx(0.02)
    assert measured["bag_gap_q95_m"] > planner.config.maximum_bag_gap_m


def test_final_shape_sampling_cannot_omit_a_recorded_failed_checkpoint():
    rest = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.005], [0.1, 0, 0.005], [0, 0.1, 0.005]]
    )
    positions = np.repeat(rest[None], 5, axis=0)
    positions[1, 3:, 2] = 0.02
    args = (positions, rest, np.array([[0, 1, 2], [3, 4, 5]]), np.ones(6), 3, {"left": range(5)})
    assert all(audit_held_shape(*args)["checks"].values())
    report = audit_held_shape(*args, extra_samples={"left": [1]})
    assert not report["checks"]["held_bag_gap_bounded"]
    assert {row["sample_index"] for row in report["samples"]} == {0, 1, 2, 4}


@pytest.mark.parametrize("half_size", [(2.0, 2.0), (0.2, 0.2)])
def test_contact_refinement_preserves_surface_mass_uv_provenance_and_closed_edges(half_size):
    from real2sim2real.utils.package_refinement import lumped_areas, refine_surface
    from real2sim2real.utils.sorting_quality import unit_box_mesh

    vertices, triangles = unit_box_mesh()
    areas = lumped_areas(vertices, triangles)
    masses = 0.2 * areas / areas.sum()
    refined = refine_surface(vertices, triangles, masses, half_size_xy=half_size)
    np.testing.assert_array_equal(refined.vertices[: len(vertices)], vertices)
    assert len(refined.vertices) > len(vertices)
    assert np.all(refined.masses > 0)
    assert refined.masses.sum() == pytest.approx(0.2)
    np.testing.assert_allclose(
        np.average(refined.vertices, axis=0, weights=refined.masses),
        np.average(vertices, axis=0, weights=masses),
        atol=1e-14,
    )
    np.testing.assert_allclose(lumped_areas(refined.vertices, refined.triangles).sum(), areas.sum())
    expected = np.einsum("fci,fij->fcj", refined.barycentric, vertices[triangles[refined.parents]])
    np.testing.assert_allclose(refined.vertices[refined.triangles], expected, atol=1e-14)
    old_face = vertices[triangles[refined.parents]]
    new_face = refined.vertices[refined.triangles]
    old_normal = np.cross(old_face[:, 1] - old_face[:, 0], old_face[:, 2] - old_face[:, 0])
    new_normal = np.cross(new_face[:, 1] - new_face[:, 0], new_face[:, 2] - new_face[:, 0])
    assert np.all(np.sum(old_normal * new_normal, axis=1) > 0)
    edges = np.sort(
        np.concatenate(
            [refined.triangles[:, :2], refined.triangles[:, 1:], refined.triangles[:, [2, 0]]]
        ),
        axis=1,
    )
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    uv = np.tile(np.array([[0, 0], [1, 0], [0, 1]], float), (len(triangles), 1, 1))
    uv += np.arange(len(triangles))[:, None, None] * 2
    mapped = np.einsum("fci,fij->fcj", refined.barycentric, uv[refined.parents])
    assert np.all(refined.barycentric >= 0)
    np.testing.assert_allclose(refined.barycentric.sum(2), 1)
    seam = {}
    for index, value in zip(refined.triangles.ravel(), mapped.reshape(-1, 2), strict=True):
        if index >= len(vertices):
            seam.setdefault(index, []).append(value)
    assert any(np.ptp(values, axis=0).max() > 1 for values in seam.values())
    with pytest.raises(ValueError, match="uniform areal density"):
        changed = masses.copy()
        changed[0] *= 2
        refine_surface(vertices, triangles, changed)


def test_refined_sampling_also_checks_the_original_probe_population():
    garment = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    original_bag = np.array([[0.1, 0.1, 0.02], [0.2, 0.1, 0.005], [0.1, 0.2, 0.005]])
    grid = np.array(
        [[x, y, 0.005] for x in np.linspace(0.2, 0.4, 10) for y in np.linspace(0.2, 0.4, 10)]
    )
    bag_tri = [[0, 1, 2]]
    for row in range(9):
        for col in range(9):
            a = 3 + row * 10 + col
            bag_tri.extend([[a, a + 1, a + 10], [a + 1, a + 11, a + 10]])
    vertices = np.concatenate([garment, original_bag, grid])
    triangles = np.concatenate([[[0, 1, 2]], np.array(bag_tri) + 3])
    args = (vertices[None], vertices, triangles, np.ones(len(vertices)), 3, {"left": [0]})
    assert all(audit_held_shape(*args)["checks"].values())
    reference = dict(garment_masses=[1, 1, 1], bag_vertex_count=3, bag_triangles=[[0, 1, 2]])
    report = audit_held_shape(*args, quality_reference=reference)
    assert report["checks"]["held_bag_gap_bounded"]
    assert not report["checks"]["held_original_sampling_bounded"]
    assert report["samples"][0]["reference_bag_gap_q95_m"] > 0.01


def test_original_face_probe_follows_the_refined_material_not_a_coarse_chord():
    from real2sim2real.utils.package_refinement import map_material_probes, refine_surface

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    triangles = np.array([[0, 1, 2]])
    refined = refine_surface(vertices, triangles, np.ones(3), half_size_xy=(2, 2))
    indices, weights = map_material_probes(refined, triangles, triangles, np.full((1, 3), 1 / 3))
    np.testing.assert_allclose(
        (refined.vertices[indices] * weights[:, :, None]).sum(1), [[1 / 3, 1 / 3, 0]]
    )
    deformed = refined.vertices.copy()
    deformed[3:, 2] = 1
    point = (deformed[indices] * weights[:, :, None]).sum(1)[0]
    assert point[2] == pytest.approx(1)
    assert deformed[triangles].mean(1)[0, 2] == 0


@pytest.mark.skipif(
    not __import__("pathlib")
    .Path("assets/PhysicSolver/Package/v7_sorting/parcel_two_mesh.usd")
    .is_file(),
    reason="Local V7 source asset required",
)
def test_refined_usd_keeps_source_unchanged_and_carries_original_quality_reference(tmp_path):
    import hashlib
    import sys
    from pathlib import Path

    source = Path("assets/PhysicSolver/Package/v7_sorting")
    digest = hashlib.sha256((source / "parcel_two_mesh.usd").read_bytes()).hexdigest()
    target = tmp_path / "refined"
    subprocess.run(
        [
            sys.executable,
            "scripts/refine_package_assets.py",
            "--source",
            str(source),
            "--target",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((target / "provenance.json").read_text())
    assert hashlib.sha256((source / "parcel_two_mesh.usd").read_bytes()).hexdigest() == digest
    assert manifest["source_asset_sha256"] == digest
    assert manifest["original_sampling_retained"] and not manifest["materials_modified"]
    assert manifest["usd_roundtrip_verified"]
    assert manifest["reference_probe_counts"]["garment_vertices"] == 3300
    assert manifest["reference_probe_counts"]["bag_vertices"] == 1650
    assert not manifest["loft_spring_anchors_modified"]
    assert manifest["refinement"]["Garment"]["mass_kg"] == pytest.approx(0.2, abs=1e-7)
    assert manifest["refinement"]["Garment"]["refined_vertices"] > 3300
    assert manifest["refinement"]["Bag"]["bag_closed"]


def test_replay_uses_exact_physics_times_and_excludes_the_terminal_sample():
    from scripts.render_sorting_package import replay_indices

    np.testing.assert_array_equal(replay_indices(np.arange(61) / 60, 15), np.arange(0, 60, 4))
    np.testing.assert_array_equal(replay_indices(np.array([0.0]), 15), [0])
    with pytest.raises(ValueError, match="coincide"):
        replay_indices(np.arange(61) / 60, 17)
    with pytest.raises(ValueError, match="strictly"):
        replay_indices(np.array([0.0, 0.2, 0.1]), 15)


def test_workflow_gap_policy_retains_grasp_and_original_shape_failures():
    from real2sim2real.utils.sorting_quality import without_bag_gap_checks

    checks = dict(
        sequence_completed=True,
        left_bilateral_transfer=True,
        held_shape_rms_bounded=True,
        held_bag_gap_bounded=False,
        held_original_sampling_bounded=False,
    )
    shape = dict(
        maximum_shape_rms_m=0.037413,
        samples=[dict(reference_aligned_garment_shape_rms_m=0.03)],
    )
    assert all(without_bag_gap_checks(checks, shape).values())
    assert checks["held_bag_gap_bounded"] is False
    checks["left_bilateral_transfer"] = False
    assert not all(without_bag_gap_checks(checks, shape).values())
    checks["left_bilateral_transfer"] = True
    shape["samples"][0]["reference_aligned_garment_shape_rms_m"] = 0.04
    assert not without_bag_gap_checks(checks, shape)["held_original_shape_rms_bounded"]
    checks["held_shape_rms_bounded"] = False
    assert not without_bag_gap_checks(checks, shape)["held_shape_rms_bounded"]


def test_replay_optical_overrides_reject_nonoptical_fields_and_preserve_source():
    from scripts.render_sorting_package import override_camera_poses

    source = {
        "scene": {"cameras": [], "garment_mass_kg": 0.05},
        "robot": {
            "cameras": [{"name": "left_wrist", "pose": {"position": [0, 0, 0]}}],
            "usd_path": "native.usda",
        },
        "planner": {"gripper_package_friction": 2.0},
    }
    updated = override_camera_poses(source, {"left_wrist": {"position": [0.22, 0, 0.1]}})
    assert source["robot"]["cameras"][0]["pose"]["position"] == [0, 0, 0]
    assert updated["scene"] == source["scene"]
    assert updated["planner"] == source["planner"]
    assert updated["robot"]["usd_path"] == source["robot"]["usd_path"]
    assert updated["robot"]["cameras"][0]["pose"]["position"] == [0.22, 0, 0.1]
    with pytest.raises(ValueError, match="Unknown replay cameras"):
        override_camera_poses(source, {"not_a_camera": {}})
    with pytest.raises(ValueError):
        override_camera_poses(source, {"left_wrist": {"usd_path": "modified.usda"}})
    with pytest.raises(ValueError):
        override_camera_poses(source, {"left_wrist": {"position": [float("nan"), 0, 0]}})


def test_presentation_profile_cannot_modify_physics_or_camera_mounts():
    import copy

    from scripts.render_sorting_package import ReplayRenderProfile, override_render_profile

    source = {
        "sim_world": {"physics_dt": 1 / 60, "renderer": "RayTracedLighting"},
        "scene": {
            "garment_mass_kg": 0.025,
            "cloth_usd_path": "original.usd",
            "cameras": [{"name": "overview", "resolution": [640, 480], "pose": {"x": 1}}],
        },
        "robot": {
            "usd_path": "native.usda",
            "cameras": [{"name": "head", "resolution": [640, 480], "pose": {"x": 2}}],
        },
        "planner": {"gripper_package_friction": 2.0},
    }
    original = copy.deepcopy(source)
    rendered = override_render_profile(source, ReplayRenderProfile())
    assert source == original
    assert rendered["sim_world"]["physics_dt"] == source["sim_world"]["physics_dt"]
    assert rendered["planner"] == source["planner"]
    for section in ("scene", "robot"):
        camera = rendered[section]["cameras"][0]
        assert camera["resolution"] == [1920, 1440]
        camera["resolution"] = source[section]["cameras"][0]["resolution"]
        assert rendered[section] == source[section]
    for invalid in (
        {"physics_dt": 0.1},
        {"garment_mass_kg": 1},
        {"resolution": [1921, 1440]},
        {"samples_per_pixel_per_frame": 0},
    ):
        with pytest.raises(ValueError):
            ReplayRenderProfile(**invalid)


def test_sorting_ik_honors_the_configured_closing_direction():
    from real2sim2real.planner.planner_sorting import SortingPlannerConfig
    from real2sim2real.planner.sorting_runtime import SortingRuntime

    runtime = object.__new__(SortingRuntime)
    runtime.config = SortingPlannerConfig(tcp_orientation_wxyz=(0, 1, 0, 0))
    np.testing.assert_allclose(runtime.grasp_rotation[:, 1], [0, -1, 0], atol=1e-12)
    runtime.config.tcp_orientation_wxyz = (0, 2**-0.5, 2**-0.5, 0)
    np.testing.assert_allclose(runtime.grasp_rotation[:, 1], [1, 0, 0], atol=1e-12)


def test_right_approach_switches_the_measured_tcp_and_contact_aliases():
    from types import SimpleNamespace

    from real2sim2real.planner.sorting_runtime import SortingRuntime

    runtime = object.__new__(SortingRuntime)
    runtime.active_side = "left"
    runtime.hand_bodies = dict(left=10, right=20)
    runtime.contact_body_ids = dict(left_0=11, left_1=12, right_0=21, right_1=22, palm=10)
    runtime.target_q = np.zeros(14)
    runtime.robot = SimpleNamespace(arm_indices=dict(right=list(range(7, 14))))
    runtime.approach_start, runtime.approach_goal = np.zeros(7), np.ones(7)
    runtime.set_gripper = lambda side, closed: None
    runtime.command_approach("right", 0.5)
    assert runtime.active_side == "right"
    assert runtime.contact_body_ids["palm"] == 20
    assert (runtime.contact_body_ids["left"], runtime.contact_body_ids["right"]) == (21, 22)
    np.testing.assert_array_equal(runtime.target_q[:7], 0)
    np.testing.assert_array_equal(runtime.target_q[7:], 0.5)


def test_external_contacts_query_current_rigids_and_previous_particles():
    from types import SimpleNamespace

    from real2sim2real.planner.sorting_runtime import SortingRuntime

    runtime = object.__new__(SortingRuntime)
    runtime.state_0 = SimpleNamespace(
        body_q=object(), body_qd=object(), particle_q=object(), particle_qd=object()
    )
    runtime.state_1 = SimpleNamespace(body_q=object(), body_qd=object())
    runtime._collision_views = {}
    runtime.newton = SimpleNamespace(State=SimpleNamespace)
    runtime.model = SimpleNamespace(device="cpu")
    runtime.wp = SimpleNamespace(
        launch=lambda *a, **k: None, transform=lambda *a: a, vec3=lambda *a: a
    )
    runtime._belt_motion, runtime.belt_body = object(), 0
    runtime.belt_rest, runtime.belt_speed = np.zeros(7), 0
    runtime.contacts, runtime.control = object(), object()
    queries, solves = [], []
    runtime.pipeline = SimpleNamespace(collide=lambda *args: queries.append(args))
    runtime.cloth_solver = SimpleNamespace(step=lambda *args: solves.append(args))
    runtime._advance_soft(1 / 1920)
    view = queries[0][0]
    assert view.particle_q is runtime.state_0.particle_q
    assert view.body_q is runtime.state_1.body_q
    assert view.body_q is not runtime.state_0.body_q
    assert solves[0][:2] == (runtime.state_0, runtime.state_1)


def test_jaw_audit_detects_a_face_crossing_even_when_all_vertices_are_outside():
    from real2sim2real.utils.sorting_quality import audit_jaw_penetration

    cube = np.array(
        [[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (-0.01, 0.01)]
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ]
    )
    vertices = np.array([[[-0.02, 0, 0], [0.02, -0.02, 0], [0.02, 0.02, 0]]])
    poses = np.array([[[0, 0, 0, 0, 0, 0, 1]]], float)
    jaws = [dict(body_id=0, vertices=cube, triangles=faces)]
    report = audit_jaw_penetration(vertices, np.array([[0, 1, 2]]), poses, jaws)
    assert not report["passed"]
    assert report["maximum_penetration_m"] == pytest.approx(0.01 - 0.02 / 3)
    assert report["worst_penetration"]["probe_kind"] == "face_centre"
    assert report["worst_penetration"]["probe_index"] == 3
    assert audit_jaw_penetration(vertices + 0.1, np.array([[0, 1, 2]]), poses, jaws)["passed"]


@pytest.mark.skipif(
    not __import__("pathlib").Path("assets/Robot/pro_m6_696/pro_m6_696.usda").is_file(),
    reason="Local Pro-M6 USD required for native gripper regression",
)
def test_native_mimics_are_not_duplicated_and_closed_fingers_clear_the_support():
    import os
    import sys
    from pathlib import Path

    usd = next(Path("app/extscache").glob("omni.usd.libs-*/pxr/Gf")).parent.parent.resolve()
    newton = Path("app/exts/isaacsim.pip.newton/pip_prebundle").resolve()
    warp_sdk = next(Path("app/extscache").glob("omni.warp.core-*/warp")).parent.resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(newton), str(warp_sdk), str(usd), environment.get("PYTHONPATH", "")]
    )
    environment["LD_LIBRARY_PATH"] = (
        str(usd / "bin") + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    )
    code = """
import numpy as np
import newton,warp as wp
wp.set_device('cpu')
from newton.solvers import SolverMuJoCo
from real2sim2real.robot.robot_sorting import SortingRobot
from real2sim2real.planner.sorting_kinematics import SortingArmIK
from real2sim2real.planner.planner_physic_solver import _rotation_wxyz
from real2sim2real.utils.sorting_quality import unit_box_mesh
robot=SortingRobot();robot.source_path='assets/Robot/pro_m6_696/pro_m6_696.usda'
builder=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(builder)
robot.add_newton_robot(builder)
model=builder.finalize(device='cpu')
assert model.constraint_mimic_count==10
pairs=list(zip(model.constraint_mimic_joint0.numpy(),model.constraint_mimic_joint1.numpy()))
assert len(set(pairs))==10
initial=model.joint_q.numpy().copy()
q=initial.copy()
for side,followers in robot.mimic.items():
    q[robot.gripper_indices[side]]=0.675
    for index,ratio in followers:q[index]=ratio*0.675
state=model.state()
newton.eval_fk(model,wp.array(q,dtype=float,device='cpu'),model.joint_qd,state)
ik=SortingArmIK(robot,'left')
hand=state.body_q.numpy()[model.body_label.index(ik.body_path)]
transform=np.eye(4);transform[:3,:3]=_rotation_wxyz(hand[[6,3,4,5]]);transform[:3,3]=hand[:3]
tcp=transform@ik.tip
bounds=[]
for shape,body in enumerate(builder.shape_body):
    if (body<0 or '/robotiq_left/' not in builder.body_label[body]
        or not builder.body_label[body].endswith('inner_finger')):
        continue
    mesh=builder.shape_source[shape]
    if builder.shape_type[shape]==newton.GeoType.BOX:
        vertices,_=unit_box_mesh()
    else:
        assert mesh is not None
        vertices=mesh.vertices
    pose=state.body_q.numpy()[body]
    transform=wp.transform(pose[:3],pose[3:])
    scale=np.asarray(builder.shape_scale[shape])
    points=np.array([wp.transform_point(transform,wp.transform_point(
        builder.shape_transform[shape],wp.vec3(*(v*scale)))) for v in vertices])
    local=(np.c_[points,np.ones(len(points))]@np.linalg.inv(tcp).T)[:,:3]
    bounds.append((local.min(0),local.max(0)))
assert len(bounds)==4  # both native pad boxes and both metal finger meshes
assert 0.010<0.048-max(high[2] for low,high in bounds)<0.020
bounds.sort(key=lambda pair:pair[0][1])
negative=[high[1] for low,high in bounds if high[1]<0]
positive=[low[1] for low,high in bounds if low[1]>0]
gap=min(positive)-max(negative)
assert 0.004<gap<0.006
np.testing.assert_array_equal(model.joint_q.numpy(),initial)

# Batching only the constant inputs must preserve every solved substep. This
# compares native integration, mimic equalities and FK, rather than final poses.
from types import SimpleNamespace
from real2sim2real.planner.sorting_runtime import SortingRuntime
from real2sim2real.planner.planner_sorting import SortingPlannerConfig
runtime=object.__new__(SortingRuntime)
runtime.model=model;runtime.robot=robot;runtime.wp=wp;runtime.newton=newton
runtime.state_0=state;runtime.config=SortingPlannerConfig(grasp_pad_bottom_m=.03)
runtime._create_ik()
runtime.contact_body_ids=runtime._contact_bodies()
runtime.pad_colliders=[]
for shape,body in enumerate(builder.shape_body):
    if builder.shape_type[shape]!=newton.GeoType.BOX:continue
    vertices,_=unit_box_mesh()
    points=np.array([wp.transform_point(builder.shape_transform[shape],
        wp.vec3(*(v*np.asarray(builder.shape_scale[shape])))) for v in vertices])
    runtime.pad_colliders.append(dict(body_id=body,vertices=points))
before=state.body_q.numpy().copy()
runtime._calibrate_pad_bottom()
np.testing.assert_array_equal(state.body_q.numpy(),before)
np.testing.assert_array_equal(model.joint_q.numpy(),initial)
for angle in [.2825,.6425,.6752]:
    check_q=initial.copy()
    for followers in robot.mimic.values():
        for index,ratio in followers:check_q[index]=ratio*angle
    scratch=model.state()
    newton.eval_fk(model,wp.array(check_q,dtype=float,device='cpu'),model.joint_qd,scratch)
    poses=scratch.body_q.numpy()
    for side in ('left','right'):
        pose=poses[runtime.hand_bodies[side]]
        transform=np.eye(4);transform[:3,:3]=_rotation_wxyz(pose[[6,3,4,5]])
        transform[:3,3]=pose[:3];tcp=transform@runtime.iks[side].tip
        height=runtime.pad_bottom_tcp_height(side,angle,.03)
        bottoms=[]
        for shape,body in enumerate(builder.shape_body):
            if (builder.shape_type[shape]!=newton.GeoType.BOX
                or f'/robotiq_{side}/' not in builder.body_label[body]):continue
            vertices,_=unit_box_mesh();pose=poses[body]
            transform=wp.transform(pose[:3],pose[3:])
            points=np.array([wp.transform_point(transform,wp.transform_point(
                builder.shape_transform[shape],wp.vec3(*(v*np.asarray(builder.shape_scale[shape])))))
                for v in vertices])
            local=(np.c_[points,np.ones(len(points))]@np.linalg.inv(tcp).T)[:,:3]
            bottoms.append((local@runtime.grasp_rotation.T)[:,2].min()+height)
        assert abs(min(bottoms)-.03)<2e-6
assert runtime.pad_bottom_tcp_height('left',.28,.03)<runtime.pad_bottom_tcp_height('left',.64,.03)
robot.sim_world=SimpleNamespace(config=SimpleNamespace(physics_dt=1/60))
histories=[]
for batched,soft_steps,rigid_steps in (
    (False,32,1),(True,32,1),(True,8,4),(True,32,4),(True,64,2)
):
    runtime=object.__new__(SortingRuntime)
    runtime.model=model;runtime.robot=robot;runtime.wp=wp
    runtime.belt_speed=0.0
    runtime.config=SimpleNamespace(newton_substeps=soft_steps,mujoco_cpu=True,
        mujoco_substeps_per_vbd_step=rigid_steps,
        mujoco_control_per_frame=batched,mujoco_iterations=100,
        mimic_time_constant_s=.0015,mimic_impedance=.999,newton_use_cuda_graph=False)
    runtime.state_0=model.state();runtime.state_1=model.state()
    for state in (runtime.state_0,runtime.state_1):
        newton.eval_fk(model,model.joint_q,model.joint_qd,state)
    runtime.control=model.control()
    runtime.rigid_solver=runtime._create_mujoco_solver()
    runtime.cloth_solver=SimpleNamespace(rebuild_bvh=lambda state:None)
    history=[]
    runtime._advance_soft=lambda dt:history.append(runtime.state_1.joint_q.numpy().copy())
    for frame in range(3):
        target=initial.copy()
        target[robot.arm_indices['left']]+=0.01*(frame+1)
        runtime.control.joint_target_q.assign(target)
        runtime._advance_direct(1/60)
    if batched:
        residuals=np.concatenate(runtime.native_mimic_residuals)
        assert residuals.shape==(3*soft_steps*rigid_steps,12)
        assert runtime.maximum_native_mimic_error==np.max(abs(residuals))
    histories.append(history)
assert len(histories[0])==96
np.testing.assert_allclose(histories[0],histories[1],atol=1e-9,rtol=0)
assert len(histories[2])==24
np.testing.assert_allclose(histories[1][3::4],histories[2],atol=1e-9,rtol=0)
np.testing.assert_allclose(histories[3],histories[4][1::2],atol=1e-9,rtol=0)

# Only the original robot geometry is allowed, including its native pads.
from real2sim2real.robot.robot_sorting import SortingRobotConfig
from pydantic import ValidationError
try:
    SortingRobotConfig(pad_width_m=.06)
except ValidationError:
    pass
else:
    raise AssertionError("Widened pads must be rejected")
other=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(other)
robot.add_newton_robot(other)
native=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(native)
native.add_usd(robot.source_stage, floating=False, collapse_fixed_joints=True,
    enable_self_collisions=False, load_visual_shapes=False)
for field in ("body_mass", "body_inertia", "body_com", "shape_type", "shape_scale",
              "shape_body", "shape_transform"):
    np.testing.assert_array_equal(np.asarray(getattr(other,field)),np.asarray(getattr(native,field)))
assert other.shape_count==native.shape_count

opened=SortingRobot(SortingRobotConfig(initial_gripper_rad=.28))
opened.source_path=robot.source_path
open_builder=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(open_builder)
opened.add_newton_robot(open_builder)
for side,pairs in opened.mimic.items():
    assert open_builder.joint_q[opened.gripper_indices[side]]==.28
    for index,ratio in pairs:assert open_builder.joint_q[index]==ratio*.28
np.testing.assert_array_equal(open_builder.body_mass,native.body_mass)
np.testing.assert_array_equal(open_builder.body_inertia,native.body_inertia)
assert open_builder.shape_count==native.shape_count

# Translating the entire solver frame must preserve articulated FK, free-body
# poses, world-static geometry, masses and the native MuJoCo dynamics.
origin=np.array([.55,-.9,1.78])
floor=other.add_shape_box(-1,xform=wp.transform((-3,-.8,1.78),wp.quat_identity()),
    hx=.3,hy=.3,hz=.02)
belt=other.add_body(xform=wp.transform((-1,-.8,1.78),wp.quat_identity()),is_kinematic=True)
other.add_shape_box(belt,hx=.3,hy=.3,hz=.02)
other.add_particle(pos=wp.vec3(.55,-.9,1.8),vel=wp.vec3(0),mass=.1)
other.color(include_bending=True)
reference=other.finalize(device='cpu')
runtime=object.__new__(SortingRuntime)
runtime.newton=newton;runtime.wp=wp;runtime.physics_origin=origin
runtime.robot=SimpleNamespace(sim_world=SimpleNamespace(config=SimpleNamespace(device='cpu')))
local=runtime._finalize_model(other)
np.testing.assert_array_equal(local.body_mass.numpy(),reference.body_mass.numpy())
np.testing.assert_array_equal(local.body_label,reference.body_label)
assert local.constraint_mimic_count==reference.constraint_mimic_count==10
np.testing.assert_allclose(local.shape_transform.numpy()[floor,:3]+origin,
    reference.shape_transform.numpy()[floor,:3],atol=3e-7)
world_states=[]
for model,offset in [(reference,np.zeros(3)),(local,origin)]:
    a,z=model.state(),model.state()
    for state in (a,z):newton.eval_fk(model,model.joint_q,model.joint_qd,state)
    runtime.state_0=a;runtime.physics_origin=offset
    initial_poses=runtime.body_poses()
    initial_particles=runtime.vertices()
    solver=SolverMuJoCo(model,use_mujoco_cpu=True,update_data_interval=0,
        integrator='implicitfast',iterations=100)
    solver.reset(a)
    control=model.control()
    for _ in range(32):
        count=model.particle_count;model.particle_count=0
        try:solver.step(a,z,control,None,1/1920)
        finally:model.particle_count=count
        a,z=z,a
    runtime.state_0=a
    world_states.append((initial_poses,initial_particles,runtime.body_poses(),a.joint_q.numpy()))
for i in range(3):
    np.testing.assert_allclose(world_states[0][i],world_states[1][i],atol=8e-7,rtol=0)
np.testing.assert_allclose(world_states[0][3][:26],world_states[1][3][:26],atol=2e-7,rtol=0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_stream_config_keeps_successful_soft_asset_and_native_head_mount(tmp_path):
    from real2sim2real.job.job_sorting import SortingJob, SortingJobConfig
    from real2sim2real.planner.planner_sorting_stream import SortingStreamPlanner

    try:
        apply_settings_file("configs/PhysicSolver_mujoco_vbd_sorting_stream.toml")
        config = SortingJobConfig.from_settings("job")
        config.output_dir = str(tmp_path)
        job = SortingJob(config=config)
        assert isinstance(job.planner, SortingStreamPlanner)
        assert "v10_station" in job.scene.config.cloth_usd_path
        assert job.scene.config.clear_legacy_parcels
        assert job.scene.config.infeed_travel_m == 0.75
        assert job.planner.config.continuous_outfeed
        head = next(camera for camera in job.robot.config.cameras if camera.name == "head")
        assert head.pose_relative_to.endswith("/tn__Headdepth_Link_oE/Camera_Head")
        assert head.pose.position == (0.0, 0.0, 0.0)
        assert len(job.recording_cameras) == 4
    finally:
        reset_settings()


def test_showcase_config_uses_four_hd_views_without_starting_simulation(tmp_path):
    from real2sim2real.job.job_sorting import SortingJobConfig

    try:
        apply_settings_file("configs/PhysicSolver_mujoco_vbd_sorting_stream.toml")
        base_config = SortingJobConfig.from_settings("job")
        base_config.output_dir = str(tmp_path / "base")
        original = SortingJob(config=base_config)
        apply_settings_file("configs/PhysicSolver_mujoco_vbd_sorting_stream_showcase.toml")
        config = SortingJobConfig.from_settings("job")
        assert config.output_dir == "outputs/sorting_stream_showcase_sync"
        config.output_dir = str(tmp_path / "showcase")
        showcase = SortingJob(config=config)
        assert showcase.sim_world.app is None
        assert showcase.sim_world.config.renderer == "PathTracing"
        assert showcase.sim_world.config.synchronous_capture
        assert not showcase.sim_world.config.pathtracing_cached
        assert not showcase.sim_world.config.pathtracing_many_light_sampling
        assert not showcase.sim_world.config.pathtracing_temporal_denoising
        assert not showcase.sim_world.config.auto_exposure
        assert showcase.config.recorder.fps == 15
        assert showcase.record_stride == 4
        assert [camera.name for camera in showcase.recording_cameras] == [
            "overview",
            "head",
            "left_wrist",
            "right_wrist",
        ]
        for camera, source in zip(
            showcase.recording_cameras, original.recording_cameras, strict=True
        ):
            assert camera.resolution == (1920, 1440)
            assert camera.model_dump(exclude={"resolution"}) == source.model_dump(
                exclude={"resolution"}
            )
        for section in ("scene", "robot"):
            assert getattr(showcase, section).config.model_dump(exclude={"cameras"}) == getattr(
                original, section
            ).config.model_dump(exclude={"cameras"})
        assert showcase.planner.config == original.planner.config
    finally:
        reset_settings()


@pytest.mark.parametrize("advance_physics", [False, True])
def test_all_four_views_share_one_physics_step(monkeypatch, tmp_path, advance_physics):
    import sys
    from types import ModuleType, SimpleNamespace

    import real2sim2real.job.job_physic_solver as module

    viewport = SimpleNamespace(camera_path="/World/GUI", resolution=(800, 600))
    utility = ModuleType("omni.kit.viewport.utility")
    utility.get_active_viewport = lambda: viewport
    monkeypatch.setitem(sys.modules, utility.__name__, utility)
    job = object.__new__(SortingJob)
    job.sim_world = SimpleNamespace(physics_steps=0)
    names = ["overview", "head", "left_wrist", "right_wrist"]
    job.recording_cameras = [
        SimpleNamespace(name=name, prim_path=f"/World/{name}", resolution=(1920, 1440))
        for name in names
    ]
    recorder = SimpleNamespace(frame_path=lambda name: tmp_path / f"{name}.png")
    captured = []

    def capture(world, path, **kwargs):
        captured.append((world.physics_steps, path.stem))
        viewport.camera_path = kwargs["camera_prim"]
        viewport.resolution = (kwargs["width"], kwargs["height"])
        if advance_physics:
            world.physics_steps += 1

    monkeypatch.setattr(module, "capture_viewport_frame", capture)
    if advance_physics:
        with pytest.raises(RuntimeError, match="advanced physics between views"):
            job._capture_cameras(recorder)
        assert captured == [(0, "overview")]
    else:
        job._capture_cameras(recorder)
        job.sim_world.physics_steps = 4
        job._capture_cameras(recorder)
        assert captured == [(step, name) for step in (0, 4) for name in names]
    assert (viewport.camera_path, viewport.resolution) == ("/World/GUI", (800, 600))


@pytest.mark.parametrize("fail_capture", [False, True])
@pytest.mark.parametrize("camera_sequence", [False, True])
def test_synchronous_pt_flushes_view_before_locking_history(
    monkeypatch, tmp_path, fail_capture, camera_sequence
):
    import asyncio
    import sys
    from pathlib import Path
    from types import ModuleType, SimpleNamespace
    from unittest.mock import Mock

    from real2sim2real.simworld.isaacsim import IsaacsimWorld, IsaacsimWorldConfig
    from real2sim2real.utils.lerobot_v3 import capture_viewport_frame

    loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "ensure_future", lambda coroutine: loop.create_task(coroutine))
    control = "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges"
    values = {}
    settings = SimpleNamespace(
        get=values.get,
        set=lambda key, value: values.__setitem__(key, value),
        set_bool=lambda key, value: values.__setitem__(key, value),
        set_int=lambda key, value: values.__setitem__(key, value),
    )
    rendered, captured = [], []
    viewport = SimpleNamespace(camera_path="/World/GUI", resolution=(640, 480))

    async def wait_for_rendered_frames(count):
        rendered.append((count, values[control]))
        await asyncio.sleep(0)

    viewport.wait_for_rendered_frames = wait_for_rendered_frames

    def capture_to_file(viewport, filename):
        captured.append(values["/rtx/externalFrameCounter"])

        async def wait_for_result(**kwargs):
            if fail_capture:
                raise RuntimeError("capture unavailable")
            Path(filename).write_bytes(b"mock image")
            await asyncio.sleep(0)

        return SimpleNamespace(
            wait_for_result=wait_for_result, view=np.eye(4), resolution=viewport.resolution
        )

    modules = {
        name: ModuleType(name)
        for name in (
            "omni",
            "omni.usd",
            "omni.kit",
            "omni.kit.viewport",
            "omni.kit.viewport.utility",
            "omni.kit.renderer_capture",
            "pxr",
        )
    }
    reset = Mock()
    modules["omni"].usd = modules["omni.usd"]
    modules["omni.usd"].get_context = lambda: SimpleNamespace(reset_renderer_accumulation=reset)
    utility = modules["omni.kit.viewport.utility"]
    utility.get_active_viewport = lambda: viewport
    utility.capture_viewport_to_file = capture_to_file
    renderer = Mock()
    modules["omni.kit.renderer_capture"].acquire_renderer_capture_interface = lambda: renderer
    modules["pxr"].UsdGeom = SimpleNamespace(
        Camera=object,
        XformCache=lambda: SimpleNamespace(
            GetLocalToWorldTransform=lambda prim: SimpleNamespace(GetInverse=lambda: np.eye(4))
        ),
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    world = IsaacsimWorld(
        IsaacsimWorldConfig(
            renderer="PathTracing",
            samples_per_pixel_per_frame=64,
            rt_subframes=2,
            synchronous_capture=True,
            pathtracing_cached=False,
            pathtracing_many_light_sampling=False,
        )
    )
    world._settings = settings
    world.stage = SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsA=lambda cls: True))
    world.update = lambda: loop.run_until_complete(asyncio.sleep(0))
    try:
        world._configure_rendering()
        assert values["/app/asyncRendering"] is False
        assert values["/rtx/materialDb/syncLoads"] is True
        assert values["/rtx/pathtracing/cached/enabled"] is False
        assert values["/rtx/pathtracing/lightcache/cached/enabled"] is False
        assert values["/rtx/pathtracing/optixDenoiser/temporalMode/enabled"] is False
        renderer.set_capture_sync.assert_called_once_with(True)
        if camera_sequence:
            # The camera-batched caller already warmed this view and keeps
            # accumulation externally controlled across the full sequence.
            viewport.camera_path, viewport.resolution = "/World/overview", (1920, 1440)
            values[control] = True
        if fail_capture:
            with pytest.raises(RuntimeError, match="capture unavailable"):
                capture_viewport_frame(
                    world,
                    tmp_path / "failed.png",
                    width=1920,
                    height=1440,
                    camera_sequence=camera_sequence,
                )
        else:
            for name in ("overview", "head", "left_wrist", "right_wrist"):
                capture_viewport_frame(
                    world,
                    tmp_path / f"{name}.png",
                    width=1920,
                    height=1440,
                    camera_prim="/World/overview" if camera_sequence else f"/World/{name}",
                    camera_sequence=camera_sequence,
                )
            assert captured == [1, 2, 3, 4]
        assert rendered == ([(3, True)] if camera_sequence else [(8, False), (3, True)]) * len(
            captured
        )
        assert values[control] is camera_sequence
        assert reset.call_count == (0 if camera_sequence else len(captured))
        assert world.physics_steps == 0
    finally:
        loop.close()


@pytest.mark.parametrize("advance_physics", [False, True])
def test_camera_batches_keep_exact_source_samples_and_commit_only_complete_groups(
    monkeypatch, tmp_path, advance_physics
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import scripts.render_sorting_package as module
    from real2sim2real.utils.cameras import CameraConfig

    names = ["overview", "head", "left_wrist", "right_wrist"]
    job = object.__new__(module.SortingReplayJob)
    job.output = tmp_path
    job.indices = np.array([0, 4])
    job.saved = dict(
        times=np.arange(5) / 60,
        state=np.arange(10).reshape(5, 2),
        action=np.arange(10, 20).reshape(5, 2),
    )
    values = {}
    control = "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges"
    job.sim_world = SimpleNamespace(
        physics_steps=0,
        config=SimpleNamespace(synchronous_capture=True, renderer="PathTracing"),
        _settings=SimpleNamespace(
            get=values.get, set_bool=lambda key, value: values.__setitem__(key, value)
        ),
    )
    job.recording_cameras = [
        CameraConfig(
            name=name, prim_path=f"/World/{name}", pose_relative_to="/World", resolution=(64, 48)
        )
        for name in names
    ]
    warmed, captured, current = [], [], {}
    job._prepare_offline_render = lambda index, camera: warmed.append((index, camera.name))
    job._publish_sample = lambda index: current.update(index=index)
    recorder = LeRobotV3Recorder(
        tmp_path / "lerobot_v3",
        fps=15,
        width=64,
        height=48,
        task="mock source replay",
        cameras={name: (64, 48) for name in names},
        joint_names=["a", "b"],
    )

    def capture(world, path, **kwargs):
        assert kwargs["camera_sequence"]
        captured.append((path.parent.name, current["index"]))
        Image.new("RGB", (64, 48), (current["index"], 10, 20)).save(path)
        if advance_physics:
            world.physics_steps += 1

    monkeypatch.setattr(module, "capture_viewport_frame", capture)
    # Leave the normal logger lazy; no Isaac Sim resources are created.
    if advance_physics:
        with pytest.raises(RuntimeError, match="advanced physics"):
            job._capture_camera_sequences(recorder, 0)
        assert recorder.rows == []
    else:
        original_add = recorder.add_frame

        def add_frame(state, action):
            assert len(captured) == 8
            original_add(state, action)

        recorder.add_frame = Mock(side_effect=add_frame)
        job._capture_camera_sequences(recorder, 0)
        assert warmed == [(0, name) for name in names]
        assert captured == [(name, index) for name in names for index in (0, 4)]
        assert [row["timestamp"] for row in recorder.rows] == [0, 1 / 15]
        assert [row["observation.state"] for row in recorder.rows] == [[0, 1], [8, 9]]
        assert [row["action"] for row in recorder.rows] == [[10, 11], [18, 19]]
        progress = json.loads((tmp_path / "render_progress.json").read_text())
        assert progress["camera"] == "right_wrist"
        assert progress["source_index"] == 4
        assert progress["physics_steps_advanced"] == 0
    assert values[control] is False


@pytest.mark.parametrize("exit_code", [0, 3])
def test_manual_camera_render_cli_only_launches_when_invoked(monkeypatch, tmp_path, exit_code):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock

    from typer.testing import CliRunner

    from real2sim2real.job.cli import app

    source = tmp_path / "source"
    source.mkdir()
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    launch = Mock(return_value=SimpleNamespace(returncode=exit_code))
    monkeypatch.setattr(subprocess, "run", launch)
    result = CliRunner().invoke(
        app,
        [
            "render-sorting",
            "--source",
            str(source),
            "--output",
            str(tmp_path / "render"),
            "--render-profile",
            str(profile),
        ],
    )
    assert result.exit_code == exit_code, result.output
    arguments = launch.call_args.args[0]
    assert arguments[0] == sys.executable
    assert Path(arguments[1]).name == "render_sorting_package.py"
    assert arguments[2] == str(source)
    assert arguments[-2:] == ["--capture-order", "camera"]


def test_cached_soft_success_cannot_hide_damage_during_the_later_rigid_relay():
    from types import SimpleNamespace

    from real2sim2real.planner.planner_sorting_stream import (
        SortingStreamPlanner,
        SortingStreamPlannerConfig,
    )

    planner = object.__new__(SortingStreamPlanner)
    planner.config = SortingStreamPlannerConfig()
    planner.soft_result = dict(success=True, strict_quality_success=True)
    planner.rigid = SimpleNamespace(result=lambda: dict(success=True), failure_reason=None)
    planner.belt_speeds = [[0.0, planner.config.belt_speed_mps]]
    planner.stream_samples = [np.zeros((3, 3))]
    planner.stream_bodies = [np.zeros((2, 7))]
    planner.stream_states = [np.zeros(16)]
    planner.stream_actions = [np.zeros(16)]
    planner.global_max_stretch = 1.66
    planner.global_min_area = 0.051
    planner.global_min_floor_z = -0.00265
    planner.global_failure_reason = "Later ground collision"
    planner.scene = SimpleNamespace(config=SimpleNamespace(floor_z_m=0.0))
    planner.runtime = SimpleNamespace(
        cloth_solver=SimpleNamespace(maximum_contacts=SimpleNamespace(numpy=lambda: np.zeros(3)))
    )
    planner.carton_spawn_time_s, planner.elapsed = 67.25, 134.5
    result = planner.result()
    assert result["checks"]["soft_parcel"] and result["checks"]["rigid_parcel"]
    assert not result["success"]
    assert not result["strict_quality_success"]
    assert not result["checks"]["soft_entire_stream_stretch_bounded"]
    assert not result["checks"]["soft_entire_stream_floor_clear"]


def test_recorded_float32_carton_pose_publishes_to_usd():
    import os
    import sys
    from pathlib import Path

    usd = next(Path("app/extscache").glob("omni.usd.libs-*/pxr/Gf")).parent.parent.resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(usd) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["LD_LIBRARY_PATH"] = (
        str(usd / "bin") + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy;runpy.run_path('tests/test_sorting_package.py')"
            "['_check_float32_carton_pose']()",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _check_float32_carton_pose():
    from pxr import Gf, Usd, UsdGeom

    from real2sim2real.scene.scene_sorting import SortingScene

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Carton")
    scene = object.__new__(SortingScene)
    scene.rigid_visual_op = root.AddTransformOp()
    scene.rigid_visual = UsdGeom.Imageable(root)
    pose = np.array([1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
    scene.publish_rigid_pose(pose, visible=True)
    np.testing.assert_allclose(root.GetLocalTransformation().ExtractTranslation(), [1, 2, 3])
    np.testing.assert_allclose(
        root.GetLocalTransformation().Transform(Gf.Vec3d(1, 0, 0)), [1, 3, 3], atol=1e-6
    )
    assert scene.rigid_visual.GetVisibilityAttr().Get() == UsdGeom.Tokens.inherited


def test_native_rigid_belt_force_is_contact_limited_and_stops_in_free_fall():
    import os
    import sys
    from pathlib import Path

    newton = Path("app/exts/isaacsim.pip.newton/pip_prebundle").resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(newton) + os.pathsep + environment.get("PYTHONPATH", "")
    code = '''
import mujoco
import numpy as np
from types import SimpleNamespace
from real2sim2real.planner.sorting_stream_runtime import SortingStreamRuntime
model = mujoco.MjModel.from_xml_string("""
<mujoco><option timestep="0.001" integrator="implicitfast"/>
<worldbody><geom name="belt" type="box" size="1 .3 .02" pos="0 0 -.02"
 friction="0 0 0"/>
<body name="carton" pos="0 0 .036"><freejoint/>
<geom name="box" type="box" size=".0315 .05 .035" mass=".03125" friction="0 0 0"/>
</body></worldbody></mujoco>""")
data = mujoco.MjData(model)
runtime = object.__new__(SortingStreamRuntime)
runtime.carton_spawned = True
runtime.carton_mj_body = model.body('carton').id
runtime.carton_geom = model.geom('box').id
runtime.belt_geoms = {model.geom('belt').id: 'outfeed'}
runtime._native_belt_mu = {model.geom('belt').id: .5}
runtime.infeed_speed, runtime.belt_speed = 0., .18
runtime.maximum_belt_force_n = 0.
runtime.rigid_solver = SimpleNamespace(mj_model=model, mj_data=data, _mujoco=mujoco)
for _ in range(1000):
    runtime._before_native_substep(.001)
    mujoco.mj_step(model, data)
assert -.19 < data.qpos[0] < -.15, data.qpos
assert abs(data.qvel[0] + .18) < .01, data.qvel
assert np.linalg.norm(data.xfrc_applied[runtime.carton_mj_body, :3]) < .03125 * 9.81 * .6
# Airborne parcels receive gravity and no conveyor force.
data.qpos[2] = .5
mujoco.mj_forward(model, data)
assert data.ncon == 0
runtime._before_native_substep(.001)
np.testing.assert_array_equal(data.xfrc_applied[runtime.carton_mj_body], 0.)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_native_fixed_belt_cannot_drift_during_batched_cpu_steps():
    import os
    import sys
    from pathlib import Path

    newton = Path("app/exts/isaacsim.pip.newton/pip_prebundle").resolve()
    warp = next(Path("app/extscache").glob("omni.warp.core-*/warp")).parent.resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(newton), str(warp), environment.get("PYTHONPATH", "")]
    )
    code = """
import newton, warp as wp, numpy as np
from newton.solvers import SolverMuJoCo
from real2sim2real.planner.sorting_runtime import SortingRuntime
wp.init()
b = newton.ModelBuilder()
SolverMuJoCo.register_custom_attributes(b)
r = object.__new__(SortingRuntime)
r.wp = wp
belt = r._add_fixed_belt(b, np.array([0., 0., .5]), 'belt')
b.add_shape_box(belt, hx=.5, hy=1., hz=.5)
box = b.add_body(xform=wp.transform((0, 0, 1.036), wp.quat_identity()))
b.add_shape_box(box, hx=.0315, hy=.05, hz=.035,
                cfg=newton.ModelBuilder.ShapeConfig(density=140.))
b.add_ground_plane()
b.color()
m = b.finalize(device='cpu')
s = SolverMuJoCo(m, use_mujoco_cpu=True, update_data_interval=0)
s.reset(m.state())
s.mj_model.opt.timestep = 1/7680
body_map = s.mjc_body_to_newton.numpy()[0]
belt_id = int(np.flatnonzero(body_map == belt)[0])
box_id = int(np.flatnonzero(body_map == box)[0])
assert s.mj_model.body_mocapid[belt_id] >= 0
for _ in range(1500):
    s._mujoco.mj_step(s.mj_model, s.mj_data)
np.testing.assert_allclose(s.mj_data.xpos[belt_id], [0,0,.5], atol=1e-9)
assert s.mj_data.xpos[box_id, 2] > 1.0335
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_fallen_or_edge_overhanging_parcel_is_not_below_an_infinite_support_plane():
    from real2sim2real.scene.scene_sorting import SortingScene

    scene = object.__new__(SortingScene)
    scene.supports = {"belt": (np.array([0.0, 0.0, 0.96]), np.array([1.0, 1.0, 1.0]))}
    # Below the belt's underside is free space, and the nearest end face limits depth.
    assert scene.support_penetration(np.array([[0.5, 0.5, 0.5]])) == 0.0
    assert scene.support_penetration(np.array([[0.9999, 0.5, 0.98]])) == pytest.approx(0.0001)
    assert scene.support_penetration(np.array([[0.5, 0.5, 0.999]])) == pytest.approx(0.001)
