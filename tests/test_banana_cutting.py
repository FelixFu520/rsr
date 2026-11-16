"""Runtime clipping conserves closed geometry; migrated backends remain lazy and isolated."""

import json
from pathlib import Path

import numpy as np
import pytest

from real2sim2real.core.config import apply_settings_file, reset_settings
from real2sim2real.job.base import AutoJob
from real2sim2real.planner.planner_physic_solver import BananaPlannerConfig
from real2sim2real.utils.physic_solver_utils import (
    CuttingMesh,
    cutting_banana,
    cutting_evaluate,
    cutting_knife_pose,
    cutting_mesh_report,
    cutting_split,
)


def test_five_oblique_cuts_are_closed_and_conserve_volume_and_materials():
    mesh = cutting_banana()
    initial = cutting_mesh_report(mesh)["volume_m3"]
    fragments = []
    config = BananaPlannerConfig()
    for x, angle in zip(config.cut_positions_m, config.cut_angles_deg, strict=True):
        normal = [np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0]
        left, mesh = cutting_split(mesh, [x, 0, 0], normal)
        for half in (left, mesh):
            report = cutting_mesh_report(half)
            assert report["nonmanifold_edges"] == 0
            assert report["volume_m3"] > 0
            assert 3 in half.materials
            assert set(half.materials) <= {0, 1, 2, 3}
        fragments.append(left)
    volume = sum(cutting_mesh_report(piece)["volume_m3"] for piece in [*fragments, mesh])
    assert abs(volume - initial) / initial < config.maximum_volume_relative_error


def test_missed_plane_preserves_geometry():
    mesh = cutting_banana()
    assert cutting_split(mesh, [2, 0, 0], [1, 0, 0]) is None
    assert cutting_mesh_report(mesh)["nonmanifold_edges"] == 0


@pytest.mark.parametrize("debug", [False, True])
def test_banana_job_constructs_without_kit_and_debug_keeps_physics(debug, monkeypatch, tmp_path):
    suffix = "_debug" if debug else ""
    path = Path(f"configs/physic_solver/kinematic_tgs_banana01_five_cuts{suffix}.toml")
    monkeypatch.setenv("RSR_JOB__OUTPUT_DIR", str(tmp_path / "fresh"))
    try:
        apply_settings_file(path)
        job = AutoJob.from_config()
        assert job.sim_world.app is None
        assert job.robot.config.backend == "scripted_knife"
        assert len(job.recording_cameras) == 1
        assert job.record_stride == (32 if debug else 16)
        assert job.planner.config == BananaPlannerConfig()
        assert job.scene.config.density_kg_m3 == 950
        assert job.sim_world.config.physics_dt == 1 / 480
        assert job.sim_world.config.renderer == ("MinimalRendering" if debug else "PathTracing")
    finally:
        reset_settings()


def test_evaluation_rejects_open_meshes_and_stale_fragments():
    mesh = cutting_banana()
    broken = CuttingMesh(mesh.vertices, mesh.faces[:-1], mesh.materials[:-1])
    result = cutting_evaluate(
        [],
        [dict(time_s=0, knife=[0, 0, 0, 0], pieces=[])],
        expected_cuts=5,
        maximum_volume_relative_error=1e-6,
        minimum_fragment_motion_m=0.001,
        maximum_support_penetration_m=0.002,
        meshes={"banana": broken},
        duration_s=13,
        physics_dt=1 / 120,
    )
    assert not result["success"]
    assert {"cut_count", "closed_meshes", "fragment_motion", "full_duration"} <= set(
        result["failed_checks"]
    )


def test_knife_is_raised_then_sweeps_below_fruit():
    config = BananaPlannerConfig()
    _, _, first, _, _ = cutting_knife_pose(0, config.cut_positions_m, config.cut_angles_deg)
    cycle, phase, cut, normal, _ = cutting_knife_pose(
        2.2, config.cut_positions_m, config.cut_angles_deg
    )
    assert first[2] == 0.15
    assert cycle == 0 and 0.62 < phase < 0.82
    assert cut[2] == 0.033
    np.testing.assert_allclose(normal, [1, 0, 0])


def test_asset_manifest_is_complete_and_source_unchanged():
    import hashlib

    root = Path("assets/PhysicSolver/Banana")
    if not (root / "provenance.json").exists():
        pytest.skip("Prepare task-local banana assets first")
    manifest = json.loads((root / "provenance.json").read_text())
    for entry in manifest["files"]:
        target = root / entry["path"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("tip_drop_m", [None, 0.0, 0.003])
def test_acceptance_checks_gravity_for_last_tip_too(tip_drop_m):
    """A moving left slice must not conceal an omitted or frozen sixth piece."""
    left, tip = cutting_split(cutting_banana(), [-0.096, 0, 0], [1, 0, 0])
    meshes = {"left": left, "tip": tip}
    rows = []
    for time_s in (0.0, 1.0):
        pieces = []
        for path, drop in (("left", 0.003), ("tip", tip_drop_m)):
            if drop is None:
                continue
            center = meshes[path].vertices.mean(axis=0)
            position = center - np.array([0, 0, drop * time_s])
            transform = np.eye(4)
            transform[3, :3] = position
            pieces.append(
                dict(
                    path=path,
                    birth_center_m=center.tolist(),
                    position_m=position.tolist(),
                    transform=transform.tolist(),
                )
            )
        rows.append(dict(time_s=time_s, knife=[0, 0, 0, 0], pieces=pieces))
    result = cutting_evaluate(
        [
            dict(
                old_path="original",
                fragment_path="left",
                remaining_path="tip",
                birth_offset_m=[0, 0, 0],
                initial_velocity_mps=[0, 0, 0],
                initial_angular_velocity_deg_s=[0, 0, 0],
            )
        ],
        rows,
        expected_cuts=1,
        maximum_volume_relative_error=1e-6,
        minimum_fragment_motion_m=0.001,
        maximum_support_penetration_m=0.002,
        meshes={"original": cutting_banana(), **meshes},
        duration_s=1,
        physics_dt=1,
    )
    assert result["checks"]["all_final_pieces_dynamic"] == (tip_drop_m is not None)
    assert result["checks"]["gravity_drop"] == (tip_drop_m is not None and tip_drop_m >= 0.001)


def test_final_cut_releases_remaining_tip_once():
    from types import SimpleNamespace

    from real2sim2real.planner.planner_physic_solver import BananaPlanner

    releases = []
    planner = BananaPlanner(BananaPlannerConfig(cut_positions_m=(-0.096,), cut_angles_deg=(0,)))
    planner.begin_episode(
        SimpleNamespace(
            sim_world=SimpleNamespace(
                physics_steps=528, config=SimpleNamespace(physics_dt=1 / 240)
            ),
            mesh=cutting_banana(),
            split=lambda *args: {},
            release_remaining=lambda: releases.append("tip"),
        )
    )
    robot = SimpleNamespace(command=lambda coordinates: None)
    assert not planner.on_physics_step(robot)
    assert not planner.on_physics_step(robot)
    assert releases == ["tip"]
