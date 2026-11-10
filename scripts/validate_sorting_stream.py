"""Independently audit continuous soft/rigid sorting and synchronized LeRobot v3 videos."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from real2sim2real.core.logger import get_logger
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.sorting_quality import (
    audit_held_shape,
    audit_jaw_penetration,
    without_bag_gap_checks,
)
from scripts.collect_physic_solver_cases import validate_dataset
from scripts.render_sorting_package import (
    ReplayRenderProfile,
    override_render_profile,
    replay_indices,
)


def soft_geometry_quality(positions: np.ndarray, rest: np.ndarray, triangles: np.ndarray) -> dict:
    """Check soft geometry through the later rigid relay, including floor interactions."""
    edges = np.unique(
        np.sort(
            np.concatenate(
                [
                    triangles[:, :2],
                    triangles[:, 1:],
                    triangles[:, [2, 0]],
                ]
            ),
            axis=1,
        ),
        axis=0,
    )
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=-1)
    faces = rest[triangles]
    areas = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=-1)
    maximum_stretch, minimum_area = 1.0, 1.0
    for start in range(0, len(positions), 128):
        chunk = positions[start : start + 128]
        current = np.linalg.norm(chunk[:, edges[:, 0]] - chunk[:, edges[:, 1]], axis=-1)
        maximum_stretch = max(maximum_stretch, float((current / lengths).max()))
        faces = chunk[:, triangles]
        current = np.linalg.norm(
            np.cross(
                faces[:, :, 1] - faces[:, :, 0],
                faces[:, :, 2] - faces[:, :, 0],
            ),
            axis=-1,
        )
        minimum_area = min(minimum_area, float((current / areas).min()))
    return dict(maximum_edge_stretch=maximum_stretch, minimum_triangle_area_ratio=minimum_area)


def outfeed_departure(points: np.ndarray, edge: float, floor: float) -> dict:
    """Require complete belt departure before landing, allowing the parcel to spread below it."""
    exited = points[:, :, 0].max(1) < edge
    # A soft parcel can rotate/spread back underneath the belt after fully leaving
    # its edge. Landing must follow departure; their XY predicates need not coincide.
    departed = np.logical_or.accumulate(exited)
    grounded = departed & (points[:, :, 2].max(1) < floor + 0.12)
    exit_samples = np.flatnonzero(exited)
    ground_samples = np.flatnonzero(grounded)
    return dict(
        fully_exits_outfeed=bool(len(exit_samples)),
        falls_to_ground=bool(len(ground_samples)),
        first_full_exit_sample=int(exit_samples[0]) if len(exit_samples) else None,
        first_ground_sample=int(ground_samples[0]) if len(ground_samples) else None,
    )


def audit(output: Path) -> dict:
    """Recompute source continuity, motion, geometry, and video/state synchronization."""
    result = json.loads((output / "result.json").read_text())
    run = sorted((output / "diagnostics").glob("*/stream.npz"))[-1].parent
    config = json.loads((run / "configuration.json").read_text())
    if json.loads((run / "result.json").read_text()) != result:
        raise ValueError("Rendered run and dataset results disagree")
    source = Path(result["source_physics"])
    if not source.is_absolute():
        source = output / source
    for filename, field in (
        ("stream.npz", "source_stream_sha256"),
        ("particles.npz", "source_trajectory_sha256"),
        ("configuration.json", "source_configuration_sha256"),
    ):
        digest = hashlib.sha256((source / filename).read_bytes()).hexdigest()
        if result[field] != digest:
            raise ValueError(f"Physical source changed: {filename}")
        if (
            filename != "configuration.json"
            and hashlib.sha256((run / filename).read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"Replay copy changed: {filename}")
    expected = copy.deepcopy(json.loads((source / "configuration.json").read_text()))
    overrides = result.get("observation_camera_pose_overrides", {})
    observed = set()
    for section in ("scene", "robot"):
        for camera in expected[section]["cameras"]:
            if camera["name"] in overrides:
                camera["pose"] = overrides[camera["name"]]
                observed.add(camera["name"])
    if result.get("observation_render_profile") is not None:
        expected = override_render_profile(
            expected, ReplayRenderProfile.model_validate(result["observation_render_profile"])
        )
    if expected != config or observed != set(overrides):
        raise ValueError("Replay configuration differs beyond declared optical/render settings")
    outcome = json.loads((output / "lerobot_v3/meta/task_outcome.json").read_text())
    if outcome["physical_result"] != result or outcome["configuration"] != config:
        raise ValueError("LeRobot outcome metadata differs from the rendered source")
    with np.load(run / "stream.npz") as archive:
        stream = {key: archive[key] for key in archive.files}
    with np.load(run / "particles.npz") as archive:
        soft = {key: archive[key] for key in archive.files}
    times = stream["times"]
    info = json.loads((output / "lerobot_v3/meta/info.json").read_text())
    cameras = {
        camera["name"]: tuple(camera["resolution"])
        for section in ("scene", "robot")
        for camera in config[section]["cameras"]
    }
    dataset = validate_dataset(output, allow_failure=False, expected_cameras=cameras)
    data = pq.read_table(output / "lerobot_v3/data/chunk-000/file-000.parquet")
    indices = replay_indices(times, info["fps"])
    if len(data) != len(indices):
        raise ValueError("Video/data frame count differs from the continuous source")
    for feature, key in (("observation.state", "state"), ("action", "action")):
        np.testing.assert_allclose(
            data[feature].to_pylist(), stream[key][indices], atol=1e-6, rtol=0
        )
        if info["features"][feature]["shape"] != [16]:
            raise ValueError("Expected sixteen native joint coordinates")
    dt = config["sim_world"]["physics_dt"]
    poses = stream["body_poses"][:, int(stream["carton_body"])]
    visible = stream["carton_visible"].astype(bool)
    # Transform every original carton vertex, rather than a COM-only proxy.
    quaternion = poses[:, 3:] / np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
    x, y, z, w = quaternion.T
    matrices = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    carton = np.einsum("fij,vj->fvi", matrices, stream["carton_rest_vertices"]) + poses[:, None, :3]
    residuals = stream["native_mimic_residuals"]
    expected_substeps = (
        (len(times) - 1)
        * config["sim_world"]["newton_substeps"]
        * config["planner"]["mujoco_substeps_per_vbd_step"]
    )
    checks = dict(
        continuous_accepted_source=bool(
            result["success"] and result["completed"] and result["continuous_world"]
        ),
        two_parcels_completed=result["sequence"] == ["soft", "rigid"]
        and all(p["success"] for p in result["parcels"].values()),
        finite_physical_samples=all(
            np.isfinite(stream[key]).all() for key in ("positions", "body_poses", "state", "action")
        ),
        uninterrupted_60hz_samples=bool(np.allclose(np.diff(times), dt, atol=1e-9, rtol=0)),
        single_carton_spawn=bool(
            visible.any()
            and np.count_nonzero(np.diff(visible.astype(int)) == 1) == 1
            and not np.any(np.diff(visible.astype(int)) < 0)
        ),
        rigid_no_post_spawn_teleport=bool(
            np.linalg.norm(np.diff(poses[visible, :3], axis=0), axis=1).max() < 0.20
        ),
        constant_outfeed_speed=bool(
            np.all(stream["belt_speeds_mps"][:, 1] == config["planner"]["belt_speed_mps"])
        ),
        painted_belt_motion_matches_physics=bool(
            np.allclose(
                np.diff(stream["belt_offsets_m"], axis=0),
                stream["belt_speeds_mps"][:-1] * dt,
                atol=1e-8,
            )
        ),
        native_substep_audit_complete=residuals.shape
        == (expected_substeps, len(soft["mimic_pairs"])),
        native_mimics_bounded=bool(np.isfinite(residuals).all() and np.abs(residuals).max() < 0.03),
        obsolete_parcels_removed=bool(config["scene"]["clear_legacy_parcels"]),
        original_soft_asset_retained="v10_station" in config["scene"]["cloth_usd_path"],
        native_stream_backend=config["planner"]["backend"] == "sorting_stream"
        and info["robot_type"] == "pro_m6_696_newton",
    )
    for kind, field in (("soft", "cloth_usd_path"), ("rigid", "rigid_usd_path")):
        checks[f"{kind}_asset_hash_matches_source"] = (
            hashlib.sha256(Path(resolve_usd_path(config["scene"][field])).read_bytes()).hexdigest()
            == result["parcels"][kind]["asset_sha256"]
        )
    head = next(camera for camera in config["robot"]["cameras"] if camera["name"] == "head")
    checks["head_uses_native_optical_frame"] = (
        head["pose_relative_to"].endswith("/tn__Headdepth_Link_oE/Camera_Head")
        and head["pose"]["position"] == [0.0, 0.0, 0.0]
        and head["pose"]["orientation_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    )
    rows = json.loads((source / "stream_trajectory.json").read_text())
    infeed = {}
    for kind in ("soft", "rigid"):
        incoming = [row for row in rows if row["parcel"] == kind and row["phase"] == "left_infeed"]
        displacement = incoming[-1]["garment_com_m"][1] - incoming[0]["garment_com_m"][1]
        checks[f"{kind}_arrives_from_remote_infeed"] = (
            displacement >= config["scene"]["infeed_travel_m"] - 0.03
        )
        infeed[kind] = displacement
    floor = config["scene"]["floor_z_m"]
    edge = soft["support_bounds"][2, 0, 0]
    floor_penetration = {}
    departures = {}
    for kind, points in (("soft", stream["positions"]), ("rigid", carton[visible])):
        departure = outfeed_departure(points, edge, floor)
        for name in ("fully_exits_outfeed", "falls_to_ground"):
            checks[f"{kind}_{name}"] = departure[name]
        sample_times = times if kind == "soft" else times[visible]
        for event in ("full_exit", "ground"):
            index = departure[f"first_{event}_sample"]
            departure[f"first_{event}_time_s"] = (
                float(sample_times[index]) if index is not None else None
            )
        departures[kind] = departure
        checks[f"{kind}_floor_penetration_bounded"] = bool(points[:, :, 2].min() >= floor - 0.0015)
        floor_penetration[kind] = max(0.0, floor - float(points[:, :, 2].min()))
    separation = np.maximum(
        np.maximum(
            stream["positions"][-1].min(0) - carton[-1].max(0),
            carton[-1].min(0) - stream["positions"][-1].max(0),
        ),
        0.0,
    )
    support_depths = {}
    for kind, points, triangles in (
        ("soft", stream["positions"], soft["triangles"]),
        ("rigid", carton[visible], stream["carton_triangles"]),
    ):
        maximum = 0.0
        for start in range(0, len(points), 128):
            chunk = points[start : start + 128]
            probes = np.concatenate([chunk, chunk[:, triangles].mean(axis=2)], axis=1)
            for low, high in soft["support_bounds"]:
                depth = np.minimum(probes - low, high - probes).min(axis=-1)
                maximum = max(maximum, float(depth.max()))
        support_depths[kind] = maximum
        checks[f"{kind}_finite_support_geometry_clear"] = maximum <= 0.0015
    jaws = [
        dict(
            body_id=int(body),
            vertices=soft[f"jaw_vertices_{i}"],
            triangles=soft[f"jaw_triangles_{i}"],
        )
        for i, body in enumerate(soft["jaw_body_ids"])
    ]
    rigid_jaws = audit_jaw_penetration(
        carton[visible],
        stream["carton_triangles"],
        stream["body_poses"][visible],
        jaws,
        tolerance_m=0.0015,
    )
    checks["rigid_original_mesh_clear_of_fingers"] = rigid_jaws["passed"]
    soft_jaws = audit_jaw_penetration(
        stream["positions"],
        soft["triangles"],
        stream["body_poses"],
        jaws,
        tolerance_m=0.0015,
    )
    checks["soft_entire_stream_clear_of_fingers"] = soft_jaws["passed"]
    video_samples = set(indices.tolist())
    held = audit_held_shape(
        soft["positions"],
        soft["rest"],
        soft["triangles"],
        soft["masses"],
        int(soft["garment_count"]),
        {side: soft[f"{side}_transfer_indices"] for side in ("left", "right")},
        maximum_shape_rms_m=config["planner"]["maximum_shape_rms_m"],
        maximum_bag_gap_m=config["planner"]["maximum_bag_gap_m"],
        quality_reference=json.loads(str(soft["quality_reference_json"])),
        extra_samples={
            side: sorted(set(soft[f"{side}_transfer_indices"].astype(int).tolist()) & video_samples)
            for side in ("left", "right")
        },
    )
    held_required = (
        held["checks"]
        if config["planner"]["require_bag_gap_quality"]
        else without_bag_gap_checks(held["checks"], held)
    )
    checks["soft_held_shape_bounded"] = all(held_required.values())
    soft_geometry = soft_geometry_quality(stream["positions"], soft["rest"], soft["triangles"])
    checks["soft_entire_stream_stretch_bounded"] = (
        soft_geometry["maximum_edge_stretch"] < config["planner"]["maximum_edge_stretch"]
    )
    checks["soft_entire_stream_triangles_valid"] = (
        soft_geometry["minimum_triangle_area_ratio"]
        > config["planner"]["minimum_triangle_area_ratio"]
    )
    checks = {name: bool(value) for name, value in checks.items()}
    report = dict(
        passed=all(checks.values()),
        checks=checks,
        failed_checks=[name for name, value in checks.items() if not value],
        frames=len(data),
        physics_samples=len(times),
        native_substeps=len(residuals),
        infeed_displacement_m=infeed,
        outfeed_departure=departures,
        maximum_finite_support_depth_m=support_depths,
        maximum_floor_penetration_m=floor_penetration,
        final_parcel_aabb_separation_m=float(np.linalg.norm(separation)),
        rigid_finger_audit=rigid_jaws,
        soft_finger_audit=soft_jaws,
        soft_held_shape=held,
        soft_entire_stream_geometry=soft_geometry,
        dataset=dataset,
        scope=(
            "Continuous source, sampled original meshes, all native substeps, all video frames; "
            "no official LeRobot SDK loading test"
        ),
    )
    destination = output / "qa"
    destination.mkdir(exist_ok=True)
    (destination / "stream_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit(args.output)
    get_logger(__name__).info(
        "Sorting stream audit", passed=report["passed"], failed_checks=report["failed_checks"]
    )
    raise SystemExit(0 if report["passed"] else 1)
