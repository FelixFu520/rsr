"""Run migrated parcel cases serially, preserve every attempt, and independently audit physics.

./app/python.sh scripts/physic_solver/run_package_matrix.py --cases v9_mass100g shake_spring
./app/python.sh scripts/physic_solver/run_package_matrix.py --collect
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from real2sim2real.core.config import apply_settings_file, reset_settings
from real2sim2real.core.logger import get_logger
from real2sim2real.job.job_physic_solver import PhysicSolverJobConfig
from real2sim2real.planner.planner_physic_solver import PackagePlannerConfig
from real2sim2real.scene.scene_physic_solver import PackageSceneConfig
from real2sim2real.utils.physic_solver_utils import check_package_containment, proxy_pose_errors

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/physic_solver/package_cases"
DEBUG = ROOT / "outputs/debug/physic_solver/package_migration"


def _audit_shake(output: Path, rows: list[dict], configuration: dict) -> dict:
    """Verify measured TCP translation and rotation using actual joints and independent URDF FK."""
    from real2sim2real.planner.planner_physic_solver import _rotation_wxyz, _URDFHandIK
    from real2sim2real.utils.physic_solver_utils import package_grasp_quaternion

    indices = [i for i, row in enumerate(rows) if row["phase"] == "shake"]
    if not indices or not (output / "full_physics.npz").exists():
        return dict(motion_verified=False)
    frames = np.load(output / "full_physics.npz")
    ik = _URDFHandIK(SimpleNamespace(config=SimpleNamespace(**configuration["robot"])))
    poses = np.array([ik._fk(frames["states"][i, :7])[0] for i in indices])
    rotation = poses[:, :3, :3] @ _rotation_wxyz(package_grasp_quaternion()[[3, 0, 1, 2]]).T
    angles = np.stack(
        [
            np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])),
            np.arcsin(np.clip(-rotation[:, 2, 0], -1, 1)),
        ],
        axis=1,
    )
    tcp = np.array([rows[i]["tcp_m"] for i in indices])
    translation_range = np.ptp(tcp, axis=0)
    angle_range = np.rad2deg(np.ptp(angles, axis=0))
    settings = configuration["planner"]
    requested = np.abs(
        [*settings["shake_translation_m"], settings["shake_yaw_deg"], settings["shake_pitch_deg"]]
    )
    measured = np.r_[translation_range, angle_range]
    error = float(np.linalg.norm(poses[:, :3, 3] - tcp, axis=1).max())
    return dict(
        motion_verified=bool(
            np.any(requested > 0) and np.all(measured + 1e-6 >= requested) and error <= 1e-5
        ),
        translation_range_m=translation_range.tolist(),
        yaw_pitch_range_deg=angle_range.tolist(),
        maximum_fk_tcp_error_m=error,
        criterion="Actual peak-to-peak ranges reach at least half the configured nominal ranges.",
    )


def _audit_assistance(output: Path, result: dict, configuration: dict) -> dict:
    """Recompute capture locality and no-snap coordinates from full-rate saved physics."""
    from real2sim2real.planner.planner_physic_solver import _rotation_wxyz

    groups = result.get("attachment_groups", [])
    if (
        len(groups) != 2
        or any(g["material"] != "bag" for g in groups)
        or len({int(g["body_id"]) for g in groups}) != 2
        or not (output / "full_physics.npz").exists()
    ):
        return dict(local_capture_verified=False)
    frames = np.load(output / "full_physics.npz")
    garment_count = int(np.load(output / "particles.npz")["garment_count"])
    particle_count = frames["positions"].shape[1]
    robot, planner = configuration["robot"], configuration["planner"]
    if robot["backend"] == "franka_package_pads":
        center, half = np.array(robot["pad_center_m"]), np.array(robot["pad_half_extents_m"])
    else:
        tree = ET.parse(robot["urdf_path"]).getroot()
        candidates = []
        for collision in tree.find("./link[@name='fr3_leftfinger']").findall("collision"):
            box, origin = collision.find("geometry/box"), collision.find("origin")
            if box is not None and np.allclose(
                np.fromstring(origin.get("rpy", "0 0 0"), sep=" "), 0
            ):
                candidates.append(
                    (
                        np.fromstring(origin.get("xyz"), sep=" "),
                        np.fromstring(box.get("size"), sep=" ") / 2,
                    )
                )
        center, half = max(candidates, key=lambda pair: pair[0][2])
    used, local_errors, distances = [], [], []
    for group in groups:
        ids = np.asarray(group["ids"], int)
        if ids.ndim != 1 or np.any((ids < garment_count) | (ids >= particle_count)):
            return dict(local_capture_verified=False, reason="Attachment IDs must belong to bag")
        frame = int(np.argmin(abs(frames["times"] - group["capture_time_s"])))
        pose = frames["bodies"][frame, group["body_id"]]
        local = (frames["positions"][frame, ids].astype(float) - pose[:3]) @ _rotation_wxyz(
            pose[[6, 3, 4, 5]]
        )
        local_errors.append(float(np.max(np.abs(local - np.array(group["local_points_m"])))))
        delta = abs(local - center) - half
        distances.extend(
            (np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(1), 0)).tolist()
        )
        used.extend(ids.tolist())
    distances = np.abs(distances)
    return dict(
        local_capture_verified=bool(
            max(local_errors) < 5e-6
            and max(distances) <= planner["attachment_capture_distance_m"] + 5e-6
            and len(used) == len(set(used))
            and all(
                3 <= len(group["ids"]) <= planner["attachment_vertices_per_group"]
                for group in groups
            )
        ),
        maximum_capture_local_error_m=max(local_errors),
        maximum_capture_pad_distance_m=float(max(distances)),
        bag_vertex_indices_verified=True,
        distinct_jaws_verified=True,
    )


def audit_physics(output: Path, config: PackagePlannerConfig) -> dict:
    """Recompute shape, clearance, lift, release, contact retention and containment from disk."""
    result = json.loads((output / "result.json").read_text())
    configuration = json.loads((output / "configuration.json").read_text())
    samples = np.load(output / "particles.npz")
    points, rest, tri = samples["positions"], samples["rest"], samples["triangles"]
    n, masses = int(samples["garment_count"]), samples["masses"]
    times = samples["times"]
    rows_path = output / "trajectory.json"
    if not rows_path.exists():
        rows_path = max(output.glob("diagnostics/*/trajectory.jsonl"))
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    else:
        rows = json.loads(rows_path.read_text())
    edges = np.unique(
        np.sort(np.concatenate([tri[:, :2], tri[:, 1:], tri[:, [2, 0]]]), axis=1), axis=0
    )
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    stretch = float(
        np.max(np.linalg.norm(points[:, edges[:, 0]] - points[:, edges[:, 1]], axis=2) / lengths)
    )
    full_stretch = stretch
    full_penetration = max(
        0.0, float(configuration["robot"]["base_position"][2] - points[:, :, 2].min())
    )
    full_path = output / "full_physics.npz"
    proxy_audit = None
    if config.rigid_soft_coupling == "proxy" and not full_path.exists():
        raise ValueError("Two-way coupling requires saved full-rate proxy geometry")
    if full_path.exists():
        full = np.load(full_path)
        if config.rigid_soft_coupling == "proxy":
            position, angle = proxy_pose_errors(full["bodies"], full["proxy_bodies"])
            substeps = np.asarray(result["maximum_substep_proxy_position_rotation_error"])
            efforts = full["actuator_forces"]
            maximum_efforts = np.asarray(result["maximum_source_joint_actuator_efforts"])
            force_limit = configuration["robot"]["finger_max_force"]
            force_bounded = bool(
                efforts.shape == full["states"].shape
                and np.isfinite(efforts).all()
                and np.isfinite(full["native_external_wrenches"]).all()
                and np.isfinite(full["native_controls"]).all()
                and np.max(abs(efforts[:, -2:])) <= force_limit + 1e-5
                and max(maximum_efforts[-2:]) <= force_limit + 1e-5
                and np.all(maximum_efforts + 1e-5 >= np.max(abs(efforts), axis=0))
            )
            proxy_audit = dict(
                maximum_saved_position_error_m=position,
                maximum_saved_rotation_error_rad=angle,
                maximum_substep_errors=substeps.tolist(),
                source_finger_effort_limit_n=force_limit,
                maximum_source_joint_efforts=maximum_efforts.tolist(),
                finger_force_limit_respected=force_bounded,
                passed=bool(
                    np.isfinite(substeps).all()
                    and position <= 0.0015
                    and angle <= np.deg2rad(1.0)
                    and substeps[0] <= 0.0015
                    and substeps[1] <= np.deg2rad(1.0)
                    and substeps[0] + 1e-6 >= position
                    and substeps[1] + 1e-5 >= angle
                ),
            )
        full_points = full["positions"]
        full_times = full["times"]
        np.testing.assert_allclose(
            full_times,
            np.arange(len(full_times)) * configuration["sim_world"]["physics_dt"],
            atol=1e-8,
        )
        for start in range(0, len(full_points), 32):
            chunk = full_points[start : start + 32]
            if not np.isfinite(chunk).all():
                raise ValueError("Non-finite full-rate physical particles")
            full_stretch = max(
                full_stretch,
                float(
                    np.max(
                        np.linalg.norm(chunk[:, edges[:, 0]] - chunk[:, edges[:, 1]], axis=2)
                        / lengths
                    )
                ),
            )
            full_penetration = max(
                full_penetration,
                float(configuration["robot"]["base_position"][2] - chunk[:, :, 2].min()),
            )
    containment = check_package_containment(
        points, tri, n, times=times, checkpoint_times_s=samples["checkpoint_times_s"].tolist()
    )
    com = np.stack(
        [
            np.average(points[:, :n], axis=1, weights=masses[:n]),
            np.average(points[:, n:], axis=1, weights=masses[n:]),
        ],
        axis=1,
    )
    baseline = int(np.argmin(abs(times - config.approach_s)))
    retained = [row for row in rows if row["phase"] in ("transfer", "shake")]
    fraction = (
        float(np.mean([row["contacts_left"] > 0 and row["contacts_right"] > 0 for row in retained]))
        if retained
        else 0.0
    )
    release = next((i for i, row in enumerate(rows) if row["phase"] == "release"), None)
    release_drop = 0.0
    if release is not None:
        heights = [
            np.mean([row["garment_com_m"][2], row["bag_com_m"][2]]) for row in rows[release:]
        ]
        release_drop = float(heights[0] - min(heights))
    checks = dict(
        source_physics_passed=bool(result["success"]),
        all_particles_finite=bool(np.isfinite(points).all()),
        saved_surface_stretch_bounded=stretch < 1.3,
        full_rate_surface_stretch_bounded=full_stretch < 1.3,
        full_rate_table_penetration_bounded=full_penetration <= config.maximum_table_penetration_m,
        saved_containment_passed=containment["passed"],
        both_surfaces_lifted=bool(
            np.all(np.max(com[:, :, 2] - com[baseline, :, 2], axis=0) >= config.minimum_lift_m)
        ),
        retained_contacts=bool(len(retained) and fraction >= 0.95),
        sustained_transfer_lift=bool(
            retained
            and all(
                row["garment_com_m"][2] - com[baseline, 0, 2] >= config.minimum_lift_m
                and row["bag_com_m"][2] - com[baseline, 1, 2] >= config.minimum_lift_m
                for row in retained
            )
        ),
        release_drop_measured=release_drop >= config.minimum_release_drop_m,
        both_surfaces_rest_on_table=bool(
            rows
            and max(rows[-1]["garment_com_m"][2], rows[-1]["bag_com_m"][2])
            < configuration["robot"]["base_position"][2] + 0.07
        ),
        no_final_jaw_contacts=bool(
            rows and rows[-1]["contacts_left"] == rows[-1]["contacts_right"] == 0
        ),
        attachment_mode_reported=result["attachments_used"] == (config.attachment_mode != "none"),
        source_asset_unchanged=hashlib.sha256(
            Path(configuration["scene"]["cloth_usd_path"]).read_bytes()
        ).hexdigest()
        == result["asset_sha256"],
    )
    for part, weights in (("garment", masses[:n]), ("bag", masses[n:])):
        requested = configuration["scene"].get(f"{part}_mass_kg")
        if requested is not None:
            checks[f"{part}_mass_matches_configuration"] = bool(
                np.isclose(weights.sum(), requested, rtol=1e-6)
            )
    assistance = None
    if config.attachment_mode == "spring":
        assistance = _audit_assistance(output, result, configuration)
        checks["local_attachment_capture_verified"] = assistance["local_capture_verified"]
        released_rows = [row for row in rows if row["phase"] in ("release", "retract", "settle")]
        checks["spring_forces_disabled_on_release"] = bool(
            released_rows and all(not row.get("attachment_active", True) for row in released_rows)
        )
    measured_shake = None
    if config.shake_s:
        shake = [row for row in rows if row["phase"] == "shake"]
        checks["shake_was_executed"] = bool(len(shake) > 0)
        checks["shake_retention"] = bool(
            shake
            and all(
                min(
                    row["garment_com_m"][2] - com[baseline, 0, 2],
                    row["bag_com_m"][2] - com[baseline, 1, 2],
                )
                >= config.minimum_lift_m
                for row in shake
            )
        )
        measured_shake = _audit_shake(output, rows, configuration)
        checks["actual_robot_shake_verified"] = measured_shake["motion_verified"]
    if proxy_audit is not None:
        checks["proxy_geometry_consistent"] = proxy_audit["passed"]
        checks["finger_force_limit_respected"] = proxy_audit["finger_force_limit_respected"]
    audit = dict(
        success=all(checks.values()),
        checks=checks,
        maximum_saved_edge_stretch=stretch,
        maximum_full_rate_edge_stretch=full_stretch,
        maximum_full_rate_table_penetration_m=full_penetration,
        contact_fraction=float(fraction),
        release_drop_m=release_drop,
        containment=containment,
        assistance=assistance,
        measured_shake=measured_shake,
        proxy_geometry=proxy_audit,
    )
    (output / "physics_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> int:
    """Never overwrite attempts or accepted datasets; promote only independently passed results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument(
        "--collect", action="store_true", help="Also record dual-camera LeRobot demonstrations"
    )
    args = parser.parse_args()
    paths = (
        [CONFIGS / f"{name}.toml" for name in args.cases]
        if args.cases
        else sorted(CONFIGS.glob("*.toml"))
    )
    DEBUG.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        apply_settings_file(path)
        config = PackagePlannerConfig.from_settings("planner")
        scene = PackageSceneConfig.from_settings("scene")
        job = PhysicSolverJobConfig.from_settings("job")
        reset_settings()
        attempt = DEBUG / path.stem / run_id
        attempt.mkdir(parents=True)
        output = attempt / "physics" / path.stem
        if args.collect and Path(job.output_dir).resolve().exists():
            raise FileExistsError(f"Refusing to overwrite accepted result: {job.output_dir}")
        environment = os.environ.copy()
        environment.pop("RSR_SETTINGS_FILE", None)
        command = [
            str(ROOT / "app/python.sh"),
            "scripts/physic_solver/check_package_matrix.py",
            "--cases",
            path.stem,
            "--output",
            str(attempt / "physics"),
        ]
        with (attempt / "worker.log").open("w") as stream:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=job.timeout_s + 300,
            )
        row = dict(
            case=path.stem,
            output=str(output),
            returncode=process.returncode,
            scene=scene.model_dump(mode="json"),
            planner=config.model_dump(mode="json"),
        )
        try:
            row["physics_audit"] = audit_physics(output, config)
            row["success"] = process.returncode == 0 and row["physics_audit"]["success"]
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            row.update(success=False, error=f"{type(exc).__name__}: {exc}")
        if row["success"] and args.collect:
            accepted = Path(job.output_dir).resolve()
            if accepted.exists():
                raise FileExistsError(f"Refusing to overwrite accepted result: {accepted}")
            capture = attempt / "capture"
            with (attempt / "capture.log").open("w") as stream:
                captured = subprocess.run(
                    [
                        str(ROOT / "app/python.sh"),
                        "scripts/physic_solver/render_package_trajectory.py",
                        str(output),
                        "--output",
                        str(capture),
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=job.timeout_s + 300,
                )
            try:
                from scripts.physic_solver.render_package_trajectory import (
                    validate_rendered_trajectory,
                )

                validation = validate_rendered_trajectory(output, capture)
                if captured.returncode or not validation["success"]:
                    raise ValueError("Captured physical demonstration failed")
                accepted.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(capture, accepted)
                validate_rendered_trajectory(output, accepted)
                row["dataset"] = str(accepted)
            except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as exc:
                row.update(success=False, capture_error=str(exc))
        records.append(row)
        (DEBUG / f"matrix_{run_id}.json").write_text(
            json.dumps(
                dict(
                    success=all(item["success"] for item in records),
                    completed=len(records),
                    requested=len(paths),
                    cases=records,
                ),
                indent=2,
            )
            + "\n"
        )
        get_logger(__name__).info("Package matrix case", case=path.stem, success=row["success"])
    return 0 if all(row["success"] for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
