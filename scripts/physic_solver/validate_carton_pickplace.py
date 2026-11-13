"""Recompute rigid-carton transfer, release, support and placement checks without Kit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from real2sim2real.planner.planner_physic_solver import _rotation_wxyz
from real2sim2real.utils.lerobot_v3 import validate_dataset
from real2sim2real.utils.physic_solver_utils import isaac_newton_sdk_environment


def _source_dimensions(source: Path) -> np.ndarray:
    """Measure the original USD bound with Isaac's USD SDK in an isolated, renderer-free reader."""
    reader = """
import json, sys
from pxr import Usd, UsdGeom
stage = Usd.Stage.Open(sys.argv[1])
assert UsdGeom.GetStageMetersPerUnit(stage) == 1
meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
assert len(meshes) == 1
cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
size = cache.ComputeWorldBound(meshes[0]).ComputeAlignedRange().GetSize()
sys.stdout.write(json.dumps(list(size)))
"""
    root = Path(__file__).resolve().parents[2]
    encoded = subprocess.check_output(
        [str(root / "app/python.sh"), "-c", reader, str(source.resolve())],
        env=isaac_newton_sdk_environment(),
        cwd=root,
        text=True,
    )
    return np.array(json.loads(encoded))


def validate(output: Path) -> dict:
    """Use saved measured rigid poses and full-rate rows, rather than copying the Job checks."""
    result = json.loads((output / "result.json").read_text())
    config = json.loads((output / "configuration.json").read_text())
    rows = json.loads((output / "trajectory.json").read_text())
    held = [row for row in rows if row["phase"] == "transfer"]
    released = [row for row in rows if row["phase"] in ("release", "retract", "rest")]
    if not held or not released:
        raise ValueError("Incomplete carton transfer/release measurements")
    initial = np.array(rows[0]["target_xyz_m"])
    final = np.array(rows[-1]["target_xyz_m"])
    lifts = np.array([row["target_xyz_m"][2] - initial[2] for row in held])
    distances = np.array(
        [np.linalg.norm(np.array(row["target_xyz_m"]) - row["tcp_xyz_m"]) for row in held]
    )
    drop = released[0]["target_xyz_m"][2] - min(row["target_xyz_m"][2] for row in released)
    error = np.linalg.norm(
        final[:2] - initial[:2] - np.array(config["planner"]["place_offset_m"][:2])
    )
    settings = config["planner"]
    dimensions = np.array(result["carton_dimensions_m"])
    corners = (
        np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * dimensions / 2
    )
    minimum_z = np.array(
        [
            (
                corners @ _rotation_wxyz(np.array(row["quaternion_xyzw"])[[3, 0, 1, 2]]).T
                + row["target_xyz_m"]
            )[:, 2].min()
            for row in rows
        ]
    )
    np.testing.assert_allclose(minimum_z, [row["minimum_z_m"] for row in rows], atol=1e-7)
    # CartonScene places the unchanged box 1 mm above its measured support.
    table_z = initial[2] - dimensions[2] / 2 - 0.001
    penetration = float(max(0.0, table_z - minimum_z.min()))
    support_error = float(abs(minimum_z[-1] - table_z))
    source = Path(config["scene"]["carton_usd_path"])
    provenance = json.loads((source.parent / "provenance.json").read_text())
    hashes_match = all(
        hashlib.sha256((source.parent / name).read_bytes()).hexdigest() == expected
        for name, expected in provenance["sha256"].items()
    )
    checks = dict(
        physical_outcome_passed=bool(result["success"]),
        sustained_lift=bool(lifts.min() >= settings["minimum_lift_m"]),
        held_near_gripper=bool(distances.max() < 0.04),
        release_drop=bool(drop >= settings["minimum_release_drop_m"]),
        placement=bool(error <= settings["maximum_place_error_m"]),
        no_grasp_attachments=result["attachments_used"] is False,
        original_carton_dimensions=bool(
            np.allclose(dimensions, _source_dimensions(source), atol=1e-7)
        ),
        source_assets_unchanged=hashes_match and provenance["geometry_modified"] is False,
        declared_mass=bool(np.isclose(result["carton_mass_kg"], config["scene"]["carton_mass_kg"])),
        bounded_table_penetration=penetration <= settings["maximum_table_penetration_m"],
        final_support=support_error <= 0.002,
        measured_poses_finite=bool(np.isfinite(minimum_z).all()),
    )
    report = dict(
        success=all(checks.values()),
        checks=checks,
        minimum_transfer_lift_m=float(lifts.min()),
        release_drop_m=float(drop),
        place_error_m=float(error),
        maximum_table_penetration_m=penetration,
        final_support_error_m=support_error,
    )
    if (output / "lerobot_v3").exists():
        cameras = {
            camera["name"]: tuple(camera["resolution"]) for camera in config["scene"]["cameras"]
        }
        report["dataset"] = validate_dataset(output, expected_cameras=cameras)
        import pyarrow.parquet as pq

        table = pq.read_table(output / "lerobot_v3/data/chunk-000/file-000.parquet")
        indices = np.rint(
            np.array(table["timestamp"].to_pylist()) / config["sim_world"]["physics_dt"]
        ).astype(int)
        for key, measured in (("observation.state", "robot_state"), ("action", "robot_target")):
            np.testing.assert_array_equal(
                np.array(table[key].to_pylist(), np.float32),
                np.array([rows[i][measured] for i in indices], np.float32),
            )
        report["recorded_state_action_alignment"] = True
        review = output / "qa/visual_review.json"
        report["visual_inspection"] = (
            "passed"
            if review.exists() and json.loads(review.read_text()).get("passed")
            else "pending"
        )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report = validate(parser.parse_args().output)
    raise SystemExit(0 if report["success"] else 1)
