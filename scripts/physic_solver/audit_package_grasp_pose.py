"""Independently check achieved grasp poses from saved physical joints and URDF FK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from real2sim2real.core.logger import get_logger
from real2sim2real.planner.planner_physic_solver import _URDFHandIK


def audit_grasp_pose(source: Path) -> dict:
    """Compare the last closed grasp with independently composed requested world rotations."""
    configuration = json.loads((source / "configuration.json").read_text())
    rows = json.loads((source / "trajectory.json").read_text())
    with np.load(source / "full_physics.npz") as frames:
        index = [i for i, row in enumerate(rows) if row["phase"] == "close"][-1]
        np.testing.assert_allclose(frames["times"][index], rows[index]["time_s"], atol=1e-8)
        joints = frames["states"][index, :7]
    planner, robot = configuration["planner"], configuration["robot"]
    actual, _ = _URDFHandIK(SimpleNamespace(config=SimpleNamespace(**robot)))._fk(joints)
    yaw, pitch = np.deg2rad([planner["grasp_yaw_deg"], planner["grasp_pitch_deg"]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    downward = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    requested = rz @ ry @ downward
    angle_error = float(
        np.rad2deg(np.arccos(np.clip((np.trace(requested.T @ actual[:3, :3]) - 1) / 2, -1, 1)))
    )
    target = np.array([*configuration["scene"]["target_xy"], robot["base_position"][2]])
    target += np.asarray(planner["grasp_offset_m"]) + [0, 0, planner["grasp_height_m"]]
    position_error = float(np.linalg.norm(actual[:3, 3] - target))
    fk_error = float(np.linalg.norm(actual[:3, 3] - rows[index]["tcp_m"]))
    relative = actual[:3, :3] @ downward.T
    measured = np.rad2deg(
        [np.arctan2(relative[1, 0], relative[0, 0]), np.arcsin(np.clip(-relative[2, 0], -1, 1))]
    )
    report = dict(
        success=bool(position_error <= 0.005 and angle_error <= 1 and fk_error <= 1e-5),
        source=str(source.resolve()),
        frame=index,
        time_s=rows[index]["time_s"],
        requested_yaw_pitch_deg=[planner["grasp_yaw_deg"], planner["grasp_pitch_deg"]],
        measured_yaw_pitch_deg=measured.tolist(),
        requested_tcp_m=target.tolist(),
        measured_tcp_m=actual[:3, 3].tolist(),
        position_error_m=position_error,
        orientation_error_deg=angle_error,
        fk_recorded_tcp_error_m=fk_error,
        thresholds=dict(position_error_m=0.005, orientation_error_deg=1, fk_error_m=1e-5),
        method="Actual physical joints + source URDF FK; independent Rz @ Ry @ downward matrix.",
        scope="Table-mounted parcel task, robot base Z equals the table support plane.",
    )
    (source / "grasp_pose_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    """Audit one or more completed source trajectories, without changing physical data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    reports = [audit_grasp_pose(source) for source in parser.parse_args().sources]
    for report in reports:
        get_logger(__name__).info("Measured package grasp pose", **report)
    return 0 if all(report["success"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
