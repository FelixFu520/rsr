"""Independently compare USD replay visibility and poses with the recorded physical trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim2real.core.logger import get_logger


def validate(output: Path) -> dict:
    """Check actual replay samples with USD; call only after Kit has loaded pxr."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    path = output / "banana_cutting_replay.usdc"
    stage = Usd.Stage.Open(str(path.resolve()))
    rows = json.loads((output / "trajectory.json").read_text())
    events = json.loads((output / "cut_events.json").read_text())
    fps = stage.GetTimeCodesPerSecond()
    counts = []
    maximum_error = 0.0
    knife_error = 0.0
    for row in rows:
        t = row["time_s"] * fps
        visible = [
            prim
            for prim in stage.Traverse()
            if prim.IsA(UsdGeom.Mesh)
            and str(prim.GetPath()).startswith("/World/Food/")
            and UsdGeom.Imageable(prim).ComputeVisibility(t) != "invisible"
        ]
        expected = 1 + sum(event["time_s"] <= row["time_s"] + 1e-9 for event in events)
        counts.append(len(visible) == expected)
        for piece in row["pieces"]:
            matrix = np.asarray(
                UsdGeom.Xformable(stage.GetPrimAtPath(piece["path"])).GetLocalTransformation(t)
            )
            maximum_error = max(maximum_error, float(np.max(np.abs(matrix - piece["transform"]))))
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Knife")).GetLocalTransformation(t)
        knife_error = max(
            knife_error,
            float(np.max(np.abs(np.asarray(matrix.ExtractTranslation()) - row["knife"][:3]))),
        )
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path.resolve()))
    files = [*(layer.realPath for layer in layers), *assets]
    task_root = Path("assets/PhysicSolver").resolve()
    local = all(
        Path(file).resolve().is_relative_to(task_root) or Path(file).resolve() == path.resolve()
        for file in files
        if file
    )
    checks = dict(
        visible_piece_counts=all(counts),
        measured_poses=maximum_error < 1e-8,
        knife_poses=knife_error < 1e-8,
        resolved_dependencies=not unresolved,
        task_local_dependencies=local,
        rigid_integration_disabled=all(
            not UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ),
    )
    report = dict(
        success=all(checks.values()),
        checks=checks,
        maximum_pose_error=maximum_error,
        maximum_knife_position_error_m=knife_error,
        checked_physics_samples=len(rows),
    )
    (output / "replay_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Start a read-only USD process and return an explicit exit code even with fast shutdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "active_gpu": 1, "multi_gpu": False})
    status = 1
    try:
        report = validate(args.output)
        get_logger(__name__).info("Banana replay validation", **report)
        status = 0 if report["success"] else 1
    finally:
        app.close(exit_code=status)


if __name__ == "__main__":
    main()
