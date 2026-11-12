"""Audit a fixed follow-camera mount against a complete patrol's warehouse geometry.

Run with Isaac python.sh. Search uses the saved 120 Hz trajectory, checks every video
frame, and writes the selected mount plus any occluding collider paths. It does not
change the warehouse or robot motion. Final RGB validation is still required.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from real2sim2real.core.config import apply_settings_file
from real2sim2real.core.logger import get_logger
from real2sim2real.job.base import AutoJob
from real2sim2real.utils.cameras import CameraConfig
from real2sim2real.utils.fire_inspection_utils import pose_matrix, project_point

log = get_logger(__name__)


def _look_at_camera(template: CameraConfig, offset: tuple[float, float, float]) -> CameraConfig:
    from pxr import Gf

    matrix = (
        Gf.Matrix4d()
        .SetLookAt(Gf.Vec3d(*offset), Gf.Vec3d(0.0, 0.0, 0.05), Gf.Vec3d(0.0, 0.0, 1.0))
        .GetInverse()
    )
    q = matrix.ExtractRotationQuat()
    return template.model_copy(
        update={
            "pose": template.pose.model_copy(
                update={"position": offset, "orientation_wxyz": (q.GetReal(), *q.GetImaginary())}
            ),
            "focal_length": 12.0,
            "horizontal_aperture": 27.94,
        }
    )


def _audit(camera: CameraConfig, rows: list[dict], height: float, query) -> dict:
    import carb

    local = pose_matrix(camera.pose.position, camera.pose.orientation_wxyz)
    # A conservative envelope around the robot, inspection module, beacon and moving feet.
    envelope = np.array(
        [[x, y, z] for x in (-0.68, 0.63) for y in (-0.32, 0.32) for z in (0.0, 1.18)]
    )
    # Visibility rays cover the torso, inspection module and four lower-leg regions.
    targets = np.array(
        [[x, y, z] for x in (-0.48, 0.45) for y in (-0.22, 0.22) for z in (0.12, 0.95)]
        + [[0.0, 0.0, 1.05], [0.0, 0.0, 0.55]]
    )
    worst_margin = 1.0
    # Inspect the reported clipping location first, then every recorded instant.
    order = [1754, *range(len(rows))] if len(rows) > 1754 else range(len(rows))
    for index in dict.fromkeys(order):
        row = rows[index]
        x, y = row["xy"]
        yaw = row["yaw"]
        c, s = math.cos(yaw), math.sin(yaw)
        base = pose_matrix((x, y, height), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
        matrix = local @ base
        eye = matrix[3, :3]
        hits = []

        def overlap(hit, found=hits):
            found.append(str(hit.collision))
            return False

        query.overlap_box(
            carb.Float3(0.10, 0.10, 0.10),
            carb.Float3(*eye),
            carb.Float4(0.0, 0.0, 0.0, 1.0),
            overlap,
            True,
        )
        if hits:
            return {
                "passed": False,
                "frame": index,
                "reason": "camera_near_geometry",
                "prim": hits[0],
            }

        def world(points, px=x, py=y, cosine=c, sine=s):
            result = points.copy()
            result[:, 0] = px + cosine * points[:, 0] - sine * points[:, 1]
            result[:, 1] = py + sine * points[:, 0] + cosine * points[:, 1]
            return result

        for target in world(targets):
            delta = target - eye
            distance = float(np.linalg.norm(delta))
            blockers = []

            def ray_hit(hit, found=blockers):
                found.append(str(hit.collision))
                return False

            query.raycast_all(
                carb.Float3(*eye),
                carb.Float3(*(delta / distance)),
                max(0.0, distance - 0.08),
                ray_hit,
                True,
            )
            if blockers:
                return {
                    "passed": False,
                    "frame": index,
                    "reason": "robot_occluded",
                    "prim": blockers[0],
                }
        for corner in world(envelope):
            p = project_point(
                corner,
                matrix,
                focal_length=camera.focal_length,
                horizontal_aperture=camera.horizontal_aperture,
                resolution=camera.resolution,
                near=camera.clipping_range[0],
                far=camera.clipping_range[1],
                margin=0.92,
            )
            if not p.in_view:
                return {"passed": False, "frame": index, "reason": "robot_outside_frame"}
            width, image_height = camera.resolution
            worst_margin = min(
                worst_margin,
                p.u / width,
                1 - p.u / width,
                p.v / image_height,
                1 - p.v / image_height,
            )
    return {
        "passed": True,
        "frames_checked": len(rows),
        "minimum_image_margin": round(worst_margin, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--low-camera", action="store_true", help="Search lower, more level mounts")
    args = parser.parse_args()
    apply_settings_file(args.config)
    job = AutoJob.from_config()
    rows = [json.loads(line) for line in args.trajectory.read_text().splitlines()][
        :: job.record_stride
    ]
    template = job.robot.camera_config("follow")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        job.init_world()
        job.arrange_scene()
        job.sim_world.reset()
        from omni.physx import get_physx_scene_query_interface

        query = get_physx_scene_query_interface()
        candidates = [template]
        if args.low_camera:
            candidates = [
                _look_at_camera(template, (back, side, up))
                for up in (0.8, 0.9, 1.0, 1.1)
                for back in (-1.6, -1.8, -1.4, -1.2)
                for side in (0.35, -0.35, 0.0, 0.6, -0.6)
            ]
        elif args.search:
            candidates += [
                _look_at_camera(template, (back, side, up))
                for up in (1.6, 1.9, 2.2)
                for back in (-1.2, -1.0, -0.8, -0.6)
                for side in (0.0, -0.35, 0.35)
            ]
        reports = []
        selected = None
        for camera in candidates:
            report = _audit(camera, rows, job.robot.config.stance_height_m, query)
            report["camera"] = camera.model_dump(mode="json")
            reports.append(report)
            log.info(
                "跟随相机全路线检查",
                position=camera.pose.position,
                **{k: v for k, v in report.items() if k != "camera"},
            )
            if report["passed"]:
                selected = camera.model_dump(mode="json")
                break
        result = {
            "passed": selected is not None,
            "trajectory": str(args.trajectory.resolve()),
            "frames": len(rows),
            "camera_clearance_box_half_extent_m": 0.10,
            "robot_envelope_xyz_m": [[-0.68, -0.32, 0.0], [0.63, 0.32, 1.18]],
            "image_margin_fraction": 0.04,
            "selected_camera": selected,
            "candidates": reports,
        }
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        if selected is None:
            raise RuntimeError(f"No camera passed; see {args.output}")
    finally:
        job.close()


if __name__ == "__main__":
    main()
