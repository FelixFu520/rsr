"""World-frame Pro-M6 IK using the same authored USD joints as Newton."""

from __future__ import annotations

from typing import Any

import numpy as np

from real2sim2real.planner.planner_physic_solver import _rotation_wxyz


def pose_matrix(position: Any, quaternion: Any) -> np.ndarray:
    """Convert a position and wxyz quaternion to a column-vector transform."""
    result = np.eye(4)
    result[:3, :3] = _rotation_wxyz(np.asarray(quaternion, float))
    result[:3, 3] = position
    return result


class SortingArmIK:
    """Seven-joint analytic Jacobian; never writes measured articulation coordinates."""

    def __init__(self, robot: Any, side: str) -> None:
        from pxr import UsdGeom, UsdPhysics

        stage, root = robot.source_stage, robot.source_root
        suffix = "L" if side == "left" else "R"
        self.frames, self.axes, self.lower, self.upper = [], [], [], []
        for i in range(1, 8):
            joint = UsdPhysics.RevoluteJoint.Get(stage, f"{root}/Physics/Joint{i}_{suffix}")
            if not joint:
                raise ValueError(f"Missing Pro-M6 arm joint {i}_{suffix}")
            frames = []
            for number in (0, 1):
                p = getattr(joint, f"GetLocalPos{number}Attr")().Get()
                q = getattr(joint, f"GetLocalRot{number}Attr")().Get()
                frames.append(pose_matrix(p, [q.GetReal(), *q.GetImaginary()]))
            self.frames.append((frames[0], np.linalg.inv(frames[1])))
            self.axes.append(np.eye(3)[{"X": 0, "Y": 1, "Z": 2}[str(joint.GetAxisAttr().Get())]])
            self.lower.append(np.deg2rad(joint.GetLowerLimitAttr().Get()))
            self.upper.append(np.deg2rad(joint.GetUpperLimitAttr().Get()))
            if i == 1:
                parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
                self.base = np.asarray(UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(0)).T
            if i == 7:
                end = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
                world_end = np.asarray(UsdGeom.Xformable(end).ComputeLocalToWorldTransform(0)).T
                self.body_path = str(end.GetPath())
        candidates = [p for p in stage.Traverse() if p.GetName() == f"ToolCenter_{suffix}"]
        candidates.sort(key=lambda p: ("/Geometry/" not in str(p.GetPath()), len(str(p.GetPath()))))
        if not candidates:
            raise ValueError(f"Missing native ToolCenter_{suffix}")
        tcp = candidates[0]
        world_tcp = np.asarray(UsdGeom.Xformable(tcp).ComputeLocalToWorldTransform(0)).T
        self.tip = np.linalg.inv(world_end) @ world_tcp
        self.q = np.array([0.1, 0.1, 0.1, -1.5, 0, 0, -1.3])
        if side == "right":
            self.q[[0, 2, 6]] *= -1

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the physical TCP transform and its spatial Jacobian."""
        transform, origins, axes = self.base.copy(), [], []
        for angle, (parent, child), axis in zip(q, self.frames, self.axes, strict=True):
            transform = transform @ parent
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, :3] @ axis)
            skew = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            rotation = np.eye(4)
            rotation[:3, :3] += np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
            transform = transform @ rotation @ child
        transform = transform @ self.tip
        jacobian = np.column_stack(
            [
                np.r_[np.cross(axis, transform[:3, 3] - origin), axis]
                for axis, origin in zip(axes, origins, strict=True)
            ]
        )
        return transform, jacobian

    def solve(
        self, position: np.ndarray, rotation: np.ndarray, iterations: int = 30
    ) -> tuple[np.ndarray, float]:
        """Find bounded joint targets and report remaining TCP error in metres."""
        for _ in range(iterations):
            pose, jacobian = self.forward(self.q)
            relative = rotation @ pose[:3, :3].T
            angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
            skew = np.array(
                [
                    relative[2, 1] - relative[1, 2],
                    relative[0, 2] - relative[2, 0],
                    relative[1, 0] - relative[0, 1],
                ]
            )
            if angle > np.pi - 1e-5:
                _, axes = np.linalg.eigh((relative + np.eye(3)) / 2)
                angular = axes[:, -1] * angle
            else:
                angular = skew * (angle / (2 * np.sin(angle)) if angle > 1e-8 else 0.5)
            error = np.r_[position - pose[:3, 3], angular]
            if np.linalg.norm(error) < 1e-5:
                break
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * 1e-4, error)
            self.q = np.clip(self.q + np.clip(delta, -0.12, 0.12), self.lower, self.upper)
        pose, _ = self.forward(self.q)
        return self.q.copy(), float(np.linalg.norm(position - pose[:3, 3]))

    def solve_pregrasp(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        *,
        waypoints: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Screen continuous grasp/transfer paths, including wrist joint limits."""
        initial = self.q.copy()
        rng = np.random.default_rng(42)
        for seed in [initial, *rng.uniform(self.lower, self.upper, (41, 7))]:
            self.q = seed.copy()
            q, distance = self.solve(position, rotation, 180)
            pose, _ = self.forward(q)
            angular = float(
                np.arccos(np.clip((np.trace(rotation @ pose[:3, :3].T) - 1) / 2, -1, 1))
            )
            if distance >= 0.001 or angular >= 0.005:
                continue
            valid = True
            path = waypoints or []
            for start, end in zip(path, path[1:], strict=False):
                for blend in np.linspace(0, 1, 13)[1:]:
                    current, error = self.solve(start * (1 - blend) + end * blend, rotation, 60)
                    actual, _ = self.forward(current)
                    angle = np.arccos(
                        np.clip((np.trace(rotation @ actual[:3, :3].T) - 1) / 2, -1, 1)
                    )
                    if error > 0.005 or angle > 0.02:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                self.q = q.copy()
                return q, distance
        self.q = initial
        raise RuntimeError("No continuous joint-limit-respecting Pro-M6 grasp/transfer path")
