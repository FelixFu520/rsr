"""分拣包裹任务的全部 Planner：Pro-M6 双臂入料 → 平台交接 → 出料，单件与连续双包裹。

本文件按依赖顺序包含：

1. ``SortingArmIK``：与 Newton 相同 USD 关节的世界系 Pro-M6 IK。
2. ``SortingRuntime``：原生 MuJoCo 机器人 + V7/V10 双网格 VBD 包裹、输送带接触和逐子步记录
   （继承物理求解器任务的 ``PackageRuntime``）。
3. ``sorting_package``：``SortingPlanner``，单个柔性包裹的双臂中继、抓点/深度、在线质量门禁。
4. ``SortingStreamRuntime``：同一 MuJoCo/VBD 世界中追加 V10 纸箱刚体、柔体休眠与唤醒。
5. ``sorting_stream``：``SortingStreamPlanner``，先软包后纸箱的连续编排与落地验收。

MuJoCo 在 CPU 上求解机器人与纸箱，VBD 在 CUDA 上求解袋膜、衣物和刚软接触（单向耦合）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from real2sim2real.core.logger import get_logger
from real2sim2real.planner.planner_physic_solver import (
    PackagePlannerConfig,
    PackageRuntime,
    PickPlaceClothPlanner,
    _rotation_wxyz,
)
from real2sim2real.utils.physic_solver_utils import check_package_containment
from real2sim2real.utils.sorting_utils import (
    audit_held_shape,
    audit_jaw_penetration,
    unit_box_mesh,
    without_bag_gap_checks,
)

# ================================================================================================
# World-frame Pro-M6 IK using the same authored USD joints as Newton.
# ================================================================================================


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


# ================================================================================================
# MuJoCo Pro-M6 articulation coupled to V7's two-mesh VBD contact model.
# ================================================================================================


@cache
def _belt_kernel() -> Any:
    import warp as wp

    @wp.kernel
    def belt_motion(
        previous: wp.array[wp.transform],
        current: wp.array[wp.transform],
        body: int,
        rest: wp.transform,
        velocity: wp.vec3,
        dt: float,
    ):
        p = wp.transform_get_translation(rest)
        q = wp.transform_get_rotation(rest)
        previous[body] = wp.transform(p - velocity * dt, q)
        current[body] = rest

    return belt_motion


class SortingRuntime(PackageRuntime):
    """Preserve native robot geometry and solve contacts without cloth attachments."""

    def __init__(self, robot: Any, scene: Any, config: Any) -> None:
        self.active_side = "left"
        self.belt_speed = 0.0
        self.infeed_speed = 0.0
        self.physics_origin = np.asarray(config.physics_origin_m, dtype=float)
        super().__init__(robot, scene, config)
        self.belt_rest = self.model.body_q.numpy()[self.belt_body].copy()
        if hasattr(self, "infeed_body"):
            self.infeed_rest = self.model.body_q.numpy()[self.infeed_body].copy()
        if config.continuous_outfeed:
            self.belt_speed = config.belt_speed_mps
        self._configure_contact_friction()
        self._belt_motion = _belt_kernel()
        self._masses = self.model.particle_mass.numpy()
        self.direct_use_graph = self.direct_use_graph and not config.mujoco_cpu
        self._soft_graphs = {}
        self._collision_views = {}
        # Return above the source/exit belts through the already screened grasp
        # workspace. A joint-space return to READY can sweep an open pad into
        # the platform even when both end poses are collision-free.
        self.parking_positions = {
            "left": np.r_[scene.config.target_xy, scene.supports["infeed"][1][2] + 0.25],
            "right": np.r_[
                scene.config.outfeed_target_xy,
                scene.supports["outfeed"][1][2] + config.place_height_m + 0.18,
            ],
        }
        self.jaw_colliders = []
        self.pad_colliders = []
        self.fixed_robot_colliders = []
        finger_bodies = {
            self.contact_body_ids[f"{side}_{i}"] for side in ("left", "right") for i in (0, 1)
        }
        for shape, body in enumerate(self.model.shape_body.numpy()):
            source = self.model.shape_source[shape]
            fixed_robot = shape < self.robot_shape_count and int(body) < 0
            if int(body) not in finger_bodies and not fixed_robot:
                continue
            is_pad = int(self.model.shape_type.numpy()[shape]) == int(self.newton.GeoType.BOX)
            if is_pad:
                points, triangles = unit_box_mesh()
            elif source is not None:
                points = np.asarray(source.vertices)
                triangles = np.asarray(source.indices, dtype=int).reshape(-1, 3)
            else:
                raise ValueError("Unsupported native finger collider; cannot certify geometry")
            transform = self.model.shape_transform.numpy()[shape]
            rotation = _rotation_wxyz(transform[[6, 3, 4, 5]])
            vertices = (points * self.model.shape_scale.numpy()[shape]) @ rotation.T
            if fixed_robot:
                self.fixed_robot_colliders.append(
                    dict(
                        shape_id=shape,
                        vertices=vertices + transform[:3] + self.physics_origin,
                        triangles=triangles,
                    )
                )
                continue
            self.jaw_colliders.append(
                dict(
                    body_id=int(body),
                    vertices=vertices + transform[:3],
                    triangles=triangles,
                )
            )
            if is_pad:
                self.pad_colliders.append(self.jaw_colliders[-1])
        self._pad_bottom_offsets = {}
        if config.grasp_pad_bottom_m is not None:
            self._calibrate_pad_bottom()

    def _configure_contact_friction(self) -> None:
        """Apply finger and support materials using Newton's geometric mixing."""
        shape_mu = self.model.shape_material_mu.numpy()
        bodies = self.model.shape_body.numpy()
        fingers = np.isin(
            bodies,
            [self.contact_body_ids[f"{side}_{i}"] for side in ("left", "right") for i in (0, 1)],
        )
        target = self.config.gripper_package_friction
        particle_mu = float(self.model.soft_contact_mu)
        if target is not None:
            if target > 0 and particle_mu <= 0:
                raise ValueError(
                    "Positive pair friction requires nonzero particle material friction"
                )
            shape_mu[fingers] = target**2 / particle_mu if target > 0 else 0.0
        shape_mu[self.belt_shape] = self.config.table_friction
        if hasattr(self, "infeed_shape"):
            shape_mu[self.infeed_shape] = self.config.table_friction
        if self.config.platform_friction is not None:
            shape_mu[self.platform_shape] = self.config.platform_friction
        self.model.shape_material_mu.assign(shape_mu)
        if hasattr(self, "floor_shape") and self.config.floor_contact_stiffness is not None:
            stiffness = self.model.shape_material_ke.numpy()
            stiffness[self.floor_shape] = self.config.floor_contact_stiffness
            self.model.shape_material_ke.assign(stiffness)
        if hasattr(self, "floor_shape") and self.config.floor_contact_margin_m is not None:
            margins = self.model.shape_margin.numpy()
            margins[self.floor_shape] = self.config.floor_contact_margin_m
            self.model.shape_margin.assign(margins)
        supports = (bodies < 0) | (bodies == self.belt_body)
        self.contact_friction = dict(
            particle_material_mu=particle_mu,
            finger_material_mu=np.unique(shape_mu[fingers]).tolist(),
            effective_gripper_package_mu=np.unique(
                np.sqrt(particle_mu * shape_mu[fingers])
            ).tolist(),
            effective_support_package_mu=np.unique(
                np.sqrt(particle_mu * shape_mu[supports])
            ).tolist(),
            mixing="geometric_mean",
            platform_shape=self.platform_shape,
            platform_material_mu=float(shape_mu[self.platform_shape]),
            effective_platform_package_mu=float(
                np.sqrt(particle_mu * shape_mu[self.platform_shape])
            ),
        )

    def _calibrate_pad_bottom(self) -> None:
        """Evaluate native linkage FK in scratch buffers without advancing dynamics."""
        angles = np.linspace(0.0, 0.7, 141)
        offsets = {side: [] for side in self.hand_bodies}
        q = self.state_0.joint_q.numpy().copy()
        scratch = self.model.state()
        coordinates = self.wp.array(q, dtype=self.wp.float32, device=self.model.device)
        for angle in angles:
            for pairs in self.robot.mimic.values():
                for index, ratio in pairs:
                    q[index] = ratio * angle
            coordinates.assign(q)
            self.newton.eval_fk(self.model, coordinates, self.state_0.joint_qd, scratch)
            poses = scratch.body_q.numpy()
            for side, hand in self.hand_bodies.items():
                pose = poses[hand]
                rotation = _rotation_wxyz(pose[[6, 3, 4, 5]])
                tcp = pose[:3] + rotation @ self.iks[side].tip[:3, 3]
                tcp_rotation = rotation @ self.iks[side].tip[:3, :3]
                bodies = {self.contact_body_ids[f"{side}_{i}"] for i in (0, 1)}
                heights = []
                for pad in self.pad_colliders:
                    if pad["body_id"] not in bodies:
                        continue
                    pose = poses[pad["body_id"]]
                    world = pad["vertices"] @ _rotation_wxyz(pose[[6, 3, 4, 5]]).T
                    local = (world + pose[:3] - tcp) @ tcp_rotation
                    heights.append((local @ self.grasp_rotation.T)[:, 2].min())
                if not heights:
                    raise ValueError(f"No native pad geometry found for {side}")
                offsets[side].append(min(heights))
        self._pad_bottom_offsets = {
            side: (angles, np.asarray(values)) for side, values in offsets.items()
        }

    def pad_bottom_tcp_height(self, side: str, angle: float, bottom: float) -> float:
        """Return target TCP height for a specified native pad bottom plane."""
        angles, offsets = self._pad_bottom_offsets[side]
        if not angles[0] <= angle <= angles[-1]:
            raise ValueError("Pad calibration angle is outside the native linkage range")
        return float(bottom - np.interp(angle, angles, offsets))

    def pad_bottom_z(self, side: str, poses: np.ndarray) -> float:
        """Measure the lowest pad vertex from actual world-frame body poses."""
        bodies = {self.contact_body_ids[f"{side}_{i}"] for i in (0, 1)}
        return min(
            float(
                (
                    pad["vertices"] @ _rotation_wxyz(poses[pad["body_id"]][[6, 3, 4, 5]]).T
                    + poses[pad["body_id"]][:3]
                )[:, 2].min()
            )
            for pad in self.pad_colliders
            if pad["body_id"] in bodies
        )

    def _create_mujoco_solver(self) -> Any:
        from newton.solvers import SolverMuJoCo

        solver = SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            iterations=self.config.mujoco_iterations,
            use_mujoco_cpu=self.config.mujoco_cpu,
            update_data_interval=0 if self.config.mujoco_cpu else 1,
        )
        # Native MuJoCo joint equalities replace the USD's unapplied PhysX mimic
        # attributes. Keep their compliance explicit in the saved configuration.
        time_constant = max(
            self.config.mimic_time_constant_s,
            2
            * self.robot.sim_world.config.physics_dt
            / (self.config.newton_substeps * self.config.mujoco_substeps_per_vbd_step),
        )
        solver.mj_model.eq_solref[:, 0] = time_constant
        solver.mj_model.eq_solimp[:, 0] = self.config.mimic_impedance
        solver.mj_model.eq_solimp[:, 1] = 0.9999
        if not self.config.mujoco_cpu:
            solver.mjw_model.eq_solref.assign(solver.mj_model.eq_solref[None])
            solver.mjw_model.eq_solimp.assign(solver.mj_model.eq_solimp[None])
        else:
            # VBD does not feed joint state back into this one-way rigid model.
            # Keep MuJoCo's continuous double-precision state between substeps.
            solver.reset(self.state_0)
            mapping = {
                int(q): int(mj)
                for q, mj, dim in zip(
                    self.model.joint_q_start.numpy()[:-1],
                    solver.mj_q_start.numpy(),
                    self.model.joint_dof_dim.numpy(),
                    strict=True,
                )
                if sum(dim) > 0
            }
            pairs = [
                (mapping[index], mapping[self.robot.gripper_indices[side]], ratio)
                for side, joints in self.robot.mimic.items()
                for index, ratio in joints
            ]
            self._native_mimic_pairs = np.asarray(pairs)
            self.native_mimic_residuals: list[np.ndarray] = []
            self.maximum_native_mimic_error = 0.0
        return solver

    def _configure_builder(self, builder: Any) -> None:
        # Import the authored convex collision approximations first. Requesting SDFs
        # before USD approximation would cause Newton to reject hull conversion.
        pass

    def _finalize_model(self, builder: Any) -> Any:
        if np.any(self.physics_origin):
            # Newton's native merge transforms particles, fixed world shapes,
            # articulated roots and free joints together. A constant frame
            # translation adds no force and changes no relative geometry.
            local = self.newton.ModelBuilder()
            local.add_builder(
                builder,
                xform=self.wp.transform(-self.physics_origin, self.wp.quat_identity()),
            )
            builder = local
        return super()._finalize_model(builder)

    def vertices(self) -> np.ndarray:
        """Return world coordinates while the optional solver frame stays local."""
        return super().vertices() + self.physics_origin

    def body_poses(self) -> np.ndarray:
        """Return solved rigid poses in the same world frame as the recorded particles."""
        poses = self.state_0.body_q.numpy().astype(float)
        poses[:, :3] += self.physics_origin
        return poses

    def _add_deformables(self, builder: Any, *, xpbd: bool) -> None:
        wp = self.wp
        for shape, source in enumerate(builder.shape_source):
            if source is not None and hasattr(source, "vertices"):
                builder.shape_force_sdf[shape] = True
                builder.shape_sdf_max_resolution[shape] = self.config.contact_sdf_resolution
        for name in ("platform", "outfeed"):
            low, high = self.scene.supports[name]
            center, half = (low + high) / 2, (high - low) / 2
            body = -1
            transform = wp.transform(center, wp.quat_identity())
            if name == "outfeed":
                body = (
                    self._add_fixed_belt(builder, center, "outfeed_belt")
                    if self.config.continuous_outfeed
                    else builder.add_body(xform=transform, is_kinematic=True, label="outfeed_belt")
                )
                self.belt_body = body
                transform = wp.transform_identity()
            shape = builder.add_shape_box(body, xform=transform, hx=half[0], hy=half[1], hz=half[2])
            if name == "platform":
                self.platform_shape = shape
            if name == "outfeed":
                self.belt_shape = shape
        super()._add_deformables(builder, xpbd=xpbd)

    def _add_table(self, builder: Any) -> None:
        if self.scene.config.infeed_travel_m <= 0:
            return super()._add_table(builder)
        low, high = self.scene.supports["infeed"]
        center, half = (low + high) / 2, (high - low) / 2
        self.infeed_body = self._add_fixed_belt(builder, center, "infeed_belt")
        self.infeed_shape = builder.add_shape_box(
            self.infeed_body, hx=half[0], hy=half[1], hz=half[2]
        )
        self.floor_shape = builder.add_ground_plane(height=self.scene.config.floor_z_m)

    def _add_fixed_belt(self, builder: Any, center: np.ndarray, label: str) -> int:
        """Use fixed-root mocap geometry so native CPU batching cannot integrate a belt."""
        transform = self.wp.transform(center, self.wp.quat_identity())
        body = builder.add_link(xform=transform, is_kinematic=True, label=label)
        joint = builder.add_joint_fixed(-1, body, parent_xform=transform, label=f"{label}_root")
        builder.add_articulation([joint], label=f"{label}_articulation")
        return body

    def _create_ik(self) -> None:
        self.iks = {side: SortingArmIK(self.robot, side) for side in ("left", "right")}
        self.hand_bodies = {
            side: self.model.body_label.index(ik.body_path) for side, ik in self.iks.items()
        }
        self.hand_body = self.hand_bodies["left"]

    def _contact_bodies(self) -> dict[str, int]:
        result = {}
        for side in ("left", "right"):
            for index, name in enumerate(("left_inner_finger", "right_inner_finger")):
                result[f"{side}_{index}"] = next(
                    i
                    for i, label in enumerate(self.model.body_label)
                    if label.endswith(f"/robotiq_{side}/{name}")
                )
        result.update(left=result["left_0"], right=result["left_1"], palm=self.hand_body)
        return result

    def _create_robot_visuals(self) -> None:
        from pxr import UsdGeom

        robot = self.robot
        self._body_ops = []
        for label in self.model.body_label[: self.robot_body_count]:
            path = label.replace(robot.source_root, robot.config.prim_path, 1)
            xf = UsdGeom.Xformable(robot.sim_world.stage.GetPrimAtPath(path))
            if not xf:
                raise ValueError(f"Missing robot visual body: {path}")
            xf.ClearXformOpOrder()
            xf.SetResetXformStack(True)
            self._body_ops.append(xf.AddTransformOp())
        self._gripper_ops = {}
        for side, ik in self.iks.items():
            path = f"{robot.source_root}/robotiq_{side}"
            source = robot.source_stage.GetPrimAtPath(path)
            from pxr import UsdGeom

            world = np.asarray(UsdGeom.Xformable(source).ComputeLocalToWorldTransform(0)).T
            end = robot.source_stage.GetPrimAtPath(ik.body_path)
            end_world = np.asarray(UsdGeom.Xformable(end).ComputeLocalToWorldTransform(0)).T
            xf = UsdGeom.Xformable(
                robot.sim_world.stage.GetPrimAtPath(
                    path.replace(robot.source_root, robot.config.prim_path, 1)
                )
            )
            xf.ClearXformOpOrder()
            xf.SetResetXformStack(True)
            self._gripper_ops[side] = (xf.AddTransformOp(), np.linalg.inv(end_world) @ world)
        self._pad_visual_checks = []
        unit, _ = unit_box_mesh()
        shape_poses = self.model.shape_transform.numpy()
        shape_scales = self.model.shape_scale.numpy()
        for pad in robot.native_pad_geometry:
            path = pad["visual_path"].replace(robot.source_root, robot.config.prim_path, 1)
            cube = UsdGeom.Cube(robot.sim_world.stage.GetPrimAtPath(path))
            pose = shape_poses[pad["shape_id"]]
            local = (unit * shape_scales[pad["shape_id"]]) @ _rotation_wxyz(
                pose[[6, 3, 4, 5]]
            ).T + pose[:3]
            self._pad_visual_checks.append((cube, pad["body_id"], local))
        self.visual_alignment_maximum_m = 0.0
        self.visual_alignment_samples = 0
        self._publish_robot_visuals()
        robot.initialize_cameras()

    def _publish_robot_visuals(self, transforms: np.ndarray | None = None) -> None:
        from pxr import Gf

        if transforms is None:
            transforms = self.body_poses()
        matrices = []
        for pose in transforms[: self.robot_body_count]:
            matrix = np.eye(4)
            matrix[:3, :3] = _rotation_wxyz(pose[[6, 3, 4, 5]])
            matrix[:3, 3] = pose[:3]
            matrices.append(matrix)
        for op, matrix in zip(self._body_ops, matrices, strict=True):
            op.Set(Gf.Matrix4d(matrix.T))
        for side, (op, adapter) in self._gripper_ops.items():
            op.Set(Gf.Matrix4d((matrices[self.hand_bodies[side]] @ adapter).T))
        self._check_pad_visual_alignment(matrices)

    def _check_pad_visual_alignment(self, matrices: list[np.ndarray]) -> None:
        """Reject a published frame if its visible pads differ from the solved colliders."""
        from pxr import UsdGeom

        cache = UsdGeom.XformCache()
        unit, _ = unit_box_mesh()
        for cube, body, local in self._pad_visual_checks:
            transform = np.asarray(cache.GetLocalToWorldTransform(cube.GetPrim())).T
            visible = (unit * float(cube.GetSizeAttr().Get()) / 2) @ transform[:3, :3].T
            visible += transform[:3, 3]
            solved = local @ matrices[body][:3, :3].T + matrices[body][:3, 3]
            error = float(np.linalg.norm(visible - solved, axis=1).max())
            if error > 2e-6:
                raise ValueError(f"Visible gripper pad differs from physics by {error:.6g} m")
            self.visual_alignment_maximum_m = max(self.visual_alignment_maximum_m, error)
        self.visual_alignment_samples += 1

    def publish_recorded_visuals(self, vertices: np.ndarray, body_poses: np.ndarray) -> None:
        """Publish cached observations without modifying either solver's state buffers."""
        self.scene.publish_cloth_vertices(vertices)
        self._publish_robot_visuals(body_poses)

    def _publish(self) -> None:
        super()._publish()
        shapes = getattr(self, "last_active_contact_shapes", np.empty(0, int))
        bodies = getattr(self, "last_active_contact_bodies", np.empty(0, int))
        self.last_contact_counts["outfeed"] = int(np.count_nonzero(bodies == self.belt_body))
        self.last_contact_counts["platform"] = int(np.count_nonzero(shapes == self.platform_shape))
        if hasattr(self, "infeed_body"):
            self.last_contact_counts["infeed"] = int(np.count_nonzero(bodies == self.infeed_body))

    def tcp_position(self, side: str | None = None) -> np.ndarray:
        """Read the actual solved TCP, including MuJoCo tracking error."""
        side = side or self.active_side
        pose = self.state_0.body_q.numpy()[self.hand_bodies[side]]
        return (
            pose[:3]
            + self.physics_origin
            + _rotation_wxyz(pose[[6, 3, 4, 5]]) @ self.iks[side].tip[:3, 3]
        )

    def command_arm(self, side: str, target: np.ndarray, gripper_angle_rad: float) -> float:
        """Drive native joint targets; soft-body positions are never commanded."""
        self._activate_side(side)
        q, error = self.iks[side].solve(np.asarray(target), self.grasp_rotation)
        self.target_q[self.robot.arm_indices[side]] = q
        self.set_gripper_angle(side, gripper_angle_rad)
        self.control.joint_target_q.assign(self.target_q)
        return error

    def _activate_side(self, side: str) -> None:
        self.active_side = side
        self.contact_body_ids.update(
            left=self.contact_body_ids[f"{side}_0"],
            right=self.contact_body_ids[f"{side}_1"],
            palm=self.hand_bodies[side],
        )

    def prepare_approach(
        self, side: str, target: np.ndarray, waypoints: list[np.ndarray] | None = None
    ) -> float:
        """Find a reachable top-down posture before moving from the ready posture."""
        self.approach_start = self.target_q[self.robot.arm_indices[side]].copy()
        self.approach_goal, error = self.iks[side].solve_pregrasp(
            target, self.grasp_rotation, waypoints=waypoints
        )
        return error

    @property
    def grasp_rotation(self) -> np.ndarray:
        """Honor the configured physical TCP orientation in both IK stages."""
        return _rotation_wxyz(np.asarray(self.config.tcp_orientation_wxyz, dtype=float))

    def command_approach(self, side: str, blend: float) -> None:
        """Interpolate only arm joint targets; allow the articulation to solve actual motion."""
        self._activate_side(side)
        self.target_q[self.robot.arm_indices[side]] = (
            self.approach_start * (1 - blend) + self.approach_goal * blend
        )
        self.set_gripper(side, False)

    def set_gripper(self, side: str, closed: bool | float) -> None:
        """Set the master drive and matching passive mimic targets for metadata."""
        closure = float(np.clip(closed, 0.0, 1.0))
        angle = self.config.open_gripper_rad + closure * (
            self.config.closed_gripper_rad - self.config.open_gripper_rad
        )
        self.set_gripper_angle(side, angle)

    def set_gripper_angle(self, side: str, angle: float) -> None:
        """Command the actual master angle within the source USD's 0–0.7 rad limits."""
        if not 0 <= angle <= 0.7:
            raise ValueError("Pro-M6 gripper angle must remain within the authored joint limits")
        for index, ratio in self.robot.mimic[side]:
            self.target_q[index] = ratio * angle
        self.control.joint_target_q.assign(self.target_q)

    def recording_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return fourteen measured arm angles and two measured master gripper angles."""
        indices = [
            *self.robot.arm_indices["left"],
            *self.robot.arm_indices["right"],
            self.robot.gripper_indices["left"],
            self.robot.gripper_indices["right"],
        ]
        return self.state_0.joint_q.numpy()[indices], self.target_q[indices]

    def _advance_direct(self, frame_dt: float) -> None:
        dt = frame_dt / self.config.newton_substeps
        if not getattr(self, "soft_asleep", False):
            self.cloth_solver.rebuild_bvh(self.state_0)
        batch_control = self.config.mujoco_cpu and self.config.mujoco_control_per_frame
        if batch_control:
            # Every joint is a scalar hinge, targets are constant over this
            # physics frame and this coupling has no soft-to-rigid force path.
            # Newton's CPU bridge otherwise downloads identical control/zero
            # force buffers at every substep, synchronizing the CUDA soft solve.
            self.state_0.clear_forces()
            solver = self.rigid_solver
            solver._apply_mjc_control(self.model, self.state_0, self.control, solver.mj_data)
            rigid_substeps = self.config.mujoco_substeps_per_vbd_step
            solver.mj_model.opt.timestep = dt / rigid_substeps
            # Advance the independent rigid system first and upload its complete
            # solved substep trajectory together. Repeated small CPU→CUDA copies
            # otherwise wait for the preceding soft solve on the same stream.
            positions, velocities, forces, mimic_residuals = [], [], [], []
            pairs = self._native_mimic_pairs
            for _ in range(self.config.newton_substeps):
                for _ in range(rigid_substeps):
                    self._before_native_substep(dt / rigid_substeps)
                    solver._mujoco.mj_step(solver.mj_model, solver.mj_data)
                    solver._step += 1
                    q = solver.mj_data.qpos
                    mimic_residuals.append(
                        q[pairs[:, 0].astype(int)] - q[pairs[:, 1].astype(int)] * pairs[:, 2]
                    )
                positions.append(solver.mj_data.qpos.copy())
                velocities.append(solver.mj_data.qvel.copy())
                forces.append(solver.mj_data.qfrc_actuator.copy())
            residuals = np.asarray(mimic_residuals)
            self.native_mimic_residuals.append(residuals)
            self.maximum_native_mimic_error = max(
                self.maximum_native_mimic_error, float(np.max(np.abs(residuals)))
            )
            arrays = [
                np.asarray(values, dtype=np.float32) for values in (positions, velocities, forces)
            ]
            if not hasattr(self, "_rigid_frame_buffers"):
                self._rigid_frame_buffers = [
                    self.wp.empty(values.shape, dtype=float, device=self.model.device)
                    for values in arrays
                ]
                q, qd, force = self._rigid_frame_buffers
                self._rigid_substeps = [
                    SimpleNamespace(
                        nworld=1,
                        qpos=q[i : i + 1],
                        qvel=qd[i : i + 1],
                        qfrc_actuator=force[i : i + 1],
                    )
                    for i in range(self.config.newton_substeps)
                ]
            for buffer, values in zip(self._rigid_frame_buffers, arrays, strict=True):
                buffer.assign(values)
        for substep in range(self.config.newton_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            count = self.model.particle_count
            self.model.particle_count = 0
            try:
                if batch_control:
                    # Keep all native MuJoCo integration, equalities and rigid
                    # contacts at the original substep rate (Newton 1.6.1 API).
                    solver._update_newton_state(
                        self.model,
                        self.state_1,
                        self._rigid_substeps[substep],
                        state_prev=self.state_0,
                    )
                else:
                    self.rigid_solver.step(self.state_0, self.state_1, self.control, None, dt)
            finally:
                self.model.particle_count = count
            # A treadmill changes only the support's tangential contact displacement.
            # Actual parcel displacement is produced entirely by VBD friction.
            key = (
                id(self.state_0),
                self.belt_speed,
                getattr(self, "infeed_speed", 0.0),
                getattr(self, "soft_asleep", False),
            )
            if self.config.mujoco_cpu and self.config.newton_use_cuda_graph and self.direct_warmed:
                if key not in self._soft_graphs:
                    with self.wp.ScopedCapture(device=self.model.device) as capture:
                        self._advance_soft(dt)
                    self._soft_graphs[key] = capture.graph
                self.wp.capture_launch(self._soft_graphs[key])
            else:
                self._advance_soft(dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _before_native_substep(self, dt: float) -> None:
        """Allow contact-driven conveyor forces for an optional dynamic rigid parcel."""

    def _advance_soft(self, dt: float) -> None:
        self.wp.launch(
            self._belt_motion,
            1,
            inputs=[
                self.state_0.body_q,
                self.state_1.body_q,
                self.belt_body,
                self.wp.transform(self.belt_rest[:3], self.belt_rest[3:]),
                self.wp.vec3(-self.belt_speed, 0.0, 0.0),
                dt,
            ],
            device=self.model.device,
        )
        if hasattr(self, "infeed_body"):
            self.wp.launch(
                self._belt_motion,
                1,
                inputs=[
                    self.state_0.body_q,
                    self.state_1.body_q,
                    self.infeed_body,
                    self.wp.transform(self.infeed_rest[:3], self.infeed_rest[3:]),
                    self.wp.vec3(0.0, self.infeed_speed, 0.0),
                    dt,
                ],
                device=self.model.device,
            )
        key = id(self.state_0)
        if key not in self._collision_views:
            # External VBD evaluates particle contacts against state_out's
            # already integrated rigid poses, while particles start in state_in.
            # Preserve the old rigid poses separately for frictional motion.
            view = self.newton.State()
            view.particle_q = self.state_0.particle_q
            view.particle_qd = self.state_0.particle_qd
            view.body_q = self.state_1.body_q
            view.body_qd = self.state_1.body_qd
            self._collision_views[key] = view
        self.pipeline.collide(self._collision_views[key], self.contacts)
        self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)


# ================================================================================================
# Two physically independent grasps: infeed → platform → outfeed, then belt transport.
# ================================================================================================


def _motion_blend(progress: float, profile: Literal["cubic", "quintic"]) -> float:
    """Blend a bounded trajectory; quintic also has zero endpoint acceleration."""
    t = float(np.clip(progress, 0, 1))
    if profile == "quintic":
        return t**3 * (10 - 15 * t + 6 * t**2)
    return t * t * (3 - 2 * t)


def _robot_contact_count(runtime: Any) -> int:
    """Count authored robot shapes, including fixed-root colliders with body id -1."""
    shapes = getattr(runtime, "last_active_contact_shapes", None)
    if shapes is not None:
        return int(np.count_nonzero((shapes >= 0) & (shapes < runtime.robot_shape_count)))
    bodies = getattr(runtime, "last_active_contact_bodies", np.empty(0, dtype=int))
    return int(np.count_nonzero((bodies >= 0) & (bodies < runtime.robot_body_count)))


def _finite_metrics(value: Any) -> Any:
    """Preserve nested measured coordinates and encode unavailable metrics as null."""
    if isinstance(value, np.ndarray):
        return _finite_metrics(value.tolist())
    if isinstance(value, dict):
        return {key: _finite_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_metrics(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _regrasp_pad_bottom(
    garment: np.ndarray,
    xy: np.ndarray,
    support_z: float,
    minimum_m: float,
    indent_m: float,
    indent_fraction: float | None = None,
) -> float:
    """Choose the pad bottom from the current central garment surface, in support metres."""
    selected = np.all(np.abs(garment[:, :2] - xy) <= 0.03, axis=1)
    if not selected.any():
        raise ValueError("No live garment surface in the regrasp region")
    surface_z = float(np.quantile(garment[selected, 2], 0.9))
    surface_height = surface_z - support_z
    if indent_fraction is not None:
        indent_m = max(indent_m, surface_height * indent_fraction)
    return max(minimum_m, surface_height - indent_m)


@dataclass
class StrainGripController:
    """Relieve a position-controlled grip using measured deformation, with hysteresis."""

    target: float
    maximum_relief_rad: float
    rate_rad_s: float
    recovery_rate_rad_s: float | None = None
    hysteresis: float = 0.03
    relief_rad: float = 0.0
    peak_relief_rad: float = 0.0

    def command(
        self, nominal_rad: float, stretch: float, dt: float, *, minimum_rad: float = 0.0
    ) -> float:
        """Relieve high strain and restore grip below the configured dead band."""
        if stretch > self.target:
            self.relief_rad += self.rate_rad_s * dt
        elif stretch < self.target - self.hysteresis:
            recovery = self.recovery_rate_rad_s
            self.relief_rad -= (0.1 * self.rate_rad_s if recovery is None else recovery) * dt
        available = min(self.maximum_relief_rad, max(0.0, nominal_rad - minimum_rad))
        self.relief_rad = float(np.clip(self.relief_rad, 0, available))
        self.peak_relief_rad = max(self.peak_relief_rad, self.relief_rad)
        return max(0.0, nominal_rad - self.relief_rad)


class SortingPlannerConfig(PackagePlannerConfig):
    """Reuse V7 material/contact settings, with separate relay and conveyor acceptance."""

    backend: str = "sorting_package"
    gripper_package_friction: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    platform_friction: FiniteFloat | None = Field(default=None, ge=0)
    tcp_orientation_wxyz: tuple[float, float, float, float] = (0, 2**-0.5, 2**-0.5, 0)
    mujoco_cpu: bool = True
    mujoco_control_per_frame: bool = True
    mujoco_substeps_per_vbd_step: int = Field(default=1, ge=1, le=16)
    mujoco_iterations: int = Field(default=100, gt=0)
    mimic_time_constant_s: float = Field(default=0.0015, gt=0)
    mimic_impedance: float = Field(default=0.999, gt=0, lt=0.9999)
    open_gripper_rad: float = Field(default=0.0, ge=0, le=0.7)
    closed_gripper_rad: float = Field(default=0.64, gt=0, le=0.7)
    hold_gripper_rad: float = Field(default=0.675, gt=0, le=0.7)
    grasp_height_m: float = Field(default=0.048, ge=0, le=0.10)
    grasp_pad_bottom_m: float | None = Field(default=None, ge=0, le=0.08)
    infeed_grasp_pad_bottom_m: FiniteFloat | None = Field(default=None, ge=0, le=0.08)
    regrasp_surface_indent_m: float | None = Field(default=None, gt=0, le=0.08, allow_inf_nan=False)
    regrasp_surface_indent_fraction: FiniteFloat | None = Field(default=None, gt=0, lt=1)
    grasp_offset_xy_m: tuple[FiniteFloat, FiniteFloat] = (0.0, 0.0)
    regrasp_offset_xy_m: tuple[FiniteFloat, FiniteFloat] | None = None
    place_height_m: float = Field(default=0.20, gt=0)
    release_lowering_m: float = Field(default=0.0, ge=0, le=0.1, allow_inf_nan=False)
    lift_height_m: float = Field(default=0.25, gt=0)
    belt_speed_mps: float = Field(default=0.35, gt=0)
    belt_transport_s: float = Field(default=6, gt=0)
    infeed_speed_mps: FiniteFloat = Field(default=0.06, gt=0)
    infeed_settle_s: FiniteFloat = Field(default=2.0, gt=0)
    continuous_outfeed: bool = False
    outfeed_place_height_m: FiniteFloat | None = Field(default=None, gt=0)
    floor_contact_stiffness: FiniteFloat | None = Field(default=None, gt=0)
    floor_contact_margin_m: FiniteFloat | None = Field(default=None, ge=0)
    minimum_belt_displacement_m: float = Field(default=0.30, gt=0)
    maximum_edge_stretch: float = Field(default=1.3, gt=1)
    minimum_triangle_area_ratio: float = Field(default=0.03, gt=0, lt=1)
    maximum_shape_rms_m: float = Field(default=0.037413, gt=0)
    maximum_bag_gap_m: float = Field(default=0.01, gt=0)
    require_bag_gap_quality: bool = True
    early_stop_on_shape_failure: bool = True
    maximum_tcp_error_m: float = Field(default=0.03, gt=0)
    return_s: float = Field(default=2.0, gt=0)
    transfer_profile: Literal["cubic", "quintic"] = "cubic"
    strain_limited_grip: bool = False
    grip_strain_target: float = Field(default=1.18, gt=1, lt=1.3)
    grip_maximum_relief_rad: float = Field(default=0.01, gt=0, le=0.05)
    grip_relief_rate_rad_s: float = Field(default=0.05, gt=0, le=0.5)
    grip_recovery_rate_rad_s: float = Field(default=0.005, gt=0, le=0.5)
    grip_strain_hysteresis: float = Field(default=0.03, gt=0, lt=0.3)
    hold_s: float = Field(default=0.0, ge=0)
    physics_origin_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.0, 0.0, 0.0)
    label_up_min_cos: float = Field(default=0.9, ge=0, le=1)

    @model_validator(mode="after")
    def _pair_friction(self) -> SortingPlannerConfig:
        if self.mujoco_substeps_per_vbd_step > 1 and not (
            self.mujoco_cpu and self.mujoco_control_per_frame
        ):
            raise ValueError("MuJoCo subcycling requires batched native CPU control")
        if self.regrasp_surface_indent_m is not None and self.grasp_pad_bottom_m is None:
            raise ValueError("Live regrasp height requires native pad-bottom calibration")
        if self.infeed_grasp_pad_bottom_m is not None and self.grasp_pad_bottom_m is None:
            raise ValueError("Infeed grasp height requires native pad-bottom calibration")
        if (
            self.regrasp_surface_indent_fraction is not None
            and self.regrasp_surface_indent_m is None
        ):
            raise ValueError("Relative regrasp depth requires a minimum surface indentation")
        if self.gripper_package_friction and self.cloth_friction <= 0:
            raise ValueError("Positive gripper_package_friction requires nonzero cloth_friction")
        if self.release_lowering_m >= self.place_height_m:
            raise ValueError("Release TCP must remain above the destination support")
        return self


class SortingPlanner(PickPlaceClothPlanner):
    """Carry the unchanged two-mesh parcel through both arms without attachments."""

    config_cls = SortingPlannerConfig
    runtime_cls = SortingRuntime

    @property
    def expected_duration_s(self) -> float:
        """Return the finite full relay and conveyor duration."""
        return (
            2
            * sum(
                getattr(self.config, f"{phase}_s")
                for phase in (
                    "approach",
                    "descend",
                    "close",
                    "lift",
                    "hold",
                    "transfer",
                    "lower",
                    "release",
                    "retract",
                    "return",
                    "rest",
                )
            )
            + self.config.belt_transport_s
            + self._infeed_duration_s
        )

    @property
    def _infeed_duration_s(self) -> float:
        travel = getattr(getattr(self.scene, "config", None), "infeed_travel_m", 0.0)
        return (
            travel / self.config.infeed_speed_mps + self.config.infeed_settle_s if travel else 0.0
        )

    @property
    def action_names(self) -> list[str]:
        """Commands are native joint position targets in radians."""
        return [name.replace("_rad", "_target_rad") for name in self.robot.recording_joint_names]

    def begin_episode(self, *, register_stepper: bool = True) -> None:
        """Build one continuous physical model shared by both grasps and the conveyors."""
        if (
            self.robot.sim_world.config.newton_solver != "mujoco"
            or self.config.cloth_solver != "vbd"
        ):
            raise ValueError("Sorting requires MuJoCo robot + VBD parcel/contact")
        super().begin_episode(register_stepper=register_stepper)
        self._initialize_tracking()

    def _initialize_tracking(self) -> None:
        """Initialize measurements on the already constructed continuous runtime."""
        phases = (
            "approach",
            "descend",
            "close",
            "lift",
            "hold",
            "transfer",
            "lower",
            "release",
            "retract",
            "return",
            "rest",
        )
        self.schedule = [
            (side, phase, getattr(self.config, f"{phase}_s"))
            for side in ("left", "right")
            for phase in phases
            if getattr(self.config, f"{phase}_s") > 0
        ]
        self.schedule.append(("right", "conveyor", self.config.belt_transport_s))
        if self._infeed_duration_s > 0:
            self.schedule.insert(
                0,
                (
                    "left",
                    "infeed",
                    self._infeed_duration_s,
                ),
            )
        self.ends = np.cumsum([row[2] for row in self.schedule])
        self._index = -1
        self._frame = 0
        self.command_tcp: np.ndarray | None = None
        self.command_side: str | None = None
        self.samples, self.sample_times, self.joints, self.actions, self.hand_poses = (
            [],
            [],
            [],
            [],
            [],
        )
        self.checkpoint_times = [0.0]
        self.transfer_samples = []
        self.transfer_groups = {"left": [], "right": []}
        self.legs = {
            side: dict(
                peak_lifts=np.zeros(2),
                contact_frames=0,
                transfer_frames=0,
                hold_frames=0,
                hold_contact_frames=0,
                minimum_hold_lift=np.full(2, np.inf),
                minimum_transfer_lift=np.full(2, np.inf),
                released=False,
                release_drop=0.0,
                place_error=np.inf,
                minimum_height=np.inf,
                footprint_supported=False,
                label_up_cos=None,
            )
            for side in ("left", "right")
        }
        self.rest = self.runtime.vertices().copy()
        tri = self.rest[self.scene.triangle_indices.reshape(-1, 3)]
        self.rest_areas = np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
        self.max_stretch = 1.0
        self.min_area = 1.0
        self.max_penetration = self.max_tcp_error = self.max_mimic_error = 0.0
        self.max_contact_penetration = 0.0
        self.max_finger_geometry_penetration = 0.0
        self.online_held_shape_samples: list[dict[str, Any]] = []
        self._belt_start = None
        self.belt_displacement = 0.0
        self.failure_reason: str | None = None
        self.belt_exited = False
        self.articulation_samples: list[np.ndarray] = []
        self.body_samples: list[np.ndarray] = []
        self.grip_limiters = {
            side: StrainGripController(
                self.config.grip_strain_target,
                self.config.grip_maximum_relief_rad,
                self.config.grip_relief_rate_rad_s,
                self.config.grip_recovery_rate_rad_s,
                self.config.grip_strain_hysteresis,
            )
            for side in ("left", "right")
        }
        self.instantaneous_stretch = 1.0

    def _limited_gripper_angle(
        self, side: str, nominal_rad: float, *, closing: bool = False
    ) -> float:
        if not self.config.strain_limited_grip:
            return nominal_rad
        minimum = (
            self.config.open_gripper_rad
            if closing
            else self.legs[side].get("acquired_gripper_rad", self.config.closed_gripper_rad)
        )
        return self.grip_limiters[side].command(
            nominal_rad,
            self.instantaneous_stretch,
            self.robot.sim_world.config.physics_dt,
            minimum_rad=min(nominal_rad, minimum),
        )

    def _coms(self, vertices: np.ndarray) -> np.ndarray:
        n, mass = self.scene.garment_count, self.runtime._masses
        return np.array(
            [
                np.average(vertices[:n], axis=0, weights=mass[:n]),
                np.average(vertices[n:], axis=0, weights=mass[n:]),
            ]
        )

    def _grasp_tcp_height(self, side: str, angle: float) -> float:
        if self.config.grasp_pad_bottom_m is None:
            return self.config.grasp_height_m
        bottom = getattr(self, "_episode_grasp_pad_bottom_m", self.config.grasp_pad_bottom_m)
        return self.runtime.pad_bottom_tcp_height(side, angle, bottom)

    def _measure_held_shape(self, vertices: np.ndarray, side: str) -> None:
        """Retain strict measurements even when a diagnostic run continues past a failure."""
        held = audit_held_shape(
            vertices[None],
            self.rest,
            self.scene.triangle_indices.reshape(-1, 3),
            self.runtime._masses,
            self.scene.garment_count,
            {side: [0]},
            maximum_shape_rms_m=self.config.maximum_shape_rms_m,
            maximum_bag_gap_m=self.config.maximum_bag_gap_m,
            quality_reference=getattr(self.scene, "quality_reference", None),
        )
        self.online_held_shape_samples.append(
            dict(
                held["samples"][0],
                time_s=self.elapsed,
                sample_index=len(self.samples) - 1,
            )
        )
        get_logger(__name__).info(
            "Sorting held shape check",
            time_s=self.elapsed,
            measurements=held["samples"][0],
            checks=held["checks"],
        )
        required = (
            held["checks"]
            if self.config.require_bag_gap_quality
            else without_bag_gap_checks(held["checks"], held)
        )
        if not all(required.values()) and self.config.early_stop_on_shape_failure:
            self.failure_reason = f"{side} held parcel exceeded required shape-quality limits"

    def _measure_moving_landing(
        self, vertices: np.ndarray, coms: np.ndarray, robot_contacts: int, outfeed_contacts: int
    ) -> None:
        """Record initial touchdown separately from settling on a continuously moving belt."""
        low, high = self.destination_bounds
        supported = bool(np.all((vertices[:, :2] >= low[:2]) & (vertices[:, :2] <= high[:2])))
        if not supported or robot_contacts or not outfeed_contacts:
            return
        leg = self.legs["right"]
        if "landing_time_s" not in leg:
            leg.update(
                landing_time_s=self.elapsed,
                landing_xy_m=coms[:, :2].tolist(),
                place_error=float(np.max(np.linalg.norm(coms[:, :2] - self.goal[:2], axis=1))),
            )
        height = float(np.max(coms[:, 2] - self.goal[2]))
        label = self.scene.label_up_cos(vertices)
        if (
            "release_verified_time_s" not in leg
            and leg["place_error"] <= self.config.maximum_place_error_m
            and height < 0.07
            and np.min(coms[:, 2] - self.goal[2]) >= -self.config.maximum_table_penetration_m
            and label >= self.config.label_up_min_cos
        ):
            leg.update(
                release_verified_time_s=self.elapsed,
                minimum_height=height,
                released=True,
                footprint_supported=True,
                label_up_cos=label,
            )

    def _enter(self, index: int, coms: np.ndarray) -> None:
        side, phase, _ = self.schedule[index]
        self._index = index
        self.checkpoint_times.append(self.elapsed)
        self.start_tcp = self.runtime.tcp_position(side)
        if phase == "infeed":
            self.runtime.infeed_speed = self.config.infeed_speed_mps
            self.infeed_start_y = float(coms[0, 1])
            return
        if phase != "approach" and self.command_side == side and self.command_tcp is not None:
            self.start_tcp = self.command_tcp.copy()
        if phase == "approach":
            if side == "left":
                self.runtime.infeed_speed = 0.0
            offset = (
                self.config.regrasp_offset_xy_m
                if side == "right" and self.config.regrasp_offset_xy_m is not None
                else self.config.grasp_offset_xy_m
            )
            self.pick = coms[0].copy()
            self.pick[:2] += offset
            self.support_z = float(
                self.scene.supports["infeed" if side == "left" else "platform"][1][2]
            )
            self._episode_grasp_pad_bottom_m = self.config.grasp_pad_bottom_m
            if side == "left" and self.config.infeed_grasp_pad_bottom_m is not None:
                self._episode_grasp_pad_bottom_m = self.config.infeed_grasp_pad_bottom_m
            if side == "right" and self.config.regrasp_surface_indent_m is not None:
                self._episode_grasp_pad_bottom_m = _regrasp_pad_bottom(
                    self.runtime.vertices()[: self.scene.garment_count],
                    self.pick[:2],
                    self.support_z,
                    self.config.grasp_pad_bottom_m,
                    self.config.regrasp_surface_indent_m,
                    self.config.regrasp_surface_indent_fraction,
                )
                if (
                    self._grasp_tcp_height(side, self.config.closed_gripper_rad)
                    + self.config.minimum_lift_m
                    > self.config.lift_height_m
                ):
                    raise ValueError("Live regrasp surface leaves insufficient lift clearance")
                self.legs[side]["grasp_pad_bottom_m"] = self._episode_grasp_pad_bottom_m
                get_logger(__name__).info(
                    "Sorting live regrasp height",
                    side=side,
                    pad_bottom_above_support_m=self._episode_grasp_pad_bottom_m,
                    surface_indent_m=self.config.regrasp_surface_indent_m,
                    surface_indent_fraction=self.config.regrasp_surface_indent_fraction,
                )
            self.legs[side]["baseline"] = coms[:, 2].copy()
            low, high = self.scene.supports["platform" if side == "left" else "outfeed"]
            self.destination_bounds = low, high
            self.goal = (low + high) / 2
            if side == "right":
                self.goal[:2] = self.scene.config.outfeed_target_xy
            elif (point := getattr(self.scene.config, "platform_target_xy", None)) is not None:
                if not np.all((np.asarray(point) > low[:2]) & (np.asarray(point) < high[:2])):
                    raise ValueError("Platform target must be inside the measured support bounds")
                self.goal[:2] = point
            self.goal[2] = high[2]
        previous_target = getattr(self, "target", self.start_tcp).copy()
        self.target = self.start_tcp.copy()
        if phase == "approach":
            self._place_height_m = (
                self.config.outfeed_place_height_m
                if side == "right" and self.config.outfeed_place_height_m is not None
                else self.config.place_height_m
            )
            approach_height = max(
                0.20, self._grasp_tcp_height(side, self.config.open_gripper_rad) + 0.10
            )
            self.target = np.r_[self.pick[:2], self.support_z + approach_height]
            nominal_goal = self.goal.copy()
            nominal_goal[:2] += offset
            self.max_tcp_error = max(
                self.max_tcp_error,
                self.runtime.prepare_approach(
                    side,
                    self.target,
                    [
                        self.target,
                        np.r_[
                            self.pick[:2],
                            self.support_z
                            + self._grasp_tcp_height(side, self.config.open_gripper_rad),
                        ],
                        np.r_[
                            self.pick[:2],
                            self.support_z
                            + self._grasp_tcp_height(side, self.config.closed_gripper_rad),
                        ],
                        np.r_[self.pick[:2], self.support_z + self.config.lift_height_m],
                        np.r_[nominal_goal[:2], self.support_z + self.config.lift_height_m],
                        nominal_goal + [0, 0, self._place_height_m],
                        *(
                            [
                                nominal_goal
                                + [
                                    0,
                                    0,
                                    self._place_height_m - self.config.release_lowering_m,
                                ]
                            ]
                            if self.config.release_lowering_m > 0
                            else []
                        ),
                        nominal_goal
                        + [
                            0,
                            0,
                            self._place_height_m - self.config.release_lowering_m + 0.18,
                        ],
                        self.runtime.parking_positions[side],
                    ],
                ),
            )
            self.command_tcp = self.target.copy()
            self.command_side = side
        elif phase in ("descend", "close"):
            angle = (
                self.config.open_gripper_rad
                if phase == "descend"
                else self.config.closed_gripper_rad
            )
            self.target = np.r_[self.pick[:2], self.support_z + self._grasp_tcp_height(side, angle)]
        elif phase == "lift":
            # Preserve the grip acquired with strain relief instead of forcing
            # the nominal closed angle back onto the parcel at the phase boundary.
            if self.config.strain_limited_grip:
                self.legs[side]["acquired_gripper_rad"] = min(
                    self.config.closed_gripper_rad,
                    float(self.runtime.target_q[self.robot.gripper_indices[side]]),
                )
            self.target[2] = self.support_z + self.config.lift_height_m
        elif phase == "hold":
            self.target = previous_target
        elif phase == "transfer":
            self.target[:2] += self.goal[:2] - coms[0, :2]
        elif phase == "lower":
            self.target[:2] += self.goal[:2] - coms[0, :2]
            self.target[2] = self.goal[2] + self._place_height_m
        elif phase == "release":
            self.legs[side]["release_z"] = coms[:, 2].copy()
            self.target[2] -= self.config.release_lowering_m
            self.release_start_angle = float(
                self.runtime.target_q[self.robot.gripper_indices[side]]
            )
        elif phase == "retract":
            self.target[2] += 0.18
        elif phase == "return":
            self.target = self.runtime.parking_positions[side].copy()
        elif phase == "conveyor":
            self._belt_start = np.asarray(
                self.legs[side].get("landing_xy_m", coms[:, :2]), float
            ).copy()
            self.runtime.belt_speed = self.config.belt_speed_mps

    def on_physics_step(self, robot: Any) -> bool:
        """Measure actual states, then change only robot/belt control commands."""
        del robot
        vertices = self.runtime.vertices()
        coms = self._coms(vertices)
        index = int(np.searchsorted(self.ends, self.elapsed, side="right"))
        q, action = self.runtime.recording_coordinates()
        self.samples.append(vertices.astype(np.float32))
        self.sample_times.append(self.elapsed)
        self.joints.append(q)
        self.actions.append(action)
        body_poses = self.runtime.body_poses()
        self.hand_poses.append(body_poses[list(self.runtime.hand_bodies.values())])
        self.max_penetration = max(self.max_penetration, self.scene.support_penetration(vertices))
        lengths = np.linalg.norm(
            vertices[self._edge_indices[:, 0]] - vertices[self._edge_indices[:, 1]], axis=1
        )
        self.instantaneous_stretch = float(np.max(lengths / self._edge_lengths))
        self.max_stretch = max(self.max_stretch, self.instantaneous_stretch)
        tri = vertices[self.scene.triangle_indices.reshape(-1, 3)]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        self.min_area = min(self.min_area, float(np.min(area / self.rest_areas)))
        full_q = self.runtime.state_0.joint_q.numpy()
        self.articulation_samples.append(full_q)
        self.body_samples.append(body_poses)
        jaw = audit_jaw_penetration(
            vertices[None],
            self.scene.triangle_indices.reshape(-1, 3),
            self.body_samples[-1][None],
            self.runtime.jaw_colliders,
            tolerance_m=self.config.maximum_table_penetration_m,
        )
        self.max_finger_geometry_penetration = max(
            self.max_finger_geometry_penetration, jaw["maximum_penetration_m"]
        )
        for side, pairs in self.robot.mimic.items():
            master = full_q[self.robot.gripper_indices[side]]
            self.max_mimic_error = max(
                self.max_mimic_error, max(abs(full_q[i] - ratio * master) for i, ratio in pairs)
            )
        self.max_mimic_error = max(
            self.max_mimic_error, getattr(self.runtime, "maximum_native_mimic_error", 0.0)
        )
        if index >= len(self.schedule):
            if not self.config.continuous_outfeed:
                self.runtime.belt_speed = 0.0
            self.state.update(done=True, phase="done")
            return True
        if self.max_mimic_error >= 0.03:
            self.failure_reason = "Native gripper mimic constraints exceeded 0.03 rad"
        if self.max_stretch >= self.config.maximum_edge_stretch:
            self.failure_reason = "Surface stretch exceeded the fixed V7 limit"
        if self.max_penetration > self.config.maximum_table_penetration_m:
            self.failure_reason = "Parcel penetrated a support beyond the fixed tolerance"
        if self.max_finger_geometry_penetration > self.config.maximum_table_penetration_m:
            self.failure_reason = "Parcel penetrated native finger geometry beyond tolerance"
        if self.max_contact_penetration > self.config.maximum_table_penetration_m:
            self.failure_reason = "Sampled finger contact penetration exceeded tolerance"
        if index != self._index and self.schedule[index][1] == "transfer":
            previous_side = self.schedule[index][0]
            if not np.all(self.legs[previous_side]["peak_lifts"] >= self.config.minimum_lift_m):
                self.failure_reason = f"{previous_side} arm did not lift both physical meshes"
        if (
            index != self._index
            and self.schedule[index][1] in {"approach", "conveyor"}
            and (self.schedule[index][0] == "right")
            and index > 0
        ):
            previous_side = "left" if self.schedule[index][1] == "approach" else "right"
            previous = self.legs[previous_side]
            if not (
                previous["released"]
                and previous["place_error"] <= self.config.maximum_place_error_m
                and previous["minimum_height"] < 0.07
                and previous["footprint_supported"]
                and previous["label_up_cos"] is not None
                and previous["label_up_cos"] >= self.config.label_up_min_cos
            ):
                self.failure_reason = (
                    f"{previous_side} arm did not release onto the destination support"
                )
        if self.failure_reason is not None:
            self.runtime.infeed_speed = 0.0
            self.state.update(done=False, phase="failed")
            return True
        if index != self._index:
            self._enter(index, coms)
        side, phase, duration = self.schedule[index]
        if phase == "infeed":
            arrived = coms[0, 1] >= self.scene.config.target_xy[1]
            if arrived:
                self.runtime.infeed_speed = 0.0
            self.rows.append(
                dict(
                    time_s=self.elapsed,
                    phase="left_infeed",
                    garment_com_m=coms[0].tolist(),
                    bag_com_m=coms[1].tolist(),
                    infeed_speed_mps=self.runtime.infeed_speed,
                    state=q.tolist(),
                    action=action.tolist(),
                )
            )
            self.state.update(
                phase="left_infeed",
                progress={
                    "arrived": bool(arrived),
                    "parcel_y_m": float(coms[0, 1]),
                    "infeed_speed_mps": self.runtime.infeed_speed,
                },
            )
            self._frame += 1
            self.elapsed = self._frame * self.robot.sim_world.config.physics_dt
            return False
        leg = self.legs[side]
        leg["peak_lifts"] = np.maximum(leg["peak_lifts"], coms[:, 2] - leg["baseline"])
        counts = self.runtime.last_contact_counts
        distances = getattr(self.runtime, "last_contact_distances_m", {})
        self.max_contact_penetration = max(
            self.max_contact_penetration,
            max(
                (
                    -value
                    for name, value in distances.items()
                    if name in {"left_0", "left_1", "right_0", "right_1"} and value is not None
                ),
                default=0.0,
            ),
        )
        bilateral = all(counts.get(f"{side}_{i}", 0) > 0 for i in (0, 1))
        robot_contacts = _robot_contact_count(self.runtime)
        if (
            phase == "lift"
            and robot_contacts == 0
            and self.runtime.pad_bottom_z(side, body_poses) > float(vertices[:, 2].max()) + 0.01
        ):
            self.failure_reason = f"{side} arm lost grasp below the rising native pads"
            self.runtime.belt_speed = 0.0
            self.state.update(done=False, phase="failed")
            return True
        if phase == "hold":
            leg["hold_frames"] += 1
            leg["hold_contact_frames"] += int(bilateral)
            leg["minimum_hold_lift"] = np.minimum(
                leg["minimum_hold_lift"], coms[:, 2] - leg["baseline"]
            )
        if phase == "transfer":
            leg["transfer_frames"] += 1
            leg["contact_frames"] += int(bilateral)
            leg["minimum_transfer_lift"] = np.minimum(
                leg["minimum_transfer_lift"], coms[:, 2] - leg["baseline"]
            )
            self.transfer_samples.append(len(self.samples) - 1)
            self.transfer_groups[side].append(len(self.samples) - 1)
            expected_frames = np.ceil(duration / self.robot.sim_world.config.physics_dt) + 1
            missed = leg["transfer_frames"] - leg["contact_frames"]
            # These conditions already make final acceptance impossible. Stop
            # promptly while preserving synchronized failure evidence.
            if missed > 0.05 * expected_frames:
                self.failure_reason = f"{side} arm lost bilateral grip during transfer"
            elif np.any(leg["minimum_transfer_lift"] < self.config.minimum_lift_m):
                self.failure_reason = f"{side} arm lost sustained lift during transfer"
            frames = max(1, int(np.ceil(duration / self.robot.sim_world.config.physics_dt - 1e-8)))
            if leg["transfer_frames"] in {1, (frames - 1) // 2 + 1, frames}:
                self._measure_held_shape(vertices, side)
            if self.failure_reason is not None:
                self.runtime.belt_speed = 0.0
                self.state.update(done=False, phase="failed")
                return True
        if "release_z" in leg and phase in ("release", "retract", "rest"):
            leg["release_drop"] = max(
                leg["release_drop"], float(np.min(leg["release_z"] - coms[:, 2]))
            )
        moving_destination = side == "right" and self.config.continuous_outfeed
        if moving_destination and phase in {"release", "retract", "return", "rest"}:
            self._measure_moving_landing(vertices, coms, robot_contacts, counts.get("outfeed", 0))
        if phase == "rest" and not moving_destination:
            leg["place_error"] = float(np.max(np.linalg.norm(coms[:, :2] - self.goal[:2], axis=1)))
            leg["minimum_height"] = float(np.max(coms[:, 2] - self.goal[2]))
            leg["released"] = robot_contacts == 0
            low, high = self.destination_bounds
            leg["footprint_supported"] = bool(
                np.all((vertices[:, :2] >= low[:2]) & (vertices[:, :2] <= high[:2]))
            )
            leg["label_up_cos"] = self.scene.label_up_cos(vertices)
        start = 0.0 if index == 0 else self.ends[index - 1]
        blend = _motion_blend(
            (self.elapsed - start) / duration,
            self.config.transfer_profile if phase == "transfer" else "cubic",
        )
        grip_blend = float(np.clip((self.elapsed - start) / (duration * 0.7), 0, 1))
        grip_blend = grip_blend * grip_blend * (3 - 2 * grip_blend)
        target = self.start_tcp * (1 - blend) + self.target * blend
        if phase == "hold":
            target = self.target
        if phase == "approach":
            self.runtime.command_approach(side, blend)
        elif phase in {"close", "release"}:
            opening = self.config.open_gripper_rad
            angle = (
                opening + grip_blend * (self.config.closed_gripper_rad - opening)
                if phase == "close"
                else opening + (1 - grip_blend) * (self.release_start_angle - opening)
            )
            if phase == "close":
                angle = self._limited_gripper_angle(side, angle, closing=True)
            if phase == "close" and self.config.grasp_pad_bottom_m is not None:
                target = self.start_tcp.copy()
                target[2] = self.support_z + self._grasp_tcp_height(side, angle)
                error = self.runtime.command_arm(side, target, angle)
                self.command_tcp = target.copy()
                self.command_side = side
                self.max_tcp_error = max(self.max_tcp_error, error)
            elif phase == "release" and self.config.release_lowering_m > 0:
                error = self.runtime.command_arm(side, target, angle)
                self.command_tcp = target.copy()
                self.command_side = side
                self.max_tcp_error = max(self.max_tcp_error, error)
            else:
                self.runtime.set_gripper_angle(side, angle)
        elif phase not in {"conveyor", "rest", "hold"}:
            angle = self.config.open_gripper_rad
            if phase == "lift":
                angle = self.config.closed_gripper_rad + blend * (
                    self.config.hold_gripper_rad - self.config.closed_gripper_rad
                )
            elif phase in {"transfer", "lower"}:
                angle = self.config.hold_gripper_rad
            if phase in {"lift", "transfer", "lower"}:
                angle = self._limited_gripper_angle(side, angle)
            error = self.runtime.command_arm(side, target, angle)
            self.command_tcp = target.copy()
            self.command_side = side
            self.max_tcp_error = max(self.max_tcp_error, error)
        elif phase == "hold":
            self.runtime.set_gripper_angle(
                side, self._limited_gripper_angle(side, self.config.hold_gripper_rad)
            )
        elif phase == "conveyor":
            self.belt_displacement = float(np.min(self._belt_start[:, 0] - coms[:, 0]))
            low, high = self.scene.supports["outfeed"]
            self.belt_exited = self.belt_exited or bool(
                vertices[:, 0].max() < low[0]
                and np.all((coms[:, 1] >= low[1]) & (coms[:, 1] <= high[1]))
            )
            if self.belt_exited:
                if not self.config.continuous_outfeed:
                    self.runtime.belt_speed = 0.0
                    self.state.update(done=True, phase="done")
                    return True
        state, action = self.runtime.recording_coordinates()
        self.actions[-1] = action
        self.rows.append(
            dict(
                time_s=self.elapsed,
                phase=f"{side}_{phase}",
                garment_com_m=coms[0].tolist(),
                bag_com_m=coms[1].tolist(),
                tcp_m=self.runtime.tcp_position(side).tolist(),
                pad_bottom_m=self.runtime.pad_bottom_z(side, body_poses) - self.support_z,
                target_tcp_m=(
                    self.command_tcp.tolist()
                    if self.command_tcp is not None and phase != "approach"
                    else None
                ),
                contacts=counts,
                robot_contacts=robot_contacts,
                grip_relief_rad=self.grip_limiters[side].relief_rad,
                state=state.tolist(),
                action=action.tolist(),
            )
        )
        self.state.update(
            phase=f"{side}_{phase}",
            progress=dict(
                maximum_edge_stretch=self.max_stretch,
                instantaneous_edge_stretch=self.instantaneous_stretch,
                grip_relief_rad=self.grip_limiters[side].relief_rad,
                minimum_triangle_area_ratio=self.min_area,
                belt_displacement_m=self.belt_displacement,
                contacts=counts,
                robot_contacts=robot_contacts,
                com_lift_m=(coms[:, 2] - leg["baseline"]).tolist(),
                gripper_angle_rad=float(full_q[self.robot.gripper_indices[side]]),
                maximum_mimic_error_rad=float(self.max_mimic_error),
                maximum_support_penetration_m=self.max_penetration,
                maximum_finger_geometry_penetration_m=self.max_finger_geometry_penetration,
            ),
        )
        self._frame += 1
        self.elapsed = self._frame * self.robot.sim_world.config.physics_dt
        return False

    def write_diagnostics(self, directory: Path) -> None:
        """Save every physics-frame particle sample and frame-aligned native joint data."""
        collision_data = {}
        for index, jaw in enumerate(self.runtime.jaw_colliders):
            collision_data[f"jaw_vertices_{index}"] = jaw["vertices"]
            collision_data[f"jaw_triangles_{index}"] = jaw["triangles"]
        fixed = getattr(self.runtime, "fixed_robot_colliders", [])
        for index, collider in enumerate(fixed):
            collision_data[f"fixed_robot_vertices_{index}"] = collider["vertices"]
            collision_data[f"fixed_robot_triangles_{index}"] = collider["triangles"]
        for name in ("particle_radius", "shape_margin"):
            if hasattr(self.runtime.model, name):
                collision_data[name] = getattr(self.runtime.model, name).numpy()
        np.savez_compressed(
            directory / "particles.npz",
            positions=np.asarray(self.samples),
            times=np.asarray(self.sample_times),
            triangles=self.scene.triangle_indices.reshape(-1, 3),
            rest=self.rest,
            masses=self.runtime._masses,
            garment_count=self.scene.garment_count,
            quality_reference_json=json.dumps(getattr(self.scene, "quality_reference", None)),
            checkpoint_times_s=sorted(set([*self.checkpoint_times, self.sample_times[-1]])),
            state=np.asarray(self.joints),
            action=np.asarray(self.actions),
            hand_poses=np.asarray(self.hand_poses),
            transfer_indices=np.asarray(self.transfer_samples),
            left_transfer_indices=np.asarray(self.transfer_groups["left"]),
            right_transfer_indices=np.asarray(self.transfer_groups["right"]),
            support_bounds=np.array(list(self.scene.supports.values())),
            label_triangles=self.scene.label_triangles,
            articulation_q=np.asarray(self.articulation_samples),
            body_poses=np.asarray(self.body_samples),
            jaw_body_ids=[jaw["body_id"] for jaw in self.runtime.jaw_colliders],
            shape_material_mu=self.runtime.model.shape_material_mu.numpy(),
            shape_body=self.runtime.model.shape_body.numpy(),
            soft_contact_mu=self.runtime.model.soft_contact_mu,
            belt_body=self.runtime.belt_body,
            platform_shape=getattr(self.runtime, "platform_shape", -1),
            robot_shape_count=getattr(self.runtime, "robot_shape_count", 0),
            fixed_robot_shape_ids=[collider["shape_id"] for collider in fixed],
            mimic_pairs=[
                (index, self.robot.gripper_indices[side], ratio)
                for side, pairs in self.robot.mimic.items()
                for index, ratio in pairs
            ],
            native_mimic_residuals=np.concatenate(
                getattr(self.runtime, "native_mimic_residuals", [])
                or [np.empty((0, sum(len(pairs) for pairs in self.robot.mimic.values())))],
                axis=0,
            ),
            native_mimic_dt_s=self.robot.sim_world.config.physics_dt
            / (self.runtime.config.newton_substeps * self.config.mujoco_substeps_per_vbd_step),
            **collision_data,
        )

    def result(self) -> dict[str, Any]:
        """Require both grasps, the platform release, the second release and belt motion."""
        capacity = self.runtime.cloth_solver.maximum_contacts.numpy()
        containment = check_package_containment(
            np.asarray(self.samples),
            self.scene.triangle_indices.reshape(-1, 3),
            self.scene.garment_count,
            times=np.asarray(self.sample_times),
            checkpoint_times_s=sorted(set([*self.checkpoint_times, self.sample_times[-1]])),
        )
        checks = dict(
            sequence_completed=bool(self.state.get("done")),
            finite_state=bool(np.isfinite(self.samples).all()),
            garment_mass_matches_config=abs(
                float(self.runtime._masses[: self.scene.garment_count].sum())
                - self.scene.config.garment_mass_kg
            )
            < 1e-6,
            bag_mass_matches_config=abs(
                float(self.runtime._masses[self.scene.garment_count :].sum())
                - (
                    self.scene.config.bag_mass_kg
                    if self.scene.config.bag_mass_kg is not None
                    else self.scene.authored_bag_mass_kg
                )
            )
            < 1e-6,
            no_pinned_vertices=self.runtime.pinned_vertex_count == 0,
            surface_stretch_bounded=self.max_stretch < self.config.maximum_edge_stretch,
            no_collapsed_triangles=self.min_area > self.config.minimum_triangle_area_ratio,
            support_penetration_bounded=self.max_penetration
            <= self.config.maximum_table_penetration_m,
            garment_contained_at_checkpoints=containment["passed"],
            no_contact_buffer_overflow=bool(
                max(capacity[:2]) < self.config.self_contact_buffer and capacity[2] < 16384
            ),
            ik_targets_reachable=self.max_tcp_error < self.config.maximum_tcp_error_m,
            mimic_constraints_maintained=self.max_mimic_error < 0.03,
            conveyor_transport_observed=self.belt_displacement
            >= self.config.minimum_belt_displacement_m,
            parcel_exited_outfeed=self.belt_exited,
            sampled_finger_contact_penetration_bounded=self.max_contact_penetration
            <= self.config.maximum_table_penetration_m,
        )
        shape = audit_held_shape(
            np.asarray(self.samples),
            self.rest,
            self.scene.triangle_indices.reshape(-1, 3),
            self.runtime._masses,
            self.scene.garment_count,
            self.transfer_groups,
            maximum_shape_rms_m=self.config.maximum_shape_rms_m,
            maximum_bag_gap_m=self.config.maximum_bag_gap_m,
            quality_reference=getattr(self.scene, "quality_reference", None),
            extra_samples={
                side: [
                    row["sample_index"]
                    for row in self.online_held_shape_samples
                    if row["side"] == side
                ]
                for side in self.transfer_groups
            },
        )
        checks.update(shape["checks"])
        jaw_audit = audit_jaw_penetration(
            np.asarray(self.samples),
            self.scene.triangle_indices.reshape(-1, 3),
            np.asarray(self.body_samples),
            self.runtime.jaw_colliders,
            tolerance_m=self.config.maximum_table_penetration_m,
        )
        checks["all_saved_vertices_and_face_centres_clear_of_fingers"] = jaw_audit["passed"]
        legs = {}
        for side, leg in self.legs.items():
            fraction = leg["contact_frames"] / max(1, leg["transfer_frames"])
            checks.update(
                {
                    f"{side}_both_meshes_lifted": bool(
                        np.all(leg["peak_lifts"] >= self.config.minimum_lift_m)
                    ),
                    f"{side}_sustained_transfer_lift": bool(
                        np.all(leg["minimum_transfer_lift"] >= self.config.minimum_lift_m)
                        and leg["transfer_frames"] > 0
                    ),
                    f"{side}_bilateral_transfer": fraction >= 0.95,
                    f"{side}_pretransfer_hold": self.config.hold_s == 0
                    or (
                        leg["hold_frames"] > 0
                        and leg["hold_contact_frames"] / leg["hold_frames"] >= 0.95
                        and np.all(leg["minimum_hold_lift"] >= self.config.minimum_lift_m)
                    ),
                    f"{side}_released": leg["released"]
                    and leg["release_drop"] >= self.config.minimum_release_drop_m,
                    f"{side}_placed_on_destination": leg["place_error"]
                    <= self.config.maximum_place_error_m
                    and leg["minimum_height"] < 0.07
                    and leg["footprint_supported"],
                    f"{side}_label_up": leg["label_up_cos"] is not None
                    and leg["label_up_cos"] >= self.config.label_up_min_cos,
                }
            )
            legs[side] = _finite_metrics(leg)
            legs[side]["bilateral_transfer_fraction"] = fraction
        checks = {key: bool(value) for key, value in checks.items()}
        strict_checks = checks.copy()
        workflow_checks = without_bag_gap_checks(strict_checks, shape)
        if not self.config.require_bag_gap_quality:
            checks = workflow_checks
        return dict(
            success=all(checks.values()),
            workflow_success=all(workflow_checks.values()),
            strict_quality_success=all(strict_checks.values()),
            acceptance_profile=(
                "strict_v7" if self.config.require_bag_gap_quality else "workflow_gap_advisory"
            ),
            checks=checks,
            failed_checks=[key for key, passed in checks.items() if not passed],
            strict_checks=strict_checks,
            strict_failed_checks=[key for key, passed in strict_checks.items() if not passed],
            legs=legs,
            object=(
                f"{getattr(self.scene, 'package_parameters', {}).get('version', 'V7')}"
                "_two_mesh_package"
            ),
            robot_backend=self.robot.config.backend,
            robot_asset_sha256=hashlib.sha256(
                Path(self.robot.source_path).read_bytes()
            ).hexdigest(),
            gripper_geometry="authored_ProM6_Robotiq_with_visual_matched_27x7.5x65mm_pads",
            native_pad_geometry=self.robot.native_pad_geometry,
            pad_width_m=self.robot.config.pad_width_m,
            garment_mass_kg=float(self.runtime._masses[: self.scene.garment_count].sum()),
            bag_mass_kg=float(self.runtime._masses[self.scene.garment_count :].sum()),
            target_garment_mass_kg=self.scene.config.garment_mass_kg,
            authored_garment_mass_kg=self.scene.authored_garment_mass_kg,
            garment_mass_scale=self.scene.garment_mass_scale,
            authored_bag_mass_kg=self.scene.authored_bag_mass_kg,
            target_bag_mass_kg=(
                self.scene.config.bag_mass_kg
                if self.scene.config.bag_mass_kg is not None
                else self.scene.authored_bag_mass_kg
            ),
            bag_mass_scale=self.scene.bag_mass_scale,
            contact_friction=self.runtime.contact_friction,
            coupling="one_way_rigid_to_deformable",
            attachments_used=False,
            strain_limited_grip=self.config.strain_limited_grip,
            peak_grip_relief_rad={
                side: limiter.peak_relief_rad for side, limiter in self.grip_limiters.items()
            },
            pinned_vertex_count=self.runtime.pinned_vertex_count,
            maximum_edge_stretch=self.max_stretch,
            minimum_triangle_area_ratio=self.min_area,
            maximum_support_penetration_m=self.max_penetration,
            maximum_sampled_finger_contact_penetration_m=self.max_contact_penetration,
            maximum_ik_target_error_m=self.max_tcp_error,
            maximum_mimic_error_rad=float(self.max_mimic_error),
            belt_displacement_m=self.belt_displacement,
            maximum_substep_contacts=capacity.tolist(),
            containment=containment,
            held_shape_audit=shape,
            online_held_shape_samples=self.online_held_shape_samples,
            finger_geometry_audit=jaw_audit,
            physics_duration_s=self.elapsed,
            failure_reason=self.failure_reason,
            newton_version=self.runtime.newton.__version__,
            warp_version=self.runtime.wp.__version__,
            asset_sha256=hashlib.sha256(
                Path(self.scene._asset(self.scene.config.cloth_usd_path)).read_bytes()
            ).hexdigest(),
        )


# ================================================================================================
# One shared MuJoCo/VBD world for a soft parcel followed by a dynamic V10 carton.
# ================================================================================================


class SortingStreamRuntime(SortingRuntime):
    """Keep both parcels in the same model and insert the carton only once, at spawn."""

    def __init__(self, robot: Any, scene: Any, config: Any) -> None:
        self.carton_spawned = False
        self.soft_asleep = False
        self.belt_offsets = np.zeros(2)
        super().__init__(robot, scene, config)
        solver = self.rigid_solver
        native_dt = (
            robot.sim_world.config.physics_dt
            / config.newton_substeps
            / config.mujoco_substeps_per_vbd_step
        )
        if config.rigid_contact_time_constant_s < 2 * native_dt:
            raise ValueError("Rigid contact time constant must cover at least two native substeps")
        bodies = solver.mjc_body_to_newton.numpy()[0]
        shapes = solver.mjc_geom_to_newton_shape.numpy()[0]
        self.carton_mj_body = int(np.flatnonzero(bodies == self.carton_body)[0])
        self.carton_geom = int(np.flatnonzero(shapes == self.carton_shape)[0])
        self._carton_masks = (
            int(solver.mj_model.geom_contype[self.carton_geom]),
            int(solver.mj_model.geom_conaffinity[self.carton_geom]),
        )
        solver.mj_model.geom_contype[self.carton_geom] = 0
        solver.mj_model.geom_conaffinity[self.carton_geom] = 0
        solver.mj_model.body_gravcomp[self.carton_mj_body] = 1.0
        # Rigid contact uses its own critically damped native material. VBD's
        # soft-contact damping has a different meaning and is left unchanged.
        solver.mj_model.geom_solmix[:] = 0.0
        solver.mj_model.geom_solmix[self.carton_geom] = 1.0
        solver.mj_model.geom_solref[self.carton_geom] = [config.rigid_contact_time_constant_s, 1.0]
        solver.mj_model.geom_solimp[self.carton_geom, :2] = [0.95, 0.99]
        self._mj_shapes = shapes
        self._mj_bodies = bodies
        self.belt_geoms = {
            int(np.flatnonzero(shapes == self.infeed_shape)[0]): "infeed",
            int(np.flatnonzero(shapes == self.belt_shape)[0]): "outfeed",
        }
        # MuJoCo's fixed mocap bodies have no tangential surface velocity. Use
        # contact-normal-limited Coulomb belt forces in their tangent plane;
        # disable only their native sliding friction to avoid counting it twice.
        self._native_belt_mu = {}
        for geom in self.belt_geoms:
            self._native_belt_mu[geom] = self.config.table_friction
            solver.mj_model.geom_friction[geom, 0] = 0.0
        # MuJoCo combines native sliding friction with max(), whereas the VBD
        # path uses Newton's geometric mean. Let the counter-surface specify
        # the rigid contact coefficient explicitly.
        solver.mj_model.geom_friction[self.carton_geom, 0] = 0.0
        platform_geom = int(np.flatnonzero(shapes == self.platform_shape)[0])
        solver.mj_model.geom_friction[platform_geom, 0] = (
            self.config.table_friction
            if self.config.platform_friction is None
            else self.config.platform_friction
        )
        finger_bodies = {
            self.contact_body_ids[f"{side}_{i}"] for side in ("left", "right") for i in (0, 1)
        }
        shape_bodies = self.model.shape_body.numpy()
        for geom, shape in enumerate(shapes):
            if int(shape_bodies[shape]) in finger_bodies:
                solver.mj_model.geom_friction[geom, 0] = (
                    self.config.gripper_package_friction or self.config.shape_contact_friction
                )
        self.carton_contacts: dict[str, int] = {}
        self.carton_contact_shapes = np.empty(0, int)
        self.carton_contact_distances: dict[str, float | None] = {}
        self.maximum_carton_contact_penetration_m = 0.0
        self.maximum_belt_force_n = 0.0

    def _add_deformables(self, builder: Any, *, xpbd: bool) -> None:
        super()._add_deformables(builder, xpbd=xpbd)
        half = self.scene.rigid_dimensions / 2
        self.carton_body = builder.add_body(
            xform=self.wp.transform(
                (100.0, 0.0, half[2] + self.scene.config.floor_z_m), self.wp.quat_identity()
            ),
            label="stream_carton",
        )
        cfg = self.newton.ModelBuilder.ShapeConfig(
            density=float(self.scene.config.rigid_mass_kg / np.prod(2 * half)),
            mu=self.config.shape_contact_friction,
            margin=0.0002,
        )
        self.carton_shape = builder.add_shape_box(
            self.carton_body,
            hx=float(half[0]),
            hy=float(half[1]),
            hz=float(half[2]),
            cfg=cfg,
            label="stream_carton_collision",
        )

    def spawn_carton(self) -> None:
        """Initialize a new parcel at the remote infeed end; never reset it after insertion."""
        if self.carton_spawned:
            raise RuntimeError("A carton can only be spawned once per continuous run")
        solver = self.rigid_solver
        solver.mj_model.opt.solver = (
            solver._mujoco.mjtSolver.mjSOL_PGS
            if self.config.rigid_mujoco_solver == "pgs"
            else solver._mujoco.mjtSolver.mjSOL_NEWTON
        )
        if self.config.rigid_disable_warmstart:
            solver.mj_model.opt.disableflags |= int(solver._mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        solver.mj_model.opt.iterations = self.config.rigid_mujoco_iterations
        solver.mj_model.body_gravcomp[self.carton_mj_body] = 0.0
        solver.mj_model.geom_contype[self.carton_geom] = self._carton_masks[0]
        solver.mj_model.geom_conaffinity[self.carton_geom] = self._carton_masks[1]
        joint = int(solver.mj_model.body_jntadr[self.carton_mj_body])
        q = int(solver.mj_model.jnt_qposadr[joint])
        v = int(solver.mj_model.jnt_dofadr[joint])
        position = np.array(
            [
                self.scene.config.target_xy[0],
                self.scene.config.target_xy[1] - self.scene.config.infeed_travel_m,
                self.scene.supports["infeed"][1][2]
                + self.scene.rigid_dimensions[2] / 2
                + self.scene.config.initial_clearance_m,
            ]
        )
        solver.mj_data.qpos[q : q + 3] = position - self.physics_origin
        solver.mj_data.qpos[q + 3 : q + 7] = [1, 0, 0, 0]
        solver.mj_data.qvel[v : v + 6] = 0.0
        solver.mj_data.qacc_warmstart[:] = 0.0
        solver._mujoco.mj_forward(solver.mj_model, solver.mj_data)
        for state in (self.state_0, self.state_1):
            solver._update_newton_state(self.model, state, solver.mj_data, state_prev=state)
        self.carton_spawned = True
        self.infeed_speed = self.config.infeed_speed_mps
        self._publish()

    def carton_vertices(self, poses: np.ndarray | None = None) -> np.ndarray:
        """Return original mesh vertices transformed by the solved free rigid body."""
        pose = (self.body_poses() if poses is None else poses)[self.carton_body]
        return self.scene.rigid_vertices @ _rotation_wxyz(pose[[6, 3, 4, 5]]).T + pose[:3]

    def _before_native_substep(self, dt: float) -> None:
        if not self.carton_spawned:
            return
        solver, body = self.rigid_solver, self.carton_mj_body
        model, data = solver.mj_model, solver.mj_data
        data.xfrc_applied[body] = 0.0
        contacts = []
        total_normal = 0.0
        for index in range(data.ncon):
            contact = data.contact[index]
            g0, g1 = map(int, contact.geom)
            if self.carton_geom not in (g0, g1):
                continue
            other = g1 if g0 == self.carton_geom else g0
            self.maximum_carton_contact_penetration_m = max(
                getattr(self, "maximum_carton_contact_penetration_m", 0.0),
                max(0.0, -float(contact.dist)),
            )
            if other not in self.belt_geoms:
                continue
            force = np.zeros(6)
            solver._mujoco.mj_contactForce(model, data, index, force)
            normal = max(0.0, float(force[0]))
            if normal > 0:
                contacts.append((other, normal, contact.pos.copy()))
                total_normal += normal
        if not contacts:
            return
        velocity = np.zeros(6)
        solver._mujoco.mj_objectVelocity(
            model, data, solver._mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
        )
        for geom, normal, point in contacts:
            speed = (
                np.array([0.0, self.infeed_speed, 0.0])
                if self.belt_geoms[geom] == "infeed"
                else np.array([-self.belt_speed, 0.0, 0.0])
            )
            arm = point - data.xipos[body]
            slip = speed - (velocity[3:] + np.cross(velocity[:3], arm))
            slip[2] = 0.0
            # Apply one critically limited momentum correction across all belt
            # contacts, bounded by each contact's measured Coulomb capacity.
            force = model.body_mass[body] * slip / max(dt, 0.01) * normal / total_normal
            limit = self._native_belt_mu[geom] * normal
            force *= min(1.0, limit / max(np.linalg.norm(force), 1e-12))
            data.xfrc_applied[body, :3] += force
            data.xfrc_applied[body, 3:] += np.cross(arm, force)
        self.maximum_belt_force_n = max(
            self.maximum_belt_force_n, float(np.linalg.norm(data.xfrc_applied[body, :3]))
        )

    def step(self, frame_dt: float) -> None:
        """Advance both solvers, allowing a settled floor parcel to sleep until approached."""
        if self.carton_spawned and self.config.sleep_soft_when_settled:
            vertices = self.vertices()
            carton = self.carton_vertices()
            separation = np.maximum(
                np.maximum(vertices.min(0) - carton.max(0), carton.min(0) - vertices.max(0)),
                0.0,
            )
            floor = self.scene.config.floor_z_m
            settled = (
                vertices[:, 2].max() < floor + 0.10
                and vertices[:, 2].min() >= floor - 0.0015
                and np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max() < 0.02
            )
            asleep = settled and np.linalg.norm(separation) > 0.20
            if asleep and not self.soft_asleep:
                self.state_0.particle_qd.zero_()
                self.state_1.particle_qd.zero_()
            self.soft_asleep = bool(asleep)
        self.belt_offsets += np.array([self.infeed_speed, self.belt_speed]) * frame_dt
        super().step(frame_dt)

    def _advance_soft(self, dt: float) -> None:
        if self.soft_asleep:
            self.wp.copy(self.state_1.particle_q, self.state_0.particle_q)
            self.wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
        else:
            super()._advance_soft(dt)

    def _publish(self) -> None:
        super()._publish()
        if not hasattr(self, "carton_mj_body"):
            return
        self.scene.publish_rigid_pose(
            self.body_poses()[self.carton_body], visible=self.carton_spawned
        )
        if self.scene.config.belt_markers:
            self.scene.publish_belt_markers(self.belt_offsets)
        self.carton_contacts = dict.fromkeys(self.contact_body_ids, 0)
        self.carton_contacts.update(infeed=0, outfeed=0, platform=0)
        self.carton_contact_distances = dict.fromkeys(self.contact_body_ids)
        shapes = []
        data = self.rigid_solver.mj_data
        for contact in data.contact[: data.ncon]:
            g0, g1 = map(int, contact.geom)
            if self.carton_geom not in (g0, g1) or contact.dist > 0.0005:
                continue
            other = g1 if g0 == self.carton_geom else g0
            shape = int(self._mj_shapes[other])
            shapes.append(shape)
            body = int(self.model.shape_body.numpy()[shape])
            for name, target in (
                ("infeed", self.infeed_shape),
                ("outfeed", self.belt_shape),
                ("platform", self.platform_shape),
            ):
                self.carton_contacts[name] += int(shape == target)
            for name, target in self.contact_body_ids.items():
                if body == target:
                    self.carton_contacts[name] += 1
                    old = self.carton_contact_distances[name]
                    self.carton_contact_distances[name] = min(
                        float(contact.dist), old if old is not None else np.inf
                    )
            self.maximum_carton_contact_penetration_m = max(
                self.maximum_carton_contact_penetration_m, max(0.0, -float(contact.dist))
            )
        self.carton_contact_shapes = np.asarray(shapes, int)

    def publish_recorded_visuals(
        self,
        vertices: np.ndarray,
        body_poses: np.ndarray,
        *,
        carton_visible: bool = False,
        belt_offsets: np.ndarray | None = None,
    ) -> None:
        """Replay the same dynamic bodies and VBD particles without stepping physics."""
        super().publish_recorded_visuals(vertices, body_poses)
        self.scene.publish_rigid_pose(body_poses[self.carton_body], visible=carton_visible)
        if self.scene.config.belt_markers and belt_offsets is not None:
            self.scene.publish_belt_markers(belt_offsets)


# ================================================================================================
# Sequential soft/rigid relay, with a single continuously integrated physical world.
# ================================================================================================


class SortingStreamPlannerConfig(SortingPlannerConfig):
    """Two parcels, inserted one at a time, sharing unchanged native robot hardware."""

    backend: str = "sorting_stream"
    rigid_closed_gripper_rad: FiniteFloat = Field(default=0.40, ge=0, le=0.7)
    rigid_hold_gripper_rad: FiniteFloat = Field(default=0.42, ge=0, le=0.7)
    rigid_pad_bottom_m: FiniteFloat = Field(default=0.015, ge=0, le=0.08)
    sleep_soft_when_settled: bool = True
    rigid_contact_time_constant_s: FiniteFloat = Field(default=0.0005, gt=0)
    rigid_mujoco_iterations: int = Field(default=100, gt=0)
    rigid_mujoco_solver: Literal["newton", "pgs"] = "pgs"
    rigid_disable_warmstart: bool = False
    separate_ground_drop_lanes: bool = False
    ground_lane_margin_m: FiniteFloat = Field(default=0.012, ge=0)
    lower_drop_tcp_orientation_wxyz: (
        tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] | None
    ) = None

    @model_validator(mode="after")
    def validate_lower_drop_orientation(self) -> SortingStreamPlannerConfig:
        """Require a unit quaternion for the optional lower-lane tool attitude."""
        if self.lower_drop_tcp_orientation_wxyz is not None and not np.isclose(
            np.linalg.norm(self.lower_drop_tcp_orientation_wxyz), 1.0, atol=1e-6
        ):
            raise ValueError("Lower drop TCP orientation must be a unit quaternion")
        return self


def rigid_relay_config(config: SortingStreamPlannerConfig) -> SortingStreamPlannerConfig:
    """Keep rigid grasps independent of the soft parcel's infeed depth and offsets."""
    return config.model_copy(
        update={
            "closed_gripper_rad": config.rigid_closed_gripper_rad,
            "hold_gripper_rad": config.rigid_hold_gripper_rad,
            "grasp_pad_bottom_m": config.rigid_pad_bottom_m,
            "infeed_grasp_pad_bottom_m": None,
            "grasp_offset_xy_m": (0.0, 0.0),
            "regrasp_offset_xy_m": (0.0, 0.0),
            "regrasp_surface_indent_m": None,
            "regrasp_surface_indent_fraction": None,
            "strain_limited_grip": False,
        }
    )


class _CartonView:
    """Expose the active rigid parcel to the shared motion controller, without altering physics."""

    def __init__(self, runtime: SortingStreamRuntime) -> None:
        object.__setattr__(self, "shared", runtime)

    def __getattr__(self, name: str) -> Any:
        aliases = {
            "vertices": "carton_vertices",
            "last_contact_counts": "carton_contacts",
            "last_active_contact_shapes": "carton_contact_shapes",
            "last_contact_distances_m": "carton_contact_distances",
        }
        return getattr(self.shared, aliases.get(name, name))

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.shared, name, value)


class _CartonScene:
    def __init__(self, scene: Any, outfeed_target: np.ndarray) -> None:
        self.shared = scene
        self.config = scene.config.model_copy(update={"outfeed_target_xy": outfeed_target.tolist()})
        self.triangle_indices = scene.rigid_triangles
        self.garment_count = len(scene.rigid_vertices)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.shared, name)

    def label_up_cos(self, vertices: np.ndarray) -> float:
        triangles = self.shared.rigid_triangles
        rest = self.shared.rigid_vertices[triangles]
        normals = np.cross(rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0])
        faces = vertices[triangles[normals[:, 2] > 1e-8]]
        normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]).sum(0)
        return float(normal[2] / max(np.linalg.norm(normal), 1e-12))


class _RigidRelay(SortingPlanner):
    """Use the same native-arm path with rigid contact measurements and rigid acceptance."""

    def initialize(self, robot: Any, scene: Any, runtime: SortingStreamRuntime) -> None:
        target = np.asarray(scene.config.outfeed_target_xy, float).copy()
        self.predicted_ground_lane_clearance_m = None
        if self.config.separate_ground_drop_lanes:
            soft = runtime.vertices()
            low, high = scene.supports["outfeed"]
            half = scene.rigid_dimensions[1] / 2
            candidates = [
                low[1] + half + self.config.ground_lane_margin_m,
                high[1] - half - self.config.ground_lane_margin_m,
            ]
            if candidates[0] > candidates[1]:
                raise ValueError("No full-width rigid drop lane fits on the outfeed belt")
            scores = [
                max(float(soft[:, 1].min()) - (y + half), (y - half) - float(soft[:, 1].max()))
                for y in candidates
            ]
            selected = int(np.argmax(scores))
            target[1] = candidates[selected]
            self.predicted_ground_lane_clearance_m = scores[selected]
            if selected == 0 and self.config.lower_drop_tcp_orientation_wxyz is not None:
                self.config = self.config.model_copy(
                    update={"tcp_orientation_wxyz": self.config.lower_drop_tcp_orientation_wxyz}
                )
        orientation_changed = not np.allclose(
            runtime.config.tcp_orientation_wxyz, self.config.tcp_orientation_wxyz
        )
        runtime.config.tcp_orientation_wxyz = self.config.tcp_orientation_wxyz
        if orientation_changed:
            runtime._calibrate_pad_bottom()
        self.outfeed_target = target
        self.robot, self.scene, self.runtime = (
            robot,
            _CartonScene(scene, target),
            _CartonView(runtime),
        )
        self.elapsed, self.rows = 0.0, []
        self.state.update(ready=True, done=False)
        self._initialize_tracking()
        triangles = scene.rigid_triangles
        edges = np.concatenate([triangles[:, :2], triangles[:, 1:], triangles[:, [2, 0]]])
        self._edge_indices = np.unique(np.sort(edges, axis=1), axis=0)
        self._edge_lengths = np.linalg.norm(
            self.rest[self._edge_indices[:, 0]] - self.rest[self._edge_indices[:, 1]], axis=1
        )

    def _coms(self, vertices: np.ndarray) -> np.ndarray:
        del vertices
        center = self.runtime.body_poses()[self.runtime.carton_body, :3]
        # Shared path code tracks two height channels for the soft parcel. Both
        # channels read the one measured rigid COM here; no duplicate body is built.
        return np.tile(center, (2, 1))

    def _measure_held_shape(self, vertices: np.ndarray, side: str) -> None:
        # Rigidity is measured from exact edge lengths every physics frame.
        pass

    def result(self) -> dict[str, Any]:
        checks = dict(
            sequence_completed=bool(self.state.get("done")),
            finite_state=bool(np.isfinite(self.samples).all()),
            rigid_geometry_preserved=abs(self.max_stretch - 1.0) < 1e-4,
            support_penetration_bounded=self.max_penetration
            <= self.config.maximum_table_penetration_m,
            finger_geometry_penetration_bounded=self.max_finger_geometry_penetration
            <= self.config.maximum_table_penetration_m,
            native_contact_penetration_bounded=self.runtime.maximum_carton_contact_penetration_m
            <= self.config.maximum_table_penetration_m,
            ik_targets_reachable=self.max_tcp_error < self.config.maximum_tcp_error_m,
            mimic_constraints_maintained=self.max_mimic_error < 0.03,
            conveyor_transport_observed=self.belt_displacement
            >= self.config.minimum_belt_displacement_m,
            parcel_exited_outfeed=self.belt_exited,
        )
        for side, leg in self.legs.items():
            checks.update(
                {
                    f"{side}_lifted": bool(np.all(leg["peak_lifts"] >= self.config.minimum_lift_m)),
                    f"{side}_bilateral_transfer": leg["transfer_frames"] > 0
                    and leg["contact_frames"] / max(leg["transfer_frames"], 1) >= 0.95,
                    f"{side}_released": bool(leg["released"]),
                    f"{side}_landed_on_destination": leg["footprint_supported"]
                    and leg["place_error"] <= self.config.maximum_place_error_m,
                }
            )
        checks = {key: bool(value) for key, value in checks.items()}
        return dict(
            success=all(checks.values()),
            checks=checks,
            failed_checks=[key for key, passed in checks.items() if not passed],
            failure_reason=self.failure_reason,
            belt_displacement_m=self.belt_displacement,
            maximum_contact_penetration_m=self.runtime.maximum_carton_contact_penetration_m,
            maximum_finger_geometry_penetration_m=self.max_finger_geometry_penetration,
            maximum_ik_target_error_m=self.max_tcp_error,
            actual_mass_kg=float(self.runtime.model.body_mass.numpy()[self.runtime.carton_body]),
            dimensions_m=self.scene.rigid_dimensions.tolist(),
            asset_sha256=hashlib.sha256(
                Path(self.scene._asset(self.scene.config.rigid_usd_path)).read_bytes()
            ).hexdigest(),
            rigid_scale=self.scene.config.rigid_scale,
            collision_geometry="measured_V10_carton_bounding_box",
            conveyor_model="contact_normal_limited_Coulomb_surface_velocity",
            native_solver_algorithm=self.config.rigid_mujoco_solver,
            native_warmstart_disabled=self.config.rigid_disable_warmstart,
            native_contact_time_constant_s=self.config.rigid_contact_time_constant_s,
            maximum_belt_force_n=self.runtime.maximum_belt_force_n,
            physics_duration_s=self.elapsed,
            attachments_used=False,
            outfeed_target_xy_m=self.outfeed_target.tolist(),
            predicted_ground_lane_clearance_m=self.predicted_ground_lane_clearance_m,
            tcp_orientation_wxyz=list(self.config.tcp_orientation_wxyz),
        )


class SortingStreamPlanner(SortingPlanner):
    """Run the successful soft relay, then spawn and relay one rigid parcel without reset."""

    config_cls = SortingStreamPlannerConfig
    runtime_cls = SortingStreamRuntime

    def begin_episode(self, *, register_stepper: bool = True) -> None:
        if not self.scene.config.rigid_usd_path or self.scene.config.infeed_travel_m <= 0:
            raise ValueError("The stream requires a V10 rigid asset and a positive infeed distance")
        if not self.config.continuous_outfeed:
            raise ValueError("The stream requires a continuously running outfeed belt")
        super().begin_episode(register_stepper=register_stepper)
        self.rigid: _RigidRelay | None = None
        self.soft_result: dict | None = None
        self.stream_samples: list[np.ndarray] = []
        self.stream_bodies: list[np.ndarray] = []
        self.stream_states: list[np.ndarray] = []
        self.stream_actions: list[np.ndarray] = []
        self.stream_rows: list[dict] = []
        self.carton_visibility: list[bool] = []
        self.belt_speeds: list[list[float]] = []
        self.belt_offsets: list[np.ndarray] = []
        self.soft_sleeping: list[bool] = []
        self.carton_spawn_time_s: float | None = None
        self.soft_native_frames = 0
        self.global_max_stretch = 1.0
        self.global_min_area = 1.0
        self.global_min_floor_z = float("inf")
        self.global_failure_reason = None

    @property
    def expected_duration_s(self) -> float:
        return 2 * super().expected_duration_s

    def on_physics_step(self, robot: Any) -> bool:
        time_s = len(self.stream_samples) * self.robot.sim_world.config.physics_dt
        vertices = self.runtime.vertices()
        self.stream_samples.append(vertices.astype(np.float32))
        lengths = np.linalg.norm(
            vertices[self._edge_indices[:, 0]] - vertices[self._edge_indices[:, 1]], axis=1
        )
        self.global_max_stretch = max(
            self.global_max_stretch, float((lengths / self._edge_lengths).max())
        )
        faces = vertices[self.scene.triangle_indices.reshape(-1, 3)]
        areas = np.linalg.norm(
            np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1
        )
        self.global_min_area = min(self.global_min_area, float((areas / self.rest_areas).min()))
        self.global_min_floor_z = min(self.global_min_floor_z, float(vertices[:, 2].min()))
        self.stream_bodies.append(self.runtime.body_poses().astype(np.float32))
        state, action = self.runtime.recording_coordinates()
        self.stream_states.append(state.copy())
        self.carton_visibility.append(self.runtime.carton_spawned)
        self.belt_offsets.append(self.runtime.belt_offsets.copy())
        self.soft_sleeping.append(self.runtime.soft_asleep)
        if self.rigid is None:
            completed = super().on_physics_step(robot)
            if self.rows:
                self.stream_rows.append(dict(self.rows[-1], parcel="soft", time_s=time_s))
            if completed and self.failure_reason is None and self.state.get("done"):
                self.soft_result = super().result()
                if self.soft_result["success"]:
                    self.soft_native_frames = len(self.runtime.native_mimic_residuals)
                    self.runtime.spawn_carton()
                    self.carton_spawn_time_s = time_s
                    config = rigid_relay_config(self.config)
                    self.rigid = _RigidRelay(config=config)
                    self.rigid.initialize(self.robot, self.scene, self.runtime)
                    self.state.update(done=False, phase="rigid_spawn")
                    completed = False
        else:
            completed = self.rigid.on_physics_step(robot)
            self.state.update(self.rigid.state)
            self.state["phase"] = "rigid_" + self.state.get("phase", "unknown")
            self.elapsed = time_s + (0 if completed else self.robot.sim_world.config.physics_dt)
            if self.rigid.rows:
                self.stream_rows.append(dict(self.rigid.rows[-1], parcel="rigid", time_s=time_s))
        _, action = self.runtime.recording_coordinates()
        self.stream_actions.append(action.copy())
        self.belt_speeds.append([self.runtime.infeed_speed, self.runtime.belt_speed])
        if (
            self.global_max_stretch >= self.config.maximum_edge_stretch
            or self.global_min_area <= self.config.minimum_triangle_area_ratio
            or self.global_min_floor_z
            < self.scene.config.floor_z_m - self.config.maximum_table_penetration_m
        ):
            self.global_failure_reason = (
                "Soft geometry or floor contact exceeded limits in the continuous stream"
            )
            self.state.update(done=False, phase="failed")
            completed = True
        return completed

    def write_diagnostics(self, directory: Path) -> None:
        native = self.runtime.native_mimic_residuals
        try:
            if self.soft_native_frames:
                self.runtime.native_mimic_residuals = native[: self.soft_native_frames]
            super().write_diagnostics(directory)
        finally:
            self.runtime.native_mimic_residuals = native
        np.savez_compressed(
            directory / "stream.npz",
            positions=np.asarray(self.stream_samples),
            body_poses=np.asarray(self.stream_bodies),
            times=np.arange(len(self.stream_samples)) * self.robot.sim_world.config.physics_dt,
            state=np.asarray(self.stream_states),
            action=np.asarray(self.stream_actions),
            carton_visible=np.asarray(self.carton_visibility),
            belt_speeds_mps=np.asarray(self.belt_speeds),
            belt_offsets_m=np.asarray(self.belt_offsets),
            soft_sleeping=np.asarray(self.soft_sleeping),
            native_mimic_residuals=(np.concatenate(native) if native else np.empty((0, 10))),
            carton_body=self.runtime.carton_body,
            carton_rest_vertices=self.scene.rigid_vertices,
            carton_triangles=self.scene.rigid_triangles,
        )
        (directory / "stream_trajectory.json").write_text(json.dumps(self.stream_rows) + "\n")
        if self.rigid is not None:
            (directory / "rigid_result.json").write_text(
                json.dumps(self.rigid.result(), indent=2) + "\n"
            )

    def result(self) -> dict[str, Any]:
        soft = self.soft_result if self.soft_result is not None else super().result()
        rigid = self.rigid.result() if self.rigid is not None else {"success": False}
        checks = dict(soft_parcel=soft["success"], rigid_parcel=rigid["success"])
        checks["outfeed_always_running"] = bool(
            self.belt_speeds
            and np.all(np.asarray(self.belt_speeds)[:, 1] == self.config.belt_speed_mps)
        )
        checks["stream_arrays_synchronized"] = (
            len(self.stream_samples)
            == len(self.stream_bodies)
            == len(self.stream_states)
            == len(self.stream_actions)
        )
        checks["finite_entire_stream"] = bool(
            np.isfinite(self.stream_samples).all() and np.isfinite(self.stream_bodies).all()
        )
        checks["soft_entire_stream_stretch_bounded"] = (
            self.global_max_stretch < self.config.maximum_edge_stretch
        )
        checks["soft_entire_stream_triangles_valid"] = (
            self.global_min_area > self.config.minimum_triangle_area_ratio
        )
        checks["soft_entire_stream_floor_clear"] = (
            self.global_min_floor_z
            >= self.scene.config.floor_z_m - self.config.maximum_table_penetration_m
        )
        capacity = self.runtime.cloth_solver.maximum_contacts.numpy()
        checks["no_contact_buffer_overflow"] = bool(
            max(capacity[:2]) < self.config.self_contact_buffer and capacity[2] < 16384
        )
        return dict(
            soft,
            success=all(checks.values()),
            workflow_success=all(checks.values()),
            strict_quality_success=bool(soft["strict_quality_success"] and all(checks.values())),
            checks=checks,
            failed_checks=[key for key, passed in checks.items() if not passed],
            parcels={"soft": soft, "rigid": rigid},
            sequence=["soft", "rigid"],
            carton_spawn_time_s=self.carton_spawn_time_s,
            continuous_world=True,
            physics_duration_s=self.elapsed,
            failure_reason=self.global_failure_reason
            or (self.rigid.failure_reason if self.rigid else self.failure_reason),
            maximum_edge_stretch=self.global_max_stretch,
            minimum_triangle_area_ratio=self.global_min_area,
            maximum_floor_penetration_m=max(
                0.0, self.scene.config.floor_z_m - self.global_min_floor_z
            ),
        )
