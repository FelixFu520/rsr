"""物理求解器验证任务的全部 Planner：衣服 / 刚体方块 / V7 包裹抓放与 Newton 运行时。

本文件按依赖顺序包含：

1. ``pickplace_cloth``：``PickPlaceClothPlanner`` 与 ``NewtonClothRuntime``（Featherstone /
   MuJoCo / XPBD + VBD 衣服，hybrid 下 PhysX 手指代理驱动 VBD；PhysX 刚体方块也用此 Planner）。
2. 官方 Newton cloth_franka 关键帧 ``OFFICIAL_KEYFRAMES``。
3. ``newton_cloth_franka``：``OfficialClothPlanner`` / ``OfficialClothRuntime``，官方关键帧 +
   雅可比伪逆控制。
4. ``pickplace_package``：``PackagePlanner`` / ``PackageRuntime``，V7 袋膜与衣物双网格抓放；
   分拣任务的 ``SortingRuntime`` 继承 ``PackageRuntime``。

调用流程：Job 绑定 scene/robot → ``begin_episode`` 创建 runtime → ``on_physics_step`` 生成
阶段目标 → ``runtime.command`` 求 IK/设置关节目标 → ``IsaacsimWorld.step`` 调用 ``runtime.step``
推进物理；Job 负责同步录制状态、动作和图像。Newton/Warp 只在创建运行时时导入。
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.planner.base import BasePlanner
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.physic_solver_utils import (
    KinematicVBDStepper,
    check_package_containment,
    moving_frame_vbd_class,
)

# ================================================================================================
# 衣服/刚体抓放流程与 Newton 衣服运行时。
#
# 本节保存任务配置、关节控制 kernel、URDF IK、NewtonClothRuntime 和 Planner。
#
# 当前没有使用 cuRobo：Newton 机器人使用本节的 URDF 雅可比 IK，PhysX 机器人调用
# Robot 中的 Lula IK；TCP 轨迹由预设动作阶段插值生成，不执行障碍物避让路径搜索。
# Newton 模型内部使用厘米，任务接口与输出数据使用米/rad。Newton/Warp 仅在创建
# 运行时或控制 kernel 时导入，读取配置与列举后端不会启动仿真或加载 GPU 依赖。
# ================================================================================================


class PickPlaceClothPlannerConfig(BaseConfig):
    """Motion uses metres/seconds; material values follow Newton's centimetre model."""

    timeout_s: float = Field(default=1200.0, gt=0)
    settle_s: float = Field(default=1.0, gt=0)
    approach_s: float = Field(default=3.0, gt=0)
    descend_s: float = Field(default=2.0, gt=0)
    close_s: float = Field(default=2.0, gt=0)
    lift_s: float = Field(default=3.0, gt=0)
    hold_s: float = Field(default=2.0, gt=0)
    transfer_s: float = Field(default=2.5, gt=0)
    lower_s: float = Field(default=1.5, gt=0)
    place_offset_m: tuple[float, float, float] = (-0.05, -0.12, 0.0)
    place_height_m: float = Field(default=0.08, gt=0)
    maximum_place_error_m: float = Field(default=0.08, gt=0)
    release_s: float = Field(default=1.0, gt=0)
    retract_s: float = Field(default=1.5, gt=0)
    retract_height_m: float = Field(default=0.12, gt=0)
    rest_s: float = Field(default=2.0, gt=0)
    approach_height_m: float = Field(default=0.18, gt=0)
    lift_height_m: float = Field(default=0.25, gt=0)
    grasp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tcp_orientation_wxyz: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    open_finger_m: float = Field(default=0.04, ge=0, le=0.04)
    closed_finger_m: float = Field(default=0.0, ge=0, le=0.04)
    open_gripper_control: bool = False
    minimum_lift_m: float = Field(default=0.08, gt=0)
    minimum_release_drop_m: float = Field(default=0.04, ge=0)
    maximum_patch_distance_m: float = Field(default=0.08, gt=0)
    maximum_table_penetration_m: float = Field(default=0.02, ge=0)
    grasp_patch_vertices: int = Field(default=20, ge=1)
    cloth_solver: Literal["vbd", "xpbd"] = "vbd"
    ik_iterations: int = Field(default=24, ge=1)
    particle_radius_m: float = Field(
        default=0.008,
        gt=0,
        description="Particle contact envelope radius in metres; does not thicken the visual mesh",
    )
    cloth_mass_kg: float = Field(default=0.2, gt=0)
    cloth_density_cm: float | None = Field(default=None, gt=0)
    cloth_stretch_stiffness: float = Field(default=1e4, gt=0)
    cloth_stretch_damping: float = Field(default=1.5e-2, ge=0)
    cloth_bend_stiffness: float = Field(default=5.0, gt=0)
    cloth_bend_damping: float = Field(default=0.5, ge=0)
    contact_stiffness: float = Field(default=5e4, gt=0)
    contact_damping: float = Field(default=50.0, ge=0)
    cloth_friction: float = Field(default=0.25, ge=0)
    table_friction: float = Field(default=0.4, ge=0)
    friction_epsilon_cm_s: float = Field(default=0.1, gt=0)
    xpbd_spring_stiffness: float = Field(default=1000.0, gt=0)
    xpbd_spring_damping: float = Field(default=1.0, ge=0)
    xpbd_mass_lumping: Literal["area", "uniform"] = "area"
    xpbd_self_contact: bool = Field(
        default=False,
        description="Diagnostic raw particle repulsion; no cloth adjacency exclusion in XPBD",
    )


@cache
def _get_track_joint_velocity_kernel() -> Any:
    """首次需要时创建 Warp kernel；复用缓存，避免重置 episode 时重复注册。

    Featherstone 路径用它限制机械臂和手指的跟踪速度；它不负责运动路径搜索。
    局部导入使本模块能在尚未启用 Newton/Warp 扩展时读取配置。
    """
    import warp as wp

    @wp.kernel
    def track_joint_velocity(
        target: wp.array[float],
        position: wp.array[float],
        velocity: wp.array[float],
    ):
        """Limit arm rates to rad/s and finger rates to centimetres per second."""
        index = wp.tid()
        limit = 2.0
        if index >= 7:
            limit = 8.0
        velocity[index] = wp.clamp(5.0 * (target[index] - position[index]), -limit, limit)

    return track_joint_velocity


def _rotation_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class _URDFHandIK:
    """Small geometric-Jacobian IK using the same official URDF as the solver.

    Keeping this CPU controller independent of Warp's IK kernels also avoids
    relying on an articulation-only IK model containing the cloth particles.
    """

    def __init__(self, robot: Any) -> None:
        root = ET.parse(resolve_usd_path(robot.config.urdf_path)).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, child = [], "fr3_hand"
        while child in by_child:
            joint = by_child[child]
            chain.append(joint)
            child = joint.find("parent").get("link")
        self.chain = list(reversed(chain))
        self.base = np.eye(4)
        self.base[:3, :3] = _rotation_wxyz(np.asarray(robot.config.base_orientation))
        self.base[:3, 3] = robot.config.base_position
        self.lower, self.upper = [], []
        for joint in self.chain:
            if joint.get("type") == "revolute":
                limit = joint.find("limit")
                self.lower.append(float(limit.get("lower")))
                self.upper.append(float(limit.get("upper")))
        self.q = np.asarray(robot.config.initial_joint_positions[:7], dtype=float)

    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transform = self.base.copy()
        origins, axes = [], []
        index = 0
        for joint in self.chain:
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            sr, sp, sy = np.sin([roll, pitch, yaw])
            cr, cp, cy = np.cos([roll, pitch, yaw])
            fixed = np.eye(4)
            fixed[:3, 3] = xyz
            fixed[:3, :3] = np.array(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr],
                ]
            )
            transform = transform @ fixed
            if joint.get("type") == "revolute":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                origins.append(transform[:3, 3].copy())
                axes.append(transform[:3, :3] @ axis)
                cross = np.array(
                    [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
                )
                rotation = np.eye(4)
                rotation[:3, :3] += np.sin(q[index]) * cross + (1 - np.cos(q[index])) * (
                    cross @ cross
                )
                transform = transform @ rotation
                index += 1
        tcp = transform[:3, 3] + transform[:3, :3] @ np.array([0.0, 0.0, 0.1034])
        jacobian = np.zeros((6, 7))
        for index, (origin, axis) in enumerate(zip(origins, axes, strict=True)):
            jacobian[:3, index] = np.cross(axis, tcp - origin)
            jacobian[3:, index] = axis
        pose = transform.copy()
        pose[:3, 3] = tcp
        return pose, jacobian

    def solve(self, position: np.ndarray, quaternion: np.ndarray, iterations: int) -> np.ndarray:
        rotation = _rotation_wxyz(quaternion)
        for _ in range(iterations):
            pose, jacobian = self._fk(self.q)
            angular = 0.5 * sum(
                (np.cross(pose[:3, i], rotation[:, i]) for i in range(3)), start=np.zeros(3)
            )
            error = np.concatenate((position - pose[:3, 3], angular))
            if np.linalg.norm(error) < 1e-5:
                break
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * 1e-4, error)
            self.q = np.clip(self.q + np.clip(step, -0.15, 0.15), self.lower, self.upper)
        return self.q.copy()


class NewtonClothRuntime:
    """构建 Newton 模型，推进机器人/衣服子步并同步 USD 显示与实测状态。

    Newton direct 使用独立机器人求解器配合 VBD 衣服，或使用统一 XPBD 模型。
    hybrid 将 PhysX 实测的手指/手掌位姿作为 Newton 碰撞代理输入 VBD。
    VBD 外部刚体模式是刚体驱动衣服的单向耦合；衣服没有固定顶点或抓取附件。
    """

    scale = 100.0
    color_bending = False

    def __init__(self, robot: Any, scene: Any, config: Any) -> None:
        import newton
        import warp as wp
        from newton.solvers import SolverFeatherstone, SolverMuJoCo, SolverXPBD

        self.newton, self.wp = newton, wp
        self.robot, self.scene, self.config = robot, scene, config
        self.pure_newton = robot.pure_newton
        self.solver_name = str(robot.sim_world.config.newton_solver).lower()
        if not self.pure_newton and self.solver_name != "vbd":
            raise ValueError("The hybrid cloth driver requires newton_solver='vbd'")
        if self.solver_name == "semi_implicit":
            raise ValueError("The cloth grasp demo supports VBD, XPBD, MuJoCo, or Featherstone")
        xpbd = self.pure_newton and self.solver_name == "xpbd" and config.cloth_solver == "xpbd"
        if config.cloth_solver == "xpbd" and not xpbd:
            raise ValueError("cloth_solver='xpbd' requires pure Newton with newton_solver='xpbd'")
        self.scale = type(self).scale
        builder = newton.ModelBuilder(
            gravity=tuple(value * self.scale for value in robot.sim_world.config.gravity)
        )
        self._configure_builder(builder)
        if self.pure_newton and self.solver_name == "mujoco":
            SolverMuJoCo.register_custom_attributes(builder)
        self.proxy_body_ids: list[int] = []
        if self.pure_newton:
            robot.add_newton_robot(builder, scale=self.scale)
            self.robot_body_count = builder.body_count
            self.robot_shape_count = builder.shape_count
        else:
            # These are measured links of the running PhysX Panda. Each is
            # represented by its pad/palm collision box, never by a cloth pin.
            transforms = robot.collider_transforms()
            # Official Panda rubber tips occupy local +Y on the left and -Y
            # on the right (the USD mirrors its geometry, not its body frame).
            # Their inner faces lie at the measured prismatic joint opening.
            offsets = ([0.0, 0.00758, 0.04525], [0.0, -0.00758, 0.04525], [0.0, 0.0, 0.025])
            extents = ([0.00875, 0.0076, 0.00925], [0.00875, 0.0076, 0.00925], [0.03, 0.045, 0.025])
            for index, (transform, offset, extent) in enumerate(
                zip(transforms, offsets, extents, strict=True)
            ):
                body = builder.add_body(
                    xform=wp.transform(transform[:3] * self.scale, transform[3:]),
                    is_kinematic=True,
                    label=f"physx_finger_proxy_{index}",
                )
                self.proxy_body_ids.append(body)
                builder.add_shape_box(
                    body,
                    xform=wp.transform(np.asarray(offset) * self.scale, wp.quat_identity()),
                    hx=extent[0] * self.scale,
                    hy=extent[1] * self.scale,
                    hz=extent[2] * self.scale,
                )
            self.robot_body_count = 0
            self.robot_shape_count = 0
        self._add_table(builder)
        self._add_deformables(builder, xpbd=xpbd)
        if xpbd and config.xpbd_mass_lumping == "uniform":
            # Preserve total mass while improving XPBD's unweighted spring solve.
            total_mass = float(sum(builder.particle_mass))
            mass_per_vertex = total_mass / builder.particle_count
            builder.particle_mass[:] = [mass_per_vertex] * builder.particle_count
        builder.color(include_bending=self.color_bending)
        self.model = self._finalize_model(builder)
        if xpbd and not config.xpbd_self_contact:
            # This kernel has no cloth-adjacency exclusion. Preserve the
            # particle radii for shape contacts while disabling its all-pairs
            # repulsion between neighboring garment vertices.
            self.model.particle_grid = None
            self.model.particle_max_radius = 0.0
        self.model.soft_contact_ke = config.contact_stiffness
        self.model.soft_contact_kd = config.contact_damping
        self.model.soft_contact_mu = config.cloth_friction
        bodies = self.model.shape_body.numpy()
        self.model.shape_material_mu.assign(
            np.where(bodies >= 0, robot.config.finger_friction, config.table_friction)
        )
        self.model.shape_material_ke.fill_(config.contact_stiffness)
        self.model.shape_material_kd.fill_(config.contact_damping)
        self.pinned_vertex_count = int(np.count_nonzero(self.model.particle_inv_mass.numpy() == 0))
        if self.pinned_vertex_count:
            raise RuntimeError("The cloth grasp demonstration requires zero pinned vertices")
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.rigid_previous_body_q = None
        self.rigid_previous_body_qd = None
        if self.pure_newton and self.solver_name == "xpbd" and not xpbd:
            # Native XPBD uses its input body buffers as Jacobi scratch space.
            # VBD's external-rigid path needs the untouched previous pose to
            # evaluate the fingers' displacement and contact friction.
            self.rigid_previous_body_q = wp.empty_like(self.state_0.body_q)
            self.rigid_previous_body_qd = wp.empty_like(self.state_0.body_qd)
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.pipeline = self._create_collision_pipeline()
        self.contacts = self.pipeline.contacts()
        self.unified_xpbd = xpbd
        if self.unified_xpbd:
            self.cloth_solver = SolverXPBD(self.model, iterations=config.solver_iterations)
            self.rigid_solver = None
        else:
            self.cloth_solver = self._create_cloth_solver()
            if self.pure_newton and self.solver_name == "mujoco":
                self.rigid_solver = self._create_mujoco_solver()
            elif self.pure_newton and self.solver_name == "xpbd":
                self.rigid_solver = SolverXPBD(self.model, iterations=config.solver_iterations)
            elif self.pure_newton:
                self.rigid_solver = SolverFeatherstone(
                    self.model, update_mass_matrix_interval=config.newton_substeps
                )
            else:
                self.rigid_solver = None
        self.initial_q = self.model.joint_q.numpy().copy()
        self.target_q = self.initial_q.copy()
        self.tcp_target = np.asarray(scene.target_position).copy()
        self.finger_target = 0.04
        self.gravity_earth = wp.clone(self.model.gravity)
        self.gravity_zero = wp.zeros_like(self.model.gravity)
        self.sim_time = 0.0
        self._last_proxy = None
        self._visual_bodies: list[Any] = []
        if self.pure_newton:
            self._create_ik()
            self._create_robot_visuals()
        self.last_contacts = 0
        self.last_contact_counts = dict(left=0, right=0, palm=0, table=0)
        if self.pure_newton:
            self.contact_body_ids = self._contact_bodies()
        else:
            self.contact_body_ids = dict(
                zip(("left", "right", "palm"), self.proxy_body_ids, strict=True)
            )
        self.direct_graph: Any = None
        self.direct_warmed = False
        self.direct_use_graph = (
            self.pure_newton
            and config.newton_use_cuda_graph
            and self.model.device.is_cuda
            and config.newton_substeps % 2 == 0
        )
        if self.pure_newton and not self.unified_xpbd and self.solver_name != "xpbd":
            self.track_joint_velocity = _get_track_joint_velocity_kernel()
        self.kinematic_stepper = None
        if not self.pure_newton:
            self.kinematic_stepper = KinematicVBDStepper(
                self.model,
                self.cloth_solver,
                self.pipeline,
                self.contacts,
                self.state_0,
                self.state_1,
                self.control,
                config.newton_substeps,
                robot.sim_world.config.physics_dt / config.newton_substeps,
                use_graph=config.newton_use_cuda_graph,
            )

    def _contact_bodies(self) -> dict[str, int]:
        """Identify the native Franka fingers; other robot runtimes override this mapping."""
        labels = self.model.body_label
        return {
            "left": next(i for i, name in enumerate(labels) if name.endswith("fr3_leftfinger")),
            "right": next(i for i, name in enumerate(labels) if name.endswith("fr3_rightfinger")),
            "palm": self.hand_body,
        }

    def _create_mujoco_solver(self) -> Any:
        """Create the standard GPU MuJoCo solver; task runtimes may select native CPU."""
        from newton.solvers import SolverMuJoCo

        return SolverMuJoCo(self.model, solver="newton", integrator="implicitfast", iterations=30)

    def _add_table(self, builder: Any) -> None:
        """Add the measured support; conveyor runtimes may provide a moving surface."""
        low, high = np.asarray(self.scene.table_bounds, dtype=float)
        center, half = (low + high) * 0.5, (high - low) * 0.5
        builder.add_shape_box(
            -1,
            xform=self.wp.transform(center * self.scale, self.wp.quat_identity()),
            hx=half[0] * self.scale,
            hy=half[1] * self.scale,
            hz=half[2] * self.scale,
        )

    def _add_deformables(self, builder: Any, *, xpbd: bool) -> None:
        """Add the physical cloth; specialized assets may contain multiple meshes."""
        config, scene, wp = self.config, self.scene, self.wp
        vertices_cm = np.asarray(scene.cloth_vertices) * self.scale
        triangles = np.asarray(scene.triangle_indices, dtype=np.int32).reshape(-1, 3)
        triangle_points = vertices_cm[triangles]
        area_cm2 = float(
            np.linalg.norm(
                np.cross(
                    triangle_points[:, 1] - triangle_points[:, 0],
                    triangle_points[:, 2] - triangle_points[:, 0],
                ),
                axis=1,
            ).sum()
            * 0.5
        )
        if area_cm2 <= 0:
            raise ValueError("The cloth surface has no positive triangle area")
        density = config.cloth_density_cm or config.cloth_mass_kg / area_cm2
        builder.add_cloth_mesh(
            vertices=[wp.vec3(*v) for v in vertices_cm],
            indices=triangles.reshape(-1),
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            scale=1.0,
            density=density,
            tri_ke=config.cloth_stretch_stiffness,
            tri_ka=config.cloth_stretch_stiffness,
            tri_kd=config.cloth_stretch_damping,
            edge_ke=config.cloth_bend_stiffness,
            edge_kd=config.cloth_bend_damping,
            # SolverXPBD projects edge springs; it does not solve the FEM
            # triangle stretch energies consumed by SolverVBD.
            add_springs=xpbd,
            spring_ke=config.xpbd_spring_stiffness,
            spring_kd=config.xpbd_spring_damping,
            particle_radius=config.particle_radius_m * self.scale,
        )

    def _create_cloth_solver(self) -> Any:
        """Create VBD independently of the rigid articulation solver."""
        from newton.solvers import SolverVBD

        config = self.config
        return SolverVBD(
            self.model,
            iterations=config.solver_iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.2,
            particle_self_contact_gap=0.0,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=32,
            particle_collision_detection_interval=-1,
            friction_epsilon=config.friction_epsilon_cm_s,
        )

    def _configure_builder(self, builder: Any) -> None:
        """Allow specialized contact geometry before importing rigid assets."""

    def _finalize_model(self, builder: Any) -> Any:
        """Finalize the assembled model without changing the default coordinate frame."""
        return builder.finalize(device=self.robot.sim_world.config.device or "cuda:0")

    def _create_collision_pipeline(self) -> Any:
        return self.newton.CollisionPipeline(
            self.model, soft_contact_gap=self.config.particle_radius_m * self.scale
        )

    def _create_ik(self) -> None:
        hand_bodies = [
            i for i, label in enumerate(self.model.body_label) if label.endswith("fr3_hand")
        ]
        if hand_bodies:
            self.hand_body, self.tcp_offset_cm = hand_bodies[0], 10.34
        else:
            # Fixed hand/flange/tool frames collapse into link7. Its TCP is
            # 0.107 + 0.1034 m along Z, as authored by the same official URDF.
            self.hand_body = next(
                i for i, label in enumerate(self.model.body_label) if label.endswith("fr3_link7")
            )
            self.tcp_offset_cm = 21.04
        self.ik = _URDFHandIK(self.robot)

    def command(self, position: np.ndarray, quaternion_wxyz: np.ndarray, finger: float) -> None:
        """Change only robot control targets; garment state remains unconstrained."""
        self.tcp_target = np.asarray(position, dtype=float)
        self.finger_target = float(finger)
        if not self.pure_newton:
            self.robot.command_tcp(position, quaternion_wxyz, finger)
            return
        self.target_q[:7] = self.ik.solve(
            self.tcp_target, quaternion_wxyz, self.config.ik_iterations
        )
        self.target_q[-2:] = finger * self.scale
        self.control.joint_target_q.assign(self.target_q)

    def step(self, frame_dt: float) -> None:
        """Advance all Newton substeps once and publish the resulting visual geometry."""
        substeps = self.config.newton_substeps
        proxy = None if self.pure_newton else self.robot.collider_transforms().copy()
        if proxy is not None:
            proxy[:, :3] *= self.scale
            previous = proxy if self._last_proxy is None else self._last_proxy
            q_inputs, q_outputs, velocities = [], [], []
            for index in range(substeps):
                alpha_0, alpha_1 = index / substeps, (index + 1) / substeps
                q_in = previous * (1 - alpha_0) + proxy * alpha_0
                q_out = previous * (1 - alpha_1) + proxy * alpha_1
                q_in[:, 3:] /= np.linalg.norm(q_in[:, 3:], axis=1, keepdims=True)
                q_out[:, 3:] /= np.linalg.norm(q_out[:, 3:], axis=1, keepdims=True)
                velocity = np.zeros((len(proxy), 6), dtype=np.float32)
                velocity[:, :3] = (proxy[:, :3] - previous[:, :3]) / frame_dt
                q_inputs.append(q_in)
                q_outputs.append(q_out)
                velocities.append(velocity)
            self.kinematic_stepper.step(
                np.asarray(q_inputs), np.asarray(q_outputs), np.asarray(velocities)
            )
            self.state_0 = self.kinematic_stepper.state_0
            self.state_1 = self.kinematic_stepper.state_1
            self._last_proxy = proxy
            self.sim_time += frame_dt
            self._publish()
            return
        if self.direct_use_graph and self.direct_warmed:
            if self.direct_graph is None:
                with self.wp.ScopedCapture(device=self.model.device) as capture:
                    self._advance_direct(frame_dt)
                self.direct_graph = capture.graph
            self.wp.capture_launch(self.direct_graph)
        else:
            self._advance_direct(frame_dt)
            self.direct_warmed = True
        self.sim_time += frame_dt
        self._publish()

    def _advance_direct(self, frame_dt: float) -> None:
        """Run a capture-compatible fixed number of shared-model substeps."""
        substeps = self.config.newton_substeps
        dt = frame_dt / substeps
        if hasattr(self.cloth_solver, "rebuild_bvh"):
            self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            if self.rigid_solver is not None:
                count = self.model.particle_count
                self.model.particle_count = 0
                try:
                    if self.solver_name != "mujoco":
                        # Gravity compensation for the mounted robot's driven
                        # trajectory; the freely simulated cloth keeps gravity.
                        self.model.gravity.assign(self.gravity_zero)
                    if self.solver_name not in {"mujoco", "xpbd"}:
                        self._apply_rigid_control()
                    if self.rigid_previous_body_q is not None:
                        self.wp.copy(self.rigid_previous_body_q, self.state_0.body_q)
                        self.wp.copy(self.rigid_previous_body_qd, self.state_0.body_qd)
                    self.rigid_solver.step(self.state_0, self.state_1, self.control, None, dt)
                    if self.rigid_previous_body_q is not None:
                        self.wp.copy(self.state_0.body_q, self.rigid_previous_body_q)
                        self.wp.copy(self.state_0.body_qd, self.rigid_previous_body_qd)
                finally:
                    self.model.particle_count = count
                    self.model.gravity.assign(self.gravity_earth)
                self.state_0.particle_f.zero_()
            self.pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _apply_rigid_control(self) -> None:
        self.wp.launch(
            self.track_joint_velocity,
            dim=self.model.joint_dof_count,
            inputs=[self.control.joint_target_q, self.state_0.joint_q],
            outputs=[self.state_0.joint_qd],
            device=self.model.device,
        )

    def _publish(self) -> None:
        """Publish and validate the latest actual state after one complete frame."""
        vertices = self.vertices()
        if not np.isfinite(vertices).all():
            raise FloatingPointError("Newton cloth generated non-finite vertex positions")
        self.scene.publish_cloth_vertices(vertices)
        if self.pure_newton:
            self._publish_robot_visuals()
        if hasattr(self.contacts, "soft_contact_count"):
            self.last_contacts = int(self.contacts.soft_contact_count.numpy()[0])
            shape_ids = self.contacts.soft_contact_shape.numpy()[: self.last_contacts]
            shape_ids = shape_ids[(shape_ids >= 0) & (shape_ids < self.model.shape_count)]
            bodies = self.model.shape_body.numpy()[shape_ids]
            self.last_contact_counts = {
                name: int(np.count_nonzero(bodies == body))
                for name, body in self.contact_body_ids.items()
            }
            self.last_contact_counts["table"] = int(np.count_nonzero(bodies < 0))

    def vertices(self) -> np.ndarray:
        """Return freely simulated garment positions in world metres."""
        return self.state_0.particle_q.numpy().astype(float) / self.scale

    def recording_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Read physical joints; XPBD maximal body coordinates require inverse FK projection.

        XPBD does not update state.joint_q. Use separate observation buffers so collecting
        measurements cannot change the solver state, then unwrap revolute coordinates near
        the commanded branch. Finger coordinates are converted from centimetres to metres.
        """
        if self.solver_name == "xpbd":
            if not hasattr(self, "_observation_q"):
                self._observation_q = self.wp.clone(self.state_0.joint_q)
                self._observation_qd = self.wp.clone(self.state_0.joint_qd)
            self.newton.eval_ik(self.model, self.state_0, self._observation_q, self._observation_qd)
            state = self._observation_q.numpy().astype(float)
            state[:7] += 2 * np.pi * np.round((self.target_q[:7] - state[:7]) / (2 * np.pi))
        else:
            state = self.state_0.joint_q.numpy().astype(float)
        action = self.target_q.copy()
        state[-2:] /= self.scale
        action[-2:] /= self.scale
        return state, action

    def finger_gap_m(self) -> float:
        """Read actual finger separation, including XPBD's body-based states."""
        if not self.pure_newton:
            return float(np.sum(self.robot.finger_positions()))
        transforms = self.state_0.body_q.numpy()
        left, right = (self.contact_body_ids[name] for name in ("left", "right"))
        return float(np.linalg.norm(transforms[left, :3] - transforms[right, :3]) / self.scale)

    def tcp_position(self) -> np.ndarray:
        """Read actual simulated hand pose in world metres."""
        if not self.pure_newton:
            return self.robot.tcp_position()
        hand = self.state_0.body_q.numpy()[self.hand_body]
        point = self.wp.transform_point(
            self.wp.transform(hand[:3], hand[3:]), self.wp.vec3(0.0, 0.0, self.tcp_offset_cm)
        )
        return np.asarray(point, dtype=float) / self.scale

    def _create_robot_visuals(self) -> None:
        from pxr import Gf, Tf, UsdGeom, Vt

        model = self.model
        stage = self.robot.sim_world.stage
        body_paths = [
            f"{self.robot.config.prim_path}/{Tf.MakeValidIdentifier(label.rsplit('/', 1)[-1])}"
            for label in model.body_label[: self.robot_body_count]
        ]
        if len(set(body_paths)) != len(body_paths):
            raise ValueError("Robot body labels must have unique USD link names")
        for path in body_paths:
            xform = UsdGeom.Xform.Define(stage, path)
            xform.ClearXformOpOrder()
            self._visual_bodies.append(xform.AddTransformOp())
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_flags = model.shape_flags.numpy()
        transforms, scales = model.shape_transform.numpy(), model.shape_scale.numpy()
        visible = int(self.newton.ShapeFlags.VISIBLE)
        for index, body in enumerate(shape_body):
            if index >= self.robot_shape_count or not shape_flags[index] & visible:
                continue
            source = model.shape_source[index]
            if source is None or int(shape_type[index]) not in (
                int(self.newton.GeoType.MESH),
                int(self.newton.GeoType.CONVEX_MESH),
            ):
                continue
            if body < 0:
                # The URDF's fixed base collapses into the world. Its visual
                # transform already includes the table-mounted base pose.
                UsdGeom.Xform.Define(stage, f"{self.robot.config.prim_path}/fixed_base")
                path = f"{self.robot.config.prim_path}/fixed_base/shape_{index}"
            else:
                path = f"{body_paths[body]}/shape_{index}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(source.vertices, np.float32)))
            indices = np.asarray(source.indices, np.int32).reshape(-1)
            mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
            mesh.CreateFaceVertexIndicesAttr(indices.tolist())
            mesh.CreateDisplayColorAttr([Gf.Vec3f(0.88, 0.9, 0.92)])
            xf = UsdGeom.Xformable(mesh)
            tf = transforms[index]
            xf.AddTranslateOp().Set(Gf.Vec3d(*(tf[:3] / self.scale).tolist()))
            xf.AddOrientOp().Set(Gf.Quatf(float(tf[6]), *tf[3:6].tolist()))
            xf.AddScaleOp().Set(Gf.Vec3f(*(scales[index] / self.scale).tolist()))
        self._publish_robot_visuals()
        self.robot.initialize_cameras()

    def _publish_robot_visuals(self) -> None:
        from pxr import Gf

        for op, transform in zip(self._visual_bodies, self.state_0.body_q.numpy(), strict=False):
            matrix = Gf.Matrix4d(1.0)
            matrix.SetRotate(Gf.Quatd(float(transform[6]), *transform[3:6].astype(float).tolist()))
            matrix.SetTranslateOnly(Gf.Vec3d(*(transform[:3] / self.scale).astype(float).tolist()))
            op.Set(matrix)


class PickPlaceClothPlanner(BasePlanner):
    """Drive joints and evaluate lifted geometry without attaching the grasp target."""

    config_cls = PickPlaceClothPlannerConfig
    runtime_cls = NewtonClothRuntime

    def __init__(self, config: PickPlaceClothPlannerConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("planner")
        self.scene: Any = None
        self.runtime: Any = None
        self.elapsed = 0.0
        self.rows: list[dict[str, Any]] = []
        self._phases: list[tuple[str, float]] = []
        self._phase_index = -1
        self._phase_start_tcp: np.ndarray | None = None
        self._grasp_position: np.ndarray | None = None
        self._baseline_position: np.ndarray | None = None
        self._patch: np.ndarray | None = None
        self._baseline_vertices: np.ndarray | None = None
        self._edge_indices: np.ndarray | None = None
        self._edge_lengths: np.ndarray | None = None
        self._ik_failures = 0
        self.state["timeout_s"] = self.config.timeout_s

    def init_planner(self, robot: Any) -> None:
        """Bind the already arranged robot; defer handles until after world reset."""
        self.robot = robot

    def configure_scene(self, scene: Any) -> None:
        """Bind the independently registered scene's target and support geometry."""
        self.scene = scene

    def begin_episode(self, *, register_stepper: bool = True) -> None:
        """Create a fresh cloth runtime and install its single world-step callback."""
        if self.scene is None:
            raise RuntimeError("Call configure_scene before begin_episode")
        if self.scene.config.object_kind == "cloth":
            world_config = self.robot.sim_world.config
            runtime_config = SimpleNamespace(
                **{
                    **self.config.model_dump(),
                    "newton_substeps": world_config.newton_substeps,
                    "newton_use_cuda_graph": world_config.newton_use_cuda_graph,
                    "solver_iterations": world_config.newton_iterations,
                }
            )
            self.runtime = self.runtime_cls(self.robot, self.scene, runtime_config)
            if register_stepper:
                self.robot.sim_world.register_newton_stepper(self.runtime.step)
            triangles = np.asarray(self.scene.triangle_indices).reshape(-1, 3)
            edges = np.concatenate((triangles[:, :2], triangles[:, 1:], triangles[:, [2, 0]]))
            self._edge_indices = np.unique(np.sort(edges, axis=1), axis=0)
            vertices = self.runtime.vertices()
            self._edge_lengths = np.linalg.norm(
                vertices[self._edge_indices[:, 0]] - vertices[self._edge_indices[:, 1]], axis=1
            )
        self.elapsed = 0.0
        self.rows.clear()
        self._phase_index = -1
        self._phases = [
            ("settle", self.config.settle_s),
            ("approach", self.config.approach_s),
            ("descend", self.config.descend_s),
            ("close", self.config.close_s),
            ("lift", self.config.lift_s),
            ("hold", self.config.hold_s),
            ("transfer", self.config.transfer_s),
            ("lower", self.config.lower_s),
            ("release", self.config.release_s),
            ("retract", self.config.retract_s),
            ("rest", self.config.rest_s),
        ]
        self.state.update(ready=True, done=False, phase="settle", success=False)

    def _tcp_position(self) -> np.ndarray:
        return self.runtime.tcp_position() if self.runtime else self.robot.tcp_position()

    def _target_position(self) -> np.ndarray:
        if self.runtime is not None and self._patch is not None:
            return self.runtime.vertices()[self._patch].mean(axis=0)
        return np.asarray(self.scene.target_position, dtype=float)

    def _enter_phase(self, phase: str) -> None:
        self._phase_start_tcp = self._tcp_position().copy()
        if phase == "approach":
            self._grasp_position = np.asarray(self.scene.target_position).copy()
            self._grasp_position += np.asarray(self.config.grasp_offset_m)
            # A freely simulated shirt rests one particle radius above the
            # support. Grasp its measured settled position, never its spawn Z.
        if phase == "close":
            if self.runtime:
                vertices = self.runtime.vertices()
                self._patch = np.argsort(np.linalg.norm(vertices - self._grasp_position, axis=1))[
                    : self.config.grasp_patch_vertices
                ]
                self._baseline_vertices = vertices.copy()
            self._baseline_position = self._target_position().copy()
        self.state["phase"] = phase

    def on_physics_step(self, robot: Any) -> bool:
        """Read the previous actual state, then command the next robot target."""
        del robot
        if self.state.get("done"):
            return True
        ends = np.cumsum([duration for _, duration in self._phases])
        index = int(np.searchsorted(ends, self.elapsed, side="right"))
        if index >= len(self._phases):
            self.state.update(done=True, phase="done")
            self.state["success"] = self.result()["success"]
            return True
        phase, duration = self._phases[index]
        start_time = 0.0 if index == 0 else float(ends[index - 1])
        if index != self._phase_index:
            self._phase_index = index
            self._enter_phase(phase)
        self._record(phase)
        fraction = min(1.0, max(0.0, (self.elapsed - start_time) / duration))
        fraction = fraction * fraction * (3 - 2 * fraction)
        finger = self.config.open_finger_m
        destination = self._phase_start_tcp.copy()
        if self._grasp_position is not None:
            destination = self._grasp_position.copy()
            if phase == "approach":
                destination[2] += self.config.approach_height_m
            elif phase in {"lift", "hold", "transfer"}:
                destination[2] += self.config.lift_height_m
                if phase == "transfer":
                    destination += np.asarray(self.config.place_offset_m)
            elif phase in {"lower", "release", "retract", "rest"}:
                destination += np.asarray(self.config.place_offset_m)
                destination[2] += self.config.place_height_m
                if phase in {"retract", "rest"}:
                    destination[2] += self.config.retract_height_m
        if phase in {"close", "lift", "hold", "transfer", "lower"}:
            finger = self.config.closed_finger_m
        if self.config.open_gripper_control:
            finger = self.config.open_finger_m
        if phase in {"settle", "close", "hold", "release", "rest"}:
            target = destination
        else:
            target = self._phase_start_tcp * (1 - fraction) + destination * fraction
        if phase == "settle":
            target = self._phase_start_tcp
        quaternion = np.asarray(self.config.tcp_orientation_wxyz, dtype=float)
        if self.runtime:
            self.runtime.command(target, quaternion, finger)
        elif not self.robot.command_tcp(target, quaternion, finger):
            self._ik_failures += 1
        self.state["last_action"] = dict(tcp=target.tolist(), finger_m=finger)
        self.elapsed += float(self.robot.sim_world.config.physics_dt)
        return False

    def _record(self, phase: str) -> None:
        point = self._target_position()
        tcp = self._tcp_position()
        row = dict(
            time_s=float(self.elapsed),
            phase=phase,
            target_xyz_m=point.tolist(),
            tcp_xyz_m=tcp.tolist(),
            target_z_m=float(point[2]),
            tcp_z_m=float(tcp[2]),
            patch_distance_m=float(np.linalg.norm(point - tcp)),
            actual_finger_gap_m=float(
                self.runtime.finger_gap_m()
                if self.runtime
                else np.sum(self.robot.finger_positions())
            ),
        )
        if self.runtime:
            vertices = self.runtime.vertices()
            row["cloth_min_z_m"] = float(vertices[:, 2].min())
            row["contacts"] = self.runtime.last_contacts
            row.update(
                {
                    f"contacts_{name}": count
                    for name, count in self.runtime.last_contact_counts.items()
                }
            )
            edges = self._edge_indices
            lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
            row["edge_stretch_p99"] = float(
                np.quantile(lengths / np.maximum(self._edge_lengths, 1e-9), 0.99)
            )
            if self._baseline_vertices is not None:
                # Remove the best rigid motion before measuring actual shape
                # change, so translation alone cannot satisfy deformation.
                a = self._baseline_vertices - self._baseline_vertices.mean(axis=0)
                b = vertices - vertices.mean(axis=0)
                u, _, vt = np.linalg.svd(a.T @ b)
                if np.linalg.det(u @ vt) < 0:
                    u[:, -1] *= -1
                row["shape_rms_m"] = float(
                    np.sqrt(np.mean(np.sum((a @ (u @ vt) - b) ** 2, axis=1)))
                )
        self.rows.append(row)

    def result(self) -> dict[str, Any]:
        """Return JSON-native measured checks; missing hold/release evidence fails."""
        held = [row for row in self.rows if row["phase"] == "hold"]
        rested = [row for row in self.rows if row["phase"] == "rest"]
        minimum_lift = 0.0
        release_drop = 0.0
        maximum_distance = 0.0
        if held and self._baseline_position is not None:
            minimum_lift = min(row["target_z_m"] for row in held) - float(
                self._baseline_position[2]
            )
            maximum_distance = max(row["patch_distance_m"] for row in held)
        if held and rested:
            release_drop = float(np.mean([row["target_z_m"] for row in held])) - float(
                np.mean([row["target_z_m"] for row in rested[len(rested) // 2 :]])
            )
        checks = {
            "hold_observed": bool(held),
            "target_lifted": minimum_lift >= self.config.minimum_lift_m,
            "target_stays_near_gripper": bool(held)
            and maximum_distance <= self.config.maximum_patch_distance_m,
            "target_falls_after_release": bool(rested)
            and release_drop >= self.config.minimum_release_drop_m,
            "target_supported_after_release": bool(rested)
            and all(
                row["target_z_m"]
                >= self.scene.table_top_z - self.config.maximum_table_penetration_m
                for row in rested[len(rested) // 2 :]
            ),
            "no_attachments": True,
        }
        place_error = float("inf")
        if rested and self._grasp_position is not None:
            actual_xy = np.mean(
                [row["target_xyz_m"][:2] for row in rested[len(rested) // 2 :]], axis=0
            )
            desired_xy = self._grasp_position[:2] + np.asarray(self.config.place_offset_m)[:2]
            place_error = float(np.linalg.norm(actual_xy - desired_xy))
        checks["target_reached_place"] = place_error <= self.config.maximum_place_error_m
        result = dict(
            success=all(checks.values()),
            checks=checks,
            minimum_hold_lift_m=float(minimum_lift),
            maximum_hold_patch_distance_m=float(maximum_distance),
            release_drop_m=float(release_drop),
            physics_duration_s=float(self.elapsed),
            attachments_used=False,
            ik_failed_frames=self._ik_failures,
            open_gripper_control=self.config.open_gripper_control,
            commanded_closed_finger_m=self.config.closed_finger_m,
            commanded_open_finger_m=self.config.open_finger_m,
            minimum_hold_finger_gap_m=min(
                (row["actual_finger_gap_m"] for row in held), default=0.0
            ),
            maximum_rest_finger_gap_m=max(
                (row["actual_finger_gap_m"] for row in rested), default=0.0
            ),
            samples=len(self.rows),
            place_error_m=place_error if np.isfinite(place_error) else None,
        )
        if self.runtime:
            checks["no_pinned_vertices"] = self.runtime.pinned_vertex_count == 0
            checks["cloth_deforms"] = (
                max((row.get("shape_rms_m", 0.0) for row in held), default=0.0) > 0.003
            )
            checks["cloth_stretch_bounded"] = (
                max((row.get("edge_stretch_p99", 0.0) for row in self.rows), default=0.0) < 1.5
            )
            result.update(
                pinned_vertex_count=self.runtime.pinned_vertex_count,
                particle_count=int(self.runtime.model.particle_count),
                cloth_total_mass_kg=float(self.runtime.model.particle_mass.numpy().sum()),
                newton_version=str(self.runtime.newton.__version__),
                warp_version=str(self.runtime.wp.__version__),
                physics_substeps=self.runtime.config.newton_substeps,
                solver_iterations=self.runtime.config.solver_iterations,
                cuda_graph_active=bool(
                    self.runtime.direct_graph is not None
                    or (
                        self.runtime.kinematic_stepper is not None
                        and self.runtime.kinematic_stepper.graph is not None
                    )
                ),
                solver=self.runtime.solver_name,
                rigid_solver=(
                    type(self.runtime.rigid_solver).__name__
                    if self.runtime.rigid_solver is not None
                    else None
                ),
                cloth_solver=type(self.runtime.cloth_solver).__name__,
                particle_radius_m=self.config.particle_radius_m,
                cloth_self_contact_enabled=not self.runtime.unified_xpbd,
                xpbd_particle_repulsion_enabled=(
                    self.config.xpbd_self_contact if self.runtime.unified_xpbd else None
                ),
                stretch_spring_count=int(self.runtime.model.spring_count),
                particle_mass_lumping=(
                    self.config.xpbd_mass_lumping if self.runtime.unified_xpbd else "area"
                ),
                coupling="unified" if self.runtime.unified_xpbd else "one_way_rigid_to_cloth",
                maximum_cloth_stretch_p99=max(
                    (row.get("edge_stretch_p99", 0.0) for row in self.rows), default=0.0
                ),
                maximum_hold_contacts={
                    name: max((row.get(f"contacts_{name}", 0) for row in held), default=0)
                    for name in ("left", "right", "palm", "table")
                },
            )
            result["success"] = all(checks.values())
        return result


# ================================================================================================
# Official cloth_franka keyframes from Newton 1.6.1 (seconds, cm, xyzw, activation).
#
# Source: newton/examples/cloth/example_cloth_franka.py, create_articulation().
# https://github.com/newton-physics/newton
# The trajectory is data, not a trained reinforcement-learning policy.
# ================================================================================================

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

OFFICIAL_KEYFRAMES = (
    (4, 31.0, -60.0, 40.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.8),
    (2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.8),
    (2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.1),
    (2, 26.0, -60.0, 26.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.1),
    (2, 12.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.1),
    (3, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.1),
    (1, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.8),
    (2, 15.0, -33.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.8),
    (3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.8),
    (3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.1),
    (2, 15.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.1),
    (3, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.1),
    (1, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, 0.8),
    (2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, 0.8),
    (2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, 0.8),
    (2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, 0.1),
    (2, -18.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, 0.1),
    (3, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, 0.1),
    (1, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, 0.8),
    (3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, 0.8),
    (3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, 0.1),
    (2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, 0.1),
    (3, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, 0.1),
    (2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, 0.8),
    (2, 0.0, -20.0, 30.0, 0.9239, -0.3827, 0.0, 0.0, 0.8),
    (2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, 0.8),
    (2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, 0.1),
    (2, 0.0, -20.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, 0.1),
    (1, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, 0.1),
    (1.5, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, 0.1),
    (1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, 0.1),
    (1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, 0.8),
    (2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, 0.8),
)


# ================================================================================================
# Adapt Newton 1.6.1 cloth_franka's keyframe/Jacobian controller to the RSR lifecycle.
#
# Reference: newton/examples/cloth/example_cloth_franka.py (Apache-2.0).
# Physics stays in NewtonClothRuntime; the Job owns stepping, recording and cleanup.
# ================================================================================================

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0


class OfficialClothPlannerConfig(PickPlaceClothPlannerConfig):
    """Official controller/material profile; the first seven keyframes perform one fold."""

    backend: str = "newton_cloth_franka"
    keyframe_limit: int = Field(default=7, ge=7, le=len(OFFICIAL_KEYFRAMES))
    final_settle_s: float = Field(default=2.0, gt=0)
    cloth_density_cm: float | None = Field(default=None, gt=0)
    contact_stiffness: float = Field(default=1e4, gt=0)
    contact_damping: float = Field(default=10.0, ge=0)
    shape_contact_stiffness: float = Field(default=5e4, gt=0)
    shape_contact_damping: float = Field(default=50.0, ge=0)
    shape_contact_friction: float = Field(default=1.5, ge=0)
    table_friction: float = Field(default=1.5, ge=0)
    friction_epsilon_cm_s: float = Field(default=0.01, gt=0)
    minimum_lift_m: float = Field(default=0.03, gt=0)
    minimum_release_drop_m: float = Field(default=0.01, ge=0)
    open_finger_m: float = Field(default=0.032, gt=0, le=0.04)
    closed_finger_m: float = Field(default=0.004, ge=0, le=0.04)
    full_surface_contacts: bool = False
    contact_sdf_resolution: int = Field(default=64, ge=16, le=256)
    keyframe_offset_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.0, 0.0, 0.0)
    finger_velocity_gain: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _finger_range(self) -> OfficialClothPlannerConfig:
        if self.closed_finger_m >= self.open_finger_m:
            raise ValueError("closed_finger_m must be smaller than open_finger_m")
        return self


def _soft_contact_geometry(
    particles: np.ndarray,
    radii: np.ndarray,
    body_q: np.ndarray,
    shape_body: np.ndarray,
    shape_margin: np.ndarray,
    shape_ids: np.ndarray,
    indices: np.ndarray,
    barycentric: np.ndarray,
    body_points: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bodies, raw signed distances and active masks, in Newton model units.

    Mirror VBD's particle/rigid contact geometry: collision candidates outside
    particle radius + shape margin exert no contact force and must not count.
    """
    bodies = shape_body[shape_ids]
    valid = indices >= 0
    safe = np.maximum(indices, 0)
    weights = np.where(valid, barycentric, 0.0)
    points = np.sum(particles[safe] * weights[:, :, None], axis=1)
    radius = np.max(np.where(valid, radii[safe], 0.0), axis=1)
    surface = body_points.copy()
    moving = bodies >= 0
    transforms = body_q[bodies[moving]]
    xyz, w = transforms[:, 3:6], transforms[:, 6:7]
    local = surface[moving]
    cross = 2 * np.cross(xyz, local)
    surface[moving] = local + w * cross + np.cross(xyz, cross) + transforms[:, :3]
    distance = np.sum(normals * (points - surface), axis=1)
    margin = shape_margin[shape_ids] if shape_margin.size else 0.0
    return bodies, distance, distance < radius + margin


@cache
def _control_kernels() -> tuple[Any, Any]:
    import warp as wp

    @wp.kernel
    def body_velocity(
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_com: wp.array[wp.vec3],
        body_id: int,
        offset: wp.vec3,
        output: wp.array[float],
    ):
        transform = body_q[body_id]
        radius = wp.transform_vector(transform, offset - body_com[body_id])
        velocity = body_qd[body_id]
        omega = wp.spatial_bottom(velocity)
        linear = wp.spatial_top(velocity) + wp.cross(omega, radius)
        output[0] = linear[0]
        output[1] = linear[1]
        output[2] = linear[2]
        output[3] = omega[0]
        output[4] = omega[1]
        output[5] = omega[2]

    @wp.kernel
    def end_effector_error(
        body_q: wp.array[wp.transform],
        body_id: int,
        offset: wp.transform,
        target: wp.transform,
        output: wp.array[wp.spatial_vector],
    ):
        actual = body_q[body_id] * offset
        position = wp.transform_get_translation(target) - wp.transform_get_translation(actual)
        rotation = wp.transform_get_rotation(target) * wp.quat_inverse(
            wp.transform_get_rotation(actual)
        )
        output[0] = wp.spatial_vector(
            position[0], position[1], position[2], rotation[0], rotation[1], rotation[2]
        )

    return body_velocity, end_effector_error


class OfficialClothRuntime(NewtonClothRuntime):
    """Use the official velocity controller and flat bending rest angles with shared stepping."""

    def __init__(self, robot: Any, scene: Any, config: Any) -> None:
        if not robot.pure_newton or robot.sim_world.config.newton_solver != "featherstone":
            raise ValueError("Official cloth_franka profile requires direct Newton Featherstone")
        if robot.config.control_mode != "velocity":
            raise ValueError("Official cloth_franka profile requires robot.control_mode=velocity")
        super().__init__(robot, scene, config)
        self.model.edge_rest_angle.zero_()
        self.model.shape_material_ke.fill_(config.shape_contact_stiffness)
        self.model.shape_material_kd.fill_(config.shape_contact_damping)
        self.model.shape_material_mu.assign(
            np.where(
                self.model.shape_body.numpy() < 0,
                config.table_friction,
                config.shape_contact_friction,
            )
        )

    def _configure_builder(self, builder: Any) -> None:
        if self.config.full_surface_contacts:
            builder.default_shape_cfg.configure_sdf(
                force_sdf=True, max_resolution=self.config.contact_sdf_resolution
            )

    def _create_collision_pipeline(self) -> Any:
        return self.newton.CollisionPipeline(
            self.model,
            soft_contact_gap=self.config.particle_radius_m * self.scale,
            enable_rigid_soft_full_surface_contact=self.config.full_surface_contacts,
        )

    def _publish(self) -> None:
        super()._publish()
        contacts, model = self.contacts, self.model
        count = min(self.last_contacts, contacts.soft_contact_shape.shape[0])
        if not count:
            self.last_active_contact_bodies = np.empty(0, dtype=int)
            self.last_active_contact_shapes = np.empty(0, dtype=int)
            self.last_contact_counts["robot_root"] = 0
            self.last_contact_distances_m = dict.fromkeys(self.contact_body_ids)
            self.last_contact_particle_ids = {
                name: np.empty(0, dtype=int) for name in ("left", "right")
            }
            return
        shape_ids = contacts.soft_contact_shape[:count].numpy()
        valid = (shape_ids >= 0) & (shape_ids < model.shape_count)
        indices = contacts.soft_contact_indices[:count].numpy()[valid]
        bodies, distances, active = _soft_contact_geometry(
            self.state_0.particle_q.numpy(),
            model.particle_radius.numpy(),
            self.state_0.body_q.numpy(),
            model.shape_body.numpy(),
            model.shape_margin.numpy(),
            shape_ids[valid],
            indices,
            contacts.soft_contact_barycentric[:count].numpy()[valid],
            contacts.soft_contact_body_pos[:count].numpy()[valid],
            contacts.soft_contact_normal[:count].numpy()[valid],
        )
        self.last_contact_counts = {
            name: int(np.count_nonzero(active & (bodies == body)))
            for name, body in self.contact_body_ids.items()
        }
        self.last_active_contact_bodies = bodies[active].copy()
        self.last_active_contact_shapes = shape_ids[valid][active].copy()
        root = (bodies < 0) & (shape_ids[valid] < getattr(self, "robot_shape_count", 0))
        self.last_contact_counts["robot_root"] = int(np.count_nonzero(active & root))
        self.last_contact_counts["table"] = int(np.count_nonzero(active & (bodies < 0) & ~root))
        self.last_contact_particle_ids = {}
        for name in ("left", "right"):
            ids = indices[active & (bodies == self.contact_body_ids[name])]
            self.last_contact_particle_ids[name] = np.unique(ids[ids >= 0])
        self.last_contact_distances_m = {
            name: float(distances[mask].min() / self.scale) if mask.any() else None
            for name, body in self.contact_body_ids.items()
            for mask in [bodies == body]
        }

    def _create_ik(self) -> None:
        super()._create_ik()
        # The reference targets a point 22 cm from the collapsed final arm link.
        self.tcp_offset_cm = 22.0
        wp = self.wp
        self._velocity_kernel, self._error_kernel = _control_kernels()
        self._jacobian_state = self.model.state(requires_grad=True)
        self._body_velocity = wp.empty(6, dtype=float, requires_grad=True)
        self._jacobian = wp.empty(6 * self.model.joint_dof_count, dtype=float)
        self._error = wp.empty(1, dtype=wp.spatial_vector)
        self._onehots = [wp.array(row, dtype=float) for row in np.eye(6, dtype=np.float32)]
        self._velocity_target = wp.zeros(self.model.joint_dof_count, dtype=float)
        self._posture = self.model.joint_q.numpy().copy()

    def track_keyframe(
        self, target_m: np.ndarray, quaternion_xyzw: np.ndarray, activation: float
    ) -> None:
        """Compute the reference pseudoinverse/nullspace velocity command once per physics frame."""
        wp = self.wp
        q, qd = self.state_0.joint_q, self.state_0.joint_qd
        q.requires_grad = qd.requires_grad = True
        offset = wp.vec3(0.0, 0.0, self.tcp_offset_cm)
        wp.launch(
            self._error_kernel,
            dim=1,
            inputs=[
                self.state_0.body_q,
                self.hand_body,
                wp.transform(offset, wp.quat_identity()),
                wp.transform(target_m * self.scale, quaternion_xyzw),
            ],
            outputs=[self._error],
            device=self.model.device,
        )
        tape = wp.Tape()
        with tape:
            self.newton.eval_fk(self.model, q, qd, self._jacobian_state)
            wp.launch(
                self._velocity_kernel,
                dim=1,
                inputs=[
                    self._jacobian_state.body_q,
                    self._jacobian_state.body_qd,
                    self.model.body_com,
                    self.hand_body,
                    offset,
                ],
                outputs=[self._body_velocity],
                device=self.model.device,
            )
        for row in range(6):
            tape.backward(grads={self._body_velocity: self._onehots[row]})
            wp.copy(
                self._jacobian[
                    row * self.model.joint_dof_count : (row + 1) * self.model.joint_dof_count
                ],
                qd.grad,
            )
            tape.zero()
        jacobian = self._jacobian.numpy().reshape(6, -1)
        inverse = np.linalg.pinv(jacobian)
        current = q.numpy()
        posture = self._posture.copy()
        posture[0] = current[0]
        velocity = inverse @ self._error.numpy()[0] + (
            np.eye(len(current)) - inverse @ jacobian
        ) @ (posture - current)
        # Reference activations 0.1/0.8 mean 4/32 mm per finger. Keep the
        # trajectory while allowing the physical opening to match cloth thickness.
        blend = np.clip((activation - 0.1) / 0.7, 0.0, 1.0)
        opening = self.config.closed_finger_m + blend * (
            self.config.open_finger_m - self.config.closed_finger_m
        )
        velocity[-2:] = self.config.finger_velocity_gain * (opening * self.scale - current[-2:])
        limits = self.model.joint_velocity_limit.numpy()[-2:]
        velocity[-2:] = np.clip(velocity[-2:], -limits, limits)
        if not np.isfinite(velocity).all():
            raise FloatingPointError("Official controller generated a non-finite velocity")
        self._velocity_target.assign(velocity)
        self.target_q = current + velocity * self.robot.sim_world.config.physics_dt

    def hold(self) -> None:
        """Stop joint motion after the final reference keyframe."""
        self._velocity_target.zero_()
        self.target_q = self.state_0.joint_q.numpy().copy()

    def recording_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Record actual positions and the commanded joint velocities in rad/s and m/s."""
        state, _ = super().recording_coordinates()
        velocity = self._velocity_target.numpy().astype(float).copy()
        velocity[-2:] /= self.scale
        return state, velocity

    def _apply_rigid_control(self) -> None:
        self.state_0.joint_qd.assign(self._velocity_target)


class OfficialClothPlanner(PickPlaceClothPlanner):
    """Execute the official folding sequence; report measured contact, lift and release."""

    config_cls = OfficialClothPlannerConfig
    runtime_cls = OfficialClothRuntime
    action_names = [f"fr3_joint{i}_velocity_rad_s" for i in range(1, 8)] + [
        "left_finger_velocity_m_s",
        "right_finger_velocity_m_s",
    ]

    @property
    def expected_duration_s(self) -> float:
        """Return the reference sequence and final settling duration in simulation seconds."""
        return sum(row[0] for row in OFFICIAL_KEYFRAMES[: self.config.keyframe_limit]) + (
            self.config.final_settle_s
        )

    def begin_episode(self, *, register_stepper: bool = True) -> None:
        """Build the shared runtime, then initialize the official reference schedule."""
        super().begin_episode(register_stepper=register_stepper)
        self._keyframes = np.asarray(OFFICIAL_KEYFRAMES[: self.config.keyframe_limit], dtype=float)
        self._ends = np.cumsum(self._keyframes[:, 0])
        self._reference_origin = self.scene.official_frame_translation
        self._initial_z: np.ndarray | None = None
        self._peak_lift = 0.0
        self._peak_contact_count = dict(left=0, right=0)
        self._bilateral_grasp = False
        self._release_initial_z: float | None = None
        self._release_particle_ids = np.empty(0, dtype=int)
        self._release_drop = 0.0
        self._previous_closed = False
        self._transfer_frames = 0
        self._transfer_contact_frames = 0

    def on_physics_step(self, robot: Any) -> bool:
        """Advance control time exactly once; physics integration belongs to sim_world.step()."""
        del robot
        index = int(np.searchsorted(self._ends, self.elapsed, side="right"))
        vertices = self.runtime.vertices()
        if self.elapsed >= self._ends[-1] + self.config.final_settle_s:
            self.runtime.hold()
            self.state.update(done=True, phase="done")
            return True
        if self._initial_z is None and self.elapsed >= self._ends[0]:
            self._initial_z = vertices[:, 2].copy()
        if self._initial_z is not None:
            # Robust lifted patch metric: average the twenty most lifted physical vertices.
            lifted = vertices[:, 2] - self._initial_z
            self._peak_lift = max(self._peak_lift, float(np.sort(lifted)[-20:].mean()))
        counts = self.runtime.last_contact_counts
        if self._ends[3] <= self.elapsed < self._ends[5]:
            self._transfer_frames += 1
            self._transfer_contact_frames += int(counts["left"] > 0 and counts["right"] > 0)
        for side in self._peak_contact_count:
            self._peak_contact_count[side] = max(self._peak_contact_count[side], counts[side])
        if index < len(self._keyframes):
            row = self._keyframes[index]
            if row[8] < 0.5 and counts["left"] > 0 and counts["right"] > 0:
                self._bilateral_grasp = True
            target = row[1:4] / 100.0 + self._reference_origin + self.config.keyframe_offset_m
            self.runtime.track_keyframe(target, row[4:8], row[8])
            phase = f"official_{index:02d}_{'close' if row[8] < 0.5 else 'open'}"
        else:
            self.runtime.hold()
            phase = "settle_after_release"
        closed = index < len(self._keyframes) and self._keyframes[index, 8] < 0.5
        if not closed and self._previous_closed:
            self._release_particle_ids = np.union1d(
                self.runtime.last_contact_particle_ids["left"],
                self.runtime.last_contact_particle_ids["right"],
            )
            if self._release_particle_ids.size:
                self._release_initial_z = float(vertices[self._release_particle_ids, 2].mean())
        if closed:
            self._release_initial_z = None
        elif self._release_initial_z is not None:
            # Follow the same patch that was held at opening, rather than selecting
            # different highest vertices from the rest of the shirt every frame.
            patch_z = float(vertices[self._release_particle_ids, 2].mean())
            self._release_drop = max(self._release_drop, self._release_initial_z - patch_z)
        self._previous_closed = bool(closed)
        self.state["phase"] = phase
        self.rows.append(
            {
                "phase": phase,
                "time_s": self.elapsed,
                "target_z_m": float(vertices[:, 2].max()),
                "contacts_left": counts["left"],
                "contacts_right": counts["right"],
            }
        )
        self.elapsed += self.robot.sim_world.config.physics_dt
        return False

    def result(self) -> dict[str, Any]:
        """Describe the measured demonstration without claiming a learned policy or full fold."""
        vertices = self.runtime.vertices()
        release_drop = self._release_drop
        transfer_contact_fraction = self._transfer_contact_frames / max(1, self._transfer_frames)
        checks = {
            "sequence_completed": bool(self.state.get("done")),
            "finite_cloth_state": bool(np.isfinite(vertices).all()),
            "bilateral_contact_observed": self._bilateral_grasp,
            "grasp_maintained_during_transfer": transfer_contact_fraction >= 0.95,
            "cloth_lifted": self._peak_lift >= self.config.minimum_lift_m,
            "cloth_dropped_after_release": release_drop >= self.config.minimum_release_drop_m,
            "no_pinned_vertices": self.runtime.pinned_vertex_count == 0,
            "no_attachments": True,
        }
        return {
            "success": all(checks.values()),
            "checks": checks,
            "strategy": "newton_cloth_franka_keyframes",
            "keyframes_executed": self.config.keyframe_limit,
            "full_official_sequence": self.config.keyframe_limit == len(OFFICIAL_KEYFRAMES),
            "maximum_patch_lift_m": self._peak_lift,
            "release_drop_m": release_drop,
            "release_measurement": "mean_height_of_last_grasped_contact_patch",
            "maximum_contacts": self._peak_contact_count,
            "transfer_bilateral_contact_fraction": transfer_contact_fraction,
            "contact_counts_kind": "inside_particle_radius_plus_shape_margin",
            "full_surface_contacts": self.config.full_surface_contacts,
            "closed_finger_m": self.config.closed_finger_m,
            "keyframe_offset_m": self.config.keyframe_offset_m,
            "particle_radius_m": self.config.particle_radius_m,
            "cloth_total_mass_kg": float(self.runtime.model.particle_mass.numpy().sum()),
            "physics_duration_s": self.elapsed,
            "attachments_used": False,
            "pinned_vertex_count": self.runtime.pinned_vertex_count,
            "coupling": "one_way_rigid_to_cloth",
        }


# ================================================================================================
# Contact-driven Franka grasp of the V7 bag and its independent garment.
# ================================================================================================


class PackagePlannerConfig(OfficialClothPlannerConfig):
    """Metre-based V7 materials are read from USD; these fields control the robot."""

    backend: str = "pickplace_package"
    full_surface_contacts: bool = True
    contact_stiffness: float = Field(default=20000.0, gt=0)
    contact_damping: float = Field(default=30.0, ge=0)
    shape_contact_stiffness: float = Field(default=20000.0, gt=0)
    shape_contact_damping: float = Field(default=30.0, ge=0)
    shape_contact_friction: float = Field(default=1.4, ge=0)
    table_friction: float = Field(default=0.5, ge=0)
    cloth_friction: float = Field(default=1.4, ge=0)
    open_finger_m: float = Field(default=0.04, gt=0, le=0.04)
    closed_finger_m: float = Field(default=0.00625, ge=0, le=0.04)
    finger_velocity_gain: float = Field(default=6.0, gt=0)
    grasp_height_m: float = Field(default=0.0704, gt=0)
    minimum_lift_m: float = Field(default=0.08, gt=0)
    maximum_table_penetration_m: float = Field(default=0.0015, ge=0)
    maximum_place_error_m: float = Field(default=0.08, gt=0)
    minimum_release_drop_m: float = Field(default=0.01, ge=0)
    lift_height_m: float = Field(default=0.18, gt=0)
    place_offset_m: tuple[float, float, float] = (0.0, -0.12, 0.0)
    place_height_m: float = Field(default=0.17, gt=0)
    self_contact_buffer: int = Field(default=512, ge=32)
    self_contact_gap_m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    friction_epsilon_m_s: float = Field(default=0.001, gt=0)
    arm_velocity_gain: float = Field(default=3.0, gt=0)
    approach_s: float = Field(default=2.0, gt=0)
    descend_s: float = Field(default=1.5, gt=0)
    close_s: float = Field(default=1.5, gt=0)
    lift_s: float = Field(default=2.0, gt=0)
    transfer_s: float = Field(default=1.5, gt=0)
    lower_s: float = Field(default=1.0, gt=0)
    release_s: float = Field(default=1.0, gt=0)
    retract_s: float = Field(default=1.0, gt=0)
    rest_s: float = Field(default=1.5, gt=0)
    align_place_to_com: bool = False


class PackageRuntime(OfficialClothRuntime):
    """Use Featherstone velocities or MuJoCo position drives with two-mesh VBD."""

    scale = 1.0
    color_bending = True

    def __init__(self, robot: Any, scene: Any, config: Any) -> None:
        solver = robot.sim_world.config.newton_solver
        mode = {"featherstone": "velocity", "mujoco": "position"}.get(solver)
        if not robot.pure_newton or mode is None:
            raise ValueError("Package requires direct Newton Featherstone or MuJoCo")
        if robot.config.control_mode != mode or config.cloth_solver != "vbd":
            raise ValueError(f"Package {solver} requires {mode} control and VBD")
        # Preserve authored bending rest angles; the shirt profile flattens them.
        NewtonClothRuntime.__init__(self, robot, scene, config)
        self.model.shape_material_ke.fill_(config.shape_contact_stiffness)
        self.model.shape_material_kd.fill_(config.shape_contact_damping)
        self.model.shape_material_mu.assign(
            np.where(
                self.model.shape_body.numpy() < 0,
                config.table_friction,
                config.shape_contact_friction,
            )
        )
        self.model.shape_margin.fill_(0.0002)

    def _create_ik(self) -> None:
        if self.robot.config.control_mode == "position":
            NewtonClothRuntime._create_ik(self)
            self.tcp_offset_cm = 0.2104  # Same TCP as the URDF geometric IK, in metres.
        else:
            super()._create_ik()
            self.tcp_offset_cm = 0.22  # In this runtime all Newton lengths are metres.

    def _create_robot_visuals(self) -> None:
        """Display any V7 jaw extensions using exactly their collision dimensions."""
        from pxr import Gf, UsdGeom

        super()._create_robot_visuals()
        model = self.model
        for shape in getattr(self.robot, "package_pad_shapes", []):
            body = int(model.shape_body.numpy()[shape])
            name = model.body_label[body].rsplit("/", 1)[-1]
            cube = UsdGeom.Cube.Define(
                self.robot.sim_world.stage, f"{self.robot.config.prim_path}/{name}/v7_pad"
            )
            cube.CreateSizeAttr(2.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.12, 0.12)])
            transform = model.shape_transform.numpy()[shape]
            cube.AddTranslateOp().Set(Gf.Vec3d(*(transform[:3] / self.scale).tolist()))
            cube.AddOrientOp().Set(Gf.Quatf(float(transform[6]), *transform[3:6].tolist()))
            cube.AddScaleOp().Set(
                Gf.Vec3f(*(model.shape_scale.numpy()[shape] / self.scale).tolist())
            )

    def track_keyframe(
        self, target_m: np.ndarray, quaternion_xyzw: np.ndarray, activation: float
    ) -> None:
        """Track the parcel path faster while respecting authored arm velocity limits."""
        if self.robot.config.control_mode == "position":
            blend = np.clip((activation - 0.1) / 0.7, 0.0, 1.0)
            opening = self.config.closed_finger_m + blend * (
                self.config.open_finger_m - self.config.closed_finger_m
            )
            self.command(target_m, quaternion_xyzw[[3, 0, 1, 2]], opening)
            return
        super().track_keyframe(target_m, quaternion_xyzw, activation)
        velocity = self._velocity_target.numpy()
        limits = self.model.joint_velocity_limit.numpy()[:7]
        velocity[:7] = np.clip(velocity[:7] * self.config.arm_velocity_gain, -limits, limits)
        self._velocity_target.assign(velocity)
        self.target_q = self.state_0.joint_q.numpy() + (
            velocity * self.robot.sim_world.config.physics_dt
        )

    def hold(self) -> None:
        """Hold the actual posture using the active controller's command type."""
        if self.robot.config.control_mode == "position":
            self.target_q = self.state_0.joint_q.numpy().copy()
            self.control.joint_target_q.assign(self.target_q)
        else:
            super().hold()

    def recording_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Record measured joints and either position or velocity commands."""
        if self.robot.config.control_mode == "position":
            return NewtonClothRuntime.recording_coordinates(self)
        return super().recording_coordinates()

    def _add_deformables(self, builder: Any, *, xpbd: bool) -> None:
        if xpbd:
            raise ValueError("V7 package materials require VBD")
        wp, scene = self.wp, self.scene
        p = scene.package_parameters
        for part in scene.package_parts:
            film = part["name"] == "Bag"
            start, end = part["start"], part["end"]
            stiffness = p["bag_stiffness"] if film else p["cloth_stiffness"]
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=scene.cloth_vertices[start:end].tolist(),
                indices=part["triangles"].reshape(-1).tolist(),
                density=1.0,
                tri_ke=stiffness,
                tri_ka=stiffness,
                tri_kd=p["bag_damping"] if film else p["cloth_damping"],
                edge_ke=0.00001 if film else p["bend_stiffness"],
                edge_kd=p["bag_bend_damping"] if film else p["cloth_bend_damping"],
                particle_radius=0.00035 if film else 0.0008,
            )
            builder.particle_mass[start:end] = part["masses"].tolist()
        for (i, j), weight in zip(scene.loft_springs, scene.loft_weights, strict=True):
            builder.add_spring(
                int(i), int(j), p["loft_stiffness"] * weight, p["loft_damping"] * weight, 0.0
            )

    def _create_cloth_solver(self) -> Any:
        return moving_frame_vbd_class()(
            self.model,
            iterations=self.config.solver_iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.0016,
            particle_self_contact_gap=(
                self.scene.package_parameters["self_gap"]
                if self.config.self_contact_gap_m is None
                else self.config.self_contact_gap_m
            ),
            particle_topological_contact_filter_threshold=2,
            particle_vertex_contact_buffer_size=self.config.self_contact_buffer,
            particle_edge_contact_buffer_size=self.config.self_contact_buffer,
            rigid_body_particle_contact_buffer_size=16384,
            friction_epsilon=self.config.friction_epsilon_m_s,
        )

    def _create_collision_pipeline(self) -> Any:
        return self.newton.CollisionPipeline(
            self.model,
            soft_contact_gap=0.004,
            soft_contact_max=16384,
            enable_rigid_soft_full_surface_contact=True,
            deterministic=True,
        )


class PackagePlanner(OfficialClothPlanner):
    """Move the physical parcel with finger contacts and verify both surfaces lift."""

    config_cls = PackagePlannerConfig
    runtime_cls = PackageRuntime

    @property
    def expected_duration_s(self) -> float:
        """Return the duration of the complete approach, grasp and placement."""
        return sum(
            getattr(self.config, name)
            for name in (
                "approach_s",
                "descend_s",
                "close_s",
                "lift_s",
                "transfer_s",
                "lower_s",
                "release_s",
                "retract_s",
                "rest_s",
            )
        )

    @property
    def action_names(self) -> list[str]:
        """Describe commands in the actual controller's units."""
        if self.robot.config.control_mode == "position":
            return [f"fr3_joint{i}_target_rad" for i in range(1, 8)] + [
                "left_finger_target_m",
                "right_finger_target_m",
            ]
        return OfficialClothPlanner.action_names

    def begin_episode(self, *, register_stepper: bool = True) -> None:
        """Build both physical meshes and the robot, then initialize the task schedule."""
        PickPlaceClothPlanner.begin_episode(self, register_stepper=register_stepper)
        center = np.array([*self.scene.config.target_xy, self.scene.table_top_z])
        center += self.config.grasp_offset_m
        grasp = center + [0, 0, self.config.grasp_height_m]
        raised = grasp + [0, 0, self.config.lift_height_m]
        destination = raised + self.config.place_offset_m
        lower = center + self.config.place_offset_m + [0, 0, self.config.place_height_m]
        self._schedule = [
            ("approach", self.config.approach_s, center + [0, 0, 0.20], False),
            ("descend", self.config.descend_s, grasp, False),
            ("close", self.config.close_s, grasp, True),
            ("lift", self.config.lift_s, raised, True),
            ("transfer", self.config.transfer_s, destination, True),
            ("lower", self.config.lower_s, lower, True),
            ("release", self.config.release_s, lower, False),
            ("retract", self.config.retract_s, lower + [0, 0, 0.15], False),
            ("settle", self.config.rest_s, lower + [0, 0, 0.15], False),
        ]
        self._ends = np.cumsum([row[1] for row in self._schedule])
        self._start_tcp = self.runtime.tcp_position().copy()
        self._baseline: np.ndarray | None = None
        self._peak_lifts = np.zeros(2)
        self._bilateral = False
        self._transfer_count = self._transfer_contacts = 0
        self._release_z: float | None = None
        self._release_drop = self._penetration = self._stretch = 0.0
        self._particle_samples: list[np.ndarray] = []
        self._sample_times: list[float] = []
        self._placement_correction: np.ndarray | None = None

    def _align_placement(self, garment_com: np.ndarray) -> None:
        """Translate the remaining robot path so the hanging parcel COM reaches its goal."""
        goal = np.asarray(self.scene.config.target_xy) + np.asarray(self.config.place_offset_m[:2])
        destination = self._schedule[4][2]
        correction = np.zeros(3)
        correction[:2] = (
            goal - garment_com[:2] - (destination[:2] - self.runtime.tcp_position()[:2])
        )
        for index in range(4, len(self._schedule)):
            phase, duration, target, closed = self._schedule[index]
            self._schedule[index] = (phase, duration, target + correction, closed)
        self._placement_correction = correction

    def on_physics_step(self, robot: Any) -> bool:
        """Command only robot joints and measure the unconstrained bag and garment."""
        del robot
        index = int(np.searchsorted(self._ends, self.elapsed, side="right"))
        vertices = self.runtime.vertices()
        if len(self.rows) % 4 == 0:
            self._particle_samples.append(vertices.astype(np.float32))
            self._sample_times.append(self.elapsed)
        n = self.scene.garment_count
        masses = self.runtime.model.particle_mass.numpy()
        coms = np.asarray(
            [
                np.average(vertices[:n], axis=0, weights=masses[:n]),
                np.average(vertices[n:], axis=0, weights=masses[n:]),
            ]
        )
        if self._baseline is None and self.elapsed >= self._ends[0]:
            self._baseline = coms.copy()
        if self._baseline is not None:
            self._peak_lifts = np.maximum(self._peak_lifts, coms[:, 2] - self._baseline[:, 2])
        self._penetration = max(self._penetration, self.scene.table_top_z - vertices[:, 2].min())
        lengths = np.linalg.norm(
            vertices[self._edge_indices[:, 0]] - vertices[self._edge_indices[:, 1]], axis=1
        )
        self._stretch = max(self._stretch, float((lengths / self._edge_lengths).max()))
        if index >= len(self._schedule):
            self.runtime.hold()
            self.state.update(done=True, phase="done")
            return True
        if (
            self._schedule[index][0] == "transfer"
            and self.config.align_place_to_com
            and self._placement_correction is None
        ):
            self._align_placement(coms[0])
        phase, duration, target, closed = self._schedule[index]
        contacts = self.runtime.last_contact_counts
        self.state["progress"] = dict(
            com_lifts_m=self._peak_lifts.tolist(),
            finger_contacts=[contacts["left"], contacts["right"]],
            maximum_edge_stretch=float(self._stretch),
            maximum_table_penetration_m=float(self._penetration),
            tcp_m=self.runtime.tcp_position().tolist(),
        )
        bilateral = contacts["left"] > 0 and contacts["right"] > 0
        if closed and bilateral:
            self._bilateral = True
        if phase == "transfer":
            self._transfer_count += 1
            self._transfer_contacts += int(bilateral)
        if phase == "release" and self._release_z is None:
            self._release_z = float(coms[:, 2].mean())
        if self._release_z is not None:
            self._release_drop = max(self._release_drop, self._release_z - float(coms[:, 2].mean()))
        start_time = 0.0 if index == 0 else self._ends[index - 1]
        previous = self._start_tcp if index == 0 else self._schedule[index - 1][2]
        blend = np.clip((self.elapsed - start_time) / (duration * 0.6), 0, 1)
        blend = blend * blend * (3 - 2 * blend)
        command = previous * (1 - blend) + target * blend
        self.runtime.track_keyframe(
            command, np.array([2**-0.5, 2**-0.5, 0, 0]), 0.1 if closed else 0.8
        )
        self.state["phase"] = phase
        self.rows.append(
            dict(
                time_s=self.elapsed,
                phase=phase,
                garment_com_m=coms[0].tolist(),
                bag_com_m=coms[1].tolist(),
                tcp_m=self.runtime.tcp_position().tolist(),
                contacts_left=contacts["left"],
                contacts_right=contacts["right"],
                minimum_z_m=float(vertices[:, 2].min()),
            )
        )
        self.elapsed += self.robot.sim_world.config.physics_dt
        return False

    def write_diagnostics(self, directory: Path) -> None:
        """Save actual solver particles for independent geometry checks and replay."""
        np.savez_compressed(
            directory / "particles.npz",
            positions=np.asarray(self._particle_samples),
            times=np.asarray(self._sample_times),
            checkpoint_times_s=[0.0, *self._ends.tolist()],
            triangles=self.scene.triangle_indices.reshape(-1, 3),
            rest=self.runtime.model.particle_q.numpy(),
            masses=self.runtime.model.particle_mass.numpy(),
            garment_count=self.scene.garment_count,
        )

    def result(self) -> dict[str, Any]:
        """Reject partial lifts, lost transfers, penetration and misplaced releases."""
        vertices = self.runtime.vertices()
        n = self.scene.garment_count
        masses = self.runtime.model.particle_mass.numpy()
        com = np.average(vertices[:n], axis=0, weights=masses[:n])
        goal = np.array(self.scene.config.target_xy) + np.array(self.config.place_offset_m[:2])
        place_error = float(np.linalg.norm(com[:2] - goal))
        fraction = self._transfer_contacts / max(1, self._transfer_count)
        capacity = self.runtime.cloth_solver.maximum_contacts.numpy().tolist()
        containment = check_package_containment(
            np.asarray(self._particle_samples),
            self.scene.triangle_indices.reshape(-1, 3),
            n,
            times=np.asarray(self._sample_times),
            checkpoint_times_s=[0.0, *self._ends.tolist()],
        )
        checks = dict(
            sequence_completed=bool(self.state.get("done")),
            finite_state=bool(np.isfinite(vertices).all()),
            bilateral_contact_observed=self._bilateral,
            garment_lifted=bool(self._peak_lifts[0] >= self.config.minimum_lift_m),
            bag_lifted=bool(self._peak_lifts[1] >= self.config.minimum_lift_m),
            transfer_contact_maintained=fraction >= 0.95,
            released_and_dropped=self._release_drop >= self.config.minimum_release_drop_m,
            placed_near_goal=place_error <= self.config.maximum_place_error_m,
            rests_on_table=float(com[2]) < self.scene.table_top_z + 0.07,
            table_penetration_bounded=self._penetration <= self.config.maximum_table_penetration_m,
            surface_stretch_bounded=self._stretch < 1.3,
            no_pinned_vertices=self.runtime.pinned_vertex_count == 0,
            garment_contained_at_checkpoints=containment["passed"],
            no_contact_buffer_overflow=(
                max(capacity[:2]) < self.config.self_contact_buffer
                and capacity[2] < min(16384, self.runtime.contacts.soft_contact_max)
            ),
        )
        checks = {name: bool(value) for name, value in checks.items()}
        return dict(
            success=all(checks.values()),
            checks=checks,
            object="V7_two_mesh_package",
            maximum_com_lift_m=self._peak_lifts.tolist(),
            release_drop_m=self._release_drop,
            place_error_m=place_error,
            transfer_bilateral_contact_fraction=fraction,
            maximum_table_penetration_m=float(self._penetration),
            maximum_edge_stretch=self._stretch,
            garment_mass_kg=float(masses[:n].sum()),
            bag_mass_kg=float(masses[n:].sum()),
            particles=len(vertices),
            garment_particles=n,
            bag_particles=len(vertices) - n,
            internal_loft_springs=len(self.scene.loft_springs),
            attachments_used=False,
            pinned_vertex_count=self.runtime.pinned_vertex_count,
            coupling="one_way_rigid_to_deformable",
            physics_duration_s=self.elapsed,
            maximum_substep_contacts=capacity,
            containment=containment,
            newton_version=self.runtime.newton.__version__,
            warp_version=self.runtime.wp.__version__,
            asset_sha256=hashlib.sha256(
                Path(self.scene._asset(self.scene.config.cloth_usd_path)).read_bytes()
            ).hexdigest(),
            robot_backend=self.robot.config.backend,
            gripper_geometry=(
                "V7_14x60x84mm_parallel_pads"
                if hasattr(self.robot, "package_pad_shapes")
                else "native_Franka_fingers"
            ),
            placement_correction_m=(
                self._placement_correction.tolist()
                if getattr(self, "_placement_correction", None) is not None
                else None
            ),
        )
