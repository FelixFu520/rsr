"""HALO task robot: native 29-DOF Unitree G1 (three-finger hands) on a free base in PhysX.

The asset is Isaac Sim 6.2 ``Robots/Unitree/G1/g1.usd`` (Physics=PhysX, ThreeFinger hands),
flattened with the HALO reference silver materials. This backend reproduces the HALO
``physx_world.build`` robot setup on the runtime stage only:

- articulation root on ``pelvis``, self collisions, 32/8 solver iterations, no world joint;
- force drives on all 43 revolute joints: the 29 body motors use the GEAR-SONIC gains
  (armature × (2π·10 Hz)², ankles and waist pitch/roll ×2) and motor effort limits, shoulders
  and elbows use ``arm_stiffness/arm_damping``; the 14 finger joints hold 0 rad at 20/2/5 N·m;
- speculative CCD, 2 m/s depenetration, contact reports on every link, 4/0.5 mm offsets;
- a D455-like pinhole head camera fixed to ``torso_link``.

After the initial state is written once, every motion comes from PhysX integrating the motor
targets; the base is never written again (``root_pose_writes_during_capture = 0``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from real2sim2real.core.config import BaseConfig
from real2sim2real.robot.base import BaseRobot
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.cameras import CameraConfig, CameraPose, create_cameras
from real2sim2real.utils.halo_utils import (
    ARMATURE,
    EFFORT_LIMIT,
    KDS,
    KPS,
    MUJOCO_JOINT_NAMES,
)

LANDMARK_BODIES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
STATE_NAMES = (
    [f"joint_pos.{n}" for n in MUJOCO_JOINT_NAMES]
    + [f"joint_vel.{n}" for n in MUJOCO_JOINT_NAMES]
    + ["base.x", "base.y", "base.z", "base.qw", "base.qx", "base.qy", "base.qz"]
    + ["base.vx", "base.vy", "base.vz", "base.wx", "base.wy", "base.wz"]
)


class G1HeadCameraConfig(BaseModel):
    """Head RGB camera rigidly mounted on a robot link (HALO: D455 colour optics, 25° down).

    Width/height are scalars so a debug config can inherit and override them.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = "head"
    link: str = "torso_link"
    position_m: tuple[float, float, float] = (0.12, 0.0, 0.43)
    orientation_wxyz: tuple[float, float, float, float] = (
        0.59636781,
        0.3799282,
        -0.3799282,
        -0.59636781,
    )  # USD camera axes; optical axis 25° below the torso x axis
    width: int = Field(default=640, gt=0, multiple_of=2)
    height: int = Field(default=360, gt=0, multiple_of=2)
    focal_length: float = Field(default=1.93, gt=0)
    horizontal_aperture: float = Field(default=3.896, gt=0)
    clipping_range: tuple[float, float] = (0.02, 100.0)


class G1SonicRobotConfig(BaseConfig):
    """Asset, drives, contact settings and cameras of the HALO G1."""

    backend: str = "g1_sonic"
    usd_path: str = "assets/Halo/Robot/g1_29_reference.usd"
    manifest_path: str = Field(
        default="assets/Halo/Robot/robot_manifest.json",
        description="HALO robot manifest: body order, collider count and mass to verify",
    )
    prim_path: str = "/World/robots/G1"
    arm_stiffness_nm_rad: float = Field(default=40.0, gt=0, description="shoulders and elbows")
    arm_damping_nms_rad: float = Field(default=3.0, ge=0, description="HALO crawl_v3 uses 4.0")
    auxiliary_stiffness_nm_rad: float = Field(default=20.0, gt=0, description="finger joints")
    auxiliary_damping_nms_rad: float = Field(default=2.0, ge=0)
    auxiliary_effort_nm: float = Field(default=5.0, gt=0)
    auxiliary_armature_kgm2: float = Field(default=0.00425, ge=0)
    self_collisions: bool = True
    solver_position_iterations: int = Field(default=32, ge=1)
    solver_velocity_iterations: int = Field(default=8, ge=0)
    max_depenetration_velocity_mps: float = Field(default=2.0, gt=0)
    speculative_ccd: bool = True
    contact_offset_m: float = Field(default=0.004, gt=0)
    rest_offset_m: float = Field(default=0.0005, ge=0)
    static_friction: float = Field(default=1.0, ge=0)
    dynamic_friction: float = Field(default=0.8, ge=0)
    restitution: float = Field(default=0.0, ge=0, le=1)
    place_at_start: bool = Field(
        default=False,
        description="Move the USD reference so pelvis starts at the start pose; HALO left the "
        "reference at the asset origin and only wrote the start state through PhysX",
    )
    landmark_bodies: list[str] = Field(default_factory=lambda: list(LANDMARK_BODIES))
    head_camera: G1HeadCameraConfig = Field(default_factory=G1HeadCameraConfig)


class G1SonicRobot(BaseRobot):
    """Free-base G1 whose 29 body motors follow the planner's position targets."""

    config_cls = G1SonicRobotConfig
    recording_robot_type = "unitree_g1_29dof_physx"
    recording_joint_names = STATE_NAMES
    action_names = list(MUJOCO_JOINT_NAMES)

    def __init__(self, config: G1SonicRobotConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("robot")
        self.initial_qpos: np.ndarray | None = None
        self.body_names: list[str] = []
        self.manifest: dict = {}
        self.dof_names: list[str] = []
        self.body_index = np.zeros(0, dtype=int)
        self.default_targets = np.zeros(0)
        self.motor_targets = np.zeros(0)
        self.joint_limits = np.zeros((29, 2))
        self.state["root_pose_writes_during_capture"] = 0
        self._views: dict[str, Any] = {}
        head = self.config.head_camera
        link = f"{self.config.prim_path}/{head.link}"
        self.cameras = [
            CameraConfig(
                name=head.name,
                prim_path=f"{link}/{head.name}_camera",
                pose_relative_to=link,
                pose=CameraPose(position=head.position_m, orientation_wxyz=head.orientation_wxyz),
                resolution=(head.width, head.height),
                clipping_range=head.clipping_range,
                focal_length=head.focal_length,
                horizontal_aperture=head.horizontal_aperture,
            )
        ]

    # --- setup ----------------------------------------------------------------------------
    def set_initial_qpos(self, qpos: np.ndarray) -> None:
        """Start pose ``[x, y, z, qw, qx, qy, qz, 29 joints]``; set before ``init_robot``."""
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (36,) or not np.isfinite(qpos).all():
            raise ValueError("G1 initial qpos must be 36 finite values")
        self.initial_qpos = qpos.copy()

    @property
    def prefix(self) -> str:
        """Prim-path prefix of every robot link (contact classification)."""
        return self.config.prim_path + "/"

    def init_robot(self, sim_world: Any) -> None:
        """Reference the G1, author HALO physics settings, place it and mount the head camera."""
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        if sim_world.physics_engine != "physx":
            raise ValueError("g1_sonic requires sim_world.physics_engine='physx'")
        if self.initial_qpos is None:
            raise RuntimeError("Call set_initial_qpos before init_robot")
        self.sim_world = sim_world
        stage = sim_world.stage
        cfg = self.config
        self.manifest = json.loads(Path(resolve_usd_path(cfg.manifest_path)).read_text())
        root = stage.DefinePrim(cfg.prim_path, "Xform")
        root.GetReferences().AddReference(resolve_usd_path(cfg.usd_path))
        material = UsdShade.Material.Define(stage, "/World/robots/PhysicsMaterial")
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(cfg.static_friction)
        api.CreateDynamicFrictionAttr(cfg.dynamic_friction)
        api.CreateRestitutionAttr(cfg.restitution)
        arm = {"shoulder", "elbow"}
        articulation_roots, world_joints, drives = [], [], {}
        bodies = []
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(path)
                pa = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                pa.CreateEnabledSelfCollisionsAttr(cfg.self_collisions)
                pa.CreateSolverPositionIterationCountAttr(cfg.solver_position_iterations)
                pa.CreateSolverVelocityIterationCountAttr(cfg.solver_velocity_iterations)
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                if not joint.GetBody0Rel().GetTargets() and joint.GetJointEnabledAttr().Get():
                    world_joints.append(path)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                name = prim.GetName()
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr("force")
                if name in MUJOCO_JOINT_NAMES:
                    i = MUJOCO_JOINT_NAMES.index(name)
                    kp, kd, effort = KPS[i], KDS[i], EFFORT_LIMIT[i]
                    if any(part in name for part in arm):
                        kp, kd = cfg.arm_stiffness_nm_rad, cfg.arm_damping_nms_rad
                else:
                    kp, kd = cfg.auxiliary_stiffness_nm_rad, cfg.auxiliary_damping_nms_rad
                    effort = cfg.auxiliary_effort_nm
                # USD angular drives are per degree.
                drive.CreateStiffnessAttr(float(kp * math.pi / 180))
                drive.CreateDampingAttr(float(kd * math.pi / 180))
                drive.CreateMaxForceAttr(float(effort))
                revolute = UsdPhysics.RevoluteJoint(prim)
                drives[name] = {
                    "kp_nm_rad": float(kp),
                    "kd_nms_rad": float(kd),
                    "max_force_nm": float(effort),
                    "limits_deg": [
                        float(revolute.GetLowerLimitAttr().Get()),
                        float(revolute.GetUpperLimitAttr().Get()),
                    ],
                }
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(path[len(self.prefix) :])
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rb.CreateMaxDepenetrationVelocityAttr(cfg.max_depenetration_velocity_mps)
                rb.CreateEnableSpeculativeCCDAttr(cfg.speculative_ccd)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                pc.CreateContactOffsetAttr(cfg.contact_offset_m)
                pc.CreateRestOffsetAttr(cfg.rest_offset_m)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, UsdShade.Tokens.weakerThanDescendants, "physics"
                )
        if articulation_roots != [f"{cfg.prim_path}/pelvis"]:
            raise RuntimeError(f"G1 articulation root must be pelvis: {articulation_roots}")
        if world_joints:
            raise RuntimeError(f"G1 has enabled world joints: {world_joints}")
        if bodies != self.manifest["body_order"]:
            raise RuntimeError("G1 rigid-body order differs from the HALO robot manifest")
        missing = sorted(set(MUJOCO_JOINT_NAMES) - set(drives))
        if missing or len(drives) != 43:
            raise RuntimeError(f"G1 needs 29 body + 14 finger joints; missing {missing}")
        self.body_names = bodies
        self.state["drives"] = drives
        self.state["mass_kg"] = float(self.manifest["mass_kg"])
        if cfg.place_at_start:
            self._place_reference(stage, root, Gf, UsdGeom)
        self.sensors.update(create_cameras(stage, self.cameras))

    def _place_reference(self, stage: Any, root: Any, Gf: Any, UsdGeom: Any) -> None:
        """Move the whole reference so ``pelvis`` starts at the initial base pose."""
        q = self.initial_qpos
        cache = UsdGeom.XformCache()
        pelvis_in_root = (
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{self.config.prim_path}/pelvis"))
            * cache.GetLocalToWorldTransform(root).GetInverse()
        )
        target = Gf.Matrix4d(1.0)
        target.SetRotate(Gf.Quatd(float(q[3]), Gf.Vec3d(*map(float, q[4:7]))))
        target.SetTranslateOnly(Gf.Vec3d(*map(float, q[:3])))
        parent = cache.GetLocalToWorldTransform(root.GetParent())
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(pelvis_in_root.GetInverse() * target * parent.GetInverse())
        placed = UsdGeom.XformCache().GetLocalToWorldTransform(
            stage.GetPrimAtPath(f"{self.config.prim_path}/pelvis")
        )
        if not np.allclose(np.asarray(placed), np.asarray(target), atol=1e-6):
            raise RuntimeError("G1 pelvis placement failed")

    def arrange(self) -> None:
        """Cache the start pose."""
        q = self.initial_qpos
        self.poses["g1"] = (q[:3].copy(), q[3:7].copy())

    def adjust(self) -> None:
        """Keep the reference start pose."""

    # --- runtime (after world reset + play) ------------------------------------------------
    def initialize_runtime(self) -> None:
        """Create tensor views, write the start state once and set motor armatures."""
        from isaacsim.core.experimental.prims import Articulation, RigidPrim

        path = self.config.prim_path
        articulation = Articulation(path)
        self._views = {
            "articulation": articulation,
            "pelvis": RigidPrim(f"{path}/pelvis"),
            "torso": RigidPrim(f"{path}/torso_link"),
            "landmarks": RigidPrim([f"{path}/{n}" for n in self.config.landmark_bodies]),
            "bodies": RigidPrim([f"{path}/{n}" for n in self.body_names]),
        }
        self.dof_names = list(articulation.dof_names)
        self.body_index = np.array([self.dof_names.index(n) for n in MUJOCO_JOINT_NAMES])
        n = len(self.dof_names)
        q = self.initial_qpos
        default = np.zeros(n)
        default[self.body_index] = q[7:]
        articulation.set_world_poses(positions=q[:3], orientations=q[3:7])
        articulation.set_velocities(linear_velocities=np.zeros(3), angular_velocities=np.zeros(3))
        articulation.set_dof_positions(default)
        articulation.set_dof_velocities(np.zeros(n))
        articulation.set_dof_position_targets(default)
        armature = np.full(n, self.config.auxiliary_armature_kgm2)
        armature[self.body_index] = ARMATURE
        articulation.set_dof_armatures(armature)
        lower, upper = articulation.get_dof_limits()
        # Keep PhysX's float32: HALO clipped targets to float32 bounds (lo + 1 mm-rad in float32);
        # float64 bounds differ by ~1e-8 rad and change the closed-loop trajectory.
        self.joint_limits = np.stack(
            [
                np.asarray(lower.numpy(), dtype=np.float32)[0, self.body_index],
                np.asarray(upper.numpy(), dtype=np.float32)[0, self.body_index],
            ],
            axis=1,
        )
        kp, kd = articulation.get_dof_gains()
        self.state["runtime_gains"] = {
            "kp": _numpy(kp)[0, self.body_index].tolist(),
            "kd": _numpy(kd)[0, self.body_index].tolist(),
        }
        self.default_targets = default
        self.motor_targets = default.copy()
        self.state["ready"] = True

    def read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``qpos`` (base pose + 29 joints), 29 joint velocities, pelvis world twist."""
        views = self._views
        position, orientation = views["pelvis"].get_world_poses()
        linear, angular = views["pelvis"].get_velocities()
        dof_pos = _numpy(views["articulation"].get_dof_positions())[0]
        dof_vel = _numpy(views["articulation"].get_dof_velocities())[0]
        qpos = np.r_[_numpy(position)[0], _numpy(orientation)[0], dof_pos[self.body_index]]
        return qpos, dof_vel[self.body_index], np.r_[_numpy(linear)[0], _numpy(angular)[0]]

    def set_body_targets(self, targets: np.ndarray) -> np.ndarray:
        """Send 29 body position targets (rad); fingers keep their start targets."""
        full = self.default_targets.copy()
        full[self.body_index] = targets
        self._views["articulation"].set_dof_position_targets(full)
        self.motor_targets = full
        return full

    def torso_pose(self) -> np.ndarray:
        position, orientation = self._views["torso"].get_world_poses()
        return np.r_[_numpy(position)[0], _numpy(orientation)[0]]

    def measure(self) -> dict[str, np.ndarray]:
        """Torso pose, landmark origins, every body pose and all 43 native joint states."""
        views = self._views
        landmarks, _ = views["landmarks"].get_world_poses()
        body_p, body_q = views["bodies"].get_world_poses()
        return {
            "torso_pose": self.torso_pose(),
            "landmark_positions": _numpy(landmarks),
            "robot_body_poses": np.concatenate([_numpy(body_p), _numpy(body_q)], axis=1),
            "all_native_joint_pos": _numpy(views["articulation"].get_dof_positions())[0],
            "all_native_joint_vel": _numpy(views["articulation"].get_dof_velocities())[0],
            "all_native_motor_targets": self.motor_targets.copy(),
        }

    def drive_report(self) -> dict[str, Any]:
        """Force-limited drives on the 29 body motors (acceptance)."""
        drives = self.state.get("drives", {})
        body = [drives.get(name) for name in MUJOCO_JOINT_NAMES]
        ok = all(
            d is not None and 0 < d["max_force_nm"] <= EFFORT_LIMIT[i] + 1e-3
            for i, d in enumerate(body)
        )
        return {
            "body_motor_drives": len([d for d in body if d is not None]),
            "native_joints": len(drives),
            "force_limited_body_motors": bool(ok),
        }


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=float)
    return array.reshape(1, -1) if array.ndim == 1 else array
