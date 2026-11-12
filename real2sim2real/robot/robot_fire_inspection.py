"""X30 fire-inspection quadruped on PhysX: kinematic base with a scripted trot.

The official DeepRobotics X30 release (deep_robotics_model) ships URDF/MJCF/USD models only,
no locomotion policy. This backend therefore moves the floating base along the planner's
velocity command (root pose and matching velocity written to PhysX every physics step) and
drives the 12 leg joints with the asset's PD drives toward a scripted trot: diagonal leg
pairs alternate, stance feet move backwards at the body's ground speed, swing feet lift on a
sine arc. Foot targets are converted to joint targets with the X30 leg kinematics.
Robot links ignore gravity, so legs only follow the targets; the feet do not carry the body.
Recorded data is labelled ``locomotion="kinematic_base+scripted_trot"``: it shows plausible leg
motion but is not a dynamic gait and says nothing about X30 walking performance.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.robot.base import BaseRobot
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.cameras import CameraConfig, create_cameras, validate_camera_list
from real2sim2real.utils.fire_inspection_utils import (
    pose_matrix,
    quat_wxyz_from_yaw,
    yaw_from_quat_wxyz,
)

LEGS = ("FL", "FR", "HL", "HR")
_PARTS = ("HipX", "HipY", "Knee")
_TROT_PHASE = {"FL": 0.0, "HR": 0.0, "FR": 0.5, "HL": 0.5}


class X30LegGeometry(BaseModel):
    """Leg dimensions from the official X30 URDF (metres); checked against the USD at runtime."""

    model_config = ConfigDict(extra="forbid")
    hip_x_m: float = 0.291  # HipX joint, ±x from TORSO (front +, hind -)
    hip_y_m: float = 0.08  # HipX joint, ±y from TORSO (left +, right -)
    hip_offset_m: float = 0.11675  # HipY joint, ±y from HipX joint
    thigh_m: float = 0.30
    shank_m: float = 0.31


class X30GaitConfig(BaseModel):
    """Scripted trot; stride follows the commanded body velocity."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["trot", "stand"] = "trot"
    period_s: float = Field(default=0.5, gt=0.1, description="Full trot cycle")
    lift_m: float = Field(default=0.07, ge=0, le=0.15)
    max_stride_m: float = Field(default=0.13, gt=0, description="Max |foot offset| from nominal")
    full_speed_mps: float = Field(default=0.15, gt=0, description="Full lift above this speed")
    full_yaw_rate_rad_s: float = Field(default=0.3, gt=0)
    max_tracking_error_rad: float = Field(
        default=0.35,
        gt=0,
        description="Jam limit. Feet touch real colliders: stepping on the floor cables in the "
        "aisles deflects a leg by ~0.2 rad for a few tenths of a second.",
    )


class X30BeaconConfig(BaseModel):
    """Red warning beacon on the inspection cage; visual only (no collider, no mass)."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    # TORSO frame (m): centre of the cage top, above the gimbal, below the lidar scan plane.
    base_position_m: tuple[float, float, float] = (0.14, 0.0, 0.462)
    radius_m: float = Field(default=0.04, gt=0)
    height_m: float = Field(default=0.07, gt=0)
    color: tuple[float, float, float] = (1.0, 0.04, 0.02)
    light_intensity: float = Field(default=6000.0, gt=0, description="Normalized SphereLight")
    emissive_strength: float = Field(default=20.0, gt=0)


class X30RobotConfig(BaseConfig):
    """Asset, initial base pose, leg drives, gait and cameras for the X30 inspection robot."""

    backend: str = "x30_kinematic"
    usd_path: str = "assets/FireInspection/robot/x30/x30_fire_inspection.usdc"
    prim_path: str = "/World/robots/X30"
    base_xy: tuple[float, float] = (0.0, 0.0)
    base_yaw_deg: float = 0.0
    stance_height_m: float = Field(
        default=0.5226, gt=0, description="TORSO height above floor; asset manifest value."
    )
    floor_z: float = 0.0
    disable_gravity: bool = True
    joint_stiffness: float | None = Field(
        default=None, gt=0, description="Override asset PD stiffness; None keeps it."
    )
    joint_damping: float | None = Field(default=None, gt=0)
    stance_joint_positions: dict[str, float] = Field(
        default_factory=dict, description="rad; empty uses each joint's authored drive target."
    )
    leg: X30LegGeometry = Field(default_factory=X30LegGeometry)
    gait: X30GaitConfig = Field(default_factory=X30GaitConfig)
    beacon: X30BeaconConfig = Field(default_factory=X30BeaconConfig)
    camera_for_detection: str = "front_rgb"
    cameras: list[CameraConfig] = Field(default_factory=list)

    @field_validator("cameras")
    @classmethod
    def _validate_cameras(cls, value: list[CameraConfig]) -> list[CameraConfig]:
        return validate_camera_list(value)


# --- leg kinematics (TORSO frame, metres / rad; all X30 leg axes are -x, -y, -y) -------------
def _joint(leg: str, part: str) -> str:
    return f"{leg}_{part}_joint"


def _signs(leg: str) -> tuple[float, float]:
    return (1.0 if leg[0] == "F" else -1.0), (1.0 if leg[1] == "L" else -1.0)


def leg_fk(geometry: X30LegGeometry, leg: str, q: np.ndarray) -> np.ndarray:
    """Foot centre in the TORSO frame for joint angles (HipX, HipY, Knee)."""
    sx, sy = _signs(leg)
    qx, qy, qk = (float(v) for v in q)
    l1, l2 = geometry.thigh_m, geometry.shank_m
    x = l1 * math.sin(qy) + l2 * math.sin(qy + qk)
    z = -l1 * math.cos(qy) - l2 * math.cos(qy + qk)
    y = sy * geometry.hip_offset_m
    # Rotation by qx about -x.
    c, s = math.cos(qx), math.sin(qx)
    return np.array(
        [sx * geometry.hip_x_m + x, sy * geometry.hip_y_m + c * y + s * z, -s * y + c * z]
    )


def leg_ik(geometry: X30LegGeometry, leg: str, foot: np.ndarray) -> np.ndarray:
    """Joint angles (HipX, HipY, Knee) that put the foot centre at ``foot`` (TORSO frame)."""
    sx, sy = _signs(leg)
    vx = float(foot[0]) - sx * geometry.hip_x_m
    vy = float(foot[1]) - sy * geometry.hip_y_m
    vz = float(foot[2])
    d = sy * geometry.hip_offset_m
    rho = math.hypot(vy, vz)
    qx = -math.atan2(vz, vy) - math.acos(max(-1.0, min(1.0, d / rho)))
    qx = math.remainder(qx, math.tau)
    z = math.sin(qx) * vy + math.cos(qx) * vz
    l1, l2 = geometry.thigh_m, geometry.shank_m
    cos_k = (vx * vx + z * z - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    qk = math.acos(max(-1.0, min(1.0, cos_k)))
    qy = math.atan2(vx, -z) - math.atan2(l2 * math.sin(qk), l1 + l2 * math.cos(qk))
    return np.array([qx, qy, qk])


class TrotGait:
    """Foot targets for a trot whose stride matches the commanded body twist."""

    def __init__(self, config: X30GaitConfig, nominal: dict[str, np.ndarray]) -> None:
        self.config = config
        self.nominal = nominal
        self.phase = 0.0

    def feet(self, command: np.ndarray, dt: float) -> dict[str, np.ndarray]:
        """Advance the cycle by ``dt`` and return TORSO-frame foot targets per leg."""
        vx, vy, wz = (float(v) for v in command)
        cfg = self.config
        amplitude = min(
            1.0,
            max(math.hypot(vx, vy) / cfg.full_speed_mps, abs(wz) / cfg.full_yaw_rate_rad_s),
        )
        if cfg.mode == "stand" or amplitude < 1e-3:
            return {leg: foot.copy() for leg, foot in self.nominal.items()}
        self.phase = (self.phase + dt / cfg.period_s) % 1.0
        stance = cfg.period_s / 2
        targets = {}
        for leg, foot in self.nominal.items():
            # Ground speed of the foot relative to the body is -(v + w x p).
            velocity = np.array([vx - wz * foot[1], vy + wz * foot[0]])
            stride = velocity * stance
            norm = float(np.linalg.norm(stride))
            if norm > 2 * cfg.max_stride_m:
                stride *= 2 * cfg.max_stride_m / norm
            s = (self.phase + _TROT_PHASE[leg]) % 1.0
            if s < 0.5:  # stance: foot still on the ground, moving back relative to the body
                offset = stride * (0.5 - 2 * s)
                lift = 0.0
            else:  # swing: smooth return to the front with a sine lift
                u = 2 * (s - 0.5)
                offset = stride * (0.5 * (1 - math.cos(math.pi * u)) - 0.5)
                lift = cfg.lift_m * amplitude * math.sin(math.pi * u)
            targets[leg] = foot + np.array([offset[0], offset[1], lift])
        return targets


class X30Robot(BaseRobot):
    """PhysX articulation whose base follows planner velocity commands."""

    config_cls = X30RobotConfig
    recording_robot_type = "deeprobotics_x30_fire_inspection"
    locomotion = "kinematic_base+scripted_trot"

    def __init__(self, config: X30RobotConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("robot")
        self.articulation: Any = None
        self.root_path = ""
        self.joint_names: list[str] = []
        self.stance = np.zeros(0)
        self.targets = np.zeros(0)
        self.gait: TrotGait | None = None
        self._stance_by_name: dict[str, float] = {}
        self._nominal_feet: dict[str, np.ndarray] = {}
        self._leg_index: dict[str, list[int]] = {}
        self._camera_offsets: dict[str, np.ndarray] = {}
        self._camera_configs = {camera.name: camera for camera in self.config.cameras}
        self.command = np.zeros(3)  # body-frame vx, vy (m/s), wz (rad/s)
        self._beacon_on: bool | None = None  # None until the beacon prims exist
        self.state["beacon_switches"] = 0
        self.state["max_tracking_error_rad"] = 0.0
        if self.config.camera_for_detection not in self._camera_configs:
            raise ValueError(
                f"robot.camera_for_detection={self.config.camera_for_detection!r} "
                "must name one of [[robot.cameras]]"
            )

    @property
    def initial_position(self) -> np.ndarray:
        x, y = self.config.base_xy
        return np.array([x, y, self.config.floor_z + self.config.stance_height_m])

    @property
    def initial_orientation(self) -> np.ndarray:
        return quat_wxyz_from_yaw(math.radians(self.config.base_yaw_deg))

    def init_robot(self, sim_world: Any) -> None:
        """Reference the X30 USD, disable link gravity and mount cameras."""
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import PhysxSchema, Usd, UsdPhysics

        if sim_world.physics_engine != "physx":
            raise ValueError("x30_kinematic requires sim_world.physics_engine='physx'")
        self.sim_world = sim_world
        stage = sim_world.stage
        add_reference_to_stage(resolve_usd_path(self.config.usd_path), self.config.prim_path)
        root = stage.GetPrimAtPath(self.config.prim_path)
        bodies = 0
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                self.root_path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies += 1
                if self.config.disable_gravity:
                    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
        if bodies == 0 or not self.root_path:
            raise RuntimeError(f"X30 asset has no articulation: {self.config.usd_path}")
        self._place_reference(root)
        # No pose here: the legs are siblings of TORSO, so only the whole reference may move.
        self.articulation = SingleArticulation(prim_path=self.config.prim_path, name="x30")
        self.sensors.update(create_cameras(stage, self.config.cameras))
        self._camera_offsets = self._mount_offsets(stage)
        self._stance_by_name = self._stance_targets()
        self._nominal_feet = self._check_leg_kinematics()
        if self.config.beacon.enabled:
            self._create_beacon(stage)

    def _create_beacon(self, stage: Any) -> None:
        """Bracket across the cage top tubes, dark housing, red lens and a red sphere light."""
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        cfg = self.config.beacon
        root = f"{self.root_path}/rsr_beacon"
        xform = UsdGeom.Xform.Define(stage, root)
        xform.AddTranslateOp().Set(Gf.Vec3d(*cfg.base_position_m))
        # Bracket: 6 mm plate spanning the two longitudinal cage tubes (y = ±0.067 m).
        plate = UsdGeom.Cube.Define(stage, f"{root}/Bracket")
        plate.CreateSizeAttr(1.0)
        plate.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.003))
        plate.AddScaleOp().Set(Gf.Vec3f(0.06, 0.15, 0.006))
        plate.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.12, 0.13)])
        housing = UsdGeom.Cylinder.Define(stage, f"{root}/Housing")
        housing.CreateRadiusAttr(cfg.radius_m)
        housing.CreateHeightAttr(cfg.height_m * 0.4)
        housing.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, cfg.height_m * 0.2))
        housing.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.08, 0.08)])
        lens_center = Gf.Vec3d(0.0, 0.0, cfg.height_m * 0.4 + cfg.radius_m * 0.6)
        lens = UsdGeom.Sphere.Define(stage, f"{root}/Lens")
        lens.CreateRadiusAttr(cfg.radius_m * 0.9)
        lens.AddTranslateOp().Set(lens_center)
        # The light sits inside the lens; the lens must not shadow it.
        lens.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(
            True
        )
        material = UsdShade.Material.Define(stage, f"{root}/LensMaterial")
        shader = UsdShade.Shader.Define(stage, f"{root}/LensMaterial/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.35, 0.02, 0.02)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
        self._beacon_emissive = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(lens.GetPrim()).Bind(material)
        light = UsdLux.SphereLight.Define(stage, f"{root}/Light")
        light.CreateRadiusAttr(cfg.radius_m * 0.5)
        light.CreateColorAttr(Gf.Vec3f(*cfg.color))
        light.CreateNormalizeAttr(True)
        light.AddTranslateOp().Set(lens_center)
        self._beacon_light = light.CreateIntensityAttr(0.0)
        self._beacon_emissive.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        self._beacon_on = False

    def set_beacon(self, on: bool) -> None:
        """Switch the warning beacon; USD is only touched when the state changes."""
        if self._beacon_on is None:
            raise RuntimeError("robot.beacon.enabled is false or init_robot has not run")
        if on == self._beacon_on:
            return
        from pxr import Gf

        cfg = self.config.beacon
        self._beacon_light.Set(cfg.light_intensity if on else 0.0)
        glow = [c * cfg.emissive_strength for c in cfg.color] if on else [0.0, 0.0, 0.0]
        self._beacon_emissive.Set(Gf.Vec3f(*glow))
        self._beacon_on = on
        self.state["beacon_on_now"] = on
        self.state["beacon_switches"] += 1

    def _place_reference(self, root: Any) -> None:
        """Move the referenced asset so TORSO lands on the configured base pose.

        The asset's root prim carries the standing height; TORSO is a child body. The root
        gets one transform op equal to ``inverse(TORSO-in-root) * target TORSO pose``.
        """
        from pxr import Gf, UsdGeom

        stage = root.GetStage()
        cache = UsdGeom.XformCache()
        root_world = cache.GetLocalToWorldTransform(root)
        torso_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(self.root_path))
        torso_in_root = torso_world * root_world.GetInverse()
        target = Gf.Matrix4d(pose_matrix(self.initial_position, self.initial_orientation).tolist())
        parent = cache.GetLocalToWorldTransform(root.GetParent())
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(torso_in_root.GetInverse() * target * parent.GetInverse())
        placed = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(self.root_path))
        if not np.allclose(np.asarray(placed), np.asarray(target), atol=1e-6):
            raise RuntimeError(f"X30 TORSO placement failed: {np.asarray(placed).round(4)}")

    def _mount_offsets(self, stage: Any) -> dict[str, np.ndarray]:
        """Camera pose relative to the articulation root, measured in USD before simulation."""
        from pxr import UsdGeom

        cache = UsdGeom.XformCache()
        root = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(self.root_path)))
        offsets = {}
        for name, camera in self._camera_configs.items():
            prim = stage.GetPrimAtPath(camera.prim_path)
            world = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float)
            offsets[name] = world @ np.linalg.inv(root)
        return offsets

    def arrange(self) -> None:
        """Cache the configured base pose."""
        self.poses["x30"] = (self.initial_position, self.initial_orientation)

    def adjust(self) -> None:
        """Keep the configured base pose."""

    def initialize_runtime(self) -> None:
        """After world reset and play: create tensors, map leg joints and stand in place."""
        self.articulation.initialize()
        self.joint_names = list(self.articulation.dof_names)
        if sorted(self.joint_names) != sorted(self._stance_by_name):
            raise RuntimeError(f"Unexpected X30 joints: {self.joint_names}")
        self._leg_index = {
            leg: [self.joint_names.index(_joint(leg, part)) for part in _PARTS] for leg in LEGS
        }
        self.stance = np.array([self._stance_by_name[name] for name in self.joint_names])
        self.gait = TrotGait(self.config.gait, self._nominal_feet)
        self._configure_drives()
        self.targets = self.stance.copy()
        self.set_base(
            self.initial_position[:2], math.radians(self.config.base_yaw_deg), np.zeros(3)
        )
        self.articulation.set_joint_positions(self.stance)
        self.articulation.set_joint_velocities(np.zeros_like(self.stance))
        self.state["max_tracking_error_rad"] = 0.0
        self.state["ready"] = True

    def _stance_targets(self) -> dict[str, float]:
        """Leg joint targets (rad): config overrides, else the asset's authored drive targets."""
        from pxr import Usd, UsdPhysics

        authored: dict[str, float] = {}
        for prim in Usd.PrimRange(self.sim_world.stage.GetPrimAtPath(self.config.prim_path)):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                target = drive.GetTargetPositionAttr().Get() if drive else None
                if target is not None:
                    authored[prim.GetName()] = math.radians(float(target))
        names = [_joint(leg, part) for leg in LEGS for part in _PARTS]
        unknown = set(self.config.stance_joint_positions) - set(names)
        if unknown:
            raise ValueError(f"Unknown stance joints: {sorted(unknown)}")
        stance = {}
        for name in names:
            if name in self.config.stance_joint_positions:
                stance[name] = float(self.config.stance_joint_positions[name])
            elif name in authored:
                stance[name] = authored[name]
            else:
                raise RuntimeError(f"No stance target for joint {name}; set stance_joint_positions")
        overrides = bool(self.config.stance_joint_positions)
        self.state["stance_source"] = "config+asset" if overrides else "asset"
        return stance

    def _check_leg_kinematics(self) -> dict[str, np.ndarray]:
        """Stance foot positions from FK; must match the authored USD feet (before physics).

        Only valid before the physics view is built: PhysX writes its own link poses back.
        """
        from pxr import UsdGeom

        cache = UsdGeom.XformCache()
        stage = self.sim_world.stage
        root = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(self.root_path)))
        nominal, errors = {}, {}
        for leg in LEGS:
            q = np.array([self._stance_by_name[_joint(leg, part)] for part in _PARTS])
            foot = leg_fk(self.config.leg, leg, q)
            prim = stage.GetPrimAtPath(f"{self.config.prim_path}/{leg}_FOOT")
            if not prim:
                raise RuntimeError(f"X30 asset has no {leg}_FOOT body")
            world = np.asarray(cache.GetLocalToWorldTransform(prim))[3]
            usd = (world @ np.linalg.inv(root))[:3]
            errors[leg] = round(float(np.linalg.norm(usd - foot)), 5)
            if np.max(np.abs(leg_ik(self.config.leg, leg, foot) - q)) > 1e-6:
                raise RuntimeError(f"X30 leg IK does not invert FK for {leg}")
            if errors[leg] > 0.005:
                raise RuntimeError(
                    f"X30 leg geometry does not match the USD {leg} foot: "
                    f"usd={usd.round(4)} fk={foot.round(4)}"
                )
            nominal[leg] = foot
        self.state["leg_fk_error_m"] = errors
        return nominal

    def _configure_drives(self) -> None:
        if self.config.joint_stiffness is None and self.config.joint_damping is None:
            return
        controller = self.articulation.get_articulation_controller()
        kps, kds = controller.get_gains()
        n = len(self.joint_names)
        if self.config.joint_stiffness is not None:
            kps = np.full(n, self.config.joint_stiffness)
        if self.config.joint_damping is not None:
            kds = np.full(n, self.config.joint_damping)
        controller.set_gains(kps=np.asarray(kps), kds=np.asarray(kds))

    def set_base(self, xy: np.ndarray, yaw: float, command: np.ndarray, dt: float = 0.0) -> None:
        """Write root pose and matching world velocity; advance the gait by ``dt``."""
        from isaacsim.core.utils.types import ArticulationAction

        vx, vy, wz = (float(v) for v in command)
        c, s = math.cos(yaw), math.sin(yaw)
        position = np.array([xy[0], xy[1], self.initial_position[2]])
        self.articulation.set_world_pose(position=position, orientation=quat_wxyz_from_yaw(yaw))
        self.articulation.set_linear_velocity(np.array([c * vx - s * vy, s * vx + c * vy, 0.0]))
        self.articulation.set_angular_velocity(np.array([0.0, 0.0, wz]))
        targets = self.stance.copy()
        velocities = np.zeros_like(targets)
        if self.gait is not None and dt > 0:
            for leg, foot in self.gait.feet(np.array([vx, vy, wz]), dt).items():
                targets[self._leg_index[leg]] = leg_ik(self.config.leg, leg, foot)
            # Velocity feedforward: a PD drive with zero velocity target lags by kd/kp * qdot.
            velocities = (targets - self.targets) / dt
        self.articulation.apply_action(
            ArticulationAction(joint_positions=targets, joint_velocities=velocities)
        )
        self.targets = targets
        self.command = np.array([vx, vy, wz])

    def after_physics_step(self) -> float:
        """Track the worst joint error against the targets in force during the last step."""
        error = float(np.max(np.abs(self.joint_positions() - self.targets)))
        self.state["max_tracking_error_rad"] = max(self.state["max_tracking_error_rad"], error)
        return error

    def joint_positions(self) -> np.ndarray:
        return _numpy(self.articulation.get_joint_positions())

    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Actual root position and wxyz orientation from PhysX."""
        position, orientation = self.articulation.get_world_pose()
        return _numpy(position), _numpy(orientation)

    def base_xy_yaw(self) -> tuple[np.ndarray, float]:
        position, orientation = self.base_pose()
        return position[:2].copy(), yaw_from_quat_wxyz(orientation)

    def camera_matrix(self, name: str) -> np.ndarray:
        """Camera world matrix from the actual root pose and the fixed mount offset."""
        position, orientation = self.base_pose()
        return self._camera_offsets[name] @ pose_matrix(position, orientation)

    def camera_config(self, name: str) -> CameraConfig:
        return self._camera_configs[name]

    @property
    def recording_joint_names(self) -> list[str]:
        return ["base_x", "base_y", "base_z", "base_yaw", *self.joint_names]

    def recording_state(self) -> np.ndarray:
        position, orientation = self.base_pose()
        return np.concatenate([position, [yaw_from_quat_wxyz(orientation)], self.joint_positions()])


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float).reshape(-1).copy()
