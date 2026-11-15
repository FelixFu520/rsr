"""Two KUKA KR210 L150 arms (PhysX) with physically jointed end effectors.

Sections:

1. Configuration of the arms, the compliant screwdriver and the parallel gripper.
2. ``Kr210DualRobot``: references the KR210 USD twice, keeps the asset's joint drives and
   collisions, and extends each articulation below ``tool0``:

   - screwdriver arm: ``tool0 -(prismatic float, spring)- float -(revolute spindle, velocity
     drive with torque limit)- spindle`` carrying a cross bit (box blades) and the visual
     screwdriver; the magnetic holding force of the bit is applied by the planner;
   - gripper arm: two prismatic, force-limited fingers with rubber pads; the bridge and flange
     are colliders of ``tool0``.

   All motion comes from joint drive targets; no body pose is written after the start.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import LoggerMixin
from real2sim2real.robot.base import BaseRobot
from real2sim2real.utils.asset_utils import prim_aabb, resolve_usd_path
from real2sim2real.utils.battery_disassembly_utils import (
    SerialChain,
    ToolFrame,
    matrix_to_quat,
    transform,
    yaw_matrix,
)
from real2sim2real.utils.cameras import CameraConfig, create_cameras, validate_camera_list

Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
ARM_JOINTS = ["joint_a1", "joint_a2", "joint_a3", "joint_a4", "joint_a5", "joint_a6"]
# gripper root axes in tool0 (columns): G x = tool0 Z, G y = tool0 Y, G z = -tool0 X
GRIPPER_IN_TOOL0 = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

# ================================================================================================
# 1. Configuration
# ================================================================================================


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArmConfig(_Strict):
    """One KR210 base pose relative to ``pose_relative_to``."""

    prim_path: str
    position_m: Vec3
    yaw_deg: FiniteFloat


class ScrewdriverConfig(_Strict):
    """Compliant electric screwdriver with a magnetic cross bit (tool axis = tool0 +X)."""

    visual_usd: str = "assets/BatteryDisassembly/Tools/crosshead_screwdriver.usdz"
    tool_length_m: FiniteFloat = Field(default=0.30, gt=0, description="tool0 → bit tip")
    tool0_mass_kg: FiniteFloat = Field(default=2.0, gt=0, description="motor and float housing")
    float_mass_kg: FiniteFloat = Field(default=0.3, gt=0)
    spindle_mass_kg: FiniteFloat = Field(default=0.15, gt=0)
    float_stiffness_n_m: FiniteFloat = Field(default=1500.0, gt=0)
    float_damping_ns_m: FiniteFloat = Field(default=30.0, ge=0)
    float_preload_extension_m: FiniteFloat = Field(
        default=0.008, gt=0, description="spring target beyond nominal; preload at nominal"
    )
    float_limits_m: tuple[FiniteFloat, FiniteFloat] = (-0.03, 0.012)
    spindle_damping_nms_rad: FiniteFloat = Field(default=0.5, gt=0)
    spindle_max_torque_nm: FiniteFloat = Field(default=2.0, gt=0)
    spindle_armature_kgm2: FiniteFloat = Field(default=2e-4, ge=0)
    blade_length_m: FiniteFloat = Field(default=0.0035, gt=0)
    blade_span_m: FiniteFloat = Field(default=0.012, gt=0)
    blade_thickness_m: FiniteFloat = Field(default=0.0018, gt=0)
    shaft_radius_m: FiniteFloat = Field(default=0.0045, gt=0)
    contact_offset_m: FiniteFloat = Field(default=0.0008, gt=0)
    magnet_max_force_n: FiniteFloat = Field(default=3.0, gt=0)
    magnet_range_m: FiniteFloat = Field(default=0.006, gt=0)


class GripperConfig(_Strict):
    """Two-finger parallel gripper (visual usdz parts, prismatic force-limited fingers)."""

    visual_usd: str = "assets/BatteryDisassembly/Tools/kr210_gripper.usdz"
    mount_offset_m: FiniteFloat = Field(default=0.1244, gt=0, description="tool0 → gripper root")
    tool0_mass_kg: FiniteFloat = Field(default=15.0, gt=0)
    finger_mass_kg: FiniteFloat = Field(default=1.5, gt=0)
    stroke_m: FiniteFloat = Field(default=0.06, gt=0, description="closing travel per finger")
    stiffness_n_m: FiniteFloat = Field(default=2.0e4, gt=0)
    damping_ns_m: FiniteFloat = Field(default=400.0, ge=0)
    max_force_n: FiniteFloat = Field(default=200.0, gt=0, description="per finger")
    pad_static_friction: FiniteFloat = Field(default=1.0, ge=0)
    pad_dynamic_friction: FiniteFloat = Field(default=0.9, ge=0)
    contact_offset_m: FiniteFloat = Field(default=0.002, gt=0)


class Kr210DualRobotConfig(BaseConfig):
    """Screwdriver arm and gripper arm."""

    backend: str = "kr210_dual"
    usd_path: str = "assets/BatteryDisassembly/Robot/KR210_L150/kr210_l150.usd"
    urdf_path: str = "assets/BatteryDisassembly/Robot/KR210_L150/kr210.urdf"
    pose_relative_to: str = "/World/scene"
    screwdriver_arm: ArmConfig = Field(
        default_factory=lambda: ArmConfig(
            prim_path="/World/robots/kr210_01", position_m=(-2.05, 0.0, 0.0), yaw_deg=90.0
        )
    )
    gripper_arm: ArmConfig = Field(
        default_factory=lambda: ArmConfig(
            prim_path="/World/robots/kr210_02", position_m=(2.05, 0.0, 0.0), yaw_deg=-90.0
        )
    )
    solver_position_iterations: int = Field(default=32, ge=1)
    solver_velocity_iterations: int = Field(default=4, ge=0)
    link_contact_offset_m: FiniteFloat = Field(default=0.01, gt=0)
    screwdriver: ScrewdriverConfig = Field(default_factory=ScrewdriverConfig)
    gripper: GripperConfig = Field(default_factory=GripperConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)


# ================================================================================================
# 2. Robot
# ================================================================================================


def _matrix(prim: Any) -> np.ndarray:
    from pxr import UsdGeom

    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=float).T


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=float)
    return array.reshape(1, -1) if array.ndim == 1 else array


class Kr210DualRobot(BaseRobot, LoggerMixin):
    """Screwdriver KR210 (``screw``) and gripper KR210 (``grip``)."""

    config_cls = Kr210DualRobotConfig
    recording_robot_type = "kuka_kr210_l150_dual"

    def __init__(self, config: Kr210DualRobotConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config: Kr210DualRobotConfig = config or self.config_cls.from_settings("robot")
        self.cameras = validate_camera_list(list(self.config.cameras))
        self.chain = SerialChain.from_urdf(resolve_usd_path(self.config.urdf_path))
        if self.chain.names != ARM_JOINTS:
            raise ValueError(f"KR210 URDF joints differ: {self.chain.names}")
        sd, gr = self.config.screwdriver, self.config.gripper
        self.tools = {
            "screw": ToolFrame(np.array([sd.tool_length_m, 0.0, 0.0])),
            "grip": ToolFrame(np.array([gr.mount_offset_m, 0.0, 0.0]), GRIPPER_IN_TOOL0),
        }
        self.arms = {"screw": self.config.screwdriver_arm, "grip": self.config.gripper_arm}
        self.bases: dict[str, np.ndarray] = {}
        self._views: dict[str, Any] = {}
        self._dof: dict[str, dict[str, int]] = {}
        self.targets: dict[str, np.ndarray] = {}
        self._spindle_raw = 0.0
        self.spindle_angle = 0.0

    @property
    def recording_joint_names(self) -> list[str]:
        names = [f"screw_arm.{j}" for j in ARM_JOINTS] + [f"grip_arm.{j}" for j in ARM_JOINTS]
        return names + ["screwdriver.float_m", "screwdriver.spindle_rad", "gripper.finger_p_m",
                        "gripper.finger_n_m"]  # fmt: skip

    def tool_paths(self) -> dict[str, str]:
        screw, grip = self.arms["screw"].prim_path, self.arms["grip"].prim_path
        return {
            "screw_tool0": f"{screw}/tool0",
            "float": f"{screw}/rsr_sd_float",
            "spindle": f"{screw}/rsr_sd_spindle",
            "grip_tool0": f"{grip}/tool0",
            "finger_p": f"{grip}/rsr_finger_p",
            "finger_n": f"{grip}/rsr_finger_n",
        }

    # --- lifecycle -----------------------------------------------------------------------------
    def init_robot(self, sim_world: Any) -> None:
        """Reference both arms, fix their bases, author tools and cameras."""
        if sim_world.physics_engine != "physx":
            raise ValueError("kr210_dual requires sim_world.physics_engine='physx'")
        self.sim_world = sim_world
        for key in ("screw", "grip"):
            self._reference_arm(key)
        self._author_screwdriver()
        self._author_gripper()
        self.sensors.update(create_cameras(sim_world.stage, self.cameras))
        self.log.info(
            "KR210 双臂已装配",
            bases={k: np.round(v[:3, 3], 4).tolist() for k, v in self.bases.items()},
        )

    def arrange(self) -> None:
        for key, base in self.bases.items():
            self.poses[key] = (base[:3, 3].copy(), matrix_to_quat(base[:3, :3]))

    def adjust(self) -> None:
        """Base poses are fixed by configuration."""

    def _reference_arm(self, key: str) -> None:
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        stage = self.sim_world.stage
        arm = self.arms[key]
        parent = _matrix(stage.GetPrimAtPath(self.config.pose_relative_to))
        base = parent @ transform(yaw_matrix(arm.yaw_deg), arm.position_m)
        root = UsdGeom.Xform.Define(stage, arm.prim_path)
        if not root.GetPrim().GetReferences().AddReference(resolve_usd_path(self.config.usd_path)):
            raise RuntimeError(f"KR210 reference failed: {arm.prim_path}")
        holder = _matrix(stage.GetPrimAtPath(arm.prim_path).GetParent())
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        local = np.linalg.inv(holder) @ base
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d(local.T.tolist()))
        prim = root.GetPrim()
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise RuntimeError(f"KR210 articulation root expected at {arm.prim_path}")
        art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        art.CreateEnabledSelfCollisionsAttr(False)
        art.CreateSolverPositionIterationCountAttr(self.config.solver_position_iterations)
        art.CreateSolverVelocityIterationCountAttr(self.config.solver_velocity_iterations)
        art.CreateSleepThresholdAttr(0.0)
        # the world anchor of the fixed base follows the placed reference
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"{arm.prim_path}/root_joint"))
        if not joint or joint.GetBody0Rel().GetTargets():
            raise RuntimeError(f"KR210 root_joint to the world expected below {arm.prim_path}")
        q = matrix_to_quat(base[:3, :3])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*base[:3, 3].tolist()))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(*q.tolist()))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        for link in Usd.PrimRange(prim):
            if link.HasAPI(UsdPhysics.CollisionAPI):
                pc = PhysxSchema.PhysxCollisionAPI.Apply(link)
                pc.CreateContactOffsetAttr(self.config.link_contact_offset_m)
                pc.CreateRestOffsetAttr(0.0)
            if link.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(link).CreateThresholdAttr(0.0)
        self.bases[key] = base

    def _body(self, path: str, world: np.ndarray, mass: float, inertia: Any = None) -> Any:
        """Rigid body prim at a world pose (parent transform compensated)."""
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        stage = self.sim_world.stage
        xform = UsdGeom.Xform.Define(stage, path)
        holder = _matrix(stage.GetPrimAtPath(path).GetParent())
        local = np.linalg.inv(holder) @ world
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d(local.T.tolist()))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(mass)
        if inertia is not None:
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        return prim

    def _collider(
        self,
        path: str,
        kind: Literal["box", "cylinder"],
        center: Any,
        size: Any,
        offset: float,
        material: Any | None = None,
        axis: str = "X",
    ) -> Any:
        """Invisible collider in its parent body's frame (box: size xyz; cylinder: r, h)."""
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        stage = self.sim_world.stage
        if kind == "box":
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            geom.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
            geom.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        else:
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateRadiusAttr(float(size[0]))
            geom.CreateHeightAttr(float(size[1]))
            geom.CreateAxisAttr(axis)
            geom.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        UsdGeom.Imageable(geom).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        pc = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
        pc.CreateContactOffsetAttr(offset)
        pc.CreateRestOffsetAttr(0.0)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
        return geom.GetPrim()

    def _joint(
        self, kind: Any, path: str, body0: str, body1: str, pos0: Any, rot0: Any, axis: str
    ) -> Any:
        from pxr import Gf

        joint = kind.Define(self.sim_world.stage, path)
        joint.CreateBody0Rel().SetTargets([body0])
        joint.CreateBody1Rel().SetTargets([body1])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
        joint.CreateLocalRot0Attr(Gf.Quatf(*map(float, rot0)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateAxisAttr(axis)
        return joint

    def _author_screwdriver(self) -> None:
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        stage = self.sim_world.stage
        cfg = self.config.screwdriver
        paths = self.tool_paths()
        tool0 = stage.GetPrimAtPath(paths["screw_tool0"])
        mass = UsdPhysics.MassAPI.Apply(tool0)
        mass.CreateMassAttr(cfg.tool0_mass_kg)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.01, 0.01))
        tip = _matrix(tool0) @ transform(None, (cfg.tool_length_m, 0.0, 0.0))
        self._body(paths["float"], tip, cfg.float_mass_kg, (1e-4, 1e-4, 1e-4))
        spindle = self._body(paths["spindle"], tip, cfg.spindle_mass_kg)
        fl = self._joint(
            UsdPhysics.PrismaticJoint, f"{paths['float']}/rsr_float_joint", paths["screw_tool0"],
            paths["float"], (cfg.tool_length_m, 0, 0), (1, 0, 0, 0), "X",
        )  # fmt: skip
        fl.CreateLowerLimitAttr(cfg.float_limits_m[0])
        fl.CreateUpperLimitAttr(cfg.float_limits_m[1])
        drive = UsdPhysics.DriveAPI.Apply(fl.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(cfg.float_stiffness_n_m)
        drive.CreateDampingAttr(cfg.float_damping_ns_m)
        drive.CreateTargetPositionAttr(cfg.float_preload_extension_m)
        drive.CreateMaxForceAttr(1.0e4)
        sp = self._joint(
            UsdPhysics.RevoluteJoint, f"{paths['spindle']}/rsr_spindle_joint", paths["float"],
            paths["spindle"], (0, 0, 0), (1, 0, 0, 0), "X",
        )  # fmt: skip
        drive = UsdPhysics.DriveAPI.Apply(sp.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(cfg.spindle_damping_nms_rad * math.pi / 180.0)
        drive.CreateMaxForceAttr(cfg.spindle_max_torque_nm)
        drive.CreateTargetVelocityAttr(0.0)
        PhysxSchema.PhysxJointAPI.Apply(sp.GetPrim()).CreateArmatureAttr(cfg.spindle_armature_kgm2)
        offset = cfg.contact_offset_m
        b = cfg.blade_length_m
        self._collider(
            f"{paths['spindle']}/rsr_blade_y", "box", (-0.5 * b, 0, 0),
            (b, cfg.blade_span_m, cfg.blade_thickness_m), offset,
        )  # fmt: skip
        self._collider(
            f"{paths['spindle']}/rsr_blade_z", "box", (-0.5 * b, 0, 0),
            (b, cfg.blade_thickness_m, cfg.blade_span_m), offset,
        )  # fmt: skip
        shaft = cfg.tool_length_m - b - 0.02
        self._collider(
            f"{paths['spindle']}/rsr_shaft", "cylinder", (-b - 0.5 * shaft, 0, 0),
            (cfg.shaft_radius_m, shaft), offset,
        )  # fmt: skip
        visual = UsdGeom.Xform.Define(stage, f"{paths['spindle']}/visual")
        visual.GetPrim().GetReferences().AddReference(resolve_usd_path(cfg.visual_usd))
        # mesh +Z (tip at z=0, 0.3 m long) along spindle -X
        rotation = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        UsdGeom.Xformable(visual).AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Matrix4d(transform(rotation).T.tolist())
        )
        self._disable_visual_physics(visual.GetPrim())
        self.state["spindle_prim"] = str(spindle.GetPath())

    def _disable_visual_physics(self, prim: Any) -> None:
        from pxr import Usd, UsdPhysics

        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(child).CreateRigidBodyEnabledAttr(False)

    def _author_gripper(self) -> None:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        stage = self.sim_world.stage
        cfg = self.config.gripper
        paths = self.tool_paths()
        asset = resolve_usd_path(cfg.visual_usd)
        tool0_path = paths["grip_tool0"]
        tool0 = stage.GetPrimAtPath(tool0_path)
        mass = UsdPhysics.MassAPI.Apply(tool0)
        mass.CreateMassAttr(cfg.tool0_mass_kg)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        mount = transform(GRIPPER_IN_TOOL0, (cfg.mount_offset_m, 0.0, 0.0))
        frame = UsdGeom.Xform.Define(stage, f"{tool0_path}/rsr_gripper")
        frame.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d(mount.T.tolist()))
        pad = UsdShade.Material.Define(stage, f"{self.arms['grip'].prim_path}/rsr_pad_material")
        api = UsdPhysics.MaterialAPI.Apply(pad.GetPrim())
        api.CreateStaticFrictionAttr(cfg.pad_static_friction)
        api.CreateDynamicFrictionAttr(cfg.pad_dynamic_friction)
        api.CreateRestitutionAttr(0.0)
        world_root = _matrix(tool0) @ mount
        for part in ("Bridge", "Flange"):
            visual = stage.DefinePrim(f"{tool0_path}/rsr_gripper/{part}")
            visual.GetReferences().AddReference(asset, Sdf.Path(f"/root/Gripper/{part}"))
            self._disable_visual_physics(visual)
            lo, hi = self._local_bounds(visual, world_root)
            # colliders belong to tool0: express them in the tool0 frame
            center = (mount @ np.r_[0.5 * (lo + hi), 1.0])[:3]
            size = np.abs(GRIPPER_IN_TOOL0 @ (hi - lo))
            self._collider(
                f"{tool0_path}/rsr_{part.lower()}_collider",
                "box",
                center,
                size,
                cfg.contact_offset_m,
            )
        for side, sign in (("p", 1.0), ("n", -1.0)):
            body_path = paths[f"finger_{side}"]
            body = self._body(body_path, world_root, cfg.finger_mass_kg)
            visual = stage.DefinePrim(f"{body_path}/visual")
            visual.GetReferences().AddReference(
                asset, Sdf.Path(f"/root/Gripper/Finger{side.upper()}")
            )
            self._disable_visual_physics(visual)
            for child in visual.GetChildren():
                lo, hi = self._local_bounds(child, world_root)
                is_pad = child.GetName().startswith("Pad")
                self._collider(
                    f"{body_path}/rsr_{child.GetName().lower()}_collider", "box", 0.5 * (lo + hi),
                    hi - lo, cfg.contact_offset_m, pad if is_pad else None,
                )  # fmt: skip
            joint = self._joint(
                UsdPhysics.PrismaticJoint, f"{body_path}/rsr_finger_{side}_joint", tool0_path,
                body_path, mount[:3, 3], matrix_to_quat(GRIPPER_IN_TOOL0), "Y",
            )  # fmt: skip
            lower, upper = (-cfg.stroke_m, 0.0) if sign > 0 else (0.0, cfg.stroke_m)
            joint.CreateLowerLimitAttr(lower)
            joint.CreateUpperLimitAttr(upper)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(cfg.stiffness_n_m)
            drive.CreateDampingAttr(cfg.damping_ns_m)
            drive.CreateMaxForceAttr(cfg.max_force_n)
            drive.CreateTargetPositionAttr(0.0)
            del body
        self.state["gripper_root_in_tool0"] = mount

    def _local_bounds(self, prim: Any, frame_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """AABB of a visual subtree in a body frame (the subtree is axis aligned with it)."""
        stage = self.sim_world.stage
        lo, hi = prim_aabb(stage, str(prim.GetPath()), gprims_only=True)
        inv = np.linalg.inv(frame_world)
        corners = np.array(
            [[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        local = (inv @ corners.T).T[:, :3]
        return local.min(axis=0), local.max(axis=0)

    # --- runtime -------------------------------------------------------------------------------
    def initialize_runtime(self) -> None:
        """Tensor views, joint indices; arms start at home (all joints zero)."""
        from isaacsim.core.experimental.prims import Articulation, RigidPrim

        paths = self.tool_paths()
        for key in ("screw", "grip"):
            art = Articulation(self.arms[key].prim_path)
            names = list(art.dof_names)
            index = {name: names.index(name) for name in ARM_JOINTS}
            self._views[key] = art
            self._dof[key] = index
            self.state[f"{key}_dof_names"] = names
            zeros = np.zeros((1, len(names)))
            art.set_dof_positions(zeros)
            art.set_dof_velocities(zeros)
            art.set_dof_position_targets(zeros)
            art.set_dof_velocity_targets(zeros)
        grip_names = self.state["grip_dof_names"]
        self._finger_index = {
            side: grip_names.index(f"rsr_finger_{side}_joint") for side in ("p", "n")
        }
        screw_names = self.state["screw_dof_names"]
        self._float_index = screw_names.index("rsr_float_joint")
        self._spindle_index = screw_names.index("rsr_spindle_joint")
        self._views["spindle"] = RigidPrim(paths["spindle"])
        self._views["fingers"] = RigidPrim([paths["finger_p"], paths["finger_n"]])
        self.targets = {
            "screw": np.zeros(6), "grip": np.zeros(6),
            "float": np.array([self.config.screwdriver.float_preload_extension_m]),
            "spindle_velocity": np.zeros(1), "finger_p": np.zeros(1), "finger_n": np.zeros(1),
        }  # fmt: skip
        self._spindle_raw = 0.0
        self.spindle_angle = 0.0
        self.state["ready"] = True

    def _arm_indices(self, key: str) -> list[int]:
        return [self._dof[key][name] for name in ARM_JOINTS]

    def set_arm_targets(self, key: str, q: np.ndarray, qd: np.ndarray) -> None:
        """Joint position (rad) and feed-forward velocity (rad/s) targets of one arm."""
        idx = self._arm_indices(key)
        art = self._views[key]
        art.set_dof_position_targets(np.asarray(q, dtype=float).reshape(1, 6), dof_indices=idx)
        art.set_dof_velocity_targets(np.asarray(qd, dtype=float).reshape(1, 6), dof_indices=idx)
        self.targets[key] = np.asarray(q, dtype=float).copy()

    def set_screwdriver(self, float_target_m: float, spindle_velocity_rad_s: float) -> None:
        art = self._views["screw"]
        art.set_dof_position_targets(np.array([[float_target_m]]), dof_indices=[self._float_index])
        art.set_dof_velocity_targets(
            np.array([[spindle_velocity_rad_s]]), dof_indices=[self._spindle_index]
        )
        self.targets["float"] = np.array([float_target_m])
        self.targets["spindle_velocity"] = np.array([spindle_velocity_rad_s])

    def set_fingers(self, closing_m: float) -> None:
        """Symmetric finger target: 0 open, ``stroke_m`` fully closed (force limited)."""
        art = self._views["grip"]
        p, n = self._finger_index["p"], self._finger_index["n"]
        art.set_dof_position_targets(np.array([[-closing_m, closing_m]]), dof_indices=[p, n])
        self.targets["finger_p"] = np.array([-closing_m])
        self.targets["finger_n"] = np.array([closing_m])

    def read_state(self) -> dict[str, np.ndarray]:
        """Measured joint states (SI units); spindle angle is unwrapped."""
        out = {}
        for key in ("screw", "grip"):
            art = self._views[key]
            q = _numpy(art.get_dof_positions())[0]
            qd = _numpy(art.get_dof_velocities())[0]
            idx = self._arm_indices(key)
            out[f"{key}_q"], out[f"{key}_qd"] = q[idx], qd[idx]
            out[f"{key}_all_q"] = q
        screw_q = out["screw_all_q"]
        raw = float(screw_q[self._spindle_index])
        delta = (raw - self._spindle_raw + math.pi) % (2 * math.pi) - math.pi
        self.spindle_angle += delta
        self._spindle_raw = raw
        out["float"] = float(screw_q[self._float_index])
        out["spindle"] = self.spindle_angle
        out["spindle_qd"] = float(
            _numpy(self._views["screw"].get_dof_velocities())[0][self._spindle_index]
        )
        grip_q = out["grip_all_q"]
        out["finger_p"] = float(grip_q[self._finger_index["p"]])
        out["finger_n"] = float(grip_q[self._finger_index["n"]])
        return out

    def tip_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Bit tip position and the tool axis (spindle +X) in world."""
        from real2sim2real.utils.battery_disassembly_utils import quat_to_matrix

        p, q = self._views["spindle"].get_world_poses()
        rotation = quat_to_matrix(_numpy(q)[0])
        return _numpy(p)[0], rotation[:, 0]

    def apply_spindle_force(self, force: np.ndarray) -> None:
        """Reaction of the magnet on the bit (world frame)."""
        if np.any(force):
            self._views["spindle"].apply_forces(np.asarray(force, dtype=np.float32).reshape(1, 3))

    def finger_poses(self) -> np.ndarray:
        p, _ = self._views["fingers"].get_world_poses()
        return _numpy(p)

    def contact_owner(self, path: str) -> str | None:
        """Contact actor category: ``screw_tool``, ``grip_tool``, ``screw_arm`` or ``grip_arm``."""
        paths = self.tool_paths()
        if any(path.startswith(paths[k]) for k in ("float", "spindle", "screw_tool0")):
            return "screw_tool"
        if any(path.startswith(paths[k]) for k in ("finger_p", "finger_n", "grip_tool0")):
            return "grip_tool"
        for key in ("screw", "grip"):
            if path.startswith(self.arms[key].prim_path + "/"):
                return f"{key}_arm"
        return None
