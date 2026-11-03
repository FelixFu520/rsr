"""分拣包裹后端：配置与实现。"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import get_logger
from real2sim2real.robot.base import BaseRobot
from real2sim2real.simworld.base import BaseSimWorld
from real2sim2real.utils.isaacsim_utils import (
    bind_camera_sensor,
    normalize_prim_path,
    place_asset,
    pump_frames,
    resolve_usd_path,
    spawn_reference,
    spawn_subprim_reference,
)
from real2sim2real.utils.isaacsim_utils.camera_utils import CAMERA_ANNOTATORS
from real2sim2real.utils.pose import PoseConfig as Pose
from real2sim2real.utils.pose import read_prim_world_pose


class RobotCameraConfig(BaseModel):
    """机器人相机配置(``[[robot.cameras]]``),字段与场景相机对齐."""

    name: str = Field(default="", description="相机名称;缺省取 prim_path 末段")
    usd_path: str = Field(default="", description="相机 USD;空则创建 Camera prim")
    prim_path: str = Field(
        ...,
        description=(
            "光学 prim.桌面臂挂在手上;移动机写 /World/Cameras/<name>,跟随看 pose_relative_to"
        ),
    )
    authored_prim_path: str = Field(
        default="",
        description="可选额外匹配路径,给 job 视口/落盘筛选",
    )
    pose: Pose = Field(default_factory=Pose, description="相机位姿")
    pose_relative_to: str = Field(..., description="相机相对参考 prim")
    resolution: list[int] = Field(
        default_factory=lambda: [640, 480],
        description="采集分辨率 [width, height]",
    )
    annotators: list[str] = Field(
        default_factory=lambda: ["rgb"],
        description="CameraSensor 采集项,如 rgb / distance_to_image_plane",
    )
    tick_rate: float | None = Field(
        default=None,
        description="传感器 Hz;空则保留 USD 内 omni:sensor:tickRate",
    )
    clipping_range: list[float] | None = Field(
        default=None,
        description="近/远裁剪面 [near, far];空则保留 USD 内 clippingRange",
    )
    hide_visuals: bool = Field(
        default=True,
        description="隐藏壳体 Mesh,避免相机自己的外观进入画面",
    )
    collect_all_cameras: bool = Field(
        default=False,
        description=(
            "True:壳体 USD 下每个 Camera prim 各绑一路传感器并落盘"
            "(如 RealSense Color/Left/Right/Depth)"
        ),
    )
    stabilize_level: bool = Field(
        default=False,
        description=(
            "开稳定机架:相机脱离 link 层级,改由滤波后的机器人根位姿驱动,"
            "只保留航向,步态的俯仰/横滚/上下起伏都不进画面"
        ),
    )
    stabilize_tau: float = Field(
        default=0.25,
        ge=0.0,
        description="稳定机架的低通时间常数(秒);越大越稳越滞后,0 表示不滤波",
    )
    stabilize_lock_height: bool = Field(
        default=True,
        description="True:相机高度锁在出生高度,不跟骨盆上下起伏(平地采集推荐)",
    )
    focal_length: float | None = Field(
        default=None,
        description="裸相机(usd_path 为空时)焦距 mm;空则用 USD Camera 默认值",
    )
    horizontal_aperture: float | None = Field(
        default=None,
        description="裸相机水平感光尺寸 mm,与 focal_length 一起决定视场角",
    )
    vertical_aperture: float | None = Field(
        default=None,
        description="裸相机垂直感光尺寸 mm",
    )

    @model_validator(mode="after")
    def _default_name(self) -> Self:
        """未写 name 时用 prim 路径最后一段,方便 get_image("wrist_cam").

        Returns:
            补全 ``name`` 后的自身.
        """
        if not self.name:  # TOML 可省略 name,采集仍要用稳定键
            self.name = self.prim_path.rstrip("/").rsplit("/", 1)[-1]  # .../wrist_cam → wrist_cam
        return self

    @field_validator("resolution")
    @classmethod
    def _check_resolution(cls, value: list[int]) -> list[int]:
        """分辨率必须是两个正整数 ``[width, height]``.

        Args:
            value: TOML 里的 ``resolution``;须长度为 2 且均为正整数.

        Returns:
            ``[width, height]`` 整数列表.

        Raises:
            ValueError: 长度不是 2 或存在非正整数.
        """
        if len(value) != 2 or int(value[0]) <= 0 or int(value[1]) <= 0:  # 防 0/负数让传感器崩
            raise ValueError("resolution 须为 [width, height],且均为正整数")
        return [int(value[0]), int(value[1])]  # 统一成 int,避免 toml 读成 float

    @field_validator("annotators")
    @classmethod
    def _check_annotators(cls, value: list[str]) -> list[str]:
        """至少一项,且必须在场景相机同一白名单内.

        Args:
            value: ``CameraSensor`` 采集项列表,如 ``rgb``.

        Returns:
            拷贝后的采集项列表.

        Raises:
            ValueError: 为空或不在 ``CAMERA_ANNOTATORS`` 白名单内.
        """
        if not value:  # 空列表会导致 CameraSensor 无输出
            raise ValueError("annotators 不能为空")
        unknown = [item for item in value if item not in CAMERA_ANNOTATORS]  # 与场景相机同一集合
        if unknown:  # 拼写错误尽早失败,不要等到 Isaac 绑 sensor
            allowed = ", ".join(sorted(CAMERA_ANNOTATORS))  # 给调用方可读白名单
            raise ValueError(f"未知 annotators {unknown};可选: {allowed}")
        return list(value)  # 拷贝,避免调用方事后改原 list

    @field_validator("clipping_range")
    @classmethod
    def _check_clipping_range(cls, value: list[float] | None) -> list[float] | None:
        """None 表示沿用 USD;否则须 ``0 < near < far``.

        Args:
            value: ``[near, far]`` 或 None;腕部 near 过小会看到夹爪内部.

        Returns:
            ``[near, far]`` 浮点列表;未配置则为 ``None``.

        Raises:
            ValueError: 已配置但不是 ``0 < near < far``.
        """
        if value is None:  # 未配置则保留资产 clippingRange
            return None
        # near 须为正且严格小于 far,否则 USD Camera 裁剪非法
        if len(value) != 2 or float(value[0]) <= 0 or float(value[1]) <= float(value[0]):
            raise ValueError("clipping_range 须为 [near, far],且 0 < near < far")
        return [float(value[0]), float(value[1])]  # 统一 float,避免 int/str 混入 USD


class ProM6RobotConfig(BaseConfig):
    """``[robot]``:Pro-M6 双臂分拣机器人配置.

    USD 挂到 ``/World/robots/Pro_m6_696_dynamics_fixed``.视觉网格须按场景包
    里的源 prim 引用(``usd_prim_path``);不要用只有碰撞方块的代理 USD.
    """

    name: str = Field(default="pro_m6", description="机器人名称")
    usd_path: str = Field(
        default="assets/Robot/pro_m6_696/pro_m6_696.usda",
        description="Pro-M6 USD;空则假定已在 stage 上",
    )
    prim_path: str = Field(
        default="/World/robots/Pro_m6_696_dynamics_fixed",
        description="articulation 根 prim;子 prim 引用时即此路径,否则 USD 挂在其父级 /World/robots",
    )
    usd_prim_path: str = Field(
        default="/Pro_m6_696_dynamics_fixed",
        description=(
            "被引用文件里的源 prim;非空则按子 prim 引用(保留 instanceable 视觉网格)."
            "空则挂 USD 默认 prim"
        ),
    )
    left_arm_joint_names: list[str] = Field(
        default_factory=lambda: [f"Joint{i}_L" for i in range(1, 8)],
        description="左臂 7 个关节名,顺序即 DOF 顺序",
    )
    right_arm_joint_names: list[str] = Field(
        default_factory=lambda: [f"Joint{i}_R" for i in range(1, 8)],
        description="右臂 7 个关节名,顺序即 DOF 顺序",
    )
    left_gripper_prim_path: str = Field(
        default="", description="左夹爪 prim;空则 <root>/robotiq_left"
    )
    right_gripper_prim_path: str = Field(
        default="", description="右夹爪 prim;空则 <root>/robotiq_right"
    )

    pose: Pose = Field(default_factory=Pose, description="相对 pose_relative_to 的位姿")
    pose_relative_to: str = Field(default="/World", description="机器人相对参考 prim")
    cameras: list[RobotCameraConfig] = Field(default_factory=list, description="机器人相机")


class ProM6Robot(BaseRobot):
    """Pro-M6 双臂:USD 挂在 ``/World/robots``,articulation 用其子 prim.

    ``place_robot``(分拣 Job)::

        AutoRobot.from_config()
        init_robot(sim_world)                  # 把带视觉网格的机器人 prim 挂到 /World/robots
        Job.adjust_robot()                     # 子 prim 引用已带原机位,不再叠平移
        arrange()                              # 登记位姿、绑机载相机、建 articulation
    """

    config_cls = ProM6RobotConfig

    def __init__(self, config: ProM6RobotConfig | Any | None = None, **kwargs: Any) -> None:
        """写入 ``config``,占位 articulation.

        Args:
            config: 已构造配置;None 则从 ``[robot]`` 加载.
            **kwargs: 工厂多余字段,丢弃.

        Returns:
            None.
        """
        super().__init__()
        del kwargs
        self.articulation_obj: Any | None = None
        self.config: ProM6RobotConfig = config or self.config_cls.from_settings("robot")

    @property
    def spawn_prim_path(self) -> str:
        """USD 引用根:articulation 根的父级,通常是 ``/World/robots``.

        Returns:
            规范化后的父级 prim 路径.
        """
        root = normalize_prim_path(self.config.prim_path)
        parent = root.rsplit("/", 1)[0]
        return parent if parent and parent != "/" else root

    # ============= 机器人流水线 <==========

    def init_robot(self, sim_world: BaseSimWorld) -> None:
        """绑定世界并挂机器人 USD;不写位姿.

        Args:
            sim_world: 已 ``init_world`` 的 Isaac 世界,须带可用 stage.

        Returns:
            None.

        Raises:
            RuntimeError: 未配置 usd_path,或引用后找不到 articulation 根.
        """
        from isaacsim.core.utils.stage import add_reference_to_stage

        self.sim_world = sim_world
        assert self.sim_world is not None, "Sim world is not initialized"
        assert self.sim_world.stage is not None, "Stage is not initialized"

        usd = (self.config.usd_path or "").strip()
        src_prim = (self.config.usd_prim_path or "").strip()
        if usd:
            resolved = resolve_usd_path(usd)
            if src_prim:
                spawn_subprim_reference(
                    self.sim_world.stage, self.config.prim_path, resolved, src_prim
                )
            else:
                spawn_reference(
                    self.sim_world.stage, self.spawn_prim_path, resolved, add_reference_to_stage
                )
            pump_frames(self.sim_world, 4)
            self.log.info(
                "引用 Pro-M6 成功",
                spawn_prim=self.spawn_prim_path,
                prim_path=self.config.prim_path,
                usd=resolved,
                usd_prim_path=src_prim or None,
            )

        root = normalize_prim_path(self.config.prim_path)
        prim = self.sim_world.stage.GetPrimAtPath(root)
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"环境 USD 缺少机器人 prim: {root}")

    def arrange(self) -> None:
        """按配置摆实例根(不改 USD 内部尺度),绑机载相机,再建 articulation.

        Returns:
            None.
        """
        root = normalize_prim_path(self.config.prim_path)
        src_prim = (self.config.usd_prim_path or "").strip()
        if src_prim:
            # 源 prim 已带世界位姿,再写父级会叠一次平移
            child_pose = read_prim_world_pose(self.sim_world.stage, root)
            if child_pose is None:
                raise RuntimeError(f"读不到 Pro-M6 世界位姿: {root}")
            pos, quat = child_pose
        else:
            pos, quat = place_asset(
                self.sim_world.stage,
                self.poses,
                self.spawn_prim_path,
                self.config.pose,
                self.config.pose_relative_to,
            )
            pump_frames(self.sim_world, 1)
            child_pose = read_prim_world_pose(self.sim_world.stage, root)
            if child_pose is not None:
                pos, quat = child_pose
        self.poses[root] = (pos, quat)
        self.log.info("摆放 Pro-M6 成功", prim_path=root, position=pos.tolist())

        for cam in self.config.cameras:
            self.sensors[cam.name] = bind_camera_sensor(self.sim_world.stage, cam)
            pump_frames(self.sim_world, 1)
        if self.config.cameras:
            self.log.info("绑定机载相机成功", cameras=[cam.name for cam in self.config.cameras])

        self.seed_poses_realtime()
        self._build_articulation()

    def post_physics_init(self, *, warmup_steps: int = 30) -> None:
        """Play 后只步进出帧;双臂关节由分拣规划器每步接管,这里不下发目标.

        Args:
            warmup_steps: 热身物理步数;至少 1.

        Returns:
            None.
        """
        world = None if self.sim_world is None else self.sim_world.world
        if world is None:
            return
        for _ in range(max(1, int(warmup_steps))):
            try:
                world.step(render=True)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("热身步进失败", error=str(exc))
                break

    # ============= 辅助函数 <==========

    def _build_articulation(self) -> None:
        """``arrange`` 末尾:建 ``isaacsim.core.experimental.prims.Articulation``.

        Returns:
            None.
        """
        from isaacsim.core.experimental.prims import Articulation

        root = normalize_prim_path(self.config.prim_path)
        try:
            self.articulation_obj = Articulation(root)
            self.log.debug("构建 Pro-M6 articulation 成功", prim_path=root)
        except Exception as exc:  # noqa: BLE001
            self.log.error("构建 Pro-M6 articulation 失败", error=str(exc))
            self.articulation_obj = None

    @property
    def left_gripper_prim_path(self) -> str:
        """左夹爪 prim;未配置则取 ``<root>/robotiq_left``.

        Returns:
            左夹爪 prim 路径.
        """
        root = normalize_prim_path(self.config.prim_path)
        return self.config.left_gripper_prim_path or f"{root}/robotiq_left"

    @property
    def right_gripper_prim_path(self) -> str:
        """右夹爪 prim;未配置则取 ``<root>/robotiq_right``.

        Returns:
            右夹爪 prim 路径.
        """
        root = normalize_prim_path(self.config.prim_path)
        return self.config.right_gripper_prim_path or f"{root}/robotiq_right"

    def get_state(self) -> dict[str, Any]:
        """落盘用的双臂观测:全量 DOF 位置 / 速度 / 目标.

        Returns:
            ``joint_positions`` / ``joint_velocities`` / ``joint_position_targets``.
        """
        empty = {
            "joint_positions": np.zeros(0, dtype=float),
            "joint_velocities": np.zeros(0, dtype=float),
            "joint_position_targets": np.zeros(0, dtype=float),
        }
        art = self.articulation_obj
        if art is None:
            return empty
        out: dict[str, Any] = {}
        for key, getter in (
            ("joint_positions", "get_dof_positions"),
            ("joint_velocities", "get_dof_velocities"),
            ("joint_position_targets", "get_dof_position_targets"),
        ):
            try:
                raw = getattr(art, getter)()
                value = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
                out[key] = np.asarray(value, dtype=float).reshape(-1)
            except Exception:  # noqa: BLE001
                out[key] = empty[key]
        return out

    def layout_prim_paths(self) -> list[str]:
        """视口框选:引用根、articulation 根与机载相机.

        Returns:
            prim 路径列表.
        """
        paths = [self.spawn_prim_path, self.config.prim_path]
        paths.extend(cam.prim_path for cam in self.config.cameras)
        return paths

    def seed_poses_realtime(self) -> None:
        """用 ``poses`` 覆盖 ``poses_realtime``;``arrange`` 结束时调用."""
        self.poses_realtime = dict(self.poses)

    def _root_prim_key(self) -> str | None:
        """根 prim 的规范化路径;配置尚未写入则为 ``None``."""
        prim = getattr(self.config, "prim_path", None)
        if prim is None:
            return None
        return normalize_prim_path(str(prim))

    def get_prim_pose(self, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``arrange`` 后缓存的世界位姿.

        Args:
            prim_path: USD prim 路径.

        Returns:
            ``(世界 pos, wxyz)`` 元组.

        Raises:
            KeyError: 该 prim 尚未摆放.
        """
        pose = self.poses.get(normalize_prim_path(prim_path))
        if pose is None:
            raise KeyError(f"尚未摆放: {prim_path}")
        return pose

    @property
    def root_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """当前根世界位姿:优先 ``poses_realtime``,否则 ``poses``."""
        key = self._root_prim_key()
        if key is None:
            return None
        return self.poses_realtime.get(key) or self.poses.get(key)

    @root_pose.setter
    def root_pose(self, value: tuple[np.ndarray, np.ndarray] | None) -> None:
        """只写 ``poses_realtime``;``None`` 则删掉 realtime 根键."""
        key = self._root_prim_key()
        if key is None:
            return
        if value is None:
            self.poses_realtime.pop(key, None)
            return
        pos, quat = value
        self.poses_realtime[key] = (
            np.asarray(pos, dtype=float).reshape(-1)[:3].copy(),
            np.asarray(quat, dtype=float).reshape(-1)[:4].copy(),
        )

    def get_meta(self) -> dict[str, Any]:
        """写入 episode meta 的机器人快照.

        Returns:
            含 ``robot_type``;配置有 ``usd_path`` 时再带 ``robot_usd``.
        """
        usd = getattr(self.config, "usd_path", None)
        meta: dict[str, Any] = {"robot_type": self.robot_type}
        if usd is not None:
            meta["robot_usd"] = usd
        return meta

    @property
    def robot_type(self) -> str:
        """机型名,写入 episode meta.

        Returns:
            机型名字符串.
        """
        return str(getattr(self.config, "name", None) or "robot")

    def get_image(self, name: str, annotator: str = "rgb") -> np.ndarray | None:
        """按相机名取一帧;``annotator`` 选 RGB / 深度 / 其它已绑定项.

        Args:
            name: ``sensors`` 字典里的相机名(如 ``wrist`` / ``head_cam``).
            annotator: ``rgb``(默认)、``depth``(针孔深度别名)或
                ``CAMERA_ANNOTATORS`` 里的官方名;须已写入该相机 ``annotators``.

        Returns:
            numpy 数组;RGB 多为 ``uint8`` ``HxWx3``,深度多为浮点 ``HxW``.
            未绑定、未挂该 annotator 或读失败则为 ``None``.
        """
        sensor = self.sensors.get(name)
        if sensor is None:
            return None
        from real2sim2real.utils.isaacsim_utils.camera_utils import read_sensor_data

        return read_sensor_data(sensor, annotator=annotator, name=name)

    @property
    def articulation(self) -> Any:
        """主臂关节体句柄;尚未构建时为 ``None``.

        Returns:
            主臂 articulation;未构建时为 ``None``.
        """
        return self.articulation_obj

    @property
    def articulation_controller(self) -> Any:
        """主臂关节 PD 控制器;无 articulation 时为 ``None``.

        Returns:
            关节控制器;无 articulation 时为 ``None``.
        """
        art = self.articulation_obj
        if art is None:
            return None
        getter = getattr(art, "get_articulation_controller", None)
        return getter() if callable(getter) else None

    @property
    def log(self) -> Any:
        """返回绑定组件名的日志器。"""
        return get_logger(f"{type(self).__module__}.{type(self).__qualname__}").bind(
            component=type(self).__qualname__
        )
