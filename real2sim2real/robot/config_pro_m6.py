"""``[robot]`` Pro-M6 配置(新增文件,不改 ``robot/config.py``)."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.scene.config_sorting import CAMERA_ANNOTATORS
from real2sim2real.utils.pose import PoseConfig as Pose


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


class UsdRobotConfig(BaseConfig):
    """USD 机器人共用:路径 / 位姿 / 相机.Franka / G1 / TeleAvatar 都继承此类."""

    name: str = Field(default="robot", description="机器人名称")
    usd_path: str = Field(default="", description="机器人 USD 路径")
    prim_path: str = Field(default="/World/robots/Robot", description="机器人 prim 路径")
    pose: Pose = Field(default_factory=Pose, description="相对 pose_relative_to 的位姿")
    pose_relative_to: str = Field(default="/World", description="机器人相对参考 prim")
    cameras: list[RobotCameraConfig] = Field(default_factory=list, description="机器人相机")


class ProM6RobotConfig(UsdRobotConfig):
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
