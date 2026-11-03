"""``[scene]`` 分拣台配置与场景/机器人共用的相机配置(新增文件,不改 ``scene/config.py``)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.utils.pose import PoseConfig as Pose

# 与 isaacsim.sensors.experimental.rtx.CameraSensor 的 annotators 对齐
CAMERA_ANNOTATORS = frozenset(
    {
        "rgb",  # 彩色图,采集主通路
        "rgba",
        "distance_to_image_plane",  # 针孔深度
        "distance_to_camera",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation",
        "instance_id_segmentation",
        "bounding_box_2d_tight",
        "bounding_box_2d_loose",
        "bounding_box_3d",
        "pointcloud",
    }
)


class SceneCameraConfig(BaseModel):
    """场景相机配置(``[[scene.cameras]]``)."""

    name: str = Field(..., description="相机名称")
    usd_path: str = Field(default="", description="相机 USD;空则创建 Camera prim")
    prim_path: str = Field(..., description="相机实例根 prim;壳体 USD 的 Camera 可在子树")
    authored_prim_path: str = Field(
        default="",
        description="TOML 里写的 prim_path;运行时改挂后仍用它做 job 视口/落盘筛选",
    )
    pose: Pose = Field(..., description="相机位姿")
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
        description="隐藏壳体 Mesh,避免相机自己的外观进入采集画面;视口仍显示 Camera gizmo",
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

    @field_validator("resolution")
    @classmethod
    def _check_resolution(cls, value: list[int]) -> list[int]:
        """分辨率必须是两个正整数 ``[width, height]``.

        Args:
            value: TOML 里的 ``resolution``;须长度为 2 且均为正整数.

        Returns:
            规范化后的 ``[width, height]`` 整数列表.
        """
        if len(value) != 2 or int(value[0]) <= 0 or int(value[1]) <= 0:  # 防 0/负数让传感器崩
            raise ValueError("resolution 须为 [width, height],且均为正整数")
        return [int(value[0]), int(value[1])]  # 统一成 int,避免 toml 读成 float

    @field_validator("annotators")
    @classmethod
    def _check_annotators(cls, value: list[str]) -> list[str]:
        """至少一项,且必须在 ``CAMERA_ANNOTATORS`` 白名单内.

        Args:
            value: ``CameraSensor`` 采集项列表,如 ``rgb``.

        Returns:
            采集项列表的浅拷贝.
        """
        if not value:  # 空列表会导致 CameraSensor 无输出
            raise ValueError("annotators 不能为空")
        unknown = [item for item in value if item not in CAMERA_ANNOTATORS]  # 不在白名单的项
        if unknown:  # 拼写错误尽早失败,不要等到 Isaac 绑 sensor
            allowed = ", ".join(sorted(CAMERA_ANNOTATORS))  # 给调用方可读白名单
            raise ValueError(f"未知 annotators {unknown};可选: {allowed}")
        return list(value)  # 拷贝,避免调用方事后改原 list

    @field_validator("clipping_range")
    @classmethod
    def _check_clipping_range(cls, value: list[float] | None) -> list[float] | None:
        """None 表示沿用 USD;否则须 ``0 < near < far``.

        Args:
            value: ``[near, far]`` 或 None;未写则保留资产原值.

        Returns:
            ``[near, far]`` 浮点列表;未配置则为 ``None``.
        """
        if value is None:  # 未配置则保留资产 clippingRange
            return None
        # near 须为正且严格小于 far,否则 USD Camera 裁剪非法
        if len(value) != 2 or float(value[0]) <= 0 or float(value[1]) <= float(value[0]):
            raise ValueError("clipping_range 须为 [near, far],且 0 < near < far")
        return [float(value[0]), float(value[1])]  # 统一 float,避免 int/str 混入 USD


class StagePrimConfig(BaseModel):
    """场景包里已有的 prim(``[[scene.objects]]`` 等),只记名字与路径.

    分拣台把场景 USD 引用到 ``[scene].prim_path`` 下,本配置只登记子树里的物体路径.
    """

    name: str = Field(..., description="物体名称,写入 episode meta")
    prim_path: str = Field(..., description="环境 USD 里已存在的 prim 路径")
    grasp_type: str = Field(default="other", description="target=本轮搬运目标;other=其它")


class SortingSceneConfig(BaseConfig):
    """``[scene]``:分拣台场景,把场景包 USD 引用到 ``prim_path`` 下."""

    usd_path: str = Field(
        default="assets/Scene/sorting_station/composed/scene.usdc",
        description="分拣台场景包 USD;挂到 prim_path,不改资产内部尺度",
    )
    prim_path: str = Field(
        default="/World/scene", description="场景引用根;传送带/平台/包裹都在此子树"
    )
    hide_prim_paths: list[str] = Field(
        default_factory=list,
        description="引用后关掉的 prim(如场景包自带的机器人,改由 [robot] 单独引用)",
    )
    platform_prim_path: str = Field(
        default="/World/scene/platform/Empty/Plane/Plane_002", description="平台支撑面 prim"
    )
    conveyor_in_prim_path: str = Field(
        default="/World/scene/ConveyorBelt_A06_01", description="入带 prim;包裹初始位置"
    )
    conveyor_out_prim_path: str = Field(
        default="/World/scene/ConveyorBelt_A06", description="出带 prim;包裹终点"
    )
    objects: list[StagePrimConfig] = Field(
        default_factory=list, description="要跟踪位姿的物体,第一个 target 作为搬运目标"
    )
    cameras: list[SceneCameraConfig] = Field(
        default_factory=list, description="第三人称等场景相机;机载相机写在 [[robot.cameras]]"
    )
