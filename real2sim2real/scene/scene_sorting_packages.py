"""分拣包裹后端：配置与实现。"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import get_logger
from real2sim2real.scene.base import BaseScene
from real2sim2real.simworld.base import BaseSimWorld
from real2sim2real.utils.isaacsim_utils import (
    bind_camera_sensor,
    ensure_xform,
    normalize_prim_path,
    place_asset,
    prim_aabb,
    pump_frames,
    read_sensor_data,
    resolve_usd_path,
    spawn_camera,
    spawn_reference,
)
from real2sim2real.utils.isaacsim_utils.camera_utils import CAMERA_ANNOTATORS
from real2sim2real.utils.misc import jitter_world_xy
from real2sim2real.utils.pose import PoseConfig as Pose
from real2sim2real.utils.pose import read_prim_world_pose


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


_INFEED_JITTER_MARGIN_M = 0.02

_INFEED_JITTER_MAX_TRIES = 24


class SortingScene(BaseScene):
    """分拣台场景:``init_scene`` 挂 USD,``arrange`` 按 pose 摆位.

    ``arrange_scene``(分拣 Job)::

        AutoScene.from_config()
        init_scene(sim_world)                  # 把场景 USD 挂到 /World/scene
        arrange()                              # 登记位姿、隐藏包内机器人、绑相机
          └─ seed_poses_realtime()
    """

    config_cls = SortingSceneConfig

    def __init__(self, config: SortingSceneConfig | Any | None = None, **kwargs: Any) -> None:
        """加载 ``[scene]`` 配置;``sim_world`` 在 ``init_scene`` 时绑定.

        Args:
            config: 已构造的分拣场景配置;None 则从 settings 的 ``scene`` 段加载.
            **kwargs: 工厂多余字段,本后端不用.

        Returns:
            None.
        """
        super().__init__()
        self._teleport_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        del kwargs
        self.config: SortingSceneConfig = config or self.config_cls.from_settings("scene")
        self.embedded_robot_pose: tuple[np.ndarray, np.ndarray] | None = None

    # ============= 场景流水线 <==========

    def init_scene(self, sim_world: BaseSimWorld | None = None) -> None:
        """把场景包 USD 引用到 ``prim_path``,再校验关键 prim.

        Args:
            sim_world: 已 ``init_world`` 的 Isaac 世界;必须带可用 stage.

        Returns:
            None.

        Raises:
            RuntimeError: 未配置 usd_path,或引用后仍缺少平台 / 传送带 / 物体.
        """

        self.sim_world = sim_world
        assert self.sim_world is not None, "Sim world is not initialized"
        assert self.sim_world.stage is not None, "Stage is not initialized"
        from isaacsim.core.utils.stage import add_reference_to_stage

        ensure_xform(self.sim_world.stage, self.config.prim_path)  # /World/scene,与背景/机器人兄弟

        usd = resolve_usd_path(self.config.usd_path)
        spawn_reference(self.sim_world.stage, self.config.prim_path, usd, add_reference_to_stage)
        pump_frames(self.sim_world, 4)
        self.log.info("引用分拣台场景成功", prim_path=self.config.prim_path, usd=usd)

        for cam in self.config.cameras:
            spawn_camera(self.sim_world.stage, cam.prim_path, cam.usd_path, add_reference_to_stage)
            pump_frames(self.sim_world, 1)
        if self.config.cameras:
            self.log.info("加载场景相机成功", cameras=[cam.name for cam in self.config.cameras])

        missing = [path for path in self.stage_prim_paths() if not self._prim_valid(path)]
        if missing:
            raise RuntimeError(f"场景 USD 缺少分拣台 prim: {missing}")

    def arrange(self) -> None:
        """登记工位位姿,关掉场景包自带机器人,再摆第三人称相机.

        Returns:
            None.
        """
        for path in self.stage_prim_paths():
            pose = read_prim_world_pose(self.sim_world.stage, path)
            if pose is not None:
                self.poses[normalize_prim_path(path)] = pose
        self.log.info("登记分拣台位姿成功", n_prims=len(self.poses))

        self._hide_embedded_robot()

        for cam in self.config.cameras:
            pos, _ = place_asset(
                self.sim_world.stage,
                self.poses,
                cam.prim_path,
                cam.pose,
                cam.pose_relative_to,
            )
            self.sensors[cam.name] = bind_camera_sensor(self.sim_world.stage, cam)
            pump_frames(self.sim_world, 1)
            self.log.info(
                "摆放场景相机成功",
                name=cam.name,
                prim_path=cam.prim_path,
                position=pos.tolist(),
            )

        self.seed_poses_realtime()

    # ============= 辅助函数 <==========

    def _hide_embedded_robot(self) -> None:
        """关掉场景包里自带的机器人,并把其世界位姿留给 Job 去摆独立引用.

        Returns:
            None.
        """
        for raw in self.config.hide_prim_paths:
            path = normalize_prim_path(raw)
            prim = self.sim_world.stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                self.log.warning("要隐藏的 prim 不存在", prim_path=path)
                continue
            pose = read_prim_world_pose(self.sim_world.stage, path)
            if pose is not None and self.embedded_robot_pose is None:
                self.embedded_robot_pose = pose
            prim.SetActive(False)
            self.log.info("已关闭场景包内 prim", prim_path=path)

    def stage_prim_paths(self) -> list[str]:
        """平台 / 两条传送带 / 全部物体的 prim 路径.

        Returns:
            prim 路径列表,顺序固定:平台 → 入带 → 出带 → 物体.
        """
        paths = [
            self.config.platform_prim_path,
            self.config.conveyor_in_prim_path,
            self.config.conveyor_out_prim_path,
        ]
        paths.extend(obj.prim_path for obj in self.config.objects)
        return [path for path in paths if path]

    def _prim_valid(self, prim_path: str) -> bool:
        """stage 上是否存在该 prim.

        Args:
            prim_path: USD prim 路径.

        Returns:
            存在且有效为 True.
        """
        prim = self.sim_world.stage.GetPrimAtPath(normalize_prim_path(prim_path))
        return bool(prim is not None and prim.IsValid())

    def _infeed_surface_path(self) -> str:
        """入带优先用 ``.../Belt`` 表面,避免整段机架 AABB 把抖动范围撑得过大."""
        raw = normalize_prim_path(self.config.conveyor_in_prim_path)
        belt = raw.rstrip("/") + "/Belt"
        return belt if self._prim_valid(belt) else raw

    def apply_object_jitter(
        self,
        prim_path: str,
        radius: float,
        rng: np.random.Generator,
        *,
        clothing_surface_prim_path: str = "",
        margin_m: float = _INFEED_JITTER_MARGIN_M,
        max_tries: int = _INFEED_JITTER_MAX_TRIES,
        previous_offset: np.ndarray | None = None,
        min_distance_m: float = 0.0,
    ) -> np.ndarray:
        """相对 ``poses`` 摆放位做入带 XY 抖动,写入 stage 与 ``poses_realtime``.

        采样会把包裹世界 AABB 限制在入带 XY 内(加边距),避免掉到皮带外.
        调用前须已把袋表面恢复成静止网格,否则 AABB 是上一集变形结果.

        Args:
            prim_path: 包裹根 prim 路径.
            radius: XY 半边长(米);<=0 则不偏移.
            rng: 随机数生成器,保证多集可复现.
            clothing_surface_prim_path: 用于量 AABB 的可视 Mesh;空则用根 prim.
            margin_m: 抖动后 AABB 相对入带 XY 的边距(米).
            max_tries: 越界则重采样次数;用尽则退回摆放位.
            previous_offset: 上一成功集相对原位的 XY 偏移.
            min_distance_m: 相对上一成功集的最小 XY 间距;不可满足时抛错.

        Returns:
            ``(dx, dy)`` 米.

        Raises:
            AssertionError: 该 prim 尚未 ``arrange``,或 stage 未绑定.
        """
        path = normalize_prim_path(prim_path)
        base = self.poses.get(path)
        assert base is not None, f"尚未摆放: {path}"
        assert self.sim_world is not None and self.sim_world.stage is not None
        stage = self.sim_world.stage
        base_pos = np.asarray(base[0], dtype=float)
        base_quat = np.asarray(base[1], dtype=float)
        self.teleport_object(path, base_pos, base_quat)
        if radius <= 0:
            return np.zeros(2, dtype=float)

        aabb_path = normalize_prim_path(clothing_surface_prim_path or path)
        clothing_aabb = prim_aabb(stage, aabb_path, gprims_only=True)
        infeed_aabb = prim_aabb(stage, self._infeed_surface_path(), gprims_only=True)
        assert clothing_aabb is not None, f"量不出包裹 AABB: {aabb_path}"
        clothing_min, clothing_max = clothing_aabb
        in_min = in_max = None
        if infeed_aabb is not None:
            in_min, in_max = infeed_aabb
            in_half = 0.5 * (in_max[:2] - in_min[:2])
            if np.any(in_half > 2.0):
                self.log.warning(
                    "入带 AABB 过大,抖动只按半径限制",
                    half_xy=in_half.tolist(),
                    infeed=self._infeed_surface_path(),
                )
                in_min = in_max = None

        margin = float(margin_m)
        chosen = (base_pos.copy(), 0.0, 0.0)
        for _ in range(max(1, int(max_tries))):
            pos, dx, dy = jitter_world_xy(base_pos, float(radius), rng)
            if (
                previous_offset is not None
                and np.linalg.norm(np.array([dx, dy]) - previous_offset) < min_distance_m
            ):
                continue
            if in_min is not None and in_max is not None:
                new_min = clothing_min.copy()
                new_max = clothing_max.copy()
                new_min[0] += dx
                new_min[1] += dy
                new_max[0] += dx
                new_max[1] += dy
                inside = (
                    float(new_min[0]) >= float(in_min[0]) + margin
                    and float(new_max[0]) <= float(in_max[0]) - margin
                    and float(new_min[1]) >= float(in_min[1]) + margin
                    and float(new_max[1]) <= float(in_max[1]) - margin
                )
                if not inside:
                    continue
            chosen = (pos, dx, dy)
            break
        else:
            if previous_offset is not None and min_distance_m > 0:
                raise RuntimeError("入带范围内未采到满足最小间距的初始位置，请减小最小间距")
            self.log.warning(
                "入带抖动多次越界,退回摆放位",
                radius=float(radius),
                tries=int(max_tries),
            )
            chosen = (base_pos.copy(), 0.0, 0.0)

        pos, dx, dy = chosen
        self.teleport_object(path, pos, base_quat)
        self.log.info(
            "包裹入带抖动",
            prim=path,
            offset_xy=[round(float(dx), 4), round(float(dy), 4)],
        )
        return np.array([dx, dy], dtype=float)

    def grasp_target(self) -> StagePrimConfig | None:
        """配置里第一个 ``grasp_type=target`` 的物体;没有则退回第一个物体.

        Returns:
            搬运目标物体配置;场景没有物体则为 ``None``.
        """
        for obj in self.config.objects:
            if obj.grasp_type == "target":
                return obj
        return self.config.objects[0] if self.config.objects else None

    def goal_object(self) -> StagePrimConfig | None:
        """终点物体;分拣任务与搬运目标同一件.

        Returns:
            搬运目标物体配置;场景没有物体则为 ``None``.
        """
        return self.grasp_target()

    def layout_prim_paths(self) -> list[str]:
        """视口框选:场景根 + 分拣台三大件 + 物体 + 场景相机.

        Returns:
            prim 路径列表.
        """
        paths = [self.config.prim_path, *self.stage_prim_paths()]
        paths.extend(cam.prim_path for cam in self.config.cameras)
        return paths

    def get_meta(self) -> dict[str, Any]:
        """写入 episode meta 的场景快照.

        Returns:
            场景后端、USD 与物体清单.
        """
        return {
            "scene_backend": self.config.backend,
            "scene_usd": self.config.usd_path,
            "scene_objects": [obj.name for obj in self.config.objects],
        }

    def get_image(self, name: str, annotator: str = "rgb"):  # noqa: ANN201
        """按相机名取一帧;未绑定该相机则为 ``None``.

        Args:
            name: ``sensors`` 里的相机名.
            annotator: ``rgb`` / ``distance_to_image_plane`` 等已绑定项.

        Returns:
            numpy 数组;未绑定或读失败则为 ``None``.
        """
        sensor = self.sensors.get(name)
        if sensor is None:
            return None
        return read_sensor_data(sensor, annotator=annotator, name=name)

    def seed_poses_realtime(self) -> None:
        """用 ``poses`` 覆盖 ``poses_realtime``;``arrange`` 结束时调用.

        Returns:
            None.
        """
        self.poses_realtime = dict(self.poses)

    def get_prim_pose(self, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``arrange`` 后缓存的世界位姿.

        Args:
            prim_path: USD prim 路径;会先 ``normalize_prim_path`` 再查 ``poses``.

        Returns:
            ``(世界 pos, wxyz)`` 元组.

        Raises:
            KeyError: 该 prim 尚未摆放.
        """
        pose = self.poses.get(normalize_prim_path(prim_path))  # 键与 arrange 写入时一致
        if pose is None:
            raise KeyError(f"尚未摆放: {prim_path}")  # 调用方应先 arrange
        return pose  # (世界 pos, wxyz)

    def set_object_world_xy(
        self, prim_path: str, xy: np.ndarray, *, z: float | None = None
    ) -> None:
        """``arrange_scene`` 首次、每集重采样后、以及 ``_run_episode`` 的 reset 之后调用.

        Args:
            prim_path: 物体 USD prim 路径.
            xy: 世界系 ``[x, y]``.
            z: 显式世界 z;None 则沿用缓存高度或 ``floor_z``.
        """
        del prim_path, xy, z  # 基类无物体,子类覆盖真正挪物
        return None

    def teleport_object(
        self,
        prim_path: str,
        pos: np.ndarray,
        quat: np.ndarray | None = None,
    ) -> None:
        """把物体写到世界位姿,只更新 ``poses_realtime``,不改 arrange 快照.

        Args:
            prim_path: 物体 USD prim 路径.
            pos: 世界平移 ``(x, y, z)``.
            quat: 世界朝向 ``wxyz``;``None`` 则沿用 realtime / arrange 缓存.
        """
        from real2sim2real.utils.isaacsim_utils import apply_pose

        path = normalize_prim_path(prim_path)
        if quat is None:
            cached = self.poses_realtime.get(path) or self.poses.get(path)
            if cached is None:
                raise KeyError(f"尚未摆放: {prim_path}")
            quat = cached[1]
        pos_arr = np.asarray(pos, dtype=float).reshape(3).copy()
        quat_arr = np.asarray(quat, dtype=float).reshape(4).copy()
        stage = None if self.sim_world is None else getattr(self.sim_world, "stage", None)
        if stage is not None:
            apply_pose(stage, path, pos_arr, quat_arr)
        pose = (pos_arr, quat_arr)
        self.poses_realtime[path] = pose
        self._teleport_poses[path] = pose

    def restore_teleported_object(self, prim_path: str) -> None:
        """写回最近一次 ``teleport_object``;没有则退回 ``poses`` 摆放位.

        Args:
            prim_path: 物体 USD prim 路径.
        """
        path = normalize_prim_path(prim_path)
        pose = self._teleport_poses.get(path) or self.poses.get(path)
        if pose is None:
            return
        self.teleport_object(path, pose[0], pose[1])

    def refresh_poses_realtime(self, prim_paths: list[str] | None = None) -> None:
        """从 stage 读世界位姿写入 ``poses_realtime``.

        Args:
            prim_paths: 要刷新的 prim;``None`` 则刷新 ``poses`` 的全部键.
        """
        if self.sim_world is None or getattr(self.sim_world, "stage", None) is None:
            return
        from real2sim2real.utils.pose import read_prim_world_pose

        keys = prim_paths if prim_paths is not None else list(self.poses.keys())
        for raw in keys:
            path = normalize_prim_path(raw)
            pose = read_prim_world_pose(self.sim_world.stage, path)
            if pose is not None:
                self.poses_realtime[path] = pose

    def get_prim_pose_realtime(self, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
        """返回 play 中的当前世界位姿;缺 realtime 则回退 ``poses``.

        Args:
            prim_path: USD prim 路径;会先 ``normalize_prim_path``.

        Returns:
            ``(世界 pos, wxyz)`` 元组.

        Raises:
            KeyError: ``poses_realtime`` 与 ``poses`` 都没有该 prim.
        """
        path = normalize_prim_path(prim_path)
        pose = self.poses_realtime.get(path) or self.poses.get(path)
        if pose is None:
            raise KeyError(f"尚未摆放: {prim_path}")
        return pose

    @property
    def log(self) -> Any:
        """返回绑定组件名的日志器。"""
        return get_logger(f"{type(self).__module__}.{type(self).__qualname__}").bind(
            component=type(self).__qualname__
        )
