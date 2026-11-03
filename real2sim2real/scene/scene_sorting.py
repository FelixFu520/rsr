"""分拣台场景:把场景包 USD 引用到 ``/World/scene``,登记位姿并绑第三人称相机.

由 ``SortingJob.arrange_scene`` 调用.工厂背景由 SimWorld 挂到 ``/World/background``;
本模块只挂传送带 / 平台 / 包裹所在的场景包,不改引用资产的内部尺度.
场景包里若仍带机器人,经 ``hide_prim_paths`` 关掉,改由 Robot 模块挂到 ``/World/robots``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from real2sim2real.core.logger import LoggerMixin
from real2sim2real.scene.config_sorting import SortingSceneConfig, StagePrimConfig
from real2sim2real.scene.scene_stage import StageScene
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
from real2sim2real.utils.misc import jitter_world_xy
from real2sim2real.utils.pose import read_prim_world_pose

_INFEED_JITTER_MARGIN_M = 0.02
_INFEED_JITTER_MAX_TRIES = 24


class SortingScene(StageScene, LoggerMixin):
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
