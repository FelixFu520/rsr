"""分拣台采集 Job:编排 + 双臂中继搬运 + 落盘 + 校验.

流水线由 ``BaseJob.run`` 固定;本类只覆盖钩子.工厂背景 / 分拣场景 / 机器人
分别引用到 ``/World/background``、``/World/scene``、``/World/robots``.
抓放由 ``SortingPlanner`` 推进,Job 负责开集、抽帧落盘与成败判定.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from real2sim2real.core.logger import LoggerMixin
from real2sim2real.job.config_sorting import SortingJobConfig
from real2sim2real.job.job_collect import CollectJob
from real2sim2real.scene.scene_sorting import SortingScene


class SortingJob(CollectJob, LoggerMixin):
    """分拣台采集:入带 → 平台 → 出带的双臂中继,逐步落盘并校验.

    ``BaseJob.run`` 固定五段;本类只覆盖钩子.本类覆盖标为 [本类].

    完整流程::

        run
          ├─ (构造) IsaacsimSimWorld / SortingScene / ProM6Robot / SortingPlanner 由 BaseJob 装配
          ├─ 1. init_world  # 基类: 工厂背景挂到 /World/background
          ├─ 2. arrange_scene  # SortingScene: 场景包挂到 /World/scene,关掉包内机器人
          ├─ 3. place_robot
          │    ├─ before_place_robot_hook       # [本类] 写入场景包原机位
          │    ├─ robot.init_robot / arrange    # ProM6Robot: 机器人 USD 挂到 /World/robots
          │    └─ planner.init_planner
          ├─ 4. start_simulation  # CollectJob.simulate 循环;本类覆盖目标与开集钩子
          │    ├─ assert_ready_hook          # [本类] 须是 SortingScene
          │    ├─ collect_target_hook        # [本类] 第一个 grasp_type=target
          │    └─ for episode / attempt:
          │         ├─ before_episode_hook            # [本类] 入带 XY 抖动 + 重建 Newton
          │         ├─ after_reset_hook                # [本类] Play 前写回本集抖动
          │         ├─ before_plan_hook                # [本类] 注册物理回调 + 相机热身
          │         ├─ plan_hook               # [本类] 控制器已就绪即可进循环
          │         ├─ stride_hook              # [本类] 按 physics_hz / fps 抽帧
          │         ├─ playing_hook   # [本类] 状态机 DONE 会 Stop 时间轴
          │         └─ retry_hook            # [本类] 恢复静止袋 + 同集抖动 + 重建 Newton
          └─ 5. close  # [本类] 先清控制器再 destroy_world
    """

    config_cls = SortingJobConfig

    def __init__(self, config: SortingJobConfig | Any | None = None, **kwargs: Any) -> None:
        """占位组件;``run`` 的 finally 总会 close.

        Args:
            config: 分拣 job 配置;None 则从 settings 加载.
            **kwargs: 丢弃.

        Returns:
            None.
        """
        del kwargs
        super().__init__(config=config)
        self.config: SortingJobConfig
        seed = int(self.config.object_jitter_seed)
        self._jitter_rng = np.random.default_rng(None if seed == 0 else seed)
        self._object_xy_offset = np.zeros(2, dtype=float)
        self._jitter_prim_path = ""

    # ============= 采集流水线钩子 <==========

    def before_place_robot_hook(self) -> None:
        """``place_robot`` 开头:把场景包里原机位写进机器人配置,再 ``init_robot`` / ``arrange``.

        Returns:
            None.
        """
        assert isinstance(self.scene, SortingScene), "sorting 需要 SortingScene"
        assert self.robot is not None
        if (getattr(self.robot.config, "usd_prim_path", "") or "").strip():
            return  # 子 prim 引用已带场景包原机位,再写会叠平移
        pose = getattr(self.scene, "embedded_robot_pose", None)
        if pose is None:
            return
        from real2sim2real.utils.pose import PoseConfig

        pos, quat = pose
        self.robot.config.pose = PoseConfig(
            position=[float(v) for v in np.asarray(pos, dtype=float).reshape(3)],
            orientation_wxyz=[float(v) for v in np.asarray(quat, dtype=float).reshape(4)],
        )
        self.robot.config.pose_relative_to = "/World"

    def assert_ready_hook(self) -> None:
        """``start_simulation`` 开头:分拣采集须是 ``SortingScene``.

        Returns:
            None.
        """
        assert isinstance(self.scene, SortingScene), "sorting 需要 SortingScene"

    def collect_target_hook(self) -> Any | None:
        """配置里第一个 ``grasp_type=target`` 的物体;没有则整轮放弃.

        Returns:
            搬运目标;场景没有物体则为 ``None``.
        """
        assert isinstance(self.scene, SortingScene)
        target = self.scene.grasp_target()
        if target is None:
            self.log.error("场景没有可搬运的包裹")
            return None
        return target

    def before_episode_hook(self, target: Any, episode: int) -> None:
        """``collect_episodes``:每集开盘之前,相对入带摆放位 XY 抖动包裹.

        必须先恢复静止网格再量 AABB / 重建 Newton:上一集写回的变形表面
        不能再做星形四面体.

        Args:
            target: 本轮搬运目标.
            episode: 1-based 集号;抖动只吃 RNG,不吃集号.

        Returns:
            None.
        """
        del episode
        self._place_clothing_for_episode(target, resample=True)

    def _place_clothing_for_episode(self, target: Any, *, resample: bool) -> None:
        """恢复静止袋 → 入带抖动(或写回本集偏移) → 停时间轴 → 重建 Newton.

        Args:
            target: 本轮搬运目标.
            resample: True 时抽新的 XY 偏移;False 时重试本集已有偏移.

        Returns:
            None.
        """
        from real2sim2real.planner.newton_sorting import restore_cached_rest_mesh

        assert isinstance(self.scene, SortingScene)
        assert self.planner is not None
        assert self.sim_world is not None and self.sim_world.stage is not None
        surface = str(self.planner.config.clothing_surface_prim_path)
        restore_cached_rest_mesh(self.sim_world.stage, surface)
        self._jitter_prim_path = target.prim_path
        if resample:
            self._object_xy_offset = self.scene.apply_object_jitter(
                target.prim_path,
                float(self.config.object_jitter_xy),
                self._jitter_rng,
                clothing_surface_prim_path=surface,
            )
        else:
            self.scene.restore_teleported_object(target.prim_path)
        self.sim_world.stop()
        self.planner.reset()

    def after_reset_hook(self) -> None:
        """``reset`` 可能把包裹 xform 打回 USD 原位,Play 前写回本集抖动.

        Returns:
            None.
        """
        if self._jitter_prim_path and self.scene is not None:
            self.scene.restore_teleported_object(self._jitter_prim_path)

    def before_plan_hook(self, target: Any, *, episode: int, attempt: int) -> None:
        """``reset_and_play`` 之后:先注册状态机回调,再热身相机.

        顺序不能反:回调注册后双臂立刻被写到 ready 位,热身期间才不会因重力下垂.

        Args:
            target: 本轮搬运目标;热身不依赖它.
            episode: 1-based 集号.
            attempt: 1-based 尝试序号.

        Returns:
            None.
        """
        del target
        assert self.planner is not None
        assert self.robot is not None
        self.planner.begin_episode(episode - 1, attempt=attempt)  # 规划器要 0-based 集号
        self.robot.post_physics_init(warmup_steps=self.config.camera_warmup_steps)

    def plan_hook(self, target: Any, *, episode: int, attempt: int) -> bool:
        """本任务无独立规划段:控制器已装配即可进入执行循环.

        Args:
            target: 本轮搬运目标.
            episode: 1-based 集号.
            attempt: 1-based 尝试序号.

        Returns:
            规划器已就绪则为 True.
        """
        del target, episode, attempt
        assert self.planner is not None
        if not self.planner.ready:
            self.log.error("分拣规划器未就绪")
            return False
        return True

    def stride_hook(self) -> int:
        """抽帧间隔:显式配置优先,否则按 ``physics_hz / recorder.fps`` 计算.

        Returns:
            至少 1 的物理步间隔.
        """
        if self.config.record_stride > 0:
            return self.config.record_stride
        fps = max(1, int(self.config.recorder.fps))
        return max(1, round(float(self.config.physics_hz) / fps))

    def playing_hook(self) -> bool:
        """状态机 DONE / FAILED 时会 Stop 时间轴,据此结束本集.

        Returns:
            True.
        """
        return True

    def episode_fields_hook(self, target: Any, episode: int, attempt: int) -> dict[str, Any]:
        """写入 recorder meta / extra 的任务字段.

        Args:
            target: 本轮搬运目标.
            episode: 1-based 集号.
            attempt: 1-based 尝试序号.

        Returns:
            ``sorting_object`` 与 ``object_xy_offset``;集号/尝试由外层补.
        """
        del episode, attempt
        return {"sorting_object": target.name, "object_xy_offset": self._object_xy_offset.tolist()}

    def retry_hook(self, target: Any) -> None:
        """校验失败或未验收时:恢复静止袋、写回本集抖动、重建 Newton.

        Args:
            target: 本轮搬运目标.

        Returns:
            None.
        """
        self._place_clothing_for_episode(target, resample=False)

    def close(self) -> None:
        """先清控制器留下的临时 prim 与碰撞覆盖,再销毁 World.

        Returns:
            None.
        """
        planner = self.planner
        cleanup = getattr(planner, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception as exc:  # noqa: BLE001  # 收尾失败不阻断 destroy_world
                self.log.warning("清理分拣规划器失败", error=str(exc))
        super().close()
