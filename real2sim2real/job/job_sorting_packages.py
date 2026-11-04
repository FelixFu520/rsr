"""分拣包裹后端：配置与实现。"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
from pydantic import Field, model_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.job.base import BaseJob
from real2sim2real.planner.planner_sorting_packages import restore_cached_rest_mesh
from real2sim2real.scene.scene_sorting_packages import SortingScene
from real2sim2real.utils.recorder import LeRobotRecorder, LeRobotV2Recorder, create_recorder
from real2sim2real.utils.recorder_config import JobRecorderConfig
from real2sim2real.utils.validate import validate_episode


class SortingJobConfig(BaseConfig):
    """``[job]``:分拣台双臂搬运采集配置."""

    inspect_gui: bool = Field(default=False, description="是否在各流水线阶段后进入 GUI 可编辑等待")
    num_episodes: int = Field(default=1, ge=1, description="本轮连续采集多少条 episode")
    object_jitter_xy: float = Field(
        default=0.06,
        ge=0.0,
        allow_inf_nan=False,
        description="每集包裹相对入带摆放位的 XY 随机偏移半边长(m);0 则不抖",
    )
    object_jitter_seed: int = Field(default=0, description="偏移 RNG 种子;0 表示不固定")
    object_jitter_min_distance_m: float = Field(
        default=0.04,
        ge=0.0,
        allow_inf_nan=False,
        description="本轮相邻成功集初始包裹位置最小 XY 间距(m);0 关闭",
    )
    max_attempts: int = Field(default=4, ge=1, le=20, description="每集重采样/执行次数上限")

    @model_validator(mode="after")
    def _sampling_distance(self) -> SortingJobConfig:
        if self.object_jitter_xy > 0 and self.object_jitter_min_distance_m >= (
            2 * np.sqrt(2) * self.object_jitter_xy
        ):
            raise ValueError("初始位置最小间距不能大于扰动区域的直径")
        return self

    camera_warmup_steps: int = Field(
        default=30, description="Play 后热身步数;关节由规划器接管,这里只等相机出帧"
    )
    max_steps: int = Field(
        default=6000, description="落盘帧数硬上限;分拣一趟约需数千物理步,按抽帧后计"
    )
    record_stride: int = Field(
        default=0,
        ge=0,
        description="落盘抽帧间隔(物理步);0 表示按 physics_hz / recorder.fps 自动计算",
    )
    physics_hz: float = Field(
        default=60.0, gt=0.0, description="物理频率,自动抽帧时与 [sim_world].physics_dt 对齐"
    )
    viewport_camera_prims: list[str] = Field(
        default_factory=list,
        description="GUI 视口相机 prim/名;空=全部已绑定传感器",
    )
    recorder: JobRecorderConfig = Field(
        default_factory=lambda: JobRecorderConfig(
            output_dir="outputs/sorting_pro_m6", robot_type="pro_m6"
        ),
        description="落盘配置,见 [job.recorder]",
    )


class SortingJob(BaseJob):
    """分拣台采集:入带 → 平台 → 出带的双臂中继,逐步落盘并校验.

    ``BaseJob.run`` 固定五段;采集逻辑与任务钩子均在本类中实现。

    完整流程::

        run
          ├─ (构造) IsaacsimSimWorld / SortingScene / ProM6Robot / SortingPlanner 由 BaseJob 装配
          ├─ 1. init_world  # 基类: 工厂背景挂到 /World/background
          ├─ 2. arrange_scene  # SortingScene: 场景包挂到 /World/scene,关掉包内机器人
          ├─ 3. place_robot
          │    ├─ before_place_robot_hook       # [本类] 写入场景包原机位
          │    ├─ robot.init_robot / arrange    # ProM6Robot: 机器人 USD 挂到 /World/robots
          │    └─ planner.init_planner
          ├─ 4. start_simulation  # simulate 循环;本类覆盖目标与开集钩子
          │    ├─ assert_ready_hook          # [本类] 须是 SortingScene
          │    ├─ collect_target_hook        # [本类] 第一个 grasp_type=target
          │    └─ for episode / attempt:
          │         ├─ before_episode_hook            # [本类] 入带 XY 抖动 + 重建 Newton
          │         ├─ after_reset_hook                # [本类] Play 前写回本集抖动
          │         ├─ before_plan_hook                # [本类] 注册物理回调 + 相机热身
          │         ├─ plan_hook               # [本类] 控制器已就绪即可进循环
          │         ├─ stride_hook              # [本类] 按 physics_hz / fps 抽帧
          │         ├─ playing_hook   # [本类] 状态机 DONE 会 Stop 时间轴
          │         └─ retry_hook            # [本类] 恢复静止袋 + 重采样位置 + 重建 Newton
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
        self.recorder: LeRobotRecorder | None = None
        self.last_stats: dict[str, Any] = {}
        self.config: SortingJobConfig
        seed = int(self.config.object_jitter_seed)
        self._jitter_rng = np.random.default_rng(None if seed == 0 else seed)
        self._object_xy_offset = np.zeros(2, dtype=float)
        self._previous_object_xy_offset: np.ndarray | None = None
        self._object_initial_pose: dict[str, Any] = {}
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
                previous_offset=self._previous_object_xy_offset,
                min_distance_m=self.config.object_jitter_min_distance_m,
                max_tries=128,
            )
        else:
            self.scene.restore_teleported_object(target.prim_path)
        position, orientation = self.scene.get_prim_pose_realtime(target.prim_path)
        orientation_norm = float(np.linalg.norm(orientation))
        if not np.isfinite(orientation_norm) or orientation_norm < 1e-12:
            raise RuntimeError("初始包裹朝向不是有效四元数")
        self._object_initial_pose = {
            "object_initial_position": position.tolist(),
            "object_initial_orientation_wxyz": (orientation / orientation_norm).tolist(),
        }
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
        return {
            "sorting_object": target.name,
            "object_xy_offset": self._object_xy_offset.tolist(),
            **self._object_initial_pose,
        }

    def retry_hook(self, target: Any) -> None:
        """校验失败或未验收时:恢复静止袋、重采样初始位置、重建 Newton.

        Args:
            target: 本轮搬运目标.

        Returns:
            None.
        """
        self._place_clothing_for_episode(target, resample=True)

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
        exc = sys.exc_info()[1]
        if exc is not None:
            self.log.error("采集流程异常", error=str(exc), exc_info=exc)
        super().close()

    _MIN_CAMERA_FRAMES = 5

    def simulate(self) -> None:
        """视口 → recorder → 目标 → 按集 / 尝试循环 → 汇总.

        无 ``[job.recorder]`` 或 ``collect_target_hook`` 返回 None 时直接返回,不打汇总.

        Returns:
            None.写 ``last_stats`` 与 recorder 产物.
        """
        assert self.sim_world is not None, "sim_world 未初始化"
        self.assert_ready_hook()  # 默认空;子类查场景类型

        self.show_layout_cameras()  # 把场景 / 机器人 prim 框进默认视口
        self.sim_world.open_camera_viewports(self.camera_viewport_targets())  # GUI 相机窗

        if not self._setup_recorder():  # 无 [job.recorder] 则整轮不采集
            return

        target = self.collect_target_hook()  # 子类给搬运目标;默认 None
        if target is None:  # 没有可采目标则整轮放弃
            return

        n_ok, n_timeout, n_ep = self.collect_episodes(target)
        self._log_collect_finished(n_ok=n_ok, n_timeout=n_timeout, n_ep=n_ep)
        self.after_collect_hook(n_ok=n_ok, n_timeout=n_timeout, n_ep=n_ep)

    def show_layout_cameras(self) -> None:
        """把场景 / 机器人相关 prim 框进默认视口,便于 GUI 检查布局.

        Returns:
            None.
        """
        cam_paths: list[str] = []  # 待框进默认视口的 prim;先场景后机器人
        if self.scene is not None:  # init_world 后、arrange 前可能仍无 scene
            cam_paths.extend(self.scene.layout_prim_paths())  # 桌子/物体/导航道具
        if self.robot is not None:  # place_robot 前可能仍无 robot
            layout = getattr(self.robot, "layout_prim_paths", None)  # 非 ABC,机型自带
            if callable(layout):  # 有的 stub 机器人没有该方法
                cam_paths.extend(layout())  # 机械臂/腕部相机
        cam_paths = [p for p in cam_paths if p]  # 丢掉空字符串,避免 Kit 报无效 prim
        if self.sim_world is not None:  # stub 基类 show_cameras 为 no-op
            self.sim_world.show_cameras(cam_paths)  # 把 prim 框进默认视口

    def camera_viewport_targets(self) -> dict[str, str]:
        """按 ``[job].viewport_camera_prims`` 筛相机给 GUI 视口;空=全部.

        Returns:
            相机名到 prim 路径;读不到的相机会跳过.
        """
        from real2sim2real.utils.isaacsim_utils import filter_camera_targets, iter_bound_cameras

        prims = list(getattr(self.config, "viewport_camera_prims", None) or [])  # 空=全部
        return filter_camera_targets(  # 按 prim 白名单筛;读不到的相机会跳过
            iter_bound_cameras(self.scene, self.robot),  # 先场景相机,后机器人相机
            prims,
        )

    def _setup_recorder(self) -> bool:
        """按 ``[job.recorder]`` 创建落盘器.

        Returns:
            已创建则为 True;配置无 ``recorder`` 则为 False.
        """
        recorder_cfg = getattr(self.config, "recorder", None)  # stub / 未配则没有落盘
        if recorder_cfg is None:  # 无 [job.recorder] 时整轮不采集
            return False  # start_simulation 据此直接返回
        self.recorder = create_recorder(recorder_cfg, camera_names=self.expected_camera_names())
        return True  # 已写入 self.recorder,后续按集 open/finalize

    def expected_camera_names(self) -> list[str]:
        """按 ``[job.recorder].record_camera_prims`` 筛落盘相机;空=全部.

        Returns:
            本集应落盘的相机名列表;先场景后机器人.
        """
        from real2sim2real.utils.isaacsim_utils import filter_camera_targets, iter_bound_cameras

        recorder = getattr(self.config, "recorder", None)  # 无 recorder 时下面 prims 为空
        prims = list(getattr(recorder, "record_camera_prims", None) or [])  # 空=全部
        return list(  # 转 list:落盘 schema / validate 都要稳定顺序
            filter_camera_targets(
                iter_bound_cameras(self.scene, self.robot),  # 先场景后机器人,与视口一致
                prims,
            )
        )

    def collect_episodes(self, target: Any) -> tuple[int, int, int]:
        """按集开盘、尝试、计数.子类不覆盖本方法.

        流程::

            begin_collect_hook()
            n_ep = episode_count_hook()
            for episode in 1..n_ep:
                stop_collect_hook → break
                before_episode_hook(target, episode)
                outcome = _collect_one_episode(target, episode)
                stop_collect_hook → break   # 不把本集计入成功/timeout
                # outcome: success / timeout / failed

        Args:
            target: ``collect_target_hook`` 返回的本轮目标物.

        Returns:
            ``(成功集数, 验收为 timeout 的集数, 计划集数)``.
        """
        self.begin_collect_hook()  # 默认空;搜拍在此清退出标志
        n_ep = self.episode_count_hook()  # 默认 num_episodes;搜拍用 max_missions
        n_ok = 0  # 校验通过且 status==success 的集数
        n_timeout = 0  # 导航等验收为 timeout 的集数(不重试)
        for ep_i in range(1, n_ep + 1):  # 1-based,写入 recorder meta.episode
            if self.stop_collect_hook():  # 默认 False;搜拍对话退出则停
                break
            self.before_episode_hook(target, ep_i)  # 默认空;桌面抖动 / 导航重采样
            outcome = self._collect_one_episode(target, ep_i)  # success/timeout/failed
            if self.stop_collect_hook():  # 本集中途退出不计入成功或 timeout
                break
            if outcome == "success":  # 计入成功,进入下一集
                n_ok += 1
                self._previous_object_xy_offset = self._object_xy_offset.copy()
            elif outcome == "timeout":  # 仅验收路径会返回 timeout
                n_timeout += 1
        return n_ok, n_timeout, n_ep  # 计划集数仍是 episode_count_hook 的返回值

    def begin_collect_hook(self) -> None:
        """``collect_episodes`` 开头.默认空.

        Returns:
            None.
        """

    def episode_count_hook(self) -> int:
        """``collect_episodes`` 计划集数.默认读 ``num_episodes``.

        Returns:
            至少 1 的计划集数;缺省为 1.
        """
        return int(getattr(self.config, "num_episodes", 1))  # 搜拍改读 max_missions

    def stop_collect_hook(self) -> bool:
        """``collect_episodes`` 每集开始前、以及计数前.默认继续.

        Returns:
            True 则停下,且计数前停下时本集不计入成功或 timeout.
        """
        return False  # 搜拍在对话退出时返回 True

    def _collect_one_episode(self, target: Any, ep_i: int) -> str:
        """单集:打开 recorder → 执行 → 落盘校验 → 成功 / 验收 / 重试.

        流程(每 attempt)::

            episode_fields_hook → recorder.open
              → run_episode                     # 异常记为 error,不炸整轮
              → recorder.finalize
              → validate_episode(require_success_hook)
              ├─ success 且校验通过 → 返回 success
              ├─ 校验失败 → retry_hook, 下一 attempt
              ├─ accept_hook → accepted_hook, 验收
              └─ 否则 retry_hook, 下一 attempt

        Args:
            target: 本轮目标物,原样传给 ``run_episode`` / ``retry_hook``.
            ep_i: 1-based 集号.

        Returns:
            ``success`` / ``timeout`` / ``failed``.
        """
        assert self.recorder is not None, "recorder 未初始化"  # start_simulation 已建
        last_error: str | None = None  # 跨 attempt 保留,写入下一份 extra
        max_attempts = self.config.max_attempts
        for attempt in range(1, max_attempts + 1):
            self.last_stats = {}
            task_fields = self.episode_fields_hook(target, ep_i, attempt)  # 抖动/起终点等
            episode_dir = self.recorder.open(  # 开新集目录,返回路径供 validate
                meta={  # 写入本集任务快照
                    **self.episode_meta(),  # Job/scene/robot/planner 公共元数据
                    "episode": ep_i,  # 1-based 集号
                    "attempt": attempt,  # 1-based 重试序号
                    **task_fields,  # 任务字段追加;勿重复写 episode/attempt
                }
            )

            try:
                status = self.run_episode(target, episode=ep_i, attempt=attempt)
                if status == "success":
                    last_error = None
            except Exception as exc:  # noqa: BLE001  # 单次异常不炸整个 Job
                last_error = f"{type(exc).__name__}: {exc}"  # 留给 finalize.extra
                status = "error"  # 不当 success,走校验/重试
                self.log.exception("采集异常", episode=ep_i, attempt=attempt, error=last_error)

            extra: dict[str, Any] = dict(self.last_stats)  # 循环统计;run_episode 已写入
            extra.update({"episode": ep_i, "attempt": attempt, **task_fields})  # 与 meta 对齐
            if last_error:  # 运行异常或上一轮校验错误
                extra["error"] = last_error
            if status != "success" and isinstance(self.recorder, LeRobotV2Recorder):
                extra["status"] = status
                extra.update(self.planner.get_meta())
                self.recorder.discard(extra=extra)
                self.log.warning("丢弃未成功尝试，重新采样", episode=ep_i, attempt=attempt)
                if attempt < max_attempts:
                    self.retry_hook(target)
                continue
            recorder_cfg = getattr(self.config, "recorder", None)  # 取导出视频开关
            self.recorder.finalize(  # 关本集、写 status/extra
                status=status,
                extra=extra,
                export_video=bool(getattr(recorder_cfg, "export_video", False)),
                video_fps=int(getattr(recorder_cfg, "fps", 15)),
            )

            report = validate_episode(  # 查步数/每路相机帧;桌面还要求 success
                episode_dir,
                expected_cameras=self.expected_camera_names(),
                min_steps=1,
                min_frames_per_camera=self._MIN_CAMERA_FRAMES,
                require_success_status=self.require_success_hook(),
            )

            if status == "success" and report.ok:  # 规划成功且落盘合格
                self.log.info(
                    "采集本集成功",
                    episode=ep_i,
                    attempt=attempt,
                    episode_dir=str(episode_dir),
                )
                return "success"  # 计入 n_ok,不再重试

            if not report.ok:  # 帧数不够或相机缺失,本 attempt 作废
                last_error = "; ".join(report.errors)  # 带入下一 attempt extra
                self.log.warning(
                    "采集本次验收失败",
                    episode=ep_i,
                    attempt=attempt,
                    errors=report.errors,
                    failure_reason=extra.get("failure_reason"),
                )
                if attempt < max_attempts:  # 还有重试额度才复位
                    self.retry_hook(target)  # 子类清 planner / 回起点 / 回写物体
                continue  # 落盘不合格不走验收,直接下一 attempt

            if self.accept_hook(status):  # 导航 timeout 验收本集
                self.accepted_hook(status, episode=ep_i)  # 打 timeout 警告
                return status  # 通常是 timeout,计入 n_timeout

            if attempt < max_attempts:  # 未验收且还能重试
                self.retry_hook(target)  # 清状态后再跑一次
                last_error = None  # 本轮无落盘错误,下一 extra 不再带旧 error
        else:  # for 正常走完(未 return),所有 attempt 都失败
            self.log.error("采集本集失败", episode=ep_i, attempts=max_attempts)
        return "failed"  # 用尽 attempt 或不验收且无更多重试

    def episode_meta(self) -> dict[str, Any]:
        """写入 recorder 的任务快照:后端、输出目录、相机,以及 scene/robot/planner.

        Returns:
            写入 recorder 的元数据字典.stub 无 ``recorder`` 时 ``output_dir`` 为 None.
        """
        recorder = getattr(self.config, "recorder", None)  # stub 无 recorder 时目录为 None
        meta: dict[str, Any] = {
            "job_backend": getattr(self.config, "backend", None),  # tabletop / navigation
            "output_dir": getattr(recorder, "output_dir", None),  # 本轮落盘根目录
            "cameras": self.expected_camera_names(),  # 与 schema / validate 对齐
        }
        if self.scene is not None:  # arrange_scene 之后才有
            meta.update(self.scene.get_meta())  # 场景后端与物体清单
        if self.robot is not None:  # place_robot 之后才有
            meta.update(self.robot.get_meta())  # 机型 / 关节名等
        if self.planner is not None and self.planner.config is not None:
            meta["planner"] = self.planner.config.backend  # curobo / replay / nav
        return meta  # 写入 recorder.open.meta,便于事后对照配置

    def run_episode(self, target: Any, *, episode: int, attempt: int) -> str:
        """单次尝试:reset/play → 开集世界 → 规划 → 逐步落盘.

        子类覆盖下列钩子,不要改本方法顺序::

            reset_and_play
              → episode_world_hook(target)
              → before_plan_hook(target, episode, attempt)
              → plan_hook(...)               # False → 直接返回 failed
              → after_plan_hook(target)
              → _run_collect_loop(target)

        Args:
            target: 本轮目标物.
            episode: 1-based 集号.
            attempt: 1-based 尝试序号.

        Returns:
            ``success`` / ``failed`` / ``timeout``.规划失败直接 ``failed``.
        """
        self.reset_and_play()  # reset 清物理(失败只警告),再 play
        self.episode_world_hook(target)  # 默认空;写回抖动位或目标物到 goal
        self.before_plan_hook(target, episode=episode, attempt=attempt)  # 默认空;桌面热身
        if not self.plan_hook(target, episode=episode, attempt=attempt):  # 默认 False
            return "failed"  # 无路径/抓放失败,不进执行循环
        self.after_plan_hook(target)  # 默认空;导航摆起点并热身
        return self._run_collect_loop(target)  # 逐步下发/step/落盘,返回结束状态

    def reset_and_play(self) -> None:
        """本集开始:``reset`` 清物理(失败只警告),再 ``play``.

        ``reset`` 与 ``play`` 之间调用 ``after_reset_hook``.子类不覆盖本方法.

        Returns:
            None.
        """
        assert self.sim_world is not None, "sim_world 未初始化"  # 第1段必须已写入
        try:
            self.sim_world.reset()  # 清物理状态;个别后端会抛
        except Exception as exc:  # noqa: BLE001  # reset 失败不中断本集
            self.log.warning("reset 仿真世界失败, 跳过", error=str(exc))
        self.after_reset_hook()  # 默认空;分拣写回抖动位,KR210 恢复工位
        self.sim_world.play()  # 启动物理,后续 step 才推进

    def episode_world_hook(self, target: Any) -> None:
        """``run_episode``:``reset_and_play`` 之后、``before_plan_hook`` 之前.默认空.

        Args:
            target: 本轮目标物.

        Returns:
            None.
        """
        del target  # 基类不用;桌面回写抖动位,导航把目标物放到 goal_xy

    def after_plan_hook(self, target: Any) -> None:
        """``run_episode``:``plan_hook`` 成功之后、``_run_collect_loop`` 之前.

        默认空.导航在此摆机器人并热身.

        Args:
            target: 本轮目标物.

        Returns:
            None.
        """
        del target  # 基类空;导航在此摆起点、对齐朝向并热身

    def _run_collect_loop(self, target: Any) -> str:
        """逐步:下发 → step → 落盘 → 判定结束.

        流程::

            begin_loop_hook(target)
              → stride = stride_hook()
              → while episode_loop_running():   # 可经 playing_hook 结束
                   planner.on_physics_step(robot)
                   command_hook(target, done)
                   sim_world.step(render=抽帧或结束)
                   if 抽帧:
                       target_pose_hook → frame_hook → record_step
                   status_hook        # 非 None 则结束
                   step_limit_hook >= max_steps # timeout / max_steps
                   wall > planner.timeout_s     # timeout / wall_clock
              → last_stats = steps + loop_stats_hook + planner.get_meta
              → episode_done_hook(status)

        Args:
            target: 本轮目标物,用来读实时位姿.

        Returns:
            ``success`` / ``failed`` / ``timeout``.
        """
        assert self.sim_world is not None  # start_simulation 开头已断言
        assert self.robot is not None
        assert self.planner is not None
        assert self.recorder is not None  # _setup_recorder 失败则到不了这里
        self.begin_loop_hook(target)  # 默认空;清轨迹/抬升等本集状态
        timeout_s = float(self.planner.timeout_s)  # 墙钟上限,来自 planner 配置
        cam_names = self.expected_camera_names()  # 与 schema / validate 对齐
        t0 = time.monotonic()  # 墙钟起点,判 wall_clock timeout
        physics_i = 0  # 已跑物理步(含未落盘的)
        record_i = 0  # 已落盘帧,写入 last_stats.steps
        status = "timeout"  # 循环若未主动结束则算超时
        timeout_reason = ""  # max_steps / wall_clock;未超时为空
        ms_plan = 0.0  # on_physics_step 累计毫秒
        ms_step = 0.0  # sim.step 累计毫秒
        ms_record = 0.0  # 落盘累计毫秒
        max_steps = int(getattr(self.config, "max_steps", 2000))  # 与 step_limit_hook 比较
        stride = max(1, int(self.stride_hook()))  # 至少 1,避免除零/空转

        while self.episode_loop_running():  # 默认可经 is_running / is_playing 结束
            t_mark = time.perf_counter()  # 计 on_physics_step 耗时
            done = self.planner.on_physics_step(self.robot)  # 推进一物理步,返回是否结束
            ms_plan += (time.perf_counter() - t_mark) * 1000.0
            self.command_hook(target, done=done)  # 默认空;下发指令或记 attached

            should_record = physics_i % stride == 0 or done  # 抽帧或本集最后一步必录
            t_mark = time.perf_counter()  # 计 step 耗时
            self.sim_world.step(render=should_record)  # 仅抽帧/结束时 render,省开销
            ms_step += (time.perf_counter() - t_mark) * 1000.0

            if should_record:  # 按 stride 抽帧,或 done 时补最后一帧
                obj_pose = self.target_pose_hook(target)  # 刷新并读目标物实时位姿
                self.frame_hook(target, obj_pose)  # 默认空;跟踪抬升/抖动/占用图
                t_mark = time.perf_counter()  # 计落盘耗时
                self._record_collect_step(record_i, cam_names, obj_pose)  # 写一帧
                record_i += 1
                ms_record += (time.perf_counter() - t_mark) * 1000.0

            physics_i += 1  # 无论是否抽帧都算一步
            phase = str(self.planner.phase)  # done/failed/其它阶段
            decided = self.status_hook(done, phase)  # None 表示继续
            if decided is not None:  # success 或 failed
                status = decided
                break  # 正常结束,不再判超时
            if self.step_limit_hook(physics_i, record_i) >= max_steps:  # 桌面按帧,导航按物理步
                status = "timeout"
                timeout_reason = "max_steps"
                break
            if time.monotonic() - t0 > timeout_s:  # 墙钟超时,与物理步解耦
                status = "timeout"
                timeout_reason = "wall_clock"
                break

        self.last_stats = {
            "steps": record_i,  # 落盘帧数;子类勿在 loop_stats 里重复写
            **self.loop_stats_hook(  # 默认空;导航写 rtf/抖动,桌面写抬升
                physics_i=physics_i,
                record_i=record_i,
                t0=t0,
                timeout_reason=timeout_reason,
                ms_plan=ms_plan,
                ms_step=ms_step,
                ms_record=ms_record,
            ),
            **self.planner.get_meta(),  # 路径长度/phase 等
        }
        self.episode_done_hook(status)  # 默认空;导航打本集结束日志
        return status  # 交给 _collect_one_episode 做校验/验收

    def begin_loop_hook(self, target: Any) -> None:
        """``_run_collect_loop`` 开头,逐步循环之前.默认空.

        Args:
            target: 本轮目标物.

        Returns:
            None.
        """
        del target  # 基类空;导航清轨迹/抖动,桌面清抬升/合爪

    def episode_loop_running(self) -> bool:
        """``_run_collect_loop`` while 条件;默认看 ``is_running``.

        桌面可再经 ``playing_hook`` 要求 ``is_playing``.

        Returns:
            True 则继续 step.
        """
        if self.sim_world is None or not self.sim_world.is_running():  # App 已关则停
            return False
        if self.playing_hook() and not self.sim_world.is_playing():  # 桌面 Stop 即结束
            return False
        return True  # 继续 step

    def command_hook(self, target: Any, *, done: bool) -> None:
        """``_run_collect_loop``:``on_physics_step`` 之后、``step`` 之前.默认空.

        Args:
            target: 本轮目标物.
            done: 规划器是否宣告结束.

        Returns:
            None.
        """
        del target, done  # 基类空;导航稳相机,桌面记 attached

    def target_pose_hook(self, target: Any) -> Any | None:
        """刷新并读取目标物实时世界位姿;没有则为 None.

        Args:
            target: 带 ``prim_path`` 的本轮目标.

        Returns:
            ``(pos, quat)``;prim 不在 ``poses_realtime`` 则为 None.
        """
        if self.scene is None:  # 理论上 start_simulation 已断言,防御
            return None
        try:
            self.scene.refresh_poses_realtime([target.prim_path])  # 从 stage 刷实时位姿
            return self.scene.get_prim_pose_realtime(target.prim_path)  # (pos, quat)
        except KeyError:  # prim 不在 poses_realtime
            return None  # 落盘 object_pose 写 None,不中断本帧

    def frame_hook(self, target: Any, obj_pose: Any) -> None:
        """``_run_collect_loop`` 抽帧时,``record_step`` 之前.默认空.

        跟踪抬升 / 占用图 / 步态.

        Args:
            target: 本轮目标物.
            obj_pose: ``target_pose_hook`` 的结果.

        Returns:
            None.
        """
        del target, obj_pose  # 基类空;桌面跟踪最高 z,导航采样抖动/刷新占用图

    def _record_collect_step(self, record_i: int, cam_names: list[str], obj_pose: Any) -> None:
        """写一帧 recorder:状态 / 动作 / phase / 物体位姿 / RGB.

        Args:
            record_i: 落盘序号,从 0 计.
            cam_names: 本集要采的相机名.
            obj_pose: 目标物世界位姿;读不到则为 None.

        Returns:
            None.
        """
        assert self.recorder is not None  # 循环入口已断言
        assert self.robot is not None
        assert self.planner is not None
        recorder_cfg = getattr(self.config, "recorder", None)  # 读 save_rgb
        save_rgb = bool(getattr(recorder_cfg, "save_rgb", True))  # 默认存 RGB
        rgb_by_camera = self.collect_rgb(cam_names) if save_rgb else {}  # 关 RGB 则空 dict
        self.recorder.record_step(  # 写一帧:状态/动作/phase/物体位姿/RGB
            step_idx=record_i,  # 落盘序号,从 0 计
            robot_state=self.robot.get_state(),  # 关节/根位姿等
            action=self.planner.last_action,  # 本步下发的动作
            planner_phase=str(self.planner.phase),  # 便于事后对齐轨迹段
            object_pose=obj_pose,  # 读不到则为 None
            rgb_by_camera=rgb_by_camera,  # 相机名 → 数组或 None
        )

    def collect_rgb(self, names: list[str]) -> dict[str, np.ndarray | None]:
        """按名取 RGB;先场景后机器人,两处都没有则为 None.

        Args:
            names: 本集要采的相机名,与落盘 schema 对齐.

        Returns:
            相机名 → RGB 数组;传感器未出帧则为 ``None``.
        """
        out: dict[str, np.ndarray | None] = {}  # 与落盘 schema 对齐,缺帧写 None
        for name in names:  # 顺序与 expected_camera_names 一致
            frame = None  # 先假设本路未出帧
            if self.scene is not None:  # 场景相机优先(桌面 overhead 等)
                frame = self.scene.get_image(name)
            if frame is None and self.robot is not None:  # 再试腕部/机载相机
                frame = self.robot.get_image(name)
            out[name] = frame  # None 也写入,validate 才能发现缺相机
        return out

    def status_hook(self, done: bool, phase: str) -> str | None:
        """``_run_collect_loop`` 每步 ``step`` 之后.默认 ``done`` 或 phase 为 done/failed.

        Args:
            done: ``on_physics_step`` 返回值.
            phase: 当前规划阶段.

        Returns:
            结束状态;None 表示继续.
        """
        if done or phase in ("done", "failed"):  # 规划器宣告结束或进入终态
            return "success" if phase == "done" else "failed"  # done 以外都算失败
        return None  # 继续循环;桌面还要求见过合爪

    def step_limit_hook(self, physics_i: int, record_i: int) -> int:
        """``_run_collect_loop`` 每步与 ``max_steps`` 比较;默认用落盘帧数.

        Args:
            physics_i: 已跑物理步.
            record_i: 已落盘帧.

        Returns:
            用来判超时的步数.
        """
        del physics_i  # 基类按落盘帧计;导航改用物理步
        return record_i  # 与 max_steps 比较,抽帧时帧数 < 物理步

    def loop_stats_hook(
        self,
        *,
        physics_i: int,
        record_i: int,
        t0: float,
        timeout_reason: str,
        ms_plan: float,
        ms_step: float,
        ms_record: float,
    ) -> dict[str, Any]:
        """``_run_collect_loop`` 结束时并入 ``last_stats``;默认空.

        Args:
            physics_i: 已跑物理步.
            record_i: 已落盘帧.
            t0: 循环墙钟起点.
            timeout_reason: ``max_steps`` / ``wall_clock``;未超时为空.
            ms_plan: ``on_physics_step`` 累计毫秒.
            ms_step: ``sim.step`` 累计毫秒.
            ms_record: 落盘累计毫秒.

        Returns:
            任务统计;不要重复写 ``steps``.
        """
        del physics_i, record_i, t0, timeout_reason, ms_plan, ms_step, ms_record
        return {}  # 基类只写 steps;子类补 rtf/抬升等,勿重复 steps

    def episode_done_hook(self, status: str) -> None:
        """执行成功后保留抓点,供下一集采样时避开其邻域.

        导航打 ``collect_episode_done``.

        Args:
            status: 本集结束状态.

        Returns:
            None.
        """
        if status == "success":
            self.planner.remember_successful_grasps()

    def require_success_hook(self) -> bool:
        """``_collect_one_episode`` 里 ``validate_episode`` 调用;默认 True.

        True 则失败/超时集不算合格(桌面抓放);导航覆盖为 False.

        Returns:
            True 则校验要求 ``status == success``.
        """
        return True  # 桌面抓放:失败/超时集不算合格;导航覆盖为 False

    def accept_hook(self, status: str) -> bool:
        """``_collect_one_episode``:校验通过但非 success 时;默认 False.

        导航 timeout 覆盖为 True,验收本集、不重试.

        Args:
            status: ``run_episode`` 返回的状态串.

        Returns:
            True 则验收本集并计入对应计数.
        """
        del status  # 基类不验收非 success;导航 timeout 覆盖为 True
        return False  # False → 复位再试或记 failed

    def accepted_hook(self, status: str, *, episode: int) -> None:
        """``_collect_one_episode``:``accept_hook`` 为真之后.默认空.

        Args:
            status: 被验收的状态(如 ``timeout``).
            episode: 1-based 集号.

        Returns:
            None.
        """
        del status, episode  # 基类空;导航 timeout 时打墙钟/步数警告

    def _log_collect_finished(self, *, n_ok: int, n_timeout: int, n_ep: int) -> None:
        """``start_simulation`` 末尾打汇总.父类步骤,子类不覆盖.

        Args:
            n_ok: 成功集数.
            n_timeout: 验收为 timeout 的集数.
            n_ep: 计划集数.

        Returns:
            None.
        """
        recorder = getattr(self.config, "recorder", None)  # 取 output_dir 便于对照产物
        self.log.info(  # 整轮汇总;n_timeout 仅导航验收路径非 0
            "采集完成",
            ok=n_ok,
            timeout=n_timeout,
            total=n_ep,
            output_dir=getattr(recorder, "output_dir", None),
        )

    def after_collect_hook(self, *, n_ok: int, n_timeout: int, n_ep: int) -> None:
        """汇总之后检查计划集数是否完成,未完成时让 CLI 明确报错.

        Args:
            n_ok: 成功集数.
            n_timeout: 验收为 timeout 的集数.
            n_ep: 计划集数.

        Returns:
            None.
        """
        if n_ok + n_timeout < n_ep:
            raise RuntimeError(f"采集仅成功 {n_ok}/{n_ep} 集；拒绝原因见 rejected_attempts.jsonl")
