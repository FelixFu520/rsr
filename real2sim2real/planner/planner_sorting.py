"""分拣规划器:装配并驱动 Pro-M6 双臂中继状态机.

状态机本体在 :mod:`real2sim2real.planner.prom6_relay`.
本后端只做四件事:

1. ``init_planner``:注入 prim 路径、配 GPU 物理、Play 前去掉 PhysX 柔性并建 Newton 软袋
2. ``begin_episode``:注册 ``PHYSICS_POST_STEP`` 回调
3. ``on_physics_step``:把控制器阶段翻译成 Job 认识的 ``phase`` / ``done`` / 动作
4. ``reset`` / ``cleanup``:清临时 prim、夹爪碰撞覆盖与回调
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from real2sim2real.core.logger import LoggerMixin
from real2sim2real.planner.config_sorting import SortingPlannerConfig
from real2sim2real.planner.planner_manipulation import BaseManipulationPlanner


class SortingPlanner(BaseManipulationPlanner, LoggerMixin):
    """Pro-M6 双臂分拣:入带抓包裹 → 平台 → 出带,Newton 接触承重.

    ``place_robot``::

        init_planner(robot)                    # 配 GPU 物理 + 去 PhysX 柔性 + 建 Newton 软袋

    每集 ``run_episode``::

        begin_episode(ep, attempt)             # 注册物理回调,状态机开始跑
        loop: on_physics_step(robot)           # 读阶段/动作,判断本集是否结束
        reset()                                # 重试前清回调,再重新装配
    """

    config_cls = SortingPlannerConfig

    def __init__(self, config: SortingPlannerConfig | Any | None = None, **kwargs: Any) -> None:
        """写入 ``[planner]`` 配置;控制器留到 ``init_planner`` 再建.

        Args:
            config: 已构造配置;None 则从 ``[planner]`` 加载.
            **kwargs: 工厂多余字段,丢弃.

        Returns:
            None.
        """
        super().__init__()
        del kwargs
        self.config: SortingPlannerConfig = config or self.config_cls.from_settings("planner")
        self.state["timeout_s"] = float(self.config.timeout_s)

        self._controller: Any | None = None  # prom6_relay 控制器实例
        self._robot: Any | None = None  # 已 arrange 的 ProM6Robot
        self._callback_ids: list[int] = []  # 已注册的 SimulationManager 回调
        self._physics_dt = 1.0 / float(self.config.physics_hz)  # manual 模式下的步长

    # ============= 规划流水线 <==========

    def init_planner(self, robot: Any) -> None:
        """``place_robot`` 末尾:配 GPU 物理,去掉袋子 PhysX 柔性,建 Newton 接触世界.

        Args:
            robot: 已 ``arrange`` 的 Pro-M6 机器人.

        Returns:
            None.
        """
        from real2sim2real.planner import prom6_relay as relay
        from real2sim2real.planner.newton_sorting import (
            NewtonSortingParams,
            NewtonSortingWorld,
            discover_finger_meshes,
            discover_tool_centers,
            resolve_infeed_prim_path,
        )

        self._robot = robot
        sim_world = robot.sim_world
        assert sim_world is not None, "robot 尚未绑定 sim_world"
        stage = sim_world.stage
        assert stage is not None, "stage 未初始化"

        cfg = self.config
        relay.apply_settings(
            robot_path=cfg.robot_prim_path,
            clothing_root_path=cfg.clothing_root_prim_path,
            clothing_surface_path=cfg.clothing_surface_prim_path,
            clothing_volume_proxy_path=cfg.clothing_volume_proxy_prim_path,
            support_path=cfg.support_prim_path,
            belt_path=cfg.belt_prim_path,
            left_gripper_path=cfg.left_gripper_prim_path,
            right_gripper_path=cfg.right_gripper_prim_path,
            runtime_root_path=cfg.runtime_root_prim_path,
            replay_cache_dir=cfg.replay_cache_dir,
            physics_device=cfg.physics_device,
            physics_hz=cfg.physics_hz,
        )

        sim_world.stop()
        relay.require_timeline_stopped(stage)
        relay.require_uninitialized_physics()
        relay.remove_owned_runtime_prims(stage)
        relay.configure_gpu_physics(stage)

        timing = self._build_timing(relay)
        self._controller = relay.ProM6LeftClothingDragController(
            stage,
            timing,
            articulation=getattr(robot, "articulation", None),
        )
        self._controller.prepare_before_play()

        finger_paths = list(cfg.left_finger_prim_paths) + list(cfg.right_finger_prim_paths)
        if not finger_paths:
            try:
                finger_paths = discover_finger_meshes(
                    stage, relay.LEFT_GRIPPER_PATH
                ) + discover_finger_meshes(stage, relay.RIGHT_GRIPPER_PATH)
            except RuntimeError:
                finger_paths = []
        left_tcp, right_tcp = discover_tool_centers(stage, relay.ROBOT_PATH)
        tcp_jaws = [(left_tcp, "left"), (right_tcp, "right")]
        infeed_path = resolve_infeed_prim_path(stage, cfg.infeed_prim_path)
        params = NewtonSortingParams(
            youngs_modulus=float(cfg.youngs_modulus),
            poisson_ratio=float(cfg.poisson_ratio),
            density=float(cfg.density),
            particle_radius=float(cfg.particle_radius),
            soft_contact_mu=float(cfg.soft_contact_mu),
            shape_mu=float(cfg.shape_mu),
            physics_hz=float(cfg.physics_hz),
            sim_substeps=int(cfg.newton_substeps),
            vbd_iterations=int(cfg.vbd_iterations),
            device=str(cfg.physics_device),
            belt_speed_mps=float(cfg.belt_speed_mps),
            min_lift_height_m=float(cfg.min_lift_height_m),
        )
        newton_world = NewtonSortingWorld(
            stage,
            params,
            clothing_surface_path=cfg.clothing_surface_prim_path,
            support_prim_path=cfg.support_prim_path,
            belt_prim_path=cfg.belt_prim_path,
            infeed_prim_path=infeed_path,
            finger_prim_paths=finger_paths,
            tcp_jaw_prims=tcp_jaws,
        )
        self._controller.bind_newton(newton_world, min_lift_height_m=float(cfg.min_lift_height_m))
        self.state["ready"] = True
        self._physics_dt = 1.0 / float(cfg.physics_hz)
        self.log.info(
            "分拣规划器已就绪",
            drive_mode=cfg.drive_mode,
            grasp_point=self._controller.grasp_point.tolist(),
            support_center=self._controller.support_center.tolist(),
            newton_particles=int(newton_world.model.particle_count),
            infeed=infeed_path,
            fingers=len(finger_paths),
            left_tcp=left_tcp,
            right_tcp=right_tcp,
        )

    def begin_episode(self, episode_index: int, *, attempt: int = 1) -> None:
        """每集 Play 之后调用:注册物理回调,状态机从 SETTLE 开始跑.

        Args:
            episode_index: 0-based 集号,仅用于日志.
            attempt: 1-based 尝试序号,仅用于日志.

        Returns:
            None.
        """
        if self._controller is None:
            self.log.warning("尚未 init_planner,跳过 begin_episode")
            return
        if self.config.drive_mode == "callback":
            self._register_callbacks()
        self.log.info(
            "分拣本集开始",
            episode=episode_index,
            attempt=attempt,
            drive_mode=self.config.drive_mode,
        )

    def on_physics_step(self, robot: Any) -> bool:
        """每步:manual 模式推进状态机,并把阶段/动作同步给 Job.

        Args:
            robot: 当前机器人;关节由控制器直接写,这里不用.

        Returns:
            本集是否已结束(DONE / FAILED).
        """
        del robot  # 关节由控制器经 Articulation 直接下发
        controller = self._controller
        if controller is None:
            return True
        if self.config.drive_mode == "manual" and not controller.cleaned:
            try:
                controller.step(self._physics_dt, None)
            except Exception as exc:  # noqa: BLE001  # 与原脚本一致:失败即收尾
                controller.fail(exc)
        self.state["last_action"] = self._read_commanded_action(controller)
        return self.phase in ("done", "failed")

    # ============= 辅助函数 <==========

    @property
    def phase(self) -> str:
        """控制器当前阶段;Job 用它判定 success / failed;控制器未建时取 ``state``.

        Returns:
            ``done`` / ``failed`` / 小写阶段名;控制器未建时为占位阶段.
        """
        controller = self._controller
        if controller is None:
            return str(self.state["phase"])
        return str(controller.phase.name).lower()

    def _build_timing(self, relay: Any) -> Any:
        """按 ``timing_overrides`` 覆盖默认时长表.

        Args:
            relay: 已导入的 :mod:`prom6_relay` 模块.

        Returns:
            校验过的 ``TimingConfig``.

        Raises:
            ValueError: 覆盖了不存在的时长字段.
        """
        overrides = dict(self.config.timing_overrides)
        if not overrides:
            return relay.TASK_TIMING
        known = set(relay.TimingConfig.__dataclass_fields__)
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"timing_overrides 含未知字段 {unknown};可选: {sorted(known)}")
        return replace(relay.TASK_TIMING, **{k: float(v) for k, v in overrides.items()})

    def _register_callbacks(self) -> None:
        """注册 ``PHYSICS_POST_STEP`` 与 ``SIMULATION_STOPPED`` 回调.

        与原脚本一致:物理步后推进状态机,时间轴停止时收尾.

        Returns:
            None.
        """
        from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

        self._deregister_callbacks()
        controller = self._controller
        assert controller is not None

        def physics_callback(dt: float, context: object) -> None:
            try:
                controller.step(dt, context)
            except Exception as exc:  # noqa: BLE001  # 单帧异常转为 FAILED,不炸进程
                controller.fail(exc)

        def stopped_callback(*_args: Any, **_kwargs: Any) -> None:
            if controller.cleaned:
                return
            controller.cleanup(hold_left=True)

        self._callback_ids = [
            SimulationManager.register_callback(
                physics_callback, event=SimulationEvent.PHYSICS_POST_STEP
            ),
            SimulationManager.register_callback(
                stopped_callback, event=SimulationEvent.SIMULATION_STOPPED
            ),
        ]
        controller.callback_id = self._callback_ids[0]
        controller.stop_callback_id = self._callback_ids[1]

    def _deregister_callbacks(self) -> None:
        """注销本规划器注册过的回调;重复调用安全.

        Returns:
            None.
        """
        if not self._callback_ids:
            return
        from isaacsim.core.simulation_manager import SimulationManager

        for callback_id in self._callback_ids:
            try:
                SimulationManager.deregister_callback(callback_id)
            except Exception:  # noqa: BLE001  # 已随时间轴停止注销过
                pass
        self._callback_ids = []

    @staticmethod
    def _read_commanded_action(controller: Any) -> np.ndarray | None:
        """本步下发的双臂关节目标(左 7 + 右 7),供 recorder 落盘.

        Args:
            controller: prom6_relay 控制器.

        Returns:
            长度 14 的关节目标;状态机尚未初始化则为 ``None``.
        """
        left = getattr(controller, "left_commanded_q", None)
        right = getattr(controller, "right_commanded_q", None)
        if left is None and right is None:
            return None
        parts = [np.asarray(q, dtype=float).reshape(-1) for q in (left, right) if q is not None]
        return np.concatenate(parts)

    def reset(self) -> None:
        """重试前:收掉旧控制器与回调,再按同一配置重新装配.

        Returns:
            None.
        """
        super().reset()
        robot = self._robot
        self.cleanup()
        if robot is not None:
            self.init_planner(robot)

    def cleanup(self) -> None:
        """Job ``close`` / 重试时调用:清临时 prim、碰撞覆盖与回调.

        Returns:
            None.
        """
        controller = self._controller
        if controller is not None:
            try:
                controller.cleanup(hold_left=False)
            except Exception as exc:  # noqa: BLE001  # 收尾失败不阻断 Job 退出
                self.log.warning("清理分拣控制器失败", error=str(exc))
        self._deregister_callbacks()
        self._controller = None
        self.state["ready"] = False

    def get_meta(self) -> dict[str, Any]:
        """并入 ``last_stats`` 的规划器快照.

        Returns:
            阶段、右臂拖拽距离与回放帧数.
        """
        controller = self._controller
        if controller is None:
            return {"planner_phase": self.phase}
        recorder = getattr(controller, "replay_recorder", None)
        return {
            "planner_phase": self.phase,
            "right_drag_distance_m": float(
                getattr(controller, "right_actual_drag_distance", 0.0) or 0.0
            ),
            "replay_frames": int(getattr(recorder, "frame_count", 0) or 0),
        }
