"""``BaseManipulationPlanner``:抓放 / 搬运规划家族的能力子类.

``BasePlanner`` 只保留 ``init_planner`` / ``on_physics_step`` 与 ``state`` 缓存;
操作类任务另需的 ``begin_episode`` / ``pick_and_place`` / ``reset`` / ``get_meta``
以及 ``phase`` / ``ready`` / ``timeout_s`` / ``last_action`` 只读视图放在这里.
全身 loco、导航不要继承此类.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from real2sim2real.planner.base import BasePlanner


class BaseManipulationPlanner(BasePlanner):
    """操作 / 抓放规划家族(单臂或双臂);状态仍存放在 ``self.state``.

    Job 采集循环::

        begin_episode(ep, attempt)             # 每集 / 每次尝试开始
        loop: on_physics_step(robot)           # True=本集结束
        last_action / phase / get_meta()       # recorder 与统计
        reset()                                # 重试清状态,不卸载求解器
    """

    def __init__(self) -> None:
        """在基类状态上增加抓放标志与放置位姿占位.

        Returns:
            None.
        """
        super().__init__()
        self.attached: bool = False  # 合爪后视为已抓住
        self.place_pose: Any | None = None  # 相对支撑面的放置位姿;无则 None

    @property
    def timeout_s(self) -> float:
        """Job 墙钟超时(秒)."""
        return float(self.state["timeout_s"])

    @property
    def ready(self) -> bool:
        """求解器是否已加载成功."""
        return bool(self.state["ready"])

    @property
    def phase(self) -> str:
        """当前执行阶段名,recorder 落盘用."""
        return str(self.state["phase"])

    @property
    def last_action(self) -> np.ndarray | None:
        """最近一次下发的动作副本;尚未规划时为 None."""
        action = self.state["last_action"]
        return None if action is None else np.asarray(action).copy()

    # ============= 规划流水线 <==========

    def begin_episode(self, episode_index: int, *, attempt: int = 1) -> None:
        """Job 在每集每次尝试开始时调用;默认 no-op.

        Args:
            episode_index: Job 内 0-based 集序号.
            attempt: 本集第几次尝试(1-based).

        Returns:
            None.
        """
        del episode_index, attempt

    def pick_and_place(
        self,
        pick_pos: np.ndarray,
        place_pos: np.ndarray,
        start_joint_angles: np.ndarray,
    ) -> bool:
        """编排抓放并规划第一段轨迹;默认不支持,返回 False.

        Args:
            pick_pos: 抓取点世界坐标.
            place_pos: 放置点世界坐标.
            start_joint_angles: 规划起点关节角.

        Returns:
            True 表示已编排并规划成功;默认不支持.
        """
        del pick_pos, place_pos, start_joint_angles
        return False

    # ============= 辅助函数 <==========

    def reset(self) -> None:
        """清空抓放状态并回到 ``idle``;不卸载规划器.

        Returns:
            None.
        """
        self.state["phase"] = "idle"
        self.state["last_action"] = None
        self.attached = False

    def get_meta(self) -> dict[str, Any]:
        """写入 episode 统计的规划器快照;默认为空.

        Returns:
            规划器快照字典.
        """
        return {}
