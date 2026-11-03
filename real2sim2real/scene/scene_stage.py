"""``StageScene``:在 ``BaseScene`` 上补 Isaac stage 位姿读写与相机取帧.

``BaseScene`` 只保留位姿缓存与 ``init_scene`` / ``arrange``;采集循环需要的
实时位姿、挪物、取图等通用能力放在这里,由具体场景(如 ``SortingScene``)继承.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from real2sim2real.core.logger import LoggerMixin
from real2sim2real.scene.base import BaseScene
from real2sim2real.utils.isaacsim_utils import normalize_prim_path


class StageScene(BaseScene, LoggerMixin):
    """带 stage 位姿读写的场景基础实现;``init_scene`` / ``arrange`` 仍由子类实现."""

    def __init__(self) -> None:
        """在 ``BaseScene`` 缓存上增加最近一次 teleport 位姿.

        Returns:
            None.
        """
        super().__init__()
        self._teleport_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # 最近一次显式 teleport

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

    def goal_object(self) -> Any | None:
        """返回目标物体配置;没有任何物体则为 ``None``.

        Returns:
            Any | None.
        """
        return None

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

    def layout_prim_paths(self) -> list[str]:
        """视口框选 / 布局检查用的 prim 路径;stub / 未覆盖时为空列表.

        Returns:
            prim 路径列表;基类为空.
        """
        return []

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

    def get_meta(self) -> dict[str, Any]:
        """写入 episode meta 的场景快照;基类为空,桌面/loco 各自填自己的键.

        Returns:
            场景快照字典;基类为空 dict.
        """
        return {}

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

    def get_image(self, name: str, annotator: str = "rgb") -> np.ndarray | None:
        """按相机名取一帧;``annotator`` 选 RGB / 深度 / 其它已绑定项.

        Args:
            name: ``sensors`` 字典里的相机名(如 ``wrist`` / ``front``).
            annotator: ``rgb``(默认)、``depth`` 或官方 annotator 名.

        Returns:
            numpy 数组;默认无相机,恒为 ``None``.
        """
        del name, annotator  # 默认无相机,子类覆盖真正读帧
        return None  # stub / 未绑定时 Job 应跳过该相机
