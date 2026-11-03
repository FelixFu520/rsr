"""``ManipulatorRobot``:在 ``BaseRobot`` 上补机械臂采集所需的通用能力.

``BaseRobot`` 只保留位姿、传感器、状态缓存及 ``init_robot`` / ``arrange``;
实时位姿、相机取帧、主臂 articulation 句柄与 episode 元数据放在这里,
由具体机型(如 ``ProM6Robot``)继承.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from real2sim2real.core.logger import LoggerMixin
from real2sim2real.robot.base import BaseRobot
from real2sim2real.utils.isaacsim_utils.openusd_utils import normalize_prim_path


class ManipulatorRobot(BaseRobot, LoggerMixin):
    """机械臂基础实现:位姿查询、相机取帧、articulation 句柄与 episode 元数据."""

    def __init__(self, **kwargs: Any) -> None:
        """占位主臂 articulation;机型 ``arrange`` 后再写入句柄.

        Args:
            **kwargs: 工厂多余字段,丢弃.

        Returns:
            None.
        """
        super().__init__()
        del kwargs
        self.articulation_obj: Any | None = None  # 尚未构建则为 None

    # ============= 机器人流水线 <==========

    def post_physics_init(self, *, warmup_steps: int = 30) -> None:
        """``world.reset()`` 之后的热身(出帧 / 稳定接触);默认空.

        Args:
            warmup_steps: 热身物理步数;默认 30.

        Returns:
            None.
        """

    def get_state(self) -> dict[str, Any]:
        """当前状态字典;键由各机型自定,recorder 落盘用.

        Returns:
            机型状态字典.
        """
        raise NotImplementedError

    # ============= 辅助函数 <==========

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

    def layout_prim_paths(self) -> list[str]:
        """视口框选 / 布局检查用的 prim 路径.

        Returns:
            根与相机 prim 路径列表;无配置则为空.
        """
        cfg = self.config
        if cfg is None:
            return []
        prim = getattr(cfg, "prim_path", None)
        paths = [str(prim)] if prim else []
        paths.extend(cam.prim_path for cam in (getattr(cfg, "cameras", None) or []))
        return paths

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
