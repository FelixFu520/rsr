"""Pro-M6 双臂机器人:把机器人 USD 引用到 ``/World/robots``,再绑相机、建 articulation.

分拣台场景包里的机器人会被 Scene 关掉;本后端把场景包里带视觉网格的机器人
prim 引用到 ``/World/robots``,不改资产内部尺度.碰撞方块代理 USD 不能当本体.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from real2sim2real.robot.config_pro_m6 import ProM6RobotConfig
from real2sim2real.robot.robot_manipulator import ManipulatorRobot
from real2sim2real.simworld.base import BaseSimWorld
from real2sim2real.utils.isaacsim_utils import (
    bind_camera_sensor,
    normalize_prim_path,
    place_asset,
    pump_frames,
    resolve_usd_path,
    spawn_reference,
    spawn_subprim_reference,
)
from real2sim2real.utils.pose import read_prim_world_pose


class ProM6Robot(ManipulatorRobot):
    """Pro-M6 双臂:USD 挂在 ``/World/robots``,articulation 用其子 prim.

    ``place_robot``(分拣 Job)::

        AutoRobot.from_config()
        init_robot(sim_world)                  # 把带视觉网格的机器人 prim 挂到 /World/robots
        Job.adjust_robot()                     # 子 prim 引用已带原机位,不再叠平移
        arrange()                              # 登记位姿、绑机载相机、建 articulation
    """

    config_cls = ProM6RobotConfig

    def __init__(self, config: ProM6RobotConfig | Any | None = None, **kwargs: Any) -> None:
        """写入 ``config``,占位 articulation.

        Args:
            config: 已构造配置;None 则从 ``[robot]`` 加载.
            **kwargs: 工厂多余字段,丢弃.

        Returns:
            None.
        """
        super().__init__(**kwargs)
        self.config: ProM6RobotConfig = config or self.config_cls.from_settings("robot")

    @property
    def spawn_prim_path(self) -> str:
        """USD 引用根:articulation 根的父级,通常是 ``/World/robots``.

        Returns:
            规范化后的父级 prim 路径.
        """
        root = normalize_prim_path(self.config.prim_path)
        parent = root.rsplit("/", 1)[0]
        return parent if parent and parent != "/" else root

    # ============= 机器人流水线 <==========

    def init_robot(self, sim_world: BaseSimWorld) -> None:
        """绑定世界并挂机器人 USD;不写位姿.

        Args:
            sim_world: 已 ``init_world`` 的 Isaac 世界,须带可用 stage.

        Returns:
            None.

        Raises:
            RuntimeError: 未配置 usd_path,或引用后找不到 articulation 根.
        """
        from isaacsim.core.utils.stage import add_reference_to_stage

        self.sim_world = sim_world
        assert self.sim_world is not None, "Sim world is not initialized"
        assert self.sim_world.stage is not None, "Stage is not initialized"

        usd = (self.config.usd_path or "").strip()
        src_prim = (self.config.usd_prim_path or "").strip()
        if usd:
            resolved = resolve_usd_path(usd)
            if src_prim:
                spawn_subprim_reference(
                    self.sim_world.stage, self.config.prim_path, resolved, src_prim
                )
            else:
                spawn_reference(
                    self.sim_world.stage, self.spawn_prim_path, resolved, add_reference_to_stage
                )
            pump_frames(self.sim_world, 4)
            self.log.info(
                "引用 Pro-M6 成功",
                spawn_prim=self.spawn_prim_path,
                prim_path=self.config.prim_path,
                usd=resolved,
                usd_prim_path=src_prim or None,
            )

        root = normalize_prim_path(self.config.prim_path)
        prim = self.sim_world.stage.GetPrimAtPath(root)
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"环境 USD 缺少机器人 prim: {root}")

    def arrange(self) -> None:
        """按配置摆实例根(不改 USD 内部尺度),绑机载相机,再建 articulation.

        Returns:
            None.
        """
        root = normalize_prim_path(self.config.prim_path)
        src_prim = (self.config.usd_prim_path or "").strip()
        if src_prim:
            # 源 prim 已带世界位姿,再写父级会叠一次平移
            child_pose = read_prim_world_pose(self.sim_world.stage, root)
            if child_pose is None:
                raise RuntimeError(f"读不到 Pro-M6 世界位姿: {root}")
            pos, quat = child_pose
        else:
            pos, quat = place_asset(
                self.sim_world.stage,
                self.poses,
                self.spawn_prim_path,
                self.config.pose,
                self.config.pose_relative_to,
            )
            pump_frames(self.sim_world, 1)
            child_pose = read_prim_world_pose(self.sim_world.stage, root)
            if child_pose is not None:
                pos, quat = child_pose
        self.poses[root] = (pos, quat)
        self.log.info("摆放 Pro-M6 成功", prim_path=root, position=pos.tolist())

        for cam in self.config.cameras:
            self.sensors[cam.name] = bind_camera_sensor(self.sim_world.stage, cam)
            pump_frames(self.sim_world, 1)
        if self.config.cameras:
            self.log.info("绑定机载相机成功", cameras=[cam.name for cam in self.config.cameras])

        self.seed_poses_realtime()
        self._build_articulation()

    def post_physics_init(self, *, warmup_steps: int = 30) -> None:
        """Play 后只步进出帧;双臂关节由分拣规划器每步接管,这里不下发目标.

        Args:
            warmup_steps: 热身物理步数;至少 1.

        Returns:
            None.
        """
        world = None if self.sim_world is None else self.sim_world.world
        if world is None:
            return
        for _ in range(max(1, int(warmup_steps))):
            try:
                world.step(render=True)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("热身步进失败", error=str(exc))
                break

    # ============= 辅助函数 <==========

    def _build_articulation(self) -> None:
        """``arrange`` 末尾:建 ``isaacsim.core.experimental.prims.Articulation``.

        Returns:
            None.
        """
        from isaacsim.core.experimental.prims import Articulation

        root = normalize_prim_path(self.config.prim_path)
        try:
            self.articulation_obj = Articulation(root)
            self.log.debug("构建 Pro-M6 articulation 成功", prim_path=root)
        except Exception as exc:  # noqa: BLE001
            self.log.error("构建 Pro-M6 articulation 失败", error=str(exc))
            self.articulation_obj = None

    @property
    def left_gripper_prim_path(self) -> str:
        """左夹爪 prim;未配置则取 ``<root>/robotiq_left``.

        Returns:
            左夹爪 prim 路径.
        """
        root = normalize_prim_path(self.config.prim_path)
        return self.config.left_gripper_prim_path or f"{root}/robotiq_left"

    @property
    def right_gripper_prim_path(self) -> str:
        """右夹爪 prim;未配置则取 ``<root>/robotiq_right``.

        Returns:
            右夹爪 prim 路径.
        """
        root = normalize_prim_path(self.config.prim_path)
        return self.config.right_gripper_prim_path or f"{root}/robotiq_right"

    def get_state(self) -> dict[str, Any]:
        """落盘用的双臂观测:全量 DOF 位置 / 速度 / 目标.

        Returns:
            ``joint_positions`` / ``joint_velocities`` / ``joint_position_targets``.
        """
        empty = {
            "joint_positions": np.zeros(0, dtype=float),
            "joint_velocities": np.zeros(0, dtype=float),
            "joint_position_targets": np.zeros(0, dtype=float),
        }
        art = self.articulation_obj
        if art is None:
            return empty
        out: dict[str, Any] = {}
        for key, getter in (
            ("joint_positions", "get_dof_positions"),
            ("joint_velocities", "get_dof_velocities"),
            ("joint_position_targets", "get_dof_position_targets"),
        ):
            try:
                raw = getattr(art, getter)()
                value = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
                out[key] = np.asarray(value, dtype=float).reshape(-1)
            except Exception:  # noqa: BLE001
                out[key] = empty[key]
        return out

    def layout_prim_paths(self) -> list[str]:
        """视口框选:引用根、articulation 根与机载相机.

        Returns:
            prim 路径列表.
        """
        paths = [self.spawn_prim_path, self.config.prim_path]
        paths.extend(cam.prim_path for cam in self.config.cameras)
        return paths
