"""Isaac 机器人装配:夹爪臂、贴桌.

机型策略绑定在 ``robot_franka`` / ``robot_g1`` / ``robot_teleavatar``.
pxr / isaacsim 仅在函数内 import,避免 stub 通路被重依赖拖垮.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from real2sim2real.utils.isaacsim_utils.openusd_utils import normalize_prim_path


def add_parallel_gripper_manipulator(
    world: Any,
    *,
    prim_path: str,
    name: str,
    ee_prim_path: str,
    finger_joint_names: list[str],
    open_position: float,
    closed_position: float,
    action_delta: float = 0.02,
) -> Any:
    """把已摆好的臂包成 ``SingleManipulator`` + ``ParallelGripper`` 并加入 scene.

    须在 pose 摆好之后调用;真正 ``initialize`` 要等 ``world.reset()``.

    Args:
        world: Isaac ``World``;None 则返回 None.
        prim_path: 机械臂根 prim.
        name: World 内物体名,须唯一.
        ee_prim_path: 末端 prim.
        finger_joint_names: 两指关节名.
        open_position: 单指张开位置(m).
        closed_position: 单指闭合位置(m).
        action_delta: 每步开合增量(m).

    Returns:
        ``SingleManipulator``;``world`` 为 None 则为 ``None``.
    """
    if world is None:
        return None
    from isaacsim.robot.manipulators import SingleManipulator
    from isaacsim.robot.manipulators.grippers import ParallelGripper

    root = normalize_prim_path(prim_path)
    ee_prim = normalize_prim_path(ee_prim_path)
    gripper = ParallelGripper(
        end_effector_prim_path=ee_prim,
        joint_prim_names=list(finger_joint_names),
        joint_opened_positions=np.array([open_position, open_position]),
        joint_closed_positions=np.array([closed_position, closed_position]),
        action_deltas=np.array([action_delta, action_delta]),
    )
    return world.scene.add(
        SingleManipulator(
            prim_path=root,
            name=name,
            end_effector_prim_path=ee_prim,
            gripper=gripper,
        )
    )


MANIPULATOR_BASE_SUFFIXES: tuple[str, ...] = (
    "panda_link0",
    "g_link0",
    "link0",
    "base_link",
    "panda/panda_link0",
)
MANIPULATOR_BASE_NAMES: frozenset[str] = frozenset(
    {"panda_link0", "g_link0", "link0", "base_link", "base"}
)


def snap_base_to_support(
    robot: Any,
    world_pos: np.ndarray,
    world_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把底座底面落到桌面,并把底座 XY 收进桌面 AABB.

    Args:
        robot: 机械臂实例,用 ``support_z`` / ``config`` / ``sim_world``.
        world_pos: ``place_asset`` 合成的世界位置.
        world_quat: ``place_asset`` 合成的世界朝向 ``wxyz``.

    Returns:
        贴合后的 ``(世界 pos, wxyz)``;量不到底座则原样返回.
    """
    from real2sim2real.utils.isaacsim_utils.modify_stage_utils import apply_pose, pump_frames
    from real2sim2real.utils.isaacsim_utils.openusd_utils import (
        find_prim_path,
        get_stage,
        is_world_frame,
        prim_aabb,
        prim_aabb_z_max,
    )
    from real2sim2real.utils.misc import emit_log
    from real2sim2real.utils.pose import snap_aabb_onto_support

    cfg = robot.config
    if is_world_frame(cfg.pose_relative_to):
        return world_pos, world_quat
    stage = get_stage(robot.sim_world)
    support_z = robot.support_z
    if support_z is not None:
        support_z = float(support_z)
    else:
        support_z = prim_aabb_z_max(stage, cfg.pose_relative_to)
    table_bounds = prim_aabb(stage, cfg.pose_relative_to, gprims_only=True)
    root = normalize_prim_path(cfg.prim_path)
    base_path = (
        find_prim_path(
            stage,
            root,
            suffixes=MANIPULATOR_BASE_SUFFIXES,
            names=MANIPULATOR_BASE_NAMES,
        )
        or root
    )
    apply_pose(stage, cfg.prim_path, world_pos, world_quat)
    pump_frames(robot.sim_world, 2)
    base_bounds = prim_aabb(stage, base_path, gprims_only=True)
    if base_bounds is None:
        emit_log(robot, "warning", "robot_base_aabb_missing", base_path=base_path)
        return world_pos, world_quat
    tmin = tmax = None
    if table_bounds is not None:
        tmin, tmax = table_bounds
    snapped = snap_aabb_onto_support(
        world_pos,
        base_bounds[0],
        base_bounds[1],
        support_z=support_z,
        rel_z=float(cfg.pose.position[2]),
        table_min=tmin,
        table_max=tmax,
    )
    return snapped, world_quat
