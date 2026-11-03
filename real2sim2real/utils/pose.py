"""位姿配置与齐次变换工具.

TOML 里用欧拉角(度);内部统一 ``(position, wxyz)``.
``compose_pose`` 给 Scene/Robot 相对摆放;``quat_*_rotate`` 给 cuRobo 世界↔基座.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_ORIENTATION_WXYZ = (1.0, 0.0, 0.0, 0.0)  # 单位四元数,无旋转


class PoseConfig(BaseModel):
    """``position`` + 欧拉角 ``pitch/roll/yaw``(度)."""

    model_config = ConfigDict(extra="forbid")  # 拼写 pitch 成 pithc 会直接报错

    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    pitch: float = Field(default=0.0, description="绕 Y 轴旋转,度")
    roll: float = Field(default=0.0, description="绕 X 轴旋转,度")
    yaw: float = Field(default=0.0, description="绕 Z 轴旋转,度")
    orientation_wxyz: list[float] | None = Field(
        default=None, description="若提供则优先于 pitch/roll/yaw"
    )

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(position, orientation_wxyz)`` numpy 数组.

        Returns:
            ``(position, orientation_wxyz)`` 两个 float ndarray;
            未提供 ``orientation_wxyz`` 时由欧拉角换算.
        """
        pos = np.asarray(self.position, dtype=float)  # TOML list → ndarray
        if self.orientation_wxyz:  # 显式四元数覆盖欧拉
            return pos, np.asarray(self.orientation_wxyz, dtype=float)  # 配置已是 wxyz
        return pos, rpy_to_quat_wxyz(self.roll, self.pitch, self.yaw, degrees=True)  # TOML 用度


def rpy_to_quat_wxyz(
    roll: float,
    pitch: float,
    yaw: float,
    *,
    degrees: bool = True,
) -> np.ndarray:
    """内旋 ZYX(yaw → pitch → roll) → ``(w,x,y,z)``.

    Args:
        roll: 绕 X 轴旋转.
        pitch: 绕 Y 轴旋转.
        yaw: 绕 Z 轴旋转.
        degrees: True 时输入为度(TOML 约定);False 时为弧度.

    Returns:
        单位四元数 ``(w,x,y,z)`` 的 1D ndarray.
    """
    if degrees:
        roll, pitch, yaw = (float(np.deg2rad(v)) for v in (roll, pitch, yaw))  # 半角公式用弧度
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)  # roll 半角
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)  # pitch 半角
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)  # yaw 半角
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,  # w
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy,  # z
        ],
        dtype=float,
    )


def quat_wxyz_to_rpy(q: Sequence[float], *, degrees: bool = True) -> tuple[float, float, float]:
    """``(w,x,y,z)`` → 内旋 ZYX 的 ``(roll, pitch, yaw)``,与 ``rpy_to_quat_wxyz`` 互逆.

    Args:
        q: 四元数 ``(w,x,y,z)``.
        degrees: True 返回度(TOML / 日志约定);False 返回弧度.

    Returns:
        ``(roll, pitch, yaw)``;单位由 ``degrees`` 决定.
    """
    w, x, y, z = (float(v) for v in q)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    # 万向锁附近 asin 的入参会被浮点误差顶出 [-1,1],先夹住
    pitch = np.arcsin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    if degrees:
        return float(np.rad2deg(roll)), float(np.rad2deg(pitch)), float(np.rad2deg(yaw))
    return float(roll), float(pitch), float(yaw)


def quat_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray:
    """单位四元数 ``(w,x,y,z)`` → 3x3 旋转矩阵.

    Args:
        q: 四元数 ``(w,x,y,z)``;模过小则当无旋转.

    Returns:
        3x3 旋转矩阵;退化四元数返回单位阵.
    """
    w, x, y, z = (float(v) for v in q)  # 接受 list/tuple/ndarray
    n = w * w + x * x + y * y + z * z  # 模平方;过小视为零旋转
    if n < 1e-12:  # 退化四元数当无旋转
        return np.eye(3, dtype=float)  # 当成单位旋转,避免 NaN
    s = 2.0 / n  # 非单位四元数也按此归一进矩阵
    wx, wy, wz = s * w * x, s * w * y, s * w * z  # 旋转矩阵交叉项
    xx, xy, xz = s * x * x, s * x * y, s * x * z  # x 相关对角/交叉项
    yy, yz, zz = s * y * y, s * y * z, s * z * z  # y/z 对角与 yz
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )


def matrix_to_quat_wxyz(rm: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 → ``(w,x,y,z)``(Shepperd 稳定分支).

    Args:
        rm: 3x3 旋转矩阵;返回前会再单位化以防舍入.

    Returns:
        单位四元数 ``(w,x,y,z)`` 的 1D ndarray.
    """
    m = np.asarray(rm, dtype=float)  # 允许非 ndarray
    t = float(np.trace(m))  # 迹最大时走 w 分支,数值最稳
    if t > 0.0:  # w 最大,数值最稳
        s = np.sqrt(t + 1.0) * 2.0  # Shepperd 缩放
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:  # x 最大
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0  # Shepperd 缩放
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:  # y 最大
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0  # Shepperd 缩放
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:  # z 最大
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0  # Shepperd 缩放
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)  # 组装后再单位化
    return q / max(np.linalg.norm(q), 1e-12)  # 再单位化,防舍入


def look_at_quat_wxyz(
    eye: Sequence[float],
    target: Sequence[float],
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """USD 相机朝向:本地 ``-Z`` 指向 ``target``.

    Args:
        eye: 相机世界位置.
        target: 注视点世界位置.
        up: 世界上方向,默认 ``+Z``.

    Returns:
        单位四元数 ``(w,x,y,z)``.
    """
    eye_v = np.asarray(eye, dtype=float).reshape(-1)[:3]
    tgt = np.asarray(target, dtype=float).reshape(-1)[:3]
    fwd = tgt - eye_v
    norm = float(np.linalg.norm(fwd))
    if norm < 1e-8:
        return rpy_to_quat_wxyz(90.0, 0.0, 0.0, degrees=True)
    fwd = fwd / norm
    up_v = np.asarray(up, dtype=float).reshape(-1)[:3]
    z_axis = -fwd
    x_axis = np.cross(up_v, z_axis)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-8:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0], dtype=float), z_axis)
        x_norm = float(np.linalg.norm(x_axis))
    x_axis = x_axis / max(x_norm, 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    return matrix_to_quat_wxyz(np.stack([x_axis, y_axis, z_axis], axis=1))


def pose_to_matrix(
    position: Sequence[float],
    orientation_wxyz: Sequence[float] = _DEFAULT_ORIENTATION_WXYZ,
) -> np.ndarray:
    """位姿 → 4x4 齐次矩阵.

    Args:
        position: 平移 ``(x,y,z)``.
        orientation_wxyz: 旋转 ``(w,x,y,z)``;默认单位四元数.

    Returns:
        4x4 齐次变换矩阵.
    """
    t = np.eye(4, dtype=float)  # 齐次单位阵,再填旋转与平移
    t[:3, :3] = quat_wxyz_to_matrix(orientation_wxyz)  # 旋转块
    t[:3, 3] = np.asarray(position, dtype=float)  # 平移列
    return t  # 给 compose_pose 做矩阵乘法


def matrix_to_pose(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4x4 → ``(position, orientation_wxyz)``.

    Args:
        t: 4x4 齐次变换矩阵.

    Returns:
        ``(position, orientation_wxyz)`` 两个 ndarray.
    """
    m = np.asarray(t, dtype=float)  # 允许 list/tuple 传入
    return m[:3, 3].copy(), matrix_to_quat_wxyz(m[:3, :3])  # 平移列 + 旋转块


def compose_pose(
    parent_pos: Sequence[float],
    parent_quat: Sequence[float],
    rel_pos: Sequence[float],
    rel_quat: Sequence[float] = _DEFAULT_ORIENTATION_WXYZ,
) -> tuple[np.ndarray, np.ndarray]:
    """``T_world = T_parent @ T_rel``.

    Args:
        parent_pos: 父坐标系原点的世界平移.
        parent_quat: 父坐标系世界姿态 ``(w,x,y,z)``.
        rel_pos: 相对父系的平移(如物体相对桌面).
        rel_quat: 相对父系的姿态;默认无额外旋转.

    Returns:
        合成后的世界 ``(position, orientation_wxyz)``.
    """
    # 先变齐次再相乘:T_world = T_parent @ T_rel
    tw = pose_to_matrix(parent_pos, parent_quat) @ pose_to_matrix(rel_pos, rel_quat)  # 右乘相对
    return matrix_to_pose(tw)  # 拆回 (pos, wxyz) 给 Scene/Robot 缓存


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Hamilton 积 ``a ⊗ b``,两个四元数均为 ``(w,x,y,z)``.

    Args:
        a: 左乘四元数.
        b: 右乘四元数.

    Returns:
        单位化后的 ``(w,x,y,z)``.
    """
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    q = np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )
    return q / max(float(np.linalg.norm(q)), 1e-12)


def quat_rotate(quat_wxyz: Sequence[float], vec: Sequence[float]) -> np.ndarray:
    """用四元数 ``(w,x,y,z)`` 旋转向量(不做平移).

    Args:
        quat_wxyz: 旋转 ``(w,x,y,z)``.
        vec: 被旋转的 3D 向量.

    Returns:
        旋转后的 3D 向量,shape ``(3,)``.
    """
    return quat_wxyz_to_matrix(quat_wxyz) @ np.asarray(vec, dtype=float)  # 纯旋转,无平移


def quat_inv_rotate(quat_wxyz: Sequence[float], vec: Sequence[float]) -> np.ndarray:
    """用四元数 ``(w,x,y,z)`` 的逆旋转向量(旋转矩阵为正交阵,逆即转置).

    Args:
        quat_wxyz: 旋转 ``(w,x,y,z)``;本函数用其转置.
        vec: 被逆旋转的 3D 向量.

    Returns:
        逆旋转后的 3D 向量,shape ``(3,)``.
    """
    return quat_wxyz_to_matrix(quat_wxyz).T @ np.asarray(vec, dtype=float)  # R.T ≡ R^{-1}


def read_prim_world_pose(stage: object, prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """从 USD stage 读取 prim 当前世界位姿(含物理驱动后的实时变换);无效则 ``None``.

    与摆放阶段缓存的静态位姿不同,本函数每次都重新计算,
    适合在物理仿真步进过程中轮询被抓物体等动态 prim。

    Args:
        stage: 当前 USD stage;None 时直接返回 None(stub).
        prim_path: 目标 prim 路径,会先 ``normalize_prim_path``.

    Returns:
        世界 ``(position, orientation_wxyz)``;stage/prim 无效或非 Xformable
        则为 ``None``.
    """
    if stage is None:
        return None  # 未绑定 stage(stub)时无法查
    # 函数内 import,避免 pose 配置路径拉起整个 isaacsim_utils
    from real2sim2real.utils.isaacsim_utils import normalize_prim_path  # noqa: PLC0415

    # Kit 运行时对象:用字符串路径取 prim,不存在则后面判 Invalid
    prim = stage.GetPrimAtPath(normalize_prim_path(prim_path))  # type: ignore[attr-defined]
    if prim is None or not prim.IsValid():
        return None  # 路径尚未 spawn 或已被删
    try:
        from pxr import Usd, UsdGeom  # noqa: PLC0415  # 仅 Isaac 进程有 pxr

        xf = UsdGeom.Xformable(prim)  # 非 Xformable 会在下面 except
        mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())  # 含父链
        trans = mat.ExtractTranslation()  # 世界平移
        quat = mat.ExtractRotationQuat()  # pxr 四元数
        imag = quat.GetImaginary()  # (x, y, z)
        pos = np.array([trans[0], trans[1], trans[2]], dtype=float)  # pxr → numpy
        ori = np.array([quat.GetReal(), imag[0], imag[1], imag[2]], dtype=float)  # wxyz
        return pos, ori  # 与 poses 缓存同约定 (pos, wxyz)
    except Exception:  # noqa: BLE001  # 非 Xformable 等
        return None  # 读失败不当致命错误,Job 可跳过该 prim


def axis_shift_into_span(
    item_min: float,
    item_max: float,
    span_min: float,
    span_max: float,
) -> float:
    """把区间 ``[item_min, item_max]`` 平移进 ``[span_min, span_max]``.

    单轴计算.物体比支撑面更宽时居中.

    Args:
        item_min: 物体 AABB 在该轴上的下沿.
        item_max: 物体 AABB 在该轴上的上沿.
        span_min: 支撑面 AABB 在该轴上的下沿.
        span_max: 支撑面 AABB 在该轴上的上沿.

    Returns:
        应加到物体世界坐标上的位移;已在区间内则为 ``0``.
    """
    item_span = item_max - item_min
    host_span = span_max - span_min
    if item_span >= host_span:
        return 0.5 * (span_min + span_max) - 0.5 * (item_min + item_max)
    shift = 0.0
    if item_min < span_min:
        shift += span_min - item_min
    if item_max + shift > span_max:
        shift += span_max - (item_max + shift)
    return shift


def snap_aabb_onto_support(
    world_pos: np.ndarray,
    base_min: np.ndarray,
    base_max: np.ndarray,
    *,
    support_z: float | None,
    rel_z: float,
    table_min: np.ndarray | None = None,
    table_max: np.ndarray | None = None,
) -> np.ndarray:
    """按底座 AABB 把根位置贴到支撑面,并可选把 XY 收进桌子范围.

    Args:
        world_pos: 当前世界位置.
        base_min: 底座 AABB 下沿.
        base_max: 底座 AABB 上沿.
        support_z: 支撑面世界 z;None 则不改高度.
        rel_z: 相对支撑面的间隙(米),0 表示贴合.
        table_min: 桌子 AABB 下沿;None 则不收 XY.
        table_max: 桌子 AABB 上沿;None 则不收 XY.

    Returns:
        贴合后的世界位置副本;朝向由调用方保持.
    """
    snapped = np.asarray(world_pos, dtype=float).copy()
    if support_z is not None:
        snapped[2] += (float(support_z) + float(rel_z)) - float(base_min[2])
    if table_min is not None and table_max is not None:
        snapped[0] += axis_shift_into_span(
            float(base_min[0]), float(base_max[0]), float(table_min[0]), float(table_max[0])
        )
        snapped[1] += axis_shift_into_span(
            float(base_min[1]), float(base_max[1]), float(table_min[1]), float(table_max[1])
        )
    return snapped


def alpha_beta_track(
    state: dict[str, Any] | None,
    target_pos: np.ndarray,
    target_yaw: float,
    *,
    elapsed: float,
    tau: float,
) -> tuple[dict[str, Any], np.ndarray, float]:
    """alpha-beta 跟踪位置与航向:压振荡,匀速输入无稳态滞后.

    Args:
        state: 上次滤波状态;None 或 ``tau≈0`` 则初始化为当前测量.
        target_pos: 本步测量位置.
        target_yaw: 本步测量航向(弧度).
        elapsed: 距上次调用的时间(秒).
        tau: 时间常数(秒);``<=0`` 表示不滤波.

    Returns:
        ``(新状态, 滤波位置, 滤波航向弧度)``.
    """
    target = np.asarray(target_pos, dtype=float)[:3].copy()
    if state is None or tau <= 1e-6:
        fresh = {
            "pos": target.copy(),
            "vel": np.zeros(3, dtype=float),
            "yaw": float(target_yaw),
            "yaw_rate": 0.0,
        }
        return fresh, target, float(target_yaw)
    if elapsed <= 0.0:
        return state, np.asarray(state["pos"], dtype=float).copy(), float(state["yaw"])
    alpha = float(1.0 - np.exp(-elapsed / tau))
    beta = alpha * alpha / (2.0 - alpha)
    pos = np.asarray(state["pos"], dtype=float) + np.asarray(state["vel"]) * elapsed
    pos_err = target - pos
    state["pos"] = pos + alpha * pos_err
    state["vel"] = np.asarray(state["vel"], dtype=float) + (beta / elapsed) * pos_err
    yaw = float(state["yaw"]) + float(state["yaw_rate"]) * elapsed
    yaw_err = float((target_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)
    state["yaw"] = yaw + alpha * yaw_err
    state["yaw_rate"] = float(state["yaw_rate"]) + (beta / elapsed) * yaw_err
    return state, np.asarray(state["pos"], dtype=float).copy(), float(state["yaw"])
