"""位姿配置与齐次变换工具.

TOML 里用欧拉角(度);内部统一 ``(position, wxyz)``.
``compose_pose`` 用于 Scene/Robot 的相对摆放.
"""

from __future__ import annotations

from collections.abc import Sequence

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
