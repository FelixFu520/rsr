"""可复现的场景位置抖动与包裹主体抓点采样。"""

from __future__ import annotations

import numpy as np


def sample_grasp_offsets(
    minimum: np.ndarray,
    maximum: np.ndarray,
    fraction_xy: tuple[float, float],
    count: int,
    rng: np.random.Generator,
    *,
    min_distance_m: float = 0.0,
    previous_offset: np.ndarray | None = None,
) -> list[np.ndarray]:
    """在包裹中心矩形内采样 XY 偏移,避开中心及上一成功抓点的邻域."""
    half = 0.5 * (np.asarray(maximum) - np.asarray(minimum))
    extent = half[:2] * np.asarray(fraction_xy)
    if not np.any(extent > 0):
        return []
    offsets: list[np.ndarray] = []
    for _ in range(max(1, count) * 64):
        xy = rng.uniform(-extent, extent)
        if np.linalg.norm(xy) < min_distance_m:
            continue
        if previous_offset is not None:
            if np.linalg.norm(xy - np.asarray(previous_offset)[:2]) < min_distance_m:
                continue
        offsets.append(np.array([*xy, 0.0]))
        if len(offsets) >= count:
            break
    return offsets


def jitter_world_xy(
    base_xyz: np.ndarray,
    radius: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """在 XY 上均匀抖动,Z 不变;radius<=0 则不偏移.

    Args:
        base_xyz: 世界位置 ``(x, y, z)``.
        radius: XY 半边长(米);<=0 则原样返回.
        rng: 随机数生成器,保证可复现.

    Returns:
        ``(pos, dx, dy)``:抖动后世界坐标与实际 XY 偏移(米).
    """
    pos = np.asarray(base_xyz, dtype=float).reshape(-1).copy()  # copy,避免改调用方缓存
    dx = dy = 0.0  # radius<=0 时调用方仍要写 offset
    if radius > 0:  # 均匀方框抖动,不是圆盘
        dx = float(rng.uniform(-radius, radius))
        dy = float(rng.uniform(-radius, radius))  # 与 dx 独立均匀,方框而非圆盘
        pos[0] += dx  # 只动 XY,Z 保持原高度
        pos[1] += dy
    return pos, dx, dy
