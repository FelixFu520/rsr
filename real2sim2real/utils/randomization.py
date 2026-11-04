"""有界且可复现的包裹主体抓点采样。"""

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
