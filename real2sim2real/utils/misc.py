from typing import Any

import numpy as np


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


def math_atan2_deg(w: float, x: float, y: float, z: float) -> float:
    """四元数 yaw(度),ZYX 内旋.

    Args:
        w: 四元数 w.
        x: 四元数 x.
        y: 四元数 y.
        z: 四元数 z.

    Returns:
        航向角(度).
    """
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return float(np.rad2deg(np.arctan2(siny, cosy)))


def extract_joint_path(result: Any, n_dof: int) -> np.ndarray | None:
    """从 cuRobo ``plan_pose`` 结果取出 ``(N, n_dof)`` 关节轨迹.

    Args:
        result: ``MotionPlanner.plan_pose`` 的返回值,字段名随 cuRobo 版本而变.
        n_dof: 截取的臂关节列数,Franka 为 7.

    Returns:
        ``(N, n_dof)`` 关节轨迹;无解或解析失败则为 None.
    """
    if result is None:  # plan_pose 返回空
        return None  # 调用方当无解
    success = getattr(result, "success", None)  # 有的版本是 bool,有的是 tensor
    try:
        if success is not None and hasattr(success, "any"):  # GPU tensor:任一 batch 成功即可
            if not bool(success.any().item()):  # 全失败
                return None  # 无轨迹
        elif success is False:  # 纯 Python bool
            return None  # 明确失败
    except Exception:  # noqa: BLE001  # success 形状/设备异常,当未知,继续抽轨迹
        pass

    candidates: list[Any] = []  # 按优先级收集可能的轨迹对象
    if hasattr(result, "get_interpolated_plan"):  # 插值后的稠密路点,优先
        try:
            candidates.append(result.get_interpolated_plan())  # 插值关节路点
        except Exception:  # noqa: BLE001  # 方法存在但调用失败,忽略
            pass
    for attr in ("interpolated_plan", "js_path", "position", "solution"):  # 各版本字段名
        val = getattr(result, attr, None)  # 缺字段则为 None
        if val is not None:  # 有值才进候选
            candidates.append(val)

    for val in candidates:  # 逐个试,能转成 (N, n_dof) 就用
        try:
            if hasattr(val, "position"):  # JointState 包一层 position
                val = val.position  # 剥到关节角张量
            if hasattr(val, "detach"):  # torch.Tensor → numpy
                arr = val.detach().cpu().numpy()  # 拷回 CPU,规划结果在 GPU
            else:
                arr = np.asarray(val)  # 已是 ndarray / list
            arr = np.asarray(arr, dtype=float)  # 统一 float
            while arr.ndim > 2:  # 去掉 batch/horizon 等多余维
                arr = arr.reshape(-1, arr.shape[-1]) if arr.shape[-1] >= n_dof else arr[0]
            if arr.ndim == 1:  # 单路点 (n_dof,)
                arr = arr.reshape(1, -1)  # 升成 (1, n_dof)
            if arr.ndim >= 2 and arr.shape[-1] >= 1:  # 至少有一列关节
                return arr.reshape(-1, arr.shape[-1])[:, :n_dof]  # 行=路点,列截到臂 DOF
        except Exception:  # noqa: BLE001  # 这个候选解析失败,试下一个
            continue
    return None  # 所有候选都抽不出轨迹


def wobble_stats(roll: list[float], pitch: list[float], z: list[float]) -> dict[str, float]:
    """骨盆晃动幅度,用来判断"画面抖"是步态本身还是相机挂法的问题.

    人形正常行走时骨盆横滚/俯仰大约在几度、抬降在 2~3cm;明显超出说明是
    速度策略在打晃,该去调控制而不是调相机.

    Args:
        roll: 逐帧骨盆横滚(度).
        pitch: 逐帧骨盆俯仰(度).
        z: 逐帧骨盆高度(米).

    Returns:
        峰峰值/RMS 字典;``roll`` 为空则 ``{}``.
    """
    if not roll:
        return {}
    arr_r, arr_p, arr_z = np.asarray(roll), np.asarray(pitch), np.asarray(z)
    return {
        "roll_pp_deg": round(float(arr_r.max() - arr_r.min()), 2),
        "pitch_pp_deg": round(float(arr_p.max() - arr_p.min()), 2),
        "roll_rms_deg": round(float(arr_r.std()), 2),
        "pitch_rms_deg": round(float(arr_p.std()), 2),
        "z_pp_cm": round(float(arr_z.max() - arr_z.min()) * 100.0, 2),
    }


def emit_log(obj: Any, level: str, event: str, **kwargs: Any) -> None:
    """有 ``log`` 才打;stub 静默.

    Args:
        obj: 带 ``log`` 的对象.
        level: ``info`` / ``warning`` / ``error``.
        event: 事件名.
        **kwargs: 结构化字段.

    Returns:
        None.
    """
    log = getattr(obj, "log", None)
    if log is None:
        return
    getattr(log, level)(event, **kwargs)
