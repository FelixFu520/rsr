"""对 LeRobot 落盘结果做事后数据校验(纯 numpy,无需 Isaac).

供 ``TabletopJob`` 采集结束后自检。
"""

from __future__ import annotations  # 延后求值注解

import json  # 读 meta.json
from dataclasses import dataclass, field  # ValidationResult 聚合
from pathlib import Path  # episode 目录 / 相对视频路径
from typing import Any  # details / meta 为自由 dict

import numpy as np  # 关节有限性

_TRAJ_REQUIRED_KEYS = (  # trajectory.npz 必须有的字段
    "step",
    "joint_positions",
    "ee_position",
    "ee_orientation_wxyz",
    "gripper_position",
    "action",
)


@dataclass
class ValidationResult:
    """单次校验的聚合结果,``to_dict`` 给 CLI JSON 输出."""

    ok: bool  # 所有 checks 为 True 且 errors 为空
    episode_dir: str
    checks: dict[str, bool] = field(default_factory=dict)  # 分项是否通过
    details: dict[str, Any] = field(default_factory=dict)  # 帧数等
    errors: list[str] = field(default_factory=list)  # 人读失败原因

    def to_dict(self) -> dict[str, Any]:
        """序列化,供 ``json.dumps``.

        Returns:
            含 ``ok`` / ``episode_dir`` / ``checks`` / ``details`` / ``errors`` 的 dict.
        """
        return {  # CLI json.dumps;键与字段对齐,Exit code 看 ok
            "ok": self.ok,
            "episode_dir": self.episode_dir,
            "checks": self.checks,
            "details": self.details,
            "errors": self.errors,
        }


def validate_episode(
    episode_dir: str | Path,
    *,
    expected_cameras: list[str] | None = None,
    min_steps: int = 1,
    min_frames_per_camera: int = 1,
    require_success_status: bool = True,
) -> ValidationResult:
    """校验单个 episode 目录下落盘数据的完整性.

    自动识别 native(``meta.json`` + ``trajectory.npz``)与 LeRobot v2.1
    (``meta/info.json`` + parquet/mp4).

    Args:
        episode_dir: ``open()`` 返回的目录(native 为时间戳子目录,lerobot 为数据集根).
        expected_cameras: 期望存在的相机名(如 ``["overhead_cam", "wrist_cam"]``);
            为空则校验目录下实际存在的所有相机子目录.
        min_steps: 轨迹最少步数.
        min_frames_per_camera: 每路相机最少帧数;lerobot 改为检查 mp4 存在.
        require_success_status: 是否要求 ``status == "success"``.

    Returns:
        聚合结果;目录缺失或任一分项失败时 ``ok`` 为 False.
    """
    ep_dir = Path(episode_dir)
    result = ValidationResult(ok=True, episode_dir=str(ep_dir))  # 先假设通过,分项失败再翻

    if not ep_dir.is_dir():
        result.ok = False  # 目录都不在,后续文件检查无意义
        result.errors.append(f"episode 目录不存在: {ep_dir}")
        result.checks["episode_dir_exists"] = False
        return result  # 后续文件都不必查
    result.checks["episode_dir_exists"] = True

    if (ep_dir / "meta" / "info.json").is_file():
        _validate_lerobot_v2(  # v2.1 布局,走 parquet/mp4 而非 npz
            ep_dir,
            result,
            expected_cameras=expected_cameras,
            min_steps=min_steps,
            require_success_status=require_success_status,
        )
        result.ok = all(result.checks.values()) and not result.errors  # 任一分项失败则整集失败
        return result

    meta = _check_meta(ep_dir, result, require_success_status=require_success_status)
    traj = _check_trajectory(ep_dir, result, min_steps=min_steps)
    _check_cameras(
        ep_dir,
        result,
        expected_cameras=expected_cameras or _discover_cameras(ep_dir),  # 未指定则扫 rgb/
        min_frames=min_frames_per_camera,
        num_steps=traj.get("num_steps") if traj else None,  # 轨迹缺则相机不比对步数
    )
    _check_videos(ep_dir, meta, result)

    result.ok = all(result.checks.values()) and not result.errors  # 任一分项失败则整集失败
    return result


def _validate_lerobot_v2(
    root: Path,
    result: ValidationResult,
    *,
    expected_cameras: list[str] | None,
    min_steps: int,
    require_success_status: bool,
) -> None:
    """校验 LeRobot v2.1 数据集里最近一条 episode.

    Args:
        root: 数据集根(含 ``meta/info.json``).
        result: 就地写入 checks / errors / details.
        expected_cameras: 期望相机短名或 ``observation.images.*``;空则用 features.
        min_steps: ``episodes.jsonl`` 最近一条 ``length`` 下限.
        require_success_status: 是否要求最近一条 ``status == success``.

    Returns:
        None.
    """
    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        result.checks["meta_json_valid"] = False
        result.errors.append(f"meta/info.json 解析失败: {exc}")  # 截断/手工改坏
        return  # 后续路径模板都依赖 info,无法继续
    result.checks["meta_json_exists"] = True
    result.checks["meta_json_valid"] = True
    result.details["meta"] = info  # CLI JSON 里带上,便于对照 fps / features
    version = str(info.get("codebase_version", ""))
    result.checks["lerobot_version"] = version.startswith("v2")
    if not version.startswith("v2"):
        result.errors.append(f"期望 LeRobot v2.x,实际 codebase_version={version!r}")  # v3 布局不同

    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    if not episodes:
        result.checks["trajectory_exists"] = False
        result.errors.append("meta/episodes.jsonl 为空")  # 尚未 finalize 或写盘失败
        return  # 没有集号就无法定位 parquet
    last = episodes[-1]  # 采集刚 finalize 的是最后一条
    ep_index = int(last.get("episode_index", len(episodes) - 1))
    length = int(last.get("length", 0))
    status = str(last.get("status", ""))
    result.details["status"] = status
    result.details["episode_index"] = ep_index
    result.details["num_steps"] = length
    if require_success_status:
        result.checks["status_success"] = status == "success"
        if status != "success":
            result.errors.append(f"status 非 success: {status!r}")  # 失败集默认不算合格

    result.checks["trajectory_has_min_steps"] = length >= min_steps
    if length < min_steps:
        result.errors.append(f"轨迹步数不足: {length} < {min_steps}")  # 空集/刚 open 就 finalize

    chunk = ep_index // int(info.get("chunks_size", 1000))  # 与 recorder._CHUNKS_SIZE 对齐
    data_tpl = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    parquet_path = root / data_tpl.format(episode_chunk=chunk, episode_index=ep_index)
    result.checks["trajectory_exists"] = parquet_path.is_file()
    if not parquet_path.is_file():
        result.errors.append(f"parquet 不存在: {parquet_path}")  # jsonl 有记录但 data/ 缺失
        return  # 没表就无法查列
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415  # 延迟导入,native 校验不必装

        table = pq.read_table(parquet_path)  # 整表读入,列名检查 + state 质量
    except Exception as exc:  # noqa: BLE001
        result.checks["trajectory_loadable"] = False
        result.errors.append(f"parquet 加载失败: {exc}")
        return
    result.checks["trajectory_loadable"] = True
    required = ("observation.state", "action", "timestamp", "episode_index", "frame_index")
    missing = [k for k in required if k not in table.column_names]
    result.checks["trajectory_keys_present"] = not missing
    if missing:
        result.errors.append(f"parquet 缺少字段: {missing}")  # schema 不完整,replay 会炸
    if "observation.state" in table.column_names and table.num_rows:
        state = np.asarray(table.column("observation.state").to_pylist(), dtype=float)
        finite_ok = bool(np.all(np.isfinite(state)))
        result.checks["joint_positions_finite"] = finite_ok
        if not finite_ok:
            result.errors.append("observation.state 含 NaN/Inf")  # 仿真读关节失败
        nonzero_ok = bool(np.any(np.abs(state) > 1e-6))
        result.checks["joint_positions_nonzero"] = nonzero_ok
        if not nonzero_ok:
            result.errors.append("observation.state 全为 0,机器人状态疑似未采集")

    video_tpl = str(
        info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
    )
    features = info.get("features") or {}
    video_keys = [
        k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"
    ]
    if expected_cameras:
        wanted = [
            c if c.startswith("observation.images.") else f"observation.images.{c}"
            for c in expected_cameras
        ]  # Job 传短名,info 里是 feature 键
    else:
        wanted = video_keys  # 未指定则 features 里声明的都要有 mp4
    cam_ok = True
    result.details["cameras"] = {}
    for key in wanted:
        rel = video_tpl.format(episode_chunk=chunk, episode_index=ep_index, video_key=key)
        mp4 = root / rel
        exists = mp4.is_file() and mp4.stat().st_size > 0  # 空文件当缺失
        result.details["cameras"][key] = {"exists": exists, "path": str(mp4)}
        if not exists:
            cam_ok = False  # 任一路空文件则 cameras_valid_frames 失败
            result.errors.append(f"相机视频缺失或为空: {key} -> {mp4}")
    result.checks["cameras_present"] = True  # 目录约定存在;有效性看下一键
    result.checks["cameras_valid_frames"] = cam_ok if wanted else True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读 jsonl;缺失则空列表.

    Args:
        path: jsonl 路径;文件不存在视为空,不抛错.

    Returns:
        按文件顺序的 dict 列表;文件不存在则为空列表.
    """
    if not path.is_file():  # 尚未 finalize / 空数据集
        return []  # 当空列表,调用方 len=0
    rows: list[dict[str, Any]] = []  # 一行一条 dict,保持文件顺序
    for line in path.read_text(encoding="utf-8").splitlines():  # 集数很小,整文件读即可
        line = line.strip()  # 去掉手工编辑留下的空白
        if line:  # 空行不是合法 JSON
            rows.append(json.loads(line))  # 空行跳过,兼容手工编辑
    return rows  # episodes[-1] 才是最近一集


def _check_meta(
    ep_dir: Path, result: ValidationResult, *, require_success_status: bool
) -> dict[str, Any] | None:
    """检查 meta.json 存在、可解析,以及可选的 status==success.

    Args:
        ep_dir: native episode 目录(含 ``meta.json``).
        result: 就地写入 checks / errors.
        require_success_status: True 时 ``status`` 必须为 ``success``.

    Returns:
        解析后的 meta dict;文件缺失或 JSON 坏则为 ``None``.
    """
    meta_path = ep_dir / "meta.json"
    if not meta_path.is_file():
        result.checks["meta_json_exists"] = False
        result.errors.append("meta.json 不存在")  # native 布局;lerobot 走 info.json
        return None  # 后续视频检查依赖 meta.videos
    result.checks["meta_json_exists"] = True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        result.checks["meta_json_valid"] = False
        result.errors.append(f"meta.json 解析失败: {exc}")
        return None  # 坏 JSON 不继续,避免 status 误判
    result.checks["meta_json_valid"] = True
    result.details["meta"] = meta
    status = str(meta.get("status", ""))
    result.details["status"] = status
    if require_success_status:
        result.checks["status_success"] = status == "success"
        if status != "success":
            result.errors.append(f"status 非 success: {status!r}")  # 失败集默认不算合格
    return meta


def _check_trajectory(
    ep_dir: Path, result: ValidationResult, *, min_steps: int
) -> dict[str, Any] | None:
    """检查 npz 存在、字段齐全、步数、关节有限且非全零.

    Args:
        ep_dir: native episode 目录(含 ``trajectory.npz``).
        result: 就地写入 checks / errors / details.
        min_steps: ``step`` 列长度下限.

    Returns:
        含 ``num_steps`` 的 dict;npz 缺失或无法加载则为 ``None``.
    """
    traj_path = ep_dir / "trajectory.npz"
    if not traj_path.is_file():
        result.checks["trajectory_exists"] = False
        result.errors.append("trajectory.npz 不存在")  # native 才有 npz;lerobot 是 parquet
        return None  # 相机步数比对需要 num_steps
    result.checks["trajectory_exists"] = True
    try:
        data = np.load(traj_path)
    except Exception as exc:  # noqa: BLE001
        result.checks["trajectory_loadable"] = False
        result.errors.append(f"trajectory.npz 加载失败: {exc}")  # 截断/非 npz
        return None
    result.checks["trajectory_loadable"] = True

    missing = [k for k in _TRAJ_REQUIRED_KEYS if k not in data.files]
    result.checks["trajectory_keys_present"] = not missing
    if missing:
        result.errors.append(f"trajectory.npz 缺少字段: {missing}")  # 旧采集格式

    num_steps = int(data["step"].shape[0]) if "step" in data.files else 0
    result.details["num_steps"] = num_steps
    result.checks["trajectory_has_min_steps"] = num_steps >= min_steps
    if num_steps < min_steps:
        result.errors.append(f"轨迹步数不足: {num_steps} < {min_steps}")

    if "joint_positions" in data.files and num_steps:
        jp = data["joint_positions"]
        finite_ok = bool(np.all(np.isfinite(jp)))
        result.checks["joint_positions_finite"] = finite_ok
        if not finite_ok:
            result.errors.append("joint_positions 含 NaN/Inf")  # Isaac 读关节失败
        nonzero_ok = bool(np.any(np.abs(jp) > 1e-6))  # 全 0 说明没读到 articulation
        result.checks["joint_positions_nonzero"] = nonzero_ok
        if not nonzero_ok:
            result.errors.append("joint_positions 全为 0,机器人状态疑似未采集")

    if "ee_position" in data.files and num_steps:
        ee = data["ee_position"]
        # 只记 check,不追加 error:末端可能未采但仍可训关节
        result.checks["ee_position_finite"] = bool(np.all(np.isfinite(ee)))

    return {"num_steps": num_steps}


def _discover_cameras(ep_dir: Path) -> list[str]:
    """``rgb/`` 下的子目录名即相机名.

    Args:
        ep_dir: native episode 目录;无 ``rgb/`` 则空列表.

    Returns:
        ``rgb/`` 下子目录名的排序列表;无该目录则为空列表.
    """
    rgb_root = ep_dir / "rgb"
    if not rgb_root.is_dir():
        return []  # 未采 RGB 或目录名不同
    return sorted(p.name for p in rgb_root.iterdir() if p.is_dir())


def _check_cameras(
    ep_dir: Path,
    result: ValidationResult,
    *,
    expected_cameras: list[str],
    min_frames: int,
    num_steps: int | None,
) -> None:
    """每路相机目录存在、PNG 够数.

    Args:
        ep_dir: native episode 目录(含 ``rgb/<cam>/*.png``).
        result: 就地写入 checks / errors / details.
        expected_cameras: 要检查的相机名;空则直接判失败.
        min_frames: 每路 PNG 总数下限.
        num_steps: 轨迹步数;本实现未用于比对,保留给调用方签名.

    Returns:
        None.
    """
    if not expected_cameras:  # Job 未指定且 rgb/ 下也扫不到
        result.checks["cameras_present"] = False  # 无法宣称相机齐全
        result.errors.append("未发现任何相机 RGB 输出目录")  # 未采 RGB 或 rgb/ 名不对
        return  # 没有相机名就无法逐路扫 PNG
    result.details["cameras"] = {}  # 每路帧数,CLI JSON 用
    all_ok = True  # 任一路不够数则整项失败
    for cam in expected_cameras:  # 缺一路也扫完,一次报全
        cam_dir = ep_dir / "rgb" / cam  # native 布局;lerobot 走 mp4 不进这里
        cam_report: dict[str, Any] = {}  # 写入 details,CLI JSON 可看每路帧数
        if not cam_dir.is_dir():  # 相机名对不上或没开 save_rgb
            cam_report["exists"] = False  # details 仍占键,便于对照 expected
            all_ok = False  # 整项失败,其它相机继续
            result.errors.append(f"相机 {cam} 输出目录不存在: {cam_dir}")  # 人读路径
            result.details["cameras"][cam] = cam_report  # 先写下失败报告再 continue
            continue  # 其它相机继续查,一次报全
        frames = sorted(cam_dir.glob("*.png"))  # 文件名排序,与 step 序号一致
        cam_report["exists"] = True  # 目录在,下面再数帧
        cam_report["n_frames"] = len(frames)
        if len(frames) < min_frames:  # 空目录或采帧中断
            all_ok = False  # 其它相机继续扫
            result.errors.append(f"相机 {cam} 帧数不足: {len(frames)} < {min_frames}")
        result.details["cameras"][cam] = cam_report  # 成功失败都留该路报告
    result.checks["cameras_present"] = True  # 期望列表非空且都查过
    result.checks["cameras_valid_frames"] = all_ok  # 任一路不够则 False


def _check_videos(ep_dir: Path, meta: dict[str, Any] | None, result: ValidationResult) -> None:
    """若 meta 声明了 videos,检查文件存在且非空.

    Args:
        ep_dir: native episode 目录,用于相对路径回退.
        meta: ``meta.json`` 内容;无 ``videos`` 则跳过本项.
        result: 就地写入 ``videos_present``.

    Returns:
        None.
    """
    if not meta or not meta.get("videos"):
        return  # 没导出视频则跳过此项,不判失败
    ok = True
    for name, rel in meta["videos"].items():
        candidates = [Path(rel), ep_dir / Path(rel).name]  # 绝对或只取文件名
        exists = any(c.is_file() and c.stat().st_size > 0 for c in candidates)
        if not exists:
            ok = False
            result.errors.append(f"视频缺失或为空: {name} -> {rel}")  # 声明了却没导出
    result.checks["videos_present"] = ok
