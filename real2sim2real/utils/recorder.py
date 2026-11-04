"""LeRobot 落盘:工厂 + v2.1 parquet/mp4 + PNG/视频辅助.

不依赖 Isaac;PNG 优先 PIL,失败再 imageio/cv2,最后纯 zlib.
视频优先 ffmpeg,否则 imageio.

LeRobot v2.1 布局与 HuggingFace ``codebase_version=v2.1`` 对齐,不依赖 ``lerobot`` 包::

    <output_dir>/
        meta/info.json
        meta/episodes.jsonl
        meta/tasks.jsonl
        meta/episodes_stats.jsonl
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/observation.images.<cam>/episode_000000.mp4
"""

from __future__ import annotations  # 延后求值注解

import json  # info.json / jsonl
import shutil  # which(ffmpeg) / rmtree
import subprocess  # 调 ffmpeg
from abc import ABC, abstractmethod  # 落盘器公共接口
from pathlib import Path  # 数据集根与相对 chunk 路径
from typing import Any  # parquet table / 运行时 dict

import numpy as np  # 状态向量与图像统计

from real2sim2real.core.logger import LoggerMixin  # self.log
from real2sim2real.utils.recorder_config import JobRecorderConfig  # [job.recorder] 选型

# 每个 chunk 目录最多多少条 episode;与官方 DEFAULT_CHUNKS_SIZE 一致
_CHUNKS_SIZE = 1000
_CODEBASE_VERSION = "v2.1"
_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
# Franka Panda 关节名;长度不够时截断,超出则补 motor_i
_FRANKA_MOTORS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]


def create_recorder(
    rec_cfg: JobRecorderConfig,
    *,
    camera_names: list[str] | None = None,
) -> LeRobotRecorder:
    """按 ``[job.recorder].format`` 构造落盘器.

    新格式(如 ``lerobot_v3``)在此分支挂实现类即可.

    Args:
        rec_cfg: ``[job.recorder]`` 嵌套配置(路径 / fps / 是否写 RGB).
        camera_names: 期望相机名;空则 finalize 时按实际 RGB 键推断.

    Returns:
        ``LeRobotRecorder`` 子类实例.

    Raises:
        NotImplementedError: ``format`` 为 ``lerobot_v3``.
        ValueError: 未知 ``format``.
    """
    fmt = rec_cfg.format  # 小写字符串,与 Literal 白名单对齐
    if fmt == "lerobot_v2":  # 当前唯一落地实现
        return LeRobotV2Recorder(  # 把 [job.recorder] 字段摊进构造
            rec_cfg.output_dir,
            save_rgb=rec_cfg.save_rgb,
            save_depth=rec_cfg.save_depth,
            fps=rec_cfg.fps,
            task=rec_cfg.task,
            robot_type=rec_cfg.robot_type,
            camera_names=camera_names,
        )
    if fmt == "lerobot_v3":  # 配置 Literal 已占位,实现未接
        raise NotImplementedError("job.recorder.format=lerobot_v3 尚未实现")  # 别 silently 落成 v2
    raise ValueError(f"未知 job.recorder.format: {fmt!r}")  # extra=forbid 仍可能手改 settings


class LeRobotRecorder(ABC, LoggerMixin):
    """LeRobot 落盘器公共接口:``open`` → ``record_step``* → ``finalize``.

    Job 只依赖本契约;写盘格式由子类决定(v2.1 parquet/mp4 等).
    多次 ``open``/``finalize`` 追加 episode.本类只放各版本共用的构造字段
    与生命周期槽位,不写具体 schema.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_rgb: bool = True,
        save_depth: bool = False,
        fps: int = 15,
        task: str = "pick and place",
        robot_type: str = "franka",
        camera_names: list[str] | None = None,
    ) -> None:
        """只记根目录与开关;真正写盘在子类 ``finalize``.

        Args:
            output_dir: 数据集根(各版本的 ``meta`` / 数据文件都在其下).
            save_rgb: 是否写相机画面.
            save_depth: 是否写深度;子类可不支持.
            fps: 数据集与视频帧率.
            task: 任务语言描述.
            robot_type: 机型名,写入数据集元数据.
            camera_names: 期望相机名;空则按实际 RGB 键推断.

        Returns:
            None.
        """
        super().__init__()  # LoggerMixin
        self.root = Path(output_dir)  # 统一 Path,后续相对路径都相对此根
        self.save_rgb = save_rgb  # finalize 时是否编码画面
        self.save_depth = save_depth  # 子类可不写深度列
        self.fps = int(fps)  # timestamp 与视频帧率共用
        self.task = str(task)  # 任务语言描述
        self.robot_type = str(robot_type)  # 数据集元数据机型
        self.camera_names = list(camera_names or [])  # copy,避免调用方事后改列表
        self.episode_dir: Path | None = None  # Job/validate 当数据集根或本集目录用
        self._finalized = True  # 尚未 open 时视为已收尾,finalize no-op
        self._meta: dict[str, Any] = {}  # open 传入的任务快照 + status

    @abstractmethod
    def open(self, *, meta: dict[str, Any] | None = None) -> Path:
        """开始一条新 episode;返回供 ``validate_episode`` 使用的路径.

        Args:
            meta: 写入本集的任务快照(集号/抓取物等);None 则空 dict.

        Returns:
            数据集根或本集目录,供 Job 传给 ``validate_episode``.
        """

    @abstractmethod
    def record_step(
        self,
        *,
        step_idx: int,
        robot_state: dict[str, Any],
        action: np.ndarray | list[float] | None,
        planner_phase: str = "",
        object_pose: tuple[np.ndarray, np.ndarray] | None = None,
        rgb_by_camera: dict[str, np.ndarray | None] | None = None,
        depth_by_camera: dict[str, np.ndarray | None] | None = None,
    ) -> None:
        """记录一步状态/动作与可选 RGB/深度.

        Args:
            step_idx: 本集第几帧,从 0 计;同时用来算 ``timestamp``.
            robot_state: 机型观测,至少含 ``joint_positions``.
            action: 本步下发的控制目标;None 则回退到当前观测,避免空列.
            planner_phase: 规划阶段名,便于事后对齐轨迹段.
            object_pose: 目标物世界 ``(pos, quat_wxyz)``;None 则不写物体列.
            rgb_by_camera: 相机名 → RGB;``None`` 帧跳过该路.
            depth_by_camera: 相机名 → 深度;子类可不写.

        Returns:
            None.

        Raises:
            RuntimeError: 尚未 ``open``.
        """

    @abstractmethod
    def finalize(
        self,
        *,
        status: str,
        extra: dict[str, Any] | None = None,
        export_video: bool = True,
        video_fps: int = 15,
    ) -> None:
        """把当前 episode 落盘并更新元数据.

        未 open 或已 finalize 则 no-op.步数为 0 时跳过写盘.

        Args:
            status: 本集结束状态(``success`` / ``failed`` / ``timeout`` 等).
            extra: 写入本集的附加统计(步数/抬升等);None 则只用 status.
            export_video: 调用方开关;子类可忽略(有 RGB 就必须出视频).
            video_fps: 覆盖构造时 fps;0 则回退 ``self.fps``.

        Returns:
            None.
        """

    def _require_open(self) -> None:
        """``record_step`` 前确认本集已 ``open`` 且未 ``finalize``.

        Returns:
            None.

        Raises:
            RuntimeError: 尚未 ``open`` 或已经 ``finalize``.
        """
        if self.episode_dir is None or self._finalized:
            raise RuntimeError(f"{type(self).__name__}.open() 尚未调用")  # 避免写到错误集


class LeRobotV2Recorder(LeRobotRecorder):
    """将采集过程写入 LeRobot v2.1 数据集根目录.

    多次 ``open``/``finalize`` 会追加 episode(可接续已有数据集).
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_rgb: bool = True,
        save_depth: bool = False,
        fps: int = 15,
        task: str = "pick and place",
        robot_type: str = "franka",
        camera_names: list[str] | None = None,
    ) -> None:
        """公共字段见 ``LeRobotRecorder``;本类另缓存 parquet / 图像统计.

        Args:
            output_dir: 数据集根(``meta``/``data``/``videos`` 都在其下).
            save_rgb: 是否写相机 mp4.
            save_depth: 预留;v2.1 本实现不写深度.
            fps: ``info.json`` 与视频帧率.
            task: 任务语言描述.
            robot_type: ``info.json`` 机型.
            camera_names: 期望相机名;空则按实际 RGB 键推断.

        Returns:
            None.
        """
        super().__init__(
            output_dir,
            save_rgb=save_rgb,
            save_depth=save_depth,
            fps=fps,
            task=task,
            robot_type=robot_type,
            camera_names=camera_names,
        )
        self._info: dict[str, Any] = {}  # 当前 info.json 缓存,finalize 时回写
        self._steps: list[dict[str, Any]] = []  # 本集逐步缓存,finalize 堆成 parquet
        self._episode_index = 0  # 下一条要写的 0-based 集号
        self._image_stats: dict[str, _ChannelStats] = {}  # 每路相机累积,写 episodes_stats
        if save_depth:
            self.log.warning("lerobot_v2_depth_unsupported")  # 开关开着也不写,避免 silent 丢数据

    def open(self, *, meta: dict[str, Any] | None = None) -> Path:
        """开始一条新 episode 的内存缓存;返回数据集根路径.

        Args:
            meta: 写入本集的任务快照(集号/抓取物等);None 则空 dict.

        Returns:
            数据集根,供 Job 传给 ``validate_episode``.
        """
        self.root.mkdir(parents=True, exist_ok=True)  # 首次采集时父目录可能还不存在
        (self.root / "meta").mkdir(exist_ok=True)  # jsonl / info.json 都在 meta/
        self._info = _load_or_init_info(
            self.root,
            fps=self.fps,
            robot_type=self.robot_type,
        )  # 接续已有数据集时沿用 total_episodes
        self._episode_index = int(self._info.get("total_episodes", 0))
        self.episode_dir = self.root  # v2.1 无时间戳子目录,Job 把根当 episode_dir
        self._steps = []  # 清掉上一集缓存
        self._finalized = False  # 允许 record_step
        self._meta = dict(meta or {})  # copy,避免调用方事后改同一 dict
        self._image_stats = {}  # 本集图像统计从零累积
        self.log.info(
            "lerobot_episode_open",
            path=str(self.root),
            episode_index=self._episode_index,
        )
        return self.root

    def record_step(
        self,
        *,
        step_idx: int,
        robot_state: dict[str, Any],
        action: np.ndarray | list[float] | None,
        planner_phase: str = "",
        object_pose: tuple[np.ndarray, np.ndarray] | None = None,
        rgb_by_camera: dict[str, np.ndarray | None] | None = None,
        depth_by_camera: dict[str, np.ndarray | None] | None = None,
    ) -> None:
        """缓存一步状态/动作,RGB 先写成临时 PNG.

        Args:
            step_idx: 本集第几帧,从 0 计;同时用来算 ``timestamp``.
            robot_state: 机械臂观测,至少含 ``joint_positions``.
            action: 本步下发的关节目标;None 则回退到当前关节,避免空列.
            planner_phase: 规划阶段名,便于事后对齐轨迹段.
            object_pose: 抓取物世界 ``(pos, quat_wxyz)``;None 则不写物体列.
            rgb_by_camera: 相机名 → RGB;``None`` 帧跳过该路.
            depth_by_camera: 预留;v2.1 本实现丢弃.

        Returns:
            None.

        Raises:
            RuntimeError: 尚未 ``open``.
        """
        del depth_by_camera  # v2.1 本实现不写深度
        self._require_open()  # 尚未 open / 已 finalize 则拒绝写入

        entry: dict[str, Any] = {  # 一步一行,finalize 堆 parquet
            "step": step_idx,
            "planner_phase": planner_phase,
            "observation.state": _as_float_list(robot_state.get("joint_positions")),
            "action": _as_float_list(
                action if action is not None else robot_state.get("joint_positions")
            ),  # None 时用观测占位,parquet 列宽才能对齐
            "timestamp": float(step_idx) / float(self.fps),  # 秒;与 info.json fps 一致
        }
        if object_pose is not None:  # 无物体列时 replay 仍可用关节
            entry["object_position"] = _as_float_list(object_pose[0])
            entry["object_orientation_wxyz"] = _as_float_list(object_pose[1])  # wxyz 与 Isaac 一致

        if self.save_rgb and rgb_by_camera:  # 关 RGB 时不写临时 PNG
            tmp_root = _tmp_rgb_dir(self.root, self._episode_index)  # 编码 mp4 前的 PNG 缓存
            for name, rgb in rgb_by_camera.items():
                if rgb is None:
                    continue  # 该帧该路没出图,不占 PNG,stats 也不更新
                key = _video_key(name)  # 短名 → observation.images.<name>
                cam_dir = tmp_root / key
                cam_dir.mkdir(parents=True, exist_ok=True)  # 每路相机一个子目录
                path = cam_dir / f"{step_idx:06d}.png"  # ffmpeg %06d 序列约定
                _save_rgb_png(path, rgb)
                arr = np.asarray(rgb)  # stats 用原数组,不依赖刚写的 PNG
                self._image_stats.setdefault(key, _ChannelStats()).update(arr)  # 按像素累通道

        self._steps.append(entry)  # 先缓存,finalize 再写 parquet

    def finalize(
        self,
        *,
        status: str,
        extra: dict[str, Any] | None = None,
        export_video: bool = True,
        video_fps: int = 15,
    ) -> None:
        """把当前 episode 写成 parquet/mp4,并更新 ``meta/``.

        未 open 或已 finalize 则 no-op.步数为 0 时跳过写盘.

        Args:
            status: 本集结束状态(``success`` / ``failed`` / ``timeout`` 等).
            extra: 写入 jsonl 的附加统计(步数/抬升等);None 则只用 status.
            export_video: 调用方开关;本实现忽略,有 RGB 就必须出 mp4.
            video_fps: 覆盖构造时 fps;0 则回退 ``self.fps``.

        Returns:
            None.
        """
        del export_video  # 有 RGB 就必须出 mp4,否则不是合法 v2.1
        if self.episode_dir is None or self._finalized:
            return  # 重复 finalize / 未 open,避免把空壳写进数据集
        self._finalized = True  # 先置位,后续异常也不允许再 record_step
        self._meta["status"] = status
        if extra:
            self._meta.update(extra)  # Job 的 last_stats 并入,validate 可读
        if not self._steps:
            self.log.warning("lerobot_empty_episode_skipped", episode_index=self._episode_index)
            return  # 空集不写 parquet,避免污染 total_frames

        fps = int(video_fps) if video_fps else self.fps  # 0 视为未指定
        self.fps = fps
        ep_index = self._episode_index  # 本集号,下面 chunk/文件名都用它
        chunk = ep_index // _CHUNKS_SIZE  # 每 1000 集一个 chunk 目录
        n_frames = len(self._steps)
        global_start = int(self._info.get("total_frames", 0))  # 跨集聚引,replay 用
        task_index = _ensure_task(self.root, self.task)  # 同描述复用已有 task_index

        state = _stack_feature(self._steps, "observation.state")  # (T, Ds)
        action = _stack_feature(self._steps, "action")  # (T, Da)
        video_keys = self._resolved_video_keys()
        if "features" not in self._info or not self._info["features"]:
            # 首集才写 schema,后续集不得改维,否则 HF datasets 对不上
            self._info["features"] = _build_features(
                state_dim=int(state.shape[1]),
                action_dim=int(action.shape[1]),
                video_keys=video_keys,
                image_shapes=self._image_shapes(),
                fps=fps,
                robot_type=self.robot_type,
            )
        _write_parquet(  # 一行一帧;失败则本集不算进 total_*
            self.root / _DATA_PATH.format(episode_chunk=chunk, episode_index=ep_index),
            state=state,
            action=action,
            timestamps=np.asarray([s["timestamp"] for s in self._steps], dtype=np.float32),
            episode_index=ep_index,
            global_start=global_start,
            task_index=task_index,
        )
        videos_written = 0  # 无 RGB 时保持 0,info.total_videos 不加
        if self.save_rgb:
            videos_written = self._encode_videos(chunk=chunk, ep_index=ep_index, fps=fps)

        stats = _episode_stats(  # jsonl 一条,给 lerobot 可视化均值方差
            state=state,
            action=action,
            timestamps=np.asarray([s["timestamp"] for s in self._steps], dtype=np.float32),
            image_stats=self._image_stats,
            episode_index=ep_index,
            n_frames=n_frames,
            global_start=global_start,
            task_index=task_index,
        )
        _append_jsonl(  # 追加而非覆盖,接续已有数据集
            self.root / "meta" / "episodes.jsonl",
            {
                "episode_index": ep_index,
                "tasks": [self.task],
                "length": n_frames,
                "status": status,
                "metadata": dict(self._meta),
            },
        )
        _append_jsonl(self.root / "meta" / "episodes_stats.jsonl", stats)  # 与 episodes 同步追加

        self._info["fps"] = fps  # 本集可能覆盖构造时 fps
        self._info["total_episodes"] = ep_index + 1  # 下一条 open 用
        self._info["total_frames"] = global_start + n_frames
        self._info["total_tasks"] = _count_jsonl(self.root / "meta" / "tasks.jsonl")
        self._info["total_videos"] = int(self._info.get("total_videos", 0)) + videos_written
        self._info["total_chunks"] = chunk + 1  # 1-based 目录数
        self._info["splits"] = {"train": f"0:{ep_index + 1}"}  # 半开区间,官方约定
        _write_json(self.root / "meta" / "info.json", self._info)
        self._cleanup_tmp_rgb(ep_index)  # mp4 已出,PNG 缓存可删
        self.log.info(
            "lerobot_episode_finalize",
            path=str(self.root),
            episode_index=ep_index,
            status=status,
            steps=n_frames,
            videos=videos_written,
        )

    def _resolved_video_keys(self) -> list[str]:
        """配置的相机名优先,否则用本集实际写过的临时目录.

        Returns:
            ``observation.images.*`` 键列表;配置优先,否则扫临时目录或 stats.
        """
        names = self.camera_names or []  # 构造时注入的期望相机,空则推断
        if names:
            return [_video_key(n) for n in names]  # 即使某路没出帧也占 schema
        tmp = _tmp_rgb_dir(self.root, self._episode_index)
        if not tmp.is_dir():
            return list(self._image_stats)  # 无 PNG 目录时退回已累积的相机
        return sorted(p.name for p in tmp.iterdir() if p.is_dir())

    def _image_shapes(self) -> dict[str, tuple[int, int, int]]:
        """各路相机 (H, W, C);读第一张临时 PNG,失败则 480x640.

        Returns:
            各路 ``(H, W, C)``;读不到第一张 PNG 时为 ``(480, 640, 3)``.
        """
        shapes: dict[str, tuple[int, int, int]] = {}
        tmp = _tmp_rgb_dir(self.root, self._episode_index)  # 读第一张 PNG 拿分辨率
        for key in self._resolved_video_keys():  # 与 features 视频键对齐
            frames = sorted((tmp / key).glob("*.png")) if tmp.is_dir() else []
            hw = (480, 640, 3)  # 缺帧时仍要写 features,用常见 VGA
            if frames:
                arr = _read_png_hwc(frames[0])
                if arr is not None:
                    hw = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))  # HWC
            shapes[key] = hw  # 缺图也占键,schema 完整
        return shapes

    def _encode_videos(self, *, chunk: int, ep_index: int, fps: int) -> int:
        """把临时 PNG 序列编成每相机一条 mp4;返回成功条数.

        Args:
            chunk: episode 所在 chunk 序号(每 ``_CHUNKS_SIZE`` 集一个目录).
            ep_index: 0-based 集号,写入文件名 ``episode_XXXXXX.mp4``.
            fps: 视频帧率,应与 ``info.json`` 一致.

        Returns:
            成功写出的非空 mp4 条数.
        """
        tmp = _tmp_rgb_dir(self.root, ep_index)  # PNG 序列根;编码失败也要告警每路
        written = 0  # 成功条数累加进 info.total_videos
        for key in self._resolved_video_keys():
            cam_dir = tmp / key
            rel = _VIDEO_PATH.format(episode_chunk=chunk, episode_index=ep_index, video_key=key)
            mp4 = self.root / rel  # 相对数据集根,info.video_path 模板一致
            mp4.parent.mkdir(parents=True, exist_ok=True)  # 相机子目录首次采集才有
            if cam_dir.is_dir() and _ffmpeg_png_to_mp4(cam_dir, mp4, fps=fps):
                written += 1  # 空文件不算成功
                self.log.info("lerobot_video_exported", camera=key, path=str(mp4))
            else:
                self.log.warning("lerobot_video_export_failed", camera=key, path=str(mp4))
        return written

    def _cleanup_tmp_rgb(self, ep_index: int) -> None:
        """删掉本集临时 PNG,失败只告警.

        Args:
            ep_index: 与 ``_tmp_rgb_dir`` 一致的 0-based 集号.

        Returns:
            None.
        """
        tmp = _tmp_rgb_dir(self.root, ep_index)
        if tmp.is_dir():  # 编码失败时目录可能仍在,下次 open 会换集号
            try:
                shutil.rmtree(tmp)  # mp4 已出,PNG 不必进 git / 磁盘
            except Exception as exc:  # noqa: BLE001
                self.log.warning("lerobot_tmp_cleanup_failed", error=str(exc))


class LeRobotV3Recorder(LeRobotRecorder):
    """LeRobot v3 落盘器占位;``create_recorder`` 尚未接线."""


def is_lerobot_v2_dataset(path: str | Path) -> bool:
    """目录下有 ``meta/info.json`` 即视为 LeRobot v2 数据集根.

    Args:
        path: 候选数据集根;可以是 str 或 ``Path``.

    Returns:
        ``True`` 当且仅当 ``meta/info.json`` 存在.
    """
    return (Path(path) / "meta" / "info.json").is_file()  # validate/replay 据此认 v2 根


def list_lerobot_episodes(root: str | Path) -> list[int]:
    """按 parquet 文件名列出数据集中的 episode_index,不依赖可能过期的 info.json.

    Args:
        root: 数据集根,其下应有 ``data/chunk-*/episode_*.parquet``.

    Returns:
        按文件名解析出的 ``episode_index`` 升序列表;无 data 目录则为空.
    """
    root_path = Path(root)
    indices: list[int] = []
    data_dir = root_path / "data"
    if not data_dir.is_dir():
        return indices  # 空数据集 / 尚未 finalize
    for path in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
        try:
            indices.append(int(path.stem.split("_")[1]))  # episode_000012 → 12
        except (IndexError, ValueError):
            continue  # 非约定文件名跳过,不炸整个列表
    return indices


def load_lerobot_episode(root: str | Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    """读取一集 ``observation.state`` 与 ``action``.

    Args:
        root: 数据集根.
        episode_index: 0-based 集号;chunk 由 ``_CHUNKS_SIZE`` 推算.

    Returns:
        ``(state, action)`` 形状 ``(T, Ds)`` / ``(T, Da)`` 的 float32.

    Raises:
        FileNotFoundError: parquet 不存在.
        ValueError: 表中缺少必要列.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415  # 延迟导入,CI stub 不必装 pyarrow

    root_path = Path(root)
    chunk = int(episode_index) // _CHUNKS_SIZE
    path = root_path / _DATA_PATH.format(episode_chunk=chunk, episode_index=int(episode_index))
    if not path.is_file():
        raise FileNotFoundError(f"episode parquet 不存在: {path}")  # 调用方(replay)需区分缺集
    table = pq.read_table(path, columns=["observation.state", "action"])  # 只读需要的列
    names = set(table.column_names)
    if "observation.state" not in names or "action" not in names:
        raise ValueError(f"parquet 缺少 state/action 列: {path}")  # 旧格式/写盘中断
    state = _parquet_list_column(table, "observation.state")
    action = _parquet_list_column(table, "action")
    return state, action


def _parquet_list_column(table: Any, name: str) -> np.ndarray:
    """把 parquet list 列堆成 ``(T, D)`` float32.

    Args:
        table: ``pyarrow.Table``.
        name: 列名,如 ``observation.state``.

    Returns:
        ``(T, D)`` float32;空表为 ``(0, 0)``,变长 list 按最宽 pad.
    """
    rows = table.column(name).to_pylist()  # list[list[float]]
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)  # 空表仍给二维,调用方按 shape[1] 取维
    max_n = max((len(r or []) for r in rows), default=0)  # 变长 list pad 到最宽
    out = np.zeros((len(rows), max_n), dtype=np.float32)
    for i, row in enumerate(rows):
        if row:
            arr = np.asarray(row, dtype=np.float32).reshape(-1)
            out[i, : arr.size] = arr[:max_n]  # 超宽截断,与 pad 对称
    return out


def _video_key(name: str) -> str:
    """相机短名 → ``observation.images.<name>``.

    Args:
        name: 传感器名或已带前缀的 feature 键.

    Returns:
        已带 ``observation.images.`` 前缀的 feature 键.
    """
    if name.startswith("observation.images."):
        return name  # 已经是 feature 键,避免双重前缀
    return f"observation.images.{name}"


def _tmp_rgb_dir(root: Path, episode_index: int) -> Path:
    """单集临时 PNG 根目录.

    Args:
        root: 数据集根.
        episode_index: 0-based 集号,写入 ``.cache/episode_XXXXXX``.

    Returns:
        ``<root>/.cache/episode_XXXXXX``.
    """
    return root / ".cache" / f"episode_{episode_index:06d}"  # 点目录,默认 gitignore


def _read_png_hwc(path: Path) -> np.ndarray | None:
    """读一张 PNG 为 ``(H, W, C)``;失败返回 None.

    Args:
        path: 临时 PNG;损坏时返回 None,调用方走默认分辨率.

    Returns:
        RGB ``(H, W, C)``;损坏或非三通道则为 ``None``.
    """
    try:
        from PIL import Image  # noqa: PLC0415  # 延迟导入,无 PIL 时走失败分支

        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"))  # 丢掉 alpha,features 只要 3 通道
        if arr.ndim == 3:
            return arr
    except Exception:  # noqa: BLE001
        return None  # 损坏文件不炸 finalize
    return None  # 灰度等非 HWC 也视为失败


def _as_float_list(value: Any) -> list[float]:
    """ndarray/None → 一维 float 列表.

    Args:
        value: 关节向量 / None / 已是 list;None 给空列表,parquet 再 pad.

    Returns:
        一维 float 列表;``None`` 为空列表.
    """
    if value is None:  # 未采到关节时不写 None,否则 parquet 列类型崩
        return []  # 空列表,finalize 再 pad 到最宽维
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()  # 压成 1D list,parquet 才能对齐


def _stack_feature(steps: list[dict[str, Any]], key: str) -> np.ndarray:
    """按最大长度 pad 成 ``(T, D)`` float32.

    Args:
        steps: ``record_step`` 缓存的逐步 dict.
        key: 要堆叠的字段,如 ``observation.state`` / ``action``.

    Returns:
        ``(T, D)`` float32;短行左侧对齐,右侧 pad 0.
    """
    rows = [s.get(key, []) for s in steps]  # 缺键当空,避免 KeyError 中断落盘
    max_n = max((len(r) for r in rows), default=0)
    out = np.zeros((len(rows), max_n), dtype=np.float32)
    for i, r in enumerate(rows):
        if r:
            out[i, : len(r)] = np.asarray(r, dtype=np.float32)  # 短行左侧对齐 pad 0
    return out


def _motor_names(n: int) -> list[str]:
    """Franka 名优先,否则 ``motor_i``.

    Args:
        n: 需要的电机数;超过 Panda 9 轴则补 ``motor_i``.

    Returns:
        长度 ``n`` 的电机名列表.
    """
    if n <= len(_FRANKA_MOTORS):
        return _FRANKA_MOTORS[:n]  # 只要臂 7 轴时截掉手指
    extra = [f"motor_{i}" for i in range(len(_FRANKA_MOTORS), n)]
    return _FRANKA_MOTORS + extra


def _build_features(
    *,
    state_dim: int,
    action_dim: int,
    video_keys: list[str],
    image_shapes: dict[str, tuple[int, int, int]],
    fps: int,
    robot_type: str,
) -> dict[str, Any]:
    """构造 ``info.json`` 的 ``features`` 段.

    Args:
        state_dim: ``observation.state`` 维数.
        action_dim: ``action`` 维数.
        video_keys: ``observation.images.*`` 键.
        image_shapes: 各路 ``(H, W, C)``;缺省 480x640.
        fps: 写入每路 video.info.
        robot_type: 机型;本实现未写入 features,仅占位对齐调用方.

    Returns:
        ``info.json`` 的 ``features`` dict(state/action/索引列 + 各路 video).
    """
    del robot_type  # features 段不写机型,机型在 info 顶层
    features: dict[str, Any] = {  # 首集 schema;后续集不得改维
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": {"motors": _motor_names(state_dim)},
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": {"motors": _motor_names(action_dim)},
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key in video_keys:  # 每路相机一条 video feature
        h, w, c = image_shapes.get(key, (480, 640, 3))  # 缺 PNG 时用 VGA,避免 schema 空洞
        features[key] = {  # dtype=video,HF 才去 videos/ 找 mp4
            "dtype": "video",
            "shape": [h, w, c],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": float(fps),
                "video.height": h,
                "video.width": w,
                "video.channels": c,
                "video.codec": "avc1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return features  # 写入 info.json,load 时按此校验列


def _load_or_init_info(root: Path, *, fps: int, robot_type: str) -> dict[str, Any]:
    """已有数据集则读 ``info.json``,否则给空壳.

    Args:
        root: 数据集根.
        fps: 新建空壳时写入;已有文件则尊重盘上值.
        robot_type: 新建空壳时写入;接续数据集不覆盖.

    Returns:
        已有 ``info.json`` 的内容;否则带路径模板的空壳(``features`` 为空).
    """
    path = root / "meta" / "info.json"  # 接续采集时已有此文件
    if path.is_file():  # 已有数据集则沿用 total_* / features
        info = json.loads(path.read_text(encoding="utf-8"))  # 坏 JSON 让调用方炸,避免静默空壳
        return info  # 接续采集,保留 total_* / features
    return {  # 空壳;首集 finalize 才填 features
        "codebase_version": _CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": _CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": "0:0"},
        "data_path": _DATA_PATH,
        "video_path": _VIDEO_PATH,
        "features": {},  # 首集 finalize 才按实际维数填写
    }


def _write_parquet(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    timestamps: np.ndarray,
    episode_index: int,
    global_start: int,
    task_index: int,
) -> None:
    """一集一行一帧写入 parquet.

    Args:
        path: 目标 parquet,父目录不存在时创建.
        state: ``(T, Ds)`` 观测.
        action: ``(T, Da)`` 动作.
        timestamps: 长度 T 的秒时间戳.
        episode_index: 本集 0-based 集号,每行重复.
        global_start: 跨集 ``index`` 列的起始值.
        task_index: ``tasks.jsonl`` 中的任务序号.

    Returns:
        None.
    """
    import pyarrow as pa  # noqa: PLC0415  # 延迟导入,CI stub 不必装 pyarrow
    import pyarrow.parquet as pq  # noqa: PLC0415  # 只在 finalize 写盘时才需要

    n = int(state.shape[0])  # 帧数 T,后面各列都要这么长
    path.parent.mkdir(parents=True, exist_ok=True)  # chunk 目录首次才有
    state_type = pa.list_(pa.float32(), int(state.shape[1]))  # 定长 list,HF datasets 更稳
    action_type = pa.list_(pa.float32(), int(action.shape[1]))  # 与 state 同样定长 list
    table = pa.table(  # 列名必须与 features 对齐,缺列 replay 会炸
        {
            "observation.state": pa.array(state.tolist(), type=state_type),
            "action": pa.array(action.tolist(), type=action_type),
            "timestamp": pa.array(timestamps.astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
            "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64), type=pa.int64()),
            "index": pa.array(
                np.arange(global_start, global_start + n, dtype=np.int64), type=pa.int64()
            ),  # 全数据集连续帧号
            "task_index": pa.array(np.full(n, task_index, dtype=np.int64), type=pa.int64()),
        }
    )
    pq.write_table(table, path)  # 覆盖写本集;父目录已 mkdir


def _episode_stats(
    *,
    state: np.ndarray,
    action: np.ndarray,
    timestamps: np.ndarray,
    image_stats: dict[str, _ChannelStats],
    episode_index: int,
    n_frames: int,
    global_start: int,
    task_index: int,
) -> dict[str, Any]:
    """v2.1 ``episodes_stats.jsonl`` 的一条记录.

    Args:
        state: ``(T, Ds)`` 观测,算 min/max/mean/std.
        action: ``(T, Da)`` 动作.
        timestamps: 长度 T 的秒时间戳.
        image_stats: 各路相机累积器;无 RGB 时为空.
        episode_index: 本集 0-based 集号.
        n_frames: 本集帧数,用于构造 index 类统计.
        global_start: 跨集 ``index`` 列起点.
        task_index: 任务序号,统计形状与其它标量列一致.

    Returns:
        ``{"episode_index": ..., "stats": ...}``,供 ``episodes_stats.jsonl``.
    """
    frame_index = np.arange(n_frames, dtype=np.float64)  # 本集 0..T-1,与 parquet 列对齐
    ep = np.full(n_frames, episode_index, dtype=np.float64)  # 每行重复集号,才能算 min/max
    idx = np.arange(global_start, global_start + n_frames, dtype=np.float64)  # 跨集连续 index
    task = np.full(n_frames, task_index, dtype=np.float64)  # 标量列也走 (T,) 再 reshape
    stats: dict[str, Any] = {  # 与 parquet 列同名,lerobot 可视化读此
        "observation.state": _vector_stats(state),
        "action": _vector_stats(action),
        "timestamp": _vector_stats(timestamps.reshape(-1, 1)),  # 标量列也走 (T, 1)
        "frame_index": _vector_stats(frame_index.reshape(-1, 1)),
        "episode_index": _vector_stats(ep.reshape(-1, 1)),
        "index": _vector_stats(idx.reshape(-1, 1)),
        "task_index": _vector_stats(task.reshape(-1, 1)),
    }
    for key, acc in image_stats.items():  # 无 RGB 时跳过,不写空视频 stats
        stats[key] = acc.as_dict()  # (C, 1, 1) 图像统计
    return {"episode_index": episode_index, "stats": stats}  # jsonl 一条,与 episodes 同步


def _vector_stats(array: np.ndarray) -> dict[str, Any]:
    """``(T, D)`` → min/max/mean/std/count;空数组给零.

    Args:
        array: 至少一维;空数组按列宽填 0,避免 jsonl 缺键.

    Returns:
        ``min/max/mean/std/count`` dict;空数组按列宽填 0.
    """
    if array.size == 0:
        dim = int(array.shape[1]) if array.ndim == 2 else 1
        zeros = [0.0] * dim
        return {"min": zeros, "max": zeros, "mean": zeros, "std": zeros, "count": [0]}
    arr = np.asarray(array, dtype=np.float64)  # 用 float64 减累计误差
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)  # 统一成 (T, D)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],  # 官方用 list,不是标量
    }


def _ensure_task(root: Path, task: str) -> int:
    """保证 ``tasks.jsonl`` 有该描述,返回 ``task_index``.

    Args:
        root: 数据集根.
        task: 任务语言描述;相同字符串复用已有 index.

    Returns:
        该任务描述对应的 ``task_index``(已有则复用,否则追加).
    """
    path = root / "meta" / "tasks.jsonl"
    tasks = _read_jsonl(path)
    for row in tasks:
        if str(row.get("task", "")) == task:
            return int(row["task_index"])  # 同描述不重复追加
    task_index = len(tasks)  # 新描述接到末尾
    _append_jsonl(path, {"task_index": task_index, "task": task})
    return task_index


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读 jsonl;文件不存在则空列表.

    Args:
        path: jsonl 路径;缺失视为空数据集,不抛错.

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


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    """追加一行 JSON.

    Args:
        path: jsonl 路径;父目录不存在时创建.
        obj: 要追加的对象;``ensure_ascii=False`` 保留中文任务描述.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")  # 一行一条,HF 约定


def _count_jsonl(path: Path) -> int:
    """jsonl 行数.

    Args:
        path: jsonl 路径;缺失为 0.

    Returns:
        非空行数;文件缺失为 0.
    """
    return len(_read_jsonl(path))  # 缺失/空文件为 0,写入 info.total_tasks


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    """覆盖写 JSON.

    Args:
        path: 目标文件,通常 ``meta/info.json``.
        obj: 可 JSON 序列化的 dict.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)  # meta/ 首次采集才有
    # indent=2 便于手工 diff;ensure_ascii=False 保留中文 task
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class _ChannelStats:
    """逐帧累积 RGB 通道统计,输出 v2.1 图像 stats 形状 ``(C, 1, 1)``."""

    def __init__(self) -> None:
        """像素计数与通道 min/max/sum 从零开始;第一帧 update 会覆盖初值.

        Returns:
            None.
        """
        self.n_pix = 0  # 累计像素数,mean = sum / n_pix
        self.n_frames = 0  # count 字段用帧数,不是像素
        self.ch_sum = np.zeros(3, dtype=np.float64)
        self.ch_sumsq = np.zeros(3, dtype=np.float64)  # 用于无偏之外的总体 std
        self.ch_min = np.ones(3, dtype=np.float64)  # 1 作为哨兵,第一帧 min 会更小
        self.ch_max = np.zeros(3, dtype=np.float64)

    def update(self, rgb: np.ndarray) -> None:
        """吃一帧 uint8/float RGB.

        Args:
            rgb: ``(H, W)`` 或 ``(H, W, C)``;float 按 0-1 或 0-255 自动归一.

        Returns:
            None.
        """
        arr = np.asarray(rgb)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)  # 灰度扩成三通道,stats 形状固定 C=3
        arr = arr[..., :3]  # 丢掉 alpha
        if arr.dtype != np.uint8:
            mx = float(np.nanmax(arr)) if arr.size else 0.0  # 空帧当 0,走 0-1 分支
            if mx <= 1.0 + 1e-3:
                x = np.clip(arr, 0.0, 1.0).astype(np.float64)  # 已是 0-1
            else:
                x = np.clip(arr, 0.0, 255.0).astype(np.float64) / 255.0  # 0-255 float
        else:
            x = arr.astype(np.float64) / 255.0  # uint8 → 0-1,与 LeRobot 官方一致
        self.ch_sum += x.sum(axis=(0, 1))  # 按通道累像素,后面除 n_pix
        self.ch_sumsq += np.square(x).sum(axis=(0, 1))  # E[x^2],用来算 std
        self.ch_min = np.minimum(self.ch_min, x.min(axis=(0, 1)))
        self.ch_max = np.maximum(self.ch_max, x.max(axis=(0, 1)))
        self.n_pix += int(x.shape[0] * x.shape[1])
        self.n_frames += 1  # count 用帧数,与官方 episodes_stats 一致

    def as_dict(self) -> dict[str, Any]:
        """序列化成 LeRobot 图像 stats.

        Returns:
            ``min/max/mean/std`` 为 ``(C, 1, 1)`` 嵌套 list,``count`` 为本集帧数.
        """
        if self.n_pix == 0:
            z = [[[0.0]], [[0.0]], [[0.0]]]  # (C, 1, 1) 空壳,避免缺键
            return {"min": z, "max": z, "mean": z, "std": z, "count": [0]}
        mean = self.ch_sum / self.n_pix  # 像素级均值,不是帧级
        var = np.maximum(self.ch_sumsq / self.n_pix - np.square(mean), 0.0)  # 截负误差
        std = np.sqrt(var)

        def cube(v: np.ndarray) -> list[list[list[float]]]:
            """把 ``(3,)`` 通道向量收成官方 ``(C, 1, 1)`` 嵌套 list.

            Args:
                v: 长度 3 的通道统计(min/max/mean/std).

            Returns:
                ``[[[c0]], [[c1]], [[c2]]]``.
            """
            return [[[float(v[0])]], [[float(v[1])]], [[float(v[2])]]]  # (C,1,1),HF 可视化要这嵌套

        return {
            "min": cube(self.ch_min),
            "max": cube(self.ch_max),
            "mean": cube(mean),
            "std": cube(std),
            "count": [self.n_frames],  # 官方 count 是帧数 list
        }


def _ffmpeg_png_to_mp4(frame_dir: Path, out_mp4: Path, *, fps: int = 15) -> bool:
    """``%06d.png`` 序列转 mp4;无 ffmpeg 或失败则退 imageio.

    Args:
        frame_dir: 含 ``000000.png`` 起的连续序号 PNG.
        out_mp4: 输出路径;成功时必须非空文件.
        fps: 视频帧率.

    Returns:
        ``True`` 当且仅当写出非空 mp4(ffmpeg 失败则再试 imageio).
    """
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        return False  # 该路没出过 RGB
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        pattern = str(frame_dir / "%06d.png")  # 必须连续,缺号会停
        cmd = [
            ffmpeg,
            "-y",  # 覆盖已有
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(int(fps)),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",  # 播放器兼容
            str(out_mp4),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if out_mp4.is_file() and out_mp4.stat().st_size > 0:
                return True  # 空文件视为失败,走 imageio
        except Exception:  # noqa: BLE001  # 缺 libx264 等
            pass
    return _imageio_png_to_mp4(frames, out_mp4, fps=fps)


def _imageio_png_to_mp4(frames: list[Path], out_mp4: Path, *, fps: int = 15) -> bool:
    """ffmpeg 不可用时的 PNG→mp4 回退.

    Args:
        frames: 已排序的 PNG 路径.
        out_mp4: 输出路径.
        fps: 视频帧率;二次尝试时丢掉 codec 参数以提高兼容.

    Returns:
        ``True`` 当且仅当写出非空 mp4;无 imageio 或编码失败则为 ``False``.
    """
    try:
        import imageio.v2 as imageio  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        try:
            import imageio  # noqa: PLC0415  # 旧包布局
        except Exception:  # noqa: BLE001
            return False  # 无编码器,调用方记 export_failed
    try:
        writer = imageio.get_writer(str(out_mp4), fps=int(fps), codec="libx264", format="FFMPEG")
        try:
            for path in frames:
                writer.append_data(imageio.imread(path))
        finally:
            writer.close()  # 必须关,否则 mp4 可能不 flush
        return out_mp4.is_file() and out_mp4.stat().st_size > 0
    except Exception:  # noqa: BLE001  # codec 参数不被接受时再试默认
        try:
            writer = imageio.get_writer(str(out_mp4), fps=int(fps))
            try:
                for path in frames:
                    writer.append_data(imageio.imread(path))
            finally:
                writer.close()
            return out_mp4.is_file() and out_mp4.stat().st_size > 0
        except Exception:  # noqa: BLE001
            return False


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    """把 RGB 写成 PNG:归一化 → 优先 PIL → imageio → cv2 → 纯 zlib.

    Args:
        path: 目标 PNG,父目录须已存在.
        rgb: ``(H, W)`` 或 ``(H, W, C)``;float 按 0-1 / 0-255 归一到 uint8.

    Returns:
        None.
    """
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        if mx <= 1.0 + 1e-3:  # 0-1 float
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:  # 灰度扩成三通道,后续编码器统一
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.ascontiguousarray(arr[..., :3])  # 丢掉 alpha
    try:
        from PIL import Image  # noqa: PLC0415

        Image.fromarray(arr).save(path)
        return  # 最常见路径,成功即停
    except Exception:  # noqa: BLE001
        pass
    try:
        import imageio.v2 as imageio  # noqa: PLC0415

        imageio.imwrite(path, arr)
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2  # noqa: PLC0415

        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))  # OpenCV 要 BGR
        return
    except Exception:  # noqa: BLE001
        pass
    _write_png_rgb(path, arr)  # 无第三方也能落盘


def _write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    """无第三方依赖的最后手段:纯 zlib PNG 编码.

    Args:
        path: 目标 PNG.
        rgb: 连续 ``(H, W, 3)`` uint8,调用方已切掉 alpha.

    Returns:
        None.
    """
    import struct  # PNG chunk 大端长度
    import zlib  # IDAT 压缩

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))  # 每行前加 filter=None

    def chunk(tag: bytes, data: bytes) -> bytes:
        """PNG chunk:长度 + 类型 + 数据 + CRC.

        Args:
            tag: 4 字节类型,如 ``IHDR`` / ``IDAT`` / ``IEND``.
            data: chunk 载荷.

        Returns:
            带长度与 CRC 的 PNG chunk 字节.
        """
        return (  # PNG: 长度+类型+数据+CRC,否则解码器拒读
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8bit RGB
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")  # 空 IEND 结束文件
    path.write_bytes(png)
