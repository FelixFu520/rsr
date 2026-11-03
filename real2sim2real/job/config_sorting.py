"""``[job]`` 分拣采集配置与 ``[job.recorder]``(新增文件,不改 ``job/config.py``)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from real2sim2real.core.config import BaseConfig

# 落盘格式;新增版本在此扩面,并在 ``create_recorder`` 挂实现
RecorderFormat = Literal["lerobot_v2", "lerobot_v3"]


class JobRecorderConfig(BaseModel):
    """``[job.recorder]``:采集落盘格式与路径.

    ``format`` 决定实现类(``lerobot_v2``; ``lerobot_v3`` 预留).
    """

    model_config = ConfigDict(extra="forbid")

    format: RecorderFormat = Field(
        default="lerobot_v2",
        description="lerobot_v2: v2.1 parquet+mp4; lerobot_v3: 预留",
    )
    output_dir: str = Field(default="outputs/tabletop", description="LeRobot 数据集根目录")
    task: str = Field(default="pick and place", description="LeRobot 任务语言描述")
    robot_type: str = Field(default="franka", description="LeRobot info.json 的 robot_type")
    save_rgb: bool = Field(default=True, description="是否落盘 RGB")
    save_depth: bool = Field(default=False, description="是否落盘深度;lerobot_* 暂不写")
    export_video: bool = Field(default=True, description="是否导出各相机 mp4")
    fps: int = Field(default=15, description="数据集 fps,同时用于导出 mp4")
    record_camera_prims: list[str] = Field(
        default_factory=list,
        description=(
            "落盘相机,按 prim/名筛选;空=全部.写组内 Camera prim 只录那一路;写壳体根则组内全录"
        ),
    )


class SortingJobConfig(BaseConfig):
    """``[job]``:分拣台双臂搬运采集配置."""

    inspect_gui: bool = Field(default=False, description="是否在各流水线阶段后进入 GUI 可编辑等待")
    num_episodes: int = Field(default=1, ge=1, description="本轮连续采集多少条 episode")
    object_jitter_xy: float = Field(
        default=0.06,
        ge=0.0,
        description="每集包裹相对入带摆放位的 XY 随机偏移半边长(m);0 则不抖",
    )
    object_jitter_seed: int = Field(default=0, description="偏移 RNG 种子;0 表示不固定")
    camera_warmup_steps: int = Field(
        default=30, description="Play 后热身步数;关节由规划器接管,这里只等相机出帧"
    )
    max_steps: int = Field(
        default=6000, description="落盘帧数硬上限;分拣一趟约需数千物理步,按抽帧后计"
    )
    record_stride: int = Field(
        default=0,
        ge=0,
        description="落盘抽帧间隔(物理步);0 表示按 physics_hz / recorder.fps 自动计算",
    )
    physics_hz: float = Field(
        default=60.0, gt=0.0, description="物理频率,自动抽帧时与 [sim_world].physics_dt 对齐"
    )
    viewport_camera_prims: list[str] = Field(
        default_factory=list,
        description="GUI 视口相机 prim/名;空=全部已绑定传感器",
    )
    recorder: JobRecorderConfig = Field(
        default_factory=lambda: JobRecorderConfig(
            output_dir="outputs/sorting_pro_m6", robot_type="pro_m6"
        ),
        description="落盘配置,见 [job.recorder]",
    )
