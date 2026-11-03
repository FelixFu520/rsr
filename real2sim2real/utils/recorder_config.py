"""采集落盘格式与路径配置。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
