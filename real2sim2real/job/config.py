"""任务编排配置:公共基类 + 各 Job 专用 Config.

``StubJobConfig`` 只有 ``backend`` 等公共字段;
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from real2sim2real.core.config import BaseConfig

# 落盘格式;新增版本在此扩面,并在 ``create_recorder`` 挂实现
RecorderFormat = Literal["lerobot_v2", "lerobot_v3"]


class StubJobConfig(BaseConfig):
    """``[job]``:stub 任务编排配置."""


