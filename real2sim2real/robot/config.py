"""``[robot]`` 模块配置.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from real2sim2real.core.config import BaseConfig


class StubRobotConfig(BaseConfig):
    """``[robot]``:stub 机器人配置."""

