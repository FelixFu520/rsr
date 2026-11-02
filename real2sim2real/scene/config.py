"""``[scene]`` 模块配置."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from real2sim2real.core.config import BaseConfig


class StubSceneConfig(BaseConfig):
    """``[scene]``:stub 场景配置."""

