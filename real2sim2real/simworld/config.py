"""``[sim_world]`` 模块配置."""

from __future__ import annotations

from pydantic import Field

from real2sim2real.core.config import BaseConfig


class StubSimWorldConfig(BaseConfig):
    """``[sim_world]``:stub 仿真世界配置."""
