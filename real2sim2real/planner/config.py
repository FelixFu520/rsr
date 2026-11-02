"""``[planner]`` 模块配置.

stub 仅公共 ``backend``;curobo 含超时、抓取高度偏移、段间静止步数与放置点;
replay 按已采集 LeRobot episode 回放关节动作/状态.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from real2sim2real.core.config import BaseConfig


class StubPlannerConfig(BaseConfig):
    """``[planner]``:stub 规划器配置."""


