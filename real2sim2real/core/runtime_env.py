"""运行时路径

"""

from __future__ import annotations

import os
from pathlib import Path


def assets_root() -> Path:
    """从 ``RSR_ASSETS_ROOT`` 解析资产根。

    Returns:
        资产根 ``Path``.

    Raises:
        RuntimeError: 未设置 ``RSR_ASSETS_ROOT``.
    """
    override = os.environ.get("RSR_ASSETS_ROOT", "").strip()
    if not override:
        raise RuntimeError("未设置环境变量 RSR_ASSETS_ROOT")
    return Path(override).expanduser().resolve()


def isaacsim_home() -> Path:
    """从 ``RSR_ISAACSIM_HOME`` 解析 Isaac Sim 安装目录。

    Returns:
        Isaac Sim 安装目录 ``Path``.

    Raises:
        RuntimeError: 未设置 ``RSR_ISAACSIM_HOME``.
    """
    override = os.environ.get("RSR_ISAACSIM_HOME", "").strip()
    if not override:
        raise RuntimeError("未设置环境变量 RSR_ISAACSIM_HOME")
    return Path(override).expanduser().resolve()
