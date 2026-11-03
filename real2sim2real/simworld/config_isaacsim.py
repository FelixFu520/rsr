"""``[sim_world]`` Isaac Sim 配置(新增文件,不改 ``simworld/config.py``)."""

from __future__ import annotations

from pydantic import Field

from real2sim2real.core.config import BaseConfig


class IsaacsimSimWorldConfig(BaseConfig):
    """``[sim_world]``:仿真世界配置."""

    headless: bool = Field(
        default=True, 
        description="是否无头模式."
    )
    renderer: str = Field(
        default="RayTracedLighting", 
        description="RayTracedLighting速度快, PathTracing渲染效果逼真"
    )
    rt_subframes: int = Field(
        default=8, 
        description="光线追踪渲染器会在后台迭代多少次子步."
    )
    physics_dt: float = Field(
        default=1.0 / 60.0, 
        description="物理步长."
    )
    rendering_dt: float = Field(
        default=1.0 / 30.0, 
        description="渲染步长."
    )
    device: str = Field(
        default="",
        description="物理设备,如 cuda / cpu;空则沿用 Isaac 默认.人形策略须 cuda",
    )
    physics_backend: str = Field(
        default="",
        description="SimulationManager backend,如 torch;空则沿用默认",
    )
    stage_usd_path: str = Field(
        default="",
        description=(
            "整份环境 USD:直接 open 成 stage,prim 路径与原文件一致(分拣台等成品场景用);"
            "非空时忽略 background_usd_path"
        ),
    )
    background_usd_path: str = Field(
        default="", 
        description="背景 USD 路径."
    )
    background_prim_path: str = Field(
        default="/World/background",
        description=(
            "背景引用挂到的 prim.默认 /World/background,"
            "与 /World/scene、/World/robots 互为兄弟,避免背景尺度传到工位和机器人"
        ),
    )
    background_gravity: bool = Field(
        default=False,
        description="背景是否受重力影响. False 时将背景刚体设为 kinematic,保留碰撞但不掉落.",
    )
    background_skip_metrics_assembler: bool = Field(
        default=False,
        description=(
            "True 时用 pxr AddReference 挂背景,不走 Isaac Metrics Assembler."
            "SimReady 工厂常标 metersPerUnit=0.01 但几何已是米,Assembler 会再缩 100 倍."
        ),
    )
    wait_init_time: int = Field(
        default=30, 
        description="仿真世界初始化后等待时间."
    )
    dlss: bool = Field(
        default=False, 
        description="是否启用 DLSS."
    )
    auto_exposure: bool = Field(
        default=False, 
        description="是否启用自动曝光."
    )
    enable_newton: bool = Field(
        default=False,
        description=(
            "启动后启用 Isaac 自带 isaacsim.pip.newton."
        ),
    )
