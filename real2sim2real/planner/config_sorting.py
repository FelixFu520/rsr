"""``[planner]`` 分拣中继配置(新增文件,不改 ``planner/config.py``)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from real2sim2real.core.config import BaseConfig


class SortingPlannerConfig(BaseConfig):
    """``[planner]``:Pro-M6 双臂分拣中继(柔性包裹 入带 → 平台 → 出带).

    状态机在 :mod:`real2sim2real.planner.prom6_relay`.刚体仍走 PhysX;袋装包裹走
    Newton VBD 接触抓取.本配置暴露场景 prim、Newton 材料与提起验收.
    """

    timeout_s: float = Field(
        default=1800.0,
        description="单集分拣超时(s,墙钟);RTX 渲染 + GPU 柔性体下实时率远低于 1x",
    )
    robot_prim_path: str = Field(
        default="/World/robots/Pro_m6_696_dynamics_fixed", description="机器人根 prim"
    )
    clothing_root_prim_path: str = Field(
        default="/World/scene/ImportedBaggedClothing", description="柔性包裹根 prim"
    )
    clothing_surface_prim_path: str = Field(
        default="/World/scene/ImportedBaggedClothing/Asset/visual/visual",
        description="袋装包裹可视 Mesh;Play 前会去掉 PhysX 柔性 API,改由 Newton 写回顶点",
    )
    clothing_volume_proxy_prim_path: str = Field(
        default="/World/scene/ImportedBaggedClothing/Asset/visual/visual_proxy",
        description="体积代理 prim;存在时必须 inactive,避免与 Newton 软体抢显示",
    )
    support_prim_path: str = Field(
        default="/World/scene/platform/Empty/Plane/Plane_002", description="平台支撑面 prim"
    )
    belt_prim_path: str = Field(
        default="/World/scene/ConveyorBelt_A06/Belt", description="出带皮带 prim,判定包裹是否送出"
    )
    infeed_prim_path: str = Field(
        default="/World/scene/ConveyorBelt_A06_01",
        description="入带 prim;Newton 里做成静态碰撞盒,空则跳过",
    )
    left_gripper_prim_path: str = Field(
        default="", description="左夹爪 prim;空则 <root>/robotiq_left"
    )
    right_gripper_prim_path: str = Field(
        default="", description="右夹爪 prim;空则 <root>/robotiq_right"
    )
    left_finger_prim_paths: list[str] = Field(
        default_factory=list,
        description="左指垫 Mesh prim;空则在左夹爪下按 inner_finger 自动发现",
    )
    right_finger_prim_paths: list[str] = Field(
        default_factory=list,
        description="右指垫 Mesh prim;空则在右夹爪下按 inner_finger 自动发现",
    )
    runtime_root_prim_path: str = Field(
        default="/World/ProM6_SortingRuntime",
        description="运行期临时 prim 根,收尾时整棵删掉",
    )
    youngs_modulus: float = Field(default=10000.0, gt=0.0, description="Newton 包裹杨氏模量 Pa")
    poisson_ratio: float = Field(default=0.3, gt=0.0, lt=0.5, description="Newton 包裹泊松比")
    density: float = Field(default=70.0, gt=0.0, description="Newton 包裹等效密度 kg/m^3")
    particle_radius: float = Field(default=0.0015, gt=0.0, description="Newton 粒子接触半径 m")
    soft_contact_mu: float = Field(default=0.5, ge=0.0, description="Newton 软接触摩擦")
    shape_mu: float = Field(default=1.5, ge=0.0, description="Newton 刚体代理摩擦")
    newton_substeps: int = Field(default=10, ge=1, description="每物理帧 Newton 子步数")
    vbd_iterations: int = Field(default=10, ge=1, description="VBD 每子步迭代次数")
    min_lift_height_m: float = Field(
        default=0.08,
        ge=0.0,
        description="左右提起结束时包裹 COM 相对提起前至少升高这么多,否则 FAILED",
    )
    belt_speed_mps: float = Field(
        default=0.5, ge=0.0, description="松爪后出带运动学速度 m/s,沿世界 -X"
    )
    physics_device: str = Field(default="cuda:0", description="PhysX 设备;柔性体必须 GPU")
    physics_hz: float = Field(default=60.0, gt=0.0, description="物理频率,与 physics_dt 对齐")
    replay_cache_dir: str = Field(
        default="outputs/sorting_pro_m6/replay_cache",
        description="成功一次后落盘的 npz 回放目录",
    )
    drive_mode: Literal["callback", "manual"] = Field(
        default="callback",
        description=(
            "callback:注册 PHYSICS_POST_STEP 回调推进状态机(与原脚本一致);"
            "manual:由 Job 循环每步显式调用 step(dt)"
        ),
    )
    timing_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="覆盖 prom6_relay.TimingConfig 的字段(秒),如 left_drag_seconds",
    )
