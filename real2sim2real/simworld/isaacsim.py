"""Isaac Sim 6.2 的统一世界：PhysX、Newton USD 与显式混合步进。"""

from __future__ import annotations

import math
import select
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, FiniteFloat, field_validator, model_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import get_logger
from real2sim2real.simworld.base import BaseSimWorld

Vector3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
PhysicsEngine = Literal["physx", "newton", "hybrid"]


class IsaacsimWorldConfig(BaseConfig):
    """读取 settings 的 ``[sim_world]`` 段，配置统一的 Isaac Sim 世界。

    设置 ``backend="isaacsim"`` 选中本实现，再用 ``physics_engine`` 选择引擎；
    ``newton_execution`` 仅区分纯 Newton 的 USD/direct 路径，不改变 hybrid 的调度方式。
    模式选择及完整调用流程见 ``IsaacsimWorld`` 的类注释。

    世界只负责运行时、环境和步进。机器人、刚体、布料/柔体的资产、材料及接触代理，
    由 Scene/Robot/Planner 组件装配。所有路径共用此配置和世界类，无需按物体类型派生世界。
    """

    # 引擎与全局物理参数：步长单位为秒，重力单位为 m/s²；世界使用米、Z-up。
    backend: str = "isaacsim"
    physics_engine: PhysicsEngine = "physx"
    physics_dt: FiniteFloat = Field(default=1 / 120, gt=0, le=1)
    rendering_dt: FiniteFloat = Field(default=1 / 30, gt=0)
    gravity: Vector3 = (0.0, 0.0, -9.81)  # m/s²；世界使用米、Z-up
    device: str = "cuda:0"
    physx_solver: Literal["TGS", "PGS"] = "TGS"
    physx_enable_ccd: bool = False
    physx_enable_stabilization: bool = True

    # Newton USD 由世界应用这些参数；direct/hybrid 由模型回调读取并应用，世界不重复建模。
    # newton_execution 控制纯 Newton 的接入方式，仅 physics_engine="newton" 时生效：
    # - usd：Isaac 从 USD 场景建立 Newton 模型；世界应用配置，通过 SimulationManager 步进。
    # - direct：Scene/Robot/Planner 自行建立 Newton 模型；世界调用注册的 stepper，
    #   Kit 时间轴保持停止，只负责显示。运行前必须调用 register_newton_stepper(...)。
    # - hybrid 固定采用“PhysX 步进 → Newton 回调”，忽略此字段；physx 模式也不使用它。
    # newton_solver 则选择实际求解器（如 mujoco/xpbd/vbd），与上述接入方式分别配置。
    # 例如自建 VBD 布料模型：physics_engine="newton", newton_execution="direct",
    # newton_solver="vbd"，再由案例组件注册负责模型步进和状态同步的回调。
    newton_execution: Literal["usd", "direct"] = "usd"
    newton_solver: Literal["mujoco", "xpbd", "vbd", "featherstone", "semi_implicit"] = "mujoco"
    newton_substeps: int = Field(default=1, ge=1)
    newton_iterations: int = Field(default=10, ge=1)
    newton_use_cuda_graph: bool = True
    newton_solver_options: dict[str, Any] = Field(default_factory=dict)
    newton_config_options: dict[str, Any] = Field(default_factory=dict)

    # 应用与渲染参数：rendering_dt 不决定一次 step 内的物理步数。
    headless: bool = True
    renderer: Literal["RayTracedLighting", "PathTracing"] = "RayTracedLighting"
    width: int = Field(default=1280, gt=0)
    height: int = Field(default=720, gt=0)
    rt_subframes: int = Field(default=1, ge=1)
    samples_per_pixel_per_frame: int = Field(default=64, ge=1)
    warmup_frames: int = Field(default=2, ge=0, description="初始化时只刷新 UI/渲染的帧数")
    dlss: bool = False
    auto_exposure: bool = False
    fast_shutdown: bool = True
    extensions: list[str] = Field(default_factory=list)

    # 环境加载二选一：整份打开 stage，或在新建 stage 中引用背景。
    # 桌面物体与机器人由其他组件添加，不能借背景挂载覆盖它们的根路径。
    stage_usd_path: str = ""
    background_usd_path: str = ""
    background_prim_path: str = "/World/background"
    background_offset: Vector3 = (0.0, 0.0, 0.0)
    background_gravity: bool = False
    background_skip_metrics_assembler: bool = True
    background_light_scale: FiniteFloat = Field(default=1.0, ge=0)

    @field_validator("background_prim_path")
    @classmethod
    def _validate_background_path(cls, value: str) -> str:
        parts = value.split("/")
        if (
            not value.startswith("/World/")
            or not all(part.isidentifier() for part in parts[1:])
            or parts[2] in {"scene", "robots", "PhysicsScene"}
        ):
            raise ValueError(
                "背景必须挂在 /World 下独立的 prim，不能覆盖 scene/robots/PhysicsScene"
            )
        return value

    @model_validator(mode="after")
    def _validate_modes(self) -> Self:
        if (
            self.physics_engine == "newton"
            and self.newton_execution == "usd"
            and self.newton_solver not in {"mujoco", "xpbd", "vbd"}
        ):
            raise ValueError("Newton USD 后端仅提供 mujoco/xpbd/vbd；其他求解器需 direct")
        if self.stage_usd_path and self.background_usd_path:
            raise ValueError("stage_usd_path 与 background_usd_path 只能选择一个")
        # USD 场景将 Hz 存为整数，拒绝会被 SDK 静默截断的步长。
        if not (self.physics_engine == "newton" and self.newton_execution == "direct"):
            frequency = 1 / self.physics_dt
            if not math.isclose(frequency, round(frequency), abs_tol=1e-8, rel_tol=0):
                raise ValueError("USD 物理步长必须对应整数 Hz，例如 1/60、1/120、1/240")
        reserved = {
            "num_substeps",
            "physics_frequency",
            "time_step_app",
            "solver_cfg",
            "use_cuda_graph",
        }
        if reserved.intersection(self.newton_config_options):
            raise ValueError("newton_config_options 不能覆盖步长、子步数、步进所有者或 solver_cfg")
        if "solver_type" in self.newton_solver_options:
            raise ValueError("请用 newton_solver 选择求解器，不能在 options 中覆盖 solver_type")
        return self


class IsaacsimWorld(BaseSimWorld):
    """所有案例共用的 6.2 世界；构造时不启动 Kit 或加载仿真依赖。

    模式与选择
    ----------
    三种 ``physics_engine``，其中纯 Newton 有两条执行路径，共四种用法：

    1. ``physics_engine="physx"``：USD 场景统一交给 PhysX。
       适合继续使用 PhysX 机器人、碰撞及该引擎支持的刚体/柔体资产。
       ``physx_solver`` 可选 TGS/PGS；不需要注册 Newton 回调。
    2. ``physics_engine="newton", newton_execution="usd"``：官方 Newton USD 后端。
       适合让 Isaac 从兼容的 USD 资产建立整份 Newton 模型。
       ``newton_solver`` 可选 mujoco/xpbd/vbd，具体资产能力取决于求解器与安装的 schema；
       不能仅凭“柔体”就认定任意求解器都可用。本机缺少 VBD USD schema 时会明确报错。
    3. ``physics_engine="newton", newton_execution="direct"``：应用直接管理 Newton 模型。
       适合自行建立机器人、布料/柔体或组合多个 Newton 求解器的案例。
       必须注册 ``register_newton_stepper``；Kit 时间轴始终停止，只负责显示，
       不再由 USD 物理后端积分同一批对象。
    4. ``physics_engine="hybrid"``：PhysX USD + 应用管理的 Newton 模型。
       适合保留 PhysX 机器人，同时由 Newton 计算布料/柔体。
       必须注册 Newton stepper；每步先执行 PhysX，再调用 Newton stepper。
       ``newton_execution`` 不参与此模式的选择，hybrid 始终走显式联合步进。

    hybrid 的回调负责把 PhysX 状态同步到 Newton 接触代理、计算 Newton 动力学，
    再同步显示状态。世界不会自动创建跨引擎碰撞，也不会自动提供双向力反馈。
    一般先按资产和所需求解器选模式；刚体、柔体抓取任务均使用本类，不另建案例世界。
    最小配置为 ``[sim_world] backend="isaacsim", physics_engine="physx"``；
    更多字段和 TOML 写法见 ``configs/isaacsim_world.toml``。

    被 Job 调用的流程
    ----------------
    ``BaseJob.__init__``
      → ``AutoSimWorld.from_config()`` 读取 backend、加载 entry point 和本配置类
      → ``IsaacsimWorld(config)`` 只保存配置与状态，不启动仿真资源
    ``BaseJob.run()``
      → ``init_world`` 阶段调用本类 ``init_world()``：启动 Kit、加载环境、配置引擎
      → ``arrange_scene`` 阶段由 Scene 添加物体，读取本类的 stage
      → ``place_robot`` 阶段由 Robot 添加机器人，再初始化 Planner
      → ``start_simulation`` 阶段进入具体 Job 的 ``simulate()`` 执行仿真循环
      → 无论成功或异常，``finally: BaseJob.close()`` 调用本类 ``destroy_world()``

    需要 direct/hybrid 的案例组件应在 reset/play 之前构建 Newton runtime，并注册：
    ``register_newton_stepper(runtime.step, reset=runtime.reset, close=runtime.close)``。
    step 必需，reset/close 可选；回调自己应用本配置中的求解器、子步数、迭代数和重力。
    ``BaseJob.simulate()`` 默认为空，具体 Job 需要实现以下循环，基类不会自动播放：

    .. code-block:: python

        # 以下代码放在具体 Job.simulate() 中；此时场景、机器人与 Planner 已装配。
        sim_world = self.sim_world
        sim_world.reset()                   # 建立物理视图，并调用已注册的 Newton reset
        sim_world.play()                    # 允许显式步进；首次 play 也会自动补 reset
        while sim_world.is_running() and sim_world.is_playing():
            # 在此读取状态、计算并下发控制；任务结束时 break。
            sim_world.step(render=True)    # 一个 physics_dt，可按采集频率选择是否渲染
        # 资源销毁由 BaseJob.run() 的 finally 统一负责。

    步进和资源约定
    --------------
    - PhysX / Newton USD 的 ``world`` 句柄为 6.2 ``SimulationManager``；direct 为 None。
      ``app`` 是 SimulationApp，``stage`` 是 Scene/Robot 共用的 USD stage。
    - ``step()`` 只在运行中积分；PhysX/Newton USD 通过 Manager 推进一步，
      direct 只调用 Newton 回调，hybrid 按 PhysX → Newton 顺序执行。
      ``render=True`` 最后调用 update，不因物理/渲染步长不同而增加物理步数。
    - ``update()`` 只刷新 Kit/UI/渲染；``warmup_frames`` 也是只渲染。
      USD 模型 reset 时可能包含 SDK 初始化步，``physics_steps`` 只统计显式 step。
    - ``pause()`` 后 play 继续当前状态；``stop()`` 后 play 会重新初始化模型。
      direct/hybrid 的模型重置效果由注册的 reset 回调负责。
    - 独立使用本类时自行用 try/finally 调用 ``destroy_world()``。
      销毁时停止时间轴、执行 Newton close、关闭 Kit，可重复调用。
      ``fast_shutdown=True`` 沿用 Isaac 快速退出行为，关闭 Kit 时退出进程。
    """

    config_cls = IsaacsimWorldConfig

    def __init__(self, config: IsaacsimWorldConfig | None = None) -> None:
        """保存世界配置和运行状态；仿真资源在 init_world 中创建。"""
        super().__init__()
        self.config = config if config is not None else self.config_cls.from_settings("sim_world")
        self.log = get_logger(__name__)
        self.physics_steps = 0
        self._playing = False
        self._initialized = False
        self._newton_config: Any | None = None
        self._newton_stepper: Callable[[float], None] | None = None
        self._newton_reset: Callable[[], None] | None = None
        self._newton_close: Callable[[], None] | None = None

    @property
    def physics_engine(self) -> PhysicsEngine:
        """配置指定的引擎模式。"""
        return self.config.physics_engine

    @property
    def uses_direct_newton(self) -> bool:
        """全部动力学是否由应用的 Newton 模型负责。"""
        return self.physics_engine == "newton" and self.config.newton_execution == "direct"

    def register_newton_stepper(
        self,
        stepper: Callable[[float], None],
        *,
        reset: Callable[[], None] | None = None,
        close: Callable[[], None] | None = None,
    ) -> None:
        """挂载 direct/hybrid 模型；dt 为秒，回调完成一个完整 physics_dt。

        回调拥有 Newton 模型，负责应用 config 中的求解器、子步数、重力等参数。
        hybrid 在 PhysX 一步后调用；回调同步接触代理、步进 Newton 并写回显示状态。
        可选 reset/close 分别随世界重置/销毁执行；不允许重复注册或在运行中更换。
        """
        if not (self.uses_direct_newton or self.physics_engine == "hybrid"):
            raise ValueError("只有 direct Newton 或 hybrid 模式接受 Newton stepper")
        if self._newton_stepper is not None or self._playing:
            raise RuntimeError("Newton stepper 已注册或世界正在运行")
        if not callable(stepper) or any(
            fn is not None and not callable(fn) for fn in (reset, close)
        ):
            raise TypeError("Newton stepper/reset/close 必须可调用")
        self._newton_stepper, self._newton_reset, self._newton_close = stepper, reset, close

    def init_world(self) -> None:
        """启动 SimulationApp，加载环境并配置引擎；资产装配后再 reset/play。"""
        if self.app is not None:
            raise RuntimeError("IsaacsimWorld 已初始化")
        from isaacsim import SimulationApp

        try:
            self.app = SimulationApp(
                {
                    "headless": self.config.headless,
                    "renderer": self.config.renderer,
                    "width": self.config.width,
                    "height": self.config.height,
                    "samples_per_pixel_per_frame": self.config.samples_per_pixel_per_frame,
                    "fast_shutdown": self.config.fast_shutdown,
                    "extra_args": [
                        "--/exts/isaacsim.physics.newton/auto_switch_on_startup=false",
                    ],
                }
            )
            self._initialize_runtime()
        except BaseException:
            self.destroy_world()
            raise

    def _initialize_runtime(self) -> None:
        import carb
        import omni.timeline
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.extensions import enable_extension

        self._timeline = omni.timeline.get_timeline_interface()
        self._manager = SimulationManager
        self._renderer = RenderingManager
        self._settings = carb.settings.get_settings()
        self._timeline_action("stop")
        self._settings.set_bool("/exts/isaacsim.physics.newton/auto_switch_on_startup", False)
        required = list(self.config.extensions)
        if self.physics_engine != "physx":
            required.append("isaacsim.pip.newton")
        if self.physics_engine == "newton" and not self.uses_direct_newton:
            required.extend(["isaacsim.physics.newton", "isaacsim.physics.newton.tensors"])
        for extension in dict.fromkeys(required):
            if not enable_extension(extension):
                raise RuntimeError(f"Isaac Sim 6.2 扩展不可用: {extension}")
        self.update()
        self._load_environment()
        self._renderer.set_dt(self.config.rendering_dt)
        self._settings.set("/rtx/rendermode", self.config.renderer)
        self._settings.set_bool("/rtx/post/dlss/enabled", self.config.dlss)
        # SimulationApp defaults to AA mode 3 (DLSS). Select spatial FXAA when
        # DLSS is disabled, avoiding incompatible temporal history across cameras.
        self._settings.set_int("/rtx/post/aa/op", 3 if self.config.dlss else 2)
        self._settings.set_bool("/rtx/post/tonemap/autoExposure/enabled", self.config.auto_exposure)
        if self.config.headless:
            # Dataset RGB must not include viewport grids, axes, or selection guides.
            self._settings.set_int("/persistent/app/viewport/displayOptions", 0)
        if not self.uses_direct_newton:
            self._configure_physics()
            self.world = self._manager
        for _ in range(self.config.warmup_frames):
            self.update()
        self.log.info(
            "世界初始化完成", engine=self.physics_engine, execution=self.config.newton_execution
        )

    @staticmethod
    def _resolve_asset(path: str) -> str:
        if "://" in path:
            return path
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return str(resolved)

    def _load_environment(self) -> None:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

        context = omni.usd.get_context()
        if self.config.stage_usd_path:
            if not context.open_stage(self._resolve_asset(self.config.stage_usd_path)):
                raise RuntimeError("环境 USD 打开失败")
        else:
            context.new_stage()
        self.stage = context.get_stage()
        if self.stage is None:
            raise RuntimeError("无法创建 USD stage")
        if self.config.stage_usd_path:
            if (
                not math.isclose(UsdGeom.GetStageMetersPerUnit(self.stage), 1.0)
                or UsdGeom.GetStageUpAxis(self.stage) != UsdGeom.Tokens.z
            ):
                raise ValueError("整份环境 stage 必须使用米、Z-up；引用背景可单独配置位移")
        else:
            UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        for path in ("/World", "/World/scene", "/World/robots"):
            if not self.stage.GetPrimAtPath(path):
                UsdGeom.Xform.Define(self.stage, path)
        if not self.stage.GetDefaultPrim():
            self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))
        if self.config.background_usd_path:
            path = self.config.background_prim_path
            asset = self._resolve_asset(self.config.background_usd_path)
            if self.config.background_skip_metrics_assembler:
                background = UsdGeom.Xform.Define(self.stage, path).GetPrim()
                if not background.GetReferences().AddReference(asset):
                    raise RuntimeError(f"背景引用失败: {asset}")
            else:
                from isaacsim.core.utils.stage import add_reference_to_stage

                background = add_reference_to_stage(asset, path)
            self.stage.Load(path)
            UsdGeom.Xformable(background).AddTranslateOp(opSuffix="rsr_offset").Set(
                Gf.Vec3d(*self.config.background_offset)
            )
            if self.config.background_light_scale != 1.0:
                for prim in Usd.PrimRange(background):
                    if prim.HasAPI(UsdLux.LightAPI):
                        light = UsdLux.LightAPI(prim)
                        light.GetIntensityAttr().Set(
                            light.GetIntensityAttr().Get() * self.config.background_light_scale
                        )
            if not self.config.background_gravity:
                # 仅修改背景子树的动态刚体，保留静态碰撞和后续 Scene/Robot 的物理属性。
                for prim in Usd.PrimRange(background):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)

    def _configure_physics(self) -> None:
        from isaacsim.core.simulation_manager import PhysicsScene, PhysxScene

        engine = "newton" if self.physics_engine == "newton" else "physx"
        if not self._manager.switch_physics_engine(engine):
            raise RuntimeError(f"Isaac Sim 6.2 无法启用物理引擎: {engine}")
        paths = PhysicsScene.get_physics_scene_paths() or ["/World/PhysicsScene"]
        if engine == "newton":
            self._configure_newton(paths)
        else:
            for path in paths:
                scene = PhysxScene(path)
                scene.set_dt(self.config.physics_dt)
                scene.set_gravity(self.config.gravity)
                scene.set_solver_type(self.config.physx_solver)
                scene.set_enabled_ccd(self.config.physx_enable_ccd)
                scene.set_enabled_stabilization(self.config.physx_enable_stabilization)
        self._manager.set_default_physics_scene(paths[0])
        if self.config.device:
            self._manager.set_device(self.config.device)

    def _configure_newton(self, paths: list[str]) -> None:
        from isaacsim.core.simulation_manager import NewtonMjcScene, PhysicsScene
        from isaacsim.physics.newton import (
            MuJoCoSolverConfig,
            NewtonConfig,
            VBDSolverConfig,
            XPBDSolverConfig,
            configure_newton,
            get_newton_solver_to_physics_scene_object,
            switch_newton_solver,
        )

        solver = self.config.newton_solver
        options = dict(self.config.newton_solver_options)
        if solver in {"xpbd", "vbd"}:
            options["iterations"] = self.config.newton_iterations
        elif self.config.device == "cpu":
            options.setdefault("use_mujoco_cpu", True)
        solver_cls = {
            "mujoco": MuJoCoSolverConfig,
            "xpbd": XPBDSolverConfig,
            "vbd": VBDSolverConfig,
        }
        self._newton_config = NewtonConfig(
            **self.config.newton_config_options,
            num_substeps=self.config.newton_substeps,
            physics_frequency=1 / self.config.physics_dt,
            use_cuda_graph=self.config.newton_use_cuda_graph,
            time_step_app=False,
            solver_cfg=solver_cls[solver](**options),
        )
        configure_newton(self._newton_config)
        for path in paths:
            scene = PhysicsScene(path)
            try:
                # 已有环境常带 PhysX scene schema；保留它会让通用 set_dt 只写 PhysX Hz。
                # 仅在当前 stage 移除该场景 API，不改环境文件或物体的碰撞/关节属性。
                if scene.prim.HasAPI("PhysxSceneAPI"):
                    scene.prim.RemoveAPI("PhysxSceneAPI")
                if not switch_newton_solver(solver, scene.prim):
                    raise RuntimeError(f"无法选择 Newton 求解器 {solver}")
                wrapper = get_newton_solver_to_physics_scene_object()[solver](scene.prim)
                wrapper.set_dt(self.config.physics_dt)
                wrapper.set_gravity(self.config.gravity)
                if isinstance(wrapper, NewtonMjcScene):
                    # 6.2 parser 读 Newton Hz，Manager.step 读 mjc dt，必须同时设置。
                    scene.set_dt(self.config.physics_dt)
                    wrapper.set_iterations(self.config.newton_iterations)
                else:
                    wrapper.set_max_solver_iterations(self.config.newton_iterations)
            except Exception as exc:
                raise RuntimeError(
                    f"当前安装的 Newton USD schema 无法配置 {solver}；"
                    "检查 6.2 安装，或选择 direct 并注册模型 stepper"
                ) from exc

    def _require_app(self) -> None:
        if self.app is None:
            raise RuntimeError("请先调用 init_world()")

    def _require_stepper(self) -> None:
        if (
            self.uses_direct_newton or self.physics_engine == "hybrid"
        ) and not self._newton_stepper:
            raise RuntimeError("请先注册 Newton stepper，不能在缺少 Newton 模型时开始仿真")

    def _timeline_action(self, action: str) -> None:
        getattr(self._timeline, action)()
        self._timeline.commit()

    def reset(self) -> None:
        """资产装配完成后重建物理视图；direct/hybrid 同时调用注册的模型 reset。"""
        self._require_app()
        self._require_stepper()
        self.stop()
        if not self.uses_direct_newton:
            self._manager.invalidate_physics()
            self._manager.initialize_physics()
            if self._manager.get_physics_simulation_view() is None:
                raise RuntimeError("USD 物理模型初始化失败，请检查资产和求解器兼容性")
            if self.physics_engine == "newton":
                from isaacsim.physics.newton import acquire_stage

                runtime = acquire_stage()
                if runtime is None or not runtime.initialized:
                    raise RuntimeError("Newton USD 模型初始化失败")
        if self._newton_reset is not None:
            self._newton_reset()
        self.physics_steps = 0
        self._initialized = True

    def play(self) -> None:
        """开始显式步进；首次调用时自动初始化已装配的模型。"""
        self._require_app()
        self._require_stepper()
        if not self._initialized:
            self.reset()
        self._timeline_action("stop" if self.uses_direct_newton else "play")
        self._playing = True

    def pause(self) -> None:
        """暂停动力学，仍可调用 update 刷新 GUI。"""
        self._require_app()
        self._timeline_action("stop" if self.uses_direct_newton else "pause")
        self._playing = False

    def stop(self) -> None:
        """停止并使下次 play 重建模型；注册的 Newton runtime 保留到销毁。"""
        self._require_app()
        self._timeline_action("stop")
        self._playing = False
        self._initialized = False

    def is_running(self) -> bool:
        """Kit 窗口或无头应用是否仍运行。"""
        return self.app is not None and bool(self.app.is_running())

    def is_playing(self) -> bool:
        """世界是否允许运行下一物理步。"""
        if self.app is None or not self._playing:
            return False
        return self.uses_direct_newton or bool(self._timeline.is_playing())

    def step(self, *, render: bool = True) -> None:
        """运行中推进一个 physics_dt；渲染开关不改变物理步数。"""
        self._require_app()
        if self.is_playing():
            self._require_stepper()
            if self.uses_direct_newton:
                self._timeline_action("stop")
            else:
                expected = "newton" if self.physics_engine == "newton" else "physx"
                if self._manager.get_active_physics_engine() != expected:
                    raise RuntimeError("GUI/外部代码更换了物理引擎，请停止并重新初始化世界")
                self._manager.step(steps=1, update_fabric=self._manager.is_fabric_enabled())
            if self._newton_stepper is not None:
                self._newton_stepper(self.config.physics_dt)
            self.physics_steps += 1
        if render:
            self.update()

    def update(self) -> None:
        """只刷新 Kit/渲染，不推进任何引擎；设置在异常时也恢复。"""
        self._require_app()
        if self.uses_direct_newton:
            self._timeline_action("stop")
        key = "/app/player/playSimulations"
        previous = self._settings.get(key)
        self._settings.set_bool(key, False)
        try:
            self.app.update()
        finally:
            self._settings.set(key, previous if previous is not None else True)

    def wait_for_gui_edit(self, *, reason: str = "") -> None:
        """GUI 检查期间停止物理，终端 Enter 或关闭窗口结束检查。"""
        self._require_app()
        if self.config.headless:
            return
        self.stop()
        self.log.info("GUI 检查：终端按 Enter 继续", reason=reason)
        while self.is_running():
            self._timeline_action("stop")
            self.update()
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break

    def destroy_world(self) -> None:
        """逆序释放模型和 Kit；部分初始化失败和重复关闭均可调用。"""
        active_error = sys.exc_info()[1]
        cleanup_error: Exception | None = None
        self._playing = False
        self._initialized = False
        if self.app is not None and hasattr(self, "_timeline"):
            try:
                self._timeline_action("stop")
            except Exception as exc:
                cleanup_error = exc
                self.log.exception("停止时间轴失败")
        try:
            if self._newton_close is not None:
                close, self._newton_close = self._newton_close, None
                close()
        except Exception as exc:
            cleanup_error = cleanup_error or exc
            self.log.exception("清理 Newton 模型失败")
        finally:
            self._newton_stepper = self._newton_reset = self._newton_close = None
            self._newton_config = None
            self.world = self.stage = None
            if self.app is not None:
                app, self.app = self.app, None
                if active_error is not None:
                    self.log.error("仿真异常，关闭应用", error=str(active_error), exc_info=True)
                try:
                    app.close(exit_code=int(active_error is not None or cleanup_error is not None))
                except Exception:
                    if active_error is None and cleanup_error is None:
                        raise
                    self.log.exception("关闭 SimulationApp 失败")
        if cleanup_error is not None and active_error is None:
            raise cleanup_error
