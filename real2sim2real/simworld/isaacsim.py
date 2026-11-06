"""Isaac Sim 世界配置与生命周期实现。"""

from __future__ import annotations

import select
import sys
from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field, model_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.core.logger import get_logger
from real2sim2real.simworld.base import BaseSimWorld
from real2sim2real.utils import (
    WORLD_BACKGROUND_PRIM,
    ensure_world_mount_roots,
    resolve_usd_path,
)


class IsaacsimSimWorldConfig(BaseConfig):
    """``[sim_world]``:仿真世界配置."""

    headless: bool = Field(default=True, description="是否无头模式.")
    renderer: str = Field(
        default="RayTracedLighting", description="RayTracedLighting速度快, PathTracing渲染效果逼真"
    )
    rt_subframes: int = Field(default=8, description="光线追踪渲染器会在后台迭代多少次子步.")
    physics_dt: float = Field(default=1.0 / 60.0, ge=0, description="物理步长.")
    rendering_dt: float = Field(default=1.0 / 30.0, ge=0, description="渲染步长.")
    physics_engine: Literal["", "physx", "newton", "hybrid"] = Field(
        default="",
        description=(
            "physx:仅 PhysX;newton:仅 Newton;hybrid:PhysX 刚体 + 应用手动 Newton 柔体."
            "空值兼容旧 enable_newton,默认选择 physx."
        ),
    )
    physx_solver: Literal["TGS", "PGS"] = Field(
        default="TGS", description="PhysX 刚体与关节约束求解器,仅 physx/hybrid 使用."
    )
    newton_execution: Literal["usd", "direct"] = Field(
        default="usd",
        description=(
            "纯 Newton 的执行方式:usd 使用 Isaac 官方 Newton 后端;"
            "direct 由注册的 stepper 独占整个 Newton 模型,Kit 只负责显示."
        ),
    )
    newton_solver: Literal["mujoco", "xpbd", "vbd", "featherstone", "semi_implicit"] = Field(
        default="mujoco",
        description="Newton 求解器;Isaac USD 后端支持 mujoco/xpbd/vbd,其余仅用于 direct.",
    )
    newton_substeps: int = Field(default=1, ge=1, description="每个 Newton 物理步的子步数.")
    newton_iterations: int = Field(default=10, ge=1, description="Newton 约束求解迭代数.")
    newton_use_cuda_graph: bool = Field(
        default=True, description="官方 Newton 后端使用 CUDA Graph."
    )
    newton_solver_options: dict[str, Any] = Field(
        default_factory=dict, description="传给安装版本 Newton solver config 的额外参数."
    )
    newton_config_options: dict[str, Any] = Field(
        default_factory=dict, description="传给安装版本 NewtonConfig 的额外参数."
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
    background_usd_path: str = Field(default="", description="背景 USD 路径.")
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
    wait_init_time: int = Field(default=30, description="仿真世界初始化后等待时间.")
    dlss: bool = Field(default=False, description="是否启用 DLSS.")
    auto_exposure: bool = Field(default=False, description="是否启用自动曝光.")
    enable_newton: bool = Field(
        default=False,
        description="兼容旧配置:physics_engine 为空时 True 选择 hybrid,False 选择 physx.",
    )

    @property
    def resolved_physics_engine(self) -> Literal["physx", "newton", "hybrid"]:
        """返回显式引擎,或从旧 ``enable_newton`` 推导兼容模式."""
        return self.physics_engine or ("hybrid" if self.enable_newton else "physx")

    @model_validator(mode="after")
    def _validate_newton_execution(self) -> IsaacsimSimWorldConfig:
        if self.resolved_physics_engine != "physx" and self.physics_dt <= 0:
            raise ValueError("Newton/hybrid 模式需要大于零的 physics_dt")
        if (
            self.resolved_physics_engine == "newton"
            and self.newton_execution == "usd"
            and self.newton_solver not in {"mujoco", "xpbd", "vbd"}
        ):
            raise ValueError("Isaac Newton USD 后端仅支持 mujoco/xpbd/vbd;其他求解器须选择 direct")
        return self


class IsaacsimSimWorld(BaseSimWorld):
    """Isaac Sim 世界:管理 ``SimulationApp`` / ``World`` 生命周期.

    职责:
    - 环境检查与 App / World 启停、步进
    - 加载环境 USD 或背景引用
    - 视口辅助(camera gizmo / 独立预览窗)
    - 为 direct/hybrid 模式调用应用注册的 Newton stepper

    额外场景资产与机器人分别由 Scene/Robot 初始化和编排位姿。
    """

    config_cls = IsaacsimSimWorldConfig

    def __init__(self, config: IsaacsimSimWorldConfig | Any | None = None, **kwargs: Any) -> None:
        """读取 ``[sim_world]`` 配置,占位 App / World / Stage(真正启动在 ``init_world``).

        Args:
            config: 已构造的 Isaacsim 配置;为 None 时从 settings 的 ``sim_world`` 段加载.
            **kwargs: 工厂可能传入的多余字段,本后端不使用.

        Returns:
            None.
        """
        super().__init__()
        del kwargs
        self.config: IsaacsimSimWorldConfig = config or self.config_cls.from_settings("sim_world")

        self.app: Any | None = None  # isaacsim.SimulationApp;init_world 后才有
        self.world: Any | None = None  # isaacsim.core.api.World;承载物理/渲染步进
        self.stage: Any | None = None  # 当前 USD stage,Scene/Robot 往这里挂 prim
        self.physics_steps = 0
        self._direct_playing = False
        self._newton_stepper: Callable[[float], None] | None = None
        self._newton_runtime_config: Any | None = None

    @property
    def physics_engine(self) -> Literal["physx", "newton", "hybrid"]:
        """当前配置的实际物理模式;hybrid 不会启动官方 Newton USD 后端."""
        return self.config.resolved_physics_engine

    @property
    def uses_direct_newton(self) -> bool:
        """是否由应用直接拥有全部 Newton 动力学,PhysX 不参与积分."""
        return self.physics_engine == "newton" and self.config.newton_execution == "direct"

    def register_newton_stepper(self, stepper: Callable[[float], None] | None) -> None:
        """注册 direct/hybrid 的 Newton 步进函数;hybrid 在 PhysX 步进后调用.

        stepper 接收以秒为单位的 ``physics_dt``.应用负责模型、接触、状态同步与清理;
        hybrid 是显式联合仿真,不会自动产生跨引擎的反作用力.
        """
        if stepper is not None and not callable(stepper):
            raise TypeError("Newton stepper 必须可调用")
        if stepper is not None and not (self.uses_direct_newton or self.physics_engine == "hybrid"):
            raise ValueError("只有 direct Newton 或 hybrid 模式接受应用 Newton stepper")
        self._newton_stepper = stepper

    # ==========> 仿真世界生命周期 <==========

    def init_world(self) -> None:
        """建立仿真运行时(App / World),并按配置引用背景 USD.

        Isaac 要求先启动 ``SimulationApp``,再 import ``isaacsim.core.*``,
        否则 Kit 插件未就绪会直接 ImportError.
        由 ``BaseJob.init_world`` 调用;之后 Scene/Robot 才能访问 stage.

        Returns:
            None.
        """
        from isaacsim import SimulationApp  # 必须在本函数内 import:App 尚未启动时顶层 import 会失败

        # (1) 启动SimulationAPP
        # 渲染器未写则回退光线追踪照明(比 PathTracing 快,适合采集)
        renderer = self.config.renderer or "RayTracedLighting"
        # 组装 Kit 启动参数;launch_config 只在构造时生效一次
        launch_config = {
            "headless": self.config.headless,  # True 不弹 GUI,CI/无显示器用
            "renderer": renderer,  # 视口光照算法名,如 RayTracedLighting
            "rt_subframes": self.config.rt_subframes,  # RTX 每帧后台子迭代次数
        }
        # 启动 Kit 进程内嵌 App,之后才能用 isaacsim.core / carb
        self.app: SimulationApp = SimulationApp(launch_config=launch_config)
        import carb

        # 显式模式由本组件选择.避免用户 Kit 设置在扩展加载时自动改成另一动力学所有者.
        carb.settings.get_settings().set_bool(
            "/exts/isaacsim.physics.newton/auto_switch_on_startup", False
        )

        # (2) 启用可选插件
        # 占用图 / RTX 实验相机都是可选插件,须显式 enable 后再 update
        from isaacsim.core.utils.extensions import enable_extension  # App 起来后才能 import

        enable_extension("isaacsim.asset.gen.omap")  # 占用图生成,规划/避障可能用到
        enable_extension("isaacsim.sensors.experimental.rtx")  # RTX CameraSensor 依赖此扩展
        if self.physics_engine != "physx":
            # hybrid/direct 只加载库,不能启用会替代 PhysX 的 USD backend.
            self._enable_newton_extension("isaacsim.pip.newton")
        if self.physics_engine == "newton" and not self.uses_direct_newton:
            self._enable_newton_extension("isaacsim.physics.newton")
            self._enable_newton_extension("isaacsim.physics.newton.tensors")
        self.update()  # 只泵 UI/扩展,不允许启动期间隐式推进物理

        # (3) 配置渲染设置
        # carb: Kit 运行时全局设置(RTX 视口).launch_config 只影响启动参数,
        # 这里再写一遍,避免 App 起来后仍沿用默认 PathTracing / DLSS / 自动曝光.
        try:
            import carb  # noqa: PLC0415  # Kit 设置总线,须在 App 启动后 import

            settings = carb.settings.get_settings()  # 全局 settings 单例
            # 与配置一致的渲染模式(如 RayTracedLighting),决定视口光照算法
            settings.set("/rtx/rendermode", renderer)
            # 关闭 DLSS:半分辨率重建易导致画面发糊/视角错位,布局调试时不需要
            settings.set_bool("/rtx/post/dlss/enabled", self.config.dlss)
            # 关闭自动曝光:室内场景亮度波动时避免画面闪白/闪黑
            settings.set_bool("/rtx/post/tonemap/autoExposure/enabled", self.config.auto_exposure)
            # 默认透视视口画出 Camera prim 的视锥/图标,方便对齐镜头
            self._enable_camera_gizmos(settings)
        except Exception as exc:  # noqa: BLE001  # carb 在部分无头构建里可能缺失
            self.log.warning("配置渲染设置失败", error=str(exc))  # 渲染设置失败不阻断世界启动

        # (4) 创建 World 并加载背景 USD
        # 先建空 World(含 /World),再在 /World 下建 background / scene / robots
        # 三个独立挂载点,把背景 USD 引用到 /World/background,避免背景尺度传到场景和机器人.
        # stage_usd_path 是例外:成品场景自带 /PhysicsScene 与绝对 prim 路径,
        # 必须整份 open 成 stage,否则 reference 只带进默认 prim 子树,路径与物理设置都会变。
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

        self._open_stage_usd()  # 未配置 stage_usd_path 时是 no-op,继续走背景 reference 通路

        # 物理设备 / 张量后端须在 World 构造前写入 SimulationManager 单例;
        # World 与后续 Articulation / RigidPrimView 都读这份全局状态,建完再改当前世界不会跟着换.
        # 空字符串不碰 Isaac 默认(桌面抓取通常够用);人形策略须 cuda + torch,否则观测/动作设备不一致.
        device = (self.config.device or "").strip()
        physics_backend = (self.config.physics_backend or "").strip()
        if device or physics_backend:
            from isaacsim.core.simulation_manager import SimulationManager

            if physics_backend:
                SimulationManager.set_backend(
                    physics_backend
                )  # 如 torch:关节/刚体 View 读写走 PyTorch,非默认 numpy
            if device:
                SimulationManager.set_physics_sim_device(
                    device
                )  # 如 cuda:PhysX 在 GPU 积分,非默认 CPU
            self.log.debug("设备和后端", device=device or "default", backend=physics_backend)

        if self.physics_engine == "newton":
            import omni.usd
            from pxr import UsdGeom

            # deprecated World/PhysicsContext 会无条件应用 PhysxSceneAPI;纯 Newton 不能创建它.
            if get_current_stage() is None:
                omni.usd.get_context().new_stage()
            UsdGeom.SetStageMetersPerUnit(get_current_stage(), 1.0)
            if not self.uses_direct_newton:
                self._configure_newton_usd()
        else:
            from isaacsim.core.api import World  # 仅 PhysX/hybrid 使用 legacy World

            self._select_physics_engine("physx")
            world_kwargs: dict[str, Any] = {
                "stage_units_in_meters": 1.0,
                "physics_dt": self.config.physics_dt,
                "rendering_dt": self.config.rendering_dt,
            }
            if device:
                world_kwargs["device"] = device
            if physics_backend:
                world_kwargs["backend"] = physics_backend
            try:
                self.world = World(**world_kwargs)
            except TypeError:
                world_kwargs.pop("device", None)
                world_kwargs.pop("backend", None)
                self.world = World(**world_kwargs)
            self.world.get_physics_context().set_solver_type(self.config.physx_solver)
        self.stage = get_current_stage()  # 记下当前 stage,供 Scene/Robot 查 prim
        ensure_world_mount_roots(self.stage)  # /World/background、/scene、/robots 互为兄弟
        # 整份 open 过 stage 就不再挂背景;两者都空时只留 World 自建的空 /World 与三个挂载点
        raw_background = "" if self.config.stage_usd_path else self.config.background_usd_path
        usd_path = resolve_usd_path(raw_background) if raw_background else ""
        if usd_path:  # 空字符串表示不要背景,只留空挂载点
            from real2sim2real.utils import (
                add_usd_reference,
                ensure_prim_loaded,
                mark_ignore_metrics_assembler,
                spawn_reference,
            )

            bg_prim = (
                self.config.background_prim_path or WORLD_BACKGROUND_PRIM
            ).strip() or WORLD_BACKGROUND_PRIM
            if self.config.background_skip_metrics_assembler:
                spawn_reference(self.stage, bg_prim, usd_path, add_usd_reference)
                mark_ignore_metrics_assembler(self.stage, bg_prim)
            else:
                spawn_reference(self.stage, bg_prim, usd_path, add_reference_to_stage)
            ensure_prim_loaded(self.stage, bg_prim)
            self._wait_stage_loaded()
            self.log.info("引用背景成功", prim_path=bg_prim, usd=usd_path)
            if not self.config.background_gravity:  # False:背景当静态碰撞,不随重力掉落
                self._make_background_static()  # 只锁背景子树,不碰 /World/scene 与机器人
        if self.world is not None:
            self.world.reset()  # Newton 等待 Scene/Robot 装配完成后再解析模型

        # (6) 停止仿真,避免 Play 时 Kit 锁定编辑
        self.stop()

        # (7) 多泵几帧只保证 UI/Hydra 就绪;解决不了「无灯/贴图失败」导致的黑屏
        for _ in range(max(1, int(self.config.wait_init_time))):  # 至少泵 1 帧
            self.update()  # 推进 Kit 主循环,让材质/灯光流式加载
        self.log.info(
            "physics_mode_initialized",
            physics_engine=self.physics_engine,
            newton_execution=self.config.newton_execution,
            newton_solver=self.config.newton_solver if self.physics_engine != "physx" else None,
        )

    def _enable_newton_extension(self, name: str) -> None:
        """加载所选模式必需的扩展,不可用时立即报错而不静默退回 PhysX."""
        from isaacsim.core.utils.extensions import enable_extension

        if not enable_extension(name):
            raise RuntimeError(f"无法启用 {name};请使用包含 Newton 的 Isaac Sim 6.2 安装")

    def _select_physics_engine(self, name: Literal["physx", "newton"]) -> None:
        from isaacsim.core.simulation_manager import SimulationManager

        if SimulationManager.get_active_physics_engine() != name:
            if not SimulationManager.switch_physics_engine(name):
                raise RuntimeError(f"无法切换 Isaac 物理后端到 {name}")

    def _configure_newton_usd(self) -> None:
        """使用本机 6.2 公开接口配置 MuJoCo/XPBD/VBD 的 USD 后端."""
        from isaacsim.core.simulation_manager import PhysicsScene, SimulationManager
        from isaacsim.physics.newton import (
            MuJoCoSolverConfig,
            NewtonConfig,
            VBDSolverConfig,
            XPBDSolverConfig,
            configure_newton,
            switch_newton_solver,
        )

        self._select_physics_engine("newton")
        solver = self.config.newton_solver
        solver_options = dict(self.config.newton_solver_options)
        solver_options.pop("solver_type", None)
        if solver in {"xpbd", "vbd"}:
            solver_options.setdefault("iterations", self.config.newton_iterations)
        solver_cls = {
            "mujoco": MuJoCoSolverConfig,
            "xpbd": XPBDSolverConfig,
            "vbd": VBDSolverConfig,
        }[solver]
        config_options = dict(self.config.newton_config_options)
        config_options.update(
            num_substeps=self.config.newton_substeps,
            physics_frequency=1.0 / self.config.physics_dt,
            use_cuda_graph=self.config.newton_use_cuda_graph,
            time_step_app=False,
            solver_cfg=solver_cls(**solver_options),
        )
        self._newton_runtime_config = NewtonConfig(**config_options)
        configure_newton(self._newton_runtime_config)
        paths = PhysicsScene.get_physics_scene_paths() or ["/World/PhysicsScene"]
        for path in paths:
            scene = PhysicsScene(path)
            try:
                switched = switch_newton_solver(solver, scene.prim)
            except Exception as exc:
                raise RuntimeError(
                    f"安装的 Isaac Newton USD schemas 无法启用 {solver};"
                    "请检查该版本功能支持,或使用 newton_execution='direct' 和应用 stepper"
                ) from exc
            if not switched:
                raise RuntimeError(f"当前 Isaac Newton 后端不支持求解器: {solver}")
            scene.set_dt(self.config.physics_dt)
            scene.set_max_solver_iterations(self.config.newton_iterations)
            if solver == "mujoco":
                from isaacsim.core.simulation_manager.impl.mjc_scene import NewtonMjcScene

                # Newton parser 读通用 Hz,SimulationManager.step 则通过 MJC wrapper 读秒.
                # 只写前者会让实际积分静默保留 MuJoCo 默认的 0.002 s.
                mjc_scene = NewtonMjcScene(scene.prim)
                mjc_scene.set_dt(self.config.physics_dt)
                mjc_scene.set_iterations(self.config.newton_iterations)
        SimulationManager.set_default_physics_scene(paths[0])

    @staticmethod
    def _stop_timeline() -> None:
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            # 无条件覆盖延迟的 Play 命令;is_stopped 只描述已提交状态,不足以检查 pending Play.
            timeline.stop()
            commit = getattr(timeline, "commit", None)
            if callable(commit):
                commit()
        except ImportError:
            pass

    def destroy_world(self) -> None:
        """销毁仿真运行时(App/World).

        Job ``close`` 的 finally 会调到这里;先清 World 单例再 close App,
        否则 Kit 退出时可能残留 PhysX / 渲染线程.

        Returns:
            None.
        """
        active_error = sys.exc_info()[1]
        exit_code = 1 if active_error is not None else 0
        if active_error is not None:
            # SimulationApp 默认 fast_shutdown 会 os._exit,先记录 traceback 并保留失败退出码.
            self.log.error(
                "simulation_failed_before_cleanup", error=str(active_error), exc_info=True
            )
        self._direct_playing = False
        self._newton_stepper = None
        self._newton_runtime_config = None
        if self.world is not None:  # 有 World 才需要清实例
            try:
                self.world.clear_instance()  # 释放 World 单例与物理场景
            except Exception:  # noqa: BLE001  # 已部分销毁时忽略
                pass
            self.world = None  # 断开引用,避免 close 后再 step
        if self.app is not None:  # App 仍在则关 Kit
            try:
                import omni.kit.app  # noqa: PLC0415

                omni.kit.app.get_app().post_quit(exit_code)
            except Exception:  # noqa: BLE001
                pass
            try:
                try:
                    self.app.close(exit_code=exit_code)
                except TypeError:
                    # 老版 close 无 exit_code;关闭快速退出,让调用方的原异常正常传播.
                    if active_error is not None:
                        config = getattr(self.app, "config", None)
                        if isinstance(config, dict):
                            config["fast_shutdown"] = False
                        try:
                            import carb

                            carb.settings.get_settings().set_bool("/app/fastShutdown", False)
                        except ImportError:
                            pass
                    self.app.close()
            except Exception as exc:  # noqa: BLE001
                self.log.warning("SimulationApp.close 失败", error=str(exc))
            self.app = None  # 标记已关闭,is_running 将返回 False
        self.stage = None

    def play(self) -> None:
        """开始仿真(Play):物理与时间轴同时跑.

        采集循环在 ``start_simulation`` 里调用;Play 后 Kit 会锁定部分编辑.
        World 与 omni.timeline 都写一遍,避免只 Play 了其中一个导致步进不同步.

        Returns:
            None.
        """
        if self.uses_direct_newton:
            # 直接 Newton 独占积分.时间轴必须保持 Stop,否则 app.update 会额外推进 PhysX.
            self._stop_timeline()
            self._direct_playing = True
            return
        if self.world is not None:  # World 未建时跳过,避免 AttributeError
            self.world.play()  # Isaac World 层开始物理
        try:
            import omni.timeline  # noqa: PLC0415  # Kit 时间轴接口,无 GUI 构建可能没有

            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            commit = getattr(timeline, "commit", None)
            if callable(commit):
                # timeline.play 是延迟命令;立即提交后无渲染的第一步也必须能积分.
                commit()
        except Exception:  # noqa: BLE001  # 无头/精简包没有 timeline 时忽略
            pass

    def pause(self) -> None:
        """暂停仿真(Pause);仍可能限制部分编辑,布局请用 stop().

        Pause 冻结物理但保持 Playing 会话;自由挪 prim 应调用 ``stop()``.

        Returns:
            None.
        """
        if self.uses_direct_newton:
            self._direct_playing = False
            self._stop_timeline()
            return
        if self.world is not None:  # World 未建则只尝试 timeline
            self.world.pause()  # 暂停物理步进
        try:
            import omni.timeline  # noqa: PLC0415  # 与 play() 对称,两边都 pause

            timeline = omni.timeline.get_timeline_interface()
            timeline.pause()
            commit = getattr(timeline, "commit", None)
            if callable(commit):
                commit()
        except Exception:  # noqa: BLE001  # timeline 不可用时静默
            pass

    def stop(self) -> None:
        """停止时间轴(Stop),便于在 GUI 中自由编辑场景.

        布局 / ``wait_for_gui_edit`` 必须 Stop:Play/Pause 都会锁住部分 xform 编辑.
        World.stop 失败时降级 pause,再停 timeline.

        Returns:
            None.
        """
        self._direct_playing = False
        if self.uses_direct_newton:
            self._stop_timeline()
            return
        if self.world is not None:  # 有 World 才尝试 stop
            try:
                self.world.stop()  # 理想路径:完全停止物理会话
            except Exception:  # noqa: BLE001  # 部分版本 World 无 stop 或已销毁
                try:
                    self.world.pause()  # 降级:至少停住物理
                except Exception:  # noqa: BLE001  # pause 也失败则只靠 timeline
                    pass
        self._stop_timeline()

    def is_running(self) -> bool:
        """Kit App 进程是否仍在跑(关窗口后为 False).

        ``wait_for_gui_edit`` 用它判断循环是否该退出;App 未启动视为未运行.

        Returns:
            Kit App 仍在跑为 ``True``;未启动或已关闭为 ``False``.
        """
        if self.app is not None and hasattr(self.app, "is_running"):  # 部分 stub App 无此方法
            return bool(self.app.is_running())  # True=窗口还开着 / 无头进程未 close
        return False  # 尚未 init 或 App 无 is_running,一律当未运行

    def is_playing(self) -> bool:
        """时间轴是否处于 Play 状态(采集循环据此判断是否继续步进).

        World 未建或查询抛错时返回 False,避免采集循环空转崩溃.

        Returns:
            时间轴处于 Play 为 ``True``;未建或查询失败为 ``False``.
        """
        if self.uses_direct_newton:
            return self._direct_playing
        if self.world is None:
            if self.physics_engine == "newton" and self.app is not None:
                import omni.timeline

                return bool(omni.timeline.get_timeline_interface().is_playing())
            return False
        try:
            return bool(self.world.is_playing())  # 问 World 当前是否 Playing
        except Exception:  # noqa: BLE001  # World 已销毁或 API 变化
            return False

    def reset(self) -> None:
        """``World.reset()``:激活新加入 stage 的物理体(articulation / rigid body).

        Robot ``_build_articulation`` 之后必须 reset,``SingleManipulator.initialize``
        才会真正生效;Job 在 Play 前调用.

        Returns:
            None.
        """
        self.physics_steps = 0
        if self.uses_direct_newton:
            self.stop()
            return  # Newton 模型的 reset 归拥有模型的 planner/runtime 负责
        if self.physics_engine == "newton" and self.app is not None:
            from isaacsim.core.simulation_manager import SimulationManager
            from isaacsim.physics.newton import acquire_stage, configure_newton

            self.stop()
            configure_newton(self._newton_runtime_config)
            SimulationManager.invalidate_physics()
            SimulationManager.initialize_physics()
            stage = acquire_stage()
            if stage is None or not stage.initialized:
                raise RuntimeError("Newton USD 模型初始化失败,请检查 USD/solver 兼容性和 Kit 日志")
            return
        if self.world is not None:  # 未 init 时 no-op
            self.world.reset()  # 重建 PhysX 视图,新 articulation 从此可控

    def step(self, *, render: bool = True) -> None:
        """推进恰好一个 ``physics_dt``;可选再刷一帧渲染.

        Isaac ``World.step(render=True)`` 会走 ``app.update()``,按 ``rendering_dt``
        连跑多步物理(0.005/0.02 时是 4 步).Go2/G1 策略按物理步 decimation,
        那样会跟丢、身体逐渐趴下并被判摔倒.

        Args:
            render: True 时同步渲染(相机/视口有画面);False 只跑物理更快.

        Returns:
            None.
        """
        if self.uses_direct_newton:
            self._stop_timeline()
            if self._direct_playing:
                if self._newton_stepper is None:
                    raise RuntimeError("direct Newton 模式尚未注册 stepper")
                self._newton_stepper(self.config.physics_dt)
                self.physics_steps += 1
            if render:
                self.update()
            return
        if self.physics_engine == "newton" and self.app is not None:
            from isaacsim.core.simulation_manager import SimulationManager

            if self.is_playing():
                SimulationManager.step(steps=1, update_fabric=bool(render))
                self.physics_steps += 1
            if render:
                self.update()
            return
        if self.world is None:  # 未建 World 则无法步进
            return
        was_playing = self.is_playing()
        try:
            self.world.step(render=False, update_fabric=bool(render))
        except TypeError:
            self.world.step(render=False)
        if was_playing:
            if self._newton_stepper is not None:
                self._newton_stepper(self.config.physics_dt)
            self.physics_steps += 1
        if render:
            render_fn = getattr(self.world, "render", None)
            if callable(render_fn):
                render_fn()
            elif self.app is not None:
                self.app.update()

    def update(self) -> None:
        """推进一帧 Kit 主循环(GUI 交互依赖此调用).

        与 ``step`` 不同:不跑物理,只泵 UI/Hydra,USD 引用加载、视口刷新靠它.
        Scene/Robot ``_pump`` 会调到这里.

        Returns:
            None.
        """
        if self.app is not None:  # App 未启动则无法泵循环
            if self.uses_direct_newton:
                self._stop_timeline()
            import carb

            settings = carb.settings.get_settings()
            key = "/app/player/playSimulations"
            previous = settings.get(key)
            # 与 Isaac SimulationContext.render 一致:刷新时不再隐式推进物理.
            settings.set_bool(key, False)
            try:
                self.app.update()
            finally:
                settings.set(key, previous if previous is not None else True)

    # ==========> 仿真世界调试/可视化接口 <==========

    def wait_for_gui_edit(self, *, reason: str = "") -> None:
        """Stop + 泵主循环,让 GUI 可编辑;不能用 debug 断点代替本方法.

        debugpy 断点会冻住 Python 主线程,``app.update()`` 停转,视口无法交互。
        编辑完成后在运行该进程的终端按 Enter,或关闭 Isaac 窗口结束等待。
        headless 时直接跳过(没有可交互视口).

        Args:
            reason: 日志里的阶段名,如 ``after_place_robot``,便于对照流水线.

        Returns:
            None.
        """
        if self.app is None:  # 世界没起来,无从等待
            return
        if self.config.headless:  # 无头模式没有 GUI 可编
            self.log.info("gui_inspect_skipped_headless", reason=reason)  # 记录跳过,流水线继续
            return
        self.stop()  # 必须 Stop,否则视口里拖 prim 会被时间轴锁住
        # 提示操作员:终端 Enter 继续,关窗口退出
        self.log.info(
            "gui_inspect_begin",
            reason=reason or "manual",  # 没传 reason 就标 manual
            tip="可在 Isaac GUI 中编辑;终端按 Enter 继续流水线,或关闭窗口退出",
        )
        while self.is_running():  # 关窗口 → is_running False → 结束等待
            # 防止误点 Play 后无法改场景
            try:
                import omni.timeline  # noqa: PLC0415  # 每帧检查时间轴,挡住误点 Play

                tl = omni.timeline.get_timeline_interface()  # 当前时间轴
                if tl.is_playing():  # 用户在 GUI 点了 Play
                    tl.stop()  # 立刻停回可编辑状态
            except Exception:  # noqa: BLE001  # 无 timeline 时只泵 update
                pass
            self.update()  # 泵一帧,视口才能响应鼠标
            try:
                # select 超时 0:非阻塞看 stdin 是否有一行
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()  # 吃掉 Enter,避免残留给后续 input
                    # 记下是按 Enter 离开,不是关窗口
                    self.log.info("gui_inspect_continue", reason=reason or "manual")
                    break  # 用户确认继续流水线
            except Exception:  # noqa: BLE001  # 无 stdin(被重定向)时忽略,靠关窗口退出
                pass
        self.stop()  # 离开等待前再 Stop,保证后续代码从静止时间轴开始

    def show_cameras(self, prim_paths: list[str]) -> None:
        """在默认透视视口显示相机 gizmo,并把视角框到这些 prim.

        Args:
            prim_paths: 要框进视口的 prim 路径(相机/桌子/机械臂均可).

        Returns:
            None.
        """
        if self.config.headless:  # 无头没有活动视口
            return
        self._enable_camera_gizmos()  # 先打开 gizmo,否则框过去也看不见相机锥
        paths = [p for p in prim_paths if p]  # 丢掉空字符串,避免 frame API 报错
        if not paths:  # 调用方没给任何有效路径
            return
        try:
            # App 起来后才能 import 视口工具
            from omni.kit.viewport.utility import frame_viewport_prims, get_active_viewport

            viewport = get_active_viewport()  # 当前透视视口
            if viewport is not None:  # 启动瞬间可能还没有活动视口
                frame_viewport_prims(viewport_api=viewport, prims=paths)  # 相机看向这些 prim
        except Exception as exc:  # noqa: BLE001  # 视口工具在部分构建缺失
            self.log.warning("frame_cameras_failed", error=str(exc), prims=paths)  # 框失败不阻断
        self.update()  # 泵一帧让 framing 立刻画出来

    def open_camera_viewports(self, cameras: dict[str, str]) -> None:
        """为每路相机开独立 GUI viewport,顶替默认透视窗(headless 时跳过).

        三路视口同时渲染时 RTX 常丢 ``LdrColorSDBuff``,主透视和 overhead 会黑屏.
        先停掉默认 Viewport 的 Hydra,再让相机窗占它的停靠位.

        Args:
            cameras: ``{相机名: 实际 Camera prim 路径}``.

        Returns:
            None.
        """
        if self.config.headless:  # 无头没有窗口系统
            return
        if not cameras:  # 没有要打开的相机
            return
        try:  # 视口工具须在 App 启动后 import
            from omni.kit.viewport.utility import (
                create_viewport_window,  # 新建视口窗口
                get_active_viewport_window,  # 开相机窗之前先记住主透视视口
                get_viewport_from_window_name,  # 按窗口名找回已有视口,避免重复开窗
            )
        except Exception as exc:  # noqa: BLE001  # 精简包可能没有 viewport.utility
            self.log.warning("open_camera_viewports_import_failed", error=str(exc))
            return  # 开不了预览窗不阻断采集

        main_vp = get_active_viewport_window()  # 新建前的活动窗=主透视
        self._disable_viewport_hydra(main_vp)  # 先停透视 Hydra,窗口先留着当停靠锚点
        preview_windows: list[Any] = []  # 能拿到的 Window 实例,后面停靠
        window_names: list[str] = []  # 标题,复用已有窗时用 Workspace 再查一次
        # 左侧工具栏约 48px;停靠前也不贴 (0,0),避免闪一帧盖住 Play
        x = 80
        for name, cam_path in cameras.items():  # 一路相机一个窗口
            window_name = f"Camera: {name}"  # 窗口标题,方便和配置里的 name 对上
            try:
                viewport = get_viewport_from_window_name(window_name)  # 已存在则复用
                if viewport is not None:  # 上次采集留下的同名窗口
                    viewport.camera_path = cam_path  # 只改绑定的 Camera prim
                    window_names.append(window_name)  # 供停靠时按标题找回
                else:
                    window = create_viewport_window(
                        window_name,  # 新窗口标题
                        width=640,  # 与采集分辨率同宽,避免拉伸后看起来像黑图
                        height=480,
                        position_x=x,  # 先错开,停靠失败时也不盖工具栏
                        position_y=40,  # 略低于菜单栏
                        camera_path=cam_path,  # 一开始就绑到该 Camera
                    )
                    if window is None:  # Kit 拒绝创建(数量上限等)
                        self.log.warning("创建相机视口失败", name=name)  # 单路失败继续
                        continue  # 试下一路相机
                    preview_windows.append(window)  # 新建实例,停靠时优先用
                    window_names.append(window_name)  # 标题与 Workspace 查找对齐
                x += 650  # 下一扇浮窗往右挪,略大于宽度 640 留缝
                # 复用或新建都记一笔,方便对照 prim 路径
                self.log.debug("打开相机视口", name=name, camera_path=cam_path, window=window_name)
            except Exception as exc:  # noqa: BLE001  # 单路失败不影响其它相机
                self.log.warning("打开相机视口失败", name=name, error=str(exc))
        self._layout_camera_viewports(preview_windows, window_names, main_vp)  # 占 Viewport 槽
        self._rebind_camera_viewports(cameras)  # 停靠后 Kit 可能把相机打回透视,再绑一次

    # ==========> 仿真世界调试/可视化 辅助函数 <==========

    def _disable_viewport_hydra(self, window: Any | None) -> None:
        """停掉指定 Viewport 的 Hydra,释放 ``LdrColor`` 渲染槽.

        Args:
            window: 视口窗口实例;None 则跳过(启动瞬间可能还没有活动窗).

        Returns:
            None.
        """
        if window is None:  # 主透视尚未创建,无从停 Hydra
            return
        try:
            api = getattr(window, "viewport_api", None)  # 有的 Window 把 API 挂在属性上
            if api is not None:  # 拿到渲染接口才改 updates_enabled
                api.updates_enabled = False  # 停 LdrColor 占用,给相机窗腾槽
        except Exception as exc:  # noqa: BLE001  # 精简包 Window 形状不同
            self.log.warning("disable_viewport_hydra_failed", error=str(exc))  # 不阻断开相机窗

    def _hide_default_viewport(self, main: Any | None) -> None:
        """隐藏默认透视 Viewport.

        不 ``destroy``:Isaac 内部仍可能 ``get_active_viewport()``.停渲染 + 藏窗即可
        让相机预览窗出画,Play 栏会跟到相机 Viewport 上.

        Args:
            main: 开相机窗之前的活动透视窗;None 则跳过.

        Returns:
            None.
        """
        if main is None:  # 没有锚点窗,无从隐藏
            return
        self._disable_viewport_hydra(main)  # 先停 Hydra,避免与相机窗抢 LdrColor
        try:
            main.visible = False  # 藏窗但不 destroy,Kit 仍可 get_active_viewport
        except Exception as exc:  # noqa: BLE001  # 部分 Window 无 visible
            self.log.warning("hide_default_viewport_failed", error=str(exc))
            return  # 藏失败就别假装成功打 info
        self.log.debug("默认透视窗已隐藏")  # 确认透视窗已让出停靠位

    def _rebind_camera_viewports(self, cameras: dict[str, str]) -> None:
        """停靠后再写一次 ``camera_path``,避免窗口退回 ``/OmniverseKit_Persp``.

        退回透视时画面是原点处黑场+坐标轴,看起来像 overhead 坏了.

        Args:
            cameras: ``{相机名: 实际 Camera prim 路径}``,与开窗时同一份.

        Returns:
            None.
        """
        try:
            from omni.kit.viewport.utility import get_viewport_from_window_name  # 按标题找回
        except Exception as exc:  # noqa: BLE001  # 精简包可能没有此工具
            self.log.warning("rebind_camera_viewports_import_failed", error=str(exc))
            return  # 绑不上就保持停靠后的默认相机
        for name, cam_path in cameras.items():  # 与开窗循环同一套名字
            window_name = f"Camera: {name}"  # 必须与 create 时的标题一致
            try:
                viewport = get_viewport_from_window_name(window_name)  # 停靠后实例可能换了
                if viewport is None:  # 窗已关或标题被改
                    continue
                viewport.camera_path = cam_path  # 写回真实 Camera,别停在透视
                viewport.updates_enabled = True  # 停靠/藏透视窗时 Kit 可能把预览也停了
                self.log.debug(
                    "重新绑定相机视口", name=name, camera_path=cam_path, window=window_name
                )
            except Exception as exc:  # noqa: BLE001  # 单路失败继续绑其它
                self.log.warning("重新绑定相机视口失败", name=name, error=str(exc))
        self.update()  # 让新绑定的相机出一帧

    def _layout_camera_viewports(
        self,
        preview_windows: list[Any],
        window_names: list[str],
        main_vp: Any | None = None,
    ) -> None:
        """让相机预览窗占原 Viewport 停靠位,上下分栏.

        第一扇 ``SAME`` 进默认 Viewport 槽,再把透视窗藏掉;其余叠在第一扇下面.
        停靠失败则保持已错开的浮窗(创建时已避开左侧工具栏).

        Args:
            preview_windows: 新建时拿到的 ``ViewportWindow`` 实例.
            window_names: 窗口标题;复用已有窗时靠这个从 Workspace 找回.
            main_vp: 开相机窗之前的活动视口;None 则按标题 ``Viewport`` 查找.

        Returns:
            None.
        """
        if not preview_windows and not window_names:  # 一路都没开成
            return
        try:
            import omni.ui as ui  # noqa: PLC0415  # dock_in / Workspace
        except Exception as exc:  # noqa: BLE001  # 无 GUI 构建没有 omni.ui
            self.log.warning("布局相机视口失败", error=str(exc))
            return  # 保持创建时已错开的浮窗

        try:
            self.update()  # 新窗进 Workspace 后再按标题查找
            resolved: list[Any] = []  # 按标题去重后的 Window,避免代理重复
            for title in window_names:  # 按标题取 Window,避免新建实例与 Workspace 代理重复
                win = ui.Workspace.get_window(title)  # Workspace 里才是可 dock 的对象
                if win is not None:  # 标题对得上才进列表
                    resolved.append(win)
            if not resolved:  # 查找失败则退回 create 返回的实例
                resolved = list(preview_windows)
            if not resolved:  # create 也没留下窗口
                return
            main = ui.Workspace.get_window("Viewport") or main_vp  # 优先按标题找原透视槽
            # 读标题/名字,避免把相机窗误当成原 Viewport 槽
            main_title = getattr(main, "title", None) or getattr(main, "name", None)
            if main is None or main_title in set(window_names):  # 找不到原 Viewport 就只叠相机窗
                # 找不到原透视槽:相机窗彼此上下停靠,不碰未知窗口
                self.log.warning("layout_camera_viewports_no_main", main_title=main_title)
                n = len(resolved)  # 分栏比例按剩余窗数算
                for i in range(1, n):  # 从第二扇起叠到上一扇下面
                    resolved[i].dock_in(
                        resolved[i - 1], ui.DockPosition.BOTTOM, 1.0 / (n - i + 1)
                    )  # 剩余窗均分高度
                self.update()  # 泵一帧让叠放生效
                return  # 没有 Viewport 槽可替换,到此结束
            # 第一扇成为 Viewport 槽里的 tab,再藏掉透视窗,相机窗接管整格
            resolved[0].dock_in(main, ui.DockPosition.SAME, 1.0)  # SAME=占原透视停靠位
            try:
                resolved[0].focus()  # 让相机窗成为活动视口,Play 栏跟过来
            except Exception:  # noqa: BLE001  # 部分 Window 没有 focus
                pass
            self._hide_default_viewport(main)  # 再藏一次,停靠过程可能又把透视窗带出来
            n = len(resolved)  # 第一扇已占槽
            if n == 2:  # 两路相机左右并排,方便同时看
                resolved[1].dock_in(resolved[0], ui.DockPosition.RIGHT, 0.5)
            elif n > 2:
                # 两列网格:0/2/4... 左,1/3/5... 右,D455 组 + follow 能同时看见
                resolved[1].dock_in(resolved[0], ui.DockPosition.RIGHT, 0.5)
                left = [resolved[i] for i in range(0, n, 2)]
                right = [resolved[i] for i in range(1, n, 2)]
                for col in (left, right):
                    for i in range(1, len(col)):
                        col[i].dock_in(col[i - 1], ui.DockPosition.BOTTOM, 1.0 / (len(col) - i + 1))
            self.update()  # 泵一帧让 dock 立刻生效
            self.log.debug("相机视口已停靠", n=n, replaced="Viewport")  # 已顶替默认透视槽
        except Exception as exc:  # noqa: BLE001  # 停靠失败仍可拖浮窗,不阻断采集
            self.log.warning("停靠相机视口失败", error=str(exc))

    def _make_background_static(self) -> None:
        """给背景补静态碰撞,并锁刚体/关节,避免 Play 后整座厂房下落.

        由 ``[sim_world].background_gravity=false`` 触发.SimReady 工厂常常只有
        可视 Mesh 和动态 RigidBody、没有 CollisionAPI;omap 扫不到碰撞,物体
        也会穿过地面掉下去.这里只处理背景子树,不碰 ``/World/scene`` 与机器人.

        Returns:
            None. stage 未就绪或背景 prim 无效时直接返回.
        """
        if self.stage is None:  # init_world 尚未拿到 stage,无法改物理 API
            return
        from pxr import Usd, UsdPhysics

        from real2sim2real.utils import (
            ensure_static_collider,
            normalize_prim_path,
            set_subtree_puppet,
        )

        bg_prim = (
            self.config.background_prim_path or WORLD_BACKGROUND_PRIM
        ).strip() or WORLD_BACKGROUND_PRIM
        root = self.stage.GetPrimAtPath(normalize_prim_path(bg_prim))
        if root is None or not root.IsValid():  # 空 stage 或路径写错
            return
        n_col = ensure_static_collider(self.stage, bg_prim)
        n_bodies, n_joints, n_art = set_subtree_puppet(self.stage, bg_prim, enabled=True)
        n_rb = 0
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            # 整座厂房若是一个刚体,凸包会把室内扫成实心;去掉刚体后 Mesh 碰撞变静态
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            n_rb += 1
        self.log.info(
            "背景静态化",
            prim_path=bg_prim,
            colliders=n_col,
            bodies=n_bodies,
            rigid_bodies_removed=n_rb,
            joints=n_joints,
            articulations=n_art,
        )

    def _wait_stage_loaded(self, *, max_frames: int = 600) -> None:
        """泵 Kit 直到背景引用/payload 加载完.

        Args:
            max_frames: 最多等待帧数.

        Returns:
            None.
        """
        try:
            import isaacsim.core.experimental.utils.stage as stage_utils
        except Exception:  # noqa: BLE001
            for _ in range(8):
                self.update()
            return
        for _ in range(max(1, int(max_frames))):
            if not stage_utils.is_stage_loading():
                break
            self.update()

    def _open_stage_usd(self) -> None:
        """把 ``[sim_world].stage_usd_path`` 整份 open 成当前 stage;未配置则 no-op.

        成品场景(分拣台等)把机器人 / 传送带 / 柔性体和 ``/PhysicsScene`` 都写在
        同一份 USD 里,脚本按绝对 prim 路径寻址.``add_reference_to_stage`` 只会带进
        默认 prim 子树,既丢掉根级 ``/PhysicsScene`` 的 GPU 设置,也会改变 prim 路径,
        因此这类场景必须 open 而不是 reference.

        Returns:
            None.
        """
        raw = (self.config.stage_usd_path or "").strip()
        if not raw:  # 未配置整份 stage,走背景 reference 通路
            return
        path = resolve_usd_path(raw)  # 相对路径 → 仓库内绝对路径
        import isaacsim.core.experimental.utils.stage as stage_utils  # App 启动后才能 import

        opened, _ = stage_utils.open_stage(path)  # 顶替当前空 stage
        if not opened:
            raise RuntimeError(f"打开环境 USD 失败: {path}")
        for _ in range(600):  # 等资产流式加载完,否则 World 建不出物理体
            if not stage_utils.is_stage_loading():
                break
            self.update()  # 泵 Kit 主循环推进加载
        self.log.info("打开环境 USD 成功", usd=path)

    def _enable_camera_gizmos(self, settings: Any | None = None) -> None:
        """打开默认视口的相机 gizmo,使 scene/robot cameras 可见.

        Args:
            settings: 已拿到的 carb settings;None 则内部再 get_settings().

        Returns:
            None.
        """
        try:
            import carb  # noqa: PLC0415  # 视口显示选项走 carb 持久化键

            settings = settings or carb.settings.get_settings()  # 复用调用方的 settings
            settings.set("/app/viewport/show/camera", True)  # 显示单个 camera 图标
            settings.set("/app/viewport/show/cameras", True)  # 显示全部 cameras(复数键,版本差异)
            # 读出当前显示位,后面只或上 Camera gizmo,不冲掉其它
            opts = int(settings.get_as_int("/persistent/app/viewport/displayOptions") or 0)
            # bit 5 = Camera gizmos;或进去以免冲掉用户其它显示位
            settings.set_int("/persistent/app/viewport/displayOptions", opts | (1 << 5))
            # 图标/相机锥不随视距缩小,布局时更容易对准
            settings.set_bool("/persistent/app/viewport/gizmo/constantScaleEnabled", True)
            settings.set_bool("/persistent/app/viewport/gizmo/constantScaleCamera", True)
            settings.set_float("/persistent/app/viewport/gizmo/constantScale", 10.0)  # 相机图标放大
        except Exception as exc:  # noqa: BLE001  # 无头或旧 Kit 没有这些键
            self.log.warning("enable_camera_gizmos_failed", error=str(exc))  # gizmo 失败不影响仿真

    @property
    def log(self) -> Any:
        """返回绑定组件名的日志器。"""
        return get_logger(f"{type(self).__module__}.{type(self).__qualname__}").bind(
            component=type(self).__qualname__
        )
