# AGENTS.md — Real2Sim2Real

给 Agent / 贡献者的工作手册。人类入门见 [`README.md`](README.md)。改代码前先对齐本文约定。

## 项目是什么

可插拔的 **Real → Sim → Real** 编排框架（数字孪生、仿真采集、策略回流）。  
本仓库提供：**抽象接口 + 配置/Registry + stub 通路**；重型仿真后端通过 entry points 外挂。

## 仓库地图

| 路径 | 何时改 |
|------|--------|
| `real2sim2real/{simworld,scene,robot,planner,job}/` | ABC（`base.py`）、模块工厂、本包 stub、每个任务一个实现文件 |
| `real2sim2real/core/` | 配置、Registry、CLI、日志、运行时、`default_settings.toml` |
| `real2sim2real/utils/` | 共用辅助（asset_utils、cameras、kit_process、lerobot_v3）+ 每个任务一个 `<task>_utils.py` |
| `configs/<task>/` | 任务配置|
| `scripts/<task>/` | 资产准备、采集、独立校验；`scripts/rsr` 是 CLI 包装 |
| `docs/<task>.md` | 任务说明、运行方法、问题与解决方案、当前结果 |
| `tests/` | 无 Kit 单元测试 |
| `pyproject.toml` | 依赖、uv、entry points、脚本 `rsr` |

项目用 **uv** 管锁文件与 wheel；**唯一运行时是 Isaac Sim `python.sh`**。不要 `uv sync` 出 `.venv` 当仿真环境。

## 架构契约（必守）

### 1. 模块组织

每个能力域（simworld / scene / robot / planner / job）保持：

- `{domain}/base.py` — ABC + `Auto*.from_config()` 工厂：`peek_backend` + 实现类 `config_cls` → Registry
- `{domain}/{domain}_stub.py`（simworld 为 `stub.py`）— 无依赖的 stub 后端
- `{domain}/{domain}_<task>.py` — 该任务在本能力域的全部实现类及其 pydantic 配置（继承 `BaseConfig`），
  实现类通过 `config_cls` 指定配置；在 `pyproject.toml` entry points 注册

### 1.1 任务布局

每个任务在 scene / robot / planner / job / utils 中**各只有一个文件**，文件名后缀即任务名：

| 任务 | 后缀 | 资产目录 | 后端名（scene / robot / planner / job） |
|------|------|----------|------------------------------------------|
| 物理求解器验证（含包裹矩阵／螺纹装配） | `physic_solver` | `assets/PhysicSolver/` | `pickplace_cloth`、`pickplace_rigid`、`newton_cloth_franka`、`pickplace_package`、`pickplace_carton`、`screw_assembly` / `screw_spindle`、`franka_carton_gripper`、`franka_newton`、`franka_physx`、`franka_package_pads` / `pickplace_cloth`、`newton_cloth_franka`、`pickplace_package`、`pickplace_carton`、`screw_assembly` / `pickplace_cloth`、`pickplace_carton`、`screw_assembly` |
| 分拣包裹 | `sorting` | `assets/SortingPackage/` | `sorting_package` / `pro_m6_newton` / `sorting_package`、`sorting_stream` / `sorting_package` |
| 机器狗巡检 | `fire_inspection` | `assets/FireInspection/` | `fire_warehouse` / `x30_kinematic` / `fire_patrol` / `fire_inspection` |

- 同一任务的多个后端、运行时和辅助类放在同一文件中，按依赖顺序分节（节首注释说明内容）。
- 新任务依次新增 `{domain}/{domain}_<task>.py`、`utils/<task>_utils.py`、`configs/<task>/`、
  `scripts/<task>/`、`docs/<task>.md` 和资产目录 `assets/<Task>/`；后端名保持配置里的小写字符串，
  改名需同步配置与文档。
- 资产：每个任务只引用自己资产目录里的文件（含 USD 依赖、MDL 与贴图），可以单独打包发给别人；
  任务之间共用的资产各自复制一份。不属于任何任务的资产放 `assets/bak/`，任务配置和运行代码不引用它
  （资产准备脚本可以从这里复制）。
- 任务之间可以继承（分拣复用物理求解器任务的 `PackageScene` / `PackageRuntime` / `PhysicSolverJob`），
  不要复制代码。
- 包内模块顶层不导入 Kit / pxr / Newton / Warp；Warp kernel 和依赖 Newton 的子类在首次使用时定义
  （见 `utils/physic_solver_utils.moving_frame_vbd_class`），保证 `list-backends` 和单元测试不启动 Kit。
- 输出：`outputs/<task>/` 只放通过验收的结果；调试、对照和失败轨迹放 `outputs/debug/<task>/`。

模块职责（分离，由 Job 编排）：

| 模块 | 职责 |
|------|------|
| **simworld** | 仿真运行时（`init_world` / play / step / close）+ 环境定义（加载环境 USD） |
| **scene** | 在已有环境中添加额外资产（桌面物体、地形道具等） |
| **robot** | 机器人定义与初始化（单臂 / 复合 / 人形均走此模块） |
| **job** | 装配组件并按流水线调用上述 API；不存位姿，读 `scene` / `robot` 的 `poses` 与 `poses_realtime` |

能力分层（**通用 ABC 不要塞子任务字段**；新场景/机型按需挂能力子类）：

| 通用 ABC | 能力子类（按需继承） | 不要放进通用 ABC |
|----------|----------------------|------------------|
| `BaseScene` | （桌面语义在 `TabletopScene`：`table_pose` 读 `poses[table]` / `grasp_target`） | 桌子语义不要放进通用 ABC；loco 场景直接继承 `BaseScene` |
| `BaseRobot` | 机型专属能力由后端按需扩展 | 通用接口只保留位姿、传感器、状态缓存及 `init_robot` / `arrange` / `adjust` |
| `BasePlanner` | `BaseManipulationPlanner`（`pick_and_place` / `begin_episode`） | 抓放；全身 loco / 导航继承 `BasePlanner` |

### 2. Job 流水线顺序

`BaseJob.run()` 固定：

```
init_world → arrange_scene → place_robot → start_simulation
finally: close
```

`BaseJob.__init__(config=None)` 先加载任务配置，再调用四个 `Auto*.from_config()` 创建
`sim_world`、`scene`、`robot`、`planner`。组件构造时不启动仿真资源，后续阶段复用这些实例。
子类构造函数通过 `super().__init__(config=config)` 传入配置。

各阶段顺序固定：

| 阶段 | 执行顺序 |
|------|----------|
| `init_world` | `before_init_world_hook` → `sim_world.init_world` → GUI 检查 → `after_init_world_hook` |
| `arrange_scene` | `before_arrange_scene_hook` → `build_occ` → `scene.init_scene` → `scene.arrange` → `adjust_scene` → GUI 检查 → `after_arrange_scene_hook` |
| `place_robot` | `before_place_robot_hook` → `robot.init_robot` → `robot.arrange` → `adjust_robot` → `planner.init_planner` → GUI 检查 → `after_place_robot_hook` |
| `start_simulation` | `before_start_simulation_hook` → `setup_gui_hook` → GUI 检查 → `simulate` → `after_start_simulation_hook` |

`build_occ` 是空的建图扩展接口；具体建图和采样逻辑由子类实现。
`adjust_scene` / `adjust_robot` 默认分别调用 `scene.adjust()` / `robot.adjust()`。
修改机器人初始摆放配置的操作放到 `before_place_robot_hook`。
GUI 检查调用 `maybe_gui_inspect`，仅当 `inspect_gui` 开启且后端支持时执行。

`simulate` 是空的任务执行接口，具体循环由子类实现。基类不包含 recorder、episode、
重试、校验、相机采集、统计或任务目标等逻辑。所有阶段的 after 钩子只在正常完成时执行；
异常向上传播，由 `run` 的 `finally: close` 清理资源。stub 复用整个基础流程。

通用任务状态放在 `state`；不要在通用接口里加入桌面、导航等任务专用字段。

### 3. 选型路径

```
settings[section].backend
  → AutoX.from_config()
  → ensure_*_backends()
  → *_REGISTRY.get(name)(...)
```

组件默认从各自 section 的 `backend` 选型。`AutoJob` 只依赖 `BaseConfig`：用 `peek_backend` + 各 Job 的 `config_cls`（如 `TabletopJob.config_cls = TabletopJobConfig`）选型；组件装配由各 Job 后端自行完成。

### 4. 配置优先级

后者覆盖前者：

1. `core/default_settings.toml`
2. 项目 `inherit` / `dynaconf_include` 父配置（先父后子），再加载 `RSR_SETTINGS_FILE` 或 cwd `settings.toml`（CLI `-c` 会设置前者）
3. 同目录 `.secrets.toml`
4. 环境变量 `RSR_*`（嵌套 `__`）

`BaseConfig`：含 `backend` 字段、`extra="forbid"`；用 `from_settings(section)` / `peek_backend(section)`。配置类由各后端 `config_cls` 指定，不经 Registry。

## 扩展后端（推荐做法）

**优先独立包 + entry points**，避免把 Isaac 等重依赖塞进本包。

```toml
# 实现包 pyproject.toml
[project.entry-points."real2sim2real.planner.backends"]
curobo = "my_pkg.planner.curobo:CuroboPlanner"

[project.entry-points."real2sim2real.cli.commands"]
my_cmd = "my_pkg.cli:register"   # def register(main_app: typer.Typer) -> None
```

本包内验证通路用 stub；新后端名称用配置里的小写字符串（如 `"stub"`、`"isaac"`）。

Entry point 组名以 `pyproject.toml` / `core/registry.py` 为准，勿自创别名。

## CLI 约定

- 入口：`rsr` → `real2sim2real.core.cli:main`（须由 Isaac `python.sh` 拉起）
- 根命令保持薄：`version` / `list-backends` / `settings`
- 业务：`real2sim2real.cli.commands` 插件，`register(app)` 挂子命令
- stub 示例：`./app/python.sh -m real2sim2real.core.cli collect run`
- 指定配置：`./app/python.sh -m real2sim2real.core.cli collect run -c settings.toml`
- 任务入口：
  - 物理求解器验证：`./app/python.sh scripts/physic_solver/collect_physic_solver_cases.py`
  - 分拣包裹：`./scripts/sorting/run_sorting_showcase.sh outputs/sorting`（内部调用 `rsr collect render-sorting`）
  - 机器狗巡检：`./scripts/rsr collect run -c configs/fire_inspection/x30_physx_warehouse.toml`

调试子命令时使用 `./app/python.sh -m real2sim2real.core.cli`（`main()` 内加载插件）。

## 环境与常用命令

```bash
uv lock
./app/python.sh -m pip install -e ".[dev]"
./app/python.sh -m ruff check real2sim2real scripts
./app/python.sh -m ruff format <改动的文件>   # 不要整包 format：会改写未改动的 base 文件
./app/python.sh -m pytest -q
./app/python.sh -m real2sim2real.core.cli list-backends
./app/python.sh -m real2sim2real.core.cli collect run
```

| 用途 | 做法 |
|------|------|
| 开发 | `./app/python.sh -m pip install -e ".[dev]"` |
| 发布 | `uv build` → 对方用 `isaacsim/python.sh -m pip install *.whl` |

改依赖只动 `pyproject.toml`，再 `uv lock`，然后用 Isaac `pip install -e ".[dev]"`。

## 仿真调试 Skills

项目内 skills 位于 `.agents/skills/`。按当前主要问题选择入口；先读该 skill 的 `SKILL.md` 和相关本地 reference，不需要一次加载全部 skills。

| 主要问题 | 入口 |
|----------|------|
| 吞吐、步耗时、CPU fallback、碰撞内存增长 | [isaac-physics-performance](.agents/skills/isaac-physics-performance/SKILL.md) |
| 穿透、弹飞、接触抖动、碰撞几何/SDF | [isaac-collision-debug](.agents/skills/isaac-collision-debug/SKILL.md) |
| 机器人关节发散、驱动振荡、闭环/mimic 稳定性 | [isaac-articulation-stability](.agents/skills/isaac-articulation-stability/SKILL.md) |
| 夹爪驱动力、闭合速度、夹持不足与具体调参 | [isaac-gripper-tuning](.agents/skills/isaac-gripper-tuning/SKILL.md) |
| 功能组合不支持、版本行为差异、官方已知限制 | [isaac-physics-faq](.agents/skills/isaac-physics-faq/SKILL.md) |
| 迁移任务、新增任务、丰富已有任务（引擎选型、组件接入、资产打包、回归、采集验收） | [rsr-task-development](.agents/skills/rsr-task-development/SKILL.md) |

混合问题先按直接证据选主入口：夹爪参数估算从 gripper 开始，接触几何异常从 collision 开始，驱动/约束异常从 articulation 开始；只有证据涉及其他领域时再读取对应 skill，避免循环加载。

本地 references 是可离线阅读的摘要和工作笔记，**不是完整教程镜像**。具体 API、schema、单位约定及版本限制需查对应安装版本的官方资料或本机实现。无法核实时标明待验证；`latest` 页面不构成当前安装版本的兼容保证。

## 编码约定

- Python 3.12（与 Isaac Sim 6.2 一致）；类型标注；公共 API 写简短 docstring
- Lint：`ruff`（`E,F,I,UP,B`），行宽 100
- 日志：`get_logger` / `LoggerMixin`（structlog），勿散落 `print`
- 插件加载失败：告警并继续，勿拖垮 CLI / Registry

## 改动时注意

- **改 `{domain}/base.py`**：同步 stub、相关 Config，并考虑外部插件兼容
- **改 entry point 组名 / section 名**：同步 `pyproject.toml`、`registry.py`、文档
- **加配置字段**：更新对应 `*Config`、`default_settings.toml`及适用的配置示例
- **移动配置 / 脚本**：`inherit` 相对配置文件所在目录；脚本用 `from scripts.<task>.<name> import ...` 互相引用
- **不要**提交密钥、`.secrets.toml`、大体积生成物；遵守 `.gitignore`
- **不要**在未要求时创建 git commit / push

## 快速自检清单

- [ ] `./app/python.sh -m ruff check real2sim2real scripts` 与 `./app/python.sh -m pytest -q` 通过
- [ ] 新后端已在 `pyproject.toml` entry points 注册，`./app/python.sh -m real2sim2real.core.cli list-backends` 能看到
- [ ] 配置字段已写入对应 `*Config`、`default_settings.toml`及适用的配置示例
- [ ] Job 仍走固定五段生命周期（含 `close`）
- [ ] 任务代码仍是每个能力域一个文件；对应 `docs/<task>.md` 已更新

