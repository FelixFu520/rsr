# AGENTS.md — Real2Sim2Real

给 Agent / 贡献者的工作手册。人类入门见 [`README.md`](README.md)。改代码前先对齐本文约定。

## 项目是什么

可插拔的 **Real → Sim → Real** 编排框架（数字孪生、仿真采集、策略回流）。  
本仓库提供：**抽象接口 + 配置/Registry + stub 通路**；重型仿真后端通过 entry points 外挂。

## 仓库地图

| 路径 | 何时改 |
|------|--------|
| `real2sim2real/{simworld,scene,robot,planner,job}/` | ABC（`base.py`）、模块工厂、配置、本包 stub |
| `real2sim2real/core/` | 配置、Registry、CLI、日志、运行时、`default_settings.toml` |
| `real2sim2real/utils/` | 辅助函数（isaacsim_utils、pose、recorder、validate） |
| `pyproject.toml` | 依赖、uv、entry points、脚本 `rsr` |

项目用 **uv** 管锁文件与 wheel；**唯一运行时是 Isaac Sim `python.sh`**。不要 `uv sync` 出 `.venv` 当仿真环境。

## 架构契约（必守）

### 1. 模块组织

每个能力域（simworld / scene / robot / planner / job）保持：

- `{domain}/base.py` — ABC + `Auto*.from_config()` 工厂：`peek_backend` + 实现类 `config_cls` → Registry
- `{domain}/config.py` — section 级 pydantic（继承 `BaseConfig`，含公共字段）
- 各 stub 文件（如 `job_stub.py`）— 实现类通过 `config_cls` 指定 `config.py` 中的配置；在 `pyproject.toml` entry points 注册。新增插件可将实现类与专用配置放在同文件

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

调试子命令时使用 `./app/python.sh -m real2sim2real.core.cli`（`main()` 内加载插件）。

## 环境与常用命令

```bash
uv lock
./app/python.sh -m pip install -e ".[dev]"
./app/python.sh -m ruff check real2sim2real
./app/python.sh -m ruff format real2sim2real
./app/python.sh -m pytest -q
./app/python.sh -m real2sim2real.core.cli list-backends
./app/python.sh -m real2sim2real.core.cli collect run
```

| 用途 | 做法 |
|------|------|
| 开发 | `./app/python.sh -m pip install -e ".[dev]"` |
| 发布 | `uv build` → 对方用 `isaacsim/python.sh -m pip install *.whl` |

改依赖只动 `pyproject.toml`，再 `uv lock`，然后用 Isaac `pip install -e ".[dev]"`。

## 编码约定

- Python 3.12（与 Isaac Sim 6.0.1 一致）；类型标注；公共 API 写简短 docstring
- Lint：`ruff`（`E,F,I,UP,B`），行宽 100
- 日志：`get_logger` / `LoggerMixin`（structlog），勿散落 `print`
- 插件加载失败：告警并继续，勿拖垮 CLI / Registry

## 改动时注意

- **改 `{domain}/base.py`**：同步 stub、相关 Config，并考虑外部插件兼容
- **改 entry point 组名 / section 名**：同步 `pyproject.toml`、`registry.py`、文档
- **加配置字段**：更新对应 `*Config`、`default_settings.toml`及适用的配置示例
- **不要**提交密钥、`.secrets.toml`、大体积生成物；遵守 `.gitignore`
- **不要**在未要求时创建 git commit / push

## 快速自检清单

- [ ] `./app/python.sh -m ruff check real2sim2real` 通过
- [ ] 新后端已在 `pyproject.toml` entry points 注册，`./app/python.sh -m real2sim2real.core.cli list-backends` 能看到
- [ ] 配置字段已写入对应 `*Config`、`default_settings.toml`及适用的配置示例
- [ ] Job 仍走固定五段生命周期（含 `close`）

