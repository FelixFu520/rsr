# Real2Sim2Real

虚实闭环平台：**场景重建与数字孪生 → 仿真数据采集 → 策略训练 → 真机评测回流**。

当前仓库是可插拔的仿真编排框架：抽象基类 + Registry + stub 后端，真实仿真引擎（Isaac Sim）跑在 **唯一的 Python 环境**——Isaac 自带的 `python.sh`。

> Agent / 贡献者：请先读 [`AGENTS.md`](AGENTS.md)（目录约定、扩展后端、配置与 CLI 约定）。

## 环境模型

| 角色 | 用什么 |
|------|--------|
| 运行时 / 开发 / 测试 | **仅** Isaac Sim：`./app/python.sh`（或 `$RSR_ISAACSIM_HOME/python.sh`） |
| 项目元数据 / 锁文件 / 打 wheel | [uv](https://docs.astral.sh/uv/) 读根目录 [`pyproject.toml`](pyproject.toml) |

## 配置 Isaac Sim 路径

使用已安装的 Isaac Sim standalone，将其根目录链接到本仓库：

```bash
cd /path/to/rsr
ln -sfn ~/isaacsim app
```

运行、开发与测试均使用 `./app/python.sh`，无需另建运行时 venv。
stub 流程只使用 Python 包，不启动 Isaac Sim 应用。

## 快速开始（开发）

```bash
# 安装 uv: https://docs.astral.sh/uv/getting-started/installation/
# Isaac Sim 已按上一节装好，且仓库根有 app -> isaacsim

cd /path/to/rsr
uv lock                                    # 生成或更新 uv.lock（不创建运行时 venv）
./app/python.sh -m pip install -e ".[dev]" # editable 装进 Isaac
./app/python.sh -c "import real2sim2real, sys; print(real2sim2real.__file__); print(sys.version)"
```

录制工具依赖 Pillow、PyArrow；视频导出需要系统提供 `ffmpeg`。

常用命令（都走 Isaac）：

```bash
./app/python.sh -m real2sim2real.core.cli --help
```

安装后的 `rsr` 入口与 `python.sh -m real2sim2real.core.cli` 使用同一个 CLI。

## 发布与他人安装

维护者打 wheel（uv，不依赖 Isaac 解释器）：

```bash
uv build
# 产物：dist/real2sim2real-*.whl
```

使用方只需 Isaac Sim，**不要**再建 Pixi / conda / uv venv：

```bash
# 1. 按上文「安装 Isaac Sim 6.0.1」装好 standalone
# 2. 本包
/path/to/isaacsim/python.sh -m pip install real2sim2real-0.1.0-py3-none-any.whl

# 3. 验证内置 stub 流程
/path/to/isaacsim/python.sh -m real2sim2real.core.cli collect run
```

相对路径的 USD 按工作区根解析（源码树、`RSR_REPO_ROOT`，或当前目录上的 `assets/` + `configs/`）。

## CLI 速览

```bash
./app/python.sh -m real2sim2real.core.cli version
./app/python.sh -m real2sim2real.core.cli list-backends                 # 列出已注册后端
./app/python.sh -m real2sim2real.core.cli list-backends -k planner
./app/python.sh -m real2sim2real.core.cli settings                      # 打印合并后的 settings
./app/python.sh -m real2sim2real.core.cli settings -s job
./app/python.sh -m real2sim2real.core.cli collect run                 # 零配置 stub 流水线
```

`collect` 等业务子命令通过 entry points 挂到 `rsr`（见下方「插件」）。

## 架构概览

```
                    +-------------------------------------+
  settings.toml --->|  Dynaconf + pydantic BaseConfig     |
  RSR_* env    --->|  core/config.py                      |
                    +------------------+------------------+
                                       |
         Auto*Factory.from_config      |  Registry.get(backend)
                                       v
  +----------+  +-------+  +-------+  +---------+
  | SimWorld |  | Scene |  | Robot |  | Planner |
  +----+-----+  +-------+  +---+---+  +----+----+
       |                       |           |
       +----------- Job(config) -----------+
                                       |
                              BaseJob.run()
     init_world -> arrange_scene -> place_robot -> start_simulation -> close
```

### 核心层

| 层 | 路径 | 职责 |
|----|------|------|
| 抽象接口 | 各模块 `{domain}/base.py` | `BaseSimWorld` / `BaseScene` / `BaseRobot` / `BasePlanner` / `BaseJob` |
| 模块实现 | `real2sim2real/{simworld,scene,robot,planner,job}/` | ABC + section `config.py`；工厂合并在 `base.py`，后端通过 `config_cls` 指定配置类 |
| 基础设施 | `real2sim2real/core/` | 配置、Registry、CLI、日志、运行时路径 |
| 辅助函数 | `real2sim2real/utils/` | pose、recorder、validate、isaacsim_utils |

### 设计约定

1. **接口与实现分离**：业务只依赖各模块 `{domain}.base`；具体引擎放在插件包或本包 stub。
2. **配置驱动选型**：每个 section 的 `backend = "..."` 决定 Registry 取哪个实现。
3. **工厂统一入口**：`AutoSimWorld` / `AutoScene` / … / `AutoJob.from_config()`。
4. **Job 编排固定顺序**：`init_world` → `arrange_scene` → `place_robot` → `start_simulation`，`finally` 里 `close`。

## 配置系统

查找与覆盖顺序（**后者优先**）：

1. 包内 `real2sim2real/core/default_settings.toml`
2. 项目配置的 `inherit` / `dynaconf_include` 父配置（先父后子），再加载项目 `settings.toml`（cwd）或 `RSR_SETTINGS_FILE` / CLI `-c`
3. 同目录 `.secrets.toml`（可选）
4. 环境变量前缀 `RSR_`（嵌套用双下划线，如 `RSR_JOB__BACKEND=stub`）

常用 section：`logging` / `sim_world` / `scene` / `robot` / `planner` / `job`。

各组件从对应 section 的 `backend` 选型。CLI 可用 `collect run -c settings.toml` 指定配置文件。

## 插件（entry points）

在实现包的 `pyproject.toml` 中声明即可挂载，无需改平台代码：

| Group | 用途 |
|-------|------|
| `real2sim2real.simworld.backends` | SimWorld 后端 |
| `real2sim2real.scene.backends` | Scene 后端 |
| `real2sim2real.robot.backends` | Robot 后端 |
| `real2sim2real.planner.backends` | Planner 后端 |
| `real2sim2real.job.backends` | Job 后端 |
| `real2sim2real.cli.commands` | `rsr` 子命令（`register(app)`） |

本包已注册名为 `stub` 的各后端，用于无仿真依赖的通路验证。

## 目录速查

```text
rsr/
├── pyproject.toml          # 依赖、entry points、工具链
├── app                    # 指向 Isaac Sim 安装目录的链接
├── real2sim2real/
│   ├── core/              # config / registry / cli / logger / runtime_env
│   ├── simworld|scene|robot|planner|job/  # base.py（ABC + Auto*）、config、stub
│   └── utils/             # pose / recorder / validate / isaacsim_utils
├── README.md
└── AGENTS.md
```


