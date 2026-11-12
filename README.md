# Real2Sim2Real

虚实闭环平台：**场景重建与数字孪生 → 仿真数据采集 → 策略训练 → 真机评测回流**。

本仓库是可插拔的仿真编排框架：抽象基类 + 配置 / Registry + stub 后端，真实仿真跑在
**唯一的 Python 环境**——Isaac Sim(当前使用6.2版本) 自带的 `python.sh`。每个任务的场景、机器人、规划器、
Job 都是独立注册的后端，复用同一个 `IsaacsimWorld` 和固定的 Job 生命周期，结果导出为 LeRobot v3。

> Agent / 贡献者：先读 [`AGENTS.md`](AGENTS.md)（目录约定、扩展后端、配置与 CLI 约定）。

## 任务

| 任务 | 内容 | 物理 | 配置 | 文档 |
|---|---|---|---|---|
| 求解器验证 | 桌面 Franka 抓放衣服 / 刚体方块 / 自定义柔性包裹，8 个“机器人求解器 + 接触求解器”组合 | Newton（Featherstone、MuJoCo、XPBD、VBD）、PhysX、hybrid | [configs/physic_solver/](configs/physic_solver/) | [docs/physic_solver.md](docs/physic_solver.md) |
| 分拣包裹 | Pro-M6 双臂把软包和纸箱依次从入料带经平台送到出料带，四路 1K 视频 | MuJoCo（CPU）+ VBD（CUDA） | [configs/sorting/](configs/sorting/) | [docs/sorting.md](docs/sorting.md) |
| 机器狗巡检 | X30 一句话指令巡航仓库，发现火点、拍照、警示灯报警 | PhysX（CPU）+ Flow 火焰 | [configs/fire_inspection/](configs/fire_inspection/) | [docs/fire_inspection.md](docs/fire_inspection.md) |

采集命令（仓库根目录）：

```bash
# 物理求解器验证：串行采集 8 个组合
./app/python.sh scripts/physic_solver/collect_physic_solver_cases.py
# 分拣包裹：连续物理 → 四路 1K 渲染 → LeRobot v3 → 独立验收
./scripts/sorting/run_sorting_showcase.sh outputs/sorting
# 机器狗巡检：采集 + 独立验收
./scripts/rsr collect run -c configs/fire_inspection/x30_physx_warehouse.toml
./app/python.sh scripts/fire_inspection/validate_fire_inspection.py outputs/fire_inspection
```

### 输出约定

| 目录 | 内容 |
|---|---|
| `outputs/physic_solver/<case>/` | 通过验收的求解器组合，含 `lerobot_v3/`、`result.json`、`validation.json` |
| `outputs/sorting/` | 分拣连续物理源、渲染缓存、`lerobot_v3/`、`qa/` 独立验收 |
| `outputs/fire_inspection/` | 巡检 `lerobot_v3/`、照片、地图、`validation.json` |
| `outputs/debug/<task>/` | 调试与未通过验收的结果（`diagnostic` 数据、局部物理、对照实验） |

`outputs/` 下非 debug 目录里的 `lerobot_v3/` 都是通过全部物理验收的 `demonstration` 数据；
已有正式数据集的目录拒绝覆盖。

## 环境

| 角色 | 用什么 |
|------|--------|
| 运行时 / 开发 / 测试 | **仅** Isaac Sim：`./app/python.sh`（或 `$RSR_ISAACSIM_HOME/python.sh`） |
| 项目元数据 / 锁文件 / 打 wheel | [uv](https://docs.astral.sh/uv/) 读根目录 [`pyproject.toml`](pyproject.toml) |

```bash
cd /path/to/rsr
ln -sfn ~/isaacsim app                     # 仓库根的 app 指向 Isaac Sim 安装目录
uv lock                                    # 更新 uv.lock（不创建运行时 venv）
./app/python.sh -m pip install -e ".[dev]" # editable 装进 Isaac，注册 entry points
./app/python.sh -m real2sim2real.core.cli list-backends
./app/python.sh -m pytest -q
```

不要 `uv sync` 出 `.venv` 当仿真环境。录制依赖 Pillow、PyArrow，视频编码与校验需要系统 `ffmpeg` / `ffprobe`。
资产（`assets/`）与输出（`outputs/`）不进 git；各任务的资产准备脚本见对应文档。
每个任务的资产自成一个目录，可单独打包：`assets/PhysicSolver/`（物理求解器验证）、
`assets/SortingPackage/`（分拣包裹）、`assets/FireInspection/`（机器狗巡检）；其余资产在 `assets/bak/`。
本项目同时运行的 Isaac Sim 实例不超过两个。

## 目录结构

```text
rsr/
├── pyproject.toml              # 依赖、entry points、脚本 rsr
├── app -> ~/isaacsim           # Isaac Sim 安装目录
├── real2sim2real/
│   ├── core/                   # 配置、Registry、CLI、日志、default_settings.toml
│   ├── simworld/               # base.py、stub.py、isaacsim.py（统一世界）
│   ├── scene/                  # base.py、scene_stub.py、scene_{physic_solver,sorting,fire_inspection}.py
│   ├── robot/                  # base.py、robot_stub.py、robot_{physic_solver,sorting,fire_inspection}.py
│   ├── planner/                # base.py、planner_stub.py、planner_{physic_solver,sorting,fire_inspection}.py
│   ├── job/                    # base.py、cli.py、job_stub.py、job_{physic_solver,sorting,fire_inspection}.py
│   └── utils/                  # 共用：asset_utils、cameras、kit_process、lerobot_v3
│                               # 任务：{physic_solver,sorting,fire_inspection}_utils.py
├── configs/
│   ├── physic_solver/          # 8 个求解器组合
│   ├── sorting/                # mujoco_vbd_sorting.toml + render_profile.json
│   └── fire_inspection/        # 正式 + 调试
├── scripts/
│   ├── rsr                     # 用 Isaac python.sh 启动 CLI
│   ├── physic_solver/          # 资产准备、批量采集、包裹物理检查与校验、世界模式冒烟测试
│   ├── sorting/                # 一体流程、物理求解、回放渲染、独立验收、同步检查、预览
│   └── fire_inspection/        # 资产准备、独立验收
├── tests/                      # 无 Kit 单元测试
└── docs/                       # 每个任务一份文档
```

每个任务在 scene / robot / planner / job / utils 中各只有一个文件，文件名后缀即任务名。
分拣任务复用物理求解器任务的类（`SortingScene ← PackageScene`、`SortingRuntime ← PackageRuntime`、
`SortingJob ← PhysicSolverJob`）。

## 架构

```
                    +-------------------------------------+
  settings.toml --->|  Dynaconf + pydantic BaseConfig     |
  RSR_* env    --->|  core/config.py                      |
                    +------------------+------------------+
                                       |
         Auto*.from_config()           |  Registry.get(backend)
                                       v
  +----------+  +-------+  +-------+  +---------+
  | SimWorld |  | Scene |  | Robot |  | Planner |
  +----+-----+  +-------+  +---+---+  +----+----+
       |                       |           |
       +----------- Job(config) -----------+
                                       |
                              BaseJob.run()
     init_world -> arrange_scene -> place_robot -> start_simulation -> finally: close
```

| 模块 | 职责 |
|------|------|
| simworld | 仿真运行时（`init_world` / play / step / close）+ 加载环境 USD |
| scene | 在已有环境中添加额外资产（桌面物体、工位、相机等） |
| robot | 机器人定义与初始化 |
| planner | 每个物理步生成控制目标，记录实测量并给出验收结果 |
| job | 装配组件，按固定流水线调用上述 API，负责采集与导出 |

`BaseJob.__init__` 先加载任务配置，再用四个 `Auto*.from_config()` 创建组件；组件构造时不启动 Kit，
`init_world()` 才创建应用。各阶段钩子和顺序见 [`AGENTS.md`](AGENTS.md)。

### 统一仿真世界

[IsaacsimWorld](real2sim2real/simworld/isaacsim.py) 由 `sim_world.physics_engine` 选择 `physx`、
`newton`（`newton_execution="usd"` 或 `"direct"`）或 `hybrid`；`step()` 每次推进一个 `physics_dt`，
`update()` 只刷新渲染。direct / hybrid 由组件注册 Newton stepper。完整字段见
[configs/isaacsim_world.toml](configs/isaacsim_world.toml)，各模式的用法见
[物理求解器验证文档](docs/physic_solver.md#12-世界模式)。

固定相机写在 `[[scene.cameras]]`，机载相机写在 `[[robot.cameras]]`：每项含唯一 `name`、`prim_path`、
`pose_relative_to`、`pose`（米 + wxyz 四元数，USD 相机坐标 -Z 朝前）和 `resolution`。录制按相机名
生成 `observation.images.<name>`，各路共享帧号和时间戳，切换采集视角不推进物理。

## 配置系统

查找与覆盖顺序（后者优先）：

1. 包内 `real2sim2real/core/default_settings.toml`（全部 stub，零配置可跑）
2. 项目配置的 `inherit` / `dynaconf_include` 父配置（先父后子，相对路径相对该文件），再加载
   `RSR_SETTINGS_FILE`（CLI `-c` 会设置）或 cwd `settings.toml`
3. 同目录 `.secrets.toml`（可选，不进 git）
4. 环境变量 `RSR_*`（嵌套用双下划线，如 `RSR_JOB__OUTPUT_DIR=outputs/debug/x`）

常用 section：`logging` / `sim_world` / `scene` / `robot` / `planner` / `job`，各组件按 section 的
`backend` 选型。配置类与实现放在同一文件，由实现类的 `config_cls` 指定（`extra="forbid"`）。

## CLI

```bash
./app/python.sh -m real2sim2real.core.cli version
./app/python.sh -m real2sim2real.core.cli list-backends [-k planner]
./app/python.sh -m real2sim2real.core.cli settings [-s job]
./app/python.sh -m real2sim2real.core.cli collect run [-c settings.toml]   # 不带 -c 时为 stub 流水线
./scripts/rsr collect render-sorting --output outputs/sorting              # 分拣：物理 + 回放渲染
```

`./scripts/rsr` 与已安装的 `rsr` 入口是同一个 CLI。`collect` 等业务子命令通过
`real2sim2real.cli.commands` 插件挂载。

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

## 发布与安装

```bash
uv build                                                        # 产物：dist/real2sim2real-*.whl
/path/to/isaacsim/python.sh -m pip install real2sim2real-0.1.0-py3-none-any.whl
/path/to/isaacsim/python.sh -m real2sim2real.core.cli collect run   # 验证内置 stub 流程
```

相对路径的 USD 按工作区根解析（源码树、`RSR_REPO_ROOT`，或当前目录上的 `assets/` + `configs/`）。
