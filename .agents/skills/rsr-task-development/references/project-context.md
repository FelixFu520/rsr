# 项目背景、依赖与约定

整理日期 2026-11-12，依据仓库代码、`AGENTS.md`、`README.md` 和 `docs/*.md`。数值与版本以运行时实测为准。

## 项目是什么

Real2Sim2Real（包名 `real2sim2real`，CLI `rsr`）是虚实闭环平台的仿真编排层：场景重建与数字孪生 →
仿真数据采集 → 策略训练 → 真机评测回流。本仓库负责中间两步：在 Isaac Sim 里装配场景、机器人和控制器，
跑出**由真实物理驱动、通过验收的**演示数据，导出为 LeRobot v3。框架是 ABC + 配置 + Registry + stub，
任务实现按能力域注册为后端，共用一个 `IsaacsimWorld` 和固定 Job 生命周期。

## 现有任务（新任务的参照物）

| 任务 | 后缀 / 资产目录 | 物理 | 机器人与控制 | 被操作物体 | 可借鉴 |
|---|---|---|---|---|---|
| 物理求解器验证 | `physic_solver` / `assets/PhysicSolver/` | Newton direct（Featherstone/MuJoCo/XPBD + VBD）、hybrid（PhysX TGS + VBD）、PhysX | 固定 Franka；URDF 雅可比 IK 或 Lula IK；阶段机插值 | 衣服、刚体方块、双网格柔性包裹 | 求解器组合矩阵、柔体验收项、批量采集并把失败移入 debug |
| 分拣包裹 | `sorting` / `assets/SortingPackage/` | Newton direct：原生 MuJoCo CPU + CUDA VBD | Pro-M6 双臂 + Robotiq 2F-140，mimic 用原生等式约束 | 软包 + 纸箱，传送带 | 继承上一任务的类；物理与渲染分离；逐子步记录；渲染同步审计 |
| 机器狗巡检 | `fire_inspection` / `assets/FireInspection/` | PhysX CPU TGS 120 Hz + Flow 火焰（仅视觉） | X30 运动学机身 + 脚本化 trot | 无（导航 + 拍照） | 迁移流程与对照表；PhysX 场景查询建图；资产依赖闭包（含 MDL） |

后端名（scene / robot / planner / job）见 `AGENTS.md` 1.1 节与 `pyproject.toml`。

## 运行环境与依赖

| 项 | 内容 | 核实方式 |
|---|---|---|
| Isaac Sim | 6.2（本机 `6.2.0-rc.14`），仓库根 `app -> ~/isaacsim` 软链 | `cat app/VERSION` |
| Python | 3.12，只用 `./app/python.sh` 或 `./scripts/rsr` | — |
| Newton / MuJoCo | 由 Isaac 扩展 `isaacsim.pip.newton` 提供（文档记录 Newton 1.6.1）；`physics_engine != "physx"` 时 `IsaacsimWorld` 才启用该扩展，裸 `python.sh` 里 `import newton` 会失败 | 在 Kit 进程里打印 `newton.__version__` |
| Warp | Isaac 自带（site-packages 为 1.18.0；分拣文档记录 1.17.0，以 Kit 内实际导入为准） | 同上 |
| pxr | 裸 `python.sh` 无法 `import pxr`，需要先启动 headless `SimulationApp` | 参考 `scripts/fire_inspection/prepare_fire_inspection_assets.py::_import_pxr` |
| 包依赖 | numpy、pillow、pyarrow、dynaconf、pydantic、pyyaml、structlog、typer；dev：pytest、ruff | `pyproject.toml` |
| 系统工具 | `ffmpeg` / `ffprobe`（视频编码与校验） | `which ffmpeg ffprobe` |
| uv | 只用于 `uv lock` 和 `uv build`；`uv.lock` 在 `.gitignore` 中 | — |

安装与自检：

```bash
ln -sfn ~/isaacsim app
./app/python.sh -m pip install -e ".[dev]"     # 注册 entry points；新增后端后必须重跑
./app/python.sh -m real2sim2real.core.cli list-backends
./app/python.sh -m ruff check real2sim2real scripts
./app/python.sh -m pytest -q
```

改依赖只动 `pyproject.toml`，然后 `uv lock`，再 `pip install -e ".[dev]"`。任务专用依赖见
[code-and-assets.md](code-and-assets.md#额外依赖)。

## 约定速查

- **单位与坐标**：米、Z-up、四元数 wxyz；整份打开的 stage 必须 `metersPerUnit=1`、Z-up（世界会校验）。
  USD 相机 -Z 朝前、+Y 朝上；`CameraConfig` 的焦距/光圈按毫米填，`create_cameras` 负责换算。
  物理求解器任务的 Newton 衣服模型内部用 `scale=100`（厘米），新任务优先用 SI，确需缩放时在运行时类内封装。
- **prim 路径**：`/World/scene`（Scene）、`/World/robots`（Robot）、`/World/background`（背景引用）、
  `/World/PhysicsScene`；背景不能挂在 scene/robots/PhysicsScene 下。相机必须在 `/World` 下，且
  `prim_path` 在 `pose_relative_to` 之下（跟随该 prim 运动）。
- **步长**：USD 路径（PhysX、Newton usd、hybrid）的 `physics_dt` 必须对应整数 Hz；录制 fps 必须整除物理频率；
  H.264 分辨率为偶数。
- **配置**：`default_settings.toml` → `inherit` 父配置（相对该文件目录，先父后子）→ `-c` 文件 →
  `.secrets.toml` → `RSR_*` 环境变量（嵌套 `__`，例 `RSR_JOB__OUTPUT_DIR=outputs/debug/x`）。
  所有 Config `extra="forbid"`：拼错或不存在的字段直接报错，环境变量覆盖也一样。
- **GUI 调试**：`RSR_SIM_WORLD__HEADLESS=false`；`job.inspect_gui=true` 在各阶段停下等 Enter。
  `keep_open` 只有 `FireInspectionJobConfig` 有。
- **Kit 退出码**：`fast_shutdown=True` 时 `SimulationApp.close()` 会直接退出进程、吞掉异常。脚本里先记录异常，
  再 `close(exit_code=1)`；父进程用 `utils/kit_process.run_kit_worker` 校验本次运行写出的结果。
- **日志**：`get_logger` / `LoggerMixin`（structlog），不用 `print`（独立 CLI 脚本输出结果除外）。
  长任务每约 30 s 记录一次进度与分项耗时（参考 `PhysicSolverJob.simulate`）。
- **资源**：本项目同时运行的 Isaac Sim 不超过两个（暂停实例也算）；不停止其它项目的任务；
  PathTracing 正式渲染期间 GPU 上不跑其它 Kit 任务（分拣任务实测会让刚体位姿错帧）。
- **git**：`assets/`、`outputs/`、`bak/`、`tests/*`（白名单除外）、`uv.lock` 不进 git；未经要求不 commit。

## 用户偏好（来自历次交付）

- 树要干净、好审计：每任务每能力域一个文件；遗留代码移到 `bak/`（git 忽略）而不是删除。
- 用户只检查非 debug 目录下的 `outputs/<task>/**/lerobot_v3`；诊断、失败、探索性数据一律放 `outputs/debug/<task>/`。
- 边缘用例重跑直到通过时，记录每一次运行（参考 `outputs/physic_solver/mujoco_vbd_pickplace_cloth/run_note.json`）并明确告知。
- 抓取效果要“看起来真实”，引擎可以换，但不能靠附着、隐藏夹块或放宽门限换取成功。
- 门限放宽、口径变化需用户授权（分拣任务的形状 RMS 37.4 → 45 mm 是用户授权后改的）。

## 已知文档漂移

动手前按代码核实；交付时提醒用户是否修正。

| 位置 | 文档说法 | 实际 |
|---|---|---|
| `AGENTS.md` 能力分层 | `TabletopScene`、`BaseManipulationPlanner`、`TabletopJob`、`maybe_gui_inspect` | 代码中不存在；GUI 检查是 `BaseJob.gui_inspect` → `sim_world.wait_for_gui_edit`；抓放类 Planner 直接继承 `BasePlanner`，`begin_episode` 在各自实现里 |
| `AGENTS.md` 1.1 任务布局表 | — | 表头重复了一行（3 列与 4 列各一行） |
| `README.md`、`simworld/isaacsim.py` 类注释 | `configs/isaacsim_world.toml` | 文件不存在；世界字段以 `IsaacsimWorldConfig` 为准 |
| `docs/physic_solver.md` 第 4 节 | GUI 用 `RSR_JOB__KEEP_OPEN=true` | `PhysicSolverJobConfig` 无此字段，设置后配置校验失败 |
| `scripts/rsr` 注释、`.vscode/launch.json` | `configs/tabletop.toml`、`configs/navigation_g1_official.toml` | 文件不存在（旧任务遗留） |
| `docs/physic_solver.md` 第 4 节 | `pytest tests/test_isaacsim_modes.py` | 该文件在 `.gitignore` 中，仅本机存在 |
