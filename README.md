# Real2Sim2Real

虚实闭环平台：**场景重建与数字孪生 → 仿真数据采集 → 策略训练 → 真机评测回流**。

当前仓库是可插拔的仿真编排框架：抽象基类 + Registry + stub 后端，真实仿真引擎（Isaac Sim）跑在 **唯一的 Python 环境**——Isaac 自带的 `python.sh`。

> Agent / 贡献者：请先读 [`AGENTS.md`](AGENTS.md)（目录约定、扩展后端、配置与 CLI 约定）。

## Franka 桌面抓取 Demo

Demo 的 world、scene、robot、planner、job 分别以 `franka_demo` 注册；复用固定的
`BaseJob.run()` 生命周期。机器人固定在桌面上，执行接近、闭合、抬升、保持、释放与撤离。
从仓库根目录运行，资产位于 `assets/Demo`（复制资产和依赖，不改源文件）。
组件实现分别位于 [world](real2sim2real/simworld/franka_demo.py)、
[scene](real2sim2real/scene/scene_franka_demo.py)、
[robot](real2sim2real/robot/robot_franka_demo.py)、
[planner](real2sim2real/planner/planner_franka_demo.py) 和
[job](real2sim2real/job/job_franka_demo.py)；
共享引擎模式由 [isaacsim.py](real2sim2real/simworld/isaacsim.py) 提供。

```bash
./app/python.sh -m pip install -e ".[dev]"
./app/python.sh scripts/prepare_franka_demo_assets.py

# 刚性方块：PhysX + TGS
./app/python.sh -m real2sim2real.core.cli collect run -c configs/franka_rigid_physx.toml

# 衣服：PhysX 机械臂 + Newton VBD 衣服
./app/python.sh -m real2sim2real.core.cli collect run -c configs/franka_cloth_hybrid.toml

# 衣服：全部物理由 Newton 执行，Featherstone 机械臂 + VBD 衣服
./app/python.sh -m real2sim2real.core.cli collect run -c configs/franka_cloth_newton.toml
```

无头运行可在命令前加 `RSR_SIM_WORLD__HEADLESS=true`；GUI 完成后保持窗口用
`RSR_JOB__KEEP_OPEN=true`。每个示例写入独立的 `outputs/franka_*` 目录，
`result.json` 保存实测抓取结果，启用 `capture_frames` 时保存保持和结束画面。
验收失败默认抛出异常并关闭仿真，避免把仅完成动作序列当作成功抓取。
`configuration.json` 记录本次生效配置，`trajectory.jsonl` 保存逐步测量值。

录制全部六个配置（含统一 XPBD 抓取失败对照）：

```bash
./app/python.sh scripts/record_franka_demos.py
```

脚本依次启动仿真，避免同时运行多个 demo。视频汇总在 `outputs/videos/`，
各案例目录也保存 `grasp.mp4`、`video.json` 和逐帧仿真时间 `video_frames.jsonl`。
默认视频为 1280×720；每 `render_stride` 个物理步采一帧，当前配置对应 30 fps，
播放速度与仿真时间一致。录制过程中的额外渲染刷新不推进物理。
单独录制一个配置可用 `RSR_JOB__RECORD_VIDEO=true`；
宽高由 `[job].video_width`、`video_height` 控制，系统需提供 `ffmpeg`。

张开夹爪的对照可用环境变量运行，输出写到单独目录；预期
`result.json` 中 `target_lifted=false`、`target_stays_near_gripper=false`，
用于核对闭合动作对持续抓持的作用：

```bash
RSR_SIM_WORLD__HEADLESS=true RSR_PLANNER__OPEN_GRIPPER_CONTROL=true \
RSR_JOB__FAIL_ON_UNSUCCESSFUL=false RSR_JOB__CAPTURE_FRAMES=false \
RSR_JOB__OUTPUT_DIR=outputs/franka_cloth_open_control \
./app/python.sh -m real2sim2real.core.cli collect run -c configs/franka_cloth_newton.toml
```

PhysX/hybrid 使用官方 Panda USD，纯 Newton 使用随 Newton 示例提供的官方 Franka
FR3 URDF。两者都固定基座，双指行程命令以米为单位。示例中的摩擦系数、驱动增益和
衣服材料参数是仿真起点：方块质量 `0.1 kg`，衣服默认总质量 `0.2 kg`，
衣服密度根据实际三角面面积反算。Newton 内部使用厘米尺度，渲染和输出统一为米。
衣服抓点采用实心侧缝顶点 `1922`。纯 Newton 抓取高度偏移为 `6 mm`；
hybrid 为 `12 mm`，并提高机械臂位置驱动增益，保证实际指尖在桌面上方闭合。
这些值对应随示例复制的资产与桌面尺寸，更换模型时应重新核对碰撞几何。

`[sim_world].physics_engine` 支持：

| 值 | 动力学归属 | 使用场景 |
|---|---|---|
| `physx` | PhysX 负责全部刚体和关节 | 刚性方块 demo |
| `newton` | Newton 负责全部动力学 | 衣服 demo 使用 `newton_execution="direct"` 的共享模型 |
| `hybrid` | PhysX 机器人，显式 Newton 步进负责衣服 | 衣服 demo，保留项目现有集成方式 |

纯 Newton 的 `newton_execution="usd"` 使用 Isaac Sim 官方 Newton 后端，
MuJoCo、XPBD 已通过本机步进验证。VBD 的 USD 路径还依赖匹配的
`NewtonVbdSceneAPI` schema，本机 6.2 RC 安装未提供，选择时会明确报错。
VBD 衣服 demo 使用 `direct`，由注册的 stepper 管理 Newton 模型，
暂停 Kit 时间轴且不调用 PhysX World 步进。
旧配置 `enable_newton=true` 在没有显式模式时仍选择 `hybrid`。

布料 demo 使用 Newton 原生三角面片模型和夹指接触，不使用把衣服绑定到夹爪的附件。
VBD 的 `integrate_with_external_rigid_solver` 路线，以及 hybrid 的运动学接触代理，
均为单向机器人驱动衣服；衣服反力不会反馈到机械臂。需要双向力控时须另行接入并验证
Newton 耦合求解器。[官方 Isaac Sim 后端说明](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/physics/newton_physics.html)
和 [Newton 耦合说明](https://newton-physics.github.io/newton/1.6.1/concepts/coupling.html)
解释了后端切换与显式耦合的区别。

另提供 `franka_cloth_newton_xpbd.toml`（XPBD 机器人 + VBD 衣服）和
`franka_cloth_newton_mujoco.toml`（MuJoCo 机器人 + VBD 衣服）供求解器对照；
衣服求解器由 `[planner].cloth_solver` 显式指定。MuJoCo 本身不求解衣服。
Featherstone 路径采用速度驱动；Featherstone、XPBD 机器人均补偿重力，衣服保留重力。
MuJoCo 路径保留机器人重力，因此使用独立的驱动增益与抓取高度配置。
`franka_cloth_newton_xpbd_unified.toml` 另外保留 `cloth_solver="xpbd"` 的统一求解对照：
本机整件衣服可稳定步进，尚未通过抓持验收，失败会正常返回非零状态。
统一 XPBD 衣服使用距离弹簧（`xpbd_spring_stiffness` / `xpbd_spring_damping`），
通过 `xpbd_mass_lumping="uniform"` 保持总质量并均匀分配到粒子，区别于 VBD 的面积质量分配；
默认关闭原始粒子间排斥；Newton 1.6.1 的 XPBD 不支持布料自碰撞。
`xpbd_self_contact=true` 仅供诊断，不具备 VBD 的网格邻接过滤，密集网格可能发散。
这些求解器对照配置属于实验路径。Newton 1.6.1 的 Featherstone、XPBD
不执行 `joint_effort_limit`；本示例的 `finger_max_force` 限力只在 PhysX/MuJoCo
路径生效。布料的单向耦合路径均不能据此评价真实夹爪承载能力。
[Newton 求解器能力表](https://newton-physics.github.io/newton/1.6.1/solvers/index.html#joint-feature-support)

衣服 demo 的子步数与迭代数由 `[planner].newton_substeps`、`solver_iterations`
控制；`[sim_world]` 下同名 Newton 配置用于官方 USD 后端。
如需单独检查引擎步进、重力下落和渲染刷新是否误推进物理，可逐条运行：

```bash
./app/python.sh scripts/validate_physics_modes.py --engine physx
./app/python.sh scripts/validate_physics_modes.py --engine newton --solver mujoco
./app/python.sh scripts/validate_physics_modes.py --engine newton --solver xpbd
./app/python.sh scripts/validate_physics_modes.py --engine newton --execution direct --solver vbd
./app/python.sh scripts/validate_physics_modes.py --engine newton --execution direct
./app/python.sh scripts/validate_physics_modes.py --engine hybrid --solver vbd
```

本机参考验收（Isaac Sim `6.2.0-rc.14`、Newton `1.6.1`、Warp `1.17.0`）：

| 配置 | 保持阶段最低抬升 | 松爪后下降 | 结果 |
|---|---:|---:|---|
| `franka_rigid_physx.toml` | 24.1 cm | 24.4 cm | 通过 |
| `franka_cloth_newton.toml` | 24.2 cm | 23.9 cm | 通过 |
| `franka_cloth_hybrid.toml` | 25.5 cm | 25.2 cm | 通过 |
| `franka_cloth_newton_mujoco.toml` | 25.2 cm | 24.0 cm | 通过（含视频） |
| `franka_cloth_newton_xpbd.toml` | 26.2 cm | 25.3 cm | 通过（含视频） |

衣服包含 6436 个自由粒子，无固定顶点或抓取附件。上述数值来自各示例目录中的
`result.json`；保持阶段持续检查实际几何位置，释放后检查回落和桌面支撑。
MuJoCo+VBD、XPBD+VBD 此前也用相同生产 runtime 和完整资产完成无 Kit 的 GPU 物理验收，
各自的张开夹爪对照均未达到持续抓持条件。当前六配置视频采集流程会在 Kit 中
重新执行完整动作序列；每次采集的实际结果见对应案例的 `result.json`，
视频和解码检查汇总见 `outputs/videos/recording_summary.json`。
Demo 的配置、资产、步进与验收回归可单独运行：

```bash
./app/python.sh -m pytest -q tests/test_isaacsim_modes.py tests/test_franka_demo_scene.py tests/test_franka_demo_job.py tests/test_franka_demo_acceptance.py tests/test_franka_demo_coupling.py
```

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

### 分拣包裹采集后端

采集配置为 `configs/arm-sorting-package_pro-m6-696_curobo_factory002.toml`。

该配置在入带范围内做 ±8 cm 的 XY 初始位置扰动，本轮相邻成功集的初始位置至少相差
5 cm。双臂抓点在包裹中心附近随机采样，由 `grasp_jitter_fraction_xy` 限制为包裹
半宽/半长的比例；`[0, 0]` 则恢复确定性的中心优先抓取。每个候选必须通过下探、
提起的 IK 和关节余量检查，执行后仍检查接触、抬升、离带和标签朝向。不可达或
验收失败会重新采样，最多 `max_attempts` 次，不保证任意初始位置都能成功。
失败尝试只写 `rejected_attempts.jsonl` 诊断，成功集才进入 LeRobot 训练索引。
`meta/episodes.jsonl` 保存初始位姿、左右实际抓点、相对中心偏移和抓点随机种子，
可直接比较数据差异。固定 `object_jitter_seed` / `grasp_random_seed` 可复现采样序列，
不同集和重试仍使用不同抓点。
各后端将专用配置与实现放在同一文件，直接继承对应的 `Base*`：

| 后端 | 实现文件 | 公开导入 |
|------|----------|----------|
| Job | `job/job_sorting_packages.py` | `from real2sim2real.job.job_sorting_packages import SortingJob` |
| Planner | `planner/planner_sorting_packages.py` | `from real2sim2real.planner.planner_sorting_packages import SortingPlanner` |
| Robot | `robot/robot_pro_m6.py` | `from real2sim2real.robot.robot_pro_m6 import ProM6Robot` |
| Scene | `scene/scene_sorting_packages.py` | `from real2sim2real.scene.scene_sorting_packages import SortingScene` |
| SimWorld | `simworld/isaacsim.py` | `from real2sim2real.simworld.isaacsim import IsaacsimSimWorld` |

表中文件路径均相对于 `real2sim2real/`。entry points 直接指向各实现模块中的类。配置中的 `sorting`、`pro_m6`、`isaacsim` 后端名保持不变。
Kit、USD 和控制器运行时依赖在世界初始化后加载，构造组件不会启动仿真资源。
通用录制配置与落盘实现位于 `utils/recorder.py`，相机标注常量位于
`utils/isaacsim_utils/camera_utils.py`。

分拣示例启用了 `[planner].keep_label_up`：包裹入带时须已标签朝上。
左臂从主体中部夹取，使用 `left_lift_height_m = 0.14` 和
`left_setdown_offset_m = 0.0` 减少悬垂与过压翻折。平台松爪、左臂撤离后，
按 `label_uv_bounds` 跟踪实际标签三角面；其面积加权法线的世界 Z 分量
须达到 `label_up_min_cos = 0.9`，否则本次尝试失败，不进入右臂接力。
默认 UV 范围对应 `bagged_clothing_000`，更换包裹纹理时须重新标定，
这项功能不负责将入带时标签朝下的包裹主动翻面。

```bash
./app/python.sh -m real2sim2real.core.cli collect run -c configs/arm-sorting-package_pro-m6-696_curobo_factory002.toml
```

### 核心层

| 层 | 路径 | 职责 |
|----|------|------|
| 抽象接口 | 各模块 `{domain}/base.py` | `BaseSimWorld` / `BaseScene` / `BaseRobot` / `BasePlanner` / `BaseJob` |
| 模块实现 | `real2sim2real/{simworld,scene,robot,planner,job}/` | ABC 与工厂位于 `base.py`；job/planner/robot/scene 的配置与各后端实现在同一文件，SimWorld 保留 `config.py`；后端通过 `config_cls` 指定配置类 |
| 基础设施 | `real2sim2real/core/` | 配置、Registry、CLI、日志、运行时路径 |
| 辅助函数 | `real2sim2real/utils/` | pose、recorder、validate、isaacsim_utils |

工具的公共符号统一在 `utils/__init__.py` 导出，外部使用 `from real2sim2real.utils import ...`；实现仍按职责保留在子模块中。

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


