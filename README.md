# Real2Sim2Real

虚实闭环平台：**场景重建与数字孪生 → 仿真数据采集 → 策略训练 → 真机评测回流**。

当前仓库是可插拔的仿真编排框架：抽象基类 + Registry + stub 后端，真实仿真引擎（Isaac Sim）跑在 **唯一的 Python 环境**——Isaac 自带的 `python.sh`。

> Agent / 贡献者：请先读 [`AGENTS.md`](AGENTS.md)（目录约定、扩展后端、配置与 CLI 约定）。

当前整理状态：历史任务实现保留在 `bak/rsr-bak/`。主包包含基础框架、stub 通路、统一的 `IsaacsimWorld` 和衣服抓放采集案例。
下文旧 Franka / sorting 任务作为历史参考保留；新衣服案例使用下述独立配置。

## 求解器组合与 LeRobot v3 采集

求解器案例统一命名为 `PhysicSolver_<机器人求解器>_<接触求解器>_<任务>.toml`。
任务参数写入各自配置，`core/default_settings.toml` 保持原来的 stub 默认值。
PhysX 的实际求解器在文件名中写为 `tgs`，引擎归属仍由 `sim_world.physics_engine` 指定。

这些案例的实现集中在 [planner_physic_solver.py](real2sim2real/planner/planner_physic_solver.py)、
[robot_physic_solver.py](real2sim2real/robot/robot_physic_solver.py)、
[scene_physic_solver.py](real2sim2real/scene/scene_physic_solver.py) 和
[job_physic_solver.py](real2sim2real/job/job_physic_solver.py)。模块文件统一命名，
各后端仍通过原有 `backend` 名称选择。

| 配置后缀（均有 `PhysicSolver_` 前缀和 `.toml` 后缀） | 世界模式 | 机器人 / 接触 | 任务 |
|---|---|---|---|
| `featherstone_vbd_pickplace_cloth` | Newton direct | Featherstone / VBD | 衣服抓放 |
| `featherstone_vbd_pickplace_package` | Newton direct | Featherstone / VBD | V7 袋膜与内部衣物抓放 |
| `mujoco_vbd_pickplace_cloth` | Newton direct | MuJoCo / VBD | 衣服抓放 |
| `xpbd_vbd_pickplace_cloth` | Newton direct | XPBD / VBD | 衣服抓放 |
| `tgs_vbd_pickplace_cloth` | hybrid | PhysX TGS / Newton VBD | 衣服抓放 |
| `tgs_tgs_pickplace_rigid` | PhysX | TGS / TGS | 刚体方块抓放 |
| `xpbd_xpbd_pickplace_cloth` | Newton direct | XPBD / XPBD | 衣服抓放实验组 |

统一串行采集和校验：

```bash
./app/python.sh scripts/collect_physic_solver_cases.py
# 也可只选择一个组合：
./app/python.sh scripts/collect_physic_solver_cases.py xpbd_vbd_pickplace_cloth
```

采集脚本复核并复用已有数据，其余组合逐个启动 Kit。汇总位于
`outputs/PhysicSolver/collection_summary.json`，逐组记录采集状态和抓取验收结果。
各组合仍使用独立的 `output_dir`、日志和 LeRobot v3 数据目录。
实验组可通过 `job.recorder.save_failed_episodes=true` 保留失败轨迹：
episode 元数据的 `success=false`、`dataset_use="diagnostic"` 会明确区分实测失败与成功示教。
数据内 `meta/task_outcome.json` 同时保存物理验收和生效配置。

XPBD 使用 maximal body coordinates，关节观测由实时 link 位姿反算到独立观测缓冲区；
采集不会修改求解器状态。PhysX 和 Newton 的数据均记录实际关节位置及同一帧的控制目标。

[PhysicSolver_featherstone_vbd_pickplace_cloth.toml](configs/PhysicSolver_featherstone_vbd_pickplace_cloth.toml)
使用统一的 `IsaacsimWorld`，选择 Newton direct：Featherstone 负责固定基座的 Franka FR3，
VBD 负责衣服。动作包括接近、闭合、抬升、保持、横向转运、降低、松开与撤离。
默认无头运行，所有资产使用 `assets/PhysicSolver/` 中的独立副本。

```bash
./app/python.sh -m pip install -e ".[dev]"
# 资产已复制；需要重新准备时执行（保留 USD/URDF/MDL 依赖和许可证）
./app/python.sh scripts/prepare_physic_solver_assets.py
./app/python.sh -m real2sim2real.core.cli collect run -c configs/PhysicSolver_featherstone_vbd_pickplace_cloth.toml
```

默认结果位于 `outputs/PhysicSolver/featherstone_vbd_pickplace_cloth/`：

- `lerobot_v3/`：一次成功采集的一个 episode，包含 `meta/info.json`、`meta/stats.json`、
  `meta/tasks.parquet`、`meta/episodes/chunk-000/file-000.parquet`、
  `data/chunk-000/file-000.parquet` 与对应 RGB MP4。
- `result.json`：提起、持续抓持、衣服变形、松开回落、放置误差等实测验收结果。
- `diagnostics/<run>/`：生效配置与逐步测量。未通过验收的采集保存在此处，
  不发布为训练数据集。

数据为 [LeRobot v3.0 布局](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)，
以 15 fps 同步记录 `observation.images.overview`、`observation.state` 与 `action`。
状态是实际关节位置，动作是同一帧下发的关节目标：前七维为机械臂 rad，后两维为手指开度 m。
采集仅刷新渲染，不额外推进物理。已有成功数据不会被覆盖；再次采集时更换
`job.output_dir`，例如使用环境变量 `RSR_JOB__OUTPUT_DIR=outputs/PhysicSolver/run2`。

该案例采用机器人驱动衣服的单向耦合；衣服无固定顶点、无抓取附件。
Featherstone 的轨迹控制包含重力补偿和限速跟踪，当前不执行 URDF effort limit；
因此这些数据用于该仿真控制方式，不能据此评价真实夹爪限力性能。

### V7 包裹抓放

[原装 Franka 夹爪配置](configs/PhysicSolver_mujoco_vbd_pickplace_package.toml) 使用
MuJoCo 位置驱动机械臂和 URDF 原装短指，接触与袋膜/衣物仍使用 VBD。
初始姿态在中央抓取点上方，抓取偏移、各阶段时长可配置；动作数据记录关节位置目标。
可按转运开始时的实测衣物质心修正剩余路径，使悬垂包裹对齐放置目标。
结果单独保存在 `outputs/PhysicSolver/mujoco_vbd_pickplace_package/`，保留失败诊断视频。

```bash
./app/python.sh -m real2sim2real.core.cli collect run -c configs/PhysicSolver_mujoco_vbd_pickplace_package.toml
./app/python.sh scripts/validate_package_pickplace.py outputs/PhysicSolver/mujoco_vbd_pickplace_package
```

原装短指配置已完成一次通过全部 14 项验收的采集：衣物/袋膜最大质心提升约
236/232 mm，转运双侧接触保持率 100%，释放回落约 106 mm，落点误差约 18.8 mm。
局部最大边长伸长比 1.223，无桌面穿透；转运期间抽样的最低质心提升仍分别为
231/226 mm。12.3 秒、双相机各 185 帧（15 fps）数据标记为 `demonstration`。
视频解码、帧对齐、动作单位和十个关键时刻的袋内包含关系均已复核。

[长夹块对照配置](configs/PhysicSolver_featherstone_vbd_pickplace_package.toml) 使用
`pickplace_package` 场景和 Planner，沿用采集 Job 的完整生命周期和双相机 LeRobot v3 输出。
独立资产在 `assets/PhysicSolver/Package/v7/`，保留 Bag、Garment 两个物理网格、
原始 UV/材质、逐点质量（衣物 200 g、袋膜约 20.54 g）和衣物内部的被动弹簧。
Newton 模型使用米/千克/秒，V7 的材料参数从资产读取，不将衣物压扁或固定在夹爪上。
机器人使用 `franka_package_pads`：在 Featherstone 手指刚体上安装 V7 的
14 × 60 × 84 mm 长夹块，显示尺寸与碰撞几何一致。原 Franka 短指在本任务的中央/边缘
试抓中滑脱，因此不将长夹块数据标成原装夹爪数据。夹块密度取 1000 kg/m³，
每块约 70.56 g，是仿真假设，未做硬件标定。

```bash
# 重新复制时拒绝覆盖不同文件；源 V7 改变后应选新的 --target 保存快照。
./app/python.sh scripts/prepare_package_assets.py
# 可先不录 RGB，验证抓取几何和物理行为。
./app/python.sh scripts/check_package_pickplace.py
./app/python.sh -m real2sim2real.core.cli collect run -c configs/PhysicSolver_featherstone_vbd_pickplace_package.toml
# 已导出数据的帧对齐、双相机解码和粒子几何复核（含失败诊断数据）。
./app/python.sh scripts/validate_package_pickplace.py
```

结果在 `outputs/PhysicSolver/featherstone_vbd_pickplace_package/`。
验收分别检查袋膜与衣物的质心提升、转运双侧接触、释放回落、落点、逐帧表面伸长与桌面穿透，
并累计每个物理子步的接触容量。`diagnostics/<run>/particles.npz` 保存 15 Hz 的实际粒子状态，
用于在十个关键时刻独立检查全部衣物顶点及面中心是否仍在封闭袋膜内（容差 1 mm）。
几何抽样不能证明连续时间无穿透。源 V7 的 USD 复算尚未全部通过，因此本任务独立验收，
不继承源轨迹的成功标记；失败轨迹留在 diagnostics，不进入成功数据集。
需要将完整失败视频也导出为 LeRobot 时，设置 `job.recorder.save_failed_episodes=true`；
episode 会标记 `success=false`、`dataset_use="diagnostic"`。
复核脚本输出 `validation.json`、`qa/physical_audit.json` 和双相机关键帧 `qa/keyframes.jpg`。

长夹块对照配置的实测验收未通过：32 子步 / 28 次迭代下，衣物与袋膜质心分别提升约
115.7 / 115.4 mm 后滑脱，转运双侧接触占比约 2.25%，局部最大边长伸长比为 1.422。
13 秒、双相机各 195 帧的采集保留为 `diagnostic`。十个几何抽样时刻的衣物包含检查通过，
未检测到桌面穿透或接触容量溢出；这些通过项不代表抓放任务成功。

Featherstone 与 VBD 在这里采用单向刚体驱动柔体耦合，夹爪不接收包裹的反作用力；
参见 [Newton VBD 耦合说明](https://newton-physics.github.io/newton/1.6.0/api/_generated/newton.solvers.SolverVBD.html)。
这套控制方式不能用于验证实物夹爪的额定负载或限力性能。

## 统一仿真世界（Isaac Sim 6.2）

[IsaacsimWorld](real2sim2real/simworld/isaacsim.py) 与 `IsaacsimWorldConfig` 放在同一文件，
由 `AutoSimWorld.from_config()` 创建。构造阶段不启动 Kit，`init_world()` 才创建应用、
加载环境并配置引擎。世界使用米、Z-up；桌子、刚性/柔性物体和机器人由 Scene/Robot 添加，
无需为每个抓取案例派生世界类。`world` 句柄是 6.2 的 `SimulationManager`，
后续组件通过 experimental/tensor API 访问物理状态。

| `physics_engine` | 动力学执行方式 |
|---|---|
| `physx` | USD 场景全部由 PhysX 积分，支持该引擎可用的刚体/柔体资产 |
| `newton` + `newton_execution="usd"` | 官方 Newton USD 后端；求解器选择 `mujoco`、`xpbd` 或 `vbd`，具体资产支持取决于求解器和安装的 schema |
| `newton` + `newton_execution="direct"` | 应用 Newton 模型负责全部动力学，Kit 时间轴保持停止，只用于显示 |
| `hybrid` | PhysX 推进一步后调用应用的 Newton 模型，供 PhysX 机器人 + Newton 柔体等组合使用 |

Isaac 的引擎切换接口互斥启用一个 USD 动力学后端；hybrid 使用显式联合仿真，
不会自动创建跨引擎碰撞或双向力反馈。接触代理、状态同步及模型构建由案例组件负责。
Newton USD 的 VBD 路径还要求安装对应 schema；本机 `6.2.0-rc.14` 缺少该 schema 时会明确报错，
柔体可接入 direct/hybrid 模型。[6.2 官方 Newton 说明](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/physics/newton_physics.html)。

完整配置见 [configs/isaacsim_world.toml](configs/isaacsim_world.toml)，可以设置引擎、求解器、
物理/渲染步长、重力、设备、窗口、渲染器、额外扩展、环境 USD 与背景位置。
默认配置仍走 stub；仅检查真实世界初始化可运行：

```bash
./app/python.sh -m pip install -e ".[dev]"
./app/python.sh -m real2sim2real.core.cli collect run -c configs/isaacsim_world.toml
```

装配后的执行顺序为 `reset()` → `play()` → `step(render=...)`，结束调用 `destroy_world()`。
`step()` 每次推进一个 `physics_dt`，`update()` 只刷新 UI/渲染；`rendering_dt` 设置 Kit 帧周期，
不会自动增加物理步数。USD 模型初始化包含 SDK 的 warmup 步，`physics_steps` 只统计应用显式调用。
`fast_shutdown=true` 沿用 Isaac 的快速退出方式，关闭 Kit 时退出进程；异常退出码为 1。

direct/hybrid 的组件在启动仿真前挂载 Newton 模型：

```python
sim_world.register_newton_stepper(
    runtime.step,   # 接收 dt 秒，完成一整步；hybrid 在此同步 PhysX 接触代理
    reset=runtime.reset,
    close=runtime.close,
)
```

该 runtime 从 `sim_world.config` 读取求解器、子步数、迭代数、重力等参数；世界只调度回调，
不重复积分模型。缺少 stepper 时 direct/hybrid 会在运行前报错。USD 路径由世界直接应用 Newton 配置。
已有案例配置使用同一个世界，但归档中的任务后端仍需逐一整理接入。

PhysicSolver 使用 `[[scene.cameras]]` 配置固定相机，使用 `[[robot.cameras]]` 配置机载相机，
可分别添加多项。相机由所属 Scene/Robot 创建，世界配置不含相机字段。
每项包含唯一的 `name`、`prim_path`、`pose_relative_to`、`pose` 和 `resolution`，
可选 `clipping_range`、`focal_length` 和 `horizontal_aperture`（镜头参数单位 mm）。
`pose.position` 单位为米，`pose.orientation_wxyz` 是相对参考 prim 的四元数；
使用 USD 相机坐标（-Z 朝前、+Y 朝上），相机路径须位于参考 prim 下以跟随运动。

六份 PhysicSolver 案例默认包含固定 `overview` 和腕部 `wrist` 两路 RGB 相机。
Newton 腕部相机挂在 `fr3_link7`，PhysX 挂在 `panda_link7`；偏离手掌侧面并看向夹爪中心。
Newton 在创建可视连杆后挂载相机，逐步继承实际求解后的腕部位姿。
录制按相机名生成 `observation.images.overview`、`observation.images.wrist` 等独立视频，
每路使用自身分辨率，共享关节数据帧号和时间戳；切换采集视角不会推进物理。
GUI 默认显示列表中的第一路，可在视口中切换到对应 Camera prim。
Featherstone 示例默认使用 `RayTracedLighting`、640×480 和 `rt_subframes=4`；
采集会等待视口实际完成渲染，并核对截图的相机矩阵与分辨率，不推进物理。
关闭 DLSS 时采用空间 FXAA，避免不同相机共享不合适的时域历史；无头 RGB 不含视口网格和坐标轴。
`samples_per_pixel_per_frame=64` 仅在切换为 `PathTracing` 时生效。
相机 `resolution` 决定视频分辨率，
`sim_world.width/height` 决定初始渲染尺寸；无相机列表时才使用 `job.recorder.width/height`。
完整示例见 [Featherstone 布料配置](configs/PhysicSolver_featherstone_vbd_pickplace_cloth.toml)。
`job.timeout_s` 是采集循环中包含渲染与采集的墙钟时限，不是仿真时长。双相机逐路采集时，
每路都会执行 `rt_subframes`；128 spp × 8 帧的旧设置曾在录到 16.68/18 秒时耗尽 1800 秒。
当前保留双路 15 fps，并移除采集之间不写入数据的重复渲染。日志每约 30 秒报告仿真进度、
已录帧数、预计剩余耗时及控制/物理/采集/I/O 分项耗时；这些耗时也写入 `result.json`。
需要手动提高采样时，应同时根据实测耗时设置墙钟预算。

该 Featherstone 配置现使用 `newton_cloth_franka` 场景及 Planner，参考
[NVIDIA cloth_franka 说明](https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton/)
及本机 Newton 1.6.1 的官方示例。运行前执行
`./app/python.sh scripts/prepare_official_newton_cloth.py`，从 Isaac Sim 官方发行包复制
`unisex_shirt.usd`、许可证和带 SHA-256/版本的 `provenance.json` 到独立的
`assets/PhysicSolver/Cloth/newton_official/`。原始衣服几何保持不变，不压扁或重建。
添加 `--verify-upstream` 可在线核对 GitHub 官方资产哈希；本次已核验完全一致。
衣服与关键帧整体平移到项目桌面，保留官方机器人/桌面的相对高度和 80 cm 工作区。

控制采用官方关键帧、雅可比伪逆与零空间速度控制；当前为前 7 个关键帧的一轮抓放，
`planner.keyframe_limit=33` 可执行完整参考序列。它是预设控制器，不是训练好的 RL 策略。
动作视频与观测位置对齐，`action` 明确记录实际关节速度（rad/s、m/s）。
Featherstone → VBD 保持官方单向耦合，不增加布料固定点或抓取附件。
衣服原始 USD 只有几何和显示颜色，没有完整织物材质。场景为它绑定显式哑光材质，
`cloth_roughness=0.92`、`cloth_specular=0.02`，并把仓库灯和桌面补光强度降为原来的
四分之一（`sim_world.background_light_scale`、`scene.fill_light_intensity/key_light_intensity`）。
这会减少塑料般的高光和过亮区域；它仍是均匀表面材质，没有织纹贴图或专用织物 BRDF。
碰撞粒子半径决定物理包络；默认官方参数为 8 mm，不能把检测到接触当作可视表面已经贴合。
`full_surface_contacts` 可增加网格边、面与刚体的接触；刚体网格需要 SDF，分辨率由
`contact_sdf_resolution` 控制。`keyframe_offset_m` 以米平移参考路径，
`open_finger_m/closed_finger_m` 控制单侧实际开度，便于匹配物理厚度。
`finger_velocity_gain` 控制夹指响应速度，实际命令受 URDF 速度上限约束；
减小碰撞半径后必须确认夹指在提起前已经闭合，不能只缩小厚度。
本次 1–4 mm 半径、边/面接触、闭合时序、弯曲刚度和摩擦对比中，薄接触方案仍发生中途滑脱。
示例继续保留通过完整抓放验收的 8 mm 参数；约 3 mm 的可见表面间隙尚未解决。
物理包络内的接触验收与可见表面零间隙是两个独立条件。
项目保留 `cloth_mass_kg=0.2`，按三角形面积分配质量；官方演示的 `density=0.02`
在当前厘米模型中对应约 141 kg，因此未直接沿用。验收报告只统计进入物理接触包络的接触，
排除检测余量内但未进入接触包络的候选；同时要求首轮转移阶段至少 95% 的帧保持双侧接触，
释放下落跟踪松开前被夹住的同一片衣服顶点，并检查提起及释放，
不将执行一轮抓放宣称为完成整件衣服的折叠。

引擎自由落体和“只渲染不积分”检查（每条命令独立启动 Kit）：

```bash
./app/python.sh -m pytest -q tests/test_isaacsim_modes.py
./app/python.sh scripts/validate_physics_modes.py --engine physx
./app/python.sh scripts/validate_physics_modes.py --engine newton --solver mujoco
./app/python.sh scripts/validate_physics_modes.py --engine newton --solver xpbd
./app/python.sh scripts/validate_physics_modes.py --engine newton --execution direct --solver vbd
./app/python.sh scripts/validate_physics_modes.py --engine hybrid --solver vbd
./app/python.sh scripts/validate_physics_modes.py --engine hybrid --solver vbd --cloth
```

## Franka 桌面抓取 Demo（历史归档）

以下保留历史设计与验收记录；旧配置和任务后端位于 `bak/rsr-bak/`，
旧资产准备与视频录制脚本已清理。当前采集使用上方 PhysicSolver 配置和脚本。

Demo 的世界统一选 `isaacsim`；scene、robot、planner、job 的后端名为 `franka_demo`；复用固定的
`BaseJob.run()` 生命周期。机器人固定在桌面上，执行接近、闭合、抬升、保持、释放与撤离。
从仓库根目录运行，资产位于 `assets/Demo`（复制资产和依赖，不改源文件）。
世界实现位于 [isaacsim.py](real2sim2real/simworld/isaacsim.py)，任务组件归档为
[scene](real2sim2real/scene/scene_franka_demo.py)、
[robot](real2sim2real/robot/robot_franka_demo.py)、
[planner](real2sim2real/planner/planner_franka_demo.py) 和
[job](real2sim2real/job/job_franka_demo.py)；
共享引擎模式由 [isaacsim.py](real2sim2real/simworld/isaacsim.py) 提供。

```bash
./app/python.sh -m pip install -e ".[dev]"

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

旧版录制流程依次启动六个案例（含统一 XPBD 抓取失败对照）。历史视频汇总在 `outputs/videos/`，
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
| SimWorld | `simworld/isaacsim.py` | `from real2sim2real.simworld.isaacsim import IsaacsimWorld` |

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


