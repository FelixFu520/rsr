# Real2Sim2Real

虚实闭环平台：**场景重建与数字孪生 → 仿真数据采集 → 策略训练 → 真机评测回流**。

当前仓库是可插拔的仿真编排框架：抽象基类 + Registry + stub 后端，真实仿真引擎（Isaac Sim）跑在 **唯一的 Python 环境**——Isaac 自带的 `python.sh`。

> Agent / 贡献者：请先读 [`AGENTS.md`](AGENTS.md)（目录约定、扩展后端、配置与 CLI 约定）。

当前整理状态：历史任务实现保留在 `bak/rsr-bak/`。主包包含基础框架、stub 通路、统一的 `IsaacsimWorld`、衣服/包裹抓放及 Pro-M6 双臂分拣案例。
下文旧 Franka / sorting 任务作为历史参考保留；新衣服案例使用下述独立配置。

Pro-M6 双臂包裹分拣的当前实现、复现命令、错位排查和最新交付见
[包裹分拣完整文档](docs/sorting_package.md)。

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

### Pro-M6 双臂包裹分拣

当前优先验证[完整流程配置](configs/PhysicSolver_mujoco_vbd_sorting_package_workflow.toml)：
原装机器人与夹爪，MuJoCo＋VBD，衣物 50 g、袋膜约 5.14 g，有效指端摩擦 2.0。
`require_bag_gap_quality=false` 将袋衣贴合间隙作为记录项；完整抓取、交接、放手、出带，
以及丢抓、拉伸、形状和穿透检查仍须通过。结果区分 `workflow_success` 与
`strict_quality_success`；流程成功不代表严格贴合质量已通过。
放置时在台面附近张爪并同步缓慢下降 64 mm，再抬起空夹爪；放手以无机器人接触、完整受目标台面
支撑判定，实际下落量仍记录，`minimum_release_drop_m=0` 允许受支撑的平稳放手。
本项目同时运行的 Isaac Sim 实例不超过两个，包含暂停的实例；建议先单实例完成物理验证，
再回放录制四视角 LeRobot v3。

```bash
./app/python.sh scripts/check_sorting_package.py --config configs/PhysicSolver_mujoco_vbd_sorting_package_workflow.toml --output outputs/PhysicSolver/sorting_native_workflow
./app/python.sh scripts/render_sorting_package.py outputs/PhysicSolver/sorting_native_workflow --output outputs/PhysicSolver/sorting_native_workflow_collection
./app/python.sh scripts/validate_sorting_package.py outputs/PhysicSolver/sorting_native_workflow_collection
```

[分拣配置](configs/PhysicSolver_mujoco_vbd_sorting_package.toml) 恢复旧任务的
**入料带 → 左臂抓取 → 平台放手交接 → 右臂抓取 → 出料带放手 → 摩擦输送出带**。
场景、机器人、Planner 和 Job 分别通过 `sorting_package` / `pro_m6_newton` 注册，
复用固定 Job 生命周期，不依赖运行时导入 `bak/`。左右臂放手撤离后返回安全停靠位；
第二次抓取读取同一物理模型的实时包裹状态，不恢复或压平网格。
阶段之间延续上一段的指令目标；闭合和松开默认只更新夹爪目标，避免把实测跟踪误差反复加进
下一段的笛卡尔轨迹。诊断同时记录实测 TCP 和指令 TCP；任务时钟由整数物理帧计算。
下放前根据实际质心修正横向落点；转运首帧、中点和末帧即时检查形状与袋衣间隙，
不可通过的候选提前保存诊断并停止。物理检查与缓存渲染脚本使用独立 Kit 进程，
依据本次运行标识核对结果和退出码，避免快速关闭跳过失败判断或误用旧结果。
`planner.transfer_profile="quintic"` 可将转运段改为起止速度、加速度均为零的五次曲线；
默认 `"cubic"` 保留原三次曲线。它只改变机器人运动目标，仍逐步求解实际关节和接触，
需要重新检查形状、袋衣间隙、拉伸和落点；在线日志输出各形状检查点的原始数值。
`planner.strain_limited_grip=true` 可启用仿真网格反馈：瞬时边长伸长超过
`grip_strain_target` 时，以 `grip_relief_rate_rad_s` 小幅松爪，最大松开量为
`grip_maximum_relief_rad`；低于目标减去 `grip_strain_hysteresis` 后，按
`grip_recovery_rate_rad_s` 恢复。默认死区 0.03、恢复速度为松开速度的十分之一。
左右臂独立反馈，抬升、保持、转运和下放阶段只调整真实关节目标，释放从最后的夹爪目标
连续打开。该控制使用仿真真实网格，不是仅靠机器人本体传感器的策略；不改写粒子状态或
放宽物理验收。诊断记录每帧的 `grip_relief_rad`，LeRobot 状态和动作仍为实际 16 维关节量。

机器人从原 Pro-M6 USD 导入，使用原生 MuJoCo CPU 求解关节位置驱动及 Robotiq mimic
等式约束，袋膜、衣物和刚软接触使用 CUDA VBD。机器人自碰撞沿用源资产的
`newton:selfCollisionEnabled=false`。源 USD 的碰撞近似先转换为凸包，再生成
接触 SDF；不增加隐藏夹块。源工位的每根夹指同时有金属网格和 30 × 7.5 × 70 mm 白色指垫碰撞盒。
高度与穿透验收包含两种碰撞体；不能只根据金属网格估算指尖。夹块闭合时比 ToolCenter
向前伸出约 36 mm，本任务设置 TCP 离台 48 mm，使夹块下缘离台约 12 mm。
主轴从 0.64 rad 闭合到提升后的 0.675 rad，对应约 12 → 4.8 mm 的实际夹块间隙，
没有增加隐藏夹块或把旧虚拟间隙当作实测几何。
MuJoCo mimic 的时间常数和阻抗写在配置及采集快照中。
`planner.hold_s` 可在提升后保持原关节目标静置，再开始水平转运；示例为 1 s。
这与 V7 在提升结束后静置评估形状的阶段相对应，转运期的原有形状门限保持不变。
`planner.grasp_pad_bottom_m` 可改用指垫底边目标：从原生连杆 FK 计算指垫相对 TCP 的
高度曲线，下降时按打开角度定位，闭合时同步补偿机械臂高度。校准使用独立 FK 缓冲，
不推进动力学或改写实际状态。可与 `open_gripper_rad=0.28`（名义开口约 81 mm）配合，
减少 Robotiq 闭合时向下扫过袋膜的运动。该选项默认不设置，实际关节跟踪误差仍需测量；
`trajectory.json` 另记录每帧实际指垫最低点相对抓取支撑面的高度 `pad_bottom_m`。
`planner.grasp_offset_xy_m` 可在世界 XY 中相对实时衣物质心选择抓取点；抓取、转运、下放
的名义 TCP 路径一起做关节限位预检，放置修正仍以实际质心和原目标平台中心为准。
机器人与夹爪结构固定为原 USD：30 × 7.5 × 70 mm 原生指垫，不添加扩展件，保留原质量及惯量。
`scene.garment_mass_kg` 默认 0.2，`scene.bag_mass_kg` 默认保留资产的袋膜质量；
可分别声明两者质量，各自的逐点质量按相同比例缩放，几何、UV 和弹簧保持原样。
`robot.initial_gripper_rad` 仅指定仿真开始前的夹爪姿态，同时按原 mimic 比例初始化从动关节；
后续姿态完全由 MuJoCo 积分求解。轻载诊断从 0.28 rad 打开状态开始，以减少启动阶跃。实际质量、原始质量与缩放比例写入结果；独立验收还读取 USD，核对两部分的声明质量及各自的质量分布，
不将轻载实验称为原 200 g 案例。
`planner.gripper_package_friction` 可单独设定原生内指与包裹的有效摩擦系数。
本机 Newton 1.6.1 按两侧材料系数的几何平均合成；运行时反算内指材料值，保留其它
机器人部位、袋衣自接触及支撑面的参数。默认不覆盖，此时有效系数为 1.4；设为 2.0
时，衣物侧仍为 1.4，内指侧约为 2.857。实际材料值与合成结果写入结果和诊断缓存。
摩擦实验仍须通过原拉伸、穿透、形状、贴合及完整分拣验收，参数需独立于实物标定记录。
[摩擦对照配置](configs/PhysicSolver_mujoco_vbd_sorting_package_friction.toml) 使用原装夹爪、
100 g 衣物和有效摩擦 2.0，并先夹紧再提升；目前仍为待验收实验。
刚软耦合接触检测使用 MuJoCo 推进后的刚体位姿和推进前的柔体状态；VBD 同时保留
刚体前后位姿，以计算接触与摩擦运动。
`planner.physics_origin_m` 可指定固定的局部求解原点；Newton 整体平移机器人、自由刚体、
静态支撑和粒子，控制目标、USD 及缓存仍使用世界坐标。这改变浮点条件而不改变相对几何
或添加外力；不同坐标设置的轨迹不保证逐位相同。默认 `[0,0,0]` 保留世界坐标求解。
每帧的控制目标在 32 个子步内保持恒定；CPU 桥接可批量上传逐子步的实际求解结果，
避免重复同步。所有 MuJoCo 积分、刚体碰撞和 VBD 接触仍逐子步求解，
`mujoco_control_per_frame=false` 可切回 Newton 原始桥接作对照。

本任务独立保存当前 `/home/fufa/Projects/package/output/simulation_v7` 到
`assets/PhysicSolver/Package/v7_sorting/`。已有 `v7/` 抓放资产保持原样；两份快照的网格和
逐点质量一致，源控制参数与验证记录有差异。衣物 200 g、袋膜约 20.54 g，保留原 UV、材质
及衣物内部被动弹簧，不固定粒子，不使用抓取附件。传送带用支撑面的切向接触位移产生摩擦，
不会直接给包裹写入平移或速度。

窄指垫的接触离散对照可使用 `scripts/refine_package_assets.py` 另存
`assets/PhysicSolver/Package/v7_sorting_refined/`，再设置 `scene.cloth_usd_path` 选择该资产。
默认局部细分将衣物／袋膜节点从 3300／1650 增至 5076／2538；保持原三角面的初始表面、
面密度、总质量和原蓬松弹簧锚点，节点质量按新三角面积重新集中。UV 与材质分区随父面映射。
派生资产的 `provenance.json` 记录源哈希、拓扑变化、USD 回读及封闭性检查；历史报告以
`SOURCE_` 文件名保留。它需要重新求解和验收，不继承原资产的结果。
验收同时检查全部新网格探针和原 V7 的材料采样点：原衣物节点使用原质量权重，原袋膜面中心
按材料坐标映射到细分后的表面，采样数量保持不变，两套结果都必须满足原门限。

```bash
# 独立快照已准备；在新机器上准备时选择对应的 V7 源目录。
./app/python.sh scripts/prepare_package_assets.py --target assets/PhysicSolver/Package/v7_sorting
# 可选的独立接触网格细分对照；不改写上面的原始快照。
./app/python.sh scripts/refine_package_assets.py
# 完整物理试验，不采集 RGB。
./app/python.sh scripts/check_sorting_package.py
# 从真实物理缓存渲染视频，避免重复求解。失败/截断缓存需显式加 --allow-failure。
./app/python.sh scripts/render_sorting_package.py outputs/PhysicSolver/sorting_physics_check --output outputs/PhysicSolver/mujoco_vbd_sorting_package
# 四路同步 LeRobot v3：全景、头部、左腕、右腕。
./app/python.sh -m real2sim2real.core.cli collect run -c configs/PhysicSolver_mujoco_vbd_sorting_package.toml
./app/python.sh scripts/validate_sorting_package.py outputs/PhysicSolver/mujoco_vbd_sorting_package
```

缓存渲染读取物理运行的配置快照及真实粒子、刚体位姿、关节观测和动作；渲染期间保持物理
暂停，不插值或重新求解轨迹。数据集记录 `observation_generation=rendered_from_recorded_physics`
及源轨迹 SHA256，验收仍以原始物理结果为准。

LeRobot 的状态和动作均为 16 维：左臂七关节、右臂七关节、左右夹爪主轴，全部使用 rad。
状态读取 MuJoCo 实际求解结果，动作记录同一物理时刻下发的位置目标。四路视频共享 15 fps
帧号和时间戳；诊断另外保存每个 60 Hz 物理帧的两个网格、完整关节状态及碰撞体位姿。
独立校验会完整解码各路 H.264，逐帧核对视频/Parquet/物理状态，并生成 `qa/keyframes.jpg`
、`qa/synchronized_four_view.mp4` 和 `qa/physical_audit.json`。
腕部虚拟相机采用侧向外置观察位置，避开原生指节遮挡。回放旧缓存时可添加
`--camera-poses configs/PhysicSolver_sorting_camera_poses.json`，仅覆盖已命名相机的光学位姿。
回放保存原始配置 SHA256 和相机覆盖记录；独立校验拒绝其他配置变化，物理轨迹保持原样。

验收分别检查两次提起、整个转运期的双侧接触及两个网格离台、平台/出带放手、完整包裹支撑
范围、标签朝上及真实出带位移。物理检查还包括质量、有限状态、三角面塌陷、边长伸长、
台面/指垫穿透、子步接触容量、袋内包含关系，以及转运期衣物形状 RMS 和袋衣间隙。
独立视频复核覆盖转运期间每个视频帧的形状与间隙，并保留物理转运检查点。
标签方向使用原 FrontPhotograph 中标签 UV 区域的真实网格法线。
失败轨迹保留为 `diagnostic`，只有所有检查通过才标记为 `demonstration`；
验收脚本遇到物理失败返回非零退出码。当前尚未产出通过全部验收的完整双臂示范，
现有回放均为诊断数据，不继承旧抓放数据或源 V7 的成功标记。平指垫对照仍发生过
滑脱、局部拉伸或袋衣间隙越限；平台释放、右臂交接及出带不能据此声明已经验证。
[反馈诊断配置](configs/PhysicSolver_mujoco_vbd_sorting_package_diagnostic.toml)
使用原生 30 mm 指垫、100 g 衣物和局部细分，继承主配置且保持原形状与接触验收门限。
它是待验证的轻载实验入口；历史加宽指垫数据不符合当前机器人结构要求，不能作为合格示范。
诊断时可设 `planner.early_stop_on_shape_failure=false`，继续观察形状或袋衣间隙失败后的
交接与出带过程。该选项只关闭这两项的提前终止；原始测量、最终验收和数据标签保持不变，
穿透、过度拉伸、丢抓及不安全落点仍提前停止。`result.json` 的 `failed_checks` 列出未通过项。

MuJoCo → VBD 沿用参考案例的单向刚体驱动柔体耦合，包裹反力不回馈机器人关节。
这能验证所声明模型下的抓持、变形和释放，不能据此验证真实夹爪额定负载。
几何检查只覆盖保存的物理帧，不构成连续时间无穿透的证明。

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


