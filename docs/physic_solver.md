# 物理求解器验证

更新日期：2026-11-12。

## 1. 任务介绍

固定在桌面上的 Franka 机械臂执行一次完整抓放：接近 → 下探 → 闭合 → 抬升 → 保持 → 横向转运 →
下放 → 松开 → 撤离 → 静置。被抓物体分三类：

- **衣服**：一件 0.2 kg 的衬衫（质量按三角面积分配），无固定顶点、无抓取附件。
- **刚体方块**：0.1 kg 方块，全部由 PhysX 求解，作为刚体对照。
- **自定义柔性包裹**：袋膜（约 20.54 g）+ 内部衣物（200 g）两个物理网格，衣物内部有被动弹簧。

任务目的不是训练策略，而是在同一套 `IsaacsimWorld` 和固定 Job 生命周期下，比较不同
“机器人求解器 + 接触/柔体求解器”组合能否完成真实接触驱动的抓放，并把每次成功的过程采集为
LeRobot v3 数据。每个组合的验收都基于实测几何（提升高度、双侧接触、释放回落、落点误差、
拉伸、穿透），只完成动作序列不算成功。

### 1.1 求解器组合

配置统一放在 `configs/physic_solver/`，文件名为 `<机器人求解器>_<接触/柔体求解器>_<任务>.toml`。
PhysX 的实际求解器写作 `tgs`；引擎归属由 `sim_world.physics_engine` 指定。

| 配置 | 世界模式 | 机器人 | 机器人 / 物体求解器 | 物体 | 预期 |
|---|---|---|---|---|---|
| `featherstone_vbd_pickplace_cloth` | Newton direct | FR3 URDF，速度控制 | Featherstone / VBD | 官方 Newton 衬衫，官方关键帧控制器 | 通过 |
| `mujoco_vbd_pickplace_cloth` | Newton direct | FR3 URDF | MuJoCo / VBD | 衣服 | 通过 |
| `xpbd_vbd_pickplace_cloth` | Newton direct | FR3 URDF | XPBD / VBD | 衣服 | 通过 |
| `tgs_vbd_pickplace_cloth` | hybrid | Panda USD + Lula IK | PhysX TGS / Newton VBD | 衣服 | 通过 |
| `tgs_tgs_pickplace_rigid` | PhysX | Panda USD + Lula IK | TGS / TGS | 刚体方块 | 通过 |
| `mujoco_vbd_pickplace_package` | Newton direct | FR3 原装短指，位置控制 | MuJoCo / VBD |  柔性包裹 | 通过 |
| `featherstone_vbd_pickplace_package` | Newton direct | FR3 + 长夹块 | Featherstone / VBD | 柔性包裹 | 对照组，未通过验收 |
| `xpbd_xpbd_pickplace_cloth` | Newton direct | FR3 URDF | 统一 XPBD（机器人与衣服同一求解器） | 衣服 | 实验组，未通过验收 |

求解器区别：

- **PhysX TGS**：迭代求解关节、刚体接触和摩擦约束。
- **Featherstone**：用关节坐标计算机械臂动力学；本任务只求解机器人。
- **MuJoCo**：求解机器人关节、驱动和刚体接触，不求解衣服。
- **VBD**：逐顶点局部能量优化，处理衣服/袋膜的拉伸、弯曲、接触和自碰撞。
- **XPBD**：带柔顺度的位置约束，可同时处理刚体和柔体；统一 XPBD 的衣服用距离弹簧，
  与 VBD 的三角面材料模型不同。

各组合除了求解器，还有机器人模型、控制方式、重力补偿、抓取高度偏移和子步预算的差异
（见各配置文件注释）。因此结果只说明“该配置能否稳定抓放”，不能直接给求解器精度排名。
统一 XPBD 未通过，也不能证明 XPBD 算法本身不适合抓衣服。

### 1.2 世界模式

`sim_world.physics_engine` 与 `newton_execution` 决定动力学归属：

| 值 | 动力学执行方式 |
|---|---|
| `physx` | USD 场景全部由 PhysX 积分 |
| `newton` + `newton_execution="direct"` | Planner 自建 Newton 模型负责全部动力学，Kit 时间轴保持停止，只用于显示 |
| `newton` + `newton_execution="usd"` | Isaac 官方 Newton USD 后端（`mujoco` / `xpbd`）；VBD 需要本机缺失的 `NewtonVbdSceneAPI` schema |
| `hybrid` | PhysX 推进一步后调用 Newton stepper；PhysX 手指/手掌位姿作为 VBD 的运动学碰撞代理 |

direct / hybrid 的组件在启动仿真前注册 stepper：
`sim_world.register_newton_stepper(runtime.step, reset=..., close=...)`。缺少 stepper 时世界在
运行前报错。hybrid 不会自动创建跨引擎碰撞或双向力反馈。

## 2. 当前结果

本节数据来自 2026-11-11 用整理后代码重新采集的结果（`collect_physic_solver_cases.py`，每个组合一个 Kit
进程）。汇总见 [collection_summary.json](../outputs/physic_solver/collection_summary.json)。

| 组合 | 结果 | 帧数 / 时长 | 关键实测 | 目录 |
|---|---|---|---|---|
| `featherstone_vbd_pickplace_cloth` | 通过 | 270 帧 / 18.0 s | 接触片抬升 100.6 mm，释放回落 83.4 mm，转运双侧接触 100% | [outputs/physic_solver/featherstone_vbd_pickplace_cloth](../outputs/physic_solver/featherstone_vbd_pickplace_cloth/) |
| `mujoco_vbd_pickplace_cloth` | 通过（见下方说明） | 323 帧 / 21.5 s | 保持最低抬升 251.6 mm，回落 239.3 mm，落点误差 44.7 mm | [outputs/physic_solver/mujoco_vbd_pickplace_cloth](../outputs/physic_solver/mujoco_vbd_pickplace_cloth/) |
| `xpbd_vbd_pickplace_cloth` | 通过 | 323 帧 / 21.5 s | 抬升 261.1 mm，回落 164.4 mm，落点误差 52.1 mm | [outputs/physic_solver/xpbd_vbd_pickplace_cloth](../outputs/physic_solver/xpbd_vbd_pickplace_cloth/) |
| `tgs_vbd_pickplace_cloth` | 通过 | 323 帧 / 21.5 s | 抬升 253.4 mm，回落 191.1 mm，落点误差 9.1 mm | [outputs/physic_solver/tgs_vbd_pickplace_cloth](../outputs/physic_solver/tgs_vbd_pickplace_cloth/) |
| `tgs_tgs_pickplace_rigid` | 通过 | 323 帧 / 21.5 s | 抬升 241.3 mm，回落 243.1 mm，落点误差 2.6 mm | [outputs/physic_solver/tgs_tgs_pickplace_rigid](../outputs/physic_solver/tgs_tgs_pickplace_rigid/) |
| `mujoco_vbd_pickplace_package` | 通过 14/14 | 185 帧 / 12.3 s | 衣物/袋膜质心提升 238.5/235.2 mm，转运双侧接触 100%，回落 111.0 mm，落点误差 51.4 mm，最大边长比 1.234 | [outputs/physic_solver/mujoco_vbd_pickplace_package](../outputs/physic_solver/mujoco_vbd_pickplace_package/) |
| `featherstone_vbd_pickplace_package` | 未通过（对照组） | 195 帧 / 13.0 s | 质心提升 139 mm、转运接触 100%；撤离后静置阶段（约 12 s）包裹突然加速飞离桌面，`placed_near_goal`、`table_penetration_bounded` 失败 | [outputs/debug/physic_solver/featherstone_vbd_pickplace_package](../outputs/debug/physic_solver/featherstone_vbd_pickplace_package/) |
| `xpbd_xpbd_pickplace_cloth` | 未通过（实验组） | 323 帧 / 21.5 s | 衣服没有被提起（保持期抬升 -5.5 mm），导出为 `diagnostic` | [outputs/debug/physic_solver/xpbd_xpbd_pickplace_cloth](../outputs/debug/physic_solver/xpbd_xpbd_pickplace_cloth/) |

正式目录 `outputs/physic_solver/` 中的 6 个 `lerobot_v3/` 都是 `demonstration`，已通过
`validate_dataset`（帧号/时间戳、有限关节值、视频逐帧解码、H.264、分辨率、episode 时间范围）。

**`mujoco_vbd_pickplace_cloth` 的结果在多次运行间波动。** 第一次批量采集时落点误差 81.3 mm，超过
80 mm 门限，被移到 debug。为区分整理代码引入的回归和运行间差异，用整理前的提交（`96eec55`）和整理后
代码各重复运行：

| 代码 | 落点误差 | 结果 |
|---|---:|---|
| 整理后，第 1 次（批量采集） | 81.3 mm | 未通过 |
| 整理后，第 2 次 | 72.3 mm | 通过 |
| 整理后，第 3 次（当前正式数据） | 44.7 mm | 通过 |
| 整理前 `96eec55`，第 1 次 | 21.9 mm | 通过 |
| 整理前 `96eec55`，第 2 次 | 84.2 mm | 未通过 |

两套代码都在 22–84 mm 之间波动，抬升、回落和拉伸几乎不变，说明这是 GPU 求解的运行间差异
（衣服松开后的落点对微小差异敏感），不是整理引入的回归。正式目录保存第 3 次运行，运行记录见
[run_note.json](../outputs/physic_solver/mujoco_vbd_pickplace_cloth/run_note.json)；其余运行留在
`outputs/debug/physic_solver/`。该组合离门限余量小，重新采集可能需要多次运行。

`featherstone_vbd_pickplace_package` 另跑了两次对照：整理前代码提升约 109 mm 后滑脱、转运双侧接触 0%、
最大边长比 1.362；整理后代码第 2 次提升约 127 mm 后滑脱、转运双侧接触 18%、边长比 1.354。两者都与早期记录
（约 115 mm 后滑脱）一致；批量采集那次的“抓稳后静置失稳”是同一失败组合的另一种表现，不是整理引入的回归。

## 3. 组成

| 模块 | 后端名 | 实现 |
|---|---|---|
| SimWorld | `isaacsim` | [simworld/isaacsim.py](../real2sim2real/simworld/isaacsim.py) |
| Scene | `pickplace_cloth` / `pickplace_rigid` / `newton_cloth_franka` / `pickplace_package` | [scene/scene_physic_solver.py](../real2sim2real/scene/scene_physic_solver.py) |
| Robot | `franka_newton` / `franka_physx` / `franka_package_pads` | [robot/robot_physic_solver.py](../real2sim2real/robot/robot_physic_solver.py) |
| Planner | `pickplace_cloth` / `newton_cloth_franka` / `pickplace_package` | [planner/planner_physic_solver.py](../real2sim2real/planner/planner_physic_solver.py) |
| Job | `pickplace_cloth` | [job/job_physic_solver.py](../real2sim2real/job/job_physic_solver.py) |
| 辅助函数 | — | [utils/physic_solver_utils.py](../real2sim2real/utils/physic_solver_utils.py)：hybrid VBD 步进、移动坐标系 VBD、袋膜包含检查 |

`planner_physic_solver.py` 按依赖顺序包含四部分：

1. `PickPlaceClothPlanner` + `NewtonClothRuntime`：通用抓放阶段机、URDF 雅可比 IK、
   Featherstone / MuJoCo / XPBD + VBD 模型构建、hybrid 代理插值、USD 显示同步和验收。
   PhysX 刚体方块也使用这个 Planner，此时不创建 Newton 运行时。
2. `OFFICIAL_KEYFRAMES`：Newton 1.6.1 官方 `example_cloth_franka.py` 的关键帧。
3. `OfficialClothPlanner` + `OfficialClothRuntime`：官方关键帧 + 雅可比伪逆 + 零空间速度控制。
4. `PackagePlanner` + `PackageRuntime`：V7 双网格包裹、`MovingFrameVBD`、逐子步接触容量审计。
   分拣任务的 `SortingRuntime` 继承 `PackageRuntime`。

当前没有使用 cuRobo：Newton 机器人使用本文件的 URDF IK，PhysX 机器人使用 Lula IK；
TCP 轨迹由预设阶段插值生成，不做避障路径搜索。

### 3.1 脚本

| 脚本 | 用途 |
|---|---|
| [collect_physic_solver_cases.py](../scripts/physic_solver/collect_physic_solver_cases.py) | 串行采集全部 8 个组合并校验；未通过的组合移到 `outputs/debug/physic_solver/` |
| [prepare_physic_solver_assets.py](../scripts/physic_solver/prepare_physic_solver_assets.py) | 从 `assets/bak/Demo` 复制桌子、衣服、方块、Franka 等到 `assets/PhysicSolver` |
| [prepare_official_newton_cloth.py](../scripts/physic_solver/prepare_official_newton_cloth.py) | 从 Isaac Sim 自带 Newton 复制官方 `unisex_shirt.usd`、许可证和来源记录 |
| [prepare_package_assets.py](../scripts/physic_solver/prepare_package_assets.py) | 复制 V7 包裹快照；`--source/--target` 也用于分拣任务的 V10 快照 |
| [check_package_pickplace.py](../scripts/physic_solver/check_package_pickplace.py) | 包裹抓放只跑物理、不录 RGB，用于低成本调参 |
| [validate_package_pickplace.py](../scripts/physic_solver/validate_package_pickplace.py) | 不启动 Kit，复核包裹数据集的帧对齐、双相机解码和粒子几何 |
| [validate_physics_modes.py](../scripts/physic_solver/validate_physics_modes.py) | 各世界模式的自由落体与“只渲染不积分”冒烟测试，每次一个 Kit 进程 |

### 3.2 资产

所有资产放在 `assets/PhysicSolver/`（git 忽略，约 578 MB），全部是独立副本，只引用该目录内的文件，
整个目录可以单独打包发给别人：

| 目录 | 内容 |
|---|---|
| `Table/`、`Cube/`、`Cloth/` | 桌子、刚体方块、项目衣服 |
| `Cloth/newton_official/` | 官方 Newton 衬衫 + `LICENSE.md` + 带 SHA-256 的 `provenance.json` |
| `Robot/franka_emika_panda/` | FR3 URDF 与网格（Newton） |
| `Isaac/Robots/FrankaRobotics/FrankaPanda/` | Panda USD（PhysX） |
| `Isaac/Environments/Simple_Warehouse/` | 背景仓库 |
| `Package/v7/` | V7 包裹：Bag、Garment 两个网格，原始 UV/材质、逐点质量和被动弹簧 |

## 4. 运行

```bash
./app/python.sh -m pip install -e ".[dev]"
# 资产已在 assets/PhysicSolver；需要重新准备时：
./app/python.sh scripts/physic_solver/prepare_physic_solver_assets.py
./app/python.sh scripts/physic_solver/prepare_official_newton_cloth.py
./app/python.sh scripts/physic_solver/prepare_package_assets.py

# 串行采集全部 8 个组合（每个组合一个 Kit 进程），汇总写到 outputs/physic_solver/collection_summary.json
./app/python.sh scripts/physic_solver/collect_physic_solver_cases.py
# 只采一个组合
./app/python.sh scripts/physic_solver/collect_physic_solver_cases.py mujoco_vbd_pickplace_cloth
# 或直接运行单个配置
./scripts/rsr collect run -c configs/physic_solver/mujoco_vbd_pickplace_cloth.toml

# 包裹：先只跑物理，再复核数据集
./app/python.sh scripts/physic_solver/check_package_pickplace.py \
    --config configs/physic_solver/mujoco_vbd_pickplace_package.toml \
    --output outputs/debug/physic_solver/package_physics_check
./app/python.sh scripts/physic_solver/validate_package_pickplace.py \
    outputs/physic_solver/mujoco_vbd_pickplace_package

# 世界模式冒烟测试
./app/python.sh -m pytest -q tests/test_isaacsim_modes.py
./app/python.sh scripts/physic_solver/validate_physics_modes.py --engine physx
./app/python.sh scripts/physic_solver/validate_physics_modes.py --engine newton --solver mujoco
./app/python.sh scripts/physic_solver/validate_physics_modes.py --engine hybrid --solver vbd --cloth
```

已有 `lerobot_v3/` 的输出目录拒绝覆盖；重新采集要换 `job.output_dir`，例如
`RSR_JOB__OUTPUT_DIR=outputs/debug/physic_solver/run2`。无头运行是默认值；打开 GUI 用
`RSR_SIM_WORLD__HEADLESS=false RSR_JOB__KEEP_OPEN=true`。

## 5. 输出与数据格式

```text
outputs/physic_solver/
├── collection_summary.json        # 每个组合的采集状态、验收结果、日志路径
├── capture_logs/<case>_<UTC>.log
└── <case>/                        # 只保留通过验收的组合
    ├── lerobot_v3/
    │   ├── meta/info.json, stats.json, tasks.parquet, episodes/chunk-000/file-000.parquet
    │   ├── meta/task_outcome.json  # 物理验收结果 + 生效配置
    │   ├── data/chunk-000/file-000.parquet
    │   └── videos/observation.images.{overview,wrist}/chunk-000/file-000.mp4
    ├── result.json                # 实测验收与分项耗时
    ├── validation.json            # 数据集独立校验
    └── diagnostics/<run>/         # configuration.json、result.json、trajectory.jsonl（包裹另有 particles.npz）
outputs/debug/physic_solver/<case>/  # 未通过验收的组合（含 diagnostic 数据集）
```

- 15 fps 同步录制 `observation.images.overview`（固定全景）、`observation.images.wrist`（腕部）、
  `observation.state`、`action`，两路均为 640×480 H.264。
- `observation.state` 是实际关节位置：前 7 维臂关节 rad，后 2 维手指开度 m。
- `action` 是同一帧下发的关节目标。官方衣服控制器（`featherstone_vbd_pickplace_cloth`）记录的是
  实际下发的关节速度（rad/s、m/s）。
- 采集只刷新渲染，不额外推进物理；多相机逐路采集时核对物理步数不变。
- 失败轨迹默认不导出；`job.recorder.save_failed_episodes=true` 时导出为
  `success=false`、`dataset_use="diagnostic"`，采集脚本会把它移到 debug 目录。

## 6. 验收

衣服 / 方块（`PickPlaceClothPlanner.result`）：保持阶段已观测、目标抬升 ≥ `minimum_lift_m`、
保持期贴近夹爪、松开后回落 ≥ `minimum_release_drop_m`、回落后受桌面支撑、落点误差 ≤
`maximum_place_error_m`、无附件；衣服另检查无固定顶点、确实变形（形状 RMS > 3 mm）、
边长拉伸 p99 < 1.5。

官方衬衫（`OfficialClothPlanner.result`）：关键帧序列完成、状态有限、出现双侧接触、首轮转运
≥ 95% 帧保持双侧接触、接触片抬升、松开后同一片顶点回落、无固定顶点、无附件。接触只统计进入
物理接触包络的点，不统计检测余量内的候选。

V7 包裹（`PackagePlanner.result`，14 项）：序列完成、状态有限、双侧接触、衣物与袋膜分别提升、
转运期双侧接触 ≥ 95%、释放回落、落点误差、落在桌面上、桌面穿透、表面拉伸 < 1.3、无固定顶点、
十个检查时刻衣物顶点与面中心都在封闭袋膜内（1 mm 容差）、子步接触容量未溢出。

## 7. 遇到的问题与解决方案

| 问题 | 原因 | 处理 |
|---|---|---|
| XPBD 组合的关节观测不变 | XPBD 用最大坐标（body 位姿），不更新 `state.joint_q` | 用 `eval_ik` 写入独立观测缓冲区，再按指令分支展开转角；采集不改求解器状态 |
| XPBD 步进会改写上一帧刚体位姿 | XPBD 把输入 body 缓冲区当作 Jacobi 临时空间，而 VBD 外部刚体路径要用未改动的上一帧位姿计算手指位移和接触摩擦 | XPBD 步进前复制、步进后恢复 `body_q/body_qd` |
| 双相机采集超时（1800 s 只录到 16.68/18 s） | 128 spp × 8 子帧，且每次采集后还有一次不写数据的重复渲染 | 物理步进改为 `render=False`，只保留采集渲染；日志每 30 s 报告进度和控制/物理/采集/I/O 分项耗时 |
| 多相机交替采集的画面一致性 | 不同相机之间不能共享时域抗锯齿历史；视口叠加层会进入截图 | 关闭 DLSS 时用空间 FXAA；等待实际渲染完成并核对相机矩阵与分辨率；无头 RGB 不含视口网格和坐标轴 |
| 衣服像塑料、画面过亮 | 官方衬衫 USD 只有几何和显示颜色 | 绑定哑光材质（roughness 0.92、specular 0.02），仓库灯与补光降为 1/4 |
| 官方演示密度不可直接用 | `density=0.02` 在厘米模型中对应约 141 kg | 保留 `cloth_mass_kg=0.2`，按三角面积分配 |
| 夹指与衣服之间约 3 mm 可见间隙 | 粒子碰撞半径 8 mm 构成物理包络 | 试过 1–4 mm 半径、边/面接触、闭合时序、弯曲刚度和摩擦，薄接触都在转运中滑脱；保留 8 mm。接触验收与可见零间隙是两个独立条件，后者未解决 |
| 指尖在桌面以下闭合或抓空 | FR3 URDF 与 Panda USD 的几何、驱动增益不同 | 按组合设置 `grasp_offset_m` 的 Z：XPBD+VBD 6 mm，MuJoCo+VBD 与统一 XPBD 10 mm，hybrid 12 mm 并提高机械臂驱动增益 |
| hybrid 下 VBD 需要手指碰撞体 | VBD 模型里没有 PhysX 刚体 | 读取 PhysX 实测手指/手掌位姿，作为 Newton 运动学碰撞盒，逐子步插值位姿和速度 |
| Panda 指垫位置左右镜像 | USD 镜像的是几何，不是 body 坐标系 | 代理盒偏移分别取 ±Y 7.58 mm |
| 原装短指抓 V7 包裹滑脱（Featherstone） | 速度控制 + 短指在中央/边缘试抓都滑脱 | 新增 `mujoco_vbd_pickplace_package`：MuJoCo 位置控制 + 原装短指 + 中央抓点 + 慢抬升/转运 + `align_place_to_com` 按实测衣物质心修正落点，通过全部 14 项 |
| 长夹块对照仍失败 | 32 子步/28 迭代下提升约 115 mm 后滑脱，转运双侧接触约 2.25%，拉伸 1.422 | 保留为 `diagnostic` 对照组，不当作原装夹爪数据 |
| VBD 自接触界限随包裹整体平移变得保守 | DAT 截断按世界坐标计算 | `MovingFrameVBD` 先减去质量加权公共位移、截断后再加回；相对轨迹不变，并逐子步记录接触容量峰值 |
| Newton USD 路径选 VBD 直接报错 | Isaac Sim 6.2 RC 未提供 `NewtonVbdSceneAPI` schema | 柔体统一走 direct / hybrid，USD 路径只用于 MuJoCo / XPBD 冒烟测试 |
| 同一配置多次运行结果不同 | Newton VBD / MuJoCo（GPU）求解有运行间差异，衣服松开后的落点对微小差异敏感（`mujoco_vbd_pickplace_cloth` 落点误差 22–84 mm） | 不放宽门限；采集脚本把未通过的运行移到 `outputs/debug/physic_solver/`，正式目录只放通过全部验收的运行；用整理前代码复跑确认不是回归 |
| 统一 XPBD 衣服抓不住 | Newton 1.6.1 XPBD 不支持布料自碰撞；原始粒子排斥没有网格邻接过滤 | 默认关闭粒子排斥、均匀分配质量；保留为实验组，失败正常返回非零 |

## 8. 局限

- 既有衣服及包裹基准采用单向耦合：机器人推动柔体，柔体反力不回馈机械臂。新增包裹矩阵可显式选择双向body代理，详见第9节。单向数据能验证动作流程、
  形变和持续抓持，不能评价真实夹爪的额定负载或限力性能。
- Newton 1.6.1 的 Featherstone、XPBD 不执行 `joint_effort_limit`，`finger_max_force` 只在
  PhysX / MuJoCo 路径生效。
- V7 长夹块密度 1000 kg/m³（每块约 70.56 g）是仿真假设，未做硬件标定。
- 几何检查只覆盖保存的物理帧，不构成连续时间无穿透的证明。
- 每个组合只采一条固定初始状态的 episode。

## 9. package V7–V10 迁移矩阵与摇晃、附着对照

新增实现延续本任务已有的 scene/robot/planner/job/utils 文件，没有拆成独立任务。
源项目 `/home/fufa/Projects/package` 只读，资产复制到 `assets/PhysicSolver/Package/`，不依赖源脚本运行。
物理引擎图的整理、求解器能力边界和选型说明见 [physics_engines.md](physics_engines.md)。

### 9.1 迁移对照与验收口径

| 来源／新增要求 | 本仓库配置／实现 | 迁移范围 |
|---|---|---|
| V7 首次可保持的包裹基准 | `package_cases/v7_center.toml` | 保留双网格、VBD 材料和内部被动弹簧；矩阵采用源 V7 的显式长夹块与 MuJoCo 位置控制 |
| V8 抓取位置／角度 | `v8_offset_x35`、`v8_offset_y55`、`v8_yaw45`、`v8_yaw90`、`v8_offset_tilt10` | 相对包裹抓点35/55mm；45°优化至X35；90°采用V7长夹垫／Y175mm边缘、59.5mm抓高、零预紧及双向body代理；组合偏航30°／俯仰10°由原(-25,40)mm改为X35；五项均独立通过，原位置失败保留，非单角度对照 |
| V9 内容物重量 | `v9_mass100g`、`v9_mass200g`、`v9_mass350g`、`v9_mass500g` | 按原始逐点比例归一化衣物总质量；袋膜独立计重；四组统一X35mm抓点 |
| V10 柔性包裹种类 | `v10_station_clothing`、`v10_station_green`、`v10_pink_tshirt`、`v10_frosted_top`、`v10_black_hoodie` | 五款自包含 USD、UV/材质；绿色与原分拣站款源几何相同、外观不同，在本项目仍分别求解 |
| 摇晃夹爪 | `shake_none.toml` | 转运后2s、1.5Hz；XYZ 振幅15/10/8mm、偏航12°、俯仰6°；平滑包络在两端位移／速度为零 |
| 包裹是否附着夹爪 | `shake_spring_x35.toml` 对照 `shake_none.toml` | 同一X35mm抓点，仅改变附着模式；另保留已通过的中央抓点`shake_spring.toml` |

V10 的两个刚性纸箱分别由 `mujoco_rigid_v10_box01.toml` 和 `mujoco_rigid_v10_box02.toml` 接入。
保留201×201×141mm／126×201×141mm原尺寸、350g／250g总质量和贴图；使用显式260mm行程、90mm长夹指。
`pickplace_carton` 场景／规划／Job 与 `franka_carton_gripper` 均放在本任务文件；MuJoCo 刚体接触为双向。
碰撞采用源项目同样的刚性包围盒，不模拟纸箱压溃。夹块及滑轨明确可见；它不是原装 Franka 夹爪。
机械臂按长夹指修正TCP和雅可比，放置前求预抓取初始关节，避免初始姿态扫过箱体。

源 V7 完整 USD 独立复算形状 RMS 曾超过原目标，V8 的45°和组合倾斜工况提前失持，V9/V10也有贴合或应变失败。
这些是源项目既有结论；本项目每个新配置重新求解，不能继承源“成功”标签。

本轮从复制后的USD独立读取的衣物／外袋边界与质量如下；所有网格父级世界变换均为单位变换，
导入没有遗漏额外平移或缩放。每款均为3300个衣物节点、1650个袋膜节点，但尺寸与质量不同。
原始盘点保留在 `outputs/debug/physic_solver/migration_logs/package_mesh_inventory.jsonl`。

| 资产 | 外袋XYZ尺寸（mm） | 衣物质量（g） | 外袋质量（g） |
|---|---:|---:|---:|
| V7 | 300×400×55.0 | 200 | 20.542 |
| 分拣站衣物／绿色款 | 252.5×321.1×45.9 | 200 | 13.908 |
| 粉色T恤 | 254×330.2×38.0 | 150 | 13.926 |
| 磨砂上衣 | 240×300×45.0 | 180 | 12.436 |
| 黑色连帽衫 | 349.2×450.9×80.0 | 450 | 28.254 |

V10黑色款初始TCP高度按外袋最高点相对V7的差值设为95.4mm。
该组合的首次完整求解、数据和视觉验收已通过：衣物／袋膜抬升257.4／255.4mm，
转运双侧接触100%、边长比1.11287、台面穿入0、释放回落100.5mm、落点误差24.2mm，无附着。
[黑色款正式结果](../outputs/physic_solver/package_cases/v10_black_hoodie/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v10_black_hoodie/validation.json)和
[关键帧](../outputs/physic_solver/package_cases/v10_black_hoodie/qa/keyframes.jpg)均通过，
双相机222帧／15fps，14.8s物理墙钟耗时27.0min；长夹块和源BLACKHOODIE贴图在腕部画面可见。
薄款不能简单照搬相同夹入深度，否则夹块接近桌面、可能夹住全部下层。
分拣站衣物调试后，粉色、磨砂与绿色款最初测试原装FR3指垫的中央抓点／22mm TCP／单指2.5mm闭合／
12.3s基线；各款分别求解，后续宽夹垫候选见下表。
它们是几何选取的待验证起点，不是已通过的结论；
磨砂款的原装指垫／22mm／5mm夹缝首次运行在1.867s闭合初段达到边长比1.30629，按原门限拒绝，
原始失败在 `package_migration/v10_frosted_top/20261112T200151_572773Z`。
下一候选仅将夹缝增至10mm，其余材料、质量和控制保持，仍须完整重跑。
10mm夹缝将夹紧拉伸保持在1.2917，但最高抬升110.8／118.2mm后失持，4.8s两部分已回到台面，拒绝。
失败前缀在 `package_migration/v10_frosted_top/20261112T223742_642536Z`。
曾准备的短指X35候选未执行，后续改用下文源宽夹垫的配套几何与控制组合。
[短指失败工况的实际网格／碰撞盒截面](../outputs/debug/physic_solver/migration_logs/thin_native_grasp_sections.png)
将45°／X35、粉色中央和磨砂中央的闭合、最高抬升及拒绝帧放在一起；
手指编号从当前Newton的URDF构建器独立核对。截面以实际指间中心随动，取距中面25mm以内的点，
可区分局部夹持、滑落和下放时仍持住的情况，但二维投影不代替三维穿透验收。
粉色款的原装指垫／22mm／5mm夹缝中央抓点曾抬升231.3／226.9mm，但5.63s时已失持回到台面，拒绝。
失败前缀位于同一UTC运行编号的 `v10_pink_tshirt`；下一候选只沿X偏移35mm，保持其余控制。
该X35短指候选最高仅55.0／50.9mm后失去双侧接触，抬升阶段即被拒绝，
保留在 `package_migration/v10_pink_tshirt/20261112T223742_642536Z`。
绿色款原装指垫／22mm／5mm夹缝的独立求解在3.367s提升初段达到边长比1.30150，拒绝，
失败保留在同一UTC运行编号的 `v10_station_green`；后续采用下文宽夹垫组合。
即使与分拣站衣物的源几何相同，也不能继承其成功或将单次运行差异归因于颜色。

宽夹垫候选恢复源V10的60×14×84mm几何，结合实际Franka关节与刚体帧换算抓高。
已通过V7轨迹的夹紧帧中，TCP指令高70.4mm、实测TCP高67.581mm，夹垫最低点15.394mm；
配置TCP高与实测夹垫底的有效差为55.006mm，其中实测TCP至夹垫底52.187mm、跟踪误差2.819mm。
该换算包含当前姿态与驱动的跟踪误差，是候选估计，不能当成所有机型的固定几何常数。
复算记录在 `outputs/debug/physic_solver/migration_logs/long_pad_height_calibration.json`。

| 宽夹垫候选 | 源V10夹垫底高度 | 本项目TCP指令高 | 净夹缝 |
|---|---:|---:|---:|
| 粉色T恤 | 10.0mm | 65.0mm | 7mm，介于源初始8mm／保持6mm之间 |
| 磨砂上衣 | 22.91mm | 77.9mm | 10mm，源保持值 |
| 绿色款 | 23.35mm | 78.35mm | 10mm，源保持值 |

源`half_gap`表示夹垫中心半间距，净夹缝为`2*half_gap-14mm`；
本项目`closed_finger_m`表示单指内面距离，净夹缝为两倍该值，不能直接复制两者数值。
三款均采用中央抓点、0.6s平滑闭合、无额外悬停、2s转运、1.5s下放与0.25m TCP释放高度；
这是宽夹垫配套控制组合，各款仍须从头完整验收，不把源抓取成功或某个参数单独当作本项目结论。
绿色款该组合首次完整求解抬升259.6／257.6mm、转运接触100%、拉伸1.20792、穿入0，
但释放后落点误差94.55mm，超过80mm，拒绝；源在 `20261112T223742_642536Z`。
释放开始时衣物质心与目标仅差约2.3mm，最终沿X／Y偏62.4／71.1mm，问题出现在释放后。
下一绿色候选只设置 `gripper_open_ramp_s=0.5`；粉色宽夹垫候选也采用0.5s平滑张开。
磨砂宽夹垫实验保留原张开方式，先完整验证再决定是否调整。
该磨砂宽夹垫完整物理、数据和实际视觉验收已通过：抬升267.6／265.0mm、转运双侧接触100%，
最大边长比1.13881、台面穿入0、释放回落137.8mm、落点误差6.18mm，无附着。
[磨砂正式结果](../outputs/physic_solver/package_cases/v10_frosted_top/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v10_frosted_top/validation.json)和
[关键帧](../outputs/physic_solver/package_cases/v10_frosted_top/qa/keyframes.jpg)均通过，
768物理帧／双相机192帧／15fps，12.8s物理耗时32.9min；源透明外观、标签与可见宽夹垫均已复核。
该款保留原阶跃张开，不将绿色款的释放偏移直接推广为所有包裹的结论。
粉色宽夹垫候选随后完整通过物理、独立数据和实际视觉验收：抬升260.8／257.2mm、
转运双侧接触100%、最大边长比1.265344、台面穿入0、释放回落141.8mm、落点误差14.44mm。
[粉色正式结果](../outputs/physic_solver/package_cases/v10_pink_tshirt/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v10_pink_tshirt/validation.json)与
[关键帧](../outputs/physic_solver/package_cases/v10_pink_tshirt/qa/keyframes.jpg)通过；
768物理帧／双相机192帧／15fps，TCP偏差3.910mm、姿态偏差0.441°，源哈希／逐值状态动作／FK一致性通过。
12.8s物理墙钟耗时50.5min；3次物理尝试保留在 `run_note.json`，该组合仍只采集一条通过episode。
绿色宽夹垫将张开改为0.5s后，完整物理、独立数据和实际视觉验收通过：
抬升265.8／262.2mm、转运双侧接触100%、边长比1.162893、台面穿入0、
释放回落121.2mm、落点误差43.98mm，低于原80mm门限。
[绿色正式结果](../outputs/physic_solver/package_cases/v10_station_green/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v10_station_green/validation.json)与
[关键帧](../outputs/physic_solver/package_cases/v10_station_green/qa/keyframes.jpg)均通过，
768物理帧／双相机192帧／15fps，物理墙钟43.0min；3次物理尝试均写入 `run_note.json`。
该结果支持当前配置可完成一次抓放，尚无重复运行成功率的统计保证。
从通过轨迹的实际刚体帧复算，最低夹垫点为22.905mm，源V10设定为22.910mm；
[高度复算记录](../outputs/physic_solver/package_cases/v10_frosted_top/qa/actual_pad_geometry.json)
确认本次TCP换算后的实际夹入深度与源配置匹配。
源材质、尺寸、质量和验收门限均保留。分拣站衣物中央抓点曾在抬升约30mm后失持，
失败前缀保留；仅改X偏移35mm、维持原70.4mm高度也在抬升约27mm后失持。
TCP降至61.3mm后闭合拉伸达到1.530，严格拒绝；台面穿入0.413mm虽未超限，也显示夹入较深。
X35mm／65.9mm中间高度的闭合边长比1.164，但提升时两部分最高仅4.6／5.5mm后失持，仍拒绝。
下一诊断改用既有成功基线的原装FR3指垫、22mm TCP高度与单指2.5mm闭合目标，仍保留X35mm；
这是与指垫几何配套的控制组合，不单独归因于某个参数，仍须完整验收。
该组合抬升238.4／234.2mm、边长比1.2746，但在9.85s转运后段滑落，双侧接触仅93.4%，拒绝。
下一轮恢复原装指垫的完整12.3s基线：中央抓点、取消额外悬停、转运2s、下放1s、释放高度0.20m。
此前迁移矩阵的14.8s在原控制上增加2.5s持物时间；当前是完整控制组合的候选，仍须重新求解，
不将短时抬起视作成功，也不单独归因于某个参数。
该12.3s组合的完整物理及独立审计已通过：抬升245.0／241.7mm、转运双侧接触100%，
边长比1.29969835（严格小于1.3）、穿入0、释放回落111.0mm、落点误差20.9mm。
源轨迹在 `outputs/debug/physic_solver/package_batches/station_clothing_native_baseline_01/v10_station_clothing`，
[正式结果](../outputs/physic_solver/package_cases/v10_station_clothing/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v10_station_clothing/validation.json)与
[关键帧](../outputs/physic_solver/package_cases/v10_station_clothing/qa/keyframes.jpg)均已通过：
738个物理帧，185帧／15fps／双相机，原装短指在腕部视角可见。
最大边长比余量较小，只代表本次完整轨迹通过，不作为多次运行稳健性的证据。
[失败工况的实测夹块／包裹截面投影](../outputs/debug/physic_solver/migration_logs/station_grip_cross_sections.png)
使用保存的刚体和粒子位姿显示高抓点脱离、低抓点闭合时的变形；投影只用于诊断，不代替三维穿透检查。

本轮沿用已有包裹门限：衣物与袋膜质心均抬升至少80mm，转运／摇晃双侧接触至少95%，释放回落至少10mm，
落点误差不超过80mm，台面最大穿入不超过1.5mm，最大边长比小于1.3，关键帧袋内包含误差不超过1mm，
无接触容量溢出。摇晃全阶段两部分最低抬升也必须达到80mm。每组先单次完整求解；重跑须保留全部尝试。
独立审计还检查最终两部分均回到桌面附近、夹爪接触归零，避免把内容物落下但外袋仍挂在手指上算作完全释放。
它们是仿真验收指标，不是硬件负载认证，也不是源项目所有形态指标的逐项等价复现。

### 9.2 附着模型

`attachment_mode` 默认 `none`，`shake_s` 默认0，旧配置保持原动作和纯接触路径。
`spring` 在闭合完成后，以实际手指刚体为参照，选取距指垫表面不超过4mm、每侧最多8个且不重复的袋膜节点。
只附着外袋，不直接粘住内容物；衣物仍依靠袋内接触支撑。
绑定局部坐标使用当时的实测位置，不吸到夹面。每节点30N/m、0.015N·s/m，力上限0.5N；参数全部写入生效配置。
独立审计直接检查节点编号属于保存轨迹的外袋区间，并要求两组对应不同夹指，不能只凭材料标签通过。
每个子步由实际手指位姿计算弹簧力，释放时关闭力。全程不改节点位置、速度、质量、逆质量或活动标志；
不增设固定粒子，仍接受原拉伸、穿透和包含门限。局部点不足时明确报错。
这是人为的可释放抓持辅助，未标定黏性材料、断裂或真实胶黏，结果会记录 `attachments_used`、分组节点和距离。

### 9.3 执行与输出

```bash
# 资产准备：每款仍使用只复制、不修改源资产的入口
./app/python.sh scripts/physic_solver/prepare_package_assets.py \
  --source /home/fufa/Projects/package/output/simulation_v10/cases/station_clothing \
  --target assets/PhysicSolver/Package/v10/station_clothing
# 其余四款替换末尾款名；V7 使用该脚本默认参数。

# 整个矩阵：逐工况独立Newton SDK进程，先物理与独立复核
./app/python.sh scripts/physic_solver/run_package_matrix.py
# --collect 将已通过的连续物理轨迹渲染为双相机 LeRobot v3，不重复求解物理
./app/python.sh scripts/physic_solver/run_package_matrix.py --cases v9_mass100g shake_none shake_spring --collect
```

每次尝试都在 `outputs/debug/physic_solver/package_migration/<case>/<UTC运行编号>/`，物理文件位于 `physics/<case>/`，含日志、原始粒子、实测轨迹、
`result.json` 和重新计算的 `physics_audit.json`。新鲜运行目录避免复用旧成功文件。
`--collect` 只有在渲染结果通过源文件哈希、关节／刚体对齐和独立数据检查时，才移动到配置声明的正式目录；正式目录存在时拒绝覆盖。
默认双相机640×480、15fps、H.264、LeRobot v3，固定全景与腕部相机沿用已有配置。
渲染入口使用Linux文件锁串行启动Kit；独立物理尝试同时结束时，后来的渲染等待前一个完成，
避免本机曾出现的并发RTX初始化错误。锁不改变已保存的物理轨迹。

矩阵共18个柔性工况，另有2个刚性纸箱。`package_matrix_base.toml` 为V7派生工况采用源V7的
60×14×84mm显式夹块、单指闭合6.25mm、刚软接触容量32768；个别位置及V10按各款记录实际夹爪和控制组合。
V7派生工况的起始TCP抓取高度为70.4mm；V10按各款厚度设置的高度见9.1与生效配置。
已有原装短指成功基线配置不修改。V7派生配置在提升后保持1s、转运3s、下放1.5s；
薄款原装指垫候选采用无额外悬停、转运2s、下放1s；摇晃配置在摇晃后另保持1s，
下放前依据实际衣物质心修正后续机械臂航点。没有修改粒子位姿或放宽质量门限。

采集前与导出后另用 `audit_package_grasp_pose.py` 读取最后一个夹紧帧的实际关节，
通过源URDF正向运动学复算TCP；独立矩阵 `Rz @ Ry @ downward` 验证配置角度，
位置偏差须≤5mm、角度偏差≤1°、FK与记录TCP差≤0.01mm。此入口只检查抓取位姿，不代替物理抓放验收。
本任务机器人基座Z与桌面支撑平面一致；该假设写入审计报告。
新增检查补审的12个已通过柔性工况全部通过，最大位置偏差4.368mm、姿态偏差0.477°；
篡改请求偏航45°或X偏移35mm而保持实际关节不变的两个负对照均被拒绝，
结果位于 `outputs/debug/physic_solver/grasp_pose_negative_controls_01/negative_control.json`。
报告随正式数据保存为 `qa/grasp_pose_audit.json`，手动入口如下：

```bash
./app/python.sh scripts/physic_solver/audit_package_grasp_pose.py \
  outputs/debug/physic_solver/package_batches/<run>/<case>
```

推荐批量物理与渲染分开：

```bash
# Isaac python.sh 内运行本机 Newton 1.6.1 / Warp 1.17 / USD；不启动 Kit / RTX
./app/python.sh scripts/physic_solver/check_package_matrix.py
# 将某个已通过物理和独立审计的连续轨迹渲染为正式数据；目标必须不存在
./app/python.sh scripts/physic_solver/render_package_trajectory.py \
  outputs/debug/physic_solver/package_batches/<run>/<case> \
  --output outputs/physic_solver/package_cases/<case>
```

`newton_diagnostic` 世界后端只负责真实 Newton 物理诊断，通过同一 BaseJob 生命周期装配组件。
SDK库来自当前Isaac安装，不创建虚拟环境或更换Newton版本。正式相机渲染使用原 `isaacsim` 世界。
本机本轮14.8s物理的实测墙钟耗时为：V7约28.9min、X35位置约36.2min、500g约44.4min，
另需视频渲染与独立复算时间。这些运行的GPU并发负载不同，仅用于估计预算，不构成可比性能基准；
多进程隔离也不保证吞吐随并发数增长。
批处理各工况有独立模型、控制和碰撞，之间无物理交互；控制上传后有同步屏障。
多模型并发CUDA图出现过非法内存访问，不能把短冒烟通过当作完整路径已稳定；当前父进程为每个工况启动独立子进程，逐个求解。
超过不可恢复的拉伸／台面穿透门限时早停该工况，保存失败前缀，不把它算作完成或成功。
进度日志同时记录当前质心抬升与历史最高值；仅看最高值不能判断物体后来是否滑落。
异常、超时和中断也保存已有轨迹；新运行从头独立求解，不将前缀拼接为成功轨迹。
本轮8子步／28迭代的对照在提升时双侧接触丢失，主体抬升不足2mm；32子步参考已抬升约240mm。
因此8子步对照不采用、不晋升正式结果。弹性核改为scalar的短诊断也未观察到收益，默认仍用tiled。
32子步／12迭代的对照也在提升阶段失去接触，衣物／袋膜最高仅30／47mm；常规工况保留28迭代。
16子步／28迭代的完整任务对照曾抬升213／200mm，但转运时失去双侧接触并跌破80mm保持门限，
同样拒绝采用。短时抬起不代表完整抓放成功；正式矩阵均为32子步，常规28迭代，困难姿态40迭代，详见9.7节各次记录。

`full_physics.npz` 保存每个物理帧的粒子、刚体、实际关节、实际下发动作和时刻。
形变与台面穿入复算覆盖保存的全部60Hz物理帧；接触容量峰值另由求解器累计各子步统计。
渲染只发布同一个源帧的几何，物理步数保持零；独立FK核对关节／刚体对齐，逐值核对导出状态和动作，
保留完整源文件及SHA-256。它是连续真实物理的可视化，不是脚本生成的物体动作。
本轮多Kit初始化曾出现RTX管线断言；无Kit的多个GPU MuJoCo模型也曾崩溃，不能把原因全部归于RTX。
新矩阵显式使用 `mujoco_cpu=true`：原生双精度MuJoCo解机器人，32个真实子步批量上传，再由CUDA VBD解包裹。
单向基准的控制在一个60Hz帧内保持，全部32子步的真实刚体位姿仍用于接触；不减少其28次VBD迭代，不加柔体到刚体的反馈。
显式启用双向body代理的工况则在各子步回传接触反力，使用独立的代理几何、原生驱动力限及实际位姿验收；不与单向结果混作负载对照。
完整GPU软体帧可捕获为CUDA图；CPU求解与上传发生在图外。已保存的生效配置标明实际执行路径。

纸箱先用真实CPU动力学探针迭代，之后用同一个原生CPU MuJoCo模型做完整Isaac视频验收：

```bash
./app/python.sh scripts/physic_solver/probe_carton_cpu.py --output outputs/debug/physic_solver/carton_probe
./scripts/rsr collect run -c configs/physic_solver/mujoco_rigid_v10_box01.toml
./app/python.sh scripts/physic_solver/validate_carton_pickplace.py outputs/physic_solver/package_cases/v10_box01
```

纸箱原生MuJoCo接触显式使用Newton 1.6.1的force-space材料模式，刚度1e6N/m、阻尼1000N·s/m；
未沿用默认2500N/m导致的大穿入。大纸箱按实际抓法使用更高夹指驱动增益，非质量单因素实验。

### 9.4 本轮已完成的纸箱验收（2026-11-12 UTC）

| 工况 | 转运最低抬升 | 最大TCP距离 | 释放回落 | 落点误差 | 独立校验／视觉复核 |
|---|---:|---:|---:|---:|---|
| [box01正式结果](../outputs/physic_solver/package_cases/v10_box01/result.json) | 200.0mm | 17.3mm | 96.8mm | 8.5mm | 通过，165帧／15fps／640×480 |
| [box02正式结果](../outputs/physic_solver/package_cases/v10_box02/result.json) | 178.3mm | 26.2mm | 33.3mm | 22.8mm | 通过，165帧／15fps／640×480 |

两者均无附着、原尺寸和质量、全程有限，角点独立重算的台面穿入与最终支撑通过。
首帧材质仍偏暗，后续关键帧显示源贴图。box02首轮完整采集落地翻滚、偏离123.6mm，保留在debug；
将TCP释放高度由0.12m降至0.08m后，从头重新求解与采集通过，门限未变。
每个正式目录保留 `validation.json`、`qa/visual_review.json`、`run_note.json` 和原始全频率轨迹。
柔性矩阵的进度与结果须以各次 `batch_summary.json`／`physics_audit.json` 为准；未通过的工况不放入正式目录。

### 9.5 新柔性基准正式验收

`package_cases/v7_center.toml` 已完整求解、独立复算、双相机渲染及视觉复核通过：
[结果](../outputs/physic_solver/package_cases/v7_center/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v7_center/validation.json)、
[关键帧](../outputs/physic_solver/package_cases/v7_center/qa/keyframes.jpg)。
衣物／袋膜最高抬升242.0／239.9mm，转运双侧接触100%，释放回落98.3mm，落点误差20.1mm，
最大边长比1.1442、台面最大穿入0；11个检查时刻的9588个衣物顶点／面中心探针均在袋内1mm容差范围。
32子步／28迭代、14.8s连续物理、888个物理帧、222个视频帧／15fps／两路640×480 H.264。
无附着、无固定粒子；完整源物理文件保留并校验SHA-256，渲染物理步数为零，导出状态与动作逐值一致。
腕部视野部分被手掌遮挡，全景覆盖完整任务；其余重量、位置、种类及摇晃工况另行逐个验收，不能沿用本例成功标签。

`shake_spring.toml` 的[正式结果](../outputs/physic_solver/package_cases/shake_spring/result.json)与
[独立验收](../outputs/physic_solver/package_cases/shake_spring/validation.json)也已通过，并复核摇晃中的相反方向关键帧。
摇晃期间衣物／袋膜最低抬升208.5／210.1mm，转运／摇晃双侧接触100%，最大边长比1.1851，
落点误差26.0mm、释放回落93.5mm，267帧／15fps／双相机。
外袋每侧8个附着节点，捕获点距指垫最大2.174mm，局部坐标复算误差小于0.000024mm；释放后力关闭、末帧接触为零。
实测关节FK核对了真实摇晃：XYZ往复范围25.9／17.6／13.1mm，偏航／俯仰范围20.4°／10.2°，并非只有动作标签。
这是有限弹簧辅助的成功记录。中央抓点无附着摇晃在11.47s跌破80mm持续抬升门限，
最低衣物／外袋抬升约70.5／69.3mm，按原门限拒绝。
`shake_none`改为X35mm后，[正式结果](../outputs/physic_solver/package_cases/shake_none/result.json)、
[独立验收](../outputs/physic_solver/package_cases/shake_none/validation.json)与摇晃相反方向关键帧均通过：
最低摇晃抬升232.5／228.7mm，双侧接触100%，边长比1.1262、穿入0、回落114.2mm、落点误差12.7mm。
真实XYZ往复范围25.9／17.6／13.0mm，偏航／俯仰20.4°／10.2°；17.8s／1068物理帧／双相机267帧。
同抓点`shake_spring_x35`的[正式结果](../outputs/physic_solver/package_cases/shake_spring_x35/result.json)、
[独立验收](../outputs/physic_solver/package_cases/shake_spring_x35/validation.json)与视觉复核也已通过：
最低摇晃抬升234.1／231.0mm、双侧接触100%、边长比1.1249、穿入0、回落118.6mm、落点误差14.9mm。
同为17.8s／1068物理帧／双相机267帧；每侧8个外袋节点，捕获点距夹面最大0.350mm，
独立核对外袋编号、不同夹指、零捕获位移及释放关闭，实测XYZ与角度往复范围和无附着组相近。
不将不同抓点的记录当作附着单因素对照，也不据单次结果推断统计收益。

`v8_offset_x35.toml` 的[正式结果](../outputs/physic_solver/package_cases/v8_offset_x35/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v8_offset_x35/validation.json)和
[关键帧](../outputs/physic_solver/package_cases/v8_offset_x35/qa/keyframes.jpg)已通过。
无附着，衣物／袋膜最高抬升246.1／242.1mm，转运最低约243.1／238.4mm，双侧接触100%；
最大边长比1.1348、台面穿入0、释放回落121.9mm、落点误差27.0mm。
同为14.8s连续物理、888帧、双相机222帧；完整源文件、状态／动作对齐和可见的偏移抓取／释放均已核对。

### 9.6 500g 夹持调试记录与控制单位

本轮按项目 gripper skill 工作表核对，参考的是诊断方法，没有采用 PhysX Robotiq 示例的增益。
实际为FR3 URDF的两个直线夹指，附加60×14×84mm可见夹块，非Robotiq四连杆。
URDF第二指含mimic标签；本路径显式给两个滑动坐标相同目标，不用标签本身证明实际耦合。
原生MuJoCo位置驱动：夹指刚度1600N/m、阻尼40N·s/m、执行器上限60N，指令60Hz、物理1920Hz。
这些是任务仿真参数；包裹VBD反力不回馈刚体，因此60N不是已验证的真实接触夹力。

| 输入／参数 | 来源与单位 | 本轮观测／用途 |
|---|---|---|
| 内容物500g、袋膜约20.542g、摩擦1.4 | 质量配置、源USD与任务接触假设 | 理想对称静摩擦下每指至少约1.824N；柔性多接触不由此公式直接认证 |
| 单指开度40→6.25mm | 实际写入 `control.joint_target_q`，米 | 原阶跃命令的60Hz实测差分峰值约0.936m/s；可能漏掉更高的子步尖峰 |
| `gripper_close_ramp_s=0.6` | 三次平滑位置目标，秒 | 目标峰值速度 `1.5×(0.04−0.00625)/0.6=0.084375m/s`；实际效果另由轨迹验收 |
| `gripper_open_ramp_s=0.5` | 释放阶段平滑目标，秒 | 释放末值仍40mm，不修改物体状态；两个ramp默认0，旧配置行为不变 |

本机Newton 1.6.1的 `SolverMuJoCo` 实现明确列出不支持 `joint_velocity_limit`；
不能因URDF的0.2m/s被复制到模型，就假定该求解器会强制限速。
依据为 `app/exts/isaacsim.pip.newton/pip_prebundle/newton/_src/solvers/mujoco/solver_mujoco.py` 的Joint support说明。

500g原抓法最高抬升约185mm，保持时失持，未通过；降低抓点10mm的阶跃闭合尝试最大边长比1.468，早停。
深抓点加平滑闭合将60Hz实测指速峰值降至约0.0838m/s，但边长比1.30062仍超标，按原门限拒绝。
恢复原抓取高度、仅增加平滑闭合／释放的尝试也在夹紧阶段达到边长比1.315，按原门限早停。
因此质量组保留原闭合方式；平滑控制保留作诊断选项，45°姿态的独立尝试另见9.7。
[四次失败的实测曲线](../outputs/debug/physic_solver/migration_logs/500g_gripper_attempts.png)
对比实际单指位置、全频率最大边长比和内容物抬升，区分闭合应变与提升后失持。
只把500g抓点沿X偏移35mm后，完整物理和独立审计均通过：衣物／袋膜抬升245.0／241.6mm，
转运双侧接触100%，边长比1.1488、台面穿入0、释放回落121.3mm、落点误差5.23mm。
源轨迹在 `outputs/debug/physic_solver/package_batches/weight500_x35_03/v9_mass500g`；
[正式结果](../outputs/physic_solver/package_cases/v9_mass500g/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v9_mass500g/validation.json)与
[关键帧](../outputs/physic_solver/package_cases/v9_mass500g/qa/keyframes.jpg)均已通过。
222帧／15fps／双相机；源文件哈希、状态／动作逐值对齐、FK误差0和渲染物理步数0均核对通过。
100／200／350／500g四个质量配置统一采用同一X35mm抓点与原闭合方式，每个质量均独立求解，
不从500g或V8的位置结果继承成功标签。
100g的[正式结果](../outputs/physic_solver/package_cases/v9_mass100g/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v9_mass100g/validation.json)与视觉复核也已通过：
抬升244.2／241.5mm，转运双侧接触100%，边长比1.1341、台面穿入0、回落119.0mm、落点误差19.4mm。
同为888个物理帧、双相机222帧；第一条完整物理与视频均通过，旧长跑控制器的配置类缓存导致自动晋升失败，
经新进程复核后晋升，未重复或修改物理轨迹。
350g的[正式结果](../outputs/physic_solver/package_cases/v9_mass350g/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v9_mass350g/validation.json)与视觉复核通过：
抬升245.3／240.5mm、转运双侧接触100%、边长比1.1464、穿入0、回落115.9mm、落点误差22.3mm。
同为888物理帧／双相机222帧；该物理子进程完成后才更换旧调度进程，连续轨迹未重跑或修改。
200g的[正式结果](../outputs/physic_solver/package_cases/v9_mass200g/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v9_mass200g/validation.json)与视觉复核通过：
抬升242.8／239.4mm、转运双侧接触100%、边长比1.1270、穿入0、回落114.1mm、落点误差16.4mm。
四组均完成独立求解、双相机222帧与数据／视觉验收，后续矩阵使用新调度进程。
[四种重量的实测衣物／外袋抬升曲线](../outputs/physic_solver/package_cases/mass_sweep_curves.png)
中灰色区为转运，水平虚线为该阶段80mm门限，竖点线为开始释放。每质量仅一条通过轨迹，
单向耦合下的曲线不能用于认证真实机械臂额定负载。
没有放宽1.3拉伸、1.5mm台面穿入和80mm抬升门限；不同抓点的调参结果不作为质量单因素对照。
旋转抓点在 `before_place_robot_hook` 先用同一URDF的IK对齐初始预抓姿态，避免开始后的方向阶跃；
45°／90°／组合倾斜的离线预抓IK已核对可达，实际抓放仍须运行验收。

### 9.7 位置／角度工况的后续验证

Y55mm长夹块工况最高抬升111.8／117.8mm，随后失去双侧接触，5.2s时两部分已回到桌面，拒绝。
失败前缀保留在 `package_migration/v8_offset_y55/20261112T200151_572773Z`。
原装FR3指垫的12.3s基线在3.417s提升初段达到边长比1.30068235，按原1.3门限拒绝，
保留在 `package_batches/y55_native_baseline_01`。不四舍五入视作通过。
从保存网格独立定位，超限边位于衣物的Y55mm夹持截面，原长12.406mm；并非数值NaN或台面穿入触发。
下一轮保持Y55mm、22mm抓高和其余控制，仅将单指闭合目标由2.5改为5mm，减小局部挤压；
数值门限不变，仍须完整独立验收。

上述10mm夹缝的Y55完整重跑已通过物理、独立数据和实际视觉验收：抬升226.4／222.0mm，
转运双侧接触100%、采样最低抬升221.3／216.8mm、最大边长比1.20845、台面穿入0，
释放回落98.5mm、落点误差35.2mm，两部分最终落回桌面、指间接触归零，无附着。
[正式结果](../outputs/physic_solver/package_cases/v8_offset_y55/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v8_offset_y55/validation.json)与
[关键帧](../outputs/physic_solver/package_cases/v8_offset_y55/qa/keyframes.jpg)均通过；
738物理帧／双相机185帧／15fps，源轨迹哈希、状态／动作逐值对齐与FK一致性均通过。
原始源为 `package_batches/y55_native_gap10_01/v8_offset_y55`，12.3s物理耗时32.3min，
该控制组合仍只完成单条episode。

45°长夹块工况在4.383s提升阶段达到边长比1.31347，抬升107.0／104.7mm仍不足以算作完成，拒绝。
原始失败保留在 `package_migration/v8_yaw45/20261112T200151_572773Z`。
独立定位的最坏边属于袋膜，原长17.673mm，邻近衣物边也达到1.30035。
下一候选保持45°和预抓对齐，换为原装FR3指垫的中央抓点／22mm／单指5mm／12.3s组合，
与Y55及90°候选统一10mm夹缝，各项独立验收。
该22mm短指候选在1.967s夹紧初段达到边长比1.30348，拒绝；
失败保留在 `package_migration/v8_yaw45/20261112T213300_511763Z`。
下一轮只将TCP抓高由22改为28mm，姿态、10mm夹缝及其余控制保持，仍须完整验收。
28mm候选仍在1.967s达到边长比1.30370，拒绝，保留在 `20261112T213953_088707Z`。
下一候选保持28mm几何，仅设置 `gripper_close_ramp_s=0.6` 检查快速闭合的瞬态影响；
之前500g长指实验未采用该选项，不能直接推断短指45°的结果。
0.6s平滑闭合将夹紧阶段峰值保持在1.2972，但提升至4.30s时仍达到1.30188，按原门限拒绝，
最高抬升79.4／77.3mm，保留在 `20261112T214613_669612Z`。
下一候选只将夹缝由10改为13mm，保留28mm抓高、0.6s闭合、45°及其余控制。
13mm夹缝在夹紧阶段为1.1963，但4.60s提升时达到1.30569，最高抬升157.1／155.8mm，拒绝，
保留在 `20261112T215803_910310Z`。下一轮仅将抓点沿X偏移35mm；保持45°和当前短指控制。
这是对45°抓法的抓点优化，不能再称为原中央抓点的单角度对照。
X35候选完整保持至下放，最高抬升233.4／230.1mm，转运期间边长比1.2079；
8.883s下放减速时达到1.30013882，仍严格拒绝，保留在 `20261112T220803_113935Z`。
下一候选仅将下放由1放慢为1.5s，其余抓点、角度、夹缝和闭合设置保持。
该次独立重跑在6.85s转运初段达到边长比1.30205而拒绝，保留在 `20261112T223159_814518Z`。
本次提升峰值233.2／230.7mm、双侧接触仍在；下放尚未开始，因此不能据此评价下放变慢的效果。
下一轮保持抓法，单独将VBD迭代由28提高为40做精度加密，仍保留32子步与全部原门限；
其他工况不改精度，40迭代仅为该姿态候选。增加的7200s超时预算仅覆盖更长计算，不改变物理门限。
该40迭代完整物理与独立审计已通过：抬升245.1／243.1mm、转运双侧接触100%，
边长比1.29432、台面穿入0、释放回落112.6mm、落点误差51.4mm，无附着；
12.8s物理墙钟耗时40.7min。源轨迹在 `20261112T224633_533970Z`。
[45°正式结果](../outputs/physic_solver/package_cases/v8_yaw45/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v8_yaw45/validation.json)与
[关键帧](../outputs/physic_solver/package_cases/v8_yaw45/qa/keyframes.jpg)均已通过；
768物理帧、双相机192帧／15fps、FK／源哈希／逐值状态动作对齐与实际视觉复核通过。
源URDF FK实测偏航45.0104°，位置偏差4.411mm、完整姿态偏差0.469°；
另从两指实际刚体中心连线独立交叉核对偏航45.0104°，见 `qa/jaw_axis_crosscheck.json`。
8次实际尝试的有效配置与失败结果均保留，记录于正式结果的 `run_note.json`，该组合仍只采集一条通过episode。

90°长夹块工况最高抬升80.5／84.4mm后失去双侧接触，4.77s时两部分已回到台面，拒绝并保存失败前缀。
后续90°配置继承当前45°控制组合，采用同一X35抓点，保留90°与预抓对齐；
本轮队列仅在45°完整物理及数据通过后启动，90°仍须独立求解，不继承成功标签。
该X35短指／40迭代候选在3.70s提升初段达到边长比1.30112，最高仅2.7／3.4mm，拒绝，
保留在 `20261112T233129_431132Z`。
下一轮90°配置改为继承已通过的 `v8_offset_x35.toml` 宽夹垫／32子步28迭代组合，
仅增加90°偏航与预抓对齐；原X35成功不能替代本次独立验证。
该宽夹垫X35／90°尝试最高76.4／74.5mm后失去接触，未达到80mm抬升门限，拒绝，
源在 `20261112T235420_125387Z`。下一候选仅将净夹缝12.5收紧至10mm，并设置0.5s平滑张开。
10mm夹缝的阶跃闭合在1.90s达到边长比1.32275，拒绝，源在 `20261113T002424_376451Z`。
下一候选仅增加0.6s平滑闭合；独立物理尝试在 `package_batches/yaw90_gap10_ramp_01`，完整通过后再采集。
该候选最高抬升89.1／87.0mm后完全滑脱，5.82s时接触归零且两部分回到桌面，拒绝并保存前缀。
核对源V8的 `yaw90/trial_final/validation.json`，原中央抓取确已完整通过，夹垫最低点为28mm。
按实测指令TCP到夹垫最低点约55mm换算，下一候选采用中央TCP83mm、10mm夹缝、0.6s平滑闭合，
并使用32子步／40迭代；这是对齐源几何与求解精度的组合验证，不把效果归因于单个参数。
首次启动因迭代字段拼写错误未进入物理，配置检查已捕获并修正为 `newton_iterations`，保留启动失败记录。
83mm中央抓取／40迭代随后完成提升和全部转运：最高291.2／291.1mm、最大边长比1.238338。
但释放后明显偏移，12.3s触发台面穿入早停，落点误差已达424.6mm，整条拒绝；
源在 `20261113T004833_971582Z`。释放前质心距目标约9mm，水平速度约0.02m/s；
释放后约0.48s达到约3m/s。包裹下缘在松开前已接触台面，问题集中于释放与着台过程。
短时重建初态的诊断 `seeded_yaw90_release_control01` 也复现约404mm偏移、3.04m/s峰值及13.3mm穿入，
并在两夹指接触归零后记录到手掌接触；下一诊断只提高释放初态高度，检查着台时序。
这些重建初态的诊断仅用于定位，`full_episode_accepted=false`，不拼接或晋升为正式轨迹；
最终候选必须从头完成完整物理与数据／视觉验收。
提高初态释放TCP至350mm的短诊断仍产生478mm偏移、4.29m/s峰值，拒绝该假设；
源移动参考系实现已核对与迁移实现一致。40迭代／阶跃张开短诊断也出现201mm偏移和2.13mm穿入。
另从源V8的原始位置／速度重新复算，6.9s保持末帧至结束也漂移180.5mm，释放后水平速度峰值1.168m/s；
源验收没有XY落点项，这与当前80mm落点目标是不同口径，不把源“成功”推定为精确放置成功。
复算见 `migration_logs/source_v8_yaw90_release_drift.json`。后续短诊断也全部拒绝：
28迭代偏移255.5mm／穿入4.48mm，1s张开偏移369.5mm／穿入38.1mm，
控制器先降至180mm后张开仍偏移508.1mm／穿入3.06mm；原生世界坐标DAT偏移478.6mm，
粒子侧摩擦降至0.5则偏移3442.6mm。提高释放高度没有手掌接触仍失稳，手掌接触不足以解释所有案例。
十一个诊断的初态来源、参数、全部实测采样与范围说明见 `migration_logs/release_diagnostics_index.json`，均未采用。
下一完整候选改用已通过Y55的原装短指家族：中央22mm／10mm夹缝、90°预抓对齐、
32子步／40迭代、闭合0.6s／张开0.5s、保持0s／转运2s／下放1.5s／放置TCP200mm。
这是独立夹爪／控制组合验证，不能把结果归因于单一参数，也不改变已有通过工况的配置或验收门限。
原装中央22mm／10mm夹缝候选在4.85s提升减速段达到边长比1.31086，最高202.6／201.2mm，拒绝，
源在 `20261113T020755_489543Z`。下一轮只将提升阶段从3放慢至4.5s，实际平滑运动从1.8延长至2.7s，
其余夹爪、位置、角度、夹缝和门限保持；仍须完整重跑。
4.5s慢速提升在5.267s达到边长比1.306698，最高170.3／168.7mm，仍拒绝，
源在 `20261113T022221_062667Z`。两轮最坏边均属于袋膜；慢速工况的最坏边中点距原生指垫内面约0.49mm，
处于接触区，而非衣物悬垂部分。定位见 `migration_logs/yaw90_native_overstrain_edges.json`。
下一轮只将单指目标5改为6.5mm，即净夹缝10改为13mm，保留中央22mm抓高、90°和4.5s提升。
13mm候选最大边长比保持1.27920，但在约173mm抬升后失去双侧接触，随后跌落，
5.90s台面穿入14.12mm而早停，拒绝，源在 `20261113T023140_367215Z`。
下一轮只把净夹缝取为11.5mm（单指5.75mm），在10mm拉伸受限和13mm滑脱之间寻找可行区间；
完整保持、释放和原门限仍须独立验收。
11.5mm候选仍在5.167s达到边长比1.31559，最高147.8／146.5mm，拒绝，
源在 `20261113T024115_453404Z`。随后将URDF每个碰撞盒的局部RPY和实际刚体帧一起复算：
22mm指令TCP对应实际TCP19.545mm，而最低碰撞盒仅距台面9.933／10.001mm。
实测指令TCP到最低指垫约12.067mm；源90°夹垫最低点为28mm，因此下一中央短指候选取TCP40.1mm，
并恢复源10mm保持夹缝，保留90°、4.5s提升、40迭代及原门限。
这是对齐接触几何的组合调整，不作单因素归因；校准见 `migration_logs/native_pad_height_calibration.json`。
40.1mm／10mm夹缝候选最高抬升92.1／88.4mm，最大边长比1.2143，但随后失去双侧接触，
6.37s时接触仍为零、当前抬升降至66.1／61.6mm，确认无法完成保持后停止并保留前缀，
源为 `20261113T025330_275612Z`。下一轮仅将TCP取为31mm，在22mm拉伸受限与40.1mm滑脱之间寻找抓取深度；
10mm夹缝、4.5s提升及原门限保持。
31mm中央候选在5.167s达到边长比1.310917，最高170.7／169.9mm，拒绝，
源为 `20261113T030323_320764Z`。下一轮只将抓点沿Y偏移55mm，保持31mm抓高、10mm夹缝和90°；
这是对90°抓法的位置优化，不再作为源中央位置的单角度对照，原中央失败均保留。
Y55mm／原装31mm抓高最高仅5.9／5.8mm后滑脱，7.15s时仍无夹指接触且已回到桌面，
随后进入转运即触发80mm保持门限早停，失败前缀保留。下一轮回到此前中央宽夹垫83mm／10mm夹缝／40迭代的完整保持组合，
相对该组合仅将位置改为X35mm，原3s提升／1s保持／3s转运／1.5s下放／250mm放置高度均恢复。
该比较针对抓点对宽夹垫释放的影响，不将多轮原装短指候选混作单因素基线。
宽夹垫X35／83mm在约186.5／184.3mm抬升后失持，5.083s跌落并触发台面穿入门限，
最大边长比1.22050，仍拒绝，源为 `20261113T032400_193260Z`。
随后针对原中央83mm、已完整保持转运但释放失稳的轨迹做时间离散加密诊断：32子步提高至64子步，
其余材料、几何、控制与门限不改；诊断仍为重建初态，不能直接晋升正式轨迹。
64子步诊断在1.2s内向上及水平方向失稳，水平偏移8.575m、峰值11.859m/s，未采用。
另将整套场景的数值原点平移至初始抓点与桌面高度，保持所有相对几何不变；
32子步诊断仍偏移460.8mm、峰值3.301m/s、穿入4.38mm，未采用。
随后按本机 Newton 官方 `example_proxy_joint_gripper.py` 核对原生双向耦合接口，
在调试目录另行验证 MuJoCo/VBD 的 body/joint proxy 反馈及有限夹指驱动。
该诊断尚未进入正式配置，不将单向接触下已有通过结果外推为双向耦合结果。
首轮有限夹指驱动200N/m、8N·s/m下，闭合末实际净指缝23.76mm，而目标为10mm；
衣物／袋最高抬升仅0.010／0.036mm，提升初段失去接触，4.4s停止并保存失败前缀。
调试记录为 `proxy_coupling_yaw90_01/configuration`；该轮没有通过，不能宣称双向反馈已经解决问题。
第二轮仅提高有限夹指驱动至800N/m、20N·s/m，仍未抬起；新增逐帧核对发现，
VBD关节代理的夹指位置相对源机器人偏离约12–14mm，部分代理姿态相差数十度。
因此该整臂关节代理组合不满足物理与可视几何一致性，不能将失败只归因于夹指驱动力，
也未加入生产运行时。诊断位于 `proxy_coupling_yaw90_02/configuration/proxy_residuals.json`。
下一完整单向候选以此前中央83mm／10mm夹缝组合为基准，只将抓高降低至70.4mm，
保留平滑闭合／张开和32子步／40迭代，从头验证抓取、转运和释放。
中央70.4mm／10mm夹缝在2.4s闭合阶段达到边长比1.311586，拒绝，
源为 `20261113T040824_786823Z`。下一轮只将抓高取两者中点76.7mm；其余组合和原门限保持。
76.7mm在提升至275.4／275.2mm后，于5.75s达到边长比1.339372，仍拒绝，
源为 `20261113T041215_224519Z`。随后回到已完成全转运的中央83mm轨迹，
仅做重建初态的同步张开／向上退爪诊断；该诊断不能直接晋升为完整正式轨迹。
该同步退爪诊断仍偏移419.5mm、峰值3.869m/s、穿入11.82mm，拒绝且未采用。
继续核对源V8配置与V7 USD的材料参数：膜／衣物／弯曲／内部弹簧阻尼一致，未发现遗漏的材料覆盖。

原生刚体代理的4帧独立冒烟取消VBD整臂关节代理，仅由MuJoCo求解全部机器人关节，
VBD通过官方 `SolverCoupledProxy` 回传接触反力；两侧body标签顺序相同，位置差最大约0.00029mm。
该冒烟不构成完整抓取通过。可选配置 `planner.rigid_soft_coupling="proxy"` 已接入，默认仍为 `"one_way"`，
要求 `mujoco_cpu=true`、`attachment_mode="none"`；采用lagged body代理、质量比例1、单次耦合迭代，
在每个物理子步消费CPU刚体反馈，仅把GPU VBD子步捕获为CUDA graph。
额外验收要求逐子步最大代理位置／姿态差≤1.5mm／1°，并保存全60Hz的源／代理刚体位姿供独立复算。
当前从中央83mm、10mm指缝、原材料和原控制时序开始完整重跑；尚未将双向方案标为成功。
该83mm双向尝试闭合末实际净指缝约49.88mm，目标10mm，未抓起包裹，4.2s停止并保存前缀，
源为 `20261113T042642_805166Z`。逐子步代理误差为0.0959mm／0.0108°，
但离线原始位姿另检出起始帧代理输出缓存尚未分发（后续帧最大0.0875mm）；
已在reset后显式同步entry输入状态，避免保存未初始化的初始代理帧。
下一轮只将抓高降至70.4mm，保留原1600N/m夹指驱动、60N力限和全部材料，验证更深的有限受力抓取。
70.4mm双向尝试最高衣物／袋抬升仅0.011／0.030mm，提升初段失持，4.417s保存失败前缀，
源为 `20261113T043345_554204Z`。闭合末实测净指缝17.20mm，目标10mm；
最大边长比1.12110，逐子步代理误差0.1504mm／0.0183°。
初始缓冲同步后，离线全部原始位姿最大差0.1284mm／0.0137°，没有再遗漏起始帧。
下一轮保持70.4mm和60N力限，仅把夹指驱动由1600N/m、40N·s/m调整为6000N/m、60N·s/m，
检查有限驱动力下的闭合与持物；这组参数是显式候选，不修改17项已通过柔性工况。
6000N/m／60N·s/m下闭合末净指缝仍为17.49mm，未抬起；原生静态审计确认增益生效，
当时实际驱动力约21.58／23.45N，关节总驱动力限制为已启用的±60N。
仅将闭合目标设为0的宽夹垫尝试最高抬升0.116／1.996mm后失持，边长比1.08949，
源为 `20261113T044918_652587Z`。随后使用原装短指／22mm抓高、同一有限驱动和零目标，
也未抬起，源为 `20261113T045659_083013Z`。

新增 `proxy_mass_scale` 只调整目标代理的有效虚拟质量／惯量，不改变源机器人或粒子质量，
默认1；初始化逐项核对源质量、惯量与粒子质量未被覆盖。原装短指将比例提高为100后，
几何误差降至0.00539mm／0.000079°，但仍未持物，源为 `20261113T050759_994417Z`。
新增60Hz的实际原生控制量、电机力、外部反力／力矩原始记录，逐子步另保存最大真实电机力并检查60N力限。
首次记录发现t=0的MuJoCo控制缓存尚为reset默认值、尚未用于实际子步，已在采样前加载真实初始控制并做forward；
后续初始电机力为零，逐子步最大值覆盖所有已保存样本，不混用未执行的初始默认控制。
源反力记录显示中央深抓时主要向上压反力、手指却近乎完全闭合；表层52mm候选仍未持物，
源为 `20261113T051613_403348Z`，最高1.243／2.043mm、边长比1.10043。
当前恢复原装22mm／10mm目标指缝，保持虚拟惯量100与全部真实质量、材料、力限，检验过闭合是否排出夹持材料；
仍待完整物理与独立验收，不把一致性检查通过视为抓放通过。
中央22mm／10mm目标指缝仍未持物，源为 `20261113T052534_211281Z`。
随后选择Y侧边缘175mm／32mm抓高，使一根张开的手指处于400mm包裹之外；
提升增至450mm／4.5s，保证计划搬运时整包离台，下放150mm／3s并按原物体朝向规划落点。
该边缘组合的10mm目标指缝未持物；改为零目标后，两部分曾实际抬升262.4／263.7mm、
边长比1.15398，但提升末段滑落，7.6s触发台面穿入门限，仍拒绝，源为 `20261113T054043_843545Z`。
滑落前实际夹指电机力约10–15N；配置60N是上限，并非持续夹紧力。

新增 `grip_preload_n=0`，默认不改变既有控制。仅在无附着的双向proxy模式允许非零值，且不超过每指真实力限。
以 `q_ref=q_nominal-F_preload/kp` 表达每指闭合预紧，张开时平滑取消；
它是PD驱动平衡点，不是直接指定关节实际位置，机械行程与原生MuJoCo关节总力限保持不变。
运行时检查原生增益与输入范围确实允许该参考值，并逐子步／离线复核实际电机力没有超过60N。
每指20N预紧的完整尝试最高抬升71.9／73.7mm后失持，未采用，源为 `20261113T055630_357991Z`。
恢复零预紧，将原装短指抓高降低至12.5mm（最低碰撞点约2.5mm）的尝试，
最高抬升202.1／203.5mm后滑脱，5.85s触发5.54mm台面穿入早停；
边长比1.26742，源为 `20261113T060808_968532Z`，仍按原门限拒绝。
下一完整候选保持Y175mm边缘、零目标指缝、零预紧、6000N/m与60N·s/m夹指驱动、每指60N力限，
换回源V7的可见60×14×84mm长夹垫，按实测55mm的TCP到最低垫角偏差取57.5mm抓高；
释放TCP相应由150改为200mm，维持约145mm最低垫角高度，其余材料、真实质量与验收门限不改。
该抓法是明确的边缘位置与夹具组合优化，不能称为原中央90°条件的成功复现。
该200mm释放高度候选完成全转运：最高244.6／245.6mm、双侧接触100%、
全部顶点在整个转运期间最低离桌57.24mm，逐子步代理位置／姿态差6.39µm／0.000707°。
但下放至TCP约286mm时提前失去夹持，计划松爪尚未开始；13.85s保存失败前缀，
此时落点误差65.37mm、最大边长比1.15719、台面穿入0，仍不晋升，源为 `20261113T061810_993462Z`。
下一完整候选将释放TCP提高至350mm，避免继续向下压送悬挂包裹；
同时打开 `align_place_to_com`，在转运开始以实测悬挂质心修正剩余机器人路径。
只修正机器人目标，不改物体状态；原真实质量、材料、耦合、驱动力限和验收门限保持。
`20261113T063838_171652Z` 是读取前轮保存结果过早引起的旧配置重复启动，4步／0.067s即取消，
另记 `operator_note.json`，不作为夹持效果比较。新配置经工厂校验后另起连续完整尝试。
350mm／质心修正的完整尝试在下放末段13.583s达到边长比1.301166，按原1.3门限拒绝，
源为 `20261113T063907_686994Z`；此前全转运双侧接触100%、最大抬升239.3／239.9mm、
所有顶点最低离台50.14mm，最终采样落点误差36.75mm、台面穿入0。
下一完整尝试只把释放TCP由350提高至500mm，几乎保持提升终点高度，在着台前松爪。
另做转运末态重建的短时双向释放诊断，仅用于定位；有限差分速度和重置的ALM历史不构成原轨迹延续。
有效的直接释放诊断 `proxy_reconstructed_release02` 在3s内回落228.08mm、最大边长比1.026598、
台面穿入0，但按正式目标复算落点误差204.34mm；原始采样保留，不能直接晋升。
首个重建诊断被默认reset清除初态，已标记无效；修正后用flags=0并逐项断言初始粒子与关节状态等于源检查点。
有效诊断的初版摘要漏加 `place_offset_m`，随后按原始采样与完整目标独立复算修正，原摘要值另保留，未据错误摘要判通过。
新增默认零的 `release_compensation_xy_m`，在实测质心对齐的机器人目标上加XY前馈补偿，
不改变验收目标、物体状态、质量、材料或门限；无质心对齐时禁止启用非零补偿。
下一候选为500mm释放高度与(0,-200)mm补偿；同一URDF的补偿终点IK位置残差0.0024mm，仍须完整物理验收。
未补偿的500mm完整候选在释放前因方案替换取消，另记 `operator_note.json`，不当作夹持失败或通过。
500mm／(0,-200)mm补偿的完整物理与独立物理审计随后均通过，源为 `20261113T071102_032546Z`：
抬升239.7／240.4mm、转运接触100%、回落222.56mm、落点39.19mm、边长比1.111955、穿入0。
但完整源关节FK复算的闭合TCP误差6.612mm超过原5mm门限，姿态0.839°在原1°以内；
渲染器因此在创建数据集前拒绝，整条尚未正式验收。不能把物理成功视为全部成功。
闭合末原生记录显示一腕关节已达12N·m上限，其余近端关节仍低于力限，末0.5s仍有跟踪变化。
调参时曾误把类默认值1200／80当成生效值；完整源配置实际从父配置继承6000N·m/rad、300N·m·s/rad。
因此2400／120尝试实际降低了增益，闭合前缀独立检查为9.181mm／1.022°而拒绝，
源为 `20261113T074032_769488Z`；不能将其解释为“提高增益无效”。
Y200mm／2400／120分支在查明该配置错误后取消，未做抓取效果结论。
下一候选恢复已完整物理通过的Y175mm位置，真正将6000／300提高至12000／450，所有力限保持原值。
该12000／450尝试的闭合TCP误差降至5.949mm、姿态0.897°，仍超过原5mm门限而早停，
源为 `20261113T074850_427573Z`。实际关节与URDF雅可比分解显示，腕关节仍达12N·m，
双指向上反力合计约173N；不是配置未生效，原生控制目标与记录动作逐值相同。
下一轮只把抓高57.5提高至65mm，计划夹垫最低点由约2.5提高至10mm，
减轻深抓时的竖直挤压；Y175、12000／450、全部真实质量、材料和力限保持。
新增闭合前缀的同一独立5mm／1°FK检查，在提升前早停不合格候选；
测试用真实小URDF覆盖正确位姿通过、6.6mm误差拒绝，完整结束后的独立检查仍保留。
65mm抓高候选 `20261113T075617_845334Z` 在电脑关机时中断，最后持久化进度为2.65s闭合阶段；
尚无完整状态文件或闭合位姿检查点，不据此判断物理成功或失败。
恢复后以同一配置从初态独立重跑，编号 `20261113T081219_836453Z`；不拼接中断前后的轨迹。
重跑闭合末3.3s的独立实际关节FK检查为0.332mm／0.030°，通过原5mm／1°门限；完整抓放仍按原门限继续验收。
该65mm候选最高抬升257.1／258.3mm，但约6.4s双侧接触丢失，6.917s自然触发台面穿入早停；
边长比1.079624，未通过，不晋升。检查时工作进程已退出，未发送终止信号。
下一轮仅将抓高从65降至59.5mm，计划夹垫下缘约4.5mm，在此前57.5mm跟踪误差与65mm滑脱之间增加有效夹持深度；其余参数及原门限保持。
59.5mm候选 `20261113T082315_132207Z` 随后完整物理、独立审计、双相机数据和实际视觉复核全部通过：
闭合实际FK误差3.121mm／0.361°，衣物／袋膜最高抬升244.3／245.2mm，转运双侧接触100%，
整包最低顶点在全部转运期间离台至少55.77mm，释放回落226.63mm，落点误差36.35mm，
最大边长比1.106904、台面穿入0；无附着、无固定粒子，11个检查时刻的衣物包含检查通过。
逐子步代理误差最大6.14µm／0.000587°，实际原生夹指电机力全程不超过原60N上限；
曲线区分实际力与上限，不将上限当成持续夹紧力。
17.8s连续物理／1068帧，物理墙钟1549.48s，双相机各267帧／15fps／640×480 H.264；
原始源哈希、零渲染物理步、实际关节／可见几何和导出状态／动作逐值对齐均通过。
[90°正式结果](../outputs/physic_solver/package_cases/v8_yaw90/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v8_yaw90/validation.json)、
[关键帧](../outputs/physic_solver/package_cases/v8_yaw90/qa/keyframes.jpg)及
[实际力与整包净空曲线](../outputs/physic_solver/package_cases/v8_yaw90/qa/force_measurements.png)已实际查看；
腕部画面受到夹指／手掌遮挡，全景覆盖完整任务。
`run_note.json`保留本轮所有已保存的90°连续尝试／失败前缀、关机中断和启动取消记录，并另索引短时重建诊断。
当前成功仅为一条优化抓点episode，不是原中央90°的复现，也不证明硬件额定负载或统计可靠性。

组合偏移(-25,40)mm／偏航30°／俯仰10°的长夹块工况在4.95s达到边长比1.30551，
最高抬升209.2／204.1mm，仍按原门限拒绝，原始失败与前两项位于同一UTC运行编号。
后续候选保留位置和角度，共用45°控制组合，也在45°通过后独立运行。
该短指／40迭代候选在3.717s提升初段达到边长比1.30056，最高仅3.5／4.0mm，拒绝，
源在 `20261112T233129_431132Z`。
下一候选保留(-25,40)mm／30°／10°，继承已通过X35的宽夹垫基线，单独使用40迭代与0.5s平滑张开；
夹缝仍为12.5mm，避免将90°为解决滑落而收紧的参数直接用于此前拉伸受限的组合倾斜。
宽夹垫／40迭代在5.467s达到边长比1.32357，最高265.3／265.1mm，拒绝，
源在 `20261112T235420_125387Z`。
实际刚体帧复算夹紧后两垫最低点为12.878／17.674mm，最低角比已通过平抓基准15.394mm低2.516mm；
下一轮只将TCP抓高70.4提高为73.4mm，保留位置、角度、夹缝和精度。
复算记录为 `outputs/debug/physic_solver/migration_logs/tilt_pad_height_measurements.json`。
提高3mm的工况最高抬升284.2／283.8mm，仍于5.483s提升末段达到边长比1.33067，拒绝，
源在 `20261113T002424_376451Z`。接触仍在，但质心在末段反复升降；下一轮仅将 `lift_s` 从3提高为4.5s，
保留几何、位置、角度及32子步／40迭代。实际运动采用阶段前60%的三次平滑插值，
故运动时间从1.8延长为2.7s，其余时间保持；仍须完成转运、释放与原门限独立验收。
该慢速提升候选完成提升，但在8.750s的静止保持末段达到边长比1.300714，仍拒绝，
源在 `20261113T004913_209337Z`，最高抬升270.3／272.4mm。保持阶段TCP不再运动而质心仍振荡，
说明仅放慢提升不足以消除该抓法的形变尖峰。下一轮只将抓点从(-25,40)mm改为X35mm，
保留30°／10°、73.4mm抓高、4.5s提升、32子步／40迭代及原门限。
这是倾斜抓法的抓点优化，不再作为原(-25,40)mm位置的单角度对照；原位置失败仍保留。
X35抓点随后完整物理与独立审计通过：抬升250.3／249.5mm、最大边长比1.180762、
台面穿入0、释放回落121.1mm、落点误差18.70mm；16.3s物理墙钟55.6min。
源为 `package_migration/v8_offset_tilt10/20261113T012316_197098Z`。
[倾斜正式结果](../outputs/physic_solver/package_cases/v8_offset_tilt10/result.json)、
[独立验收](../outputs/physic_solver/package_cases/v8_offset_tilt10/validation.json)与
[关键帧](../outputs/physic_solver/package_cases/v8_offset_tilt10/qa/keyframes.jpg)均通过，
978物理帧／双相机245帧／15fps；实际偏航30.038°、俯仰10.467°、位置误差4.503mm、完整姿态误差0.496°。
源哈希、逐值状态／动作、FK对齐与实际视觉复核通过；6次物理尝试记录在 `run_note.json`，仍仅一条通过episode。
共享配置后的90°／组合倾斜预抓IK已重新检查可达，位置残差均小于0.0001mm；
记录在 `outputs/debug/physic_solver/migration_logs/shared_rotated_pregrasp_ik.json`，该检查不代替物理抓放验收。

<!-- PACKAGE_ACCEPTANCE_TABLE_START -->
### 9.8 本轮迁移矩阵的最终验收（2026-11-13 UTC）

18个柔性工况、2个原尺寸刚性纸箱及1个固定螺母／连续旋转主轴装配工况均通过物理、独立数据校验和实际视觉复核。
[完整验收索引](../outputs/physic_solver/package_migration_acceptance.json)包含各目录、配置和实测值。
每个配置仅有1条通过episode；失败、取消和关机中断均保留在debug，不据此估算成功率。

柔性工况沿用原门限：转运／摇晃两部分质心持续抬升≥80mm、双侧接触≥95%、最大边长比≤1.3、
台面穿入≤1.5mm、落点≤80mm、释放回落≥10mm，结束受台面支撑且夹指接触归零；衣物仍在袋内、无固定粒子。
附着仅在两项明确标记的spring配置中启用，采用有限力局部弹簧且在释放时关闭；其余16项无附着。

| 柔性配置 | 衣物／袋膜最高抬升 mm | 最大边长比 | 落点 mm | 夹指／耦合 | 子步×迭代 | 独立验收 |
|---|---:|---:|---:|---|---:|---|
| [shake_none](../configs/physic_solver/package_cases/shake_none.toml) | 248.4／244.4 | 1.126202 | 12.71 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/shake_none/validation.json) |
| [shake_spring](../configs/physic_solver/package_cases/shake_spring.toml) | 235.2／234.0 | 1.185115 | 26.00 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/shake_spring/validation.json) |
| [shake_spring_x35](../configs/physic_solver/package_cases/shake_spring_x35.toml) | 247.7／243.6 | 1.124944 | 14.90 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/shake_spring_x35/validation.json) |
| [v10_black_hoodie](../configs/physic_solver/package_cases/v10_black_hoodie.toml) | 257.4／255.4 | 1.112867 | 24.22 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v10_black_hoodie/validation.json) |
| [v10_frosted_top](../configs/physic_solver/package_cases/v10_frosted_top.toml) | 267.6／265.0 | 1.138811 | 6.18 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v10_frosted_top/validation.json) |
| [v10_pink_tshirt](../configs/physic_solver/package_cases/v10_pink_tshirt.toml) | 260.8／257.2 | 1.265344 | 14.44 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v10_pink_tshirt/validation.json) |
| [v10_station_clothing](../configs/physic_solver/package_cases/v10_station_clothing.toml) | 245.0／241.7 | 1.299698 | 20.86 | 原装短指／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v10_station_clothing/validation.json) |
| [v10_station_green](../configs/physic_solver/package_cases/v10_station_green.toml) | 265.8／262.2 | 1.162893 | 43.98 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v10_station_green/validation.json) |
| [v7_center](../configs/physic_solver/package_cases/v7_center.toml) | 242.0／239.9 | 1.144225 | 20.07 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v7_center/validation.json) |
| [v8_offset_tilt10](../configs/physic_solver/package_cases/v8_offset_tilt10.toml) | 250.3／249.5 | 1.180762 | 18.70 | V7长垫／单向 | 32×40 | [通过](../outputs/physic_solver/package_cases/v8_offset_tilt10/validation.json) |
| [v8_offset_x35](../configs/physic_solver/package_cases/v8_offset_x35.toml) | 246.1／242.1 | 1.134801 | 26.97 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v8_offset_x35/validation.json) |
| [v8_offset_y55](../configs/physic_solver/package_cases/v8_offset_y55.toml) | 226.4／222.0 | 1.208452 | 35.24 | 原装短指／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v8_offset_y55/validation.json) |
| [v8_yaw45](../configs/physic_solver/package_cases/v8_yaw45.toml) | 245.1／243.1 | 1.294322 | 51.37 | 原装短指／单向 | 32×40 | [通过](../outputs/physic_solver/package_cases/v8_yaw45/validation.json) |
| [v8_yaw90](../configs/physic_solver/package_cases/v8_yaw90.toml) | 244.3／245.2 | 1.106904 | 36.35 | V7长垫／双向proxy | 32×40 | [通过](../outputs/physic_solver/package_cases/v8_yaw90/validation.json) |
| [v9_mass100g](../configs/physic_solver/package_cases/v9_mass100g.toml) | 244.2／241.5 | 1.134073 | 19.42 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v9_mass100g/validation.json) |
| [v9_mass200g](../configs/physic_solver/package_cases/v9_mass200g.toml) | 242.8／239.4 | 1.127032 | 16.38 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v9_mass200g/validation.json) |
| [v9_mass350g](../configs/physic_solver/package_cases/v9_mass350g.toml) | 245.3／240.5 | 1.146418 | 22.31 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v9_mass350g/validation.json) |
| [v9_mass500g](../configs/physic_solver/package_cases/v9_mass500g.toml) | 245.0／241.6 | 1.148829 | 5.23 | V7长垫／单向 | 32×28 | [通过](../outputs/physic_solver/package_cases/v9_mass500g/validation.json) |

配置包含不同抓点、夹垫、控制和耦合组合，不能当成严格单因素求解器排名。
17项柔性记录为刚体影响柔体的单向耦合；90°为有独立代理几何和真实驱动力限检查的双向body代理。
两者均不构成真实机械臂额定负载认证。station_clothing的1.299698接近1.3门限，需更多重复试验评估余量。
原中央90°及原(-25,40)mm组合倾斜失败保留；当前通过的是文档中明确记录的优化抓点。
纸箱为刚性包围盒近似，不包含纸板压溃。螺纹装配已测得3.488圈／8.805mm，采用固定螺母与导向主轴；
尚未覆盖自由空间拾取、工具移走后的保持或指定预紧力认证。纸箱为单相机，柔性包裹与螺纹装配为双相机。
本地LeRobot v3文件、状态／动作和视频已校验，尚未使用官方LeRobot SDK加载。

<!-- PACKAGE_ACCEPTANCE_TABLE_END -->

## 10. 连续旋转主轴拧螺丝子任务

场景／规划／Job 后端为 `screw_assembly`，机器人为 `screw_spindle`，均在本任务既有文件末尾新增分节。
采用固定螺母、竖直导向装配臂、连续旋转主轴，螺栓头由显式刚性卡盘夹持。
导向机构限制横移和倾斜；该阶段不包含自由空间寻找、抓取螺栓、工具脱离或扭矩达到指定预紧力的工艺。
用户允许其他机械臂类型，本方案先验证真实螺纹旋入能力，不伪称已实现双臂装配。

### 10.1 资产、动力学与成功标准

使用 Newton 官方示例同源的 NVIDIA IsaacGymEnvs Factory M20 loose 螺栓／螺母三角网格，
复制到 `assets/PhysicSolver/Screw/M20/`，附来源 URL、SHA-256 和许可证。
本机官方 UI 教程使用M16；本任务用M20，不混用两者尺寸及螺距。初始啮合4mm匹配Newton M20示例；M16示例的0.5mm不适用于这里。
求解路径为 direct Newton：MuJoCo 刚体／关节 + Newton hydroelastic SDF 接触；120Hz、4子步，实际接触480Hz。
SDF 128、刚度1e11、摩擦0.01、零 margin/gap、elliptic 摩擦锥、接触容量40000、iso缓冲倍率2。

只有主轴转速目标与恒定轴向弹簧目标，没有“角度→轴向位置”的控制规则，没有螺旋关节，也不写螺栓位姿。
螺纹接触决定实际轴向位移。验收至少2圈、前进6mm，实测前进与圈数×2.5mm导程的相对误差不超过25%，
旋入阶段至少95%的物理帧观察到螺纹接触，停止旋转保持1s时轴向漂移不超过1mm，状态有限且无接触溢出。
全程轴向位移与连续角度×导程的最大偏差不超过2mm，防止先穿入再旋转的误判。
保留卡盘和导向机构的最终保持不等于移走工具后仍已达到真实拧紧预载。

### 10.2 运行

```bash
./app/python.sh scripts/physic_solver/prepare_screw_assets.py
./scripts/rsr collect run -c configs/physic_solver/mujoco_hydroelastic_screw_debug.toml
./scripts/rsr collect run -c configs/physic_solver/mujoco_hydroelastic_screw.toml
```

debug 配置先只跑物理；正式配置记录装配近景、轴向位移m、连续主轴角rad、轴向弹簧目标m与转速目标rad/s。
独立验收入口为 `scripts/physic_solver/validate_screw_assembly.py`。

### 10.3 本轮正式结果（2026-11-12 UTC）

[正式结果](../outputs/physic_solver/screw_assembly/result.json) 与
[独立校验](../outputs/physic_solver/screw_assembly/validation.json) 均通过：
3.488圈、前进8.805mm、导程误差0.983%、全程非导程位移最大0.323mm、保持漂移0.200mm，
旋入接触帧97.02%、逐子步最大接触108（容量40000），135帧／15fps／两路640×480 H.264，LeRobot v3。
实际配置为轴向200N/m／8N·s/m、主轴-3rad/s、建立接触1s／旋入7s／保持1s。
装配相机与螺纹相机的初、中、末帧均已视觉复核，记录见 `qa/visual_review.json`。
由原始测量绘制的 [导程、全程偏差与接触曲线](../outputs/physic_solver/screw_assembly/qa/thread_measurements.png)
展示实际轴向进给与实测转数×螺距的关系，保持阶段以浅色标出。
曲线从初始帧计零；上面的8.805mm是旋入阶段的进给，不含建立接触阶段约0.25mm的轴向回退。

失败尝试留在 `outputs/debug/physic_solver/screw_*`：0.5mm初始啮合导致入口非导程移动；
降低hydroelastic刚度导致明显穿入；单看旋入阶段导程会漏掉该问题，已增加全程检查。
修正M20初始布局后物理与双相机采集均重复通过，再进行正式采集。
未完成工具脱离、螺栓自由抓取及预紧扭矩验证；未使用官方LeRobot SDK加载，已做本地全视频解码及数据校验。

另做了两个真实物理负对照，保持相同轴向预载、螺纹几何、接触和时长：

- 只将主轴速度指令归零时，原0.02N·m·s/rad有限阻尼允许预载反驱，实测0.17869圈／0.44125mm，
  与2.5mm螺距相符；“主轴近乎不动”判据未通过，原失败记录保留于 `screw_no_rotation_01`。
- 归零指令并将实际阻尼增至2N·m·s/rad制动，转动仅0.001837圈、旋入阶段进给0.00458mm，
  全程非导程位移最大0.266mm；[负对照验收](../outputs/debug/physic_solver/screw_braked_spindle_01/negative_control.json)通过。
  未规定轴向位移或转角，扭矩上限仍0.3N·m。该结果支持正式工况的进给来自螺纹旋转耦合，
  而非预载直接压穿螺纹；制动配置只用于debug，不替换正式控制参数。

## 11. 本轮代码与配置回归

2026-11-13 UTC：新增可选双向body代理、有界驱动预紧、机器人释放目标补偿及闭合位姿早期检查后，`./app/python.sh -m pytest -q` 为261通过、3跳过（28.15s）。
三个跳过项需要USD／Newton导入路径；本轮另在本机Isaac SDK环境逐项运行，3项全部通过（6.12s），
覆盖材质引用、长夹垫替换碰撞体和原始轴向几何；后续增量未修改这些资产及几何路径。没有以stub替代物理采集。
`ruff check real2sim2real scripts tests/test_physic_solver_migration.py`、`list-backends`、
任务skill的全配置检查脚本与 `git diff --check` 均通过；新增后端可见，通用ABC没有增加子任务字段。
最近日志保留在 `outputs/debug/physic_solver/migration_logs/rsr_posecheckpoint_*0743.log`，此前基线和SDK专项日志也保留。

额外依赖未变，仍使用Isaac `python.sh`；只跑SDK物理时，启动器选择本机Newton1.6.1／Warp1.17，
避免误用普通Python路径里的其他Warp版本。没有创建`.venv`，没有commit或push。
