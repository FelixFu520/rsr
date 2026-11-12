# 物理求解器验证：Franka 抓放衣服、刚体方块与 V7 包裹

更新日期：2026-11-11。

## 1. 任务介绍

固定在桌面上的 Franka 机械臂执行一次完整抓放：接近 → 下探 → 闭合 → 抬升 → 保持 → 横向转运 →
下放 → 松开 → 撤离 → 静置。被抓物体分三类：

- **衣服**：一件 0.2 kg 的衬衫（质量按三角面积分配），无固定顶点、无抓取附件。
- **刚体方块**：0.1 kg 方块，全部由 PhysX 求解，作为刚体对照。
- **V7 包裹**：袋膜（约 20.54 g）+ 内部衣物（200 g）两个物理网格，衣物内部有被动弹簧。

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
| `mujoco_vbd_pickplace_package` | Newton direct | FR3 原装短指，位置控制 | MuJoCo / VBD | V7 包裹 | 通过 |
| `featherstone_vbd_pickplace_package` | Newton direct | FR3 + V7 长夹块 | Featherstone / VBD | V7 包裹 | 对照组，未通过验收 |
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
| [prepare_physic_solver_assets.py](../scripts/physic_solver/prepare_physic_solver_assets.py) | 从 `assets/Demo` 复制桌子、衣服、方块、Franka 等到 `assets/PhysicSolver` |
| [prepare_official_newton_cloth.py](../scripts/physic_solver/prepare_official_newton_cloth.py) | 从 Isaac Sim 自带 Newton 复制官方 `unisex_shirt.usd`、许可证和来源记录 |
| [prepare_package_assets.py](../scripts/physic_solver/prepare_package_assets.py) | 复制 V7 包裹快照；`--source/--target` 也用于分拣任务的 V10 快照 |
| [check_package_pickplace.py](../scripts/physic_solver/check_package_pickplace.py) | 包裹抓放只跑物理、不录 RGB，用于低成本调参 |
| [validate_package_pickplace.py](../scripts/physic_solver/validate_package_pickplace.py) | 不启动 Kit，复核包裹数据集的帧对齐、双相机解码和粒子几何 |
| [validate_physics_modes.py](../scripts/physic_solver/validate_physics_modes.py) | 各世界模式的自由落体与“只渲染不积分”冒烟测试，每次一个 Kit 进程 |

### 3.2 资产

所有资产放在 `assets/PhysicSolver/`（git 忽略），全部是独立副本：

| 目录 | 内容 |
|---|---|
| `Table/`、`Cube/`、`Cloth/` | 桌子、刚体方块、项目衣服 |
| `Cloth/newton_official/` | 官方 Newton 衬衫 + `LICENSE.md` + 带 SHA-256 的 `provenance.json` |
| `Robot/franka_emika_panda/` | FR3 URDF（Newton）与 Panda USD（PhysX） |
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

- 衣服、包裹都是单向耦合：机器人推动柔体，柔体反力不回馈机械臂。数据能验证动作流程、
  形变和持续抓持，不能评价真实夹爪的额定负载或限力性能。
- Newton 1.6.1 的 Featherstone、XPBD 不执行 `joint_effort_limit`，`finger_max_force` 只在
  PhysX / MuJoCo 路径生效。
- V7 长夹块密度 1000 kg/m³（每块约 70.56 g）是仿真假设，未做硬件标定。
- 几何检查只覆盖保存的物理帧，不构成连续时间无穿透的证明。
- 每个组合只采一条固定初始状态的 episode。
