# 桌面抓放：Franka 在 home_000 家居桌面上用 cuMotion 抓起木块放到放置位

更新日期：2026-11-14（运行时间记 UTC）。

## 1. 任务介绍

从 Real2Sim2Real（`/home/fufa/Projects/Real2Sim2Real`，`configs/tabletop-pickplace_franka_curobo_home000.toml`，
`TabletopJob` + `TabletopCuroboPlanner` + `TabletopScene` + `FrankaRobot`）迁入的桌面抓放采集。家居背景里放一张桌子、
一块 5 cm 木块（抓取目标）和一个苹果（干扰物），Franka Panda 站在桌面上，把木块抓起放到桌面放置位。
共 2 集，每集木块在名义位置附近 XY 均匀抖动 ±0.12 m（seed 42，与源项目同一随机序列）。俯视和腕部两路 RGB 与
关节状态、动作一起导出为 LeRobot v3。

用户要求（2026-11-14）：抓放采集与轨迹回放拆成两个任务；规划器用 Isaac Sim 自带的 cuMotion，不另装库。
回放任务见 [tabletop_replay.md](tabletop_replay.md)。

| 环节 | 源项目 | 本任务 |
|---|---|---|
| 规划 | cuRobo `MotionPlanner.plan_pose`（需 pip 安装 `nvidia-curobo`），每段到点规划，无世界障碍 | Isaac Sim 6.2 扩展 `isaacsim.robot_motion.cumotion` 的 `TrajectoryOptimizer`；竖直段与转运段加直线路径约束；苹果与桌面作为障碍 |
| 执行 | `SingleManipulator` + PD 关节目标，每个路点停 2 步 | `isaacsim.core.experimental.prims.Articulation` 关节位置目标 + 速度前馈，按 120 Hz 采样连续轨迹 |
| 抓取 | 合爪后置 `attached=True`，“合过爪且走完”即成功 | 纯摩擦夹持（USD 指驱动 7.2 N 上限，第二指 mimic），无附着；成功由物理验收决定 |
| 放置高度 | 放置点 z = 桌面 + 0.025 m（木块中心） | 桌面实测高度 + 木块半高 + 0；抓取后实测的手-物偏移保留到放置段 |
| 数据 | LeRobot v2.1，状态 9 维、动作 7 维（无夹爪），15 fps | LeRobot v3，每集一个数据集；状态 9 维、动作 8 维（含指目标），正式 30 fps |

### 1.1 流程

```
init_world      新建 stage，引用 home_000 背景（唯一的动态刚体设为 kinematic）；PhysX CPU TGS 120 Hz；
                启用 isaacsim.robot_motion.cumotion 扩展
arrange_scene   桌子引用到 /World/scene/Table（源配置位姿），桌面网格加静态三角网格碰撞，量桌面高度
                木块、苹果按桌面系偏移摆放，AABB 底面高出桌面 1 mm；运行时覆盖质量（木块 0.1 kg、苹果 0.18 kg）
                俯视相机（源 SG5 光学中心位姿与内参的针孔 Camera，3.3 节）
place_robot     before hook：把实测桌面交给机器人
                Franka USD 引用；底座落到桌面，脚印超出桌边时整体移回桌面内（源项目 snap_base_to_support 同口径）
                腕部相机挂在 panda_hand（同上）
                after hook：加载 cuMotion Franka 描述，加入苹果与桌面障碍，检查起始姿态无碰撞
start_simulation
  before hook   机器人各连杆和两个物体加 PhysX 接触报告
  simulate      每集：停止 → 写本集木块位置 → reset → play → 张量视图 → 写起始关节
                → 每个物理步：规划器下发关节目标 → 记 120 Hz 状态 → 每 4 步（30 fps，调试每 8 步）两路相机各采一张 → PhysX 一步
                → 结束后在完整状态记录上验收 → 导出 episode_XXX/lerobot_v3 + meta/physics
finally: close
```

规划阶段：`start`（0.25 s 保持，木块落稳）→ `approach_pick`（到木块中心上方 0.223 m）→ `descend`（直线下降到 0.103 m）
→ `close`（0.6 s，源 36 步）→ `lift`（直线抬 0.12 m）→ `transfer`（直线到放置位上方）→ `lower`（直线下降）
→ `open`（0.2 s，源 12 步）→ `retreat`（直线抬 0.12 m）→ `rest`（1 s 保持，供静止验收）。
每段运动结束后保持 0.067 s（源 4 步）；下一段是开合爪时保持 0.267 s（源 16 步）。

### 1.2 约束与验收口径

- 数据只来自物理：初始状态之后不写任何刚体位姿，不加附着约束，运行中不改 USD。
- 机器人用 Isaac Sim Franka Panda 资产及其自带驱动（臂刚度 400/°、阻尼 80，力矩 87/12 N·m；指 400 N/m、7.2 N），原装夹爪。
- 木块资产自带质量 1 g，按 5 cm 实木块改为 0.1 kg；苹果资产无质量（按默认密度约 0.4 kg），设为 0.18 kg。只改运行时 stage。
- 验收项与门限是本任务新定的（源项目没有物理验收），见第 10 节。

## 2. 当前结果

### 正式（1K / 30 fps PathTracing）

已完成：2026-11-14 21:54–23:50 UTC（America/New_York 当日 16:54–18:50），GPU 0 单卡，墙钟 6943.6 s。
输出：[franka_physx_home000](../outputs/tabletop_pickplace/franka_physx_home000/)，
两集均通过全部 **14 项物理验收**和独立校验；共 546 帧组，每帧组为俯视与腕部各一帧。

| 项 | 第 0 集 | 第 1 集 | 门限 |
|---|---|---|---|
| 帧数 / 仿真时长 | 276 / 9.175 s | 270 / 8.975 s | 1024×768 / 30 fps |
| 木块抬升 | 119.77 mm | 119.77 mm | ≥ 80 mm |
| 持握滑移（手系） | 0.204 mm | 0.165 mm | ≤ 10 mm |
| 放置误差（XY） | 0.225 mm | 0.345 mm | ≤ 20 mm |
| 苹果位移（机械臂运动后） | 0.183 mm | 0.001 mm | ≤ 10 mm |
| 臂连杆 / 手指异物 / 木块异物接触冲量 | 0 / 0 / 0 N·s | 0 / 0 / 0 N·s | ≤ 1e-3 / 0.01 / 0.01 |
| 直线段最大偏差 | 2.36 mm | 1.63 mm | ≤ 5 mm |
| RGB 采集墙钟 | 3491.56 s | 3393.99 s | 实测平均 12.6 s / 帧组 |

独立校验：[validation.json](../outputs/tabletop_pickplace/franka_physx_home000/validation.json)，
覆盖视频完整解码、H.264、帧数/fps/分辨率、数据与 120 Hz 物理记录一致、重算验收一致、非黑帧。
接手后 agent 查看了两集 `qa/keyframes.jpg`（每集两路相机各 6 个关键帧）：木块由两指夹起、转运、释放后落在放置位，
苹果未见被碰动；该检查不代表用户视觉复核。

### 调试（640×480 / 15 fps MinimalRendering）

`outputs/debug/tabletop_pickplace/franka_physx_home000_debug/`，2026-11-14 21:52 UTC，最终相机与采集设置
（针孔相机、`parallel_cameras`）。两集都通过 Job 内 14 项验收和独立校验（视频解码 / 帧数 / fps / 分辨率、
数据行与 120 Hz 物理记录逐帧一致、重算验收一致、关键帧无黑帧）。

| 项 | 第 0 集 | 第 1 集 | 门限 |
|---|---|---|---|
| 木块抖动 (dx, dy) | (0.0657, −0.0147) m | (0.0861, 0.0474) m | 源项目同一序列 |
| 帧数 / 仿真时长 | 138 / 9.18 s | 135 / 8.98 s | — |
| 抬升 | 0.120 m | 0.120 m | ≥ 0.08 m |
| 持握滑移（手系） | 0.20 mm | 0.16 mm | ≤ 10 mm |
| 放置误差（XY） | 0.23 mm | 0.35 mm | ≤ 20 mm |
| 落桌间隙 / 倾斜 | 0.0 mm / 0.00° | 0.0 mm / 0.00° | ≤ 3 mm / ≤ 5° |
| 撤离时带起 / 最终指开度 | 0.0 mm / 0.040 m | 0.0 mm / 0.040 m | ≤ 3 mm / ≥ 0.036 m |
| 苹果位移（机械臂运动后） | 0.18 mm | 0.0 mm | ≤ 10 mm |
| 臂连杆 / 手指异物 / 木块异物接触冲量 | 0 / 0 / 0 N·s | 0 / 0 / 0 N·s | ≤ 1e-3 / 0.01 / 0.01 |
| 两指与木块接触冲量 | 14.5 / 15.9 N·s | 15.3 / 16.9 N·s | > 0 |
| 直线段最大偏差 | 2.4 mm（retreat） | 1.6 mm（lift） | ≤ 5 mm |
| 墙钟（控制 / 采集 / 物理 / 导出） | 1.8 / 19.1 / 2.4 / 0.5 s | 1.7 / 18.6 / 2.3 / 0.5 s | — |

视觉复核：本次由 agent 查看了两集的 `qa/keyframes.jpg`（俯视与腕部各 6 帧：木块被两指夹起、转运、落在放置位，苹果未动），
不代表用户复核。调试输出中 `outputs/debug/tabletop_pickplace/` 另保留了冒烟、探针与旧相机设置的运行（见第 11 节）。

## 3. 设计选择

### 3.1 物理引擎与求解器

PhysX CPU TGS，120 Hz。场景只有刚体与关节（木块、苹果、Franka），Franka USD 带 PhysX 驱动与 mimic 指关节，
源项目也是 PhysX；按 [engine-selection](../.agents/skills/rsr-task-development/references/engine-selection.md) 决策顺序选 PhysX。
物理求解器验证任务的 `tgs_tgs_rigid01_baseline` 在同一引擎、同一频率下抓放刚体方块落点误差 2.6 mm。

### 3.2 规划：cuMotion

`isaacsim.robot_motion.cumotion` 是 Isaac Sim 6.2 自带扩展（`app/exts/isaacsim.robot_motion.cumotion`，1.5.0，原生库 1.1.0a2），
提供 `cumotion` 原生库和 Franka 的 URDF + XRDF（`robot_configurations/franka`，cspace 为 7 个臂关节，工具系含 `panda_hand`）。
本任务直接调用原生 API，不使用 `WorldBinding` / `SceneQuery`：障碍只有两个盒体（苹果 AABB 外扩 1 cm，位置取 `start`
保持结束时的实测值；桌面板顶面在实测桌面下 1 cm、厚 0.2 m），写在基座系里。末端朝向固定为基座系 (0, 1, 0, 0)（手掌朝下），
与源项目 `_DOWN_QUAT_WXYZ` 相同。

| 段 | 做法 |
|---|---|
| `approach_pick`（自由段） | `solve_ik` 以当前关节为种子求预抓位（保持同一臂构型，失败再用起始姿态做种子），终点做碰撞检查；再用 `TrajectoryOptimizer.plan_to_cspace_target` 规划带障碍的关节轨迹 |
| `descend` / `lift` / `transfer` / `lower` / `retreat`（直线段） | `TaskSpacePathSpec.add_linear_path` → `convert_task_space_path_spec_to_cspace`（以当前关节为 IK 种子）→ `CSpaceTrajectoryGenerator` 按关节速度 / 加速度 / 加加速度限的 50% 做时间参数化；采样后用 FK 检查手到直线的偏差（> 5 mm 即规划失败）、终点误差（> 1 mm 失败）和碰撞 |

- 轨迹按 120 Hz 采样位置与一阶导数，作为关节位置目标与速度目标下发。
- 抓取点在 `start` 保持结束时从 PhysX 读木块 AABB 中心；`lift` 开始时读实际手-物偏移，后续放置目标按这个偏移补偿，
  使木块中心落在放置位。

被否决：

- `TrajectoryOptimizer` 的 `linear_path_constraint`：冒烟运行 1 中第 1 集下降段的指令手轨迹偏离直线 20 mm、关节 1 摆动约 1 rad，
  PhysX 跟踪滞后最大 0.137 rad，手停在抓取高度上方 3.7 cm，未抓到木块（第 11 节）。约束在优化里是代价项，不是硬约束。
- cuRobo（需另装 `nvidia-curobo`，用户要求不装库）；RMPflow 控制器（反应式，不保证直线段与到达时间）；Lula IK 逐步跟踪（无碰撞规划）。

### 3.3 相机与采集

源项目两路相机都引用 Sensing SG5-IMX490C H110SA USD（壳体隐藏，用其 `Camera_SG5_IMX490C_5300_GMSL2_H110SA` 渲染）。
本任务改为同位姿、同内参的针孔 USD Camera：

- 位姿：源配置壳体位姿与壳体内 Camera prim 的局部旋转 (-90, 180, 90) 合成后的光学中心位姿（配置里的四元数）。
- 内参：SG5 Camera prim 的焦距 4.26 mm、水平光圈 8.64 mm（水平视场 90.8°）。分辨率保持源配置的 4:3：调试 640×480，正式 1024×768。
- 理由：Isaac Sim 6.2 RTX 不支持其 f-theta 鱼眼投影，日志 "Unknown projection type, defaulting to pinhole"，实际就按上面的针孔内参渲染；
  而对 SG5 Camera prim 建固定 RenderProduct 时 Replicator 报 "renderer failed to advance to the scheduled frame"，连续 30 次取不到图像
  （改投影为 pinhole 也一样），同一位姿的普通 Camera 第一次就成功。渲染画面与源项目帧的取景一致（agent 对比查看）。
- 采集：`capture_mode="parallel_cameras"`（复用 `battery_disassembly_utils.BatteryCameraCapture`），每路相机一个固定 RenderProduct，
  每次采集前后核对物理步数与相机矩阵。视口模式在两路相机间切换，每张图都要重置路径追踪累积，1K PathTracing 实测约 38 s / 帧组；
  固定 RenderProduct 约 23 s / 帧组。调试配置继承同一采集方式。
- `SensorCameraConfig` 仍支持引用传感器 USD（仅视口采集），当前配置不使用；`assets/TabletopPickplace/Sensor/` 保留源资产。

## 4. 组成

| 模块 | 后端 | 实现 | 迁移对照 |
|---|---|---|---|
| SimWorld | `isaacsim` | [simworld/isaacsim.py](../real2sim2real/simworld/isaacsim.py) | `IsaacsimSimWorld` |
| Scene | `tabletop_home` | [scene/scene_tabletop_pickplace.py](../real2sim2real/scene/scene_tabletop_pickplace.py) | `TabletopScene` |
| Robot | `franka_tabletop` | [robot/robot_tabletop_pickplace.py](../real2sim2real/robot/robot_tabletop_pickplace.py) | `FrankaRobot` |
| Planner | `cumotion_pickplace` | [planner/planner_tabletop_pickplace.py](../real2sim2real/planner/planner_tabletop_pickplace.py) | `TabletopCuroboPlanner` |
| Job | `tabletop_pickplace` | [job/job_tabletop_pickplace.py](../real2sim2real/job/job_tabletop_pickplace.py) | `TabletopJob` |
| 辅助 | — | [utils/tabletop_pickplace_utils.py](../real2sim2real/utils/tabletop_pickplace_utils.py) | `misc.jitter_world_xy`、`snap_base_to_support` 等 |

迁移后的调整：桌面语义（桌面高度、桌面系偏移）放在本任务的场景类里，不进通用 ABC；
源项目的 `TabletopScene.apply_object_jitter` 写 `poses_realtime` 后瞬移物体，这里改为每集停止时间轴后写 USD 再 reset，
episode 中途不瞬移；抖动位置若与苹果脚印间隙小于 2 cm 或离桌边小于 5 cm 则重抽（seed 42 的两集都一次通过）。

### 4.1 脚本

| 脚本 | 用途 |
|---|---|
| [prepare_tabletop_pickplace_assets.py](../scripts/tabletop_pickplace/prepare_tabletop_pickplace_assets.py) | 复制资产与依赖闭包，写 `provenance.json` |
| [collect_tabletop_pickplace.py](../scripts/tabletop_pickplace/collect_tabletop_pickplace.py) | 正式采集入口：进程内单调预算时钟，debug 暂存，独立校验通过后发布；拒绝覆盖旧结果 |
| [validate_tabletop_pickplace.py](../scripts/tabletop_pickplace/validate_tabletop_pickplace.py) | 不启动 Kit 的独立校验（视频、数据行与物理记录一致、重算验收、关键帧） |

## 5. 资产

`assets/TabletopPickplace/`，976 MB，564 个文件，目录与源项目 `assets/` 相同：

```text
assets/TabletopPickplace/
├── Background/Home/home_000/     interior_template.usdc + textures（207 个网格碰撞体，11 盏矩形灯）
├── Scene/Table/table_000/        桌子（0.60 × 1.00 × 0.73 m，只有视觉网格）
├── Scene/Cube/cube_000/          5 cm 木块（刚体，boundingCube 碰撞，摩擦 2.0）
├── Scene/Fruit/apple/apple_000/  苹果（刚体在 /World/body，凸分解碰撞）
├── Robot/FrankaPanda/            franka.usd + configuration/ Materials/ Props/ + franka-LICENSE.txt
├── Sensor/Sensing/SG5/H100SA/    SG5 相机 USD（源资产；当前配置不引用，见 3.3 节）
└── provenance.json               来源、许可证、依赖检查、每个文件的 SHA-256
```

准备命令：`./app/python.sh scripts/tabletop_pickplace/prepare_tabletop_pickplace_assets.py`（源只读、逐字节复制、
目标已有不同文件时拒绝覆盖）。复制后逐个打开 USD，`GetUsedLayers()` 全在本目录内，必需 prim 都在。

| 来源 | 内容 | 许可证 |
|---|---|---|
| Real2Sim2Real `assets/Background/Home/home_000` | 家居背景 | 源项目未记录，再分发前需确认 |
| Real2Sim2Real `assets/Scene/{Table,Cube,Fruit}` | 桌子、木块、苹果 | 源项目未记录，再分发前需确认 |
| Real2Sim2Real `assets/Robot/FrankaPanda` | Isaac Sim Franka Panda | `franka-LICENSE.txt`（随资产复制） |
| Real2Sim2Real `assets/Sensor/Sensing/SG5/H100SA` | SG5 相机 | Isaac Sim 传感器资产条款（源项目未记录） |

未复制：Kit 内置 `OmniPBR.mdl` / `OmniPBR_Opacity.mdl`（Kit MDL 搜索路径解析）；cuMotion Franka 描述（Isaac Sim 扩展自带）。
依赖检查唯一未解析项是 Franka USD 未选中的 `Robotiq_2F_85` 夹爪变体引用的 `Robotiq_2F_85_edit.usd`（源项目也没有），
本任务用 `Default` 夹爪变体，不受影响。对源资产无改动。

## 6. 依赖与安装

无额外 pip 依赖。cuMotion 来自 Isaac Sim 扩展 `isaacsim.robot_motion.cumotion`，由配置 `sim_world.extensions` 启用；
验证：Kit 进程里 `import cumotion` 成功、`load_cumotion_supported_robot("franka")` 返回 7 个关节。
正式采集器默认复用电池任务的 Linux x86_64 单调预算时钟补丁，需要 `cc`（Debian/Ubuntu：`build-essential`）；
源码、共享库及 SHA-256 留在运行目录的 `capture/native/`。补丁只替换 GPU 库的计时，不改物理时钟或画质。
其它平台可显式传 `--system-budget-clock` 使用系统计时；该模式不能规避本机已出现的回跳问题。

## 7. 配置

| 文件 | 用途 | 输出 |
|---|---|---|
| [franka_physx_home000.toml](../configs/tabletop_pickplace/franka_physx_home000.toml) | 正式：1024×768 / 30 fps PathTracing 64 spp、6 次反弹、OptiX 降噪、同步采集，GPU 0，两路固定 RenderProduct | `outputs/tabletop_pickplace/franka_physx_home000/` |
| [franka_physx_home000_debug.toml](../configs/tabletop_pickplace/franka_physx_home000_debug.toml) | 调试：640×480 / 15 fps MinimalRendering（RTX 纹理预览），可覆盖、失败也导出 diagnostic | `outputs/debug/tabletop_pickplace/franka_physx_home000_debug/` |

调试与正式只差采集设置，物理参数与验收门限相同。主要字段：

| 字段 | 取值与理由 |
|---|---|
| `sim_world.physics_dt` | 1/120 s；30 fps 每 4 步一帧，15 fps 每 8 步 |
| `scene.objects[].mass_kg` | 木块 0.1、苹果 0.18（见 1.2 节） |
| `robot.base_offset_m` | (-0.35, 0)，源配置；超出桌边时移回桌面内 |
| `planner.hand_above_object_m` / `approach_clearance_m` | 0.103 / 0.12 m，源配置 |
| `planner.place_offset_m` | (0.15, -0.15)，源配置 `place_on_table` 的 XY |
| `planner.linear_speed_scale` / `path_deviation_m` | 0.5 / 0.005 m：直线段按关节限的一半时间参数化，偏差超过 5 mm 视为规划失败 |
| `job.recorder.capture_mode` | `parallel_cameras`（3.3 节） |
| `job.object_jitter_xy_m` / `object_jitter_seed` | 0.12 / 42，源配置 |

## 8. 运行

```bash
./app/python.sh -m pip install -e ".[dev]"
./app/python.sh scripts/tabletop_pickplace/prepare_tabletop_pickplace_assets.py
# 只跑物理（不录 RGB）
RSR_JOB__RECORDER__record_rgb=false ./scripts/rsr collect run -c configs/tabletop_pickplace/franka_physx_home000_debug.toml
# 调试全流程
./scripts/rsr collect run -c configs/tabletop_pickplace/franka_physx_home000_debug.toml
# 正式采集与独立校验
./app/python.sh scripts/tabletop_pickplace/collect_tabletop_pickplace.py
./app/python.sh scripts/tabletop_pickplace/validate_tabletop_pickplace.py outputs/tabletop_pickplace/franka_physx_home000
```

GUI：`RSR_SIM_WORLD__HEADLESS=false`，`RSR_JOB__INSPECT_GUI=true` 在各阶段停下等 Enter。
正式输出目录已有内容时拒绝运行；`overwrite_dataset=true` 仅允许输出到 `outputs/debug/` 的子目录，不能用于覆盖正式结果。
正式运行未通过验收时整个输出目录移到 `outputs/debug/tabletop_pickplace/`。
正式采集器在 `outputs/debug/<task>/<UTC>/episodes/` 暂存，全部集独立校验通过后才移动到正式目录；
`status.json` 给出状态与日志路径，正式目录的 `capture_manifest.json` 记录生效世界配置与时钟补丁。
发布只更新结果/校验回执里的路径，`meta/physics` 和 `task_outcome.configuration` 保留真实执行配置。
历史正式抓放结果来自直接 CLI 采集，已独立验收，未追溯补写时钟清单。

## 9. 输出与数据格式

```text
outputs/tabletop_pickplace/franka_physx_home000/
├── result.json                    全部集汇总
├── validation.json                独立校验汇总
├── diagnostics/<UTC>/             configuration.json、run_result.json、每集 state_log.npz / contacts.json / layout.json / acceptance.json
└── episode_000/, episode_001/
    ├── result.json, validation.json, qa/keyframes.jpg
    └── lerobot_v3/                data/、videos/observation.images.{overhead_cam,wrist_cam}/、meta/（含 physics/episode_000000/）
```

| 列 | 维度 | 内容与单位 |
|---|---|---|
| `observation.state` | 9 | `panda_joint1..7`（rad）、`panda_finger_joint1/2`（m），PhysX 实测 |
| `action` | 8 | 同一帧下发的 7 个臂关节位置目标（rad）+ `panda_finger_joint1` 目标（m，第二指由 mimic 跟随） |
| `observation.object_pose` | 7 | 木块 AABB 中心 xyz（m，世界系）+ 朝向 wxyz |
| `observation.ee_pose` | 7 | `panda_hand` 世界位姿 |
| `observation.phase_index` | 1 | 规划阶段序号（`utils/tabletop_pickplace_utils.PHASES`） |
| `next.done` / `next.success` | 1 | 最后一帧为 true / 验收结果 |

`task` = "pick the cube and place it on the table"，`robot_type` = `franka_panda`。

## 10. 验收

Job 在完整 120 Hz 状态记录与接触报告上计算，独立校验脚本从落盘文件重算并比对：

| 检查 | 含义（门限） |
|---|---|
| `finite_states` | 记录无 NaN |
| `sequence_completed` | 规划器走完全部阶段 |
| `object_lifted` | 持握期木块最高点比起始高 ≥ 0.08 m |
| `object_held_without_slip` | 持握期木块中心在手系中的漂移 ≤ 0.01 m |
| `placed_at_target` | 最终 XY 与放置位误差 ≤ 0.02 m |
| `resting_on_table` | 最终底面与桌面间隙 ≤ 3 mm |
| `object_upright` | 最终倾斜 ≤ 5° |
| `object_at_rest` | 最后 0.5 s 位移 ≤ 2 mm |
| `object_released` | 撤离时木块被带起 ≤ 3 mm，最终指开度 ≥ 0.036 m |
| `distractors_undisturbed` | 机械臂开始运动后苹果位移 ≤ 0.01 m（`start` 保持期的落稳位移单独记为 `distractor_settle_m`） |
| `arm_links_collision_free` | 臂与手掌连杆接触冲量 ≤ 1e-3 N·s |
| `fingers_touch_only_target` | 手指与桌面、苹果、环境的接触冲量 ≤ 0.01 N·s |
| `target_touches_only_table_and_fingers` | 木块与苹果、环境、臂的接触冲量 ≤ 0.01 N·s |
| `both_fingers_contacted_target` | 两指都与木块有接触 |

独立校验另查：视频帧数 / fps / 分辨率 / H.264 / 完整解码；数据行与物理记录逐帧一致（≤ 1e-5，float32 存储）；
关键帧非黑帧并写 `qa/keyframes.jpg`。人工视觉复核状态写为 `pending_visual_inspection`，查看后再更新。
agent 查看后另存 `qa/visual_review.json`（含拼图 SHA-256、检查范围和结论），并将 `validation.json` 的
`visual_review` 标为 `agent_reviewed_keyframes`；用户复核状态独立记录。重新执行独立校验会重建拼图并重置复核状态。

## 11. 问题与解决方案

调试与探针输出都在 `outputs/debug/tabletop_pickplace/`（日志同名 `.log`）。

| 现象 | 原因 | 处理 |
|---|---|---|
| 源项目底座名义 x = 0.10 m 在桌边外 | 桌子 x 范围 0.15–0.75 m | 与源项目一样把底座脚印移回桌面内（源实测基座 x = 0.304 m，本任务相同） |
| 木块资产质量 1 g | 资产作者未按实物设定 | 运行时覆盖为 0.1 kg（见 1.2 节） |
| 木块 / 苹果资产 `xformOp:orient` 为 float 四元数，追加 double 变换操作冲突 | 资产自带 translate/orient/scale | 运行时只写一个 `xformOp:transform:rsr`，清掉原操作顺序（资产文件不改） |
| `RSR_JOB__RECORDER__RECORD_RGB=false` 报 extra_forbidden（`physics_smoke1_envcase.log`） | dynaconf 保留嵌套键的大小写 | 嵌套字段名保持小写：`RSR_JOB__RECORDER__record_rgb=false` |
| 冒烟 1 第 1 集未抓到木块（`physics_smoke1`） | `TrajectoryOptimizer` 直线约束是软约束：下降段偏离直线 20 mm、关节 1 摆约 1 rad，跟踪滞后 0.137 rad，手停在抓取高度上方 3.7 cm | 直线段改为任务空间直线路径 + 时间参数化，并检查偏差（3.2 节）；冒烟 2 两集通过 |
| 冒烟 1 第 0 集只因 `distractors_undisturbed` 失败 | 苹果（圆底）在 `start` 0.25 s 保持期滚动 1.5–1.7 cm，机械臂尚未运动 | 该项从机械臂开始运动时算起，`distractor_settle_m` 单独记录；规划障碍用苹果实测位置。门限 10 mm 未变 |
| 1K PathTracing 视口采集约 38 s / 帧组（`pt_timing_probe_viewport`） | 两路相机共用一个视口，每次切换都重置累积并多渲染 8 帧以上 | 改为固定 RenderProduct 并行采集，约 23 s / 帧组（`pt_timing_probe_parallel_plain`） |
| SG5 Camera prim 的 RenderProduct 一直取不到图像（`pt_timing_probe_parallel_{firstframe,retry,pinhole}`） | Replicator "renderer failed to advance to the scheduled frame"，与 f-theta 投影属性无关（改 pinhole 仍失败） | 改用同位姿同内参的普通 Camera（3.3 节）；`parallel_cameras` 遇到传感器 USD 相机在构造时报错 |

## 12. 局限

- 物体位姿来自 PhysX 真值，没有视觉感知；抓取点是 AABB 中心，朝向固定朝下。
- 只有 2 集，布局只抖动木块位置。
- 苹果和桌面在规划中是盒体近似；背景家具不作为规划障碍（工作空间内没有）。
- 相机按针孔渲染，不是 SG5 实物的鱼眼畸变（Isaac Sim 6.2 RTX 不支持其 f-theta 投影）。
- 历史正式抓放仅使用 GPU 0；现在正式采集器已接入单调预算时钟，回放正式配置使用两张 GPU（见回放文档）。
- 未用官方 LeRobot SDK 加载数据集。

## 13. 交接复核与回归

2026-11-14 接手现有采集链，未重复启动 Kit 或改动采集的物理参数、验收门限。
正式抓放已通过独立校验。原回放进程后续复现 GPU 预算循环故障，保留证据后从头重跑并通过正式验收（见回放文档）。

- 修复 `overwrite_dataset` 仅有注释约束的问题：现在非 `outputs/debug/` 子目录一律拒绝覆盖；
  两种任务的正式结果保护、调试重跑均有无 Kit 测试。
- 回放加载源记录的 `physics_dt` 并与当前运行比较；即使 30 fps 同时整除 60 Hz 和 120 Hz，
  也不能把两种物理步长混用。正式源数据两集均实测为 120 Hz / 30 fps。
- 独立校验额外拒绝非有限的位姿列与形状不匹配，避免 NaN 比较或 NumPy 广播掩盖损坏的记录；
  正式抓放两集已用修复后的校验脚本重新通过。
- 正式采集器新增“独立校验失败不发布”和“发布保留原始执行配置”的无 Kit 回归测试。
- 回归：`pytest -q` **350 passed, 3 skipped**（接手基线 340 passed, 3 skipped），
  `ruff check real2sim2real scripts` 通过；全部 48 份任务配置通过 `check_task_configs.py`；
  `list-backends` 中旧后端与新增桌面后端均可见。
- 两份资产各 564 个文件与 `provenance.json` 的 SHA-256 全部匹配，每份 1,021,352,989 字节。
  未解析依赖仍仅为未选中的 Robotiq 夹爪变体（第 5 节）；源资产、其它任务代码/配置/输出均未修改。
- 文档原记为“15 项抓放验收”，实测代码与落盘结果均为 **14 项**，已更正文档计数，检查与门限未改。
