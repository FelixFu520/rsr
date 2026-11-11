# X30 仓库消防巡检

操作员输入一句话（例如“巡检仓库”），X30 机器狗沿巡航路线巡检仓库北区货架。
着火纸箱进入视野后，机器狗停下并转向火点，静止 4 s，然后打开红色警示灯爆闪报警 3 s。
报警结束后，机器狗走到观察点、对准火点，用云台可见光相机拍照。
整个过程同步录制为 LeRobot v3 数据集。

更新日期：2026-11-11。

## 1. 当前结果

| 项目 | 结果 |
|---|---|
| 正式输出 | [outputs/xunjian](../outputs/xunjian/)，`success=true`，`dataset_use=demonstration` |
| 调试输出 | [outputs/xunjian-debug](../outputs/xunjian-debug/)，同一配置继承而来，轨迹与正式输出逐步一致 |
| 数据集 | 1 个 episode，1070 帧，15 fps，71.3 s；3 路 640×480 H.264 视频 |
| 指令 | `巡检仓库`（作为 LeRobot `task` 原样保存） |
| 巡航 | 规划路线 37.6 m，最小净空 0.65 m（要求 ≥ 0.6 m） |
| 发现火点 | 巡航开始后 42.0 s，位置 (-3.14, 46.09)，相机距火点 6.44 m |
| 停留与报警 | 刹停于 (-3.14, 45.70)，转向火点后静止 4.0 s；警示灯爆闪 3.0 s 共 6 次，跟随视角红色像素峰值 9.5%（静止时 0.007%） |
| 拍照 | 位置 (-5.406, 39.448)，距火点 2.13 m，警示灯已熄灭；火点窗口内火焰色像素 5.7%（1456 个，门限 1%） |
| 物理 | PhysX CPU，120 Hz，共 9036 步；墙钟约 9 min（渲染采集占 65%） |
| 验收 | Job 内 12 项全部通过；`validate_fire_inspection.py` 35 项全部通过 |

直接查看：[巡检照片](../outputs/xunjian/photos/)、
[巡航路线与实际轨迹](../outputs/xunjian/map/patrol_map_result.png)、
[各视角关键帧](../outputs/xunjian/qa/keyframes.jpg)、
[验收报告](../outputs/xunjian/validation.json)。

## 2. 任务流程

```
操作员一句话 ──► 关键词解析（巡检/巡逻/检查/找…，或提到“火”）
                │
init_world      打开火焰仓库 stage，启用 omni.flowusd，配置 PhysX
arrange_scene   build_occ：PhysX 场景查询扫描北区占用图（机器人尚未加载）
                scene：校验 /FireFX 与着火纸箱，读取火点，创建固定相机
                after hook：航点之间 A* 连通，检查净空，输出 map/patrol_map.png
place_robot     引用 X30，TORSO 落到出发位姿，挂载相机，校验腿部运动学
start_simulation
  before hook   重建物理视图（含机器人）→ play → 站立
  simulate      火焰预热 4 s 并预检 → 巡航 → 确认发现火点 → 刹停 → 转向火点
                → 静止 4 s → 警示灯爆闪报警 3 s → A* 走到观察点 → 转向火点
                → 静止 1 s → 拍照验收 → 静止 1 s → 导出 LeRobot v3
finally: close
```

巡航路线：西侧货架过道南端出发向北 → 北端横移到中间过道 → 中间过道向南 → 出口转向西南空地。
着火纸箱在出发点身后，被环形货架挡住。中间过道走到南端出口附近时，火点才首次进入前视相机视野。

## 3. 设计选择

### 3.1 物理引擎：PhysX

选 PhysX（CPU，TGS，120 Hz），不用 Newton。理由：

- 任务只有一台刚体关节机器人和静态仓库，没有布料或柔体，Newton VBD 的优势用不上。
- 仓库 6268 个碰撞体是三角网格（`physics:approximation` 未设置），X30 资产按 PhysX
  关节/驱动 schema 制作。x30 项目的站立和传感器验证在 Isaac Sim 6.2 PhysX 上完成。
- 建图和视线判断直接使用 PhysX 场景查询（`overlap_box`、`raycast_all`），查询对象就是仿真用的碰撞几何。
- 本仓库在整场景级别使用 Newton 时仍有发散问题（见分拣任务记录）。
- Flow 火焰属于渲染与时间轴功能，和物理引擎无关。
- 速度足够：每个物理步约 5.2 ms，整段物理只占墙钟的 9%，瓶颈是 RTX 渲染。

### 3.2 X30 行走：运动学机身 + 脚本化 trot

官方 [deep_robotics_model](https://github.com/DeepRoboticsLab/deep_robotics_model/tree/main/X30)
只提供 X30 的 URDF/MJCF/USD 模型（本机副本 commit `52352f5`），没有行走策略。按此前约定的方案 A 处理：

- 机身：规划器每个物理步给出 `(vx, 0, wz)`。Robot 把当前参考位姿和对应的世界速度写入 PhysX，
  PhysX 在这一步内按该速度积分，得到下一位姿。机器人连杆关闭重力，脚不承重。
- 腿：12 个腿关节使用资产自带的 PD 驱动，跟踪脚本化 trot。对角腿成对交替，周期 0.5 s，抬脚 7 cm。
  支撑脚相对机身以 `-(v + ω×p)` 后移，在地面上基本不滑；摆动脚沿正弦弧线回到前方。
  脚的目标位置由 X30 腿部逆运动学换算为关节目标，同时下发速度前馈。
- 启动时用官方 URDF 腿长做正运动学，与 USD 中站姿下的四只脚位置比对，误差为 0（门限 5 mm）。
- 腿与环境碰撞是真实的。中间过道和北端横移处地面有线缆槽（`SM_WallWire_57/55`），
  踩过时腿被顶起约 0.19 rad，PD 随后恢复。`result.json` 的 `leg_contact_segments` 记录了位置。

数据标注为 `locomotion="kinematic_base+scripted_trot"`。数据只说明腿部动作看起来合理，不能用来评价
X30 的步态或通过能力。以后有 X30 行走策略时，只需替换 Robot 后端，规划器输出的速度指令不变。

### 3.3 路线与发现火点

- 路线：`planner.route_xy` 给出 5 个航点。`build_occ` 先在机器人加载前扫描北区占用图：
  0.05 m 网格，机身高度带 0.10–1.15 m 内与碰撞体重叠即为占据，向下射线无地面即为不可走。
  相邻航点之间用 A* 连通，膨胀半径 0.6 m（X30 足迹 1.03 m × 0.47 m 的外接圆）。
  净空不足 0.9 m 的格子代价为 1.3 倍，路径尽量居中，但不会为此绕远路。
- 发现火点使用仿真真值，不做图像识别。判定条件：火点投影在前视相机视场的 80% 以内，距离不超过 10 m，
  从相机到火点的 PhysX 射线不被遮挡（忽略机器人自身和着火纸箱），并且连续满足 12 个物理步。
- 发现后走到 `planner.standoff_xy`（纸箱正前方约 2.1 m，与原场景 `Fire_Front` 机位一致）。
  如果不配置，会在火点周围 2.1 m 的圆上找有净空、视线通畅、A* 可达的最近点。
- 照片验收：火点投影周围 160×160 px 窗口内，火焰色像素占比 ≥ 1% 且 ≥ 50 个。

### 3.4 报警：红色警示灯爆闪

发现火点后的报警方式选的是警示灯，没有选跳跃或犬吠：

- 真实消防巡检机器人通常用声光报警，警示灯是其中能被相机录下来的部分。
- LeRobot 视频没有音轨，犬吠不会进入数据。机身是运动学移动，跳跃只能做成动画，不是物理跳跃。
- 仓库很暗，红灯爆闪在跟随视角里非常明显，前视画面中周围货架也会被照红。

警示灯装在护笼顶部两根纵向管之间（TORSO 系 (0.14, 0, 0.462) m），由支架板、黑色灯座、
直径 8 cm 的红色灯罩和一个 normalized SphereLight（强度 6000）组成。它只是视觉部件，
没有碰撞体和质量，也不遮挡前视相机。报警期间以 2 Hz、50% 占空比爆闪，共 6 次。

时序：确认发现火点后沿巡航路径刹停 → 原地转向火点 → 静止 `detect_hold_s=4` s →
报警 `alarm_s=3` s（`action.alarm=1`）→ 关灯后再规划接近路径。拍照前灯已经熄灭，
避免红光被照片验收误判为火焰。如果刹停后火点被挡住，规划器会沿路线继续缓行，
直到转身后能看到火点再重新停下（`reacquire` 阶段）。本次刹停点 (-3.14, 45.70) 可以直接看到火点，没有触发这一步。

### 3.5 指令解析：关键词，不接 LLM

当前只有一种任务，关键词足够。包含巡检、巡逻、巡查、巡航、巡视、检查、查看、排查、搜索、寻找、找、
inspect、patrol、search、find、check 之一，或提到火、烟、fire、smoke、flame，就判定为巡检任务。
例如“巡逻仓库，找到着火点并拍照”“去仓库看看哪里着火了”“Find the fire”都能识别。
“你好”“拍张照片”会报错，并给出示例。

没有接入 LLM，配置和代码中也没有保存 ark API key。以后有多种任务时，可以在
`parse_inspection_command` 前加 LLM 意图解析，key 放在环境变量或 `.secrets.toml`。

## 4. 组成与迁移对照

| 模块 | 后端名 | 实现 | 对应原项目 Real2Sim2Real |
|---|---|---|---|
| SimWorld | `isaacsim` | [simworld/isaacsim.py](../real2sim2real/simworld/isaacsim.py)（未改动） | `isaacsim` 世界 |
| Scene | `fire_warehouse` | [scene/scene_fire_inspection.py](../real2sim2real/scene/scene_fire_inspection.py) | `NavigationScene` |
| Robot | `x30_kinematic` | [robot/robot_x30.py](../real2sim2real/robot/robot_x30.py) | `Go2Robot` + RL 步态 |
| Planner | `fire_patrol` | [planner/planner_fire_patrol.py](../real2sim2real/planner/planner_fire_patrol.py) | `NavigationPlanner` |
| Job | `fire_inspection` | [job/job_fire_inspection.py](../real2sim2real/job/job_fire_inspection.py) | `FactorySearchJob` |
| 几何工具 | — | [utils/patrol_geometry.py](../real2sim2real/utils/patrol_geometry.py) | `utils/occupancy.py` |

原项目是 Go2 在仓库中按对话找物体拍照：omap 建占用图、A* 导航、RL 步态、LLM 理解指令、LeRobot v2。
迁移后保留了“一句话指令 → 导航 → 拍照 → 数据集”的结构，并按本项目约定调整：

- 场景换成 x30 项目的 Flow 火焰仓库，机器人换成改装 X30。
- 目标从“找指定物体”变为“巡航中发现火点”。路线由配置航点给出，A* 只负责连通相邻航点。
- 占用图改用 PhysX 场景查询直接扫描，不依赖 `isaacsim.asset.gen.omap`。
- X30 没有步态策略，改为运动学机身 + 脚本化 trot。
- 指令解析去掉 LLM，数据格式由 LeRobot v2 改为 v3（复用 [utils/lerobot_v3.py](../real2sim2real/utils/lerobot_v3.py)）。
- 生命周期沿用 `BaseJob.run()` 固定流程，新后端已在 [pyproject.toml](../pyproject.toml) entry points 注册。

## 5. 资产

所有资产放在 `assets/FireInspection/`（git 忽略，约 355 MB），由脚本从 x30 项目复制，源文件不改动：

```bash
./app/python.sh scripts/prepare_fire_inspection_assets.py
```

```text
assets/FireInspection/
├── warehouse/
│   ├── warehouse_cardbox_fire.usda   # Flow 火焰层；只把 sublayer 改为复制后的仓库
│   ├── 003_StorageHouse/             # Demonstration.usd 及全部依赖（与源目录文件完全一致）
│   └── fire_manifest.json            # 着火目标 /Root/SM_CardBoxA_425、起火点、原场景相机
├── robot/x30/
│   ├── x30_fire_inspection.usdc      # 已展平整机：12 个腿关节、附件碰撞与质量、云台相机
│   ├── asset_manifest.json
│   └── LICENSE_OEM.txt               # 云深处 BSD-3-Clause
└── provenance.json                   # 来源、改动、540 个文件的 SHA-256、依赖检查
```

脚本先用 `UsdUtils.ComputeAllDependencies` 收集 USD 依赖，再递归解析 MDL 中的 `.::OmniUe4Base`
模块导入和 `texture_2d("...")` 贴图。USD 依赖扫描看不到这两类文件，漏掉后材质会编译失败。
已有文件内容不同时，脚本拒绝覆盖。复制后会打开新 stage，确认没有加载资产目录之外的 layer。
`provenance.json` 中的 140 个 unresolved 条目是源仓库残留的 Unreal `/Game/...` 材质名，不是缺失文件。

| 来源 | 内容 |
|---|---|
| `/home/fufa/Projects/x30/output/warehouse_fire/` | 纸箱 `/Root/SM_CardBoxA_425` 上的 Flow 火焰与烟雾，起火点 (-7.54, 39.46, 0.99) m |
| `/home/fufa/Projects/x30/output/x30_fire_inspection.usdc` | X30 改装巡检机型，附件约 8.4 kg（概念估计），TORSO 站立高度 0.5226 m |

## 6. 配置

| 文件 | 用途 | 输出 |
|---|---|---|
| [fire_inspection_x30_physx_warehouse.toml](../configs/fire_inspection_x30_physx_warehouse.toml) | 正式采集；只发布全部验收通过的数据；已有数据集时拒绝覆盖 | `outputs/xunjian/` |
| [fire_inspection_x30_physx_warehouse_debug.toml](../configs/fire_inspection_x30_physx_warehouse_debug.toml) | 继承正式配置；可重复运行并覆盖上次数据集；失败轨迹也导出为 `diagnostic` | `outputs/xunjian-debug/` |

主要字段：

| 字段 | 含义 |
|---|---|
| `sim_world.stage_usd_path` | 火焰仓库 stage；`extensions = ["omni.flowusd"]` 启用 Flow 运行扩展 |
| `scene.flow_max_steps_per_update` | 运行时覆盖 Flow `maxStepsPerSimulate`（资产为 2）。Flow 以 60 Hz 步进，录制间隔 1/15 s 需要 4 步，取 8 |
| `[[scene.cameras]] fire_overview` | 着火点附近的固定视角，也用于预热后的火焰预检 |
| `robot.base_xy` / `base_yaw_deg` | TORSO 出发位姿；必须等于 `planner.route_xy[0]` |
| `robot.gait.*` | trot 周期、抬脚高度、最大步幅、卡腿门限（0.35 rad） |
| `[[robot.cameras]] front_rgb` | 挂在原生云台 `RGB_Camera` 下的零偏移相机，18 mm / 36 mm，水平视场约 90°，用于发现火点和拍照 |
| `[[robot.cameras]] follow` | TORSO 后 1.6 m、高 1.0 m 的跟随视角 |
| `planner.map_*` | 占用图扫描范围、分辨率和机身高度带 |
| `planner.route_xy` | 巡航航点（世界 XY，米） |
| `planner.robot_radius_m` / `clearance_margin_m` | 硬净空 0.6 m，软净空再加 0.3 m |
| `planner.detect_*` | 发现火点的距离、视场比例、连续确认步数 |
| `planner.detect_hold_s` / `alarm_s` / `alarm_flash_hz` | 发现火点后静止时长、报警时长、爆闪频率 |
| `robot.beacon.*` | 警示灯位置、尺寸、颜色、灯光强度 |
| `planner.standoff_xy` | 拍照位置；不填时自动选点 |
| `job.command` / `default_command` | 巡检指令；为空且有终端时交互输入，否则用默认值 |
| `job.fire_warmup_s` | 开始巡航前让 Flow 起火的时间（2–4 s 后可见） |
| `job.photo.*` | 照片分辨率和火焰像素验收门限 |

各后端的字段默认值同时写在 [default_settings.toml](../real2sim2real/core/default_settings.toml) 的注释中。

## 7. 运行

```bash
./app/python.sh -m pip install -e ".[dev]"            # 注册新后端
./app/python.sh scripts/prepare_fire_inspection_assets.py
./scripts/rsr list-backends                           # 应看到 fire_inspection、fire_patrol、x30_kinematic、fire_warehouse
./app/python.sh -m pytest -q tests/test_fire_inspection.py   # 无 Kit 单元测试：解析、腿部运动学、trot、A*、巡检与报警状态机

# 1. 调试（约 8 min）
./scripts/rsr collect run -c configs/fire_inspection_x30_physx_warehouse_debug.toml
./app/python.sh scripts/validate_fire_inspection.py outputs/xunjian-debug

# 2. 正式采集（约 9 min；outputs/xunjian 中已有 lerobot_v3 时拒绝覆盖）
./scripts/rsr collect run -c configs/fire_inspection_x30_physx_warehouse.toml
./app/python.sh scripts/validate_fire_inspection.py outputs/xunjian
```

终端提示 `请输入巡检指令 [巡检仓库]:` 时可以直接回车。
非交互运行时用环境变量指定：`RSR_JOB__COMMAND="巡逻仓库，找到着火点拍照"`。
打开 GUI 观察：`RSR_SIM_WORLD__HEADLESS=false RSR_JOB__KEEP_OPEN=true`。

## 8. 每步时序与同步

规划器是带加速度限制的单轮车模型（0.8 m/s²、2.0 rad/s²，巡航 0.8 m/s，接近 0.5 m/s）。
航向误差超过 0.6 rad 时原地转向；经过路径拐点前按 v² = v_corner² + 2ad 减速。

| 阶段 | 行为 | 结束条件 |
|---|---|---|
| `patrol` | 沿 A* 连通后的巡航路径行进 | 确认看到火 → `stop`；走完仍没看到 → `not_found` |
| `stop` | 沿路径减速 | 停稳 |
| `turn_to_fire` | 原地转向火点 | 对准后火点可见 → `hold`；被挡 → `reacquire` |
| `reacquire` | 沿路线缓行，直到转身后能看到火点 | 回到 `stop` |
| `hold` | 静止，前视相机对准火点 | 4 s |
| `alarm` | 静止，`alarm=1`，警示灯 2 Hz 爆闪 | 3 s 后关灯 |
| `approach` | 选拍照点并 A* 规划，沿路径前往 | 到达并停稳 |
| `face` | 原地转向火点 | 航向误差 < 0.03 rad |
| `settle` | 静止 1 s | 发出拍照请求 |
| `photo` | Job 拍照并验收，再静止 1 s | `done` |

每个物理步（120 Hz）的顺序：

1. 录制步（每 8 步一次，即 15 fps）：三路相机依次渲染当前物理时刻，读取机身与关节状态，作为观测。
2. 规划器根据这一观测给出本步指令和警示灯状态；如果有拍照请求，此时拍 1280×960 照片。
3. 录制步把（观测，本步指令）写成一帧。拍照后的第一帧 `photo_trigger=1`。
4. PhysX 推进一步。
5. 把时间轴（Flow 的时钟）设为新的物理时间。

多相机采集只刷新渲染，不推进物理和 Flow。Job 会核对采集前后的物理步数和 Flow 时间，二者必须不变。
因此同一帧的三路图像、状态和火焰形态都对应同一个仿真时刻；`fire_time_s` 与 `simulation_time_s` 相等（65.6083 s）。

## 9. 输出

```text
outputs/xunjian[-debug]/
├── lerobot_v3/                 # 全部验收通过后原子发布
│   ├── meta/info.json, stats.json, tasks.parquet, episodes/..., task_outcome.json
│   ├── data/chunk-000/file-000.parquet
│   └── videos/observation.images.{fire_overview,front_rgb,follow}/chunk-000/file-000.mp4
├── photos/fire_<UTC>.png/.json # 1280×960 巡检照片及其验收数据（位置、距离、火焰像素）
├── map/
│   ├── occupancy.npz           # PhysX 扫描的占用图
│   ├── patrol_map.png/.json    # 占用图 + 巡航路径 + 航点 + 火点
│   └── patrol_map_result.png   # 再叠加实际轨迹、接近路径和发现位置
├── diagnostics/<run>/          # configuration.json、result.json、trajectory.jsonl（每物理步一行）、预检图
├── logs/collect.log
├── result.json
├── validation.json             # validate_fire_inspection.py 生成
└── qa/keyframes.jpg            # 各视角首帧 / 拍照帧 / 末帧
```

LeRobot v3 数据：

- `task`：操作员输入的原句。`robot_type="deeprobotics_x30_fire_inspection"`。
- `observation.state`（16 维）：`base_x, base_y, base_z`（m）、`base_yaw`（rad），
  以及 12 个腿关节实际角度（rad，按 PhysX 顺序：4 个 HipX、4 个 HipY、4 个 Knee）。
- `action`（5 维）：`cmd_vx, cmd_vy`（机体系 m/s，`cmd_vy` 恒为 0）、`cmd_wz`（rad/s）、`photo_trigger`、
  `alarm`（报警期间为 1）。警示灯在第 k 帧的指令，会出现在第 k+1 帧的图像中。
- 视频：`fire_overview`（固定）、`front_rgb`（云台可见光）、`follow`（跟随），各 640×480、15 fps、H.264。
- `meta/task_outcome.json`：完整配置和 `result.json` 内容，包括发现位置、照片、验收结果。

## 10. 验收

Job 中以下 12 项全部通过，数据才标记为 `demonstration` 并发布到正式目录：

| 检查 | 含义 | 本次 |
|---|---|---|
| `episode_completed` | 在 `max_steps` 内走完全部阶段 | 8556 步 |
| `route_clear` | 巡航路径每个采样点净空 ≥ 0.6 m | 最小 0.65 m |
| `fire_found` | 巡航中按真值规则确认看到火 | 6.44 m 处 |
| `held_before_alarm` | 对准火点后静止至少 `detect_hold_s` | 4.0 s |
| `alarm_raised` | 报警阶段走完，警示灯开关次数 ≥ 2 × 爆闪次数 | 12 次 |
| `beacon_off_for_photo` | 拍照时警示灯已熄灭 | 熄灭 |
| `approach_clear` | 接近路径净空 ≥ 0.6 m | 最小 0.66 m |
| `reached_photo_phase` | 到达观察点、对准火点并完成拍照 | — |
| `photo_taken` | 照片文件已写出 | — |
| `photo_shows_fire` | 火点窗口内火焰色像素 ≥ 1% 且 ≥ 50 个 | 5.7%，1456 个 |
| `base_follows_command` | PhysX 机身位姿与指令积分的单步偏差 < 1 mm | 最大 0.53 mm |
| `legs_track_gait` | 腿关节与 trot 目标的最大偏差 < 0.35 rad（不卡腿） | 最大 0.19 rad（踩线缆槽） |

另有两道前置检查：航点之间 A* 规划失败时，在 `arrange_scene` 阶段直接报错并输出地图；
火焰预热后用 `fire_overview` 检查火焰像素，Flow 没有生效时直接失败。

[validate_fire_inspection.py](../scripts/validate_fire_inspection.py) 不启动 Kit，独立复核 35 项：
v3 元数据与帧数、时间戳等间隔、数值有限、单 episode、`task` 等于指令原句、只有一帧拍照触发、
机身速度不超过上限、按录制指令积分能复现录制的位移（误差 2 mm）和航向（0.004 rad）、
拍照帧机器人静止且对准火点（0.024 rad）、各路视频能完整解码且分辨率正确、地图文件存在，
并从照片重新计算火焰像素。

报警部分单独核对：`alarm` 连续 45 帧（3.0 s）、报警期间机身静止、之前静止 4.0 s、报警时对准火点、
报警在拍照之前；并从跟随视角视频解码报警前后各帧，按红色像素比例数出爆闪 6 次，与 2 Hz × 3 s 一致。

## 11. 调试中发现并修复的问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 仓库材质大量 MDL 编译错误 | `ComputeAllDependencies` 不解析 MDL 内部的模块导入和贴图 | 资产脚本递归解析 MDL 依赖；复制后与源目录文件完全一致 |
| 资产脚本出错却返回 0 | `SimulationApp.close()` 快速退出时吞掉异常 | 先记录异常，再以 `exit_code=1` 关闭 |
| 原路线 x=-5 穿过墙和立柱 | 旧路线按相机机位推测，未按实测布局 | PhysX 扫描真实碰撞几何建图，航点间 A* 连通 |
| 世界步进时火焰不出现 | `IsaacsimWorld.update()` 关闭 `playSimulations` 时，时间轴自动推进也不驱动 Flow | 关闭时间轴自动推进，每个物理步后手动把时间轴设为物理时间 |
| 15 s 后火焰重置 | stage `endTimeCode=900`（15 s），时间轴循环 | 时间轴结束时间设为 1e7 s，关闭循环 |
| 机身实际速度是指令的 3 倍，拍照点冲过 0.32 m | 两处各多走一次：规划器写入的是已积分的下一位姿，PhysX 又按速度积分一次；提交时间轴时间变化会让 PhysX stage-update 节点多走一步 | 写入当前参考位姿 + 速度，由 PhysX 积分；提交时间轴时临时禁用 `Physics` 节点；增加 `base_follows_command` 检查 |
| 加载机器人后关节状态读取失败 | 建图时的无机器人物理视图被缓存，新的 tensor view 绑定到它 | 建图后调用 `invalidate_physics()` |
| X30 落到地面以下 | 资产根 prim 带有 0.5226 m 站立高度，TORSO 本身无偏移 | 按 TORSO 相对根的变换反算根位姿 |
| 腿部正运动学校验失败 | `reset()` 后 PhysX 把直腿姿态写回 USD | 在建立物理视图之前校验运动学 |

## 12. 局限

- 机身是运动学移动，trot 只是脚本化的腿部动作，脚不承重。数据不能用来评价 X30 的行走能力或越障能力。
- 发现火点用的是仿真真值（视场 + PhysX 射线），没有用视觉检测。照片验收只是颜色阈值，不是火焰分类器。
- 占用图只看 0.10–1.15 m 高度带。地面线缆槽等更低的物体视为可通行，脚会与它们真实接触。
- 当前只采集一条固定路线的 episode；更换出发点或路线后，需要重新查看 `map/patrol_map.png`。
- Flow 只是视觉效果，没有热辐射或温度数据。热像相机是光学占位，不参与采集。
- 仓库整体偏暗（原场景灯光），前视画面中远处火点像素较少，因此用固定相机做预热检查。
- 报警只有警示灯，没有声音，也没有向外部系统发送消息。报警信息记录在 `result.json` 的 `alarm` 字段。
