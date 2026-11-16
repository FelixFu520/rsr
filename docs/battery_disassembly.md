# 电池拆解：双 KUKA KR210 在 c1003b1a1 工厂里拧出四颗螺丝、揭下电池盖

更新日期：2026-11-16（环境核对；运行时间记 UTC）。

## 1. 任务介绍

从 Real2Sim2Real（`/home/fufa/Projects/Real2Sim2Real`，`configs/arm_kr210_kr210_unscrew/`，
`Kr210BatteryJob`）迁入的双臂拆解任务。robot01 用电动螺丝刀拧出电池盖上的四颗 M10 螺丝，逐颗放进篮子；
robot02 再用平行夹爪夹起 12 kg 电池盖，放到前方托盘上。三路 RGB（俯视、螺丝刀腕部、夹爪）与状态、动作一起
导出为 LeRobot v3。

源项目是运动学演示：双臂每步直接写关节位置并关掉碰撞，螺丝刀和夹爪是每步摆位的可视模型，螺丝和盖板由脚本
贴在工具上，只有松开之后才交给 PhysX。用户选择“全物理”口径（2026-11-14），本任务改为全部由 PhysX 驱动：

| 环节 | 源项目 | 本任务 |
|---|---|---|
| 机械臂 | 写关节位置，碰撞与重力关闭 | 资产自带关节驱动跟踪 IK 轨迹（位置 + 速度前馈），碰撞开启 |
| 螺丝刀 | 每步摆位的可视模型，`Spin` 只转画面 | `tool0` 下的浮动弹簧（棱柱关节）+ 主轴（转动关节，速度驱动、2 N·m 限矩），十字批头碰撞体 |
| 拧松 | 螺丝副本吸附到批头上 | 批头叶片与螺丝十字槽接触传递扭矩；螺纹是 mimic 螺旋约束，转满 16 圈（24 mm）到止挡后由求解器断开保持关节 |
| 运螺丝 | 吸附运输，松开时注入初速度 | 电磁批头吸力（外力，随距离线性衰减，3 N），到篮子上方断电自由落下 |
| 揭盖 | 盖板变 kinematic 跟随夹爪，松开时新建刚体 | 12 kg 动态刚体，两指限力 200 N、橡胶垫摩擦夹持；松开后自由落到托盘 |

### 1.1 流程

```
init_world      新建 stage，引用 C1003B1A1 工厂背景；PhysX CPU TGS 240 Hz
arrange_scene   Battery.usd 引用到 /World/scene（场景位姿取源配置），电池 rotateZ = -90°
                关闭电池包 / 盖板 / 篮子的资产碰撞体，换成盒体代理（盖板 18 块含四个螺丝孔、电池壳 1 块、篮子 5 块）
                四颗螺丝改为独立刚体（头、十字槽、螺杆碰撞体，30 g）+ 螺纹 articulation + 保持关节
                托盘（静态盒体）、工位灯光、俯视相机
place_robot     两台 KR210 引用、基座固定到配置位姿；螺丝刀与夹爪关节链挂到 tool0；两路腕部相机
                after hook：numpy IK 解出两臂全部名义轨迹（不可达直接报错，不进仿真）
start_simulation
  before hook   reset → play → 张量视图 → 两臂回零 → 订阅接触报告
  simulate      每个物理步：读关节传感器 → 规划器下发驱动目标与磁力 → 螺纹走完则断开保持关节
                → 正式每 8 步（30 fps）、调试每 16 步（15 fps）三路相机各采一张 → PhysX 一步
                → 结束后在完整状态记录上验收 → 导出 LeRobot v3 + meta/physics
finally: close
```

螺丝刀臂依次处理四颗螺丝：移到螺丝上方 → 快速下降 → 最后 4 cm 慢速接近（弹簧压在螺丝头上）→
寻位（主轴 60°/s 慢转，浮动行程显示叶片落入十字槽即判定插入）→ 拧松（3 r/s，臂按浮动压缩量向上跟随）→
抬升 → 移到篮子上方 → 下探到篮口下 10 cm → 磁力断电 → 抬起。四颗完成后回零，夹爪臂开始：
移到盖板上方 → 下降到夹持高度 → 两指限力闭合 → 提升 0.2 m → 平移到托盘上方 → 下降到盖底离托盘 5 mm →
张开 → 撤离 → 回零。

### 1.2 约束与验收口径

- 正式数据只来自物理：初始状态之后不写任何刚体位姿，不加吸附约束，运行中不改 USD。螺纹脱开是
  PhysX 可断裂关节在螺纹止挡处被批头扭矩拉断，电磁批头是外力（反作用力加在主轴上），两者都写进局限（第 12 节）。
- 机器人用 Isaac Sim KR210 L150 资产与其关节驱动参数（刚度 1e15/°），末端工具是本任务新加的关节链。
- 碰撞代理：电池包和篮子的资产网格是三角网格 / 80 万顶点凸分解，CPU 上接触不稳且慢，换成按资产尺寸测量的
  盒体（数值在 `BatteryWorkcellSceneConfig` 和 `docs` 第 5 节），渲染仍用资产网格。
- 验收项和门限是本任务新定的（源项目只检查螺丝在篮子 AABB ±3 cm、盖底 ±15 mm），见第 10 节。

## 2. 当前结果

### 当前：1K 双 GPU 采集，单调时钟补丁长测中

当前采集三路 **1024×768、30 fps**，保留 PathTracing 64 spp、6 次反弹、OptiX 降噪和
PhysX CPU TGS 240 Hz。单个 Kit 进程使用双 GPU；三路固定 RenderProduct 并行采集。
多卡利用率骤降已定位到墙钟回跳引发的显存预算计时异常，进程内单调时钟补丁正在连续验证。
原因、证据、使用方法和未通过的方案集中见[第 11.1 节](#render-budget-clock)。

本轮运行目录为 `outputs/debug/battery_disassembly/clock_1k_20261114T1233Z/`；实时进度见
[status.json](../outputs/debug/battery_disassembly/clock_1k_20261114T1233Z/status.json)，
持续验证见 [verification.json](../outputs/debug/battery_disassembly/clock_1k_20261114T1233Z/verification.json)。
已越过先前第 36 / 50 / 60 / 124 组的退化位置，并在实际墙钟回跳时继续渲染。
这属于在途验证；完整 1K 视频、整段物理验收和最终视频关键帧复核仍待完成。

通过物理及独立视频校验后，采集器才发布至
`outputs/battery_disassembly/kr210_physx_c1003b1a1_1k/`，提供
`overview.mp4`、`screwdriver.mp4`、`gripper.mp4`。最终关键帧视觉复核状态记录在
`capture_manifest.json`，未查看前明确标为 pending；已有 640 正式结果保留。

### 已停止：2K 单卡 PathTracing 采集

2026-11-14 09:29 UTC 按用户“PathTracing 优先最高画质、接受长时间渲染”的选择启动新采集。
三路 2048×1536、30 fps、64 spp、6 次反弹、OptiX 降噪；原估计 40–60 h，超时 72 h。
第 49 组开始发生渲染调度等待，每组三路从约 43 s 增至约 446 s，原估计失效。
最后完成 50 组（约 1.64 s 仿真时间），用户改为 1K 多卡后停止；原帧、日志与线程栈保留，不作为正式结果。
运行记录以
[status.json](../outputs/debug/battery_disassembly/pathtracing_2k_20261114T0929Z/status.json) 为准，
[capture.log](../outputs/debug/battery_disassembly/pathtracing_2k_20261114T0929Z/capture.log) 留存完整日志。
采集脚本自动重算物理验收、核对帧数/分辨率/fps、完整解码并生成关键帧拼图；
未通过和被取消的结果保留在 debug，已有 640 正式结果不覆盖。直接视频入口指向 LeRobot 原视频，无二次压缩。

### 历史 640 采集

历史 640 采集 2026-11-14 08:29–08:52 UTC，单个 Kit 进程。物理与 08:05 UTC 的调试运行 `rgb4` 逐位相同
（状态记录逐项最大差 0），PhysX CPU 在本机可复现。

| 项目 | 结果 |
|---|---|
| 输出 | [outputs/battery_disassembly/kr210_physx_c1003b1a1](../outputs/battery_disassembly/kr210_physx_c1003b1a1/)（`dataset_use = demonstration`） |
| 数据集 | LeRobot v3，1 episode，2411 帧，15 fps，160.7 s；三路相机 overview / screwdriver / gripper，640×480，H.264；90 MB |
| 渲染 | `RayTracedLighting`，`rt_subframes = 4`，每路相机录制前 3 张不记录的预热图 |
| 物理 | PhysX CPU TGS 240 Hz，38571 步；墙钟 1368 s（物理 288 s、控制 94 s、采集 954 s、导出 15 s） |
| Job 验收 | 16 / 16 通过 |
| 独立校验 | `validate_battery_disassembly.py` 通过：视频解码 / 帧数 / fps / 分辨率、数据行与物理记录一致（状态最大差 1.5e-5，float32 存储）、重算验收一致、无黑帧 |

| 指标 | 实测 | 门限 |
|---|---|---|
| 每颗螺丝转出（螺纹转角，结束时） | 15.62 / 15.69 / 15.77 / 15.85 圈（止挡 16 圈；脱开后空 carrier 在重力下回退约 0.3 圈） | ≥ 0.95 × 16 圈 |
| 导程（mimic 约束决定，作一致性检查） | 1.4997 mm（四颗相同） | 1.5 mm ± 2 % |
| 批头打滑（主轴转角与螺丝转角之差） | 2.4–2.6 % | ≤ 10 % |
| 螺丝最终位置 | 4 颗都在篮子内腔（1 颗立着，3 颗躺倒），最后 1.5 s 位移 0 | 在内腔；≤ 2 mm |
| 盖板底面 - 托盘顶面 | 0.06 µm（落下 5 mm 后静止） | ≤ 10 mm |
| 盖板倾斜 | 0.0° | ≤ 3° |
| 盖板在夹爪中的滑移（提升到松开） | 1.35 mm | ≤ 10 mm |
| 机械臂连杆接触冲量 | 0 | ≤ 1e-3 N·s |
| 工具接触非目标物体 | 0 | ≤ 0.05 N·s |
| 螺丝接触盖板 / 批头 / 篮子以外的物体 | 0 | 0 |

调试与对照（`outputs/debug/battery_disassembly/`）：`probe/`（螺纹机构探针）、`physics1`–`physics6`（只跑物理的迭代，
问题见第 11 节）、`rgb1`、`rgb2`（运行中改 USD 关节导致 Kit 变慢后终止，各有 `run_note.json`）、`rgb3`（第 0 帧偏暗）、
`rgb4`（与正式运行逐位相同的调试运行）。

## 3. 设计选择

### 3.1 物理引擎与求解器

PhysX CPU TGS，240 Hz。理由：

- 只有刚体和关节；KR210 资产是 PhysX schema（`PhysicsArticulationRootAPI`、角驱动），源项目也用 PhysX；
- 需要 PhysX mimic 关节做螺纹螺旋约束，需要接触报告做碰撞验收；
- 探针（`outputs/debug/battery_disassembly/probe/`）：240 Hz 下 1.8 mm 叶片 / 2.5 mm 槽的接触稳定传递扭矩，
  15.6 圈拧出 23.4 mm，导程 1.4997 mm（名义 1.5 mm）。

被否决的方案：

| 方案 | 结果 |
|---|---|
| PhysX `PhysxPhysicsRackAndPinionJoint`（最大坐标）做螺纹 | 位置约束用的是包裹到 ±180° 的转角，每转一圈平移复位，螺丝只在 0–0.9 mm 间来回（探针 `screw_probe_screw_slider_1_240.json`） |
| mimic gearing 按 m/rad 写 | PhysX mimic 用 USD 单位（m/°），按 m/rad 写大了 57 倍，螺丝被锁死、主轴抖动 |
| 螺纹网格 + SDF / hydroelastic 接触（物理求解器任务的 M20 方案） | 需要 Newton；KR210 + 工位 + 四颗螺丝在 Newton 中没有现成路径，CPU PhysX 不支持该接触模型 |
| 保留资产三角网格 / 凸分解碰撞 | 篮子凸分解 80 万顶点，盖板凸分解贴在三角网格电池壳上；改用盒体代理 |

### 3.2 螺纹与拧松

每颗螺丝一个小 articulation：`anchor`（固定到世界）→ `slide`（棱柱，沿螺丝轴）→ `slider` → `turn`（转动）→
`carrier`。`slide` 上的 `PhysxMimicJointAPI` 引用 `turn`，gearing = −pitch/360 m/°，即 slide = 1.5 mm × 圈数
（右旋螺纹，从上往下看逆时针为松）。`turn` 上的角阻尼 0.002 N·m·s/rad 代表螺纹摩擦，`turn` 的上限
360° × 24/1.5 = 5760° 是螺纹末端的硬止挡。螺丝本体是独立刚体，用一个最大坐标固定关节 `hold` 连在 `carrier`
上，`hold` 的断裂扭矩 0.5 N·m、断裂力 500 N：正常拧松只传约 0.04 N·m，转到止挡后批头继续施加最高 2 N·m，
`hold` 由 PhysX 求解器断开，螺丝从螺纹中脱出。扭矩只能经批头叶片与十字槽四个象限块的接触传给螺丝，
验收用螺纹转角与主轴编码器转角比较打滑。Scene 只读状态判断“已脱出”（行程到末端且螺丝与 carrier 分离 > 0.5 mm）。

### 3.3 螺丝刀与传感器

`tool0 -(rsr_float_joint，棱柱，1500 N/m，阻尼 30，目标伸出 8 mm)- float -(rsr_spindle_joint，转动，
速度驱动，阻尼 0.5 N·m·s/rad，限矩 2 N·m，转子惯量 2e-4 kg·m²)- spindle`。插入时弹簧预压 12 N。
规划器只用机器人侧传感器：浮动行程（判定插入、拧松时向上跟随）、主轴编码器（拧出所需圈数 = 24/1.5 + 0.5），
最大上升量 ≥ 23.5 mm 时结束拧松。螺丝脱出后浮动弹簧目标改为 0，避免把自由螺丝压回孔里。

### 3.4 夹爪与盖板

两指为 `tool0` 下的棱柱关节（各 1.5 kg，行程 60 mm，刚度 2e4 N/m、阻尼 400、限力 200 N），胶垫 μ 1.0/0.9，
盖板代理 μ 0.6/0.5（PhysX 平均组合 0.8/0.7）。夹持点取盖板代理盒体的体积质心（x 偏 +1.8 cm，凸起段一侧），
夹持高度让胶垫底边在法兰上方约 2.3 cm、横梁在盖顶上方，胶垫与盖板侧壁接触高度约 47 mm。
两指闭合超过 58 mm 判定没夹到盖板（规划器直接失败）。

### 3.5 规划

`kr210.urdf`（Isaac Sim 6.2 弃用扩展 motion_generation 中的 xacro 生成文件，关节原点与 USD 一致，
单元测试核对 q=0 时 tool0 与 joint_a2 位置）上的 numpy 阻尼最小二乘 IK。螺丝刀是 5 自由度任务（工具轴竖直向下，
绕轴转角不约束，寻位阶段找槽）；夹爪是 6 自由度，在偏航 0°/180° 两种等价朝向中取关节路程短的。所有直线段按 5 mm
（拧松段 1 mm）采样逐点求解，相邻样本关节变化超过 0.05 rad 就报错；轨迹按弧长平滑插值（smoothstep），
同时下发位置和速度目标。

### 3.6 相机

三路相机保持 4:3 宽高比，D455 彩色相机内参（f 1.93 mm / 孔径 3.896 mm，水平视场 90.6°）。
俯视相机沿用源配置位姿；源配置的夹爪相机在 tool0 后上方 0.5 m，被机器人腕部遮挡，
改到 tool0 下 0.24 m（夹爪横梁下沿以下）、沿夹爪 x 外移 0.6 m，从盖板一端平视两指夹持；螺丝刀相机改到
批头侧面 0.26 m。录制前每路相机先渲染 3 张不记录的图（RTX 第一帧明显偏暗，调试运行 rgb3 的第 0 帧平均亮度 7/255）。

#### 调试与正式渲染规范

| 用途 | 分辨率 / 输出帧率 | 渲染设置 | 配置 |
|---|---|---|---|
| 调试 | 640×480 / 15 fps | `MinimalRendering`、纹理漫反射模式 2、环境补光 0.5、无间接光照 | `kr210_physx_c1003b1a1_debug.toml` |
| 正式 2K | 2048×1536 / 30 fps | `PathTracing`、64 spp、6 次反弹、OptiX 降噪、同步采集 | `kr210_physx_c1003b1a1.toml` |
| 正式 1K（当前默认） | 1024×768 / 30 fps | 与 2K 相同画质参数，单进程双 GPU | `kr210_physx_c1003b1a1_1k.toml` |

调试优先非光线追踪渲染，但当前安装的 Isaac Sim 6.2 受支持渲染入口均为 RTX；
`MinimalRendering` 是简化预览，**仍属于 RTX，不能当作纯非光追**。本机实现
`app/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py` 与
[6.2 官方渲染模式说明](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/reference_material/rendering_modes.html)
均确认支持 Minimal 模式。只调物理可设 `record_rgb=false`，完全跳过 RGB 采集。

正式采集保留材质、间接光照、反射和阴影；关闭跨相机光照缓存、时域降噪及 DLSS，避免切换相机时残留旧画面。
同步采集屏障等待实际渲染完成，每次拍摄前后核对物理步数不变。30 fps 是仿真时间采样率，离线渲染可以慢于实时。
调试与正式配置的物理参数、轨迹与验收门限相同。历史 640 数据保留，新结果分别写入 `_2k` / `_1k` 目录。

本机 2K 短测：64 spp PathTracing 每路预热采集约 11–15 s，完整三路视频估算超过 40 h；
实时光追 5 帧的采集耗时 14.79 s，完整 4822 帧约 4 h。用户明确选择 PathTracing，接受长时间渲染；
因此停止实时光追完整试跑，保留 64 spp PathTracing 正式设置，超时设为 72 h。
短测留在 `outputs/debug/battery_disassembly/render_profiles/`，两次有限步长试跑的物理验收因未完成任务而失败，
不能作为演示数据；被停止的完整实时光追采集留在 `outputs/debug/battery_disassembly/capture_2k_rtx/`。

## 4. 组成

| 能力域 | 后端 | 实现 | 迁移对照（源项目） |
|---|---|---|---|
| SimWorld | `isaacsim` | [simworld/isaacsim.py](../real2sim2real/simworld/isaacsim.py)（新增可选 Minimal 预览，保留原默认行为） | `IsaacsimSimWorld` |
| Scene | `battery_workcell` | [scene_battery_disassembly.py](../real2sim2real/scene/scene_battery_disassembly.py) | `BatteryKr210Scene` |
| Robot | `kr210_dual` | [robot_battery_disassembly.py](../real2sim2real/robot/robot_battery_disassembly.py) | `Kr210Robot` |
| Planner | `battery_unscrew` | [planner_battery_disassembly.py](../real2sim2real/planner/planner_battery_disassembly.py) | `Kr210Planner` + `kr210_ik.py` |
| Job | `battery_disassembly` | [job_battery_disassembly.py](../real2sim2real/job/job_battery_disassembly.py) | `Kr210BatteryJob` |
| 辅助 | — | [battery_disassembly_utils.py](../real2sim2real/utils/battery_disassembly_utils.py) | `kr210_usd.py` |

迁移后的调整：

- 源项目依赖 Lula 的 `Kuka_KR210` 配置；Isaac Sim 6.2 不再提供该配置，改用 URDF 上的 numpy IK。
- 源配置引用的 `assets/misc/...` 在源仓库已移到 `assets/tmp/misc/...`；源配置的输出目录在源仓库里不存在，
  没有可对照的源项目成功输出。
- 源项目录制 LeRobot v2，本任务统一为 LeRobot v3。
- 源项目有 30 个分场景配置（c1003b1a1–c1003b1a30）和整包工位 `screwsim.toml`；按用户选择先迁 c1003b1a1，
  资产准备脚本可按场景 ID 追加背景（`--scenes`）。

### 4.1 脚本

| 脚本 | 用途 |
|---|---|
| [prepare_battery_disassembly_assets.py](../scripts/battery_disassembly/prepare_battery_disassembly_assets.py) | 复制资产及其依赖闭包，写 `provenance.json` |
| [collect_battery_disassembly.py](../scripts/battery_disassembly/collect_battery_disassembly.py) | 独立 Kit 子进程采集、保存进度、独立校验、通过后发布；不覆盖旧结果 |
| [validate_battery_disassembly.py](../scripts/battery_disassembly/validate_battery_disassembly.py) | 独立校验数据集、数据行与物理记录一致、重算验收、关键帧 |
| [gpu_budget_clock.c](../scripts/battery_disassembly/gpu_budget_clock.c) | GPU 库专用单调时钟补丁，由采集器编译并仅加载到 Kit 子进程 |

## 5. 资产

`assets/BatteryDisassembly/`（1.9 GB，183 个文件，逐文件 SHA-256 见 `provenance.json`）：

```text
assets/BatteryDisassembly/
├── Background/C1003B1A1/      # C1003B1A1.usd + Assets/（173 MB）；preview/、README.md
├── Scene/Battery/             # Battery.usd、Pure_Battery/Pure_Battery.usd、textures/（1.7 GB）
├── Robot/KR210_L150/          # kr210_l150.usd、configuration/、kr210.urdf、kr210_urdf-LICENSE.md
├── Tools/                     # crosshead_screwdriver.usdz、kr210_gripper.usdz（仅可视）
└── provenance.json
```

```bash
./app/python.sh scripts/battery_disassembly/prepare_battery_disassembly_assets.py            # c1003b1a1
./app/python.sh scripts/battery_disassembly/prepare_battery_disassembly_assets.py --scenes C1003B1A2  # 追加场景
```

| 内容 | 来源 | 许可证 |
|---|---|---|
| 工厂背景 | Real2Sim2Real `assets/Background/SimScenes/C1003B1A1` | 源项目未记录 |
| 工位 / 电池 / 篮子 | Real2Sim2Real `assets/Scene/Battery/Battery.usd` | 源项目未记录 |
| KR210 USD | Real2Sim2Real `assets/Robot/Kuka/KR210_L150`（与 Isaac 6.2 自带版本 SHA 不同，用源项目版本） | NVIDIA Isaac Sim 资产条款 |
| KR210 URDF | Isaac Sim 6.2 `extsDeprecated/isaacsim.robot_motion.motion_generation` | 扩展许可证（随附）；几何来自 ROS-Industrial kr210l150 xacro |
| 螺丝刀 / 夹爪 | Real2Sim2Real `assets/tmp/misc/kr210_tools` | 源项目未记录 |

源文件不改。背景依赖扫描有 Unreal 导出遗留的 `/Game/...`、`/Engine/...` 字符串（不是文件），记在
`provenance.json` 的 `unresolved_non_file_paths`。KR210 的 `OmniPBR.mdl` 由 Kit 搜索路径解析。

碰撞代理（资产坐标，电池旋转前；运行时写入 stage，不改文件）：

| 代理 | 尺寸 |
|---|---|
| 盖板法兰 | z 0.8147–0.8190，x ±0.4221，y −0.6519–0.6419，四个 13 mm 方孔 |
| 盖板壳体 | 侧壁 x ±0.4015、y −0.6324–0.6224、壁厚 4 mm；主顶面 0.899，凸起段 y 0.198–0.6224 顶面 0.960 |
| 电池壳 | x ±0.425，y ±0.629，z 0.7504–0.8147（与螺丝过滤碰撞） |
| 篮子 | 资产包围盒内 8 mm 壁、15 mm 底 |
| 螺丝 | 头 r 10.1 mm（底 1.5 mm + 3 mm 深十字槽，槽宽 2.5 mm），螺杆 r 5.1 mm × 32 mm |

## 6. 运行环境、依赖与安装

在仓库根使用 Isaac SDK 的 `./app/python.sh`；本机为 Linux x86_64、Isaac Sim
`6.2.0-rc.14+release.52805.02189a84.gl`、Python 3.12.14，`app -> /home/fufa/isaacsim`。
不使用旧 Real2Sim2Real 项目环境，不建立 `.venv`。公共依赖为 NumPy、Pillow、PyArrow、Dynaconf、Pydantic、
PyYAML、structlog、Typer，开发依赖 pytest / Ruff；完整版本、用途和路径见
[公共依赖表](../README.md#公共依赖与安装位置)。本任务没有额外任务专用 pip 包。

| 组件 | 用途与本机情况 | 安装 / 输入位置 |
|---|---|---|
| PhysX / USD / Kit | CPU TGS 240 Hz、有限力驱动、接触和机器人状态 | `/home/fufa/isaacsim/` 的物理、USD、Kit 扩展 |
| NumPy | 本任务 URDF 雅可比 IK、轨迹与几何计算，2.3.1 | Isaac `extscache/omni.kit.pip_archive-0.0.0+187cf21b.lx64.cp312/pip_prebundle/` |
| RTX | PathTracing、双相机和材质；物理用 CPU 也仍需渲染 GPU | Isaac SDK 渲染插件 |
| ffmpeg / ffprobe | H.264 编码、预览和独立校验，6.1.1 | `/usr/bin/ffmpeg`、`/usr/bin/ffprobe` |
| `cc` 与系统 C/C++ 运行库 | 正式 1K 配置的单调 GPU 预算时钟补丁，GCC 13.3.0 | `/usr/bin/cc`；Ubuntu/Debian 的 `build-essential` |
| KR210 URDF 与资产 | 初始化模型和 IK 依据，不是额外 Python 库 | 运行时在 `assets/BatteryDisassembly/Robot/KR210_L150/`；准备时从 SDK 的 `extsDeprecated/isaacsim.robot_motion.motion_generation/` 复制 |

本任务的 KR210 IK 已改为 NumPy 实现，不要求 Lula 的 `Kuka_KR210` 配置，也不调用 cuRobo / cuMotion、
Newton、MuJoCo、Kimodo 或官方 LeRobot 包。旧源项目中的依赖不等于 rsr 当前依赖。
工厂、工位、机器人、工具、材质贴图都在 `assets/BatteryDisassembly/`；只在重新准备资产时需要第 5 节的源项目输入。

当前正式入口默认选 `kr210_physx_c1003b1a1_1k.toml`：CPU 物理、双 GPU RTX 渲染，
`monotonic_render_budget_clock=true`。只有单张 GPU 时可按第 8 节改为单 GPU 并选择新的输出目录，
不能把某次历史采集的硬件数量视为所有变体的最低要求。
预算时钟补丁在采集前用 `cc` 编译，链接 libc/libdl/pthread，并拦截当前 libstdc++ 的计时 ABI，
只在 Kit 子进程用 `LD_PRELOAD` 加载，不需要安装为系统库或全局设置 `LD_PRELOAD`。
源码/共享库/哈希先存于本次运行的 `capture/native/`，发布时保存为输出的 `native/`。
该补丁仅验证了当前 Linux x86_64 SDK；其它 SDK/平台须按第 11.1 节重新验证，关闭补丁只改变计时方案。

安装和无 Kit 自检：

```bash
./app/python.sh -m pip install -e '.[dev]'
./app/python.sh -c 'import numpy, PIL, pyarrow, pydantic'
command -v ffmpeg ffprobe cc
cc --version
./app/python.sh .agents/skills/rsr-task-development/scripts/check_task_configs.py configs/battery_disassembly
```

视频工具缺失时安装系统 `ffmpeg` 包；需要补丁而缺编译器时安装 `build-essential`。
2026-11-16 公共包、工具路径和任务配置核对通过，本轮没有重新执行物理/视频采集；
系统 `pip check` 的既有 3 项报告见公共环境说明，不将其记为本任务新缺包。

## 7. 配置

| 文件 | 用途 | 输出 |
|---|---|---|
| [kr210_physx_c1003b1a1.toml](../configs/battery_disassembly/kr210_physx_c1003b1a1.toml) | 正式 2K / 30 fps | `outputs/battery_disassembly/kr210_physx_c1003b1a1_2k/` |
| [kr210_physx_c1003b1a1_1k.toml](../configs/battery_disassembly/kr210_physx_c1003b1a1_1k.toml) | 正式 1K / 30 fps / 双 GPU（当前默认） | `outputs/battery_disassembly/kr210_physx_c1003b1a1_1k/` |
| [kr210_physx_c1003b1a1_debug.toml](../configs/battery_disassembly/kr210_physx_c1003b1a1_debug.toml) | 调试：允许覆盖、保存失败集 | `outputs/debug/battery_disassembly/kr210_physx_c1003b1a1_debug/` |

主要字段：`scene.position_m`（源配置工位位姿）、`scene.battery_yaw_deg=-90`、`scene.screw.*`（螺纹、尺寸）、
`scene.cover.*`（代理、12 kg）、`robot.screwdriver.*` / `robot.gripper.*`、`planner.*`（速度、高度、阶段参数）、
`job.acceptance.*`（门限）、`job.recorder.record_rgb`（false 只跑物理）。
`job.monotonic_render_budget_clock` 默认 false，当前 1K 配置启用；其启动要求见[第 11.1 节](#render-budget-clock)。

## 8. 运行

```bash
./app/python.sh scripts/battery_disassembly/prepare_battery_disassembly_assets.py
./app/python.sh -m pip install -e ".[dev]"
# 只跑物理（约 6 min）
RSR_JOB__RECORDER__RECORD_RGB=false ./scripts/rsr collect run -c configs/battery_disassembly/kr210_physx_c1003b1a1_debug.toml
# 当前默认：正式 1K / 30 fps / 双 GPU PathTracing + 独立校验
./app/python.sh scripts/battery_disassembly/collect_battery_disassembly.py
# 保留的 2K 单卡备选，同样自动独立验收
./app/python.sh scripts/battery_disassembly/collect_battery_disassembly.py --config configs/battery_disassembly/kr210_physx_c1003b1a1.toml
```

GUI：`RSR_SIM_WORLD__HEADLESS=false`，`RSR_JOB__INSPECT_GUI=true` 在各阶段停下等 Enter。
当前 1K 采集还需要 Linux x86_64 的 C 编译器 `cc`（Debian/Ubuntu：`build-essential`）；
须通过上述采集脚本加载时钟补丁，直接执行 `rsr collect run -c ..._1k.toml` 会因缺少补丁而报错。

## 9. 输出与数据格式

```text
outputs/battery_disassembly/kr210_physx_c1003b1a1_2k/
├── result.json, validation.json, qa/keyframes.jpg
├── diagnostics/<UTC>/           # configuration、state_log.npz、layout、contacts、trajectory.jsonl、result
└── lerobot_v3/                  # meta/info.json、data/、videos/observation.images.{overview,screwdriver,gripper}/
    └── meta/physics/episode_000000/  # state_log.npz（60 Hz）、layout.json、contacts.json、acceptance.json
```

- `robot_type = "kuka_kr210_l150_dual"`；`task` 见配置。
- `observation.state`（16）：`screw_arm.joint_a1..a6`、`grip_arm.joint_a1..a6`（rad）、
  `screwdriver.float_m`（浮动行程，m，正为伸出）、`screwdriver.spindle_rad`（主轴累计转角）、
  `gripper.finger_p_m` / `finger_n_m`（m，闭合为负 / 正）。
- `action`（16）：两臂关节位置目标（rad）、浮动目标（m）、主轴速度目标（rad/s）、夹爪闭合目标（m）、磁力开关。
- 额外列：`observation.screw_positions`（12）、`observation.thread_slide_m`（4）、`observation.cover_pose`（7）、
  `observation.phase_index`、`next.done`、`next.success`。

## 10. 验收

Job 在完整状态记录（60 Hz）和接触报告上检查，`scripts/battery_disassembly/validate_battery_disassembly.py`
从发布的 `meta/physics` 文件重算同一组检查：

| 检查 | 含义 |
|---|---|
| `finite_states` | 状态记录全部有限 |
| `sequence_completed` | 两臂程序全部执行完、规划器没有报失败（寻位 / 拧松 / 夹持各有超时与判据） |
| `all_threads_disengaged` | 四个螺纹都走到末端、螺丝与 carrier 分离 |
| `screws_turned_out_by_contact` | 螺纹转角 ≥ 0.95 × 16 圈 |
| `thread_lead_matches_pitch` | 上升 / 圈数 与 1.5 mm 相差 ≤ 2 % |
| `bit_drove_screw_without_slip` | 拧松期间螺丝转角 / 主轴转角 ≥ 90 % |
| `screws_in_basket` | 四颗螺丝最终在篮子内腔包围盒内 |
| `screws_at_rest`、`cover_at_rest` | 最后 1.5 s 位移 ≤ 2 mm |
| `cover_rests_on_pallet` | 盖底与托盘顶面相差 ≤ 10 mm |
| `cover_level` | 盖板倾斜 ≤ 3° |
| `cover_on_pallet_footprint` | 盖板质心在托盘范围内 |
| `cover_held_by_friction_without_slip` | 闭合后到张开前，盖板相对两指中点的位移 ≤ 10 mm |
| `screws_touch_only_cover_bit_basket` | 螺丝没有碰到桌面、电池壳等其它物体（没有掉落） |
| `arm_links_collision_free` | KR210 连杆与任何物体的接触冲量 ≤ 1e-3 N·s |
| `tools_touch_only_their_targets` | 螺丝刀只碰螺丝、夹爪只碰盖板（≤ 0.05 N·s） |

前置检查（启动 Kit 前或首个物理步前失败）：PhysX、物理频率为整数 Hz 且被 fps 整除、相机名唯一、已有数据集拒绝覆盖；
两臂全部名义轨迹 IK 可达且连续。独立校验另查数据行与物理记录逐帧一致、额外列一致、三路视频完整解码且无黑帧，
并写 `qa/keyframes.jpg`；`qa/visual_review.json` 记录关键帧人工查看（由 agent 查看，非用户复核）。

## 11. 问题与解决方案

<a id="render-budget-clock"></a>

### 11.1 墙钟回跳导致 GPU 渲染停滞

#### 现象与适用环境

本机环境为 Isaac Sim 6.2 rc.14 / Kit 110.4（构建 `187cf21b`）、Linux x86_64，
两张 NVIDIA RTX 5880 Ada 48Q vGPU，驱动 570.172.18。
1K 三路 PathTracing 正常约 6 s/组，运行一段后骤降至约 81 s/组；一条 `carb.tasking` 线程
占满一个 CPU 核，GPU 0 / GPU 1 利用率约 4% / 0%，主线程在渲染更新中等待。
更早的视口采集还出现过约 452.6 s/组，记录见
[render_slowdown.json](../outputs/debug/battery_disassembly/pathtracing_1k_multigpu_20261114T102315Z/render_slowdown.json)。
这里一“组”指同一采集时刻三路相机各一帧。

#### 已确认的故障链与证据

1. 只读 GDB 栈、寄存器和反汇编将忙线程定位到 `libgpu.foundation.plugin.so` 的显存预算历史更新循环，
   栈偏移为 `+0x193bf6`、`+0x19609b`，底层反复执行 `memmove`。
2. 当场捕获“当前时间 − 上次时间”为 **−277,478,771 ns（约 −277.48 ms）**。
   负差值转换为无符号微秒后回绕，单精度累计秒数达到约 **1.8446744e13 s**。
3. 历史循环每次减去 0.5 s，在这个量级已无法改变 float32 的值，因此反复删除、搬移、追加历史样本，
   任务无法完成；主线程等不到预算任务，GPU 随之空闲。
4. 后续补丁在锁内直接审计 `CLOCK_REALTIME`，实际记录到约 42.7 ms、103.8 ms、43.3 ms 的回退读数，
   说明本机确实存在墙钟回跳。墙钟尚未追上之前可能连续记录多行，日志行数不能当作独立回跳次数。

原生分析见 [native_finding.json](../outputs/debug/battery_disassembly/mgpu_investigation/native_finding.json)、
[stalled_native_02.txt](../outputs/debug/battery_disassembly/mgpu_investigation/stalled_native_02.txt) 和
[hot_function.asm](../outputs/debug/battery_disassembly/mgpu_investigation/hot_function.asm)。
实际回退读数保存在本轮 `capture.log` 和
[verification.json](../outputs/debug/battery_disassembly/clock_1k_20261114T1233Z/verification.json)。

**已确认的是墙钟异常和预算循环的数值处理故障；主机墙钟为何回跳仍待系统层定位。**
没有证据将其归因于桌面最小化、远程连接、某个时间同步服务或驱动；用户确认桌面与终端始终开启。
本项目没有调整系统时钟、时间同步服务或 NVIDIA SDK 二进制文件。

#### 解决方案：只为 GPU 库提供单调计时

项目补丁 [gpu_budget_clock.c](../scripts/battery_disassembly/gpu_budget_clock.c) 通过 `LD_PRELOAD`
拦截 C++ `std::chrono::system_clock::now()`，按返回地址识别调用方：

- 来自 `libgpu.foundation.plugin.so` 的调用返回“首次墙钟值 + `CLOCK_MONOTONIC` 增量”；
  初始化与状态更新受锁保护，生成的时间值严格递增，避免预算统计得到负时间差。
- 其它调用方仍返回原始 `CLOCK_REALTIME`。补丁作用于该 GPU 库被拦截的计时调用，
  不修改 PhysX 240 Hz 步长、视频 30 fps、PathTracing 采样和反弹次数。
- 采集器仅在 Kit 子进程的环境中追加此预加载库，并保留已有 `LD_PRELOAD`；
  父进程的独立验收流程不加载它。

这是本项目针对当前 SDK 的进程内规避方案，主机墙钟回跳的来源仍需另行排查。
实现依赖 Linux x86_64 下 libstdc++ 的符号与 64 位返回值 ABI；更换 SDK、平台或标准库后须重新验证，
不能仅凭配置开关认定拦截仍然生效。

#### 启用方法与归档

当前 [1K 配置](../configs/battery_disassembly/kr210_physx_c1003b1a1_1k.toml) 已设置：

```toml
[job]
monotonic_render_budget_clock = true
```

其它配置默认关闭。系统需要 `cc`；Debian/Ubuntu 可安装 `build-essential`。
在仓库根目录通过采集器启动：

```bash
./app/python.sh scripts/battery_disassembly/collect_battery_disassembly.py \
  --config configs/battery_disassembly/kr210_physx_c1003b1a1_1k.toml
```

采集器在启动 Kit 前用 `cc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared` 编译共享库，
链接 `dl`、`pthread`，随后设置该子进程的 `LD_PRELOAD`。
Job 在初始化 Kit 前检查 `rsr_budget_clock_version() == 1`；开启此选项但未加载补丁时立即报错，
因此不能绕过采集器直接用 `rsr collect run` 启动该配置。

每次运行的审计位置如下（`<run>` 位于 `outputs/debug/battery_disassembly/`）：

| 文件 / 日志 | 用途 |
|---|---|
| `<run>/capture/native/gpu_budget_clock.c` | 本次实际使用的源码副本 |
| `<run>/capture/native/libgpu_budget_clock.so` | 本次编译、预加载的共享库 |
| `<run>/capture/native/clock.json` | 补丁模式、作用范围、源码与共享库 SHA-256、执行库路径 |
| `<run>/status.json`、发布后的 `capture_manifest.json` | 进度及 `render_budget_clock` 记录 |
| `<run>/capture.log` 的 `[rsr-budget-clock] active` | 首次命中目标库的证据；同时含进程和调用库路径 |
| 同日志的 `[rsr-budget-clock] realtime reversal ... ns=...` | 原墙钟低于此前最大读数超过 1 ms 的审计；预算计时仍使用单调值 |

验收发布时，`<run>/capture/` 整体移动到正式输出目录，补丁文件随之位于 `<output>/native/`；
`clock.json` 的 `execution_library` 保留原执行路径，源码和共享库的相对路径及 SHA-256 用于发布后核对。
`status.json` 与 `capture.log` 留在原调试运行目录。

本次 `clock_1k_20261114T1233Z` 长测在正式开关与版本检查加入前，已用同一计时算法的原型启动；
实际运行的原型源码和共享库已单独归档，原执行配置保持原样。后续新采集使用上述正式入口。

#### 验证结果与完成边界

原生回归测试在无 Kit 的独立子进程中模拟墙钟回退 **1 s**：未加补丁的目标库计时回退 1 s；
加补丁后目标库计时继续递增，而其它库调用方仍观察到原回退。另有测试确认请求补丁但未加载时，
Job 在 Kit 初始化前拒绝启动。此前实现的检查结果为 `ruff check` 通过、无 Kit 测试
**317 通过 / 3 跳过**，三个电池任务配置与后端发现检查通过。
时钟专项回归位于 [test_battery_disassembly.py](../tests/test_battery_disassembly.py)，可单独运行：

```bash
./app/python.sh -m pytest -q tests/test_battery_disassembly.py \
  -k 'native_budget_clock or requested_native_clock'
```

2026-11-14 **08:00 EST（13:00 UTC）附近**的长测快照：三路各完成 269 帧，近期约 6.60 s/组，
最慢进度窗口 6.68 s/组；326 次 GPU 采样平均 **82.1% / 44.3%**，峰值 **99% / 94%**，
显存峰值 7428 / 4689 MiB。实际捕获上述墙钟回退后仍继续采集，未检测到先前的单核空转。
三路第 100 帧已实际查看，视角和画面正常。持续数据见本轮 `utilization.jsonl` 与 `verification.json`。

这些结果验证了异常输入下的处理及此前退化位置之后的持续运行；完整 episode 尚未完成。
整段物理验收、三路视频完整解码与最终关键帧复核仍待完成，不能据此将完整 1K 数据记为已验收。

#### 先前对照与排除结果

| 尝试 | 实测与结论 | 最终处理 |
|---|---|---|
| 关闭 `/rtx/multiThreading/enabled` | 121 组短测约 11.5 s/组，但后续正式运行仍在第 36 / 124 组附近退化；不能解决计时故障 | 1K 保留 `renderer_multithreading=false`，仅控制 CPU 命令录制，不关闭多 GPU |
| 三路固定 RenderProduct 并行采集 | 121 组短测平均 5.80 s/组、最慢窗口 6.26 s/组，两卡平均 80.4% / 46.1%；提高吞吐，但不能防止墙钟下溢 | 保留 `capture_mode="parallel_cameras"`，配合单调时钟补丁 |
| 关闭贴图流送、贴图常驻 | 初始化预算子系统为 0，之后仍注册 `GeometryStreaming...`，约第 50 组复现同一循环；原故障子系统为 `TextureStreaming` | 已撤回启动开关，恢复默认贴图流送 |
| Carb 只用一个工作线程 | PhysX 初始化时持续等待；首次写入 `extra_args` 的覆盖又被 SDK 后追加的默认 16 线程冲掉，该次试验无效 | 已撤回线程数覆盖，恢复默认 CPU 线程池 |

并行采集使用本机 Isaac Sim 6.2 的 `ViewportManager.optimize_render_products()` 分配 GPU，
当前 GPU 0 处理 screwdriver / gripper，GPU 1 处理 overview，关闭闲置预览视口渲染；
两卡负载不要求相等。1K 配置设置 `active_gpu="@none"` 清除父配置主卡选择、`multi_gpu=true`，
不并行启动多个 Kit。每次采集 `delta_time=0`，并核对 PhysX 步数、相机矩阵、分辨率和 PT 采样。
接口依据见[Isaac Sim 6.2 多 GPU 相机分配](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/py/source/extensions/isaacsim.core.rendering_manager/docs/index.html#multi-gpu-camera-allocation)。

两个 121 组对照各有 241 条物理记录、25 类字段与已验收 640 轨迹对应前缀逐位相同（最大差 0），
并行对照三路第 120 帧已查看。但后续回归推翻了“短测已解决长期停滞”的结论，不能用这些部分轨迹替代完整验收。
详细记录见 [threading_comparison.json](../outputs/debug/battery_disassembly/mgpu_investigation/threading_comparison.json)、
[parallel_comparison.json](../outputs/debug/battery_disassembly/mgpu_investigation/parallel_comparison.json)、
[regression.json](../outputs/debug/battery_disassembly/pathtracing_1k_parallel_20261114T113928Z/regression.json) 和
[resident_native.txt](../outputs/debug/battery_disassembly/mgpu_investigation/resident_native.txt)。
所有失败、对照和未完成轨迹均保留在 `outputs/debug/battery_disassembly/`。

### 11.2 物理与早期采集问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 螺纹探针每圈复位 | rack-and-pinion 关节用包裹转角 | 改为 articulation mimic 螺旋约束 |
| 螺丝锁死、主轴速度 ±400 rad/s 抖动 | mimic gearing 单位是 m/°，按 m/rad 写；主轴惯量太小 | gearing = −pitch/360；主轴关节加 2e-4 kg·m² armature |
| 两个手指 DOF 名重复（`rsr_finger_joint`、`rsr_finger_joint_0`） | 同名关节被 PhysX 改名 | 关节名带侧别 `rsr_finger_{p,n}_joint` |
| 拧松阶段超时（测得上升 13 mm） | 螺纹脱开后浮动弹簧把批头和螺丝压回 11 mm，上升量按当前值计算 | 用阶段内最大上升量判定 |
| 抬升时螺丝掉到桌上 | 磁力参考点是螺丝头顶，叶片插入槽底时反而向下拉，螺丝悬在叶片外 | 参考点改为槽底，贴合时沿工具轴向上拉，叶片留在槽内限位 |
| 篮子里螺丝永不停（质心 1.3 cm/s，位置不变） | PhysX 默认没有扭转摩擦，躺倒的螺丝绕竖轴一直转 | 螺丝碰撞体 `torsionalPatchRadius` 3 mm、角阻尼 0.2；静止判据改为最后 1.5 s 位移 ≤ 2 mm（速度读数在接触中有 6 mm/s 抖动） |
| RGB 调试运行变慢到每张图约 170 s（rgb1 在 t=80 s，rgb2 在 t=104 s） | 主线程停在 `SimulationApp.update`（`stall_tracebacks.txt`），一个 `carb.tasking` 线程占满一核、GPU 空闲；当时均发生在运行中将 `hold` 的 `physics:jointEnabled` 设为 false 之后，PhysX 报 `CreateJoint ... disjointed body transforms`，只跑物理未变慢；缺少当时的原生预算栈，不能据此确认 USD 修改是原因，也不能确认与第 11.1 节同因 | 终止两次运行（各自 `run_note.json`）；当时改为 PhysX 可断裂关节，运行中不再改 USD；Job 加 `stall_dump_s` 看门狗 |
| 只用断裂扭矩时螺丝不脱出 | 行程上限 + mimic 的止挡太软（角度方向刚度按 gearing² 缩小），转角越过止挡 176° 仍只有不到 0.5 N·m | `turn` 加 5760° 硬上限，`slide` 上限放宽 0.5 mm |

## 12. 局限

- 螺纹是 mimic 螺旋约束加粘性摩擦，不是螺纹接触；没有自锁、预紧力或拧紧扭矩模型，脱扣是到长度后断开关节。
- 电磁批头是按距离线性衰减的外力（3 N，6 mm），不是磁场计算。
- 电池包、盖板、篮子用盒体碰撞代理；视觉网格与代理的误差在毫米级（例如代理螺丝头比视觉顶面高约 1.4 mm）。
- KR210 关节驱动刚度沿用资产（1e15/°），相当于理想位置伺服；没有电机力矩上限与柔性。
- 只有 c1003b1a1 一个场景、一个 episode；其余 29 个分场景和整包工位未迁。
- 未用官方 LeRobot SDK 加载验证；人工视觉复核只覆盖关键帧。
