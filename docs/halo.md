# HALO G1：原生 29 关节 G1 在客厅爬过矮桌、弯腰走过颈高桌、走到沙发坐下

更新日期：2026-11-13（America/New_York；运行时间均记 UTC）。

## 1. 任务介绍

从 HALO 项目（`/home/fufa/Projects/halo`）迁入的三个子任务，对应源项目的最终版本：

| 子任务 | 源版本 | 动作 | 参考时长 |
|---|---|---|---|
| crawl | `crawl_v3` | 走近 0.83 m 矮桌 → 降身 → 手膝着地爬过桌下 → 全身出桌后及时起身 → 站稳停留 | 20.0 s |
| crouch | `crouch_v3` | 走近 1.13 m 颈高桌 → 屈髋屈膝、躯干前倾、双脚交替走过桌下（手膝不着地）→ 及时直起 → 站稳 | 15.8 s |
| sit | `sit_v1` | 平滑起步 → 走到电视前原生双人沙发 → 转身 → 双脚约束下坐 → 接触触发保持 → 面向电视坐稳 | 24.0 s |

机器人是 Isaac Sim 6.2 自带的 Unitree G1（29 个身体关节、三指手、46 个刚体、34.02 kg），银白参考材质。
GEAR-SONIC 编码器/解码器以 50 Hz 闭环跟踪一段参考动作，输出 29 个电机的位置目标；PhysX 以 500 Hz 积分
自由基座和有限力电机。参考动作只是 SONIC 的目标输入，机器人状态全部来自物理，初始状态之后不再写入基座。
头部相机和第三视角相机两路 RGB 与 50 Hz 控制记录一起导出为 LeRobot v3。

### 1.1 流程

```
init_world      新建 stage，引用 home_010（z −0.015 m），PhysX CPU TGS 500 Hz、CCD 开、稳定化关
arrange_scene   客厅全部网格设为静态三角网格碰撞（μs 1.0 / μd 0.8，4 / 0.5 mm 偏移）；
                薄地板与 Decals 关闭，隐形 FloorCollider 承重；唯一的刚体移除
                crawl / crouch：引用 Blender 桌子并设碰撞；sit：校验原生沙发未被改动
                创建第三视角相机（crouch / sit 固定，crawl 跟随）
place_robot     before hook：机器人起始位姿 = 参考第 0 帧
                引用 G1，articulation 根在 pelvis，43 个 force 驱动（29 身体 + 14 手指），
                接触报告、推测式 CCD；pelvis 摆到起始位姿；头部相机挂在 torso_link
                planner：加载 SONIC ONNX（CPU）并把参考重采样到 50 Hz
start_simulation
  before hook   reset → play → 写一次起始状态与电机 armature → SONIC 历史复位 → 订阅接触报告 → 锁曝光
  simulate      1 s 不记录的起始姿态跟踪 → 每 20 ms：读实测状态 → SONIC → 目标调理 → 下发
                → 记录 50 Hz 行；图像时刻（i/fps 最近的控制帧）两路相机各采一张 → 10 个 PhysX 步
                → 结束后在完整 50 Hz 记录上做验收 → 导出 LeRobot v3 + meta/physics
finally: close
```

### 1.2 约束与验收口径

- 原装 G1 资产与原生关节，不加隐藏碰撞体、不固定基座、不回放参考位姿；电机力矩不超过 SONIC 电机表的上限。
- 图像不插值：每张图取离 `i / fps` 最近的实测控制帧，名义时间与实测时间都记录，量化误差 ≤ 6.67 ms（30 fps）。
- 验收项与门限照搬 HALO 的 `validate_physx.py` 和 `camera_geometry.py`（第 10 节），没有放宽。
- 当前采集规格：验证 640×360 / 15 fps、实时 RTX `RayTracedLighting`；正式 1024×576 / 30 fps、`PathTracing` 16 spp。两阶段均为 head + rear、三个子任务各 1 个 episode。实时 RTX 仍使用光线追踪，已获用户确认。

## 2. 当前结果

### 2.1 本次采集（2026-11-13 本地时间 / 2026-11-14 UTC）

本次按“640×360 / 15 fps 实时 RTX 验证 → 1024×576 / 30 fps PathTracing 正式采集”顺序执行，
三个子任务各 1 个 episode、每集 head + rear 两路 RGB，数据格式为 LeRobot v3。实时 RTX 仍使用光线追踪，
已获用户确认。正式渲染为 16 spp、6 次反弹、OptiX 降噪，始终只有一个 Kit 进程。

| 项目 | crawl | crouch | sit |
|---|---|---|---|
| 正式输出 | [outputs/halo/crawl](../outputs/halo/crawl/) | [outputs/halo/crouch](../outputs/halo/crouch/) | [outputs/halo/sit](../outputs/halo/sit/) |
| 验证帧数（15 fps） | 300 | 237 | 360 |
| 正式帧数（30 fps） | 600 | 474 | 720 |
| 时长 | 20.0 s | 15.8 s | 24.0 s |
| 完整 50 Hz 控制记录 | 999 行 | 789 行 | 1199 行 |
| 验证耗时（含启动、独立校验） | 160.9 s | 132.4 s | 189.6 s（完整重跑） |
| 正式耗时（含启动、独立校验） | 1988.0 s（33.1 min） | 1565.3 s（26.1 min） | 2238.6 s（37.3 min） |
| 正式 Job 耗时 / 其中图像采集 | 1956.3 / 1889.25 s | 1535.7 / 1484.22 s | 2208.4 / 2123.67 s |
| 任务验收（两个阶段） | 31/31 | 33/33 | 25/25 |
| 独立校验（两个阶段） | 16/16 | 16/16 | 16/16 |
| 正式 qpos / action 与本次验证 | 逐位一致 | 逐位一致 | 逐位一致 |

三个正式集共 **1794 组双相机图像、59.8 s**，即 3588 张 RGB 图像编码成六路 H.264 视频。
各集的 `result.json.success=true`、`validation.json.passed=true`、`dataset_use=demonstration`。
独立校验从落盘的完整控制记录重新计算物理验收，并全量解码视频核对帧数、fps、尺寸、黑帧、状态与动作行。
三个验证集的 qpos 也分别与第 2.2 节的历史通过记录逐位一致，所以该节的物理实测指标同样适用于本次采集。

运行窗口：验证从 01:38:22 UTC 开始，正式采集 01:51:51–03:28:23 UTC；
验证中 sit 第一次运行在第 330 张 head / 第 329 张 rear 后停止推进，终止后保留全部落盘证据，
保持相同配置从头重跑，重跑通过，未拼接轨迹、未改变门限。三个正式集均一次通过。
验证目录位于 [capture_1k30_20261113/validation](../outputs/debug/halo/capture_1k30_20261113/validation/)，
通过的 sit 验证为 `sit_retry01/`，原 `sit/` 为未完成证据。

完整运行记录、配置快照、命令与日志见
[collection.json](../outputs/debug/halo/capture_1k30_20261113/collection.json)、
[run_note.json](../outputs/debug/halo/capture_1k30_20261113/run_note.json) 和同目录的 `collect.py`；
正式集汇总见 [collection_manifest.json](../outputs/halo/collection_manifest.json)。
已实际查看六个通过数据集各自 `qa/keyframes.jpg` 的两路各 6 帧（共 72 张缩略图），
覆盖接近、桌下通过或下坐、起身或坐稳末段；未做人工逐帧复核，也未对本次新数据重新运行官方 LeRobot reader。

回归：全仓 `ruff check real2sim2real scripts` 通过，`pytest -q` 为 288 passed / 3 skipped，
六份 HALO 配置通过 `check_task_configs.py`，`list-backends` 可见 HALO 四域后端。
本次只调整 HALO 采集配置、对应既有配置测试和文档；物理、控制、资产与验收逻辑未改。

### 2.2 迁移阶段验证（历史记录）

调试验证于 2026-11-14 01:07–01:31 UTC 依次运行（一次一个进程；crouch / sit 之后只给 Job 加了“卡住时写线程栈”的诊断，
crawl 用最终代码重跑）。当时使用 PathTracing；旧 2K 正式配置未运行。逐次运行记录见 `outputs/debug/halo/run_note.json`。

| 项目 | crawl | crouch | sit |
|---|---|---|---|
| 输出 | [outputs/debug/halo/crawl](../outputs/debug/halo/crawl/) | [outputs/debug/halo/crouch](../outputs/debug/halo/crouch/) | [outputs/debug/halo/sit](../outputs/debug/halo/sit/) |
| 数据集 | 300 帧 / 15 fps / 20.0 s | 237 帧 / 15 fps / 15.8 s | 360 帧 / 15 fps / 24.0 s |
| 50 Hz 控制记录 | 999 行 | 789 行 | 1199 行 |
| 相机 | head + rear（跟随），640×360 H.264 | head + rear（固定） | head + rear（固定） |
| 物理步 | 10480（含 1 s 预热） | 8380 | 12480 |
| 墙钟（其中采集） | 459 s（395 s） | 360 s（311 s） | 525 s（443 s） |
| Job 验收 | 31 项全部通过 | 33 项全部通过 | 25 项全部通过 |
| 独立校验 | 16 项全部通过 | 16 项全部通过 | 16 项全部通过 |
| 与 HALO 记录对比 | 999 行 qpos 逐位相同 | 789 行逐位相同 | 1199 行逐位相同 |

对比对象：crawl_v3 `demo_reference_material_2k_30fps_001`、crouch_v3 `demo_reference_g1_2k_30fps_002`、
sit_v1 `demo_2k_30fps_001` 的 `control_recording.npz`。当时三次调试运行与这些源记录的物理轨迹逐位相同；关键指标与 HALO README 报告的正式结果一致：

| 指标 | 实测 | 门限 |
|---|---|---|
| crawl 碰桌冲量 / 最大穿透 | 0 N·s / 0.40 mm | < 0.001 N·s / ≤ 5 mm |
| crawl 最小头部碰撞体净空 / 桌腿通道余量 | 4.04 cm / 14.79 cm | ≥ 0 / > 1 cm |
| crawl 躯干居中：均值 / 中位 / P95 / 最大 | 0.58 / 1.30 / 3.40 / 4.17 cm | 中位 ≤ 1.5、P95 ≤ 4、最大 ≤ 5 cm |
| crawl 全身出桌 / 明显起身 | 14.98 s / 15.60 s | 起身在出桌后 −0.25…1.2 s |
| crawl 非足部地面冲量 | 3443 N·s | > 5 N·s（手膝承重） |
| crouch 碰桌冲量 / 最大穿透 | 0 N·s / 0 mm | 同上 |
| crouch 最小头部净空 / 桌腿通道余量 | 2.95 cm / 4.80 cm | ≥ 0 / > 1 cm |
| crouch 全身出桌 / 明显起身 | 10.54 s / 11.28 s | 起身在上身出桌后 −0.02…1.2 s |
| crouch 非足部地面冲量 / 左右脚卸载摆动 | 0 N·s / 4、5 次 | < 0.02 N·s / ≥ 2 次 |
| crouch 躯干前倾中位 / 最低膝 / 最低腕 | 37.0° / 0.208 m / 0.469 m | ≥ 25° / > 0.10 m / > 0.20 m |
| sit 末 2 s 骨盆均值 | (3.68512, −3.13649, 0.52709) m | — |
| sit 座垫中心误差 / 横向误差 | 1.53 cm / 0.36 cm | ≤ 10 cm / ≤ 5.5 cm |
| sit 沙发承重 / 竖直力平衡误差 | 82.2 % / 1.29 % | > 35 % / < 10 % |
| sit 末 2 s 位置极差（X / Y / Z） | 0.053 / 0.012 / 0.010 mm | < 25 mm |
| sit 躯干竖直 / 面向电视余弦最小值 | 0.99666 / 0.99679 | > 0.95 |
| sit 坐姿保持触发 | 15.5 s（实测座面接触） | 触发 |
| 第三视角最小画幅余量（640 宽） | crawl 38.1 px、crouch 49.3 px、sit 51.6 px（沙发 83.6 px） | ≥ 12.8 px（2 %） |

调试过程中的失败运行只保留日志（`outputs/debug/halo/logs/`），原因与处理见第 11 节。
三个数据集都能被官方 LeRobot reader（HALO 环境里的 lerobot 0.7.1，`LeRobotDataset(..., video_backend="pyav")`）
读取，首、中、末帧维度为 state 71、action 29、两路图像 3×360×640。`validate_halo.py` 另写出
`qa/keyframes.jpg`（每路相机 6 帧），未做人工逐帧视觉复核。

## 3. 设计选择

### 3.1 物理引擎：PhysX CPU TGS 500 Hz（沿用 HALO）

任务只有刚体与关节，HALO 的全部正式数据都在 PhysX 下标定（桌子位置、出桌时机、净空、坐姿保持门限），
所以不换引擎。CPU 版 PhysX 在本机逐位可复现（同一 HALO 脚本今天重跑与它记录的数据完全相同），
GPU 只用于渲染。没有考虑 Newton：SONIC 已在 HALO 的 PhysX 设置下验证，换引擎需要重新标定所有场景参数。

### 3.2 与 HALO 逐位一致的初始化

HALO 在 `timeline.play()` 之后先调用一次 `app.update()`（时间轴在播放，物理会走几步，此时 G1 还在资产原点），
然后才通过 PhysX 写入起始状态。这几步不出现在数据里，但会让后续轨迹在 mrad 级别不同；爬行等多接触动作对
这种差异敏感，20 s 后基座位置可差 18 cm，HALO 按自己轨迹标定的桌子位置因此不再居中（迁移第一次调试运行的
躯干中位偏差 7.2 cm，门限 1.5 cm）。迁移后保留同样的顺序：`robot.place_at_start=false`、
`job.startup_app_updates=1`。这一次更新的物理步不计入 `physics_steps`，起始状态随后完整覆盖位姿、关节和速度。

另一个需要保持的细节是关节限位的精度：HALO 把目标裁剪到 PhysX 返回的 float32 限位（`lo + 0.001` 也按
float32 计算），迁移后同样保留 float32。两处都对齐后，三个子任务的完整 50 Hz 记录（qpos、关节速度、动作、
接触等）与 HALO 的正式记录逐位相同（第 2 节）。

### 3.3 控制：GEAR-SONIC 编码器/解码器

`utils/halo_utils.SonicTracker` 逐行移植 HALO 的 `sonic_tracker.py` 与 `sonic_contract.py`：参考动作按 50 Hz
重采样（关节线性插值、基座四元数 slerp），编码器取未来 10 帧（步长 5）的关节位置、速度与锚点 6D 姿态，
解码器取 10 帧历史。用 HALO crouch_v3 记录的实测状态回放，移植后的输出与 HALO 记录的动作中位差 2×10⁻⁷ rad，
差异只出现在 HALO 已按关节限位裁剪的帧。每帧下发的是调理后的目标，并用它更新 SONIC 的 last-action 历史。

### 3.4 图像采样与渲染

图像不插值：第 i 张图取离 `i / fps` 最近的 50 Hz 控制帧（30 fps 量化 ≤ 6.67 ms，15 fps ≤ 6.67 ms），
渲染前后核对 `physics_steps` 不变。两路相机依次用 `capture_viewport_frame` 采集，
`synchronous_capture=true`；验证用实时 RTX `RayTracedLighting`，正式用 `PathTracing`、16 spp、6 次反弹、
OptiX 降噪、关闭两项光照缓存与自动曝光。HALO 用 Replicator
双 render product，并因长段采集变慢而按 75 个控制点分段、每段从头重跑物理；迁移后一次连续运行、不分段。
历史 640×360 PathTracing 验证中，两路相机合计约 1.2–1.4 s/图像时刻（采集占墙钟 85 %）；
当前实时 RTX 验证和 1K PathTracing 正式采集的实测耗时见第 2 节。

### 3.5 相机

两路相机都用 `CameraConfig` 复现 D455 彩色镜头（焦距 1.93 mm、水平光圈 3.896 mm、16:9 裁剪竖直光圈、
无畸变），没有引用 `rsd455.usd`：RGB 渲染只用到焦距、光圈和裁剪面，D455 资产里的 `omni:sensor:tickRate`
等属性 HALO 本身也关掉了。头部相机挂在 `torso_link` 下 (0.12, 0, 0.43) m、光轴下俯 25°。第三视角：crawl
按实测骨盆平滑跟随（每个控制帧改写相机 USD 位姿，录制时核对写入与读回误差 < 0.1 mm），crouch / sit 固定。

## 4. 组成

| 模块 | 后端名 | 实现 | HALO 对应 |
|---|---|---|---|
| SimWorld | `isaacsim`（PhysX CPU） | [simworld/isaacsim.py](../real2sim2real/simworld/isaacsim.py) | `collect_physx.py` 的 SimulationApp 与 PhysicsScene |
| Scene | `halo_home` | [scene/scene_halo.py](../real2sim2real/scene/scene_halo.py) | `physx_world.build`：客厅碰撞、地面代理、桌子、第三视角 |
| Robot | `g1_sonic` | [robot/robot_halo.py](../real2sim2real/robot/robot_halo.py) | `physx_world.build` 的 G1 部分、`collect_physx.read()`、头部相机 |
| Planner | `sonic_tracking` | [planner/planner_halo.py](../real2sim2real/planner/planner_halo.py) | `sonic_tracker.py`、`native_motor_control.py`、sit 的坐姿保持 |
| Job | `halo` | [job/job_halo.py](../real2sim2real/job/job_halo.py) | `collect_physx.py` 主循环、`validate_physx.py`、`export_lerobot_v3.py` |
| 辅助 | — | [utils/halo_utils.py](../real2sim2real/utils/halo_utils.py) | SONIC 契约、采样、跟随相机、`ContactMonitor`、`camera_geometry.py` |

迁移后的调整：

- 路径改为 rsr 约定：客厅 `/World/background`、桌子 `/World/scene/Desk`、G1 `/World/robots/G1`；
  接触分类按路径前缀（座椅 → 桌子 → 机器人自身 → 客厅障碍 → 其余为地面），与 HALO 的子串判断等价。
- 三个子任务共用同一套代码，差异全部放进配置：桌子有无、相机固定或跟随、手臂阻尼（crawl 4.0，其余 3.0）、
  手臂目标平滑（crawl）、坐姿保持与 seated 低通（sit）、验收种类。
- HALO crawl_v3 的腕部–拇指碰撞过滤对当前三指手资产不生效（目标 prim `L_thumb_proximal` 不存在），未迁移。
- 共享代码只加了默认不生效的选项：`IsaacsimWorldConfig.active_gpu` / `multi_gpu`（单卡渲染）、
  `LeRobotV3Recorder(extra_features=...)`（HALO 的附加列）。其它任务的默认行为不变。
- HALO 的参考动作生成（Kimodo、IK、拼接）、Blender 建桌和机器人材质烘焙没有迁移；迁入的是它们的最终产物，
  来源和 SHA-256 见 `assets/Halo/provenance.json`。

### 4.1 脚本

| 脚本 | 用途 |
|---|---|
| [scripts/halo/prepare_halo_assets.py](../scripts/halo/prepare_halo_assets.py) | 从 HALO 项目逐字节复制资产到 `assets/Halo/`，检查 USD 依赖闭包，写 `provenance.json` |
| [scripts/halo/validate_halo.py](../scripts/halo/validate_halo.py) | 不启动 Kit 的独立校验；`--compare` 与 HALO 源记录逐帧对比 |

## 5. 资产

```text
assets/Halo/                        907 MB，363 个文件
├── Home010/home_010.usdc           客厅（224 MB）+ textures/ + forest.exr
├── Robot/g1_29_reference.usd       29 关节 G1，银白参考材质 + Textures/G1Reference/、robot_manifest.json
├── Desk/crawl/crawl_desk.usdc      黑色矮桌 1.10×0.60×0.83 m（+ .blend 源文件、dimensions.json）
├── Desk/crouch/crouch_desk.usdc    银灰双柱桌 1.15×0.60×1.13 m
├── reference/{crawl,crouch,sit}/   qpos.npy + spec.json（30 fps 参考动作与阶段表）
├── models/sonic/                   model_encoder.onnx、model_decoder.onnx、observation_config.yaml、LICENSE
└── provenance.json                 来源、许可证、依赖检查、每个文件的 SHA-256
```

准备命令：`./app/python.sh scripts/halo/prepare_halo_assets.py`（源文件只读，目标已存在且内容不同则拒绝覆盖）。

| 资产 | 来源 | 许可证 |
|---|---|---|
| home_010 | HALO `output/v5/assets/Home010`（与 `halo/home_010` 输入逐字节相同） | HALO 项目未记录，再分发前需确认 |
| G1 | Isaac Sim 6.2 `Robots/Unitree/G1/g1.usd`（PhysX、ThreeFinger），HALO `prepare_robot.py` 展平并换材质 | Isaac Sim 资产条款 |
| 桌子 | HALO `build_desk.py`（Blender 5.2.3） | HALO 项目自产 |
| 参考动作 | crawl_v3 `crawl_29_native`、crouch_v3 `crouch_29_native`、sit_v1 `walk_then_sit` | HALO 项目自产 |
| SONIC 模型 | NVlabs/GR00T-WholeBodyControl @ 1e1ec4f5 | 代码 Apache-2.0，权重 NVIDIA Open Model License |

依赖检查：四个 USD 无 unresolved 引用，组合后只加载 `assets/Halo/` 内的 layer；G1 材质用到的
`OmniPBR.mdl` 是 Kit 内置模块，不复制。资产文件都没有修改，碰撞、摩擦、驱动等物理设置只写在运行时 stage 上。

## 6. 依赖与安装

不需要额外安装。SONIC 用 Isaac Sim 6.2 自带的 `onnxruntime`（本机 1.26.0，CPUExecutionProvider），
参考重采样与坐姿保持用自带的 `scipy`（1.17.0）。两者都只在方法内导入；缺失时 `SonicTracker` 报出安装命令。
验证：`./app/python.sh -c "import onnxruntime, scipy"`。

## 7. 配置

| 文件 | 用途 | 输出 |
|---|---|---|
| [g1_physx_crawl.toml](../configs/halo/g1_physx_crawl.toml) | crawl 正式，1024×576 / 30 fps，PathTracing，GPU 0 | `outputs/halo/crawl` |
| [g1_physx_crouch.toml](../configs/halo/g1_physx_crouch.toml) | crouch 正式，1024×576 / 30 fps，PathTracing，GPU 1 | `outputs/halo/crouch` |
| [g1_physx_sit.toml](../configs/halo/g1_physx_sit.toml) | sit 正式，1024×576 / 30 fps，PathTracing，GPU 0 | `outputs/halo/sit` |
| `g1_physx_{crawl,crouch,sit}_debug.toml` | 继承正式配置，改为 640×360 / 15 fps、实时 RTX、debug 输出、可覆盖、失败也导出 | `outputs/debug/halo/<子任务>` |

调试配置只覆盖标量：本项目的配置合并会把继承链中的 TOML 数组拼接而不是替换，所以相机尺寸写成
`width` / `height` 两个标量。物理、控制、场景和验收在正式与调试配置中完全相同；渲染不推进物理，
因此同一子任务的调试与正式运行物理轨迹相同。

| 字段 | 取值与理由 |
|---|---|
| `sim_world.physics_dt` | 0.002 s；SONIC 50 Hz，每个动作 10 个物理步 |
| `sim_world.background_offset` | (0, 0, −0.015) m，HALO 的地面对齐 |
| `scene.desk.*` | HALO 按 29 关节 G1 实测通道包络标定的位置：crawl (2.66920, −1.40134) m、86°；crouch (2.54258, −1.68) m、90° |
| `scene.seat.*` | sit 目标座垫中心 (3.7, −3.13288, 0.37678) m，原生沙发世界包围盒（校验未被移动或缩放） |
| `robot.arm_damping_nms_rad` | crawl 4.0，crouch / sit 3.0（与 HALO 各版本一致） |
| `planner.motor_smoothing` | crawl：手臂目标 15 ms 平滑、6 rad/s、150 rad/s² |
| `planner.seated_hold` | sit：骨盆 0.49–0.58 m、离座垫中心 < 0.12 m、座面竖直冲量 > 2 N·s 时触发，0.6 s 过渡，腰部扶正朝 180° |
| `robot.place_at_start`、`job.startup_app_updates` | false / 1：复现 HALO 的初始化，见 3.2 |
| `job.acceptance.*` | 子任务种类、桌下阶段名、头部碰撞体、门限（与 HALO 相同） |

## 8. 运行

```bash
./app/python.sh -m pip install -e ".[dev]"                 # 注册 halo 后端
./app/python.sh scripts/halo/prepare_halo_assets.py         # 生成 assets/Halo（约 1 分钟）
# 验证（640×360 / 15 fps，RayTracedLighting；一次只运行一个，见第 11 节）
./scripts/rsr collect run -c configs/halo/g1_physx_crawl_debug.toml
./scripts/rsr collect run -c configs/halo/g1_physx_crouch_debug.toml
./scripts/rsr collect run -c configs/halo/g1_physx_sit_debug.toml
./app/python.sh scripts/halo/validate_halo.py outputs/debug/halo/crawl \
    --compare /home/fufa/Projects/halo/output/crawl_v3/trials/demo_reference_material_2k_30fps_001/crawl/control_recording.npz
# 正式（1024×576 / 30 fps，PathTracing 16 spp；先通过验证再运行）
./scripts/rsr collect run -c configs/halo/g1_physx_crawl.toml
./app/python.sh scripts/halo/validate_halo.py outputs/halo/crawl
```

本机上两个 halo 进程并行（即使在两张卡上）多次卡住，所以一次只运行一个。GUI：`RSR_SIM_WORLD__HEADLESS=false`，
`RSR_JOB__INSPECT_GUI=true` 在各阶段停下。冒烟测试：`RSR_JOB__MAX_CONTROL_FRAMES=60`（提前结束，验收必然不通过）。

## 9. 输出与数据格式

```text
outputs/<halo|debug/halo>/<子任务>/
├── result.json                     验收结果（checks、metrics、reports、scene_report、耗时）
├── validation.json                 validate_halo.py 的独立校验（另写 qa/keyframes.jpg：两路相机各 6 帧）
├── lerobot_v3/                     1 个 episode；data/、videos/observation.images.{head,rear}/、meta/
│   └── meta/physics/episode_000000/ control_recording.npz（完整 50 Hz）、frame_mapping.csv、robot_geometry.npz、
│                                    camera_calibration.json、configuration.json、acceptance.json
└── diagnostics/<UTC>/              同上 + physical_scene.usda、contacts.json、actor_impulses.json、trajectory.jsonl
                                    （某帧卡住超过 stall_dump_s 时另有 stall_tracebacks.txt）
```

| 字段 | 内容 |
|---|---|
| `observation.state` | 71 维：29 关节位置（rad）、29 关节速度（rad/s）、骨盆位置（m）与四元数 wxyz、骨盆线速度（m/s）与角速度（rad/s，世界系） |
| `action` | 29 维：同一控制帧下发的电机位置目标（rad），关节顺序为 SONIC 的 MuJoCo 顺序 |
| `observation.images.head` / `.rear` | 头部 / 第三视角 RGB，H.264 yuv420p |
| `observation.contacts` | 7 维：上一 20 ms 内地面、桌子、座椅、箱子（恒 0）、自身、客厅障碍的接触冲量（N·s）与最大穿透（m） |
| `observation.contact_vertical_impulses` | 6 维：同上分类的竖直冲量（N·s） |
| `observation.foot_support_impulses` | 左脚、右脚、非足部的地面冲量（N·s） |
| `observation.body_landmarks` | 双踝、双膝、双腕原点（m） |
| `observation.torso_pose`、`observation.camera_{head,rear}_pose` | 位置 + wxyz；相机为 USD 轴（−Z 朝前），转 OpenCV 见 `camera_calibration.json` |
| `observation.phase_index` | 参考阶段序号（`result.json` 的 `planner.phases`） |
| `observation.measured_timestamp`、`observation.control_frame_index` | 该行实际物理时间（s）与 50 Hz 控制帧序号；`timestamp` 为名义时间 i/fps |
| `next.done`、`next.success` | 最后一行为 true（success 只在 episode 通过验收时为 true） |

`robot_type = "unitree_g1_29dof_physx"`，`task` 为配置中的英文指令。与 HALO 数据集的列名、维度和含义相同；
接触冲量属于 20 ms 控制周期，按图像行求和不等于总冲量，需用 `control_recording.npz`。

## 10. 验收

Job 结束后在完整 50 Hz 记录上检查（门限与 HALO 相同），结果写入 `result.json` 的 `checks`：

| 类别 | 检查 |
|---|---|
| 通用 | 控制帧完整、数值有限、50 Hz 均匀、单位四元数、穿透 ≤ 5 mm、实测关节超限 ≤ 0.01 rad、目标不超限、29 个有限力电机、43 个原生关节都记录、初始化后不写基座、头部相机刚性跟随 torso（< 1 mm）、图像全部采集、时间量化 ≤ 10 ms |
| 场景 | 客厅静态碰撞体 ≥ 100 且含落地灯、薄地板由代理承重 |
| 相机 | 第三视角中每帧所有碰撞体投影离画面边缘 ≥ 2 % 宽度、FK 标记点误差 < 2 mm；固定相机不动 / 跟随相机与按实测骨盆重放的滤波器一致 |
| crawl / crouch | 两侧桌沿都穿过、头部相机原点在桌下 ≥ 50 帧、站姿头顶高于横梁 10 cm 以上、头部碰撞体全程低于横梁、桌腿通道余量 > 1 cm、出桌与起身时机、零碰桌、客厅障碍冲量 < 0.05 N·s、末尾直立且 1 s 内位移 < 5 cm、站高 > 0.65 m |
| crawl | 躯干前倾中位 ≥ 30°、非足部地面冲量 > 5 N·s（手膝真正承重）、躯干居中（中位 ≤ 1.5 cm、P95 ≤ 4 cm、最大 ≤ 5 cm） |
| crouch | 前倾中位 ≥ 25° 且 P95 < 70°、膝 > 0.10 m、腕 > 0.20 m、非足部冲量 < 0.02 N·s、每只脚 ≥ 2 次卸载摆动 |
| sit | 原生沙发可见、可碰撞、网格未改、包围盒误差 < 0.1 mm；末 2 s 骨盆中心误差 ≤ 10 cm、横向 ≤ 5.5 cm、位置极差 < 2.5 cm、高度在座面上 8–25 cm；沙发承重 > 35 % 且 ≥ 95 帧有座椅接触；竖直力平衡误差 < 10 %；躯干竖直与面向电视余弦 > 0.95；非足部落地与其它家具冲量 < 0.05 N·s；坐姿保持由接触触发 |

独立校验（`validate_halo.py`，不启动 Kit）：数据集结构与附加列、每行 state / action / 附加列与 50 Hz 记录逐位相同、
名义与实测时间、两路视频（编码、帧数、帧率、分辨率、完整解码、无黑帧）、用归档文件重算的验收结果与
`result.json` 相同；`--compare` 给出与 HALO 源记录的逐帧差。

## 11. 问题与解决方案

| 现象 | 原因 | 处理 |
|---|---|---|
| 第一次 crawl 调试只有“躯干居中”未通过：中位偏差 7.2 cm（门限 1.5 cm），20 s 后基座与 HALO 记录相差最多 18 cm | HALO 在 `play()` 后、写起始状态前跑了一次 `app.update()`（G1 留在资产原点，物理走几步）；缺了这一步，第 0 帧关节差 2.6 mrad，爬行接触把差异放大。把 HALO 脚本复制到 `/tmp` 去掉这一步，HALO 自己也偏离同样量级 | 加 `robot.place_at_start=false`、`job.startup_app_updates=1`，复现同样的初始化；前 60 帧与 HALO 逐位相同 |
| 加上初始化步骤后 crawl 全程逐位一致，crouch 从第 161 帧开始出现 float32 末位差异，随后起身晚 0.84 s，“出桌后及时起身”未通过 | HALO 把目标裁剪到 PhysX 返回的 float32 关节限位（`lo + 0.001` 也按 float32 计算）；移植时转成了 float64，限位值差约 1×10⁻⁸ rad，目标碰到限位的帧开始分叉 | 关节限位保持 float32；crouch 前 260 帧与 HALO 逐位相同 |
| 运行中偶发卡住几分钟：主线程在等待、一个 `carb.tasking` 线程占满一个核、GPU 利用率 0 %。两个进程并行时出现 3 次（含两张卡上同时停住），单进程出现 1 次（crawl 第 960 帧附近），同配置重跑都没有复现 | 未定位（本机 ptrace 受限，无法附加取栈；py-spy 全程采样的那次没有卡住）。只影响墙钟，不影响数据：渲染不推进物理，轨迹仍与 HALO 逐位相同 | 一次只运行一个进程；Job 在单帧超过 `job.stall_dump_s`（120 s）时把所有 Python 线程栈写到 `diagnostics/<run>/stall_tracebacks.txt` |
| 本次实时 RTX 落座验证末段停止推进，留下 head 330 张、rear 329 张，120 s 线程栈文件为空 | 根因仍未确定，属于流程未完成，没有验收结果 | 保留原目录和日志，终止该进程并按相同配置完整重跑；重跑以及后续 1K 正式集全部通过，详见本次 `run_note.json` |
| 继承的调试配置里改 `resolution=[640,360]` 变成 4 个数 | 配置合并 `merge_enabled=True` 会拼接 TOML 数组 | 相机尺寸改用标量 `width` / `height` |
| HALO 正式 2K 需要分段采集 | Replicator 长时间 PathTracing 调度变慢 | rsr 用视口采集，不分段（未在 2K 下实测本任务的耗时） |

## 12. 局限

- 本次已完成 640×360 / 15 fps 实时 RTX 验证与 1024×576 / 30 fps PathTracing 正式采集；未运行 2K。视觉复核为每路 6 个关键帧，非人工逐帧检查。
- 参考动作、桌子和机器人材质是 HALO 的产物，迁移只复制；要改桌子位置或座高，需要按 HALO 流程重新生成参考并标定。
- 与 HALO 逐位一致依赖相同的 Isaac Sim 6.2 RC、CPU PhysX 和初始化顺序；换版本或改 `startup_app_updates` 后
  物理仍有效，但轨迹会变，需要重新验收（crawl 的躯干居中最敏感）。
- 净空、穿透和接触来自配置的碰撞体与 PhysX 接触报告，不是可见网格的穷举检测，也不是实机验证。
- 画质：未迁移 HALO 的时域画质审计；HALO 记录的 crouch_v3 门框局部噪声未通过（P99 6.64 > 6）仍是已知项。
- 每个子任务一个 episode；home_010 的许可证在 HALO 项目中没有记录。
