# 桌面轨迹回放：在 PhysX 中回放桌面抓放数据集的动作并复验

更新日期：2026-11-14（运行时间记 UTC）。

## 1. 任务介绍

从 Real2Sim2Real（`configs/arm_franka_replay_pickplace/home000.toml`，`TabletopJob` + `ReplayPlanner`）迁入。
源任务继承采集配置，把规划器换成 `ReplayPlanner`：读取已采集 LeRobot episode 的 `action`（或 `observation.state`），
逐步下发给机械臂，夹爪开合由指开度推断，成功仍按“合过爪且走完”判定。

本任务读取 [桌面抓放](tabletop_pickplace.md) 正式数据集每集的 LeRobot v3 `action`（7 个臂关节目标 + 指目标），
在源数据记录的同一布局下由 PhysX 重新执行，不调用规划器；通过抓放物理验收、且关节与木块终点与录制一致，
才导出为 LeRobot v3。用途是检验已采动作在物理中的可复现性（开环回放）。

### 1.1 流程

```
init_world / arrange_scene / place_robot   与桌面抓放相同（复用 tabletop_home / franka_tabletop 后端）
  after place hook   读取源数据集各集（info、data、meta/physics/layout.json、task_outcome.json）
                     检查关节名、源集是否为 demonstration、物理步长与源记录一致、fps 整除物理频率且等于本任务录制 fps
start_simulation
  simulate           每集：按源 layout 的木块偏移写 USD → reset → play → 核对桌面高度、基座位置、木块起始中心
                     与源记录一致（≤ 1e-4 m）→ 每个物理步下发回放目标 → 记录、采集、验收、导出
finally: close
```

回放目标：源数据每帧动作（30 fps 时每 4 个物理步一帧）之间线性插值，臂关节速度前馈用相邻两帧差分；
`interpolation="hold"` 则每帧保持并给零速度目标。线性插值用到下一帧动作，是离线回放，不是因果策略。

### 1.2 验收

桌面抓放的全部物理检查（[tabletop_pickplace.md](tabletop_pickplace.md) 第 10 节）加回放一致性：

| 检查 | 含义（门限） |
|---|---|
| `source_was_demonstration` | 源集通过了抓放验收 |
| `arm_follows_recording` | 回放与源 `observation.state` 臂关节 RMS ≤ 0.02 rad，最大 ≤ 0.05 rad |
| `fingers_follow_recording` | 指开度最大差 ≤ 5 mm |
| `object_ends_where_recorded` | 木块最终中心与源最后一帧距离 ≤ 0.01 m |

## 2. 当前结果

### 正式（1K / 30 fps PathTracing）

已完成：2026-11-15 00:23–01:27 UTC（America/New_York 11 月 14 日 19:23–20:27），
单调预算时钟、双 GPU 固定相机，1024×768 / 30 fps PathTracing 64 spp、6 次反弹、降噪。
采集、独立校验与发布共 3863.7 s（64.4 分钟）。两集各通过 **14 项抓放物理检查 + 4 项回放一致性检查**，
且在发布到正式目录后再次通过独立校验。

输出：[franka_physx_home000](../outputs/tabletop_replay/franka_physx_home000/)，
校验：[validation.json](../outputs/tabletop_replay/franka_physx_home000/validation.json)，
采集清单：[capture_manifest.json](../outputs/tabletop_replay/franka_physx_home000/capture_manifest.json)。

| 项 | 第 0 集 | 第 1 集 | 门限 |
|---|---|---|---|
| 帧数 / 仿真时长 | 276 / 9.167 s | 270 / 8.967 s | 与源采样帧数一致 |
| 臂关节 RMS / 最大偏差 | 0.000810 / 0.003669 rad | 0.000879 / 0.004096 rad | ≤ 0.02 / 0.05 rad |
| 指开度最大偏差 | 1.544 mm | 1.551 mm | ≤ 5 mm |
| 木块终点与源记录距离 | 0.087 mm | 0.081 mm | ≤ 10 mm |
| 木块抬升 / 持握滑移 | 119.69 / 0.143 mm | 119.72 / 0.178 mm | ≥ 80 / ≤ 10 mm |
| 放置误差（XY） | 0.309 mm | 0.376 mm | ≤ 20 mm |
| 臂连杆 / 手指异物 / 木块异物接触冲量 | 0 / 0 / 0 N·s | 0 / 0 / 0 N·s | ≤ 1e-3 / 0.01 / 0.01 |
| RGB 采集墙钟 | 1955.57 s | 1853.95 s | 整体平均约 7.0 s / 帧组 |

agent 已查看两集 `qa/keyframes.jpg`（俯视、腕部各 6 帧），确认夹持转运、开爪释放、木块留在桌面；
未见黑帧或苹果被碰动。每集 `qa/visual_review.json` 记录拼图 SHA-256 和检查范围，用户视觉复核仍待进行。
原单卡回放的未完成输出与修复记录见第 6 节。

### 调试（640×480 / 15 fps MinimalRendering）

`outputs/debug/tabletop_replay/franka_physx_home000_debug/`，2026-11-14 21:53 UTC，回放桌面抓放调试数据集两集。
两集都通过抓放 14 项验收、回放一致性 4 项和独立校验（独立校验从落盘数据重算一致性）。

| 项 | 第 0 集 | 第 1 集 | 门限 |
|---|---|---|---|
| 帧数 | 138 | 135 | 等于源集 |
| 臂关节 RMS / 最大偏差 | 0.0008 / 0.0037 rad | 0.0009 / 0.0047 rad | ≤ 0.02 / 0.05 rad |
| 指开度最大偏差 | 3.7 mm | 3.2 mm | ≤ 5 mm |
| 木块终点与源记录距离 | 0.07 mm | 0.12 mm | ≤ 10 mm |
| 放置误差 / 持握滑移 | 0.23 / 0.20 mm | 0.35 / 0.16 mm | ≤ 20 / 10 mm |

指开度偏差主要在闭合与张开的瞬间：回放在帧间线性插值指目标，源采集是阶跃目标。

## 3. 组成

| 模块 | 后端 | 实现 | 迁移对照 |
|---|---|---|---|
| Scene | `tabletop_home`（复用） | [scene/scene_tabletop_pickplace.py](../real2sim2real/scene/scene_tabletop_pickplace.py) | `TabletopScene` |
| Robot | `franka_tabletop`（复用） | [robot/robot_tabletop_pickplace.py](../real2sim2real/robot/robot_tabletop_pickplace.py) | `FrankaRobot` |
| Planner | `lerobot_replay` | [planner/planner_tabletop_replay.py](../real2sim2real/planner/planner_tabletop_replay.py) | `ReplayPlanner` |
| Job | `tabletop_replay` | [job/job_tabletop_replay.py](../real2sim2real/job/job_tabletop_replay.py)（继承 `TabletopPickPlaceJob`） | `TabletopJob` |
| 辅助 | — | [utils/tabletop_replay_utils.py](../real2sim2real/utils/tabletop_replay_utils.py) | `utils/recorder.load_lerobot_episode` |

迁移后的调整：动作包含指目标，不再从指开度推断开合；不支持源项目的 `control="state"`（下发录制的实测状态），
因为 `action` 才是录制时真正下发的目标；布局从源数据读取，不重新抽随机数。

| 脚本 | 用途 |
|---|---|
| [prepare_tabletop_replay_assets.py](../scripts/tabletop_replay/prepare_tabletop_replay_assets.py) | 复制与桌面抓放相同的资产到 `assets/TabletopReplay/`（976 MB，任务间不共用资产） |
| [collect_tabletop_replay.py](../scripts/tabletop_replay/collect_tabletop_replay.py) | 复用抓放正式采集器；单调预算时钟，debug 暂存，独立校验通过后发布 |
| [validate_tabletop_replay.py](../scripts/tabletop_replay/validate_tabletop_replay.py) | 桌面抓放的全部独立校验 + 从落盘文件重算回放一致性 |

## 4. 配置与运行

| 文件 | 源数据集 | 输出 |
|---|---|---|
| [franka_physx_home000.toml](../configs/tabletop_replay/franka_physx_home000.toml) | `outputs/tabletop_pickplace/franka_physx_home000`（1K / 30 fps） | `outputs/tabletop_replay/franka_physx_home000/` |
| [franka_physx_home000_debug.toml](../configs/tabletop_replay/franka_physx_home000_debug.toml) | `outputs/debug/tabletop_pickplace/franka_physx_home000_debug`（640 / 15 fps） | `outputs/debug/tabletop_replay/franka_physx_home000_debug/` |

```bash
./app/python.sh scripts/tabletop_replay/prepare_tabletop_replay_assets.py
./scripts/rsr collect run -c configs/tabletop_replay/franka_physx_home000_debug.toml
./app/python.sh scripts/tabletop_replay/collect_tabletop_replay.py
./app/python.sh scripts/tabletop_replay/validate_tabletop_replay.py outputs/tabletop_replay/franka_physx_home000
```

输出布局、数据列与桌面抓放相同（第 9 节）；`result.json` 的 `planner.source_dataset` 记录回放的源数据集。
正式配置 `multi_gpu=true`，两路固定 RenderProduct 分配到两张 GPU；只有单 GPU 时可用
`RSR_SIM_WORLD__multi_gpu=false` 运行上述采集器。调试仍可直接使用 CLI；正式采集器依赖与时钟适用范围见抓放文档第 6 节。

## 5. 局限

- 开环回放：不读传感器修正动作，PhysX 运行间的数值差异会累积。
- 回放和源采集必须用同一物理频率、同一资产与布局；布局不一致时直接报错。
- 未用官方 LeRobot SDK 加载；数据结构、视频完整解码、逐帧数据与物理源以及回放一致性已由独立脚本校验。

## 6. 失败与修复记录

原单卡回放在第 0 集约 114 帧组后从约 13 s/组降至约 80 s/组，GPU 利用率约 1%，
一个 `carb.tasking4` 线程占满一核。只读原生栈确认忙线程在 `libgpu.foundation.plugin.so`
`+0x193bf6` / `+0x19609b` 的 `memmove` 循环，与电池拆解任务已确认的墙钟回跳故障一致。
证据：[render_wait_native_root.txt](../outputs/debug/tabletop_replay/formal_logs/render_wait_native_root.txt)。

已用 SIGINT 停止原进程，保留未完成数据于
`outputs/debug/tabletop_replay/franka_physx_home000_interrupted_clock_20261115T002249Z/`，含 `run_note.json`。
Kit 快速退出返回了 0，但独立校验失败，因此不能仅按进程退出码认定采集成功。
新增采集器复用 [gpu_budget_clock.c](../scripts/battery_disassembly/gpu_budget_clock.c)，不修改 SDK 文件或系统时钟；
补丁在新进程日志记录 `active`，两张 GPU 均参与渲染。本轮日志实际捕获约 276 ms 的墙钟回跳，
仍完成两集并通过验收；共有 80 行回跳审计记录（同一次回跳可以产生多行，不代表 80 次独立事件）。
每次重跑从同一布局完整执行，不拼接旧帧；物理参数、画质和验收门限均未改变。

本轮状态：[status.json](../outputs/debug/tabletop_replay/monotonic_formal_20261115T0023Z/status.json)，
原始日志：[capture.log](../outputs/debug/tabletop_replay/monotonic_formal_20261115T0023Z/capture.log)。
两张 GPU 与单调预算时钟同时启用，正常采集约 7.0 s/组；不能把双卡的吞吐收益单独归因于时钟补丁。

NVIDIA 的 [RTX 资源管理说明](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html#gpu-resources-management)
解释了流式资源预算；本机具体时钟故障来自原生诊断和项目实测，不能由 `latest` 文档推断为所有版本共有的问题。
