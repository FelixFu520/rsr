# 分拣包裹：Pro-M6 双臂连续分拣软包与纸箱

更新日期：2026-11-16（环境核对）。本文合并原 `package.md`、`package01.md`、`sorting_package.md` 及夹爪错位、
1K 重渲染和 LeRobot 组装记录；逐次实验记录可通过 Git 历史查阅。

## 运行环境与依赖

在仓库根运行；唯一解释器为 `./app/python.sh`，`./scripts/rsr` 是同一解释器的 CLI 包装。
本机 Linux x86_64、Isaac Sim `6.2.0-rc.14+release.52805.02189a84.gl`、Python 3.12.14，
`app -> /home/fufa/isaacsim`，不建 `.venv`。公共依赖为 NumPy、Pillow、PyArrow、Dynaconf、Pydantic、
PyYAML、structlog、Typer，开发依赖 pytest / Ruff；版本、用途和安装路径见
[公共依赖表](../README.md#公共依赖与安装位置)。本任务无额外任务专用 pip 包。

| 组件 | 用途与本机版本 | 实际安装位置 |
|---|---|---|
| Newton | 自管模型与步进，1.6.1 | `/home/fufa/isaacsim/exts/isaacsim.pip.newton/pip_prebundle/newton/` |
| MuJoCo / MuJoCo Warp | 分拣配置的机械臂和纸箱使用原生 MuJoCo CPU；配套库均 3.12.0 | 同上 `pip_prebundle/mujoco/`、`mujoco_warp/` |
| Warp / CUDA VBD | 柔性衣物与袋膜，配套 Warp 1.17.0 | `/home/fufa/isaacsim/extscache/omni.warp.core-1.17.0+lx64/warp/` |
| Kit / USD / RTX | 资产装配、零物理步渲染、相机 | Isaac SDK 的 `exts/`、`extscache/` 和原生插件 |
| ffmpeg / ffprobe | H.264 视频编码、解码、帧数与采样率校验，6.1.1 | `/usr/bin/ffmpeg`、`/usr/bin/ffprobe` |

`configs/sorting/mujoco_vbd_sorting.toml` 选择 `physics_engine="newton"`、`newton_execution="direct"`、
`device="cuda:0"`。世界自动启用 `isaacsim.pip.newton`；刚体在 CPU，柔体在 CUDA，仍需要可用 NVIDIA GPU，
不是纯 CPU 仿真。Kit 的 RTX 采集也需要 GPU；`headless=true` 只是不显示窗口。
本机裸 Python 另能找到 site-packages 的 Warp 1.18.0，配套 Newton SDK 入口使用扩展版 1.17.0；
不要为了裸 `import newton` 安装另一套 pip Newton，也不要把 Warp 版本混用。
当前分拣采集入口不编译预算时钟补丁，不要求安装 C 编译器；同样不依赖 cuRobo、cuMotion、Kimodo 或官方 LeRobot 包。

安装和环境检查不运行完整分拣：

```bash
./app/python.sh -m pip install -e '.[dev]'
./app/python.sh -c 'import numpy, PIL, pyarrow'
./app/python.sh - <<'PY'
import subprocess, sys
from real2sim2real.utils.physic_solver_utils import isaac_newton_sdk_environment
subprocess.run(
    [sys.executable, "-c", "import newton, warp, mujoco, mujoco_warp; print(newton.__version__, warp.__version__, mujoco.__version__)"],
    env=isaac_newton_sdk_environment(), check=True,
)
PY
command -v ffmpeg ffprobe
./app/python.sh .agents/skills/rsr-task-development/scripts/check_task_configs.py configs/sorting
```

2026-11-16 专项 SDK 和公共包导入通过；本轮不重跑完整物理或正式渲染。
运行资产只来自 `assets/SortingPackage/`；模型可复用 `physic_solver` 的 Python 实现，但资产已独立复制，
不要求运行时访问 `assets/PhysicSolver/` 或源 `/home/fufa/Projects/package`。迁机需携带整个任务资产目录，
尤其保留 `Robot/` 与 `Scene/` 的相对层级。视频需要系统 `ffmpeg` 包及 `libx264` 编码能力。

## 1. 任务介绍

Pro-M6 双臂机器人（原装 Robotiq 2F-140 夹爪）在同一物理世界中依次分拣两件包裹：先柔性衣物袋，
后刚性纸箱。每件都经过远端入料、左臂抓取、中间平台放手、右臂重抓、出料带上方释放、
输送离带和自由落地。全过程同步录制四路 RGB（全景、头部、左腕、右腕）为 LeRobot v3。

### 1.1 单件流程与两件编排

```text
从入料抓取点上游 0.75 m 出现
  → 入料带以 0.12 m/s 输送，到位后暂停并静置
  → 左臂接近、闭合、抬升、悬停、转运
  → 中间平台下放、张爪、撤离、静置，确认全机器人脱离
  → 右臂根据当前包裹形状重新选抓点和抓取深度
  → 右臂闭合、提升、悬停、转运
  → 出料带上方张爪，包裹自由落到始终以 0.18 m/s 运行的带面
  → 接触摩擦输送，完整离带，再自由落地
  → 软包流程结束后生成纸箱，继续同一物理世界
```

纸箱阶段仍检查已经落地的软包，不能沿用前半程的成功标志忽略后续损伤。纸箱出料侧位根据软包
实际落地区域选择，避免后落地纸箱压坏软包；必要时采用绕夹持面法线倾斜 30° 的 TCP 朝向，
整段路径仍需通过原生关节限位检查。

### 1.2 约束与验收口径

| 需求 | 实现及边界 |
| --- | --- |
| 原装机器人与夹爪 | 不增加宽指垫、托边或隐藏夹块；碰撞胶垫与现有可视胶垫尺寸一致 |
| MuJoCo＋VBD | 原生 MuJoCo CPU 处理刚体与关节，CUDA VBD 处理柔体和刚软接触 |
| 不用附着作弊 | 不固定包裹粒子，不加抓取吸附约束，不在平台展平、瞬移或重置包裹 |
| 材料可调但固定 | 物体质量、指端摩擦显式记录；全 episode 固定材料，不在松手时临时降低摩擦 |
| 变形上限 | 衣物形状 RMS 上限经用户授权由 37.413 mm 提高到 45 mm；袋衣间隙 10 mm、穿透 1.5 mm 不放宽 |
| 单件节拍 | 清除工位旧包裹；软包完成后才生成纸箱，世界与机器人保持连续 |
| 物理与渲染分离 | 先连续求解并保存物理源，再零物理步回放渲染；渲染不能改变物理内容 |
| 资源 | 本项目同时运行的 Isaac Sim 不超过两个（暂停实例也计入），不停止其他项目任务；四路渲染期间 GPU 上不跑其它 Kit 任务（见 7.7） |

## 2. 当前结果

本节数据来自 2026-11-11 用整理后代码**从初态重新求解**并渲染的 [outputs/sorting](../outputs/sorting/)
（物理 16:36–17:55 UTC，四路渲染与组装 19:48–21:58 UTC）。

| 项目 | 当前结果 |
| --- | --- |
| 交付目录 | [outputs/sorting](../outputs/sorting/) |
| 数据集 | [lerobot_v3](../outputs/sorting/lerobot_v3/)，1 episode，`dataset_use=demonstration` |
| 四路 RGB | `overview`、`head`、`left_wrist`、`right_wrist`；各 1024×768、15 FPS、2018 帧，H.264 |
| 物理时间线 | 134.516667 s，8072 个 60 Hz 样本，1033088 个原生刚体子步记录；求解墙钟 4739 s |
| 物理运行 ID | `80447b5fd14440c3b33fb914d756b2e9`（[physics/](../outputs/sorting/physics/)） |
| 渲染 | 推进物理步数 0；渲染运行 ID `b229996ae57345809310adbf8099bdf2` |
| 结果 | `success=true`、`workflow_success=true`、`strict_quality_success=true` |
| 独立验收 | [stream_audit.json](../outputs/sorting/qa/stream_audit.json)：34 项全部通过，`failed_checks=[]` |
| 渲染同步 | 25 帧 × 4 台相机 = 100 项同实例逐像素对照全部一致（[qa/render_sync](../outputs/sorting/qa/render_sync/)） |
| 人工视觉复核 | 已生成 [四路同步预览](../outputs/sorting/qa/synchronized_four_view.mp4) 和关键帧拼图，[visual_review.json](../outputs/sorting/qa/visual_review.json) 为 `pending_visual_inspection`，尚未人工查看 |

| 指标 | 本次独立实测 | 门限／结论 |
| --- | --- | --- |
| 全时间线软包最大边长比 | 1.200493 | < 1.3 |
| 最小三角形面积比 | 0.578472 | > 0.03 |
| 持握衣物对齐 RMS 最大值（94 个样本） | 33.83 mm | < 45 mm |
| 持握袋衣间隙 q95 最大值 | 9.575 mm | < 10 mm，严格通过 |
| 原网格夹指穿透，软包／纸箱 | 0.948／0.025 mm | ≤ 1.5 mm |
| 地面穿透，软包／纸箱 | 0／0.067 mm | ≤ 1.5 mm |
| 原生 mimic 最大误差 | 0.00122 rad | < 0.03 rad |
| 实际入料位移，软包／纸箱 | 0.7517／0.7535 m | 到位 |
| 完整离带 → 落地，软包 | 63.50 s → 64.02 s | 先离带后落地 |
| 完整离带 → 落地，纸箱 | 130.75 s → 131.27 s | 先离带后落地 |
| 最终两件 AABB 间隔 | 147.5 mm | 落地分离 |
| 可视胶垫角点与物理碰撞体 | 最大 5.0×10⁻¹⁶ m，全部 2018 个视频时刻 | ≤ 2×10⁻⁶ m |

与 2026-11-11 04:27 UTC 的归档交付相比，这是一次独立的新求解：CUDA VBD 结果不逐位可复现，但所有门限
都通过；独立验收的袋衣间隙（9.575 vs 9.858 mm）和形变 RMS（33.8 vs 39.7 mm）比归档版略小。第一次渲染与另一个 Kit 实例并行时
在 overview 第 340 帧同步审计失败而中止，见 7.7；失败的渲染缓存保存在 `outputs/debug/sorting/`。
LeRobot 布局通过项目导出器及校验器验收，**尚未执行官方 LeRobot SDK 加载测试**。

## 3. 组成

### 3.1 模块

| 模块 | 后端名 | 实现 |
| --- | --- | --- |
| SimWorld | `isaacsim` | [simworld/isaacsim.py](../real2sim2real/simworld/isaacsim.py) |
| Scene | `sorting_package` | [scene/scene_sorting.py](../real2sim2real/scene/scene_sorting.py)：工位、两网格软包、纸箱、质量、支撑范围和任务落点（继承物理求解器任务的 `PackageScene`） |
| Robot | `pro_m6_newton` | [robot/robot_sorting.py](../real2sim2real/robot/robot_sorting.py)：Pro-M6 原生关节、夹爪、胶垫碰撞代理和机载相机 |
| Planner | `sorting_stream`（单件版 `sorting_package`） | [planner/planner_sorting.py](../real2sim2real/planner/planner_sorting.py) |
| Job | `sorting_package` | [job/job_sorting.py](../real2sim2real/job/job_sorting.py)：复用物理求解器任务的采集 Job，不改机器人站位 |
| 辅助函数 | — | [utils/sorting_utils.py](../real2sim2real/utils/sorting_utils.py)：形状 RMS、袋衣间隙、固定基座间距、夹指穿透等独立几何检查；袋膜包含检查在 [physic_solver_utils.py](../real2sim2real/utils/physic_solver_utils.py) |

`planner_sorting.py` 按依赖顺序包含：

1. `SortingArmIK`：与 Newton 相同 USD 关节的世界系 Pro-M6 IK。
2. `SortingRuntime`：MuJoCo/VBD 求解、角色材料、接触统计、原生子步记录、输送带接触和可视几何发布。
3. `SortingPlanner`（`sorting_package`）：单软包双臂中继、抓点和深度、连续动作、减夹反馈、在线质量门禁。
4. `SortingStreamRuntime`：纸箱原生刚体、输送接触、柔体休眠与唤醒。
5. `SortingStreamPlanner`（`sorting_stream`）：软包、纸箱连续编排，单件入料和落地验收。

仍遵循 `init_world → arrange_scene → place_robot → start_simulation → finally: close`；
任务字段保留在分拣后端，运行时不导入 `bak/`。

### 3.2 脚本

| 脚本 | 用途 |
| --- | --- |
| [run_sorting_showcase.sh](../scripts/sorting/run_sorting_showcase.sh) | 一体入口：连续物理 → 成功源检查 → 逐相机渲染 → 自动组装 LeRobot → 独立验收 |
| [check_sorting_package.py](../scripts/sorting/check_sorting_package.py) | 从初态求解并保存物理源，不采集 RGB |
| [render_sorting_package.py](../scripts/sorting/render_sorting_package.py) | 同源状态回放、分相机分批采图、缓存复用和自动 LeRobot 导出 |
| [validate_sorting_stream.py](../scripts/sorting/validate_sorting_stream.py) | 双包裹全时间线、几何、数据、视频与同步证据的独立验收 |
| [validate_sorting_render_sync.py](../scripts/sorting/validate_sorting_render_sync.py) | 同一渲染实例内连续采集／静态刷新的像素对照 |
| [review_sorting_stream.py](../scripts/sorting/review_sorting_stream.py) | 从正式视频提取关键帧，生成四路同步预览和待检查记录 |
| [prepare_sorting_stream_assets.py](../scripts/sorting/prepare_sorting_stream_assets.py) | 复制 V10 纸箱及贴图并记录哈希 |

`./scripts/rsr collect render-sorting` 是 `run_sorting_showcase.sh` 调用的 CLI 子命令：没有
`--source` 时先运行 `check_sorting_package.py` 生成 `physics/`，再运行渲染脚本。

### 3.3 配置

[configs/sorting/mujoco_vbd_sorting.toml](../configs/sorting/mujoco_vbd_sorting.toml) 是唯一的任务配置
（由原 `PhysicSolver_mujoco_vbd_sorting_package → _workflow → _collect → sorting_stream → _showcase`
五级继承链展平；展平前后的有效配置逐字段一致，并与已验收物理源的配置快照一致）。
[configs/sorting/render_profile.json](../configs/sorting/render_profile.json) 只覆盖回放渲染的显示设置：
1024×768、PathTracing、32 SPP、`rt_subframes=2`、4 次反弹、空间 OptiX 降噪；关闭自动曝光、DLSS、
时间降噪和相关光照缓存。

旧继承链、单软包实验配置（workflow / diagnostic / friction）和细分接触网格工具已移到
`bak/rsr-legacy/`，不再参与当前流程。

### 3.4 资产与求解模型

本任务的全部资产放在 `assets/SortingPackage/`（git 忽略，约 563 MB），只引用该目录内的文件，
整个目录可以单独打包发给别人：

| 资产 | 内容 |
| --- | --- |
| `Background/Factory/factory_002/` | 工厂背景 `Factory_Interior.usd` 及其材质贴图 |
| `Scene/sorting_station/` | 分拣工位：入料带、中间平台、出料带；`composed/` 是正式配置引用的组合场景 |
| `Robot/pro_m6_696/pro_m6_696.usda` | Pro-M6 + Robotiq 2F-140；保留原 USD 世界位姿、质量、惯量与力矩限值。通过相对路径引用 `../../Scene/sorting_station/composed/snapshot.usdc`，两个目录须保持同级 |
| `Package/v10_station/` | 软包：衣物与袋膜共 4950 个顶点、9584 个三角形，保留材料、UV 和内部被动弹簧 |
| `Package/v10_box02/` | 纸箱：源 USD 和贴图保留哈希，场景缩放 0.5，约 63×101×71 mm |
| `Package/v7_sorting/` | `SortingSceneConfig` 默认软包与单元测试使用的 V7 快照；`v7_sorting_refined/` 是其细分版本 |
| `Background/sorting_station/` | 早期工位灯光层，当前配置未引用 |

衣物质量 25 g、袋膜约 2.568 g，按原逐点质量比例缩放，不改源资产文件；纸箱 31.25 g。
软包快照用 `scripts/physic_solver/prepare_package_assets.py --source <V10 station_clothing> --target
assets/SortingPackage/Package/v10_station` 生成。

环境为 Isaac Sim 6.2、Newton 1.6.1、Warp 1.17.0。刚体使用原生 MuJoCo CPU，柔体使用 CUDA VBD，
不是 PhysX 接触后端，不能直接套用 PhysX 的传送带或驱动参数。刚体到柔体是单向耦合。

### 3.5 关键参数

| 参数组 | 当前值与含义 |
| --- | --- |
| 时间离散 | 60 Hz；每帧 32 个 VBD 子步，每步 28 次迭代；每 VBD 步 4 个 MuJoCo 子步，即 7680 Hz 原生刚体频率 |
| mimic | 原生等式约束；time constant 0.00027 s，impedance 0.9998；逐子步验收 |
| 机械臂／夹爪驱动 | stiffness/damping 分别为 6000/300、40/1 |
| 胶垫 | 碰撞代理与可视胶垫统一为 27×7.5×65 mm；`pad_width_m=0.027` |
| 任务落点 | 平台 `[0,-0.96]` m；出料基准 `[-0.55,-0.84]` m；左右抓点偏移为零，纸箱可使用避让侧位 |
| 软包夹爪 | open 0.28 rad，closed 0.64 rad，hold 0.675 rad |
| 纸箱夹爪 | closed 0.40 rad，hold 0.42 rad，指垫底边 15 mm |
| 软包抓取深度 | 入料指垫底边 20 mm；重抓最低 15 mm，绝对下压 25 mm、相对比例 0.3 |
| 动作 | 提升 4 s、转运 3 s 五次曲线、张爪 4 s；提升 TCP 高度 0.28 m，平台放置 0.13 m，出料释放 0.20 m |
| 减夹反馈 | 目标边长比 1.22，最大放宽 0.035 rad，放宽速率 0.1 rad/s，恢复速率 0.001 rad/s |
| 接触 | 接触／普通 shape 刚度 40000；SDF resolution 128；self-contact gap 1.6 mm |
| 地面 | 独立 shape 刚度 400000；仅地面 `floor_contact_margin_m=0.004` m |
| 质量门限 | RMS 45 mm，袋衣间隙 q95 10 mm，穿透 1.5 mm，边长比 1.3，最小三角形面积比 0.03 |

这些 TCP 高度不等于指垫最低点或包裹质心高度；夹爪角度也不是以米表示的开口。修改碰撞代理后
应重新用实际 FK 核对夹缝，不把旧标定表当通用换算式。

接触摩擦按角色分别设置。本机实现的有效摩擦为 `sqrt(mu_particle × mu_shape)`：粒子 1.4、夹指 shape
约 2.85714，对应有效指端摩擦 2.0；输送带 shape 0.5，有效摩擦约 0.83666；平台 shape 0.1，有效摩擦约
0.37417。运行时读回实际材料数组并记录 shape 映射。

## 4. 运行

所有命令在仓库根目录执行，只使用 Isaac Sim 的 `./app/python.sh`（`./scripts/rsr` 是 CLI 包装）。

### 4.1 从初态完整采集

```bash
./scripts/sorting/run_sorting_showcase.sh outputs/sorting
./app/python.sh scripts/sorting/review_sorting_stream.py outputs/sorting   # 生成人工检查用预览
```

失败返回非零并保存 `logs/pipeline_*.log`。渲染阶段不要同时运行其它 GPU 任务（见 7.7）。已有 `physics/` 时，脚本会检查成功、完整、连续、严格质量
及胶垫几何标识，拒绝恢复诊断源；已有根 `result.json` 时跳过生成、重新验收。正式数据已存在时拒绝覆盖，
要换新目录。`review_sorting_stream.py` 把 `visual_review.json` 标为 `pending_visual_inspection`，实际查看后
再记录检查范围。

### 4.2 复用已验收物理，复制到新目录再渲染

`--source` 只指定回放源，不复制 `physics/`。交付自包含目录时先复制并核对哈希：

```bash
./app/python.sh - <<'PY'
from pathlib import Path
import hashlib, json, shutil

source = Path("/path/to/accepted/physics").resolve()
output = Path("outputs/sorting").resolve()
output.mkdir(parents=True, exist_ok=False)
shutil.copytree(source, output / "physics")
files = {}
for path in sorted(source.rglob("*")):
    if path.is_file():
        relative = path.relative_to(source)
        original = hashlib.sha256(path.read_bytes()).hexdigest()
        copied = hashlib.sha256((output / "physics" / relative).read_bytes()).hexdigest()
        assert original == copied, str(relative)
        files[str(relative)] = {"sha256": original, "bytes": path.stat().st_size}
(output / "qa").mkdir()
(output / "qa/physics_copy_verification.json").write_text(json.dumps(
    {"source": str(source), "destination": str(output / "physics"),
     "identical": True, "physics_rerun": False, "files": files}, indent=2) + "\n")
PY
./scripts/sorting/run_sorting_showcase.sh outputs/sorting   # 检测到 physics/ 后只渲染与验收
```

只改显示设置使用 `--render-profile`，保持原物理配置与状态。改变碰撞几何、质量、材料或动力学控制时
必须重新连续求解，不能伪装成重渲染。新求解在 CUDA VBD 上可能产生形态差异；要求重现同一物理内容时
使用已验收缓存。

### 4.3 分阶段调试（结果放 `outputs/debug/sorting/`）

```bash
# 仅物理；物理阶段失败时停止，不继续渲染。
./app/python.sh scripts/sorting/check_sorting_package.py \
  --config configs/sorting/mujoco_vbd_sorting.toml --output outputs/debug/sorting/physics

# 只回放该源；包含分批采图和最终数据组装。失败源需显式加 --allow-failure，只能作诊断画面。
./app/python.sh scripts/sorting/render_sorting_package.py outputs/debug/sorting/physics \
  --output outputs/debug/sorting --render-profile configs/sorting/render_profile.json --fps 15

./app/python.sh scripts/sorting/validate_sorting_stream.py outputs/debug/sorting

# 同一渲染实例内的像素同步复查
./app/python.sh scripts/sorting/validate_sorting_render_sync.py outputs/sorting/physics \
  --output outputs/debug/sorting/render_sync_check --render-profile configs/sorting/render_profile.json
```

局部恢复须标记 `diagnostic_restart=true`、`accepted_demonstration=false`；恢复粒子和关节状态不等于
恢复完整求解器内部状态，局部成功不能与前半段拼接交付。

### 4.4 渲染分批、缓存与续渲

默认按 `overview → head → left_wrist → right_wrist` 渲染，每批最多 600 个新帧后换新 Kit 进程；
生成顺序不改变共享的物理采样时刻，不是每台相机重新仿真一次。`.render_cache/manifest.json` 版本 3，
绑定源配置、结果及 NPZ 哈希、最终光学／渲染设置、FPS、采样索引和
`capture_sync_contract="flush_scene_before_pt_counter_v1"`。`capture_workers/*.json` 记录每批运行标识、
累计帧数和完整性，父进程拒绝无进展批次。相同源、相同设置下重跑原命令会复用有效 PNG，只补缺失、
损坏或审计不匹配的帧；源或设置变化时拒绝复用。不要手工改 manifest。

四路齐全后**自动组装** LeRobot：检查四路图像 → 逐帧计算通道统计 → 加入共享 state/action →
编码四路 H.264 → 写 Parquet、任务与 episode 元数据 → 把临时目录重命名为 `lerobot_v3`。
1K 的 8072 张 PNG 统计约需十几分钟，日志可能长时间停在最后一条采图进度；`.lerobot-v3-*/` 临时目录
存在不代表程序失败，也不能证明进程仍在运行，应同时看进程、日志与最终产物。组装进程中途退出时，
确认没有仍在运行的组装任务后重跑同一命令即可从完整缓存继续，不重跑物理或渲染。

## 5. 输出与数据格式

```text
outputs/sorting/
├── physics/          # configuration.json、result.json、rigid_result.json、particles.npz、stream.npz、
│                     # trajectory.json、stream_trajectory.json
├── .render_cache/    # manifest.json + frames/{camera}/frame_*.png，支持续渲
├── capture_workers/  # 各相机批次结果与完整性标记
├── diagnostics/      # 渲染运行的配置、源副本与结果
├── logs/             # 物理、采图、组装、验收日志
├── result.json       # 物理结果、来源哈希、渲染与同步证据
├── render_progress.json
├── lerobot_v3/
│   ├── data/chunk-000/file-000.parquet
│   ├── videos/observation.images.{overview,head,left_wrist,right_wrist}/chunk-000/file-000.mp4
│   └── meta/info.json, stats.json, tasks.parquet, episodes/chunk-000/file-000.parquet, task_outcome.json
└── qa/               # stream_audit.json、render_sync/、四路预览、关键帧拼图、visual_review.json
```

- `observation.state`：16 个原生坐标的实际角度（左臂 7、右臂 7、左右夹爪主关节各 1）；`action` 是同顺序
  的位置目标，单位均为 rad，名称见 `meta/info.json`。
- 四路视频、状态、动作使用相同源索引，15 FPS，时间戳 `frame_index / 15`；H.264、yuv420p、无音轨。
- 数据集记录 `observation_generation=rendered_from_recorded_physics` 及源轨迹 SHA256；验收以原始物理结果为准。
- 失败轨迹只能作为 `diagnostic`，只有全部检查通过才标记为 `demonstration`。

## 6. 验收

物理阶段（`SortingStreamPlanner.result`）：两件包裹的两次提起、整个转运期双侧接触及网格离台、平台/出带
放手（无机器人接触、完整受支撑）、完整包裹支撑范围、标签朝上、真实出带位移、离带后落地；质量、有限状态、
三角面塌陷、边长伸长、台面/指垫/地面穿透、子步接触容量、袋内包含关系、原生 mimic 子步残差，以及转运期
衣物形状 RMS 和袋衣间隙。结果区分 `workflow_success` 与 `strict_quality_success`。

独立验收（`validate_sorting_stream.py`）：重新读取物理源，覆盖全时间线的软包拉伸、面积比、持握形状与间隙、
夹指/地面/支撑面穿透、联动约束、入料位移、完整离带与落地时序、两件落地分离、可视胶垫角点与物理几何一致，
并完整解码四路视频、逐帧核对帧数、时间索引与原始关节值，以及渲染同步证据。

几何检查只覆盖保存的物理帧，不是连续时间无穿透的证明；在线接触穿透与独立几何穿透的采样、算法不同，
应分别引用。LeRobot 布局通过项目导出器及校验器验收，尚未执行官方 LeRobot SDK 加载测试。

## 7. 遇到的问题与解决方案

### 7.1 抓不住、滑脱、变形与释放拉扯

早期 100 g、50 g 衣物及原袋膜质量下，原生夹爪曾空抓、提升后滑脱或释放时拉伸超限。减轻到衣物 25 g、
袋膜约 2.568 g 并提高指端有效摩擦有助于保持，但不能单独解释全流程成功。高摩擦也可能在张爪时拖住袋膜；
单独降摩擦、降低放置高度、张爪时下降 64 mm、空中张爪等局部候选都失败过。加宽指垫与结构约束冲突，未保留。

最终处理是原生 FK 抓取深度、合适抓点、连续路径、分角色摩擦和真实支撑／释放判定的组合。阶段之间延续
上一段命令目标，避免反复叠加实测 TCP 跟踪误差；撤离使用经过检查的笛卡尔路径，避免张开的夹指扫过包裹。

右臂闭合末段曾出现边长比 1.300869（超过 1.3）：`strain_limited_grip` 原先未覆盖闭合阶段。修复把反馈接入
闭合，以放宽后的角度补偿指垫底边高度，并把闭合末帧实际下发目标延续到提升阶段。没有修改拉伸门限。

### 7.2 固定机器人基座漏计，造成“已放手”的假象

固定基座及相机碰撞体的 `body_id=-1` 与世界静态体相同，旧逻辑只统计动态 body，漏掉了机器人固定根，
还可能把它当台面，出现夹爪已张开、包裹仍被基座拱起的情况。修复按导入机器人的 shape ID 范围统计全
机器人接触，把固定根分类为 `robot_root`，并用实际世界凸几何、全部顶点和面中心独立检查间距。平台目标
从 `[0,-0.93]` 前移到 `[0,-0.96]`，提升高度由 0.33 m 改为 0.28 m，避开右臂高位转运的关节限位；
没有移动机器人或台面，也没有删除基座碰撞体。

### 7.3 TCP、指垫底边和右臂重抓点混淆

原生夹指含金属凸体与胶垫，闭合时最低点随角度变化，只用 TCP 或金属指尖推断抓深会失准。现在用原生 FK
计算指垫底边；右臂在当前抓点约 ±30 mm 区域读取衣物表面 q90 高度，结合绝对下压、相对比例和最低底边
限制选高。修基座后包裹更平展，旧拱起状态筛出的偏心抓点仍导致拉伸；联合筛查平台、出料落点和全段 IK 后，
中心抓取获得连续可达路径。

### 7.4 60 Hz 记录漏掉原生 mimic 子步尖峰

USD 中有 PhysX mimic 配置不代表 MuJoCo 已建立正确约束。早期 60 Hz 记录约 0.0125 rad，逐子步却发现
0.03098 rad 峰值（门限 0.03）。修复使用原生 MuJoCo 等式约束、CPU 双精度状态及每物理帧 128 个刚体子步，
保存全部子步残差。降低刚体／柔体精度很快触发 mimic 超限或空抓，不能用粗步长替代正式精度。诊断写盘曾误读
`planner.config.newton_substeps`，已改为从运行时配置读取。

### 7.5 连续输送、纸箱落地和软包后半程损伤

移动出料带上首次触带落点与落稳位置不同，分别判定，保留 80 mm 落点门限。输送由真实接触的切向相对运动
实现，不直接平移粒子。纸箱采用独立接触参数和 PGS；软包静止且远离纸箱后才休眠，接近则唤醒。

纸箱曾落到软包上导致拉伸和地面穿透，因此增加落地区域避让及全时间线柔体检查。软包曾以约 6 m/s 落地，
单 VBD 子步行程超过 3 mm，约 1 mm 接触包络来不及捕获；只提高地面刚度到 2000000 仍失败。现在仅给地面
增加 4 mm shape margin 提前建立接触，夹指、平台和带面 margin 不变，穿透门限仍为 1.5 mm（静置软包与
可视地面可能相隔数毫米）。离带检查也改为先累计确认完整离带、再检查后续落地高度，不要求同一帧满足。

### 7.6 袋衣间隙、形变和夹爪错位是不同问题

`bag_gap_q95_m` 是袋膜到衣物表面的距离分位数，不是夹爪到包裹的距离；`aligned_garment_shape_rms_m` 衡量
对齐后的衣物形变；夹指穿透是另一项几何检查。早期“先完成流程”的配置把间隙设为 advisory，记录
`strict_quality_success=false`，不能写成严格通过。旧 1K 运行的 10.030 mm 是真实严格失败。

### 7.7 夹爪画面整体错位：根因是渲染同步

用户在旧 1K 腕部视频和 head 第 340 帧发现夹爪整体偏离抓取点。排查区分出两个独立问题：

1. **碰撞代理尺寸不一致**：原碰撞胶垫 30×7.5×70 mm，可视胶垫 27×7.5×65 mm。代理改为匹配可视几何，
   保留质量、惯量及关节；这改变了物理接触，因此重新生成了连续物理源。
2. **刚体与柔体进入渲染器的时序不一致**：连续采集过早递增 `/rtx/externalFrameCounter`，刚体变换尚未
   完成 Hydra 更新就开始 Path Tracing 累积，出现旧夹爪姿态与新袋体网格同帧。

现在每次发布样本后先等待两次**实际渲染完成事件**，再递增外部累积计数；新相机和每个续渲批次首帧至少
等待 8 次实际渲染事件。`world.update()` 调用次数不等于 GPU 完成帧数。修复后同实例静态对照像素误差为 0；
缓存升级到 v3，拒绝旧 PNG。不同进程可能存在采样或光照差异，不能把跨进程整图差异直接认定为姿态错位。

**2026-11-11 重新采集时又发现：GPU 被其它进程占用时，这个屏障仍会漏掉刚体位姿。** 渲染与另一个 Kit 实例
（物理求解器批量采集）同时运行时，生产采集在 overview 第 340 帧的同实例逐像素对照失败：差异集中在左夹爪和
包裹附近（x 524–579、y 205–279 px，最大 107）。用调试探针（`outputs/debug/sorting/capture_barrier_probe.py`）
在 head 相机的运动帧上对比不同屏障：

| 条件 | 结果 |
| --- | --- |
| 另一 Kit 实例同时占用 GPU | 两次排空：12 帧中 5 帧与静态参考不一致；把排空增加到 4、8、16 次实际渲染事件仍有失败 |
| GPU 只有本渲染进程 | 两次排空到十六次排空：10 帧 × 4 种排空全部逐像素一致 |
| 每帧完整刷新（与静态参考同一流程） | 一致，但约 7 s/帧，四路 8072 帧需约 15 h，未采用 |

原因：`/rtx/pathtracing/totalSpp=32`，计数器重置后的第一帧就累积满样本，之后不再更新；若刚体位姿在
GPU 繁忙时晚于这一帧到达渲染器，旧位姿会留在整帧中。只在采集期间排空的帧几乎不消耗渲染时间，
增加帧数并不能等到晚到的位姿。

处理：正式渲染期间本项目不同时运行其它 Kit 实例（其它项目的 GPU 任务也应暂停）；同实例像素审计从每台相机
5 帧扩大到 25 帧（第 340 帧、四个转运中点，以及第 50、150、…、1950 帧，避开每批第一帧），四路共 100 项，
任一不一致即中止采集。`render_sync_audit_frames()` 由渲染与独立验收共用。这仍是抽样证据，不是逐帧证明。

### 7.8 长时任务、缓存和验收性能

长时采图曾在单进程约 3094 帧、左腕约 1855 帧处停滞，未发现内存、显存或 I/O 耗尽。采用每批最多 600 个
新帧的独立进程，并用运行标识防止误用旧结果。`physic_solver_utils.py` 的表面距离用 NumPy AABB 树和保守
下界剪枝，叶节点仍算精确点到三角形距离，单帧从约 6.45 s 降到 0.28～0.46 s，没有减少探针或放宽门限。

## 8. 历史版本

| 阶段 | 结果与现在的关系 |
| --- | --- |
| V7／V8／V9／V10 参考 | 部分使用运动学宽指垫，不能等同原生 Pro-M6 能力 |
| 早期原生 100 g／50 g 候选 | 完成过左臂抬升，但严格间隙、释放拉伸或形变失败 |
| `sorting_package` 单软包版 | 56.8 s、852 帧、640×480；流程通过，袋衣间隙约 11.241 mm，严格项失败 |
| `sorting_stream` 双包裹版 | 134.5 s、2018 帧、640×480；间隙 advisory |
| `sorting_stream_showcase_1k_final` | 严格间隙 10.030 mm 失败，且有夹爪渲染错位 |
| `sorting_stream_showcase_aligned` | 修正胶垫、提高 RMS 上限后得到已接受物理源；旧视频仍错位 |
| `sorting_package_showcase`（2026-11-11 04:27 UTC） | 复制上述物理源，修复渲染同步后重渲染 1K；34 项独立验收通过。归档于 `/home/fufa/Projects/output/outputs-分拣包裹/sorting_package_showcase` |

## 9. 排查顺序

1. **先判断是物理失败还是画面失败**：查 `physics/result.json` 的失败原因、连续性和严格检查；物理回调提前
   停止且 `state.done=false` 不能写成完成。
2. **冻结问题时刻和物理源**：保存配置、源哈希、相机、视频帧号和源索引；纯渲染问题使用同源状态。
3. **按直接证据定位**：接触角色／基座 → 实际胶垫几何与 FK → 抓点、深度和路径 → 原生子步约束 →
   全程柔体质量。相机矩阵正确但画面仍错位时，检查 Hydra／GPU 完成事件与累积时序。
4. **先做低成本对照**：小分辨率、同实例静态参考或局部诊断；局部恢复只支持局部结论。
5. **恢复时核对进程和缓存契约**：不重复启动仍活跃的工作进程，不复用旧同步版本 PNG。
6. **交付前闭环**：成功连续源、零物理步重渲染、实际导出像素对照、四路解码、状态／动作对齐、
   严格几何验收和明确范围的选帧视觉检查都完成后，再标记交付完成。

## 10. 局限

- 刚体到柔体单向耦合，包裹不向机器人反馈完整反作用力，不能据此认证实物负载或限力能力。
- 地面 4 mm 接触皮层不是零厚度接触。
- 自动像素对照只覆盖 100 个指定的相机×帧组合；人工视觉复核是选帧检查。
- 每次只采集一条固定初始状态的 episode。
