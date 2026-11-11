# 包裹分拣：实现、完整流程、问题排查与交付

更新日期：2026-11-10（America/New_York）。最新交付完成时间为 2026-11-11 04:27 UTC。

本文合并原 `package.md`、`package01.md` 及后续夹爪错位、1K 重渲染和 LeRobot 组装记录。
当前结论以 `outputs/sorting_package_showcase` 内的配置、物理源和验收报告为准；历史实验单独列出，
不再把当时的“运行中”“尚未成功”或旧版“最终交付”作为当前状态。

## 1. 当前交付

已完成一条同一世界内连续求解的双包裹分拣 episode：先柔性衣物袋，后刚性纸箱。
两件均经过远端入料、左臂抓取、平台放手、右臂重抓、出料带上方释放、输送离带和自由落地。
当前四路 1K 视频已经重新渲染并自动组装为 LeRobot v3；严格袋衣间隙和独立验收均通过。

| 项目 | 当前结果 |
| --- | --- |
| 交付目录 | [outputs/sorting_package_showcase](../outputs/sorting_package_showcase/) |
| 数据集 | [lerobot_v3](../outputs/sorting_package_showcase/lerobot_v3/)，1 episode |
| 四路 RGB | `overview`、`head`、`left_wrist`、`right_wrist`；各 1024×768、15 FPS、2018 帧 |
| 物理时间线 | 134.516667 s，8072 个 60 Hz 样本，1033088 个原生刚体子步记录 |
| 物理运行 ID | `4081d9e3c332493a93a9cddd4653fa26` |
| 物理源 | [physics](../outputs/sorting_package_showcase/physics/)，7 个文件原样复制；交付后再次核对 SHA-256 一致 |
| 本次重渲染 | 推进物理步数为 0，没有重新求解或拼接物理轨迹 |
| 结果 | `success=true`、`workflow_success=true`、`strict_quality_success=true` |
| 独立验收 | 34 项检查全部通过，`failed_checks=[]` |
| 实际图像同步 | 5 个关键时刻 × 4 台相机，共 20 项同实例像素对照通过 |

直接查看：[四路同步预览](../outputs/sorting_package_showcase/qa/synchronized_four_view.mp4)、
[左腕视频](../outputs/sorting_package_showcase/lerobot_v3/videos/observation.images.left_wrist/chunk-000/file-000.mp4)、
[右腕视频](../outputs/sorting_package_showcase/lerobot_v3/videos/observation.images.right_wrist/chunk-000/file-000.mp4)、
[独立验收报告](../outputs/sorting_package_showcase/qa/stream_audit.json)、
[物理复制校验](../outputs/sorting_package_showcase/qa/physics_copy_verification.json)。

本次源来自旧 `sorting_stream_showcase_aligned/physics`，复制后保存于当前目录。
旧源目录后来已被清理；当前副本仍与复制时记录的原始哈希一致。复现应使用当前副本，
不要继续依赖已删除的 `sorting_stream_showcase_aligned` 或 `sorting_stream_showcase_1k_final` 路径。

## 2. 需求与完整业务流程

### 2.1 约束和验收口径的演变

| 需求 | 实现及边界 |
| --- | --- |
| 恢复原 Pro-M6 双臂分拣 | 从旧工位迁移流程，通过现有 Scene、Robot、Planner、Job 后端接入 |
| 沿用 MuJoCo＋VBD | 原生 MuJoCo CPU 处理刚体与关节，CUDA VBD 处理柔体和刚软接触 |
| 保留机器人与夹爪结构 | 不增加宽指垫、托边或隐藏夹块；后续修正碰撞代理与现有可视胶垫的尺寸不一致 |
| 可调整物体质量、指端摩擦 | 参数显式记录；全 episode 固定材料，不在松手时临时降低摩擦 |
| 不用附着作弊 | 不固定包裹粒子，不加抓取吸附约束，不在平台展平、瞬移或重置包裹 |
| 先完成流程，再处理贴合 | 早期采用 `workflow_gap_advisory`，保留严格间隙失败；当前 showcase 已恢复严格袋衣间隙要求 |
| 可以提高变形限制 | 当前衣物形状 RMS 上限由约 37.413 mm 提高到 45 mm；间隙 10 mm、穿透 1.5 mm 未放宽 |
| 清除堆积、单件节拍 | 清除工位旧包裹；柔性包裹完成后才生成纸箱，世界与机器人保持连续 |
| 保持已有物理，重新渲染到新目录 | 将已接受源逐文件复制、核对哈希，在 `sorting_package_showcase` 输出 1K 视频 |
| 降低排查成本 | 先用 512×384 做同步对照，再验证 1024×768，正式交付保持 1K |
| 资源约束 | 本项目同时运行的 Isaac Sim 不超过两个，暂停实例也计入；不停止其他项目任务 |

### 2.2 单件流程与两件连续编排

```text
从入料抓取点上游 0.75 m 出现
  → 入料带以 0.12 m/s 输送，到位后暂停并静置
  → 左臂接近、闭合、抬升、悬停、转运
  → 中间平台下放、张爪、撤离、静置，确认全机器人脱离
  → 右臂根据当前包裹形状重新选抓点和抓取深度
  → 右臂闭合、提升、悬停、转运
  → 出料带上方张爪，包裹自由落到始终以 0.18 m/s 运行的带面
  → 接触摩擦输送，完整离带，再自由落地
  → 柔性流程结束后生成纸箱，继续同一物理世界
```

纸箱阶段仍检查已经落地的柔性包裹，不能沿用前半程的成功标志忽略后续损伤。
纸箱出料侧位根据柔性包裹实际落地区域选择，避免后落地纸箱压坏软包。
必要时采用绕夹持面法线倾斜 30° 的 TCP 朝向，整段路径仍需通过原生关节限位检查。

## 3. 实现、资产和当前参数

### 3.1 模块职责与入口

| 文件 | 职责 |
| --- | --- |
| [scene_sorting.py](../real2sim2real/scene/scene_sorting.py) | 工位、两网格包裹、纸箱、质量、支撑范围和任务落点 |
| [robot_sorting.py](../real2sim2real/robot/robot_sorting.py) | Pro-M6 原生关节、夹爪、胶垫碰撞代理和四路相机中的机器人相机 |
| [planner_sorting.py](../real2sim2real/planner/planner_sorting.py) | 单包裹双臂中继、抓点和深度、连续动作、在线质量门禁 |
| [sorting_runtime.py](../real2sim2real/planner/sorting_runtime.py) | MuJoCo/VBD 求解、角色材料、接触统计、原生子步和可视几何发布 |
| [planner_sorting_stream.py](../real2sim2real/planner/planner_sorting_stream.py) | 软包、纸箱连续编排、单件入料和落地验收 |
| [sorting_stream_runtime.py](../real2sim2real/planner/sorting_stream_runtime.py) | 纸箱原生刚体、输送接触、柔体休眠与唤醒 |
| [job_sorting.py](../real2sim2real/job/job_sorting.py) | 复用项目生命周期与采集编排 |
| [package_geometry.py](../real2sim2real/utils/package_geometry.py)、[sorting_quality.py](../real2sim2real/utils/sorting_quality.py) | 表面距离、形状、袋衣间隙、夹指穿透等独立几何检查 |
| [check_sorting_package.py](../scripts/check_sorting_package.py) | 从初态求解并保存物理源，不采集 RGB |
| [render_sorting_package.py](../scripts/render_sorting_package.py) | 同源状态回放、分相机分批采图、缓存复用和自动 LeRobot 导出 |
| [lerobot_v3.py](../real2sim2real/utils/lerobot_v3.py) | 渲染同步、图像统计、视频编码、Parquet 与元数据发布 |
| [validate_sorting_render_sync.py](../scripts/validate_sorting_render_sync.py) | 同一渲染实例内的连续采集／静态刷新像素对照 |
| [validate_sorting_stream.py](../scripts/validate_sorting_stream.py) | 双包裹全时间线、几何、数据、视频与同步证据的独立验收 |
| [review_sorting_stream.py](../scripts/review_sorting_stream.py) | 从正式视频提取关键帧，生成四路同步预览与待检查记录 |
| [run_sorting_showcase.sh](../scripts/run_sorting_showcase.sh) | 连续物理、逐相机渲染／组装、独立验收的一体入口 |

Scene 和 Job 使用 `sorting_package`，Robot 使用 `pro_m6_newton`，当前 Planner 使用
`sorting_stream`；单软包 Planner 使用 `sorting_package`。后端通过现有 entry points 注册。
仍遵循 `init_world → arrange_scene → place_robot → start_simulation → finally: close`，
任务字段保留在分拣后端，运行时不导入 `bak/`。

### 3.2 资产与求解模型

早期参考来自 `bak/rsr-bak` 及 `/home/fufa/Projects/package/output/simulation_v7` 等版本。
参考案例包含运动学夹爪和自定义宽指垫，Franka 案例也使用不同机器人；其成功不能证明
迁移后的 Pro-M6 原生动力学抓取成功。

当前柔性资产为 `assets/PhysicSolver/Package/v10_station/parcel_two_mesh.usd`：
衣物和袋膜共 4950 个顶点、9584 个三角形，保留材料、UV 和内部被动弹簧。
衣物质量 25 g，袋膜约 2.568 g，按原逐点质量比例缩放，不改源资产文件。
纸箱为 `assets/PhysicSolver/Package/v10_box02/source.usd`，源 USD 和贴图保留哈希，
场景缩放 0.5 倍，约 63×101×71 mm，质量 31.25 g。

本轮环境记录为 Isaac Sim 6.2、Newton 1.6.1、Warp 1.17.0，刚体使用原生 MuJoCo CPU，
柔体使用 CUDA VBD。这里不是 PhysX 接触后端；不能直接套用 PhysX 的传送带或驱动参数。
刚体到柔体目前是单向耦合，包裹不向机器人反馈完整反作用力，不能据此认证实物负载或限力能力。

### 3.3 当前 showcase 的关键有效参数

以 [物理配置快照](../outputs/sorting_package_showcase/physics/configuration.json) 为本次交付依据。
可编辑入口为 [showcase 配置](../configs/PhysicSolver_mujoco_vbd_sorting_stream_showcase.toml)，
它继承 [stream 配置](../configs/PhysicSolver_mujoco_vbd_sorting_stream.toml) 和单包裹 collect 配置。

| 参数组 | 当前值与含义 |
| --- | --- |
| 时间离散 | 60 Hz；每帧 32 个 VBD 子步，每步 28 次迭代；每 VBD 步 4 个 MuJoCo 子步，即 7680 Hz 原生刚体频率 |
| mimic | 原生等式约束；time constant 0.00027 s，impedance 0.9998；逐子步验收 |
| 机械臂／夹爪驱动 | stiffness/damping 分别为 6000/300、40/1；保留原质量、惯量与力矩限值 |
| 胶垫 | 碰撞代理与可视胶垫统一为 27×7.5×65 mm；`pad_width_m=0.027` |
| 任务落点 | 平台 `[0,-0.96]` m；出料基准 `[-0.55,-0.84]` m；左右抓点偏移为零，纸箱可使用避让侧位 |
| 软包夹爪 | open 0.28 rad，closed 0.64 rad，hold 0.675 rad |
| 纸箱夹爪 | closed 0.40 rad，hold 0.42 rad，指垫底边 15 mm |
| 软包抓取深度 | 入料指垫底边 20 mm；重抓最低 15 mm，绝对下压 25 mm、相对比例 0.3 |
| 动作 | 提升 4 s、转运 3 s 五次曲线、张爪 4 s；提升 TCP 高度 0.28 m，平台放置 0.13 m，出料释放 0.20 m；释放附加下降 0 |
| 减夹反馈 | 开启；目标边长比 1.22，最大放宽 0.035 rad，放宽速率 0.1 rad/s，恢复速率 0.001 rad/s |
| 接触 | 接触／普通 shape 刚度 40000；SDF resolution 128；self-contact gap 1.6 mm |
| 地面 | 独立 shape 刚度 400000；仅地面 `floor_contact_margin_m=0.004` m |
| 当前质量门限 | RMS 45 mm，袋衣间隙 q95 10 mm，穿透 1.5 mm，边长比 1.3，最小三角形面积比 0.03 |

这些 TCP 高度不等于指垫最低点或包裹质心高度；夹爪角度也不是以米表示的开口。
旧原生几何对照中 0.64/0.65/0.675 rad 对应的约 11.816/9.819/4.823 mm 夹缝
只属于当时的几何标定，修改碰撞代理后应重新用实际 FK 核对，不把旧表当通用换算式。

接触摩擦按角色分别设置。本机实现的有效摩擦为 `sqrt(mu_particle × mu_shape)`：
粒子 1.4、夹指 shape 约 2.85714，对应有效指端摩擦 2.0；输送带 shape 0.5，
有效摩擦约 0.83666；平台 shape 0.1，有效摩擦约 0.37417。
运行时读回实际材料数组并记录 shape 映射；不能只看配置值判断参数是否生效。

配置继承涉及向量或相机列表时，按现有示例用 `@reset` 表达替换，避免旧值与新值拼接。
调参和复现都应检查合并后的有效配置。冻结旧实验配置后再改公共父配置，避免改变历史含义。

## 4. 从物理到 LeRobot 的运行流程

所有命令在仓库根目录执行。Python 只使用 Isaac Sim 的 `./app/python.sh`；
`./scripts/rsr` 是项目入口包装。不要用 `uv sync` 生成另一套仿真运行环境。

### 4.1 从初态完整重算

```bash
./scripts/run_sorting_showcase.sh outputs/sorting_package_showcase_new
```

依次执行：连续物理 → 成功源检查 → 分批渲染四路 → 自动组装 LeRobot → 独立流式验收。
失败返回非零，并保存 `logs/pipeline_*.log`。使用新目录，避免覆盖当前交付。
脚本省略参数时使用带纽约时间的 `sorting_stream_showcase_1k_*` 目录名；需要指定名称时显式传参。

已有 `physics/` 时，完整脚本会检查成功、完整、连续、严格质量及新胶垫几何标识，
拒绝恢复诊断源。已有根 `result.json` 时会跳过生成并重新验收，因此残留结果文件必须与
当前数据集一起核实，不能仅凭该文件存在认定交付完成。

该脚本不自动生成人工检查用的四路拼图。验收通过后执行：

```bash
./app/python.sh scripts/review_sorting_stream.py outputs/sorting_package_showcase_new
```

这会生成预览并把 `visual_review.json` 标为 `pending_visual_inspection`；实际查看后再记录检查范围。

### 4.2 保持现有物理，复制到新目录再渲染

`--source` 只指定回放源，不自动复制整个 `physics/`。需要交付自包含目录时先复制并核对哈希：

```bash
./app/python.sh - <<'PY'
from pathlib import Path
import hashlib
import json
import shutil

source = Path("outputs/sorting_package_showcase/physics").resolve()
output = Path("outputs/sorting_package_showcase_rerender").resolve()
output.mkdir(parents=True, exist_ok=False)
shutil.copytree(source, output / "physics")
files = {}
for path in sorted(source.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(source)
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    copied = hashlib.sha256((output / "physics" / relative).read_bytes()).hexdigest()
    assert original == copied, str(relative)
    files[str(relative)] = {"sha256": original, "bytes": path.stat().st_size}
(output / "qa").mkdir()
(output / "qa/physics_copy_verification.json").write_text(
    json.dumps({"source": str(source), "destination": str(output / "physics"),
                "identical": True, "physics_rerun": False, "files": files}, indent=2) + "\n"
)
shutil.copy2("configs/sorting_stream_showcase_render.json", output / "render_profile.json")
PY

./scripts/rsr collect render-sorting \
  --source outputs/sorting_package_showcase_rerender/physics \
  --render-profile outputs/sorting_package_showcase_rerender/render_profile.json \
  --output outputs/sorting_package_showcase_rerender

./app/python.sh scripts/validate_sorting_stream.py outputs/sorting_package_showcase_rerender
./app/python.sh scripts/review_sorting_stream.py outputs/sorting_package_showcase_rerender
```

只改显示设置使用 `--render-profile`，保持原物理配置与状态。正式导出不使用 `--allow-failure`。
若需要改变碰撞几何、质量、材料或动力学控制，应生成新的连续物理源，不能把这些变化伪装成重渲染。
新求解可能产生形态差异；要求重现同一物理内容时应使用已验收缓存。

### 4.3 分阶段调试

```bash
# 仅物理；物理阶段失败时停止，不继续正式渲染。
./app/python.sh scripts/check_sorting_package.py \
  --config configs/PhysicSolver_mujoco_vbd_sorting_stream_showcase.toml \
  --max-steps 16000 \
  --output outputs/sorting_package_debug/physics

# 只回放该源；脚本包含分批采图和最终数据组装。
./app/python.sh scripts/render_sorting_package.py \
  outputs/sorting_package_debug/physics \
  --output outputs/sorting_package_debug \
  --render-profile configs/sorting_stream_showcase_render.json \
  --fps 15

# 双包裹版本的独立验收；旧单软包使用 validate_sorting_package.py。
./app/python.sh scripts/validate_sorting_stream.py outputs/sorting_package_debug
```

失败源的 `--allow-failure` 仅用于诊断画面，不能转成成功示教。
局部恢复须标记 `diagnostic_restart=true`、`accepted_demonstration=false`；
恢复粒子和关节状态不等于恢复完整求解器内部状态，局部成功不能与前半段拼接交付。

### 4.4 渲染配置、分批与续渲

正式 profile 为 [sorting_stream_showcase_render.json](../configs/sorting_stream_showcase_render.json)：
1024×768、PathTracing、32 SPP、`rt_subframes=2`、4 次反弹、空间 OptiX 降噪；
关闭自动曝光、DLSS、时间降噪和相关光照缓存。时间轴停止回调后关闭 RTX ECO，避免自动暂停干扰采集。
head 使用原生 `Camera_Head` 光学坐标，局部平移为零、旋转为单位四元数；腕部光学位姿使抓持区域可见。

默认按 `overview → head → left_wrist → right_wrist` 渲染，每批最多 600 个新帧后换新 Kit 进程。
本次交付调度为 head、左腕、右腕、overview；生成顺序不改变它们共享的物理采样时刻。
不能把“按相机依次渲染”理解为每台相机重新仿真一次。

`.render_cache/manifest.json` 版本为 3，绑定源配置、结果及 NPZ 哈希、最终光学／渲染设置、
FPS、采样索引和 `capture_sync_contract="flush_scene_before_pt_counter_v1"`。
`capture_workers/*.json` 记录本批运行标识、累计帧数和完整性，父进程拒绝无进展批次。
相同源、相同设置下重新执行原命令，可以复用有效 PNG，仅补缺失、损坏或审计不匹配的帧。
源或设置变化时拒绝复用；不要手工改 manifest 来复用旧错位图像。

### 4.5 frame 会自动组装成 LeRobot 吗？

**会。正常渲染入口在四路齐全后自动组装，无需另写“拼视频”命令。**
流程为：检查四路图像 → 逐帧读取图像并计算通道统计 → 加入共享 state/action 行 →
编码四路 H.264 视频 → 写 Parquet、任务与 episode 元数据 → 将临时目录重命名为 `lerobot_v3`。

1024×768 的 8072 张 PNG 需要统计约 63 亿像素；统计阶段日志可能长时间停在最后一条采图进度。
本次恢复后的组装约耗时 13 分钟，属于本机这一轮记录，不是固定性能保证。
`.lerobot-v3-*/frames/` 是临时工作目录，尚未出现 `lerobot_v3` 不一定代表程序失败。
不过临时目录和进度 JSON 也不能证明进程仍在运行，应同时查看实际进程、日志与最终产物。

本次确实遇到组装进程退出，而 `delivery_progress.json` 仍停在 `assembling_dataset`。
退出原因没有定位；确认没有仍在运行的组装任务后，从完整 `.render_cache` 恢复，重新计算统计并完成导出，
没有重跑物理或重渲染。旧临时目录在交付确认后清理，不能把未完成的目录直接改名冒充数据集。

一般恢复可重新执行同样的 `collect render-sorting --source ... --output ...` 命令。
它会复核相机缓存并继续组装；已有正式 `lerobot_v3` 时入口拒绝覆盖，需先诊断实际状态或选择新目录。
`qa/delivery_progress.json` 是本次交付调度记录，并非所有命令都会生成；通用进度见
`render_progress.json`、`capture_workers/`、日志、`result.json` 和最终验收报告。

## 5. 遇到的问题及有效修复

### 5.1 抓不住、滑脱、变形与释放拉扯

早期 100 g、50 g 衣物候选及原袋膜质量下，原生夹爪曾空抓、提升后滑脱或释放时拉伸超限。
减轻到衣物 25 g、袋膜约 2.568 g，并提高指端有效摩擦，有助于保持，但不能单独解释全流程成功。
高摩擦也可能在张爪时拖住袋膜；单独降摩擦、降低放置高度、张爪时下降 64 mm、空中张爪等局部候选都曾失败。
加宽指垫与用户结构限制冲突，未作为合格方案保留。

最终处理组合是原生 FK 抓取深度、合适抓点、连续路径、分角色摩擦和真实支撑／释放判定。
阶段之间延续上一段命令目标，避免反复叠加实测 TCP 跟踪误差；撤离使用经过检查的笛卡尔路径，
避免张开的夹指扫过包裹。释放后仍要求完整占地、落点、标签朝上及全机器人接触归零。

后续 showcase 曾在 41.016667 s 的右臂闭合末段达到边长比 1.300869，超过 1.3。
原 `strain_limited_grip` 未覆盖闭合阶段。修复把反馈接入闭合，以放宽后的角度补偿指垫底边高度，
并把闭合末帧实际下发目标延续到提升阶段，避免瞬间恢复过紧的名义角。没有修改拉伸门限。
同起点局部开／关反馈对照支持该修复，但正式结论仍来自之后从初态求解的完整轨迹。

### 5.2 固定机器人基座漏计，造成“已放手”的假象

固定基座及相机碰撞体的 `body_id=-1` 与世界静态体相同，旧逻辑只统计动态 body，
漏掉了机器人固定根，且可能把它当台面。旧缓存有顶点距基座约 0.5 mm，位于接触包络内，
于是出现夹爪已张开、包裹仍拱起的情况。

修复按导入机器人的 shape ID 范围统计全机器人接触，把固定根分类为 `robot_root`，
并用实际世界凸几何、全部顶点和面中心独立检查间距。通过调整原支撑范围内的任务落点避让，
没有移动机器人或台面，也没有删除基座碰撞体。

单包裹历史成功落点为平台 `[0,-0.93]`。后续流式运行仍出现固定根 28 个接触，
当前平台目标前移到 `[0,-0.96]`；同时提升高度由 0.33 m 改为 0.28 m，
避开右臂高位转运的关节限位。旧版的 4.366 mm 或 25.678 mm 基座间距属于各自运行，
不作为当前轨迹的实测值直接复用。

### 5.3 TCP、指垫底边和右臂重抓点混淆

原生夹指含金属凸体与胶垫，闭合时最低点随角度变化。只用 TCP 或金属指尖推断抓深会失准。
现在通过原生 FK 计算指垫底边；右臂在当前抓点附近约 ±30 mm 区域读取衣物表面 q90 高度，
结合绝对下压、相对比例和最低底边限制选高，而不是沿用初始包裹高度。

修复基座后包裹更平展，旧拱起状态筛出的 `[-35,+55] mm` 偏心抓点仍导致拉伸。
减小补夹或改 hold 到 0.64/0.63 rad 也没有解决。联合筛查平台、出料落点和全段 IK 后，
中心抓取获得连续可达路径。局部 Y+25 mm 等成功候选仅用于选择方向，没有拼接进正式数据。

### 5.4 60 Hz 记录漏掉原生 mimic 子步尖峰

USD 中有 PhysX mimic 配置不代表 MuJoCo 已建立正确约束。早期 60 Hz 记录约 0.0125 rad，
逐子步却发现 0.03098 rad 峰值，超过 0.03 rad。
修复使用原生 MuJoCo 等式约束、CPU 双精度状态及每物理帧 128 个刚体子步，
保存全部原生子步残差，检查数量、有限性和峰值。

批处理与逐步执行核对共同时刻关节状态，不减少原始求解子步。
降低刚体／柔体精度曾很快触发 mimic 超限或空抓；不能用粗步长“跑得快”替代正式物理精度。
诊断写盘也曾误读 `planner.config.newton_substeps`，现从正确的运行时配置读取并有 NPZ 写盘回归。
没有缓存的中断运行只保留日志，不能称为可回放成功源。

### 5.5 连续输送、纸箱落地和软包后半程损伤

移动出料带上，首次触带落点与后续落稳位置不同；分别判定，保留 80 mm 落点门限。
输送由真实接触的切向相对运动实现，不直接平移包裹粒子；带面标记位移与物理速度一起记录。
纸箱采用独立接触参数和 PGS，柔体静止且远离纸箱后才能休眠，接近则唤醒。

曾有纸箱落到软包上导致柔性网格拉伸、地面穿透；因此增加落地区域避让及整个双件时间线的柔体检查。
另一次软包以约 6 m/s 落地，单 VBD 子步行程超过 3 mm，原约 1 mm 接触包络来不及捕获冲击。
仅把地面刚度提高到 2000000 的局部对照仍失败。现在仅给地面增加 4 mm shape margin，
提前建立接触；夹指、平台和带面 margin 不变，穿透门限仍为 1.5 mm。
这个接触皮层可能使静置软包与可视地面相隔数毫米，不能把它误说成零厚度接触。

离带落地检查也修正过时序：软包先完整越边，落地摊开后投影可以回到高架带边下方。
现在先累计确认完整离带，再检查后续全网格的落地高度；不要求两个条件恰好在同一帧满足。
从未完整离带、仅离带前接近地面、离带后仍过高的情况仍拒收。

### 5.6 袋衣间隙、形变和夹爪错位是不同问题

`bag_gap_q95_m` 衡量袋膜到衣物表面的距离分位数，不是夹爪到包裹的距离。
`aligned_garment_shape_rms_m` 衡量对齐后的衣物形变，夹指穿透又是另一项几何检查。

早期用户允许先完成流程，故 `require_bag_gap_quality=false` 时仍记录 `strict_checks` 和
`strict_quality_success=false`，不能把流程通过写成严格贴合通过。
旧 1K 运行的 10.030 mm 大于 10 mm 是真实严格失败，并非通过显示精度四舍五入即可消除。

随后用户授权提高变形上限，当前仅把 RMS 上限改为 45 mm，并启用严格袋衣间隙验收。
在胶垫几何修正后重新连续求解的当前源，其持握样本最大袋衣间隙 q95 为 9.858253 mm，
RMS 最大 39.717888 mm。最后一次复制重渲染没有改变这些物理内容或门限。

### 5.7 夹爪画面整体错位：真正需要修复的是渲染同步

用户在旧 1K 左右腕视频，以及 `sorting_stream_showcase_aligned` 的 head 第 340 帧指出：
夹爪整体位置或姿态偏离抓取点；提高分辨率前未观察到该问题。
此前仅检查 USD 位姿、可视角点和相机矩阵，不能证明 GPU 输出已更新，旧视频验收因此不充分。

排查区分了两个独立问题：

1. **碰撞代理尺寸不一致。** 原碰撞胶垫为 30×7.5×70 mm，可视胶垫实际为 27×7.5×65 mm。
   将代理匹配现有可视几何，保留质量、惯量及关节；这改变了物理接触，故先重新生成连续物理源。
2. **刚体与柔体进入渲染器的时序不一致。** 连续采集过早递增 `/rtx/externalFrameCounter`，
   刚体变换尚未完成 Hydra 更新就开始 Path Tracing 累积，出现旧夹爪姿态与新袋体网格同帧显示。
   相机未变化时跳过刷新的优化暴露了这个问题，单纯提高分辨率或检查 CPU 矩阵不能解决。

现在每次发布保存样本后，先等待两次**实际渲染完成事件**，再递增外部累积计数并完成采样。
新相机和每个续渲批次的首帧走完整预热，至少等待 8 次实际渲染事件。
`world.update()` 调用次数不等于 GPU 完成帧数，不能用固定调用次数替代这个同步屏障。
全过程检查推进物理步数为零。

旧快速采集对照的夹爪区域平均绝对像素误差约为 25.35（8-bit）；修复后对应静态对照为 0。
512×384 和 1024×768 均完成 20 项同实例对照，证明缩小分辨率只是排查提速手段，最终仍可交付 1K。
缓存升级至 v3 后拒绝旧 PNG；仅修代码却继续复用旧缓存，会把旧错位重新交付。

生产采集同时检查第 `326、340、724、1335、1733` 帧的四路实际导出 PNG，
与**同一个渲染实例**充分刷新的静态画面逐像素比较，保存参考图和 SHA-256。
20 项全部相等，独立验收再次检查帧号、源索引、PNG 哈希和参考图。
不同进程可能存在采样或光照差异，不能把跨进程整图像素差直接认定为姿态错位。
CPU 几何检查仍保留：全部 2018 个视频时刻、4 块胶垫、每块 8 个角点与求解器碰撞体对应。

需要复查同步时，在新的诊断目录执行：

```bash
./app/python.sh scripts/validate_sorting_render_sync.py \
  outputs/sorting_package_showcase/physics \
  --output outputs/sorting_render_sync_check \
  --render-profile configs/sorting_stream_showcase_render.json
```

证据：[修复诊断摘要](../outputs/sorting_package_showcase/qa/render_sync_diagnosis.json)、
[第 340 帧前后对比](../outputs/sorting_package_showcase/qa/head_frame340_before_after.png)、
[正式帧同步报告及参考图](../outputs/sorting_package_showcase/qa/render_sync/)。
诊断摘要中的旧 reproduction 路径是历史来源，目录已清理；当前 `qa/render_sync/` 才是本次正式输出的证据。

### 5.8 长时任务、缓存和验收性能

历史长时采图曾在单进程约 3094 帧、独立左腕进程约 1855 帧处停滞，当时未发现内存、显存或 I/O 耗尽，
Kit 内部原因未确定。采用每批最多 600 个新帧的独立进程限制单进程寿命，并用运行标识防止旧结果误判。
这属于渲染恢复；重启后的物理轨迹不能据此拼接为正式 episode。

为降低几何审计成本，`package_geometry.py` 使用 NumPy AABB 树和保守距离下界剪枝，
叶节点仍计算原精确点到三角形距离。历史真实帧对照最大差值为 0，约从 6.45 s/帧降至 0.28～0.46 s/帧，
没有减少探针或放宽门限。整帧 CUDA 图等历史提速尝试收益不足或有兼容限制，未替换正式验证路径。

## 6. 当前产物布局与数据含义

```text
outputs/sorting_package_showcase/
├── physics/                       # 7 个原始物理文件及配置、结果
├── render_profile.json            # 本次纯渲染参数
├── .render_cache/
│   ├── manifest.json              # v3 源与渲染契约
│   └── frames/{camera}/frame_*.png # 四路各 2018 张，支持续渲
├── capture_workers/               # 各相机批次结果与完整性标记
├── diagnostics/                   # 渲染运行的配置、源副本与结果
├── logs/                          # 采图、组装、验收和预览日志
├── result.json                    # 物理结果、来源哈希、渲染与同步证据
├── render_progress.json
├── lerobot_v3/
│   ├── data/chunk-000/file-000.parquet
│   ├── videos/observation.images.{camera}/chunk-000/file-000.mp4
│   └── meta/
│       ├── info.json
│       ├── stats.json
│       ├── tasks.parquet
│       ├── episodes/chunk-000/file-000.parquet
│       └── task_outcome.json
└── qa/
    ├── stream_audit.json
    ├── physics_copy_verification.json
    ├── render_sync/{camera}/frame_*.json、frame_*.png
    ├── synchronized_four_view.mp4
    ├── grasp_four_view_keyframes.jpg
    ├── soft_workflow_keyframes.jpg、rigid_workflow_keyframes.jpg
    ├── visual_review.json
    └── delivery_progress.json
```

物理源包含 `configuration.json`、`result.json`、`rigid_result.json`、`particles.npz`、
`stream.npz`、`trajectory.json`、`stream_trajectory.json`。复制后不改这些文件里的历史路径，
新渲染结果单独记录当前 `source_physics` 和哈希，保留来源可追溯性。

`observation.state` 是 16 个原生坐标的实际角度：左臂 7、右臂 7、左右夹爪主关节各 1；
`action` 是同顺序的位置目标，单位均为 rad，具体名称在 `meta/info.json`。
四路视频、状态、动作使用相同源索引，15 FPS 时间戳为 `frame_index / 15`。
本条末帧时刻为 134.466667 s，2018 帧视频时长约 134.533333 s；
源物理终点 134.516667 s。三者分别是末帧时间、编码时长、物理终点，不应强行混为一个数字。

正式视频为 H.264、yuv420p、无音轨。四路拼接预览的每格为 640×480、整体 1280×960，
这只是审阅视频；四台训练相机各自仍为 1024×768。
`meta/task_outcome.json` 保存物理结果和有效配置，与根结果及 diagnostics 一起复核。

## 7. 当前验收结果与覆盖范围

以下数值来自当前 [stream_audit.json](../outputs/sorting_package_showcase/qa/stream_audit.json)，
不是旧 `sorting_package` 或旧 1K 运行的数值。

| 指标 | 本条独立实测 | 门限／结论 |
| --- | --- | --- |
| 全时间线软包最大边长比 | 1.208587 | <1.3，通过 |
| 最小三角形面积比 | 0.566737 | >0.03，通过 |
| 持握衣物对齐 RMS 最大值 | 39.717888 mm | <45 mm，通过；本上限经用户授权调整 |
| 持握袋衣间隙 q95 最大值 | 9.858253 mm | <10 mm，严格通过 |
| 原网格夹指穿透，软包／纸箱 | 0.859124／0.024856 mm | ≤1.5 mm |
| 地面穿透，软包／纸箱 | 0／0.198107 mm | ≤1.5 mm |
| 有限支撑面穿透，软包／纸箱 | 0／约 0.000362 mm | ≤1.5 mm |
| 原生联动约束 | 1033088 个子步记录完整，残差检查通过 | <0.03 rad |
| 实际入料位移，软包／纸箱 | 0.751636／0.753451 m | 到位检查通过 |
| 完整离带时刻，软包／纸箱 | 63.783333／130.766667 s | 完整越过带边 |
| 落地判定时刻，软包／纸箱 | 64.300000／131.283333 s | 发生在完整离带之后 |
| 最终两件 AABB 间隔 | 30.236055 mm | 落地包裹分离 |
| 可视胶垫角点与物理几何误差 | 最大约 5.09×10⁻¹⁶ m | ≤2×10⁻⁶ m；全部视频时刻 |
| 正式导出 PNG／同实例静态参考 | 20/20 相等，像素误差为 0 | 同步检查通过 |
| 四路视频与状态／动作 | 全量解码、帧数、时间索引、原始关节值核对通过 | 每路 2018 帧 |
| 物理复制与重渲染 | 7 个文件 SHA-256 一致，新增物理步数 0 | 保持原物理内容 |

持握形状与袋衣间隙覆盖 94 个保存样本，包括转运视频采样点和检查点；
夹指几何覆盖软包全部 8072 个保存帧、纸箱可见后的 4036 帧，使用顶点和面中心探针。
这不是连续时间内无穿透的数学证明，也不是对每个三角形内部所有点的穷尽检测。
在线接触穿透与独立几何穿透的采样、算法不同，应分别引用。

人工视觉复核查看了两件包裹的流程拼图、四路抓取拼图，以及从正式 MP4 精确提取的
head／左腕第 340 帧、右腕第 724 帧；这些抓取画面未见先前的整体错位。
[visual_review.json](../outputs/sorting_package_showcase/qa/visual_review.json) 明确是选帧检查，
没有宣称人工观看了每一视频帧。自动像素对照也只覆盖指定的 20 个组合。

LeRobot 布局和内容通过项目导出器及校验器验收，**尚未执行官方 LeRobot SDK 加载测试**。
最后一轮功能修改记录为 113 项相关回归通过、15 个 warning，Ruff 和 diff 空白检查通过；
这不等于全仓测试通过。旧迁移阶段曾有 16 个历史缺失模块造成全仓 pytest 收集错误，
不应把该历史记录当成本次已重新执行全仓测试的结论。

## 8. 历史版本和失败实验如何解释

| 阶段 | 结果与现在的关系 |
| --- | --- |
| V7／V8／V9／V10 参考 | 有成功抓取，也有偏移、旋转、质量和贴合失败；部分使用运动学宽指垫，不能等同原生 Pro-M6 能力 |
| 早期原生 100 g／50 g 候选 | 完成过左臂抬升，但严格间隙、释放拉伸或形变失败；减重、慢动作、增摩擦均非单独充分条件 |
| 低精度与高驱动增益对照 | 粗步长出现 mimic 超限／空抓；提高增益仍掉包，未作为最终修复 |
| 多组平台缓存恢复 | 验证过降低摩擦、偏心抓点、实时深度、减夹等局部方向；全部保留诊断边界，不能拼接示教 |
| `sorting_package` 单软包版 | 56.8 s、852 帧、640×480；完整流程通过，袋衣间隙约 11.241 mm，严格项失败，按当时流程优先口径交付 |
| `sorting_stream` 双包裹版 | 134.516667 s、2018 帧、640×480；历史间隙实测约 8.393 mm，但配置仍为 advisory；不是当前 1K 交付 |
| `sorting_stream_showcase_1k_final` | 1K 流程与旧检查通过，但严格间隙 10.030 mm 失败，且后被用户发现夹爪渲染错位；不能作为当前正确视频依据 |
| `sorting_stream_showcase_aligned` | 修正胶垫并提高 RMS 上限后得到已接受物理源；旧视频第 340 帧仍错位，CPU 对齐检查不足，视频失效 |
| `sorting_package_showcase` | 复制上述已接受物理源；修复 GPU 渲染同步后重渲染 1K，自动组装、严格质量和独立验收均通过 |

早期实验曾归档到 `/home/fufa/Projects/outputs/outputs3-分拣包裹/outputs`，
旧文档记录的 `outputs/debug/*`、`outputs/PhysicSolver/*` 和部分旧交付目录可能已移走或删除。
本文保留必要结论和实验名称，不把历史路径作为仍可点击的当前证据。原两份文档的逐次记录可通过 Git 历史查阅。

## 9. 后续复现和排查顺序

1. **先判断是物理失败还是画面失败。** 查 `physics/result.json` 的失败原因、连续性和严格检查；
   不用末尾 USD／材质告警代替真正退出原因。物理回调提前停止且 `state.done=false` 不能写成完成。
2. **冻结问题时刻和物理源。** 保存配置、源哈希、相机、视频帧号和源索引。
   物理改变必须重新完整求解；纯渲染问题使用同源状态，不在不同物理轨迹之间比较画面。
3. **按直接证据定位。** 接触角色／基座 → 实际胶垫几何与 FK → 抓点、深度和路径 →
   原生子步约束 → 全程柔体质量。相机矩阵正确但画面仍错位时，检查 Hydra／GPU 完成事件与累积时序。
4. **先做低成本对照。** 使用小分辨率、同实例静态参考或局部诊断缩小原因；
   局部恢复只支持局部结论。需要调参时以单变量和有效配置对照为依据，不把单次形态波动归因于某个参数。
5. **恢复时核对进程和缓存契约。** 不重复启动仍活跃的工作进程，不把旧进度 JSON 当成进程状态；
   不复用旧同步版本 PNG，不修改 manifest 绕过检查。
6. **交付前闭环。** 成功连续源、复制哈希、零物理步重渲染、实际导出像素对照、四路解码、
   状态／动作对齐、严格几何验收及明确范围的选帧视觉检查都完成后，再标记交付完成。

进一步提升吞吐应优先优化图像统计、编码和有界采图的可观测性，保留真实渲染完成屏障及验收门限。
提高画质或减少等待都需要重新做同实例同步对照；分辨率变大本身不能成为允许姿态错位的理由。
