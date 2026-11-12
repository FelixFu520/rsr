# 场景盘点与物理引擎 / 求解器选择

结论来自本仓库三个任务的实测记录（`docs/*.md` 的“问题与解决方案”）和 `simworld/isaacsim.py`。
它们只对 Isaac Sim 6.2 RC + Newton 1.6.1 有效；换版本后重新做冒烟测试。

## 1. 先盘点场景里有什么

对环境、机器人、物体各跑一次盘点脚本（只读，不写文件），再填下表：

```bash
./app/python.sh .agents/skills/rsr-task-development/scripts/usd_inventory.py <env.usd> <robot.usd> <object.usd>
```

| 维度 | 要查的内容 | 判读要点 |
|---|---|---|
| 环境 | 整份 stage 还是背景引用；碰撞体数量与 `physics:approximation`；是否带动态刚体；灯光；Flow/粒子等特效 | 未设置近似的静态网格按三角网格碰撞（巡检仓库 6268 个）；背景动态刚体默认被设为 kinematic（`background_gravity=false`） |
| 机器人 | 格式（USD+PhysX schema / URDF / MJCF）；固定或浮动基座；关节数；驱动刚度/阻尼/力矩限；mimic 或闭环（Robotiq）；现成控制器或策略及其训练引擎 | `PhysicsArticulationRootAPI`、`PhysicsDriveAPI:*`、`PhysxJointAPI`、mimic 类 schema 决定能否直接用 PhysX 或需转换 |
| 被操作物体 | 刚体（质量、惯量、碰撞近似）、关节物体、柔体（布料表面网格 / 双网格包裹 / 体积软体）、粒子、线缆 | 有柔体就决定了接触求解器必须是 Newton VBD（本仓库唯一验证过的柔体路径） |
| 交互 | 抓刚体 / 抓柔体 / 推 / 只导航；传送带；物体堆叠；是否需要双向力反馈 | VBD 外部刚体是单向耦合：机器人推柔体，柔体反力不回传 |
| 查询与传感 | 相机路数；射线、重叠查询（建图、视线）；接触传感 | PhysX 场景查询（`overlap_box`、`raycast_all`）只在 PhysX 下可用 |
| 规模 | episode 时长、物理步数、粒子数、渲染器与 spp | 巡检 PhysX CPU 约 4 ms/步，墙钟 88% 花在 PathTracing；分拣物理求解 4739 s |

## 2. 本仓库可用的世界模式

由 `[sim_world] physics_engine` 与 `newton_execution` 决定，所有任务共用 `IsaacsimWorld`，不要另建世界类。

| 模式 | 动力学归属 | 适用 | 约束 |
|---|---|---|---|
| `physx` | USD 场景全部由 PhysX 积分（`physx_solver` TGS/PGS） | 刚体/关节任务、PhysX 制作的资产、需要场景查询、Lula IK、Isaac 关节 API | CPU tensor API 只在时间轴播放时可读写；先建物理视图再加机器人时要 `invalidate_physics()` |
| `newton` + `usd` | Isaac 从 USD 建 Newton 模型（mujoco/xpbd/vbd） | 资产已兼容 Newton，且做过冒烟测试 | 本机缺 `NewtonVbdSceneAPI`，VBD 不能走此路径；分拣的 Robotiq 夹爪闭合时 MuJoCo 发散为 NaN；Newton tensor `get_jacobians` 未实现 |
| `newton` + `direct` | Planner 自建 Newton 模型并注册 stepper；Kit 时间轴停止，只显示 | 柔体操作、需要组合多个 Newton 求解器、需要原生 MuJoCo 约束 | 组件自己负责建模、子步、USD 显示同步；`register_newton_stepper` 必须在 `reset/play` 前调用 |
| `hybrid` | 每步先 PhysX，再调 Newton stepper | 机器人必须留在 PhysX（Panda USD + Lula IK），物体用 VBD | 不会自动建跨引擎碰撞：要把 PhysX 手指/手掌位姿做成 Newton 运动学代理并逐子步插值；无力反馈 |

## 3. 求解器能力与已知限制

| 求解器 | 解什么 | 本仓库实测结论 |
|---|---|---|
| PhysX TGS | 关节、刚体接触、摩擦 | 刚体方块抓放落点误差 2.6 mm；X30 巡检全程稳定 |
| Newton MuJoCo | 机器人关节、驱动、刚体接触（不解柔体） | 位置控制 + 原装短指抓 V7 包裹通过；分拣用原生 MuJoCo **CPU** 双精度 + 等式约束实现 mimic，逐子步残差 < 0.03 rad；60 Hz 记录会漏掉子步尖峰 |
| Featherstone | 关节坐标的机器人动力学 | 官方衬衫关键帧通过；长夹块抓包裹滑脱；Newton 1.6.1 不执行 `joint_effort_limit` |
| XPBD | 带柔顺度的位置约束，刚体 + 柔体 | 最大坐标，不更新 `state.joint_q`，需自建观测；会把输入 `body_q` 当临时空间，需步前复制步后恢复；不执行力矩限；统一 XPBD 抓衣服失败（1.6.1 无布料自碰撞） |
| VBD | 布料/袋膜拉伸、弯曲、接触、自碰撞 | 衣服、包裹、软包都靠它通过；CUDA 结果不逐位可复现，落点等敏感量会在运行间波动；大位移下自接触界限变保守，用 `MovingFrameVBD` 处理 |

## 4. 决策顺序

1. **有柔体且与机器人接触** → 接触/柔体求解器选 VBD。
   - 机器人与夹爪可导入 Newton → `newton` + `direct`。夹爪有 mimic、要位置控制或要精确夹持 → MuJoCo（优先 CPU）；
     沿用官方速度控制示例 → Featherstone。
   - 机器人或控制器必须留在 PhysX → `hybrid`，接受运动学代理和单向耦合。
   - 不选统一 XPBD 做正式数据（只作实验组）。
2. **只有刚体与关节** → 默认 `physx`（TGS）。只有在有明确理由（例：策略在 MuJoCo 中训练、需要与 Newton 柔体任务共用模型）
   且冒烟测试通过时才用 Newton；PhysX 资产转 Newton 先读 `app/skills/physx-to-newton-conversion`。
   单环境、需要场景查询的任务用 PhysX CPU（`device="cpu"`）；多环境或大规模粒子再考虑 GPU。
3. **足式 / 人形移动** → 有策略：使用训练时的引擎、步长、观测与动作定义，并核对资产一致；没有策略：先问用户，
   可选运动学机身 + 脚本步态（巡检方案 A），在数据中标注 `locomotion` 并写明局限。
4. **视觉特效**（Flow 火焰、烟）只属于渲染和时间轴，不影响引擎选择，但会约束步进与采集时序（见巡检文档第 8 节）。
5. **耦合方向**写进文档局限：单向耦合的数据不能评价夹爪负载或限力性能。

## 5. 参数起点（引用来源，不要直接照搬）

| 参数 | 已验证取值 | 来源 |
|---|---|---|
| `physics_dt` | 1/120（PhysX 巡检、桌面抓放）；1/60（分拣，每帧 32 个 VBD 子步 × 28 迭代，每 VBD 步 4 个 MuJoCo 子步） | 各任务配置 |
| 录制 fps | 15（必须整除物理频率） | 全部任务 |
| VBD 接触 | shape 刚度 40000、SDF resolution 128、自接触 gap 1.6 mm；有效摩擦 `sqrt(mu_particle × mu_shape)` 按角色分别设置 | 分拣 3.5 节 |
| 粒子碰撞半径 | 8 mm（1–4 mm 在转运中滑脱） | 物理求解器 7 节 |
| 地面高速落地 | 仅地面加 4 mm shape margin，其余 margin 不变 | 分拣 7.5 节 |
| 质量 | 按真实物体给总质量，按面积或原逐点比例分配；不用示例密度（`density=0.02` 在厘米模型里约 141 kg） | 物理求解器 7 节 |

## 6. 验证选型并记录

1. 世界模式冒烟：`./app/python.sh scripts/physic_solver/validate_physics_modes.py --engine <e> [--solver <s>] [--cloth]`
   （每次一个 Kit 进程，验证自由落体与“只渲染不积分”）。
2. 任务最小探针，结果放 `outputs/debug/<task>/engine_probe*/`：机械臂闭合夹持并保持 N 秒测滑移；足式站立稳定性；
   柔体落桌拉伸与穿透。只跑物理、不录 RGB，降低成本。
3. 在 `docs/<task>.md` 的“设计选择”节写：选定的模式和求解器、步长/子步/迭代、被否决的方案与证据
   （参考 `docs/fire_inspection.md` 3.1 节）。
4. 出现穿透、发散、夹不住、太慢时，按 SKILL.md 末尾转到对应 isaac-* skill。
