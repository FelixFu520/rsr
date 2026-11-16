# Real2Sim2Real 源码精读指南

本文按**类、方法、对象引用、数组形状和状态变化**讲解代码。目标是能沿调用栈解释实现、定位修改点，并判断一次修改会影响哪些任务。
运行环境、资产准备和采集命令分别见 [README](../README.md) 与各任务文档；本文将阅读单位细化到函数。

基于 2026-11-16 工作区源码。源码链接中的行号用于定位本次版本；后续代码变化时以符号名为准。
文中标为“简化”的片段用于解释控制流，省略日志、异常处理或 SDK 细节。

## 1. 源码阅读顺序与方法

按下面的代码单元阅读，每个单元都要记下：**调用方 → 参数 → 读取的状态 → 修改的状态 → 返回值 → 后续消费者**。

| 单元 | 精读符号 | 必须能从代码回答的问题 |
|---|---|---|
| 1 | `AutoJob.from_config`、`BaseJob.__init__` | 工厂返回类还是实例？传入的 config 管哪一个组件？ |
| 2 | `BaseConfig.from_settings`、`peek_backend`、`Registry.load_entry_points` | 配置校验与选型为什么分成两步？ |
| 3 | `BaseJob.run`、四个阶段、`close` | 哪些方法发生动态派发？异常后哪些钩子还会执行？ |
| 4 | `IsaacsimWorld.reset/play/step/update` | 谁拥有物理时间？何时调用 Newton 回调？ |
| 5 | `TabletopHomeScene`、`FrankaTabletopRobot` | USD 资产如何变为运行时状态接口？ |
| 6 | `CumotionPickPlacePlanner`、`TabletopPickPlaceJob._run_episode` | 一次规划怎样变成逐步命令？一行数据对应哪一时刻？ |
| 7 | `LeRobotV3Recorder`、`ContactMonitor`、`evaluate_pickplace` | 图像、行数据、接触和验收如何连接？ |
| 8 | `TabletopReplayJob`、`replay_command` | 不重写仿真循环怎样替换整个动作来源？ |
| 9 | 巡检、求解器、分拣、电池、HALO 的关键调用链 | 各任务究竟覆盖了基类的哪一层？ |

建议先把单元 1–8 读透，再逐任务阅读第 8–12 节。第一次不要顺序通读数千行的物理运行时文件。
遇到函数调用，先确认接收对象的实际类型。例如 `self.runtime.step()` 在不同 Planner 中会派发到不同实现。

## 2. 配置、工厂和注册：字符串怎样变成对象

### 2.1 `AutoJob.from_config()` 的两阶段构造

入口：[job/base.py：AutoJob](../real2sim2real/job/base.py#L141)。先逐行读 `from_config()`，再对照其他四域的 `Auto*`。

```python
# 按实际代码压缩，保留关键顺序。
ensure_job_backends()
name = backend or (config.backend if config is not None else BaseConfig.peek_backend("job"))
job_cls = JOB_REGISTRY.get(name)
if config is None:
    cfg_cls = getattr(job_cls, "config_cls", BaseConfig)
    config = cfg_cls.from_settings("job")
return job_cls(config=config, **kwargs)
```

这里有四个不同对象：`name: str`、`job_cls: type`、`cfg_cls: type[BaseConfig]`、`config: BaseConfig`。
`Registry.get()` 返回实现类，最后一行才构造实例。`config_cls` 是类属性，会随具体 Job 子类改变。

继续进入 [BaseJob.__init__](../real2sim2real/job/base.py#L26)：

```text
config                           → self.config（仅 Job section）
AutoSimWorld.from_config()       → self.sim_world
AutoScene.from_config()          → self.scene
AutoRobot.from_config()          → self.robot
AutoPlanner.from_config()        → self.planner
```

传入 `TabletopPickPlaceJobConfig` 并不会同时传入 Scene/Robot/Planner 配置。
四个工厂仍各自从进程 settings 读取对应 section。这解释了为什么脚本先调用 `apply_settings_file()`，再构造 Job。

练习：追踪 `cumotion_pickplace`，从 [pyproject.toml](../pyproject.toml) 的 entry point 找到
`CumotionPickPlacePlanner`，再找到其 `config_cls = CumotionPickPlacePlannerConfig`，最后回到 `AutoPlanner.from_config()`。

### 2.2 `BaseConfig` 的数据转换链

精读 [core/config.py](../real2sim2real/core/config.py) 的以下函数：

| 函数 | 输入 → 输出 | 实现要点 |
|---|---|---|
| `apply_settings_file(path)` | 路径 → 绝对路径 | 检查文件；写 `RSR_SETTINGS_FILE`；`reset_settings()` 清缓存 |
| `_expand_settings_chain(project_file)` | 子配置路径 → 父先子后的路径列表 | 内部 `walk()` 深度优先；递归前放入 `seen`，阻断环 |
| `_build_settings()` | 路径链与环境 → Dynaconf | 默认文件、父/子文件、secrets 按顺序加载，开启 merge |
| `get_settings()` | 无参数 → 进程单例 | `_SETTINGS` 缓存；后续修改文件不等于对象自动重载 |
| `peek_backend(section)` | section → 后端字符串 | 只窥探 backend，不用通用 BaseConfig 校验全部任务字段 |
| `from_settings(section)` | section → 当前配置子类实例 | Box 转普通容器 → 归一化顶层键 → `model_validate()` |

为什么不能先 `BaseConfig.from_settings("planner")` 再选型？因为通用配置只有 `backend`，
`extra="forbid"` 会拒绝 `cumotion_robot` 等专属字段。必须先取名字、选配置子类，再完整校验。

`_normalize_top_level_keys()` 只处理当前 section 顶层，并移除 `inherit`、`dynaconf_merge` 等保留键。
它不会递归替你修正嵌套 Pydantic 字段名，因此 `RSR_JOB__RECORDER__record_rgb` 的末级小写有实际意义。

代码练习：给 `_expand_settings_chain()` 手工构造 `child → parent → grandparent`，在纸上写出 `seen` 与 `chain` 每次递归后的值。
然后解释为什么 `seen.add()` 放在递归之后会无法阻止循环引用。

### 2.3 `Registry` 与导入时机

入口：[core/registry.py](../real2sim2real/core/registry.py)。内部核心结构为 `_impls: dict[str, type[T]]`。

`load_entry_points(group)` 迭代 `importlib.metadata.entry_points()`，对每项调用 `ep.load()` 导入模块并获取类，
再 `register_class(ep.name, impl_cls)`。导入失败会 warning 并继续；`get()` 找不到名字才抛含可用后端的 `KeyError`。

`ensure_*_backends()` 使用模块级布尔值避免反复加载。同名注册是字典覆盖，不会创建多个对象。
`ep.load()` 仍会执行模块顶层代码，所以任务文件把 Kit、pxr、Newton、Warp 的重依赖放在方法或延迟定义内部。
这也是 `list-backends` 可以发现真实后端而不启动仿真的代码原因。

## 3. 五个基类：对象所有权与动态派发

入口：[SimWorld](../real2sim2real/simworld/base.py)、[Scene](../real2sim2real/scene/base.py)、
[Robot](../real2sim2real/robot/base.py)、[Planner](../real2sim2real/planner/base.py)、[Job](../real2sim2real/job/base.py)。

### 3.1 先画内存中的引用关系

```text
ConcreteJob
  ├── config: ConcreteJobConfig
  ├── sim_world ──────────────→ ConcreteSimWorld
  ├── scene ─────────────────→ ConcreteScene
  │                              └── sim_world ──→ 同一个世界
  ├── robot ─────────────────→ ConcreteRobot
  │                              └── sim_world ──→ 同一个世界
  └── planner ───────────────→ ConcretePlanner
                                 └── robot / scene / runtime（由具体后端绑定）
```

`scene.init_scene(self.sim_world)` 和 `robot.init_robot(self.sim_world)` 传递的是对象引用。
它们不创建新的仿真世界。Planner 基类不接收 Scene；桌面 Planner 的 `build()`、物理任务的 `configure_scene()` 等方法补上该引用。

| 基类字段/方法 | 当前实现含义 | 不应做的推断 |
|---|---|---|
| Scene/Robot `poses` | 初始摆放位姿缓存 | 不会自动等于 PhysX 每步状态 |
| `poses_realtime` | 供后端发布实时位姿 | 字典本身没有刷新机制，查具体写入点 |
| `sensors` / `state` | 传感器与后端额外状态 | 不能假设所有任务有相同 key |
| Planner `state["last_action"]` | 最近控制输出，由任务维护 | 不保证跨任务维度相同，也不保证包含速度前馈 |
| `BasePlanner.on_physics_step()` | 默认立即返回 `True` | 基类没有通用运动算法 |

这些 `ABC` 中多个方法只有 docstring，没有 `@abstractmethod`；类名里的 Base 不代表 Python 一定禁止直接实例化。
接口语义主要由后端与测试保证。阅读时要检查方法体，不能只看注释。

### 3.2 `BaseJob.run()` 中哪些调用会进入子类

`run()` 中的 `self.init_world()` 等是普通虚调用，阶段方法中的 `self.before_*()`、`self.simulate()` 同样动态派发。
例如 `TabletopPickPlaceJob` 没有重写 `run()`，却通过 `before_place_robot_hook()` 传桌面高度、
`after_place_robot_hook()` 构建 cuMotion、`simulate()` 执行多集采集。

重点读 `place_robot()` 的顺序：

```text
before_place_robot_hook()   # 此时场景已摆好，机器人尚未加载
robot.init_robot(world)
robot.arrange()
adjust_robot()
planner.init_planner(robot)
gui_inspect(...)
after_place_robot_hook()    # 具体任务可在这里绑定 Scene 并构建规划资源
```

`run()` 的 `try/finally` 包住四个阶段。任一阶段抛异常会跳过它之后的语句及 after 钩子，但执行 `close()`。
`BaseJob.close()` 调用 `destroy_world()`，随后将 `sim_world` 设为 `None`；桌面任务还先关闭相机与接触订阅，再调用 `super().close()`。

**当前实现与约定的差异：** `BaseJob.adjust_scene()` / `adjust_robot()` 目前只有 docstring；
它们没有默认调用组件 `adjust()`。通用 Scene/Robot 也没有定义该方法。
`AGENTS.md` 中的 `TabletopScene` / `BaseManipulationPlanner` 目前不在对应 `base.py` 中。
跟读现有代码时以实际覆盖方法为准，扩展时仍遵守 [AGENTS](../AGENTS.md) 的职责分层。

练习：在 `StubJob` 上逐步执行 `run()`，给四个 before/after 钩子编号；假设 `scene.arrange()` 抛异常，列出实际还能执行的方法。

### 3.3 五个 stub 是最小实现对照

逐个打开 [StubSimWorld](../real2sim2real/simworld/stub.py)、[StubScene](../real2sim2real/scene/scene_stub.py)、
[StubRobot](../real2sim2real/robot/robot_stub.py)、[StubPlanner](../real2sim2real/planner/planner_stub.py)、[StubJob](../real2sim2real/job/job_stub.py)。
先圈出 `config_cls`、`super().__init__()` 和配置赋值，再比较具体任务多添加了哪些资源与方法。
Scene/Robot 初始化保存世界引用，arrange 为空；StubJob 直接继承通用生命周期，没有自己的采集循环。

一个容易忽略的实现细节：`StubPlanner.init_planner()` 写的是 `self.ready` / `self.phase` 属性，
而 BasePlanner 构造器还创建了 `state["ready"]` / `state["phase"]`。两套存储不会自动同步。
做代码追踪时应逐个找赋值点，不能看到相同单词就认为它们是同一个状态。

## 4. `IsaacsimWorld`：物理时间的唯一推进点

入口：[simworld/isaacsim.py](../real2sim2real/simworld/isaacsim.py)。先读类注释和 `IsaacsimWorldConfig`，
再读 [reset/play/step/update](../real2sim2real/simworld/isaacsim.py#L543)。

### 4.1 初始化把哪些句柄存到对象上

`__init__()` 保存配置和回调，`init_world()` 才进入 `_initialize_runtime()`、加载环境与配置物理。
跟踪 `app`、`stage`、`_manager`、`_timeline`、`_settings` 的赋值，分清 Kit App、USD Stage、物理管理器不是同一对象。
`_load_environment()` 区分打开整个 stage 与在新 stage 引用背景；Scene/Robot 后续再挂自己的 Prim。

`register_newton_stepper(stepper, reset=..., close=...)` 存的是回调引用。
direct/hybrid 的任务模型在 Planner/Runtime 创建，世界只负责在正确时刻调用回调。

| 方法 | 主要状态变化/调用 |
|---|---|
| `reset()` | 停止 timeline；managed 路径重建物理；执行 `_newton_reset`；`physics_steps=0`、`_initialized=True` |
| `play()` | 必要时 reset；direct 保持 timeline 停止，其他路径 play；`_playing=True` |
| `stop()` | timeline stop；`_playing=False`、`_initialized=False`，下次 play 需要重建 |
| `step(render=False)` | 若正在运行则执行一次物理步并增加计数；不主动刷新画面 |
| `update()` | 临时禁用 `/app/player/playSimulations` 后 `app.update()`，finally 恢复设置 |
| `destroy_world()` | 停止、清回调/模型、关 App，处理部分初始化与原异常保护 |

### 4.2 逐行展开 `step()` 的三条分支

```python
# 简化：保留三种路径共同的调度顺序。
if self.is_playing():
    if self.uses_direct_newton:
        self._timeline_action("stop")
    else:
        self._manager.step(steps=1, update_fabric=...)
    if self._newton_stepper is not None:
        self._newton_stepper(self.config.physics_dt)
    self.physics_steps += 1
if render:
    self.update()
```

- PhysX/受管理 Newton：物理管理器推进；额外回调是否合法由注册检查限制。
- Newton direct：跳过 manager，调用任务注册的 stepper。
- hybrid：manager 先推进 PhysX，再调用 Newton stepper；顺序决定柔体使用哪一刻的机器人碰撞代理。

`physics_steps` 是外层步计数，不是 VBD/MuJoCo 原生子步数。
调用者执行 `world.step()` 后再手动 `runtime.step()` 会有重复积分风险，读代码时必须数清这两个入口。
`update()` 不推进物理的保证来自设置与 finally，而不是 `app.update()` 名称；直接调用 App 时要重新分析 timeline 状态。

练习：在纸上分别模拟 direct 与 hybrid 的一次 `step(render=True)`，列出 manager、Newton callback、App update 的调用次数与顺序。

## 5. 桌面抓放：完整追踪五个组件

配置只用于定位对象：[franka_physx_home000.toml](../configs/tabletop_pickplace/franka_physx_home000.toml)。
接下来沿实际方法阅读，不需要先启动一次高清采集。

### 5.1 Scene：引用资产、计算支撑高度、建立状态接口

入口：[TabletopHomeScene](../real2sim2real/scene/scene_tabletop_pickplace.py#L122)。

| 方法 | 读取 → 修改/返回 | 应精读的实现 |
|---|---|---|
| `init_scene(world)` | 配置 USD/Prim → stage 引用、`capture_prims`、`sensors` | 延迟导入 pxr；验证刚体子 Prim 存在 |
| `arrange()` | 几何 AABB、桌面位姿 → `layout`、`_nominal`、`poses` | 找真实桌面顶，按物体最低点修正 Z，按需覆盖质量 |
| `accept_offset(xy)` | 候选目标偏移 → bool | 目标脚印不能出桌边，与障碍脚印保持间隔 |
| `set_target_offset(xy)` | 名义位姿 + 偏移 → 新位姿 | 检查 timeline 已停止；只移动目标，供每集 reset 前使用 |
| `initialize_runtime()` | USD 与当前物理模型 → `RigidPrim` views、中心偏移 | reset 后创建 views，计算刚体系下的中心偏移 |
| `target_pose()` / `target_center()` | 物理 view → 刚体原点位姿 / 几何中心 | 二者不是同一坐标点 |

`arrange()` 先把物体原点摆在 `table_top`，量出实际 `lo.z`，再加上 `table_top - lo.z + clearance`。
因此不是假设资产原点就在底面；`layout["objects"]` 同时保存 Prim 原点、body 原点、AABB 中心与半尺寸。

`initialize_runtime()` 的关键坐标计算为：

```text
center_offset_body = R(q_body_initial).T @ (center_initial - p_body_initial)
target_center_now  = p_body_now + R(q_body_now) @ center_offset_body
```

后续 Planner 和验收读取 `target_center()`，而不是每次把 USD Prim 原点当作抓取中心。
练习：假设资产刚体原点在木块一个角，旋转 90° 后，比较上式与“只加初始世界偏移”的差别。

### 5.2 Robot：摆放与驱动是两个阶段

入口：[FrankaTabletopRobot](../real2sim2real/robot/robot_tabletop_pickplace.py#L96)。

`set_support()` 接收 Scene 测得的桌高、边界与坐标系，`arrange()` 用 `footprint_shift()` 把底座脚印移回桌面内，
并保存 `base_pose`。这些是加载阶段的 USD 操作。

`initialize_runtime()` 在 reset 后创建 `Articulation` 和手/两指的 `RigidPrim` views，
按关节名称查出 `_arm_idx`、`_finger_idx`，写入一次初始状态。名称索引避免假设 USD 原生 DOF 顺序恒定。

重点读 [command / joint_state](../real2sim2real/robot/robot_tabletop_pickplace.py#L196)：

```text
command(arm_q[7], finger_m, arm_qd[7])
  → indices = 7 个臂索引 + 第一指索引
  → set_dof_position_targets([arm_q, finger_m])
  → set_dof_velocity_targets([arm_qd, 0])

joint_state()
  → 按 7 臂 + 2 指索引读取实际 q、qd，各 9 维
```

`action_names` 只有 8 项，因为第二指由 mimic 跟随；`recording_joint_names` 有 9 项。
`command()` 写驱动目标，`initialize_runtime()` 才写初始关节状态，二者 API 的物理含义不同。
注意臂速度前馈实际也下发了，但 LeRobot `action` 向量记录的是位置目标与一指目标，没有包含这 7 个速度。

### 5.3 Planner：样本驱动的状态机

入口：[CumotionPickPlacePlanner](../real2sim2real/planner/planner_tabletop_pickplace.py)。按这个函数顺序读：

```text
init_planner → build → begin_episode → _reset_episode
on_physics_step → _enter → _hand_target → _plan → _solve
                                       ├── _solve_free → _ik → TrajectoryOptimizer
                                       └── _solve_linear → 直线路径转换 → 时间参数化
                 → robot.command → _advance
```

`init_planner()` 只保存 Robot。`build(scene, robot, dt)` 才导入 cuMotion、加载机器人描述，
将世界中的桌面/障碍转入机器人基座系，创建轨迹生成器、优化器配置与碰撞检查器。
轨迹生成器的速度/加速度/jerk 限值都乘 `linear_speed_scale`，不只是把输出数组缩短。

| 成员 | 类型/形状 | 谁写，谁读 |
|---|---|---|
| `phase` | 字符串 | `_advance()` 改，`_enter()` 决定该阶段做什么 |
| `_samples` | `(positions[N,7], velocities[N,7])` 或 None | `_enter()` 构造，每步按 `_i` 取一行 |
| `_i` | 阶段内样本索引 | 每次下发后 +1，切阶段置 0 |
| `_q_cmd` | `(7,)` | 上次命令/规划端点，用作后续规划起点和保持目标 |
| `_finger` | 单指位置目标 | close/open 阶段切换，运动阶段保留 |
| `grasp_offset_m` | `(3,)` | 进入 lift 时由实测 hand-center 计算，用于放置目标 |
| `state["last_action"]` | `(8,)` | 下发当前样本后写入，Job 在同一轮读取 |

`_enter()` 不是每步都规划：只有 `_samples is None` 才进入。
start/close/open/rest 生成固定关节目标和零速度的保持数组；运动阶段才调用 `_plan()`。
运动轨迹末尾再拼接 settle 保持段；使用 `positions[1:]` 跳过重复起点。

两种规划路径必须分开读：

- `_solve_free()`：IK 求目标关节 → 检查目标碰撞 → 关节空间优化 → 按 `dt` 采样；`time_scale` 拉长时间，速度除以同一倍率。
- `_solve_linear()`：从当前末端姿态构造任务空间直线 → 转关节路点 → 生成带时间轨迹 → FK 重算末端位置，检查横向偏差、终点误差，并抽样检查碰撞。

直线偏差的计算是 `offset - outer(offset @ axis, axis)` 的模长最大值。
理解这行 NumPy：它从每个点到起点的向量里减去沿直线方向的投影，留下垂直分量。

`on_physics_step()` 下发一个样本后才递增 `_i`，达到末尾立即 `_advance()`。
因此在阶段边界，`last_action` 是刚下发的旧阶段末样本，而 `state["phase"]` 可能已经变为下一阶段。
跟踪一条数据时要保留这个代码顺序，不要仅凭 phase 字符串推断该动作首次在哪一阶段生成。
返回 `True` 同时覆盖 done 和 failed；任务成功要读 `result()` 与物理验收。

### 5.4 Job：一轮循环精确对应哪些数据

入口：[TabletopPickPlaceJob._run_episode](../real2sim2real/job/job_tabletop_pickplace.py#L216)。先看钩子：

```text
before_place_robot_hook  → scene.layout → robot.set_support(...)
after_place_robot_hook   → planner.build(scene, robot, physics_dt)
before_start_simulation_hook → ContactMonitor(stage, bodies, classify)
simulate                → episode_plans() → 每集 _run_episode()
```

`episode_plans()` 用单个 RNG 连续采样偏移，并通过 `scene.accept_offset` 拒绝非法布局。
每集 `_run_episode()` 停止世界、设置目标偏移、reset/play、初始化两个组件的物理 views，再 `begin_controller()`。

```python
# 简化自 _run_episode()；q_t 是步进前的实测状态。
done = controller.on_physics_step(self.robot)       # 选择并下发 u_t
row = self._log_row(log, step, controller)          # 读 q_t、u_t、物体/手位姿
if recorder is not None and step % self.image_stride == 0:
    self._capture(recorder)                        # 该物理时刻的各相机
    recorder.add_frame(row["q"], row["action"], self._extras(row, controller))
if done or reached_limit:
    break
world.step(render=False)                          # 物理得到下一状态
step += 1
```

`_log_row()` 产生高频字典：`q[9]`、`qd[9]`、`action[8]`、`object_pos[3]`、`object_quat[4]`、
`hand_pos[3]`、`hand_quat[4]`、`finger_q[2]`、`distractor_pos[K,3]` 等。
循环后每个 list 转为数组，因此 `q` 变成 `(N,9)`，接触和布局单独保存，再传 `evaluate()`。

这份数据对应 `(observation_t, action_t)`，不是“action_t 执行后的 observation”。
达到 done 的末轮会记录，但不会再推进一次物理。回放结束边界与源数据比较依赖这个细节。

### 5.5 接触与验收：从 callback 变成纯数组判断

入口：[ContactMonitor](../real2sim2real/utils/tabletop_pickplace_utils.py#L239)、
[evaluate_pickplace](../real2sim2real/utils/tabletop_pickplace_utils.py#L349)。

```text
PhysX contact event
 → _callback(headers, data)
 → actor ID 转 Prim path → classify → 类别（target/table/finger/...）
 → 类别排序组成 "a|b" key
 → 对接触点冲量取模求和
 → add() 更新总量、分阶段量、首次/最后发生时间
 → Job evaluate() → evaluate_pickplace(log, layout, contacts, ...)
```

`evaluate_pickplace()` 不负责推进仿真，适合独立精读与构造输入验证。
它先用 `phase_index` 建阶段 mask，再对持握期、撤离期、末段静止窗口分别计算指标。

最值得自己推一遍的是滑移：将物体中心转入每帧手坐标系，计算相对首帧的最大位移。
用世界坐标位移会把正常转运也算作滑移；这里通过 `relative_position()` 去掉手的刚体运动。
另一处是苹果扰动：从首次机械臂运动起比较，start 保持期的自然落稳单独记为 settle 指标。

`both_fingers_contacted_target` 来自接触累计，不由“指目标已闭合”推断。
`sequence_completed` 来自 Planner；最终 `success = all(checks.values())`，两条证据在这里汇合。

## 6. 桌面回放：通过覆盖方法复用父类算法

入口：[TabletopReplayJob](../real2sim2real/job/job_tabletop_replay.py)、
[LeRobotReplayPlanner](../real2sim2real/planner/planner_tabletop_replay.py)、
[tabletop_replay_utils.py](../real2sim2real/utils/tabletop_replay_utils.py)。

### 6.1 不重写 `_run_episode()`，改变它调用的几个方法

| 覆盖点 | 父类行为 | Replay 行为 |
|---|---|---|
| `after_place_robot_hook()` | `planner.build(...)` | 仍调用 super，但实际 Planner 为 Replay，加载源集；额外检查 fps |
| `episode_plans()` | 随机抖动布局 | 从每个 `SourceEpisode.layout` 取原偏移和源索引 |
| `begin_controller()` | 开始 cuMotion 抓放 | 检查桌高/基座/目标起点，绑定当前 SourceEpisode |
| `evaluate()` | 抓放物理检查 | super 后追加源轨迹一致性，重新汇总 success |

这就是阅读继承代码的关键：调用方没有变，`self` 的动态类型改变了被调用的方法。
`TabletopHomeScene`、`FrankaTabletopRobot` 也直接复用，只有配置中的任务资产根路径不同。

### 6.2 源数据类型与插值公式

`SourceEpisode` 是 dataclass，包含 `actions[T,8]`、`states[T,9]`、`phases[T]`、`object_pose[T,7]`、
`layout`、`fps`、`physics_dt` 及坐标名称。`load_source_episode()` 从 Parquet 和 meta 中装配它。
`build()` 检查物理 dt、关节名称、源 success 与采样整除关系；这一步没有 cuMotion 调用。

精读 `replay_command(actions, step, stride, fps, interpolation)`：

```text
k     = min(step // stride, T-1)
alpha = (step - k*stride) / stride
linear: target = actions[k] + alpha*(actions[k+1]-actions[k])
        arm_velocity = (actions[k+1,:7]-actions[k,:7]) * fps
hold:   target = actions[k], arm_velocity = 0
```

末帧无下一动作时保持目标、速度为零。指目标也插值，速度前馈只取前七维臂关节。
Planner 在 `step >= (T-1)*stride` 时结束，Job 因而在源最后一帧时刻记录后停止。
`evaluate()` 用 `frames = arange(T)*stride` 从高频回放 `q` 抽取与源同一时刻的状态，传 `replay_fidelity()`。

代码练习：手算 `stride=4`、第一臂关节两帧值为 `0.0→0.4` 时 step 0/1/3/4 的位置与速度目标；
对照测试 `test_replay_command_hold_and_linear`，再解释为什么同一 fps 仍要单独检查 `physics_dt`。

## 7. 公共工具：从物理接口到文件格式

### 7.1 `LeRobotV3Recorder` 的提交边界

入口：[utils/lerobot_v3.py](../real2sim2real/utils/lerobot_v3.py)。按 `__init__ → frame_path → add_frame → finalize → _encode_video` 阅读。

`__init__()` 检查目标不存在、分辨率为正偶数、坐标名不重复、ffmpeg 可用，创建同父目录下的 `staging`。
录制过程中的图像和 Parquet 都先写 staging，成功收尾才 rename 到 root。

`frame_path(camera)` 根据 `len(rows)` 给出当前帧文件名，因此 Job 必须先把所有相机图像写到对应路径，再 `add_frame()`：

```text
rows 长度为 k
 → 每台相机 frame_path(name) 对应第 k 帧
 → Job 实际截图
 → add_frame(state, action, extras)
    ├── state/action 转 float32，校验一维长度和有限性
    ├── extras 与 feature 定义逐项匹配
    ├── 检查所有相机图像存在且尺寸正确
    ├── 更新 RGB 统计
    └── append 一行；timestamp = k/fps；下一次 frame_path 进入 k+1
```

`finalize()` 编码每路视频，使用显式 Arrow schema 写固定长度向量列，生成 features、统计和 episode 元信息，
删除临时帧目录，最后再次检查 root 不存在并 rename。
它保证文件提交完整性；任务成功由调用方传入 `episode_success`，recorder 本身不计算抓放物理是否成功。
允许保存失败集的任务会导出 diagnostic，不能把“目录成功生成”解释为任务通过。

精读 `_extra_values()`：`shape=[1]` 被转成标量，其他长度变为 list；这影响 Parquet 读取后的维度。
`update_extra()` 允许结束时补写最后一帧 `next.success`，写回前仍走同一套 extras 校验。

### 7.2 截图为何不会偷偷推进物理

入口：[capture_viewport_frame](../real2sim2real/utils/lerobot_v3.py#L382)。它会设置相机与尺寸，计算期望 view matrix，
在受控 PathTracing 下管理累积历史，并等待实际 rendered-frame 事件。
等待循环调用 `world.update()`，不是 `world.step()`；结束时核对物理计数与相机状态。

顺着 `wait_rendered_frames()` 内部看：它创建异步等待任务，但通过有限次数的 Kit update 驱动事件完成，
超时取消任务并报错。调用 `app.update()` 的次数不等于 GPU 已经完成的渲染帧数。
桌面并行相机路径还会使用 `BatteryCameraCapture`，进入该类看固定 RenderProduct 与各路 annotator 的采集/关闭方法。

### 7.3 其他公共文件怎样精读

| 文件/符号 | 代码关注点 |
|---|---|
| [asset_utils.resolve_usd_path](../real2sim2real/utils/asset_utils.py) | 绝对/相对路径如何解析；仓库根、工作目录和安装包环境的区别 |
| `asset_utils.prim_aabb` | `BBoxCache`、遍历 instance proxies、忽略不可见/空几何、逐节点 min/max 合并 |
| `asset_utils.read_prim_world_pose` | 读取世界矩阵，提取平移与 wxyz；失败返回 None 的调用方如何处理 |
| [cameras.CameraConfig / create_cameras](../real2sim2real/utils/cameras.py) | 配置校验、相机名去重、`pose_relative_to` 与 Prim 父子关系、内参写入 |
| [kit_process.run_kit_worker](../real2sim2real/utils/kit_process.py) | 外层进程与 Kit worker 的分工；用本次结果标记校验退出，避免只看进程返回码 |
| [core/logger.LoggerMixin](../real2sim2real/core/logger.py) | 实例日志怎样绑定 component；日志配置怎样从 settings 注入 |

## 8. 机器狗巡检：事件状态机怎样连接导航和拍照

### 8.1 五个组件的代码切入点

| 组件 | 源码入口与必读方法 |
|---|---|
| SimWorld | 共用 `IsaacsimWorld`：PhysX 管理器步进，Flow 另有时间同步 |
| Scene | [FireWarehouseScene](../real2sim2real/scene/scene_fire_inspection.py)：`init_scene`、`arrange`、`fire_occluder`、`start_fire_clock`、`sync_fire_time` |
| Robot | [X30Robot](../real2sim2real/robot/robot_fire_inspection.py)：`initialize_runtime`、`set_base`、`after_physics_step`、`camera_matrix` |
| Planner | [FirePatrolPlanner](../real2sim2real/planner/planner_fire_inspection.py)：`set_map`、`sight`、`_follow`、`on_physics_step`、`photo_taken` |
| Job | [FireInspectionJob](../real2sim2real/job/job_fire_inspection.py)：`build_occ`、`after_arrange_scene_hook`、`simulate`、`_physics_step`、`_take_photo` |
| 纯算法 | [fire_inspection_utils.py](../real2sim2real/utils/fire_inspection_utils.py)：`OccupancyGrid.plan`、`scan_physx_occupancy`、`project_point`、`fire_pixel_score` |

### 8.2 从地图数组跟进 A*

`build_occ()` 调用 `scan_physx_occupancy()`：在 `rows × cols` 网格上遍历，每格用 overlap box 查机器人高度带是否占据，
再向下 raycast 检查是否有地面，返回 `OccupancyGrid`。机器人此时尚未加入世界，避免把自身扫成障碍。

`OccupancyGrid.plan(start, goal, radius, margin)` 的局部变量值得逐个画出来：

```text
hard = inflate(radius)                # 禁行格
soft = inflate(radius+margin) & ~hard # 可行但代价较高
best[cell]                           # 起点到此格的最低已知代价
parent[cell]                         # 路径回溯指针
queue                                # heapq，优先级为 g + 欧氏启发距离
```

找到终点后沿 parent 回溯，格子转世界坐标，再 `_shortcut()` 做视线平滑。
这是一个能脱离 Kit 单独研究的算法单元；重点不是背 A* 名称，而是核对 row/col 与 x/y 的转换。

### 8.3 Planner、Job 的双向事件交互

[on_physics_step](../real2sim2real/planner/planner_fire_inspection.py#L275) 的状态分支：

```text
patrol → approach → face → settle → photo → hold → alarm → done
   └── not_found                     └── 失败也可能提前结束
```

`visible_steps` 连续满足可见性才触发接近；`sight()` 合并投影、距离和遮挡检查。
`settle` 写 `state["photo_request"]=True`，Job 调 `_take_photo()` 保存并检查照片，再通过 `planner.photo_taken(info)` 回传结果。
Planner 下一轮看到 `self.photo` 才进入报警。这是明确的双向调用，不是 Planner 自己保存照片。

Job 的 `photo_pending` 还会把物理步之间的一次拍照事件保留到下一张数据帧，随后清零。
否则 120 Hz 的瞬时事件可能落在两张 15 fps 视频帧之间，数据 action 就会丢掉该事件。

### 8.4 `set_base()` 写了哪些物理量

[X30Robot.set_base](../real2sim2real/robot/robot_fire_inspection.py#L461) 每步：

1. 写参考机身位姿；将局部 `(vx, vy)` 旋转为世界线速度，并写角速度。
2. `TrotGait.feet(command, dt)` 生成四脚目标，用 `leg_ik()` 转换为关节角。
3. 差分相邻目标得到速度前馈，通过 `ArticulationAction` 写腿部驱动。

Planner 同时维护 `base_xy/base_yaw` 的参考积分，读实测位姿只统计 drift；所以这里是运动学机身控制。
`camera_matrix()` 将安装偏移与实测基座矩阵相乘；该文件使用的 USD 矩阵约定要从乘法顺序跟读，不能套用另一工具库的列向量公式。

Job 与桌面循环的不同在于：采样步先捕获观测与图像，再调用 Planner，最后把同一观测对应的动作写入 recorder，随后 `_physics_step()`。
`_physics_step()` 除了推进世界，还统计腿跟踪误差并同步火焰时钟。

代码练习：在 `test_patrol_finds_fire_and_photographs` 中看 fake robot 的最小接口，
找到一次 `photo_request` 如何变成 `planner.photo`，解释为什么两者不能用同一个布尔字段替代。

## 9. 物理求解器任务：Planner 与 Runtime 的两层设计

本任务源码较大，按类族阅读：[Scene](../real2sim2real/scene/scene_physic_solver.py)、[Robot](../real2sim2real/robot/robot_physic_solver.py)、
[Planner/Runtime](../real2sim2real/planner/planner_physic_solver.py)、[Job](../real2sim2real/job/job_physic_solver.py)、
[utils](../real2sim2real/utils/physic_solver_utils.py)。

### 9.1 先区分“生成控制”与“推进模型”

```text
PhysicSolverJob
  → planner.configure_scene(scene)
  → register_newton_stepper(job._step_newton, reset=job._reset_components)
  → world.reset()
      → _reset_components()
          → 组件 initialize_runtime（如果存在）
          → planner.begin_episode(register_stepper=False)
              → 创建具体 runtime

每一轮：planner.on_physics_step(robot)  # 测量、选阶段、改控制目标
        world.step(render=False)
          → job._step_newton(dt)
              → planner.runtime.step(dt) # 真正积分与发布
```

`_step_newton()` 每次从 `self.planner.runtime` 取当前对象；reset 重建 Runtime 后无需让世界继续引用旧实例的 bound method。
`register_stepper=False` 避免 begin_episode 再注册一次。
刚体分支则使用组件初始化与 PhysX，阅读 `object_kind` 判断，不要假设所有配置都进入 Newton。

### 9.2 类族对应的是算法差异

| 对象/变体 | Scene / Robot | Planner / Runtime | Job |
|---|---|---|---|
| 普通衣服 | `PickPlaceClothScene`、`FrankaNewtonRobot` 或 `FrankaPhysxRobot` | `PickPlaceClothPlanner` / `NewtonClothRuntime` | `PhysicSolverJob` |
| PhysX 方块 | `PickPlaceRigidScene`、`FrankaPhysxRobot` | 同一抓放 Planner 的 rigid 分支 | `PhysicSolverJob` |
| 官方衣服 | `OfficialClothScene`、`FrankaNewtonRobot` | `OfficialClothPlanner` / `OfficialClothRuntime` | `PhysicSolverJob` |
| 衣物袋 | `PackageScene`、原装 Franka 或 `PackagePadRobot` | `PackagePlanner` / `PackageRuntime` | `PhysicSolverJob` |
| 纸箱 | `CartonScene`、`CartonGripperRobot` | `CartonPlanner` / `CartonRuntime` | `CartonJob` |
| 导向主轴 | `ScrewScene`、`ScrewSpindleRobot` | `ScrewPlanner` / `ScrewRuntime` | `ScrewJob` |
| 单 UR5e | `ScrewScene`、`Ur5eScrewdriverRobot` | `ScrewArmPlanner` / `ScrewRuntime` | `ScrewJob` |
| 双 UR5e / 穿出 / 松爪 | `DualScrewScene`、`DualUr5eScrewRobot` | `DualScrewPlanner` / `DualScrewRuntime` | `ScrewJob` |

同名任务文件中也定义对应 Config；`runtime_cls` 与 `config_cls` 是两个不同的类级选择点。
普通抓放用阶段表，官方衣服用关键帧，包裹重写调度；不要从共同继承关系推断它们执行同一条轨迹。

### 9.3 机器人构建与 IK

`FrankaNewtonRobot.add_newton_robot(builder, scale=...)` 向 ModelBuilder 添加机器人；
`FrankaPhysxRobot.initialize_runtime()` 创建 Isaac 物理句柄，其 `command_tcp()` 用 PhysX 机器人路径下发。
`PackagePadRobot`、`CartonGripperRobot` 在 `add_newton_robot()` 覆盖几何/关节构建；UR5e 还提供 `screw_control()` 或 `dual_control()`。

精读 `_URDFHandIK._fk()` 与 `solve()`：前者累乘关节变换，记录每个轴与原点，构造 6×7 Jacobian；
位置部分为 `cross(axis, tcp-origin)`，姿态部分为 axis。
`solve()` 由位置误差与旋转误差组成 6 维 error，用阻尼最小二乘：

```text
dq = J.T @ solve(J @ J.T + 1e-4*I, error)
q  = clip(q + clip(dq, -0.15, 0.15), lower, upper)
```

内部 `self.q` 保留上次解，下一次求解从附近继续；这影响 IK 分支连续性。
这是项目自己的数值 IK，与桌面任务调用 cuMotion 的实现不同。

### 9.4 `NewtonClothRuntime` 的双缓冲与耦合

依次读 `__init__ → command → step → _advance_direct → _publish`。
构造中建立 `model`、`state_0`、`state_1`、`control`、`rigid_solver`、`cloth_solver`、`pipeline`、`contacts`。
`state_0` 是当前状态，`state_1` 是本子步输出，求解后交换引用，避免每次分配整个状态。

| 方法/分支 | 代码作用 |
|---|---|
| `command()` | 修改 TCP/夹指目标；pure Newton 用 IK 改 control，hybrid 转发到 PhysX Robot |
| `step()` 的 hybrid 分支 | 读取本帧/上帧 PhysX 碰撞代理，按子步插值，交给 `KinematicVBDStepper` |
| `_advance_direct()` | 分割 frame_dt；推进刚体、碰撞检测、柔体求解、交换状态 |
| 临时 `model.particle_count=0` | 刚体求解时屏蔽粒子；finally 恢复，避免把同一批粒子交给错误路径 |
| CUDA graph 分支 | 首次 warmup 后捕获固定推进序列，后续 launch；目标缓冲仍可更新 |
| `_publish()` | 读取求解结果，更新可视网格/机器人，并统计实际接触 |

`scale`、Warp 的 xyzw 与项目接口 wxyz 都在数据边界转换，不能把模型坐标直接当成 Scene 世界坐标。
Scene 的 `publish_cloth_vertices()` 和 Robot 的视觉发布是 Newton 状态到 USD 的单向输出路径，不是通过 USD 反向积分。

`physic_solver_utils` 中继续读 `VBDSubstepGraph`、`KinematicVBDStepper`、`moving_frame_vbd_class()`：
最后一个在首次调用时定义依赖 Newton 的类，Warp kernel 也由函数延迟返回，从而保持模块顶层轻量。

### 9.5 衣服与包裹控制的实际数据结构

`PickPlaceClothPlanner.begin_episode()` 建 `_phases`，从三角形提取唯一边与初始边长，用于后续形变比。
`OfficialClothPlanner.on_physics_step()` 按累计关键帧时间选行，再 `runtime.track_keyframe()`；
释放指标跟踪开爪时接触的同一组粒子，避免后续换一片最高顶点造成虚假的释放结论。

[PackagePlanner.begin_episode](../real2sim2real/planner/planner_physic_solver.py#L2076) 创建：

```text
_schedule[i] = (phase, duration_s, tcp_target[3], closed)
_ends = cumsum(duration_s)
index = searchsorted(_ends, elapsed, side="right")
```

可选 hold/shake 插入表中。每步将 vertices 按 `scene.garment_count` 切成衣物与袋膜，
以 particle_mass 加权求两个 COM，累计双侧接触、边长比、穿入和释放变化。
`_align_placement()` 修正后续控制目标；验收目标仍按原布局计算，控制补偿与验收标准是不同数据。

`PackageRuntime.prepare_cpu_rigid_frame()` 将一帧原生 MuJoCo 子步的 qpos/qvel/actuator force 先在 CPU 连续算好，
批量上传缓冲，`_advance_direct()` 再逐子步消费并推进 VBD。
这是单向耦合路径的调度优化；开启 coupled solver 时转 `_advance_coupled()`，用代理反馈、实测电机力与代理位姿误差检查另一条路径。

### 9.6 纸箱与真实螺纹是独立 Runtime

`CartonRuntime` 虽继承 `PackageRuntime`，但其构造器自建刚体纸箱、原生 MuJoCo 接触，没有柔体模型。
读构造器比只看继承图更重要。`CartonPlanner` 根据解算姿态把八个角点转到世界系，记录最低点，
不能仅用纸箱中心 Z 判断支撑和穿入。

[ScrewRuntime](../real2sim2real/planner/planner_physic_solver.py#L2466) 的关键链：

```text
Scene.fastener_meshes
 → newton.Mesh.build_sdf()
 → hydroelastic ShapeConfig + CollisionPipeline
 → SolverMuJoCo(use_mujoco_contacts=False, cone="elliptic", ...)
 → command(): 主轴速度 + 轴向弹簧预载
 → step(): collide → solver.step → swap（每子步）
 → publish(): 解算 body_q 转 USD
```

这里没有按螺距写轴向位移；进给来自螺纹接触。`screw_coordinates()` 读取实际自由度给验收。
单臂版本在 Robot 的 `screw_control()` 内生成机械臂/主轴控制，Planner 复用主轴任务框架。

双臂重点读 `DualScrewRuntime._build_assembly()`、`command_dual()`、`DualScrewPlanner.on_physics_step()`：
螺母变为自由刚体，由第二臂夹持；Planner 插值夹爪、持件臂高度和驱动臂高度，只在 screw 阶段施加旋转速度。
测量用螺母系下的螺栓相对姿态/位移；相邻 yaw 差用 `atan2(sin(delta), cos(delta))` 累加为连续转角。
穿出与松爪版本继续复用这些类，由阶段配置改变行为；末段松爪需要重新看相对保持与接触，不能复用夹紧期的成功标志。

代码练习：分别在 `command()` 与 `step()` 设置断点，确认“改变控制缓冲”和“真正改变动力学状态”的位置。
再沿一份 rigid、cloth、package、screw 配置找出实际进入的 Runtime 类与分支。

### 9.7 `newton_diagnostic` 是无 Kit 的真实求解世界

入口：[simworld_physic_solver.py](../real2sim2real/simworld/simworld_physic_solver.py)：`NewtonDiagnosticWorld`。
它继承 BaseSimWorld 而不是 IsaacsimWorld，Config 则继承统一世界配置并将引擎/执行模式限制为 Newton direct。
`init_world()` 用 pxr 创建或打开 USD Stage，检查米制与 Z-up，没有 SimulationApp。

它保存 `_stepper/_reset/_close`，`reset()` 必须已有 stepper，再调用模型构建回调；
`play()` 只改变手动步进状态，`step(render=False)` 直接调用真实 Newton stepper 并增加计数。
`render=True` 明确报错。`is_running()` 判断 Stage 存在，不是检查 Kit 窗口。
这条路径改变的是 App/渲染基础设施，Runtime 仍求解真实物理，不等同于 stub。

## 10. 分拣：继承、适配器与连续状态

源码：[Scene](../real2sim2real/scene/scene_sorting.py)、[Robot](../real2sim2real/robot/robot_sorting.py)、
[Planner/Runtime](../real2sim2real/planner/planner_sorting.py)、[Job](../real2sim2real/job/job_sorting.py)、
[utils](../real2sim2real/utils/sorting_utils.py)。

### 10.1 从覆盖方法看复用边界

```text
PackageScene          ← SortingScene
PackageRuntime        ← SortingRuntime ← SortingStreamRuntime
PickPlaceClothPlanner ← SortingPlanner ← SortingStreamPlanner
                                      ← _RigidRelay
PhysicSolverJob       ← SortingJob
BaseRobot             ← SortingRobot
```

`SortingJob` 文件很短，只覆盖 `before_place_robot_hook()`；主要采集循环继承父类。
反过来，`SortingRuntime` 覆盖模型构建、IK、接触材质、控制与发布，不能用“复用 Package”跳过这些方法。

| 组件/方法 | 要跟踪的变量和行为 |
|---|---|
| `SortingScene.arrange()` | `supports` 中入料带/平台/出料带边界；供抓取高度和释放位置计算 |
| `SortingScene._arrange_cloth()` | 挂载包裹并读取几何；`label_up_cos()` 从标签三角形判断朝向 |
| `SortingRobot.add_newton_robot()` | 原生双臂/夹爪进入 builder；mimic 和胶垫碰撞体来源 |
| `SortingArmIK.forward/solve/solve_pregrasp` | 单臂 FK、求解与预抓搜索；Robot 与 Runtime 怎样共享关节索引 |
| `SortingRuntime.command_arm/set_gripper` | 活动臂、目标数组、左右接触别名切换 |
| `SortingRuntime._advance_direct()` | 原生刚体子步与 VBD 步比例；CPU 批量轨迹上传；mimic 残差记录 |
| `SortingRuntime.publish_recorded_visuals()` | 从已解算记录恢复机器人视觉；供独立渲染进程复用 |

### 10.2 `SortingPlanner`：命令连续性与形变反馈

精读 `_initialize_tracking → _enter → on_physics_step → result`。
`schedule` 每项含 side、phase、duration；`_enter()` 记录 `start_tcp`、抓取基准和本阶段落点。
右臂接近时用当前变形表面重新计算重抓深度，不是把左臂的初始抓点直接复制过去。

注意 `_enter()` 对 `command_tcp` 的使用：同一臂的后续段优先从上一段命令终点连接。
若每段都从滞后的实测 TCP 重建目标，跟踪误差会逐段累积为路径漂移。
这里既读取实测 COM/形状，也保留连续命令参考，二者不是同一个变量。

`StrainGripController.command(nominal_rad, stretch, dt, minimum_rad=...)` 是一个小型有状态控制器：

```text
stretch > target              → relief += rate*dt
stretch < target-hysteresis   → relief -= recovery_rate*dt
其他                          → 保持 relief
relief 限制在 [0, min(maximum_relief, nominal-minimum)]
输出 nominal-relief
```

读取 `relief_rad` 的跨步累积、`hysteresis` 死区和 `minimum_rad`，解释为什么不能改成一次性 if 后直接大幅开爪。
形变指标见 `sorting_utils.aligned_shape_rms()`：先去除整体刚体运动，再衡量形变；
固定基座接触、袋衣间隙、胶垫穿透另有独立审计，不等同于一个 RMS 指标。

### 10.3 流式编排为什么不用 reset 切换物体

[SortingStreamPlanner.on_physics_step](../real2sim2real/planner/planner_sorting.py#L2443) 每轮先记录软包全局几何、
body poses、state，再执行当前物体控制，并记录 action。

软包控制由父类推进；成功后 `runtime.spawn_carton()`，创建 `_RigidRelay`，
将同一个 Robot、Scene、Runtime 传给其 `initialize()`。下一轮开始走 rigid 控制，世界没有重建。

重点看 [适配器](../real2sim2real/planner/planner_sorting.py#L2253)：

```text
_CartonView.vertices                 → shared.carton_vertices
_CartonView.last_contact_counts      → shared.carton_contacts
其他读取                            → shared 同名字段
所有赋值                            → shared 同名字段
```

`__getattr__()` 改读接口，`__setattr__()` 转发写接口；构造时使用 `object.__setattr__(self, "shared", runtime)` 避免递归。
`_CartonScene` 同样替换三角形、几何数量与出料目标，让已有双臂动作控制可用于纸箱。
这是对象适配，不是复制一个物理世界。

全局边长比、最小面积比和地板穿入在纸箱阶段仍继续测软包；局部 `soft_result` 不能覆盖后半程损伤。
代码练习：沿 `stream_states`、`stream_actions` 的 append 顺序推导一条样本的 observation/action 对齐，
再看测试 `test_cached_soft_success_cannot_hide_damage_during_the_later_rigid_relay`。

### 10.4 渲染回放代码与动作回放代码有什么不同

入口：[scripts/sorting/render_sorting_package.py](../scripts/sorting/render_sorting_package.py)：
`replay_indices()`、`prepare_frame_cache()`、`SortingReplayJob`、`run_camera_workers()`。

它按源时间索引恢复粒子和刚体可视状态，然后只渲染，不重新执行机器人 action 来产生物理。
这与第 6 节“把 action 下发给 PhysX 再积分”的回放含义不同。
`ReplayRenderProfile` 与 override 函数限制渲染覆盖范围；缓存还需验证源和配置是否匹配，防止拼接不同来源的帧。

## 11. 电池拆解：程序解释器、工具自由度和传感器控制

源码：[Scene](../real2sim2real/scene/scene_battery_disassembly.py)、[Robot](../real2sim2real/robot/robot_battery_disassembly.py)、
[Planner](../real2sim2real/planner/planner_battery_disassembly.py)、[Job](../real2sim2real/job/job_battery_disassembly.py)、
[utils](../real2sim2real/utils/battery_disassembly_utils.py)。

### 11.1 Scene 把一个电池资产改造成哪些运行时对象

依次读 `init_scene → arrange → _author_cover/_author_case/_author_basket → _author_screws → _author_thread`。
场景渲染网格保留，碰撞几何通过 `_box()` 和若干布局函数创建代理。
`cover_boxes()`、`basket_boxes()`、`screw_head_boxes()` 在 utils 中计算可独立测试的几何。

[_author_thread](../real2sim2real/scene/scene_battery_disassembly.py#L541) 建立：

```text
world --fixed--> anchor --prismatic slide--> slider --revolute turn--> carrier
                                                                      |
                                                         breakable fixed hold
                                                                      |
                                                                 screw body
```

`slide` 上的 mimic 引用 `turn`，代码公式是：

```text
slide + gearing*turn_deg = 0
gearing = -pitch_m / 360
所以 slide = pitch_m * turn_deg / 360
```

转角上限设为螺纹长度对应的圈数；到止挡后扭矩可触发 hold 断裂。
`update_threads(step)` 只读 slide 与螺丝/carrier 分离距离，更新 `engaged` 和 `disengaged_step`，没有在循环中删关节。
这与上一节 M20 接触自发决定导程的模型不同，差异直接体现在这两个函数的代码里。

### 11.2 Robot 的工具关节与数据接口

精读 `Kr210DualRobot._author_screwdriver()`、`_author_gripper()`，再读运行时接口：

| 方法 | 写入或返回 |
|---|---|
| `initialize_runtime()` | 两臂 Articulation、工具和指端 views、DOF 索引、初始状态 |
| `set_arm_targets(key,q,qd)` | 某臂 6 维位置与速度驱动目标；保存 `targets[key]` |
| `set_screwdriver(float_target, spindle_velocity)` | 浮动棱柱位置目标、旋转主轴速度目标 |
| `set_fingers(closing_m)` | 两指目标分别为 `-closing_m` 与 `+closing_m` |
| `read_state()` | 两臂 q/qd、浮动行程、连续主轴角度与速度、两指位置 |
| `tip_pose()` | 批头位置与轴向，给插入和磁力计算使用 |

`read_state()` 将主轴原始角度差包到 `[-π,π)`，累加到 `spindle_angle`，处理跨 ±π 边界的多圈转动。
当一轮传感器差分恰好接近整圈时会有采样歧义；理解该算法依赖足够高的采样频率。

### 11.3 Planner 是两个阶段程序的解释器

先读 `Phase`、`ArmPlanner`、`plan_screwdriver()`、`plan_gripper()`，再读 `BatteryUnscrewPlanner`。
`ArmPlanner` 使用 utils 的 `SerialChain`、`solve_ik`、`solve_line`、`TimedPath`，把名义工具路径变为关节路径。
`build()` 尝试两种夹爪 yaw，保留可行且总路径较短的程序；它完成规划，不推进仿真。

```text
programs["screw"/"grip"]   → list[Phase]
active[key]                → 当前程序指令索引
phase_t[key]               → 当前指令内时间
entered[key]               → 是否执行过一次性进入逻辑
hold[key]                  → 上一命令/保持关节目标
blend[key]                 → 接续段 TimedPath 或 None
```

`on_physics_step()` 读一份 sensors，推进 screw 程序；只有它先前已经结束才推进 grip，否则第二臂保持。
再统一写工具、两指与磁力，更新 `last_action` 和 phase；两程序都结束后继续 settle 一段时间。

[_run](../real2sim2real/planner/planner_battery_disassembly.py#L425) 是核心解释器：

- `path`：可选 blend 接续，然后 `TimedPath.sample(t)` 得到 q 与 qd。
- `seek`：慢转主轴；浮动行程持续满足阈值后确认插入。
- `unscrew`：用浮动压缩量决定机械臂上提跟随速度。
- `release/close/open/wait`：按持续时间完成，close 末尾额外检查是否夹空。

`_enter()` 只执行一次的副作用包括开磁力、选当前螺丝、设置旋向和两指目标。
`_exit()` 停主轴、撤预载并记录事件；随后索引 +1、阶段时间清零、entered=False。

### 11.4 拧松反馈的逐行公式

[_unscrew](../real2sim2real/planner/planner_battery_disassembly.py#L523) 读取的是机器人侧传感器：

```text
compression = max(0, -float_sensor)
speed       = min(follow_max_speed, follow_gain*compression)
follow_s    = min(limit, follow_s + speed*dt)
x           = follow_s / follow_step_m
q           = (1-alpha)*follow[i] + alpha*follow[i+1]
qd          = (follow[i+1]-follow[i]) / follow_step_m * speed
```

满足实际最大抬升和主轴累计圈数才完成，超时 `_fail()`。
它不是按 nominal pitch 直接把工具每圈抬高固定距离，而是让浮动行程反馈调节跟随。
`_apply_magnet()` 把吸力施加给螺丝，并在主轴施加反作用力；去读 `magnet_force()` 看距离、方向与上限如何组合。

Job 的 `simulate()` 顺序是 Planner → `scene.update_threads()` → 高频日志 → 按采样率采图/写行 → 物理一步。
`_frame()`、`_extras()` 和 `_log_row()` 的字段不完全相同，沿三者核对哪些量进入低频数据，哪些只存高频证据。
`_evaluate()` 再读取完整记录和接触，不以程序索引走完作为唯一成功条件。

代码练习：把一次 `_run("screw")` 的 entered、phase_t、active 值逐轮写出来，
分别模拟正常 seek、seek 超时、path 起点偏离过大的分支，指出各分支是否还会下发本轮目标。

## 12. HALO：模型输入、关节重排和多频率闭环

源码：[Scene](../real2sim2real/scene/scene_halo.py)、[Robot](../real2sim2real/robot/robot_halo.py)、
[Planner](../real2sim2real/planner/planner_halo.py)、[Job](../real2sim2real/job/job_halo.py)、
[utils](../real2sim2real/utils/halo_utils.py)。

### 12.1 三个子任务共用的组件边界

| 组件 | 精读方法 | 数据流 |
|---|---|---|
| `HaloHomeScene` | `_static_colliders`、`_add_desk`、`_check_seat`、`update_third_person` | 创建静态碰撞；提供桌/座/相机几何 |
| `G1SonicRobot` | `set_initial_qpos`、`initialize_runtime`、`read_state`、`set_body_targets` | 初始化状态一次；每控制周期读取真值、写电机目标 |
| `SonicTrackingPlanner` | `init_planner`、`reset`、`warmup_step`、`on_physics_step`、`_seated_hold` | 参考与实测状态进入 Tracker；输出调理后的目标 |
| `HaloJob` | `before_start_simulation_hook`、`_physics_period`、`simulate`、`_measure`、`_evaluate` | 管初始化顺序、多频率记录、接触和几何验收 |
| 共用世界 | `IsaacsimWorld` 的 PhysX 路径 | 500 Hz 物理积分，控制周期每次推进 10 步 |

crawl/crouch/sit 通过参考、场景和验收配置改变行为；不是三份独立 Planner。
Scene 负责支撑物与桌/座几何，Job 的验收调用 utils 的 passage、centering、under_desk、seated 等函数。

### 12.2 Robot 数据顺序先于策略数学

`G1SonicRobot.initialize_runtime()` 从 `dof_names` 建 `body_index`，将 29 身体关节映射到原生全部关节。
`read_state()` 返回：

```text
qpos[36]      = base_xyz[3] + base_wxyz[4] + body_joint_pos[29]
joint_vel[29] = 同一身体关节顺序的速度
base_vel[6]   = pelvis 世界线速度[3] + 世界角速度[3]
```

`set_body_targets(target[29])` 从 `default_targets` 复制完整目标，按 body_index 替换身体项，手指保持原目标。
这是驱动目标写入；初始化后没有逐帧把基座贴到参考路径。
`joint_limits` 保留 PhysX float32 精度，后续限位裁剪也因此受 dtype 影响。

### 12.3 `SonicTracker.step()` 的完整输入输出链

精读 [SonicHistory / SonicTracker](../real2sim2real/utils/halo_utils.py#L151)，尤其两组关节排列数组
`MUJOCO_TO_ISAACLAB` 与 `ISAACLAB_TO_MUJOCO`。它们是排列转换，不能把名字当作当前引擎一定为 MuJoCo。

```text
参考 qpos
 → resample_reference()：关节线性插值、基座姿态重采样
 → 未来 ids = clip(frame + arange(10)*5, 0, T-1)
 → 参考关节位置/速度 + 相对当前基座朝向的 anchor 6D 姿态
 → 按 ENCODER_ORDER 拼接，float32、增加 batch 维
 → ONNX encoder → token

实测 qpos / joint_vel / 世界角速度
 → 角速度与重力转当前基座系
 → 关节重排，位置减默认角度
 → SonicHistory.push()，保存最近 10 控制帧
 → token + history.decoder_input()
 → ONNX decoder → 标准化动作[29]
 → 反向重排、乘 ACTION_SCALE、加 DEFAULT_ANGLES
 → 身体关节位置目标[29]
```

`SonicHistory` 五个缓冲形状分别为 `(10,3)`、`(10,29)`、`(10,29)`、`(10,29)`、`(10,3)`。
`push()` 通过 `buf[:-1]=buf[1:]` 左移，再写最后一行；`decoder_input()` 按固定顺序展平拼接，不能任意换序。

Tracker 返回原始目标后，Planner 继续做限位裁剪、滤波、可选坐姿保持，再调用 `robot.set_body_targets()`。
紧接着 `tracker.set_applied_target(target)` 将**实际发送的调理后目标**重新归一化写回 `last`。
否则下一帧策略历史里保存的是网络原始输出，与电机收到的动作不一致。

### 12.4 坐姿保持为什么要读接触

`_seated_hold()` 只有在参考阶段允许、骨盆高度/座心距离合理且实际座面竖直冲量达到条件时，
才记录 `hold_pose` 与 `hold_start_s`；然后用 smoothstep 混合跟踪目标与保持目标。
可选躯干反馈再根据 torso 实测姿态修正腰部关节目标。
因此 sit 不是参考时间到达就固定姿势；触发依赖实际接触与位置。

### 12.5 Job 的循环单位其实是控制帧

虽然接口叫 `on_physics_step()`，HALO Job 每 20 ms 调用一次，内部 `_physics_period()` 才推进 10 个 2 ms 物理步。
从真实调用频率理解接口，不能仅从函数名推断是 500 Hz 策略推理。

```text
warmup：planner.warmup_step() → _physics_period()
正式每控制帧：
  上一周期座面冲量 → planner.seat_vertical_impulse_ns
  planner.on_physics_step(robot)
    → 读 observation → 推理/调理 → 写动作
  _measure(frame) → 保存完整控制频率 row
  若 frame 位于 image_schedule → 采图并写 LeRobot 行
  contacts.end_period(...)
  除最后一帧外 → _physics_period()
```

`image_schedule()` 用 `floor(image_time*50 + 0.5)` 选择最近控制帧，30 fps 不应使用 `50//30` 作为固定 stride。
`_state_vector()` 将原始 qpos 拆开，按 **关节位置 29 + 关节速度 29 + 基座位姿 7 + 基座 twist 6** 拼成 71 维低频 state；
它不是直接连接 `qpos[36] + joint_vel[29] + base_vel[6]` 的顺序。完整控制记录还包含 body poses、相机、接触与目标。
接触 row 对应已发生的物理周期，动作对应即将执行的周期，沿 `_measure()` 与 `end_period()` 核对边界。

代码练习：在 `test_sonic_joint_orders_are_inverse_permutations` 中验证排列互逆，
再从 `SonicTracker.step()` 找出四元数顺序、坐标系、默认角度、动作缩放和 batch 维各自发生变化的行。

## 13. CLI 与外层脚本：不要漏掉调用栈最外层

### 13.1 命令只是对象装配入口

按 [core.cli.main](../real2sim2real/core/cli.py) → `_register_builtin_and_plugins()` → `load_cli_plugins()` →
[job.cli.register](../real2sim2real/job/cli.py) → `run()` 追踪。
`register(main_app)` 将业务 Typer app 挂成 `collect` 子命令，`run()` 应用配置后 `AutoJob.from_config().run()`。
业务不写在根 CLI，扩展 CLI 用独立 entry point group。

继续看 `render_sorting()`：它不是简单换一个 Job backend，还编排物理 worker、相机 worker 和数据组装。
这解释了为什么有些行为需要追进 `scripts/`，仅阅读包内 Job 看不到完整执行路径。

### 13.2 采集器函数与任务 Job 的边界

| 外层函数 | 调用与修改的对象 | 精读目的 |
|---|---|---|
| [tabletop collect()](../scripts/tabletop_pickplace/collect_tabletop_pickplace.py) | 配置、子进程环境、staging、独立 validator、发布目录 | 找出子进程成功后还需哪些校验，失败何时阻止发布 |
| `_rebase_receipts()` | 发布后回执中的结果路径 | 区分输出位置修正与保留原始执行配置 |
| [replay collector](../scripts/tabletop_replay/collect_tabletop_replay.py) | 复用桌面采集器入口 | 看默认任务如何切换，哪些逻辑没有复制 |
| [battery collect()](../scripts/battery_disassembly/collect_battery_disassembly.py) | GPU 预算时钟环境、进度、独立验收与视频入口 | 时钟处理作用于外层进程环境，不是 Planner 动作算法 |
| [screw collect()/publish()](../scripts/physic_solver/collect_screw_arm_demo.py) | debug 运行、物理/数据校验、视觉复核记录与正式目录 | 看视觉复核如何成为发布条件 |
| [sorting run_camera_workers()](../scripts/sorting/render_sorting_package.py) | 每路相机 worker、缓存与最终编码 | 看 worker 完成条件、续渲、缓存匹配与失败传播 |

阅读任一独立 validator 时分三步：读取哪些原始文件、怎样重算指标、如何把失败返回给外层调用方。
它可能复用包内纯验收函数；“独立”指从落盘证据重算，不代表每个公式都另写了一遍。
例如桌面任务的物理判断回到 `evaluate_pickplace()`，额外再检查数据/视频与物理源的对齐。

## 14. 用测试读算法，而不是先运行整套仿真

### 14.1 推荐逐个调试的现有测试

| 测试入口 | 重点看测试怎么构造输入、隔离依赖和断言 |
|---|---|
| [test_tabletop_pickplace.py](../tests/test_tabletop_pickplace.py) `test_phase_machine_runs_the_source_sequence_and_keeps_the_grasp_offset` | fake Robot、规划结果替身，驱动真正 Planner 状态机；在 `_enter` / `_advance` 单步 |
| 同文件 `test_planning_failure_stops_the_episode` | 模拟规划失败，核对失败状态与布尔返回值 |
| 同文件 `test_evaluate_pickplace_passes_and_detects_failures` | 从完整轨迹构造一次成功，再修改单个量使检查失败 |
| [test_tabletop_replay.py](../tests/test_tabletop_replay.py) `test_replay_command_hold_and_linear` | 数组插值、速度前馈、最后一帧边界 |
| [test_fire_inspection.py](../tests/test_fire_inspection.py) `test_grid_plan_keeps_clearance`、`test_patrol_finds_fire_and_photographs` | 地图算法与机器人/拍照回调替身 |
| [test_physic_solver_migration.py](../tests/test_physic_solver_migration.py) `test_proxy_coupling_requires_feedback_capable_rigid_execution` | 配置组合怎样提前拒绝不可用的反馈路径 |
| 同文件 `test_release_compensation_changes_robot_path_without_moving_acceptance_goal` | 控制补偿与验收目标的数据独立性 |
| [test_sorting_package.py](../tests/test_sorting_package.py) `test_strain_grip_opens_with_bounded_relief_and_recovers_slowly` | 有状态限幅、死区与多轮输出 |
| 同文件 `test_cached_soft_success_cannot_hide_damage_during_the_later_rigid_relay` | 局部结果缓存与全局检查的优先关系 |
| [test_battery_disassembly.py](../tests/test_battery_disassembly.py) `test_ik_round_trip_full_and_axis_targets`、`test_timed_path_ends_at_rest_and_follows_arclength` | FK/IK 往返、姿态/轴约束、弧长插值和时间导数 |
| [test_halo.py](../tests/test_halo.py) `test_sonic_joint_orders_are_inverse_permutations`、`test_reference_resampling_and_image_schedule_match_halo_counts` | 关节排列和多频率索引；不需要启动 ONNX 闭环 |

测试中的 fake 并不等于被研究算法也被替换了。要区分 mock 的 SDK/规划输出，与仍在执行的 Python 状态机、插值或验收函数。
读测试时先定位 monkeypatch 和替身对象，再看断言覆盖哪一层。

可以使用 Isaac Python 逐个执行现有测试，例如：

```bash
./app/python.sh -m pytest -q tests/test_tabletop_replay.py::test_replay_command_hold_and_linear
./app/python.sh -m pytest -q tests/test_tabletop_pickplace.py::test_phase_machine_runs_the_source_sequence_and_keeps_the_grasp_offset
```

需要断点时在 IDE 选择 Isaac Python，调试同一测试；或者使用 `pytest --pdb` 在失败处进入。
不要为了单步研究纯函数先启动 Kit。无 Kit 测试可以检验 Python 逻辑，真实接触和策略效果仍需任务运行验证。

### 14.2 两段可以直接执行的源码实验

下面只调用纯函数，不加载机器人资产或启动 Kit。

```bash
./app/python.sh - <<'PY'
import numpy as np
from real2sim2real.utils.tabletop_replay_utils import replay_command

actions = np.zeros((2, 8))
actions[1, 0] = 0.4
for step in (0, 1, 3, 4):
    target, velocity, frame = replay_command(actions, step, stride=4, fps=30)
    print(step, frame, target[0], velocity[0])
PY
```

预期第一关节位置依次为 `0 / 0.1 / 0.3 / 0.4`，速度为 `12 / 12 / 12 / 0 rad/s`。
这是在两帧间用 4 个物理步插值，不是以 `dt=4` 秒积分。

```bash
./app/python.sh - <<'PY'
from real2sim2real.utils.halo_utils import image_schedule

indices = image_schedule(control_frames=10, control_hz=50, image_fps=30)
print(indices.tolist())
print((indices[1:] - indices[:-1]).tolist())
PY
```

预期索引为 `[0, 2, 3, 5, 7, 8]`，间隔不是常数。
回到 `HaloJob.simulate()` 找到它如何将这些控制帧索引用于图像采样。

## 15. 代码级学习任务与检查方式

每完成一个单元，留下可核对的代码说明。用具体符号和数组代替“已经理解整体流程”。

| 学习任务 | 应交出的具体结果 |
|---|---|
| 配置追踪 | 给一个 backend 写出 entry point、实现类、config_cls、配置校验函数和返回实例 |
| 对象关系 | 画出 Job/Scene/Robot/Planner/Runtime 的引用，并标出哪个函数建立引用 |
| 状态机 | 为桌面的一次阶段切换写出 `_samples`、`_i`、`phase`、`last_action` 的前后值 |
| 数据时序 | 为 `_run_episode()` 画两轮 `(q_t,u_t,image_t) → step → (q_t+1,...)`，标明末轮行为 |
| 坐标变换 | 推导 Scene 目标中心恢复、world-to-base、手系滑移三处矩阵操作的输入坐标系 |
| 继承派发 | 列出 Replay 覆盖的四处方法及父类调用位置；说明哪段循环没有复制 |
| 数值算法 | 推导回放插值或 IK 阻尼最小二乘的一步，核对数组形状 |
| 求解器调度 | 画 direct/hybrid、CPU MuJoCo+VBD 的一外层步与若干子步，指出状态交换位置 |
| 连续状态 | 说明分拣切换纸箱时复用哪些对象、适配哪些字段、为何继续检查软包 |
| 工具控制 | 写出 battery `Phase` 执行时 active/entered/phase_t 的状态变化，以及 seek/unscrew 的传感器条件 |
| 策略推理 | 列出 HALO 从实测状态到 decoder 输入，再从动作到电机与历史回写的重排和缩放 |
| 失败传播 | 分别追踪规划失败、验收失败和渲染失败，直到日志、结果和最外层退出/发布条件 |

后续真正改代码时先定位这一层的最小修改点。例如：

- 改桌面轨迹：`_hand_target/_solve_*`；改记录时序：`_run_episode`，同时检查 Replay 复用影响。
- 改分拣形变反馈：`StrainGripController` 与实际 `set_gripper` 调用链；只改统计值不会改变控制。
- 改电池工具反馈：`_seek/_unscrew` 与 Robot 传感器接口；改螺纹物理模型则是 Scene `_author_thread`。
- 改 HALO 动作调理：Planner/filter；同时保持 `set_applied_target()` 与实际电机目标一致。
- 改跨任务步进：`IsaacsimWorld.step/update`，需要检查所有三种物理路径，影响范围远大于单个任务。

开发约束继续遵守 [AGENTS.md](../AGENTS.md)：每任务每能力域一个文件，重依赖延迟加载，组件构造不启动 Kit，
任务专用语义不进入通用基类。新增完整任务时再使用 [任务开发 skill](../.agents/skills/rsr-task-development/SKILL.md)；
当前代码学习不需要先复制一套后端。
