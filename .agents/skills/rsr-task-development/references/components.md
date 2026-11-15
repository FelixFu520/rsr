# 组件接入：Scene / Robot / Planner / Job、配置与注册

先读对应 `{domain}/base.py` 和一个最接近的现有实现。巡检任务（PhysX、导航）和物理求解器任务
（Newton direct、抓放）是两类最完整的参照；分拣任务示范了跨任务继承。

## 通用写法（每个后端都遵守）

```python
class FooSceneConfig(BaseConfig):           # 同文件；extra="forbid" 由 BaseConfig 提供
    backend: str = "foo_scene"              # 与 entry point 名一致的小写字符串
    usd_path: str = "assets/Foo/Scene/foo.usd"
    cameras: list[CameraConfig] = Field(default_factory=list)

class FooScene(BaseScene):
    config_cls = FooSceneConfig             # Auto*.from_config 用它校验 [scene] 段

    def __init__(self, config: FooSceneConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("scene")
        # 只保存配置和空状态：不启动 Kit，不 import pxr/omni/newton/warp
```

- Kit 相关 import 写在方法内部；Warp kernel 和依赖 Newton 的子类放进首次调用时才定义的工厂函数
  （参考 `utils/physic_solver_utils.moving_frame_vbd_class`）。这样 `list-backends` 与单元测试不启动 Kit。
- 字段带单位后缀（`_m`、`_s`、`_rad`、`_kg`、`_mps`）和 `Field(gt=..., description=...)` 约束；嵌套配置用
  `BaseModel` + `ConfigDict(extra="forbid")`（参考 `X30GaitConfig`、`PhysicSolverRecorderConfig`）。
- 能在构造时发现的组合错误立刻抛 `ValueError`（引擎不匹配、相机名重复、fps 不整除物理频率、输出已存在），
  不要等 Kit 启动几分钟后才失败。
- 公共方法写简短 docstring；日志用 `get_logger(__name__)` 或 `LoggerMixin`。

## 各能力域的职责

| 域 | 必须实现 | 可选 / 约定的扩展点 | 不要做 |
|---|---|---|---|
| SimWorld | 复用 `isaacsim`，只在任务配置里设字段 | `extensions` 加任务需要的 Kit 扩展（如 `omni.flowusd`）；`stage_usd_path` 或 `background_usd_path` 二选一 | 不新建世界类；不为单个任务改世界默认行为 |
| Scene | `init_scene(sim_world)`：保存 `sim_world`，在 `/World/scene/...` 下加引用，校验必需 prim，`create_cameras` 写入 `self.sensors`；`arrange()`：按配置或实测（`prim_aabb`）摆位，填 `self.poses` | `adjust()`；`initialize_runtime()`（世界 reset 后取物理句柄）；运行中更新 `poses_realtime`；任务查询（如 `fire_occluder`） | 不启停 App；不放机器人 |
| Robot | `init_robot(sim_world)`：先检查引擎兼容，在 `/World/robots/<Name>` 下加引用，找关节根，挂机载相机；`arrange()`：缓存 `self.poses` | `initialize_runtime()`；`recording_robot_type`、`recording_joint_names`、`recording_state()` / `recording_coordinates()`；Newton direct 下 `add_newton_robot(builder)` 与 `pure_newton=True` | 不在物理视图建立后再校验 USD 运动学（`reset()` 会把姿态写回 USD）；不移动 articulation 的子 link，只移动整个引用 |
| Planner | `init_planner(robot)` 只绑定引用；`on_physics_step(robot) -> bool`（True 表示 episode 结束）；`result() -> dict` | `state["phase"]`、`state["last_action"]`、`state["progress"]`；`action_names`；`rows`（每步一行，写 `trajectory.jsonl`）；`write_diagnostics(dir)`；Newton 运行时类（建模、`step(dt)`、reset、记录坐标） | 不在 `init_planner` 里取物理句柄（世界尚未 reset）；不在 Planner 里录像或写数据集 |
| Job | `XxxJobConfig`（`output_dir`、`max_steps`、`timeout_s`、`inspect_gui`、`recorder` 子模型）；`__init__` 中 `super().__init__(config=config)` 后做跨组件校验；`simulate()` | 各阶段钩子；`build_occ`；`setup_gui_hook`；`_evaluate()` 汇总验收 | 不改 `run()` 顺序；不吞异常（`finally` 写结果后继续抛出） |

## 逻辑放在哪个阶段

| 需要做的事 | 位置 | 例子 |
|---|---|---|
| 解析指令、校验组合、检查输出目录 | `Job.__init__` | 巡检关键词解析；`lerobot_v3` 已存在则 `FileExistsError` |
| 机器人加载前扫描环境 | `build_occ` | PhysX 场景查询建占用图，之后 `stop()` 并 `invalidate_physics()` |
| 依赖场景结果的规划 | `after_arrange_scene_hook` | A* 连通航点，失败直接报错并输出地图 |
| 修改机器人初始摆放 | `before_place_robot_hook` | Franka 底座放到实测桌面上 |
| 注册 Newton stepper、`reset()`、`play()`、取运行时句柄 | `before_start_simulation_hook` | `register_newton_stepper(step, reset=...)` 必须在 `reset/play` 之前 |
| 仿真循环、录制、验收、导出 | `simulate` | 见下 |

## 仿真循环与录制

参照 `PhysicSolverJob.simulate` 与 `FireInspectionJob.simulate`：

1. 写 `diagnostics/<UTC>/configuration.json`（五个组件配置的 `model_dump`）。
2. 创建 `LeRobotV3Recorder`（`task`、`robot_type`、`joint_names`、`action_names`、`cameras`）。
3. 每个物理步：录制步先 `_capture_cameras`（逐相机 `capture_viewport_frame`，前后核对 `physics_steps` 不变）并读观测 →
   Planner 给出本步动作 → `recorder.add_frame(state, action)` → `sim_world.step(render=False)`。
   采集已经渲染，步进时再渲染只会多画一帧未采集的画面并重置 PathTracing 历史。
4. 墙钟超时、Kit 关闭时抛异常；每约 30 s 记录进度与 control/capture/physics/export 分项耗时。
5. 结束后 `result = planner.result()`；不成功且未开启 `save_failed_episodes` 时抛错；
   成功则 `finalize(episode_success=...)`，写 `meta/task_outcome.json`，`dataset_use` 为 `demonstration` / `diagnostic`。
6. 异常时 `recorder.discard(run)` 把临时帧移到诊断目录；`finally` 写 `result.json`（运行目录和输出根各一份）与 `trajectory.jsonl`。

`observation.state` 记录实测量，`action` 记录同一帧下发的目标；维度名和单位写进 `meta/info.json` 与文档。
渲染代价远大于物理、且物理不可逐位复现时，可采用分拣的“先连续求解保存物理源，再零物理步回放渲染”方案。

## 相机与渲染

- 固定相机写 `[[scene.cameras]]`，机载相机写 `[[robot.cameras]]`；名称全局唯一，Job 用 `validate_camera_list` 合并校验。
- 调试用 `RayTracedLighting` 或低 spp；正式 PathTracing 时 `synchronous_capture=true`，`rt_subframes` ≥ `spp/32` 再加余量，
  多相机交替采集时关 `pathtracing_cached`、`pathtracing_many_light_sampling`、`dlss`、`auto_exposure`、时域降噪。
- 详细取舍见 `docs/fire_inspection.md` 3.5 节、`docs/sorting.md` 7.7 节，或 `app/skills/isaac-sim-rendering`。

## 跨任务复用

能继承就继承：`SortingScene(PackageScene)`、`SortingRuntime(PackageRuntime)`、`SortingJob(PhysicSolverJob)`。
从另一任务的模块 import 是允许的，但父类以后的改动会影响你：需要改父类行为时，在父类加默认保持原行为的钩子，
并按共享代码改动处理（见 [code-and-assets.md](code-and-assets.md#对已有任务零影响)）。

## 配置与注册步骤

1. 实现类和 Config 写完后，在 `pyproject.toml` 对应组加 entry point：
   `real2sim2real.{simworld,scene,robot,planner,job}.backends`，名称 = Config 的 `backend` 默认值。
   同时更新组名上方列出任务文件的注释。
2. `./app/python.sh -m pip install -e ".[dev]"`，再 `list-backends` 确认可见（插件导入失败只告警，要看告警）。
3. `core/default_settings.toml` 各段只追加注释行 `# <backend>：字段=默认值 …`，不改 stub 的实际值。
4. `configs/<task>/` 放正式配置和继承它的 `_debug.toml`。命名沿用 `<机器人或机器人求解器>_<引擎或接触求解器>_<场景>.toml`。
   正式配置开头写注释：任务内容、物理、资产准备命令、运行命令、输出目录。调试配置只覆盖 spp、`output_dir`
   （`outputs/debug/<task>`）、`overwrite_dataset=true`、`save_failed_episodes=true`、超时。
5. 运行 `./app/python.sh .agents/skills/rsr-task-development/scripts/check_task_configs.py configs/<task>`，
   再不带参数跑一次，检查全部任务配置。

## 单元测试

`tests/test_<task>.py`，不启动 Kit：解析、运动学 FK/IK 往返、步态或轨迹、规划（用合成占用图）、验收函数、
配置路由（`apply_settings_file` + `AutoJob` 选型，参考 `tests/test_sorting_package.py`）。需要机器人或世界时用
`SimpleNamespace` / Fake 类（参考 `tests/test_fire_inspection.py::FakeRobot`）。
`.gitignore` 忽略 `tests/*`，必须追加 `!tests/test_<task>.py`，否则测试不会进 git。
