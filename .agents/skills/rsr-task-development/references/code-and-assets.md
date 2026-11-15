# 代码组织、资产组织、额外依赖与零影响回归

## 代码组织

任务名用 snake_case 作文件后缀（`<task>`），资产目录用 CamelCase（`<Task>`）。新增任务的完整文件集：

```text
real2sim2real/scene/scene_<task>.py        # 本任务全部 Scene 类及其 Config
real2sim2real/robot/robot_<task>.py
real2sim2real/planner/planner_<task>.py
real2sim2real/job/job_<task>.py
real2sim2real/utils/<task>_utils.py        # 按需；纯函数优先，便于无 Kit 测试
configs/<task>/<name>.toml + <name>_debug.toml
scripts/<task>/prepare_<task>_assets.py, validate_<task>.py（+ 按需的只跑物理/批量采集脚本）
tests/test_<task>.py                        # 并在 .gitignore 加白名单
docs/<task>.md
assets/<Task>/                              # git 忽略
```

- 一个能力域只有一个任务文件。同一任务的多个后端、运行时和辅助类都写在这个文件里，按依赖顺序分节，
  节首用横线注释说明内容（参考 `scene/scene_physic_solver.py` 的 `# ====` 分节和 `docs/physic_solver.md` 第 3 节列出的分节）。
- 某个域完全复用其它任务的后端时，不必新建空文件，在配置里直接用已有后端名，并在文档组成表里写明。
- 丰富已有任务：在该任务文件末尾加新分节和新后端名，配置放该任务的 `configs/<task>/`；
  已有后端的新字段默认值必须保持原行为，已有配置不需要修改。
- 脚本之间互相引用用 `from scripts.<task>.<name> import ...`；脚本从仓库根用 `./app/python.sh scripts/<task>/<x>.py` 运行。
- 只在确实通用、且不改变现有行为时才往共享 `utils/`（`cameras`、`lerobot_v3`、`asset_utils`、`kit_process`）加东西。
- 遗留或替换下来的代码、配置移到 `bak/`（git 忽略），运行时代码不 import `bak/`。

## 资产组织

目标：`assets/<Task>/` 打包发给别人就能运行，不依赖源项目、网络或其它任务的资产目录。

```text
assets/<Task>/
├── robot/ 或 Robot/<name>/     # 机器人 USD/URDF/MJCF 及网格、贴图、LICENSE
├── scene/ 或 Scene/、Background/ # 环境、工位
├── objects/ 或 Package/ 等      # 被操作物体，按版本快照（如 v7/、v10_station/）
└── provenance.json              # 来源、改动、依赖检查、每个文件的 SHA-256
```

`scripts/<task>/prepare_<task>_assets.py` 的要求（参照 `scripts/fire_inspection/prepare_fire_inspection_assets.py`）：

1. 源文件只读；复制而不是软链；目标已存在且 SHA-256 不同时拒绝覆盖；重复运行结果一致。
2. 依赖闭包：`UsdUtils.ComputeAllDependencies` 收集 layer 和资产，再递归解析 MDL 里的 `.::Module` 导入和
   `texture_2d("...")` 贴图（USD 扫描看不到，漏掉会导致材质编译失败）；URDF 网格按相对路径保留目录结构。
3. 依赖落在源目录之外时报错并要求显式复制；`omniverse://`、`https://` 等远程资产下载到本地后再引用。
   Isaac 内置的 `OmniPBR.mdl` 等由 Kit 搜索路径解析，可以不复制，但在 provenance 里写明。
4. 需要改路径时用 `Sdf` 只改必要字段（如 sublayer），并在 provenance 的 `modified` 里记录。
5. 校验：打开复制后的 stage，`GetUsedLayers()` 全部位于 `assets/<Task>/` 下；任务需要的 prim 都存在；有 default prim。
6. 裸 `python.sh` 无 `pxr` 时启动 headless `SimulationApp`；出错先记录再 `close(exit_code=1)`。
7. 记下目录大小（`du -sh assets/<Task>`）写进文档；许可证文件随资产复制。

其它规则：

- 配置用相对路径 `assets/<Task>/...` 引用资产，由 cwd 或 `RSR_ASSETS_ROOT`（`scripts/rsr` 自动设置）解析。
- 任务之间共用的资产各复制一份；被引用资产之间的相对目录关系要保持（分拣机器人 USD 引用 `../../Scene/...`）。
- 不属于任何任务的资产放 `assets/bak/`；任务配置和运行代码不引用它，准备脚本可以从这里复制。
- 调物理不改资产文件：质量、灯光倍率、摩擦等在运行时 stage 上覆盖并记录。必须改资产时生成新版本快照目录。
- 不手改已记录输出里的 JSON 来适配新资产路径：会破坏记录的 SHA-256 链（分拣旧输出就是这样，需重新验证或重渲染）。

## 额外依赖

1. 先查是否已有：`./app/python.sh -m pip show <pkg>`；Kit 扩展看 `app/exts`、`app/extscache`，能用
   `enable_extension` 启用的放进任务配置的 `sim_world.extensions`，不改全局默认。
2. pip 包放进 `pyproject.toml` 的任务专用可选依赖组，例 `[project.optional-dependencies] <task> = ["pkg>=x"]`，
   然后 `uv lock`，安装 `./app/python.sh -m pip install -e ".[dev,<task>]"`。核心 `dependencies` 不加任务专用包。
3. 不升级 Isaac 自带的包（numpy、torch、warp、pxr 等）。安装前用 `pip install --dry-run` 看会不会替换它们，
   装完跑 `./app/python.sh -m pip check` 和全量回归。
4. 只在任务文件的方法内 import；缺失时抛出带安装命令的错误。`list-backends` 和其它任务的测试在未安装时仍要通过。
5. 系统工具写明 apt 包名；外部服务写明环境变量名、离线替代方案，密钥不进仓库。
6. 在 `docs/<task>.md` 的“依赖与安装”节列出：包名与版本、用途、安装命令、验证命令。

## 对已有任务零影响

改动前记下基线（pytest 通过数、`list-backends` 输出）。交付前全部满足：

1. `git status` / `git diff --stat` 只含：新任务文件、`pyproject.toml` 注册行、`default_settings.toml` 注释、
   `.gitignore` 测试白名单、`README.md` / `AGENTS.md` 索引行。其它文件的改动逐项说明理由。
2. `./app/python.sh -m ruff check real2sim2real scripts`；`./app/python.sh -m ruff format <改动的文件>`。
3. `./app/python.sh -m pytest -q` 全部通过，通过数不少于基线加新增测试。
4. `list-backends` 中原有后端全部还在。
5. `./app/python.sh .agents/skills/rsr-task-development/scripts/check_task_configs.py` 全部配置通过。
6. 改了共享运行时代码（`simworld/isaacsim.py`、`utils/lerobot_v3.py`、`utils/cameras.py`、被继承的其它任务类）时：
   保证默认路径行为不变（例：巡检只在渲染模式不同时才写 `/rtx/rendermode`）；跑
   `scripts/physic_solver/validate_physics_modes.py` 覆盖受影响的世界模式；受影响任务用调试配置跑到
   `outputs/debug/<other_task>/regression_<日期>/`。分拣全流程要数小时，先问用户。
7. 不写、不移动、不覆盖 `outputs/<其它任务>/` 和 `assets/<其它任务>/`。
8. 不与其它任务的正式渲染并行运行 Kit；同时运行的 Isaac Sim 不超过两个。
