---
name: rsr-task-development
description: 在 Real2Sim2Real（rsr）中迁移外部项目的仿真任务、新增任务，或丰富已有任务（新物体/机型/求解器组合/相机/后端）时使用。按阶段覆盖：需求对齐、场景盘点与物理引擎和求解器选择、组件与配置接入、每任务每能力域一个文件、assets/<Task>/ 自包含资产、额外依赖安装说明、对已有任务零影响的回归，以及采集结果验收和失败迭代直到通过。只排查单个物理异常（穿透、关节发散、夹爪力、性能）时改用对应 isaac-* 专项 skill。
---

# rsr 任务迁移 / 新增 / 丰富

用中文报告。先读仓库根 `AGENTS.md`，本 skill 只补充流程和经验，不替代它。文档与代码冲突时以代码为准，
并在交付时列出不一致之处（已知的见 [项目背景](references/project-context.md#已知文档漂移)）。

## 先判断模式

| 模式 | 典型输入 | 额外要求 |
|---|---|---|
| 迁移 | 外部项目或旧仓库里的任务（例：Go2 找物拍照 → `fire_inspection`） | 先审计源项目：代码、配置、资产、控制策略、数据格式；交付迁移对照表；源资产只复制不修改 |
| 新增 | 一段需求 + 资产或资产来源 | 从需求推出验收项，再设计组件 |
| 丰富 | 给已有任务加物体、机型、求解器组合、相机或流程变体 | 写进该任务已有文件的新分节；新字段默认值必须保持旧行为；已有正式输出不得覆盖 |

## 工作流（每阶段有出口门槛，不满足不进入下一阶段）

1. **读背景**：[project-context.md](references/project-context.md)。确认 `./app/python.sh` 可用、
   `list-backends` 与 `pytest -q` 在改动前通过，记下基线。
2. **需求对齐**：与用户确认下列项目，未确认的写成假设并在报告中列出。
   - 任务目标与成功标准（可测量的验收项、门限、单位）；episode 数、时长、相机路数、分辨率、fps、渲染器。
   - 机器人、被操作物体、环境资产的来源与许可；是否必须保持原装机型/夹爪（不得加隐藏夹块等）。
   - 迁移时：源项目里缺失的能力（如没有步态策略、依赖 LLM/外部 API）怎么替代，列出方案让用户选。
   - 交付物：正式输出目录、数据格式（默认 LeRobot v3）、文档。
   门槛：验收项和交付物有书面定义。
3. **场景盘点与引擎选择**：[engine-selection.md](references/engine-selection.md)。用本 skill 的
   `usd_inventory.py`（见下方“自带脚本”）盘点资产，按决策顺序选世界模式、机器人求解器、接触/柔体求解器和步长。
   门槛：选型理由和被否决方案有证据（资产 schema、项目已验证结论或冒烟测试）。
4. **资产整理**：[code-and-assets.md](references/code-and-assets.md#资产组织)。写
   `scripts/<task>/prepare_<task>_assets.py`，复制到 `assets/<Task>/` 并生成 `provenance.json`。
   门槛：组合后的 stage 不加载 `assets/<Task>/` 以外的本地 layer，缺失依赖为零或已逐项说明。
5. **组件、配置与注册**：[components.md](references/components.md)。先写无 Kit 单元测试能覆盖的纯函数
   （运动学、解析、规划、几何检查），再接 Kit 部分。
   门槛：`ruff check`、`pytest -q`、`list-backends` 通过，新后端可见，
   本 skill 的 `check_task_configs.py` 校验全部配置通过。
6. **采集与验收迭代**：[validation.md](references/validation.md)。按“物理冒烟 → 只跑物理 → 调试配置
   全流程 → 正式采集 → 独立校验 → 视觉复核”逐级推进；失败时按根因分析流程修复，直到正式运行通过。
   门槛：正式目录 `result.json` 的 `success=true`，独立校验脚本全部通过，结果已写入文档。
7. **回归与交付**：[code-and-assets.md](references/code-and-assets.md#对已有任务零影响) 的回归门槛，
   按 [task-doc-template.md](references/task-doc-template.md) 写 `docs/<task>.md`，再按下方清单交付。

## 硬性规则

- 唯一运行时是 `./app/python.sh`；不要 `uv sync` 出 `.venv`。本项目同时运行的 Isaac Sim 不超过两个；
  PathTracing 正式渲染期间不要并行其它 Kit/GPU 任务。
- 生命周期固定 `init_world → arrange_scene → place_robot → start_simulation → finally: close`，
  任务逻辑只放在钩子和 `simulate` 里；通用 ABC 不加任务字段。
- 每个任务在 scene / robot / planner / job / utils 各一个文件，后缀即任务名；能复用就继承，不复制代码。
- 包内模块顶层不导入 Kit / pxr / omni / Newton / Warp / 任务专用第三方库。
- 每个任务只引用 `assets/<Task>/` 内的资产；任务间共用的资产各复制一份。
- 正式数据只来自真实物理：不固定粒子、不加吸附约束、不瞬移物体、不在中途重置后拼接。
  需要放宽门限或改变验收口径时先征得用户同意，并在文档记录授权与前后数值。
- `outputs/<task>/` 只放通过全部验收的结果；调试、对照、失败和探针放 `outputs/debug/<task>/`。
  已有正式数据集时拒绝覆盖。边缘用例重跑才通过时，把每次运行写进 `run_note.json` 并在文档说明。
- 不改其它任务的配置、资产、输出。确需改共享代码时，默认行为必须与改动前一致，并跑完回归门槛。
- 密钥只放环境变量或 `.secrets.toml`，不写进配置、文档、日志。未经要求不 commit / push。
- `ruff format` 只对改动的文件运行，不对整个包运行。

## 自带脚本（仓库根目录运行，都不写文件）

```bash
S=.agents/skills/rsr-task-development/scripts
./app/python.sh $S/usd_inventory.py <file.usd> [...]     # 单位、prim 类型、物理 schema、碰撞近似；需要时自动起 headless Kit
./app/python.sh $S/check_task_configs.py [configs/<task>] # 按注册后端的 Config 校验配置（含 inherit 链），不启动 Kit
```

## 交付清单

- [ ] 代码：`{scene,robot,planner,job}/<domain>_<task>.py`、`utils/<task>_utils.py`（按需），分节注释清楚
- [ ] 注册：`pyproject.toml` entry points 已加，`pip install -e ".[dev]"` 后 `list-backends` 可见
- [ ] 配置：`configs/<task>/` 正式 + 调试配置；新字段写入 `*Config`、`default_settings.toml` 注释和示例
- [ ] 资产：`assets/<Task>/` 自包含，`provenance.json` 含来源、SHA-256、改动和许可证
- [ ] 脚本：`scripts/<task>/` 下资产准备、独立校验（以及按需的只跑物理/批量采集）
- [ ] 测试：`tests/test_<task>.py` 无 Kit，并已在 `.gitignore` 加 `!tests/test_<task>.py`
- [ ] 依赖：额外依赖有安装说明，缺失时报清晰错误，不影响其它任务和 `list-backends`
- [ ] 结果：正式目录通过 Job 内验收和独立校验；失败与对照在 debug 目录且有记录
- [ ] 文档：`docs/<task>.md`；`README.md` 任务表、`AGENTS.md` 任务布局表和任务入口已加一行
- [ ] 回归：ruff、pytest、`check_task_configs.py`、`git diff --stat` 只含预期文件

## 交付报告

列出：模式与需求假设、引擎选型及证据、新增/修改文件、资产来源与大小、运行命令、验收结果（每项实测值与门限）、
失败与修复记录、未验证项（如未做人工视觉复核、未用官方 LeRobot SDK 加载）、对已有任务的回归结果。
区分已实测、引用项目已有结论和推测。

## 相关 skill（按证据按需读取）

穿透/弹飞/接触 → [isaac-collision-debug](../isaac-collision-debug/SKILL.md)；关节发散、mimic →
[isaac-articulation-stability](../isaac-articulation-stability/SKILL.md)；夹爪力与闭合 →
[isaac-gripper-tuning](../isaac-gripper-tuning/SKILL.md)；步耗时、吞吐 →
[isaac-physics-performance](../isaac-physics-performance/SKILL.md)；功能组合与版本限制 →
[isaac-physics-faq](../isaac-physics-faq/SKILL.md)。Isaac Sim 自带 skill 在 `app/skills/`，与本机版本匹配，
常用：`physics-simulation`、`physx-to-newton-conversion`、`urdf-mjcf-to-usd-conversion`、`usd-articulation`、
`isaac-sim-rendering`、`isaac-camera`、`isaac-sim-assets`、`manipulation-ik`、`navigation-primitives`。
