# 物理求解器验证

更新日期：2026-11-13。

## 1. 任务与统一分类

本任务比较机器人求解器、接触／柔体求解器与被操作物体的组合，覆盖衣服、刚体方块、柔性包裹、纸箱及螺纹装配。
采集执行真实物理，按接近、闭合、提升、转运、释放、撤离验证完整过程；动作序列结束不等于成功。
物理引擎和算法层级说明见 [physics_engines.md](physics_engines.md)。

### 1.1 文件名与工况

配置全部直接放在 `configs/physic_solver/`，不设置子目录，也没有仅作继承用的 base 文件。
30份配置均为独立TOML，采用：

```text
<机器人求解器>_<接触求解器>_<物体ID>_<工况>.toml
```

例如 `mujoco_vbd_package01_mass500g_pos_x35mm.toml` 的正式输出为
`outputs/physic_solver/mujoco_vbd_package01_mass500g_pos_x35mm/`。
各配置的 `job.output_dir` 最后一段与文件 stem 完全一致。

- 机器人求解器：`featherstone`、`mujoco`、`xpbd`、`tgs`。
- 接触／柔体路径：`vbd`、`mujoco`（原生刚体接触）、`xpbd`、`tgs`、`hydroelastic`。
- 工况：`baseline`、`center_native`、`center_pads`、`pos_x35mm`、`mass500g_pos_x35mm`、
  `pos_x35mm_shake_noattach`、`center_shake_spring` 等。
- `_control` 为原有对照实验，失败不作为成功演示发布；`_debug` 仅供不录RGB的物理排错。

`package01`的90°抓法为Y175mm优化抓点、59.5mm抓高和双向body代理；
组合倾斜为X35mm、偏航30°／俯仰10°优化抓点。原中央90°和原(-25,40)mm组合倾斜的历史失败仍在debug归档，
不能把当前优化抓点称为原位置的成功复现。各组的夹爪、控制与物理模型存在差别，不是严格单因素性能排名。

### 1.2 世界模式

| 模式 | 动力学归属 |
|---|---|
| `physx` | 机器人及刚体由PhysX TGS积分 |
| `hybrid` | PhysX机器人推进后，Newton VBD求解衣服；使用单向运动学碰撞代理 |
| `newton` + `direct` | 由Planner创建真实Newton模型，Kit用于可视化 |
| `newton_diagnostic` | 使用本机Isaac安装的Newton／USD SDK求解并保存连续轨迹，不启动Kit；通过后另行零物理步回放采集 |

普通VBD外部刚体路径为刚体影响柔体的单向耦合；配置中的 `rigid_soft_coupling="proxy"` 显式启用双向body代理。
这两类实验均不构成真实机械臂额定负载认证。纸箱采用原生MuJoCo双向刚体接触。

## 2. 物体编号

| 编号 | 物体 | 任务内资产 |
|---|---|---|
| `cloth01` | Newton官方衬衫 | `assets/PhysicSolver/Cloth/newton_official/` |
| `cloth02` | 通用衬衫 | `assets/PhysicSolver/Cloth/unisex_shirt.usd` |
| `rigid01` | 0.1kg刚体方块 | 场景代码创建的刚体对照 |
| `package01` | V7双网格衣物＋袋膜 | `assets/PhysicSolver/Package/v7/` |
| `package02` | 站点衣物包裹 | `assets/PhysicSolver/Package/v10/station_clothing/` |
| `package03` | 绿色包裹 | `assets/PhysicSolver/Package/v10/station_green/` |
| `package04` | 粉色T恤包裹 | `assets/PhysicSolver/Package/v10/pink_tshirt/` |
| `package05` | 磨砂上衣包裹 | `assets/PhysicSolver/Package/v10/frosted_top/` |
| `package06` | 黑色连帽衫包裹 | `assets/PhysicSolver/Package/v10/black_hoodie/` |
| `package07` | 原尺寸纸箱01，350g | `assets/PhysicSolver/Package/v10/box01/` |
| `package08` | 原尺寸纸箱02，250g | `assets/PhysicSolver/Package/v10/box02/` |
| `screw01` | M20 loose螺栓／螺母 | `assets/PhysicSolver/Screw/M20/` |

重量工况的100／200／350／500g指衣物质量；袋膜另约20.54g，不把两者混作总负载。
纸箱采用原尺寸刚性包围盒，不模拟纸板压溃。模型、MDL、贴图、URDF、许可证和来源哈希均位于本任务资产目录，
不依赖源项目 `/home/fufa/Projects/package` 的运行时路径。

## 3. 配置目录

| 配置 | 用途 | 输出位置 |
|---|---|---|
| [featherstone_vbd_cloth01_baseline](../configs/physic_solver/featherstone_vbd_cloth01_baseline.toml) | 正式采集目标 | `outputs/physic_solver/featherstone_vbd_cloth01_baseline/` |
| [mujoco_vbd_cloth02_baseline](../configs/physic_solver/mujoco_vbd_cloth02_baseline.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_cloth02_baseline/` |
| [xpbd_vbd_cloth02_baseline](../configs/physic_solver/xpbd_vbd_cloth02_baseline.toml) | 正式采集目标 | `outputs/physic_solver/xpbd_vbd_cloth02_baseline/` |
| [tgs_vbd_cloth02_baseline](../configs/physic_solver/tgs_vbd_cloth02_baseline.toml) | 正式采集目标 | `outputs/physic_solver/tgs_vbd_cloth02_baseline/` |
| [tgs_tgs_rigid01_baseline](../configs/physic_solver/tgs_tgs_rigid01_baseline.toml) | 正式采集目标 | `outputs/physic_solver/tgs_tgs_rigid01_baseline/` |
| [mujoco_vbd_package01_center_native](../configs/physic_solver/mujoco_vbd_package01_center_native.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_center_native/` |
| [featherstone_vbd_package01_center_control](../configs/physic_solver/featherstone_vbd_package01_center_control.toml) | 失败对照，保留真实结果 | `outputs/debug/physic_solver/featherstone_vbd_package01_center_control/` |
| [xpbd_xpbd_cloth02_baseline_control](../configs/physic_solver/xpbd_xpbd_cloth02_baseline_control.toml) | 失败对照，保留真实结果 | `outputs/debug/physic_solver/xpbd_xpbd_cloth02_baseline_control/` |
| [mujoco_vbd_package01_center_pads](../configs/physic_solver/mujoco_vbd_package01_center_pads.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_center_pads/` |
| [mujoco_vbd_package01_pos_x35mm](../configs/physic_solver/mujoco_vbd_package01_pos_x35mm.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_pos_x35mm/` |
| [mujoco_vbd_package01_pos_y55mm](../configs/physic_solver/mujoco_vbd_package01_pos_y55mm.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_pos_y55mm/` |
| [mujoco_vbd_package01_pos_x35mm_yaw45deg](../configs/physic_solver/mujoco_vbd_package01_pos_x35mm_yaw45deg.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_pos_x35mm_yaw45deg/` |
| [mujoco_vbd_package01_pos_y175mm_yaw90deg_twoway](../configs/physic_solver/mujoco_vbd_package01_pos_y175mm_yaw90deg_twoway.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_pos_y175mm_yaw90deg_twoway/` |
| [mujoco_vbd_package01_pos_x35mm_yaw30deg_pitch10deg](../configs/physic_solver/mujoco_vbd_package01_pos_x35mm_yaw30deg_pitch10deg.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_pos_x35mm_yaw30deg_pitch10deg/` |
| [mujoco_vbd_package01_mass100g_pos_x35mm](../configs/physic_solver/mujoco_vbd_package01_mass100g_pos_x35mm.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_mass100g_pos_x35mm/` |
| [mujoco_vbd_package01_mass200g_pos_x35mm](../configs/physic_solver/mujoco_vbd_package01_mass200g_pos_x35mm.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_mass200g_pos_x35mm/` |
| [mujoco_vbd_package01_mass350g_pos_x35mm](../configs/physic_solver/mujoco_vbd_package01_mass350g_pos_x35mm.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_mass350g_pos_x35mm/` |
| [mujoco_vbd_package01_mass500g_pos_x35mm](../configs/physic_solver/mujoco_vbd_package01_mass500g_pos_x35mm.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_mass500g_pos_x35mm/` |
| [mujoco_vbd_package02_center](../configs/physic_solver/mujoco_vbd_package02_center.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package02_center/` |
| [mujoco_vbd_package03_center](../configs/physic_solver/mujoco_vbd_package03_center.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package03_center/` |
| [mujoco_vbd_package04_center](../configs/physic_solver/mujoco_vbd_package04_center.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package04_center/` |
| [mujoco_vbd_package05_center](../configs/physic_solver/mujoco_vbd_package05_center.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package05_center/` |
| [mujoco_vbd_package06_center](../configs/physic_solver/mujoco_vbd_package06_center.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package06_center/` |
| [mujoco_mujoco_package07_center](../configs/physic_solver/mujoco_mujoco_package07_center.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_mujoco_package07_center/` |
| [mujoco_mujoco_package08_center](../configs/physic_solver/mujoco_mujoco_package08_center.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_mujoco_package08_center/` |
| [mujoco_vbd_package01_pos_x35mm_shake_noattach](../configs/physic_solver/mujoco_vbd_package01_pos_x35mm_shake_noattach.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_pos_x35mm_shake_noattach/` |
| [mujoco_vbd_package01_center_shake_spring](../configs/physic_solver/mujoco_vbd_package01_center_shake_spring.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_center_shake_spring/` |
| [mujoco_vbd_package01_pos_x35mm_shake_spring](../configs/physic_solver/mujoco_vbd_package01_pos_x35mm_shake_spring.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_vbd_package01_pos_x35mm_shake_spring/` |
| [mujoco_hydroelastic_screw01_rotate](../configs/physic_solver/mujoco_hydroelastic_screw01_rotate.toml) | 正式采集目标 | `outputs/physic_solver/mujoco_hydroelastic_screw01_rotate/` |
| [mujoco_hydroelastic_screw01_rotate_debug](../configs/physic_solver/mujoco_hydroelastic_screw01_rotate_debug.toml) | 物理排错，不录RGB | `outputs/debug/physic_solver/mujoco_hydroelastic_screw01_rotate_debug/` |

## 4. 本轮重新采集

本轮按用户要求清空原正式输出目录，从初态重新积分并重新采集视频；不将旧视频改名充当新数据，
也不使用旧物理轨迹作为本轮演示。仅本轮新完成并通过独立审计的连续轨迹，可用来零物理步回放采集。
所有源相机和recorder均为 **640×480、15fps**。后续多画面拼接视频可以采用1080p画布，不改变源采集分辨率。

```bash
# 从当前本轮状态继续；已经完成的新轨迹可在中断后续做渲染，未完成的物理前缀从初态重跑
./app/python.sh scripts/physic_solver/recapture_physic_solver.py
# 查看实际队列状态
./app/python.sh scripts/physic_solver/recapture_physic_solver.py --status
# 单个工况正常入口
./scripts/rsr collect run -c configs/physic_solver/mujoco_vbd_package01_mass500g_pos_x35mm.toml
```

统一采集器包含27个成功目标（26个抓取＋1个螺纹装配）与2个原有对照，debug配置不加入正式录制。
PathTracing阶段独占GPU；其后最多两个独立仿真实例，纯物理工况用两张GPU隔离求解，RTX录制在GPU0串行进行。
本轮数值通过的数据先保存在debug暂存区；实际查看相机关键帧、写入视觉复核记录后才发布到同名正式目录。
原有对照若出现与历史不同的结果，保留实际证据并检查，不通过重采样挑选失败。
Featherstone包裹对照本轮保存了完整双相机原始帧，物理失败时未导出LeRobot状态／动作。
采集器校验帧数、640×480、时间覆盖及逐帧解码后，将其编码为 `diagnostic_videos/`，
只作为明确标注的失败视频纳入汇总，不补造训练数据，也不重新采样挑选失败。

完整配置与工况目录校验：

```bash
./app/python.sh .agents/skills/rsr-task-development/scripts/check_task_configs.py
./app/python.sh -m ruff check real2sim2real scripts
./app/python.sh -m pytest -q
```

## 5. 验收

18个柔性包裹矩阵工况沿用各配置中的原门限：转运／摇晃两部分质心持续抬升≥80mm、双侧接触≥95%、
最大边长比≤1.3、台面穿入≤1.5mm、落点≤80mm、释放回落≥10mm，最终受台面支撑、夹指接触归零。
衣物须仍在袋内，无固定粒子。只有明确标为spring的两个工况启用有限力局部弹簧，且在释放时关闭。
独立实际URDF FK检查闭合末TCP误差≤5mm、姿态误差≤1°；源状态与可见几何差≤0.01mm。
双向代理另检查逐子步位置／姿态一致性和原生电机实际力限，不能把60N上限当作持续夹紧力。

衣服、刚体及原装短指包裹基线沿用自己的完整checks和配置门限，不将包裹矩阵的门限套用到不同模型。
纸箱由原始角点和全频率刚体轨迹独立复算，核对原尺寸／质量、提升、释放、落点、穿入和最终支撑。
螺纹装配由真实M20螺纹SDF与hydroelastic接触决定进给，控制中没有“转角→轴向位移”规则：
至少2圈／6mm进给，导程相对误差≤25%，旋入接触帧≥95%，保持漂移≤1mm，全程非导程前进≤2mm。
固定螺母与导向主轴方案不覆盖自由拾取、工具移走后的保持或指定预紧力认证。

独立校验入口：

- `validate_package_pickplace.py`、`run_package_matrix.py::audit_physics`、`audit_package_grasp_pose.py`
- `validate_carton_pickplace.py`
- `validate_screw_assembly.py`
- 共用LeRobot v3数据校验：视频逐帧解码、H.264、分辨率、fps、帧数、状态／动作有限且同步。

## 6. 本轮状态与历史

配置平铺前后已逐份核对生效模型：物理参数和验收门限一致，仅输出／任务名、分辨率及运行超时预算更新。
本轮最新全仓回归为266通过、3跳过（26.09s），覆盖配置路由、命名／640采集约定、双GPU队列不会重复领取同一工况及中断恢复。
3个跳过项依赖USD／Newton SDK导入环境；随后在该环境以CPU补跑，3项全部通过（4.85s）。
补跑检查了USD材质引用、夹垫碰撞体SI尺寸及闭合位姿，未增加GPU仿真实例。
上述全仓回归包含失败视频恢复、首次运行初始化与存活子进程接管；任务专项42项也已通过（1.15s）。
控制进程中断时，恢复逻辑先核对各工况仍存活的进程和输出，沿用它们所在的GPU通路等待完成，
不会在原子进程仍求解时重复采集；只有中断且未完成的物理轨迹才从初态重跑。

本轮从初态重新采集已完成：**27项成功目标（26项抓取＋1项螺纹装配）**通过物理、数据及各路相机关键帧复核，
**2项失败对照**完成诊断视频验收，明确保留物理失败。正式目录共27个与配置同名的结果文件夹。
配置快照、运行ID、发布路径及视觉复核图片哈希已逐项核对，见
[发布审计](../outputs/debug/physic_solver/recapture/20261113T093020Z/publication_audit.json)。
完整结果见 [collection_summary.json](../outputs/physic_solver/collection_summary.json)及各同名目录的
`result.json`、`validation.json`、`qa/visual_review.json`、`run_note.json`。

| 分类 | 成功目标数 | 本轮结果 |
|---|---:|---|
| 机器人／接触算法基础组合 | 6 | 通过 |
| package01位置与姿态 | 6 | 通过 |
| package01衣物重量变化 | 4 | 通过 |
| package02–06种类变化 | 5 | 通过 |
| 三种摇晃／附着＋package07–08纸箱 | 5 | 通过 |
| 固定螺母＋连续旋转主轴 | 1 | 通过 |
| Featherstone包裹、统一XPBD衣服 | 2项对照 | 物理失败，诊断视频有效 |

[本轮队列状态](../outputs/debug/physic_solver/recapture/20261113T093020Z/queue.json) · [配置迁移对照](../outputs/debug/physic_solver/recapture/20261113T093020Z/config_migration.json) · [旧输出归档](../outputs/debug/physic_solver/recapture/20261113T093020Z/previous_outputs/) · [旧实验说明](../outputs/debug/physic_solver/recapture/20261113T093020Z/previous_physic_solver.md)

每个通过配置目前以一条episode为目标，失败／中断和全部重试保留在debug及run_note中，不据此估算成功率。
已有station_clothing工况历史最大边长比接近1.3门限，须保持门限并逐次验收，不直接沿用旧成功标签。
本轮package02首轮在2.15s因边长比1.30094超过1.3而提前停止，未录制为成功视频；第二次保留同一门限并通过物理和双相机视觉验收。
本轮X35mm首轮完成抓放但落点误差86.5mm，超过80mm门限而拒绝；原参数第二次完整运行误差42.8mm并通过。
两个尝试均保留，说明该通过记录不能视作确定性复现或成功率统计。
90°双向代理首轮新运行在开爪前已滑落，释放回落为0、落点误差103.9mm而拒绝；
代理几何一致性与60N夹指实际力限检查通过。第二次保持原参数完整运行，通过新的物理审计与双相机视觉验收；
该结果需配合首次失败解读，不能沿用历史通过标签或估计成功率。
package05首轮已完成夹起、转运与释放，但落点误差84.6mm超过80mm门限，已拒绝发布并保留；
原参数第二次运行误差5.7mm，通过原门限及双相机视觉验收。两次结果均保留，不能据此估计成功率。
中央弹簧摇晃首轮在12.333s因最大边长比1.35001超过1.3而提前停止；第二次保留原参数，最大边长比1.27325，
通过原门限和双相机视觉验收。所有失败及重试均保留在本轮debug尝试目录。

三种摇晃工况的本轮实测结果如下。抬升为相对初始质心位置，衣物／袋膜分别统计：

| 抓点／附着 | 摇晃最低抬升（mm，衣物／袋膜） | 双侧接触 | 最大边长比 | 释放回落（mm） | 落点误差（mm） |
|---|---:|---:|---:|---:|---:|
| X35mm／无附着 | 233.5／229.7 | 100% | 1.12420 | 117.6 | 8.7 |
| 中央／有限力弹簧 | 190.7／204.1 | 99.33% | 1.27325 | 78.4 | 24.1 |
| X35mm／有限力弹簧 | 233.2／228.9 | 100% | 1.12480 | 115.4 | 37.2 |

实际URDF FK复算的偏航／俯仰峰峰值均约20.37°／10.21°；平移及角度范围通过原有“达到配置名义峰峰范围至少一半”检查。
两种弹簧辅助均在释放时关闭，无固定粒子。
螺纹装配本轮旋转3.49124圈、轴向旋入8.68827mm，导程误差0.456%，保持漂移0.0896mm；独立审计和双视角复核通过。

## 7. 视频拼接

本轮已在全部新数据完成数值验收与实际视觉复核后生成8个视频，全部位于 `outputs/physic_solver/`。
源视频共56路（抓取54路＋螺纹装配2路），均来自本轮运行。成品为1920×1080、15fps、H.264。
拼接入口如下；已有同名成品时拒绝覆盖：

```bash
./app/python.sh scripts/physic_solver/create_video_montages.py
```

按机器人求解器／接触算法、抓取位置、重量、包裹种类、摇晃／纸箱分屏，保留全景及腕部视角；
标注物体编号、机器人型号、机器人求解器、接触／柔体算法和通过状态。
片段全程原速，短片段结束后保持末帧并显示结束标记；失败对照明确标注，螺纹装配另做附篇。
总片检查完整解码、帧数和分段顺序，另比较全部解码帧MD5，确认拼接没有丢帧或重复片段。
本轮总片1624帧全部与六个抓取分类片的顺序解码帧一致。已实际查看7个分片各3张预览、总片3张预览，
共24张，核对布局、标签、摇晃段、失败状态与结束标记；人工检查不覆盖每一帧。

| 视频 | 时长（秒） | 内容 |
|---|---:|---|
| [抓取总片](../outputs/physic_solver/grasp_all_comparison_labeled.mp4) | 108.27 | 下列01–06依次拼接，含明确标注的失败对照 |
| [01_solver_comparison](../outputs/physic_solver/01_solver_comparison.mp4) | 21.53 | 6种机器人／接触算法基础组合 |
| [02_package_positions](../outputs/physic_solver/02_package_positions.mp4) | 17.80 | 6种抓取位置与姿态 |
| [03_package_weights](../outputs/physic_solver/03_package_weights.mp4) | 14.80 | 100／200／350／500g衣物 |
| [04_package_types](../outputs/physic_solver/04_package_types.mp4) | 14.80 | package02–06 |
| [05_shake_and_cartons](../outputs/physic_solver/05_shake_and_cartons.mp4) | 17.80 | 3种摇晃／附着＋package07–08 |
| [06_failed_grasp_controls](../outputs/physic_solver/06_failed_grasp_controls.mp4) | 21.53 | 2项本轮失败对照 |
| [07_screw_assembly_appendix](../outputs/physic_solver/07_screw_assembly_appendix.mp4) | 9.00 | 螺纹装配双视角附篇，不计入抓取总片 |

来源、逐文件SHA256、分片顺序、完整解码和实际视觉复核记录见
[grasp_montage_index.json](../outputs/physic_solver/grasp_montage_index.json)。

## 8. 代码与资产

生命周期保持 `init_world → arrange_scene → place_robot → start_simulation → finally: close`。
各能力域仍每任务一个文件：

- [scene_physic_solver.py](../real2sim2real/scene/scene_physic_solver.py)
- [robot_physic_solver.py](../real2sim2real/robot/robot_physic_solver.py)
- [planner_physic_solver.py](../real2sim2real/planner/planner_physic_solver.py)
- [job_physic_solver.py](../real2sim2real/job/job_physic_solver.py)
- [physic_solver_utils.py](../real2sim2real/utils/physic_solver_utils.py)：包含统一工况名称与分组映射。
- [simworld_physic_solver.py](../real2sim2real/simworld/simworld_physic_solver.py)：真实SDK纯物理世界。

资产准备脚本为 `prepare_physic_solver_assets.py`、`prepare_official_newton_cloth.py`、
`prepare_package_assets.py` 与 `prepare_screw_assets.py`；后端注册见 `pyproject.toml`。
运行时继续使用Isaac `python.sh`，没有创建 `.venv`，无新增包依赖，不修改其他任务配置、资产和输出。
