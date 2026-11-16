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
| `screw01` | M20 loose螺栓／螺母（导向主轴） | `assets/PhysicSolver/Screw/M20/` |
| `screw02` | 同一 M20 loose螺纹副 + UR5e 电批工位 | `assets/PhysicSolver/Screw/M20/`、`Robot/ur5e/` |
| `screw03` | 双 UR5e，自由螺母拾取后旋入 | 同上，新增夹爪和支座均由代码生成 |

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
| [mujoco_hydroelastic_screw02_ur5e](../configs/physic_solver/mujoco_hydroelastic_screw02_ur5e.toml) | UR5e 电批，1K / 30fps | `outputs/physic_solver/mujoco_hydroelastic_screw02_ur5e/` |
| [mujoco_hydroelastic_screw02_ur5e_debug](../configs/physic_solver/mujoco_hydroelastic_screw02_ur5e_debug.toml) | UR5e 电批，640px / 15fps | `outputs/debug/physic_solver/mujoco_hydroelastic_screw02_ur5e_debug/` |
| [mujoco_hydroelastic_screw03_dual_ur5e](../configs/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e.toml) | 双 UR5e，夹起螺母后旋入，1K / 30fps | `outputs/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e/` |
| [mujoco_hydroelastic_screw03_dual_ur5e_debug](../configs/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e_debug.toml) | 双臂调试，640px / 15fps | `outputs/debug/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e_debug/` |

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

## 9. UR5e 电批拧螺栓 demo

本变体丰富现有 `physic_solver`，旧的 `screw01` 主轴案例和已验收输出保持原样。
目标是演示 UR5e 的法兰安装电批，以连续旋转套筒将 M20 六角头螺栓拧入固定螺母。
螺栓起始已预入牙 4 mm、已被理想刚性套筒夹住；不包含取料、视觉找孔、自动上批或退出套筒。
机械臂先进行 3 mm 的平滑加载，随后保持法兰位置；浮动轴的弹簧预压和真实螺纹接触决定进给。
这是旋入和保持演示，尚不模拟拧紧至贴合后的预紧力或扭矩停机。

运行环境沿用 Isaac Sim 6.2 rc.14、Kit 内 Newton 1.6.1 / Warp 1.17.0、MuJoCo Warp。
没有新增 pip 依赖。视频处理依赖系统 `ffmpeg` / `ffprobe`，采集器复用现有 GPU 预算单调时钟补丁，
编译该小型补丁需要 `cc`（Debian/Ubuntu 的 `build-essential`）。没有创建 `.venv`。

### 9.1 资产和物理

- UR5e 来源：用户的 `/home/fufa/Projects/screwdriving_robot_usd/screwdriving_robot_usd/universalrobots_ur5e/`。
  复制整个 45 文件、约 15 MiB 的包到 `assets/PhysicSolver/Robot/ur5e/`；不修改源目录。
  上游为 NVIDIA isaacsim-assets，固定 revision `a0d9dfbac5163a7005442ae4fe8f954ff6780093`；
  主包 BSD-3-Clause，附带依赖保留其许可。`provenance.json` 记录逐文件 SHA-256、大小和来源。
- 该源包只有机械臂，没有电批。电批机身、法兰适配器、浮动鼻部、转轴和套筒由 Robot 用 USD 几何生成，
  不引用其它任务资产；套筒橙色标记用于辨认连续旋转，机身保持固定。无需额外 Blender 或网络资产。
- 真实 M20 螺栓／螺母沿用 `assets/PhysicSolver/Screw/M20/` 的 IsaacGymEnvs Factory 网格；
  桌面沿用该任务自己的 `Table/Aligned.usd`。所有本地 layer 位于 `assets/PhysicSolver/` 内。
- 六个机械臂旋转关节、轴向浮动关节、无限旋转关节属于同一 Newton articulation，均由动力学求解。
  源 USD 的 7 个连杆质量、惯量和凸分解碰撞体被导入；角驱动显式改为 SI 增益，
  `kp=12000 N·m/rad`、`kd=400 N·m·s/rad`、力矩限值 `150 N·m`，不是实机控制器标定结果。
- 继续选择 **Newton direct + MuJoCo + hydroelastic SDF**，以保留已验证的实际螺纹接触。
  未选用电池拆解的 PhysX mimic 螺旋约束，因为该变体要保持原验证案例的真实螺纹进给。
  物理 120 Hz，每步 4 子步，SDF 128、刚度 `1e11`、摩擦 `0.01`，与原案例一致。
  导入接口已对照本机实现和 [Newton 1.6.1 ModelBuilder 文档](https://newton-physics.github.io/newton/1.6.1/api/_generated/newton.ModelBuilder.html)。
- 螺栓头与套筒为**明确的理想刚性夹持模型**，不验证套筒接触滑移或防脱。
  没有螺栓／螺母间的螺旋关节、吸附或按转角指定轴向轨迹。
  电批外壳是可视几何，尚未做整机与工位的碰撞验收；本 demo 不能替代工艺安全或负载验证。

### 9.2 组件与命令

复用 Scene `screw_assembly`、Job `screw_assembly`，新 Robot / Planner 均为 `ur5e_screwdriver`，
仍按 `BaseJob.run()` 生命周期执行。代码写入任务原有的 robot / planner / utils 文件；
`ScrewRuntime` 用实际螺栓世界高度计算旋入量，避免将浮动轴相对位移误认为总进给。
记录 8 维实测状态：六个机械臂角度、浮动轴位移、连续主轴角度；8 维动作：六个位置目标、
弹簧位移目标、主轴速度目标。逐物理步 `trajectory.json` 保留与视频采样对应的测量和命令。

```bash
# 资产准备；已存在但哈希不同的文件拒绝覆盖。
./app/python.sh scripts/physic_solver/prepare_screw_arm_assets.py

# 只跑物理（先选择新的 output_dir）。
RSR_JOB__record_rgb=false \
RSR_JOB__output_dir=outputs/debug/physic_solver/ur5e_physics_new \
./scripts/rsr collect run \
  -c configs/physic_solver/mujoco_hydroelastic_screw02_ur5e_debug.toml

# 640×480、15fps、MinimalRendering 纹理预览，仍是 RTX。
./scripts/rsr collect run \
  -c configs/physic_solver/mujoco_hydroelastic_screw02_ur5e_debug.toml

# 正式 1024×768、30fps、PathTracing：64 samples/pixel，4 次反弹，空间降噪。
# 采集器在 debug 目录暂存，进行物理、状态/动作对齐、全视频解码和关键帧校验。
./app/python.sh scripts/physic_solver/collect_screw_arm_demo.py

# 查看 run/capture/qa/keyframes.jpg 和双视角视频后，在
# run/capture/qa/visual_review.json 记录 passed=true、检查范围和观察结果，再发布：
./app/python.sh scripts/physic_solver/collect_screw_arm_demo.py --publish-run <run>

# 独立重验：
./app/python.sh scripts/physic_solver/validate_screw_assembly.py <output>
```

采集器拒绝覆盖已有输出。再次运行时通过 `RSR_JOB__output_dir` 指定新目录。
两个配置只改变渲染、分辨率、fps 和输出目录，物理与验收门限一致。
历史 `recapture_physic_solver.py` 的 640px 矩阵明确只读取其既有清单，不自动加入本 1K demo。
正式目录包含 LeRobot v3、原始轨迹、配置、验收报告、双路 MP4 和 `demo.mp4` 同步并排预览。

### 9.3 验收与调试记录

沿用原门限：旋转至少 2 圈、旋入至少 6 mm、导程误差不超过 25%、保持漂移不超过 1 mm、
全过程螺旋残差不超过 2 mm、旋入阶段至少 95% 的物理帧有接触、接触数量小于 40000。
新增机械臂角度跟踪误差不超过 0.02 rad、8 维状态/动作与原始物理日志逐帧一致、
本地资产 layer 无跨任务依赖。正式结果另需双相机关键帧视觉复核。

首次物理通过（`outputs/debug/physic_solver/ur5e_development/physics04`）：
3.4991 圈、8.7093 mm 旋入、0.4386% 导程误差、0.1416 mm 保持漂移、
0.4195 mm 最大螺旋残差、98.69% 接触帧、0.001223 rad 最大机械臂跟踪误差。
640px 调试全流程通过（143 帧/相机）：3.4942 圈、8.5792 mm 旋入、1.7882% 导程误差。
两路的起点、中点、终点关键帧已实际查看；正式版本将全景焦距由 35 mm 改为 28 mm，避免顶部裁切。

失败与修复均保存在 `outputs/debug/physic_solver/ur5e_development/`：

| 记录 | 现象与修复 |
|---|---|
| `physics01` | 相机继承时列表追加，出现同名相机；debug scene 显式替换该 section。 |
| `physics02` | 引用根已有 translate 属性；先清除 xformOpOrder 再覆盖变换。 |
| `physics03` | 临时 stage 默认 Y-up，机器人被导入器旋转，批头偏离螺母；显式设 Z-up、米。该失败未发布。 |
| `physics04` | 坐标修复后全部物理门限通过，未降低门限。 |
| `original_screw_regression` | 原主轴版本复测通过：3.4902 圈、8.6977 mm、0.3184% 导程误差。 |

GPU 接触求解在不同运行间有小幅差异，上述每次运行独立验收，没有选帧拼接或重置后拼接。
回归：无 Kit 测试 355 通过 / 3 跳过（基线 350 / 3），ruff、全任务配置校验通过；
Newton direct 自由落体/只渲染冒烟通过。未验证官方 LeRobot SDK 加载、自动取螺栓、脱离套筒和终拧扭矩。

### 9.4 正式结果（2026-11-15，EST）

[双视角 demo.mp4](../outputs/physic_solver/mujoco_hydroelastic_screw02_ur5e/demo.mp4) ·
[全景](../outputs/physic_solver/mujoco_hydroelastic_screw02_ur5e/overview.mp4) ·
[螺纹特写](../outputs/physic_solver/mujoco_hydroelastic_screw02_ur5e/threads.mp4) ·
[关键帧](../outputs/physic_solver/mujoco_hydroelastic_screw02_ur5e/qa/keyframes.jpg) ·
[独立验收](../outputs/physic_solver/mujoco_hydroelastic_screw02_ur5e/validation.json)。

正式运行 `formal01` 一次通过，无重跑筛选、无中途重置。仿真 9.5 s / 1140 物理步，
每路 285 帧、1024×768 / 30 fps，双路均完整解码；并排预览 2048×768 / 30 fps，时长 9.5 s。
采集墙钟约 716 s；主轴、螺栓和机器人均来自同一次连续求解。

| 指标 | 正式实测 | 门限 |
|---|---:|---:|
| 旋转圈数 | 3.4964 圈 | ≥ 2 圈 |
| 实际旋入量 | 8.8930 mm | ≥ 6 mm |
| 导程误差 | 1.7385% | ≤ 25% |
| 停转保持漂移 | 0.3948 mm | ≤ 1 mm |
| 最大螺旋残差 | 0.4723 mm | ≤ 2 mm |
| 旋入期接触帧比例 | 99.1657% | ≥ 95% |
| 峰值接触数 | 98 | < 40000 |
| 最大机械臂跟踪误差 | 0.0012204 rad | ≤ 0.02 rad |

Job 的 11 项检查和独立校验的 11 项检查均通过。8 维状态、动作逐帧与物理日志一致，
两路起点、中点、终点关键帧及额外的初段／中段高分辨率对照已实际查看；
`qa/visual_review.json` 记录复核范围，未声称逐帧人工检查全部视频。
旧结果均保留；本变体只新增上述正式目录。

## 10. 双 UR5e：夹起螺母后拧入螺栓

本变体在单臂 demo 之上增加第二只 UR5e 与可见平行夹爪。螺母起始放在 100 mm 高的蓝色支座上，
夹爪闭合后将它抬起约 160 mm；持稳后，电批臂下降并靠轴向弹簧预压建立接触，再旋入螺栓。
螺母是独立的 **6 自由度刚体**，没有固定到世界、夹爪或机械臂，也没有吸附力或抓取附着约束。
电批套筒仍按第 9 节采用理想刚性夹持螺栓头；取螺栓和终拧预紧力仍不在本变体范围内。

### 10.1 组件、资产与控制

| 能力域 | 后端 | 实现 |
|---|---|---|
| Scene | `dual_screw_assembly` | `DualScrewScene`，沿用桌面和 M20 网格，增加可见的拾取支座 |
| Robot | `dual_ur5e_screwdriver` | `DualUr5eScrewRobot`，两只物理 UR5e、电批、两根独立驱动的夹指 |
| Planner | `dual_ur5e_screwdriver` | `DualScrewPlanner` / `DualScrewRuntime` |
| Job | `screw_assembly` | 复用原固定生命周期、记录与导出 |

两臂复用同一份 `assets/PhysicSolver/Robot/ur5e/`；没有新增外部资产或 pip 依赖。
夹爪由代码生成，橙色可见夹指与实际 hydroelastic 碰撞盒尺寸相同：12×8×100 mm。
每侧夹指的开度坐标独立，初始 28 mm，正式配置闭合目标 12 mm，增益 30000 N/m、150 N·s/m，
力上限 100 N/侧，摩擦系数 1.5。它是本 demo 的显式平行夹爪，不宣称是原装 Robotiq。
这些值为仿真设计参数，未做实机标定。第一臂底座偏移 `[-0.45,0,0]`，第二臂 `[0.45,0.28,0]`。

仍使用 Newton direct + MuJoCo Warp（本机 3.12.0）、实际 M20 螺纹 SDF，
120 Hz × 4 子步。螺母／螺栓保持 KH=1e11、摩擦 0.01；夹指 KH=1e10、SDF 64。
两臂、夹指、自由螺母由同一求解器做双向动力学，不用运动学代理移动螺母。
动态螺母的轴向进给和转角都在**螺母坐标系**下由实际刚体位姿重新计算，防止把整物抬升误认为旋入。
螺纹导程残差只在实际旋入和保持阶段计算；拾取、抬升及空中对准不应服从螺距关系。

双臂变体显式使用 `cone="elliptic"`、`impratio=10`。
本机 Newton 1.6.1 的 `SolverMuJoCo` 构造函数与接触参数 kernel 已核对支持该选项；
[MuJoCo 防滑说明](https://mujoco.readthedocs.io/en/latest/modeling.html#preventing-slip)
解释了软接触模型在持续切向载荷下的缓慢蠕滑，以及提高切向相对阻抗的作用。
未使用 NoSlip 后处理或额外锁定约束；旧的单臂／主轴案例仍使用其原始求解器默认值。

阶段：夹紧 0.8 s → 抬升 1.5 s → 电批对准 1.5 s → 预压接合 1 s → 旋入 7 s → 停转保持 1 s。
视频采样 30 fps，两路 1024×768 / PathTracing 64 spp，4 次反弹；调试为 640×480 / 15 fps / MinimalRendering。
并排预览额外加 52 像素阶段标题栏，原始两路视频不加字、不裁切。
记录 16 维机器人状态和动作，顺序为：电批臂 6 关节、轴向浮动、主轴；持螺母臂 6 关节、负侧／正侧夹指。
`trajectory.json` 另记录螺母／螺栓／夹爪 TCP 的原始刚体位姿、双侧接触和支座接触数量。

### 10.2 运行与验收

```bash
# 只跑物理，选择未使用的输出目录。
RSR_JOB__record_rgb=false \
RSR_JOB__output_dir=outputs/debug/physic_solver/dual_ur5e_physics_new \
./scripts/rsr collect run \
  -c configs/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e_debug.toml

# 正式双视角采集，先写 debug 暂存目录；完成后进行独立验收和关键帧复核。
./app/python.sh scripts/physic_solver/collect_screw_arm_demo.py \
  --config configs/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e.toml

# 实际复核关键帧后，按第9节写 visual_review.json 并发布：
./app/python.sh scripts/physic_solver/collect_screw_arm_demo.py --publish-run <run>

# 独立校验同时检查拾取和螺纹，而非只检查动作执行完毕。
./app/python.sh scripts/physic_solver/validate_screw_assembly.py <output>
```

保留原螺纹门限：≥2圈、≥6 mm进给、导程误差≤25%、保持漂移≤1 mm、最大螺旋残差≤2 mm、
旋入期接触帧≥95%、接触容量不溢出。新增加：

- 旋入／保持期螺母相对初始位置的最小抬升 ≥130 mm；支座和桌面接触为零。
- 搬运到保持全程双侧夹指接触比例 ≥95%。
- 螺母相对持物 TCP 的最大滑移 ≤3 mm，转动 ≤0.05 rad。
- 两臂最大关节跟踪误差 ≤0.02 rad。
- 从原始刚体位姿重算相对转角、轴向位置、抬升与滑移；16维状态/动作逐帧与物理记录一致。
- 两路视频完整解码，起始、夹紧后、抬升后、旋入中段、保持末段均实际查看。

### 10.3 调试记录

记录目录为 `outputs/debug/physic_solver/dual_ur5e_development/`，失败和对照没有发布到正式目录。

| 运行 | 现象、分析与处理 |
|---|---|
| `physics01` | USD 深度优先导入按负侧、正侧排序夹指，原本按正侧、负侧断言；按实际映射记录16维坐标。 |
| `physics02`、`rgb_probe01_run` | 初始手腕配置与桌面发生干涉，夹爪被顶高约49 mm，未夹到螺母；将夹爪绕工具轴转180°，保留相同夹持方向并获得手腕净空。 |
| `physics03` | 真实抬起156.2 mm，旋入8.53 mm，但最大滑移3.69 mm，失败；未放宽3 mm门限。 |
| `physics04` | 闭合目标13→12 mm，增加驱动夹紧力；滑移2.991 mm，虽通过但余量不足，保留为对照。 |
| `physics05` | 依据本机实现和MuJoCo资料，将双臂变体切向阻抗比设为10；滑移1.057 mm，抬升159.24 mm，旋入8.346 mm，全部通过。 |
| `rgb01` | 640px完整调试、物理与独立数据验收通过，滑移0.921 mm；复核后调整特写至正前方俯视，以减少夹指遮挡。 |

以上实验保持所有验收门限不变。旧主轴和单臂配置、资产和已发布结果保留。

`rgb02` 的双视角与阶段标题已复核并发布到 debug 目录：抬升159.307 mm、旋入8.607 mm、
滑移0.691 mm、双侧接触100%。正式版另将螺母显示色提亮，方便在深色夹爪前辨认。
新增代码回归为360通过 / 3跳过，ruff、全任务配置校验通过；
单臂UR5e和原主轴的物理回归与独立校验均通过（分别旋入8.730 mm、8.874 mm）。

### 10.4 正式结果（2026-11-15，EST）

[双臂完整演示](../outputs/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e/demo.mp4) ·
[全景](../outputs/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e/overview.mp4) ·
[特写](../outputs/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e/threads.mp4) ·
[拾取前后对照](../outputs/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e/qa/pickup_review.jpg) ·
[关键帧](../outputs/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e/qa/keyframes.jpg) ·
[独立验收](../outputs/physic_solver/mujoco_hydroelastic_screw03_dual_ur5e/validation.json)。

正式 `formal01` 一次通过：12.8 s、1536 个物理步、每路384帧，1024×768 / 30 fps；
并排预览含阶段标题栏，为2048×820 / 30 fps。采集墙钟1268.13 s（约21.1分钟）。
两路原始视频及并排成品都完整解码，无跨运行拼接或中途重置。

| 指标 | 正式实测 | 门限 |
|---|---:|---:|
| 旋入／保持期最小螺母抬升 | 159.279 mm | ≥130 mm |
| 螺母相对夹爪最大滑移 | 0.612 mm | ≤3 mm |
| 最大螺母转动 | 0.0016802 rad | ≤0.05 rad |
| 双侧夹指接触帧比例 | 100% | ≥95% |
| 旋入期支座／桌面接触 | 0 | 0 |
| 螺栓相对螺母旋转 | 3.4816圈 | ≥2圈 |
| 相对轴向旋入 | 8.5795 mm | ≥6 mm |
| 导程误差 | 1.4304% | ≤25% |
| 停转后相对轴向漂移 | 0.01676 mm | ≤1 mm |
| 最大螺旋残差 | 0.1470 mm | ≤2 mm |
| 螺纹接触帧比例 | 100% | ≥95% |
| 两臂最大跟踪误差 | 0.0177843 rad | ≤0.02 rad |
| 峰值接触数 | 205 | <40000 |

Job的15项检查、独立校验的18项检查全部通过；两路各384帧的16维实测状态与动作均与物理日志对应。
两路五个阶段关键帧、拾取及旋入对照、成品末段已实际查看，范围记录在 `qa/visual_review.json`。
人工复核不代表逐帧观看所有视频。旧单臂、原主轴结果均保留；本次正式输出只新增 `screw03_dual_ur5e`。

### 10.5 加深旋入，螺栓穿出螺母

新增 `mujoco_hydroelastic_screw04_dual_through{,_debug}.toml`，复用双臂、M20资产、
接触求解器和原有全部门限，只将旋拧阶段从7 s延长为14 s，总时长19.8 s。
新增 `minimum_tip_protrusion_m=0.005`：整个停转保持阶段，螺栓尖端须在螺母下表面以外至少5 mm。
该字段默认0，不改变旧配置行为；原双臂结果末帧仍差1.079 mm才穿出，因此不能用旋入量代替穿出量验收。

穿出量由真实螺栓／螺母刚体位姿计算：把螺栓尖端变换到螺母局部坐标系，
与螺母下表面比较；独立校验重新读取OBJ边界和完整位姿，不依赖电批目标值。
螺栓尖端距其参考原点65 mm，螺母下表面局部z为−36 mm（两者网格均翻转180°）。
螺母仍是六自由度刚体，没有新增吸附或固定约束。

为便于直接辨认转动，`screw04` 在螺栓原网格上增加不对称蓝／黄／白纵向涂装、红色角度标记和短黑刻度。
UV固定在螺栓局部顶点上，随求解器输出的真实姿态一起旋转，没有独立滚动贴图或修改螺纹碰撞体。
特写相机改为约12°俯角，减少螺母对穿出尖端的遮挡。旧配置 `bolt_texture_path=None`，保留原金属外观。
贴图由 `prepare_screw_arm_assets.py` 生成于 `assets/PhysicSolver/Screw/M20/Textures/`，
1024×512 PNG、CC0、SHA-256见同目录 `provenance.json`，无需下载额外资产。
材质采用 [UsdPreviewSurface标准节点链](https://openusd.org/release/spec_usdpreviewsurface.html)，
具体效果以本机Isaac Sim 6.2预览和正式画面为准。

```bash
# 640×480 / 15 fps 调试，MinimalRendering
./app/python.sh scripts/physic_solver/collect_screw_arm_demo.py \
  --config configs/physic_solver/mujoco_hydroelastic_screw04_dual_through_debug.toml
# 1024×768 / 30 fps 正式采集，PathTracing
./app/python.sh scripts/physic_solver/collect_screw_arm_demo.py \
  --config configs/physic_solver/mujoco_hydroelastic_screw04_dual_through.toml
```

与第9节相同，采集先留在debug暂存目录，独立校验和实际视觉复核后用 `--publish-run` 发布。
开发记录：`outputs/debug/physic_solver/dual_through_development/`。
`physics01` 一次通过：6.9169圈、旋入17.158 mm、保持期最小穿出7.501 mm、
最大夹持滑移0.867 mm、导程误差0.7754%；最终浮动位移−24.410 mm，在−40 mm行程限位以内。

带纹理的 `rgb03_texture` 已发布到
`outputs/debug/physic_solver/mujoco_hydroelastic_screw04_dual_through_debug/`：
6.8643圈、旋入16.936 mm、保持期最小穿出7.355 mm、最大夹持滑移1.008 mm、导程误差1.3107%。
两路各297帧（640×480 / 15 fps），合成1280×532 / 19.8 s；完整解码和独立验收通过。
实际查看两路五个阶段关键帧、8.0/8.2/8.4/8.6 s连续旋转对照及19.6 s末帧，
能看到涂装绕轴变化和螺栓尖端穿出。此前 `rgb01` 为原俯视角，`rgb02` 为降低视角但尚无涂装的对照，均保留在开发目录。
365项单元测试通过、3项跳过，ruff与全部物理验证配置校验通过；旧单臂和双臂已发布数据重新独立校验通过。

正式 `formal01` 一次通过并发布（2026-11-16，EST）：
[完整demo](../outputs/physic_solver/mujoco_hydroelastic_screw04_dual_through/demo.mp4) ·
[旋转特写](../outputs/physic_solver/mujoco_hydroelastic_screw04_dual_through/threads.mp4) ·
[连续旋转对照](../outputs/physic_solver/mujoco_hydroelastic_screw04_dual_through/qa/rotation_review.jpg) ·
[末帧](../outputs/physic_solver/mujoco_hydroelastic_screw04_dual_through/qa/final_review.png) ·
[独立验收](../outputs/physic_solver/mujoco_hydroelastic_screw04_dual_through/validation.json)。

两路各594帧，1024×768 / 30 fps / PathTracing 64 spp、4反弹；
合成2048×820 / 19.8 s，共2376物理步，采集墙钟1366.15 s（约22.8分钟）。
Job的16项和独立校验的20项检查全部通过，包括贴图来源哈希、实际状态／动作对齐、无螺母附着和任务内资产依赖。

| 指标 | 正式实测 | 门限 |
|---|---:|---:|
| 保持期最小尖端穿出 | 7.4958 mm | ≥5 mm |
| 相对旋转 | 6.9151圈 | ≥2圈 |
| 相对旋入 | 17.1535 mm | ≥6 mm |
| 导程误差 | 0.7765% | ≤25% |
| 最小螺母抬升 | 159.1781 mm | ≥130 mm |
| 最大夹持滑移 | 0.7669 mm | ≤3 mm |
| 最大螺母转动 | 0.0016199 rad | ≤0.05 rad |
| 双侧夹指／螺纹接触比例 | 100% / 100% | ≥95% / ≥95% |
| 旋入期支座／桌面接触 | 0 | 0 |
| 停转后轴向漂移 | 0.00796 mm | ≤1 mm |
| 最大螺旋残差 | 0.1453 mm | ≤2 mm |
| 两臂最大跟踪误差 | 0.0177843 rad | ≤0.02 rad |
| 峰值接触数 | 196 | <40000 |

已实际查看两路五个阶段的编码视频关键帧、19.6 s成品末帧，以及8.0/8.2/8.4/8.6 s原始采集连续帧。
可直接辨认黄色／红色涂装转离正面、蓝色涂装转向正面，末段尖端在螺母下方清楚可见。
两路原始视频和并排成品完整解码通过；复核范围和限制见 `qa/visual_review.json`。
没有放宽原门限，也没有覆盖旧单臂、双臂结果；全部对照与采集记录见开发目录 `run_note.json`。

```bash
./app/python.sh scripts/physic_solver/validate_screw_assembly.py \
  outputs/physic_solver/mujoco_hydroelastic_screw04_dual_through
```

### 10.6 拧完后松开螺母夹爪

`mujoco_hydroelastic_screw05_dual_release{,_debug}.toml` 在带纹理、穿出螺母版本之后，
增加1 s张爪和1 s自由状态观察，总时长21.8 s。电批停转、保持夹住螺栓，第二只机械臂原地张开两指。
螺母保持六自由度，不添加固定、吸附或隐藏托块。

新增可选字段 `release_s/post_release_s` 默认均为0，保留旧六阶段流程；两者须同时为正才启用。
夹持滑移、抗旋转和双侧接触门限适用于抬升至松爪前；松爪后独立要求两指实测坐标≥25 mm、
两指与螺母零接触、螺纹接触比例≥95%，以及保持至少5 mm尖端穿出。没有放宽原有数值门限。

原参数摩擦系数0.01在失去夹指后出现重力反驱：`physics01` 螺母相对螺栓回转3.558圈，
全局下降25.274 mm，末段穿出量变为−1.548 mm，失败。
`physics02/03` 将整个episode的螺纹摩擦系数恒定设为0.1，松开后螺纹接触100%，
但原−3 rad/s电批目标只完成约5.84圈，穿出约2.2 mm，仍未达到5 mm门限。
最终将目标转速改为−4 rad/s，其他旋拧时长保持14 s；`physics04` 完成7.976圈、旋入19.841 mm，
停转期穿出10.075 mm、松爪后最小穿出9.855 mm，独立校验通过。
摩擦系数为本demo选定并验证的仿真参数，尚未用实物材料试验标定。

`latch_axial_hold=true` 在首次停转时锁存轴向位置驱动目标，避免每步把目标改成当前位姿而积累下沉。
它仍使用原有限刚度驱动和真实行程限位，允许受载弹性位移，不直接固定刚体。
该字段默认false，旧版本控制和配置保持不变。
调试、失败和对照记录保存在 `outputs/debug/physic_solver/dual_release_development/`。

```bash
./app/python.sh scripts/physic_solver/collect_screw_arm_demo.py \
  --config configs/physic_solver/mujoco_hydroelastic_screw05_dual_release_debug.toml
# 正式采集；复核后仍按第9节用 --publish-run 发布
./app/python.sh scripts/physic_solver/collect_screw_arm_demo.py \
  --config configs/physic_solver/mujoco_hydroelastic_screw05_dual_release.toml
```

`rgb01` 带图像调试一次通过：两路各327帧，640×480 / 15 fps，合成1280×532 / 21.8 s。
完成7.9208圈、旋入19.570 mm，停转保持期最小穿出7.468 mm，松爪后最小穿出7.260 mm。
两指实测坐标均≥27.9996 mm，松爪后两侧接触均为0，螺纹接触100%。
已实际查看两路七阶段关键帧、21.6 s成品末帧及19.6/20.1/20.7/21.6 s松爪连续对照；
螺母在松开后随电批浮动机构受载下沉，但仍保持穿出并悬挂在螺栓上，未直接固定到电批或场景。
该版本已发布到 `outputs/debug/physic_solver/mujoco_hydroelastic_screw05_dual_release_debug/`。
回归：372项通过、3项跳过；新增完整松爪后穿出门限后的任务测试56项通过，ruff与全配置校验通过。
旧双臂及穿出版已发布结果重新独立验收通过。

关于松爪后的整体下沉：调试版从19.808 s至21.8 s，螺栓下降15.707 mm、螺母下降15.894 mm，
两者之间的穿出量只缩回0.186 mm。电批浮动轴从−24.558 mm下移至−40.078 mm，接近−40 mm下行限位；
位置目标保持在−24.841 mm，驱动刚度仍为200 N/m。
夹爪松开后，通过螺纹承托的螺栓、螺母及部分浮动机构载荷转移至该有限刚度轴，产生受载位移。
因此本版存在可见的整体下沉，螺母仍保持螺纹接触和穿出。若要求松爪后保持原高度，需另行验证更高刚度的轴向保持策略。

正式 `formal01` 一次通过并发布（2026-11-16，EST）：
[松爪版完整demo](../outputs/physic_solver/mujoco_hydroelastic_screw05_dual_release/demo.mp4) ·
[特写](../outputs/physic_solver/mujoco_hydroelastic_screw05_dual_release/threads.mp4) ·
[松爪连续对照](../outputs/physic_solver/mujoco_hydroelastic_screw05_dual_release/qa/release_review.jpg) ·
[整体下沉实测](../outputs/physic_solver/mujoco_hydroelastic_screw05_dual_release/qa/axial_settling.json) ·
[独立验收](../outputs/physic_solver/mujoco_hydroelastic_screw05_dual_release/validation.json)。

21.8 s、2616物理步，两路各654帧，1024×768 / 30 fps、PathTracing 64 spp、4反弹；
合成2048×820 / 30 fps，采集1498.58 s（约25分钟）。Job的21项和独立校验的25项检查全部通过。

| 指标 | 正式实测 | 门限 |
|---|---:|---:|
| 相对旋转／旋入 | 7.9256圈 / 19.5281 mm | ≥2圈 / ≥6 mm |
| 导程误差 | 1.4426% | ≤25% |
| 停转期最小穿出 | 7.4240 mm | ≥5 mm |
| 松爪后最小穿出 | 7.2455 mm | ≥5 mm |
| 松爪后两指最小坐标 | 27.9996 mm | ≥25 mm |
| 松爪后双指接触 | 0 | 0 |
| 松爪后螺纹接触比例 | 100% | ≥95% |
| 最小螺母抬升 | 158.9852 mm | ≥130 mm |
| 最大夹持滑移 | 0.7831 mm | ≤3 mm |
| 最大螺母转动（松爪前） | 0.0068352 rad | ≤0.05 rad |
| 双侧夹指／旋入期螺纹接触 | 100% / 99.762% | ≥95% / ≥95% |
| 停转期轴向漂移 | 0.1576 mm | ≤1 mm |
| 最大螺旋残差 | 0.3966 mm | ≤2 mm |
| 两臂最大跟踪误差 | 0.0177843 rad | ≤0.02 rad |
| 峰值接触数 | 228 | <40000 |

正式结果同样保留整体下沉：松爪至末帧，螺栓下降15.719 mm，螺母下降15.948 mm，
穿出量缩回0.227 mm；浮动轴由−24.569 mm到−40.102 mm，保持目标为−24.806 mm。
实际查看两路七阶段编码关键帧、21.6 s成品末帧、19.6/20.1/20.7/21.6 s编码松爪对照，
以及7.0/7.2/7.4/7.6 s原始PNG旋转对照；完整解码两路源视频和成品通过。
复核范围见 `qa/visual_review.json`，未逐帧人工观看。旧结果全部保留，失败和对照留在debug目录。

### 10.7 拧完后松开螺栓端，螺母继续夹持

`mujoco_hydroelastic_screw06_bolt_release{,_debug}.toml` 增加相反的释放流程：
旋拧20 s并保持后，用0.8 s主动张开电批的双瓣夹套，2 s抬起电批120 mm，最后观察1 s，总计29.6 s。
螺母夹爪全程维持闭合目标，旧的松螺母版本单独保留。

螺栓从episode开始就是独立六自由度刚体，夹套的两瓣通过真实棱柱关节运动，
由接触传递夹持力、扭矩和进给力。夹套碰撞体与可见几何一致，没有运行中删除约束、
固定螺栓、隐藏托块或重置位姿。配置 `robot.releasable_socket=true` 与
`planner.release_target="bolt"` 必须对应；`socket_open_s` 在此模式须为正。
机器人状态和动作由16维增加至18维，新增两瓣的实测坐标与目标；自由螺栓和螺母位姿另记物理日志。

沿用原旋入、真实导程、抬升、滑移和跟踪门限；螺母双侧接触和滑移检查覆盖至末帧。
新增要求：旋拧阶段夹套接触≥95%，结束后夹套与螺栓零接触、两瓣实际各打开≥20 mm、
工具全部碰撞角点高于螺栓头部≥40 mm，螺栓相对螺母沉降≤1 mm，
末段螺纹接触≥95%且尖端持续露出≥5 mm。独立脚本由原始位姿和关节坐标重算间隙。

仅靠向上拔出过盈夹套的对照会把螺栓带走，降低接触刚度的对照还出现非有限状态，均未发布。
主动夹套首轮也出现初始夹持不稳定，保留在
`outputs/debug/physic_solver/bolt_release_development/`，作为失败记录。

夹套增加随两瓣张开的托肩，内缘半径10.5 mm，支承15 mm半径的圆柱螺栓头，避开约10 mm半径的螺杆。
闭合侧壁内切半径14.5 mm，关节刚度30000 N/m、阻尼150 N·s/m、力限100 N；
夹套摩擦0.8、hydroelastic刚度1e10，均为本demo仿真参数，未进行实物材料标定。
初始两瓣各留1 mm行程间隙，在0.8 s夹持阶段平滑闭合，避免初始重叠；
托肩宽度由14 mm缩为8 mm，使两瓣闭合时保留约3.57 mm分缝，消除约1.62 mm内部重叠。
螺栓头和螺纹仍使用原网格SDF，没有用解析头部接触替代最终模型。
电批速度驱动阻尼由0.02改为0.1 N·m·s/rad，力矩上限仍为0.3 N·m。
新增两片小质量夹套及自由螺栓后，4子步对照仍出现初始冲击和跳牙；
`physics12` 单次通过但 `rgb01` 复跑失败，未发布。解析圆柱头部接触对照也未成为最终方案。
将本配置的每物理步子步数从4增至16，外层控制仍为120 Hz。
`physics19` 全部验收通过：8.7462圈、旋入21.5007 mm，停转穿出9.5103 mm；
释放后工具间隙88.4568 mm、螺栓相对螺母沉降≤0.0627 mm。
随后 `rgb04` 的释放和导程均通过，但旋拧阶段接触比例为94.9435%，未达到95%，因此未发布。
提高压紧力至400 N/m／16 N·s/m及降低螺纹刚度的后续对照也未稳定通过，未采用。
最终恢复轴向刚度／阻尼200 N/m／8 N·s/m、螺纹hydroelastic刚度1e11，
进给目标仍为−40 mm、力限仍为10 N。旋拧时间从14延至20 s，增加实际旋转和穿出量；
`deterministic_contacts=true` 启用Newton接触排序、确定性SDF累加和接触分配，减少GPU线程调度影响。
该字段默认false，不改变旧配置。螺栓位移仍由真实螺纹接触决定，原95%接触门限未改变。
128分辨率／30次迭代的 `rgb07` 预览通过全部检查并实际复核，保留在开发目录；
其旋拧接触比例为95.0021%，余量较小。最终将本配置的螺纹SDF分辨率设为256、
MuJoCo迭代上限设为100，提高接触精度和收敛精度，其他任务默认仍为128／30。
`physics25` 全部物理及独立检查通过：12.4434圈、旋入30.8342 mm，接触比例97.7926%，
导程误差0.8815%，最大螺旋残差0.3011 mm；停转穿出18.7555 mm，
松开后工具间隙91.2007 mm、螺栓相对螺母沉降≤0.0955 mm。
调试与正式采用相同物理参数和门限；未完成的对照已中止，记录保留在debug目录。

```bash
./app/python.sh scripts/physic_solver/collect_screw_arm_demo.py \
  --config configs/physic_solver/mujoco_hydroelastic_screw06_bolt_release_debug.toml
# 正式采集；实际复核后按第9节 --publish-run 发布
./app/python.sh scripts/physic_solver/collect_screw_arm_demo.py \
  --config configs/physic_solver/mujoco_hydroelastic_screw06_bolt_release.toml
```

高精度调试 `rgb08` 已发布到
`outputs/debug/physic_solver/mujoco_hydroelastic_screw06_bolt_release_debug/`：
29.6 s，两路各444帧、640×480 / 15 fps，MinimalRendering shading2，合成1280×532。
完成12.4523圈、旋入30.8412 mm，旋拧接触比例97.1678%；释放后尖端露出≥18.8401 mm，
工具间隙≥91.2273 mm、螺栓相对螺母沉降≤0.0654 mm。Job及独立检查全部通过。
实际查看两路八阶段编码关键帧、25.6/26.2/27.6/29.4 s释放对照、7.0/7.2/7.4/7.6 s旋转对照
和29.4 s成品末帧；两路源视频与合成视频完整解码通过，未逐帧人工观看。
单元测试377项通过、3项跳过，全配置校验、ruff及旧穿出版／松螺母版独立回归通过。

`formal01` 在2026-11-16 09:02:56 UTC发现同机 `human_walk` GPU进程持续运行后停止，
没有发布正式结果，也没有中断另一任务。依据
[任务开发skill](../.agents/skills/rsr-task-development/SKILL.md) 的
“PathTracing 正式渲染期间不要并行其它 Kit/GPU 任务”，正式版须待GPU可独占时重新采集。
已通过验收的高精度预览保留；所有对照、终止原因和实际参数见开发目录的 `run_note.json`。
