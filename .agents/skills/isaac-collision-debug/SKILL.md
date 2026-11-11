---
name: isaac-collision-debug
description: 排查 Isaac Sim / Omniverse PhysX 物体穿透、弹飞、抖动、接触不稳、薄壁或 SDF 碰撞失真、GPU 接触缓冲告警；按 NVIDIA Collision Behavior Guide 定位与验证。主要耗时问题使用 isaac-physics-performance，功能兼容与版本限制使用 isaac-physics-faq。
---

# Isaac 碰撞质量排查

用中文报告。先读 [碰撞指南摘要](references/collision-guide.md)，需要具体操作时查其官方链接和当前安装版本。摘要提供候选原因，不构成固定调参顺序或通用参数。

同一症状可能跨域：本 skill 主要检查碰撞体、接触和响应；夹爪驱动力/速度估算从 [夹爪调参 skill](../isaac-gripper-tuning/SKILL.md) 开始。跨 skill 引用只按证据补读，不递归加载所有关联内容。

## 先保存故障证据

读取当前项目 AGENTS.md、启动命令、生效配置与异常日志。记录实际物理后端和 Isaac Sim / Kit / PhysX 版本；PhysX 结论不能直接套到 Newton 等后端。

确定异常发生于初始帧、持续接触、运动中还是 reset 后；保存涉及的 prim 路径、动作序列和可重复的短运行。区分“没有检测到碰撞”“检测后响应异常”“视觉与物理状态不同步”，不要把三者统称为穿模。

检查实际碰撞体与可视网格、世界变换/缩放、单位、碰撞启用状态、过滤关系、刚体类型和所属 PhysicsScene。记录有效值及配置来源；仅有材质或可视模型截图不足以证明物理设置正确。

## 选择与验证改动

把证据匹配到摘要中的相关条目，优先做能区分根因的小实验。减少无关资产或控制逻辑时保留复现所需接触与运动。对同一初始状态逐项比较，不一次套用官方示例所有参数。

参数实验前记录旧值、单位、对象作用域和生效时机。数值需与对象尺寸、速度和任务误差要求匹配。涉及质量、摩擦、几何或时间离散的变化要说明对任务物理含义的影响；先验证资产定义是否正确，再考虑近似方案。

查当前运行时 API 后再修改 USD；定位已存在的活动 PhysicsScene，避免因复制示例而另建场景。官方示例包含清空根层的演示代码，不能直接用于用户 stage；用局部、可回退的编辑复现所需设置。

在同一动作/种子/仿真时长下比较异常频率、穿透或接触误差、速度突变、任务成功率及步耗时。改进画面不等于物理质量已验证；失败则回退并检查下一假设。没有真实 Isaac 环境时标明未实测。

## 项目落地与交付

在 Real2Sim2Real 中遵守当前 AGENTS.md：用 Isaac `./app/python.sh` 运行；运行时设置归 simworld 后端，资产设置归其所属模块；不向通用 ABC 加 Isaac 专用字段，不改变 Job 生命周期与 finally 清理。新配置同步 Config、默认值和适用示例。

若症状属于功能组合限制，读取 [FAQ skill](../isaac-physics-faq/SKILL.md)；质量达标后需评估吞吐，再读取 [性能 skill](../isaac-physics-performance/SKILL.md)。只读取当前问题相关内容。

若证据指向关节驱动、mimic 或闭环约束，读取 [机器人稳定性 skill](../isaac-articulation-stability/SKILL.md)。

交付异常对象与复现方法、根因证据、最小补丁或实验、前后质量/性能结果及未验证项。区分官方建议、项目实测和推测。
