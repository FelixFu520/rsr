---
name: isaac-articulation-stability
description: 排查 Isaac Sim / Omniverse PhysX 机器人关节振荡、发散、速度或力矩尖峰、轨迹跟踪差、mimic/闭环与抓取不稳；依据 NVIDIA Articulation Stability Guide 选择并验证驱动、惯量和求解器实验。主要碰撞几何问题转 isaac-collision-debug，非 PhysX 后端需独立核对。
---

# Isaac 关节与机器人稳定性

用中文说明证据、假设和结果。先读 [官方建议与公式索引](references/articulation-guide.md)，按症状选实验，不整套复制参数。具体数值、API 和适用条件以安装版本的官方文档及本机接口为准。

## 定位不稳定发生在哪里

读取当前 AGENTS.md、启动命令、生效配置与日志；记录 Isaac Sim / Kit / PhysX 版本、实际物理引擎、求解器、物理/控制/渲染周期和机器人资产版本。不要将 PhysX 指南直接用于 Newton 等其他后端。

保存可复现的初态、目标/动作序列和异常出现的步骤，确定问题来自单关节自由运动、机器人内部约束还是环境接触。采集相关关节的目标与实测位置/速度、控制输出、饱和状态及异常时间；区分命令跳变、控制更新问题、碰撞异常和数值不稳定。

检查实际 link/joint 拓扑、关节类型/轴/限位、基座约束、mimic 与闭环、drive 类型和生效增益。检查质量/惯量、质心、坐标变换和单位是否与资产及执行器规格一致。列出值的来源与覆盖关系，避免只修改未生效的配置。

## 选择可解释的实验

把观测匹配到参考索引的条目，先处理资产定义和控制输入错误，再讨论数值调参。使用可回退的局部编辑；每轮改变一个因素或一个不可拆分组合，保留复现问题所需的负载、接触与动作。

使用公式估算时记录 drive 类型、采用的质量/惯量及单位和简化假设，避免把单自由度估算当作整机稳定性证明。自然角频率用 rad/s；若输入为 Hz，先用 ω = 2πf 转换。公式中的角量与 USD API 接口单位分别核对，不能把 deg 与 rad 混算。

力驱动的质量/惯量估算式与 acceleration drive 的增益关系分开使用，不能只复制数值切换类型。物理步长、控制周期与 TGS position iterations 分别记录；迭代数不等于真实物理子步数。

调整 link/articulation 迭代参数时同时检查 PhysicsScene 的全局上下限，确认最终生效值没有被场景限制裁剪。

涉及场景或关节属性时先在当前版本查 schema 所属对象、类型、写入 API 和生效时机。原文示例不是可直接运行的补丁；特别是 solver-order 片段，不能假定通用 USD Prim 具有文中方法。定位已有活动 PhysicsScene，确认受支持的 schema API 后再局部设置，不新建重复场景。

若实验改变执行器能力、物理模型、接触过滤或时间离散，说明它对实际任务和已训练策略的影响。诊断时关闭的业务能力不能悄悄留作最终修复。

## 验证与交付

固定初态、种子、动作序列、仿真时长及负载，对比关节误差、速度/力矩峰值、饱和、接触/抓取成功率及步耗时。依问题验证空载与原始任务负载，确认变化不是简单压制动作后掩盖故障。沿用现有验收阈值；无标准时报告观察范围，不宣称稳定性保证。无收益或质量退化则回退。

Real2Sim2Real 中按当前 AGENTS.md 使用 `./app/python.sh`；运行时求解参数归 simworld，机器人参数归 robot 后端，任务控制归相应实现。不向通用 ABC 塞机型字段，不改变 Job 生命周期。新配置同步 Config、默认配置和适用示例。

输出复现与异常关节、根因证据、实验及参数单位、前后质量/耗时、剩余风险。区分官方建议、近似推算和项目实测；没有真实 Isaac 环境时明确尚未验证。

按需读取 [碰撞 skill](../isaac-collision-debug/SKILL.md)、[性能 skill](../isaac-physics-performance/SKILL.md) 或 [FAQ skill](../isaac-physics-faq/SKILL.md)。夹爪驱动与 Robotiq 示例参考 [夹爪调参 skill](../isaac-gripper-tuning/SKILL.md)；具体机型参数仍需核对硬件资料。
