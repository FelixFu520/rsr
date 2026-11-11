---
name: isaac-physics-faq
description: 核对 Isaac Sim / Omniverse PhysX 已知限制与常见异常，适用于接触报告缺失、传送带兼容、恢复不确定性、关节/TGS 行为差异、力场无效或异步回调线程问题。依据官方 Known Physics Limitations 匹配版本和功能组合；不作为所有安装故障或非 PhysX 后端的通用 FAQ。
---

# Isaac 物理 FAQ 与已知限制

官方 [Guides, Hints and FAQs 目录](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides.html) 当前列出多篇指南及 Known Physics Limitations，没有独立问答页。本 skill 将已知限制转为排查入口，不能声称覆盖整个 Isaac Sim FAQ。

## 用法

1. 读取项目 AGENTS.md、复现过程、日志与当前生效配置，确认 Isaac Sim / Kit / PhysX 版本、实际物理引擎、对象类型和启用的功能组合。
2. 读取 [已知限制索引](references/known-limitations.md)，找与症状相符的条目。按条目关键词到官方原文核对完整条件与建议；没有匹配时如实说明，不能据此判定软件 bug。
3. `latest` 的条目只作为调查线索。变更前查目标安装版本的官方文档/发布说明及本机接口。旧版本问题可能已修复，新版本限制也可能不适用于旧版；不能因名称类似就推断适用。
4. 用最小复现区分“功能不支持/组合受限”“配置或资产错误”“尚未解释的异常”。记录支持和反驳证据；日志沉默不能单独证明支持该功能。
5. 若命中限制，解释精确作用范围和与业务的关系。选择原文适用的绕行方案并验证；不要编造不支持功能的开关。方案改变模型、传感语义、精度或执行线程时，说明对当前任务的实际影响。

## 落地与验收

先在可回退的局部实验中验证，保留问题原始复现。按目标症状验证正确性，并检查步耗时和内存是否退化。对确定性问题记录起点、随机性、执行设备和版本，不把一次相同结果当作确定性保证。没有真实运行条件时输出可执行的验证步骤，明确尚未证实。

涉及 API 或示例时先定位实际 prim、PhysicsScene 与设置作用域，检查生效时机；不覆盖用户整份 stage。Real2Sim2Real 中使用 `./app/python.sh`，遵守当前 AGENTS.md 的后端职责、Config 同步与 Job 生命周期约定；本 skill 不授权扩大通用接口。

输出：匹配条目和官方链接、版本证据、已确认/待确认判断、最小绕行方案、验证结果与限制。不能把文档中的已知问题直接写成当前项目的确诊结论。

接触质量问题转到 [碰撞 skill](../isaac-collision-debug/SKILL.md)，吞吐问题转到 [性能 skill](../isaac-physics-performance/SKILL.md)，关节与机器人不稳定转到 [稳定性 skill](../isaac-articulation-stability/SKILL.md)。夹爪驱动及 Robotiq 示例转到 [夹爪调参 skill](../isaac-gripper-tuning/SKILL.md)，按需读取。
