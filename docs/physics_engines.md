# 物理引擎、求解器和接触模型

本页整理用户提供的三张图，并按本机 Isaac Sim 6.2 / Newton 1.6.1 的实现核对。
图中的“成功”是已有实验结论，不能推导出新包裹、重量、姿态或拧螺丝也会通过。
具体配置、原始结果和新增实验见 [physic_solver 任务](physic_solver.md)。

## 先区分五种概念

- **引擎**：PhysX、Newton，负责组织模型、碰撞和积分。
- **动力学求解器**：Newton 的 MuJoCo、XPBD、VBD 等，决定怎样更新相应模型的状态。
- **约束求解策略**：PhysX PGS/TGS，决定怎样迭代处理接触与关节约束。
- **形变方法／材料模型**：FEM、布料表面、PBD 粒子等，描述材料怎样变形。
- **碰撞几何／接触模型**：SDF 用距离场表达几何；hydroelastic 用分布压力计算接触力。

这些概念涉及不同层级，不能当成一列互斥开关。FEM 与 hydroelastic 也不是同一种选择。

例如可以组合“TGS 场景 + FEM 软体”。把普通刚体场景改为 TGS，不会让刚体自动变成软体。
Hydroelastic 计算分布接触压力及其合力／力矩，可与支持该接触路径的动力学求解器组合。
结合用户给出的教程，图末“Hydrostatic”应整理为这里的“Hydroelastic”；它不属于第九种动力学求解器。

## 现有组合实验

以下为仓库已经保存的结果；不是此次新增矩阵的成绩。

| 实验 | 机器人由谁求解 | 被抓物由谁求解 | 已有结果 |
|---|---|---|---|
| 刚体 PhysX | PhysX TGS | PhysX TGS，刚性方块 | 成功 |
| 纯 Newton 基准 | Featherstone | VBD，衣服 | 成功 |
| 混合模式 | PhysX TGS | VBD，衣服 | 成功 |
| MuJoCo + VBD | Newton MuJoCo | VBD，衣服 | 成功 |
| XPBD + VBD | Newton XPBD | VBD，衣服 | 成功 |
| 统一 XPBD | 同一个 XPBD 模型求解机器人和衣服 | XPBD，衣服 | 未持续抓住，诊断结果保留 |

包裹另外已有 MuJoCo + VBD 的原装 Franka 短指抓放成功记录。上述 VBD 外部刚体路径为单向
“刚体影响柔体”，不能据此评价真实夹爪受载、最大负载或袋体反作用力。

新增包裹运行时另有默认关闭的 `planner.rigid_soft_coupling="proxy"`：原生CPU MuJoCo
与VBD通过Newton实验性 `SolverCoupledProxy` 回传接触反力，机械臂关节仍由MuJoCo求解。
它带有独立的代理几何一致性和实际驱动力限检查。本轮优化Y175mm边缘／90°包裹已完成连续抓放、
数据及实际视觉验收；其余17项柔性工况仍是单向耦合。该单次实验不构成硬件负载认证，具体参数及失败历史见任务文档。

## PhysX 路径怎样理解

| 路径 | 通俗解释 | 适用对象 | 主要边界 |
|---|---|---|---|
| PGS | 逐个修正接触、关节等约束并反复协调 | 刚体、关节、接触 | 单轮通常便宜；复杂约束可能需要更多迭代 |
| TGS | 在更细的时间推进中持续修正约束 | 机械臂、刚体抓取、刚柔场景 | 可作为起点；并非所有场景都优于 PGS |
| Volume deformable / FEM | 用体积单元计算内部形变 | 橡胶、软块、组织 | 需要体积网格、材料参数和相应 GPU 支持 |
| Surface deformable / FEM | 在三角表面计算拉伸、弯曲 | 布、薄片、壳 | 当前工作流含 XPBD FEM；直接接触和迁移能力须按版本验证 |
| PBD 粒子系统 | 通过粒子约束表达材料行为 | 流体、颗粒等 | 不直接等同于有弹性的实体软体 |

配置/API 以本机 PhysX 扩展为准。原理参考 NVIDIA [Rigid Body Simulation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html)
和 [Deformable Bodies](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/deformable_bodies.html)。
这些 `latest` 链接用于解释概念，不构成本机版本兼容保证。

## Newton 1.6.1 八种求解器

| 求解器 | 核心思路 | 主要用途 | 版本边界 |
|---|---|---|---|
| Featherstone | 利用关节树结构求动力学，半隐式推进 | 机械臂、关节机构，也可配粒子／软体 | 基础可微；布料无自碰撞，部分关节功能不支持 |
| MuJoCo | 关节动力学结合接触、摩擦约束求解 | 机械臂、灵巧手、刚体抓取 | Newton 封装不求解布料和软体 |
| SemiImplicit | 由力更新速度，再更新位置 | 刚体、粒子、基础软体 | 高刚度通常需要小步长；布料无自碰撞 |
| XPBD | 预测位置，再按柔顺约束修正 | 刚体、关节、粒子、柔体原型 | 软体支持实验性；该版本布料无自碰撞 |
| VBD | 逐顶点优化位置，协调惯性、弹性和接触 | 布料、软体、刚体、杆及混合系统 | 支持柔体自接触；实验性，关节支持有限 |
| Style3D | 基于 Projective Dynamics 的布料求解 | 布料 | 专用布料路径，不负责机器人关节 |
| ImplicitMPM | 粒子与背景网格结合计算连续材料 | 颗粒、弹塑性材料 | 不负责机器人关节，也不是通用布料求解器 |
| Kamino | 最大坐标下求受约束机构动力学 | 闭环机构、硬摩擦接触、碰撞 | 实验性，不负责柔体；API 可能变化 |

能力边界依据 [Newton 1.6.1 能力表](https://newton-physics.github.io/newton/1.6.1/solvers/index.html)
及 [对应 API](https://newton-physics.github.io/newton/1.6.1/api/newton_solvers.html)，并核对
`app/exts/isaacsim.pip.newton/pip_prebundle/newton/_src/solvers/`。
本表描述库能力；Isaac Sim 的 USD 解析、控制、传感器和数据接口是否覆盖，还需单独核实。
Isaac 的标准配置主要提供 MuJoCo/XPBD 接口及 VBD 集成，不能理解成八种求解器都能任意切换。
参见 [Isaac Sim 6.2 Newton 集成](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/physics/newton_physics.html)。

## 从任务选择路径

| 任务 | 优先评估 | 原因 |
|---|---|---|
| 普通机械臂抓刚性零件 | PhysX TGS / Newton MuJoCo | 重点是关节、接触、摩擦和控制 |
| 机械臂动力学、基础可微优化 | Featherstone | 利用关节结构，并有基础微分支持 |
| 简单刚柔约束原型 | XPBD | 建模直观；需要验证收敛、接触与功能限制 |
| 海绵、橡胶等实体形变 | PhysX volume deformable / Newton VBD | 需要内部体积形变 |
| 布料折叠、褶皱、自接触 | VBD / Style3D，或 PhysX 表面模型作基线 | 自碰撞和弯曲模型是关键 |
| 绳索、电缆弯扭 | VBD 杆模型；粗略场景可用刚体链 | 连续弯扭与离散链近似不同 |
| 颗粒、弹塑性材料 | ImplicitMPM | 材料行为决定方法 |
| 复杂闭环机构 | Kamino 或其他已验证机构方案 | 需要专门处理闭环约束 |
| 螺纹装配 | Newton MuJoCo + hydroelastic / SDF | 螺纹真实几何和接触力矩决定导程，不使用螺旋轨迹代替接触 |

以上是候选选择，不是性能排名。Featherstone 不必然比 MuJoCo 快；接触数量、自由度、步长和实现均影响耗时。
MuJoCo 仍可能发生穿透或失稳，其摩擦锥、接触容量和几何分辨率仍需验证。

## Hydroelastic 螺纹装配的特别要求

本机含官方 `standalone_examples/api/isaacsim.physics.newton/nut_bolt_hydroelastic.py` 及 Newton
`examples/contacts/example_nut_bolt_hydro.py`。前者示范 M16，后者使用 M20 loose 网格；本任务采用后者的 M20 几何，
不混用两种螺距。需核对两侧 SDF、接触刚度、零间隙、elliptic 摩擦锥、足够接触容量和子步率。
官方演示的重力旋入，只证明接触对准备方法，不等于机械臂拧紧任务已经通过。
参考 [Isaac Sim 6.2 螺母螺栓教程](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/physics/hydroelastic_contact_walkthrough.html)。
