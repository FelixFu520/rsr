# 碰撞指南建议索引

来源：[Collision Behavior Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/collision_guide.html)。检索：2026-11-04；页面更新：2026-07-23。以下为精简转述，编号用于回溯实验，具体接口与适用版本查原文。

1. **GPU contacts:** Buffer-capacity warnings can explain incorrect response; SDF generates many contacts. Check contact/patch capacity.
2. **Contact offset:** Increasing it anticipates contact; excessive values increase collision cost.
3. **Speculative CCD:** Velocity-dependent offset expansion can improve fast-object response.
4. **Mass:** Extremely light bodies or large mass ratios hinder solver convergence; keep changes physically reasonable.
5. **Depenetration:** Lower maximum depenetration velocity can reduce overshoot, slowing separation.
6. **Iterations:** More position iterations improve convergence.
7. **Friction:** Increasing friction aids settling, within physically plausible values.
8. **Timestep:** Reduce physics timestep if other adjustments fail.
9. **SDF mesh:** Prefer watertight, defect-free tessellation. Large triangles can undersample dynamic SDF contacts; remeshing may help.
10. **SDF resolution:** Low resolution loses thin features; high resolution costs memory/time. Thin shells below roughly twice grid spacing are problematic. Consider separate SDF parts instead of extreme resolution.
11. **SDF/static mesh:** Only static triangles are tested against the dynamic SDF; ensure that SDF is good.
12. **Initial overlap:** Intersecting initial colliders can cause strong separating responses. Inspect Physics Debug or version-checked `get_initial_collider_pairs`.

以上为官方建议摘要；测量流程与仓库职责划分见 SKILL.md，属于本项目排查约定。原文演示数值不是通用默认值。
