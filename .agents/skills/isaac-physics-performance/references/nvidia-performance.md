# NVIDIA 性能建议索引

来源：[Simulation Performance Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/physics-performance.html)。检索日期：2026-11-04；页面标注更新：2026-07-23。以下为精简转述，编号对应原文章节顺序。具体操作请回看原文章节并核对安装版本。

1. **Solver:** Reduce iterations subject to fidelity; compare PGS and TGS at acceptable stability, not equal iteration counts.
2. **CPU/GPU:** Benchmark CPU for small scenes; large parallel workloads favor GPU with batched state access.
3. **Fabric:** Avoid costly simulation-state writeback to USD using physics Fabric.
4. **Async:** Consider overlapping physics/rendering when per-step Python is unnecessary; separate-thread Python is problematic.
5. **Threads:** Fewer physics threads can help; zero requests single-thread execution for small-scene experiments.
6. **Overlap:** Duplicated infinite ground planes across cloned environments inflate collision work and memory. Inspect using OmniPVD.
7. **Queries:** Disable scene-query support only when unused.
8. **Multi-GPU:** Distributing physics and rendering may help; consult the linked multi-GPU documentation.
9. **Colliders:** Simpler collision geometry costs less; SDF is costlier than a sphere.
10. **GPU compatibility:** High-aspect-ratio convex hulls may trigger CPU collision fallback. Look for `failed to cook GPU-compatible mesh`; consider bounding boxes or static triangle meshes for non-dynamic geometry. Such fallback also prevents particle/deformable collisions with that mesh.
11. **Deformables/particles:** These features cost more; consider rigid approximations with compliant contacts or joints if suitable.
12. **Callbacks:** Expensive per-physics-step Python controllers can dominate runtime.

本 skill 的基线测量、实验设计、质量验收及仓库适配是排查工作流，不是原文给出的固定参数或性能保证。
