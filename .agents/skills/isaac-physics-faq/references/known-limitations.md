# 已知限制检索索引

来源：[Known Physics Limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/current_limitations.html)。检索：2026-11-04；页面更新：2026-07-23。以下为压缩线索，完整条件与绕行方案需查原文并核对安装版本。相近条目合并，非完整限制表副本。

| Symptom / feature | Documented lead |
|---|---|
| Particle/deformable reports | Contact reports unsupported. |
| Resume determinism | USD snapshots omit persistent internal contact state; restart from beginning. |
| Conveyor belts | Particle/deformable and SDF interactions limited; CPU path affects GPU throughput. |
| Isosurface | Possible memory leak; rendering feature. |
| Particle/deformable friction | No static friction or friction-combine mode. |
| Particles rolling away | Solver ghost forces possible. |
| Tendons | Fidelity issues; nonzero TGS velocity iterations distort reported forces. |
| GPU convex hulls | 64 vertices/faces limit; decomposition or SDF costs more. |
| Spherical articulation joints | Nonidentity mass frames can impair limits/drives. |
| TGS velocity iterations | Historical conversion above four changed; check version. |
| D6 drives | TGS/velocity-iteration issues; compare documented alternatives. |
| D6 loop closure | Instability can involve competing stiff drives. |
| Joint friction | PGS/TGS behavior differs; damping-drive alternative documented. |
| Force fields | Do not affect particles, deformables, articulation bodies. |
| Async callbacks | Run off main Python thread; require thread-safe calls. |
| Joint properties | Availability differs inside/outside articulations. |
| TGS articulation drives | Nonzero steady-state reported velocity; per-iteration external forces suggested, with cost. |

将这些条目写成“该版本/组合可能受限”，而非永久全局规则。具体替代 API、solver 参数与性能影响请回到原表核对。
