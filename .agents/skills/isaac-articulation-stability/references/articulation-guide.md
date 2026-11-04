# 官方稳定性建议索引

来源：[Articulation and Robot Simulation Stability Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html)。检索：2026-11-04；页面更新：2026-07-23。以下为压缩转述，具体接口与调参条件查原文。

1. **Timestep:** Complex/closed-loop systems may need smaller steps; once stable, test fewer position iterations.
2. **Mimic:** Compliance can relieve competing hard constraints; tune natural frequency/damping ratio using the linked guide.
3. **Velocity:** Joint-speed limits can stabilize early RL.
4. **Force:** Limit drives using actuator specifications and smallest relevant mass/inertia; avoid excessive accelerations.
5. **Force-drive estimates:** ω=√(k/m), ζ=c/(2√(km)); angular drives substitute inertia for mass. Consider attached subtrees. Avoid ωΔt≫1; TGS position iterations or smaller steps may help. Excessive damping slows response.
6. **Acceleration drives:** k=ω², c=2ζ√k; inertia compensation simplifies tuning, but contact-connected mass is excluded and manipulation can appear weak.
7. **Tracking:** Consider feedforward compensation instead of extreme gains; large target jumps need force limits or target-rate limiting.
8. **Mass ratios:** Avoid extreme ratios; inspect nonzero, reasonable link mass/inertia with OmniPVD.
9. **Armature:** Adds joint-space inertia/damping; geared rotor reflected inertia is JG².
10. **Self-collision:** Parent-child collisions are filtered; unintended nonadjacent collisions can destabilize. Temporarily disable for diagnosis, then inspect and filter specific pairs, preserving needed collisions.
11. **Solver order:** Contact-last ordering may improve persistent penetration/gripping at measurable performance cost; verify the scene flag API locally.

本 skill 的测量、验收、单位换算和仓库适配属于排查工作流；这些近似公式不提供整机稳定性证明，原文示例数值不作为统一默认值。
