# 教程要点与估算式

来源：[Joint Parameter Tuning Example: Robotiq 2F-85](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/gripper_tuning_example.html)。检索：2026-11-04；页面更新：2026-07-23。以下是精简转述，不是完整教程镜像。计算为粗估，示例不是硬件规格表。

1. **Preparation Work:** Test isolated gripping without gravity, then gravity/load; cover multiple openings, thin walls, difficult objects.
2. **Robotiq 2F-85:** Example: six revolute joints, one drive, five mimics, ±1 gearing, zero offset, no closing loops; driven range 0–47°.
3. **Maximum Drive Force:** Symmetric static friction-only support: Ntotal≥mg/μ; τ≈Nfinger·r. Example assumptions yield 4 Nm/joint and 24 Nm at J0; not a universal transmission model.
4. **Maximum Joint Velocity:** Example closing time≈0.28s, speed≈168°/s. Avoid restrictive mimic velocity limits. USD/PhysX angular velocity units differ.
5. **Drive Stiffness/Damping:** Parallel-axis Ijoint=Icom+mδ² (δ perpendicular); ω=√(k/I), ζ=c/(2√(kI)); k≈τmax/Δq; c=2ζ√(kI). Force-drive example: I≈3.4e−6 kg·m², k≈275 N·m/rad, c≈0.06 N·m·s/rad. ζ=1 starts estimation.
6. **Units:** Angular kUSD=kPhysX·π/180; likewise damping. Verify API conversion. ω is rad/s, not Hz.
7. **Implicit Spring:** Ignoring damping, rough force≈kΔq/[1+(Δt·ω)²]; extreme gains need not improve behavior. TGS iterations are not equivalent full timesteps.
8. **Armature:** Jmotor·G²; changes physical inertia, requiring hardware justification.
9. **Mimic Joint Compliance:** Uses natural frequency/damping ratio, not drive stiffness/damping; hard-constraint conflicts may justify compliance.
10. **Verification and Fine-Tuning:** Check speed, load retention, collider penetration, jitter; recompute damping after stiffness changes.

原文章节链接：

- [准备实验](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/gripper_tuning_example.html#preparation-work)
- [驱动力与速度](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/gripper_tuning_example.html#maximum-drive-force-maximum-joint-velocity)
- [增益推导](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/gripper_tuning_example.html#drive-stiffness-damping)
- [隐式弹簧](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/gripper_tuning_example.html#implicit-spring)
- [Armature](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/gripper_tuning_example.html#armature)
- [Mimic 柔顺性](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/gripper_tuning_example.html#mimic-joint-compliance)
- [验收](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/gripper_tuning_example.html#verification-and-fine-tuning)
