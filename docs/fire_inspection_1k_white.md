# X30 仓库巡检 1K 白模视频

参考 `x30_physx_warehouse_demo_2k.toml`，新增独立配置 `x30_physx_warehouse_demo_1k_white.toml`。

- 分辨率：1024×576（16:9），30 fps；前视 `front_rgb` 与跟随 `follow` 两路。
- 渲染：沿用参考版 PathTracing、256 spp、4 次反弹、降噪和同步采集。
- 风格：仓库与原 X30 机型的几何表面统一为白色哑光（diffuse 0.82、roughness 0.85、metallic 0）；保留原照明、Flow 火焰/烟雾、红色报警灯与阴影。
- 运动：沿用原物理、路线、相机位姿、步态与报警时序。机身仍为运动学移动、脚本化 trot。
- 照片验收：窗口从 320 缩至 128 px、最少火焰像素从 200 等面积缩至 32，火焰占比仍为 1%。
- 原代码、原配置、原资产和既有结果未修改。

新增资产（均位于本任务目录）：

- `assets/FireInspection/warehouse/warehouse_cardbox_fire_white.usda`
- `assets/FireInspection/robot/x30/x30_fire_inspection_white.usda`
- `assets/FireInspection/white_model_provenance.json`（来源及 SHA-256、覆盖数量和加载层列表）

USD 覆盖层相对引用同目录的原始 USD，依赖仍完整落在 `assets/FireInspection/` 内；不复制或改写几何和物理属性。新增覆盖层合计约 2.02 MiB。资产来源和许可沿用任务原有 `provenance.json`、机器人 `LICENSE_OEM.txt`。

运行与校验：

```bash
./scripts/rsr collect run -c configs/fire_inspection/x30_physx_warehouse_demo_1k_white.toml
./app/python.sh scripts/fire_inspection/validate_fire_inspection.py outputs/fire_inspection_demo_1k_white
```

输出位于 `outputs/fire_inspection_demo_1k_white/`，拒绝覆盖已有数据集。

本次采集与验证结果：

- 2029 帧 / 30 fps，视频时长 67.633 s；仿真时长 71.625 s（含 4 s 火焰预热），采集耗时约 25.7 min。
- Job 内 12 项验收全部通过；`validation.json` 的 34 项独立校验全部通过，双路视频完整解码、分辨率一致。
- 照片火焰窗口 851 像素 / 5.19%，门限 32 像素 / 1%；腿关节最大跟踪误差 0.1941 rad < 0.35 rad；机身单步偏差 0.53 mm < 1 mm。
- 跟随相机按全部 2029 个视频采样时刻检查净空、遮挡和全身取景，全部通过，最小画面边距 4.11%。
- 视觉抽查：两路视频每 2 s 抽一帧（各 34 帧），另查看早期/中段原帧和完整分辨率照片；已查看拼图保存在 `qa/`，不是逐帧人工复核。
- 可直接播放文件位于 `outputs/fire_inspection_demo_1k_white/demo/`：`follow_1k_white_30fps.mp4`、`front_rgb_1k_white_30fps.mp4`、`fire_photo_1k_white.png`。
- 全量无 Kit 测试 390 通过、3 跳过；Ruff 通过；全部 61 份配置检查通过；原后端保持可见。
- 未使用官方 LeRobot SDK 加载数据；独立校验读取了元数据、Parquet 和全部视频帧。
