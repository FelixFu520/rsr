# X30 仓库巡检：1K 明亮场景

当前推荐：`configs/fire_inspection/x30_physx_warehouse_demo_1k_bright_stable.toml`，
正式结果：`outputs/fire_inspection_demo_1k_bright_stable/`。原始PathTracing提高至4096 spp，保持1024×576、30 fps；
不添加视频时域滤波。旧版256 spp结果已移至`outputs/debug/fire_inspection/demo_1k_bright_256spp_flicker_retired/`保留作对照。

参考 `configs/fire_inspection/x30_physx_warehouse_demo_2k.toml`，复制仓库主 USD 为
`assets/FireInspection/warehouse/warehouse_cardbox_fire_brighter.usda`，保留原材质、几何、
碰撞、Flow 火焰与烟雾。依赖复用本任务资产目录中的本地文件，原 USD 不修改。

## 旧版256 spp采集规格与灯光

- 旧版256 spp对照配置：`configs/fire_inspection/x30_physx_warehouse_demo_1k_bright.toml`。
- 两路相机 `front_rgb` / `follow` 与照片均为 1024×576（16:9），视频 30 fps。
- 沿用参考版 PathTracing、256 spp、12 个渲染子帧、4 次反弹、降噪与同步采集。
- 75 个普通顶灯调至参考 demo 实际强度的 6 倍；其余 32 个灯光保持参考 demo 的实际强度，
  包含绿色出口灯、红色应急灯、火光与拍照补光。灯光颜色、曝光和发光材质不变。
- 参考配置的 `scene.light_scale=1.5` 已固化到新 USD，故新配置为 `light_scale=1.0`。
  “保持不变”以参考配置运行时强度为准，而非未乘 1.5 的原始资产数值。
- 机器人报警灯保持 `light_intensity=6000`、`emissive_strength=20`；路线、步态、物理参数、
  相机位姿、拍照后停留 4 s 和报警 3 s 的时序不变。
- 照片窗口按分辨率等比例调整为 128 px，火焰像素下限按面积调整为 32；占比仍为 1%。
  调试照片也保留 1K 及同一门限。

顶灯选择限定在原有普通灯组及根目录高位灯：世界高度至少 3 m，RGB 最小分量与最大分量之比
至少 0.4。其余灯一律不额外增强。逐灯变更、源/副本 SHA-256 与加载层列表记录于
`assets/FireInspection/brighter_warehouse_provenance.json`。来源及许可沿用原
`assets/FireInspection/provenance.json` 和机器人 `LICENSE_OEM.txt`。

## 旧版运行（仅供复现与对照）

```bash
# 已有 FireInspection 原资产时，只需准备一次独立场景副本；已有副本时拒绝覆盖。
./app/python.sh scripts/fire_inspection/prepare_bright_warehouse.py --multiplier 6 \
  --destination assets/FireInspection/warehouse/warehouse_cardbox_fire_brighter.usda \
  --report assets/FireInspection/brighter_warehouse_provenance.json

# 640×360、15 fps、MinimalRendering mode=2 的低成本 RTX 纹理预验证。
./scripts/rsr collect run -c configs/fire_inspection/x30_physx_warehouse_demo_1k_bright_debug.toml
./app/python.sh scripts/fire_inspection/validate_fire_inspection.py outputs/debug/fire_inspection/demo_1k_brighter_validation

# 旧版1K/256 spp复现，仅供对照；写入debug目录，不覆盖已有结果。
./scripts/rsr collect run -c configs/fire_inspection/x30_physx_warehouse_demo_1k_bright.toml
./app/python.sh scripts/fire_inspection/validate_fire_inspection.py outputs/debug/fire_inspection/demo_1k_bright_256spp_reproduction
```

旧版归档目录为 `outputs/debug/fire_inspection/demo_1k_bright_256spp_flicker_retired/`，数据格式为LeRobot v3。
机身仍为运动学移动，腿部为脚本化 trot；本次只调整场景照明与采集规格。

## 旧版采集结果（连续帧质量未通过用户复核）

首版普通顶灯为参考 demo 的 3 倍。其 640×360 预验证 1015 帧、Job 内 12 项与独立校验
34 项全部通过；随后按用户“在这个基础上再亮些”的要求改为参考 demo 的 6 倍，即首版的 2 倍。
首版的正式采集中途停止，未发布为数据集；调试目录保留其未完成轨迹和画面。
旧的 3 倍场景副本、预验证及已有 2K/白模结果均保留。

6 倍版本预验证已通过：1015 帧，Job 内 12 项和独立校验 34 项全部通过，耗时 337.8 s；
照片火焰窗口 960 像素 / 5.86%，门限为 32 像素 / 1%。

副本属性审计通过：37,138 个 prim 一致，357,535 个非灯光强度属性、关系目标及时间采样保持不变；
75 个普通灯增强，32 个其他灯维持参考 demo 强度，原资产 SHA-256 未变。
审计文件位于新预验证目录的 `qa/asset_audit.json`。

正式采集完成于 2026-11-17 UTC（本地 2026-11-16）：

- 两路 1024×576 / 30 fps 视频，各 2029 帧，时长 67.633 s；仿真时间 71.625 s（含 4 s 火焰预热）。
- 采集耗时 1613.5 s（约 26.9 min）。Job 内 12 项验收与独立校验 34 项全部通过。
- 照片火焰窗口 935 像素 / 5.71%，门限 32 像素 / 1%；拍照时报警灯关闭。
- 报警连续 7 s；视频像素独立复核 14 次闪烁，与 2 Hz 一致。
- 机身单步偏差最大 0.53 mm < 1 mm；腿关节最大跟踪误差 0.1941 rad < 0.35 rad。
- 跟随相机在全部 2029 个视频采样时刻的净空、遮挡和全身取景检查全部通过，最小画面边距 4.11%。
- 视觉复核：两路视频每 2 s 抽一帧，各 34 帧；另查看拍照/结束等关键帧、完整分辨率照片和三版首帧对照。
  范围记录于 `qa/visual_review.json`，并非逐帧人工复核。独立校验完整解码了全部视频帧。
- 同机位首帧的平均显示亮度：参考版 0.1138、首版 0.2227、最终版 0.3408。
  这是显示图像的比较指标，不是照度测量；参考图来自缩小后的 2K 压缩视频。
- Ruff、全部 63 份任务配置检查通过；全量无 Kit 测试 393 通过、3 跳过；原有后端保持可见。
- 未使用官方 LeRobot SDK 加载数据；已独立读取元数据、Parquet 和全部视频帧。

可直接播放的副本：

- `outputs/debug/fire_inspection/demo_1k_bright_256spp_flicker_retired/demo/follow_1k_bright_30fps.mp4`
- `outputs/debug/fire_inspection/demo_1k_bright_256spp_flicker_retired/demo/front_rgb_1k_bright_30fps.mp4`
- `outputs/debug/fire_inspection/demo_1k_bright_256spp_flicker_retired/demo/fire_photo_1k_bright.png`

原始 LeRobot v3 数据位于 `outputs/debug/fire_inspection/demo_1k_bright_256spp_flicker_retired/lerobot_v3/`。
验收、相机审计和亮度对照分别见 `validation.json`、`camera_validation_30fps.json` 和
`qa/brightness_comparison.jpg`。原始资产及既有 2K、白模数据均未覆盖。


## 连续帧质量复核与高采样修正版

用户复核指出：跟随视频约 58 s 的火焰看起来静止，光线闪烁贯穿全程。此前 12 项任务验收与
34 项独立数据校验不覆盖连续帧光照稳定性；每 2 s 抽帧也不足以发现这类问题。因此，
这份旧数据不能据此前验收结论认定为“无闪烁”。调查记录为 `qa/temporal_quality_review.json`。

已核实：正式配置是 PathTracing；Flow 原生流场时钟在短探针中持续前进，但原资产的
11 条发射器动画曲线只到 15 s，此后保持末值。新的场景副本把这些曲线延伸覆盖整段任务，
保留原始资产和旧输出。已从修正版57–58.9 s的原始帧与59.7–60.4 s的固定相机帧中确认火舌轮廓持续变化。

当前环境在开启 OptiX 时域降噪时记录驱动限制而强制关闭它；独立渲染输出的实验中
Flow 火焰缺失，已排除该实验方案。不会把这些探针发布为正式数据。


### 修正版设置与证据

用户选择“保留原始光追输出，提高采样，接受更长采集时间”。修正版配置为
`configs/fire_inspection/x30_physx_warehouse_demo_1k_bright_stable.toml`，输出目录为
`outputs/fire_inspection_demo_1k_bright_stable/`。采集、独立校验、相机审计和连续帧质量复核均已完成，详见下方实测结果。

- PathTracing 从 256 提升为 4096 spp，129 个渲染子帧，保留 4 次反弹和原有空间降噪。
- 单 GPU 0，同步采集，启用 many-light sampling，关闭 radiance cache 与时域降噪。
  驱动 570.172.18 会因已知问题强制关闭 OptiX temporal denoising；不绕过驱动保护。
- 不对输出视频添加时域滤波、补帧或平滑；视频仍是采集器直接编码的渲染帧。
- 新场景为 `warehouse_cardbox_fire_brighter_animated.usda`。普通灯仍是参考版的 6 倍，
  其余灯及机器人报警灯保持原强度。仅在副本中重复 11 条原有 Flow 发射器动画曲线至
  至少 260 s；不重置流场、不拼接火焰画面、不改变机器人路线或物理参数。

```bash
./app/python.sh scripts/fire_inspection/prepare_bright_warehouse.py --multiplier 6 \
  --destination assets/FireInspection/warehouse/warehouse_cardbox_fire_brighter_animated.usda \
  --report assets/FireInspection/brighter_animated_provenance.json \
  --extend-fire-animation-s 260
./scripts/rsr collect run -c configs/fire_inspection/x30_physx_warehouse_demo_1k_bright_stable.toml
./app/python.sh scripts/fire_inspection/validate_fire_inspection.py outputs/fire_inspection_demo_1k_bright_stable
```

固定物理时钟及机位的原始 PNG 对照位于
`outputs/debug/fire_inspection/pt_raw_high_spp_probe/report.json`。每档两机位各 12 帧，剔除
前 2 帧，再对 10 帧计算亮度时序统计（0–255）。在相同单 GPU、many-light sampling
条件下，256 → 4096 spp：

| 指标 | 火点前视 A | 环境机位 B |
|---|---:|---:|
| 全画面平均亮度标准差 | 0.2141 → 0.0734 | 0.3001 → 0.0349 |
| 相邻帧亮度平均绝对差 | 1.3819 → 0.8859 | 1.9481 → 1.1272 |
| 每像素时序标准差 p99 | 8.3757 → 3.4395 | 9.9682 → 4.1849 |

这些是短探针的降噪改善指标，不等同于“全程无闪烁”验收。8192 spp 的额外探针遇到现有
同步等待次数上限，未采出帧；没有把该失败探针用于正式输出。选用已经通过原始帧短测的
4096 spp，不修改共享采集实现。

真实 Job 的 1 s / 30 帧双机位运动短测位于
`outputs/debug/fire_inspection/bright_stable_1s_probe_4096/`。它有意提前结束任务，
`success=false` 是尚未走完巡检路线，不是正式数据。约 58 s 机位对应的火焰短测位于
`outputs/debug/fire_inspection/animated_fire_58s_probe/animated_fire_follow58_1k_30fps.mp4`，
固定机位、仿真 62–64 s 的 60 帧中能看到火舌连续变形；该探针没有机器人；正式视频的相应时间段已另行复核，见下方实测结果。


开头 5 s、两路各 150 帧的运动补偿对照另见修正版输出的
`qa/first5s_motion_residual_comparison.json` 和 `qa/*_first5s_side_by_side.mp4`。
相同 H.264 CRF18 编码后，在 512×288 亮度图上用 Farneback 光流对齐相邻帧，对有符号
亮度残差做 11×11 模糊再统计绝对值，避开边界和跟随机位中大致的机器人区域。
跟随残差均值 2.5064 → 0.7461，前视 2.5769 → 0.7627（0–255 标度），均降低约 70%。
这是探查渲染波动的辅助比较，仍包含光流误差、反射和真实运动，不作为绝对验收门限；
不把该分析中的模糊处理应用到交付视频。


### 高采样修正版实测结果（2026-11-17 UTC）

- 两路各2029帧，1024×576、30 fps，视频67.633 s；仿真71.625 s（含4 s预热）。
- 采集7723.4 s，约128.7 min；物理、路线、相机位姿、报警参数与旧版一致。
- Job内12项、独立校验34项全部通过；两路所有帧完整解码。
- 照片火焰窗口916像素、5.59%，超过32像素/1%门限；拍照时报警灯关闭。
- 报警7 s，独立像素复核14次闪烁；机身最大单步偏差0.53 mm，腿部最大误差0.1941 rad。
- 跟随相机全部2029个采样时刻的净空、遮挡和全身取景均通过，最小画面边距4.11%。
- 资产审计：37,138个prim一致；除灯光强度和11条延长的火焰动画外，357,524个属性保持不变，
  非火焰时间采样及关系不变，原资产SHA-256未变；75个普通灯增强，32个其他灯保持参考版强度。
- 全程运动补偿亮度残差（最终7 s预期报警闪烁单列，不纳入下表）：

| 机位 | 旧版256 spp | 新版4096 spp | 均值下降 | 改善时段 |
|---|---:|---:|---:|---:|
| 跟随 | 2.7035 | 0.9570 | 64.6% | 31/31 |
| 前视 | 2.6777 | 0.8839 | 67.0% | 31/31 |

上述残差是0–255亮度标度的辅助比较，处理方法与开头5 s对照一致。完整原始视频的所有
相邻帧均参与计算，31个约2 s时段覆盖报警前全程；反射、真实运动与光流误差也会影响结果。
这说明闪烁明显改善，不等同于绝对无噪声或零闪烁。交付视频未经该分析中的模糊或时域滤波。

火焰复核：57–58.9 s原始帧近景及59.7–60.4 s视频解码近景中可见火舌高度、轮廓和倾斜变化；
后一段保存的机身XYZ与yaw变化范围均为0，排除相机转动造成的表观变化。人工复核还覆盖
两路每2 s抽帧、0/20/30/58 s附近连续帧和原生照片，范围记录于`qa/visual_review.json`，
不声称逐帧人工播放；完整解码和运动补偿分析覆盖所有帧。

Ruff、64份任务配置、后端列表检查通过；394项单元测试通过、3项跳过。
未使用官方LeRobot SDK加载；已独立读取元数据、Parquet和全部视频帧。

最终文件：

- `outputs/fire_inspection_demo_1k_bright_stable/demo/follow_1k_bright_stable_30fps.mp4`
- `outputs/fire_inspection_demo_1k_bright_stable/demo/front_rgb_1k_bright_stable_30fps.mp4`
- `outputs/fire_inspection_demo_1k_bright_stable/demo/fire_photo_1k_bright_stable.png`
- `outputs/fire_inspection_demo_1k_bright_stable/lerobot_v3/`

验收与证据：`validation.json`、`camera_validation_30fps.json`、`qa/asset_audit.json`、
`qa/temporal_quality_review.json`、`qa/full_video_motion_residual_comparison.json`、
`qa/fire_static_camera_review.json`及`qa/logs/`。旧版原始元数据未改写，包含原采集绝对路径；
查找归档照片时按其`relocation_note.json`替换目录前缀。
