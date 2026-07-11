# SuPer grasp5 Progress

## 完整适配指南（2026-07-11）

当前最终适配路径、坐标系定义、tissue/ground 构建算法、严格 LND+dVRK PSM 位姿链、剩余偏差归因和复现命令已汇总到 `SUPER_COMPLETE_ADAPTATION_GUIDE.md`。该指南以 `bodies_v4`、`bodies_v5_table` 和 `psm_lnd_pose_driver.npz` 为当前权威产物，并明确排除历史 `bodies_v3`、frame 0 base 拟合和 direct-URDF q7 驱动路径。

## 当前数据产物

| 文件/目录 | 作用 |
|---|---|
| `data/super/grasp5_native/rgb/*-left.png` | 左目 rectified 图像序列。用于 SAM2 分割、深度估计输入、overlay 验证。 |
| `data/super/grasp5_native/rgb/*-right.png` | 右目 rectified 图像序列。和左目组成 stereo pair，用于 RAFT-Stereo/Python-SuPer 估计深度。 |
| `data/super/grasp5_native/depth/*-depth.npy` | 深度图，单位米。用于把 2D mask 像素反投影成 3D 点云，构建 tissue/ground。 |
| `data/super/grasp5_native/depth/*-disparity.npy` | 视差图。主要用于调试深度质量，正常建模主要用 depth。 |
| `data/super/grasp5_native/depth/*-depth_preview.png` | 深度图可视化预览。方便肉眼快速检查深度是否合理。 |
| `data/super/grasp5_native/depth/depth_generation_summary.json` | 深度生成记录：使用的 checkpoint、baseline、fx、每帧深度统计等。用于复现实验和排查尺度。 |
| `data/super/grasp5_native/calib_rectified.json` | rectified 后的相机标定：`K_left_rect/K_right_rect/P1/P2/Q/baseline` 等。用于深度换算、投影、后续构建相机模型。 |
| `data/super/grasp5_native/joints.json` | 从 bag 提取的 PSM1 原始关节序列，包含 5458 条 q、q_d、effort、timestamp、joint names。用于检查和生成 `robots.json`。 |
| `data/super/grasp5_native/super_dataset_metadata.json` | native 数据元信息：bag 路径、使用帧范围、输出目录、帧数、关节数等。用于记录这次提取过程。 |
| `data/super/grasp5_offline_demo/robots.json` | 当前 Embodied Gaussians `DatasetManager` 读取的机器人数据。demo 用它按时间驱动 PSM1 关节。 |
| `data/super/grasp5_offline_demo/cameras.json` | 当前项目的相机 manifest。告诉 `DatasetManager` 有哪些相机、外参、视频路径和 metadata 路径。 |
| `data/super/grasp5_offline_demo/videos/stereo_left.mp4` | 左目视频。demo 离线回放时读取它作为真实观测图像。 |
| `data/super/grasp5_offline_demo/videos/stereo_right.mp4` | 右目视频。当前主要用于双目验证/备用相机，后续也可做多视角观测。 |
| `data/super/grasp5_offline_demo/videos/stereo_left.json` / `stereo_right.json` | 每个视频的 metadata：相机内参 `K`、分辨率、每帧 timestamp。`OfflineCamera` 依赖它按时间索引视频帧。 |

## 目录定位

`data/super/grasp5_native` 是处理中间数据区，方便做深度、分割、点云和 debug。

`data/super/grasp5_offline_demo` 是项目运行数据区，目标是直接喂给 `DatasetManager` 和 demo。

## GUI 相机画面对齐（2026-07-11）

| 项目 | 状态 | 结论/产出 |
|---|---|---|
| 左目视频内容核对 | ✅ | `stereo_left.mp4` 首帧与 `rgb/000000-left.png` 均为 `1920x1080`；相位相关测得位移约 `(-0.0020, -0.0022) px`，可排除视频编码造成的画面平移。 |
| 相机模型核对 | ✅ | GUI、场景构建和 PLY 均使用 rectified 内参 `fx=fy=1742.7886`、`cx=860.4195`、`cy=682.2899`。相机位姿采用表坐标系下的 Blender camera-to-world，渲染时转换为 OpenCV 相机约定。 |
| GUI 偏差根因 | ✅ | `marsoom.CameraWireframe` 在主点不位于图像中心时错误地向纹理四角加入非对称偏移，并把四角归一化到球面，导致视频纹理平面与正确的场景投影不一致。左目主点相对图像中心约偏移 `(-99.6, +142.3) px`，旧实现四角射线误差约 `2.3~4.7 deg`。 |
| GUI 纹理平面修复 | ✅ | 在 `src/embodied_gaussians/embodied_visualizer/embodied_viewer.py` 中按 `(0,0)`、`(w,0)`、`(w,h)`、`(0,h)` 和真实 `K^-1` 重建四角射线，并将四角放在同一个成像平面上；离线相机与虚拟相机均使用该逻辑。未修改 conda 环境中的第三方包。 |
| Go To Camera 视野 | ✅ | 原始内参按当前 GUI 视口等比缩放，默认 `--camera-go-zoom 0.9`，完整包含左目画面并留约 10% 边距。`1.0` 表示刚好完整包含，值越小视野越大。 |
| 数值回投影验证 | ✅ | 修正后的纹理四角回投影为 `(0,0)`、`(1920,0)`、`(1920,1080)`、`(0,1080)`，数值误差小于 `0.001 px`，四角深度一致。 |
| GUI 人工验证 | ✅ | 用户已在虚拟显示器 GUI 中确认投影没有问题。 |

运行命令：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
bash scripts/run_demo_browser_12.sh
```

需要进一步缩小 Go 后的视角时：

```bash
bash scripts/run_demo_browser_12.sh --camera-go-zoom 0.8
```

## PSM 尖端表面 Gaussian（2026-07-11）

| 项目 | 状态 | 结论/产出 |
|---|---|---|
| 运动学路径 | ✅ | dVRK xacro 重建 URDF，fixed base 经 LND/hand-eye 对齐；运行时将 `robots.json` 的 q7 展开为含 mimic 的 q14，由 Warp FK 驱动各 link。PSM 继续关闭自身、组织和地面碰撞。 |
| 旧视觉路径问题 | ✅ | 旧实现从 Warp shape/collision 几何采样后，只保留约 20 个腕部 Gaussian，并强制设置为各向同性 `0.5 mm`；同时没有可靠应用 visual origin，因此过小、过稀且可能偏离 visual mesh。 |
| 确定的视觉路径 | ✅ | 从 URDF `<visual>` mesh 离线采样，在 link-local 坐标中应用 mesh scale 和 visual origin，按表面法线构造贴面椭球，再按 link body id 绑定并随 FK 更新。 |
| 采样范围 | ✅ | 只保留 `PSM1_tool_main_link` 及其 URDF 下游子树，即插入杆、腕部、末端执行器和夹爪；不采样 base、yaw/pitch 和上方连杆。9 个下游 link 中 7 个具有 visual mesh。 |
| Gaussian 资产 | ✅ | `data/super/psm_robot/psm_surface_gaussians.npz`：1508 个 Gaussian、7 个 link。大 mesh 密度 `30000/m^2`，小 mesh 至少 64 个。 |
| Gaussian 尺度 | ✅ | 按每个 mesh 的采样间距计算；切向约 `0.7–3.75 mm`，法向约 `0.2–0.75 mm`，形成贴合表面的薄椭球。 |
| 表面验证 | ✅ | `psm_surface_gaussians_report.json` 记录各 link 覆盖数和点到 visual mesh 距离，最坏单点约 `0.27 mm`。 |
| CUDA 接入验证 | ✅ | 环境成功加载 1508 个 PSM Gaussian，总场景 6498 个 Gaussian，绑定 18 个 robot body 后世界坐标全部 finite。 |
| PLY 检查 | ✅ | `data/super/psm_robot/psm_tip_gaussians_scene_frame0.ply`；橙色为 URDF visual mesh，青色为尖端 Gaussian 中心。 |

重新生成尖端 Gaussian：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python scripts/build_psm_surface_gaussians.py
```

## PSM 器械多帧位姿适配（2026-07-11）

| 项目 | 状态 | 结论/产出 |
|---|---|---|
| 误差归因 | ✅ | 主要是适配方法问题，不是 depth 精度。旧路径将 SuPer q7 直接送入 dVRK URDF，并只在 frame 0 拟合 base；误差会随关节运动增长。 |
| 旧路径跨帧误差 | ✅ | 400 个采样状态中，`tool_main` 位置误差 P95 `30.7 mm`、姿态误差 P95 `6.28 deg`；腕部/末端位置误差 P95 约 `14.9~15.9 mm`。 |
| LND feature 单位 | ✅ | 修复 `build_super_psm_lnd_intermediates.py` 的重复 `x0.001`。`LND.json` feature 本身使用米；同时按原注释将 `grip_far` 的 `0.09` 修正为 `0.0095 m`。已重建 5458 帧 LND motion。 |
| 时间同步 | ✅ | 视频约 `29.77 Hz`，关节约 `99.28 Hz`。验证和新 overlay 均按视频 timestamp 查询最近关节状态，最近邻误差 P95 `4.77 ms`。旧 `validate_psm_projection.py` 已修正，不再按相同帧号错误配对。 |
| 确定的位姿路径 | ✅ | SuPer `LND.json` Modified-DH FK 决定器械运动；原始 `data/dvrk_model` 决定 CAD 表面。每个尖端 link 的 `T_LND-link->dVRK-link` 只由两个源模型在标准 `q=0` 下解析，运行时由 `handeye x LND FK(q) x model-link-offset` 更新 body。 |
| 无绝对修正 | ✅ | pose-driver 明确记录 `uses_dataset_reference_frame=false`、`uses_image_based_correction=false`；不使用视频 mask、frame 0、手工 base/roll 修正或深度结果。 |
| Canonical 模型一致性 | ✅ | LND 与 dVRK 对应 link 原点在 `q=0` 下的最大残差为 `0.00079 mm`，说明两个原始模型可直接建立固定坐标关系。 |
| 位姿驱动资产 | ✅ | `data/super/psm_robot/psm_lnd_pose_driver.npz`，包含 5458 帧、7 个尖端 visual link；`psm_lnd_pose_driver_report.json` 保存 canonical 模型变换和残差；生成脚本为 `scripts/calibrate_psm_lnd_pose_driver.py`。 |
| Runtime 接入 | ✅ | `apply_psm_lnd_pose()` 在跳转时间、每次 physics step 后覆盖 7 个尖端 body，并立即更新 body-bound Gaussian。原 dVRK articulation 不再决定可见器械姿态。 |
| CUDA 验证 | ✅ | 第 0 帧和第 3000 个状态切换成功，7 个 body 位姿 finite；实际 GUI 连续运行稳定，PSM base 无漂移。 |
| 更新后投影画面 | ✅ | `data/super/psm_robot/lnd_pose_validation/frame*_left_before_after.png`。左侧橙色为旧 direct URDF，右侧青色为严格 canonical LND+dVRK pose；对比图仅验证，不参与参数计算。 |

关键对比图：

- `data/super/psm_robot/lnd_pose_validation/frame000000_left_before_after.png`
- `data/super/psm_robot/lnd_pose_validation/frame000500_left_before_after.png`
- `data/super/psm_robot/lnd_pose_validation/frame001000_left_before_after.png`
- `data/super/psm_robot/lnd_pose_validation/frame001400_left_before_after.png`

重新生成 LND 数据、位姿驱动资产和对比图：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python scripts/build_super_psm_lnd_intermediates.py
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python scripts/calibrate_psm_lnd_pose_driver.py
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python scripts/validate_psm_lnd_pose_projection.py
```

启动更新后的 GUI：

```bash
bash scripts/run_demo_browser_12.sh
```
