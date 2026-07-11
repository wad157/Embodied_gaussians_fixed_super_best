# Embodied Gaussians — SUPER grasp5 适配工作日志

> 最后更新：2026-07-10
>
> 目标：在 `Embodied_gaussians_fixed_super_best` 中完成 grasp5 手术数据集的完整接入（深度估计 → 组织构建 → PSM 基座位姿拟合 → Demo 运行）。

---

## 总体进度

| # | 阶段 | 状态 | 关键产出 |
|---|------|------|---------|
| 1 | 深度估计改进 | ✅ | `depth_v2/` (全分辨率 64 迭代 RAFT-Stereo) |
| 2 | 组织/地面构建 | ✅ | `bodies_v5_table/` (z-up table frame, tissue 1050 particles + 2490 gaussians) |
| 3 | PSM URDF 适配 | ✅ | `psm.urdf` fixed base 拟合 (mean residual 1.97mm) |
| 4 | LND 中间数据生成 | ✅ | `psm1_lnd_motion.json` (5458 帧), `psm1_lnd_model.json` |
| 5 | PSM 基座位姿拟合 | ✅ | `fit_psm_urdf_base_to_lnd.py` (从零编写) |
| 6 | 投影验证 | ✅ | 左右目叠加图, URDF_visible=4/5 |
| 7 | Demo 接入 | ✅ | `super_embodied.py` + `example_embodied_super_offline.py` |
| 8 | Builder bug 修复 | ✅ | 2 处 mesh 加载崩溃修复 |

---

## Phase 1: 深度估计改进

### 问题

原始 `generate_super_depth.py` 使用 RAFT-Stereo 半分辨率 (960×540) + 24 次迭代。生成的深度图虽然数值范围合理 (0.05-0.19m)，但用于身体构建时，RANSAC 拟合出的地面平面法向量几乎竖直 (`[0.924, -0.066, 0.377]`)，导致 tissue 粒子生成方向错误。

### 解决

编写 `scripts/generate_super_depth_v2.py`：

- **全分辨率推理** (1920×1080)，不再 resize
- **64 次 RAFT 迭代**（原 24）
- **Mixed precision** 加速
- 可选 **bilateral filter** 边缘保持平滑

```bash
python scripts/generate_super_depth_v2.py --iters 64 --count 5
```

### 效果对比

| | 原始 (half res, 24 iter) | v2 (full res, 64 iter) |
|---|---|---|
| 地面法向量 | `[0.92, -0.07, 0.38]` (几乎竖直) | `[-0.21, -0.63, -0.75]` (接近水平) |
| Z 分量占比 | 38% | 73% |

### 产出

```
data/super/grasp5_native/depth_v2/
  000000-depth.npy ~ 000004-depth.npy
  000000-disparity.npy ~ 000004-disparity.npy
  depth_generation_summary.json
```

---

## Phase 2: 组织/地面构建

### v4 重建

`scripts/build_super_bodies_from_first_frame.py` 不再使用 3D RANSAC 和固定高度凸包：

- ground 在归一化相机射线坐标中使用 inverse-depth IRLS 稳健拟合，结果可复现。
- tissue footprint 保留 mask 的凹形轮廓，不再转换成凸包。
- tissue 顶面由 `depth_v2` 的逐位置高度场决定，平滑后限制在 3--15 mm。
- 物理粒子从离平面一个粒子半径处开始填充，避免初始穿透 ground。
- Gaussian 仍从组织表面独立初始化并训练，不与物理粒子共用点集。

### 构建参数

```bash
python scripts/build_super_bodies_from_first_frame.py \
  --depth data/super/grasp5_native/depth_v2/000000-depth.npy \
  --output-dir data/super/grasp5_native/bodies_v4 \
  --voxel-size 0.0015 \
  --particle-radius 0.0015 \
  --max-particles 5000 \
  --max-tissue-gaussians 3000 \
  --tissue-thickness 0.015 \
  --minimum-tissue-thickness 0.003 \
  --gaussian-iters 600 \
  --seed 42
```

### 产出

```
data/super/grasp5_native/bodies_v4/
  tissue.json        (1050 particles + 2490 gaussians)
  ground.json        (2500 gaussians + ground_plane 无限平面)
  ground_plane.json  (平面方程)
  build_metadata.json
  combined_v4.ply    (组织、ground 和平面检查点云)
```

`scripts/convert_super_scene_to_table_frame.py` 将 v4 相机坐标数据统一转换为
右手 table frame，输出 `bodies_v5_table/`。table 原点取组织中心在平面上的投影，
Z 轴为 ground 朝向组织的法向，因此 `ground_plane=[0,0,1,0]`，重力为
`[0,0,-9.80665]`。同一变换应用到 tissue、ground、PSM 和左右相机。

### 场景装配修复

- 移除 `add_rigid_body()` 对 `X_WB.z=0.1` 的强制覆盖，按 body JSON 原位加载。
- SUPER 场景不再缩小 tissue/ground Gaussian 和 tissue 粒子半径。
- 仿真 up vector 与拟合平面法向一致，重力指向 ground。
- `set_ground_plane()` 移到 `add_builder()` 之后，避免 Warp 把真实 offset 覆盖为 0。
- table-frame CUDA 无视觉力测试 100 步状态均为有限值，tissue 位移约 0.017 mm。

---

## Phase 3: PSM URDF 适配

### Step 1: 文件复制

从 `embodied_gaussians_fixed_super_fin` 复制已处理的 URDF 和重建脚本：

```bash
cp -r super_fin/data/super/psm_robot super_best/data/super/psm_robot
cp super_fin/scripts/rebuild_super_psm_from_dvrk_xacro.py super_best/scripts/
```

### Step 2: URDF 结构验证

编写验证脚本检查 17 个 link 的 `<collision>` 和 `<inertial>` 完整性、7 个关键关节类型、所有 mesh 文件存在性。

| 检查项 | 结果 |
|--------|------|
| 17 个 link 都有 `<collision>` | ✅ |
| 17 个 link 都有 `<inertial>` | ✅ |
| insertion = **prismatic** | ✅ (0→0.240m) |
| roll limit 扩展到 ±3.5 | ✅ (原数据 max 3.07) |
| jaw limit 扩展到 -1.2→1.6 | ✅ (原数据 -0.48→1.00) |
| 所有 STL mesh 存在 | ✅ 13 个文件 |

### Step 3: joints.json 兼容性

| 检查项 | 结果 |
|--------|------|
| joint_names 匹配 `input_joint_names` | ✅ 7 个完全一致 |
| 5458 帧全部在 URDF limits 内 | ✅ |
| mimic 展开后 14 个 URDF 关节 | ✅ FK 正确 |
| robots.json 有 `PSM1` key | ✅ 5458 states |

---

## Phase 4: LND 中间数据生成

### 从零编写脚本

`scripts/build_super_psm_lnd_intermediates.py`（未从 super_fin 复制，完全新写）：

**输入:**
- `data/LND.json` — Modified DH 参数 + point features + skeleton
- `data/handeye.yaml` — PSM 基座在 raw camera 的位姿
- `data/camera_calibration.yaml` — 立体标定
- `data/super/grasp5_native/joints.json` — 5458 帧关节状态
- `data/super/grasp5_native/calib_rectified.json` — 矫正标定

**处理流程:**
1. 解析 LND Modified DH 参数，逐帧计算 FK
2. 提取 point_features、skeleton_structure、shaft_features
3. 通过 handeye 矩阵 (rvec→R, tvec×0.001 mm→m) 变换到 raw camera
4. 通过 R1 矫正矩阵变换到 rectified left camera
5. 输出每帧 keypoints 和 link transforms

**踩坑记录:**
- Prismatic joint 的 `offset` 应加到 `d` 而非 `theta`。初版错误导致只有 1/24 keypoints 在图像内；修正后恢复到 22/24。

### 产出

```
data/super/grasp5_offline_demo/instruments/
  psm1_lnd_model.json          (FK 模型 + 坐标变换矩阵)
  psm1_lnd_motion.json         (5458 帧 × 24 keypoints × link transforms)
  psm1_lnd_generation_report.json
```

---

## Phase 5: PSM 基座位姿拟合

### 从零编写脚本

`scripts/fit_psm_urdf_base_to_lnd.py`（未从 super_fin 复制，完全新写）：

**方法:**
1. 加载 URDF，从 `PSM1_psm_base_link` 开始做 FK（跳过 world→base 的 fixed joint，等效 identity base）
2. 加载 LND 第 0 帧的 link transforms，变换到 rectified camera 坐标
3. 建立 URDF link origin ↔ LND link origin 对应关系（6 对）
4. Weighted Umeyama 刚体拟合 (SVD)
5. 将拟合结果写入 URDF `fixed` joint origin

**对应关系:**

| URDF Link | LND Link | 权重 | 残差 |
|-----------|----------|------|------|
| `PSM1_psm_base_link` | LND 0 (base) | 3.0 | **0.37mm** |
| `PSM1_outer_pitch_link` | LND 2 (pitch) | 3.0 | **0.37mm** |
| `PSM1_tool_main_link` | LND 3 (insertion) | 3.0 | **0.34mm** |
| `PSM1_tool_wrist_link` | LND 4 (wrist) | 2.0 | **0.45mm** |
| `PSM1_tool_wrist_sca_shaft_link` | LND 6 (distal) | 2.0 | **0.65mm** |
| `PSM1_tool_tip_link` | LND 6 (distal) | 0.5 | **9.64mm** |
| **加权平均** | | | **1.97mm** |

**最终 URDF fixed base:**

```xml
<origin rpy="-1.052313 -1.458247 -3.029056" xyz="0.091887 -0.060131 -0.002701"/>
```

PSM 基座在 rectified left camera 坐标系下约 (92mm, -60mm, -3mm)。

### 产出

```
data/super/psm_robot/
  psm.urdf                         (更新了 fixed base origin)
  psm.urdf.bak                     (拟合前备份)
  psm_base_correction_report.json  (拟合报告)
```

---

## Phase 6: 投影验证

### 从零编写脚本

`scripts/validate_psm_projection.py`（完全新写）：

- 加载更新后的 URDF，计算 frame 0 FK
- 提取 5 个关键 URDF link 原点
- 与 LND keypoints 同时投影到左右目图像
- 生成叠加对比图

**验证结果:**
- LND 关键点: 22/24 在图像内
- URDF 关键 link: 4/5 在图像内
- 腕部对齐误差: 0.4mm
- 末端对齐误差: 0.6mm

### 产出

```
data/super/psm_robot/validation/
  frame000000_stereo_left_overlay.png
  frame000000_stereo_left_compare.png
  frame000000_stereo_right_overlay.png
  frame000000_stereo_right_compare.png
```

---

## Phase 7: Demo 接入

### 场景定义

`examples/embodied_environments/super_embodied/super_embodied.py`（参考 super_fin 编写，适配 super_best API）：

**场景组成:**

| 组件 | 类型 | 数量 | 配置 |
|------|------|------|------|
| PSM | URDF articulation | 18 body, 20 gaussians | 碰撞全关, 尖端 0.5mm 高斯 |
| Tissue | Rigid body | 1 body, ~2835 gaussians | mu=0.05, ground 碰撞 ✅ |
| Ground | Visual body | ~2500 gaussians | 渲染 + 无限碰撞平面 |
| Camera | 离线视频 | stereo_left | 1920×1080, 30fps |

**关键适配点 (super_fin → super_best):**
- `robots.json` key: `sheep` → `PSM1`
- `add_rigid_body()` 移除 `density` 和 `origin_z_override` 参数 (super_best 不支持)
- `keep_only_cameras()` 改为 `hasattr` 兼容检查 (super_best 无此方法)

### Demo 入口

`examples/example_embodied_super_offline.py`（从 super_fin 复制，改 robots key + API 兼容）：

**功能:**
- `SuperPlaybackControls`: 按时间戳驱动 PSM 关节 (q7→q_full 展开)
- 每步物理后重新锚定 PSM (`set_robot_q` + `update_gaussian_transforms`)
- PSM roll 手动偏移 (`--psm-roll-offset-deg`)
- 监控模式 (`--monitor-psm-base-q`, `--monitor-tissue-q`)
- Visual forces 可关闭 (`--visual-force-iterations 0`)

### 运行脚本

`scripts/run_demo_on_display.sh` 更新为调用 `example_embodied_super_offline.py`:

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

# 一条命令启动
bash scripts/run_demo_browser_12.sh

# 监控模式
bash scripts/run_demo_browser_12.sh \
  --visual-force-iterations 0 \
  --monitor-psm-base-q \
  --monitor-tissue-q \
  --monitor-interval 0.2
```

---

## Phase 8: Builder Bug 修复

`src/embodied_gaussians/embodied_simulator/builder.py` 修复两处崩溃：

**Bug 1: 非 mesh shape 导致 NoneType (line 135-136)**

PSM 的 `PSM1_outer_pitch_link` 等 link 只有 `<collision>` 使用 box/sphere 基元，没有 triangle mesh。`self.shape_geo_src[i]` 返回 None。

```python
# 修复: 跳过 None 或空 mesh
if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
    continue
```

**Bug 2: 微小 mesh 导致零采样点 (line 144-148)**

`tool_wrist_shaft_link.stl` (300 顶点) 面积极小，`int(area * 10000) = 0`，触发 Open3D `number_of_points <= 0` 错误。

```python
# 修复: 确保至少 1 个采样点
n_samples = max(1, int(area * points_per_unit_area))
```

---

## 辅助工具

### 场景 PLY 导出

`scripts/export_scene_ply.py` — 将完整场景导出为单个 PLY 文件：

```
data/super/scene_overview.ply  (41487 顶点)
```

颜色编码: 🟠 PSM (14 link mesh) | 🔴 Tissue gaussians | 🩷 Tissue particles | 🔵 Ground + 平面 | 🟡 相机视锥

### 场景构建验证

```bash
python -c "
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'examples')
import warp as wp; wp.init()
from embodied_environments.super_embodied.super_embodied import build_environment
env = build_environment()
print(f'Bodies: {env.sim.model.body_count}, Shapes: {len(env.sim.model.shape_body)}')
"
# 输出: Bodies: 19, Shapes: 3671
```

---

## 文件结构总览

```
Embodied_gaussians_fixed_super_best/
│
├── WORK_LOG.md                            ← 本文件
│
├── scripts/
│   ├── generate_super_depth_v2.py         ← Phase 1: 全分辨率 64 迭代深度生成
│   ├── build_super_bodies_from_first_frame.py ← Phase 2: 稳健平面 + 深度高度场组织构建
│   ├── convert_super_scene_to_table_frame.py ← Phase 2: 整体转换到 z-up table frame
│   ├── build_super_psm_lnd_intermediates.py   ← Phase 4: LND FK 中间数据 (从零编写)
│   ├── fit_psm_urdf_base_to_lnd.py            ← Phase 5: URDF base 拟合 (从零编写)
│   ├── validate_psm_projection.py             ← Phase 6: 投影验证 (从零编写)
│   ├── export_scene_ply.py                    ← 辅助: 场景 PLY 导出
│   ├── rebuild_super_psm_from_dvrk_xacro.py   ← 从 super_fin 复制
│   ├── run_demo_on_display.sh              ← 更新为 SUPER demo
│   └── run_demo_browser_12.sh              ← (无需修改)
│
├── examples/
│   ├── embodied_environments/
│   │   └── super_embodied/
│   │       ├── super_embodied.py           ← Phase 7: SUPER 场景定义
│   │       ├── objects/
│   │       │   ├── tissue.json             ← 从 bodies_v5_table 复制
│   │       │   └── ground.json             ← 从 bodies_v5_table 复制
│   │       └── environment/
│   │           └── ground_plane.json       ← 从 bodies_v5_table 复制
│   └── example_embodied_super_offline.py   ← Phase 7: Demo 入口
│
├── src/embodied_gaussians/embodied_simulator/
│   └── builder.py                         ← Phase 8: 2 处 bug 修复
│
├── data/
│   ├── LND.json                           ← (原始数据)
│   ├── handeye.yaml                       ← (原始数据)
│   ├── camera_calibration.yaml            ← (原始数据)
│   └── super/
│       ├── grasp5_native/
│       │   ├── depth_v2/                  ← Phase 1 产出
│       │   ├── bodies_v4/                 ← 相机坐标系重建产出
│       │   ├── bodies_v5_table/           ← 当前 table-frame 场景资源
│       │   ├── joints.json
│       │   ├── calib_rectified.json
│       │   ├── rgb/
│       │   └── masks/
│       ├── grasp5_offline_demo/
│       │   ├── robots.json
│       │   ├── cameras.json
│       │   ├── videos/
│       │   └── instruments/               ← Phase 4 产出
│       ├── psm_robot/
│       │   ├── psm.urdf                   ← Phase 3+5 产出
│       │   ├── psm.urdf.bak
│       │   ├── meshes/*.stl
│       │   ├── psm_mimic_map.json
│       │   ├── psm_base_correction_report.json  ← Phase 5 产出
│       │   └── validation/                ← Phase 6 产出
│       ├── table_frame.json               ← camera 到 table 的统一变换
│       └── super_tissue_plane_psm_cameras_table.ply ← 完整场景检查 PLY
│
└── third_party/Python-SuPer/              ← RAFT-Stereo / Monodepth2 代码
```

---

## 未完成 / 待优化

1. **深度图覆盖更多帧**: 当前仅生成 5 帧 v2 深度，完整 pipeline 需要全部 1644 帧
2. **Monodepth2 fine-tuned 权重**: Google Drive 链接无法访问，如能获取可进一步改善深度质量
3. **PSM tool_tip 残差 9.64mm**: 末端几何差异 (URDF vs LND) 需单独微调 link 长度
4. **Warp joint order**: 当前硬编码 `PSM_WARP_JOINT_Q_ORDER`，如 Warp 版本更新可能需要重新确定顺序
5. **完整 demo 运行**: 需要真实 GPU + X11 显示环境
