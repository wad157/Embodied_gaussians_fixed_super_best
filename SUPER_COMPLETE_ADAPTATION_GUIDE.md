# SUPER grasp5 完整适配方案

本文记录 `grasp5` 从 SuPer 原始 ROS bag 到 Embodied Gaussians 离线 GUI 的当前最终适配路径。文档以仓库中的实际代码、`PROGRESS.md`、`WORK_LOG.md` 和最终产物报告为准，重点说明坐标系、tissue/ground 构建以及 PSM 器械适配。

## 1. 当前结论

当前运行时的权威场景产物是：

- 相机坐标重建结果：`data/super/grasp5_native/bodies_v4/`
- table 坐标运行结果：`data/super/grasp5_native/bodies_v5_table/`
- 运行时场景副本：`examples/embodied_environments/super_embodied/`
- PSM 几何：`data/super/psm_robot/psm.urdf` 和 `psm_surface_gaussians.npz`
- PSM 严格位姿：`data/super/psm_robot/psm_lnd_pose_driver.npz`
- 离线数据集：`data/super/grasp5_offline_demo/`

`bodies_v3`、frame 0 base 拟合和 q7 直接驱动 dVRK URDF 都是历史调试路径，不再作为最终方案。

当前方案遵守三个原则：

1. 场景几何先在 rectified 左目 OpenCV 相机系中恢复，再整体变换到 table frame。
2. tissue 的物理粒子和视觉 Gaussian 不共用同一批最终点；ground 只提供视觉 Gaussian，物理接触使用独立的无限平面。
3. PSM 的运动由 SuPer `LND.json` 和原始 q7 决定，外观由 `data/dvrk_model` 的 CAD 决定；不使用 frame 0、图像 mask 或人工位姿修正。

## 2. 为什么严格推导后仍有偏差

仍有偏差不等于代码中的坐标链仍然错误，也不能笼统归因于 depth。

当前 canonical 对齐报告中，LND 与 dVRK 对应 link 原点在 `q=0` 下的最大残差为 `0.0007936 mm`。这证明两个数字模型可以建立稳定的固定映射，但它只验证数字模型的 link 原点一致性，不验证真实器械、相机和编码器的物理精度，也不是图像轮廓误差指标。

PSM 严格位姿链完全不使用 depth，因此器械投影偏差通常来自以下因素：

| 误差源 | 典型表现 | 当前状态 |
|---|---|---|
| hand-eye 外参残差 | 所有帧、所有 link 大致同方向平移或旋转 | 使用数据集 `handeye.yaml`，未做图像绝对修正 |
| 真实安装工具与 dVRK CAD 不同 | shaft 基本对准，但腕部、夹爪轮廓有固定局部差异 | 当前使用 dVRK SCA visual mesh |
| 电缆传动、回差、柔顺性和编码器零位 | 误差随 roll、wrist、jaw 状态变化，并可能有滞回 | LND FK 是理想刚体模型，无法表达这些物理效应 |
| 图像与关节采集延迟 | 快速运动时偏差变大，静止时减小 | 最近邻时间差 P95 为 `4.77 ms`，但未知固定采集延迟仍可能存在 |
| 标定、畸变和 rectification 残差 | 误差随画面位置变化，边缘区域通常更明显 | 使用原始标定和 OpenCV rectification，没有再标定 |
| LND 参数或数据字段误差 | 特定自由度出现系统偏差 | 已修正 feature 重复乘 `0.001` 和 `grip_far=0.0095 m`，其余参数仍依赖源数据质量 |
| Gaussian 表达误差 | Gaussian 中心正确但轮廓显得偏粗、偏薄或越界 | 表面中心到 CAD 最坏约 `0.27 mm`，渲染还受 scale/opacity 影响 |

可用偏差模式快速定位：

- 所有帧都近似同一个刚体偏差：优先检查 hand-eye、相机外参和固定工具安装变换。
- shaft 位置正确但绕轴角固定偏差：检查 roll 零位、visual origin 和实际工具安装角。
- 偏差随关节角连续变化：检查 q 定义、DH 参数、传动比以及电缆传动误差。
- 只在腕部或 jaw 出现：检查实际工具型号、jaw 映射和末端 CAD。
- 只在快速运动出现：检查图像时间戳、关节时间戳和固定延迟。
- 左右目偏差方向相反：检查 stereo 外参、baseline 和左右相机投影链。

所以当前结论是：旧方案的主要误差确实来自适配方法；严格 LND 链已经排除了这部分大误差。严格链剩余的小偏差更可能是源标定、时间延迟、真实机构非理想性或 CAD 与实物差异。由于本阶段明确“不做绝对修正”，这些物理残差会保留在画面中。

## 3. 输入数据与离线数据集

原始输入包括：

- `data/grasp5/grasp5.bag`：左右目图像和 PSM1 关节状态。
- `data/camera_calibration.yaml`：原始 stereo 内外参和畸变。
- `data/handeye.yaml`：PSM base 到 raw 左相机的 hand-eye 结果。
- `data/LND.json`：SuPer 提供的 Modified-DH、skeleton、shaft 和 point features。
- `data/dvrk_model/`：dVRK xacro、mesh 和工具 CAD。

提取后的主要数据为：

| 产物 | 数量/参数 | 用途 |
|---|---:|---|
| rectified 左右目 PNG | 1644 对，`1920x1080` | depth、SAM2、投影验证 |
| `joints.json` | 5458 个 q7 状态，约 `99.28 Hz` | LND FK 和离线回放 |
| 左右目 MP4 | 各 1441 帧，约 `29.77 Hz` | GUI 真实观测 |
| rectified K | `fx=fy=1742.788593`，`cx=860.419510`，`cy=682.289867` | 反投影和渲染 |
| stereo baseline | `0.00531613697 m` | disparity 到 depth |

`data/super/grasp5_native/` 保存 PNG、depth、mask 和场景构建中间产物；`data/super/grasp5_offline_demo/` 保存 `DatasetManager` 直接读取的 `robots.json`、`cameras.json`、视频及其时间戳 metadata。

## 4. 坐标系与变换约定

### 4.1 统一记号

全文使用：

```text
X_A_B：把 B 坐标系中的齐次坐标变换到 A 坐标系
p_A = X_A_B @ p_B
```

严禁根据变量名猜方向。所有新增矩阵都应按这一约定命名，并在 JSON 中记录 source/target frame。

### 4.2 rectified 左目相机系

depth、mask 反投影和初始场景重建均使用 rectified 左目 OpenCV 相机系：

```text
x：图像向右
y：图像向下
z：相机向前
```

这是右手系。像素 `(u,v)` 和深度 `z` 的反投影为：

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
p_camera = [x, y, z]
```

相机画面中的桌面本来就是下方近、上方远，因此桌面法向不会与相机 z 轴平行。不能通过把 depth 强行拉平来得到桌面。

### 4.3 camera 到 table frame

相机系拟合平面为：

```text
n_camera = [-0.0188160968, -0.5470410611, -0.8368942777]
d_camera =  0.0858578239
n_camera^T p_camera + d_camera = 0
```

法向被定向为组织所在一侧为正。table frame 的构造规则是：

1. 取 tissue 粒子中心，将其正交投影到拟合平面，作为 table 原点。
2. `z_table` 取平面法向，指向 tissue。
3. 将相机 `+x` 投影到平面，归一化后作为 `x_table`。
4. `y_table = z_table x x_table`，并检查旋转矩阵行列式为正。

最终使用：

```text
X_table_camera =
[[ 0.9998229616, -0.0102950002, -0.0157498721,  0.0003378840],
 [ 0.0000000000, -0.8370424663,  0.5471379255, -0.0493748731],
 [-0.0188160968, -0.5470410611, -0.8368942777,  0.0858578239],
 [ 0.0000000000,  0.0000000000,  0.0000000000,  1.0000000000]]
```

table 原点在相机系中为 `[0.0012776850, 0.0056423681, 0.0988741088] m`。转换后：

```text
ground_plane = [0, 0, 1, 0]
ground z = 0
gravity = [0, 0, -9.80665] m/s^2
```

同一个 `X_table_camera` 必须用于 tissue、ground、PSM 和左右相机，不能分别进行人工平移。

### 4.4 OpenCV 与 Blender 相机轴

`cameras.json` 的 `X_WC` 使用 Blender camera-to-world 约定，而场景重建使用 OpenCV 相机轴。写入 manifest 前使用：

```text
X_table_camera_blender =
    X_table_camera_opencv @ diag(1, -1, -1, 1)
```

`FramesBuilder` 在 rasterization 前会再转换回 OpenCV 轴。若漏掉这一步，PLY 可能正确而 GUI 视频平面与渲染不重合。

右目相机在 rectified 左目世界中的中心沿相机 `+x` 平移一个 baseline。转换 table frame 前必须先补上这项外参。

## 5. Depth 和第一帧分割

### 5.1 Depth

当前 `bodies_v4` 实际使用：

```text
data/super/grasp5_native/depth_v2/000000-depth.npy
```

它由 Python-SuPer 中的 RAFT-Stereo 在原始 `1920x1080` 分辨率、64 次迭代下生成。深度换算为：

```text
z = fx * baseline / (disparity + cx_right - cx_left)
```

前 5 帧的中位深度约 `98.3-100.6 mm`。生成后使用 depth bilateral filter 抑制局部噪声，但这不能修复错误匹配或标定误差。

`scripts/filter_super_disparity.py` 可以在 tissue/ground mask 内按 percentile、MAD、median 和 bilateral 进一步过滤 disparity。需要特别注意：当前最终 `bodies_v4` 的 metadata 指向未经过该独立脚本的 `depth_v2/000000-depth.npy`。若改用 filtered depth，必须输出到新目录重建并重新验证，不能继续沿用 `bodies_v4` 的统计结论。

增加更多 depth 帧不会自动改善第一帧模型。只有先把多帧通过相机/场景运动配准到同一三维坐标系，再做时序融合才有意义；直接逐像素平均会混合器械运动和组织形变。

### 5.2 SAM2

只在 rectified 左目第一帧上交互分割：

- `000000-tissue.png`：组织区域。
- `000000-ground.png`：可见桌面区域。
- `*_overlay.png`：肉眼验证 mask 边界。
- `000000-sam2_metadata.json`：模型、面积和文件记录。

2D overlay 已确认无明显问题。后续 3D 形状错误不能通过修改正确的 mask 来掩盖，应检查 disparity、反投影、平面拟合和体积填充。

## 6. ground plane、tissue 和 ground 构建

实现入口为 `scripts/build_super_bodies_from_first_frame.py`。

### 6.1 平面拟合

流程如下：

1. 用 ground mask 对 depth 反投影。
2. 将 mask 腐蚀 7 px，减少 tissue/器械边界错误深度对拟合的影响。
3. 每 8 个点采样一次，在归一化像素射线坐标中拟合 inverse depth。
4. 使用最多 25 次 Huber IRLS，避免固定 3D RANSAC 阈值对近处点产生偏置。
5. 根据 tissue 中位 signed distance 调整法向，使 tissue 位于正侧。

对于平面：

```text
a*x + b*y + c*z + d = 0
x = ray_x*z, y = ray_y*z
```

可整理为：

```text
1/z = beta_x*ray_x + beta_y*ray_y + beta_0
plane = normalize([beta_x, beta_y, beta_0, -1])
```

这种拟合不会假设桌面与图像平行。当前 ground 点到平面的 signed distance 中位数约 `-0.815 mm`；P95 absolute distance 约 `10.47 mm`，反映了单帧 stereo depth 中存在明显局部噪声。最终 ground Gaussian 不直接使用这些起伏点，而是用 mask 射线与拟合平面的交点生成，所以视觉 ground 被严格放在同一平面上。

### 6.2 tissue 物理粒子

tissue 不是固定长方体，也不使用 mask 的凸包。当前体积填充步骤为：

1. 将 tissue depth 点投影到拟合平面的局部二维坐标。
2. 保留 mask 对应的凹形 footprint，并用 1%/99% 分位限制异常边界。
3. 以 `2*radius = 3 mm` 建立平面网格。
4. 对每个网格位置，从邻近 8 个 tissue surface 点做反距离加权，得到局部 surface height。
5. 重新投影 top point 到第一帧 mask，剔除 mask 外和离观测表面过远的网格。
6. 对占用网格的高度做 Gaussian 平滑，并裁剪到 `3-15 mm`。
7. 每列从 `z=radius` 开始，每隔 `2*radius` 放一个粒子，直到表面下一个 radius。

粒子中心的最低高度等于 `1.5 mm`，所以半径 `1.5 mm` 的球刚好接触 `z=0` 平面而不穿透。当前结果为：

```text
footprint cells: 419
particles:       1050
particle radius: 1.5 mm
column height:   3.00-14.55 mm
```

这一步决定物理体积和碰撞。不能直接把表面点云当粒子，否则粒子只形成一层壳；也不能用规则长方体填充，否则会丢失 mask 的凹形边界和局部高度。

### 6.3 tissue 视觉 Gaussian

视觉 Gaussian 与物理粒子分支独立生成：

1. 从 tissue depth surface 点中采样 Gaussian 初始中心，而不是使用最终粒子中心。
2. 使用第一帧 RGB、depth、mask 和 K 调用 `SimpleBodyBuilder._grow_gaussians()` 训练 600 次。
3. scale 约束为 `0.5-2.0` 倍 particle radius，训练位置、颜色、opacity、旋转和椭球 scale。
4. 删除距离任一物理粒子超过 `2.5*radius` 的 Gaussian，防止训练点漂离物理组织。
5. 将最终中心高度裁剪到平面以上 `0.75-15 mm`。

当前最终结果是 1050 个物理粒子和 2490 个视觉 Gaussian。两者来源于同一份第一帧组织观测，但不共用同一批最终点：粒子是体积列填充，Gaussian 是单独训练的表面椭球。

### 6.4 ground 视觉 Gaussian 与物理平面

ground mask 内的每个像素射线直接与拟合平面求交，再 voxel downsample 和采样 2500 个点。颜色取第一帧对应像素；Gaussian 的两个切向 scale 根据局部邻居间距计算，法向 scale 固定约 `1 mm`，旋转使局部 z 对齐平面法向。

ground body 没有 particles，不参与 PBD。物理接触由独立 `ground_plane.json` 提供的无限平面负责。table frame 转换后所有 ground Gaussian 位于 `z=0`，物理平面也是 `[0,0,1,0]`。

## 7. PSM 器械适配

### 7.1 几何与运动分工

当前方案是混合源模型，但两者职责明确：

- `LND.json`：决定每一帧各器械 link 的运动学位姿。
- `data/dvrk_model`：提供完整可见 CAD 表面和 URDF visual link。
- canonical `q=0`：只用于建立 LND link 到 dVRK visual link 的固定变换。
- 图像、depth、mask 和 frame 0：不参与 PSM 位姿参数计算。

只用 LND 的 shaft/skeleton/point features 会把器械显示成线、杆或稀疏特征，无法恢复真实 CAD 表面。只用 dVRK URDF FK 又会错误假设 SuPer q7 与该 xacro 的关节语义、零位和耦合完全一致。当前组合同时保留 LND 运动定义和 dVRK 外观。

### 7.2 dVRK URDF 重建

`scripts/rebuild_super_psm_from_dvrk_xacro.py` 从 xacro 重建：

- 18 个 link、17 个 joint、13 个 STL visual/collision mesh。
- 保留 fixed joint、visual origin、mesh scale 和夹爪结构。
- 用 `psm_mimic_map.json` 描述 q7 到 Warp 14 个非 fixed joint slot 的展开。

旧 q7 到 URDF 的直接展开仍用于创建 Warp articulation、body id 和不可见的结构载体，但不再决定 7 个可见尖端 link 的最终位姿。

### 7.3 hand-eye 与 rectification

`handeye.yaml` 提供 PSM1 的 Rodrigues `rvec` 和毫米单位 `tvec`：

```text
T_raw_camera_psm_base = [Rodrigues(rvec), tvec*0.001]
```

rectified 左目相机由 calibration 的 `R1` 给出：

```text
T_rect_camera_raw_camera = [R1, 0]
T_rect_camera_psm_base =
    T_rect_camera_raw_camera @ T_raw_camera_psm_base
```

左目 rectification 只旋转坐标轴，相机中心不平移。

### 7.4 LND FK

LND 使用 Craig Modified-DH：

```text
T_i = Rx(alpha_i) @ Tx(A_i) @ Rz(theta_i) @ Tz(D_i)
```

- revolute q 加到 `theta`。
- prismatic q 加到 `D`，不能错误加到 `theta`。
- jaw 的两个 link 分别旋转 `+0.5*q_jaw` 和 `-0.5*q_jaw`。
- LND feature 坐标本身已是米，不能再次乘 `0.001`。
- 源注释对应的 `grip_far` 使用 `0.0095 m`，不是错误记录的 `0.09 m`。

对 5458 个 q7 状态生成 `T_lnd_base_lnd_link(q_t)`，并保留原始关节时间戳。

### 7.5 canonical LND 到 dVRK 映射

在 `q=0` 下分别解析 LND 和 dVRK，不使用数据 frame 0。先把 dVRK 所有 link 表达到 `PSM1_psm_base_link`：

```text
T_urdf_base_urdf_link(0) =
    inverse(T_world_urdf_base(0)) @ T_world_urdf_link(0)
```

使用以下 link 原点做加权 Umeyama 刚体配准：base、outer pitch、tool main、wrist、SCA 和 distal shaft。得到固定变换：

```text
T_lnd_base_urdf_base
```

当前对应 link 原点最大 residual 为 `0.0007936 mm`。随后为每个可见 visual link 计算：

```text
T_lnd_link_urdf_link =
    inverse(T_lnd_base_lnd_link(0))
    @ T_lnd_base_urdf_base
    @ T_urdf_base_urdf_link(0)
```

运行时 rectified 相机系中的 visual link 位姿为：

```text
T_rect_camera_visual_link(t) =
    T_rect_camera_psm_base
    @ T_lnd_base_lnd_link(q_t)
    @ T_lnd_link_urdf_link
```

最终 table frame 位姿为：

```text
T_table_visual_link(t) =
    X_table_camera @ T_rect_camera_visual_link(t)
```

`psm_lnd_pose_driver.npz` 保存 5458 帧、7 个可见尖端 link 的 `xyz+xyzw`。报告明确记录：

```text
uses_dataset_reference_frame = false
uses_image_based_correction = false
```

### 7.6 PSM 表面 Gaussian

只采样 `PSM1_tool_main_link` 及其下游尖端子树，不采样 base、yaw/pitch 和画面上方连杆。9 个下游 link 中 7 个带 visual mesh。

每个 mesh 的处理为：

1. 读取 URDF `<visual>` mesh。
2. 在 link-local 坐标中应用 mesh scale 和 visual origin。
3. 按 `30000 samples/m^2` 做表面采样，小 mesh 至少 64 个点。
4. 根据表面法向建立切平面方向，把 Gaussian 设为贴面的薄椭球。
5. 按 link id 绑定到对应 Warp body。

当前资产含 1508 个 Gaussian，其中 `tool_main` 1124 个，其余 6 个 visual link 各 64 个。切向 scale 约 `0.7-3.75 mm`，法向 scale 约 `0.2-0.75 mm`；中心到 CAD 表面的最坏距离约 `0.27 mm`。

### 7.7 Runtime 驱动与碰撞

`super_embodied.py` 加载 URDF 后，把 dVRK articulation 作为 body、mesh 和 Gaussian 绑定载体。每次跳转时间和每个 physics step 后，`apply_psm_lnd_pose()` 都会覆盖 7 个可见尖端 body 的 `body_q`，随后更新 body-bound Gaussian。

PSM 是运动学回放对象，当前关闭：

- PSM self collision。
- PSM 与 tissue 的 shape collision。
- PSM 与 ground plane 的 collision。
- PSM body gravity。

这样可以先验证几何和位姿，不让初始穿透导致 PSM/tissue 飞散。tissue 自身仍保留物理粒子和 ground plane 接触。

## 8. GUI 相机与离线回放

GUI 只启用 `stereo_left`。视频首帧与 rectified PNG 的相位位移约 `0.002 px`，可排除视频编码导致的画面平移。

原 `marsoom.CameraWireframe` 对偏心主点错误构造纹理四角；当前 viewer 使用真实 `K^-1` 计算四个像素角射线，并把它们放到同一个成像平面。`Go To Camera` 按 GUI viewport 对 K 做保持纵横比的 contain 缩放，默认 `camera_go_zoom=0.9`。

这一修复只改变 GUI 视频纹理平面的显示，不改变真实相机投影矩阵、场景坐标或 PSM 位姿。

## 9. 可复现命令

所有命令在同一个 shell 中执行。使用环境绝对路径，不要把 `eg_codex/lib` 当成命令执行：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
export ENV_PREFIX=/Media_HDD/jwshan/conda_envs/eg_codex
export PYTHON="$ENV_PREFIX/bin/python"
export PATH="$ENV_PREFIX/usr/bin:$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$ENV_PREFIX/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV_PREFIX/usr/lib/x86_64-linux-gnu:$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

### 9.1 原始数据提取

完整重新提取：

```bash
"$PYTHON" scripts/extract_grasp5_bag.py
```

若 PNG 已提取，只补 calibration、joints、manifest 和 MP4：

```bash
"$PYTHON" scripts/finalize_grasp5_partial_extract.py
```

### 9.2 Depth

```bash
"$PYTHON" scripts/generate_super_depth_v2.py \
  --iters 64 \
  --count 5 \
  --save-disparity \
  --save-disparity-png
```

可选 mask 内 disparity 过滤实验，必须使用新目录：

```bash
"$PYTHON" scripts/filter_super_disparity.py \
  --disparity-dir data/super/grasp5_native/depth_v2 \
  --output-dir data/super/grasp5_native/depth_v2_mask_filtered \
  --count 5
```

### 9.3 SAM2 第一帧交互分割

先启动虚拟显示器和 noVNC：

```bash
bash scripts/start_display_browser.sh
```

再在另一个 shell 设置上述环境并运行：

```bash
export DISPLAY=:12
"$PYTHON" scripts/segment_super_first_frame_sam2.py --device cuda
```

浏览器访问 `http://192.168.0.6:6082/vnc.html`，依次完成 tissue 和 ground 选点。

### 9.4 重建当前 bodies_v4

```bash
"$PYTHON" scripts/build_super_bodies_from_first_frame.py \
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

### 9.5 转换到 table frame

```bash
"$PYTHON" scripts/convert_super_scene_to_table_frame.py
```

该命令同时更新 `bodies_v5_table`、`table_frame.json`、离线 `cameras.json` 以及运行时 tissue/ground 副本。

### 9.6 重建 PSM

```bash
"$PYTHON" scripts/rebuild_super_psm_from_dvrk_xacro.py
"$PYTHON" scripts/build_super_psm_lnd_intermediates.py
"$PYTHON" scripts/calibrate_psm_lnd_pose_driver.py
"$PYTHON" scripts/build_psm_surface_gaussians.py
"$PYTHON" scripts/validate_psm_lnd_pose_projection.py
```

### 9.7 场景检查和 GUI

```bash
"$PYTHON" scripts/export_scene_ply.py --frame 0
bash scripts/run_demo_browser_12.sh
```

注意：`export_scene_ply.py` 当前仍使用 direct dVRK URDF FK 导出 PSM，只能用它检查 tissue、ground、相机和 table frame；其中的 PSM 不能作为 strict LND pose-driver 的验收结果。严格器械位姿应以 `validate_psm_lnd_pose_projection.py` 的多帧投影图和实际 GUI 为准。

noVNC 地址为 `http://192.168.0.6:6082/vnc.html`。需要更宽视野时：

```bash
bash scripts/run_demo_browser_12.sh --camera-go-zoom 0.8
```

## 10. 验收标准

每一层都应独立通过后再进入下一层：

| 层级 | 必须验证的内容 |
|---|---|
| 2D 数据 | 左右目配对、分辨率、时间戳单调、SAM2 overlay 边界 |
| Depth | disparity 为正、尺度由 `fx*baseline` 决定、平面区域无大块错误匹配 |
| 相机系 3D | tissue/ground 投回第一帧与 mask 重合；桌面允许有透视倾斜 |
| table frame | ground 为 `z=0`、tissue 为 `z>0`、重力为 `-z` |
| tissue physics | 最低粒子中心为一个 radius；静置 100 step 不飞散 |
| tissue visual | Gaussian 贴近表面，不呈规则长方体，不与 particles 完全重合 |
| PSM canonical | LND/dVRK link 原点 residual 保持亚微米量级，当前最大值为 `0.0007936 mm` |
| PSM projection | 按视频 timestamp 取最近 q；多帧检查 shaft、wrist、jaw，不只看 frame 0 |
| GUI | PNG/MP4 无平移，纹理四角按真实 K，场景投影与视频重合 |

PSM 多帧对比图位于 `data/super/psm_robot/lnd_pose_validation/`。左侧旧 direct-URDF 结果只作为历史对照，右侧 strict LND+dVRK 结果才是当前链路。

## 11. 后续精度提升顺序

在保持“严格原数据推导”基线不变的前提下，应按以下顺序量化剩余误差：

1. 在静止或低速帧统计 shaft 中心线和 jaw keypoint 的像素残差，区分固定偏差与状态相关偏差。
2. 估计单一固定图像-关节时间延迟，检查快速运动帧是否明显改善。
3. 核对真实安装工具型号、shaft 长度、jaw 尺寸和 roll 安装零位是否与 dVRK CAD 一致。
4. 使用左右目分别投影，判断 hand-eye 与 stereo calibration 的误差归属。
5. 最后再引入多帧图像绝对修正，并把修正矩阵作为独立 calibration layer 保存，不能改写 LND、hand-eye 或 canonical 源结果。

若后续允许绝对修正，建议只优化少量可解释参数：一个全局 SE(3) hand-eye correction、固定时间延迟、roll zero 和必要的工具安装变换。不要逐帧手工修正，否则无法区分标定误差和运动学误差，也无法迁移到其他序列。
