# SuPer grasp5 适配排障与收尾计划

本文档用于接续当前 SuPer/grasp5 适配工作。前 1-9 阶段已经基本完成，当前重点不再是继续扩展数据处理链路，而是把第 10 阶段拆成可验证的小步骤，定位真实物理运行中 PSM/tissue 飞散的原因，并最终稳定跑通 demo。

## 目录

1. 当前状态
2. 当前核心问题
3. 总体原则
4. 目标目录与关键文件
5. Phase 10A：资产核对
6. Phase 10B：非 GUI 最小场景验证
7. Phase 10C：PSM 关节与 mimic 排查
8. Phase 10D：PSM 基座、尺度、相机 overlay 验证
9. Phase 10E：tissue/ground 静态验证
10. Phase 10F：逐层打开物理
11. Phase 10G：demo 入口改造
12. Phase 10H：验收标准
13. 快速检查清单
14. 常见飞散原因对照表

## 1. 当前状态

你之前的阶段计划总体是正确的，当前可以认为数据链路和主要资产已经完成：

| # | 阶段 | 状态 | 关键产出 |
|---|------|------|---------|
| 1 | 原始数据提取 | 已完成 | `robots.json`、rectified PNG、calib |
| 2 | 深度估计 | 已完成 | `depth/000000-depth.npy` |
| 3 | SAM2 分割 | 已完成 | `masks/000000-tissue.png`、`ground.png` |
| 4 | 构建场景物体 | 已完成 | `tissue.json`、`ground.json`、`ground_plane.json` |
| 5 | PSM URDF 适配 | 已完成 | 完整 PSM URDF、mimic 派生驱动、夹爪 visual/collision |
| 6 | 组装离线数据集 | 已完成 | `robots.json`、`cameras.json`、左右目 MP4 |
| 7 | 坐标系变换 | 已完成 | 右手系 table frame、`ground_plane=[0,0,1,0]` |
| 8 | PSM 基座位姿 | 已完成 | fin-local LND/handeye 修正、overlay 图重生成 |
| 9 | `super_embodied.py` | 已完成 | SUPER 场景定义、URDF 加载、q7 到 q_full 展开 |
| 10 | demo 入口 + 测试 | 进行中 | 当前在排查真实物理运行中 PSM/tissue 飞散 |

当前的任务不是重做 Phase 1-9，而是把 Phase 10 变成分层验证流程。

## 2. 当前核心问题

现象：真实物理运行中 PSM 或 tissue 出现飞散、跳变、NaN、米级位移，或者一启动就被拉到错误位置。

这类问题一般不是单一原因，而是下面几类因素叠加：

- URDF joint 顺序和 `robots.json` 关节顺序不一致。
- `q7 -> q_full` 展开错误，特别是 jaw/mimic 关节。
- `outer_insertion` 平移关节单位或 offset 错误。
- PSM base fixed transform 和 table/camera frame 不一致。
- tissue 点云尺度或坐标系错误，导致 bbox 在米级或负深度。
- collision geometry 太复杂或和初始状态穿模。
- gravity、PD stiffness/damping、visual forces 同时打开，导致无法判断是谁造成飞散。
- tissue 被当作 dynamic rigid body 后，刚体约束和视觉力/碰撞互相拉扯。

因此排查必须分层进行，不应直接跑完整 GUI demo。

## 3. 总体原则

### 原则 1：先稳定，再真实

第一目标是让 PSM、ground、tissue 在画面中稳定出现，并且 100 帧内无 NaN、无米级位移。不要一开始就追求完整物理接触和 tissue 动态。

### 原则 2：每次只打开一个变量

排查顺序固定为：

```text
PSM kinematic
  -> PSM physics
  -> ground static
  -> tissue static visual
  -> tissue collision
  -> tissue dynamics
  -> visual forces
```

每一步都要记录通过标准。哪一步第一次飞散，就只修那一步。

### 原则 3：tissue 初版默认静态

当前仓库实现偏刚体动力学，SuPer 的 deformable tissue tracking 不应作为第一版 demo 的硬目标。

建议第一版：

- PSM 是 articulated body。
- ground 是 static visual body。
- tissue 先是 static visual body 或 kinematic body。
- visual forces 后开。
- tissue dynamic/collision 后开。

### 原则 4：坐标系先用 overlay 证明

只看 3D viewer 不够。必须用左右目 overlay 验证：

- PSM mesh/keypoints 是否贴真实器械。
- ground/tissue 点云是否落在图像对应区域。
- 相机内参、外参、table frame 是否一致。

## 4. 目标目录与关键文件

推荐最终结构如下：

```text
data/super/grasp5_processed/
├── robots.json
├── cameras.json
├── videos/
│   ├── stereo_left.mp4
│   ├── stereo_left.json
│   ├── stereo_right.mp4
│   └── stereo_right.json
├── bodies/
│   ├── tissue.json
│   ├── ground.json
│   └── ground_plane.json
├── instruments/
│   ├── psm1_lnd_model.json
│   ├── psm1_lnd_motion.json
│   └── urdf/
│       └── psm1.urdf
└── debug/
    ├── overlays/
    ├── bbox_reports/
    └── smoke_tests/
```

示例环境建议如下：

```text
examples/embodied_environments/super_embodied/
├── __init__.py
├── super_embodied.py
├── environment/
│   └── ground_plane.json
├── objects/
│   ├── tissue.json
│   └── ground.json
├── assets/
│   └── robots/
│       └── psm1.urdf
└── sample_demos/
    └── 0/
        ├── robots.json
        ├── cameras.json
        └── videos/
```

demo 入口：

```text
examples/example_embodied_super_offline.py
```

辅助检查脚本：

```text
scripts/check_super_assets.py
scripts/check_super_scene.py
scripts/render_super_overlay.py
scripts/smoke_super_physics.py
```

## 5. Phase 10A：资产核对

### 目标

确认当前 workspace 中所有 Phase 1-9 产物真实存在，并且格式能被当前代码读取。

### 输入

- `robots.json`
- `cameras.json`
- `videos/*.mp4`
- `videos/*.json`
- `tissue.json`
- `ground.json`
- `ground_plane.json`
- PSM URDF
- `super_embodied.py`
- `example_embodied_super_offline.py`

### 检查内容

#### 5.1 检查离线数据集

必须满足：

- `robots.json` 有且只有预期 robot name，例如 `PSM1` 或当前 demo 使用的名字。
- `control` shape 是 `N x 7` 或可明确转换成 q_full。
- `states[i]["q"]` 是 7 维原始 q。
- `control_timestamps` 和 `states_timestamps` 单调递增。
- 视频 timestamp 起点和 robot timestamp 起点在同一时间基准下。

#### 5.2 检查相机

必须满足：

- `cameras.json` 中 `video_path` 和 `metadata_path` 都存在。
- metadata 里的 `K` 是 3x3。
- `resolution` 顺序为 `[width, height]`。
- `timestamps` 数量等于视频帧数或只差极少数可解释帧。
- 左右目内参和 rectified 后图像一致。

#### 5.3 检查 Body JSON

必须满足：

- `Body.model_validate()` 能解析。
- `tissue.json` 至少有 `gaussians`。
- 如果 tissue 作为 dynamic body，则必须有 `particles`。
- `ground.json` 可以只有 `gaussians`，也可以有 `particles`。
- 所有坐标单位是米。
- bbox 尺度在合理范围，例如厘米到几十厘米，不应出现数米。

#### 5.4 检查 URDF

必须满足：

- pure URDF，不依赖 xacro。
- mesh 路径可被 Warp 找到。
- active joint 顺序明确。
- prismatic joint 是 `outer_insertion` 或对应 insertion joint。
- jaw/mimic 展开规则明确。
- first-frame q 在 joint limit 内。

### 输出

```text
data/super/grasp5_processed/debug/asset_report.json
```

建议字段：

```json
{
  "robot_steps": 5458,
  "video_frames_left": 1644,
  "video_frames_right": 1644,
  "camera_resolution": [1920, 1080],
  "tissue_bbox": [[...], [...]],
  "ground_bbox": [[...], [...]],
  "urdf_joint_count": 0,
  "active_joint_names": [],
  "warnings": []
}
```

### 通过标准

- 所有路径存在。
- 所有 JSON 能解析。
- 视频能被 `torchcodec.VideoDecoder` 打开。
- Body bbox 尺度合理。
- q7 第一帧和 URDF joint limits 不冲突。

## 6. Phase 10B：非 GUI 最小场景验证

### 目标

不打开 GUI，不跑完整 physics，只验证 `super_embodied.py` 能构建环境，数据集能加载，第一帧能设置进去。

### 推荐脚本

```text
scripts/check_super_scene.py
```

### 执行逻辑

1. 初始化 Warp。
2. import `build_environment()`。
3. `env = build_environment(add_gaussians=False)`。
4. `DatasetManager(dataset_path, load_frames=False)`。
5. 读取第 0 帧 q。
6. 执行 q7 到 q_full 展开。
7. 设置 articulation q。
8. 只执行 `eval_ik()`，不执行 `step()`。
9. 打印 body/joint/bbox/q 范围。

### 关键点

这一阶段不要：

- 不加载视频帧。
- 不加载 Gaussian。
- 不加 tissue dynamic。
- 不开 visual forces。
- 不循环 step。

### 输出

```text
data/super/grasp5_processed/debug/scene_check.txt
```

包含：

- builder body count
- builder joint count
- model joint count
- q7
- q_full
- body_q finite 检查
- body_q min/max

### 通过标准

- 构建环境不报错。
- `q_full` shape 正确。
- `body_q` 无 NaN/Inf。
- 第一帧设置后所有 body 位姿在合理范围。

## 7. Phase 10C：PSM 关节与 mimic 排查

### 目标

确认真实数据 q7 能稳定驱动 PSM URDF。

### 需要明确的映射

原始 q7：

```text
q[0] outer_yaw
q[1] outer_pitch
q[2] outer_insertion  # prismatic, meters
q[3] outer_roll
q[4] outer_wrist_pitch
q[5] outer_wrist_yaw
q[6] jaw
```

URDF active q_full 可能包含：

```text
yaw
pitch
insertion
roll
wrist_pitch
wrist_yaw
jaw_left
jaw_right
...
```

必须在 `super_embodied.py` 中有唯一函数负责转换，例如：

```python
def expand_psm_q(q7: np.ndarray) -> np.ndarray:
    ...
```

不要在 demo、converter、environment 多处重复写 q mapping。

### 检查项

#### 7.1 insertion 单位

`outer_insertion` 是米。禁止再做除以 1000，除非明确发现输入是毫米。

检查：

```text
min(q[:,2]), median(q[:,2]), max(q[:,2])
```

合理范围通常是厘米到十几厘米。

#### 7.2 jaw/mimic

检查：

- q7 的 jaw 是开合量还是单侧夹爪角。
- URDF 中两个 jaw link 是否需要一正一负。
- mimic multiplier 是否已经写进 URDF。
- 如果 Warp 不支持 mimic 标签，必须在 q_full 展开中手动驱动。

#### 7.3 joint order

不要假设 URDF 文件里的 XML 顺序就是 Warp builder 的 joint order。需要打印 builder 解析后的 joint 名称或通过已知结构核对。

如果当前 builder 不暴露名字，则在 URDF 适配脚本里生成 `joint_order.json`，并让 `expand_psm_q()` 读取同一个定义。

### 通过标准

- 第一帧和任意抽样帧 q_full 都在 limit 内。
- 连续 100 帧 q_full 差分无异常尖峰。
- 只设置 articulation q、不 step 时，PSM body_q 无 NaN。
- 只 PSM、无 tissue、无 visual forces，step 100 帧不飞散。

## 8. Phase 10D：PSM 基座、尺度、相机 overlay 验证

### 目标

证明 PSM mesh/keypoints 在相机图像中和真实器械对齐。

### 为什么必须做

3D viewer 中看起来合理，不代表相机坐标系正确。visual forces 使用相机图像做监督，如果 `X_WC`、base transform 或 OpenCV/Blender 坐标约定错，visual forces 会把物体推向错误方向。

### 推荐脚本

```text
scripts/render_super_overlay.py
```

### 输入

- 第 N 帧 RGB 左图
- `cameras.json`
- `robots.json`
- PSM URDF / FK / builder body transforms
- `LND.json` keypoints，可选

### 输出

```text
data/super/grasp5_processed/debug/overlays/
├── 000000_left_psm_mesh.png
├── 000000_left_keypoints.png
├── 000000_right_psm_mesh.png
└── 000000_right_keypoints.png
```

### 检查标准

- 器械 shaft 方向和真实 shaft 方向一致。
- wrist 和 jaw 在真实器械末端附近。
- 左右目 overlay 同时对齐。
- 不应只靠左目对齐，因为错误 baseline 或右相机外参可能只在右目暴露。

### 常见错误

- `X_WC` 和 `X_CW` 写反。
- OpenCV 相机坐标和项目内部 Blender 风格相机坐标差一个 `diag(1,-1,-1)`。
- table frame transform 应用于点云，但没应用于 PSM base。
- handeye tvec 单位是毫米，却按米使用。
- stereo baseline 单位错误。

### 通过标准

- 第一帧 overlay 对齐。
- 中间帧和末尾帧 overlay 仍然对齐。
- 左右目都对齐。

## 9. Phase 10E：tissue/ground 静态验证

### 目标

先确认 tissue 和 ground 的几何与相机一致，再让它们参与物理。

### tissue 初始策略

第一版建议：

- tissue 不作为 dynamic rigid body。
- tissue 不参与 collision。
- tissue 不受 gravity。
- tissue 不进入 `bodies_affected_by_visual_forces`。
- tissue 只作为 visual body 渲染。

等画面稳定后，再考虑把 tissue 作为 dynamic body。

### ground 初始策略

ground 建议：

- ground plane 用于物理约束。
- ground mesh/gaussians 只用于视觉。
- ground 不参与 visual forces。

### 检查项

#### 9.1 bbox 检查

对 `tissue.json` 和 `ground.json` 打印：

```text
min_xyz
max_xyz
extent_xyz
center_xyz
num_gaussians
num_particles
```

合理情况：

- extent 是厘米到几十厘米。
- z 范围和 table frame 一致。
- ground plane `[0,0,1,0]` 时，ground 点应接近 z=0。

#### 9.2 图像投影检查

把 tissue/ground points 投影到第一帧左图：

- tissue 应落在 tissue mask。
- ground 应落在 ground mask。
- 深度方向不能反。

### 通过标准

- tissue/ground bbox 合理。
- 投影和 mask 对齐。
- 只渲染、不 step physics 时不飞。

## 10. Phase 10F：逐层打开物理

### 目标

定位飞散的最小触发条件。

### 测试矩阵

| 测试 | PSM | ground | tissue | collision | gravity | visual forces | 通过标准 |
|------|-----|--------|--------|-----------|---------|---------------|---------|
| F1 | 开 | 关 | 关 | 关 | 关 | 关 | 100 帧无 NaN |
| F2 | 开 | 静态 | 关 | 关 | 关 | 关 | 100 帧无 NaN |
| F3 | 开 | 静态 | visual only | 关 | 关 | 关 | 100 帧稳定 |
| F4 | 开 | 静态 | visual only | 关 | 开 PSM | 关 | PSM 不掉落/爆炸 |
| F5 | 开 | 静态 | dynamic | 关 | 关 | 关 | tissue 不飞 |
| F6 | 开 | 静态 | dynamic | 开 | 关 | 关 | 无穿模爆炸 |
| F7 | 开 | 静态 | dynamic | 开 | 开 | 关 | 稳定 |
| F8 | 开 | 静态 | dynamic | 开 | 开 | 开 | visual forces 不拉爆 |

### 推荐脚本

```text
scripts/smoke_super_physics.py
```

推荐参数：

```text
--dataset examples/embodied_environments/super_embodied/sample_demos/0
--frames 100
--mode psm-only
--disable-gaussians
--disable-visual-forces
--disable-tissue-dynamics
--disable-collision
--disable-gravity
```

### 每帧记录

记录：

- frame index
- timestamp
- q7
- q_full
- body_q finite
- body_q min/max
- tissue center
- PSM tip position
- max linear velocity
- max angular velocity

输出：

```text
data/super/grasp5_processed/debug/smoke_tests/F1.json
```

### 通过标准

- body_q 全 finite。
- 没有 NaN/Inf。
- body center 不出现米级跳变。
- max velocity 没有异常尖峰。
- 同一测试可重复。

## 11. Phase 10G：demo 入口改造

### 目标

让 `example_embodied_super_offline.py` 支持调试参数，避免每次都进入完整黑箱运行。

### 建议参数

```text
--dataset PATH
--robot-name PSM1
--fps 30
--frame 0
--start-paused
--no-physics
--no-visual-forces
--no-gaussians
--no-tissue-dynamics
--no-collision
--no-gravity
--max-frames 100
```

### 行为建议

#### `--no-physics`

只根据 dataset timestamp 设置 q 和图像帧，不调用 `environment.step()`。

用途：验证回放、相机、overlay、q mapping。

#### `--no-visual-forces`

保留物理 step，但关闭 visual force 更新。

用途：区分物理爆炸和视觉力拉爆。

#### `--no-tissue-dynamics`

tissue 只渲染，不参与 physics。

用途：先确保 surgical scene 可视化稳定。

#### `--frame N`

启动时跳到指定帧，便于复现某个错误。

### 通过标准

- `--no-physics` 模式可稳定播放视频和 PSM pose。
- `--no-visual-forces` 模式可 step，不飞散。
- 默认模式至少能跑前 100 帧。

## 12. Phase 10H：验收标准

最终验收分三层。

### H1：非 GUI 验收

必须通过：

- `check_super_assets.py`
- `check_super_scene.py`
- `smoke_super_physics.py --mode psm-only`
- `smoke_super_physics.py --mode psm-ground-tissue-static`

### H2：overlay 验收

必须通过：

- 第一帧左目 PSM mesh/keypoints 对齐。
- 第一帧右目 PSM mesh/keypoints 对齐。
- 中间帧和末尾帧仍然对齐。
- tissue/ground 投影和 mask 基本一致。

### H3：GUI demo 验收

必须通过：

- GUI 能启动。
- 左目视频能加载。
- PSM、tissue、ground 能显示。
- 播放 100 帧无 NaN、无米级飞散。
- `Reset` 后能回到第一帧。
- 可选：visual forces 打开后仍稳定。

## 13. 快速检查清单

运行前逐项确认：

- `robots.json` 存在。
- `robots.json` 每个 q 是 7 个数。
- q[2] insertion 是米。
- q timestamp 单调递增。
- `cameras.json` 存在。
- `videos/stereo_left.mp4` 存在。
- `videos/stereo_left.json` 的 timestamps 长度和视频帧数一致。
- `resolution` 是 `[width, height]`。
- `K` 是 rectified 后内参。
- `X_WC` 没有写反。
- `ground_plane.json` 是当前 table frame 下的平面。
- `ground_plane=[0,0,1,0]` 时，ground points 接近 z=0。
- `tissue.json` 可被 `Body.model_validate()` 解析。
- tissue bbox 是厘米级，不是米级。
- PSM URDF 是 pure URDF。
- URDF mesh 路径有效。
- URDF active joint 顺序已记录。
- q7 到 q_full 只有一个实现函数。
- jaw mimic 已手动展开或 Warp 确认支持。
- PSM first-frame overlay 已验证。
- 左右目 overlay 都已验证。
- 先跑 `--no-physics`。
- 再跑 `--no-visual-forces`。
- 最后才跑完整 demo。

## 14. 常见飞散原因对照表

| 现象 | 最可能原因 | 优先检查 |
|------|------------|----------|
| 一启动 PSM 就跳走 | q mapping 或 base transform 错 | q_full、joint order、base pose |
| insertion 方向明显错误 | prismatic 轴或单位错 | URDF insertion joint、q[2] 单位 |
| 夹爪异常张开或穿模 | jaw mimic 错 | jaw multiplier、q_full 展开 |
| PSM overlay 左目对齐右目不对 | stereo 外参或右目 K 错 | P2、baseline、right X_WC |
| PSM overlay 整体镜像 | OpenCV/Blender 坐标约定错 | `diag(1,-1,-1)` 转换 |
| tissue 一加入就飞 | tissue dynamic/collision 问题 | 先改 visual only |
| ground 一加入就爆 | ground plane 和点云 frame 不一致 | ground bbox、plane frame |
| step 后逐渐加速飞走 | gravity 或 PD 参数不合适 | gravity_factor、stiffness/damping |
| 打开 visual forces 才飞 | 渲染相机和真实相机不一致 | K、X_WC、overlay、mask |
| body_q 出 NaN | collision 穿模或 joint limit 爆 | collision disable test、limit check |

## 推荐下一步

下一步不应该继续改完整 demo，而是先补齐以下最小脚本：

1. `scripts/check_super_assets.py`
2. `scripts/check_super_scene.py`
3. `scripts/render_super_overlay.py`
4. `scripts/smoke_super_physics.py`

然后按 F1 到 F8 顺序运行。只要记录出“第一个失败的测试编号”，后续修复会非常明确。
