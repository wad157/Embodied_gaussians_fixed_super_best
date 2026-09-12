# Baseline 适配与测评

本文固定 `Embodied_gaussians_fixed_super_best` 上的 baseline 适配口径。当前第一项是 EndoGaussian。所有正式结果必须先通过 `scripts/audit_endogaussian_super_protocol.py`；旧 SUPER 真值、旧测评点、旧留出相位以及主方法的 AllTracker 中间结果均不得作为 EndoGaussian 输入。

当前适配版本为 `endogaussian_super_v2_noninstrument_mask`。早期组织专用 mask 的调试结果不进入正式汇总。

## EndoGaussian 版本

- 官方仓库：<https://github.com/CUHK-AIM-Group/EndoGaussian>
- 固定提交：`8d12793838a1595b299df0696c8149c07329e980`
- 2026-09-12 复核最新官方 README：仍要求 Python 3.7、PyTorch 1.13.1 和 CUDA 11.7，并以官方 `train.py` 训练、`render.py`/`metrics.py` 渲染测评。
- 环境：`/Media_HDD/jwshan/conda_envs/endogaussian_baseline`，与 `eg_codex` 同级。
- 上游的 `train.py`、Gaussian renderer、deformation network、loss、optimizer 和 CUDA rasterizer保持原样。适配代码仅替换数据接口、帧协议与导出接口。
- 参数采用官方 EndoNeRF `pulling` 配置：coarse 1000 次、fine 3000 次、30,000 个初始高斯、`binocular` 米制深度监督。

`binocular` 在 EndoGaussian 官方代码中表示使用双目获得的米制深度；官方 EndoNeRF/SCARED loader 每个时刻仍只向模型提供左目图像。因此 SUPER 也按单左目时间序列训练，不把右目图像当作第二个同时间训练视角。右目只参与独立立体深度估计。

## 当前 SUPER 正式数据

| 数据集 | 固定帧范围 | Future 起点 | 当前 10 点 GT | 相机文件 |
|---|---:|---:|---|---|
| grasp5 | 0–1439 | 1152 | `data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz` | `data/super/grasp5_offline_demo/cameras.json` |
| grasp3 | 0–2061 | 1649 | `data/super/grasp3_native/evaluation_v1/manual_tissue_tracks_10_v2/ground_truth_2d3d_v1.npz` | `data/super/grasp3_native/bodies_v9_dense_0p5mm_rigid_tissue/cameras_table.json` |
| grasp1 | 0–4204 | 3364 | `data/super/grasp1_native/evaluation_v2/manual_tissue_tracks_10_no_exclusion/ground_truth_2d3d_v1.npz` | `data/super/grasp1_native/bodies_v9_dense_0p5mm_rigid_tissue/cameras_table.json` |

三套 GT 都含当前冻结的 10 个点、稀疏标注帧、`K_left_rect`、`X_world_camera_opencv` 和各自的 `future_test_start_frame`。适配器会验证 GT 相机与训练相机完全一致。grasp3 不能退回 `manual_tissue_tracks_10`，grasp1 不能退回 `evaluation_v1`。

正式联合协议为 `joint_reconstruction_7to1_future_80to20`：

- 训练帧：`t < future_start` 且 `t % 8 != 0`。
- 重建测评帧：`t < future_start` 且 `t % 8 == 0`。
- Future 测评帧：`t >= future_start`；不读取这些帧的 RGB、深度或 mask。
- 轨迹只在当前 GT 的稀疏标注帧评分；重建图像在所有 `8k` 留出帧评分，Future 图像在最后 20% 每一帧评分。
- 图像指标沿用当前 SUPER scorer：左目、0.5 倍分辨率、排除冻结 SurgicalSAM2 器械 mask，报告 PSNR/SSIM/LPIPS-Alex。目标图像通过与主评测器 `OfflineCamera` 相同的 TorchCodec CUDA 路径从离线 MP4 解码；CPU 解码的色域转换不同，不能混用。
- 轨迹指标报告 2D 像素误差和相机坐标 3D 毫米误差，不做尺度拟合、刚体对齐或后验校正。

## 输入适配

训练使用以下可观测量：

1. `data/super/<dataset>_native/rgb/%06d-left.png` 的合法训练帧。
2. 当前冻结 SurgicalSAM2 器械 mask 的左目非器械区域；这与官方 EndoGaussian loader 的 mask 语义一致，背景和组织都参与重建监督。
3. `calib_rectified.json` 与当前相机文件。
4. 由同一训练帧左右 RGB 独立计算的 FoundationStereo 深度。

本地 `depth_v4_foundation_dense_timestamped` 实际每个数据集只有 5 帧，是工程抽样，不能冒充全序列深度。GT 目录下为人工轨迹标注生成的稀疏 stereo depth 也不进入训练。AllTracker 的 `observations.npz` 属于当前主方法中间结果，EndoGaussian 不读取。`visual_force_masks_v1` 的组织专用 mask 也不用于 EndoGaussian 训练，因为官方方法只排除器械；把训练限制到组织区域会使统一指标仍然评分的背景完全没有监督。

正式深度预处理由 `scripts/prepare_endogaussian_super_depth.py` 完成。它只枚举合法训练帧，把左右图降到 640×360，按时间戳选最近右目帧，以固定 FoundationStereo checkpoint 计算稠密 disparity，再用标定的焦距和 baseline 转为米。缓存只保存一个 float32 depth map，三套数据约 5 GB；全分辨率旧生成器会保存多种中间数组，在当前仅剩约 118 GB 的磁盘上不可行。审计器要求缓存中的帧集合与合法训练帧集合严格相等，出现任一留出或 Future 帧就拒绝运行。

SUPER 原图为 1920×1080，而官方 EndoNeRF 输入约为 640×512。训练固定降采样 3 倍至 640×360，并同步缩放内参。相机对象按随机训练迭代惰性加载，避免 grasp1 的数千张图同时驻留内存；初始 30,000 个高斯从合法训练时段均匀选取的 64 个深度视图采样。模型时间严格采用官方 loader 的 `frame_index / full_sequence_frame_count`。

官方 EndoGaussian `Camera` 只用 FoV 构造中心主点投影，而 SUPER 左目的主点不在图像中心。适配层根据完整 `K` 构造非对称 3DGS 投影矩阵，并同时用于训练、重建渲染和轨迹解码；否则会产生约 100–140 像素量级的固定几何偏移。该处理只补全相机数据接口，不修改上游 renderer 或 deformation 模型。

SUPER 的米制场景距离小于上游 rasterizer 的常用裁剪尺度，所以模型内部固定使用 `1000 units/m`，导出时严格除以 1000。这个换算对所有数据集和重复实验固定，不从 GT 拟合。

## 2D/3D 轨迹导出

EndoGaussian 原代码只提供重建图像，没有点轨迹接口。轨迹读取方式采用 Shape of Motion 的“在查询时刻的几何上渲染目标时刻属性”思路：

- Shape of Motion 仓库：<https://github.com/vye16/shape-of-motion>
- 固定设计参考提交：`579753e1c7ba96f60cd7690e5b835627bd1935e9`
- 只借用轨迹解码设计，不使用它的模型权重、预处理数据、相机、深度、划分或 evaluator。

具体解码器为 `som_query_anchored_displacement_v2`：

1. EndoGaussian checkpoint 完全冻结后，才读取 GT 中 frame 0 的 10 个二维查询像素。
2. 在 frame 0 渲染 EndoGaussian 自己的 alpha 和 premultiplied depth，对查询点双线性采样并做 alpha 归一化，以此反投影查询起点。GT 深度和 GT 3D 从不进入解码器。
3. 对每个待评估时刻，计算每个高斯的“目标时刻中心减查询时刻中心”，把该 3D 位移作为属性通过固定的查询时刻高斯几何渲染。
4. 在同一查询像素采样位移并加到查询起点，得到世界坐标轨迹；再用冻结相机投影得到 2D 与相机坐标 3D。
5. 强制检查 frame 0 最大重投影误差小于 `1e-3 px`，避免出现肉眼可见的初始点偏移。

Future 段直接查询 EndoGaussian 原生时空 deformation field 的未来时间。该模型没有物理约束，未来高斯可能继续移动、冻结或回到旧位置；这是 baseline 原始外推行为，结果中保留，不增加纠正模块。

## 正式结果（2026-09-12）

`grasp5`、`grasp3`、`grasp1` 均已使用 seed 0、1、2 完成三次实际训练，共 9 次。
以下为均值 ± 总体标准差：

| 数据集 | 分区 | 方法 | 3D mean (mm) ↓ | 2D mean (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| grasp5 | Reconstruction | EndoGaussian | 10.784 ± 1.285 | 26.116 ± 0.193 | 28.307 ± 0.015 | 0.8844 ± 0.0006 | 0.2713 ± 0.0008 |
| grasp5 | Reconstruction | 当前方法 | 0.697 ± 0.026 | 9.502 ± 0.318 | 22.629 ± 0.010 | 0.7907 ± 0.0003 | 0.4743 ± 0.0008 |
| grasp5 | Future | EndoGaussian | 10.482 ± 1.298 | 21.805 ± 0.035 | 26.700 ± 0.069 | 0.8356 ± 0.0010 | 0.2883 ± 0.0019 |
| grasp5 | Future | 当前方法 | 1.173 ± 0.054 | 14.581 ± 1.114 | 21.557 ± 0.087 | 0.7469 ± 0.0016 | 0.4787 ± 0.0048 |
| grasp3 | Reconstruction | EndoGaussian | 5.959 ± 0.831 | 27.187 ± 0.690 | 28.286 ± 0.010 | 0.8829 ± 0.0000 | 0.2681 ± 0.0010 |
| grasp3 | Reconstruction | 当前方法 | 0.915 ± 0.013 | 12.635 ± 0.087 | 16.609 ± 0.004 | 0.7556 ± 0.0002 | 0.5027 ± 0.0003 |
| grasp3 | Future | EndoGaussian | 5.193 ± 0.905 | 19.143 ± 0.376 | 27.029 ± 0.048 | 0.8494 ± 0.0019 | 0.2891 ± 0.0013 |
| grasp3 | Future | 当前方法 | 1.386 ± 0.069 | 20.331 ± 0.347 | 16.762 ± 0.002 | 0.7330 ± 0.0006 | 0.5050 ± 0.0007 |
| grasp1 | Reconstruction | EndoGaussian | 6.567 ± 1.071 | 28.490 ± 0.507 | 28.342 ± 0.023 | 0.8841 ± 0.0003 | 0.2759 ± 0.0012 |
| grasp1 | Reconstruction | 当前方法 | 1.138 ± 0.000 | 19.657 ± 0.000 | 17.252 ± 0.000 | 0.7845 ± 0.0000 | 0.4834 ± 0.0000 |
| grasp1 | Future | EndoGaussian | 6.183 ± 0.984 | 35.868 ± 0.421 | 27.401 ± 0.056 | 0.8512 ± 0.0018 | 0.2838 ± 0.0006 |
| grasp1 | Future | 当前方法 | 1.937 ± 0.000 | 31.745 ± 0.000 | 16.997 ± 0.000 | 0.7628 ± 0.0000 | 0.4805 ± 0.0000 |

EndoGaussian 在三套数据和两个分区上都取得更好的图像指标。当前方法在全部 3D
轨迹指标、全部 Reconstruction 2D 指标以及 grasp5/grasp1 Future 2D 指标上更好。
唯一例外是 grasp3 Future 2D：EndoGaussian 为 `19.143 px`，当前方法为
`20.331 px`；但同一分区的 3D 仍是当前方法的 `1.386 mm` 显著优于
EndoGaussian 的 `5.193 mm`。

重建视频准确而测评点移动较小并不矛盾。检查 grasp5 repeat_01 后发现 EndoGaussian
预测点相对查询帧的平均运动只有 `5.029 px / 0.321 mm`，GT 为
`26.819 px / 1.573 mm`。单高斯、邻域高斯和局部 oracle 检查都不能恢复缺失的运动，
说明主要误差来自 checkpoint 学到的持久材料运动偏小，而不是位移属性光栅化把运动
平均掉。渲染质量同时来自高斯中心、尺度、旋转和可见性变化；因此 RGB 重建正确不能
直接证明持久材料对应正确。作为待改进 baseline，这一原生行为原样保留。

9 份正式报告的 `ground_truth_hash_matches_capture`、`complete_track_schedule`、
`scored_observations_withheld`、`render_partition_exact` 均为 `true`，没有 NaN 或
无穷数。精简发布包位于 `results/endogaussian_super_v1/`，包含逐次报告、预测轨迹、
协议审计、三次汇总和代表性视频，不包含约 13 GB 的 checkpoint、深度缓存和逐帧图像。

## 运行方法

初始化上游 checkout，并复用同级 conda 环境：

```bash
bash scripts/setup_endogaussian_super_baseline.sh
```

单独准备某个数据集的合法训练深度：

```bash
CUDA_VISIBLE_DEVICES=0 /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/prepare_endogaussian_super_depth.py \
  --dataset-key grasp5 --device cuda:0
```

正式单次训练、导出和统一评分：

```bash
ENDOGAUSSIAN_GPU_ID=0 bash scripts/run_endogaussian_super_baseline_once.sh \
  grasp5 repeat_01 0 \
  outputs/endogaussian_super_joint_v1/grasp5/repeat_01
```

grasp3、grasp1 分别替换数据集键。三次重复固定使用 seed 0、1、2 和 `repeat_01`、`repeat_02`、`repeat_03`。最终分数位于每次输出的 `capture/evaluation_results.json` 与 `capture/evaluation_results.md`。

生成与 SIM baseline 对应的轨迹和重建检查视频：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/visualize_endogaussian_super_results.py \
  --dataset-key grasp5 \
  --capture outputs/endogaussian_super_joint_v1/grasp5/repeat_01/capture \
  --output-dir outputs/endogaussian_super_joint_v1/grasp5/repeat_01/capture/visualizations
```

grasp5、grasp3、grasp1 各跑三次并和 2026-09-11 最新方法正式结果汇总比较：

```bash
ENDOGAUSSIAN_GPU_ID=0 bash \
  scripts/run_endogaussian_super_three_datasets_three_repeats.sh \
  outputs/endogaussian_super_joint_v1
```

调度器会跳过已有 `status=complete` 的重复实验，拒绝自动覆盖或忽略不完整目录。汇总写入 `outputs/endogaussian_super_joint_v1/summary/summary.{json,md}`。

可在正式运行前执行审计；默认要求深度缓存完整：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/audit_endogaussian_super_protocol.py \
  --dataset-key grasp5 \
  --output outputs/endogaussian_super_grasp5_protocol_audit.json
```

`--allow-incomplete-depth` 只用于检查数据路径和协议代码，带该参数的输出不能作为正式结果。

## 实现文件

- `baselines/endogaussian_super/protocol.py`：冻结数据路径、帧划分、坐标与时间定义。
- `baselines/endogaussian_super/common.py`：SUPER 数据、惰性相机和点云初始化适配。
- `baselines/endogaussian_super/train_super.py`：调用未修改的官方训练循环。
- `baselines/endogaussian_super/export_super.py`：轨迹解码、重建渲染与 SUPER capture 输出。
- `scripts/prepare_endogaussian_super_depth.py`：无 GT 的紧凑 FoundationStereo 深度缓存。
- `scripts/prepare_endogaussian_super_render_metrics.py`：用当前 SUPER 的 TorchCodec CUDA 图像路径与冻结器械 mask 生成统一渲染指标。
- `scripts/visualize_endogaussian_super_results.py`：生成稀疏 10 点轨迹对比和全部测评帧重建对比视频。
- `scripts/audit_endogaussian_super_protocol.py`：版本、资产、相机、10 点、划分与泄漏审计。
- `scripts/run_endogaussian_super_baseline_once.sh`：GPU 上的单次端到端运行。
- `scripts/run_endogaussian_super_three_datasets_three_repeats.sh`：三数据集各三次的串行正式调度。
- `scripts/summarize_endogaussian_super_baseline.py`：与最新方法三次实际运行结果统一比较。

版本和运行时记录在 `baselines/endogaussian_super/UPSTREAM_VERSIONS.json`。正式结果发布时只提交数值结果、协议审计和必要说明，不提交 checkpoint、深度缓存或逐帧图像。
