# EH-SurGS SUPER baseline

该适配器把固定版本 EH-SurGS 接入当前 SUPER `joint_reconstruction_7to1_future_80to20` 协议。`grasp5`、`grasp3`、`grasp1` 均已完成 seed 0、1、2 三次正式运行；按用户指定，仅为 `grasp5` seed 0 生成两个检查视频。

## 方法与版本

- 官方仓库：<https://github.com/IRMVLab/EH-SurGS>
- 固定提交：`73fa04e6f5c21cc1685f728eccb1332e81ce620c`
- 轨迹解码参考 Shape of Motion 提交：`579753e1c7ba96f60cd7690e5b835627bd1935e9`
- 适配版本：`eh_surgs_super_v1_noninstrument_mask`
- 内部尺度：固定 `1000 units/m`，不从 GT 拟合
- 训练配置：官方 3000 次迭代、30,000 个初始 Gaussians

EH-SurGS 用 canonical 3D Gaussians 表示场景，以时间条件形变模型预测 Gaussian 的位置、尺度和旋转。自适应运动层级允许不同图像区域使用不同复杂度的时间运动基函数。

上游唯一补丁位于 `patches/train_camera_intrinsics.patch`，作用是让自适应运动分块读取当前相机的 `focal/cx/cy`，而不是官方 640×512 EndoNeRF 常数。协议审计固定补丁 SHA256；形变网络、损失、优化器和 CUDA rasterizer 均不修改。

## 与现有 SUPER baseline 对齐

- grasp5 固定 1440 帧，Future 从 1152 开始。
- 前 80% 中 `frame % 8 == 0` 为 Reconstruction 留出帧。
- 训练只读取其余前缀左目 RGB、当前 FoundationStereo 深度、冻结非器械 mask 和相机。
- 训练分辨率 640×360；评分分辨率为原图 0.5 倍。
- 最后 20% 不读取 RGB、深度、mask、轨迹或器械控制。
- checkpoint 冻结后才读取 frame 0 的 10 个二维查询点。
- 解码器使用模型自身 alpha/depth 和 Gaussian 位移，不读取 GT depth/3D。
- 2D/3D tracking、PSNR、SSIM、LPIPS-Alex 均由当前 SUPER scorer 计算。

## 运行

创建/检查固定 checkout 和独立环境：

```bash
EH_SURGS_GPU_ID=0 bash scripts/setup_eh_surgs_super_baseline.sh
```

运行 grasp5 seed 0：

```bash
EH_SURGS_GPU_ID=0 bash scripts/run_eh_surgs_super_baseline_once.sh \
  grasp5 repeat_01 0 outputs/eh_surgs_super_joint_v1/grasp5/repeat_01
```

复用上述 grasp5 seed 0，并在两张 GPU 上补齐三数据集各三次：

```bash
bash scripts/run_eh_surgs_super_three_datasets_three_repeats_two_gpus.sh \
  outputs/eh_surgs_super_joint_v1
```

生成两个视频：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/visualize_endogaussian_super_results.py \
  --dataset-key grasp5 \
  --capture outputs/eh_surgs_super_joint_v1/grasp5/repeat_01/capture \
  --output-dir outputs/eh_surgs_super_joint_v1/grasp5/repeat_01/capture/visualizations \
  --method-label EH-SurGS --output-prefix eh_surgs
```

输出为：

- `eh_surgs_tracks_query_anchored_vs_gt.mp4`
- `eh_surgs_reconstruction_vs_target.mp4`

## 三数据集三次结果

以下为算术均值 ± 总体标准差；完整 JSON 同时保存总体标准差、样本标准差和三次原始值。

| 数据集 | 分区 | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| grasp5 | Reconstruction | 6.336 ± 1.021 / 7.266 ± 0.825 | 33.607 ± 0.904 / 43.322 ± 1.311 | 28.301 ± 0.010 | 0.8828 ± 0.0002 | 0.2786 ± 0.0003 |
| grasp5 | Future | 6.189 ± 1.103 / 7.140 ± 0.855 | 22.643 ± 0.573 / 25.198 ± 0.495 | 26.903 ± 0.008 | 0.8285 ± 0.0009 | 0.3254 ± 0.0013 |
| grasp3 | Reconstruction | 5.226 ± 0.765 / 6.790 ± 0.836 | 30.040 ± 1.650 / 41.171 ± 1.406 | 28.277 ± 0.008 | 0.8816 ± 0.0001 | 0.2765 ± 0.0006 |
| grasp3 | Future | 4.463 ± 0.794 / 5.940 ± 0.938 | 17.292 ± 0.969 / 19.071 ± 0.883 | 27.042 ± 0.031 | 0.8375 ± 0.0011 | 0.3301 ± 0.0015 |
| grasp1 | Reconstruction | 5.799 ± 1.303 / 6.591 ± 1.166 | 34.354 ± 1.437 / 47.783 ± 1.185 | 28.332 ± 0.026 | 0.8821 ± 0.0003 | 0.2866 ± 0.0004 |
| grasp1 | Future | 5.547 ± 1.258 / 6.238 ± 1.183 | 33.600 ± 0.332 / 37.387 ± 0.594 | 27.497 ± 0.011 | 0.8460 ± 0.0001 | 0.3244 ± 0.0013 |

9 份运行均通过协议与单次 SHA256 检查。聚合结果位于 `outputs/eh_surgs_super_joint_v1/summary/summary.{md,json}`；checkpoint、逐帧 render、预测轨迹和视频继续保留在本地，不提交 Git。
