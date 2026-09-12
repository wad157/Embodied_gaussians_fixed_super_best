# EH-SurGS SUPER baseline

该适配器把固定版本 EH-SurGS 接入当前 SUPER `joint_reconstruction_7to1_future_80to20` 协议。目前按用户指定只完成 `grasp5` seed 0，并生成两个检查视频；未运行 grasp3、grasp1 或另外两个 seed。

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

## grasp5 seed 0 结果

| 分区 | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction 7:1 | 5.898 / 7.394 | 34.877 / 44.778 | 28.310 | 0.8827 | 0.2782 |
| Future 80:20 | 5.753 / 7.295 | 22.315 / 24.952 | 26.914 | 0.8272 | 0.3238 |

结果目录包含 `protocol_audit.json`、`metadata.json`、`predicted_tracks.npz`、`render_metrics_partial.json`、`evaluation_results.json` 和 SHA256 清单。它们均保留在本地 `outputs/`；Git 只提交适配代码和说明。
