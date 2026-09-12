# EndoGaussian SUPER baseline 测评结果

本目录发布 EndoGaussian baseline 在当前 SUPER `grasp5`、`grasp3` 和
`grasp1` 数据集上的精简、可复核结果。每个数据集均按
`joint_reconstruction_7to1_future_80to20` 协议，以 seed 0、1、2 实际训练
三次。

完整适配方法与测评协议见 [`../../baselines.md`](../../baselines.md)。官方
EndoGaussian 固定到提交 `8d12793838a1595b299df0696c8149c07329e980`。轨迹
导出器参考 Shape of Motion 提交
`579753e1c7ba96f60cd7690e5b835627bd1935e9` 的查询几何属性光栅化设计，
没有运行 Shape of Motion 模型，也没有使用其权重、轨迹、深度或 evaluator。

## 文件内容

- `aggregate.json`、`aggregate.csv`：只含 EndoGaussian 的三次统计结果。
- `per_run_metrics.csv`：9 次实际运行的逐次指标。
- `runs/<dataset>/<repeat>/`：每次运行的原始测评报告、capture 元数据、预测
  2D/3D 轨迹和协议审计。发布时只把 JSON 中的本机绝对路径改成仓库相对路径，
  指标和哈希保持不变。
- `visualizations/`：grasp5 的代表性轨迹视频与重建视频。
- `SHA256SUMS`：除该校验文件自身外，所有发布结果文件的 SHA-256。

训练 checkpoint、双目深度缓存、逐帧渲染图和训练日志没有上传。完整实验目录约
13 GB；本目录保留检查轨迹、核对协议和重新统计报告所需的数值结果。

## 测评协议

- 训练帧：`frame < future_start` 且 `frame % 8 != 0`。
- 重建测评帧：`frame < future_start` 且 `frame % 8 == 0`。
- Future 测评帧：`frame >= future_start`。这些帧的 RGB、深度、mask 和轨迹观测
  不进入训练或状态更新。
- 轨迹在当前冻结的 10 点 GT 上评分，报告 2D 像素误差与相机坐标 3D 毫米
  误差，不做尺度拟合、ICP、刚体对齐或事后校正。
- 渲染采用 0.5 倍分辨率左目图像，排除冻结的 SurgicalSAM2 器械 mask，报告
  PSNR、SSIM 和 LPIPS-Alex。

9 份报告的 GT 哈希匹配、轨迹时间表完整、计分观测留出和渲染划分精确四项检查
均为 `true`，没有 NaN 或无穷数。

## 三次均值 ± 总体标准差

| 数据集 | 分区 | 3D (mm) ↓ | 2D (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|
| grasp5 | Reconstruction 7:1 | 10.784 ± 1.285 | 26.116 ± 0.193 | 28.307 ± 0.015 | 0.8844 ± 0.0006 | 0.2713 ± 0.0008 |
| grasp5 | Future 80:20 | 10.482 ± 1.298 | 21.805 ± 0.035 | 26.700 ± 0.069 | 0.8356 ± 0.0010 | 0.2883 ± 0.0019 |
| grasp3 | Reconstruction 7:1 | 5.959 ± 0.831 | 27.187 ± 0.690 | 28.286 ± 0.010 | 0.8829 ± 0.0000 | 0.2681 ± 0.0010 |
| grasp3 | Future 80:20 | 5.193 ± 0.905 | 19.143 ± 0.376 | 27.029 ± 0.048 | 0.8494 ± 0.0019 | 0.2891 ± 0.0013 |
| grasp1 | Reconstruction 7:1 | 6.567 ± 1.071 | 28.490 ± 0.507 | 28.342 ± 0.023 | 0.8841 ± 0.0003 | 0.2759 ± 0.0012 |
| grasp1 | Future 80:20 | 6.183 ± 0.984 | 35.868 ± 0.421 | 27.401 ± 0.056 | 0.8512 ± 0.0018 | 0.2838 ± 0.0006 |

EndoGaussian 的图像重建质量较好，但没有恢复同等准确的持久材料点轨迹。忠实
baseline 只使用 RGB/深度重建损失，没有轨迹方法所使用的长程对应监督，因此二者并不
矛盾。
