# TRACE SUPER 基线

本适配器将官方 [vLAR-group/TRACE](https://github.com/vLAR-group/TRACE) 固定到
提交 `a4597585bc0e56c56922abe75be9198eb119c95a`，并接入现有 SUPER
`joint_reconstruction_7to1_future_80to20` 协议。

## 基线边界

- 训练输入仅包含前 80% 中合法的非留出 `stereo-left` 校正 RGB、时间戳、静态相机位姿和
  完整的非中心主点相机矩阵。
- 实采左右视频流并不同步，因此不把相同帧号的右目图像悄悄当成同时刻训练视角。
- 训练不读取 PSM 位姿或控制、深度、分割、测评点、留出 RGB 或未来 RGB；关闭 TRACE
  的可选 FreeGAVE 路径。
- 官方 TRACE checkout 保持未修改。唯一的相机兼容改动是在适配层根据
  `fx`、`fy`、`cx`、`cy` 构造非对称投影矩阵；形变模型、损失、优化器和渲染器均不变。
- 冻结 checkpoint 后才读取 10 个固定查询点和仅供评测使用的器械 mask。轨迹采用查询锚定的
  持久 Gaussian 位移解码器，不运行 Shape-of-Motion，不做尺度拟合、刚体对齐或后验纠正。
- Future 结果完全来自 TRACE 原生动力学，不提供未来观测或外部控制。

因此，这一改动属于相机/数据兼容层，不是对 TRACE 算法的增强。

## 正式结果

三套数据分别使用 seed 0、1、2，共 9 次运行，全部完成。数值为算术均值 ± 总体标准差，
没有挑选或丢弃运行。

| 数据集 | 分区 | 3D 均值 (mm) ↓ | 2D 均值 (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|
| grasp5 | Reconstruction 7:1 | 23.364 ± 1.180 | 20.493 ± 0.447 | 27.623 ± 0.132 | 0.8665 ± 0.0026 | 0.2534 ± 0.0036 |
| grasp5 | Future 80:20 | 23.325 ± 1.482 | 26.399 ± 3.959 | 25.715 ± 0.211 | 0.8165 ± 0.0034 | 0.2911 ± 0.0056 |
| grasp3 | Reconstruction 7:1 | 20.651 ± 0.959 | 19.648 ± 0.699 | 27.586 ± 0.162 | 0.8645 ± 0.0041 | 0.2512 ± 0.0052 |
| grasp3 | Future 80:20 | 21.314 ± 0.885 | 16.414 ± 0.811 | 25.026 ± 0.103 | 0.8376 ± 0.0044 | 0.2869 ± 0.0035 |
| grasp1 | Reconstruction 7:1 | 23.046 ± 4.692 | 25.779 ± 0.621 | 27.391 ± 0.161 | 0.8571 ± 0.0051 | 0.2596 ± 0.0069 |
| grasp1 | Future 80:20 | 23.323 ± 4.577 | 31.960 ± 0.775 | 26.307 ± 0.227 | 0.8348 ± 0.0085 | 0.2733 ± 0.0078 |

机器可读汇总包含每项指标的三次原始数值，见
[机器可读汇总](../../outputs/trace_super_joint_v1/summary/summary.json)。

## 复现方法

将 `TRACE_ROOT` 指向固定提交的干净 TRACE checkout；`TRACE_ENV_PREFIX` 指定兼容官方
环境的 Python 3.7/PyTorch 1.13 环境，`EVAL_ENV_PREFIX` 指定现有 SUPER 评测环境。

运行单个数据集和种子：

```bash
TRACE_ROOT=/path/to/TRACE TRACE_SUPER_GPU_ID=0 \
  bash scripts/run_trace_super_baseline_once.sh \
  grasp5 repeat_01 0 outputs/trace_super_joint_v1/grasp5/repeat_01
```

使用双 GPU 运行全部 9 次实验并自动汇总：

```bash
TRACE_ROOT=/path/to/TRACE \
  bash scripts/run_trace_super_all_repeats_scheduler.sh \
  outputs/trace_super_joint_v1
```
