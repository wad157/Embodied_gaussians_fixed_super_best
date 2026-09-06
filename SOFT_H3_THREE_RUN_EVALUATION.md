# 较软参数强 H3：三次重复评估记录

## 固定配置

- `distance=0.01`
- `shape=0.0005`
- `volume=100000`
- 关闭 post-trajectory RGB residual
- 关闭局部 distance/shape 刚度，只使用全局 H3
- H3 单次最大 log 步长 `0.08`
- Reconstruction/Future 累计 log 范围 `±0.60/±0.80`
- Reconstruction 使用 7:1；Future 使用前 80% 训练、后 20% 无视觉更新预测

## Run 1（已完成）

| 协议 | 方法 | 3D Tracking error (mm) ↓ | 2D Tracking error (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|
| Reconstruction | Pure PBD | 1.554715 | 26.909597 | 22.544836 | 0.778188 | 0.473964 |
| Reconstruction | PBD + trajectory | 0.892225 | 11.981952 | 22.369505 | 0.782041 | 0.477438 |
| Reconstruction | PBD + trajectory + strong H3 | **0.843536** | **10.505631** | 22.308129 | 0.780302 | 0.477297 |
| Future | Pure PBD | 1.669120 | 27.585644 | 21.877307 | 0.750583 | 0.477036 |
| Future | PBD + trajectory | 1.365690 | 18.234671 | 21.423114 | 0.743473 | 0.484259 |
| Future | PBD + trajectory + strong H3 | **0.846128** | **11.365065** | 21.740002 | **0.756266** | **0.470620** |

Run 1：`outputs/super_h3_no_post_rgb_soft_strong_h3_20260904_v1`。

## 三次统计协议

当前结果作为 Run 1；Run 2、Run 3 使用完全相同配置和完整六组评估。汇总前必须验证
`CODE_AND_INPUT_SHA256.txt`、`RUN_CONFIGURATION.txt` 完全一致，并要求六个结果均通过
GT hash、完整轨迹计划和渲染分区检查。

每项指标记录三次算术平均：

\[
\bar m=\frac{1}{3}\sum_{r=1}^{3}m_r,
\]

同时报告样本标准差：

\[
s=\sqrt{\frac{1}{2}\sum_{r=1}^{3}(m_r-\bar m)^2}.
\]

最终输出：

- `outputs/super_h3_no_post_rgb_soft_strong_h3_three_run_average_20260904_v1/average_results.md`
- `outputs/super_h3_no_post_rgb_soft_strong_h3_three_run_average_20260904_v1/average_results.json`

## 三次正式结果（已完成）

三次运行状态均为 `reconstruction_status=0`、`future_status=0`，代码/输入哈希与配置
完全一致，GT hash、完整轨迹计划和渲染分区检查全部通过。以下为
`mean ± sample std`。

### Reconstruction 7:1

| 方法 | 3D (mm) ↓ | 2D (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Pure PBD | 1.5699 ± 0.0173 | 27.1018 ± 0.2005 | **22.5333 ± 0.0115** | 0.778051 ± 0.000200 | **0.474427 ± 0.000476** |
| PBD + trajectory | 0.9103 ± 0.0634 | 12.6217 ± 0.7731 | 22.3055 ± 0.0864 | 0.780189 ± 0.001768 | 0.477992 ± 0.000747 |
| PBD + trajectory + strong H3 | **0.8791 ± 0.0361** | **11.8945 ± 1.2145** | 22.3173 ± 0.0295 | **0.780752 ± 0.000480** | 0.477626 ± 0.000376 |

### Future 80:20

| 方法 | 3D (mm) ↓ | 2D (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Pure PBD | 1.7737 ± 0.0916 | 28.8022 ± 1.0568 | **21.8378 ± 0.0349** | 0.749554 ± 0.000979 | 0.478783 ± 0.001678 |
| PBD + trajectory | 1.2836 ± 0.1769 | 17.1502 ± 2.0488 | 21.4792 ± 0.1115 | 0.745853 ± 0.003038 | 0.481951 ± 0.005118 |
| PBD + trajectory + strong H3 | **0.9066 ± 0.1217** | **11.8987 ± 1.1069** | 21.7299 ± 0.1303 | **0.756029 ± 0.004701** | **0.471774 ± 0.004615** |

按均值比较，强 H3 相对仅轨迹在 Reconstruction 的 3D/2D 分别改善约
`3.4%/5.8%`，在 Future 分别改善约 `29.4%/30.6%`；两个协议的 PSNR、SSIM、
LPIPS 均也同时改善。Future 相对 Pure PBD 的 3D/2D 改善约 `48.9%/58.7%`。
