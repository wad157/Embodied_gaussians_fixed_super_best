# SUPER grasp1 / grasp3 三次 A/B/C 评测

本页汇总 grasp1 与 grasp3 在 Reconstruction `7:1` 和 Future `80:20` 协议下的三次
独立 A/B/C 评测。表中为 `mean ± sample std`（`ddof=1`）；2D、3D 与 LPIPS 越低越好，
PSNR 与 SSIM 越高越好。

- A：Pure PBD；
- B：PBD + AllTracker/深度轨迹修正；
- C：B + 在线刚度辨识。

grasp1 使用 stride 5、reanchored AllTracker、逐可见物理面查询、线性 confidence
authority，并关闭光流速度写入。Reconstruction f1 覆盖 `0..4204`；Future f2 只在训练
前缀 `0..3363` 提供观测，`3364..4204` 严格开放环。grasp3 Reconstruction f1 覆盖
`0..2061`，Future f2 只在训练前缀 `0..1648` 提供观测。

## grasp1

| Protocol | Method | 2D mean (px) | 3D mean (mm) | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| Reconstruction | A | 31.740 ± 0.581 | 1.850 ± 0.031 | 17.202 ± 0.008 | 0.7731 ± 0.0010 | 0.4877 ± 0.0005 |
| Reconstruction | B | 23.573 ± 0.313 | **1.350 ± 0.016** | 17.199 ± 0.004 | 0.7800 ± 0.0002 | **0.4834 ± 0.0003** |
| Reconstruction | C | **23.552 ± 0.277** | 1.356 ± 0.027 | 17.198 ± 0.004 | **0.7805 ± 0.0006** | 0.4839 ± 0.0002 |
| Future | A | 37.988 ± 1.106 | 2.111 ± 0.069 | 16.994 ± 0.007 | 0.7549 ± 0.0012 | 0.4830 ± 0.0007 |
| Future | B | 34.703 ± 0.000 | 1.971 ± 0.000 | 16.973 ± 0.000 | 0.7564 ± 0.0000 | 0.4819 ± 0.0000 |
| Future | C | **29.976 ± 0.633** | **1.797 ± 0.036** | **17.010 ± 0.005** | **0.7671 ± 0.0008** | **0.4775 ± 0.0007** |

grasp1 Reconstruction 中，B 相对 A 的 2D/3D 平均误差分别降低约 25.7%/27.0%；C
相对 B 没有稳定的整体收益，因此 Reconstruction 推荐 B。Future 中，C 相对 A 的
2D/3D 分别降低约 21.1%/14.9%，且三个渲染指标全部改善，因此 Future 推荐 C。

机器可读数据：
[results/super_grasp1_reconstruction_f1_future_f2_three_runs_v1.csv](results/super_grasp1_reconstruction_f1_future_f2_three_runs_v1.csv)

## grasp3

| Protocol | Method | 2D mean (px) | 3D mean (mm) | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| Reconstruction | A | 31.176 ± 0.317 | 1.978 ± 0.065 | 16.547 ± 0.000 | 0.7435 ± 0.0000 | 0.5339 ± 0.0000 |
| Reconstruction | B | 16.831 ± 0.397 | 1.043 ± 0.014 | **16.606 ± 0.000** | **0.7490 ± 0.0000** | **0.5321 ± 0.0001** |
| Reconstruction | C | **16.717 ± 0.431** | **1.040 ± 0.019** | 16.597 ± 0.004 | 0.7488 ± 0.0001 | 0.5330 ± 0.0004 |
| Future | A | 22.241 ± 0.794 | 1.728 ± 0.026 | 16.841 ± 0.000 | 0.7455 ± 0.0000 | 0.5302 ± 0.0000 |
| Future | B | 22.100 ± 1.450 | 1.228 ± 0.033 | 16.786 ± 0.001 | 0.7436 ± 0.0013 | 0.5362 ± 0.0010 |
| Future | C | **16.326 ± 1.496** | **1.140 ± 0.008** | **16.874 ± 0.010** | **0.7531 ± 0.0006** | **0.5230 ± 0.0006** |

grasp3 Reconstruction 中 B/C 的几何结果接近：C 的 2D/3D 略优，B 的三个渲染指标
略优。Future 中 C 明显优于 A/B，并在五项主指标上均为最佳。

机器可读数据：
[results/super_grasp3_reconstruction_f1_future_f2_three_runs_v1.csv](results/super_grasp3_reconstruction_f1_future_f2_three_runs_v1.csv)

全部运行均正常退出；评分覆盖、GT 哈希、训练/测试观测隔离与渲染分区完整性检查通过。
