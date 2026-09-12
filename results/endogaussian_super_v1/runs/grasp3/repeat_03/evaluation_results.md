# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 5.969 / 7.862 | 27.885 / 39.240 | 28.272 | 0.8829 | 0.2673 |
| Future: final 20% open-loop | 5.101 / 6.902 | 19.031 / 20.427 | 26.974 | 0.8467 | 0.2894 |
