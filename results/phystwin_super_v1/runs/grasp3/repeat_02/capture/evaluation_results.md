# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 5.741 / 6.202 | 104.649 / 114.702 | 14.952 | 0.2051 | 0.6313 |
| Future: final 20% open-loop | 7.841 / 9.192 | 133.365 / 155.051 | 13.549 | 0.1918 | 0.6555 |
