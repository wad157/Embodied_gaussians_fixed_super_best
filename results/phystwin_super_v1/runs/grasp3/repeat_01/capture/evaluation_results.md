# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 7.596 / 8.098 | 147.974 / 161.746 | 14.689 | 0.1856 | 0.6474 |
| Future: final 20% open-loop | 6.366 / 6.776 | 115.184 / 128.089 | 14.923 | 0.1835 | 0.6467 |
