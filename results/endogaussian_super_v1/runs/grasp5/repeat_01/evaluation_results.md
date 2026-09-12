# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 12.347 / 13.686 | 26.362 / 38.539 | 28.288 | 0.8836 | 0.2718 |
| Future: final 20% open-loop | 12.089 / 13.378 | 21.836 / 24.139 | 26.787 | 0.8371 | 0.2863 |
