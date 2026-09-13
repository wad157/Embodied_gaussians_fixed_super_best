# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 4.475 / 5.212 | 36.018 / 49.023 | 28.362 | 0.8824 | 0.2861 |
| Future: final 20% open-loop | 4.256 / 4.806 | 33.775 / 38.159 | 27.504 | 0.8460 | 0.3255 |
