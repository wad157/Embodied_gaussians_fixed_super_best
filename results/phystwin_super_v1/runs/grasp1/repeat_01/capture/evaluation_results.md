# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 6.102 / 6.302 | 88.010 / 91.722 | 16.326 | 0.2247 | 0.6007 |
| Future: final 20% open-loop | 5.998 / 6.057 | 94.532 / 95.442 | 16.372 | 0.2240 | 0.5875 |
