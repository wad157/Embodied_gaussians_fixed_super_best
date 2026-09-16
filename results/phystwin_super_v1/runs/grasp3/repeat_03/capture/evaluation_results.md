# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 8.999 / 9.444 | 164.183 / 178.579 | 14.634 | 0.1648 | 0.6595 |
| Future: final 20% open-loop | 8.762 / 9.152 | 165.103 / 174.065 | 14.538 | 0.1612 | 0.6660 |
