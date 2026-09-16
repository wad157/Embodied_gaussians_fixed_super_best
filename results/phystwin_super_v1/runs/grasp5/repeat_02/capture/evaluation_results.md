# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 7.672 / 8.160 | 118.923 / 134.993 | 14.690 | 0.1880 | 0.6589 |
| Future: final 20% open-loop | 7.469 / 7.981 | 131.110 / 139.862 | 14.107 | 0.1723 | 0.6756 |
