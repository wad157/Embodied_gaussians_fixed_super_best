# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 17.698 / 19.633 | 293.868 / 334.522 | 12.986 | 0.1940 | 0.6662 |
| Future: final 20% open-loop | 36.580 / 36.680 | 637.630 / 639.182 | 9.338 | 0.1137 | 0.7654 |
