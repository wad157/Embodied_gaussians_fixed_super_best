# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 6.088 / 6.272 | 83.667 / 87.238 | 16.324 | 0.2267 | 0.5981 |
| Future: final 20% open-loop | 5.948 / 5.996 | 89.996 / 90.650 | 16.368 | 0.2261 | 0.5851 |
