# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 7.224 / 7.485 | 98.263 / 109.831 | 14.062 | 0.1533 | 0.6741 |
| Future: final 20% open-loop | 10.314 / 10.707 | 181.505 / 200.887 | 13.259 | 0.1269 | 0.6976 |
