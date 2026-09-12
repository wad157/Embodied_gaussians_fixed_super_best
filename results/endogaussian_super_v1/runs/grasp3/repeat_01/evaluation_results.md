# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 6.972 / 9.509 | 27.429 / 38.087 | 28.292 | 0.8830 | 0.2675 |
| Future: final 20% open-loop | 6.344 / 8.630 | 19.648 / 20.597 | 27.023 | 0.8503 | 0.2874 |
