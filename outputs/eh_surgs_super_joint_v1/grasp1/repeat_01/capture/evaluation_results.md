# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 7.572 / 8.064 | 34.533 / 48.138 | 28.299 | 0.8817 | 0.2872 |
| Future: final 20% open-loop | 7.253 / 7.703 | 33.890 / 37.287 | 27.505 | 0.8461 | 0.3226 |
