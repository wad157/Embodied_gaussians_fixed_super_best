# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 5.805 / 6.811 | 27.896 / 42.886 | 28.373 | 0.8844 | 0.2752 |
| Future: final 20% open-loop | 5.424 / 6.272 | 35.414 / 38.861 | 27.419 | 0.8507 | 0.2836 |
