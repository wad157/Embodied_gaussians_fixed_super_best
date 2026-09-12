# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 10.807 / 12.009 | 26.096 / 38.994 | 28.310 | 0.8844 | 0.2718 |
| Future: final 20% open-loop | 10.446 / 11.574 | 21.822 / 24.413 | 26.619 | 0.8347 | 0.2908 |
