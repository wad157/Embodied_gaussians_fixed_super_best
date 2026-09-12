# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 4.936 / 6.662 | 26.246 / 36.489 | 28.296 | 0.8829 | 0.2695 |
| Future: final 20% open-loop | 4.133 / 5.688 | 18.748 / 20.615 | 27.090 | 0.8511 | 0.2906 |
