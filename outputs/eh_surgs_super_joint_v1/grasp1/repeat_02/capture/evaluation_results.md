# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 5.351 / 6.496 | 32.512 / 46.186 | 28.333 | 0.8820 | 0.2866 |
| Future: final 20% open-loop | 5.132 / 6.203 | 33.135 / 36.713 | 27.482 | 0.8458 | 0.3251 |
