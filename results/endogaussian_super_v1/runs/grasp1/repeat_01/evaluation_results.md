# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 5.815 / 6.587 | 28.437 / 43.072 | 28.317 | 0.8837 | 0.2749 |
| Future: final 20% open-loop | 5.552 / 6.118 | 35.760 / 39.095 | 27.326 | 0.8492 | 0.2846 |
