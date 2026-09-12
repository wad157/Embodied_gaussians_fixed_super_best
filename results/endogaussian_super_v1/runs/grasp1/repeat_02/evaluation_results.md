# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 8.081 / 9.170 | 29.135 / 44.149 | 28.335 | 0.8841 | 0.2775 |
| Future: final 20% open-loop | 7.572 / 8.600 | 36.428 / 39.698 | 27.459 | 0.8535 | 0.2832 |
