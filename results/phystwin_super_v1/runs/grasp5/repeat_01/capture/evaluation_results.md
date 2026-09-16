# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 5.409 / 5.684 | 64.590 / 71.081 | 15.767 | 0.2187 | 0.5900 |
| Future: final 20% open-loop | 6.603 / 7.056 | 76.463 / 81.758 | 15.861 | 0.2074 | 0.6146 |
