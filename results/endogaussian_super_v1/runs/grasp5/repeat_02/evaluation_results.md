# SUPER joint 80/20 evaluation

Both rows come from the same continuous rollout.

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction: 7:1 in first 80% | 9.199 / 10.805 | 25.890 / 38.291 | 28.324 | 0.8851 | 0.2701 |
| Future: final 20% open-loop | 8.910 / 10.490 | 21.755 / 24.197 | 26.696 | 0.8350 | 0.2876 |
