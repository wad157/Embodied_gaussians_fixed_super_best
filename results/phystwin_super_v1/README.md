# PhysTwin SUPER results

This package contains all three formal runs (seeds 0, 1, and 2) for each SUPER dataset.
Values are arithmetic mean ± population standard deviation; no run was selected or
discarded.

| Dataset | Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| grasp5 | Reconstruction 7:1 | 6.768 ± 0.978 / 7.110 ± 1.045 | 93.925 ± 22.392 / 105.301 ± 26.288 | 14.839 ± 0.704 | 0.1867 ± 0.0267 | 0.6410 ± 0.0366 |
| grasp5 | Future 80:20 | 8.129 ± 1.585 / 8.581 ± 1.550 | 129.693 ± 42.895 / 140.836 ± 48.639 | 14.409 ± 1.083 | 0.1689 ± 0.0329 | 0.6626 ± 0.0351 |
| grasp3 | Reconstruction 7:1 | 7.445 ± 1.334 / 7.915 ± 1.330 | 138.935 ± 25.131 / 151.676 ± 27.032 | 14.758 ± 0.138 | 0.1852 ± 0.0165 | 0.6461 ± 0.0115 |
| grasp3 | Future 80:20 | 7.656 ± 0.987 / 8.374 ± 1.130 | 137.884 ± 20.628 / 152.402 ± 18.863 | 14.337 ± 0.578 | 0.1788 ± 0.0129 | 0.6561 ± 0.0079 |
| grasp1 | Reconstruction 7:1 | 9.963 ± 5.470 / 10.736 ± 6.291 | 155.182 ± 98.082 / 171.161 ± 115.528 | 15.212 ± 1.574 | 0.2151 ± 0.0150 | 0.6217 ± 0.0315 |
| grasp1 | Future 80:20 | 16.176 ± 14.428 / 16.244 ± 14.450 | 274.053 ± 257.095 / 275.091 ± 257.458 | 14.026 ± 3.315 | 0.1879 ± 0.0525 | 0.6460 ± 0.0844 |

All nine runs have 100% 2D and 3D coverage and pass the ground-truth hash, withheld
observation, complete track schedule, and exact render partition checks.

- `summary_three_repeats/summary.json`: final machine-readable aggregate and all source values.
- `summary_three_repeats/summary.md`: final compact table.
- `summary_two_repeats/`: interim historical snapshot published before seed 2 completed.
- `runs/`: per-run evaluation reports, metadata, protocol audits, and predicted tracks.
- `SHA256SUMS`: hashes for every published artifact.

See the [adapter documentation](../../baselines/phystwin_super/README.md) for the method
boundary, trajectory choice, protocol, and reproduction commands.
