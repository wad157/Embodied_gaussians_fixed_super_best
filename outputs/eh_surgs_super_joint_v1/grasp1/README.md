# EH-SurGS SUPER grasp1: three-repeat evidence

This directory publishes the lightweight, independently inspectable reports for the three formal `grasp1` runs. The runs use seeds 0/1/2 and the same joint reconstruction 7:1 plus future open-loop 80:20 protocol as the other EH-SurGS SUPER evaluations. The aggregate uses the arithmetic mean and population standard deviation, with no best-run selection.

## Aggregate result

| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Reconstruction 7:1 | 5.799 ± 1.303 / 6.591 ± 1.166 | 34.354 ± 1.437 / 47.783 ± 1.185 | 28.332 ± 0.026 | 0.8821 ± 0.0003 | 0.2866 ± 0.0004 |
| Future 80:20 | 5.547 ± 1.258 / 6.238 ± 1.183 | 33.600 ± 0.332 / 37.387 ± 0.594 | 27.497 ± 0.011 | 0.8460 ± 0.0001 | 0.3244 ± 0.0013 |

The cross-dataset aggregate is available in [`../summary/summary.md`](../summary/summary.md).

## Published files

Each `repeat_0N` directory contains:

- `status.txt`: completion status, dataset, repeat, seed, method, and protocol.
- `protocol_audit.json`: split and trajectory-protocol audit.
- `capture/metadata.json`: capture and evaluation metadata.
- `capture/capture_summary.json`: compact capture summary.
- `capture/evaluation_results.json`: full machine-readable per-frame and aggregate metrics.
- `capture/evaluation_results.md`: concise human-readable metrics.

Checkpoints, rendered frame pairs, videos, source data, and full local artifact manifests are intentionally excluded from this lightweight evidence package.
