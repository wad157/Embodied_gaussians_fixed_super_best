# PhysTwin SUPER baseline

This adapter evaluates the official [PhysTwin](https://github.com/jianghanxiao/phystwin)
implementation on SUPER with the repository's frozen
`joint_reconstruction_7to1_future_80to20` protocol.

## Method boundary

- Upstream commit: `81c718790a37e5e0102eb77af2c6edd34a9db25f`.
- The spring-mass simulator, differentiable physics optimization, Gaussian appearance
  model, and upstream KNN-LBS implementation remain PhysTwin components.
- The only upstream source patch makes optional GUI imports safe in a headless process;
  the protocol audit rejects any other tracked source modification.
- Trajectories come directly from PhysTwin's persistent moving particles and its
  Gaussian KNN-LBS binding with `K=16`. Shape of Motion is not run. Adding a second
  learned motion representation would no longer measure PhysTwin's native physical
  trajectory.
- The 10 GT query pixels are read only after physics and appearance are frozen. GT
  depth and GT 3D are never used for binding, fitting, scale correction, alignment,
  or post-processing.

## Frozen protocol

- Training observations: `t < future_start` and `t % 8 != 0`.
- Reconstruction: all `t < future_start` with `t % 8 == 0`.
- Future: the final 20%, with no RGB, depth, mask, or track observation.
- Initialization uses the first legal frame, frame 1; frame 0 remains held out.
- Metrics use the left camera at 0.5 scale and exclude the frozen instrument mask.
- Reported metrics are 3D mean/RMSE in millimetres, 2D mean/RMSE in pixels,
  PSNR, SSIM, LPIPS-Alex, and 2D/3D coverage.

The adapter reuses the exact legal-frame FoundationStereo depth cache already audited
for the other SUPER baselines. CoTracker3 scaled-offline observations are generated in
120-frame legal chunks and downsampled to 768 physical points. Formal optimization uses
20 CMA iterations (220 evaluations), 200 Adam iterations, 667 simulator substeps, and
1,000 appearance iterations with 50,000 Gaussians.

## Run

Prepare the pinned checkout and validate the shared environment:

```bash
bash scripts/setup_phystwin_super_baseline.sh
```

Run one seed:

```bash
PHYSTWIN_GPU_ID=0 bash scripts/run_phystwin_super_baseline_once.sh \
  grasp5 repeat_01 0 outputs/phystwin_super_joint_v1/grasp5/repeat_01
```

Run the complete two-GPU queue:

```bash
bash scripts/run_phystwin_super_all_auto.sh
```

Aggregate any explicitly selected repeats without best-run selection:

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/summarize_phystwin_super_baseline.py \
  --campaign outputs/phystwin_super_joint_v1 \
  --output outputs/phystwin_super_joint_v1/summary_two_repeats \
  --repeats 1 2 --baseline-only
```

## Published result

Seeds 0, 1, and 2 are complete on grasp5, grasp3, and grasp1. Their arithmetic mean
and population standard deviation are published in
[`results/phystwin_super_v1/summary_three_repeats`](../../results/phystwin_super_v1/summary_three_repeats/summary.md).
All nine requested runs are included without best-run selection. The earlier
two-repeat summary remains in the result package only as an interim historical snapshot.
