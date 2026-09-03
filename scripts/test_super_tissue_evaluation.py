#!/usr/bin/env python3
"""CPU-only checks for the SUPER evaluation metric contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from embodied_gaussians.physics_simulator.super_tissue_benchmark import (  # noqa: E402
    SuperTissueBenchmarkRecorder,
    _masked_psnr,
    _masked_ssim,
    _quat_rotate_inverse_wxyz,
    _quat_rotate_wxyz,
)
from prepare_super_tissue_evaluation_gt import strict_depth_at  # noqa: E402
from score_super_tissue_evaluation import distribution, tap_position_accuracy  # noqa: E402


def main() -> None:
    half = np.sqrt(0.5)
    quaternion = np.asarray([[half, 0.0, 0.0, half]], dtype=np.float32)
    vector = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    import torch

    rotated = _quat_rotate_wxyz(
        torch.from_numpy(quaternion), torch.from_numpy(vector)
    )
    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1.0e-6)
    restored = _quat_rotate_inverse_wxyz(
        torch.from_numpy(quaternion), rotated
    )
    assert torch.allclose(restored, torch.from_numpy(vector), atol=1.0e-6)

    depth = np.full((7, 7), np.nan, dtype=np.float32)
    depth[3, 4] = 0.091
    value, radius = strict_depth_at(depth, np.asarray([3.0, 3.0]), 2)
    assert np.isclose(value, 0.091) and radius == 1
    missing, missing_radius = strict_depth_at(depth, np.asarray([0.0, 0.0]), 2)
    assert np.isnan(missing) and missing_radius == -1

    image = np.zeros((32, 32, 3), dtype=np.float32)
    mask = np.ones((32, 32), dtype=bool)
    mse, psnr = _masked_psnr(image, image, mask)
    assert mse == 0.0 and np.isinf(psnr)
    assert np.isclose(_masked_ssim(image, image, mask), 1.0)
    changed = image.copy()
    changed[16, 16] = 1.0
    mse_changed, psnr_changed = _masked_psnr(changed, image, mask)
    assert mse_changed > 0.0 and np.isfinite(psnr_changed)

    stats = distribution(np.asarray([1.0, 2.0, 3.0]))
    assert stats["count"] == 3 and np.isclose(stats["mean"], 2.0)
    tap = tap_position_accuracy(np.asarray([0.5, 3.0, 20.0]))
    assert np.isclose(tap["delta_1px"], 1.0 / 3.0)
    assert 0.0 <= tap["delta_avg"] <= 1.0

    recorder = SuperTissueBenchmarkRecorder.__new__(SuperTissueBenchmarkRecorder)
    recorder.protocol = "reconstruction_7to1"
    recorder.reconstruction_test_phase = 0
    recorder.future_test_start = 1152
    recorder.capture_renders = True
    assert not recorder.observation_allowed(0)
    assert recorder.observation_allowed(1)
    assert recorder.should_render(8)
    recorder.protocol = "future_80to20"
    assert recorder.observation_allowed(1151)
    assert not recorder.observation_allowed(1152)
    assert recorder.should_render(1439)
    print("SUPER tissue evaluation tests: PASS")


if __name__ == "__main__":
    main()
