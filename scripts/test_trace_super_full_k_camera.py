#!/usr/bin/env python3
"""Numerically verify TRACE adapter projection for the off-center SUPER K."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
TRACE_ROOT = Path(
    "/Media_HDD/jwshan/wad/embodied_gaussians_fixed_super_best_sim/baselines/TRACE"
)
sys.path = [
    str(REPO_ROOT / "baselines" / "trace_super"),
    str(TRACE_ROOT),
] + [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_ROOT]
from common import projection_matrix_from_k  # noqa: E402


def main() -> None:
    width, height = 1920, 1080
    intrinsic = np.asarray(
        [
            [1742.7885930026016, 0.0, 860.4195098876953],
            [0.0, 1742.7885930026016, 682.289867401123],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    points = np.asarray(
        [[0.0, 0.0, 0.08], [0.01, -0.01, 0.10], [-0.02, 0.015, 0.12]],
        dtype=np.float64,
    )
    expected = (intrinsic @ points.T).T
    expected = expected[:, :2] / expected[:, 2:3]
    projection = projection_matrix_from_k(
        0.01, 100.0, intrinsic, height, width
    ).numpy()
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float64)), axis=1
    )
    clip = (projection @ homogeneous.T).T
    ndc = clip[:, :2] / clip[:, 3:4]
    actual = np.stack(
        (
            (ndc[:, 0] + 1.0) * width / 2.0,
            (ndc[:, 1] + 1.0) * height / 2.0,
        ),
        axis=1,
    )
    error = np.linalg.norm(actual - expected, axis=1)
    if float(error.max()) > 1.0e-3:
        raise AssertionError("full-K projection error {} px".format(error.max()))
    print("max_projection_error_px={:.9f}".format(float(error.max())))


if __name__ == "__main__":
    main()
