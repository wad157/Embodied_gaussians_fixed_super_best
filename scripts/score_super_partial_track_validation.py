#!/usr/bin/env python3
"""Compare incomplete track-only SUPER captures for parameter validation.

This is deliberately separate from the formal scorer: it accepts only a
prefix, reports absolute 2D/3D error on the configured reconstruction holdout
phase, and marks the result as non-formal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--pure-pbd", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def distribution(values: np.ndarray, scale: float = 1.0) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)] * scale
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "maximum": float(np.max(finite)),
    }


def score(capture: Path, ground_truth: dict[str, np.ndarray]) -> dict:
    metadata = json.loads(
        (capture / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata["protocol"] != "reconstruction_7to1":
        raise ValueError("Partial validation requires reconstruction_7to1")
    phase = int(
        metadata["reconstruction_split"]["test_rule"].rsplit(" ", 1)[-1]
    )
    with np.load(capture / "predicted_tracks.npz", allow_pickle=False) as loaded:
        prediction = {name: np.asarray(loaded[name]) for name in loaded.files}
    gt_frame = ground_truth["frame_indices"].astype(np.int32)
    pred_frame = prediction["frame_indices"].astype(np.int32)
    gt_lookup = {int(value): index for index, value in enumerate(gt_frame)}
    pred_lookup = {int(value): index for index, value in enumerate(pred_frame)}
    frames = sorted(
        frame
        for frame in set(gt_lookup) & set(pred_lookup)
        if frame != 0 and frame % 8 == phase
    )
    if not frames:
        raise ValueError(f"No partial holdout frames in {capture}")
    gt_ids = np.asarray([gt_lookup[frame] for frame in frames], dtype=np.int64)
    pred_ids = np.asarray(
        [pred_lookup[frame] for frame in frames], dtype=np.int64
    )
    if np.any(prediction["observation_used"][pred_ids]):
        raise ValueError("A validation frame used a visual observation")
    error_2d = np.linalg.norm(
        prediction["uv"][pred_ids] - ground_truth["uv"][gt_ids], axis=2
    )
    error_2d[~ground_truth["visible"][gt_ids].astype(bool)] = np.nan
    error_3d = np.linalg.norm(
        prediction["xyz_camera_m"][pred_ids]
        - ground_truth["xyz_camera_m"][gt_ids],
        axis=2,
    )
    error_3d[~ground_truth["valid_3d"][gt_ids].astype(bool)] = np.nan
    capture_summary = json.loads(
        (capture / "capture_summary.json").read_text(encoding="utf-8")
    )
    return {
        "frames": frames,
        "2d_error_px": distribution(error_2d),
        "3d_error_mm": distribution(error_3d, 1000.0),
        "runtime": {
            "elapsed_s": float(capture_summary["elapsed_s"]),
            "video_frames_per_s": (
                int(pred_frame[-1] - pred_frame[0] + 1)
                / float(capture_summary["elapsed_s"])
            ),
        },
    }


def main() -> None:
    args = parse_args()
    with np.load(args.ground_truth, allow_pickle=False) as loaded:
        ground_truth = {name: np.asarray(loaded[name]) for name in loaded.files}
    pure = score(args.pure_pbd, ground_truth)
    trajectory = score(args.trajectory, ground_truth)
    if pure["frames"] != trajectory["frames"]:
        raise ValueError("The two methods do not share the same scored frames")
    comparison = {
        "2d_mean_reduction_percent": 100.0
        * (
            pure["2d_error_px"]["mean"]
            - trajectory["2d_error_px"]["mean"]
        )
        / pure["2d_error_px"]["mean"],
        "3d_mean_reduction_percent": 100.0
        * (
            pure["3d_error_mm"]["mean"]
            - trajectory["3d_error_mm"]["mean"]
        )
        / pure["3d_error_mm"]["mean"],
    }
    report = {
        "schema": "super_partial_track_validation_v1",
        "formal_result": False,
        "pure_pbd": pure,
        "trajectory_absolute_joint_alltracker": trajectory,
        "trajectory_vs_pure_pbd": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
