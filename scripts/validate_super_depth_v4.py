#!/usr/bin/env python3
"""Validate timestamp-synchronized SUPER v4 depth and its confidence products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]
CONFIDENCE_KEYS = {
    "foundation_dense_valid",
    "foundation_lr_valid",
    "foundation_lr_error_px",
    "foundation_temporal_span_px",
    "raft_valid",
    "raft_lr_error_px",
    "raft_temporal_span_px",
    "absolute_model_difference_mm",
    "raft_contradiction",
    "high_confidence",
    "confidence_level",
}
REQUIRED_SUFFIXES = (
    "comparison.png",
    "confidence.npz",
    "confidence_level.png",
    "dense_valid_mask.png",
    "depth.npy",
    "depth_high_confidence.npy",
    "depth_lr_consistent.npy",
    "disparity.npy",
    "high_confidence_mask.png",
    "lr_consistent_mask.png",
    "raft_depth.npy",
    "raft_disparity.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("grasp1", "grasp3", "grasp5"))
    parser.add_argument("--native-dir", type=Path, default=None)
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frames", default="0,1,2,3,4")
    parser.add_argument("--min-dense-fraction", type=float, default=0.999)
    parser.add_argument("--max-ground-p95-mm", type=float, default=10.0)
    parser.add_argument("--max-plane-normal-deg", type=float, default=5.0)
    parser.add_argument("--max-plane-offset-mm", type=float, default=10.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mask(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return value > 0


def fit_plane(
    depth: np.ndarray,
    selected: np.ndarray,
    K: np.ndarray,
) -> dict[str, Any]:
    eroded = cv2.erode(
        selected.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    ).astype(bool)
    rows, columns = np.nonzero(eroded & np.isfinite(depth) & (depth > 0.0))
    rows = rows[::4]
    columns = columns[::4]
    z = depth[rows, columns].astype(np.float64)
    points = np.column_stack(
        (
            (columns - K[0, 2]) * z / K[0, 0],
            (rows - K[1, 2]) * z / K[1, 1],
            z,
        )
    )
    kept = np.ones(len(points), dtype=bool)
    for _ in range(3):
        center = points[kept].mean(axis=0)
        _, _, vh = np.linalg.svd(points[kept] - center, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        if normal[2] < 0.0:
            normal = -normal
        offset = -float(normal @ center)
        residual = np.abs(points @ normal + offset)
        threshold = max(float(np.quantile(residual[kept], 0.90)), 0.0005)
        kept = residual <= threshold
    residual_mm = np.abs(points @ normal + offset) * 1000.0
    return {
        "sample_count": int(len(points)),
        "fit_inlier_count": int(kept.sum()),
        "plane_camera": [*normal.tolist(), offset],
        "absolute_residual_mm": {
            "p50": float(np.quantile(residual_mm, 0.50)),
            "p90": float(np.quantile(residual_mm, 0.90)),
            "p95": float(np.quantile(residual_mm, 0.95)),
            "p99": float(np.quantile(residual_mm, 0.99)),
        },
    }


def main() -> None:
    args = parse_args()
    native = (args.native_dir or REPO / "data/super" / f"{args.dataset}_native").resolve()
    depth_dir = (
        args.depth_dir or native / "depth_v4_foundation_dense_timestamped"
    ).resolve()
    output = args.output or depth_dir / "depth_validation_report.json"
    frames = [int(value) for value in args.frames.split(",") if value.strip()]
    summary = read_json(depth_dir / "depth_generation_summary.json")
    calibration = read_json(native / "calib_rectified.json")
    K = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    tissue = mask(native / "masks/000000-tissue.png")
    ground = mask(native / "masks/000000-ground.png")
    expected_shape = tissue.shape
    min_depth, max_depth = np.asarray(
        summary["parameters"]["depth_range_mm"], dtype=np.float64
    ) / 1000.0
    summary_frames = {int(item["left_frame"]): item for item in summary["frames"]}

    frame_reports: list[dict[str, Any]] = []
    gates: dict[str, bool] = {}
    planes: list[np.ndarray] = []
    for frame in frames:
        prefix = depth_dir / f"{frame:06d}"
        files = {suffix: Path(f"{prefix}-{suffix}") for suffix in REQUIRED_SUFFIXES}
        files_complete = all(path.is_file() for path in files.values())
        depth = np.load(files["depth.npy"])
        disparity = np.load(files["disparity.npy"])
        with np.load(files["confidence.npz"], allow_pickle=False) as confidence:
            keys = set(confidence.files)
            arrays_have_shape = all(
                confidence[key].shape == expected_shape for key in CONFIDENCE_KEYS & keys
            )
            dense_mask = confidence["foundation_dense_valid"].astype(bool)
            lr_mask = confidence["foundation_lr_valid"].astype(bool)
            raft_mask = confidence["raft_valid"].astype(bool)
            high_mask = confidence["high_confidence"].astype(bool)
            levels = confidence["confidence_level"]
            confidence_consistent = bool(
                np.array_equal(dense_mask, np.isfinite(depth))
                and np.all(lr_mask <= dense_mask)
                and np.all(high_mask <= lr_mask)
                and np.all(high_mask <= raft_mask)
                and np.all((levels >= 0) & (levels <= 3))
            )
        valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
        finite_fraction = float(np.mean(np.isfinite(depth)))
        tissue_coverage = float(np.mean(valid[tissue]))
        ground_coverage = float(np.mean(valid[ground]))
        plane_report = fit_plane(depth, ground, K)
        plane = np.asarray(plane_report["plane_camera"], dtype=np.float64)
        planes.append(plane)
        summary_frame = summary_frames.get(frame, {})
        mode = summary_frame.get("timestamp_sampling_mode")
        before = summary_frame.get("right_before_frame")
        after = summary_frame.get("right_after_frame")
        alpha = summary_frame.get("temporal_interpolation_alpha")
        timestamp_valid = bool(
            (mode == "nearest_boundary_zero_order_hold" and before == after and alpha == 0.0)
            or (
                mode == "bracketed_linear_disparity_interpolation"
                and isinstance(before, int)
                and isinstance(after, int)
                and before < after
                and isinstance(alpha, (int, float))
                and 0.0 <= float(alpha) <= 1.0
            )
        )
        frame_report = {
            "frame": frame,
            "files_complete": files_complete,
            "shape": list(depth.shape),
            "dtype": str(depth.dtype),
            "disparity_shape": list(disparity.shape),
            "finite_fraction": finite_fraction,
            "physical_range_valid_fraction": float(np.mean(valid)),
            "tissue_dense_coverage": tissue_coverage,
            "ground_dense_coverage": ground_coverage,
            "confidence_keys_complete": keys == CONFIDENCE_KEYS,
            "confidence_arrays_have_expected_shape": arrays_have_shape,
            "confidence_semantics_consistent": confidence_consistent,
            "timestamp_sampling_mode": mode,
            "timestamp_sampling_valid": timestamp_valid,
            "ground_plane": plane_report,
        }
        frame_reports.append(frame_report)
        gates[f"frame_{frame:06d}_files"] = files_complete
        gates[f"frame_{frame:06d}_shape"] = bool(
            depth.shape == expected_shape and disparity.shape == expected_shape
        )
        gates[f"frame_{frame:06d}_dense"] = bool(
            finite_fraction >= args.min_dense_fraction
            and tissue_coverage >= args.min_dense_fraction
            and ground_coverage >= args.min_dense_fraction
        )
        gates[f"frame_{frame:06d}_confidence"] = bool(
            keys == CONFIDENCE_KEYS and arrays_have_shape and confidence_consistent
        )
        gates[f"frame_{frame:06d}_timestamp"] = timestamp_valid
        gates[f"frame_{frame:06d}_ground_residual"] = bool(
            plane_report["absolute_residual_mm"]["p95"] <= args.max_ground_p95_mm
        )

    reference = planes[0]
    stability = []
    for frame, plane in zip(frames, planes, strict=True):
        cosine = float(np.clip(reference[:3] @ plane[:3], -1.0, 1.0))
        normal_degrees = float(np.degrees(np.arccos(cosine)))
        offset_mm = float((plane[3] - reference[3]) * 1000.0)
        stability.append(
            {
                "frame": frame,
                "normal_difference_degrees": normal_degrees,
                "offset_difference_mm": offset_mm,
            }
        )
    gates["all_requested_frames_in_summary"] = set(frames) == set(summary_frames)
    gates["ground_plane_normal_stability"] = all(
        item["normal_difference_degrees"] <= args.max_plane_normal_deg
        for item in stability
    )
    gates["ground_plane_offset_stability"] = all(
        abs(item["offset_difference_mm"]) <= args.max_plane_offset_mm
        for item in stability
    )
    report = {
        "dataset": args.dataset,
        "depth_dir": str(depth_dir),
        "frames": frame_reports,
        "ground_plane_stability_relative_to_frame_0": stability,
        "thresholds": {
            "min_dense_fraction": args.min_dense_fraction,
            "max_ground_p95_mm": args.max_ground_p95_mm,
            "max_plane_normal_deg": args.max_plane_normal_deg,
            "max_plane_offset_mm": args.max_plane_offset_mm,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gates": gates, "all_gates_passed": report["all_gates_passed"]}, indent=2))
    if not report["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
