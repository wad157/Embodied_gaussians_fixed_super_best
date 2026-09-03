#!/usr/bin/env python3
"""Summarize held-out phase-4 SUPER residual parameter validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    "pure_pbd",
    "residual_step_0p05mm",
    "residual_step_0p10mm",
    "residual_step_0p20mm",
    "residual_step_0p30mm",
    "residual_step_0p40mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def reduction(baseline: float, value: float) -> float:
    return 100.0 * (baseline - value) / baseline


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows = {}
    scored_frames = None
    ground_truth = None
    for method in METHODS:
        report_path = root / method / "evaluation_results.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["protocol"] != "reconstruction_7to1":
            raise ValueError(f"Unexpected protocol for {method}")
        current_frames = report["point_tracking"]["scored_frames"]
        if scored_frames is None:
            scored_frames = current_frames
            ground_truth = report["ground_truth"]
        elif current_frames != scored_frames or report["ground_truth"] != ground_truth:
            raise ValueError(f"Validation contract mismatch for {method}")
        tracking = report["point_tracking"]
        rows[method] = {
            "2d_mean_px": float(tracking["2d_error_px"]["mean"]),
            "2d_rmse_px": float(tracking["2d_error_px"]["rmse"]),
            "3d_mean_mm": float(tracking["3d_error_mm"]["mean"]),
            "3d_rmse_mm": float(tracking["3d_error_mm"]["rmse"]),
            "tap_delta_avg": float(
                tracking["2d_tap_position_accuracy"]["delta_avg"]
            ),
            "sample_count": int(tracking["2d_error_px"]["count"]),
        }
    baseline = rows["pure_pbd"]
    for method, row in rows.items():
        row["vs_pure_pbd"] = {
            "2d_mean_error_reduction_pct": reduction(
                baseline["2d_mean_px"], row["2d_mean_px"]
            ),
            "3d_mean_error_reduction_pct": reduction(
                baseline["3d_mean_mm"], row["3d_mean_mm"]
            ),
        }
    candidates = METHODS[1:]
    selected = min(
        candidates,
        key=lambda method: (rows[method]["3d_mean_mm"], rows[method]["2d_mean_px"]),
    )
    output = {
        "schema": "super_residual_phase4_tuning_v1",
        "protocol": {
            "split": "reconstruction_7to1",
            "held_out_phase": 4,
            "formal_test_phase": 0,
            "track_only": True,
        },
        "scored_frames": scored_frames,
        "methods": rows,
        "selected_by_validation_3d_then_2d": selected,
        "selected_beats_pure_pbd_on_both_means": bool(
            rows[selected]["2d_mean_px"] < baseline["2d_mean_px"]
            and rows[selected]["3d_mean_mm"] < baseline["3d_mean_mm"]
        ),
    }
    (root / "TUNING_SUMMARY.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SUPER visual residual phase-4 tuning",
        "",
        "Formal test phase 0 was not used for parameter selection.",
        "",
        "| Method | 2D mean / RMSE (px) | 3D mean / RMSE (mm) | vs PBD 2D | vs PBD 3D |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = rows[method]
        effect = row["vs_pure_pbd"]
        lines.append(
            f"| {method} | {row['2d_mean_px']:.3f} / {row['2d_rmse_px']:.3f} "
            f"| {row['3d_mean_mm']:.3f} / {row['3d_rmse_mm']:.3f} "
            f"| {effect['2d_mean_error_reduction_pct']:+.2f}% "
            f"| {effect['3d_mean_error_reduction_pct']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Selected: `{selected}`.",
            f"Beats pure PBD on both means: `{output['selected_beats_pure_pbd_on_both_means']}`.",
        ]
    )
    (root / "TUNING_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
