#!/usr/bin/env python3
"""Summarize prefix-only open-loop residual validation before formal test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    "pure_pbd",
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
    for method in METHODS:
        report = json.loads(
            (root / method / "evaluation_results.json").read_text(
                encoding="utf-8"
            )
        )
        if report["protocol"] != "future_80to20":
            raise ValueError(f"Unexpected protocol for {method}")
        tracking = report["point_tracking"]
        if scored_frames is None:
            scored_frames = tracking["scored_frames"]
        elif tracking["scored_frames"] != scored_frames:
            raise ValueError(f"Scored frame mismatch for {method}")
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
    for row in rows.values():
        row["vs_pure_pbd"] = {
            "2d_mean_error_reduction_pct": reduction(
                baseline["2d_mean_px"], row["2d_mean_px"]
            ),
            "3d_mean_error_reduction_pct": reduction(
                baseline["3d_mean_mm"], row["3d_mean_mm"]
            ),
        }
    selected = min(
        METHODS[1:],
        key=lambda method: (
            rows[method]["3d_mean_mm"],
            rows[method]["2d_mean_px"],
        ),
    )
    output = {
        "schema": "super_residual_prefix_future_validation_v1",
        "frame_range": [0, 1151],
        "train_range": [0, 920],
        "open_loop_validation_range": [921, 1151],
        "formal_future_test_start_untouched": 1152,
        "scored_frames": scored_frames,
        "methods": rows,
        "selected_by_3d_then_2d": selected,
        "selected_beats_pure_pbd_on_both_means": bool(
            rows[selected]["2d_mean_px"] < baseline["2d_mean_px"]
            and rows[selected]["3d_mean_mm"] < baseline["3d_mean_mm"]
        ),
    }
    (root / "FUTURE_VALIDATION_SUMMARY.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SUPER residual prefix future validation",
        "",
        "Frames 921..1151 are open-loop validation; formal future frames 1152+ are not run or scored.",
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
    (root / "FUTURE_VALIDATION_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
