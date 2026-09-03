#!/usr/bin/env python3
"""Summarize non-formal causal fixed-lag reconstruction/future validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    "pure_pbd",
    "visual_residual",
    "visual_residual_online_causal",
)
PROTOCOLS = ("reconstruction_7to1", "future_80to20")


def reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    point = report["point_tracking"]
    return {
        "protocol": report["protocol"],
        "scored_frames": point["scored_frames"],
        "2d_mean_px": float(point["2d_error_px"]["mean"]),
        "2d_rmse_px": float(point["2d_error_px"]["rmse"]),
        "3d_mean_mm": float(point["3d_error_mm"]["mean"]),
        "3d_rmse_mm": float(point["3d_error_mm"]["rmse"]),
        "tap_delta_avg": float(
            point["2d_tap_position_accuracy"]["delta_avg"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    rows: dict[str, dict[str, dict]] = {}
    comparisons: dict[str, dict] = {}
    for protocol in PROTOCOLS:
        rows[protocol] = {
            method: load(
                root / method / protocol / "evaluation_results.json"
            )
            for method in METHODS
        }
        identities = {
            (row["protocol"], tuple(row["scored_frames"]))
            for row in rows[protocol].values()
        }
        if len(identities) != 1:
            raise ValueError(f"Scored-frame mismatch for {protocol}")
        online = rows[protocol]["visual_residual_online_causal"]
        effects = {}
        for reference_method in ("pure_pbd", "visual_residual"):
            reference = rows[protocol][reference_method]
            effects[f"online_vs_{reference_method}"] = {
                "2d_mean_reduction_pct": reduction(
                    reference["2d_mean_px"], online["2d_mean_px"]
                ),
                "3d_mean_reduction_pct": reduction(
                    reference["3d_mean_mm"], online["3d_mean_mm"]
                ),
            }
        effects["online_beats_both_on_2d3d_means"] = all(
            online[key] < rows[protocol][reference][key]
            for reference in ("pure_pbd", "visual_residual")
            for key in ("2d_mean_px", "3d_mean_mm")
        )
        comparisons[protocol] = effects
    report = {
        "schema": "super_causal_fixed_lag_validation_v1",
        "selection_uses_formal_reconstruction_phase": False,
        "selection_uses_formal_future_segment": False,
        "reconstruction_validation_phase": 2,
        "future_prefix_train_range": [0, 920],
        "future_prefix_validation_range": [921, 1151],
        "formal_future_test_start_untouched": 1152,
        "methods": rows,
        "comparisons": comparisons,
        "passes_requested_dominance": all(
            comparisons[protocol]["online_beats_both_on_2d3d_means"]
            for protocol in PROTOCOLS
        ),
    }
    (root / "CAUSAL_FIXED_LAG_VALIDATION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Causal fixed-lag non-formal validation",
        "",
        "| Protocol | Method | 2D mean/RMSE px | 3D mean/RMSE mm | TAP |",
        "|---|---|---:|---:|---:|",
    ]
    for protocol in PROTOCOLS:
        for method in METHODS:
            row = rows[protocol][method]
            lines.append(
                f"| {protocol} | {method} | "
                f"{row['2d_mean_px']:.3f}/{row['2d_rmse_px']:.3f} | "
                f"{row['3d_mean_mm']:.3f}/{row['3d_rmse_mm']:.3f} | "
                f"{row['tap_delta_avg']:.5f} |"
            )
        effect = comparisons[protocol]["online_vs_visual_residual"]
        lines.extend(
            [
                "",
                f"{protocol}: online vs residual-only = "
                f"{effect['2d_mean_reduction_pct']:+.2f}% 2D, "
                f"{effect['3d_mean_reduction_pct']:+.2f}% 3D.",
                "",
            ]
        )
    lines.append(
        "Passes requested 2D/3D dominance in both protocols: "
        f"`{report['passes_requested_dominance']}`."
    )
    (root / "CAUSAL_FIXED_LAG_VALIDATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
