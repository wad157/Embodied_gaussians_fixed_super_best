#!/usr/bin/env python3
"""Select a residual gain profile on non-formal reconstruction phase 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


INITIALIZATIONS = ("extreme_soft", "moderate", "extreme_hard")
METHODS = (
    "pure_pbd",
    "residual_current",
    "residual_conservative",
    "residual_micro",
)


def load(path: Path) -> dict[str, float]:
    point = json.loads(path.read_text(encoding="utf-8"))["point_tracking"]
    return {
        "2d_mean_px": float(point["2d_error_px"]["mean"]),
        "2d_rmse_px": float(point["2d_error_px"]["rmse"]),
        "3d_mean_mm": float(point["3d_error_mm"]["mean"]),
        "3d_rmse_mm": float(point["3d_error_mm"]["rmse"]),
        "tap_delta_avg": float(point["2d_tap_position_accuracy"]["delta_avg"]),
    }


def reduction(reference: float, value: float) -> float:
    return 100.0 * (reference - value) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    metrics = {
        initialization: {
            method: load(root / initialization / method / "evaluation_results.json")
            for method in METHODS
        }
        for initialization in INITIALIZATIONS
    }
    candidates = {}
    for method in METHODS[1:]:
        effects = {
            initialization: {
                "2d_mean_reduction_pct": reduction(
                    metrics[initialization]["pure_pbd"]["2d_mean_px"],
                    metrics[initialization][method]["2d_mean_px"],
                ),
                "3d_mean_reduction_pct": reduction(
                    metrics[initialization]["pure_pbd"]["3d_mean_mm"],
                    metrics[initialization][method]["3d_mean_mm"],
                ),
            }
            for initialization in INITIALIZATIONS
        }
        all_reductions = [
            value
            for initialization in INITIALIZATIONS
            for value in effects[initialization].values()
        ]
        candidates[method] = {
            "effects": effects,
            "minimum_reduction_pct": min(all_reductions),
            "mean_reduction_pct": sum(all_reductions) / len(all_reductions),
            "all_2d3d_means_better_than_pure": all(
                value > 0.0 for value in all_reductions
            ),
        }
    selected = max(
        METHODS[1:],
        key=lambda method: (
            candidates[method]["all_2d3d_means_better_than_pure"],
            candidates[method]["minimum_reduction_pct"],
            candidates[method]["mean_reduction_pct"],
        ),
    )
    report = {
        "schema": "super_reconstruction_residual_gain_validation_v1",
        "selection_partition": "reconstruction_7to1_phase_2_nonformal",
        "formal_test_phase_used_for_selection": False,
        "metrics": metrics,
        "candidates": candidates,
        "selected_method": selected,
    }
    (root / "RESIDUAL_GAIN_VALIDATION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Reconstruction residual gain validation (phase 2)",
        "",
        "| Initialization | Method | 2D mean/RMSE px | 3D mean/RMSE mm | TAP delta_avg |",
        "|---|---|---:|---:|---:|",
    ]
    for initialization in INITIALIZATIONS:
        for method in METHODS:
            row = metrics[initialization][method]
            lines.append(
                f"| {initialization} | {method} | "
                f"{row['2d_mean_px']:.3f}/{row['2d_rmse_px']:.3f} | "
                f"{row['3d_mean_mm']:.3f}/{row['3d_rmse_mm']:.3f} | "
                f"{row['tap_delta_avg']:.5f} |"
            )
    lines.extend(["", f"Selected: `{selected}`", ""])
    (root / "RESIDUAL_GAIN_VALIDATION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
