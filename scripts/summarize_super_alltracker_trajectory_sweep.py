#!/usr/bin/env python3
"""Rank the held-out AllTracker observer parameter sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIGS = {
    "current": (1.0, 0.75, 1.0, 0.001, 20.0, 20.0, 0.25),
    "bounded5": (1.0, 0.75, 1.0, 0.001, 5.0, 5.0, 0.10),
    "bounded3": (1.0, 0.75, 1.0, 0.001, 3.0, 3.0, 0.10),
    "confidence": (1.0, 0.75, 1.0, 0.020, 5.0, 5.0, 0.10),
    "velocity_low": (1.0, 0.30, 1.0, 0.005, 5.0, 5.0, 0.10),
    "velocity_high": (1.0, 1.00, 1.0, 0.005, 5.0, 5.0, 0.10),
    "blend85": (1.0, 0.50, 0.85, 0.005, 5.0, 5.0, 0.10),
    "position075": (0.75, 0.50, 1.0, 0.005, 5.0, 5.0, 0.10),
}
PROTOCOLS = ("reconstruction_7to1", "future_80to20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows: dict[str, dict] = {}
    for name, parameters in CONFIGS.items():
        rows[name] = {"parameters": parameters, "protocols": {}}
        for protocol in PROTOCOLS:
            report = json.loads(
                (root / name / protocol / "validation_results.json").read_text(
                    encoding="utf-8"
                )
            )
            rows[name]["protocols"][protocol] = {
                "2d_mean_px": float(report["2d_error_px"]["mean"]),
                "2d_rmse_px": float(report["2d_error_px"]["rmse"]),
                "3d_mean_mm": float(report["3d_error_mm"]["mean"]),
                "3d_rmse_mm": float(report["3d_error_mm"]["rmse"]),
                "tap": float(report["2d_tap_position_accuracy"]["delta_avg"]),
            }
    baseline = rows["current"]["protocols"]
    for row in rows.values():
        ratios = []
        for protocol in PROTOCOLS:
            for key in (
                "2d_mean_px",
                "2d_rmse_px",
                "3d_mean_mm",
                "3d_rmse_mm",
            ):
                ratios.append(
                    row["protocols"][protocol][key] / baseline[protocol][key]
                )
        row["joint_normalized_error"] = float(sum(ratios) / len(ratios))
    selected = min(rows, key=lambda name: rows[name]["joint_normalized_error"])
    output = {
        "schema": "super_alltracker_trajectory_parameter_sweep_v1",
        "formal_result": False,
        "reconstruction_validation_phase": 4,
        "future_open_loop_validation_frames": [921, 1151],
        "formal_test_frames_used": False,
        "parameter_order": [
            "position_gain",
            "velocity_gain",
            "absolute_position_weight",
            "regularization",
            "robust_residual_mm",
            "maximum_position_correction_mm",
            "maximum_velocity_correction_m_s",
        ],
        "methods": rows,
        "selected": selected,
    }
    (root / "TRAJECTORY_PARAMETER_SWEEP.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# AllTracker trajectory observer validation sweep",
        "",
        "No formal phase-0 reconstruction or frame-1152+ future result was used.",
        "",
        "| Config | Recon 2D / 3D mean | Future 2D / 3D mean | Joint normalized error |",
        "|---|---:|---:|---:|",
    ]
    for name, row in sorted(
        rows.items(), key=lambda item: item[1]["joint_normalized_error"]
    ):
        reconstruction = row["protocols"]["reconstruction_7to1"]
        future = row["protocols"]["future_80to20"]
        lines.append(
            f"| {name} | {reconstruction['2d_mean_px']:.3f} px / "
            f"{reconstruction['3d_mean_mm']:.3f} mm | "
            f"{future['2d_mean_px']:.3f} px / {future['3d_mean_mm']:.3f} mm | "
            f"{row['joint_normalized_error']:.5f} |"
        )
    lines.extend(["", f"Selected: `{selected}`."])
    (root / "TRAJECTORY_PARAMETER_SWEEP.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
