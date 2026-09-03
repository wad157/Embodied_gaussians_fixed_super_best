#!/usr/bin/env python3
"""Compare pure PBD, repaired residual, and repaired residual plus online stiffness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROTOCOLS = ("reconstruction_7to1", "future_80to20")
LABELS = {
    "pure_pbd": "Pure PBD",
    "repaired_residual": "PBD + repaired residual",
    "repaired_residual_online": "PBD + repaired residual + online stiffness",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pure-pbd-root", type=Path, required=True)
    parser.add_argument("--repaired-residual-root", type=Path, required=True)
    return parser.parse_args()


def flatten(path: Path, protocol: str) -> tuple[dict, dict]:
    report = json.loads((path / "evaluation_results.json").read_text(encoding="utf-8"))
    if report["protocol"] != protocol:
        raise ValueError(f"Protocol mismatch in {path}")
    tracking = report["point_tracking"]
    rendering = report["rendering"]
    values = {
        "scored_track_frames": int(tracking["scored_frame_count"]),
        "scored_track_samples": int(tracking["2d_error_px"]["count"]),
        "2d_mean_px": float(tracking["2d_error_px"]["mean"]),
        "2d_rmse_px": float(tracking["2d_error_px"]["rmse"]),
        "tap_delta_avg": float(tracking["2d_tap_position_accuracy"]["delta_avg"]),
        "3d_mean_mm": float(tracking["3d_error_mm"]["mean"]),
        "3d_rmse_mm": float(tracking["3d_error_mm"]["rmse"]),
        "3d_coverage": float(tracking["3d_coverage"]["fraction"]),
        "render_frames": int(rendering["frame_count"]),
        "psnr_db": float(rendering["psnr_db"]),
        "ssim": float(rendering["ssim"]),
        "lpips_alex": float(rendering["lpips_alex"]),
    }
    identity = {
        "ground_truth": report["ground_truth"],
        "scored_frames": tracking["scored_frames"],
    }
    return values, identity


def reduction(baseline: float, value: float) -> float:
    return 100.0 * (baseline - value) / baseline


def compare(baseline: dict, candidate: dict) -> dict:
    return {
        "2d_mean_error_reduction_pct": reduction(
            baseline["2d_mean_px"], candidate["2d_mean_px"]
        ),
        "3d_mean_error_reduction_pct": reduction(
            baseline["3d_mean_mm"], candidate["3d_mean_mm"]
        ),
        "psnr_change_db": candidate["psnr_db"] - baseline["psnr_db"],
        "ssim_change": candidate["ssim"] - baseline["ssim"],
        "lpips_reduction_pct": reduction(
            baseline["lpips_alex"], candidate["lpips_alex"]
        ),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    method_roots = {
        "pure_pbd": args.pure_pbd_root.resolve(),
        "repaired_residual": args.repaired_residual_root.resolve(),
        "repaired_residual_online": output_root,
    }
    protocols: dict[str, dict[str, dict]] = {}
    comparisons: dict[str, dict[str, dict]] = {}
    for protocol in PROTOCOLS:
        protocols[protocol] = {}
        reference_identity = None
        for method, root in method_roots.items():
            values, identity = flatten(root / protocol, protocol)
            if reference_identity is None:
                reference_identity = identity
            elif identity != reference_identity:
                raise ValueError(f"Ground truth or scored frames differ for {method}/{protocol}")
            protocols[protocol][method] = values
        values = protocols[protocol]
        comparisons[protocol] = {
            "online_vs_pure_pbd": compare(
                values["pure_pbd"], values["repaired_residual_online"]
            ),
            "online_vs_repaired_residual": compare(
                values["repaired_residual"], values["repaired_residual_online"]
            ),
            "repaired_residual_vs_pure_pbd": compare(
                values["pure_pbd"], values["repaired_residual"]
            ),
        }

    output = {
        "schema": "super_repaired_residual_online_comparison_v1",
        "frame_range": [0, 1439],
        "residual_settings": {
            "iterations": 8,
            "learning_rate_m": 2.0e-5,
            "maximum_step_m": 2.0e-4,
            "previous_residual_carry": 0.0,
            "temporal_weight": 0.10,
            "magnitude_weight": 0.01,
        },
        "online_stiffness_settings": {"log_learning_rate": 0.18},
        "methods": LABELS,
        "protocols": protocols,
        "comparisons": comparisons,
        "warning": "Single-run effect sizes are not repeated-run significance claims.",
    }
    (output_root / "REPAIRED_ONLINE_COMPARISON.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# SUPER repaired residual + online stiffness comparison",
        "",
        "All methods use the same 10-point ground truth, tool controls, 1440 frames, and three physics steps per video frame.",
        "",
    ]
    for protocol in PROTOCOLS:
        lines.extend(
            [
                f"## {protocol}",
                "",
                "| Method | 2D mean / RMSE (px) | 3D mean / RMSE (mm) | PSNR | SSIM | LPIPS |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, label in LABELS.items():
            row = protocols[protocol][method]
            lines.append(
                f"| {label} | {row['2d_mean_px']:.3f} / {row['2d_rmse_px']:.3f} "
                f"| {row['3d_mean_mm']:.3f} / {row['3d_rmse_mm']:.3f} "
                f"| {row['psnr_db']:.3f} | {row['ssim']:.4f} "
                f"| {row['lpips_alex']:.4f} |"
            )
        effect = comparisons[protocol]["online_vs_repaired_residual"]
        lines.extend(
            [
                "",
                "Online stiffness relative to repaired residual only:",
                "",
                f"- 2D mean error reduction: `{effect['2d_mean_error_reduction_pct']:+.3f}%`",
                f"- 3D mean error reduction: `{effect['3d_mean_error_reduction_pct']:+.3f}%`",
                f"- PSNR change: `{effect['psnr_change_db']:+.3f} dB`",
                f"- SSIM change: `{effect['ssim_change']:+.5f}`",
                f"- LPIPS reduction: `{effect['lpips_reduction_pct']:+.3f}%`",
                "",
            ]
        )
    lines.append("These are single-run effect sizes, not repeated-run significance claims.")
    (output_root / "REPAIRED_ONLINE_COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
