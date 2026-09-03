#!/usr/bin/env python3
"""Summarize complete 1440-frame future residual-step ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pure-pbd", type=Path, required=True)
    parser.add_argument("--residual-0p20", type=Path, required=True)
    return parser.parse_args()


def reduction(baseline: float, value: float) -> float:
    return 100.0 * (baseline - value) / baseline


def read_metrics(path: Path) -> dict[str, float | int]:
    report = json.loads((path / "evaluation_results.json").read_text(encoding="utf-8"))
    if report["protocol"] != "future_80to20":
        raise ValueError(f"Unexpected protocol in {path}")
    tracking = report["point_tracking"]
    rendering = report["rendering"]
    return {
        "2d_mean_px": float(tracking["2d_error_px"]["mean"]),
        "2d_rmse_px": float(tracking["2d_error_px"]["rmse"]),
        "3d_mean_mm": float(tracking["3d_error_mm"]["mean"]),
        "3d_rmse_mm": float(tracking["3d_error_mm"]["rmse"]),
        "tap_delta_avg": float(tracking["2d_tap_position_accuracy"]["delta_avg"]),
        "track_sample_count": int(tracking["2d_error_px"]["count"]),
        "psnr_db": float(rendering["psnr_db"]),
        "ssim": float(rendering["ssim"]),
        "lpips_alex": float(rendering["lpips_alex"]),
        "render_frame_count": int(rendering["frame_count"]),
    }


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    paths = {
        "pure_pbd": args.pure_pbd.resolve(),
        "residual_step_0p20mm": args.residual_0p20.resolve(),
        "residual_step_0p30mm": root / "residual_step_0p30mm",
        "residual_step_0p40mm": root / "residual_step_0p40mm",
    }
    rows = {name: read_metrics(path) for name, path in paths.items()}
    baseline = rows["pure_pbd"]
    for row in rows.values():
        row["vs_pure_pbd"] = {
            "2d_mean_error_reduction_pct": reduction(
                baseline["2d_mean_px"], row["2d_mean_px"]
            ),
            "3d_mean_error_reduction_pct": reduction(
                baseline["3d_mean_mm"], row["3d_mean_mm"]
            ),
            "psnr_delta_db": row["psnr_db"] - baseline["psnr_db"],
            "ssim_delta": row["ssim"] - baseline["ssim"],
            "lpips_reduction_pct": reduction(
                baseline["lpips_alex"], row["lpips_alex"]
            ),
        }
    residual_methods = tuple(name for name in paths if name != "pure_pbd")
    selected = min(
        residual_methods,
        key=lambda name: (rows[name]["3d_mean_mm"], rows[name]["2d_mean_px"]),
    )
    output = {
        "schema": "super_residual_future_full_ablation_v1",
        "frame_range": [0, 1439],
        "observed_train_range": [0, 1151],
        "open_loop_test_range": [1152, 1439],
        "methods": rows,
        "selected_residual_by_3d_then_2d": selected,
        "selected_beats_pure_pbd_on_both_means": bool(
            rows[selected]["2d_mean_px"] < baseline["2d_mean_px"]
            and rows[selected]["3d_mean_mm"] < baseline["3d_mean_mm"]
        ),
    }
    (root / "FULL_FUTURE_COMPARISON.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SUPER complete 80/20 future residual-step ablation",
        "",
        "All methods run frames 0..1439. Frames 0..1151 receive observations; frames 1152..1439 are open-loop test.",
        "",
        "| Method | 2D mean / RMSE (px) | 3D mean / RMSE (mm) | PSNR | SSIM | LPIPS | vs PBD 2D / 3D |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        effect = row["vs_pure_pbd"]
        lines.append(
            f"| {name} | {row['2d_mean_px']:.3f} / {row['2d_rmse_px']:.3f} "
            f"| {row['3d_mean_mm']:.3f} / {row['3d_rmse_mm']:.3f} "
            f"| {row['psnr_db']:.3f} | {row['ssim']:.4f} | {row['lpips_alex']:.4f} "
            f"| {effect['2d_mean_error_reduction_pct']:+.2f}% / "
            f"{effect['3d_mean_error_reduction_pct']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Selected residual: `{selected}`.",
            f"Beats pure PBD on both trajectory means: `{output['selected_beats_pure_pbd_on_both_means']}`.",
        ]
    )
    (root / "FULL_FUTURE_COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
