#!/usr/bin/env python3
"""Summarize three material initializations, three methods, and two protocols."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


INITIALIZATIONS = ("extreme_soft", "moderate", "extreme_hard")
METHODS = ("pure_pbd", "residual_only", "residual_online_robust")
PROTOCOLS = ("reconstruction_7to1", "future_80to20")


def load_metrics(path: Path) -> dict[str, float | int | None]:
    result = json.loads(path.read_text(encoding="utf-8"))
    tracking = result["point_tracking"]
    rendering = result["rendering"]
    return {
        "scored_track_frames": int(tracking["scored_frame_count"]),
        "2d_mean_px": float(tracking["2d_error_px"]["mean"]),
        "2d_rmse_px": float(tracking["2d_error_px"]["rmse"]),
        "tap_delta_avg": float(tracking["2d_tap_position_accuracy"]["delta_avg"]),
        "3d_mean_mm": float(tracking["3d_error_mm"]["mean"]),
        "3d_rmse_mm": float(tracking["3d_error_mm"]["rmse"]),
        "render_frames": int(rendering["frame_count"]),
        "psnr_db": float(rendering["psnr_db"]),
        "ssim": float(rendering["ssim"]),
        "lpips_alex": (
            None
            if rendering["lpips_alex"] is None
            else float(rendering["lpips_alex"])
        ),
    }


def percentage_lower(reference: float, value: float) -> float:
    return 100.0 * (reference - value) / reference


def improvement(reference: dict, value: dict) -> dict[str, float | None]:
    lpips = None
    if reference["lpips_alex"] is not None and value["lpips_alex"] is not None:
        lpips = percentage_lower(reference["lpips_alex"], value["lpips_alex"])
    return {
        "2d_mean_reduction_pct": percentage_lower(
            reference["2d_mean_px"], value["2d_mean_px"]
        ),
        "3d_mean_reduction_pct": percentage_lower(
            reference["3d_mean_mm"], value["3d_mean_mm"]
        ),
        "psnr_gain_db": value["psnr_db"] - reference["psnr_db"],
        "ssim_gain": value["ssim"] - reference["ssim"],
        "lpips_reduction_pct": lpips,
    }


def desired_order(rows: dict[str, dict]) -> dict[str, bool]:
    pure = rows["pure_pbd"]
    residual = rows["residual_only"]
    online = rows["residual_online_robust"]
    return {
        "2d_mean": online["2d_mean_px"] < residual["2d_mean_px"] < pure["2d_mean_px"],
        "3d_mean": online["3d_mean_mm"] < residual["3d_mean_mm"] < pure["3d_mean_mm"],
        "psnr": online["psnr_db"] > residual["psnr_db"] > pure["psnr_db"],
        "ssim": online["ssim"] > residual["ssim"] > pure["ssim"],
        "lpips": (
            online["lpips_alex"] is not None
            and residual["lpips_alex"] is not None
            and pure["lpips_alex"] is not None
            and online["lpips_alex"]
            < residual["lpips_alex"]
            < pure["lpips_alex"]
        ),
    }


def diagnostics(path: Path) -> dict[str, object]:
    statuses: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    latest: dict = {}
    if not path.exists():
        return {"available": False}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            material = event.get("material") or {}
            if event.get("event") == "stiffness_validation":
                status = str(material.get("validation_status", material.get("status")))
                statuses[status] += 1
                if status == "committed":
                    variants[str(material.get("selected_candidate_variant"))] += 1
            if int(material.get("update_count", -1) or 0) >= int(
                latest.get("update_count", -1) or 0
            ) and "distance_median" in material:
                latest = material
    return {
        "available": True,
        "validation_status": dict(statuses),
        "committed_variants": dict(variants),
        "final_update_count": latest.get("update_count"),
        "final_distance_median": latest.get("distance_median"),
        "final_shape_median": latest.get("shape_median"),
        "global_distance_direction": latest.get("global_distance_direction"),
        "global_shape_direction": latest.get("global_shape_direction"),
        "global_distance_commits": latest.get("global_distance_commits"),
        "global_shape_commits": latest.get("global_shape_commits"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    metrics = {
        protocol: {
            initialization: {
                method: load_metrics(
                    root / initialization / method / protocol / "evaluation_results.json"
                )
                for method in METHODS
            }
            for initialization in INITIALIZATIONS
        }
        for protocol in PROTOCOLS
    }
    effects: dict[str, dict] = {}
    ordering: dict[str, dict] = {}
    for protocol in PROTOCOLS:
        effects[protocol] = {}
        ordering[protocol] = {}
        for initialization in INITIALIZATIONS:
            rows = metrics[protocol][initialization]
            effects[protocol][initialization] = {
                "residual_vs_pure": improvement(
                    rows["pure_pbd"], rows["residual_only"]
                ),
                "online_vs_residual": improvement(
                    rows["residual_only"], rows["residual_online_robust"]
                ),
                "online_vs_pure": improvement(
                    rows["pure_pbd"], rows["residual_online_robust"]
                ),
            }
            ordering[protocol][initialization] = desired_order(rows)
    report = {
        "schema": "super_three_initialization_full_evaluation_v1",
        "initializations": {
            "extreme_soft": {"distance": 0.10, "shape": 0.003},
            "moderate": {"distance": 0.20, "shape": 0.004},
            "extreme_hard": {"distance": 1.60, "shape": 0.020},
        },
        "metrics": metrics,
        "effects": effects,
        "desired_order": ordering,
        "all_tracking_orders_pass": all(
            checks[metric]
            for protocol in ordering.values()
            for checks in protocol.values()
            for metric in ("2d_mean", "3d_mean")
        ),
        "all_rendering_orders_pass": all(
            checks[metric]
            for protocol in ordering.values()
            for checks in protocol.values()
            for metric in ("psnr", "ssim", "lpips")
        ),
        "online_diagnostics": {
            protocol: {
                initialization: diagnostics(
                    root
                    / initialization
                    / "residual_online_robust"
                    / protocol
                    / "stiffness_diagnostics"
                    / "events.jsonl"
                )
                for initialization in INITIALIZATIONS
            }
            for protocol in PROTOCOLS
        },
    }
    (root / "THREE_INITIALIZATION_FULL_EVALUATION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# 三种初始刚度的完整 SUPER 评估", ""]
    for protocol in PROTOCOLS:
        lines.extend(
            [
                f"## {protocol}",
                "",
                "| 初值 | 方法 | 2D mean/RMSE (px) | 3D mean/RMSE (mm) | PSNR (dB) | SSIM | LPIPS |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for initialization in INITIALIZATIONS:
            for method in METHODS:
                row = metrics[protocol][initialization][method]
                lpips = "N/A" if row["lpips_alex"] is None else f"{row['lpips_alex']:.5f}"
                lines.append(
                    f"| {initialization} | {method} | "
                    f"{row['2d_mean_px']:.3f}/{row['2d_rmse_px']:.3f} | "
                    f"{row['3d_mean_mm']:.3f}/{row['3d_rmse_mm']:.3f} | "
                    f"{row['psnr_db']:.3f} | {row['ssim']:.5f} | {lpips} |"
                )
        lines.append("")
    lines.extend(
        [
            "## 总门槛",
            "",
            f"- 所有切分和初值的 2D/3D 排序均通过：`{report['all_tracking_orders_pass']}`",
            f"- 所有切分和初值的 PSNR/SSIM/LPIPS 排序均通过：`{report['all_rendering_orders_pass']}`",
            "",
        ]
    )
    (root / "THREE_INITIALIZATION_FULL_EVALUATION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
