#!/usr/bin/env python3
"""Summarize deliberately soft/hard material-recovery experiments."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


INITIALIZATIONS = ("extreme_soft", "extreme_hard")
METHODS = ("pure_pbd", "residual_only", "residual_online_hierarchical")


def load_metrics(path: Path) -> dict[str, float]:
    tracking = json.loads(path.read_text(encoding="utf-8"))["point_tracking"]
    return {
        "2d_mean_px": float(tracking["2d_error_px"]["mean"]),
        "2d_rmse_px": float(tracking["2d_error_px"]["rmse"]),
        "3d_mean_mm": float(tracking["3d_error_mm"]["mean"]),
        "3d_rmse_mm": float(tracking["3d_error_mm"]["rmse"]),
    }


def diagnostics(path: Path) -> dict[str, object]:
    statuses: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    latest: dict = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            material = event.get("material") or {}
            if event.get("event") == "stiffness_validation":
                status = str(
                    material.get("validation_status", material.get("status"))
                )
                statuses[status] += 1
                if status == "committed":
                    variants[str(material.get("selected_candidate_variant"))] += 1
            if int(material.get("update_count", -1) or 0) >= int(
                latest.get("update_count", -1) or 0
            ) and "distance_median" in material:
                latest = material
    return {
        "validation_status": dict(statuses),
        "committed_variants": dict(variants),
        "final_distance_median": latest.get("distance_median"),
        "final_shape_median": latest.get("shape_median"),
    }


def reduction(reference: float, value: float) -> float:
    return 100.0 * (reference - value) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    table = {
        initialization: {
            method: load_metrics(
                root / initialization / method / "evaluation_results.json"
            )
            for method in METHODS
        }
        for initialization in INITIALIZATIONS
    }
    effects = {}
    order_checks = {}
    for initialization, rows in table.items():
        pure = rows["pure_pbd"]
        residual = rows["residual_only"]
        online = rows["residual_online_hierarchical"]
        effects[initialization] = {
            "residual_vs_pure_2d_pct": reduction(
                pure["2d_mean_px"], residual["2d_mean_px"]
            ),
            "residual_vs_pure_3d_pct": reduction(
                pure["3d_mean_mm"], residual["3d_mean_mm"]
            ),
            "online_vs_residual_2d_pct": reduction(
                residual["2d_mean_px"], online["2d_mean_px"]
            ),
            "online_vs_residual_3d_pct": reduction(
                residual["3d_mean_mm"], online["3d_mean_mm"]
            ),
        }
        order_checks[initialization] = {
            "2d_online_better_than_residual_better_than_pure": (
                online["2d_mean_px"]
                < residual["2d_mean_px"]
                < pure["2d_mean_px"]
            ),
            "3d_online_better_than_residual_better_than_pure": (
                online["3d_mean_mm"]
                < residual["3d_mean_mm"]
                < pure["3d_mean_mm"]
            ),
        }
    report = {
        "schema": "super_extreme_initialization_recovery_v1",
        "metrics": table,
        "effects": effects,
        "desired_order": order_checks,
        "online_diagnostics": {
            initialization: diagnostics(
                root
                / initialization
                / "residual_online_hierarchical"
                / "stiffness_diagnostics/events.jsonl"
            )
            for initialization in INITIALIZATIONS
        },
    }
    (root / "EXTREME_INITIALIZATION_RECOVERY.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Extreme initialization recovery",
        "",
        "| Initialization | Method | 2D mean / RMSE px | 3D mean / RMSE mm |",
        "|---|---|---:|---:|",
    ]
    for initialization in INITIALIZATIONS:
        for method in METHODS:
            row = table[initialization][method]
            lines.append(
                f"| {initialization} | {method} | "
                f"{row['2d_mean_px']:.3f} / {row['2d_rmse_px']:.3f} | "
                f"{row['3d_mean_mm']:.3f} / {row['3d_rmse_mm']:.3f} |"
            )
    (root / "EXTREME_INITIALIZATION_RECOVERY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
