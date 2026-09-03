#!/usr/bin/env python3
"""Summarize component-merge online-stiffness future experiments."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "pure_pbd": REPO_ROOT / "outputs/super_tissue_evaluation_formal_20260821/pure_pbd/future_80to20/evaluation_results.json",
    "residual_only": REPO_ROOT / "outputs/super_tissue_evaluation_residual_fixed_20260821/future_80to20/evaluation_results.json",
    "old_h135_online": REPO_ROOT / "outputs/super_tissue_evaluation_residual_online_bidirectional_graph_20260821_v2/future_80to20/evaluation_results.json",
    "spatial_components": REPO_ROOT / "outputs/super_stiffness_future_spatial_credit_ablation_20260822_v1/h135_corrected_source_spatial_components_12/evaluation_results.json",
}
CASES = ("merged_balanced", "merged_strong")


def metrics(path: Path) -> dict[str, float]:
    row = json.loads(path.read_text(encoding="utf-8"))["point_tracking"]
    return {
        "2d_mean_px": float(row["2d_error_px"]["mean"]),
        "2d_rmse_px": float(row["2d_error_px"]["rmse"]),
        "3d_mean_mm": float(row["3d_error_mm"]["mean"]),
        "3d_rmse_mm": float(row["3d_error_mm"]["rmse"]),
    }


def diagnostics(path: Path) -> dict[str, object]:
    status = Counter()
    variants = Counter()
    scopes = Counter()
    trial_counts = Counter()
    latest: dict = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            material = event.get("material") or {}
            if event.get("event") == "stiffness_validation":
                state = material.get("validation_status", material.get("status"))
                status[str(state)] += 1
                if state == "committed":
                    variants[str(material.get("selected_candidate_variant"))] += 1
                    scopes[str(material.get("selected_candidate_scope"))] += 1
                    trial_counts[str(material.get("candidate_search_trial_count"))] += 1
            if int(material.get("update_count", -1) or 0) >= int(
                latest.get("update_count", -1) or 0
            ) and "distance_minimum" in material:
                latest = material
    return {
        "validation_status": dict(status),
        "committed_variants": dict(variants),
        "committed_scopes": dict(scopes),
        "committed_trial_counts": dict(trial_counts),
        "commit_count": sum(variants.values()),
        "final_distance": [latest.get("distance_minimum"), latest.get("distance_median"), latest.get("distance_maximum")],
        "final_shape": [latest.get("shape_minimum"), latest.get("shape_median"), latest.get("shape_maximum")],
    }


def reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    paths = dict(PATHS)
    paths.update({name: root / name / "evaluation_results.json" for name in CASES})
    table = {name: metrics(path) for name, path in paths.items()}
    residual = table["residual_only"]
    pure = table["pure_pbd"]
    effects = {
        name: {
            "vs_residual_2d_pct": reduction(residual["2d_mean_px"], row["2d_mean_px"]),
            "vs_residual_3d_pct": reduction(residual["3d_mean_mm"], row["3d_mean_mm"]),
            "vs_pure_2d_pct": reduction(pure["2d_mean_px"], row["2d_mean_px"]),
            "vs_pure_3d_pct": reduction(pure["3d_mean_mm"], row["3d_mean_mm"]),
        }
        for name, row in table.items()
        if name not in {"pure_pbd", "residual_only"}
    }
    report = {
        "schema": "super_stiffness_future_merged_components_ablation_v1",
        "metrics": table,
        "effects": effects,
        "diagnostics": {
            name: diagnostics(root / name / "stiffness_diagnostics/events.jsonl")
            for name in CASES
        },
    }
    (root / "MERGED_COMPONENTS_ABLATION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Merged spatial-component stiffness ablation",
        "",
        "| Method | 2D mean / RMSE px | 3D mean / RMSE mm | vs residual 2D / 3D | vs Pure 2D / 3D |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in table.items():
        effect = effects.get(name)
        residual_text = "baseline" if effect is None else f"{effect['vs_residual_2d_pct']:+.3f}% / {effect['vs_residual_3d_pct']:+.3f}%"
        pure_text = "baseline" if effect is None else f"{effect['vs_pure_2d_pct']:+.3f}% / {effect['vs_pure_3d_pct']:+.3f}%"
        lines.append(
            f"| {name} | {row['2d_mean_px']:.3f} / {row['2d_rmse_px']:.3f} | "
            f"{row['3d_mean_mm']:.3f} / {row['3d_rmse_mm']:.3f} | {residual_text} | {pure_text} |"
        )
    (root / "MERGED_COMPONENTS_ABLATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
