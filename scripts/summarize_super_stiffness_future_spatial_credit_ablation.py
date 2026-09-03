#!/usr/bin/env python3
"""Compare corrected rejection credit and spatial candidate credit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES = {
    "pure_pbd": (
        REPO_ROOT
        / "outputs/super_tissue_evaluation_formal_20260821"
        / "pure_pbd/future_80to20/evaluation_results.json"
    ),
    "residual_only": (
        REPO_ROOT
        / "outputs/super_tissue_evaluation_residual_fixed_20260821"
        / "future_80to20/evaluation_results.json"
    ),
    "old_h135_bidirectional_8": (
        REPO_ROOT
        / "outputs/super_tissue_evaluation_residual_online_bidirectional_graph_20260821_v2"
        / "future_80to20/evaluation_results.json"
    ),
}
CASES = (
    "h135_corrected_source_bidirectional_8",
    "h135_corrected_source_spatial_components_12",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def read_metrics(path: Path) -> dict[str, float | int]:
    tracking = json.loads(path.read_text(encoding="utf-8"))["point_tracking"]
    return {
        "frames": int(tracking["scored_frame_count"]),
        "samples": int(tracking["2d_error_px"]["count"]),
        "2d_mean_px": float(tracking["2d_error_px"]["mean"]),
        "2d_rmse_px": float(tracking["2d_error_px"]["rmse"]),
        "3d_mean_mm": float(tracking["3d_error_mm"]["mean"]),
        "3d_rmse_mm": float(tracking["3d_error_mm"]["rmse"]),
    }


def read_diagnostics(path: Path) -> dict[str, object]:
    status = Counter()
    variants = Counter()
    scopes = Counter()
    latest: dict = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            material = event.get("material") or {}
            if event.get("event") == "stiffness_validation":
                state = material.get(
                    "validation_status", material.get("status", "unknown")
                )
                status[str(state)] += 1
                if state == "committed":
                    variants[
                        str(material.get("selected_candidate_variant", "unknown"))
                    ] += 1
                    scopes[
                        str(material.get("selected_candidate_scope", "full"))
                    ] += 1
            if int(material.get("update_count", -1) or 0) >= int(
                latest.get("update_count", -1) or 0
            ) and "distance_minimum" in material:
                latest = material
    return {
        "validation_status": dict(status),
        "committed_variants": dict(variants),
        "committed_scopes": dict(scopes),
        "commit_count": sum(variants.values()),
        "final_distance": [
            latest.get("distance_minimum"),
            latest.get("distance_median"),
            latest.get("distance_maximum"),
        ],
        "final_shape": [
            latest.get("shape_minimum"),
            latest.get("shape_median"),
            latest.get("shape_maximum"),
        ],
    }


def reduction(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / baseline


def main() -> None:
    root = parse_args().root.resolve()
    paths = dict(BASELINES)
    for name in CASES:
        paths[name] = root / name / "evaluation_results.json"
    metrics = {name: read_metrics(path) for name, path in paths.items()}
    reference = metrics["residual_only"]
    effects = {
        name: {
            "vs_residual_2d_mean_reduction_pct": reduction(
                float(reference["2d_mean_px"]), float(row["2d_mean_px"])
            ),
            "vs_residual_3d_mean_reduction_pct": reduction(
                float(reference["3d_mean_mm"]), float(row["3d_mean_mm"])
            ),
        }
        for name, row in metrics.items()
        if name not in {"pure_pbd", "residual_only"}
    }
    diagnostics = {
        name: read_diagnostics(root / name / "stiffness_diagnostics/events.jsonl")
        for name in CASES
    }
    output = {
        "schema": "super_stiffness_future_spatial_credit_ablation_v1",
        "metrics": metrics,
        "effects": effects,
        "diagnostics": diagnostics,
    }
    (root / "SPATIAL_CREDIT_ABLATION.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Online stiffness spatial-credit ablation",
        "",
        "| Method | 2D mean / RMSE (px) | 3D mean / RMSE (mm) | vs residual 2D / 3D mean |",
        "|---|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        effect = effects.get(name)
        effect_text = (
            "baseline"
            if effect is None
            else f"{effect['vs_residual_2d_mean_reduction_pct']:+.3f}% / "
            f"{effect['vs_residual_3d_mean_reduction_pct']:+.3f}%"
        )
        lines.append(
            f"| {name} | {row['2d_mean_px']:.3f} / {row['2d_rmse_px']:.3f} "
            f"| {row['3d_mean_mm']:.3f} / {row['3d_rmse_mm']:.3f} "
            f"| {effect_text} |"
        )
    (root / "SPATIAL_CREDIT_ABLATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
