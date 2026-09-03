#!/usr/bin/env python3
"""Summarize the non-formal 600-frame stiffness-consensus selection run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    "pure_pbd",
    "pbd_visual_residual",
    "pbd_visual_residual_online_stiffness",
)
PROTOCOLS = ("reconstruction_7to1", "future_80to20")


def reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def load_score(path: Path) -> dict:
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


def load_stiffness_events(path: Path) -> dict:
    counts = {"committed": 0, "deferred": 0, "rejected": 0}
    commits = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") != "stiffness_validation":
            continue
        material = event.get("material", {})
        status = str(material.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        if status == "committed":
            commits.append(
                {
                    "frame_index": int(event["frame_index"]),
                    "phase": event.get("phase"),
                    "proposal_phase": material.get(
                        "stiffness_proposal_phase"
                    ),
                    "variant": material.get("selected_candidate_variant"),
                    "distance_scale": material.get(
                        "selected_distance_scale"
                    ),
                    "shape_scale": material.get("selected_shape_scale"),
                    "scope": material.get("selected_candidate_scope"),
                    "improvement": material.get("prediction_improvement"),
                }
            )
    return {"status_counts": counts, "commits": commits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    rows: dict[str, dict[str, dict]] = {}
    comparisons: dict[str, dict] = {}
    stiffness: dict[str, dict] = {}

    for protocol in PROTOCOLS:
        rows[protocol] = {
            method: load_score(
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
        online = rows[protocol]["pbd_visual_residual_online_stiffness"]
        effects = {}
        for reference_method in ("pure_pbd", "pbd_visual_residual"):
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
            for reference in ("pure_pbd", "pbd_visual_residual")
            for key in ("2d_mean_px", "3d_mean_mm")
        )
        comparisons[protocol] = effects
        stiffness[protocol] = load_stiffness_events(
            root
            / "pbd_visual_residual_online_stiffness"
            / protocol
            / "diagnostics"
            / "events.jsonl"
        )

    has_commit_in_both = all(
        stiffness[protocol]["status_counts"].get("committed", 0) > 0
        for protocol in PROTOCOLS
    )
    report = {
        "schema": "super_stiffness_consensus_validation_v1",
        "selection_is_non_formal": True,
        "frame_count": 600,
        "reconstruction_validation_phase": 2,
        "future_prefix_train_range": [0, 479],
        "future_prefix_validation_range": [480, 599],
        "formal_reconstruction_phase_untouched": 0,
        "formal_future_test_start_untouched": 1152,
        "methods": rows,
        "stiffness": stiffness,
        "comparisons": comparisons,
        "has_stiffness_commit_in_both_protocols": has_commit_in_both,
        "passes_selection": bool(
            has_commit_in_both
            and all(
                comparisons[protocol]["online_beats_both_on_2d3d_means"]
                for protocol in PROTOCOLS
            )
        ),
    }
    (root / "STIFFNESS_CONSENSUS_VALIDATION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stiffness consensus non-formal validation",
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
        effect = comparisons[protocol]["online_vs_pbd_visual_residual"]
        lines.extend(
            [
                "",
                f"{protocol}: online vs residual = "
                f"{effect['2d_mean_reduction_pct']:+.3f}% 2D, "
                f"{effect['3d_mean_reduction_pct']:+.3f}% 3D; "
                f"commits={len(stiffness[protocol]['commits'])}, "
                f"deferred={stiffness[protocol]['status_counts'].get('deferred', 0)}.",
                "",
            ]
        )
    lines.append(f"Passes selection: `{report['passes_selection']}`.")
    (root / "STIFFNESS_CONSENSUS_VALIDATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
