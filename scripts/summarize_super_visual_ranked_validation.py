#!/usr/bin/env python3
"""Compare the legacy and later-RGB-ranked visual residual observers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


PROTOCOLS = ("reconstruction_7to1", "future_80to20")
METHODS = ("cross_frame_hold", "cross_frame_ranked_hold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "2d_mean_px": float(
            report["point_tracking"]["2d_error_px"]["mean"]
        ),
        "2d_rmse_px": float(
            report["point_tracking"]["2d_error_px"]["rmse"]
        ),
        "3d_mean_mm": float(
            report["point_tracking"]["3d_error_mm"]["mean"]
        ),
        "3d_rmse_mm": float(
            report["point_tracking"]["3d_error_mm"]["rmse"]
        ),
        "scored_frames": int(
            report["point_tracking"]["scored_frame_count"]
        ),
    }


def load_visual_diagnostics(path: Path) -> dict:
    if not path.is_file():
        return {
            "cross_frame_validation_count": 0,
            "cross_frame_accept_count": 0,
            "cross_frame_acceptance_rate": None,
            "selected_gain_counts": {},
            "candidate_count_distribution": {},
            "open_loop_prediction_count": 0,
            "open_loop_prediction_accept_count": 0,
        }
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validations = [
        event
        for event in events
        if event.get("event") == "visual_cross_frame_validation"
    ]
    accepted = [
        event
        for event in validations
        if bool(event.get("image", {}).get("accepted", False))
    ]
    predictions = [
        event
        for event in events
        if event.get("event") == "visual_open_loop_prediction"
    ]
    gain_counts = Counter(
        f"{float(event['image']['selected_gain']):g}"
        for event in accepted
        if event.get("image", {}).get("selected_gain") is not None
    )
    candidate_counts = Counter(
        str(int(event["image"].get("candidate_count", 0)))
        for event in validations
    )
    return {
        "cross_frame_validation_count": len(validations),
        "cross_frame_accept_count": len(accepted),
        "cross_frame_acceptance_rate": (
            None if not validations else len(accepted) / len(validations)
        ),
        "selected_gain_counts": dict(sorted(gain_counts.items())),
        "candidate_count_distribution": dict(
            sorted(candidate_counts.items())
        ),
        "open_loop_prediction_count": len(predictions),
        "open_loop_prediction_accept_count": sum(
            bool(event.get("image", {}).get("accepted", False))
            for event in predictions
        ),
    }


def reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows: dict[str, dict[str, dict]] = {}
    diagnostics: dict[str, dict[str, dict]] = {}
    effects: dict[str, dict] = {}
    passed = True
    for protocol in PROTOCOLS:
        rows[protocol] = {}
        diagnostics[protocol] = {}
        for method in METHODS:
            rows[protocol][method] = load_metrics(
                root / method / protocol / "evaluation_results.json"
            )
            diagnostics[protocol][method] = load_visual_diagnostics(
                root
                / method
                / protocol
                / "visual_diagnostics"
                / "events.jsonl"
            )
        baseline = rows[protocol]["cross_frame_hold"]
        ranked = rows[protocol]["cross_frame_ranked_hold"]
        protocol_effect = {
            "2d_mean_reduction_pct": reduction(
                baseline["2d_mean_px"], ranked["2d_mean_px"]
            ),
            "3d_mean_reduction_pct": reduction(
                baseline["3d_mean_mm"], ranked["3d_mean_mm"]
            ),
            "ranked_beats_control_on_2d3d_means": bool(
                ranked["2d_mean_px"] < baseline["2d_mean_px"]
                and ranked["3d_mean_mm"] < baseline["3d_mean_mm"]
            ),
        }
        effects[protocol] = protocol_effect
        passed &= protocol_effect["ranked_beats_control_on_2d3d_means"]
    report = {
        "schema": "super_visual_ranked_validation_v1",
        "selection_only": True,
        "formal_test_used_for_selection": False,
        "metrics": rows,
        "visual_diagnostics": diagnostics,
        "effects": effects,
        "passed": bool(passed),
    }
    (root / "VISUAL_RANKED_VALIDATION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Visual ranked hold validation",
        "",
        "| Protocol | Method | 2D mean px | 3D mean mm |",
        "|---|---|---:|---:|",
    ]
    for protocol in PROTOCOLS:
        for method in METHODS:
            metrics = rows[protocol][method]
            lines.append(
                f"| {protocol} | {method} | "
                f"{metrics['2d_mean_px']:.6f} | "
                f"{metrics['3d_mean_mm']:.6f} |"
            )
    lines.extend(("", f"Strict 2D+3D direction gate: **{passed}**"))
    (root / "VISUAL_RANKED_VALIDATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
