#!/usr/bin/env python3
"""Gate causal stiffness recovery at both legal material extremes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


INITIALIZATIONS = ("extreme_soft", "extreme_hard")
PROTOCOLS = ("reconstruction_7to1", "future_80to20")
METHODS = ("ranked_residual", "ranked_residual_online_causal")
MINIMUM_MEAN_GAIN = 0.01


def metrics(path: Path) -> dict[str, float | int]:
    tracking = json.loads(path.read_text(encoding="utf-8"))["point_tracking"]
    return {
        "2d_mean_px": float(tracking["2d_error_px"]["mean"]),
        "2d_rmse_px": float(tracking["2d_error_px"]["rmse"]),
        "3d_mean_mm": float(tracking["3d_error_mm"]["mean"]),
        "3d_rmse_mm": float(tracking["3d_error_mm"]["rmse"]),
        "scored_frames": int(tracking["scored_frame_count"]),
    }


def material_diagnostics(path: Path) -> dict:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validations = [
        event for event in events if event.get("event") == "stiffness_validation"
    ]
    statuses = Counter(
        str(
            event.get("material", {}).get(
                "validation_status",
                event.get("material", {}).get("status", "unknown"),
            )
        )
        for event in validations
    )
    committed = [
        event
        for event in validations
        if event.get("material", {}).get("status") == "committed"
    ]
    latest = max(
        (
            event.get("material", {})
            for event in events
            if "distance_median" in event.get("material", {})
        ),
        key=lambda value: int(value.get("update_count", 0)),
        default={},
    )
    return {
        "validation_status_counts": dict(sorted(statuses.items())),
        "commit_count": len(committed),
        "committed_variants": dict(
            sorted(
                Counter(
                    str(
                        event.get("material", {}).get(
                            "selected_candidate_variant", "unknown"
                        )
                    )
                    for event in committed
                ).items()
            )
        ),
        "final_distance_median": latest.get("distance_median"),
        "final_shape_median": latest.get("shape_median"),
    }


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    rows: dict[str, dict[str, dict[str, dict]]] = {}
    effects: dict[str, dict[str, dict]] = {}
    diagnostics: dict[str, dict[str, dict]] = {}
    passed = True
    for initialization in INITIALIZATIONS:
        rows[initialization] = {}
        effects[initialization] = {}
        diagnostics[initialization] = {}
        for protocol in PROTOCOLS:
            rows[initialization][protocol] = {
                method: metrics(
                    root
                    / initialization
                    / method
                    / protocol
                    / "evaluation_results.json"
                )
                for method in METHODS
            }
            baseline = rows[initialization][protocol]["ranked_residual"]
            online = rows[initialization][protocol][
                "ranked_residual_online_causal"
            ]
            gains = {
                "2d_mean_relative": relative_gain(
                    float(baseline["2d_mean_px"]),
                    float(online["2d_mean_px"]),
                ),
                "2d_rmse_relative": relative_gain(
                    float(baseline["2d_rmse_px"]),
                    float(online["2d_rmse_px"]),
                ),
                "3d_mean_relative": relative_gain(
                    float(baseline["3d_mean_mm"]),
                    float(online["3d_mean_mm"]),
                ),
                "3d_rmse_relative": relative_gain(
                    float(baseline["3d_rmse_mm"]),
                    float(online["3d_rmse_mm"]),
                ),
            }
            checks = {
                "2d_mean_at_least_1pct": (
                    gains["2d_mean_relative"] >= MINIMUM_MEAN_GAIN
                ),
                "3d_mean_at_least_1pct": (
                    gains["3d_mean_relative"] >= MINIMUM_MEAN_GAIN
                ),
                "2d_rmse_improved": gains["2d_rmse_relative"] > 0.0,
                "3d_rmse_improved": gains["3d_rmse_relative"] > 0.0,
            }
            case_passed = all(checks.values())
            passed &= case_passed
            effects[initialization][protocol] = {
                "gains": gains,
                "checks": checks,
                "passed": case_passed,
            }
            diagnostics[initialization][protocol] = material_diagnostics(
                root
                / initialization
                / "ranked_residual_online_causal"
                / protocol
                / "diagnostics"
                / "events.jsonl"
            )
    report = {
        "schema": "super_ranked_causal_extreme_selection_v1",
        "selection_only": True,
        "formal_test_used_for_selection": False,
        "material_initializations": {
            "extreme_soft": {"distance": 0.025, "shape": 0.001},
            "extreme_hard": {"distance": 4.0, "shape": 0.040},
        },
        "minimum_mean_relative_gain": MINIMUM_MEAN_GAIN,
        "metrics": rows,
        "effects": effects,
        "online_diagnostics": diagnostics,
        "passed": bool(passed),
    }
    (root / "RANKED_CAUSAL_EXTREME_SELECTION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
