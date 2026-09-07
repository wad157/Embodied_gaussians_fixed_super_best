#!/usr/bin/env python3
"""Validate full-f1 input and strict 80/20 causal withholding for grasp5 Future."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = (
    "pure_pbd",
    "pbd_alltracker_depth_rgb_residual",
    "pbd_alltracker_depth_rgb_residual_online_stiffness",
)
FUTURE_START = 1152
SEQUENCE_END = 1439


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    asset_validation = load_json(args.asset_root / "asset_validation.json")
    gates: dict[str, bool] = {
        "full_f1_asset_validation_passed": asset_validation.get("passed") is True,
        "asset_completion_marker_present": (args.asset_root / "COMPLETE").is_file(),
        "evaluation_completion_marker_present": (
            args.evaluation_root / "COMPLETE"
        ).is_file(),
    }
    evaluation: dict[str, dict] = {}

    expected_all_destinations = np.arange(1, SEQUENCE_END + 1)
    expected_training_destinations = np.arange(1, FUTURE_START)
    expected_future_destinations = np.arange(FUTURE_START, SEQUENCE_END + 1)

    for method in METHODS:
        result_root = args.evaluation_root / method / "future_80to20"
        result = load_json(result_root / "evaluation_results.json")
        metadata = load_json(result_root / "metadata.json")
        summary = load_json(result_root / "diagnostics" / "summary.json")
        events = load_events(result_root / "diagnostics" / "events.jsonl")
        integrity = result["integrity"]
        future_split = metadata["future_split"]
        flow_events = [
            event for event in events
            if event.get("event") == "flow_depth_state_update"
        ]
        flow_destinations = np.asarray(
            [event["image"]["destination_frame"] for event in flow_events],
            dtype=np.int64,
        )
        flow_sources = np.asarray(
            [event["image"]["source_frame"] for event in flow_events],
            dtype=np.int64,
        )
        training_events = [
            event for event in flow_events
            if int(event["image"]["destination_frame"]) < FUTURE_START
        ]
        future_events = [
            event for event in flow_events
            if int(event["image"]["destination_frame"]) >= FUTURE_START
        ]
        late_material_commits = [
            event for event in future_events
            if event.get("material", {}).get("status") == "committed"
        ]

        method_gates = {
            "protocol_is_future_80to20": result.get("protocol") == "future_80to20",
            "training_end_is_1151": (
                future_split.get("train_end_inclusive") == FUTURE_START - 1
            ),
            "future_start_is_1152": (
                future_split.get("test_start_inclusive") == FUTURE_START
            ),
            "prediction_end_is_1439": (
                integrity.get("prediction_end_inclusive") == SEQUENCE_END
            ),
            "complete_track_schedule": (
                integrity.get("complete_track_schedule") is True
            ),
            "scored_observations_withheld": (
                integrity.get("scored_observations_withheld") is True
            ),
            "render_partition_exact": (
                integrity.get("render_partition_exact") is True
            ),
        }
        if method == "pure_pbd":
            method_gates["no_flow_depth_events"] = len(flow_events) == 0
        else:
            method_gates.update(
                {
                    "full_f1_event_schedule_1_1439": np.array_equal(
                        flow_destinations, expected_all_destinations
                    ),
                    "consecutive_f1_source_destination_pairs": np.array_equal(
                        flow_sources, flow_destinations - 1
                    ),
                    "training_updates_exact_1_1151": (
                        np.array_equal(
                            np.asarray(
                                [
                                    event["image"]["destination_frame"]
                                    for event in training_events
                                ],
                                dtype=np.int64,
                            ),
                            expected_training_destinations,
                        )
                        and all(
                            event["image"].get("accepted") is True
                            for event in training_events
                        )
                    ),
                    "future_events_exact_1152_1439": np.array_equal(
                        np.asarray(
                            [
                                event["image"]["destination_frame"]
                                for event in future_events
                            ],
                            dtype=np.int64,
                        ),
                        expected_future_destinations,
                    ),
                    "future_visual_updates_all_withheld": all(
                        event["image"].get("status")
                        == "withheld_by_causal_protocol"
                        and event["image"].get("accepted") is False
                        for event in future_events
                    ),
                    "no_future_material_commit": not late_material_commits,
                }
            )

        evaluation[method] = {
            "event_type_counts": summary.get("event_type_counts", {}),
            "flow_depth_events": len(flow_events),
            "accepted_training_flow_updates": sum(
                event["image"].get("accepted") is True
                for event in training_events
            ),
            "withheld_future_flow_updates": sum(
                event["image"].get("status")
                == "withheld_by_causal_protocol"
                for event in future_events
            ),
            "future_material_commits": len(late_material_commits),
            "gates": method_gates,
        }
        gates.update(
            {
                f"evaluation_{method}_{name}": value
                for name, value in method_gates.items()
            }
        )

    report = {
        "schema": "super_grasp5_future_f1_full_validation_v1",
        "asset_root": str(args.asset_root.resolve()),
        "evaluation_root": str(args.evaluation_root.resolve()),
        "input_f1_frame_range": [0, SEQUENCE_END],
        "accepted_training_destination_range": [1, FUTURE_START - 1],
        "strictly_withheld_future_range": [FUTURE_START, SEQUENCE_END],
        "evaluation": evaluation,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("grasp5 full-f1 Future causal validation failed")


if __name__ == "__main__":
    main()
