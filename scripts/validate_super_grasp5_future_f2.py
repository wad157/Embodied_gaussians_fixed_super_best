#!/usr/bin/env python3
"""Validate grasp5 Future f2 inputs and strict 80/20 open-loop evaluation."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.bindings, allow_pickle=False) as archive:
        binding_frames = archive["source_frame_indices"].astype(np.int64)
    with np.load(args.observations, allow_pickle=False) as archive:
        current_frames = archive["current_source_frames"].astype(np.int64)
        next_frames = archive["next_source_frames"].astype(np.int64)

    expected_binding_frames = np.arange(0, 1151, 2, dtype=np.int64)
    expected_current_frames = np.arange(0, 1150, 2, dtype=np.int64)
    expected_next_frames = np.arange(2, 1151, 2, dtype=np.int64)
    gates: dict[str, bool] = {
        "binding_frames_exact_f2_0_1150": np.array_equal(
            binding_frames, expected_binding_frames
        ),
        "observation_sources_exact_f2_0_1148": np.array_equal(
            current_frames, expected_current_frames
        ),
        "observation_destinations_exact_f2_2_1150": np.array_equal(
            next_frames, expected_next_frames
        ),
        "all_f2_observations_precede_future_boundary": bool(
            len(next_frames) == 575 and int(next_frames[-1]) < FUTURE_START
        ),
        "evaluation_completion_marker_present": (
            args.evaluation_root / "COMPLETE"
        ).is_file(),
    }
    evaluation: dict[str, dict] = {}
    for method in METHODS:
        result_root = args.evaluation_root / method / "future_80to20"
        result = load_json(result_root / "evaluation_results.json")
        metadata = load_json(result_root / "metadata.json")
        summary = load_json(result_root / "diagnostics" / "summary.json")
        integrity = result["integrity"]
        future_split = metadata["future_split"]
        flow_count = int(
            summary.get("event_type_counts", {}).get(
                "flow_depth_state_update", 0
            )
        )
        expected_flow_count = 0 if method == "pure_pbd" else 575
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
            "flow_event_count_matches_f2_training_schedule": (
                flow_count == expected_flow_count
            ),
        }
        evaluation[method] = {
            "flow_depth_state_updates": flow_count,
            "expected_flow_depth_state_updates": expected_flow_count,
            "gates": method_gates,
        }
        gates.update(
            {
                f"evaluation_{method}_{name}": value
                for name, value in method_gates.items()
            }
        )

    report = {
        "schema": "super_grasp5_future_f2_validation_v1",
        "bindings": str(args.bindings.resolve()),
        "observations": str(args.observations.resolve()),
        "evaluation_root": str(args.evaluation_root.resolve()),
        "training_track_frames": 576,
        "training_observation_pairs": 575,
        "strictly_open_loop_future_range": [FUTURE_START, SEQUENCE_END],
        "evaluation": evaluation,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("grasp5 Future f2 validation failed")


if __name__ == "__main__":
    main()
