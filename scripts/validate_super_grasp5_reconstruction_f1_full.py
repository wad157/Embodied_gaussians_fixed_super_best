#!/usr/bin/env python3
"""Validate that grasp5 Reconstruction f1 covers every frame 0..1439."""

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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tracks_root = args.asset_root / "tracks"
    bindings_root = args.asset_root / "bindings"
    observations_root = args.asset_root / "observations"
    track_report = load_json(tracks_root / "report.json")
    binding_report = load_json(bindings_root / "report.json")
    observation_report = load_json(observations_root / "report.json")

    with np.load(tracks_root / "tracks.npz", allow_pickle=False) as archive:
        track_frames = archive["source_frame_indices"].astype(np.int64)
    with np.load(bindings_root / "bindings.npz", allow_pickle=False) as archive:
        binding_frames = archive["source_frame_indices"].astype(np.int64)
    with np.load(observations_root / "observations.npz", allow_pickle=False) as archive:
        current_frames = archive["current_source_frames"].astype(np.int64)
        next_frames = archive["next_source_frames"].astype(np.int64)

    expected_frames = np.arange(1440, dtype=np.int64)
    expected_current = np.arange(1439, dtype=np.int64)
    expected_next = np.arange(1, 1440, dtype=np.int64)
    gates: dict[str, bool] = {
        "track_report_passed": track_report.get("passed") is True,
        "binding_report_passed": binding_report.get("passed") is True,
        "observation_report_passed": observation_report.get("passed") is True,
        "track_report_sampled_frames_1440": (
            track_report.get("counts", {}).get("sampled_frames") == 1440
        ),
        "track_report_stride_1": (
            track_report.get("metadata", {}).get("frame_stride") == 1
        ),
        "track_report_maximum_source_frame_1439": (
            track_report.get("settings", {}).get("maximum_source_frame") == 1439
        ),
        "track_npz_exact_0_1439": np.array_equal(track_frames, expected_frames),
        "binding_npz_exact_0_1439": np.array_equal(
            binding_frames, expected_frames
        ),
        "observation_report_track_frames_1440": (
            observation_report.get("counts", {}).get("track_frames") == 1440
        ),
        "observation_report_pairs_1439": (
            observation_report.get("counts", {}).get("observation_pairs") == 1439
        ),
        "observation_report_first_0": (
            observation_report.get("frame_range", {}).get("first") == 0
        ),
        "observation_report_last_1439": (
            observation_report.get("frame_range", {}).get("last") == 1439
        ),
        "observation_report_end_1439": (
            observation_report.get("frame_range", {}).get("training_end_frame")
            == 1439
        ),
        "observation_npz_current_exact_0_1438": np.array_equal(
            current_frames, expected_current
        ),
        "observation_npz_next_exact_1_1439": np.array_equal(
            next_frames, expected_next
        ),
    }

    evaluation: dict[str, dict] = {}
    if args.evaluation_root is not None:
        for method in METHODS:
            result_root = args.evaluation_root / method / "reconstruction_7to1"
            result = load_json(result_root / "evaluation_results.json")
            summary = load_json(result_root / "diagnostics" / "summary.json")
            integrity = result["integrity"]
            flow_updates = int(
                summary.get("event_type_counts", {}).get(
                    "flow_depth_state_update", 0
                )
            )
            expected_updates = 0 if method == "pure_pbd" else 1439
            method_gates = {
                "ground_truth_hash_matches_capture": (
                    integrity.get("ground_truth_hash_matches_capture") is True
                ),
                "complete_track_schedule": (
                    integrity.get("complete_track_schedule") is True
                ),
                "render_partition_exact": (
                    integrity.get("render_partition_exact") is True
                ),
                "full_flow_depth_update_schedule": flow_updates == expected_updates,
            }
            evaluation[method] = {
                "flow_depth_state_updates": flow_updates,
                "expected_flow_depth_state_updates": expected_updates,
                "gates": method_gates,
            }
            gates.update(
                {
                    f"evaluation_{method}_{name}": value
                    for name, value in method_gates.items()
                }
            )

    report = {
        "schema": "super_grasp5_reconstruction_f1_full_validation_v1",
        "asset_root": str(args.asset_root.resolve()),
        "evaluation_root": (
            str(args.evaluation_root.resolve())
            if args.evaluation_root is not None
            else None
        ),
        "expected_frame_range": [0, 1439],
        "expected_track_frames": 1440,
        "expected_observation_pairs": 1439,
        "gates": gates,
        "evaluation": evaluation,
        "passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("grasp5 full Reconstruction f1 validation failed")


if __name__ == "__main__":
    main()
