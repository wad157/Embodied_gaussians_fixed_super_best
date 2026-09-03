#!/usr/bin/env python3
"""Score an explicitly non-formal SUPER track-only validation capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from score_super_tissue_evaluation import distribution, tap_position_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(
        (args.capture / "metadata.json").read_text(encoding="utf-8")
    )
    with np.load(args.ground_truth, allow_pickle=False) as archive:
        gt = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(
        args.capture / "predicted_tracks.npz", allow_pickle=False
    ) as archive:
        prediction = {
            name: np.asarray(archive[name]) for name in archive.files
        }
    gt_lookup = {
        int(frame): slot for slot, frame in enumerate(gt["frame_indices"])
    }
    pred_lookup = {
        int(frame): slot
        for slot, frame in enumerate(prediction["frame_indices"])
    }
    common = sorted(set(gt_lookup) & set(pred_lookup))
    protocol = str(metadata["protocol"])
    if protocol == "reconstruction_7to1":
        phase = int(
            metadata["reconstruction_split"]["test_rule"].rsplit(" ", 1)[-1]
        )
        frames = [frame for frame in common if frame != 0 and frame % 8 == phase]
    elif protocol == "future_80to20":
        start = int(metadata["future_split"]["test_start_inclusive"])
        frames = [frame for frame in common if frame >= start]
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")
    if not frames:
        raise ValueError("Validation capture has no scored frames")
    gt_slots = np.asarray([gt_lookup[frame] for frame in frames], dtype=np.int64)
    pred_slots = np.asarray(
        [pred_lookup[frame] for frame in frames], dtype=np.int64
    )
    if np.any(prediction["observation_used"][pred_slots]):
        raise ValueError("A validation frame used an observation")
    visible = gt["visible"][gt_slots].astype(bool)
    error_2d = np.linalg.vector_norm(
        prediction["uv"][pred_slots] - gt["uv"][gt_slots], axis=2
    )
    error_2d[~visible] = np.nan
    valid_3d = gt["valid_3d"][gt_slots].astype(bool)
    error_3d = np.linalg.vector_norm(
        prediction["xyz_camera_m"][pred_slots]
        - gt["xyz_camera_m"][gt_slots],
        axis=2,
    )
    error_3d[~valid_3d] = np.nan
    report = {
        "schema": "super_track_only_validation_v1",
        "formal_result": False,
        "protocol": protocol,
        "frames": frames,
        "2d_error_px": distribution(error_2d),
        "2d_tap_position_accuracy": tap_position_accuracy(error_2d),
        "3d_error_mm": distribution(error_3d, 1000.0),
        "3d_coverage": float(valid_3d.sum() / max(1, visible.sum())),
        "per_point": [
            {
                "point_id": point_id,
                "2d_error_px": distribution(error_2d[:, point_id]),
                "3d_error_mm": distribution(
                    error_3d[:, point_id], 1000.0
                ),
            }
            for point_id in range(error_2d.shape[1])
        ],
        "integrity": {
            "scored_observations_withheld": True,
            "formal_future_frames_1152_plus_used": False,
        },
    }
    (args.capture / "validation_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
