#!/usr/bin/env python3
"""Score Reconstruction and Future from one causally unified SUPER rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from score_super_tissue_evaluation import (
    compute_lpips,
    distribution,
    formatted,
    tap_position_accuracy,
)


PROTOCOL = "joint_reconstruction_7to1_future_80to20"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracking_metrics(
    frames: list[int],
    *,
    gt: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    gt_lookup: dict[int, int],
    pred_lookup: dict[int, int],
) -> dict:
    gt_slots = np.asarray([gt_lookup[frame] for frame in frames], dtype=np.int64)
    pred_slots = np.asarray(
        [pred_lookup[frame] for frame in frames], dtype=np.int64
    )
    visible = gt["visible"][gt_slots].astype(bool)
    difference_2d = prediction["uv"][pred_slots] - gt["uv"][gt_slots]
    error_2d = np.linalg.norm(difference_2d, axis=-1)
    error_2d[~visible] = np.nan
    valid_3d = gt["valid_3d"][gt_slots].astype(bool)
    difference_3d = (
        prediction["xyz_camera_m"][pred_slots]
        - gt["xyz_camera_m"][gt_slots]
    )
    error_3d = np.linalg.norm(difference_3d, axis=-1)
    error_3d[~valid_3d] = np.nan
    point_count = error_2d.shape[1]
    return {
        "scored_frame_count": len(frames),
        "scored_frames": frames,
        "2d_error_px": distribution(error_2d),
        "2d_tap_position_accuracy": tap_position_accuracy(error_2d),
        "3d_error_mm": distribution(error_3d, scale=1000.0),
        "3d_coverage": {
            "valid": int(valid_3d.sum()),
            "visible": int(visible.sum()),
            "fraction": float(valid_3d.sum() / max(1, visible.sum())),
        },
        "per_point": [
            {
                "point_id": point_id,
                "2d_error_px": distribution(error_2d[:, point_id]),
                "3d_error_mm": distribution(
                    error_3d[:, point_id], scale=1000.0
                ),
            }
            for point_id in range(point_count)
        ],
    }


def rendering_metrics(
    frames: list[int],
    *,
    render_lookup: dict[int, dict],
    lpips_lookup: dict[int, float],
) -> dict:
    records = [render_lookup[frame] for frame in frames]
    return {
        "mse": float(np.mean([record["mse"] for record in records])),
        "psnr_db": float(
            np.mean([record["psnr_db"] for record in records])
        ),
        "ssim": float(np.mean([record["ssim"] for record in records])),
        "lpips_alex": float(np.mean([lpips_lookup[frame] for frame in frames])),
        "frame_count": len(frames),
        "per_frame_lpips": [
            {"frame_index": frame, "lpips_alex": lpips_lookup[frame]}
            for frame in frames
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--lpips-device", default="cuda:0")
    args = parser.parse_args()

    capture = args.capture.resolve()
    metadata = json.loads((capture / "metadata.json").read_text())
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected {PROTOCOL}, got {metadata.get('protocol')}")
    if sha256(args.ground_truth) != metadata["ground_truth_sha256"]:
        raise ValueError("Scored ground truth differs from capture input")

    with np.load(args.ground_truth, allow_pickle=False) as archive:
        gt = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(capture / "predicted_tracks.npz", allow_pickle=False) as archive:
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
    if common != sorted(gt_lookup):
        raise ValueError("Predicted tracks do not cover the frozen GT schedule")

    future_start = int(metadata["future_split"]["test_start_inclusive"])
    phase = int(
        metadata["reconstruction_split"]["test_rule"].rsplit(" ", 1)[-1]
    )
    query_frame = int(prediction["query_frame_index"].item())
    reconstruction_track_frames = [
        frame
        for frame in common
        if frame < future_start and frame % 8 == phase and frame != query_frame
    ]
    future_track_frames = [
        frame for frame in common if frame >= future_start and frame != query_frame
    ]
    if not reconstruction_track_frames or not future_track_frames:
        raise ValueError("Both joint evaluation partitions must contain GT frames")

    observation_used = {
        int(frame): bool(prediction["observation_used"][slot])
        for frame, slot in pred_lookup.items()
    }
    scored_track_frames = reconstruction_track_frames + future_track_frames
    if any(observation_used[frame] for frame in scored_track_frames):
        raise ValueError("A scored tracking frame was exposed as observation")

    render_partial = json.loads(
        (capture / "render_metrics_partial.json").read_text()
    )
    render_records = render_partial["records"]
    render_frames = [int(record["frame_index"]) for record in render_records]
    if len(render_frames) != len(set(render_frames)):
        raise ValueError("Render metric frames are not unique")
    prediction_end = max(pred_lookup)
    expected_reconstruction_renders = [
        frame
        for frame in range(min(pred_lookup), future_start)
        if frame % 8 == phase
    ]
    expected_future_renders = list(range(future_start, prediction_end + 1))
    expected_renders = expected_reconstruction_renders + expected_future_renders
    if render_frames != expected_renders:
        raise ValueError("Renders do not match the joint scored partitions")
    if any(bool(record["observation_used"]) for record in render_records):
        raise ValueError("A scored render frame was exposed as observation")

    lpips_records, _lpips_mean = compute_lpips(
        capture / "render_pairs", render_frames, args.lpips_device
    )
    lpips_lookup = {
        int(record["frame_index"]): float(record["lpips_alex"])
        for record in lpips_records
    }
    render_lookup = {
        int(record["frame_index"]): record for record in render_records
    }

    partitions = {
        "reconstruction_7to1_within_train80": {
            "point_tracking": tracking_metrics(
                reconstruction_track_frames,
                gt=gt,
                prediction=prediction,
                gt_lookup=gt_lookup,
                pred_lookup=pred_lookup,
            ),
            "rendering": rendering_metrics(
                expected_reconstruction_renders,
                render_lookup=render_lookup,
                lpips_lookup=lpips_lookup,
            ),
        },
        "future_80to20": {
            "point_tracking": tracking_metrics(
                future_track_frames,
                gt=gt,
                prediction=prediction,
                gt_lookup=gt_lookup,
                pred_lookup=pred_lookup,
            ),
            "rendering": rendering_metrics(
                expected_future_renders,
                render_lookup=render_lookup,
                lpips_lookup=lpips_lookup,
            ),
        },
    }
    report = {
        "schema": "super_tissue_joint_evaluation_results_v1",
        "protocol": PROTOCOL,
        "ground_truth": str(args.ground_truth.resolve()),
        "capture": str(capture),
        "partitions": partitions,
        "integrity": {
            "ground_truth_hash_matches_capture": True,
            "complete_track_schedule": True,
            "scored_observations_withheld": True,
            "render_partition_exact": True,
            "training_range": [0, future_start - 1],
            "future_range": [future_start, prediction_end],
            "reconstruction_holdout_phase": phase,
            "reconstruction_render_frame_count": len(
                expected_reconstruction_renders
            ),
            "future_render_frame_count": len(expected_future_renders),
        },
        "directions": {
            "2d_error_px": "lower",
            "3d_error_mm": "lower",
            "psnr_db": "higher",
            "ssim": "higher",
            "lpips_alex": "lower",
        },
    }
    (capture / "evaluation_results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    lines = [
        "# SUPER joint 80/20 evaluation",
        "",
        "Both rows come from the same continuous rollout.",
        "",
        "| Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in (
        ("reconstruction_7to1_within_train80", "Reconstruction: 7:1 in first 80%"),
        ("future_80to20", "Future: final 20% open-loop"),
    ):
        item = partitions[name]
        track = item["point_tracking"]
        render = item["rendering"]
        lines.append(
            f"| {label} | {formatted(track['3d_error_mm']['mean'])} / "
            f"{formatted(track['3d_error_mm']['rmse'])} | "
            f"{formatted(track['2d_error_px']['mean'])} / "
            f"{formatted(track['2d_error_px']['rmse'])} | "
            f"{formatted(render['psnr_db'])} | {formatted(render['ssim'], 4)} | "
            f"{formatted(render['lpips_alex'], 4)} |"
        )
    (capture / "evaluation_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
