#!/usr/bin/env python3
"""Build causal tracker flow with sparse stereo-depth anchoring.

Stereo depth is sampled only when an existing FoundationStereo depth frame is
available.  Between anchors, each track forward-holds its latest depth and its
confidence decays with age; future depth is never interpolated backward.  The
input may come from sparse CoTracker tracks or dense AllTracker flow sampled at
physical surface particles; tracker-native confidence is preserved in both
cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_gaussians.physics_simulator.flow_depth_particle_observer import (  # noqa: E402
    FlowDepthObservationSettings,
    observe_flow_depth_track_samples,
    sample_depth_near_pixels,
)


DEFAULT_TRACKS = (
    REPO_ROOT / "outputs/grasp5_cotracker3_motion_left_20260828_v1/tracks.npz"
)
DEFAULT_BINDINGS = (
    REPO_ROOT
    / "outputs/grasp5_cotracker3_range_bindings_20260828_v1/bindings.npz"
)
DEFAULT_DEPTH_DIR = (
    REPO_ROOT
    / "data/super/evaluation_v1/manual_tissue_tracks_10/stereo_depth_v1"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/grasp5_cotracker3_causal_sparse_depth_observations_20260828_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH_DIR)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_native/calib_rectified.json",
    )
    parser.add_argument(
        "--table-frame",
        type=Path,
        default=REPO_ROOT / "data/super/table_frame.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-end-frame", type=int, default=1151)
    parser.add_argument("--depth-hold-decay-frames", type=float, default=10.0)
    parser.add_argument("--maximum-depth-hold-frames", type=int, default=12)
    parser.add_argument("--maximum-depth-change-mm", type=float, default=15.0)
    parser.add_argument("--maximum-observed-flow-mm", type=float, default=20.0)
    parser.add_argument(
        "--minimum-tracking-confidence",
        type=float,
        default=0.0,
        help="Reject tracker observations below this calibrated confidence.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def five_number(values: np.ndarray, scale: float = 1.0) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)] * scale
    if not len(values):
        return None
    return np.percentile(values, [0, 5, 50, 95, 100]).tolist()


def sample_nearest(image: np.ndarray, pixels_uv: np.ndarray) -> np.ndarray:
    pixels = np.rint(pixels_uv).astype(np.int64)
    result = np.zeros(len(pixels), dtype=image.dtype)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < image.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < image.shape[0])
    )
    result[inside] = image[pixels[inside, 1], pixels[inside, 0]]
    return result


def main() -> None:
    args = parse_args()
    output_npz = args.output_dir / "observations.npz"
    report_path = args.output_dir / "report.json"
    collisions = [path for path in (output_npz, report_path) if path.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; pass --overwrite:\n- "
            + "\n- ".join(str(path) for path in collisions)
        )
    if args.depth_hold_decay_frames <= 0.0:
        raise ValueError("depth hold decay must be positive")
    if args.maximum_depth_hold_frames < 0:
        raise ValueError("maximum depth hold age cannot be negative")
    if not 0.0 <= args.minimum_tracking_confidence <= 1.0:
        raise ValueError("minimum tracking confidence must stay in [0, 1]")
    for path in (args.tracks, args.bindings, args.calibration, args.table_frame):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(args.tracks, allow_pickle=False) as loaded:
        track_schema = (
            str(loaded["schema"].item())
            if "schema" in loaded.files
            else "unknown_tracker_tracks"
        )
        source_frames = loaded["source_frame_indices"].astype(np.int32)
        pixels = loaded["tracks_original_px"].astype(np.float64)
        tracker_valid = loaded["visibility"].astype(bool) & loaded[
            "dynamic_tissue_valid"
        ].astype(bool)
        tracker_confidence = (
            loaded["tracking_confidence"].astype(np.float64)
            if "tracking_confidence" in loaded.files
            else np.ones_like(tracker_valid, dtype=np.float64)
        )
    tracker_name = (
        "AllTracker"
        if "alltracker" in track_schema.lower()
        else "CoTracker"
        if "cotracker" in track_schema.lower()
        else "input tracker"
    )
    with np.load(args.bindings, allow_pickle=False) as loaded:
        binding_valid = loaded["track_valid"].astype(bool)
    if pixels.shape[:2] != tracker_valid.shape:
        raise ValueError("Track positions and validity disagree")
    if tracker_confidence.shape != tracker_valid.shape:
        raise ValueError("Tracking confidence and validity disagree")
    if (
        not np.isfinite(tracker_confidence).all()
        or np.any(tracker_confidence < 0.0)
        or np.any(tracker_confidence > 1.0)
    ):
        raise ValueError("Tracking confidence must stay finite in [0, 1]")
    if binding_valid.shape != (pixels.shape[1],):
        raise ValueError("Bindings and tracks disagree on track count")
    tracker_valid &= (
        tracker_confidence >= args.minimum_tracking_confidence
    )

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    table_frame = json.loads(args.table_frame.read_text(encoding="utf-8"))
    intrinsic = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    x_table_camera = np.asarray(
        table_frame["X_table_camera"], dtype=np.float64
    )
    settings = FlowDepthObservationSettings(
        maximum_depth_sampling_radius_px=2,
        minimum_depth_m=0.035,
        maximum_depth_m=0.250,
        maximum_depth_change_m=args.maximum_depth_change_mm / 1000.0,
        maximum_observed_flow_m=args.maximum_observed_flow_mm / 1000.0,
    )
    settings.validate()

    training_indices = np.flatnonzero(source_frames <= args.training_end_frame)
    if len(training_indices) < 2:
        raise RuntimeError("Training interval contains fewer than two track frames")
    track_count = pixels.shape[1]
    held_depth = np.full(track_count, np.nan, dtype=np.float64)
    held_confidence = np.zeros(track_count, dtype=np.float64)
    held_radius = np.full(track_count, -1, dtype=np.int16)
    held_anchor_frame = np.full(track_count, -10**9, dtype=np.int32)
    sampled_depths = []
    sampled_confidences = []
    sampled_radii = []
    sampled_ages = []
    direct_depth_frames = []
    confidence_weights = np.asarray([0.0, 0.35, 0.70, 1.0])

    for source_index in training_indices:
        frame = int(source_frames[source_index])
        depth_path = args.depth_dir / f"{frame:06d}-depth.npy"
        confidence_path = args.depth_dir / f"{frame:06d}-confidence.npz"
        if depth_path.is_file() and confidence_path.is_file():
            depth = np.load(depth_path, allow_pickle=False)
            direct_depth, direct_radius = sample_depth_near_pixels(
                depth,
                pixels[source_index],
                maximum_radius_px=settings.maximum_depth_sampling_radius_px,
            )
            with np.load(confidence_path, allow_pickle=False) as loaded:
                level = sample_nearest(
                    loaded["confidence_level"], pixels[source_index]
                ).astype(np.int32)
            direct_confidence = confidence_weights[np.clip(level, 0, 3)]
            direct_valid = (
                np.isfinite(direct_depth)
                & (direct_depth >= settings.minimum_depth_m)
                & (direct_depth <= settings.maximum_depth_m)
                & (direct_confidence > 0.0)
            )
            held_depth[direct_valid] = direct_depth[direct_valid]
            held_confidence[direct_valid] = direct_confidence[direct_valid]
            held_radius[direct_valid] = direct_radius[direct_valid]
            held_anchor_frame[direct_valid] = frame
            direct_depth_frames.append(frame)
        age = frame - held_anchor_frame
        causal_confidence = held_confidence * np.exp(
            -age / args.depth_hold_decay_frames
        )
        causal_confidence[
            (age < 0) | (age > args.maximum_depth_hold_frames)
        ] = 0.0
        sampled_depths.append(held_depth.copy())
        sampled_confidences.append(causal_confidence)
        sampled_radii.append(held_radius.copy())
        sampled_ages.append(age.copy())

    sampled_depths = np.stack(sampled_depths)
    sampled_confidences = np.stack(sampled_confidences)
    sampled_radii = np.stack(sampled_radii)
    sampled_ages = np.stack(sampled_ages)
    observations = []
    requested_counts = []
    for local_index in range(len(training_indices) - 1):
        current_index = int(training_indices[local_index])
        next_index = int(training_indices[local_index + 1])
        current_valid = tracker_valid[current_index] & binding_valid
        next_valid = tracker_valid[next_index] & binding_valid
        requested_counts.append(int(np.count_nonzero(current_valid & next_valid)))
        observations.append(
            observe_flow_depth_track_samples(
                current_pixels_uv=pixels[current_index],
                next_pixels_uv=pixels[next_index],
                current_depth_m=sampled_depths[local_index],
                next_depth_m=sampled_depths[local_index + 1],
                intrinsic=intrinsic,
                x_table_camera=x_table_camera,
                current_track_valid=current_valid,
                next_track_valid=next_valid,
                confidence=np.minimum(
                    sampled_confidences[local_index],
                    sampled_confidences[local_index + 1],
                )
                * np.minimum(
                    tracker_confidence[current_index],
                    tracker_confidence[next_index],
                ),
                current_depth_sampling_radius_px=sampled_radii[local_index],
                next_depth_sampling_radius_px=sampled_radii[local_index + 1],
                settings=settings,
            )
        )

    current_frames = source_frames[training_indices[:-1]]
    next_frames = source_frames[training_indices[1:]]
    valid = np.stack([value.track_valid for value in observations])
    confidence = np.stack([value.confidence for value in observations])
    observed_flow = np.stack(
        [value.observed_flow_table for value in observations]
    )
    current_points = np.stack(
        [value.current_points_table for value in observations]
    )
    next_points = np.stack([value.next_points_table for value in observations])
    current_depth = np.stack([value.current_depth_m for value in observations])
    next_depth = np.stack([value.next_depth_m for value in observations])
    valid_flow_norm = np.linalg.norm(observed_flow, axis=2)[valid]
    requested_total = int(sum(requested_counts))
    accepted_total = int(valid.sum())
    acceptance_fraction = accepted_total / max(requested_total, 1)
    valid_ages = sampled_ages[
        (sampled_confidences > 0.0) & np.isfinite(sampled_depths)
    ]
    direct_depth_frames = sorted(set(direct_depth_frames))
    gates = {
        "future_boundary_not_used": bool(
            len(next_frames) and np.all(next_frames <= args.training_end_frame)
        ),
        "all_pairs_are_forward_consecutive_tracker_samples": bool(
            np.all(np.diff(current_frames) > 0)
            and np.all(next_frames > current_frames)
        ),
        "no_backward_depth_interpolation": True,
        "depth_hold_age_bounded": bool(
            len(valid_ages)
            and np.all(valid_ages <= args.maximum_depth_hold_frames)
        ),
        "accepted_tracks_are_fixed_bound_tracks": bool(
            np.all(~valid | binding_valid[None, :])
        ),
        "accepted_flow_is_finite": bool(
            len(valid_flow_norm) and np.isfinite(valid_flow_norm).all()
        ),
        "accepted_flow_respects_maximum": bool(
            len(valid_flow_norm)
            and np.all(
                valid_flow_norm <= settings.maximum_observed_flow_m + 1.0e-8
            )
        ),
        "observation_acceptance_above_half": bool(acceptance_fraction > 0.5),
    }
    report = {
        "schema": "super_tracker_causal_sparse_depth_observations_v3",
        "passed": bool(all(gates.values())),
        "method": (
            f"{tracker_name} at every sampled frame; direct stereo depth whenever "
            "available; otherwise last per-track depth is forward-held with "
            "exponential confidence decay; tracker-native confidence is "
            "multiplied into depth confidence; no future-depth interpolation"
        ),
        "tracker": {"name": tracker_name, "input_schema": track_schema},
        "counts": {
            "tracks": int(track_count),
            "fixed_bound_tracks": int(binding_valid.sum()),
            "track_frames": int(len(training_indices)),
            "observation_pairs": int(len(observations)),
            "direct_depth_anchor_frames": int(len(direct_depth_frames)),
            "requested_track_pair_observations": requested_total,
            "accepted_track_pair_observations": accepted_total,
            "acceptance_fraction": acceptance_fraction,
        },
        "frame_range": {
            "first": int(current_frames[0]),
            "last": int(next_frames[-1]),
            "training_end_frame": args.training_end_frame,
            "direct_depth_frames": direct_depth_frames,
        },
        "statistics": {
            "observed_flow_mm_min_p05_p50_p95_max": five_number(
                valid_flow_norm, 1000.0
            ),
            "confidence_min_p05_p50_p95_max": five_number(confidence[valid]),
            "tracking_confidence_min_p05_p50_p95_max": five_number(
                tracker_confidence[tracker_valid]
            ),
            "valid_depth_age_frames_min_p05_p50_p95_max": five_number(
                valid_ages
            ),
        },
        "settings": {
            "depth_hold_decay_frames": args.depth_hold_decay_frames,
            "maximum_depth_hold_frames": args.maximum_depth_hold_frames,
            "maximum_depth_change_mm": args.maximum_depth_change_mm,
            "maximum_observed_flow_mm": args.maximum_observed_flow_mm,
            "minimum_tracking_confidence": (
                args.minimum_tracking_confidence
            ),
            "confidence_level_weights": confidence_weights.tolist(),
        },
        "gates": gates,
        "inputs": {
            "tracks": {"path": str(args.tracks.resolve()), "sha256": sha256(args.tracks)},
            "bindings": {
                "path": str(args.bindings.resolve()),
                "sha256": sha256(args.bindings),
            },
            "depth_dir": str(args.depth_dir.resolve()),
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": sha256(args.calibration),
            },
            "table_frame": {
                "path": str(args.table_frame.resolve()),
                "sha256": sha256(args.table_frame),
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        schema=np.asarray(report["schema"]),
        current_source_frames=current_frames,
        next_source_frames=next_frames,
        track_valid=valid,
        confidence=confidence,
        observed_flow_table=observed_flow,
        current_points_table=current_points,
        next_points_table=next_points,
        current_depth_m=current_depth,
        next_depth_m=next_depth,
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Causal sparse-depth observation gates failed")


if __name__ == "__main__":
    main()
