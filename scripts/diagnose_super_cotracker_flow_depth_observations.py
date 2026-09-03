#!/usr/bin/env python3
"""Build non-mutating grasp5 CoTracker + stereo-depth 3D flow observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_gaussians.physics_simulator.flow_depth_particle_observer import (  # noqa: E402
    FlowDepthObservationSettings,
    observe_flow_depth_tracks,
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
    / "data/super/grasp5_native/depth_v4_foundation_dense_timestamped"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/grasp5_cotracker3_flow_depth_observations_20260828_v1"
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
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_native/rgb",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-end-frame", type=int, default=1151)
    parser.add_argument("--maximum-depth-change-mm", type=float, default=15.0)
    parser.add_argument("--maximum-observed-flow-mm", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def five_number(values: np.ndarray, scale: float = 1.0) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)] * scale
    if not len(finite):
        return None
    return np.percentile(finite, [0, 5, 50, 95, 100]).tolist()


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


def depth_confidence(
    depth_dir: Path, frame: int, pixels_uv: np.ndarray
) -> np.ndarray:
    path = depth_dir / f"{frame:06d}-confidence.npz"
    with np.load(path, allow_pickle=False) as loaded:
        level = sample_nearest(loaded["confidence_level"], pixels_uv).astype(
            np.int32
        )
    weights = np.asarray([0.0, 0.35, 0.70, 1.0], dtype=np.float32)
    return weights[np.clip(level, 0, 3)]


def draw_overlay(
    *,
    rgb_dir: Path,
    current_frame: int,
    next_frame: int,
    current_pixels: np.ndarray,
    next_pixels: np.ndarray,
    valid: np.ndarray,
    flow_norm_m: np.ndarray,
    output_path: Path,
) -> None:
    image = cv2.imread(
        str(rgb_dir / f"{next_frame:06d}-left.png"), cv2.IMREAD_COLOR
    )
    if image is None:
        raise FileNotFoundError(next_frame)
    maximum = max(float(flow_norm_m[valid].max(initial=1.0e-12)), 1.0e-12)
    normalized = np.clip(flow_norm_m / maximum, 0.0, 1.0)
    colors = cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8)[:, None],
        cv2.COLORMAP_TURBO,
    )[:, 0]
    for track_id in np.flatnonzero(valid):
        start = tuple(np.rint(current_pixels[track_id]).astype(int))
        end = tuple(np.rint(next_pixels[track_id]).astype(int))
        color = tuple(int(value) for value in colors[track_id])
        cv2.arrowedLine(image, start, end, color, 1, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(image, end, 2, color, -1, cv2.LINE_AA)
    label = (
        f"CoTracker + depth 3D flow | {current_frame}->{next_frame} | "
        f"valid {int(valid.sum())}/{len(valid)} | color=3D flow"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write {output_path}")


def main() -> None:
    args = parse_args()
    outputs = (
        args.output_dir / "observations.npz",
        args.output_dir / "report.json",
        args.output_dir / "last_pair_overlay.png",
    )
    collisions = [path for path in outputs if path.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError(
            "Observation outputs already exist; pass --overwrite:\n- "
            + "\n- ".join(str(path) for path in collisions)
        )
    for path in (args.tracks, args.bindings, args.calibration, args.table_frame):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(args.tracks, allow_pickle=False) as loaded:
        source_frames = loaded["source_frame_indices"].astype(np.int32)
        pixels = loaded["tracks_original_px"].astype(np.float64)
        tracker_valid = loaded["visibility"].astype(bool) & loaded[
            "dynamic_tissue_valid"
        ].astype(bool)
    with np.load(args.bindings, allow_pickle=False) as loaded:
        binding_valid = loaded["track_valid"].astype(bool)
    if pixels.shape[:2] != tracker_valid.shape:
        raise ValueError("Track positions and validity disagree")
    if binding_valid.shape != (pixels.shape[1],):
        raise ValueError("Bindings and CoTracker track count disagree")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    table_frame = json.loads(args.table_frame.read_text(encoding="utf-8"))
    intrinsic = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    x_table_camera = np.asarray(table_frame["X_table_camera"], dtype=np.float64)
    settings = FlowDepthObservationSettings(
        maximum_depth_sampling_radius_px=2,
        minimum_depth_m=0.035,
        maximum_depth_m=0.250,
        maximum_depth_change_m=args.maximum_depth_change_mm / 1000.0,
        maximum_observed_flow_m=args.maximum_observed_flow_mm / 1000.0,
    )

    available = np.asarray(
        [
            frame <= args.training_end_frame
            and (args.depth_dir / f"{frame:06d}-depth.npy").is_file()
            and (args.depth_dir / f"{frame:06d}-confidence.npz").is_file()
            for frame in source_frames
        ],
        dtype=bool,
    )
    pair_indices = [
        index
        for index in range(len(source_frames) - 1)
        if available[index] and available[index + 1]
    ]
    if not pair_indices:
        raise RuntimeError("No consecutive CoTracker samples have depth")

    observations = []
    requested_counts = []
    for index in pair_indices:
        current_frame = int(source_frames[index])
        next_frame = int(source_frames[index + 1])
        current_depth = np.load(
            args.depth_dir / f"{current_frame:06d}-depth.npy",
            allow_pickle=False,
        )
        next_depth = np.load(
            args.depth_dir / f"{next_frame:06d}-depth.npy", allow_pickle=False
        )
        current_confidence = depth_confidence(
            args.depth_dir, current_frame, pixels[index]
        )
        next_confidence = depth_confidence(
            args.depth_dir, next_frame, pixels[index + 1]
        )
        confidence = np.minimum(current_confidence, next_confidence)
        current_valid = tracker_valid[index] & binding_valid
        next_valid = tracker_valid[index + 1] & binding_valid
        requested_counts.append(int(np.count_nonzero(current_valid & next_valid)))
        observations.append(
            observe_flow_depth_tracks(
                current_pixels_uv=pixels[index],
                next_pixels_uv=pixels[index + 1],
                current_depth=current_depth,
                next_depth=next_depth,
                intrinsic=intrinsic,
                x_table_camera=x_table_camera,
                current_track_valid=current_valid,
                next_track_valid=next_valid,
                confidence=confidence,
                settings=settings,
            )
        )

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
    flow_norm = np.linalg.norm(observed_flow, axis=2)
    valid_flow_norm = flow_norm[valid]
    requested_total = int(sum(requested_counts))
    accepted_total = int(valid.sum())
    acceptance_fraction = accepted_total / max(requested_total, 1)
    pair_current_frames = source_frames[pair_indices]
    pair_next_frames = source_frames[np.asarray(pair_indices) + 1]
    gates = {
        "at_least_one_depth_pair": bool(len(pair_indices) >= 1),
        "future_boundary_not_used": bool(
            np.all(pair_next_frames <= args.training_end_frame)
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
        "no_simulator_state_was_loaded_or_written": True,
    }
    report = {
        "schema": "super_cotracker_flow_depth_observations_v1",
        "passed": bool(all(gates.values())),
        "scope": "non-mutating observation diagnostic; no PBD innovation yet",
        "pairs": [
            [int(current), int(next_)]
            for current, next_ in zip(
                pair_current_frames, pair_next_frames, strict=True
            )
        ],
        "counts": {
            "tracks": int(pixels.shape[1]),
            "fixed_bound_tracks": int(binding_valid.sum()),
            "depth_pairs": len(pair_indices),
            "requested_track_pair_observations": requested_total,
            "accepted_track_pair_observations": accepted_total,
            "acceptance_fraction": acceptance_fraction,
        },
        "statistics": {
            "observed_flow_mm_min_p05_p50_p95_max": five_number(
                valid_flow_norm, 1000.0
            ),
            "absolute_depth_change_mm_min_p05_p50_p95_max": five_number(
                np.abs(next_depth - current_depth)[valid], 1000.0
            ),
            "confidence_min_p05_p50_p95_max": five_number(confidence[valid]),
        },
        "settings": {
            "maximum_depth_change_mm": args.maximum_depth_change_mm,
            "maximum_observed_flow_mm": args.maximum_observed_flow_mm,
            "training_end_frame": args.training_end_frame,
            "confidence_level_weights": [0.0, 0.35, 0.70, 1.0],
        },
        "gates": gates,
        "inputs": {
            "tracks": {"path": str(args.tracks.resolve()), "sha256": sha256(args.tracks)},
            "bindings": {
                "path": str(args.bindings.resolve()),
                "sha256": sha256(args.bindings),
            },
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
        outputs[0],
        schema=np.asarray("super_cotracker_flow_depth_observations_v1"),
        current_source_frames=pair_current_frames,
        next_source_frames=pair_next_frames,
        track_valid=valid,
        confidence=confidence,
        observed_flow_table=observed_flow,
        current_points_table=current_points,
        next_points_table=next_points,
        current_depth_m=current_depth,
        next_depth_m=next_depth,
    )
    last = len(observations) - 1
    draw_overlay(
        rgb_dir=args.rgb_dir,
        current_frame=int(pair_current_frames[last]),
        next_frame=int(pair_next_frames[last]),
        current_pixels=pixels[pair_indices[last]],
        next_pixels=pixels[pair_indices[last] + 1],
        valid=valid[last],
        flow_norm_m=flow_norm[last],
        output_path=outputs[2],
    )
    outputs[1].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("CoTracker flow-depth observation diagnostic failed")


if __name__ == "__main__":
    main()
