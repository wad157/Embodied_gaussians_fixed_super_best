#!/usr/bin/env python3
"""Lift completed SUPER manual tissue tracks to strict stereo 3D ground truth.

The manual JSON remains the source of 2D truth.  A 3D observation is emitted
only when a finite FoundationStereo depth pixel also passed left/right
consistency and the independent RAFT-Stereo agreement gate.  No temporal
interpolation or dense-depth fallback is used for the scored 3D points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data/super/evaluation_v1/manual_tissue_tracks_10"
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ROOT / "annotations.json")
    parser.add_argument("--depth-dir", type=Path, default=DEFAULT_ROOT / "stereo_depth_v1")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_native/calib_rectified.json",
    )
    parser.add_argument(
        "--cameras",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_offline_demo/cameras.json",
    )
    parser.add_argument("--maximum-sampling-radius-px", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "ground_truth_2d3d_v1.npz")
    parser.add_argument("--report", type=Path, default=DEFAULT_ROOT / "ground_truth_2d3d_v1.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def strict_depth_at(depth: np.ndarray, uv: np.ndarray, maximum_radius: int) -> tuple[float, int]:
    """Return the nearest finite strict depth, preferring the exact pixel."""
    height, width = depth.shape
    u = int(round(float(uv[0])))
    v = int(round(float(uv[1])))
    if not (0 <= u < width and 0 <= v < height):
        return float("nan"), -1
    for radius in range(maximum_radius + 1):
        x0, x1 = max(0, u - radius), min(width, u + radius + 1)
        y0, y1 = max(0, v - radius), min(height, v + radius + 1)
        patch = depth[y0:y1, x0:x1]
        finite = np.isfinite(patch) & (patch > 0.0)
        if bool(finite.any()):
            return float(np.median(patch[finite])), radius
    return float("nan"), -1


def main() -> None:
    args = parse_args()
    if args.maximum_sampling_radius_px < 0:
        raise ValueError("maximum sampling radius must be non-negative")
    annotations = read_json(args.annotations)
    if annotations.get("status") != "complete":
        raise ValueError("Manual annotations are not marked complete")
    scheduled = [int(value) for value in annotations["sampling"]["scheduled_frames"]]
    records = annotations["frames"]
    if len(records) != len(scheduled):
        raise ValueError("Manual annotation schedule is incomplete")
    calibration = read_json(args.calibration)
    annotation_calibration_hash = annotations["source"]["stereo_calibration"]["sha256"]
    if sha256(args.calibration) != annotation_calibration_hash:
        raise ValueError("Calibration differs from the file frozen by manual annotation")
    depth_summary_path = args.depth_dir / "depth_generation_summary.json"
    depth_summary = read_json(depth_summary_path)
    depth_frames = [int(value) for value in depth_summary["parameters"]["frames"]]
    if depth_frames != scheduled:
        raise ValueError("Stereo depth frame schedule differs from manual annotations")
    if depth_summary["calibration"]["sha256"] != annotation_calibration_hash:
        raise ValueError("Stereo depth used a different calibration")
    if not np.isclose(float(depth_summary["parameters"]["lr_threshold_px"]), 1.5):
        raise ValueError("Expected the frozen 1.5 px stereo LR threshold")
    if not np.isclose(float(depth_summary["parameters"]["model_agreement_mm"]), 3.0):
        raise ValueError("Expected the frozen 3 mm cross-model depth gate")
    cameras = read_json(args.cameras)
    intrinsic = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    image_size = np.asarray(calibration["image_size"], dtype=np.int32)
    x_world_camera_blender = np.asarray(cameras["stereo_left"]["X_WC"], dtype=np.float64)
    x_world_camera_opencv = x_world_camera_blender @ BLENDER_TO_OPENCV

    point_count = int(annotations["point_count"])
    frame_count = len(scheduled)
    uv = np.full((frame_count, point_count, 2), np.nan, dtype=np.float32)
    visible = np.zeros((frame_count, point_count), dtype=bool)
    xyz_camera = np.full((frame_count, point_count, 3), np.nan, dtype=np.float32)
    xyz_world = np.full_like(xyz_camera, np.nan)
    valid_3d = np.zeros((frame_count, point_count), dtype=bool)
    sampling_radius = np.full((frame_count, point_count), -1, dtype=np.int8)
    timestamps = np.empty(frame_count, dtype=np.float64)

    for frame_slot, frame_index in enumerate(scheduled):
        record = records.get(f"{frame_index:06d}")
        if record is None or not record.get("human_verified", False):
            raise ValueError(f"Frame {frame_index} is not human verified")
        timestamps[frame_slot] = float(record["timestamp_s"])
        observations = {int(item["point_id"]): item for item in record["observations"]}
        if set(observations) != set(range(point_count)):
            raise ValueError(f"Frame {frame_index} does not contain every point")
        depth_path = args.depth_dir / f"{frame_index:06d}-depth_high_confidence.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        depth = np.load(depth_path, allow_pickle=False)
        expected_shape = (int(image_size[1]), int(image_size[0]))
        if depth.shape != expected_shape:
            raise ValueError(f"Depth shape {depth.shape} != {expected_shape} at frame {frame_index}")
        for point_id, observation in observations.items():
            status = str(observation["status"])
            if status != "visible":
                continue
            point_uv = np.asarray(observation["uv"], dtype=np.float64)
            if point_uv.shape != (2,) or not np.isfinite(point_uv).all():
                raise ValueError(f"Invalid visible UV at frame {frame_index}, point {point_id}")
            uv[frame_slot, point_id] = point_uv
            visible[frame_slot, point_id] = True
            z, radius = strict_depth_at(depth, point_uv, args.maximum_sampling_radius_px)
            if not np.isfinite(z):
                continue
            camera_point = np.array(
                [
                    (point_uv[0] - intrinsic[0, 2]) * z / intrinsic[0, 0],
                    (point_uv[1] - intrinsic[1, 2]) * z / intrinsic[1, 1],
                    z,
                    1.0,
                ],
                dtype=np.float64,
            )
            world_point = x_world_camera_opencv @ camera_point
            xyz_camera[frame_slot, point_id] = camera_point[:3]
            xyz_world[frame_slot, point_id] = world_point[:3]
            valid_3d[frame_slot, point_id] = True
            sampling_radius[frame_slot, point_id] = radius

    train_mask = np.asarray(scheduled, dtype=np.int32) <= 1151
    test_mask = ~train_mask
    radius_counts = {
        str(radius): int((sampling_radius == radius).sum())
        for radius in range(args.maximum_sampling_radius_px + 1)
    }
    report = {
        "schema": "super_tissue_evaluation_ground_truth_v1",
        "passed": True,
        "annotations": {"path": str(args.annotations.resolve()), "sha256": sha256(args.annotations)},
        "depth": {
            "directory": str(args.depth_dir.resolve()),
            "summary": str(depth_summary_path.resolve()),
            "summary_sha256": sha256(depth_summary_path),
            "source_pattern": "NNNNNN-depth_high_confidence.npy",
            "policy": "strict FoundationStereo LR consistency plus RAFT agreement; no temporal interpolation or dense fallback",
            "maximum_sampling_radius_px": args.maximum_sampling_radius_px,
        },
        "coordinate_frames": {
            "uv": "full-resolution rectified stereo-left pixels",
            "xyz_camera": "rectified stereo-left OpenCV camera frame, metres",
            "xyz_world": "frozen SUPER table/world frame, metres",
        },
        "frame_count": frame_count,
        "point_count": point_count,
        "visible_2d_count": int(visible.sum()),
        "valid_3d_count": int(valid_3d.sum()),
        "valid_3d_fraction_of_visible": float(valid_3d.sum() / max(1, visible.sum())),
        "train_valid_3d_count": int(valid_3d[train_mask].sum()),
        "future_test_valid_3d_count": int(valid_3d[test_mask].sum()),
        "sampling_radius_counts": radius_counts,
        "output": str(args.output.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("super_tissue_evaluation_ground_truth_v1"),
        frame_indices=np.asarray(scheduled, dtype=np.int32),
        timestamps_s=timestamps,
        point_ids=np.arange(point_count, dtype=np.int32),
        uv=uv,
        visible=visible,
        xyz_camera_m=xyz_camera,
        xyz_world_m=xyz_world,
        valid_3d=valid_3d,
        depth_sampling_radius_px=sampling_radius,
        K_left_rect=intrinsic,
        X_world_camera_opencv=x_world_camera_opencv,
        future_train_end_frame=np.asarray(1151, dtype=np.int32),
        future_test_start_frame=np.asarray(1152, dtype=np.int32),
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
