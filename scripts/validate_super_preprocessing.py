#!/usr/bin/env python3
"""Validate a preprocessed SuPer ROS bag against its native/offline outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.rosbag1 import Reader


REPO = Path(__file__).resolve().parents[1]
LEFT_TOPIC = "/stereo/slave/left/image"
RIGHT_TOPIC = "/stereo/slave/right/image"
JOINT_TOPIC = "/dvrk/PSM1/slave/state_joint_current"
EXPECTED_JOINT_NAMES = [
    "outer_yaw",
    "outer_pitch",
    "outer_insertion",
    "outer_roll",
    "outer_wrist_pitch",
    "outer_wrist_yaw",
    "jaw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("grasp1", "grasp3", "grasp5"))
    parser.add_argument("--bag", type=Path, default=None)
    parser.add_argument("--native-dir", type=Path, default=None)
    parser.add_argument("--offline-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stereo-p95-limit-ms", type=float, default=20.0)
    parser.add_argument("--joint-p95-limit-ms", type=float, default=5.0)
    parser.add_argument("--video-mae-limit", type=float, default=15.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64) * scale
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def nearest_deltas(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(candidates, reference)
    after = np.clip(indices, 0, len(candidates) - 1)
    before = np.clip(indices - 1, 0, len(candidates) - 1)
    after_delta = np.abs(candidates[after] - reference)
    before_delta = np.abs(candidates[before] - reference)
    return np.minimum(after_delta, before_delta)


def frame_indices(rgb_dir: Path, side: str) -> list[int]:
    result = []
    for path in rgb_dir.glob(f"*-{side}.png"):
        prefix = path.name.split("-", 1)[0]
        if prefix.isdigit():
            result.append(int(prefix))
    return sorted(result)


def video_report(video_path: Path, rgb_dir: Path, side: str) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    comparisons: dict[str, dict[str, float | int]] = {}
    try:
        for label, frame_index in (("first", 0), ("last", frame_count - 1)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, decoded = capture.read()
            png = cv2.imread(
                str(rgb_dir / f"{frame_index:06d}-{side}.png"), cv2.IMREAD_COLOR
            )
            if not ok or decoded is None or png is None or decoded.shape != png.shape:
                comparisons[label] = {"frame": frame_index, "mae": float("inf")}
            else:
                comparisons[label] = {
                    "frame": frame_index,
                    "mae": float(
                        np.mean(
                            np.abs(decoded.astype(np.float32) - png.astype(np.float32))
                        )
                    ),
                }
    finally:
        capture.release()
    return {
        "path": str(video_path),
        "frame_count": frame_count,
        "resolution": [width, height],
        "fps": fps,
        "png_comparison": comparisons,
    }


def add_gate(gates: dict[str, bool], name: str, condition: bool) -> None:
    gates[name] = bool(condition)


def main() -> None:
    args = parse_args()
    bag = (args.bag or (REPO.parent / f"{args.dataset}.bag")).resolve()
    native = (args.native_dir or (REPO / "data/super" / f"{args.dataset}_native")).resolve()
    offline = (
        args.offline_dir or (REPO / "data/super" / f"{args.dataset}_offline_demo")
    ).resolve()
    output = args.output or (native / "preprocess_validation_report.json")

    metadata = read_json(native / "super_dataset_metadata.json")
    calibration = read_json(native / "calib_rectified.json")
    joints = read_json(native / "joints.json")
    cameras = read_json(offline / "cameras.json")
    robots = read_json(offline / "robots.json")
    left_video_metadata = read_json(offline / "videos/stereo_left.json")
    right_video_metadata = read_json(offline / "videos/stereo_right.json")

    with Reader(bag) as reader:
        bag_counts = {
            connection.topic: int(connection.msgcount)
            for connection in reader.connections
            if connection.topic in {LEFT_TOPIC, RIGHT_TOPIC, JOINT_TOPIC}
        }
        bag_summary = {
            "path": str(bag),
            "size_bytes": bag.stat().st_size,
            "message_count": int(reader.message_count),
            "duration_seconds": float(reader.duration * 1e-9),
            "target_topic_counts": bag_counts,
        }

    left_ids = frame_indices(native / "rgb", "left")
    right_ids = frame_indices(native / "rgb", "right")
    left_times = np.asarray(calibration["left_timestamps"], dtype=np.float64)
    right_times = np.asarray(calibration["right_timestamps"], dtype=np.float64)
    joint_times = np.asarray(joints["control_timestamps"], dtype=np.float64)
    left_meta_times = np.asarray(left_video_metadata["timestamps"], dtype=np.float64)
    right_meta_times = np.asarray(right_video_metadata["timestamps"], dtype=np.float64)
    stereo_delta_ms = stats(nearest_deltas(left_times, right_times), 1000.0)
    joint_delta_ms = stats(nearest_deltas(left_times, joint_times), 1000.0)

    image_samples: dict[str, dict[str, Any]] = {}
    for side, ids in (("left", left_ids), ("right", right_ids)):
        for frame_index in sorted({ids[0], ids[len(ids) // 2], ids[-1]}):
            path = native / "rgb" / f"{frame_index:06d}-{side}.png"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            image_samples[f"{side}_{frame_index:06d}"] = {
                "path": str(path),
                "readable": image is not None,
                "shape": list(image.shape) if image is not None else None,
                "sha256": sha256(path),
            }

    videos = {
        side: video_report(
            offline / cameras[f"stereo_{side}"]["video_path"], native / "rgb", side
        )
        for side in ("left", "right")
    }
    recovered = metadata.get("recovered_image_details", [])

    p1 = np.asarray(calibration["P1"], dtype=np.float64)
    p2 = np.asarray(calibration["P2"], dtype=np.float64)
    baseline_m = abs(float(p2[0, 3] / p2[0, 0]))
    if baseline_m > 1.0:
        baseline_m *= 1e-3

    gates: dict[str, bool] = {}
    add_gate(gates, "all_target_topics_present", set(bag_counts) == {LEFT_TOPIC, RIGHT_TOPIC, JOINT_TOPIC})
    add_gate(gates, "left_png_count_matches_bag", len(left_ids) == bag_counts.get(LEFT_TOPIC))
    add_gate(gates, "right_png_count_matches_bag", len(right_ids) == bag_counts.get(RIGHT_TOPIC))
    add_gate(gates, "joint_count_matches_bag", len(joint_times) == bag_counts.get(JOINT_TOPIC))
    add_gate(gates, "left_ids_contiguous", left_ids == list(range(len(left_ids))))
    add_gate(gates, "right_ids_contiguous", right_ids == list(range(len(right_ids))))
    add_gate(gates, "left_timestamps_match_pngs", len(left_times) == len(left_ids))
    add_gate(gates, "right_timestamps_match_pngs", len(right_times) == len(right_ids))
    add_gate(gates, "left_timestamps_strictly_increasing", bool(np.all(np.diff(left_times) > 0)))
    add_gate(gates, "right_timestamps_strictly_increasing", bool(np.all(np.diff(right_times) > 0)))
    add_gate(gates, "joint_timestamps_strictly_increasing", bool(np.all(np.diff(joint_times) > 0)))
    add_gate(gates, "left_video_metadata_exact", bool(np.array_equal(left_times, left_meta_times)))
    add_gate(gates, "right_video_metadata_exact", bool(np.array_equal(right_times, right_meta_times)))
    add_gate(gates, "stereo_sync_p95", stereo_delta_ms["p95"] <= args.stereo_p95_limit_ms)
    add_gate(gates, "joint_sync_p95", joint_delta_ms["p95"] <= args.joint_p95_limit_ms)
    add_gate(gates, "joint_schema", joints["states"][0]["names"] == EXPECTED_JOINT_NAMES)
    add_gate(gates, "offline_robot_matches_native", robots.get("PSM1") == joints)
    add_gate(gates, "positive_stereo_baseline", 0.0 < baseline_m < 0.1)
    expected_resolution = list(map(int, calibration["image_size"]))
    add_gate(
        gates,
        "sample_images_readable_at_calibrated_resolution",
        all(
            sample["readable"]
            and sample["shape"][:2] == [expected_resolution[1], expected_resolution[0]]
            for sample in image_samples.values()
        ),
    )
    for side, ids in (("left", left_ids), ("right", right_ids)):
        video = videos[side]
        add_gate(gates, f"{side}_video_count", video["frame_count"] == len(ids))
        add_gate(gates, f"{side}_video_resolution", video["resolution"] == expected_resolution)
        add_gate(
            gates,
            f"{side}_video_matches_png",
            all(
                comparison["mae"] <= args.video_mae_limit
                for comparison in video["png_comparison"].values()
            ),
        )
    add_gate(gates, "recovered_frames_documented", len(recovered) == metadata.get("counts", {}).get("recovered_images", 0))

    report = {
        "schema_version": 1,
        "dataset": args.dataset,
        "native_dir": str(native),
        "offline_dir": str(offline),
        "bag": bag_summary,
        "counts": {
            "left_png": len(left_ids),
            "right_png": len(right_ids),
            "joint_states": len(joint_times),
            "recovered_images": len(recovered),
        },
        "calibration": {
            "image_size": expected_resolution,
            "baseline_m": baseline_m,
            "fx": float(p1[0, 0]),
            "fy": float(p1[1, 1]),
            "cx": float(p1[0, 2]),
            "cy": float(p1[1, 2]),
        },
        "timestamp_sync_ms": {
            "left_to_nearest_right": stereo_delta_ms,
            "left_to_nearest_joint": joint_delta_ms,
        },
        "recovered_image_details": recovered,
        "sample_images": image_samples,
        "videos": videos,
        "provenance_sha256": {
            path.name: sha256(path)
            for path in sorted((native / "provenance").glob("*"))
            if path.is_file()
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"dataset": args.dataset, "output": str(output), "gates": gates, "all_gates_passed": report["all_gates_passed"]}, indent=2))
    if not report["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
