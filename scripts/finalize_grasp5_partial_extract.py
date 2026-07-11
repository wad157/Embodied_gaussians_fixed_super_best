#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


LEFT_TOPIC = "/stereo/slave/left/image"
RIGHT_TOPIC = "/stereo/slave/right/image"
JOINT_TOPIC = "/dvrk/PSM1/slave/state_joint_current"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a partially extracted grasp5 dataset by writing calibration, "
            "joint manifests, DatasetManager camera manifests, and MP4 videos from "
            "existing rectified PNGs."
        )
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=repo_root / "data" / "grasp5" / "grasp5.bag",
        help="Path to grasp5.bag.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=repo_root / "data" / "camera_calibration.yaml",
        help="OpenCV stereo calibration YAML.",
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=repo_root / "data" / "super" / "grasp5_native" / "rgb",
        help="Directory containing NNNNNN-left.png and NNNNNN-right.png.",
    )
    parser.add_argument(
        "--native-dir",
        type=Path,
        default=repo_root / "data" / "super" / "grasp5_native",
        help="Native output directory for calib_rectified.json and joints.json.",
    )
    parser.add_argument(
        "--offline-dir",
        type=Path,
        default=repo_root / "data" / "super" / "grasp5_offline_demo",
        help="Offline DatasetManager output directory.",
    )
    parser.add_argument(
        "--robot-name",
        default="PSM1",
        help="Robot key to write in robots.json.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on paired PNG frames to include.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only write JSON manifests; do not re-encode MP4 videos.",
    )
    return parser.parse_args()


def read_opencv_matrix(path: Path, key: str) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {path}")
    value = fs.getNode(key).mat()
    fs.release()
    if value is None:
        raise KeyError(f"Missing matrix {key} in {path}")
    return value


def read_opencv_sequence(path: Path, key: str) -> list[float]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {path}")
    node = fs.getNode(key)
    if node.empty():
        fs.release()
        raise KeyError(f"Missing sequence {key} in {path}")
    values = [node.at(i).real() for i in range(node.size())]
    fs.release()
    return values


def load_rectified_calibration(path: Path) -> dict[str, Any]:
    k1 = read_opencv_matrix(path, "K1")
    k2 = read_opencv_matrix(path, "K2")
    d1 = read_opencv_matrix(path, "D1")
    d2 = read_opencv_matrix(path, "D2")
    r = read_opencv_matrix(path, "R")
    t = np.array(read_opencv_sequence(path, "T"), dtype=np.float64).reshape(3, 1)
    image_size_hw = read_opencv_sequence(path, "ImageSize")
    height, width = int(image_size_hw[0]), int(image_size_hw[1])
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        k1, d1, k2, d2, (width, height), r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    baseline = abs(float(p2[0, 3] / p2[0, 0]))
    if baseline > 1.0:
        baseline *= 1e-3
    return {
        "image_size": [width, height],
        "K1": k1.tolist(),
        "K2": k2.tolist(),
        "D1": d1.reshape(-1).tolist(),
        "D2": d2.reshape(-1).tolist(),
        "R": r.tolist(),
        "T": t.reshape(-1).tolist(),
        "R1": r1.tolist(),
        "R2": r2.tolist(),
        "P1": p1.tolist(),
        "P2": p2.tolist(),
        "Q": q.tolist(),
        "K_left_rect": p1[:3, :3].tolist(),
        "K_right_rect": p2[:3, :3].tolist(),
        "baseline_m": baseline,
    }


def scan_paired_frames(rgb_dir: Path, max_frames: int | None) -> list[int]:
    left = {
        int(path.name.split("-")[0])
        for path in rgb_dir.glob("*-left.png")
        if path.name.split("-")[0].isdigit()
    }
    right = {
        int(path.name.split("-")[0])
        for path in rgb_dir.glob("*-right.png")
        if path.name.split("-")[0].isdigit()
    }
    common = sorted(left & right)
    if not common:
        raise FileNotFoundError(f"No paired left/right PNGs under {rgb_dir}")

    contiguous = []
    expected = common[0]
    for frame_id in common:
        if frame_id != expected:
            break
        contiguous.append(frame_id)
        expected += 1
    if max_frames is not None:
        contiguous = contiguous[:max_frames]
    return contiguous


def ros_time_to_sec(timestamp_ns: int, start_ns: int) -> float:
    return float(timestamp_ns - start_ns) * 1e-9


def extract_bag_times_and_joints(
    bag: Path, num_left: int, num_right: int
) -> tuple[list[float], list[float], dict[str, Any], int]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    left_timestamps: list[float] = []
    right_timestamps: list[float] = []
    joint_timestamps: list[float] = []
    joint_positions: list[list[float]] = []
    joint_velocities: list[list[float]] = []
    joint_efforts: list[list[float]] = []
    joint_names: list[str] | None = None
    first_timestamp_ns: int | None = None

    with Reader(bag) as reader:
        connections = [
            c for c in reader.connections if c.topic in {LEFT_TOPIC, RIGHT_TOPIC, JOINT_TOPIC}
        ]
        for conn, timestamp_ns, raw in reader.messages(connections=connections):
            timestamp_ns = int(timestamp_ns)
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            timestamp = ros_time_to_sec(timestamp_ns, first_timestamp_ns)

            if conn.topic == LEFT_TOPIC:
                if len(left_timestamps) < num_left:
                    left_timestamps.append(timestamp)
                continue
            if conn.topic == RIGHT_TOPIC:
                if len(right_timestamps) < num_right:
                    right_timestamps.append(timestamp)
                continue

            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            if joint_names is None:
                joint_names = [str(name) for name in msg.name]
            joint_timestamps.append(timestamp)
            joint_positions.append([float(v) for v in msg.position])
            joint_velocities.append([float(v) for v in msg.velocity])
            joint_efforts.append([float(v) for v in msg.effort])

    if first_timestamp_ns is None:
        raise RuntimeError(f"No target messages found in {bag}")
    if joint_names is None:
        raise RuntimeError(f"No joint states found in {bag}")
    if len(left_timestamps) < num_left or len(right_timestamps) < num_right:
        raise RuntimeError(
            f"Bag did not contain enough image timestamps: "
            f"left={len(left_timestamps)}/{num_left}, right={len(right_timestamps)}/{num_right}"
        )

    joints = {
        "control": joint_positions,
        "control_timestamps": joint_timestamps,
        "states": [
            {
                "q": q,
                "q_d": qd,
                "effort": effort,
                "names": joint_names,
            }
            for q, qd, effort in zip(joint_positions, joint_velocities, joint_efforts)
        ],
        "states_timestamps": joint_timestamps,
    }
    return left_timestamps, right_timestamps, joints, first_timestamp_ns


def estimate_fps(timestamps: list[float], fallback: float = 30.0) -> float:
    if len(timestamps) < 2:
        return fallback
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        return fallback
    return float(1.0 / np.median(diffs))


def encode_video(
    rgb_dir: Path, frame_ids: list[int], side: str, video_path: Path, fps: float
) -> tuple[int, int]:
    first = cv2.imread(str(rgb_dir / f"{frame_ids[0]:06d}-{side}.png"), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(rgb_dir / f"{frame_ids[0]:06d}-{side}.png")
    height, width = first.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")
    try:
        writer.write(first)
        for frame_id in frame_ids[1:]:
            image_path = rgb_dir / f"{frame_id:06d}-{side}.png"
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(image_path)
            writer.write(frame)
    finally:
        writer.release()
    return width, height


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame_ids = scan_paired_frames(args.rgb_dir, args.max_frames)
    num_frames = len(frame_ids)
    print(f"Using {num_frames} paired contiguous frames: {frame_ids[0]}..{frame_ids[-1]}")

    calib = load_rectified_calibration(args.calibration)
    left_timestamps, right_timestamps, joints, first_timestamp_ns = extract_bag_times_and_joints(
        args.bag, num_frames, num_frames
    )
    left_timestamps = left_timestamps[:num_frames]
    right_timestamps = right_timestamps[:num_frames]

    videos_dir = args.offline_dir / "videos"
    width, height = calib["image_size"]
    if not args.no_video:
        left_fps = estimate_fps(left_timestamps)
        right_fps = estimate_fps(right_timestamps)
        width, height = encode_video(
            args.rgb_dir, frame_ids, "left", videos_dir / "stereo_left.mp4", left_fps
        )
        width_r, height_r = encode_video(
            args.rgb_dir, frame_ids, "right", videos_dir / "stereo_right.mp4", right_fps
        )
        if (width, height) != (width_r, height_r):
            raise RuntimeError(f"Left/right video sizes differ: {(width, height)} vs {(width_r, height_r)}")

    calib["left_timestamps"] = left_timestamps
    calib["right_timestamps"] = right_timestamps
    calib["first_ros_timestamp_ns"] = first_timestamp_ns
    write_json(args.native_dir / "calib_rectified.json", calib)
    write_json(args.native_dir / "joints.json", joints)

    robots = {args.robot_name: joints}
    write_json(args.offline_dir / "robots.json", robots)

    left_metadata = {
        "serial": "stereo_left",
        "K": calib["K_left_rect"],
        "resolution": [width, height],
        "timestamps": left_timestamps,
    }
    right_metadata = {
        "serial": "stereo_right",
        "K": calib["K_right_rect"],
        "resolution": [width, height],
        "timestamps": right_timestamps,
    }
    write_json(videos_dir / "stereo_left.json", left_metadata)
    write_json(videos_dir / "stereo_right.json", right_metadata)

    cameras = {
        "stereo_left": {
            "X_WC": np.eye(4, dtype=np.float32).tolist(),
            "video_path": "videos/stereo_left.mp4",
            "metadata_path": "videos/stereo_left.json",
        },
        "stereo_right": {
            "X_WC": np.eye(4, dtype=np.float32).tolist(),
            "video_path": "videos/stereo_right.mp4",
            "metadata_path": "videos/stereo_right.json",
        },
    }
    write_json(args.offline_dir / "cameras.json", cameras)

    metadata = {
        "bag": str(args.bag),
        "calibration": str(args.calibration),
        "rgb_dir": str(args.rgb_dir),
        "native_dir": str(args.native_dir),
        "offline_dir": str(args.offline_dir),
        "frame_ids": [frame_ids[0], frame_ids[-1]],
        "num_paired_frames": num_frames,
        "num_joint_states": len(joints["control"]),
        "left_png_count_used": num_frames,
        "right_png_count_used": num_frames,
        "videos_encoded": not args.no_video,
    }
    write_json(args.native_dir / "super_dataset_metadata.json", metadata)
    write_json(args.offline_dir / "super_dataset_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
