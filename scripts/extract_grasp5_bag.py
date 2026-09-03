#!/usr/bin/env python3

import argparse
import json
import shutil
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
            "Extract SuPer grasp5 raw ROS bag data into rectified image sequences "
            "and the offline dataset format used by Embodied Gaussians."
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
        "--handeye",
        type=Path,
        default=repo_root / "data" / "handeye.yaml",
        help="Hand-eye calibration YAML copied into the output for provenance.",
    )
    parser.add_argument(
        "--lnd",
        type=Path,
        default=repo_root / "data" / "LND.json",
        help="LND model JSON copied into the output for provenance.",
    )
    parser.add_argument(
        "--native-dir",
        type=Path,
        default=repo_root / "data" / "super" / "grasp5_native",
        help="Output directory for rectified PNGs, calibration, and intermediate JSON.",
    )
    parser.add_argument(
        "--offline-dir",
        type=Path,
        default=repo_root / "data" / "super" / "grasp5_offline_demo",
        help="Output directory matching DatasetManager offline demo format.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit per image stream. Omit to extract all frames.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=3,
        choices=range(10),
        help="OpenCV PNG compression level for rectified frame dumps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output directories before extraction.",
    )
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="Only write MP4 videos and JSON manifests, not per-frame PNGs.",
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


def read_opencv_scalar_sequence(path: Path, key: str) -> list[float]:
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


def load_stereo_calibration(path: Path) -> dict[str, np.ndarray | tuple[int, int]]:
    k1 = read_opencv_matrix(path, "K1")
    k2 = read_opencv_matrix(path, "K2")
    d1 = read_opencv_matrix(path, "D1")
    d2 = read_opencv_matrix(path, "D2")
    r = read_opencv_matrix(path, "R")
    image_size_hw = read_opencv_scalar_sequence(path, "ImageSize")
    t = np.array(read_opencv_scalar_sequence(path, "T"), dtype=np.float64).reshape(3, 1)
    height, width = int(image_size_hw[0]), int(image_size_hw[1])
    return {
        "K1": k1,
        "K2": k2,
        "D1": d1,
        "D2": d2,
        "R": r,
        "T": t,
        "image_size": (width, height),
    }


def build_rectification(calib: dict[str, Any]) -> dict[str, np.ndarray]:
    width, height = calib["image_size"]
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        calib["K1"],
        calib["D1"],
        calib["K2"],
        calib["D2"],
        (width, height),
        calib["R"],
        calib["T"],
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    left_map = cv2.initUndistortRectifyMap(
        calib["K1"], calib["D1"], r1, p1, (width, height), cv2.CV_32FC1
    )
    right_map = cv2.initUndistortRectifyMap(
        calib["K2"], calib["D2"], r2, p2, (width, height), cv2.CV_32FC1
    )
    return {
        "R1": r1,
        "R2": r2,
        "P1": p1,
        "P2": p2,
        "Q": q,
        "K_left_rect": p1[:3, :3],
        "K_right_rect": p2[:3, :3],
        "left_map_x": left_map[0],
        "left_map_y": left_map[1],
        "right_map_x": right_map[0],
        "right_map_y": right_map[1],
    }


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def ros_time_to_sec(timestamp_ns: int, start_ns: int) -> float:
    return float(timestamp_ns - start_ns) * 1e-9


def image_msg_to_bgr(msg: Any) -> np.ndarray:
    row_bytes = int(msg.width) * 3
    if int(msg.step) < row_bytes:
        raise ValueError(
            f"Invalid {msg.encoding} row stride: step={msg.step}, expected at least {row_bytes}"
        )
    expected_bytes = int(msg.height) * int(msg.step)
    if len(msg.data) != expected_bytes:
        raise ValueError(
            f"Invalid image payload: bytes={len(msg.data)}, expected {expected_bytes}"
        )
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = image[:, :row_bytes].reshape(msg.height, msg.width, 3)
    encoding = str(msg.encoding).lower()
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgr8":
        return image.copy()
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def make_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def estimate_fps(timestamps: list[float], fallback: float = 30.0) -> float:
    if len(timestamps) < 2:
        return fallback
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        return fallback
    return float(1.0 / np.median(diffs))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def extract(args: argparse.Namespace) -> None:
    if not args.bag.exists():
        raise FileNotFoundError(args.bag)
    if not args.calibration.exists():
        raise FileNotFoundError(args.calibration)

    ensure_empty_dir(args.native_dir, args.overwrite)
    ensure_empty_dir(args.offline_dir, args.overwrite)

    native_rgb = args.native_dir / "rgb"
    native_rgb.mkdir(parents=True, exist_ok=True)
    native_depth = args.native_dir / "depth"
    native_depth.mkdir(parents=True, exist_ok=True)
    native_masks = args.native_dir / "masks"
    native_masks.mkdir(parents=True, exist_ok=True)
    videos_dir = args.offline_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    calib = load_stereo_calibration(args.calibration)
    rect = build_rectification(calib)
    width, height = calib["image_size"]

    fps_guess = 30.0
    left_writer = make_writer(videos_dir / "stereo_left.mp4", width, height, fps_guess)
    right_writer = make_writer(videos_dir / "stereo_right.mp4", width, height, fps_guess)

    typestore = get_typestore(Stores.ROS1_NOETIC)

    left_timestamps: list[float] = []
    right_timestamps: list[float] = []
    joint_timestamps: list[float] = []
    joint_positions: list[list[float]] = []
    joint_velocities: list[list[float]] = []
    joint_efforts: list[list[float]] = []
    joint_names: list[str] | None = None
    first_timestamp_ns: int | None = None
    counts = {LEFT_TOPIC: 0, RIGHT_TOPIC: 0, JOINT_TOPIC: 0}
    last_rectified: dict[str, np.ndarray] = {}
    recovered_images: list[dict[str, Any]] = []

    png_params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]

    with Reader(args.bag) as reader:
        connections = [
            c for c in reader.connections if c.topic in {LEFT_TOPIC, RIGHT_TOPIC, JOINT_TOPIC}
        ]
        if len(connections) != 3:
            found = sorted(c.topic for c in reader.connections)
            raise RuntimeError(f"Expected 3 target topics, found {found}")

        for conn, timestamp_ns, raw in reader.messages(connections=connections):
            if first_timestamp_ns is None:
                first_timestamp_ns = int(timestamp_ns)
            timestamp = ros_time_to_sec(int(timestamp_ns), first_timestamp_ns)
            msg = typestore.deserialize_ros1(raw, conn.msgtype)

            if conn.topic == JOINT_TOPIC:
                counts[JOINT_TOPIC] += 1
                if joint_names is None:
                    joint_names = [str(name) for name in msg.name]
                joint_timestamps.append(timestamp)
                joint_positions.append([float(v) for v in msg.position])
                joint_velocities.append([float(v) for v in msg.velocity])
                joint_efforts.append([float(v) for v in msg.effort])
                continue

            if args.max_frames is not None and counts[conn.topic] >= args.max_frames:
                continue

            try:
                bgr = image_msg_to_bgr(msg)
                if conn.topic == LEFT_TOPIC:
                    rectified = cv2.remap(
                        bgr,
                        rect["left_map_x"],
                        rect["left_map_y"],
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    rectified = cv2.remap(
                        bgr,
                        rect["right_map_x"],
                        rect["right_map_y"],
                        interpolation=cv2.INTER_LINEAR,
                    )
                last_rectified[conn.topic] = rectified
            except ValueError as exc:
                if conn.topic not in last_rectified:
                    raise
                rectified = last_rectified[conn.topic].copy()
                recovery = {
                    "topic": conn.topic,
                    "frame_index": counts[conn.topic],
                    "timestamp": timestamp,
                    "encoding": str(msg.encoding),
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "step": int(msg.step),
                    "data_bytes": len(msg.data),
                    "reason": str(exc),
                    "recovery": "duplicated_previous_rectified_frame",
                }
                recovered_images.append(recovery)
                print(f"warning: recovered malformed image: {json.dumps(recovery)}")

            if conn.topic == LEFT_TOPIC:
                frame_index = counts[LEFT_TOPIC]
                if not args.skip_png:
                    cv2.imwrite(
                        str(native_rgb / f"{frame_index:06d}-left.png"),
                        rectified,
                        png_params,
                    )
                left_writer.write(rectified)
                left_timestamps.append(timestamp)
                counts[LEFT_TOPIC] += 1
            elif conn.topic == RIGHT_TOPIC:
                frame_index = counts[RIGHT_TOPIC]
                if not args.skip_png:
                    cv2.imwrite(
                        str(native_rgb / f"{frame_index:06d}-right.png"),
                        rectified,
                        png_params,
                    )
                right_writer.write(rectified)
                right_timestamps.append(timestamp)
                counts[RIGHT_TOPIC] += 1

            total_images = counts[LEFT_TOPIC] + counts[RIGHT_TOPIC]
            if total_images > 0 and total_images % 200 == 0:
                print(
                    "extracted "
                    f"left={counts[LEFT_TOPIC]} right={counts[RIGHT_TOPIC]} "
                    f"joints={counts[JOINT_TOPIC]}"
                )

    left_writer.release()
    right_writer.release()

    if first_timestamp_ns is None:
        raise RuntimeError(f"No target messages found in {args.bag}")
    if joint_names is None:
        raise RuntimeError("No joint states were extracted.")

    robots = {
        "PSM1": {
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
    }
    write_json(args.offline_dir / "robots.json", robots)
    write_json(args.native_dir / "joints.json", robots["PSM1"])

    left_metadata = {
        "serial": "stereo_left",
        "K": rect["K_left_rect"].tolist(),
        "resolution": [width, height],
        "timestamps": left_timestamps,
    }
    right_metadata = {
        "serial": "stereo_right",
        "K": rect["K_right_rect"].tolist(),
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

    stereo_calib = {
        "image_size": [width, height],
        "K1": calib["K1"].tolist(),
        "K2": calib["K2"].tolist(),
        "D1": calib["D1"].reshape(-1).tolist(),
        "D2": calib["D2"].reshape(-1).tolist(),
        "R": calib["R"].tolist(),
        "T": calib["T"].reshape(-1).tolist(),
        "R1": rect["R1"].tolist(),
        "R2": rect["R2"].tolist(),
        "P1": rect["P1"].tolist(),
        "P2": rect["P2"].tolist(),
        "Q": rect["Q"].tolist(),
        "K_left_rect": rect["K_left_rect"].tolist(),
        "K_right_rect": rect["K_right_rect"].tolist(),
        "left_timestamps": left_timestamps,
        "right_timestamps": right_timestamps,
        "estimated_left_fps": estimate_fps(left_timestamps),
        "estimated_right_fps": estimate_fps(right_timestamps),
        "first_ros_timestamp_ns": first_timestamp_ns,
    }
    write_json(args.native_dir / "calib_rectified.json", stereo_calib)

    metadata = {
        "bag": str(args.bag),
        "calibration": str(args.calibration),
        "handeye": str(args.handeye),
        "lnd": str(args.lnd),
        "topics": {
            "left": LEFT_TOPIC,
            "right": RIGHT_TOPIC,
            "joints": JOINT_TOPIC,
        },
        "counts": {
            "left_images": len(left_timestamps),
            "right_images": len(right_timestamps),
            "joint_states": len(joint_timestamps),
            "recovered_images": len(recovered_images),
        },
        "recovered_image_details": recovered_images,
        "native_dir": str(args.native_dir),
        "offline_dir": str(args.offline_dir),
        "skip_png": args.skip_png,
    }
    write_json(args.native_dir / "super_dataset_metadata.json", metadata)
    write_json(args.offline_dir / "super_dataset_metadata.json", metadata)

    provenance_dir = args.native_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    for src in (args.calibration, args.handeye, args.lnd):
        if src.exists():
            shutil.copy2(src, provenance_dir / src.name)

    print("Extraction complete")
    print(json.dumps(metadata["counts"], indent=2))
    print(f"Native output: {args.native_dir}")
    print(f"Offline output: {args.offline_dir}")


def main() -> None:
    args = parse_args()
    extract(args)


if __name__ == "__main__":
    main()
