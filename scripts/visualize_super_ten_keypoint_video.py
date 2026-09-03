#!/usr/bin/env python3
"""Write one SUPER video overlaying all ten GT and predicted keypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = (
    ROOT
    / "data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
)
DEFAULT_ANNOTATIONS = (
    ROOT / "data/super/evaluation_v1/manual_tissue_tracks_10/annotations.json"
)
DEFAULT_PREDICTION = (
    ROOT
    / "outputs/super_alltracker_direct_online_stiffness_full_20260829_v1/"
    "direct_online_strain075/predicted_tracks.npz"
)
DEFAULT_RGB = ROOT / "data/super/grasp5_native/rgb"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/super_alltracker_direct_online_stiffness_full_20260829_v1/"
    "visualizations/ten_keypoints_strain075.mp4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--rgb-root", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0.65)
    parser.add_argument("--trail-samples", type=int, default=24)
    return parser.parse_args()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def mean_error(
    prediction: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    *,
    scale: float = 1.0,
) -> float:
    error = np.linalg.norm(prediction - truth, axis=-1)
    values = error[np.asarray(valid, dtype=bool)]
    return float(np.mean(values) * scale) if values.size else float("nan")


def darken(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(round(channel * factor)) for channel in color)


def main() -> None:
    args = parse_args()
    if args.fps <= 0.0 or not 0.0 < args.scale <= 1.0:
        raise ValueError("fps must be positive and scale must lie in (0, 1]")
    if args.trail_samples < 1:
        raise ValueError("trail-samples must be positive")

    ground_truth = read_npz(args.ground_truth.resolve())
    prediction = read_npz(args.prediction.resolve())
    frames = np.asarray(ground_truth["frame_indices"], dtype=np.int32)
    predicted_frames = np.asarray(prediction["frame_indices"], dtype=np.int32)
    if not np.array_equal(frames, predicted_frames):
        raise ValueError("Ground-truth and prediction frame schedules differ")
    gt_uv = np.asarray(ground_truth["uv"], dtype=np.float64)
    pred_uv = np.asarray(prediction["uv"], dtype=np.float64)
    if gt_uv.shape != pred_uv.shape or gt_uv.shape[1:] != (10, 2):
        raise ValueError("Expected aligned [frame, 10, 2] image tracks")
    gt_xyz = np.asarray(ground_truth["xyz_camera_m"], dtype=np.float64)
    pred_xyz = np.asarray(prediction["xyz_camera_m"], dtype=np.float64)
    visible = np.asarray(ground_truth["visible"], dtype=bool)
    valid_3d = np.asarray(ground_truth["valid_3d"], dtype=bool)
    annotations = json.loads(args.annotations.resolve().read_text(encoding="utf-8"))
    names = [str(item.get("name", f"P{index}")) for index, item in enumerate(annotations["points"])]
    if len(names) != 10:
        raise ValueError("Expected ten annotation names")
    future_start = int(np.asarray(ground_truth["future_test_start_frame"]).item())

    first_path = args.rgb_root.resolve() / f"{int(frames[0]):06d}-left.png"
    first = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(first_path)
    output_size = (
        int(round(first.shape[1] * args.scale)),
        int(round(first.shape[0] * args.scale)),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, output_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")

    colors: tuple[tuple[int, int, int], ...] = (
        (54, 67, 244),
        (38, 166, 154),
        (65, 174, 255),
        (95, 189, 89),
        (186, 99, 46),
        (211, 85, 186),
        (82, 77, 230),
        (180, 130, 70),
        (50, 205, 50),
        (200, 150, 255),
    )
    panel_height = 91
    try:
        for slot, frame_value in enumerate(frames):
            frame_index = int(frame_value)
            image_path = args.rgb_root.resolve() / f"{frame_index:06d}-left.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
            begin = max(0, slot - args.trail_samples + 1)
            gt_scaled = np.rint(args.scale * gt_uv[begin : slot + 1]).astype(np.int32)
            pred_scaled = np.rint(args.scale * pred_uv[begin : slot + 1]).astype(np.int32)

            overlay = image.copy()
            for point_id, color in enumerate(colors):
                valid_trail = visible[begin : slot + 1, point_id]
                if np.count_nonzero(valid_trail) >= 2:
                    gt_poly = np.ascontiguousarray(gt_scaled[valid_trail, point_id]).reshape(-1, 1, 2)
                    pred_poly = np.ascontiguousarray(pred_scaled[valid_trail, point_id]).reshape(-1, 1, 2)
                    cv2.polylines(overlay, [gt_poly], False, (25, 25, 25), 2, cv2.LINE_AA)
                    cv2.polylines(overlay, [pred_poly], False, darken(color, 0.86), 2, cv2.LINE_AA)
            image = cv2.addWeighted(overlay, 0.76, image, 0.24, 0.0)

            current_gt = np.rint(args.scale * gt_uv[slot]).astype(np.int32)
            current_pred = np.rint(args.scale * pred_uv[slot]).astype(np.int32)
            for point_id, color in enumerate(colors):
                if not visible[slot, point_id]:
                    continue
                gt_point = tuple(int(v) for v in current_gt[point_id])
                pred_point = tuple(int(v) for v in current_pred[point_id])
                cv2.line(image, gt_point, pred_point, (215, 215, 215), 1, cv2.LINE_AA)
                cv2.circle(image, gt_point, 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(image, gt_point, 5, color, -1, cv2.LINE_AA)
                cv2.drawMarker(image, pred_point, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 13, 4, cv2.LINE_AA)
                cv2.drawMarker(image, pred_point, color, cv2.MARKER_TILTED_CROSS, 11, 2, cv2.LINE_AA)
                cv2.putText(
                    image, f"P{point_id}", (gt_point[0] + 7, gt_point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (15, 15, 15), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    image, f"P{point_id}", (gt_point[0] + 7, gt_point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
                )

            error_2d = mean_error(pred_uv[slot], gt_uv[slot], visible[slot])
            error_3d = mean_error(
                pred_xyz[slot], gt_xyz[slot], valid_3d[slot], scale=1000.0
            )
            phase = "OPEN-LOOP FUTURE (NO VISUAL UPDATE)" if frame_index >= future_start else "OBSERVED/TRAINING PREFIX"
            cv2.rectangle(image, (0, 0), (output_size[0], panel_height), (248, 248, 248), -1)
            cv2.putText(
                image,
                f"AllTracker + direct online stiffness (strain 75%) | frame {frame_index:04d} | {phase}",
                (13, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"10 keypoints | 2D mean {error_2d:6.2f} px | 3D mean {error_3d:5.3f} mm | sample {slot + 1:03d}/{len(frames)}",
                (13, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (25, 25, 25), 1, cv2.LINE_AA,
            )
            cv2.circle(image, (19, 71), 5, (80, 80, 80), -1, cv2.LINE_AA)
            cv2.putText(image, "manual GT (circle / black trail)", (31, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (40, 40, 40), 1, cv2.LINE_AA)
            cv2.drawMarker(image, (300, 71), (142, 68, 173), cv2.MARKER_TILTED_CROSS, 11, 2, cv2.LINE_AA)
            cv2.putText(image, "prediction (cross / colored trail)", (313, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (40, 40, 40), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()

    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"Video was not written: {output}")
    print(
        f"wrote {output} ({output.stat().st_size} bytes, {len(frames)} annotated "
        f"samples, {len(frames) / args.fps:.2f}s)"
    )


if __name__ == "__main__":
    main()
