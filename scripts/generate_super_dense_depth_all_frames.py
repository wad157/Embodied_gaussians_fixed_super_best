#!/usr/bin/env python3
"""Generate spatially dense, timestamp-synchronised SUPER depth for every frame.

This is the lightweight online-observation counterpart to
``generate_super_depth_foundation_timestamped.py``.  It runs the frozen
FoundationStereo model on the two right-camera frames bracketing each left
timestamp, interpolates disparity to the left timestamp, and retains every
finite depth inside the configured range.  A cheap temporal-consistency image
is saved separately; it is a confidence signal, not a validity mask.

The strict evaluation GT must continue to use ``depth_high_confidence.npy``
from the dual-model pipeline.  Dense outputs from this script are intended for
flow-to-3D visual observations and must not silently replace strict GT.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO / "data/super/grasp5_native"
OFFLINE_ROOT = REPO / "data/super/grasp5_offline_demo"
FOUNDATION_ROOT = REPO / "third_party/FoundationStereo"

os.environ.setdefault("XFORMERS_DISABLED", "1")
sys.path.insert(0, str(REPO / "scripts"))

from generate_super_depth_foundation_timestamped import (  # noqa: E402
    depth_from_disparity,
    five_number,
    image_tensor,
    infer_foundation,
    load_foundation_model,
    read_json,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=1439)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--rgb-dir", type=Path, default=NATIVE_ROOT / "rgb")
    parser.add_argument(
        "--left-metadata",
        type=Path,
        default=OFFLINE_ROOT / "videos/stereo_left.json",
    )
    parser.add_argument(
        "--right-metadata",
        type=Path,
        default=OFFLINE_ROOT / "videos/stereo_right.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=NATIVE_ROOT / "calib_rectified.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            FOUNDATION_ROOT
            / "pretrained_models/23-51-11/model_best_bp2.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NATIVE_ROOT / "depth_dense_all_frames_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument(
        "--hierarchical",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-depth-mm", type=float, default=35.0)
    parser.add_argument("--max-depth-mm", type=float, default=250.0)
    parser.add_argument(
        "--temporal-high-threshold-px",
        type=float,
        default=1.0,
        help="Right-bracket disparity span for confidence level 3.",
    )
    parser.add_argument(
        "--temporal-medium-threshold-px",
        type=float,
        default=3.0,
        help="Right-bracket disparity span for confidence level 2.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and skip already completed frames.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def frame_paths(output_dir: Path, frame: int) -> tuple[Path, Path, Path]:
    prefix = output_dir / f"{frame:06d}"
    return (
        Path(f"{prefix}-depth.npy"),
        Path(f"{prefix}-temporal_confidence.png"),
        Path(f"{prefix}-report.json"),
    )


def completed_frame(output_dir: Path, frame: int, shape: tuple[int, int]) -> bool:
    depth_path, confidence_path, report_path = frame_paths(output_dir, frame)
    if not (depth_path.is_file() and confidence_path.is_file() and report_path.is_file()):
        return False
    try:
        depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
        confidence = cv2.imread(str(confidence_path), cv2.IMREAD_GRAYSCALE)
        report = read_json(report_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        depth.shape == shape
        and depth.dtype == np.float32
        and confidence is not None
        and confidence.shape == shape
        and int(report.get("left_frame", -1)) == frame
        and float(report.get("dense_valid_fraction", 0.0)) == 1.0
    )


def build_summary(
    *,
    args: argparse.Namespace,
    frames: list[int],
    calibration: dict,
    model_metadata: dict,
    frame_reports: list[dict],
) -> dict:
    return {
        "schema": "super_dense_timestamped_depth_all_frames_v1",
        "purpose": "dense flow-to-3D online observations; not strict evaluation GT",
        "complete": len(frame_reports) == len(frames),
        "completed_frame_count": len(frame_reports),
        "requested_frame_count": len(frames),
        "frame_range_inclusive": [frames[0], frames[-1]],
        "frame_stride": args.frame_stride,
        "foundation": model_metadata,
        "calibration": {
            "path": str(args.calibration.resolve()),
            "sha256": sha256(args.calibration),
            "fx_px": float(calibration["K_left_rect"][0][0]),
            "baseline_m": float(calibration["baseline_m"]),
            "cx_right_minus_left_px": float(
                calibration["K_right_rect"][0][2]
                - calibration["K_left_rect"][0][2]
            ),
        },
        "timestamp_method": (
            "right frames bracketing each left timestamp; linear disparity "
            "interpolation to the left timestamp"
        ),
        "parameters": {
            "iters": args.iters,
            "hierarchical": args.hierarchical,
            "depth_range_mm": [args.min_depth_mm, args.max_depth_mm],
            "temporal_high_threshold_px": args.temporal_high_threshold_px,
            "temporal_medium_threshold_px": args.temporal_medium_threshold_px,
        },
        "outputs": {
            "directory": str(args.output_dir.resolve()),
            "depth_pattern": "NNNNNN-depth.npy",
            "temporal_confidence_pattern": "NNNNNN-temporal_confidence.png",
            "confidence_encoding": {
                "85": "dense depth; bracket disparity span > medium threshold",
                "170": "span <= medium threshold",
                "255": "span <= high threshold",
            },
        },
        "warning": (
            "Spatial coverage is not accuracy. Consumers must retain confidence "
            "weighting and must not use these dense values as strict 3D GT."
        ),
        "frames": frame_reports,
    }


def main() -> None:
    args = parse_args()
    if args.frame_stride <= 0 or args.start_frame < 0 or args.end_frame < args.start_frame:
        raise ValueError("Invalid frame range or stride")
    if not 0 < args.temporal_high_threshold_px <= args.temporal_medium_threshold_px:
        raise ValueError("Temporal confidence thresholds must be positive and ordered")

    left_metadata = read_json(args.left_metadata)
    right_metadata = read_json(args.right_metadata)
    calibration = read_json(args.calibration)
    left_timestamps = np.asarray(left_metadata["timestamps"], dtype=np.float64)
    right_timestamps = np.asarray(right_metadata["timestamps"], dtype=np.float64)
    frames = list(range(args.start_frame, args.end_frame + 1, args.frame_stride))
    if frames[-1] >= len(left_timestamps):
        raise ValueError("Requested frame exceeds left-camera metadata")
    shape = (int(left_metadata["resolution"][1]), int(left_metadata["resolution"][0]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.iterdir())
    if existing and not args.resume:
        raise RuntimeError(
            f"Output directory is not empty: {args.output_dir}; use --resume"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"loading FoundationStereo on {device}", flush=True)
    model, model_metadata = load_foundation_model(args.checkpoint, device)
    fx = float(calibration["K_left_rect"][0][0])
    baseline = float(calibration["baseline_m"])
    cx_delta = float(
        calibration["K_right_rect"][0][2]
        - calibration["K_left_rect"][0][2]
    )
    min_depth = args.min_depth_mm / 1000.0
    max_depth = args.max_depth_mm / 1000.0
    summary_path = args.output_dir / "depth_generation_summary.json"
    frame_reports_by_id = {
        int(report["left_frame"]): report
        for frame in frames
        if frame_paths(args.output_dir, frame)[2].is_file()
        for report in [read_json(frame_paths(args.output_dir, frame)[2])]
    }

    for slot, frame in enumerate(frames, start=1):
        if args.resume and completed_frame(args.output_dir, frame, shape):
            print(f"[{slot}/{len(frames)}] frame {frame}: already complete", flush=True)
            continue
        started = time.perf_counter()
        timestamp = float(left_timestamps[frame])
        after = int(np.searchsorted(right_timestamps, timestamp, side="left"))
        after = int(np.clip(after, 1, len(right_timestamps) - 1))
        before = after - 1
        interval = float(right_timestamps[after] - right_timestamps[before])
        if interval <= 0.0:
            raise ValueError("Right-camera timestamps are not strictly increasing")
        alpha = float((timestamp - right_timestamps[before]) / interval)

        left = cv2.imread(str(args.rgb_dir / f"{frame:06d}-left.png"), cv2.IMREAD_COLOR)
        right_before = cv2.imread(
            str(args.rgb_dir / f"{before:06d}-right.png"), cv2.IMREAD_COLOR
        )
        right_after = cv2.imread(
            str(args.rgb_dir / f"{after:06d}-right.png"), cv2.IMREAD_COLOR
        )
        if left is None or right_before is None or right_after is None:
            raise FileNotFoundError(f"Missing stereo input around frame {frame}")
        if left.shape[:2] != shape:
            raise ValueError(f"Unexpected image shape at frame {frame}: {left.shape[:2]}")

        left_tensor = image_tensor(left, device)
        disparity_before = infer_foundation(
            model,
            left_tensor,
            image_tensor(right_before, device),
            args.iters,
            args.hierarchical,
            device,
        )
        disparity_after = infer_foundation(
            model,
            left_tensor,
            image_tensor(right_after, device),
            args.iters,
            args.hierarchical,
            device,
        )
        disparity = (
            (1.0 - alpha) * disparity_before + alpha * disparity_after
        ).astype(np.float32)
        depth, valid = depth_from_disparity(
            disparity,
            np.ones(shape, dtype=bool),
            fx,
            baseline,
            cx_delta,
            min_depth,
            max_depth,
        )
        if not bool(valid.all()):
            missing = int(valid.size - valid.sum())
            raise RuntimeError(
                f"frame {frame}: dense depth has {missing} invalid pixels; "
                "refusing to claim full coverage"
            )

        span = np.abs(disparity_after - disparity_before)
        confidence = np.full(shape, 85, dtype=np.uint8)
        confidence[span <= args.temporal_medium_threshold_px] = 170
        confidence[span <= args.temporal_high_threshold_px] = 255
        depth_path, confidence_path, report_path = frame_paths(args.output_dir, frame)
        atomic_save_npy(depth_path, depth.astype(np.float32, copy=False))
        temporary_confidence = confidence_path.with_name(
            confidence_path.stem + ".tmp.png"
        )
        if not cv2.imwrite(str(temporary_confidence), confidence):
            raise OSError(temporary_confidence)
        temporary_confidence.replace(confidence_path)
        report = {
            "left_frame": frame,
            "left_timestamp": timestamp,
            "right_before_frame": before,
            "right_after_frame": after,
            "temporal_interpolation_alpha": alpha,
            "dense_valid_fraction": float(valid.mean()),
            "depth_mm_min_p05_p50_p95_max": five_number(depth * 1000.0),
            "bracket_disparity_span_px_min_p05_p50_p95_max": five_number(span),
            "confidence_fraction": {
                "level_1": float(np.mean(confidence == 85)),
                "level_2": float(np.mean(confidence == 170)),
                "level_3": float(np.mean(confidence == 255)),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(report_path, report)
        frame_reports_by_id[frame] = report
        if slot % 25 == 0 or slot == len(frames):
            atomic_write_json(
                summary_path,
                build_summary(
                    args=args,
                    frames=frames,
                    calibration=calibration,
                    model_metadata=model_metadata,
                    frame_reports=[
                        frame_reports_by_id[key]
                        for key in sorted(frame_reports_by_id)
                    ],
                ),
            )
        print(
            f"[{slot}/{len(frames)}] frame {frame}: coverage=100.000% "
            f"level3={report['confidence_fraction']['level_3']:.3f} "
            f"elapsed={report['elapsed_seconds']:.2f}s",
            flush=True,
        )

    reports = [frame_reports_by_id[value] for value in frames]
    summary = build_summary(
        args=args,
        frames=frames,
        calibration=calibration,
        model_metadata=model_metadata,
        frame_reports=reports,
    )
    atomic_write_json(summary_path, summary)
    print(json.dumps({key: summary[key] for key in (
        "schema", "complete", "completed_frame_count", "requested_frame_count"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
