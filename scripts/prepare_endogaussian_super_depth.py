#!/usr/bin/env python3
"""Prepare compact FoundationStereo depth for legal EndoGaussian SUPER views."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = REPO_ROOT / "baselines" / "endogaussian_super"
sys.path.insert(0, str(ADAPTER_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from protocol import (  # noqa: E402
    DATASETS,
    TRAIN_DOWNSAMPLE,
    dataset_spec,
    depth_cache,
    load_json,
    resolve_path,
    sha256_file,
    training_frames,
)
from generate_super_depth_foundation_timestamped import (  # noqa: E402
    depth_from_disparity,
    image_tensor,
    infer_foundation,
    load_foundation_model,
    read_baseline_m,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--downsample", type=int, default=TRAIN_DOWNSAMPLE)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--min-depth-mm", type=float, default=35.0)
    parser.add_argument("--max-depth-mm", type=float, default=250.0)
    parser.add_argument("--limit", type=int, help="Engineering-only prefix of the legal schedule")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "third_party/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",
    )
    return parser.parse_args()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.downsample < 1 or args.iterations < 1:
        raise ValueError("downsample and iterations must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    spec = dataset_spec(args.dataset_key)
    native = resolve_path(REPO_ROOT, spec["native"])
    offline = resolve_path(REPO_ROOT, spec["offline"])
    output = (
        depth_cache(REPO_ROOT, args.dataset_key, args.downsample)
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    complete_path = output / "COMPLETE"
    schedule = training_frames(args.dataset_key)
    selected = schedule if args.limit is None else schedule[: args.limit]
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    left_meta_path = offline / "videos" / "stereo_left.json"
    right_meta_path = offline / "videos" / "stereo_right.json"
    calibration_path = native / "calib_rectified.json"
    left_meta = load_json(left_meta_path)
    right_meta = load_json(right_meta_path)
    calibration = load_json(calibration_path)
    left_timestamps = np.asarray(left_meta["timestamps"], dtype=np.float64)
    right_timestamps = np.asarray(right_meta["timestamps"], dtype=np.float64)
    K_left = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    K_right = np.asarray(calibration["K_right_rect"], dtype=np.float64)
    baseline_m, baseline_source = read_baseline_m(calibration)
    checkpoint_hash = sha256_file(args.checkpoint)
    configuration = {
        "schema": "endogaussian_super_foundation_depth_config_v1",
        "dataset_key": args.dataset_key,
        "legal_training_schedule": "frame < future_start and frame % 8 != 0",
        "frame_count": int(spec["frames"]),
        "future_start": int(spec["future_start"]),
        "downsample": args.downsample,
        "resolution_wh": [
            int(left_meta["resolution"][0]) // args.downsample,
            int(left_meta["resolution"][1]) // args.downsample,
        ],
        "foundation_iterations": args.iterations,
        "foundation_hierarchical": bool(args.hierarchical),
        "right_pairing": "nearest timestamp; ties choose earlier",
        "min_depth_mm": args.min_depth_mm,
        "max_depth_mm": args.max_depth_mm,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "left_metadata_sha256": sha256_file(left_meta_path),
        "right_metadata_sha256": sha256_file(right_meta_path),
        "calibration_sha256": sha256_file(calibration_path),
        "baseline_m": baseline_m,
        "baseline_source": baseline_source,
        "uses_ground_truth": False,
        "uses_future_or_holdout_rgb": False,
        "depth_units": "meters",
        "stored_dtype": "float32",
    }
    config_path = output / "configuration.json"
    if config_path.exists():
        existing = load_json(config_path)
        if existing != configuration:
            raise ValueError("Existing cache configuration differs: {}".format(config_path))
    else:
        write_json_atomic(config_path, configuration)
    if complete_path.exists() and args.limit is None:
        print("Depth cache already complete: {}".format(output))
        return

    pending = [frame for frame in selected if not (output / "{:06d}-depth.npy".format(frame)).is_file()]
    print("{}: {} selected, {} cached, {} pending".format(
        args.dataset_key, len(selected), len(selected) - len(pending), len(pending)
    ), flush=True)
    if pending:
        model, model_metadata = load_foundation_model(args.checkpoint, device)
    else:
        model_metadata = None
    reports = {}
    report_path = output / "frame_reports.json"
    if report_path.exists():
        reports = load_json(report_path).get("frames", {})
    width, height = configuration["resolution_wh"]
    fx = float(K_left[0, 0]) / float(args.downsample)
    cx_delta = float(K_right[0, 2] - K_left[0, 2]) / float(args.downsample)
    for progress, frame in enumerate(pending, start=1):
        started = time.perf_counter()
        timestamp = left_timestamps[frame]
        insertion = int(np.searchsorted(right_timestamps, timestamp, side="left"))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(right_timestamps)]
        right_frame = min(candidates, key=lambda index: (abs(right_timestamps[index] - timestamp), index))
        left = cv2.imread(str(native / "rgb" / "{:06d}-left.png".format(frame)), cv2.IMREAD_COLOR)
        right = cv2.imread(str(native / "rgb" / "{:06d}-right.png".format(right_frame)), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise FileNotFoundError("Missing stereo input for left {} right {}".format(frame, right_frame))
        left = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (width, height), interpolation=cv2.INTER_AREA)
        disparity = infer_foundation(
            model,
            image_tensor(left, device),
            image_tensor(right, device),
            args.iterations,
            bool(args.hierarchical),
            device,
        )
        depth, valid = depth_from_disparity(
            disparity,
            np.ones(disparity.shape, dtype=bool),
            fx,
            baseline_m,
            cx_delta,
            args.min_depth_mm / 1000.0,
            args.max_depth_mm / 1000.0,
        )
        target = output / "{:06d}-depth.npy".format(frame)
        temporary = output / (target.name + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, depth.astype(np.float32))
        os.replace(str(temporary), str(target))
        finite = depth[valid]
        reports[str(frame)] = {
            "left_frame": frame,
            "right_frame": right_frame,
            "timestamp_delta_ms": float((right_timestamps[right_frame] - timestamp) * 1000.0),
            "valid_fraction": float(valid.mean()),
            "depth_m_p05_p50_p95": np.percentile(finite, [5, 50, 95]).tolist() if len(finite) else None,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if progress % 20 == 0 or progress == len(pending):
            write_json_atomic(
                report_path,
                {
                    "schema": "endogaussian_super_foundation_depth_frames_v1",
                    "configuration": str(config_path),
                    "model": model_metadata,
                    "frames": reports,
                },
            )
        print("[{}/{}] left={} right={} valid={:.3f} elapsed={:.2f}s".format(
            progress, len(pending), frame, right_frame, float(valid.mean()), time.perf_counter() - started
        ), flush=True)

    cached = sorted(
        int(path.name[:6]) for path in output.glob("??????-depth.npy")
    )
    if args.limit is None:
        if cached != schedule:
            missing = sorted(set(schedule) - set(cached))[:20]
            extra = sorted(set(cached) - set(schedule))[:20]
            raise ValueError("Depth cache schedule mismatch; missing={} extra={}".format(missing, extra))
        complete_path.write_text(
            "dataset={}\nframes={}\nconfiguration_sha256={}\n".format(
                args.dataset_key, len(schedule), sha256_file(config_path)
            ),
            encoding="utf-8",
        )
    print("Prepared {} depth maps in {}".format(len(cached), output), flush=True)


if __name__ == "__main__":
    main()
