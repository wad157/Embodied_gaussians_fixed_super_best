#!/usr/bin/env python3
"""Generate the two raw FoundationStereo depths used by EG initialization."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from baselines.embodied_gaussians_super.common import (  # noqa: E402
    load_calibration,
    nearest_timestamp_index,
)
from baselines.embodied_gaussians_super.protocol import (  # noqa: E402
    DATASETS,
    INITIALIZATION_FRAME,
    PAPER_PARAMETERS,
    dataset_spec,
    read_json,
    resolve_path,
    sha256_file,
)
from scripts.generate_super_depth_foundation_timestamped import (  # noqa: E402
    depth_from_disparity,
    image_tensor,
    infer_foundation,
    load_foundation_model,
    read_baseline_m,
)


FOUNDATION_CHECKPOINT = (
    REPO_ROOT / "third_party/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal FoundationStereo initialization requires CUDA")
    spec = dataset_spec(args.dataset_key)
    native = resolve_path(REPO_ROOT, spec["native"])
    offline = resolve_path(REPO_ROOT, spec["offline"])
    output = (
        native / "embodied_gaussians_initial_stereo_depth_v1"
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration(REPO_ROOT, args.dataset_key)
    left_timestamps = np.asarray(calibration["left_timestamps"])
    right_timestamps = np.asarray(calibration["right_timestamps"])
    left_frame = INITIALIZATION_FRAME
    right_frame = nearest_timestamp_index(right_timestamps, float(left_timestamps[left_frame]))
    paired_left = nearest_timestamp_index(left_timestamps, float(right_timestamps[right_frame]))
    width = int(PAPER_PARAMETERS["online_width"])
    source_width, source_height = calibration["resolution_wh"]
    height = int(round(source_height * width / source_width))
    scale_x, scale_y = width / source_width, height / source_height
    calibration_path = native / "calib_rectified.json"
    rectified = read_json(calibration_path)
    baseline_m, baseline_source = read_baseline_m(rectified)
    left_path = native / "rgb" / f"{left_frame:06d}-left.png"
    right_path = native / "rgb" / f"{right_frame:06d}-right.png"
    paired_left_path = native / "rgb" / f"{paired_left:06d}-left.png"
    configuration = {
        "schema": "embodied_gaussians_super_initial_stereo_depth_v1",
        "dataset_key": args.dataset_key,
        "initialization_left_frame": left_frame,
        "initialization_right_frame": right_frame,
        "right_reference_paired_left_frame": paired_left,
        "frame_policy": "first legal reconstruction training frame; frame 0 remains withheld",
        "cameras": ["stereo_left", "stereo_right"],
        "resolution_wh": [width, height],
        "estimator": "FoundationStereo RGB-only",
        "foundation_iterations": 32,
        "foundation_hierarchical": False,
        "checkpoint": str(FOUNDATION_CHECKPOINT),
        "checkpoint_sha256": sha256_file(FOUNDATION_CHECKPOINT),
        "calibration_sha256": sha256_file(calibration_path),
        "baseline_m": baseline_m,
        "baseline_source": baseline_source,
        "filtering": "finite positive disparity and public EG max_depth=2m only",
        "dataset_specific_depth_range": None,
        "lr_consistency": False,
        "raft_used": False,
        "ground_truth_used": False,
        "rgb_sha256": {
            "left": sha256_file(left_path),
            "right": sha256_file(right_path),
            "right_reference_paired_left": sha256_file(paired_left_path),
        },
    }
    config_path = output / "configuration.json"
    if config_path.exists() and read_json(config_path) != configuration:
        raise ValueError(f"Existing depth cache has a different configuration: {config_path}")
    if not config_path.exists():
        atomic_json(config_path, configuration)
    targets = {
        "stereo_left": output / f"{left_frame:06d}-left-depth.npy",
        "stereo_right": output / f"{right_frame:06d}-right-depth.npy",
    }
    if all(path.is_file() for path in targets.values()):
        print(f"Initial stereo depth already complete: {output}")
        return

    model, model_metadata = load_foundation_model(FOUNDATION_CHECKPOINT, device)
    left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    paired_left_bgr = cv2.imread(str(paired_left_path), cv2.IMREAD_COLOR)
    if left_bgr is None or right_bgr is None or paired_left_bgr is None:
        raise FileNotFoundError("Missing initialization stereo RGB")
    left_bgr = cv2.resize(left_bgr, (width, height), interpolation=cv2.INTER_AREA)
    right_bgr = cv2.resize(right_bgr, (width, height), interpolation=cv2.INTER_AREA)
    paired_left_bgr = cv2.resize(paired_left_bgr, (width, height), interpolation=cv2.INTER_AREA)
    left_tensor = image_tensor(left_bgr, device)
    right_tensor = image_tensor(right_bgr, device)
    paired_left_tensor = image_tensor(paired_left_bgr, device)
    left_disparity = infer_foundation(model, left_tensor, right_tensor, 32, False, device)
    right_flipped = infer_foundation(
        model,
        torch.flip(right_tensor, dims=(3,)),
        torch.flip(paired_left_tensor, dims=(3,)),
        32,
        False,
        device,
    )
    right_disparity = np.flip(right_flipped, axis=1).copy()
    k_left = np.asarray(calibration["stereo_left"]["K"], dtype=np.float64)
    k_right = np.asarray(calibration["stereo_right"]["K"], dtype=np.float64)
    cx_delta = float((k_right[0, 2] - k_left[0, 2]) * scale_x)
    reports = {}
    for camera, disparity, intrinsic, target in (
        ("stereo_left", left_disparity, k_left, targets["stereo_left"]),
        ("stereo_right", right_disparity, k_right, targets["stereo_right"]),
    ):
        valid = np.isfinite(disparity) & (disparity > 0.0)
        depth, valid = depth_from_disparity(
            disparity,
            valid,
            float(intrinsic[0, 0]) * scale_x,
            baseline_m,
            cx_delta,
            0.0,
            2.0,
        )
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, depth.astype(np.float32))
        os.replace(temporary, target)
        finite = depth[valid]
        reports[camera] = {
            "output": str(target),
            "sha256": sha256_file(target),
            "valid_fraction": float(valid.mean()),
            "depth_m_p05_p50_p95": np.percentile(finite, (5, 50, 95)).tolist(),
        }
    atomic_json(
        output / "report.json",
        {"configuration": configuration, "model": model_metadata, "outputs": reports},
    )
    (output / "COMPLETE").write_text(
        f"configuration_sha256={sha256_file(config_path)}\n", encoding="utf-8"
    )
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
