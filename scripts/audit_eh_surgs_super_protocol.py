#!/usr/bin/env python3
"""Fail closed when EH-SurGS SUPER uses stale or exposed inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = REPO_ROOT / "baselines" / "eh_surgs_super"
sys.path.insert(0, str(ADAPTER_ROOT))

from protocol import (  # noqa: E402
    ADAPTER_VERSION,
    CAMERA_PATCH_SHA256,
    DATASETS,
    HOLDOUT_PHASE,
    HOLDOUT_STRIDE,
    OPENCV_FROM_OPENGL,
    QUERY_FRAME,
    TRAIN_DOWNSAMPLE,
    UPSTREAM_COMMIT,
    dataset_spec,
    depth_cache,
    future_frames,
    reconstruction_holdouts,
    resolve_native,
    resolve_path,
    sha256_file,
    training_frames,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--downsample", type=int, default=TRAIN_DOWNSAMPLE)
    parser.add_argument("--baseline-root", type=Path, default=REPO_ROOT / "baselines" / "EH-SurGS")
    parser.add_argument("--allow-incomplete-depth", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def git_output(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root)] + list(arguments), text=True).strip()


def main():
    args = parse_args()
    spec = dataset_spec(args.dataset_key)
    native = resolve_native(REPO_ROOT, args.dataset_key, args.native)
    offline = resolve_path(REPO_ROOT, spec["offline"])
    ground_truth = resolve_path(REPO_ROOT, spec["ground_truth"])
    camera_file = resolve_path(REPO_ROOT, spec["camera_file"])
    instrument_masks = resolve_path(REPO_ROOT, spec["instrument_masks"])
    baseline = args.baseline_root.expanduser().resolve()
    for path in (offline, ground_truth, camera_file, instrument_masks, baseline / ".git"):
        if not path.exists():
            raise FileNotFoundError(path)
    commit = git_output(baseline, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError("EH-SurGS commit {} != {}".format(commit, UPSTREAM_COMMIT))
    changed = [
        path
        for path in git_output(baseline, "diff", "--name-only", "HEAD").splitlines()
        if path and "/__pycache__/" not in path and not path.endswith(".pyc")
    ]
    if changed != ["train.py"]:
        raise ValueError("Unexpected EH-SurGS source changes: {}".format(changed))
    patch = subprocess.check_output(["git", "-C", str(baseline), "diff", "--", "train.py"])
    patch_hash = hashlib.sha256(patch).hexdigest()
    if patch_hash != CAMERA_PATCH_SHA256:
        raise ValueError("EH-SurGS camera patch hash differs: {}".format(patch_hash))

    frames = int(spec["frames"])
    future_start = int(spec["future_start"])
    train = training_frames(args.dataset_key)
    holdout = reconstruction_holdouts(args.dataset_key)
    future = future_frames(args.dataset_key)
    if set(train) & set(holdout) or set(train) & set(future) or set(holdout) & set(future):
        raise AssertionError("Protocol frame sets overlap")
    if sorted(train + holdout + future) != list(range(frames)):
        raise AssertionError("Protocol frame sets do not cover the fixed sequence")

    left_meta_path = offline / "videos" / "stereo_left.json"
    right_meta_path = offline / "videos" / "stereo_right.json"
    left_meta = json.loads(left_meta_path.read_text(encoding="utf-8"))
    right_meta = json.loads(right_meta_path.read_text(encoding="utf-8"))
    if left_meta["resolution"] != [1920, 1080] or right_meta["resolution"] != [1920, 1080]:
        raise ValueError("SUPER images must remain 1920x1080")
    if len(left_meta["timestamps"]) < frames or len(right_meta["timestamps"]) < frames:
        raise ValueError("Timestamp schedule is shorter than the frozen sequence")
    for frame in range(frames):
        for side in ("left", "right"):
            path = native / "rgb" / "{:06d}-{}.png".format(frame, side)
            if not path.is_file():
                raise FileNotFoundError(path)

    camera_table = json.loads(camera_file.read_text(encoding="utf-8"))
    world_from_camera = np.asarray(
        camera_table["stereo_left"]["X_WC"], dtype=np.float64
    ) @ OPENCV_FROM_OPENGL
    calibration_path = native / "calib_rectified.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    intrinsic = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    with np.load(ground_truth, allow_pickle=False) as archive:
        gt = {name: np.asarray(archive[name]) for name in archive.files}
    if int(gt["future_test_start_frame"].item()) != future_start:
        raise ValueError("Frozen future boundary differs from GT")
    if not np.array_equal(gt["point_ids"], np.arange(10, dtype=np.int32)):
        raise ValueError("Current SUPER GT must contain points 0..9")
    if not np.allclose(gt["K_left_rect"], intrinsic, atol=1.0e-12):
        raise ValueError("Training intrinsic differs from GT camera")
    if not np.allclose(gt["X_world_camera_opencv"], world_from_camera, atol=1.0e-8):
        raise ValueError("Training pose differs from GT camera")
    query_slots = np.flatnonzero(gt["frame_indices"] == QUERY_FRAME)
    if len(query_slots) != 1:
        raise ValueError("GT has no unique frame-0 query")
    query_slot = int(query_slots[0])
    if not bool(gt["visible"][query_slot].all()) or not bool(gt["valid_3d"][query_slot].all()):
        raise ValueError("All ten frame-0 query points must be visible and 3D-valid")

    with np.load(instrument_masks, allow_pickle=False) as archive:
        quality = np.asarray(archive["quality_valid"], dtype=bool)
        instrument_frames = np.asarray(archive["stereo_left_index"], dtype=np.int64)
        instrument_lookup = {
            int(frame) for frame, valid in zip(instrument_frames, quality) if bool(valid)
        }
    for frame in train + holdout + future:
        if frame not in instrument_lookup and frame - 1 not in instrument_lookup and frame + 1 not in instrument_lookup:
            raise ValueError("No formal instrument mask near frame {}".format(frame))

    cache = depth_cache(REPO_ROOT, args.dataset_key, args.downsample)
    actual_depth = sorted(
        int(path.name[:6]) for path in cache.glob("??????-depth.npy")
    ) if cache.exists() else []
    depth_complete = (cache / "COMPLETE").is_file()
    depth_exact = actual_depth == train
    if not args.allow_incomplete_depth and (not depth_complete or not depth_exact):
        raise ValueError("Depth cache incomplete or exposed")
    if any(frame in set(holdout + future) for frame in actual_depth):
        raise ValueError("Depth cache includes a withheld/future frame")

    report = {
        "schema": "eh_surgs_super_protocol_audit_v1",
        "adapter_version": ADAPTER_VERSION,
        "passed": True,
        "dataset_key": args.dataset_key,
        "frames": frames,
        "ground_truth": str(ground_truth),
        "ground_truth_sha256": sha256_file(ground_truth),
        "split": {
            "training_rule": "frame < {} and frame % {} != {}".format(
                future_start, HOLDOUT_STRIDE, HOLDOUT_PHASE
            ),
            "training_frame_count": len(train),
            "reconstruction_holdout_count": len(holdout),
            "future_frame_count": len(future),
        },
        "inputs": {
            "native": str(native),
            "rgb": "rgb/%06d-{left,right}.png",
            "left_metadata_sha256": sha256_file(left_meta_path),
            "right_metadata_sha256": sha256_file(right_meta_path),
            "calibration_sha256": sha256_file(calibration_path),
            "camera_file_sha256": sha256_file(camera_file),
            "instrument_masks_sha256": sha256_file(instrument_masks),
            "training_mask": "all depth-valid pixels outside the frozen SurgicalSAM2 instrument mask",
            "forbidden_during_training": ["ground truth", "GT stereo depth", "holdout RGB", "future RGB"],
        },
        "depth_cache": {
            "path": str(cache),
            "complete": depth_complete,
            "exact_legal_schedule": depth_exact,
            "cached_frame_count": len(actual_depth),
            "uses_ground_truth": False,
        },
        "evaluation_points": {
            "count": 10,
            "query_frame": QUERY_FRAME,
            "all_query_points_visible": True,
            "all_query_points_3d_valid": True,
        },
        "eh_surgs": {
            "root": str(baseline),
            "commit": commit,
            "algorithm_changes": ["train.py reads calibrated intrinsic for adaptive-motion block assignment"],
            "camera_compatibility_patch_sha256": patch_hash,
            "all_other_tracked_sources_clean": True,
            "time_normalization": "frame_index / full_sequence_frame_count",
        },
    }
    if args.output:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
