#!/usr/bin/env python3
"""Fail-closed audit for PhysTwin under the frozen SUPER protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = REPO_ROOT / "baselines/phystwin_super"
sys.path.insert(0, str(ADAPTER_ROOT))

from protocol import (  # noqa: E402
    COTRACKER_CHECKPOINT,
    DATASETS,
    HOLDOUT_PHASE,
    HOLDOUT_STRIDE,
    INITIALIZATION_FRAME,
    QUERY_FRAME,
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


EXPECTED_HEADLESS_PATCH_SHA256 = "4ad745d0baf8410672b95817f529bb4d9ac0cfbfe45bc6cb1626d0d08df693b8"
EXPECTED_COTRACKER_SHA256 = "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
DEFAULT_COTRACKER_WEIGHT = Path("/home/jwshan/.cache/torch/hub/checkpoints/scaled_offline.pth")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--baseline-root", type=Path, default=REPO_ROOT / "baselines/PhysTwin")
    parser.add_argument("--cotracker-weight", type=Path, default=DEFAULT_COTRACKER_WEIGHT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def git_output(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def text_sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    spec = dataset_spec(args.dataset_key)
    native = resolve_native(REPO_ROOT, args.dataset_key, args.native)
    baseline = args.baseline_root.expanduser().resolve()
    if not (baseline / ".git").is_dir():
        raise FileNotFoundError(f"Missing official PhysTwin checkout: {baseline}")
    commit = git_output(baseline, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"PhysTwin commit {commit} != {UPSTREAM_COMMIT}")
    diff = git_output(baseline, "diff", "--", "qqtt/__init__.py", "qqtt/utils/__init__.py")
    if text_sha256(diff + ("\n" if diff else "")) != EXPECTED_HEADLESS_PATCH_SHA256:
        raise ValueError("PhysTwin headless patch mismatch")
    other = git_output(baseline, "diff", "--name-only", "--", ".",
                       ":(exclude)qqtt/__init__.py", ":(exclude)qqtt/utils/__init__.py")
    if other:
        raise ValueError(f"PhysTwin algorithm source modified:\n{other}")

    frame_count = int(spec["frames"])
    future_start = int(spec["future_start"])
    train = training_frames(args.dataset_key)
    holdout = reconstruction_holdouts(args.dataset_key)
    future = future_frames(args.dataset_key)
    if train[0] != INITIALIZATION_FRAME or QUERY_FRAME not in holdout:
        raise AssertionError("SUPER frame-0 withholding/initialization invariant failed")
    if sorted(train + holdout + future) != list(range(frame_count)):
        raise AssertionError("Protocol frame sets do not exactly cover the sequence")
    depth = depth_cache(REPO_ROOT, args.dataset_key)
    actual_depth = sorted(int(path.name[:6]) for path in depth.glob("??????-depth.npy"))
    if actual_depth != train or not (depth / "COMPLETE").is_file():
        raise ValueError("FoundationStereo cache differs from exact legal training schedule")
    configuration = json.loads((depth / "configuration.json").read_text(encoding="utf-8"))
    if configuration.get("uses_ground_truth") is not False or configuration.get("uses_future_or_holdout_rgb") is not False:
        raise ValueError("Depth-cache provenance violates observation isolation")
    for frame in train:
        if not (native / "rgb" / f"{frame:06d}-left.png").is_file():
            raise FileNotFoundError(f"Missing legal RGB frame {frame}")

    pose_path = resolve_path(REPO_ROOT, spec["pose_driver"])
    pose_report_path = resolve_path(REPO_ROOT, spec["pose_report"])
    pose_report = json.loads(pose_report_path.read_text(encoding="utf-8"))
    if pose_report.get("uses_image_based_correction") is not False:
        raise ValueError("Raw control pose report uses image correction")
    surface = REPO_ROOT / "data/super/psm_robot/psm_surface_gaussians.npz"
    camera_file = resolve_path(REPO_ROOT, spec["camera_file"])
    ground_truth = resolve_path(REPO_ROOT, spec["ground_truth"])
    instrument_masks = resolve_path(REPO_ROOT, spec["instrument_masks"])
    for path in (pose_path, surface, camera_file, ground_truth, instrument_masks):
        if not path.is_file():
            raise FileNotFoundError(path)
    tracker = args.cotracker_weight.expanduser().resolve()
    if not tracker.is_file() or sha256_file(tracker) != EXPECTED_COTRACKER_SHA256:
        raise ValueError("CoTracker checkpoint missing or hash mismatch")
    with np.load(ground_truth, allow_pickle=False) as archive:
        frames = np.asarray(archive["frame_indices"], dtype=np.int32)
        slots = np.flatnonzero(frames == QUERY_FRAME)
        if len(slots) != 1 or np.asarray(archive["point_ids"]).shape != (10,):
            raise ValueError("Frozen SUPER query definition mismatch")
        slot = int(slots[0])
        if not np.asarray(archive["visible"])[slot].all() or not np.asarray(archive["valid_3d"])[slot].all():
            raise ValueError("All ten frame-0 queries must be visible and 3D-valid")

    report = {
        "schema": "phystwin_super_protocol_audit_v1", "passed": True,
        "dataset_key": args.dataset_key, "frames": frame_count, "future_start": future_start,
        "split": {
            "definition": f"train: t < {future_start} and t % {HOLDOUT_STRIDE} != {HOLDOUT_PHASE}; future: t >= {future_start}",
            "training_frame_count": len(train), "reconstruction_holdout_count": len(holdout),
            "future_frame_count": len(future), "query_frame": QUERY_FRAME,
            "initialization_frame": INITIALIZATION_FRAME,
        },
        "inputs": {
            "native": str(native), "depth_cache": str(depth),
            "depth_cache_exact_legal_schedule": True,
            "known_control_sha256": sha256_file(pose_path),
            "robot_surface_sha256": sha256_file(surface),
            "pose_uses_image_based_correction": False,
            "camera_file_sha256": sha256_file(camera_file),
            "ground_truth_sha256": sha256_file(ground_truth),
            "instrument_masks_sha256": sha256_file(instrument_masks),
            "forbidden_during_training": ["ground truth", "frame-0 RGB/depth/mask", "holdout RGB/depth/mask", "future RGB/depth/mask"],
        },
        "phystwin": {
            "root": str(baseline), "commit": commit,
            "headless_import_patch_sha256": EXPECTED_HEADLESS_PATCH_SHA256,
            "algorithm_source_files_clean": True,
            "trajectory_source": "native persistent spring-mass particles plus upstream Gaussian KNN-LBS K=16",
            "shape_of_motion_used": False,
        },
        "tracker": {"name": COTRACKER_CHECKPOINT, "checkpoint_sha256": sha256_file(tracker)},
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
