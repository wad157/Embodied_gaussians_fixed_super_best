#!/usr/bin/env python3
"""Fail-closed audit for the RGB-only, no-PSM TRACE SUPER baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = REPO_ROOT / "baselines" / "trace_super"
DEFAULT_TRACE_ROOT = Path(
    "/Media_HDD/jwshan/wad/embodied_gaussians_fixed_super_best_sim/baselines/TRACE"
)
sys.path.insert(0, str(ADAPTER_ROOT))
from protocol import (  # noqa: E402
    ADAPTER_VERSION,
    CAMERAS,
    DATASETS,
    DEFAULT_INIT_POINTS,
    DEFAULT_ITERATIONS,
    DEFAULT_TRAIN_DOWNSAMPLE,
    HOLDOUT_PHASE,
    HOLDOUT_STRIDE,
    OPENCV_FROM_OPENGL,
    QUERY_FRAME,
    UPSTREAM_COMMIT,
    dataset_spec,
    future_frames,
    max_observed_time,
    reconstruction_holdouts,
    resolve_native,
    resolve_path,
    sha256_file,
    training_frames,
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument(
        "--trace-python",
        type=Path,
        default=Path("/Media_HDD/jwshan/conda_envs/freegave/bin/python"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = dataset_spec(args.dataset_key)
    native = resolve_native(REPO_ROOT, args.dataset_key, args.native)
    offline = resolve_path(REPO_ROOT, spec["offline"])
    camera_file = resolve_path(REPO_ROOT, spec["camera_file"])
    ground_truth = resolve_path(REPO_ROOT, spec["ground_truth"])
    instrument_masks = resolve_path(REPO_ROOT, spec["instrument_masks"])
    for path in (offline, camera_file, ground_truth, instrument_masks):
        if not path.exists():
            raise FileNotFoundError(path)

    upstream = args.baseline_root.expanduser().resolve()
    if git(upstream, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError("TRACE checkout is not pinned to {}".format(UPSTREAM_COMMIT))
    changed = [
        line
        for line in git(
            upstream, "status", "--short", "--untracked-files=no"
        ).splitlines()
        if "/__pycache__/" not in line and not line.rstrip().endswith(".pyc")
    ]
    if changed:
        raise ValueError(
            "TRACE core checkout is modified:\n{}".format("\n".join(changed))
        )
    environment = subprocess.check_output(
        [
            str(args.trace_python),
            "-c",
            "import sys,torch; import diff_gaussian_rasterization,simple_knn; "
            "print(sys.version.split()[0]); print(torch.__version__); print(torch.version.cuda)",
        ],
        text=True,
    ).splitlines()
    if environment[:3] != ["3.7.16", "1.13.1", "11.6"]:
        raise ValueError("unexpected TRACE environment: {}".format(environment))

    frames = int(spec["frames"])
    future_start = int(spec["future_start"])
    train = training_frames(args.dataset_key)
    holdout = reconstruction_holdouts(args.dataset_key)
    future = future_frames(args.dataset_key)
    if sorted(train + holdout + future) != list(range(frames)):
        raise AssertionError("protocol frame sets do not cover the frozen sequence")
    if set(train) & set(holdout) or set(train) & set(future):
        raise AssertionError("protocol frame sets overlap")

    calibration_path = native / "calib_rectified.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    camera_table = json.loads(camera_file.read_text(encoding="utf-8"))
    full_k = {}
    metadata_hashes = {}
    camera_hash = sha256_file(camera_file)
    for camera_name in CAMERAS:
        suffix = "left" if camera_name == "stereo_left" else "right"
        metadata_path = offline / "videos" / (camera_name + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        timestamps = np.asarray(metadata["timestamps"], dtype=np.float64)
        if len(timestamps) < frames or not np.isfinite(timestamps[:frames]).all():
            raise ValueError("{} timestamp schedule is incomplete".format(camera_name))
        if not np.all(np.diff(timestamps[:frames]) > 0.0):
            raise ValueError("{} timestamps are not strictly increasing".format(camera_name))
        width, height = [int(value) for value in metadata["resolution"]]
        if [width, height] != [1920, 1080]:
            raise ValueError("SUPER images must remain 1920x1080")
        intrinsic = np.asarray(
            calibration["K_{}_rect".format(suffix)], dtype=np.float64
        )
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("invalid K for {}".format(camera_name))
        transform = np.asarray(camera_table[camera_name]["X_WC"], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("invalid camera pose for {}".format(camera_name))
        world_from_camera = transform @ OPENCV_FROM_OPENGL
        full_k[camera_name] = {
            "fx": float(intrinsic[0, 0]),
            "fy": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]),
            "cy": float(intrinsic[1, 2]),
            "width": width,
            "height": height,
            "principal_offset_from_center_px": [
                float(intrinsic[0, 2] - width / 2.0),
                float(intrinsic[1, 2] - height / 2.0),
            ],
            "X_WC_opencv": world_from_camera.tolist(),
        }
        metadata_hashes[camera_name] = sha256_file(metadata_path)
        for frame in range(frames):
            image = native / "rgb" / "{:06d}-{}.png".format(frame, suffix)
            if not image.is_file():
                raise FileNotFoundError(image)

    with np.load(str(ground_truth), allow_pickle=False) as archive:
        gt = {name: np.asarray(archive[name]) for name in archive.files}
    if int(gt["future_test_start_frame"].item()) != future_start:
        raise ValueError("GT future boundary differs from the frozen protocol")
    if not np.array_equal(gt["point_ids"], np.arange(10, dtype=np.int32)):
        raise ValueError("SUPER GT must contain point IDs 0..9")
    query_slots = np.flatnonzero(gt["frame_indices"] == QUERY_FRAME)
    if len(query_slots) != 1:
        raise ValueError("SUPER GT has no unique frame-0 query")
    query_slot = int(query_slots[0])
    if not bool(gt["visible"][query_slot].all()) or not bool(
        gt["valid_3d"][query_slot].all()
    ):
        raise ValueError("All frame-0 query points must be visible and 3D-valid")

    report = {
        "schema": "trace_super_protocol_audit_v1",
        "passed": True,
        "adapter_version": ADAPTER_VERSION,
        "dataset_key": args.dataset_key,
        "native": str(native),
        "frames": frames,
        "future_start": future_start,
        "max_observed_time": max_observed_time(args.dataset_key),
        "split": {
            "training_rule": "frame < {} and frame % {} != {}".format(
                future_start, HOLDOUT_STRIDE, HOLDOUT_PHASE
            ),
            "training_frame_count": len(train),
            "training_view_count": len(train) * len(CAMERAS),
            "reconstruction_holdout_count": len(holdout),
            "future_frame_count": len(future),
        },
        "training_configuration": {
            "iterations": DEFAULT_ITERATIONS,
            "downsample": DEFAULT_TRAIN_DOWNSAMPLE,
            "initial_points": DEFAULT_INIT_POINTS,
        },
        "training_inputs": {
            "allowed": [
                "legal prefix rectified stereo-left RGB",
                "complete K",
                "static stereo X_WC",
                "timestamps",
            ],
            "forbidden": [
                "PSM poses or controls",
                "depth of any kind",
                "tissue or instrument masks",
                "ground-truth trajectories",
                "evaluation point IDs/pixels",
                "held-out RGB",
                "future RGB",
            ],
            "metadata_sha256": metadata_hashes,
            "calibration_sha256": sha256_file(calibration_path),
            "camera_file_sha256": camera_hash,
            "full_k": full_k,
        },
        "evaluation_only": {
            "ground_truth": str(ground_truth),
            "ground_truth_sha256": sha256_file(ground_truth),
            "instrument_masks": str(instrument_masks),
            "instrument_masks_sha256": sha256_file(instrument_masks),
            "query_frame": QUERY_FRAME,
            "query_point_count": 10,
            "note": "GT query and frozen instrument mask are opened only after checkpoint freeze by export/scoring",
        },
        "trace": {
            "root": str(upstream),
            "commit": UPSTREAM_COMMIT,
            "tracked_core_clean": True,
            "freegave": False,
            "camera_change": "adapter-side asymmetric projection matrix from full K",
            "algorithm_changes": [],
            "shape_of_motion_used": False,
        },
        "environment": {
            "python": environment[0],
            "torch": environment[1],
            "torch_cuda": environment[2],
        },
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
