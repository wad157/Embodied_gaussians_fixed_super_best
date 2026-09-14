#!/usr/bin/env python3
"""Audit one Embodied Gaussians SUPER run before the formal rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.embodied_gaussians_super.protocol import (  # noqa: E402
    ADAPTER_VERSION,
    CAMERAS,
    DATASETS,
    HOLDOUT_PHASE,
    INITIALIZATION_FRAME,
    PAPER_PARAMETERS,
    PROTOCOL,
    UPSTREAM_COMMIT,
    dataset_spec,
    reconstruction_holdouts,
    rendering_frames,
    resolve_path,
    sha256_file,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = dataset_spec(args.dataset_key)
    body = args.body.resolve()
    body_metadata_path = body.with_suffix(".metadata.json")
    body_metadata = json.loads(body_metadata_path.read_text())
    depth_configuration_path = args.depth_dir.resolve() / "configuration.json"
    depth = json.loads(depth_configuration_path.read_text())
    ground_truth = resolve_path(REPO_ROOT, spec["ground_truth"])
    with np.load(ground_truth, allow_pickle=False) as archive:
        gt_frames = np.asarray(archive["frame_indices"], dtype=np.int32)
        future_start = int(archive["future_test_start_frame"].item())
        point_ids = np.asarray(archive["point_ids"], dtype=np.int32)
    require(body_metadata["formal"] is True, "Body initialization is not formal")
    require(body_metadata["upstream_commit"] == UPSTREAM_COMMIT, "Upstream commit mismatch")
    require(body_metadata["adapter_version"] == ADAPTER_VERSION, "Adapter version mismatch")
    require(body_metadata["initialization_cameras"] == list(CAMERAS), "Initialization is not stereo-only")
    require(body_metadata["initialization_frame_left"] == INITIALIZATION_FRAME, "Initialization frame mismatch")
    require(body_metadata["frame_zero_opened"] is False, "Reconstruction holdout frame 0 leaked into initialization")
    require(body_metadata["parameters"] == PAPER_PARAMETERS, "Paper parameter mismatch")
    require(body_metadata["dataset_specific_physical_parameters"] == {}, "Dataset-specific physics detected")
    require(depth["initialization_left_frame"] == INITIALIZATION_FRAME, "Depth frame mismatch")
    require(depth["dataset_specific_depth_range"] is None, "Dataset-specific depth range detected")
    require(depth["lr_consistency"] is False and depth["raft_used"] is False, "Extra depth filtering detected")
    require(depth["ground_truth_used"] is False, "GT was used to construct initialization depth")
    require(future_start == int(spec["future_start"]), "Future split mismatch")
    require(point_ids.shape == (10,), "Formal query count is not ten")
    require(0 in gt_frames, "GT has no frame-0 query")
    require(HOLDOUT_PHASE == 0 and reconstruction_holdouts(args.dataset_key)[0] == 0, "7:1 phase mismatch")
    required_renders = rendering_frames(args.dataset_key)
    require(required_renders[0] == 0 and required_renders[-1] == int(spec["frames"]) - 1, "Render schedule mismatch")
    pose_report_path = resolve_path(REPO_ROOT, spec["pose_report"])
    pose_report = json.loads(pose_report_path.read_text())
    require(pose_report.get("uses_image_based_correction") is False, "Robot control contains image correction")
    report = {
        "schema": "embodied_gaussians_super_protocol_audit_v1",
        "passed": True,
        "dataset_key": args.dataset_key,
        "protocol": PROTOCOL,
        "adapter_version": ADAPTER_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "initialization": {
            "cameras": list(CAMERAS),
            "left_frame": INITIALIZATION_FRAME,
            "frame_zero_withheld": True,
            "depth": "raw FoundationStereo RGB-only; no dataset-specific range/LR/RAFT filters",
            "body_sha256": sha256_file(body),
            "metadata_sha256": sha256_file(body_metadata_path),
            "depth_configuration_sha256": sha256_file(depth_configuration_path),
        },
        "physics": {
            "parameters": PAPER_PARAMETERS,
            "dataset_specific_parameters": {},
            "public_soft_code_available": False,
            "paper_soft_equations_reconstructed": True,
            "unpublished_kS_policy": "kS=1 full Eq. (5) projection; Delaunay rest adjacency",
        },
        "actuation": {
            "source": "raw LND/FK kinematics",
            "pose_driver_sha256": sha256_file(resolve_path(REPO_ROOT, spec["pose_driver"])),
            "pose_report_sha256": sha256_file(pose_report_path),
            "image_based_correction": False,
            "collision": "paper sphere collision only",
        },
        "evaluation": {
            "ground_truth_sha256": sha256_file(ground_truth),
            "query_frame": 0,
            "query_points": 10,
            "future_start": future_start,
            "reconstruction_holdout_phase": HOLDOUT_PHASE,
            "render_frame_count": len(required_renders),
            "shape_of_motion_used": False,
            "trajectory_source": "native PBD particle frames after rollout freeze",
        },
    }
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
