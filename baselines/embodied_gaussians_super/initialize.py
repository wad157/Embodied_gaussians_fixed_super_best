#!/usr/bin/env python3
"""Initialize an EG body from the first legal SUPER stereo training pair."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import warp as wp
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_FILE = (
    REPO_ROOT.parent
    / "embodied_gaussians_fixed_super_best_sim/baselines/embodied_gaussians_sim/initialize.py"
)
sys.path.insert(0, str(REPO_ROOT))

from baselines.embodied_gaussians_super.common import (  # noqa: E402
    TissueMasks,
    load_calibration,
    nearest_timestamp_index,
)
from baselines.embodied_gaussians_super.protocol import (  # noqa: E402
    ADAPTER_VERSION,
    CAMERAS,
    DATASETS,
    INITIALIZATION_FRAME,
    PAPER_PARAMETERS,
    UPSTREAM_COMMIT,
    dataset_spec,
    resolve_native,
    sha256_file,
)
from embodied_gaussians.scene_builders.domain import MaskedPosedImageAndDepth  # noqa: E402


def load_shared_initializer():
    if not SHARED_FILE.is_file():
        raise FileNotFoundError(f"Missing audited SIM EG implementation: {SHARED_FILE}")
    shared_root = str(SHARED_FILE.parent)
    sys.path.insert(0, shared_root)
    spec = importlib.util.spec_from_file_location("eg_sim_shared_initializer", SHARED_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SHARED_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_datapoints(repo_root: Path, dataset_key: str, native: Path, depth_dir: Path):
    calibration = load_calibration(repo_root, dataset_key)
    masks = TissueMasks(native)
    left_frame = INITIALIZATION_FRAME
    right_frame = nearest_timestamp_index(
        np.asarray(calibration["right_timestamps"]),
        float(np.asarray(calibration["left_timestamps"])[left_frame]),
    )
    width = int(PAPER_PARAMETERS["online_width"])
    result = []
    provenance = []
    for camera, side, frame in (
        ("stereo_left", "left", left_frame),
        ("stereo_right", "right", right_frame),
    ):
        rgb_path = native / "rgb" / f"{frame:06d}-{side}.png"
        depth_path = depth_dir / f"{frame:06d}-{side}-depth.npy"
        image = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = np.load(depth_path).astype(np.float32)
        mask = masks.get(camera, frame)
        source_height, source_width = image.shape[:2]
        target_height = int(round(source_height * width / source_width))
        size = (width, target_height)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        if depth.shape != (target_height, width):
            raise ValueError(f"Unexpected depth shape {depth.shape} for {depth_path}")
        depth = np.where(np.isfinite(depth) & (depth > 0.0) & mask, depth, 0.0).astype(np.float32)
        intrinsic = np.asarray(calibration[camera]["K"], dtype=np.float32).copy()
        intrinsic[0, :] *= width / source_width
        intrinsic[1, :] *= target_height / source_height
        intrinsic[2, :] = (0.0, 0.0, 1.0)
        result.append(
            MaskedPosedImageAndDepth(
                X_WC=np.asarray(calibration[camera]["X_WC_gl"], dtype=np.float32),
                K=intrinsic,
                image=np.ascontiguousarray(image),
                format="rgb",
                depth=np.ascontiguousarray(depth),
                depth_scale=1.0,
                mask=np.ascontiguousarray(mask),
            )
        )
        provenance.append(
            {
                "camera": camera,
                "source_frame": frame,
                "rgb_sha256": sha256_file(rgb_path),
                "depth_sha256": sha256_file(depth_path),
                "valid_depth_in_mask_fraction": float(np.count_nonzero(depth) / max(np.count_nonzero(mask), 1)),
            }
        )
    return result, provenance


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("Seed must be non-negative")
    native = resolve_native(REPO_ROOT, args.dataset_key, args.dataset)
    depth_dir = args.depth_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite initialization: {output}")
    if not (depth_dir / "COMPLETE").is_file():
        raise FileNotFoundError(f"Initial stereo depth is incomplete: {depth_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("EG public initializer requires CUDA")
    torch.cuda.manual_seed_all(args.seed)
    wp.init()
    shared = load_shared_initializer()
    shared.install_gsplat_compatibility_adapter()
    source_hashes = shared.assert_upstream_sources()
    datapoints, provenance = load_datapoints(
        REPO_ROOT, args.dataset_key, native, depth_dir
    )
    body, geometry = shared.build_body(
        datapoints,
        name=f"super_{args.dataset_key}_eg_paper_soft",
        radius=float(PAPER_PARAMETERS["particle_radius_m"]),
        particle_iterations=int(PAPER_PARAMETERS["initial_particle_iterations"]),
        gaussian_iterations=int(PAPER_PARAMETERS["initial_gaussian_iterations"]),
    )
    output.write_text(body.model_dump_json(indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema": "embodied_gaussians_super_initialization_v1",
        "formal": True,
        "dataset_key": args.dataset_key,
        "seed": args.seed,
        "adapter_version": ADAPTER_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "verified_upstream_source_sha256": source_hashes,
        "initialization_frame_left": INITIALIZATION_FRAME,
        "initialization_cameras": list(CAMERAS),
        "input": "first legal stereo training pair: RGB + raw FoundationStereo depth + tissue masks",
        "frame_zero_opened": False,
        "ground_truth_opened": False,
        "dataset_specific_physical_parameters": {},
        "parameters": PAPER_PARAMETERS,
        "depth_configuration_sha256": sha256_file(depth_dir / "configuration.json"),
        "observations": provenance,
        "particle_count": len(body.particles.means),
        "gaussian_count": len(body.gaussians.means),
        "geometry": geometry,
        "ground": "public EG z=0 ground",
        "shared_equation_implementation": str(SHARED_FILE),
        "shared_equation_implementation_sha256": sha256_file(SHARED_FILE),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
