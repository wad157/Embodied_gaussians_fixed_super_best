#!/usr/bin/env python3
"""
Improved depth generation using RAFT-Stereo at full resolution with max iterations.
Compared to generate_super_depth.py:
  - Full resolution inference (1920x1080) instead of 960x540
  - 64 iterations instead of 24
  - Bilateral filtering for edge-preserving smoothing
  - Option to merge multiple frames for temporal smoothing
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SUPER_ROOT = REPO_ROOT / "third_party" / "Python-SuPer"
if str(PYTHON_SUPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_SUPER_ROOT))

from depth.raft_core.raft_stereo import RAFTStereo  # noqa: E402
from depth.raft_core.utils.utils import InputPadder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate improved SuPer depth maps with RAFT-Stereo at full resolution."
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "rgb",
        help="Directory containing rectified NNNNNN-left.png and NNNNNN-right.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "depth_v2",
        help="Directory for output depth/disparity maps.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "data" / "camera_calibration.yaml",
        help="Original OpenCV stereo calibration YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PYTHON_SUPER_ROOT / "depth" / "raft_core" / "weights" / "raft-pretrained.pth",
        help="RAFT-Stereo checkpoint.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--iters", type=int, default=64, help="RAFT recurrent iterations (default 64).")
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
    )
    parser.add_argument("--min-depth", type=float, default=0.01)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--bilateral-d", type=int, default=5, help="Bilateral filter diameter.")
    parser.add_argument("--bilateral-sigma", type=float, default=0.05, help="Bilateral filter sigma in meters.")
    parser.add_argument("--save-disparity", action="store_true")
    parser.add_argument("--save-disparity-png", action="store_true", help="Save disparity as 16-bit PNG.")
    return parser.parse_args()


def read_opencv_matrix(path: Path, key: str) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open: {path}")
    value = fs.getNode(key).mat()
    fs.release()
    if value is None:
        raise KeyError(f"Missing matrix {key} in {path}")
    return value


def read_opencv_sequence(path: Path, key: str) -> list[float]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open: {path}")
    node = fs.getNode(key)
    if node.empty():
        fs.release()
        raise KeyError(f"Missing sequence {key} in {path}")
    values = [node.at(i).real() for i in range(node.size())]
    fs.release()
    return values


def rectified_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    k1 = read_opencv_matrix(path, "K1")
    k2 = read_opencv_matrix(path, "K2")
    d1 = read_opencv_matrix(path, "D1")
    d2 = read_opencv_matrix(path, "D2")
    r = read_opencv_matrix(path, "R")
    t = np.array(read_opencv_sequence(path, "T"), dtype=np.float64).reshape(3, 1)
    image_size_hw = read_opencv_sequence(path, "ImageSize")
    height, width = int(image_size_hw[0]), int(image_size_hw[1])
    _, _, p1, p2, _, _, _ = cv2.stereoRectify(
        k1, d1, k2, d2, (width, height), r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    baseline = abs(float(p2[0, 3] / p2[0, 0]))
    if baseline > 1.0:
        baseline *= 1e-3
    return p1, p2, baseline


def raft_args() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dims=[128, 128, 128],
        corr_levels=4,
        corr_radius=4,
        shared_backbone=False,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        corr_implementation="reg",
        mixed_precision=True,  # Enable mixed precision for speed
    )


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if device.type == "cuda":
        gpu_id = 0 if device.index is None else device.index
        model = torch.nn.DataParallel(
            RAFTStereo(raft_args()), device_ids=[gpu_id], output_device=gpu_id
        )
    else:
        model = torch.nn.DataParallel(RAFTStereo(raft_args()))
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def image_to_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load image at native resolution."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()[None]
    return tensor.to(device)


def make_preview(depth: np.ndarray, path: Path) -> None:
    valid = np.isfinite(depth)
    if not valid.any():
        preview = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.nanpercentile(depth, [2, 98])
        scaled = (depth - lo) / max(float(hi - lo), 1e-6)
        preview = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
        preview[~valid] = 0
    color = cv2.applyColorMap(preview, cv2.COLORMAP_MAGMA)
    cv2.imwrite(str(path), color)


def bilateral_filter_depth(depth: np.ndarray, d: int, sigma: float) -> np.ndarray:
    """Apply edge-preserving bilateral filter to depth map."""
    valid = np.isfinite(depth)
    if not valid.any():
        return depth
    # Fill invalid with nearest valid
    filled = depth.copy()
    filled[~valid] = np.nanmedian(depth)
    # Convert sigma from meters to intensity units for bilateral filter
    depth_range = np.nanmax(depth) - np.nanmin(depth)
    if depth_range < 1e-6:
        return depth
    sigma_color = sigma / depth_range * 255.0
    # Scale to 0-255 for OpenCV
    d_min, d_max = np.nanmin(filled), np.nanmax(filled)
    scaled = ((filled - d_min) / (d_max - d_min) * 255.0).astype(np.float32)
    filtered = cv2.bilateralFilter(scaled, d, sigma_color * 2, sigma_color)
    result = filtered / 255.0 * (d_max - d_min) + d_min
    result[~valid] = np.nan
    return result.astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p1, p2, baseline_m = rectified_calibration(args.calibration)
    fx_orig = float(p1[0, 0])
    cx_delta_orig = float(p2[0, 2] - p1[0, 2])

    print(f"Calibration: fx={fx_orig:.2f}, baseline={baseline_m*1000:.3f}mm, cx_delta={cx_delta_orig:.2f}")
    print(f"Inference: full resolution ({1920}x{1080}), {args.iters} iterations")

    model = load_model(args.checkpoint, device)

    outputs = []
    for n in range(args.count):
        frame_id = args.start + n * args.stride
        left_path = args.rgb_dir / f"{frame_id:06d}-left.png"
        right_path = args.rgb_dir / f"{frame_id:06d}-right.png"

        image1 = image_to_tensor(left_path, device)
        image2 = image_to_tensor(right_path, device)
        orig_h, orig_w = image1.shape[2], image1.shape[3]
        print(f"  Frame {frame_id}: {orig_w}x{orig_h}")

        padder = InputPadder(image1.shape)
        image1_pad, image2_pad = padder.pad(image1, image2)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                _, flow_up = model(image1_pad, image2_pad, iters=args.iters, test_mode=True)
            flow_up = padder.unpad(flow_up)

        disparity_small = (-flow_up[0, 0]).detach().float().cpu().numpy()
        scale_x = orig_w / orig_w  # = 1.0 since we're at full resolution
        fx = fx_orig * scale_x
        cx_delta = cx_delta_orig * scale_x
        denom = disparity_small + cx_delta

        # Direct depth computation at full resolution (no resize needed)
        depth = (fx * baseline_m) / np.maximum(denom, 1e-6)
        disparity = disparity_small.astype(np.float32)

        # Clip invalid depth
        invalid = (~np.isfinite(depth)) | (depth < args.min_depth) | (depth > args.max_depth)
        depth[invalid] = np.nan

        # Edge-preserving bilateral filter
        if args.bilateral_d > 0:
            depth = bilateral_filter_depth(depth, args.bilateral_d, args.bilateral_sigma)

        # Save
        depth_path = args.output_dir / f"{frame_id:06d}-depth.npy"
        np.save(depth_path, depth)
        if args.save_disparity:
            np.save(args.output_dir / f"{frame_id:06d}-disparity.npy", disparity)
        if args.save_disparity_png:
            disp_valid = disparity.copy()
            disp_valid[np.isnan(depth)] = 0
            cv2.imwrite(str(args.output_dir / f"{frame_id:06d}-disparity.png"),
                        (disp_valid * 256.0).astype(np.uint16))
        make_preview(depth, args.output_dir / f"{frame_id:06d}-depth_preview.png")

        stats = {
            "frame": frame_id,
            "valid_fraction": float(np.isfinite(depth).mean()),
            "median_depth_m": float(np.nanmedian(depth)),
            "p05_depth_m": float(np.nanpercentile(depth, 5)),
            "p95_depth_m": float(np.nanpercentile(depth, 95)),
            "median_disparity_px": float(np.nanmedian(disparity)),
            "depth_range_m": [float(np.nanmin(depth)), float(np.nanmax(depth))],
        }
        outputs.append(stats)
        print(f"    median={stats['median_depth_m']:.4f}m, "
              f"range=[{stats['depth_range_m'][0]:.4f}, {stats['depth_range_m'][1]:.4f}], "
              f"valid={stats['valid_fraction']:.2%}")

    summary = {
        "rgb_dir": str(args.rgb_dir),
        "output_dir": str(args.output_dir),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "baseline_m": baseline_m,
        "fx_orig": fx_orig,
        "resolution": "full",
        "iters": args.iters,
        "bilateral_filter": {"d": args.bilateral_d, "sigma_m": args.bilateral_sigma},
        "frames": outputs,
    }
    (args.output_dir / "depth_generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nDone. {len(outputs)} frames written to {args.output_dir}")


if __name__ == "__main__":
    main()
