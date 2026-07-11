#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask-aware filter SuPer disparity maps and regenerate depth maps."
    )
    parser.add_argument(
        "--disparity-dir",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "depth_hires_i48",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "depth_hires_i48_filtered",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "data" / "camera_calibration.yaml",
    )
    parser.add_argument(
        "--tissue-mask",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "masks" / "000000-tissue.png",
    )
    parser.add_argument(
        "--ground-mask",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "masks" / "000000-ground.png",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument("--mad-z", type=float, default=4.0)
    parser.add_argument("--median-ksize", type=int, default=5)
    parser.add_argument("--bilateral-d", type=int, default=7)
    parser.add_argument("--bilateral-sigma-color", type=float, default=1.5)
    parser.add_argument("--bilateral-sigma-space", type=float, default=9.0)
    parser.add_argument("--min-depth", type=float, default=0.01)
    parser.add_argument("--max-depth", type=float, default=2.0)
    return parser.parse_args()


def read_opencv_matrix(path: Path, key: str) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    value = fs.getNode(key).mat()
    fs.release()
    if value is None:
        raise KeyError(f"Missing matrix {key} in {path}")
    return value


def read_opencv_sequence(path: Path, key: str) -> list[float]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    node = fs.getNode(key)
    if node.empty():
        fs.release()
        raise KeyError(f"Missing sequence {key} in {path}")
    values = [node.at(i).real() for i in range(node.size())]
    fs.release()
    return values


def rectified_calibration(path: Path) -> tuple[float, float]:
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
    fx = float(p1[0, 0])
    baseline = abs(float(p2[0, 3] / p2[0, 0]))
    if baseline > 1.0:
        baseline *= 1e-3
    return fx, baseline


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def filtered_region(
    disparity: np.ndarray,
    region: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    valid = region & np.isfinite(disparity) & (disparity > 0)
    values = disparity[valid]
    if len(values) == 0:
        return disparity.copy(), valid, {"valid_before": 0, "valid_after": 0}

    lo, hi = np.percentile(values, [args.percentile_low, args.percentile_high])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * max(mad, 1e-6)
    inlier = valid & (disparity >= lo) & (disparity <= hi) & (
        np.abs(disparity - median) <= args.mad_z * robust_sigma
    )

    filled = disparity.copy()
    filled[region & ~inlier] = median
    filled[~region] = median
    if args.median_ksize > 1:
        ksize = args.median_ksize if args.median_ksize % 2 == 1 else args.median_ksize + 1
        filled = cv2.medianBlur(filled.astype(np.float32), ksize)
    if args.bilateral_d > 0:
        filled = cv2.bilateralFilter(
            filled.astype(np.float32),
            args.bilateral_d,
            args.bilateral_sigma_color,
            args.bilateral_sigma_space,
        )

    output = disparity.copy()
    output[region] = filled[region]
    stats = {
        "valid_before": int(valid.sum()),
        "valid_after": int(inlier.sum()),
        "percentile_low": float(lo),
        "percentile_high": float(hi),
        "median": median,
        "mad": mad,
    }
    return output, inlier, stats


def make_depth(disparity: np.ndarray, fx: float, baseline: float, min_depth: float, max_depth: float) -> np.ndarray:
    depth = (fx * baseline) / np.maximum(disparity, 1e-6)
    invalid = (~np.isfinite(depth)) | (depth < min_depth) | (depth > max_depth) | (~np.isfinite(disparity)) | (disparity <= 0)
    depth = depth.astype(np.float32)
    depth[invalid] = np.nan
    return depth


def make_preview(depth: np.ndarray, path: Path) -> None:
    valid = np.isfinite(depth)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.nanpercentile(depth, [2, 98])
        scaled = (depth - lo) / max(float(hi - lo), 1e-6)
        preview = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
        preview[~valid] = 0
    cv2.imwrite(str(path), cv2.applyColorMap(preview, cv2.COLORMAP_MAGMA))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tissue = load_mask(args.tissue_mask)
    ground = load_mask(args.ground_mask)
    fx, baseline = rectified_calibration(args.calibration)

    frames = []
    for n in range(args.count):
        frame_id = args.start + n * args.stride
        disparity_path = args.disparity_dir / f"{frame_id:06d}-disparity.npy"
        if not disparity_path.exists():
            raise FileNotFoundError(disparity_path)
        disparity = np.load(disparity_path).astype(np.float32)
        if disparity.shape != tissue.shape or disparity.shape != ground.shape:
            raise ValueError(f"Mask shape does not match disparity shape for frame {frame_id}.")

        filtered = disparity.copy()
        valid_map = np.zeros(disparity.shape, dtype=np.uint8)
        filtered, tissue_valid, tissue_stats = filtered_region(filtered, tissue, args)
        filtered, ground_valid, ground_stats = filtered_region(filtered, ground, args)
        valid_map[tissue_valid | ground_valid] = 255

        depth = make_depth(filtered, fx, baseline, args.min_depth, args.max_depth)
        np.save(args.output_dir / f"{frame_id:06d}-disparity_filtered.npy", filtered)
        np.save(args.output_dir / f"{frame_id:06d}-depth_filtered.npy", depth)
        cv2.imwrite(str(args.output_dir / f"{frame_id:06d}-valid_mask.png"), valid_map)
        make_preview(depth, args.output_dir / f"{frame_id:06d}-depth_filtered_preview.png")

        stats = {
            "frame": frame_id,
            "disparity_input": str(disparity_path),
            "disparity_filtered": str(args.output_dir / f"{frame_id:06d}-disparity_filtered.npy"),
            "depth_filtered": str(args.output_dir / f"{frame_id:06d}-depth_filtered.npy"),
            "valid_fraction": float(np.isfinite(depth).mean()),
            "median_depth_m": float(np.nanmedian(depth)),
            "p05_depth_m": float(np.nanpercentile(depth, 5)),
            "p95_depth_m": float(np.nanpercentile(depth, 95)),
            "tissue": tissue_stats,
            "ground": ground_stats,
        }
        frames.append(stats)
        print(json.dumps(stats, indent=2))

    summary = {
        "disparity_dir": str(args.disparity_dir),
        "output_dir": str(args.output_dir),
        "fx": fx,
        "baseline_m": baseline,
        "frames": frames,
    }
    (args.output_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
