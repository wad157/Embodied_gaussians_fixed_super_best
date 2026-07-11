#!/usr/bin/env python3

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
        description="Generate initial SuPer/grasp5 depth maps with Python-SuPer RAFT-Stereo."
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
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "depth",
        help="Directory for output depth/disparity maps.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "data" / "camera_calibration.yaml",
        help="Original OpenCV stereo calibration YAML used to reconstruct rectified K/P.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PYTHON_SUPER_ROOT / "depth" / "raft_core" / "weights" / "raft-pretrained.pth",
        help="Python-SuPer RAFT-Stereo checkpoint.",
    )
    parser.add_argument("--start", type=int, default=0, help="First frame index.")
    parser.add_argument("--count", type=int, default=5, help="Number of frame pairs to process.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride. With count=5,stride=10 this writes 0,10,20,30,40.",
    )
    parser.add_argument(
        "--infer-width",
        type=int,
        default=960,
        help="Inference width. Depth is resized back to original image size.",
    )
    parser.add_argument(
        "--infer-height",
        type=int,
        default=540,
        help="Inference height. Depth is resized back to original image size.",
    )
    parser.add_argument("--iters", type=int, default=24, help="RAFT recurrent iterations.")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.01,
        help="Depth values below this are set to NaN in saved maps.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=2.0,
        help="Depth values above this are set to NaN in saved maps.",
    )
    parser.add_argument(
        "--save-disparity",
        action="store_true",
        help="Also save NNNNNN-disparity.npy at original resolution.",
    )
    return parser.parse_args()


def read_opencv_matrix(path: Path, key: str) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {path}")
    value = fs.getNode(key).mat()
    fs.release()
    if value is None:
        raise KeyError(f"Missing matrix {key} in {path}")
    return value


def read_opencv_sequence(path: Path, key: str) -> list[float]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {path}")
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
    # The grasp5 calibration translation is in millimeters; downstream depth is meters.
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
        mixed_precision=False,
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


def image_to_tensor(path: Path, width: int, height: int, device: torch.device) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
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


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p1, p2, baseline_m = rectified_calibration(args.calibration)
    fx_orig = float(p1[0, 0])
    cx_delta_orig = float(p2[0, 2] - p1[0, 2])

    model = load_model(args.checkpoint, device)

    outputs = []
    for n in range(args.count):
        frame_id = args.start + n * args.stride
        left_path = args.rgb_dir / f"{frame_id:06d}-left.png"
        right_path = args.rgb_dir / f"{frame_id:06d}-right.png"
        left_full = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        if left_full is None:
            raise FileNotFoundError(left_path)
        orig_h, orig_w = left_full.shape[:2]

        image1 = image_to_tensor(left_path, args.infer_width, args.infer_height, device)
        image2 = image_to_tensor(right_path, args.infer_width, args.infer_height, device)
        padder = InputPadder(image1.shape)
        image1_pad, image2_pad = padder.pad(image1, image2)

        with torch.no_grad():
            _, flow_up = model(image1_pad, image2_pad, iters=args.iters, test_mode=True)
            flow_up = padder.unpad(flow_up)

        disparity_small = (-flow_up[0, 0]).detach().float().cpu().numpy()
        scale_x = args.infer_width / orig_w
        fx_small = fx_orig * scale_x
        cx_delta_small = cx_delta_orig * scale_x
        denom = disparity_small + cx_delta_small
        depth_small = (fx_small * baseline_m) / np.maximum(denom, 1e-6)

        disparity = cv2.resize(
            disparity_small / scale_x,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        depth = cv2.resize(depth_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )
        invalid = (~np.isfinite(depth)) | (depth < args.min_depth) | (depth > args.max_depth)
        depth[invalid] = np.nan

        depth_path = args.output_dir / f"{frame_id:06d}-depth.npy"
        np.save(depth_path, depth)
        if args.save_disparity:
            np.save(args.output_dir / f"{frame_id:06d}-disparity.npy", disparity)
        make_preview(depth, args.output_dir / f"{frame_id:06d}-depth_preview.png")

        stats = {
            "frame": frame_id,
            "depth_path": str(depth_path),
            "valid_fraction": float(np.isfinite(depth).mean()),
            "median_depth_m": float(np.nanmedian(depth)),
            "p05_depth_m": float(np.nanpercentile(depth, 5)),
            "p95_depth_m": float(np.nanpercentile(depth, 95)),
            "median_disparity_px": float(np.nanmedian(disparity)),
        }
        outputs.append(stats)
        print(json.dumps(stats, indent=2))

    summary = {
        "rgb_dir": str(args.rgb_dir),
        "output_dir": str(args.output_dir),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "baseline_m": baseline_m,
        "fx_orig": fx_orig,
        "infer_size": [args.infer_width, args.infer_height],
        "iters": args.iters,
        "frames": outputs,
    }
    (args.output_dir / "depth_generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
