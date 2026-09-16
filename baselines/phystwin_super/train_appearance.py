#!/usr/bin/env python3
"""Fit PhysTwin query-time tissue Gaussians on legal SUPER frame 1."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization

from common import TissueMasks, load_calibration, load_rgb
from protocol import (
    DATASETS,
    DEPTH_DOWNSAMPLE,
    INITIALIZATION_FRAME,
    UPSTREAM_COMMIT,
    depth_cache,
    resolve_native,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth-weight", type=float, default=0.001)
    return parser.parse_args()


def logit(value):
    return torch.logit(value.clamp(1.0e-4, 1.0 - 1.0e-4))


def ssim(x, y):
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(x, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(x * x, 11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 11, stride=1, padding=5) - mu_x * mu_y
    return (((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) /
            ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))).mean()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    native = resolve_native(REPO_ROOT, args.dataset_key, args.dataset)
    appearance_path = args.preprocess_dir / "appearance.npz"
    preprocess_meta = args.preprocess_dir / "metadata.json"
    if not appearance_path.is_file() or not preprocess_meta.is_file():
        raise FileNotFoundError("Preprocess output is incomplete")
    with np.load(appearance_path, allow_pickle=False) as archive:
        means_np = np.asarray(archive["means_world"], np.float32)
        colors_np = np.asarray(archive["colors_rgb"], np.float32)
    device = torch.device(args.device)
    means = torch.nn.Parameter(torch.from_numpy(means_np).to(device))
    color_logits = torch.nn.Parameter(logit(torch.from_numpy(colors_np).to(device)))
    opacity_logits = torch.nn.Parameter(torch.zeros(len(means_np), device=device))
    log_scales = torch.nn.Parameter(torch.full((len(means_np), 3), math.log(0.00022), device=device))
    quats = torch.zeros((len(means_np), 4), device=device)
    quats[:, 0] = 1.0
    optimizer = torch.optim.Adam([
        {"params": [means], "lr": 2.0e-5},
        {"params": [color_logits], "lr": 2.5e-3},
        {"params": [opacity_logits], "lr": 2.5e-3},
        {"params": [log_scales], "lr": 1.0e-3},
    ])
    masks = TissueMasks(native)
    calibration = load_calibration(REPO_ROOT, args.dataset_key)
    frame = INITIALIZATION_FRAME
    depth = np.load(depth_cache(REPO_ROOT, args.dataset_key) / f"{frame:06d}-depth.npy").astype(np.float32)
    h, w = depth.shape
    image = cv2.resize(load_rgb(native, frame).astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
    tissue = cv2.resize(masks.get(frame).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    image[~tissue] = 0.0
    intrinsic = np.asarray(calibration["stereo_left"]["K"], np.float32).copy()
    intrinsic[:2] /= DEPTH_DOWNSAMPLE
    view = np.asarray(calibration["stereo_left"]["X_CW_cv"], np.float32)
    target_rgb = torch.from_numpy(image).to(device)
    target_mask = torch.from_numpy(tissue).to(device)
    target_depth = torch.from_numpy(depth).to(device)
    K = torch.from_numpy(intrinsic).to(device)
    viewmat = torch.from_numpy(view).to(device)

    history = []
    for iteration in range(args.iterations):
        rendered, alpha, _ = rasterization(
            means=means, quats=quats, scales=log_scales.exp().clamp(5.0e-5, 0.003),
            opacities=opacity_logits.sigmoid(), colors=color_logits.sigmoid(),
            viewmats=viewmat[None], Ks=K[None], width=w, height=h,
            packed=False, render_mode="RGB+D", backgrounds=torch.zeros((1, 3), device=device),
            near_plane=0.01, far_plane=2.0,
        )
        rgb = rendered[0, ..., :3]
        alpha_image = alpha[0, ..., 0]
        predicted_depth = rendered[0, ..., 3] / alpha_image.clamp_min(1.0e-6)
        l1 = torch.abs(rgb - target_rgb).mean()
        dssim = 1.0 - ssim(rgb.permute(2, 0, 1)[None], target_rgb.permute(2, 0, 1)[None])
        valid_depth = target_mask & torch.isfinite(target_depth) & (target_depth > 0)
        depth_loss = torch.abs(predicted_depth[valid_depth] - target_depth[valid_depth]).mean()
        loss = 0.8 * l1 + 0.2 * dssim + args.depth_weight * depth_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_scales.clamp_(math.log(5.0e-5), math.log(0.003))
        if iteration % 25 == 0 or iteration == args.iterations - 1:
            entry = {"iteration": iteration, "loss": float(loss.detach()), "l1": float(l1.detach()),
                     "dssim": float(dssim.detach()), "depth_l1_m": float(depth_loss.detach())}
            history.append(entry)
            print(f"[appearance] {iteration}/{args.iterations - 1} loss={entry['loss']:.6g}", flush=True)

    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "gaussians.npz",
        means_world=means.detach().cpu().numpy().astype(np.float32),
        colors_rgb=color_logits.sigmoid().detach().cpu().numpy().astype(np.float32),
        opacities=opacity_logits.sigmoid().detach().cpu().numpy().astype(np.float32),
        scales=log_scales.exp().detach().cpu().numpy().astype(np.float32),
        quats=quats.cpu().numpy().astype(np.float32),
    )
    metadata = {
        "schema": "phystwin_super_appearance_v1", "dataset_key": args.dataset_key,
        "seed": args.seed, "phystwin_commit": UPSTREAM_COMMIT,
        "representation": "legal-frame-1 tissue Gaussians deformed by native particle LBS",
        "rasterizer": "gsplat compatibility backend", "iterations": args.iterations,
        "downsample": DEPTH_DOWNSAMPLE, "gaussians": int(len(means_np)),
        "preprocess_appearance_sha256": sha256_file(appearance_path),
        "training_frames": [INITIALIZATION_FRAME], "training_cameras": ["stereo_left"],
        "future_observations_used": False, "evaluation_truth_opened": False, "history": history,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
