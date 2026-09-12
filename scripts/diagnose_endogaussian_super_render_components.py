#!/usr/bin/env python3
"""Render EndoGaussian deformation-component ablations for SUPER diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if not entry or Path(entry).resolve() != SCRIPT_DIR]

from arguments import ModelHiddenParams, ModelParams, PipelineParams
from gaussian_renderer import render
from utils.params_utils import merge_hparams

from common import PackedNonInstrumentMasks, load_calibration, load_trained_gaussians
from export_super import make_render_camera
from protocol import DATASETS, resolve_path

REPO_ROOT = Path(__file__).resolve().parents[1]


class DeformationAblation(torch.nn.Module):
    def __init__(self, base, mode: str):
        super().__init__()
        self.base = base
        self.mode = mode

    def forward(self, means, scales, rotations, opacity, time):
        deformed = self.base(means, scales, rotations, opacity, time)
        if self.mode == "standard":
            return deformed
        if self.mode == "centers_only":
            return deformed[0], scales, rotations, opacity
        if self.mode == "shape_only":
            return means, deformed[1], deformed[2], deformed[3]
        if self.mode == "static":
            return means, scales, rotations, opacity
        raise ValueError(self.mode)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    pipeline = PipelineParams(parser)
    hidden = ModelHiddenParams(parser)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 820, 1152, 1439])
    parser.add_argument("--render-scale", type=float, default=0.5)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--configs", required=True)
    args = parser.parse_args()
    import mmcv

    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    return args, pipeline, hidden


def target_image(native: Path, frame: int, size_wh):
    bgr = cv2.imread(str(native / "rgb" / f"{frame:06d}-left.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(frame)
    bgr = cv2.resize(bgr, size_wh, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def masked_l1(a, b, mask):
    return float(np.abs(a - b)[mask].mean())


def masked_psnr(a, b, mask):
    mse = float(np.square(a - b)[mask].mean())
    return float(-10.0 * np.log10(max(mse, 1.0e-12)))


def label(image, text):
    result = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(result, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main():
    args, pipeline, hidden = parse_args()
    spec = DATASETS[args.dataset_key]
    frame_count = int(spec["frames"])
    for frame in args.frames:
        if not 0 <= frame < frame_count:
            raise ValueError(frame)
    native = resolve_path(REPO_ROOT, spec["native"])
    calibration = load_calibration(REPO_ROOT, args.dataset_key)
    masks = PackedNonInstrumentMasks(resolve_path(REPO_ROOT, spec["instrument_masks"]))
    gaussians, iteration = load_trained_gaussians(
        Path(args.model_path), args.iteration, int(args.sh_degree), hidden.extract(args)
    )
    gaussians._deformation.eval()
    base_deformation = gaussians._deformation
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    pipe = pipeline.extract(args)
    modes = ("standard", "centers_only", "shape_only", "static")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    images = {}
    targets = {}
    valid_masks = {}
    with torch.inference_mode():
        for frame in args.frames:
            camera, _ = make_render_camera(calibration, frame, frame_count, args.render_scale)
            size_wh = (camera.image_width, camera.image_height)
            targets[frame] = target_image(native, frame, size_wh)
            mask = masks.get(frame)
            valid_masks[frame] = cv2.resize(
                mask.astype(np.uint8), size_wh, interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            images[frame] = {}
            for mode in modes:
                gaussians._deformation = DeformationAblation(base_deformation, mode)
                image = render(camera, gaussians, pipe, background, stage="fine")["render"]
                images[frame][mode] = (
                    image.clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                )
    gaussians._deformation = base_deformation

    frame0 = int(args.frames[0])
    report = {
        "schema": "endogaussian_super_render_component_diagnostic_v1",
        "dataset_key": args.dataset_key,
        "checkpoint_iteration": iteration,
        "frame0": frame0,
        "note": "shape_only changes learned Gaussian scale/rotation; upstream rendering ignores learned opacity deformation.",
        "frames": {},
    }
    for frame in args.frames:
        mask = valid_masks[frame]
        rows = [label(targets[frame], f"target frame {frame}")]
        frame_report = {
            "target_temporal_l1_to_frame0": masked_l1(
                targets[frame], targets[frame0], mask & valid_masks[frame0]
            )
        }
        for mode in modes:
            image = images[frame][mode]
            rows.append(label(image, mode))
            frame_report[mode] = {
                "psnr_to_target": masked_psnr(image, targets[frame], mask),
                "l1_to_target": masked_l1(image, targets[frame], mask),
                "temporal_l1_to_own_frame0": masked_l1(
                    image, images[frame0][mode], mask & valid_masks[frame0]
                ),
                "l1_to_standard_same_frame": masked_l1(
                    image, images[frame]["standard"], mask
                ),
            }
        report["frames"][str(frame)] = frame_report
        cv2.imwrite(str(args.output_dir / f"frame_{frame:06d}.png"), np.concatenate(rows, axis=1))

    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
