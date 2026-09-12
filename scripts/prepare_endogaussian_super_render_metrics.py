#!/usr/bin/env python3
"""Use the exact current SUPER video decoder and mask for baseline render pairs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from torchcodec.decoders import VideoDecoder


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = REPO_ROOT / "baselines" / "endogaussian_super"
sys.path.insert(0, str(ADAPTER_ROOT))

from protocol import (  # noqa: E402
    DATASETS,
    PROTOCOL,
    dataset_spec,
    rendering_frames,
    resolve_path,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--render-scale", type=float, default=0.5)
    parser.add_argument("--delete-prediction-arrays", action="store_true")
    return parser.parse_args()


class InstrumentMasks:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as archive:
            self.shape = tuple(int(value) for value in archive["mask_shape"])
            self.bitorder = str(archive["bitorder"].item())
            valid = np.asarray(archive["quality_valid"], dtype=bool)
            frames = np.asarray(archive["stereo_left_index"], dtype=np.int64)
            self.packed = np.asarray(archive["left_masks_packbits"], dtype=np.uint8)
        self.lookup = {
            int(frame): slot for slot, frame in enumerate(frames) if bool(valid[slot])
        }

    def get(self, frame: int) -> tuple[np.ndarray, int, int]:
        slot = self.lookup.get(frame)
        source = frame
        if slot is None:
            for candidate in (frame - 1, frame + 1):
                if candidate in self.lookup:
                    source = candidate
                    slot = self.lookup[candidate]
                    break
        if slot is None:
            raise RuntimeError(f"No quality-valid instrument mask for frame {frame}")
        mask = np.unpackbits(
            self.packed[slot],
            bitorder=self.bitorder,
            count=self.shape[0] * self.shape[1],
        ).reshape(self.shape).astype(bool)
        return mask, source, abs(source - frame)


def masked_psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    selected = np.asarray(mask, dtype=bool)
    if not bool(selected.any()):
        return float("nan"), float("nan")
    difference = prediction[selected] - target[selected]
    mse = float(np.mean(difference * difference))
    return mse, float("inf") if mse == 0.0 else float(-10.0 * math.log10(mse))


def masked_ssim(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(3):
        x = prediction[..., channel].astype(np.float32)
        y = target[..., channel].astype(np.float32)
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        scores.append(numerator / np.maximum(denominator, 1.0e-12))
    score = np.mean(np.stack(scores, axis=-1), axis=-1)
    valid = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    return float(np.mean(score[valid])) if bool(valid.any()) else float("nan")


def main() -> None:
    args = parse_args()
    spec = dataset_spec(args.dataset_key)
    prediction_dir = args.prediction_dir.resolve()
    capture = args.capture.resolve()
    selected = rendering_frames(args.dataset_key)
    actual = sorted(int(path.stem) for path in prediction_dir.glob("??????.npy"))
    if actual != selected:
        raise ValueError("Prediction render schedule differs from the frozen protocol")
    offline = resolve_path(REPO_ROOT, spec["offline"])
    video = offline / "videos" / "stereo_left.mp4"
    # The current SUPER benchmark constructs OfflineCamera with its default
    # CUDA decoder.  TorchCodec's CPU and CUDA paths use different video range
    # conversion for these MP4s, which is visible in PSNR.  Keep this target
    # path byte-for-byte aligned with OfflineCamera instead of silently using
    # the CPU decoder in the metric-only environment.
    decoder = VideoDecoder(video, device="cuda", dimension_order="NHWC")
    instrument_path = resolve_path(REPO_ROOT, spec["instrument_masks"])
    masks = InstrumentMasks(instrument_path)
    pairs = capture / "render_pairs"
    pairs.mkdir(exist_ok=False)
    records = []
    for progress, frame in enumerate(selected, start=1):
        started = time.perf_counter()
        prediction = np.load(prediction_dir / f"{frame:06d}.npy").astype(np.float32)
        decoded_rgb = decoder.get_frame_at(frame).data.cpu().numpy().astype(np.float32) / 255.0
        target = cv2.resize(
            decoded_rgb,
            (prediction.shape[1], prediction.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        instrument, mask_source, mask_gap = masks.get(frame)
        instrument = cv2.resize(
            instrument.astype(np.uint8),
            (prediction.shape[1], prediction.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        valid = ~instrument
        mse, psnr = masked_psnr(prediction, target, valid)
        ssim = masked_ssim(prediction, target, valid)
        masked_prediction = prediction.copy()
        masked_target = target.copy()
        masked_prediction[~valid] = 0.0
        masked_target[~valid] = 0.0
        stem = f"{frame:06d}"
        cv2.imwrite(
            str(pairs / f"{stem}-prediction.png"),
            cv2.cvtColor(np.clip(masked_prediction * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(pairs / f"{stem}-target.png"),
            cv2.cvtColor(np.clip(masked_target * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(pairs / f"{stem}-valid.png"), valid.astype(np.uint8) * 255)
        records.append({
            "frame_index": frame,
            "observation_used": False,
            "mse": mse,
            "psnr_db": psnr,
            "ssim": ssim,
            "valid_pixel_count": int(valid.sum()),
            "instrument_mask_source_frame": mask_source,
            "instrument_mask_frame_gap": mask_gap,
            "render_elapsed_s": time.perf_counter() - started,
        })
        print(f"[metric-pair] {progress}/{len(selected)} frame={frame}", flush=True)
    report = {
        "schema": "super_tissue_render_metrics_without_lpips_v1",
        "protocol": PROTOCOL,
        "records": records,
        "mean": {
            key: float(np.mean([record[key] for record in records]))
            for key in ("mse", "psnr_db", "ssim", "render_elapsed_s")
        },
        "target_decoder": (
            "torchcodec VideoDecoder(device=cuda, dimension_order=NHWC), "
            "identical SUPER OfflineCamera frame index and color path"
        ),
        "target_video": str(video),
        "target_video_sha256": sha256_file(video),
        "instrument_masks": str(instrument_path),
        "instrument_masks_sha256": sha256_file(instrument_path),
        "note": "LPIPS is computed by scripts/score_super_joint_80to20_evaluation.py.",
    }
    (capture / "render_metrics_partial.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if args.delete_prediction_arrays:
        shutil.rmtree(prediction_dir)
    print(json.dumps(report["mean"], indent=2))


if __name__ == "__main__":
    main()
