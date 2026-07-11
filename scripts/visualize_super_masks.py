#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clean SuPer first-frame mask overlays.")
    parser.add_argument(
        "--image",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "rgb" / "000000-left.png",
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "masks" / "debug_overlay",
    )
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def overlay_mask(image: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image.copy()
    color = np.zeros_like(image)
    color[:, :] = color_bgr
    out[mask] = cv2.addWeighted(image, 1.0 - alpha, color, alpha, 0.0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color_bgr, 2)
    return out


def write_stats(path: Path, tissue: np.ndarray, ground: np.ndarray) -> None:
    overlap = tissue & ground
    rows = [
        f"image_pixels: {tissue.size}",
        f"tissue_pixels: {int(tissue.sum())}",
        f"ground_pixels: {int(ground.sum())}",
        f"overlap_pixels: {int(overlap.sum())}",
        f"tissue_fraction: {float(tissue.mean()):.6f}",
        f"ground_fraction: {float(ground.mean()):.6f}",
        f"overlap_fraction: {float(overlap.mean()):.6f}",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    tissue = read_mask(args.tissue_mask)
    ground = read_mask(args.ground_mask)
    if tissue.shape != image.shape[:2] or ground.shape != image.shape[:2]:
        raise ValueError("Mask and image shapes do not match.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tissue_overlay = overlay_mask(image, tissue, (40, 40, 230), 0.45)
    ground_overlay = overlay_mask(image, ground, (40, 190, 40), 0.45)

    combined = image.copy()
    combined[tissue] = cv2.addWeighted(image, 0.55, np.full_like(image, (40, 40, 230)), 0.45, 0)[tissue]
    combined[ground] = cv2.addWeighted(image, 0.55, np.full_like(image, (40, 190, 40)), 0.45, 0)[ground]
    cv2.drawContours(
        combined,
        cv2.findContours(tissue.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        -1,
        (0, 0, 255),
        2,
    )
    cv2.drawContours(
        combined,
        cv2.findContours(ground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        -1,
        (0, 255, 0),
        2,
    )

    mask_panel = np.zeros_like(image)
    mask_panel[tissue] = (40, 40, 230)
    mask_panel[ground] = (40, 190, 40)
    mask_panel[tissue & ground] = (220, 40, 220)

    cv2.imwrite(str(args.output_dir / "000000-original.png"), image)
    cv2.imwrite(str(args.output_dir / "000000-tissue-clean-overlay.png"), tissue_overlay)
    cv2.imwrite(str(args.output_dir / "000000-ground-clean-overlay.png"), ground_overlay)
    cv2.imwrite(str(args.output_dir / "000000-combined-clean-overlay.png"), combined)
    cv2.imwrite(str(args.output_dir / "000000-mask-panel.png"), mask_panel)
    write_stats(args.output_dir / "000000-mask-stats.txt", tissue, ground)

    print(f"wrote overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
