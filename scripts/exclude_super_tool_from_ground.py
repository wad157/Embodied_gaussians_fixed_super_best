#!/usr/bin/env python3
"""Subtract an image-annotated tool mask from a SuPer first-frame ground mask."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--ground", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tool-dilation-px", type=int, default=2)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    ground = read_mask(args.ground)
    tool = read_mask(args.tool)
    if ground.shape != image.shape[:2] or tool.shape != image.shape[:2]:
        raise ValueError("Image, ground, and tool shapes differ")
    if args.tool_dilation_px > 0:
        size = 2 * args.tool_dilation_px + 1
        tool_exclusion = cv2.dilate(
            tool.astype(np.uint8), np.ones((size, size), dtype=np.uint8)
        ) > 0
    else:
        tool_exclusion = tool
    removed = ground & tool_exclusion
    cleaned = ground & ~tool_exclusion
    if not removed.any():
        raise RuntimeError("The annotated tool does not overlap the raw ground mask")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "000000-ground_raw_with_tool.png"
    final_path = args.output_dir / "000000-ground.png"
    if raw_path.exists():
        raise FileExistsError(raw_path)
    shutil.copy2(args.ground, raw_path)
    cv2.imwrite(str(final_path), cleaned.astype(np.uint8) * 255)
    cv2.imwrite(
        str(args.output_dir / "000000-tool.png"), tool.astype(np.uint8) * 255
    )

    overlay = image.copy()
    ground_color = np.full_like(image, (40, 190, 40))
    tool_color = np.full_like(image, (220, 40, 220))
    overlay[cleaned] = cv2.addWeighted(image, 0.55, ground_color, 0.45, 0)[cleaned]
    overlay[tool] = cv2.addWeighted(image, 0.45, tool_color, 0.55, 0)[tool]
    cv2.drawContours(
        overlay,
        cv2.findContours(tool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        -1,
        (255, 0, 255),
        2,
    )
    cv2.imwrite(str(args.output_dir / "000000-ground-tool-exclusion-overlay.png"), overlay)
    report = {
        "method": "manual_target_image_tool_mask_subtraction",
        "lnd_used": False,
        "image": str(args.image.resolve()),
        "raw_ground": str(raw_path.resolve()),
        "manual_tool": str(args.tool.resolve()),
        "final_ground": str(final_path.resolve()),
        "tool_dilation_px": args.tool_dilation_px,
        "raw_ground_area": int(ground.sum()),
        "tool_area": int(tool.sum()),
        "removed_ground_area": int(removed.sum()),
        "final_ground_area": int(cleaned.sum()),
    }
    (args.output_dir / "000000-ground-tool-exclusion-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
