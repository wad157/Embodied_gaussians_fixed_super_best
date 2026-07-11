#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Interactively segment tissue and ground on the first SuPer/grasp5 frame with SAM2."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=repo_root / "data" / "super" / "grasp5_native" / "rgb" / "000000-left.png",
        help="RGB/BGR image to segment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "super" / "grasp5_native" / "masks",
        help="Output mask directory.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["tissue", "ground"],
        help="Mask labels to collect in order.",
    )
    parser.add_argument(
        "--model",
        default="facebook/sam2.1-hiera-large",
        help="SAM2 model id for SAM2ImagePredictor.from_pretrained.",
    )
    parser.add_argument("--device", default="cuda", help="SAM2 device.")
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="Initial OpenCV window width.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=720,
        help="Initial OpenCV window height.",
    )
    parser.add_argument(
        "--no-fill-holes",
        action="store_true",
        help="Disable hole filling before saving masks.",
    )
    return parser.parse_args()


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    inverted = cv2.bitwise_not(mask_u8)
    h, w = inverted.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inverted, flood_mask, (0, 0), 255)
    filled_region = flood_mask[1:-1, 1:-1]
    return np.logical_not(filled_region)


def draw_points(image: np.ndarray, fg_points: list[list[int]], bg_points: list[list[int]]) -> np.ndarray:
    out = image.copy()
    for x, y in fg_points:
        cv2.circle(out, (x, y), 7, (0, 255, 0), -1)
        cv2.circle(out, (x, y), 9, (0, 0, 0), 2)
    for x, y in bg_points:
        cv2.circle(out, (x, y), 7, (0, 0, 255), -1)
        cv2.circle(out, (x, y), 9, (0, 0, 0), 2)
    return out


def render_view(
    image_bgr: np.ndarray,
    mask: np.ndarray | None,
    fg_points: list[list[int]],
    bg_points: list[list[int]],
    label: str,
) -> np.ndarray:
    view = image_bgr.copy()
    if mask is not None:
        overlay = image_bgr.copy()
        overlay[mask] = (0, 191, 255)
        view = cv2.addWeighted(overlay, 0.45, image_bgr, 0.55, 0)
    view = draw_points(view, fg_points, bg_points)

    text = [
        f"SAM2: select {label}",
        "Left click: foreground   Right click: background",
        "s: save and continue   r/middle click: reset   q/esc: quit",
    ]
    y = 32
    for line in text:
        cv2.putText(view, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        y += 34
    return view


def collect_mask(
    predictor: SAM2ImagePredictor,
    image_rgb: np.ndarray,
    image_bgr: np.ndarray,
    label: str,
    window_width: int,
    window_height: int,
    do_fill_holes: bool,
) -> np.ndarray | None:
    fg_points: list[list[int]] = []
    bg_points: list[list[int]] = []
    mask: np.ndarray | None = None
    window_name = f"SAM2 {label}"

    def update_prediction() -> None:
        nonlocal mask
        all_points = fg_points + bg_points
        if not all_points:
            mask = None
            return
        point_labels = [1] * len(fg_points) + [0] * len(bg_points)
        masks, _, _ = predictor.predict(
            point_coords=np.asarray(all_points, dtype=np.float32),
            point_labels=np.asarray(point_labels, dtype=np.int32),
            multimask_output=False,
        )
        mask = masks[0].astype(bool)
        if do_fill_holes:
            mask = fill_holes(mask)

    def mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            fg_points.append([x, y])
            update_prediction()
        elif event == cv2.EVENT_RBUTTONDOWN:
            bg_points.append([x, y])
            update_prediction()
        elif event == cv2.EVENT_MBUTTONDOWN:
            fg_points.clear()
            bg_points.clear()
            update_prediction()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, render_view(image_bgr, mask, fg_points, bg_points, label))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("s"), ord("S")):
            cv2.destroyWindow(window_name)
            return mask
        if key in (ord("r"), ord("R")):
            fg_points.clear()
            bg_points.clear()
            update_prediction()
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt


def main() -> None:
    args = parse_args()
    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = SAM2ImagePredictor.from_pretrained(args.model, device=args.device)
    predictor.set_image(image_rgb)

    metadata = {
        "image": str(args.image),
        "model": args.model,
        "device": args.device,
        "labels": {},
    }

    for label in args.labels:
        print(f"Selecting {label}: left click foreground, right click background, s to save.")
        mask = collect_mask(
            predictor=predictor,
            image_rgb=image_rgb,
            image_bgr=image_bgr,
            label=label,
            window_width=args.window_width,
            window_height=args.window_height,
            do_fill_holes=not args.no_fill_holes,
        )
        if mask is None:
            print(f"No mask saved for {label}; no foreground points selected.")
            continue
        mask_path = args.output_dir / f"000000-{label}.png"
        overlay_path = args.output_dir / f"000000-{label}_overlay.png"
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(overlay_path), render_view(image_bgr, mask, [], [], label))
        metadata["labels"][label] = {
            "mask": str(mask_path),
            "overlay": str(overlay_path),
            "area": int(mask.sum()),
        }
        print(f"Saved {label}: {mask_path} area={int(mask.sum())}")

    metadata_path = args.output_dir / "000000-sam2_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
