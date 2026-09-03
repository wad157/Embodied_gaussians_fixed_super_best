#!/usr/bin/env python3
"""Segment a SuPer first frame using reviewed grasp5 masks as SAM2 prompts.

The three grasp recordings share the frozen stereo camera and tissue scene,
but the tool occupies different pixels.  The reference masks are therefore
used only as low-resolution prompts; SAM2 predicts new masks on the target
image and the resulting overlays remain explicit QC artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=REPO / "data/super/grasp5_native/masks",
    )
    parser.add_argument("--model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--tool-point",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("X", "Y"),
        help="Image-selected positive tool point. Repeat for multiple points.",
    )
    parser.add_argument(
        "--tool-box",
        type=float,
        nargs=4,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Image-selected XYXY tool box. Required when --tool-point is used.",
    )
    parser.add_argument(
        "--accept-tool-mask",
        action="store_true",
        help="Subtract the visually reviewed SAM2 tool candidate from tissue/ground.",
    )
    parser.add_argument("--foreground-points", type=int, default=12)
    parser.add_argument("--background-points", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != shape:
        raise ValueError(f"Reference mask {path} shape {mask.shape} != target {shape}")
    return mask > 0


def spread_points(mask: np.ndarray, count: int, radius: int = 90) -> np.ndarray:
    """Greedily select well-spaced, deep-interior XY points."""
    work = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    points: list[list[float]] = []
    for _ in range(count):
        _min_value, max_value, _min_location, max_location = cv2.minMaxLoc(work)
        if max_value <= 1.0:
            break
        x, y = max_location
        points.append([float(x), float(y)])
        cv2.circle(work, (x, y), radius, 0.0, thickness=-1)
    if not points:
        raise RuntimeError("Could not select prompt points from reference mask")
    return np.asarray(points, dtype=np.float32)


def mask_prompt(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA
    )
    # SAM low-resolution mask prompts are logits.  Keep boundary pixels soft
    # so the model can move the old boundary around the target tool.
    probability = np.clip(0.02 + 0.96 * resized, 1e-4, 1.0 - 1e-4)
    logits = np.log(probability / (1.0 - probability))
    return logits[None].astype(np.float32)


def bbox(mask: np.ndarray, margin: int = 8) -> np.ndarray:
    ys, xs = np.where(mask)
    height, width = mask.shape
    return np.asarray(
        [
            max(0, int(xs.min()) - margin),
            max(0, int(ys.min()) - margin),
            min(width - 1, int(xs.max()) + margin),
            min(height - 1, int(ys.max()) + margin),
        ],
        dtype=np.float32,
    )


def predict_label(
    predictor: SAM2ImagePredictor,
    foreground: np.ndarray,
    background: np.ndarray,
    foreground_count: int,
    background_count: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    fg_points = spread_points(foreground, foreground_count)
    bg_points = spread_points(background, background_count)
    points = np.concatenate([fg_points, bg_points], axis=0)
    labels = np.concatenate(
        [np.ones(len(fg_points), dtype=np.int32), np.zeros(len(bg_points), dtype=np.int32)]
    )
    logits, scores, _low_res = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=bbox(foreground),
        mask_input=mask_prompt(foreground),
        multimask_output=False,
        return_logits=True,
    )
    return logits[0], float(scores[0]), fg_points, bg_points


def predict_tool(
    predictor: SAM2ImagePredictor,
    points: np.ndarray,
    box_xyxy: np.ndarray,
    tissue_reference: np.ndarray,
    ground_reference: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    # Negative clicks come from deep tissue/ground areas, well away from the
    # upper tool box.  This prevents the box prompt from swallowing the scene.
    negative_candidates = np.logical_or(tissue_reference, ground_reference).copy()
    negative_candidates[: int(box_xyxy[3]) + 30] = False
    negative = spread_points(negative_candidates, 8, radius=150)
    all_points = np.concatenate([points, negative], axis=0)
    labels = np.concatenate(
        [np.ones(len(points), dtype=np.int32), np.zeros(len(negative), dtype=np.int32)]
    )
    logits, scores, _low_res = predictor.predict(
        point_coords=all_points,
        point_labels=labels,
        box=box_xyxy,
        multimask_output=False,
        return_logits=True,
    )
    return logits[0], float(scores[0]), negative


def overlay(image: np.ndarray, tissue: np.ndarray, ground: np.ndarray) -> np.ndarray:
    colors = image.copy()
    colors[tissue] = (0, 180, 255)
    colors[ground] = (255, 120, 0)
    return cv2.addWeighted(colors, 0.45, image, 0.55, 0.0)


def main() -> None:
    args = parse_args()
    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    shape = image_bgr.shape[:2]
    references = {
        label: load_mask(args.reference_dir / f"000000-{label}.png", shape)
        for label in ("tissue", "ground")
    }

    managed = [
        args.output_dir / "000000-tissue.png",
        args.output_dir / "000000-ground.png",
        args.output_dir / "000000-tool.png",
        args.output_dir / "000000-tissue_overlay.png",
        args.output_dir / "000000-ground_overlay.png",
        args.output_dir / "000000-tool_overlay.png",
        args.output_dir / "000000-combined_overlay.png",
        args.output_dir / "000000-sam2_metadata.json",
    ]
    collisions = [path for path in managed if path.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError("Outputs already exist:\n- " + "\n- ".join(map(str, collisions)))

    predictor = SAM2ImagePredictor.from_pretrained(args.model, device=args.device)
    predictor.set_image(image_rgb)
    tissue_logits, tissue_score, tissue_fg, tissue_bg = predict_label(
        predictor,
        references["tissue"],
        references["ground"],
        args.foreground_points,
        args.background_points,
    )
    ground_logits, ground_score, ground_fg, ground_bg = predict_label(
        predictor,
        references["ground"],
        references["tissue"],
        args.foreground_points,
        args.background_points,
    )
    if bool(args.tool_point) != bool(args.tool_box):
        raise ValueError("--tool-point and --tool-box must be supplied together")
    tool_score = None
    tool_bg = np.empty((0, 2), dtype=np.float32)
    tool_points = np.asarray(args.tool_point, dtype=np.float32)
    tool_box = (
        np.asarray(args.tool_box, dtype=np.float32) if args.tool_box is not None else None
    )
    if len(tool_points):
        tool_logits, tool_score, tool_bg = predict_tool(
            predictor,
            tool_points,
            tool_box,
            references["tissue"],
            references["ground"],
        )
        tool = tool_logits > 0.0
        tool = cv2.dilate(tool.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    else:
        tool = np.zeros(shape, dtype=bool)

    tissue = tissue_logits > 0.0
    ground = ground_logits > 0.0
    if args.accept_tool_mask:
        if not len(tool_points):
            raise ValueError("--accept-tool-mask requires image-selected tool prompts")
        tissue[tool] = False
        ground[tool] = False
    overlap = tissue & ground
    tissue[overlap] = tissue_logits[overlap] >= ground_logits[overlap]
    ground[overlap] = ~tissue[overlap]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = overlay(image_bgr, tissue, ground)
    for label, mask in (("tissue", tissue), ("ground", ground)):
        cv2.imwrite(str(args.output_dir / f"000000-{label}.png"), mask.astype(np.uint8) * 255)
        cv2.imwrite(
            str(args.output_dir / f"000000-{label}_overlay.png"),
            overlay(
                image_bgr,
                mask if label == "tissue" else np.zeros_like(mask),
                mask if label == "ground" else np.zeros_like(mask),
            ),
        )
    cv2.imwrite(str(args.output_dir / "000000-tool.png"), tool.astype(np.uint8) * 255)
    tool_overlay = image_bgr.copy()
    tool_overlay[tool] = (255, 0, 255)
    cv2.imwrite(
        str(args.output_dir / "000000-tool_overlay.png"),
        cv2.addWeighted(tool_overlay, 0.45, image_bgr, 0.55, 0.0),
    )
    cv2.imwrite(str(args.output_dir / "000000-combined_overlay.png"), combined)

    union = tissue | ground
    metadata = {
        "image": str(args.image.resolve()),
        "image_sha256": sha256(args.image),
        "model": args.model,
        "device": args.device,
        "method": "sam2_reference_scene_prompts_and_manual_image_tool_prompts",
        "reference_dir": str(args.reference_dir.resolve()),
        "reference_sha256": {
            label: sha256(args.reference_dir / f"000000-{label}.png")
            for label in ("tissue", "ground")
        },
        "scores": {"tissue": tissue_score, "ground": ground_score, "tool": tool_score},
        "tool_mask_applied": bool(args.accept_tool_mask),
        "tool_prompt_source": "manual_target_image_coordinates" if len(tool_points) else None,
        "lnd_used_for_segmentation": False,
        "areas": {
            "tissue": int(tissue.sum()),
            "ground": int(ground.sum()),
            "tool": int(tool.sum()),
            "overlap_after_resolution": int((tissue & ground).sum()),
            "uncovered": int((~union).sum()),
        },
        "prompts": {
            "tissue": {"foreground_xy": tissue_fg.tolist(), "background_xy": tissue_bg.tolist()},
            "ground": {"foreground_xy": ground_fg.tolist(), "background_xy": ground_bg.tolist()},
            "tool": {
                "foreground_xy": tool_points.tolist(),
                "background_xy": tool_bg.tolist(),
                "box_xyxy": tool_box.tolist() if tool_box is not None else None,
            },
        },
        "requires_visual_qc": not bool(args.accept_tool_mask),
    }
    (args.output_dir / "000000-sam2_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
