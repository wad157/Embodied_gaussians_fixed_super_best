#!/usr/bin/env python3
"""Repair explicit all-black SuPer frames with zero-order hold and audit backups."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--offline-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--black-max-value", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def five(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum(dtype=np.int64)
    union = np.logical_or(left, right).sum(dtype=np.int64)
    return float(intersection / union) if union else 1.0


def repair_pngs(args: argparse.Namespace, audit_dir: Path) -> list[dict]:
    rgb_dir = args.native_dir / "rgb"
    records = []
    previous_valid: np.ndarray | None = None
    previous_valid_index: int | None = None
    indices = sorted(set(args.indices))
    for index in range(indices[0] - 1, indices[-1] + 1):
        path = rgb_dir / f"{index:06d}-{args.side}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        if index not in indices:
            previous_valid = image
            previous_valid_index = index
            continue
        if int(image.max()) > args.black_max_value:
            raise ValueError(f"Frame {index} is not all-black: max={int(image.max())}")
        if previous_valid is None or previous_valid_index is None:
            raise RuntimeError(f"No valid predecessor for black frame {index}")
        backup = audit_dir / path.name
        shutil.copy2(path, backup)
        if not cv2.imwrite(str(path), previous_valid):
            raise RuntimeError(f"Could not write repaired frame {path}")
        records.append(
            {
                "frame_index": index,
                "source_frame_index": previous_valid_index,
                "original_sha256": sha256(backup),
                "repaired_sha256": sha256(path),
                "original_max_value": int(image.max()),
                "recovery": "duplicated_previous_rectified_frame",
                "reason": "all_black_structurally_valid_image",
            }
        )
    return records


def reencode_video(args: argparse.Namespace, audit_dir: Path) -> dict:
    metadata_path = args.offline_dir / "videos" / f"stereo_{args.side}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame_count = len(metadata["timestamps"])
    video_path = args.offline_dir / "videos" / f"stereo_{args.side}.mp4"
    backup = audit_dir / f"stereo_{args.side}_before_black_frame_repair.mp4"
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(video_path, backup)
    first = cv2.imread(
        str(args.native_dir / "rgb" / f"000000-{args.side}.png"), cv2.IMREAD_COLOR
    )
    if first is None:
        raise FileNotFoundError("First PNG")
    height, width = first.shape[:2]
    timestamps = np.asarray(metadata["timestamps"], dtype=np.float64)
    diffs = np.diff(timestamps)
    fps = float(1.0 / np.median(diffs[diffs > 1e-6]))
    temporary = video_path.with_name(video_path.stem + ".repairing.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {temporary}")
    try:
        for index in range(frame_count):
            frame = cv2.imread(
                str(args.native_dir / "rgb" / f"{index:06d}-{args.side}.png"),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                raise FileNotFoundError(f"Frame {index}")
            writer.write(frame)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(temporary))
    encoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if encoded_count != frame_count:
        raise RuntimeError(f"Encoded {encoded_count} frames, expected {frame_count}")
    temporary.replace(video_path)
    return {
        "path": str(video_path.resolve()),
        "backup": str(backup.resolve()),
        "frame_count": frame_count,
        "fps": fps,
        "sha256": sha256(video_path),
    }


def update_dataset_metadata(args: argparse.Namespace, records: list[dict]) -> None:
    for root in (args.native_dir, args.offline_dir):
        path = root / "super_dataset_metadata.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        details = payload.setdefault("recovered_image_details", [])
        existing = {
            (entry.get("topic"), entry.get("frame_index")) for entry in details
        }
        topic = f"/stereo/slave/{args.side}/image"
        for record in records:
            key = (topic, record["frame_index"])
            if key not in existing:
                details.append({"topic": topic, **record})
        payload.setdefault("counts", {})["recovered_images"] = len(details)
        atomic_json(path, payload)


def repair_masks(args: argparse.Namespace, audit_dir: Path) -> dict | None:
    if args.mask_dir is None:
        return None
    packed_path = args.mask_dir / "tissue_masks_packbits.npy"
    areas_path = args.mask_dir / "areas.npy"
    iou_path = args.mask_dir / "temporal_iou.npy"
    report_path = args.mask_dir / "report.json"
    for path in (areas_path, iou_path, report_path):
        shutil.copy2(path, audit_dir / ("mask_" + path.name))
    packed = np.load(packed_path, mmap_mode="r+")
    original_rows = np.stack([packed[index].copy() for index in args.indices])
    np.save(audit_dir / "mask_original_packed_rows.npy", original_rows)
    areas = np.load(areas_path)
    temporal_iou = np.load(iou_path)
    width = int(json.loads(report_path.read_text())["resolution_wh"][0])
    for index in sorted(set(args.indices)):
        packed[index] = packed[index - 1]
        areas[index] = areas[index - 1]
        temporal_iou[index] = 1.0
    after = max(args.indices) + 1
    if after < len(areas):
        previous = np.unpackbits(packed[after - 1], axis=1)[:, :width] > 0
        current = np.unpackbits(packed[after], axis=1)[:, :width] > 0
        temporal_iou[after] = binary_iou(current, previous)
    packed.flush()
    np.save(areas_path, areas)
    np.save(iou_path, temporal_iou)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ratios = areas.astype(np.float64) / float(report["manual_mask"]["area"])
    report["area_ratio"] = five(ratios)
    report["temporal_iou"] = {
        key: value
        for key, value in five(temporal_iou[1:]).items()
        if key in {"min", "p05", "median"}
    }
    report["gates"]["no_empty_masks"] = bool(np.all(areas > 0))
    report["passed"] = bool(all(report["gates"].values()))
    report["black_frame_zero_order_hold_repair"] = {
        "indices": sorted(set(args.indices)),
        "reason": "source_rgb_frames_are_all_black",
        "lnd_used": False,
    }
    atomic_json(report_path, report)
    return {
        "indices": sorted(set(args.indices)),
        "zero_masks_after": int(np.sum(areas == 0)),
        "report_passed": report["passed"],
    }


def main() -> None:
    args = parse_args()
    args.native_dir = args.native_dir.resolve()
    args.offline_dir = args.offline_dir.resolve()
    if args.mask_dir is not None:
        args.mask_dir = args.mask_dir.resolve()
    audit_dir = args.native_dir / "black_frame_repair_v1"
    if audit_dir.exists():
        raise FileExistsError(audit_dir)
    audit_dir.mkdir(parents=True)
    records = repair_pngs(args, audit_dir)
    video = reencode_video(args, audit_dir)
    update_dataset_metadata(args, records)
    masks = repair_masks(args, audit_dir)
    report = {
        "method": "zero_order_hold_from_previous_valid_rectified_frame",
        "side": args.side,
        "records": records,
        "video": video,
        "masks": masks,
    }
    atomic_json(audit_dir / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
