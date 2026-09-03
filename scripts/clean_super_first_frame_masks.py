#!/usr/bin/env python3
"""Remove only tiny disconnected SAM2 artifacts from reviewed first-frame masks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--min-component-area", type=int, default=100)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=("tissue", "ground"),
        help="Mask labels to clean (default: tissue ground).",
    )
    args = parser.parse_args()
    report = {"min_component_area": args.min_component_area, "labels": {}}
    for label in args.labels:
        path = args.mask_dir / f"000000-{label}.png"
        mask_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            raise FileNotFoundError(path)
        mask = mask_u8 > 0
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8
        )
        sizes = stats[1:, cv2.CC_STAT_AREA]
        keep_ids = np.flatnonzero(sizes >= args.min_component_area) + 1
        cleaned = np.isin(components, keep_ids)
        removed = mask & ~cleaned
        backup = args.mask_dir / f"000000-{label}_before_component_cleanup.png"
        if backup.exists():
            raise FileExistsError(backup)
        shutil.copy2(path, backup)
        cv2.imwrite(str(path), cleaned.astype(np.uint8) * 255)
        report["labels"][label] = {
            "components_before": int(count - 1),
            "component_areas_before": sorted(map(int, sizes), reverse=True),
            "components_kept": int(len(keep_ids)),
            "pixels_before": int(mask.sum()),
            "pixels_removed": int(removed.sum()),
            "pixels_after": int(cleaned.sum()),
            "backup": str(backup.resolve()),
        }
    (args.mask_dir / "000000-component-cleanup-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
