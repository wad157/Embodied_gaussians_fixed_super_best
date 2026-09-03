#!/usr/bin/env python3
"""Write the requested compact tracking/rendering evaluation table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    ("pure_pbd", "Pure PBD"),
    (
        "pbd_alltracker_depth_rgb_residual",
        "PBD + trajectory + RGB residual",
    ),
    (
        "pbd_alltracker_depth_rgb_residual_online_stiffness",
        "PBD + trajectory + RGB residual + online stiffness",
    ),
)
PROTOCOLS = (
    ("reconstruction_7to1", "Reconstruction 7:1"),
    ("future_80to20", "Future 80:20"),
)
FIELDS = (
    "3d_mean_mm",
    "2d_mean_px",
    "psnr_db",
    "ssim",
    "lpips_alex",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    comparison_path = args.root / "METHOD_COMPARISON.json"
    comparison = json.loads(comparison_path.read_text())
    compact: dict[str, object] = {
        "schema": "super_compact_tracking_rendering_v1",
        "metric_semantics": {
            "3d_mean_mm": "3D Tracking error (mm), lower is better",
            "2d_mean_px": "2D Tracking error (px), lower is better",
            "psnr_db": "PSNR (dB), higher is better",
            "ssim": "SSIM, higher is better",
            "lpips_alex": "LPIPS-Alex, lower is better",
        },
        "protocols": {},
    }
    lines = [
        "# Compact tracking and rendering evaluation",
        "",
        "Only mean 3D/2D tracking errors are retained, as requested.",
        "",
    ]
    for protocol, title in PROTOCOLS:
        protocol_rows: dict[str, object] = {}
        lines.extend(
            [
                f"## {title}",
                "",
                "| Method | 3D Tracking error (mm) ↓ | 2D Tracking error (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, label in METHODS:
            source = comparison["protocols"][protocol][method]
            row = {field: float(source[field]) for field in FIELDS}
            protocol_rows[method] = {"label": label, **row}
            lines.append(
                f"| {label} | {row['3d_mean_mm']:.6f} | "
                f"{row['2d_mean_px']:.6f} | {row['psnr_db']:.6f} | "
                f"{row['ssim']:.6f} | {row['lpips_alex']:.6f} |"
            )
        lines.append("")
        compact["protocols"][protocol] = protocol_rows
    (args.root / "METRICS_COMPACT.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False) + "\n"
    )
    (args.root / "METRICS_COMPACT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
