#!/usr/bin/env python3
"""Summarize three identical SUPER evaluations as mean +/- sample std."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METHODS = (
    ("pure_pbd", "Pure PBD"),
    ("pbd_alltracker_depth_rgb_residual", "PBD + trajectory"),
    (
        "pbd_alltracker_depth_rgb_residual_online_stiffness",
        "PBD + trajectory + strong H3",
    ),
)
PROTOCOLS = (
    ("reconstruction_7to1", "Reconstruction 7:1"),
    ("future_80to20", "Future 80:20"),
)
METRICS = (
    ("3d_error_mm", "3D (mm) down"),
    ("2d_error_px", "2D (px) down"),
    ("psnr_db", "PSNR (dB) up"),
    ("ssim", "SSIM up"),
    ("lpips_alex", "LPIPS down"),
)


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    return label, Path(raw_path).expanduser().resolve()


def load_metrics(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text())
    tracking = report["point_tracking"]
    rendering = report["rendering"]
    integrity = report["integrity"]
    required_integrity = (
        "ground_truth_hash_matches_capture",
        "complete_track_schedule",
        "render_partition_exact",
    )
    failed = [key for key in required_integrity if integrity.get(key) is not True]
    if failed:
        raise RuntimeError(f"integrity failure in {path}: {failed}")
    return {
        "3d_error_mm": float(tracking["3d_error_mm"]["mean"]),
        "2d_error_px": float(tracking["2d_error_px"]["mean"]),
        "psnr_db": float(rendering["psnr_db"]),
        "ssim": float(rendering["ssim"]),
        "lpips_alex": float(rendering["lpips_alex"]),
    }


def assert_identical_file(runs: list[tuple[str, Path]], name: str) -> None:
    reference_label, reference_root = runs[0]
    reference = (reference_root / name).read_bytes()
    for label, root in runs[1:]:
        candidate = (root / name).read_bytes()
        if candidate != reference:
            raise RuntimeError(
                f"{name} mismatch between {reference_label} and {label}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    runs: list[tuple[str, Path]] = args.run
    if len(runs) != 3:
        raise RuntimeError(f"exactly three runs are required, got {len(runs)}")
    for label, root in runs:
        if not (root / "COMPLETE").is_file():
            raise RuntimeError(f"run is incomplete: {label}={root}")
    assert_identical_file(runs, "CODE_AND_INPUT_SHA256.txt")
    assert_identical_file(runs, "RUN_CONFIGURATION.txt")

    per_run: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for label, root in runs:
        per_run[label] = {}
        for protocol, _ in PROTOCOLS:
            per_run[label][protocol] = {}
            for method, _ in METHODS:
                result_path = root / method / protocol / "evaluation_results.json"
                if not result_path.is_file():
                    raise RuntimeError(f"missing result: {result_path}")
                per_run[label][protocol][method] = load_metrics(result_path)

    aggregate: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for protocol, _ in PROTOCOLS:
        aggregate[protocol] = {}
        for method, _ in METHODS:
            aggregate[protocol][method] = {}
            for metric, _ in METRICS:
                values = [per_run[label][protocol][method][metric] for label, _ in runs]
                aggregate[protocol][method][metric] = {
                    "mean": statistics.fmean(values),
                    "sample_std": statistics.stdev(values),
                    "values": values,
                }

    payload = {
        "schema": "super_soft_strong_h3_three_run_average_v1",
        "run_count": 3,
        "standard_deviation": "sample standard deviation (ddof=1)",
        "runs": [{"label": label, "root": str(root)} for label, root in runs],
        "per_run": per_run,
        "aggregate": aggregate,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# SUPER较软参数三次重复评估",
        "",
        "固定初值：`distance=0.01`、`shape=0.0005`、`volume=100000`；",
        "RGB residual、局部刚度关闭；强 H3 单步上限 `0.08`，重建/预测累计",
        "log 范围分别为 `0.60/0.80`。表中为三次独立完整运行的均值 ± 样本标准差。",
        "",
    ]
    for protocol, protocol_label in PROTOCOLS:
        lines.extend(
            [
                f"## {protocol_label}",
                "",
                "| Method | 3D (mm) ↓ | 2D (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, method_label in METHODS:
            values = aggregate[protocol][method]
            cells = []
            for metric, _ in METRICS:
                digits = 6 if metric in {"ssim", "lpips_alex"} else 4
                mean = values[metric]["mean"]
                std = values[metric]["sample_std"]
                cells.append(f"{mean:.{digits}f} ± {std:.{digits}f}")
            lines.append(f"| {method_label} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "## Runs",
            "",
            *[f"- `{label}`: `{root}`" for label, root in runs],
            "",
            "三次运行已通过相同代码/输入哈希、相同配置、完整轨迹计划、GT 哈希和渲染",
            "分区检查。机器可读统计见同目录 `average_results.json`。",
            "",
        ]
    )
    args.output_markdown.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
