#!/usr/bin/env python3
"""Summarize three matched full-f1 SUPER reconstruction evaluations."""

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
METRICS = (
    ("3d_error_mm", "3D (mm) ↓", 4),
    ("2d_error_px", "2D (px) ↓", 4),
    ("psnr_db", "PSNR (dB) ↑", 4),
    ("ssim", "SSIM ↑", 6),
    ("lpips_alex", "LPIPS ↓", 6),
)


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    return label, Path(raw_path).expanduser().resolve()


def load_metrics(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument(
        "--allow-mixed-code-and-config",
        action="store_true",
        help=(
            "Permit an explicitly requested legacy run to be combined with "
            "new repetitions; the mismatch is recorded in the report."
        ),
    )
    args = parser.parse_args()

    runs: list[tuple[str, Path]] = args.run
    if len(runs) != 3:
        raise RuntimeError(f"exactly three runs are required, got {len(runs)}")
    if len({root for _label, root in runs}) != 3:
        raise RuntimeError("the three run roots must be distinct")
    for label, root in runs:
        if not (root / "COMPLETE").is_file():
            raise RuntimeError(f"run is incomplete: {label}={root}")
        validation_path = root / "full_reconstruction_validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("passed") is not True:
            raise RuntimeError(f"full-f1 validation failed: {validation_path}")
    mixed_code_and_config = False
    try:
        assert_identical_file(runs, "CODE_AND_INPUT_SHA256.txt")
        assert_identical_file(runs, "RUN_CONFIGURATION.txt")
    except RuntimeError:
        if not args.allow_mixed_code_and_config:
            raise
        mixed_code_and_config = True

    per_run: dict[str, dict[str, dict[str, float]]] = {}
    for label, root in runs:
        per_run[label] = {}
        for method, _method_label in METHODS:
            result_path = (
                root
                / method
                / "reconstruction_7to1"
                / "evaluation_results.json"
            )
            if not result_path.is_file():
                raise RuntimeError(f"missing result: {result_path}")
            per_run[label][method] = load_metrics(result_path)

    aggregate: dict[str, dict[str, dict[str, object]]] = {}
    for method, _method_label in METHODS:
        aggregate[method] = {}
        for metric, _metric_label, _digits in METRICS:
            values = [per_run[label][method][metric] for label, _root in runs]
            aggregate[method][metric] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "values": values,
            }

    payload = {
        "schema": "super_grasp5_reconstruction_full_f1_three_run_average_v1",
        "protocol": "reconstruction_7to1",
        "observation_schedule": "stride1_full_0_1439",
        "run_count": 3,
        "standard_deviation": "sample standard deviation (ddof=1)",
        "mixed_code_and_config": mixed_code_and_config,
        "comparability_note": (
            "One previously completed full-f1 run is combined with two new "
            "runs at the user's explicit request; code/config hashes differ."
            if mixed_code_and_config
            else "All three code/input hashes and configurations are identical."
        ),
        "runs": [{"label": label, "root": str(root)} for label, root in runs],
        "per_run": per_run,
        "aggregate": aggregate,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    headers = [label for _metric, label, _digits in METRICS]
    lines = [
        "# grasp5 Reconstruction完整f1三次评估",
        "",
        "完整f1覆盖`0..1439`；表中为三次运行的算术平均 ± 样本标准差。",
        "",
        "| Method | " + " | ".join(headers) + " |",
        "|---|" + "---:|" * len(headers),
    ]
    for method, method_label in METHODS:
        cells = []
        for metric, _metric_label, digits in METRICS:
            stats = aggregate[method][metric]
            cells.append(
                f"{stats['mean']:.{digits}f} ± "
                f"{stats['sample_std']:.{digits}f}"
            )
        lines.append(f"| {method_label} | " + " | ".join(cells) + " |")
    lines.append("")
    if mixed_code_and_config:
        lines.extend(
            [
                "按用户指定，本表由一个旧完整f1正式结果和两个当前代码新结果组成。",
                "三者均通过完整f1资产、GT哈希、轨迹计划和渲染分区检查，但旧结果与",
                "当前代码/完整配置哈希不同，因此这是指定口径的合并平均，不宣称为",
                "完全相同代码的三次重复性实验。",
            ]
        )
    else:
        lines.extend(
            [
                "三次运行已通过相同代码/输入哈希、相同配置、完整f1资产、GT哈希、",
                "轨迹计划和渲染分区检查。",
            ]
        )
    lines.extend(["逐次数据见同目录`average_results.json`。", ""])
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
