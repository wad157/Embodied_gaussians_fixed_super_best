#!/usr/bin/env python3
"""Validate and aggregate three TRACE repeats on all formal SUPER datasets."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DATASETS = ("grasp5", "grasp3", "grasp1")
REPEATS = ("repeat_01", "repeat_02", "repeat_03")
PARTITIONS = (
    "reconstruction_7to1_within_train80",
    "future_80to20",
)
FIELDS = (
    ("3d_mean_mm", "point_tracking", "3d_error_mm", "mean", "lower"),
    ("3d_rmse_mm", "point_tracking", "3d_error_mm", "rmse", "lower"),
    ("2d_mean_px", "point_tracking", "2d_error_px", "mean", "lower"),
    ("2d_rmse_px", "point_tracking", "2d_error_px", "rmse", "lower"),
    ("psnr_db", "rendering", "psnr_db", None, "higher"),
    ("ssim", "rendering", "ssim", None, "higher"),
    ("lpips_alex", "rendering", "lpips_alex", None, "lower"),
)
PROTOCOL = "joint_reconstruction_7to1_future_80to20"
EXPECTED_COMMIT = "a4597585bc0e56c56922abe75be9198eb119c95a"
EXPECTED_ADAPTER = "trace_super_v1_rgb_only_full_k"
EXPECTED_DECODER = "query_anchored_gaussian_displacement_v1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(report: dict, partition: str, field) -> float:
    _, first, second, third, _ = field
    value = report["partitions"][partition][first][second]
    return float(value if third is None else value[third])


def validate_run(root: Path, dataset: str, repeat_index: int) -> dict:
    status = (root / "status.txt").read_text(encoding="utf-8")
    if "status=complete" not in status or "seed={}".format(repeat_index) not in status:
        raise ValueError("run status/seed mismatch: {}".format(root))
    audit = load(root / "protocol_audit.json")
    metadata = load(root / "capture" / "metadata.json")
    report = load(root / "capture" / "evaluation_results.json")
    if not audit.get("passed") or audit.get("dataset_key") != dataset:
        raise ValueError("protocol audit mismatch: {}".format(root))
    if audit.get("adapter_version") != EXPECTED_ADAPTER:
        raise ValueError("adapter mismatch: {}".format(root))
    trace = audit.get("trace", {})
    if trace.get("commit") != EXPECTED_COMMIT or trace.get("algorithm_changes") != []:
        raise ValueError("TRACE source provenance mismatch: {}".format(root))
    baseline = metadata.get("baseline", {})
    tracks = baseline.get("track_report", {})
    if metadata.get("protocol") != PROTOCOL or report.get("protocol") != PROTOCOL:
        raise ValueError("protocol mismatch: {}".format(root))
    if baseline.get("trace_commit") != EXPECTED_COMMIT:
        raise ValueError("TRACE commit mismatch: {}".format(root))
    if baseline.get("external_psm_control") is not False:
        raise ValueError("run does not certify no-PSM operation: {}".format(root))
    if baseline.get("shape_of_motion_used") is not False:
        raise ValueError("unexpected Shape-of-Motion dependency: {}".format(root))
    if tracks.get("decoder_version") != EXPECTED_DECODER:
        raise ValueError("trajectory decoder mismatch: {}".format(root))
    if int(tracks.get("query_point_count", -1)) != 10 or int(
        tracks.get("query_valid_count", -1)
    ) != 10:
        raise ValueError("query coverage is not 10/10: {}".format(root))
    if float(tracks.get("initial_query_reprojection_max_px", float("inf"))) > 1.0e-3:
        raise ValueError("query anchoring failed: {}".format(root))
    integrity = report.get("integrity", {})
    for key in (
        "ground_truth_hash_matches_capture",
        "complete_track_schedule",
        "scored_observations_withheld",
        "render_partition_exact",
    ):
        if integrity.get(key) is not True:
            raise ValueError("evaluation integrity gate {} failed: {}".format(key, root))
    return report


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "values": values,
    }


def format_value(item: dict, digits: int) -> str:
    pattern = "{:0." + str(digits) + "f} ± {:0." + str(digits) + "f}"
    return pattern.format(item["mean"], item["std_population"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.expanduser().resolve()
    raw, aggregate = {}, {}
    for dataset in DATASETS:
        raw[dataset] = {}
        reports = []
        for repeat_index, repeat in enumerate(REPEATS):
            run_root = campaign / dataset / repeat
            report = validate_run(run_root, dataset, repeat_index)
            reports.append(report)
            raw[dataset][repeat] = str(
                run_root / "capture" / "evaluation_results.json"
            )
        aggregate[dataset] = {}
        for partition in PARTITIONS:
            aggregate[dataset][partition] = {}
            for field in FIELDS:
                values = [metric(report, partition, field) for report in reports]
                aggregate[dataset][partition][field[0]] = {
                    **summarize(values),
                    "direction": field[-1],
                }
    payload = {
        "schema": "trace_super_three_datasets_three_repeats_v1",
        "method": "official TRACE core + full-K RGB-only SUPER adapter",
        "trace_commit": EXPECTED_COMMIT,
        "adapter_version": EXPECTED_ADAPTER,
        "trajectory_decoder_version": EXPECTED_DECODER,
        "protocol": {
            "training": "legal first-80% non-holdout stereo-left RGB and complete calibrated camera only",
            "forbidden": "PSM, depth, masks, GT trajectories, held-out/future RGB",
            "reconstruction": "stride=8 phase=0 held-out interpolation",
            "future": "native TRACE dynamics; no future observations/control",
            "evaluation": "unchanged SUPER 10-point and masked rendering scorer",
            "alignment": "none",
            "seeds": [0, 1, 2],
            "aggregation": "arithmetic mean and population standard deviation; no best-run selection",
        },
        "raw_reports": raw,
        "aggregate": aggregate,
    }
    summary = campaign / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    json_path = summary / "summary.json"
    markdown_path = summary / "summary.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("refusing to overwrite TRACE SUPER summary")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partition_labels = {
        "reconstruction_7to1_within_train80": "前 80% 内 7:1 重建",
        "future_80to20": "后 20% Future 外推",
    }
    lines = [
        "# TRACE SUPER 基线：三数据集 × 三种子",
        "",
        "官方 TRACE `{}`；训练只使用 RGB 与相机标定，不使用 PSM、深度、mask、对齐、Shape-of-Motion 或最优运行筛选。".format(
            EXPECTED_COMMIT
        ),
        "",
        "| 数据集 | 分区 | n | 3D 均值 (mm)↓ | 2D 均值 (px)↓ | PSNR (dB)↑ | SSIM↑ | LPIPS↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for partition in PARTITIONS:
            item = aggregate[dataset][partition]
            values = [
                format_value(item["3d_mean_mm"], 3),
                format_value(item["2d_mean_px"], 3),
                format_value(item["psnr_db"], 3),
                format_value(item["ssim"], 4),
                format_value(item["lpips_alex"], 4),
            ]
            lines.append(
                "| {} | {} | 3 | {} |".format(
                    dataset, partition_labels[partition], " | ".join(values)
                )
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote {}".format(markdown_path), flush=True)


if __name__ == "__main__":
    main()
