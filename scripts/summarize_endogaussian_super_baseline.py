#!/usr/bin/env python3
"""Aggregate EndoGaussian SUPER runs and compare with the latest method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "joint_reconstruction_7to1_future_80to20"
LATEST_METHOD = "pbd_alltracker_depth_rgb_residual_online_stiffness"
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--latest-campaign",
        type=Path,
        default=REPO_ROOT / "outputs/super_joint_h2direct_persistent_three_datasets_three_repeats_20260911_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metric(report, partition, field):
    _, first, second, third, _ = field
    value = report["partitions"][partition][first][second]
    return float(value if third is None else value[third])


def load_report(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != PROTOCOL:
        raise ValueError("Protocol mismatch in {}".format(path))
    return report


def aggregate(reports):
    result = {}
    for partition in PARTITIONS:
        result[partition] = {}
        for field in FIELDS:
            values = np.asarray([metric(report, partition, field) for report in reports], dtype=np.float64)
            result[partition][field[0]] = {
                "mean": float(values.mean()),
                "std_population": float(values.std(ddof=0)),
                "values": values.tolist(),
                "direction": field[-1],
            }
    return result


def main():
    args = parse_args()
    campaign = args.campaign.resolve()
    latest_campaign = args.latest_campaign.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "endogaussian_super_vs_latest_three_repeats_v1",
        "protocol": PROTOCOL,
        "baseline": "EndoGaussian",
        "latest_method": LATEST_METHOD,
        "datasets": {},
    }
    for dataset in ("grasp5", "grasp3", "grasp1"):
        baseline_paths = [
            campaign / dataset / "repeat_0{}".format(index) / "capture/evaluation_results.json"
            for index in (1, 2, 3)
        ]
        latest_paths = [
            latest_campaign / dataset / "repeat_0{}".format(index) / LATEST_METHOD / PROTOCOL / "evaluation_results.json"
            for index in (1, 2, 3)
        ]
        for path in baseline_paths + latest_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        baseline = aggregate([load_report(path) for path in baseline_paths])
        latest = aggregate([load_report(path) for path in latest_paths])
        comparison = {}
        for partition in PARTITIONS:
            comparison[partition] = {}
            for field in FIELDS:
                name = field[0]
                baseline_mean = baseline[partition][name]["mean"]
                latest_mean = latest[partition][name]["mean"]
                comparison[partition][name] = {
                    "baseline_minus_latest": baseline_mean - latest_mean,
                    "latest_is_better": (
                        latest_mean < baseline_mean if field[-1] == "lower" else latest_mean > baseline_mean
                    ),
                }
        summary["datasets"][dataset] = {
            "baseline": baseline,
            "latest": latest,
            "comparison": comparison,
            "baseline_reports": [str(path) for path in baseline_paths],
            "latest_reports": [str(path) for path in latest_paths],
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# EndoGaussian vs latest SUPER method",
        "",
        "Values are mean ± population standard deviation across three actual runs.",
        "",
        "| Dataset | Partition | Method | 3D mean (mm) | 2D mean (px) | PSNR (dB) | SSIM | LPIPS |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("grasp5", "grasp3", "grasp1"):
        for partition in PARTITIONS:
            for method_key, label in (("baseline", "EndoGaussian"), ("latest", "Latest")):
                item = summary["datasets"][dataset][method_key][partition]
                values = []
                for name, digits in (
                    ("3d_mean_mm", 3),
                    ("2d_mean_px", 3),
                    ("psnr_db", 3),
                    ("ssim", 4),
                    ("lpips_alex", 4),
                ):
                    values.append(("{:0." + str(digits) + "f} ± {:0." + str(digits) + "f}").format(
                        item[name]["mean"], item[name]["std_population"]
                    ))
                lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    dataset, partition, label, *values
                ))
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
