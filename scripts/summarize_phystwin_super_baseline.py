#!/usr/bin/env python3
"""Aggregate selected PhysTwin SUPER runs and optionally compare methods."""

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
    parser.add_argument(
        "--repeats",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="One-based repeat numbers to aggregate (default: 1 2 3).",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Do not require or include the latest-method comparison.",
    )
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


def portable_path(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main():
    args = parse_args()
    campaign = args.campaign.resolve()
    latest_campaign = args.latest_campaign.resolve()
    output = args.output.resolve()
    repeats = tuple(args.repeats)
    if not repeats or any(index < 1 for index in repeats) or len(set(repeats)) != len(repeats):
        raise ValueError("--repeats must contain unique positive integers")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": (
            "phystwin_super_repeated_evaluation_v1"
            if args.baseline_only
            else "phystwin_super_vs_latest_repeated_evaluation_v1"
        ),
        "protocol": PROTOCOL,
        "baseline": "PhysTwin",
        "repeat_ids": ["repeat_{:02d}".format(index) for index in repeats],
        "aggregation": "arithmetic mean and population standard deviation",
        "datasets": {},
    }
    if not args.baseline_only:
        summary["latest_method"] = LATEST_METHOD
    for dataset in ("grasp5", "grasp3", "grasp1"):
        baseline_paths = [
            campaign / dataset / "repeat_{:02d}".format(index) / "capture/evaluation_results.json"
            for index in repeats
        ]
        latest_paths = [] if args.baseline_only else [
            latest_campaign / dataset / "repeat_{:02d}".format(index) / LATEST_METHOD / PROTOCOL / "evaluation_results.json"
            for index in repeats
        ]
        for path in baseline_paths + latest_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        baseline = aggregate([load_report(path) for path in baseline_paths])
        dataset_summary = {
            "baseline": baseline,
            "baseline_reports": [portable_path(path) for path in baseline_paths],
        }
        if not args.baseline_only:
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
            dataset_summary.update({
                "latest": latest,
                "comparison": comparison,
                "latest_reports": [portable_path(path) for path in latest_paths],
            })
        summary["datasets"][dataset] = dataset_summary
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        (
            "# PhysTwin SUPER: selected-repeat aggregate"
            if args.baseline_only
            else "# PhysTwin vs latest SUPER method"
        ),
        "",
        "Repeats: {}. Values are arithmetic mean ± population standard deviation; no best-run selection.".format(
            ", ".join(summary["repeat_ids"])
        ),
        "",
        "| Dataset | Partition | Method | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("grasp5", "grasp3", "grasp1"):
        for partition in PARTITIONS:
            methods = [("baseline", "PhysTwin")]
            if not args.baseline_only:
                methods.append(("latest", "Latest"))
            for method_key, label in methods:
                item = summary["datasets"][dataset][method_key][partition]
                formatted = {}
                for name, digits in ((name, 3) for name in ("3d_mean_mm", "3d_rmse_mm", "2d_mean_px", "2d_rmse_px", "psnr_db")):
                    formatted[name] = ("{:0." + str(digits) + "f} ± {:0." + str(digits) + "f}").format(
                        item[name]["mean"], item[name]["std_population"]
                    )
                for name in ("ssim", "lpips_alex"):
                    formatted[name] = "{:.4f} ± {:.4f}".format(
                        item[name]["mean"], item[name]["std_population"]
                    )
                lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    partition,
                    label,
                    "{} / {}".format(formatted["3d_mean_mm"], formatted["3d_rmse_mm"]),
                    "{} / {}".format(formatted["2d_mean_px"], formatted["2d_rmse_px"]),
                    formatted["psnr_db"],
                    formatted["ssim"],
                    formatted["lpips_alex"],
                ))
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
