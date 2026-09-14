#!/usr/bin/env python3
"""Validate and aggregate EG-Soft over three SUPER datasets and three seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


DATASETS = ("grasp5", "grasp3", "grasp1")
REPEATS = ("repeat_01", "repeat_02", "repeat_03")
PROTOCOL = "joint_reconstruction_7to1_future_80to20"
UPSTREAM_COMMIT = "c97ec671f97af25985e0af8844c0aac8d8119b97"
PARTITIONS = (
    ("reconstruction_7to1_within_train80", "Reconstruction 7:1"),
    ("future_80to20", "Future 80:20"),
)
FIELDS = (
    ("3d_mean_mm", "point_tracking", "3d_error_mm", "mean"),
    ("3d_rmse_mm", "point_tracking", "3d_error_mm", "rmse"),
    ("2d_mean_px", "point_tracking", "2d_error_px", "mean"),
    ("2d_rmse_px", "point_tracking", "2d_error_px", "rmse"),
    ("psnr_db", "rendering", "psnr_db", None),
    ("ssim", "rendering", "ssim", None),
    ("lpips_alex", "rendering", "lpips_alex", None),
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(run: Path) -> None:
    for line in (run / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        path = Path(filename)
        if not path.is_absolute():
            path = run / path
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"SHA256 mismatch: {path}")


def metric(report: dict, partition: str, field):
    _, first, second, third = field
    value = report["partitions"][partition][first][second]
    value = value if third is None else value[third]
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def read_run(run: Path, dataset: str, repeat_index: int) -> dict:
    status = (run / "status.txt").read_text()
    for item in (
        "status=complete",
        f"dataset={dataset}",
        f"repeat=repeat_0{repeat_index + 1}",
        f"seed={repeat_index}",
        "method=embodied_gaussians_super_paper_soft_v1",
        f"protocol={PROTOCOL}",
    ):
        if item not in status:
            raise ValueError(f"Run status mismatch: {run}: missing {item}")
    verify_manifest(run)
    audit = load(run / "protocol_audit.json")
    rollout = load(run / "artifacts/rollout_metadata.json")
    capture = load(run / "capture/metadata.json")
    report = load(run / "capture/evaluation_results.json")
    if not audit.get("passed") or audit.get("dataset_key") != dataset:
        raise ValueError(f"Protocol audit mismatch: {run}")
    if audit.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError(f"Upstream commit mismatch: {run}")
    if audit["physics"].get("dataset_specific_parameters") != {}:
        raise ValueError(f"Dataset-specific physics found: {run}")
    if rollout.get("parameters") != audit["physics"]["parameters"]:
        raise ValueError(f"Physics parameter mismatch: {run}")
    if rollout.get("future_rgb_depth_mask_or_track_observations_used") is not False:
        raise ValueError(f"Future observation leakage: {run}")
    if capture.get("protocol") != PROTOCOL or report.get("protocol") != PROTOCOL:
        raise ValueError(f"Protocol mismatch: {run}")
    baseline = capture.get("baseline", {})
    tracks = baseline.get("track_report", {})
    if baseline.get("shape_of_motion_used") is not False:
        raise ValueError(f"Unexpected Shape of Motion use: {run}")
    if int(tracks.get("query_point_count", -1)) != 10:
        raise ValueError(f"Query count mismatch: {run}")
    query_binding_valid = tracks.get("query_binding_valid", True)
    if query_binding_valid:
        if float(tracks.get("initial_query_reprojection_max_px", float("inf"))) > 1.0e-3:
            raise ValueError(f"Query reprojection mismatch: {run}")
    else:
        if tracks.get("query_binding_failure_reason") != "no_query_time_visible_model_gaussian":
            raise ValueError(f"Unrecognized query binding failure: {run}")
        if tracks.get("initial_query_reprojection_max_px") is not None:
            raise ValueError(f"Invalid failed-query reprojection value: {run}")
    integrity = report.get("integrity", {})
    for name in (
        "ground_truth_hash_matches_capture",
        "complete_track_schedule",
        "scored_observations_withheld",
        "render_partition_exact",
    ):
        if integrity.get(name) is not True:
            raise ValueError(f"Integrity check {name} failed: {run}")
    return report


def summarize(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(finite) != len(values):
        return {"count": len(finite), "mean": None, "population_std": None, "sample_std": None, "values": values}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "population_std": statistics.pstdev(finite),
        "sample_std": statistics.stdev(finite),
        "values": values,
    }


def formatted(item: dict, digits: int) -> str:
    if item["mean"] is None:
        return "N/A"
    return f"{item['mean']:.{digits}f} ± {item['population_std']:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.expanduser().resolve()
    output = campaign / "summary"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    raw, aggregate, rows = {}, {}, []
    for dataset in DATASETS:
        reports = []
        raw[dataset] = {}
        for repeat_index, repeat in enumerate(REPEATS):
            run = campaign / dataset / repeat
            reports.append(read_run(run, dataset, repeat_index))
            raw[dataset][repeat] = str(
                Path(dataset) / repeat / "capture/evaluation_results.json"
            )
        aggregate[dataset] = {}
        for partition, label in PARTITIONS:
            item = {
                field[0]: summarize([metric(report, partition, field) for report in reports])
                for field in FIELDS
            }
            aggregate[dataset][partition] = item
            rows.append({"dataset": dataset, "partition": label, **{name: values["mean"] for name, values in item.items()}})
    payload = {
        "schema": "embodied_gaussians_super_three_datasets_three_repeats_v1",
        "method": "Embodied Gaussians EG-Soft paper reconstruction",
        "protocol": PROTOCOL,
        "upstream_commit": UPSTREAM_COMMIT,
        "seeds": [0, 1, 2],
        "repeat_count": 3,
        "selection": "no best-run selection",
        "dispersion": "population standard deviation in table; sample standard deviation retained in JSON",
        "shape_of_motion_used": False,
        "raw_reports": raw,
        "aggregate": aggregate,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Embodied Gaussians EG-Soft SUPER: three datasets × three repeats",
        "",
        "Seeds 0/1/2; arithmetic mean ± population standard deviation; no best-run selection.",
        "",
        "| Dataset | Partition | 3D mean / RMSE (mm) | 2D mean / RMSE (px) | PSNR (dB) | SSIM | LPIPS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for partition, label in PARTITIONS:
            item = aggregate[dataset][partition]
            lines.append(
                f"| {dataset} | {label} | {formatted(item['3d_mean_mm'], 3)} / {formatted(item['3d_rmse_mm'], 3)} | "
                f"{formatted(item['2d_mean_px'], 3)} / {formatted(item['2d_rmse_px'], 3)} | "
                f"{formatted(item['psnr_db'], 3)} | {formatted(item['ssim'], 4)} | {formatted(item['lpips_alex'], 4)} |"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    for path in sorted(output.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
