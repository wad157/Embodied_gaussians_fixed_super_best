#!/usr/bin/env python3
"""Validate and aggregate three EH-SurGS repeats on all SUPER datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


PROTOCOL = "joint_reconstruction_7to1_future_80to20"
UPSTREAM_COMMIT = "73fa04e6f5c21cc1685f728eccb1332e81ce620c"
PATCH_SHA256 = "f75e5d151e7f885024e6c75ccc3a1457ebc86389d8277d433cae58175bc310d8"
DECODER = "som_query_anchored_displacement_v2"
DATASETS = ("grasp5", "grasp3", "grasp1")
REPEATS = ("repeat_01", "repeat_02", "repeat_03")
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
    manifest = run / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split("  ", 1)
        path = Path(filename)
        if not path.is_absolute():
            path = run / path
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("SHA256 mismatch: {}".format(path))


def metric(report: dict, partition: str, field) -> float:
    _, first, second, third = field
    value = report["partitions"][partition][first][second]
    result = float(value if third is None else value[third])
    if not math.isfinite(result):
        raise ValueError("Non-finite metric {} / {}".format(partition, field[0]))
    return result


def read_run(run: Path, dataset: str, repeat_index: int) -> dict:
    status = (run / "status.txt").read_text(encoding="utf-8")
    required = (
        "status=complete",
        "dataset={}".format(dataset),
        "repeat=repeat_0{}".format(repeat_index + 1),
        "seed={}".format(repeat_index),
        "method=EH-SurGS_super_v1_noninstrument_mask",
        "protocol={}".format(PROTOCOL),
    )
    if not all(item in status for item in required):
        raise ValueError("Run status mismatch: {}".format(run))
    verify_manifest(run)
    audit = load(run / "protocol_audit.json")
    metadata = load(run / "capture" / "metadata.json")
    report = load(run / "capture" / "evaluation_results.json")
    if not audit.get("passed") or audit.get("dataset_key") != dataset:
        raise ValueError("Protocol audit mismatch: {}".format(run))
    upstream = audit.get("eh_surgs", {})
    if upstream.get("commit") != UPSTREAM_COMMIT:
        raise ValueError("EH-SurGS commit mismatch: {}".format(run))
    if upstream.get("camera_compatibility_patch_sha256") != PATCH_SHA256:
        raise ValueError("Camera patch mismatch: {}".format(run))
    if metadata.get("protocol") != PROTOCOL or report.get("protocol") != PROTOCOL:
        raise ValueError("Protocol mismatch: {}".format(run))
    baseline = metadata.get("baseline", {})
    tracks = baseline.get("track_report", {})
    if baseline.get("method") != "EH-SurGS" or baseline.get("future_observations_used") is not False:
        raise ValueError("Baseline metadata mismatch: {}".format(run))
    if tracks.get("decoder_version") != DECODER:
        raise ValueError("Decoder mismatch: {}".format(run))
    if int(tracks.get("query_point_count", -1)) != 10 or int(tracks.get("query_valid_count", -1)) != 10:
        raise ValueError("Query coverage mismatch: {}".format(run))
    if float(tracks.get("initial_query_reprojection_max_px", float("inf"))) > 1.0e-3:
        raise ValueError("Query anchor reprojection mismatch: {}".format(run))
    integrity = report.get("integrity", {})
    for name in (
        "ground_truth_hash_matches_capture",
        "complete_track_schedule",
        "scored_observations_withheld",
        "render_partition_exact",
    ):
        if integrity.get(name) is not True:
            raise ValueError("Integrity check {} failed: {}".format(name, run))
    return report


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "sample_std": statistics.stdev(values),
        "values": values,
    }


def formatted(item, digits):
    return ("{:0." + str(digits) + "f} ± {:0." + str(digits) + "f}").format(
        item["mean"], item["population_std"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else campaign / "summary"
    )
    if output.exists():
        raise FileExistsError("Refusing to overwrite summary: {}".format(output))
    raw = {}
    aggregate = {}
    for dataset in DATASETS:
        reports = []
        raw[dataset] = {}
        for repeat_index, repeat in enumerate(REPEATS):
            run = campaign / dataset / repeat
            report = read_run(run, dataset, repeat_index)
            reports.append(report)
            raw[dataset][repeat] = str(run / "capture" / "evaluation_results.json")
        aggregate[dataset] = {}
        for partition, _ in PARTITIONS:
            aggregate[dataset][partition] = {
                field[0]: summarize([metric(report, partition, field) for report in reports])
                for field in FIELDS
            }
    payload = {
        "schema": "eh_surgs_super_three_datasets_three_repeats_v1",
        "method": "EH-SurGS",
        "protocol": PROTOCOL,
        "eh_surgs_commit": UPSTREAM_COMMIT,
        "camera_patch_sha256": PATCH_SHA256,
        "trajectory_decoder": DECODER,
        "seeds": [0, 1, 2],
        "repeat_count": 3,
        "selection": "no best-run selection",
        "table_dispersion": "population standard deviation, matching the existing SUPER baseline table",
        "raw_reports": raw,
        "aggregate": aggregate,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# EH-SurGS SUPER: three datasets × three repeats",
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
                "| {} | {} | {} / {} | {} / {} | {} | {} | {} |".format(
                    dataset,
                    label,
                    formatted(item["3d_mean_mm"], 3),
                    formatted(item["3d_rmse_mm"], 3),
                    formatted(item["2d_mean_px"], 3),
                    formatted(item["2d_rmse_px"], 3),
                    formatted(item["psnr_db"], 3),
                    formatted(item["ssim"], 4),
                    formatted(item["lpips_alex"], 4),
                )
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in sorted(output.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
