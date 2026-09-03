#!/usr/bin/env python3
"""Summarize the three SUPER PBD ablation groups under identical protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "outputs/super_tissue_evaluation_formal_20260821"
PROTOCOLS = ("reconstruction_7to1", "future_80to20")
LEGACY_METHODS = {
    "pure_pbd": "Pure PBD",
    "pbd_visual_residual": "PBD + visual residual",
    "pbd_visual_residual_online_stiffness": "PBD + visual residual + online stiffness",
}
TRACK_METHODS = {
    "pure_pbd": "Pure PBD",
    "pbd_cotracker_depth": "PBD + CoTracker/depth trajectory correction",
    "pbd_cotracker_depth_online_stiffness": (
        "PBD + CoTracker/depth trajectory correction + online stiffness"
    ),
}
ALLTRACKER_METHODS = {
    "pure_pbd": "Pure PBD",
    "pbd_alltracker_depth": "PBD + AllTracker/depth trajectory correction",
    "pbd_alltracker_depth_online_stiffness": (
        "PBD + AllTracker/depth trajectory correction + online stiffness"
    ),
}
ALLTRACKER_RGB_METHODS = {
    "pure_pbd": "Pure PBD",
    "pbd_alltracker_depth_rgb_residual": (
        "PBD + AllTracker/depth trajectory + RGB residual"
    ),
    "pbd_alltracker_depth_rgb_residual_online_stiffness": (
        "PBD + AllTracker/depth trajectory + RGB residual + online stiffness"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def result_directory(root: Path, method: str, protocol: str) -> Path:
    # New formal runs use one canonical method/protocol matrix.  The fallback
    # preserves readability of the frozen 2026-08-21 artifact, whose online
    # method predates the method-level directory.
    canonical = root / method / protocol
    if (canonical / "evaluation_results.json").is_file():
        return canonical
    if method == "pbd_visual_residual_online_stiffness":
        return root / protocol
    return canonical


def select_methods(root: Path) -> dict[str, str]:
    if all((root / method).is_dir() for method in ALLTRACKER_RGB_METHODS):
        return ALLTRACKER_RGB_METHODS
    if all((root / method).is_dir() for method in ALLTRACKER_METHODS):
        return ALLTRACKER_METHODS
    if all((root / method).is_dir() for method in TRACK_METHODS):
        return TRACK_METHODS
    return LEGACY_METHODS


def read_diagnostics(root: Path, method: str, protocol: str) -> dict:
    path = root / method / protocol / "diagnostics/events.jsonl"
    if not path.is_file():
        return {
            "available": False,
            "flow_depth_events": 0,
            "flow_depth_updates": 0,
            "flow_depth_accepted": 0,
            "flow_depth_rejected": 0,
            "flow_depth_withheld": 0,
            "flow_depth_update_frames": [],
            "stiffness_candidates": 0,
            "stiffness_commits": 0,
            "stiffness_rejections": 0,
            "stiffness_commit_frames": [],
        }
    result = {
        "available": True,
        "flow_depth_events": 0,
        "flow_depth_updates": 0,
        "flow_depth_accepted": 0,
        "flow_depth_rejected": 0,
        "flow_depth_withheld": 0,
        "flow_depth_update_frames": [],
        "stiffness_candidates": 0,
        "stiffness_commits": 0,
        "stiffness_rejections": 0,
        "stiffness_commit_frames": [],
    }
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            event_type = event.get("event")
            frame_index = int(event.get("frame_index", -1))
            if event_type == "flow_depth_state_update":
                result["flow_depth_events"] += 1
                image = event.get("image", {})
                status = str(image.get("status", ""))
                if status == "withheld_by_causal_protocol":
                    result["flow_depth_withheld"] += 1
                else:
                    result["flow_depth_updates"] += 1
                    accepted = bool(image.get("accepted", False))
                    key = (
                        "flow_depth_accepted"
                        if accepted
                        else "flow_depth_rejected"
                    )
                    result[key] += 1
                    result["flow_depth_update_frames"].append(frame_index)
            material = event.get("material", {})
            status = str(material.get("status", ""))
            validation_status = str(material.get("validation_status", ""))
            if status == "candidate":
                result["stiffness_candidates"] += 1
            # Observation-rate Adam commits immediately inside the trajectory
            # update transaction; unlike legacy H1/H3/H5 admission it has no
            # separate stiffness_validation event. Count that single owning
            # flow event so the formal comparison reports its true cadence.
            if (
                event_type == "flow_depth_state_update"
                and status == "committed"
            ):
                result["stiffness_candidates"] += 1
                result["stiffness_commits"] += 1
                result["stiffness_commit_frames"].append(frame_index)
            if event_type != "stiffness_validation":
                continue
            if validation_status == "committed":
                result["stiffness_commits"] += 1
                result["stiffness_commit_frames"].append(frame_index)
            elif validation_status == "rejected":
                result["stiffness_rejections"] += 1
    return result


def read_result(root: Path, method: str, protocol: str) -> dict:
    path = result_directory(root, method, protocol) / "evaluation_results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(report: dict) -> dict:
    tracking = report["point_tracking"]
    rendering = report["rendering"]
    return {
        "scored_track_frames": int(tracking["scored_frame_count"]),
        "scored_track_samples": int(tracking["2d_error_px"]["count"]),
        "2d_mean_px": float(tracking["2d_error_px"]["mean"]),
        "2d_rmse_px": float(tracking["2d_error_px"]["rmse"]),
        "tap_delta_avg": float(tracking["2d_tap_position_accuracy"]["delta_avg"]),
        "3d_mean_mm": float(tracking["3d_error_mm"]["mean"]),
        "3d_rmse_mm": float(tracking["3d_error_mm"]["rmse"]),
        "3d_coverage": float(tracking["3d_coverage"]["fraction"]),
        "render_frames": int(rendering["frame_count"]),
        "psnr_db": float(rendering["psnr_db"]),
        "ssim": float(rendering["ssim"]),
        "lpips_alex": float(rendering["lpips_alex"]),
        "video_frames_per_s": float(report["runtime"]["video_frames_per_s"]),
    }


def reduction_percent(baseline: float, value: float) -> float:
    return 100.0 * (baseline - value) / baseline


def compare(baseline: dict, value: dict) -> dict:
    return {
        "2d_mean_error_reduction_pct": reduction_percent(
            baseline["2d_mean_px"], value["2d_mean_px"]
        ),
        "3d_mean_error_reduction_pct": reduction_percent(
            baseline["3d_mean_mm"], value["3d_mean_mm"]
        ),
        "psnr_change_db": value["psnr_db"] - baseline["psnr_db"],
        "ssim_change": value["ssim"] - baseline["ssim"],
        "lpips_reduction_pct": reduction_percent(
            baseline["lpips_alex"], value["lpips_alex"]
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    methods = select_methods(root)
    reports: dict[str, dict[str, dict]] = {}
    flattened: dict[str, dict[str, dict]] = {}
    diagnostics: dict[str, dict[str, dict]] = {}
    for protocol in PROTOCOLS:
        reports[protocol] = {}
        flattened[protocol] = {}
        diagnostics[protocol] = {}
        reference_frames = None
        reference_ground_truth = None
        for method in methods:
            report = read_result(root, method, protocol)
            if report["protocol"] != protocol:
                raise ValueError(f"Protocol mismatch for {method}/{protocol}")
            frames = report["point_tracking"]["scored_frames"]
            if reference_frames is None:
                reference_frames = frames
                reference_ground_truth = report["ground_truth"]
            elif frames != reference_frames:
                raise ValueError(f"Scored frame mismatch for {method}/{protocol}")
            elif report["ground_truth"] != reference_ground_truth:
                raise ValueError(f"Ground-truth mismatch for {method}/{protocol}")
            reports[protocol][method] = report
            flattened[protocol][method] = flatten(report)
            diagnostics[protocol][method] = read_diagnostics(
                root, method, protocol
            )

    comparisons = {}
    method_ids = tuple(methods)
    pure_method, visual_method, stiffness_method = method_ids
    visual_term = (
        "trajectory correction"
        if (
            methods is TRACK_METHODS
            or methods is ALLTRACKER_METHODS
            or methods is ALLTRACKER_RGB_METHODS
        )
        else "visual residual"
    )
    for protocol in PROTOCOLS:
        values = flattened[protocol]
        comparisons[protocol] = {
            "trajectory_vs_pure_pbd": compare(
                values[pure_method], values[visual_method]
            ),
            "stiffness_vs_pure_pbd": compare(
                values[pure_method], values[stiffness_method]
            ),
            "stiffness_vs_trajectory": compare(
                values[visual_method], values[stiffness_method]
            ),
        }

    output = {
        "schema": "super_tissue_three_method_comparison_v2",
        "root": str(root),
        "methods": methods,
        "protocols": flattened,
        "runtime_diagnostics": diagnostics,
        "comparisons": comparisons,
        "interpretation": {
            "error_reduction_pct": "positive means the candidate has lower error",
            "psnr_change_db": "positive means the candidate has higher PSNR",
            "ssim_change": "positive means the candidate has higher SSIM",
            "lpips_reduction_pct": "positive means the candidate has lower LPIPS",
            "warning": "These are single-run effect sizes, not statistical significance claims.",
        },
    }
    (root / "METHOD_COMPARISON.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# SUPER three-method ablation",
        "",
        "All groups use the same ground truth, split, tool controls, 1440 frames, and three physics steps per video frame.",
        "",
    ]
    for protocol in PROTOCOLS:
        lines.extend(
            [
                f"## {protocol}",
                "",
                "| Method | 2D mean / RMSE (px) | 3D mean / RMSE (mm) | PSNR (dB) | SSIM | LPIPS | FPS |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, label in methods.items():
            item = flattened[protocol][method]
            lines.append(
                f"| {label} | {item['2d_mean_px']:.3f} / {item['2d_rmse_px']:.3f} "
                f"| {item['3d_mean_mm']:.3f} / {item['3d_rmse_mm']:.3f} "
                f"| {item['psnr_db']:.3f} | {item['ssim']:.4f} | {item['lpips_alex']:.4f} "
                f"| {item['video_frames_per_s']:.3f} |"
            )
        lines.extend(
            [
                "",
                f"Online stiffness relative to {visual_term} only:",
                "",
            ]
        )
        effect = comparisons[protocol]["stiffness_vs_trajectory"]
        lines.extend(
            [
                f"- 2D mean error reduction: `{effect['2d_mean_error_reduction_pct']:+.3f}%`",
                f"- 3D mean error reduction: `{effect['3d_mean_error_reduction_pct']:+.3f}%`",
                f"- PSNR change: `{effect['psnr_change_db']:+.3f} dB`",
                f"- SSIM change: `{effect['ssim_change']:+.5f}`",
                f"- LPIPS reduction: `{effect['lpips_reduction_pct']:+.3f}%`",
                "",
            ]
        )
        stiffness_runtime = diagnostics[protocol][stiffness_method]
        visual_runtime = diagnostics[protocol][visual_method]
        lines.extend(
            [
                "Runtime corrections:",
                "",
                "- Trajectory-only flow/depth updates: "
                f"`{visual_runtime['flow_depth_accepted']}/"
                f"{visual_runtime['flow_depth_updates']}` accepted.",
                "- Trajectory-only causally withheld pairs: "
                f"`{visual_runtime['flow_depth_withheld']}`.",
                "- Full-method flow/depth updates: "
                f"`{stiffness_runtime['flow_depth_accepted']}/"
                f"{stiffness_runtime['flow_depth_updates']}` accepted.",
                "- Full-method causally withheld pairs: "
                f"`{stiffness_runtime['flow_depth_withheld']}`.",
                "- Full-method stiffness commits/rejections: "
                f"`{stiffness_runtime['stiffness_commits']}/"
                f"{stiffness_runtime['stiffness_rejections']}`.",
                "- Stiffness commit frames: `"
                + ",".join(
                    str(value)
                    for value in stiffness_runtime["stiffness_commit_frames"]
                )
                + "`.",
                "",
            ]
        )
    lines.append("Single-run effect sizes are not statistical significance claims.")
    (root / "METHOD_COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
