#!/usr/bin/env python3
"""Detailed diagnostics for the formal dynamic-covariance q/qd three-way run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


PROTOCOLS = ("reconstruction_7to1", "future_80to20")
METHODS = (
    "pure_pbd",
    "pbd_visual_residual",
    "pbd_visual_residual_online_stiffness",
)


def numeric_stats(values) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def load_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def status_of(event: dict) -> str:
    for section in ("material", "prediction", "image"):
        value = event.get(section, {}).get("status")
        if value:
            return str(value)
    return "unknown"


def diagnostic_summary(directory: Path) -> dict | None:
    diagnostic_dir = directory / "diagnostics"
    metadata_path = diagnostic_dir / "metadata.json"
    events = load_events(diagnostic_dir / "events.jsonl")
    if not metadata_path.is_file() or not events:
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    updates = [event for event in events if event.get("event") == "visual_update"]
    cross_frame = [
        event
        for event in events
        if event.get("event") == "visual_cross_frame_validation"
    ]
    predictions = [
        event
        for event in events
        if event.get("event") == "visual_open_loop_prediction"
    ]
    stiffness = [
        event
        for event in events
        if event.get("event") == "stiffness_validation"
    ]
    trajectories = [
        event
        for event in events
        if event.get("event") == "trajectory_observation"
    ]
    physical_sections = [
        event.get("physical", {})
        for event in events
        if isinstance(event.get("physical"), dict)
        and event.get("physical")
    ]
    selected_gains = Counter(
        str(event.get("image", {}).get("selected_gain"))
        for event in cross_frame
        if event.get("image", {}).get("accepted")
    )
    candidate_counts = Counter(
        str(event.get("image", {}).get("candidate_count"))
        for event in cross_frame
    )
    stiffness_statuses = Counter(status_of(event) for event in stiffness)
    committed_variants = Counter(
        str(event.get("material", {}).get("candidate_variant", "unknown"))
        for event in stiffness
        if status_of(event) == "committed"
    )
    final_material = next(
        (
            event.get("material", {})
            for event in reversed(events)
            if event.get("material", {}).get("distance_median") is not None
        ),
        {},
    )
    return {
        "metadata": {
            key: metadata.get(key)
            for key in (
                "experiment_mode",
                "visual_residual_gain_profile",
                "visual_residual_velocity_correction_gain",
                "visual_residual_maximum_velocity_correction_m_s",
                "visual_residual_deformation_covariance_enabled",
                "online_stiffness_update",
                "stiffness_candidate_profile",
                "stiffness_admission_mode",
                "stiffness_admission_horizons",
                "stiffness_strain_signal_weight",
            )
        },
        "event_counts": dict(Counter(event.get("event", "") for event in events)),
        "visual_updates": {
            "count": len(updates),
            "same_image_accept_count": sum(
                bool(event.get("image", {}).get("accepted")) for event in updates
            ),
            "same_image_acceptance_rate": (
                sum(bool(event.get("image", {}).get("accepted")) for event in updates)
                / len(updates)
                if updates
                else 0.0
            ),
            "residual_rms_mm": numeric_stats(
                1000.0 * float(event.get("image", {}).get("residual_rms_m", 0.0))
                for event in updates
            ),
            "loss_reduction_fraction": numeric_stats(
                event.get("image", {}).get("loss_reduction_fraction", 0.0)
                for event in updates
            ),
            "velocity_correction_maximum_mm_s": numeric_stats(
                1000.0
                * float(
                    event.get("image", {}).get(
                        "velocity_correction_maximum_m_s", 0.0
                    )
                )
                for event in updates
            ),
            "velocity_correction_rms_mm_s": numeric_stats(
                1000.0
                * float(
                    event.get("image", {}).get(
                        "velocity_correction_rms_m_s", 0.0
                    )
                )
                for event in updates
            ),
            "velocity_updated_particles": numeric_stats(
                event.get("image", {}).get("velocity_updated_particles", 0)
                for event in updates
            ),
        },
        "cross_frame_validation": {
            "count": len(cross_frame),
            "accept_count": sum(
                bool(event.get("image", {}).get("accepted"))
                for event in cross_frame
            ),
            "acceptance_rate": (
                sum(
                    bool(event.get("image", {}).get("accepted"))
                    for event in cross_frame
                )
                / len(cross_frame)
                if cross_frame
                else 0.0
            ),
            "selected_gain_counts": dict(selected_gains),
            "candidate_count_distribution": dict(candidate_counts),
            "visual_improvement": numeric_stats(
                event.get("image", {}).get("visual_improvement", 0.0)
                for event in cross_frame
                if event.get("image", {}).get("accepted")
            ),
        },
        "future_boundary_prediction": {
            "count": len(predictions),
            "accept_count": sum(
                bool(event.get("image", {}).get("accepted"))
                for event in predictions
            ),
            "all_disallow_future_rgb": all(
                not bool(event.get("image", {}).get("uses_future_rgb", False))
                for event in predictions
            ),
        },
        "stiffness": {
            "validation_count": len(stiffness),
            "status_counts": dict(stiffness_statuses),
            "committed_variant_counts": dict(committed_variants),
            "final_distance_minimum": final_material.get("distance_minimum"),
            "final_distance_median": final_material.get("distance_median"),
            "final_distance_maximum": final_material.get("distance_maximum"),
            "final_shape_minimum": final_material.get("shape_minimum"),
            "final_shape_median": final_material.get("shape_median"),
            "final_shape_maximum": final_material.get("shape_maximum"),
        },
        "safety": {
            "trajectory_observation_count": len(trajectories),
            "maximum_inverted_tetrahedra": max(
                (int(section.get("inverted_tetrahedra", 0)) for section in physical_sections),
                default=0,
            ),
            "minimum_volume_ratio": min(
                (
                    float(section["minimum_volume_ratio"])
                    for section in physical_sections
                    if section.get("minimum_volume_ratio") is not None
                ),
                default=None,
            ),
            "maximum_penetration_mm": 1000.0
            * max(
                (
                    float(section.get("maximum_penetration_m", 0.0))
                    for section in physical_sections
                ),
                default=0.0,
            ),
            "maximum_anchor_error_mm": 1000.0
            * max(
                (
                    float(
                        section.get("persistent_grip_anchor_error_maximum_m", 0.0)
                    )
                    for section in physical_sections
                ),
                default=0.0,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    comparison = json.loads(
        (root / "METHOD_COMPARISON.json").read_text(encoding="utf-8")
    )
    diagnostics = {
        protocol: {
            method: diagnostic_summary(root / method / protocol)
            for method in METHODS
        }
        for protocol in PROTOCOLS
    }
    output = {
        "schema": "super_dynamic_covariance_qd_three_way_detailed_v1",
        "protocol_audit": json.loads(
            (root / "PROTOCOL_AUDIT.json").read_text(encoding="utf-8")
        ),
        "formal_metrics": comparison,
        "diagnostics": diagnostics,
    }
    (root / "DETAILED_SUMMARY.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Dynamic-covariance q/qd formal evaluation",
        "",
        "All methods use the same frozen ground truth, 1440 video frames, three physics steps per frame, and 0.5x evaluation renders.",
        "",
    ]
    for protocol in PROTOCOLS:
        lines.extend((f"## {protocol}", ""))
        values = comparison["protocols"][protocol]
        lines.extend((
            "| Method | 2D mean/RMSE px | 3D mean/RMSE mm | PSNR dB | SSIM | LPIPS |",
            "|---|---:|---:|---:|---:|---:|",
        ))
        for method in METHODS:
            item = values[method]
            lines.append(
                f"| {method} | {item['2d_mean_px']:.4f}/{item['2d_rmse_px']:.4f} "
                f"| {item['3d_mean_mm']:.4f}/{item['3d_rmse_mm']:.4f} "
                f"| {item['psnr_db']:.4f} | {item['ssim']:.5f} | {item['lpips_alex']:.5f} |"
            )
        effect = comparison["comparisons"][protocol]["online_vs_residual"]
        lines.extend((
            "",
            "Online stiffness relative to the same visual residual:",
            "",
            f"- 2D mean reduction: `{effect['2d_mean_error_reduction_pct']:+.4f}%`",
            f"- 3D mean reduction: `{effect['3d_mean_error_reduction_pct']:+.4f}%`",
            f"- PSNR change: `{effect['psnr_change_db']:+.4f} dB`",
            f"- SSIM change: `{effect['ssim_change']:+.6f}`",
            f"- LPIPS reduction: `{effect['lpips_reduction_pct']:+.4f}%`",
            "",
        ))
        for method in METHODS[1:]:
            detail = diagnostics[protocol][method]
            assert detail is not None
            visual = detail["visual_updates"]
            cross = detail["cross_frame_validation"]
            stiffness = detail["stiffness"]
            safety = detail["safety"]
            lines.extend((
                f"### {method} diagnostics",
                "",
                f"- Same-image accepted: `{visual['same_image_accept_count']}/{visual['count']}`; later-RGB accepted: `{cross['accept_count']}/{cross['count']}`.",
                f"- Selected gains: `{cross['selected_gain_counts']}`.",
                f"- Velocity max correction mean/max: `{visual['velocity_correction_maximum_mm_s']['mean']}` / `{visual['velocity_correction_maximum_mm_s']['maximum']} mm/s`.",
                f"- Stiffness validations/status: `{stiffness['validation_count']}` / `{stiffness['status_counts']}`.",
                f"- Final distance median / shape median: `{stiffness['final_distance_median']}` / `{stiffness['final_shape_median']}`.",
                f"- Safety: max inverted `{safety['maximum_inverted_tetrahedra']}`, min volume ratio `{safety['minimum_volume_ratio']}`, max penetration `{safety['maximum_penetration_mm']} mm`, max anchor error `{safety['maximum_anchor_error_mm']} mm`.",
                "",
            ))
    (root / "DETAILED_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
