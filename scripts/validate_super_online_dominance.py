#!/usr/bin/env python3
"""Fail unless residual+online is materially better than both formal baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("pure_pbd", "residual_only", "residual_online_robust")
PROTOCOLS = ("reconstruction_7to1", "future_80to20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--initializations",
        default="extreme_soft,extreme_hard",
        help="Comma-separated recovery-pressure initializations to enforce.",
    )
    parser.add_argument("--minimum-tracking-relative-gain", type=float, default=0.10)
    parser.add_argument("--minimum-psnr-gain-db", type=float, default=0.50)
    parser.add_argument("--minimum-ssim-gain", type=float, default=0.005)
    parser.add_argument("--minimum-lpips-relative-gain", type=float, default=0.05)
    return parser.parse_args()


def metrics(path: Path) -> dict[str, float]:
    result = json.loads(path.read_text(encoding="utf-8"))
    integrity = result.get("integrity", {})
    if integrity and not all(
        bool(value) for key, value in integrity.items() if key != "query_frame_excluded"
    ):
        raise ValueError(f"Metric integrity failed: {path}")
    return {
        "2d_error_px": float(result["point_tracking"]["2d_error_px"]["mean"]),
        "3d_error_mm": float(result["point_tracking"]["3d_error_mm"]["mean"]),
        "psnr_db": float(result["rendering"]["psnr_db"]),
        "ssim": float(result["rendering"]["ssim"]),
        "lpips": float(result["rendering"]["lpips_alex"]),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    initializations = tuple(
        value.strip() for value in args.initializations.split(",") if value.strip()
    )
    cases = []
    all_passed = True
    for initialization in initializations:
        for protocol in PROTOCOLS:
            by_method = {
                method: metrics(
                    root / initialization / method / protocol / "evaluation_results.json"
                )
                for method in METHODS
            }
            online = by_method["residual_online_robust"]
            baselines = (by_method["pure_pbd"], by_method["residual_only"])
            best = {
                "2d_error_px": min(value["2d_error_px"] for value in baselines),
                "3d_error_mm": min(value["3d_error_mm"] for value in baselines),
                "psnr_db": max(value["psnr_db"] for value in baselines),
                "ssim": max(value["ssim"] for value in baselines),
                "lpips": min(value["lpips"] for value in baselines),
            }
            gains = {
                "2d_error_relative": (best["2d_error_px"] - online["2d_error_px"])
                / best["2d_error_px"],
                "3d_error_relative": (best["3d_error_mm"] - online["3d_error_mm"])
                / best["3d_error_mm"],
                "psnr_db": online["psnr_db"] - best["psnr_db"],
                "ssim": online["ssim"] - best["ssim"],
                "lpips_relative": (best["lpips"] - online["lpips"])
                / best["lpips"],
            }
            checks = {
                "2d_tracking": gains["2d_error_relative"]
                >= args.minimum_tracking_relative_gain,
                "3d_tracking": gains["3d_error_relative"]
                >= args.minimum_tracking_relative_gain,
                "psnr": gains["psnr_db"] >= args.minimum_psnr_gain_db,
                "ssim": gains["ssim"] >= args.minimum_ssim_gain,
                "lpips": gains["lpips_relative"]
                >= args.minimum_lpips_relative_gain,
            }
            passed = all(checks.values())
            all_passed = all_passed and passed
            cases.append(
                {
                    "initialization": initialization,
                    "protocol": protocol,
                    "passed": passed,
                    "checks": checks,
                    "metrics": by_method,
                    "best_baseline": best,
                    "online_gain_over_best_baseline": gains,
                }
            )

    report = {
        "schema": "super_online_dominance_gate_v1",
        "passed": all_passed,
        "methods": METHODS,
        "thresholds": {
            "tracking_relative_gain": args.minimum_tracking_relative_gain,
            "psnr_gain_db": args.minimum_psnr_gain_db,
            "ssim_gain": args.minimum_ssim_gain,
            "lpips_relative_gain": args.minimum_lpips_relative_gain,
        },
        "cases": cases,
    }
    (root / "ONLINE_DOMINANCE.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SUPER residual + online stiffness dominance gate",
        "",
        "| Initialization | Protocol | 2D gain | 3D gain | PSNR gain | SSIM gain | LPIPS gain | Pass |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for case in cases:
        gain = case["online_gain_over_best_baseline"]
        lines.append(
            "| {initialization} | {protocol} | {d2:.1%} | {d3:.1%} | "
            "{psnr:+.3f} dB | {ssim:+.4f} | {lpips:.1%} | {passed} |".format(
                initialization=case["initialization"],
                protocol=case["protocol"],
                d2=gain["2d_error_relative"],
                d3=gain["3d_error_relative"],
                psnr=gain["psnr_db"],
                ssim=gain["ssim"],
                lpips=gain["lpips_relative"],
                passed="PASS" if case["passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "The online method is compared with the better of Pure PBD and residual-only for every metric.",
        ]
    )
    (root / "ONLINE_DOMINANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
