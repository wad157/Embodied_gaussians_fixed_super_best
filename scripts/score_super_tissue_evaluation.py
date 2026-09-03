#!/usr/bin/env python3
"""Score SUPER 2D/3D point tracks and masked PSNR/SSIM/LPIPS renders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = (
    REPO_ROOT
    / "data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--lpips-device", default="cuda:0")
    parser.add_argument(
        "--allow-prefix-validation",
        action="store_true",
        help=(
            "Allow a strictly pre-formal future prefix for parameter validation. "
            "The default formal scorer still requires the complete frozen schedule."
        ),
    )
    return parser.parse_args()


def distribution(values: np.ndarray, scale: float = 1.0) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)] * scale
    if len(values) == 0:
        return {
            "count": 0,
            **{
                key: None
                for key in ("mean", "rmse", "median", "p90", "p95", "maximum")
            },
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def tap_position_accuracy(errors_px: np.ndarray) -> dict:
    finite = np.asarray(errors_px, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    thresholds = (1, 2, 4, 8, 16)
    if len(finite) == 0:
        return {
            **{f"delta_{threshold}px": None for threshold in thresholds},
            "delta_avg": None,
        }
    scores = {f"delta_{threshold}px": float(np.mean(finite < threshold)) for threshold in thresholds}
    scores["delta_avg"] = float(np.mean(list(scores.values())))
    return scores


def formatted(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def compute_lpips(render_directory: Path, frame_indices: list[int], device: str) -> tuple[list[dict], float]:
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError("LPIPS package is required; install the official lpips package or pass --skip-lpips") from error
    metric = lpips.LPIPS(net="alex").to(device).eval()
    records = []
    with torch.inference_mode():
        for frame_index in frame_indices:
            stem = f"{frame_index:06d}"
            prediction = cv2.imread(str(render_directory / f"{stem}-prediction.png"), cv2.IMREAD_COLOR)
            target = cv2.imread(str(render_directory / f"{stem}-target.png"), cv2.IMREAD_COLOR)
            if prediction is None or target is None:
                raise FileNotFoundError(f"Missing render pair for frame {frame_index}")
            prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            tensors = []
            for image in (prediction, target):
                tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                tensors.append((tensor.to(device) / 127.5) - 1.0)
            value = float(metric(tensors[0], tensors[1], normalize=False).item())
            records.append({"frame_index": frame_index, "lpips_alex": value})
    return records, float(np.mean([record["lpips_alex"] for record in records]))


def main() -> None:
    args = parse_args()
    capture = args.capture.resolve()
    metadata = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
    protocol = metadata["protocol"]
    if sha256(args.ground_truth) != metadata["ground_truth_sha256"]:
        raise ValueError("Scored ground truth differs from the frozen capture input")
    with np.load(args.ground_truth, allow_pickle=False) as archive:
        gt = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(capture / "predicted_tracks.npz", allow_pickle=False) as archive:
        prediction = {name: np.asarray(archive[name]) for name in archive.files}
    gt_lookup = {int(frame): slot for slot, frame in enumerate(gt["frame_indices"])}
    pred_lookup = {int(frame): slot for slot, frame in enumerate(prediction["frame_indices"])}
    common = sorted(set(gt_lookup) & set(pred_lookup))
    prefix_validation = bool(args.allow_prefix_validation)
    formal_future_start = int(
        metadata.get("future_split", {}).get(
            "ground_truth_formal_test_start", -1
        )
    )
    prediction_end = max(pred_lookup)
    if prefix_validation:
        if protocol != "future_80to20":
            raise ValueError("Prefix validation is allowed only for future_80to20")
        validation_start = int(
            metadata["future_split"]["test_start_inclusive"]
        )
        if formal_future_start < 0:
            raise ValueError("Missing frozen formal future-test boundary")
        if not validation_start < formal_future_start:
            raise ValueError(
                "Prefix validation must start before the formal future test"
            )
        if prediction_end >= formal_future_start:
            raise ValueError(
                "Prefix validation reached or crossed the formal future test"
            )
        expected_prefix_gt = sorted(
            frame for frame in gt_lookup if frame <= prediction_end
        )
        if common != expected_prefix_gt:
            raise ValueError(
                "Predicted tracks do not cover the complete validation-prefix "
                "GT schedule"
            )
    elif common != sorted(gt_lookup):
        raise ValueError("Predicted tracks do not cover the complete frozen GT schedule")
    predicted_observation_used = {
        int(frame): bool(prediction["observation_used"][slot])
        for frame, slot in pred_lookup.items()
    }
    if protocol == "future_80to20":
        future_test_start = int(
            metadata["future_split"]["test_start_inclusive"]
        )
        scored_frames = [
            frame for frame in common if frame >= future_test_start
        ]
    elif protocol == "reconstruction_7to1":
        phase = int(metadata["reconstruction_split"]["test_rule"].rsplit(" ", 1)[-1])
        scored_frames = [frame for frame in common if frame % 8 == phase]
    else:
        raise ValueError(f"Unknown protocol {protocol}")
    query_frame = int(
        prediction.get(
            "query_frame_index",
            np.asarray(metadata.get("trajectory_query_frame", 0)),
        ).item()
    )
    scored_frames = [frame for frame in scored_frames if frame != query_frame]
    if not scored_frames:
        raise ValueError("No point-tracking frames belong to the scored partition")
    if any(predicted_observation_used[frame] for frame in scored_frames):
        raise ValueError("A scored tracking frame was exposed as an observation")

    gt_slots = np.asarray([gt_lookup[frame] for frame in scored_frames], dtype=np.int64)
    pred_slots = np.asarray([pred_lookup[frame] for frame in scored_frames], dtype=np.int64)
    visible = gt["visible"][gt_slots].astype(bool)
    difference_2d = prediction["uv"][pred_slots] - gt["uv"][gt_slots]
    error_2d = np.linalg.norm(difference_2d, axis=-1)
    error_2d[~visible] = np.nan
    valid_3d = gt["valid_3d"][gt_slots].astype(bool)
    difference_3d = prediction["xyz_camera_m"][pred_slots] - gt["xyz_camera_m"][gt_slots]
    error_3d = np.linalg.norm(difference_3d, axis=-1)
    error_3d[~valid_3d] = np.nan

    point_count = error_2d.shape[1]
    track_metrics = {
        "scored_frame_count": len(scored_frames),
        "scored_frames": scored_frames,
        "2d_error_px": distribution(error_2d),
        "2d_tap_position_accuracy": tap_position_accuracy(error_2d),
        "3d_error_mm": distribution(error_3d, scale=1000.0),
        "3d_coverage": {
            "valid": int(valid_3d.sum()),
            "visible": int(visible.sum()),
            "fraction": float(valid_3d.sum() / max(1, visible.sum())),
        },
        "per_point": [
            {
                "point_id": point_id,
                "2d_error_px": distribution(error_2d[:, point_id]),
                "3d_error_mm": distribution(error_3d[:, point_id], scale=1000.0),
            }
            for point_id in range(point_count)
        ],
    }

    render_partial = json.loads((capture / "render_metrics_partial.json").read_text(encoding="utf-8"))
    render_frames = [int(record["frame_index"]) for record in render_partial["records"]]
    if len(render_frames) != len(set(render_frames)):
        raise ValueError("Render metric frames are not unique")
    if any(bool(record["observation_used"]) for record in render_partial["records"]):
        raise ValueError("A scored render frame was exposed as an observation")
    sequence_start = int(np.min(gt["frame_indices"]))
    sequence_end_exclusive = (
        prediction_end + 1
        if prefix_validation
        else int(np.max(gt["frame_indices"])) + 1
    )
    expected_render_frames = (
        [
            frame
            for frame in range(
                int(metadata["future_split"]["test_start_inclusive"]),
                sequence_end_exclusive,
            )
        ]
        if protocol == "future_80to20"
        else [
            frame
            for frame in range(sequence_start, sequence_end_exclusive)
            if frame % 8
            == int(metadata["reconstruction_split"]["test_rule"].rsplit(" ", 1)[-1])
        ]
    )
    if metadata.get("capture_renders", True) and render_frames != expected_render_frames:
        raise ValueError("Render frames do not exactly match the frozen scored partition")
    lpips_records = []
    lpips_mean = None
    if not args.skip_lpips:
        lpips_records, lpips_mean = compute_lpips(capture / "render_pairs", render_frames, args.lpips_device)
    render_metrics = {
        **render_partial["mean"],
        "lpips_alex": lpips_mean,
        "frame_count": len(render_frames),
        "per_frame_lpips": lpips_records,
    }
    capture_summary = json.loads(
        (capture / "capture_summary.json").read_text(encoding="utf-8")
    )
    elapsed_s = float(capture_summary["elapsed_s"])
    evaluated_video_frames = int(max(pred_lookup) - min(pred_lookup) + 1)
    runtime_metrics = {
        "elapsed_s": elapsed_s,
        "evaluated_video_frames": evaluated_video_frames,
        "video_frames_per_s": float(
            evaluated_video_frames / max(elapsed_s, 1.0e-12)
        ),
    }
    report = {
        "schema": "super_tissue_evaluation_results_v2",
        "protocol": protocol,
        "ground_truth": str(args.ground_truth.resolve()),
        "capture": str(capture),
        "point_tracking": track_metrics,
        "rendering": render_metrics,
        "runtime": runtime_metrics,
        "integrity": {
            "ground_truth_hash_matches_capture": True,
            "complete_track_schedule": not prefix_validation,
            "complete_validation_prefix_track_schedule": prefix_validation,
            "prefix_validation": prefix_validation,
            "formal_future_test_start": (
                formal_future_start if prefix_validation else None
            ),
            "prediction_end_inclusive": prediction_end,
            "scored_observations_withheld": True,
            "query_frame_excluded": query_frame,
            "render_partition_exact": bool(
                not metadata.get("capture_renders", True)
                or render_frames == expected_render_frames
            ),
        },
        "directions": {"2d_error_px": "lower", "3d_error_mm": "lower", "psnr_db": "higher", "ssim": "higher", "lpips_alex": "lower"},
    }
    (capture / "evaluation_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# SUPER tissue evaluation: {protocol}",
        "",
        "| Metric | Result | Better |",
        "|---|---:|:---:|",
        f"| 2D point error mean / RMSE | {formatted(track_metrics['2d_error_px']['mean'])} / {formatted(track_metrics['2d_error_px']['rmse'])} px | ↓ |",
        f"| TAP position accuracy δ_avg | {formatted(track_metrics['2d_tap_position_accuracy']['delta_avg'], 4)} | ↑ |",
        f"| 3D point error mean / RMSE | {formatted(track_metrics['3d_error_mm']['mean'])} / {formatted(track_metrics['3d_error_mm']['rmse'])} mm | ↓ |",
        f"| 3D GT coverage | {track_metrics['3d_coverage']['valid']}/{track_metrics['3d_coverage']['visible']} ({100.0 * track_metrics['3d_coverage']['fraction']:.1f}%) | ↑ |",
        f"| PSNR | {formatted(render_metrics['psnr_db'])} dB | ↑ |",
        f"| SSIM | {formatted(render_metrics['ssim'], 4)} | ↑ |",
        f"| LPIPS (AlexNet) | {render_metrics['lpips_alex'] if render_metrics['lpips_alex'] is not None else 'not computed'} | ↓ |",
        f"| End-to-end FPS | {formatted(runtime_metrics['video_frames_per_s'])} | ↑ |",
        "",
        "All trajectory errors are absolute (no per-frame rigid alignment). Instrument pixels are excluded from rendering metrics.",
    ]
    (capture / "evaluation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
