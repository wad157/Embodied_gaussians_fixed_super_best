#!/usr/bin/env python3
"""Overlay formal SUPER GT and three evaluated methods in one motion video.

Only the scorer's held-out frame schedule is rendered.  Lines connect real
manual annotation samples; no GT or prediction is interpolated between them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION = (
    ROOT
    / "outputs/super_alltracker_rgb_recon_nonoverlap_h3_full_metrics_20260831_v10"
)
DEFAULT_GT = (
    ROOT
    / "data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
)
DEFAULT_ANNOTATIONS = (
    ROOT / "data/super/evaluation_v1/manual_tissue_tracks_10/annotations.json"
)
DEFAULT_RGB = ROOT / "data/super/grasp5_native/rgb"


METHODS = (
    (
        "pure_pbd",
        "Pure PBD",
        (235, 135, 35),
        cv2.MARKER_TRIANGLE_UP,
    ),
    (
        "pbd_alltracker_depth_rgb_residual",
        "PBD + trajectory + RGB",
        (45, 185, 70),
        cv2.MARKER_SQUARE,
    ),
    (
        "pbd_alltracker_depth_rgb_residual_online_stiffness",
        "PBD + trajectory + RGB + H3",
        (205, 65, 215),
        cv2.MARKER_DIAMOND,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--protocol",
        choices=("reconstruction_7to1", "future_80to20"),
        default="reconstruction_7to1",
    )
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--rgb-root", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--hold-frames", type=int, default=2)
    parser.add_argument("--scale", type=float, default=0.72)
    parser.add_argument("--trail-samples", type=int, default=16)
    return parser.parse_args()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_error(
    prediction: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    *,
    scale: float = 1.0,
) -> float:
    values = np.linalg.norm(prediction - truth, axis=-1)[valid]
    return float(np.mean(values) * scale) if len(values) else float("nan")


def text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    *,
    size: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    if args.fps <= 0.0 or args.hold_frames < 1:
        raise ValueError("fps and hold-frames must be positive")
    if not 0.0 < args.scale <= 1.0 or args.trail_samples < 2:
        raise ValueError("scale/trail-samples are invalid")

    evaluation_root = args.evaluation_root.resolve()
    protocol = args.protocol
    ground_truth_path = args.ground_truth.resolve()
    ground_truth = read_npz(ground_truth_path)
    gt_frames = np.asarray(ground_truth["frame_indices"], dtype=np.int32)
    gt_lookup = {int(frame): slot for slot, frame in enumerate(gt_frames)}
    annotations = json.loads(args.annotations.resolve().read_text(encoding="utf-8"))
    names = [
        str(item.get("name", f"P{index}"))
        for index, item in enumerate(annotations["points"])
    ]
    if len(names) != 10:
        raise ValueError("Expected exactly ten evaluation points")

    method_data: dict[str, dict[str, object]] = {}
    scored_frames: np.ndarray | None = None
    expected_gt_hash = sha256(ground_truth_path)
    for key, label, color, marker in METHODS:
        capture = evaluation_root / key / protocol
        report_path = capture / "evaluation_results.json"
        metadata_path = capture / "metadata.json"
        prediction_path = capture / "predicted_tracks.npz"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["ground_truth_sha256"] != expected_gt_hash:
            raise ValueError(f"Ground-truth hash mismatch: {capture}")
        current_scored = np.asarray(
            report["point_tracking"]["scored_frames"], dtype=np.int32
        )
        if scored_frames is None:
            scored_frames = current_scored
        elif not np.array_equal(scored_frames, current_scored):
            raise ValueError("The three methods use different scored schedules")
        prediction = read_npz(prediction_path)
        pred_frames = np.asarray(prediction["frame_indices"], dtype=np.int32)
        if not np.array_equal(pred_frames, gt_frames):
            raise ValueError(f"Prediction schedule mismatch: {prediction_path}")
        method_data[key] = {
            "label": label,
            "color": color,
            "marker": marker,
            "prediction": prediction,
            "report": report,
            "prediction_path": str(prediction_path),
        }
    assert scored_frames is not None
    slots = np.asarray([gt_lookup[int(frame)] for frame in scored_frames])
    gt_uv = np.asarray(ground_truth["uv"], dtype=np.float64)[slots]
    gt_xyz = np.asarray(ground_truth["xyz_camera_m"], dtype=np.float64)[slots]
    visible = np.asarray(ground_truth["visible"], dtype=bool)[slots]
    valid_3d = np.asarray(ground_truth["valid_3d"], dtype=bool)[slots]
    for key, *_ in METHODS:
        prediction = method_data[key]["prediction"]
        assert isinstance(prediction, dict)
        method_data[key]["uv"] = np.asarray(prediction["uv"])[slots]
        method_data[key]["xyz"] = np.asarray(prediction["xyz_camera_m"])[slots]
        if np.any(np.asarray(prediction["observation_used"])[slots]):
            raise ValueError(f"Scored frame consumed a visual observation: {key}")

    first_image_path = args.rgb_root.resolve() / f"{int(scored_frames[0]):06d}-left.png"
    first = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(first_image_path)
    output_size = (
        int(round(first.shape[1] * args.scale)),
        int(round(first.shape[0] * args.scale)),
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else evaluation_root
        / "visualizations"
        / f"{protocol}_gt_three_method_motion.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, output_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")

    panel_height = int(round(154 * args.scale / 0.72))
    try:
        for sample, frame in enumerate(scored_frames):
            image_path = args.rgb_root.resolve() / f"{int(frame):06d}-left.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
            begin = max(0, sample - args.trail_samples + 1)
            overlay = image.copy()

            # Trails use method color; GT is a high-contrast white/black line.
            for point_id in range(10):
                gt_valid = visible[begin : sample + 1, point_id]
                if np.count_nonzero(gt_valid) >= 2:
                    gt_poly = np.rint(
                        args.scale * gt_uv[begin : sample + 1, point_id][gt_valid]
                    ).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        overlay, [gt_poly], False, (15, 15, 15), 4, cv2.LINE_AA
                    )
                    cv2.polylines(
                        overlay, [gt_poly], False, (245, 245, 245), 2, cv2.LINE_AA
                    )
                for key, *_ in METHODS:
                    color = method_data[key]["color"]
                    uv = method_data[key]["uv"]
                    assert isinstance(color, tuple) and isinstance(uv, np.ndarray)
                    valid = gt_valid & np.isfinite(
                        uv[begin : sample + 1, point_id]
                    ).all(axis=1)
                    if np.count_nonzero(valid) >= 2:
                        poly = np.rint(
                            args.scale
                            * uv[begin : sample + 1, point_id][valid]
                        ).astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(
                            overlay, [poly], False, color, 2, cv2.LINE_AA
                        )
            image = cv2.addWeighted(overlay, 0.82, image, 0.18, 0.0)

            # Current GT and method positions, plus their instantaneous error vectors.
            for point_id in range(10):
                if not visible[sample, point_id]:
                    continue
                gt_point = tuple(
                    np.rint(args.scale * gt_uv[sample, point_id]).astype(int)
                )
                cv2.circle(image, gt_point, 8, (15, 15, 15), -1, cv2.LINE_AA)
                cv2.circle(image, gt_point, 5, (250, 250, 250), -1, cv2.LINE_AA)
                text(
                    image,
                    f"P{point_id}",
                    (gt_point[0] + 8, gt_point[1] - 8),
                    size=0.39,
                    color=(15, 15, 15),
                    thickness=3,
                )
                text(
                    image,
                    f"P{point_id}",
                    (gt_point[0] + 8, gt_point[1] - 8),
                    size=0.39,
                    color=(250, 250, 250),
                )
                for key, *_ in METHODS:
                    color = method_data[key]["color"]
                    marker = method_data[key]["marker"]
                    uv = method_data[key]["uv"]
                    assert (
                        isinstance(color, tuple)
                        and isinstance(marker, int)
                        and isinstance(uv, np.ndarray)
                    )
                    if not np.isfinite(uv[sample, point_id]).all():
                        continue
                    point = tuple(
                        np.rint(args.scale * uv[sample, point_id]).astype(int)
                    )
                    cv2.line(image, gt_point, point, color, 1, cv2.LINE_AA)
                    cv2.drawMarker(
                        image, point, (250, 250, 250), marker, 16, 4, cv2.LINE_AA
                    )
                    cv2.drawMarker(
                        image, point, color, marker, 13, 2, cv2.LINE_AA
                    )

            # Dark translucent information panel.
            panel = image.copy()
            cv2.rectangle(panel, (0, 0), (output_size[0], panel_height), (12, 12, 12), -1)
            image = cv2.addWeighted(panel, 0.82, image, 0.18, 0.0)
            title = (
                "RECONSTRUCTION 7:1 HELD-OUT"
                if protocol == "reconstruction_7to1"
                else "FUTURE 80:20 OPEN-LOOP TEST"
            )
            text(
                image,
                f"GT vs three methods | {title} | source frame {int(frame):04d} | "
                f"manual sample {sample + 1:02d}/{len(scored_frames):02d}",
                (16, 28),
                size=0.57,
                color=(245, 245, 245),
            )
            text(
                image,
                "GT: white circle/white-black trail (manual annotations only; no interpolation)",
                (16, 54),
                size=0.45,
                color=(230, 230, 230),
            )
            column_width = max(300, output_size[0] // 3)
            for method_index, (key, _label, _color, _marker) in enumerate(METHODS):
                label = method_data[key]["label"]
                color = method_data[key]["color"]
                marker = method_data[key]["marker"]
                uv = method_data[key]["uv"]
                xyz = method_data[key]["xyz"]
                assert (
                    isinstance(label, str)
                    and isinstance(color, tuple)
                    and isinstance(marker, int)
                    and isinstance(uv, np.ndarray)
                    and isinstance(xyz, np.ndarray)
                )
                error_2d = mean_error(
                    uv[sample], gt_uv[sample], visible[sample]
                )
                error_3d = mean_error(
                    xyz[sample], gt_xyz[sample], valid_3d[sample], scale=1000.0
                )
                x = 18 + method_index * column_width
                cv2.drawMarker(image, (x + 7, 82), color, marker, 13, 2, cv2.LINE_AA)
                text(image, label, (x + 22, 87), size=0.43, color=color)
                text(
                    image,
                    f"current mean: {error_2d:6.2f} px | {error_3d:5.3f} mm",
                    (x, 113),
                    size=0.43,
                    color=(238, 238, 238),
                )
                metrics = method_data[key]["report"]
                assert isinstance(metrics, dict)
                aggregate = metrics["point_tracking"]
                text(
                    image,
                    "formal mean: "
                    f"{aggregate['2d_error_px']['mean']:6.2f} px | "
                    f"{aggregate['3d_error_mm']['mean']:5.3f} mm",
                    (x, 138),
                    size=0.41,
                    color=(205, 205, 205),
                )
            for _ in range(args.hold_frames):
                writer.write(image)
    finally:
        writer.release()

    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"Video was not written: {output}")
    report = {
        "schema": "super_three_method_gt_keypoint_video_v1",
        "protocol": protocol,
        "output": str(output),
        "video_bytes": output.stat().st_size,
        "fps": args.fps,
        "hold_frames": args.hold_frames,
        "video_frames": int(len(scored_frames) * args.hold_frames),
        "duration_s": float(len(scored_frames) * args.hold_frames / args.fps),
        "source_frames": scored_frames.tolist(),
        "ground_truth": str(ground_truth_path),
        "ground_truth_sha256": expected_gt_hash,
        "methods": {
            key: {
                "label": method_data[key]["label"],
                "prediction": method_data[key]["prediction_path"],
            }
            for key, *_ in METHODS
        },
        "truth_policy": (
            "manual annotated scored frames only; trails connect samples; "
            "no interpolation"
        ),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output} ({output.stat().st_size} bytes, "
        f"{report['video_frames']} frames, {report['duration_s']:.2f}s)"
    )


if __name__ == "__main__":
    main()
