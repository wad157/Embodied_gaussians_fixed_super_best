#!/usr/bin/env python3
"""Create SUPER baseline trajectory and reconstruction review videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "baselines" / "endogaussian_super"
sys.path.insert(0, str(ADAPTER))

from protocol import DATASETS, PROTOCOL, dataset_spec, resolve_path  # noqa: E402


COLORS = (
    (54, 67, 244),
    (38, 166, 154),
    (65, 174, 255),
    (95, 189, 89),
    (186, 99, 46),
    (211, 85, 186),
    (82, 77, 230),
    (180, 130, 70),
    (50, 205, 50),
    (200, 150, 255),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-scale", type=float, default=0.6)
    parser.add_argument("--track-fps", type=float, default=10.0)
    parser.add_argument("--track-hold-frames", type=int, default=2)
    parser.add_argument("--track-trail-samples", type=int, default=18)
    parser.add_argument("--reconstruction-fps", type=float, default=15.0)
    parser.add_argument("--method-label", default="EndoGaussian")
    parser.add_argument("--output-prefix", default="endogaussian")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def dark_panel(image: np.ndarray, y0: int, y1: int, opacity: float = 0.78) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, y0), (image.shape[1], y1), (8, 8, 8), -1)
    image[:] = cv2.addWeighted(overlay, opacity, image, 1.0 - opacity, 0.0)


class H264Writer:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        if path.exists():
            raise FileExistsError("Refusing to overwrite {}".format(path))
        if width % 2 or height % 2:
            raise ValueError("H.264 output dimensions must be even")
        bundled = Path("/Media_HDD/jwshan/conda_envs/eg_codex/bin/ffmpeg")
        ffmpeg = str(bundled) if bundled.is_file() else shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is unavailable")
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "{}x{}".format(width, height),
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.path = path
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        returncode = self.process.wait()
        if returncode:
            raise RuntimeError("ffmpeg failed for {}: {}".format(self.path, stderr.strip()))


def mean_error(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, scale=1.0):
    selected = np.asarray(valid, dtype=bool) & np.isfinite(prediction).all(axis=-1)
    if not bool(selected.any()):
        return float("nan")
    return float(np.linalg.norm(prediction[selected] - target[selected], axis=-1).mean() * scale)


def phase(frame: int, future_start: int) -> tuple[str, tuple[int, int, int]]:
    if frame == 0:
        return "QUERY FRAME", (255, 210, 80)
    if frame >= future_start:
        return "FUTURE / UNSEEN RGB", (90, 90, 255)
    if frame % 8 == 0:
        return "RECONSTRUCTION HOLDOUT", (0, 225, 255)
    return "TRAINING PREFIX SAMPLE", (95, 235, 95)


def trajectory_video(
    *,
    dataset_key: str,
    gt: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    rgb_root: Path,
    future_start: int,
    output: Path,
    scale: float,
    fps: float,
    hold_frames: int,
    trail_samples: int,
    method_label: str,
) -> dict:
    frames = np.asarray(gt["frame_indices"], dtype=np.int32)
    if not np.array_equal(frames, prediction["frame_indices"]):
        raise ValueError("GT and predicted trajectory schedules differ")
    gt_uv = np.asarray(gt["uv"], dtype=np.float64)
    pred_uv = np.asarray(prediction["uv"], dtype=np.float64)
    gt_xyz = np.asarray(gt["xyz_camera_m"], dtype=np.float64)
    pred_xyz = np.asarray(prediction["xyz_camera_m"], dtype=np.float64)
    visible = np.asarray(gt["visible"], dtype=bool)
    valid_3d = np.asarray(gt["valid_3d"], dtype=bool)
    first = cv2.imread(str(rgb_root / "{:06d}-left.png".format(int(frames[0]))))
    if first is None:
        raise FileNotFoundError(rgb_root)
    size = (
        int(round(first.shape[1] * scale)) // 2 * 2,
        int(round(first.shape[0] * scale)) // 2 * 2,
    )
    writer = H264Writer(output, size[0], size[1], fps)
    try:
        for slot, frame_value in enumerate(frames):
            frame_index = int(frame_value)
            image_path = rgb_root / "{:06d}-left.png".format(frame_index)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            begin = max(0, slot - trail_samples + 1)
            overlay = image.copy()
            for point_id, color in enumerate(COLORS):
                gt_valid = visible[begin : slot + 1, point_id]
                pred_valid = gt_valid & np.isfinite(
                    pred_uv[begin : slot + 1, point_id]
                ).all(axis=1)
                if np.count_nonzero(gt_valid) >= 2:
                    points = np.rint(scale * gt_uv[begin : slot + 1, point_id][gt_valid]).astype(np.int32)
                    cv2.polylines(overlay, [points.reshape(-1, 1, 2)], False, (245, 245, 245), 2, cv2.LINE_AA)
                if np.count_nonzero(pred_valid) >= 2:
                    points = np.rint(scale * pred_uv[begin : slot + 1, point_id][pred_valid]).astype(np.int32)
                    cv2.polylines(overlay, [points.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
            image = cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0)
            for point_id, color in enumerate(COLORS):
                if not visible[slot, point_id]:
                    continue
                gt_point = tuple(np.rint(scale * gt_uv[slot, point_id]).astype(int))
                cv2.circle(image, gt_point, 7, (15, 15, 15), -1, cv2.LINE_AA)
                cv2.circle(image, gt_point, 5, (250, 250, 250), -1, cv2.LINE_AA)
                if np.isfinite(pred_uv[slot, point_id]).all():
                    pred_point = tuple(np.rint(scale * pred_uv[slot, point_id]).astype(int))
                    cv2.line(image, gt_point, pred_point, (0, 220, 255), 1, cv2.LINE_AA)
                    cv2.drawMarker(image, pred_point, (10, 10, 10), cv2.MARKER_TILTED_CROSS, 15, 4, cv2.LINE_AA)
                    cv2.drawMarker(image, pred_point, color, cv2.MARKER_TILTED_CROSS, 13, 2, cv2.LINE_AA)
                text(image, "P{}".format(point_id), (gt_point[0] + 7, gt_point[1] - 7), scale=0.4, color=(15, 15, 15), thickness=3)
                text(image, "P{}".format(point_id), (gt_point[0] + 7, gt_point[1] - 7), scale=0.4, color=(250, 250, 250))
            error_2d = mean_error(pred_uv[slot], gt_uv[slot], visible[slot])
            error_3d = mean_error(pred_xyz[slot], gt_xyz[slot], valid_3d[slot], 1000.0)
            label, label_color = phase(frame_index, future_start)
            dark_panel(image, 0, 105)
            text(image, "{} SUPER {} | frame {}/{} | {}".format(method_label, dataset_key, frame_index, int(frames[-1]), label), (14, 27), scale=0.62, color=label_color, thickness=2)
            text(image, "GT: white circle/trail | prediction: colored X/trail | connector: yellow", (14, 56), scale=0.49)
            text(image, "instant mean: 2D={:.2f} px | 3D={:.2f} mm | sparse annotated samples only".format(error_2d, error_3d), (14, 84), scale=0.52)
            for _ in range(hold_frames):
                writer.write(image)
            if (slot + 1) % 25 == 0 or slot + 1 == len(frames):
                print("[tracks] {}/{}".format(slot + 1, len(frames)), flush=True)
    finally:
        writer.close()
    return {"source_frame_count": int(len(frames)), "encoded_frame_count": int(len(frames) * hold_frames), "fps": fps, "size_wh": list(size)}


def reconstruction_video(capture: Path, future_start: int, output: Path, fps: float, method_label: str) -> dict:
    partial = json.loads((capture / "render_metrics_partial.json").read_text(encoding="utf-8"))
    records = partial["records"]
    frames = [int(record["frame_index"]) for record in records]
    if not frames:
        raise ValueError("No reconstruction records")
    first = cv2.imread(str(capture / "render_pairs" / "{:06d}-target.png".format(frames[0])))
    if first is None:
        raise FileNotFoundError("First render pair")
    panel_height, panel_width = first.shape[:2]
    writer = H264Writer(output, panel_width * 2, panel_height, fps)
    try:
        for slot, (frame_index, record) in enumerate(zip(frames, records)):
            stem = "{:06d}".format(frame_index)
            target = cv2.imread(str(capture / "render_pairs" / (stem + "-target.png")))
            prediction = cv2.imread(str(capture / "render_pairs" / (stem + "-prediction.png")))
            if target is None or prediction is None:
                raise FileNotFoundError("Missing render pair {}".format(stem))
            canvas = np.concatenate((target, prediction), axis=1)
            dark_panel(canvas, 0, 72)
            label, label_color = phase(frame_index, future_start)
            text(canvas, "TARGET (tool masked)", (14, 27), scale=0.62, thickness=2)
            text(canvas, method_label.upper(), (panel_width + 14, 27), scale=0.62, color=(95, 235, 95), thickness=2)
            text(canvas, "frame {} | {} | PSNR {:.2f} dB | SSIM {:.4f}".format(frame_index, label, float(record["psnr_db"]), float(record["ssim"])), (14, 58), scale=0.55, color=label_color, thickness=2)
            cv2.line(canvas, (panel_width, 0), (panel_width, panel_height), (255, 255, 255), 2)
            writer.write(canvas)
            if (slot + 1) % 50 == 0 or slot + 1 == len(frames):
                print("[reconstruction] {}/{}".format(slot + 1, len(frames)), flush=True)
    finally:
        writer.close()
    return {"encoded_frame_count": int(len(frames)), "fps": fps, "size_wh": [panel_width * 2, panel_height], "first_frame": frames[0], "last_frame": frames[-1]}


def main() -> None:
    args = parse_args()
    if not (0.1 <= args.track_scale <= 1.0):
        raise ValueError("track-scale must be in [0.1, 1]")
    if min(args.track_fps, args.reconstruction_fps) <= 0 or min(args.track_hold_frames, args.track_trail_samples) < 1:
        raise ValueError("Invalid fps/hold/trail")
    spec = dataset_spec(args.dataset_key)
    capture = args.capture.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metadata = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError("Capture protocol mismatch")
    gt_path = resolve_path(ROOT, spec["ground_truth"])
    if sha256(gt_path) != metadata.get("ground_truth_sha256"):
        raise ValueError("Capture and GT hashes differ")
    gt = read_npz(gt_path)
    prediction_path = capture / "predicted_tracks.npz"
    prediction = read_npz(prediction_path)
    future_start = int(spec["future_start"])
    tracks_path = output_dir / (args.output_prefix + "_tracks_query_anchored_vs_gt.mp4")
    reconstruction_path = output_dir / (args.output_prefix + "_reconstruction_vs_target.mp4")
    tracks = trajectory_video(
        dataset_key=args.dataset_key,
        gt=gt,
        prediction=prediction,
        rgb_root=resolve_path(ROOT, spec["native"]) / "rgb",
        future_start=future_start,
        output=tracks_path,
        scale=args.track_scale,
        fps=args.track_fps,
        hold_frames=args.track_hold_frames,
        trail_samples=args.track_trail_samples,
        method_label=args.method_label,
    )
    reconstruction = reconstruction_video(
        capture,
        future_start,
        reconstruction_path,
        args.reconstruction_fps,
        args.method_label,
    )
    manifest = {
        "schema": "super_baseline_review_videos_v1",
        "method": args.method_label,
        "dataset_key": args.dataset_key,
        "capture": str(capture),
        "protocol": PROTOCOL,
        "ground_truth": str(gt_path),
        "ground_truth_sha256": sha256(gt_path),
        "prediction_sha256": sha256(prediction_path),
        "tracks": {"path": str(tracks_path), **tracks},
        "reconstruction": {"path": str(reconstruction_path), **reconstruction},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
