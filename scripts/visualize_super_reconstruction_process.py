#!/usr/bin/env python3
"""Visualize SUPER reconstruction continuously from initialization to the end.

The available frozen tracks contain 149 manually annotated timestamps spanning
video frames 0--1439.  The generated MP4 is therefore an annotated-sample
time-lapse, not an interpolation to all 1440 source frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import visualize_super_ten_point_evaluation as evaluation_viz


DEFAULT_OUTPUT = evaluation_viz.DEFAULT_OUTPUT / "reconstruction_7to1"
RAW_LEFT_ROOT = evaluation_viz.REPO_ROOT / "data/super/grasp5_native/rgb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=evaluation_viz.DEFAULT_GT)
    parser.add_argument(
        "--annotations", type=Path, default=evaluation_viz.DEFAULT_ANNOTATIONS
    )
    parser.add_argument(
        "--baseline-root", type=Path, default=evaluation_viz.DEFAULT_BASELINE_ROOT
    )
    parser.add_argument(
        "--stiffness-root", type=Path, default=evaluation_viz.DEFAULT_STIFFNESS_ROOT
    )
    parser.add_argument("--raw-left-root", type=Path, default=RAW_LEFT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--video-scale", type=float, default=0.5)
    parser.add_argument("--trail-samples", type=int, default=18)
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def methods_from_args(args: argparse.Namespace) -> list[evaluation_viz.Method]:
    return [
        evaluation_viz.Method(
            "pure_pbd", "Pure PBD", "#e53935",
            args.baseline_root.resolve() / "pure_pbd",
        ),
        evaluation_viz.Method(
            "visual_residual", "PBD + RGB residual", "#1e88e5",
            args.baseline_root.resolve() / "pbd_visual_residual",
        ),
        evaluation_viz.Method(
            "visual_stiffness", "PBD + RGB residual + stiffness", "#8e44ad",
            args.stiffness_root.resolve() / "pbd_visual_residual_online_stiffness",
        ),
    ]


def full_errors(
    ground_truth: dict[str, np.ndarray],
    data: evaluation_viz.ProtocolData,
    methods: list[evaluation_viz.Method],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    errors_2d = {}
    errors_3d = {}
    visible = np.asarray(ground_truth["visible"], dtype=bool)
    valid_3d = np.asarray(ground_truth["valid_3d"], dtype=bool)
    for method in methods:
        prediction = data.predictions[method.key]
        error_2d = np.linalg.norm(
            np.asarray(prediction["uv"]) - np.asarray(ground_truth["uv"]), axis=-1
        )
        error_3d = 1000.0 * np.linalg.norm(
            np.asarray(prediction["xyz_camera_m"])
            - np.asarray(ground_truth["xyz_camera_m"]),
            axis=-1,
        )
        error_2d[~visible] = np.nan
        error_3d[~valid_3d] = np.nan
        errors_2d[method.key] = error_2d
        errors_3d[method.key] = error_3d
    return errors_2d, errors_3d


def plot_full_2d(
    ground_truth: dict[str, np.ndarray],
    data: evaluation_viz.ProtocolData,
    methods: list[evaluation_viz.Method],
    point_names: list[str],
    background: np.ndarray,
    output_dir: Path,
    dpi: int,
    write_pdf: bool,
) -> list[str]:
    gt_uv = np.asarray(ground_truth["uv"])
    visible = np.asarray(ground_truth["visible"], dtype=bool)
    frames = np.asarray(ground_truth["frame_indices"], dtype=np.int32)
    withheld = np.isin(frames, data.scored_frames)
    height, width = background.shape[:2]
    figure, axes = plt.subplots(5, 2, figsize=(13.2, 22.0), constrained_layout=True)
    for point_slot, axis in enumerate(axes.flat):
        combined = [gt_uv[:, point_slot]] + [
            np.asarray(data.predictions[method.key]["uv"])[:, point_slot]
            for method in methods
        ]
        x0, x1, y0, y1 = evaluation_viz.crop_limits(
            np.concatenate(combined, axis=0), (width, height)
        )
        axis.imshow(background, extent=(0, width, height, 0), alpha=0.60)
        valid = visible[:, point_slot]
        axis.plot(
            gt_uv[valid, point_slot, 0], gt_uv[valid, point_slot, 1],
            color="black", linewidth=2.5, marker=".", markersize=2.5,
            label="Ground truth", zorder=6,
        )
        for method in methods:
            pred_uv = np.asarray(data.predictions[method.key]["uv"])[:, point_slot]
            axis.plot(
                pred_uv[valid, 0], pred_uv[valid, 1], color=method.color,
                linewidth=1.65, label=method.label, zorder=4,
            )
            held = valid & withheld
            axis.scatter(
                pred_uv[held, 0], pred_uv[held, 1], s=11, marker="o",
                facecolor="white", edgecolor=method.color, linewidth=0.65,
                zorder=5,
            )
        # Concentric open squares make the common frame-0 initialization visible.
        sizes = (82, 60, 42, 26)
        colors = ("black",) + tuple(method.color for method in methods)
        initial = gt_uv[0, point_slot]
        for size, color in zip(sizes, colors):
            axis.scatter(
                [initial[0]], [initial[1]], s=size, marker="s",
                facecolor="none", edgecolor=color, linewidth=1.25, zorder=8,
            )
        axis.scatter(
            [gt_uv[-1, point_slot, 0]], [gt_uv[-1, point_slot, 1]],
            s=48, marker="X", facecolor="black", edgecolor="white",
            linewidth=0.8, zorder=8,
        )
        axis.set_xlim(x0, x1)
        axis.set_ylim(y1, y0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{point_names[point_slot]} — frames 0–1439")
        axis.set_xlabel("u (full-resolution pixel)")
        axis.set_ylabel("v (full-resolution pixel)")
        axis.grid(color="white", alpha=0.18, linewidth=0.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
    figure.suptitle(
        "SUPER reconstruction from initialization to end — full 2D trajectories\n"
        "concentric squares=common frame-0 initialization; open circles=7-to-1 held-out samples; X=GT end",
        fontsize=16,
    )
    return evaluation_viz.save_figure(
        figure, output_dir / "full_sequence_2d_trajectories_stereo_left",
        dpi=dpi, write_pdf=write_pdf,
    )


def plot_full_3d(
    ground_truth: dict[str, np.ndarray],
    data: evaluation_viz.ProtocolData,
    methods: list[evaluation_viz.Method],
    point_names: list[str],
    output_dir: Path,
    dpi: int,
    write_pdf: bool,
) -> list[str]:
    gt_xyz = np.asarray(ground_truth["xyz_world_m"])
    valid_3d = np.asarray(ground_truth["valid_3d"], dtype=bool)
    figure = plt.figure(figsize=(14.0, 24.0), constrained_layout=True)
    first_handles = None
    for point_slot, point_name in enumerate(point_names):
        axis = figure.add_subplot(5, 2, point_slot + 1, projection="3d")
        valid = valid_3d[:, point_slot]
        origin = gt_xyz[0, point_slot]
        gt = 1000.0 * (gt_xyz[:, point_slot] - origin)
        trajectories = [gt[valid]]
        predicted = {}
        for method in methods:
            xyz = np.asarray(data.predictions[method.key]["xyz_world_m"])[
                :, point_slot
            ]
            predicted[method.key] = 1000.0 * (xyz - origin)
            trajectories.append(predicted[method.key][valid])
        center, radius = evaluation_viz.equal_3d_limits(trajectories)
        axis.plot(
            gt[valid, 0], gt[valid, 1], gt[valid, 2], color="black",
            linewidth=2.5, label="Ground truth",
        )
        for method in methods:
            xyz = predicted[method.key]
            axis.plot(
                xyz[valid, 0], xyz[valid, 1], xyz[valid, 2],
                color=method.color, linewidth=1.65, label=method.label,
            )
        # All four start markers are concentric at (0, 0, 0).
        for size, color in zip(
            (78, 58, 40, 25), ("black",) + tuple(m.color for m in methods)
        ):
            axis.scatter(
                [0.0], [0.0], [0.0], s=size, marker="s", facecolor="none",
                edgecolor=color, linewidth=1.2,
            )
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_xlabel(r"$\Delta X_{world}$ (mm)")
        axis.set_ylabel(r"$\Delta Y_{world}$ (mm)")
        axis.set_zlabel(r"$\Delta Z_{world}$ (mm)")
        axis.set_title(f"{point_name} — relative to frame-0 GT")
        axis.view_init(elev=24, azim=-56)
        axis.grid(True, alpha=0.30)
        if first_handles is None:
            first_handles = axis.get_legend_handles_labels()
    assert first_handles is not None
    figure.legend(
        first_handles[0], first_handles[1], loc="lower center", ncol=4,
        frameon=True,
    )
    figure.suptitle(
        "SUPER reconstruction from initialization to end — full 3D trajectories\n"
        "all methods use the same frame-0 GT origin; axes preserve absolute millimeter offsets",
        fontsize=16,
    )
    return evaluation_viz.save_figure(
        figure, output_dir / "full_sequence_3d_trajectories",
        dpi=dpi, write_pdf=write_pdf,
    )


def closest_slots(frames: np.ndarray, targets: tuple[int, ...]) -> list[int]:
    return [int(np.argmin(np.abs(frames - target))) for target in targets]


def plot_checkpoints(
    ground_truth: dict[str, np.ndarray],
    data: evaluation_viz.ProtocolData,
    methods: list[evaluation_viz.Method],
    point_names: list[str],
    raw_left_root: Path,
    errors_2d: dict[str, np.ndarray],
    errors_3d: dict[str, np.ndarray],
    output_dir: Path,
    dpi: int,
    write_pdf: bool,
) -> tuple[list[str], list[int]]:
    frames = np.asarray(ground_truth["frame_indices"], dtype=np.int32)
    slots = closest_slots(frames, (0, 200, 400, 600, 800, 1000, 1152, 1280, 1439))
    gt_uv = np.asarray(ground_truth["uv"])
    figure, axes = plt.subplots(3, 3, figsize=(18.0, 10.8), constrained_layout=True)
    for axis, slot in zip(axes.flat, slots):
        frame = int(frames[slot])
        image = mpimg.imread(raw_left_root / f"{frame:06d}-left.png")
        height, width = image.shape[:2]
        axis.imshow(image, extent=(0, width, height, 0), alpha=0.78)
        current_gt = gt_uv[slot]
        axis.scatter(
            current_gt[:, 0], current_gt[:, 1], s=35, color="black",
            edgecolor="white", linewidth=0.8, label="Ground truth", zorder=6,
        )
        for point_name, (u, v) in zip(point_names, current_gt):
            axis.annotate(
                point_name, (u, v), xytext=(4, -5), textcoords="offset points",
                fontsize=6.5, color="black", fontweight="bold", zorder=7,
            )
        metric_lines = []
        for method in methods:
            pred_uv = np.asarray(data.predictions[method.key]["uv"])[slot]
            for gt_point, pred_point in zip(current_gt, pred_uv):
                axis.plot(
                    [gt_point[0], pred_point[0]], [gt_point[1], pred_point[1]],
                    color=method.color, linewidth=0.55, alpha=0.48, zorder=3,
                )
            axis.scatter(
                pred_uv[:, 0], pred_uv[:, 1], s=25, marker="x",
                color=method.color, linewidth=1.25, label=method.label, zorder=5,
            )
            metric_lines.append(
                f"{method.label}: {np.nanmean(errors_2d[method.key][slot]):.1f}px / "
                f"{np.nanmean(errors_3d[method.key][slot]):.2f}mm"
            )
        withheld = frame in set(data.scored_frames.astype(int).tolist())
        axis.set_title(
            f"frame {frame} — {'held-out' if withheld else 'visual observation allowed'}",
            fontsize=10,
        )
        axis.text(
            0.012, 0.985, "\n".join(metric_lines), transform=axis.transAxes,
            ha="left", va="top", fontsize=6.9,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.7"},
        )
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
    figure.suptitle(
        "SUPER reconstruction process checkpoints — GT dots, predicted crosses, absolute-error links",
        fontsize=16,
    )
    files = evaluation_viz.save_figure(
        figure, output_dir / "full_reconstruction_process_checkpoints_stereo_left",
        dpi=dpi, write_pdf=write_pdf,
    )
    return files, [int(frames[slot]) for slot in slots]


def draw_halo_circle(
    image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int]
) -> None:
    cv2.circle(image, center, 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, center, 4, color, -1, cv2.LINE_AA)


def generate_video(
    ground_truth: dict[str, np.ndarray],
    data: evaluation_viz.ProtocolData,
    methods: list[evaluation_viz.Method],
    point_names: list[str],
    raw_left_root: Path,
    errors_2d: dict[str, np.ndarray],
    errors_3d: dict[str, np.ndarray],
    output: Path,
    fps: float,
    scale: float,
    trail_samples: int,
) -> dict[str, object]:
    if fps <= 0.0 or not 0.0 < scale <= 1.0 or trail_samples < 1:
        raise ValueError("fps, scale, and trail-samples must be positive")
    frames = np.asarray(ground_truth["frame_indices"], dtype=np.int32)
    gt_uv = np.asarray(ground_truth["uv"], dtype=np.float64)
    first = cv2.imread(str(raw_left_root / f"{int(frames[0]):06d}-left.png"))
    if first is None:
        raise FileNotFoundError(raw_left_root / f"{int(frames[0]):06d}-left.png")
    output_size = (int(round(first.shape[1] * scale)), int(round(first.shape[0] * scale)))
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {output}")
    bgr_colors = {
        "pure_pbd": (53, 57, 229),
        "visual_residual": (229, 136, 30),
        "visual_stiffness": (173, 68, 142),
    }
    scored_frames = set(data.scored_frames.astype(int).tolist())
    try:
        for slot, frame_value in enumerate(frames):
            frame = int(frame_value)
            image = cv2.imread(str(raw_left_root / f"{frame:06d}-left.png"))
            if image is None:
                raise FileNotFoundError(raw_left_root / f"{frame:06d}-left.png")
            image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
            overlay = image.copy()
            begin = max(0, slot - trail_samples + 1)
            gt_trail = np.rint(scale * gt_uv[begin : slot + 1]).astype(np.int32)
            for point_slot in range(len(point_names)):
                gt_polyline = np.ascontiguousarray(
                    gt_trail[:, point_slot]
                ).reshape(-1, 1, 2)
                cv2.polylines(
                    overlay, [gt_polyline], False, (20, 20, 20),
                    2, cv2.LINE_AA,
                )
            for method in methods:
                pred_uv = np.asarray(data.predictions[method.key]["uv"], dtype=np.float64)
                trail = np.rint(scale * pred_uv[begin : slot + 1]).astype(np.int32)
                color = bgr_colors[method.key]
                for point_slot in range(len(point_names)):
                    method_polyline = np.ascontiguousarray(
                        trail[:, point_slot]
                    ).reshape(-1, 1, 2)
                    cv2.polylines(
                        overlay, [method_polyline], False, color, 2,
                        cv2.LINE_AA,
                    )
            image = cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0)
            current_gt = np.rint(scale * gt_uv[slot]).astype(np.int32)
            for point_slot, (u, v) in enumerate(current_gt):
                draw_halo_circle(image, (int(u), int(v)), (10, 10, 10))
                cv2.putText(
                    image, point_names[point_slot], (int(u) + 5, int(v) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA,
                )
            for method in methods:
                color = bgr_colors[method.key]
                pred_uv = np.asarray(data.predictions[method.key]["uv"])[slot]
                current_pred = np.rint(scale * pred_uv).astype(np.int32)
                for gt_point, pred_point in zip(current_gt, current_pred):
                    cv2.line(
                        image, tuple(gt_point), tuple(pred_point), color, 1,
                        cv2.LINE_AA,
                    )
                    draw_halo_circle(image, tuple(pred_point), color)

            panel_height = 119
            cv2.rectangle(image, (0, 0), (output_size[0], panel_height), (255, 255, 255), -1)
            split = "HELD-OUT 1/8" if frame in scored_frames else "VISUAL OBSERVATION ALLOWED"
            cv2.putText(
                image,
                f"SUPER reconstruction 7-to-1 | frame {frame:04d} | sample {slot + 1:03d}/{len(frames)} | {split}",
                (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 20), 1,
                cv2.LINE_AA,
            )
            legend_x = 12
            for row, method in enumerate(methods):
                y = 48 + 22 * row
                color = bgr_colors[method.key]
                cv2.circle(image, (legend_x + 5, y - 4), 4, color, -1, cv2.LINE_AA)
                text = (
                    f"{method.label}: mean 2D {np.nanmean(errors_2d[method.key][slot]):6.2f}px | "
                    f"mean 3D {np.nanmean(errors_3d[method.key][slot]):5.3f}mm"
                )
                cv2.putText(
                    image, text, (legend_x + 17, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (30, 30, 30), 1, cv2.LINE_AA,
                )
            writer.write(image)
    finally:
        writer.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Video was not written: {output}")
    return {
        "file": output.name,
        "annotated_sample_count": int(len(frames)),
        "source_frame_start": int(frames[0]),
        "source_frame_end": int(frames[-1]),
        "fps": float(fps),
        "output_size_wh": list(output_size),
        "trail_samples": int(trail_samples),
        "duration_s": float(len(frames) / fps),
        "note": "annotated-sample time-lapse; no temporal interpolation",
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path = args.ground_truth.resolve()
    ground_truth = evaluation_viz.read_npz(ground_truth_path)
    annotations = json.loads(args.annotations.resolve().read_text(encoding="utf-8"))
    point_names = [point["name"] for point in annotations["points"]]
    methods = methods_from_args(args)
    data = evaluation_viz.load_protocol(
        "reconstruction_7to1", methods, ground_truth, ground_truth_path
    )
    raw_left_root = args.raw_left_root.resolve()
    background = mpimg.imread(raw_left_root / "000000-left.png")
    errors_2d, errors_3d = full_errors(ground_truth, data, methods)

    files = []
    files.extend(
        plot_full_2d(
            ground_truth, data, methods, point_names, background, output_dir,
            args.dpi, not args.no_pdf,
        )
    )
    files.extend(
        plot_full_3d(
            ground_truth, data, methods, point_names, output_dir,
            args.dpi, not args.no_pdf,
        )
    )
    checkpoint_files, checkpoint_frames = plot_checkpoints(
        ground_truth, data, methods, point_names, raw_left_root,
        errors_2d, errors_3d, output_dir, args.dpi, not args.no_pdf,
    )
    files.extend(checkpoint_files)
    video = None
    if not args.skip_video:
        video = generate_video(
            ground_truth, data, methods, point_names, raw_left_root,
            errors_2d, errors_3d,
            output_dir / "full_reconstruction_process_stereo_left.mp4",
            args.video_fps, args.video_scale, args.trail_samples,
        )
        files.append(str(video["file"]))

    frame_zero = {}
    final_frame = {}
    for method in methods:
        frame_zero[method.key] = {
            "2d_mean_px": float(np.nanmean(errors_2d[method.key][0])),
            "3d_mean_mm": float(np.nanmean(errors_3d[method.key][0])),
        }
        final_frame[method.key] = {
            "2d_mean_px": float(np.nanmean(errors_2d[method.key][-1])),
            "3d_mean_mm": float(np.nanmean(errors_3d[method.key][-1])),
        }
    manifest = {
        "schema": "super_full_reconstruction_process_visualization_v1",
        "protocol": "reconstruction_7to1",
        "ground_truth": str(ground_truth_path),
        "ground_truth_sha256": evaluation_viz.sha256(ground_truth_path),
        "source_schedule": {
            "sample_count": int(len(ground_truth["frame_indices"])),
            "frame_start": int(ground_truth["frame_indices"][0]),
            "frame_end": int(ground_truth["frame_indices"][-1]),
            "manual_annotation_timestamps_only": True,
            "temporal_interpolation": False,
        },
        "common_initialization_check": frame_zero,
        "final_frame_error": final_frame,
        "checkpoint_frames": checkpoint_frames,
        "video": video,
        "files": files,
    }
    (output_dir / "full_reconstruction_process_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    readme = [
        "# SUPER 从初始化到结束的重建过程",
        "",
        "这组图使用 reconstruction 7-to-1 协议，覆盖第 0 帧至第 1439 帧的 149 个人工标注时刻。",
        "所有方法在第 0 帧共同初始化；完整轨迹保留绝对漂移，没有分别平移预测曲线来强制对齐。",
        "视频是标注时刻的延时过程，不对未标注帧插值。",
        "",
        "- [完整 2D 轨迹](full_sequence_2d_trajectories_stereo_left.png)",
        "- [完整 3D 轨迹](full_sequence_3d_trajectories.png)",
        "- [九阶段重建过程](full_reconstruction_process_checkpoints_stereo_left.png)",
        "- [动态重建过程](full_reconstruction_process_stereo_left.mp4)",
        "- [数据与完整性清单](full_reconstruction_process_manifest.json)",
    ]
    (output_dir / "FULL_RECONSTRUCTION_PROCESS.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
