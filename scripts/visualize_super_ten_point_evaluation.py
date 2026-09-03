#!/usr/bin/env python3
"""Visualize the formal SUPER 10-point reconstruction/future evaluation.

The plots deliberately use only the frozen manual 2D tracks and their strict
stereo-depth back-projections.  In particular, the spatial maps are sparse
10-point maps; they do not interpolate the observations into a dense surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = (
    REPO_ROOT
    / "data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
)
DEFAULT_ANNOTATIONS = (
    REPO_ROOT
    / "data/super/evaluation_v1/manual_tissue_tracks_10/annotations.json"
)
DEFAULT_BACKGROUND = REPO_ROOT / "data/super/grasp5_native/rgb/000000-left.png"
DEFAULT_BASELINE_ROOT = (
    REPO_ROOT / "outputs/super_stiffness_consensus_full_evaluation_20260825_v1"
)
DEFAULT_STIFFNESS_ROOT = (
    REPO_ROOT / "outputs/super_stiffness_dual_gradient_full_evaluation_20260826_v2"
)
DEFAULT_OUTPUT = DEFAULT_STIFFNESS_ROOT / "visualizations/ten_points_v1"
PROTOCOLS = ("reconstruction_7to1", "future_80to20")


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    color: str
    root: Path


@dataclass
class ProtocolData:
    name: str
    scored_frames: np.ndarray
    gt_slots: np.ndarray
    gt_uv: np.ndarray
    gt_xyz_world_m: np.ndarray
    visible: np.ndarray
    valid_3d: np.ndarray
    predictions: dict[str, dict[str, np.ndarray]]
    errors_2d_px: dict[str, np.ndarray]
    errors_3d_mm: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--stiffness-root", type=Path, default=DEFAULT_STIFFNESS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--no-pdf", action="store_true", help="Write PNG only instead of PNG and PDF."
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def finite_rmse(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.nanmean(values * values, axis=axis))


def finite_mean(values: np.ndarray) -> float:
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def protocol_title(protocol: str) -> str:
    if protocol == "reconstruction_7to1":
        return "Reconstruction 7-to-1 held-out frames"
    if protocol == "future_80to20":
        return "Future 80-to-20 test frames (no visual updates)"
    raise ValueError(protocol)


def save_figure(
    figure: plt.Figure,
    output_stem: Path,
    *,
    dpi: int,
    write_pdf: bool,
) -> list[str]:
    outputs = []
    png = output_stem.with_suffix(".png")
    figure.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight")
    outputs.append(png.name)
    if write_pdf:
        pdf = output_stem.with_suffix(".pdf")
        figure.savefig(pdf, facecolor="white", bbox_inches="tight")
        outputs.append(pdf.name)
    plt.close(figure)
    return outputs


def assert_close(name: str, actual: float, expected: Any, atol: float = 2.0e-5) -> None:
    if expected is None or not np.isclose(actual, float(expected), rtol=0.0, atol=atol):
        raise ValueError(f"{name}: recomputed {actual} != scorer {expected}")


def load_protocol(
    protocol: str,
    methods: list[Method],
    ground_truth: dict[str, np.ndarray],
    ground_truth_path: Path,
) -> ProtocolData:
    gt_frames = np.asarray(ground_truth["frame_indices"], dtype=np.int32)
    gt_lookup = {int(frame): slot for slot, frame in enumerate(gt_frames)}
    expected_hash = sha256(ground_truth_path)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    errors_2d_px: dict[str, np.ndarray] = {}
    errors_3d_mm: dict[str, np.ndarray] = {}
    shared_scored_frames: np.ndarray | None = None

    for method in methods:
        capture = method.root / protocol
        metadata = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
        report = json.loads(
            (capture / "evaluation_results.json").read_text(encoding="utf-8")
        )
        if metadata["protocol"] != protocol or report["protocol"] != protocol:
            raise ValueError(f"Protocol mismatch in {capture}")
        if metadata["ground_truth_sha256"] != expected_hash:
            raise ValueError(f"Ground-truth hash mismatch in {capture}")
        scored_frames = np.asarray(
            report["point_tracking"]["scored_frames"], dtype=np.int32
        )
        if shared_scored_frames is None:
            shared_scored_frames = scored_frames
        elif not np.array_equal(scored_frames, shared_scored_frames):
            raise ValueError(f"Scored schedules differ in {capture}")

        prediction = read_npz(capture / "predicted_tracks.npz")
        pred_frames = np.asarray(prediction["frame_indices"], dtype=np.int32)
        if not np.array_equal(pred_frames, gt_frames):
            raise ValueError(f"Prediction and GT schedules differ in {capture}")
        pred_lookup = {int(frame): slot for slot, frame in enumerate(pred_frames)}
        pred_slots = np.asarray(
            [pred_lookup[int(frame)] for frame in scored_frames], dtype=np.int64
        )
        if np.any(np.asarray(prediction["observation_used"])[pred_slots]):
            raise ValueError(f"A scored frame used a visual observation in {capture}")
        gt_slots = np.asarray(
            [gt_lookup[int(frame)] for frame in scored_frames], dtype=np.int64
        )
        visible = np.asarray(ground_truth["visible"])[gt_slots].astype(bool)
        valid_3d = np.asarray(ground_truth["valid_3d"])[gt_slots].astype(bool)
        error_2d = np.linalg.norm(
            np.asarray(prediction["uv"])[pred_slots]
            - np.asarray(ground_truth["uv"])[gt_slots],
            axis=-1,
        )
        error_3d = 1000.0 * np.linalg.norm(
            np.asarray(prediction["xyz_camera_m"])[pred_slots]
            - np.asarray(ground_truth["xyz_camera_m"])[gt_slots],
            axis=-1,
        )
        error_2d[~visible] = np.nan
        error_3d[~valid_3d] = np.nan
        scorer = report["point_tracking"]
        assert_close(
            f"{method.key}/{protocol}/2D mean",
            finite_mean(error_2d),
            scorer["2d_error_px"]["mean"],
        )
        assert_close(
            f"{method.key}/{protocol}/2D RMSE",
            float(finite_rmse(error_2d)),
            scorer["2d_error_px"]["rmse"],
        )
        assert_close(
            f"{method.key}/{protocol}/3D mean",
            finite_mean(error_3d),
            scorer["3d_error_mm"]["mean"],
        )
        assert_close(
            f"{method.key}/{protocol}/3D RMSE",
            float(finite_rmse(error_3d)),
            scorer["3d_error_mm"]["rmse"],
        )
        predictions[method.key] = {
            **prediction,
            "scored_slots": pred_slots,
            "capture": np.asarray(str(capture)),
        }
        errors_2d_px[method.key] = error_2d
        errors_3d_mm[method.key] = error_3d

    if shared_scored_frames is None:
        raise ValueError(f"No methods loaded for {protocol}")
    gt_slots = np.asarray(
        [gt_lookup[int(frame)] for frame in shared_scored_frames], dtype=np.int64
    )
    return ProtocolData(
        name=protocol,
        scored_frames=shared_scored_frames,
        gt_slots=gt_slots,
        gt_uv=np.asarray(ground_truth["uv"])[gt_slots],
        gt_xyz_world_m=np.asarray(ground_truth["xyz_world_m"])[gt_slots],
        visible=np.asarray(ground_truth["visible"])[gt_slots].astype(bool),
        valid_3d=np.asarray(ground_truth["valid_3d"])[gt_slots].astype(bool),
        predictions=predictions,
        errors_2d_px=errors_2d_px,
        errors_3d_mm=errors_3d_mm,
    )


def crop_limits(points_uv: np.ndarray, image_wh: tuple[int, int]) -> tuple[float, ...]:
    finite = points_uv[np.isfinite(points_uv).all(axis=-1)]
    lower = finite.min(axis=0)
    upper = finite.max(axis=0)
    center = 0.5 * (lower + upper)
    width = max(float(upper[0] - lower[0]) + 90.0, 250.0)
    height = max(float(upper[1] - lower[1]) + 90.0, 190.0)
    aspect = 1.35
    if width / height < aspect:
        width = aspect * height
    else:
        height = width / aspect
    image_width, image_height = image_wh
    x0 = max(0.0, center[0] - 0.5 * width)
    x1 = min(float(image_width), center[0] + 0.5 * width)
    y0 = max(0.0, center[1] - 0.5 * height)
    y1 = min(float(image_height), center[1] + 0.5 * height)
    return x0, x1, y0, y1


def plot_2d_trajectories(
    data: ProtocolData,
    methods: list[Method],
    point_names: list[str],
    background: np.ndarray,
    output_dir: Path,
    dpi: int,
    write_pdf: bool,
) -> list[str]:
    height, width = background.shape[:2]
    figure, axes = plt.subplots(5, 2, figsize=(13.2, 22.0), constrained_layout=True)
    for point_slot, axis in enumerate(axes.flat):
        combined = [data.gt_uv[:, point_slot]]
        for method in methods:
            prediction = data.predictions[method.key]
            combined.append(
                np.asarray(prediction["uv"])[prediction["scored_slots"], point_slot]
            )
        combined_uv = np.concatenate(combined, axis=0)
        x0, x1, y0, y1 = crop_limits(combined_uv, (width, height))
        axis.imshow(background, extent=(0, width, height, 0), alpha=0.62)
        valid = data.visible[:, point_slot]
        gt_uv = data.gt_uv[:, point_slot]
        axis.plot(
            gt_uv[valid, 0], gt_uv[valid, 1], color="black", linewidth=2.4,
            marker="o", markersize=2.6, label="Ground truth", zorder=5,
        )
        for method in methods:
            prediction = data.predictions[method.key]
            pred_uv = np.asarray(prediction["uv"])[
                prediction["scored_slots"], point_slot
            ]
            axis.plot(
                pred_uv[valid, 0], pred_uv[valid, 1], color=method.color,
                linewidth=1.7, marker=".", markersize=3.1, label=method.label,
                zorder=4,
            )
        axis.scatter(
            [gt_uv[valid][0, 0]], [gt_uv[valid][0, 1]], marker="s", s=30,
            facecolor="white", edgecolor="black", linewidth=0.9, zorder=7,
        )
        axis.scatter(
            [gt_uv[valid][-1, 0]], [gt_uv[valid][-1, 1]], marker="X", s=38,
            facecolor="black", edgecolor="white", linewidth=0.7, zorder=7,
        )
        axis.set_xlim(x0, x1)
        axis.set_ylim(y1, y0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{point_names[point_slot]} — scored samples")
        axis.set_xlabel("u (full-resolution pixel)")
        axis.set_ylabel("v (full-resolution pixel)")
        axis.grid(color="white", alpha=0.18, linewidth=0.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
    figure.suptitle(
        "SUPER fixed 10-point 2D trajectories — stereo left\n"
        f"{protocol_title(data.name)}; square=start, X=end",
        fontsize=16,
    )
    return save_figure(
        figure, output_dir / "ten_point_2d_trajectories_stereo_left",
        dpi=dpi, write_pdf=write_pdf,
    )


def equal_3d_limits(arrays: list[np.ndarray]) -> tuple[np.ndarray, float]:
    finite = np.concatenate(arrays, axis=0)
    finite = finite[np.isfinite(finite).all(axis=-1)]
    lower = finite.min(axis=0)
    upper = finite.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 0.20) * 1.10
    return center, radius


def plot_3d_trajectories(
    data: ProtocolData,
    methods: list[Method],
    point_names: list[str],
    output_dir: Path,
    dpi: int,
    write_pdf: bool,
) -> list[str]:
    figure = plt.figure(figsize=(14.0, 24.0), constrained_layout=True)
    first_handles = None
    for point_slot in range(len(point_names)):
        axis = figure.add_subplot(5, 2, point_slot + 1, projection="3d")
        valid = data.valid_3d[:, point_slot]
        origin = data.gt_xyz_world_m[valid, point_slot][0]
        gt = 1000.0 * (data.gt_xyz_world_m[:, point_slot] - origin)
        trajectories = [gt[valid]]
        method_xyz = {}
        for method in methods:
            prediction = data.predictions[method.key]
            xyz = np.asarray(prediction["xyz_world_m"])[
                prediction["scored_slots"], point_slot
            ]
            relative = 1000.0 * (xyz - origin)
            method_xyz[method.key] = relative
            trajectories.append(relative[valid])
        center, radius = equal_3d_limits(trajectories)
        axis.plot(
            gt[valid, 0], gt[valid, 1], gt[valid, 2], color="black",
            linewidth=2.4, marker="o", markersize=2.4, label="Ground truth",
        )
        for method in methods:
            xyz = method_xyz[method.key]
            axis.plot(
                xyz[valid, 0], xyz[valid, 1], xyz[valid, 2],
                color=method.color, linewidth=1.7, marker=".", markersize=2.7,
                label=method.label,
            )
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_xlabel(r"$\Delta X_{world}$ (mm)")
        axis.set_ylabel(r"$\Delta Y_{world}$ (mm)")
        axis.set_zlabel(r"$\Delta Z_{world}$ (mm)")
        axis.set_title(f"{point_names[point_slot]} — relative to first scored GT")
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
        "SUPER fixed 10-point 3D trajectories\n"
        f"{protocol_title(data.name)}; world displacement in millimeters",
        fontsize=16,
    )
    return save_figure(
        figure, output_dir / "ten_point_3d_trajectories",
        dpi=dpi, write_pdf=write_pdf,
    )


def common_percentile(errors: dict[str, np.ndarray], percentile: float = 99.0) -> float:
    values = np.concatenate([value[np.isfinite(value)] for value in errors.values()])
    return max(float(np.percentile(values, percentile)), np.finfo(float).eps)


def plot_temporal_heatmaps(
    data: ProtocolData,
    methods: list[Method],
    point_names: list[str],
    errors: dict[str, np.ndarray],
    unit: str,
    dimension: str,
    output_dir: Path,
    dpi: int,
    write_pdf: bool,
) -> tuple[list[str], float]:
    upper = common_percentile(errors)
    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_bad("white")
    norm = Normalize(vmin=0.0, vmax=upper, clip=True)
    figure, axes = plt.subplots(
        len(methods), 1, figsize=(15.5, 9.0), sharex=True, constrained_layout=True
    )
    image = None
    for axis, method in zip(np.atleast_1d(axes), methods):
        image = axis.imshow(
            np.ma.masked_invalid(errors[method.key].T), aspect="auto",
            interpolation="nearest", cmap=cmap, norm=norm, origin="upper",
        )
        axis.set_yticks(np.arange(len(point_names)), labels=point_names)
        axis.set_ylabel("Manual point")
        axis.set_title(method.label, fontsize=11)
    tick_slots = np.unique(
        np.linspace(0, len(data.scored_frames) - 1, min(9, len(data.scored_frames))).round().astype(int)
    )
    axes[-1].set_xticks(tick_slots, labels=data.scored_frames[tick_slots])
    axes[-1].set_xlabel("Scored video frame (columns are annotated samples only)")
    assert image is not None
    colorbar = figure.colorbar(image, ax=np.atleast_1d(axes), shrink=0.88, pad=0.015)
    colorbar.set_label(f"Absolute {dimension} point error ({unit}); shared P99 clip")
    figure.suptitle(
        f"{protocol_title(data.name)} — {dimension} point-error temporal heatmaps\n"
        f"shared scale 0–{upper:.2f} {unit}; white=invalid; color clipped only for display",
        fontsize=15,
    )
    files = save_figure(
        figure, output_dir / f"{dimension.lower()}_error_temporal_heatmaps",
        dpi=dpi, write_pdf=write_pdf,
    )
    return files, upper


def plot_by_point_heatmaps(
    data: ProtocolData,
    methods: list[Method],
    point_names: list[str],
    output_dir: Path,
    dpi: int,
    write_pdf: bool,
) -> list[str]:
    matrices = (
        ("2D RMSE (px)", np.stack([finite_rmse(data.errors_2d_px[m.key], axis=0) for m in methods])),
        ("3D RMSE (mm)", np.stack([finite_rmse(data.errors_3d_mm[m.key], axis=0) for m in methods])),
    )
    figure, axes = plt.subplots(2, 1, figsize=(15.5, 6.8), constrained_layout=True)
    for axis, (title, matrix) in zip(axes, matrices):
        image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        axis.set_xticks(np.arange(len(point_names)), labels=point_names)
        axis.set_yticks(np.arange(len(methods)), labels=[m.label for m in methods])
        axis.set_title(title)
        threshold = 0.56 * float(np.nanmax(matrix))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(
                    column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=8.5, color="white" if value > threshold else "black",
                )
        colorbar = figure.colorbar(image, ax=axis, shrink=0.86, pad=0.015)
        colorbar.set_label(title)
    figure.suptitle(
        f"{protocol_title(data.name)} — per-point RMSE (exact formal scored frames)",
        fontsize=15,
    )
    return save_figure(
        figure, output_dir / "error_by_point_heatmaps",
        dpi=dpi, write_pdf=write_pdf,
    )


def plot_spatial_point_heatmap(
    data: ProtocolData,
    methods: list[Method],
    point_names: list[str],
    background: np.ndarray,
    errors: dict[str, np.ndarray],
    unit: str,
    dimension: str,
    output_dir: Path,
    dpi: int,
    write_pdf: bool,
) -> list[str]:
    height, width = background.shape[:2]
    positions = np.nanmedian(data.gt_uv, axis=0)
    per_method = {
        method.key: np.asarray(finite_rmse(errors[method.key], axis=0))
        for method in methods
    }
    upper = max(float(np.max(value)) for value in per_method.values())
    norm = Normalize(vmin=0.0, vmax=upper)
    figure, axes = plt.subplots(1, len(methods), figsize=(18.0, 5.5), constrained_layout=True)
    scatter = None
    for axis, method in zip(np.atleast_1d(axes), methods):
        axis.imshow(background, extent=(0, width, height, 0), alpha=0.70)
        values = per_method[method.key]
        scatter = axis.scatter(
            positions[:, 0], positions[:, 1], c=values, cmap="turbo", norm=norm,
            s=230, marker="o", edgecolor="white", linewidth=1.2,
        )
        for point, (u, v), value in zip(point_names, positions, values):
            axis.annotate(
                f"{point}\n{value:.2f}", (u, v), xytext=(0, 0),
                textcoords="offset points", ha="center", va="center",
                fontsize=7.4, fontweight="bold", color="black",
            )
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_aspect("equal")
        axis.set_title(method.label)
        axis.set_xlabel("u (pixel)")
        axis.set_ylabel("v (pixel)")
    assert scatter is not None
    colorbar = figure.colorbar(scatter, ax=np.atleast_1d(axes), shrink=0.78, pad=0.01)
    colorbar.set_label(f"Per-point {dimension} RMSE ({unit})")
    figure.suptitle(
        f"{protocol_title(data.name)} — sparse {dimension} error map\n"
        "markers are the 10 manually annotated tissue tracks; no dense interpolation",
        fontsize=15,
    )
    return save_figure(
        figure, output_dir / f"{dimension.lower()}_error_spatial_point_maps",
        dpi=dpi, write_pdf=write_pdf,
    )


def write_error_csv(
    path: Path,
    data: ProtocolData,
    methods: list[Method],
    point_names: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            ["protocol", "method", "frame_index", "point", "error_2d_px", "error_3d_mm"]
        )
        for method in methods:
            for frame_slot, frame in enumerate(data.scored_frames):
                for point_slot, point_name in enumerate(point_names):
                    writer.writerow(
                        [
                            data.name,
                            method.key,
                            int(frame),
                            point_name,
                            float(data.errors_2d_px[method.key][frame_slot, point_slot]),
                            float(data.errors_3d_mm[method.key][frame_slot, point_slot]),
                        ]
                    )


def main() -> None:
    args = parse_args()
    ground_truth_path = args.ground_truth.resolve()
    annotations_path = args.annotations.resolve()
    background_path = args.background.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    methods = [
        Method(
            "pure_pbd", "Pure PBD", "#e53935",
            args.baseline_root.resolve() / "pure_pbd",
        ),
        Method(
            "visual_residual", "PBD + RGB residual", "#1e88e5",
            args.baseline_root.resolve() / "pbd_visual_residual",
        ),
        Method(
            "visual_stiffness", "PBD + RGB residual + stiffness", "#8e44ad",
            args.stiffness_root.resolve() / "pbd_visual_residual_online_stiffness",
        ),
    ]
    ground_truth = read_npz(ground_truth_path)
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    point_names = [point["name"] for point in annotations["points"]]
    expected_ids = np.asarray([point["point_id"] for point in annotations["points"]])
    if not np.array_equal(expected_ids, ground_truth["point_ids"]):
        raise ValueError("Annotation and frozen-GT point ordering differs")
    background = mpimg.imread(background_path)
    expected_wh = tuple(int(value) for value in annotations["image_size_wh"])
    if background.shape[1::-1] != expected_wh:
        raise ValueError(
            f"Background is {background.shape[1::-1]}, expected {expected_wh}"
        )

    manifest: dict[str, Any] = {
        "schema": "super_ten_point_evaluation_visualization_v1",
        "ground_truth": str(ground_truth_path),
        "ground_truth_sha256": sha256(ground_truth_path),
        "annotations": str(annotations_path),
        "background": str(background_path),
        "methods": [
            {"key": method.key, "label": method.label, "root": str(method.root)}
            for method in methods
        ],
        "integrity": {
            "formal_scorer_frame_lists_reused": True,
            "scored_observations_withheld_verified": True,
            "absolute_errors_no_rigid_alignment": True,
            "spatial_interpolation": False,
            "temporal_heatmap_columns": "manual annotation samples only",
        },
        "protocols": {},
    }
    readme = [
        "# SUPER 十点重建/未来预测可视化",
        "",
        "颜色与 sim 参考产物一致：黑色是真值，红色是 Pure PBD，蓝色是视觉残差，紫色是视觉残差+在线刚度。",
        "2D/3D 误差严格按正式 scorer 的绝对误差计算，不做逐帧刚体对齐；所有正式评分帧都已验证未使用视觉观测。",
        "热图列只代表人工标注时刻，空间图只画 10 个固定点，不将稀疏真值插值成稠密组织表面。",
        "",
    ]
    for protocol in PROTOCOLS:
        data = load_protocol(protocol, methods, ground_truth, ground_truth_path)
        protocol_dir = output_root / protocol
        protocol_dir.mkdir(parents=True, exist_ok=True)
        figures: list[str] = []
        figures.extend(
            plot_2d_trajectories(
                data, methods, point_names, background, protocol_dir,
                args.dpi, not args.no_pdf,
            )
        )
        figures.extend(
            plot_3d_trajectories(
                data, methods, point_names, protocol_dir,
                args.dpi, not args.no_pdf,
            )
        )
        temporal_2d, p99_2d = plot_temporal_heatmaps(
            data, methods, point_names, data.errors_2d_px, "px", "2D",
            protocol_dir, args.dpi, not args.no_pdf,
        )
        temporal_3d, p99_3d = plot_temporal_heatmaps(
            data, methods, point_names, data.errors_3d_mm, "mm", "3D",
            protocol_dir, args.dpi, not args.no_pdf,
        )
        figures.extend(temporal_2d)
        figures.extend(temporal_3d)
        figures.extend(
            plot_by_point_heatmaps(
                data, methods, point_names, protocol_dir,
                args.dpi, not args.no_pdf,
            )
        )
        figures.extend(
            plot_spatial_point_heatmap(
                data, methods, point_names, background, data.errors_2d_px,
                "px", "2D", protocol_dir, args.dpi, not args.no_pdf,
            )
        )
        figures.extend(
            plot_spatial_point_heatmap(
                data, methods, point_names, background, data.errors_3d_mm,
                "mm", "3D", protocol_dir, args.dpi, not args.no_pdf,
            )
        )
        write_error_csv(protocol_dir / "point_errors.csv", data, methods, point_names)
        method_summary = {}
        for method in methods:
            method_summary[method.key] = {
                "2d_mean_px": finite_mean(data.errors_2d_px[method.key]),
                "2d_rmse_px": float(finite_rmse(data.errors_2d_px[method.key])),
                "3d_mean_mm": finite_mean(data.errors_3d_mm[method.key]),
                "3d_rmse_mm": float(finite_rmse(data.errors_3d_mm[method.key])),
            }
        manifest["protocols"][protocol] = {
            "title": protocol_title(protocol),
            "scored_frame_count": int(len(data.scored_frames)),
            "scored_frames": data.scored_frames.astype(int).tolist(),
            "temporal_display_p99": {"2d_px": p99_2d, "3d_mm": p99_3d},
            "summary": method_summary,
            "figures": figures,
            "point_error_csv": "point_errors.csv",
        }
        readme.extend(
            [
                f"## {protocol}",
                "",
                f"正式轨迹评分时刻：{len(data.scored_frames)} 个。",
                "",
                "- [2D 轨迹](%s/ten_point_2d_trajectories_stereo_left.png)" % protocol,
                "- [3D 轨迹](%s/ten_point_3d_trajectories.png)" % protocol,
                "- [2D 逐时刻误差热图](%s/2d_error_temporal_heatmaps.png)" % protocol,
                "- [3D 逐时刻误差热图](%s/3d_error_temporal_heatmaps.png)" % protocol,
                "- [逐点 RMSE 热图](%s/error_by_point_heatmaps.png)" % protocol,
                "- [2D 稀疏空间误差图](%s/2d_error_spatial_point_maps.png)" % protocol,
                "- [3D 稀疏空间误差图](%s/3d_error_spatial_point_maps.png)" % protocol,
                "- [逐帧逐点数值](%s/point_errors.csv)" % protocol,
                "",
            ]
        )
    (output_root / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
