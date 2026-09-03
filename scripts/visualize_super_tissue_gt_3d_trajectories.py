#!/usr/bin/env python3
"""Render one standalone 3D ground-truth trajectory figure per SUPER point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = (
    REPO_ROOT
    / "data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/super_tissue_gt_3d_trajectories_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def equal_axis_limits(relative_xyz_mm: np.ndarray) -> tuple[np.ndarray, float]:
    finite = relative_xyz_mm[np.isfinite(relative_xyz_mm).all(axis=-1)]
    lower = finite.min(axis=0)
    upper = finite.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 0.25)
    return center, radius * 1.10


def trajectory_statistics(
    xyz_world_m: np.ndarray,
    valid: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, object]:
    xyz = xyz_world_m[valid]
    frames = frame_indices[valid]
    displacement_mm = (xyz - xyz[0]) * 1000.0
    steps_mm = np.linalg.norm(np.diff(displacement_mm, axis=0), axis=1)
    radial_mm = np.linalg.norm(displacement_mm, axis=1)
    return {
        "valid_samples": int(len(xyz)),
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "start_world_m": xyz[0].astype(float).tolist(),
        "end_world_m": xyz[-1].astype(float).tolist(),
        "net_displacement_mm": float(radial_mm[-1]),
        "maximum_displacement_from_start_mm": float(radial_mm.max()),
        "sampled_polyline_length_mm": float(steps_mm.sum()),
        "relative_minimum_mm": displacement_mm.min(axis=0).astype(float).tolist(),
        "relative_maximum_mm": displacement_mm.max(axis=0).astype(float).tolist(),
    }


def render_point(
    *,
    output: Path,
    point_id: int,
    xyz_world_m: np.ndarray,
    valid: np.ndarray,
    frame_indices: np.ndarray,
    future_test_start: int,
    dpi: int,
    stats: dict[str, object],
) -> None:
    xyz = xyz_world_m[valid]
    frames = frame_indices[valid]
    relative = (xyz - xyz[0]) * 1000.0
    axis_center, axis_radius = equal_axis_limits(relative)
    segments = np.stack((relative[:-1], relative[1:]), axis=1)
    segment_frames = 0.5 * (frames[:-1] + frames[1:])
    normalization = Normalize(
        vmin=float(frame_indices.min()), vmax=float(frame_indices.max())
    )
    colors = colormaps["viridis"](normalization(segment_frames))

    figure = plt.figure(figsize=(8.4, 7.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    collection = Line3DCollection(segments, colors=colors, linewidths=2.4)
    axis.add_collection3d(collection)
    axis.scatter(
        *relative[0], s=90, marker="o", color="#19a974", edgecolor="black",
        linewidth=0.8, label=f"Start (frame {int(frames[0])})", zorder=5,
    )
    split_slot = int(np.searchsorted(frames, future_test_start, side="left"))
    split_slot = min(max(split_slot, 0), len(frames) - 1)
    axis.scatter(
        *relative[split_slot], s=95, marker="D", color="#8e44ad",
        edgecolor="black", linewidth=0.8,
        label=f"80/20 split (frame {int(frames[split_slot])})", zorder=6,
    )
    axis.scatter(
        *relative[-1], s=110, marker="X", color="#e74c3c", edgecolor="black",
        linewidth=0.8, label=f"End (frame {int(frames[-1])})", zorder=7,
    )

    axis.set_xlim(axis_center[0] - axis_radius, axis_center[0] + axis_radius)
    axis.set_ylim(axis_center[1] - axis_radius, axis_center[1] + axis_radius)
    axis.set_zlim(axis_center[2] - axis_radius, axis_center[2] + axis_radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel(r"$\Delta X_{world}$ (mm)")
    axis.set_ylabel(r"$\Delta Y_{world}$ (mm)")
    axis.set_zlabel(r"$\Delta Z_{world}$ (mm)")
    axis.view_init(elev=24, azim=-56)
    axis.grid(True, alpha=0.35)
    start = np.asarray(stats["start_world_m"], dtype=float)
    axis.set_title(
        f"SUPER point {point_id}: 3D ground-truth motion\n"
        f"start world xyz = ({start[0]:.5f}, {start[1]:.5f}, {start[2]:.5f}) m",
        pad=18,
    )
    axis.legend(loc="upper left", fontsize=8)
    scalar = matplotlib.cm.ScalarMappable(norm=normalization, cmap="viridis")
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axis, shrink=0.68, pad=0.09)
    colorbar.set_label("Video frame index")
    axis.text2D(
        0.02,
        0.02,
        "samples={valid_samples}   net={net_displacement_mm:.3f} mm   "
        "max={maximum_displacement_from_start_mm:.3f} mm   "
        "path={sampled_polyline_length_mm:.3f} mm".format(**stats),
        transform=axis.transAxes,
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.75"},
    )
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ground_truth = args.ground_truth.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(ground_truth, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int32)
        point_ids = np.asarray(archive["point_ids"], dtype=np.int32)
        xyz_world_m = np.asarray(archive["xyz_world_m"], dtype=np.float64)
        valid_3d = np.asarray(archive["valid_3d"], dtype=bool)
        future_test_start = int(archive["future_test_start_frame"])
        schema = str(archive["schema"])

    points: list[dict[str, object]] = []
    for slot, point_id_value in enumerate(point_ids):
        point_id = int(point_id_value)
        valid = valid_3d[:, slot]
        if int(valid.sum()) < 2:
            raise ValueError(f"Point {point_id} has fewer than two valid 3D samples")
        stats = trajectory_statistics(
            xyz_world_m[:, slot], valid, frame_indices
        )
        filename = f"point_{point_id:02d}_gt_3d_trajectory.png"
        render_point(
            output=output_dir / filename,
            point_id=point_id,
            xyz_world_m=xyz_world_m[:, slot],
            valid=valid,
            frame_indices=frame_indices,
            future_test_start=future_test_start,
            dpi=args.dpi,
            stats=stats,
        )
        points.append({"point_id": point_id, "figure": filename, **stats})

    report = {
        "schema": "super_tissue_gt_3d_trajectory_visualization_v1",
        "ground_truth_schema": schema,
        "ground_truth": str(ground_truth),
        "coordinate_system": "world displacement relative to each point first valid sample",
        "coordinate_unit": "millimeter",
        "axis_scaling": "per-point adaptive range; equal XYZ metric scale within each figure",
        "future_test_start_frame": future_test_start,
        "point_count": len(points),
        "points": points,
    }
    (output_dir / "trajectory_statistics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SUPER 十点 3D 真值运动轨迹",
        "",
        "每个点对应一张独立图片。坐标是相对该点首个有效样本的世界坐标位移，单位 mm；",
        "每张图按该点范围自适应缩放，但图内 XYZ 保持相同毫米比例。绿色圆点为起点，紫色菱形为 80/20 分界，",
        "红色叉号为终点，线条颜色表示视频帧编号。",
        "",
    ]
    for point in points:
        lines.append(
            f"- Point {point['point_id']}: [{point['figure']}]({point['figure']}) — "
            f"net {point['net_displacement_mm']:.3f} mm, "
            f"max {point['maximum_displacement_from_start_mm']:.3f} mm, "
            f"path {point['sampled_polyline_length_mm']:.3f} mm"
        )
    (output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
