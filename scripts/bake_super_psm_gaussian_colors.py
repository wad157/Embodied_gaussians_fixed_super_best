#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = REPO_ROOT / "data/super/psm_robot/psm_surface_gaussians.npz"
DEFAULT_DRIVER = REPO_ROOT / "data/super/psm_robot/psm_lnd_pose_driver.npz"
DEFAULT_CALIB = REPO_ROOT / "data/super/grasp5_native/calib_rectified.json"
DEFAULT_RGB_DIR = REPO_ROOT / "data/super/grasp5_native/rgb"
DEFAULT_VIDEO_METADATA = (
    REPO_ROOT / "data/super/grasp5_offline_demo/videos/stereo_left.json"
)
DEFAULT_REPORT = REPO_ROOT / "data/super/psm_robot/psm_surface_gaussians_report.json"
DEFAULT_PREVIEW = REPO_ROOT / "data/super/psm_robot/psm_color_bake_frame0_overlay.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bake multiframe SUPER left-camera colors onto PSM surface Gaussians."
    )
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--video-metadata", type=Path, default=DEFAULT_VIDEO_METADATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0, 100, 200, 300, 400, 500, 650, 800, 950, 1100, 1250, 1400],
    )
    parser.add_argument("--warm-background-threshold", type=float, default=0.10)
    parser.add_argument("--minimum-value", type=float, default=0.25)
    parser.add_argument("--maximum-value", type=float, default=0.98)
    parser.add_argument("--sample-blend", type=float, default=0.90)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nearest_indices(sorted_values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    right = np.searchsorted(sorted_values, queries, side="left")
    right = np.clip(right, 0, len(sorted_values) - 1)
    left = np.clip(right - 1, 0, len(sorted_values) - 1)
    choose_left = np.abs(queries - sorted_values[left]) <= np.abs(
        sorted_values[right] - queries
    )
    return np.where(choose_left, left, right)


def project_points(points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    u = np.rint(points[:, 0] * k[0, 0] / z + k[0, 2]).astype(np.int32)
    v = np.rint(points[:, 1] * k[1, 1] / z + k[1, 2]).astype(np.int32)
    return u, v


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.sample_blend <= 1.0:
        raise ValueError("--sample-blend must be in [0, 1]")

    with np.load(args.asset, allow_pickle=False) as source:
        asset = {name: source[name].copy() for name in source.files}
    with np.load(args.driver, allow_pickle=False) as source:
        driver_timestamps = source["timestamps"].astype(np.float64)
        driver_link_names = source["link_names"].tolist()
        poses = source["poses_rect_camera_xyz_xyzw"].astype(np.float64)

    means = asset["means"].astype(np.float64)
    link_ids = asset["link_ids"].astype(np.int64)
    asset_link_names = asset["link_names"].tolist()
    if asset_link_names != driver_link_names:
        raise ValueError("PSM Gaussian and pose-driver link orders do not match")

    calibration = read_json(args.calib)
    k = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    video_metadata = read_json(args.video_metadata)
    video_timestamps = np.asarray(video_metadata["timestamps"], dtype=np.float64)
    width, height = map(int, video_metadata["resolution"])

    frame_ids = np.asarray(sorted(set(args.frames)), dtype=np.int64)
    if np.any(frame_ids < 0) or np.any(frame_ids >= len(video_timestamps)):
        raise IndexError(f"Frame ids must lie in [0, {len(video_timestamps) - 1}]")
    state_ids = nearest_indices(driver_timestamps, video_timestamps[frame_ids])

    samples = np.full((len(frame_ids), len(means), 3), np.nan, dtype=np.float32)
    visible_counts = []

    for sample_index, (frame_id, state_id) in enumerate(zip(frame_ids, state_ids)):
        image_bgr = cv2.imread(
            str(args.rgb_dir / f"{int(frame_id):06d}-left.png"), cv2.IMREAD_COLOR
        )
        if image_bgr is None:
            raise FileNotFoundError(args.rgb_dir / f"{int(frame_id):06d}-left.png")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        points_camera = np.empty_like(means)
        for link_id in range(len(asset_link_names)):
            mask = link_ids == link_id
            pose = poses[state_id, link_id]
            rotation = Rotation.from_quat(pose[3:]).as_matrix()
            points_camera[mask] = means[mask] @ rotation.T + pose[:3]

        u, v = project_points(points_camera, k)
        valid = (
            (points_camera[:, 2] > 0.0)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        ids = np.flatnonzero(valid)
        colors = image_rgb[v[ids], u[ids]]
        value = colors.max(axis=1)
        # Tissue is predominantly warm. Reject strongly warm samples while
        # retaining neutral steel and blue instrument markings.
        not_warm_background = colors[:, 0] - colors[:, 2] < args.warm_background_threshold
        plausible = (
            (value >= args.minimum_value)
            & (value <= args.maximum_value)
            & not_warm_background
        )
        ids = ids[plausible]
        samples[sample_index, ids] = colors[plausible]
        visible_counts.append(int(len(ids)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        baked = np.nanmedian(samples, axis=0)
    observed = np.all(np.isfinite(baked), axis=1)
    if not observed.any():
        raise RuntimeError("No valid PSM color samples were found")

    default_color = np.asarray([0.65, 0.67, 0.70], dtype=np.float32)
    propagated = baked.copy()
    for link_id in range(len(asset_link_names)):
        link_mask = link_ids == link_id
        observed_link = link_mask & observed
        missing_link = link_mask & ~observed
        if observed_link.any() and missing_link.any():
            tree = cKDTree(means[observed_link])
            _, nearest = tree.query(means[missing_link], k=1)
            propagated[missing_link] = baked[observed_link][nearest]
        elif missing_link.any():
            propagated[missing_link] = default_color

    luminance = propagated @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    neutral_metal = luminance[:, None] * np.asarray(
        [0.97, 1.00, 1.03], dtype=np.float32
    )[None, :]
    blue_marking = (
        (propagated[:, 2] - propagated[:, 0] > 0.12)
        & (propagated[:, 2] > 0.35)
    )
    material_colors = neutral_metal
    material_colors[blue_marking] = propagated[blue_marking]
    colors = (
        args.sample_blend * material_colors
        + (1.0 - args.sample_blend) * default_color[None, :]
    )
    colors = np.clip(colors, 0.03, 0.97).astype(np.float32)
    asset["colors"] = colors
    np.savez_compressed(args.asset, **asset)

    preview_frame = int(frame_ids[0])
    preview_state = int(state_ids[0])
    preview = cv2.imread(
        str(args.rgb_dir / f"{preview_frame:06d}-left.png"), cv2.IMREAD_COLOR
    )
    if preview is None:
        raise FileNotFoundError(args.rgb_dir / f"{preview_frame:06d}-left.png")
    points_camera = np.empty_like(means)
    for link_id in range(len(asset_link_names)):
        mask = link_ids == link_id
        pose = poses[preview_state, link_id]
        rotation = Rotation.from_quat(pose[3:]).as_matrix()
        points_camera[mask] = means[mask] @ rotation.T + pose[:3]
    u, v = project_points(points_camera, k)
    valid_preview = (
        (points_camera[:, 2] > 0.0)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    for gaussian_id in np.flatnonzero(valid_preview):
        color_bgr = tuple(
            int(channel)
            for channel in np.rint(colors[gaussian_id, ::-1] * 255.0)
        )
        cv2.circle(
            preview,
            (int(u[gaussian_id]), int(v[gaussian_id])),
            2,
            color_bgr,
            -1,
            lineType=cv2.LINE_AA,
        )
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.preview), preview)

    report = read_json(args.report) if args.report.exists() else {}
    report["color_baking"] = {
        "method": "multiframe_strict_lnd_projection_median",
        "frames": frame_ids.tolist(),
        "joint_state_indices": state_ids.tolist(),
        "valid_samples_per_frame": visible_counts,
        "directly_observed_gaussians": int(observed.sum()),
        "propagated_gaussians": int((~observed).sum()),
        "blue_marking_gaussians": int(blue_marking.sum()),
        "rgb_mean": colors.mean(axis=0).tolist(),
        "rgb_std": colors.std(axis=0).tolist(),
        "rgb_percentiles": np.percentile(colors, [0, 5, 50, 95, 100], axis=0).tolist(),
        "warm_background_threshold": args.warm_background_threshold,
        "sample_blend": args.sample_blend,
        "preview": str(args.preview),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["color_baking"], indent=2))


if __name__ == "__main__":
    main()
