#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import warp as wp
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_gaussians.scene_builders.domain import (  # noqa: E402
    Body,
    Gaussians,
    GaussianLearningRates,
    Ground,
    MaskedPosedImageAndDepth,
    Particles,
)
from embodied_gaussians.scene_builders.simple_body_builder import SimpleBodyBuilder  # noqa: E402
from embodied_gaussians.scene_builders.warp_utils import find_distant_query_points  # noqa: E402


X_WC_BLENDER_FOR_OPENCV_WORLD = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SuPer tissue/ground bodies from first-frame masks and depth."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "rgb" / "000000-left.png",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "depth" / "000000-depth.npy",
    )
    parser.add_argument(
        "--tissue-mask",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "masks" / "000000-tissue.png",
    )
    parser.add_argument(
        "--ground-mask",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "masks" / "000000-ground.png",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "calib_rectified.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "bodies",
    )
    parser.add_argument("--particle-radius", type=float, default=0.003)
    parser.add_argument("--max-particles", type=int, default=2500)
    parser.add_argument("--max-tissue-gaussians", type=int, default=2000)
    parser.add_argument("--max-ground-gaussians", type=int, default=2500)
    parser.add_argument("--voxel-size", type=float, default=0.0015)
    parser.add_argument("--gaussian-iters", type=int, default=600)
    parser.add_argument("--max-depth", type=float, default=0.25)
    parser.add_argument(
        "--tissue-thickness",
        type=float,
        default=0.015,
        help="Maximum tissue height above the fitted plane in meters.",
    )
    parser.add_argument(
        "--minimum-tissue-thickness",
        type=float,
        default=0.003,
        help="Minimum filled tissue height in meters.",
    )
    parser.add_argument(
        "--height-smoothing-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma for the tissue height field, in particle-grid cells.",
    )
    parser.add_argument("--plane-fit-subsample", type=int, default=8)
    parser.add_argument("--plane-fit-huber-k", type=float, default=1.5)
    parser.add_argument("--plane-mask-erode-px", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-gaussian-training", action="store_true")
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(args.depth).astype(np.float32)
    tissue_mask = cv2.imread(str(args.tissue_mask), cv2.IMREAD_GRAYSCALE)
    ground_mask = cv2.imread(str(args.ground_mask), cv2.IMREAD_GRAYSCALE)
    if tissue_mask is None:
        raise FileNotFoundError(args.tissue_mask)
    if ground_mask is None:
        raise FileNotFoundError(args.ground_mask)
    tissue_mask = tissue_mask > 0
    ground_mask = ground_mask > 0

    calib = json.loads(args.calib.read_text())
    k = np.asarray(calib["K_left_rect"], dtype=np.float32)
    return image_rgb, depth, tissue_mask, ground_mask, k


def backproject_mask(depth: np.ndarray, mask: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(depth) & (depth > 0)
    vs, us = np.nonzero(valid)
    z = depth[vs, us]
    x = (us.astype(np.float32) - k[0, 2]) * z / k[0, 0]
    y = (vs.astype(np.float32) - k[1, 2]) * z / k[1, 1]
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    pixels = np.stack([us, vs], axis=1).astype(np.int32)
    return points, pixels


def sample_indices(num_items: int, max_items: int, rng: np.random.Generator) -> np.ndarray:
    if num_items <= max_items:
        return np.arange(num_items)
    return rng.choice(num_items, size=max_items, replace=False)


def downsample_points(points: np.ndarray, pixels: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, pixels
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    down_points = np.asarray(down.points, dtype=np.float32)
    if len(down_points) == 0:
        return points, pixels

    # Recover source pixels by nearest neighbor from the original point cloud.
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(source)
    recovered_pixels = []
    for point in down_points:
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        recovered_pixels.append(pixels[idx[0]])
    return down_points, np.asarray(recovered_pixels, dtype=np.int32)


def fit_ground_plane(
    ground_points: np.ndarray,
    ground_pixels: np.ndarray,
    tissue_points: np.ndarray,
    k: np.ndarray,
    subsample: int,
    huber_k: float,
) -> tuple[np.ndarray, dict]:
    """Fit a plane deterministically using robust inverse-depth regression.

    For a pinhole camera, inverse depth on a 3D plane is affine in the
    normalized image ray coordinates. Fitting in this space avoids the strong
    near-point bias of a fixed 3D RANSAC distance threshold.
    """
    if len(ground_points) < 3:
        raise RuntimeError("At least three ground points are required to fit a plane.")

    step = max(int(subsample), 1)
    points = ground_points[::step].astype(np.float64)
    pixels = ground_pixels[::step].astype(np.float64)
    ray_x = (pixels[:, 0] - float(k[0, 2])) / float(k[0, 0])
    ray_y = (pixels[:, 1] - float(k[1, 2])) / float(k[1, 1])
    design = np.stack([ray_x, ray_y, np.ones_like(ray_x)], axis=1)
    inv_depth = 1.0 / points[:, 2]

    beta = np.linalg.lstsq(design, inv_depth, rcond=None)[0]
    iterations = 0
    robust_scale = 0.0
    for iterations in range(1, 26):
        residual = inv_depth - design @ beta
        residual_center = float(np.median(residual))
        robust_scale = float(1.4826 * np.median(np.abs(residual - residual_center)))
        robust_scale = max(robust_scale, 1e-9)
        cutoff = max(float(huber_k), 0.1) * robust_scale
        centered_abs = np.abs(residual - residual_center)
        weights = np.ones_like(centered_abs)
        outliers = centered_abs > cutoff
        weights[outliers] = cutoff / centered_abs[outliers]
        sqrt_weights = np.sqrt(weights)
        updated = np.linalg.lstsq(
            design * sqrt_weights[:, None], inv_depth * sqrt_weights, rcond=None
        )[0]
        if np.linalg.norm(updated - beta) < 1e-10:
            beta = updated
            break
        beta = updated

    # beta[0] * x/z + beta[1] * y/z + beta[2] - 1/z = 0
    plane = np.asarray([beta[0], beta[1], beta[2], -1.0], dtype=np.float64)
    norm = np.linalg.norm(plane[:3])
    if norm < 1e-8:
        raise RuntimeError("Invalid fitted ground plane normal.")
    plane = plane / norm

    # Orient the plane so that tissue lies on the positive side; this matches
    # SimpleBodyBuilder._prune_points_below_ground.
    tissue_side = tissue_points @ plane[:3] + plane[3]
    if np.nanmedian(tissue_side) < 0:
        plane = -plane

    ground_distances = ground_points @ plane[:3] + plane[3]
    diagnostics = {
        "method": "inverse_depth_irls",
        "sampled_points": int(len(points)),
        "iterations": int(iterations),
        "huber_k": float(huber_k),
        "inverse_depth_robust_scale": robust_scale,
        "ground_signed_distance_mm_percentiles": np.percentile(
            ground_distances * 1000.0, [0, 1, 5, 50, 95, 99, 100]
        ).tolist(),
        "ground_absolute_distance_mm_percentiles": np.percentile(
            np.abs(ground_distances) * 1000.0, [50, 90, 95, 99]
        ).tolist(),
    }
    return plane.astype(np.float32), diagnostics


def project_points(points: np.ndarray, k: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    valid_z = z > 1e-8
    u = np.zeros((len(points),), dtype=np.int32)
    v = np.zeros((len(points),), dtype=np.int32)
    u[valid_z] = np.round(points[valid_z, 0] * k[0, 0] / z[valid_z] + k[0, 2]).astype(np.int32)
    v[valid_z] = np.round(points[valid_z, 1] * k[1, 1] / z[valid_z] + k[1, 2]).astype(np.int32)
    valid = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.stack([u, v], axis=1), valid


def make_plane_frame(points: np.ndarray, plane: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal = plane[:3].astype(np.float32)
    normal /= np.linalg.norm(normal)
    signed = points @ normal + plane[3]
    projected = points - signed[:, None] * normal[None, :]
    origin = projected.mean(axis=0)

    centered = projected - origin
    covariance = centered.T @ centered / max(len(centered), 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    axis_x = eigenvectors[:, -1].astype(np.float32)
    axis_x = axis_x - np.dot(axis_x, normal) * normal
    if np.linalg.norm(axis_x) < 1e-6:
        axis_x = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if np.linalg.norm(axis_x) < 1e-6:
            axis_x = np.cross(normal, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x).astype(np.float32)
    axis_y /= np.linalg.norm(axis_y)
    return origin.astype(np.float32), axis_x, axis_y, normal


def local_plane_coordinates(points: np.ndarray, origin: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray) -> np.ndarray:
    centered = points - origin
    return np.stack([centered @ axis_x, centered @ axis_y], axis=1).astype(np.float32)


def fill_depth_shaped_particles(
    surface_points: np.ndarray,
    mask: np.ndarray,
    plane: np.ndarray,
    k: np.ndarray,
    image_rgb: np.ndarray,
    radius: float,
    maximum_height: float,
    minimum_height: float,
    smoothing_sigma: float,
    max_particles: int,
    rng: np.random.Generator,
) -> tuple[Particles, dict]:
    origin, axis_x, axis_y, normal = make_plane_frame(surface_points, plane)
    surface_heights = surface_points @ normal + plane[3]
    valid_surface = np.isfinite(surface_heights) & (surface_heights > -radius)
    if valid_surface.sum() < 3:
        raise RuntimeError("Not enough valid tissue surface points for height-field filling.")
    surface_points = surface_points[valid_surface]
    surface_heights = surface_heights[valid_surface]
    surface_heights = np.clip(surface_heights, minimum_height, maximum_height)
    projected = surface_points - (
        surface_points @ normal + plane[3]
    )[:, None] * normal[None, :]
    footprint = local_plane_coordinates(projected, origin, axis_x, axis_y)

    min_xy = np.percentile(footprint, 1.0, axis=0)
    max_xy = np.percentile(footprint, 99.0, axis=0)
    step = radius * 2.0
    xs = np.arange(min_xy[0], max_xy[0] + step * 0.5, step, dtype=np.float32)
    ys = np.arange(min_xy[1], max_xy[1] + step * 0.5, step, dtype=np.float32)
    grid_xy = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    height_tree = cKDTree(footprint)
    neighbor_count = min(8, len(footprint))
    distances, indices = height_tree.query(grid_xy, k=neighbor_count)
    if neighbor_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, step * 0.25)
    grid_heights = np.sum(weights * surface_heights[indices], axis=1) / np.sum(weights, axis=1)

    base_points = (
        origin[None, :]
        + grid_xy[:, 0:1] * axis_x[None, :]
        + grid_xy[:, 1:2] * axis_y[None, :]
    )
    top_points = base_points + grid_heights[:, None] * normal[None, :]
    height, width = mask.shape
    uv, valid = project_points(top_points, k, width, height)
    in_mask = np.zeros((len(grid_xy),), dtype=bool)
    valid_ids = np.where(valid)[0]
    in_mask[valid_ids] = mask[uv[valid_ids, 1], uv[valid_ids, 0]]
    close_to_surface = distances[:, 0] <= step * 2.5
    occupied = in_mask & close_to_surface
    if not occupied.any():
        raise RuntimeError("No tissue footprint cells left after mask projection.")

    height_grid = grid_heights.reshape(len(xs), len(ys))
    occupied_grid = occupied.reshape(len(xs), len(ys)).astype(np.float32)
    if smoothing_sigma > 0:
        smoothed_weight = gaussian_filter(occupied_grid, smoothing_sigma, mode="nearest")
        smoothed_height = gaussian_filter(
            height_grid * occupied_grid, smoothing_sigma, mode="nearest"
        ) / np.maximum(smoothed_weight, 1e-6)
        height_grid[occupied.reshape(len(xs), len(ys))] = smoothed_height[
            occupied.reshape(len(xs), len(ys))
        ]
    height_grid = np.clip(height_grid, minimum_height, maximum_height)
    grid_heights = height_grid.reshape(-1)

    particle_points = []
    particle_column_heights = []
    for base_point, column_height, is_occupied in zip(base_points, grid_heights, occupied):
        if not is_occupied:
            continue
        # A sphere centered at radius touches the plane without penetrating it.
        top_center = max(radius, float(column_height) - radius)
        layer_heights = np.arange(radius, top_center + step * 0.25, step, dtype=np.float32)
        if len(layer_heights) == 0:
            layer_heights = np.asarray([radius], dtype=np.float32)
        particle_points.append(base_point[None, :] + layer_heights[:, None] * normal[None, :])
        particle_column_heights.append(float(column_height))
    points = np.concatenate(particle_points, axis=0).astype(np.float32)
    if len(points) == 0:
        raise RuntimeError("No tissue particles left after mask/ground pruning.")

    keep = sample_indices(len(points), max_particles, rng)
    points = points[keep]
    uv, valid = project_points(points, k, width, height)
    colors = np.full((len(points), 3), 0.5, dtype=np.float32)
    colors[valid] = image_rgb[uv[valid, 1], uv[valid, 0]].astype(np.float32) / 255.0
    particles = Particles(
        means=points.tolist(),
        quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(points), 1)).tolist(),
        radii=np.full((len(points),), radius, dtype=np.float32).tolist(),
        colors=colors.tolist(),
    )
    diagnostics = {
        "grid_step_m": float(step),
        "footprint_cells": int(occupied.sum()),
        "surface_height_mm_percentiles_before_clipping": np.percentile(
            surface_points @ normal + plane[3], [0, 1, 5, 50, 95, 99, 100]
        ).tolist(),
        "column_height_mm_percentiles": (
            np.percentile(np.asarray(particle_column_heights) * 1000.0, [0, 5, 50, 95, 100]).tolist()
        ),
        "particle_signed_distance_mm_percentiles": np.percentile(
            (points @ normal + plane[3]) * 1000.0, [0, 5, 50, 95, 100]
        ).tolist(),
    }
    diagnostics["surface_height_mm_percentiles_before_clipping"] = [
        float(v * 1000.0) for v in diagnostics["surface_height_mm_percentiles_before_clipping"]
    ]
    return particles, diagnostics


def constrain_surface_height(
    points: np.ndarray, plane: np.ndarray, minimum_height: float, maximum_height: float
) -> np.ndarray:
    normal = plane[:3]
    signed = points @ normal + plane[3]
    valid = np.isfinite(signed) & (signed > -minimum_height) & (signed < maximum_height * 2.0)
    points = points[valid]
    signed = signed[valid]
    clipped = np.clip(signed, minimum_height, maximum_height)
    return (points + (clipped - signed)[:, None] * normal[None, :]).astype(np.float32)


def clip_points_to_height(
    points: np.ndarray, plane: np.ndarray, minimum_height: float, maximum_height: float
) -> np.ndarray:
    normal = plane[:3]
    signed = points @ normal + plane[3]
    clipped = np.clip(signed, minimum_height, maximum_height)
    return (points + (clipped - signed)[:, None] * normal[None, :]).astype(np.float32)


def make_datapoint(
    image_rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, k: np.ndarray
) -> MaskedPosedImageAndDepth:
    return MaskedPosedImageAndDepth(
        mask=mask.astype(np.uint8),
        X_WC=X_WC_BLENDER_FOR_OPENCV_WORLD.copy(),
        K=k.astype(np.float32),
        image=image_rgb,
        format="rgb",
        depth=depth.astype(np.float32),
        depth_scale=1.0,
    )


def train_tissue_gaussians(
    initial_points: np.ndarray,
    particle_points: np.ndarray,
    radius: float,
    datapoint: MaskedPosedImageAndDepth,
    max_depth: float,
    iterations: int,
    skip_training: bool,
) -> Gaussians:
    if skip_training:
        colors = np.full((len(initial_points), 3), 0.5, dtype=np.float32)
        return Gaussians(
            means=initial_points.tolist(),
            quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(initial_points), 1)).tolist(),
            scales=np.full((len(initial_points), 3), radius, dtype=np.float32).tolist(),
            opacities=np.full((len(initial_points),), 0.5, dtype=np.float32).tolist(),
            colors=colors.tolist(),
        )

    learning_rates = GaussianLearningRates(means=0.0001, opacities=0.001, colors=0.01, quats=0.01, scales=0.01)
    gaussians = SimpleBodyBuilder._grow_gaussians(
        initial_points=initial_points,
        radius=radius,
        num_iterations=iterations,
        learning_rates=learning_rates,
        datapoints=[datapoint],
        min_scale=0.5 * radius,
        max_scale=2.0 * radius,
        max_depth=max_depth,
        visualize=False,
    )
    distant = find_distant_query_points(radius * 2.5, np.asarray(gaussians.means), particle_points)
    return gaussians.mask(~distant)


def intersect_mask_rays_with_plane(mask: np.ndarray, plane: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vs, us = np.nonzero(mask)
    rays = np.stack(
        [
            (us.astype(np.float32) - k[0, 2]) / k[0, 0],
            (vs.astype(np.float32) - k[1, 2]) / k[1, 1],
            np.ones_like(us, dtype=np.float32),
        ],
        axis=1,
    )
    denom = rays @ plane[:3]
    valid = np.abs(denom) > 1e-6
    rays = rays[valid]
    pixels = np.stack([us, vs], axis=1).astype(np.int32)[valid]
    t = -plane[3] / denom[valid]
    valid_t = t > 0
    points = rays[valid_t] * t[valid_t, None]
    pixels = pixels[valid_t]
    return points.astype(np.float32), pixels


def quat_from_z_to_axis(axis: np.ndarray) -> list[float]:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    source = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(source, axis), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    if dot < -1.0 + 1e-8:
        return [0.0, 1.0, 0.0, 0.0]
    rot_axis = np.cross(source, axis)
    rot_axis /= np.linalg.norm(rot_axis)
    angle = np.arccos(dot)
    quat_xyzw = R.from_rotvec(rot_axis * angle).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def make_ground_gaussians(
    points: np.ndarray,
    pixels: np.ndarray,
    image_rgb: np.ndarray,
    plane: np.ndarray,
    max_gaussians: int,
    rng: np.random.Generator,
) -> Gaussians:
    keep = sample_indices(len(points), max_gaussians, rng)
    points = points[keep]
    pixels = pixels[keep]
    colors = image_rgb[pixels[:, 1], pixels[:, 0]].astype(np.float32) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    scales = []
    for point in points:
        _, _, dist2 = kdtree.search_knn_vector_3d(point, min(4, len(points)))
        if len(dist2) > 1:
            scale = float(np.sqrt(np.mean(dist2[1:])))
        else:
            scale = 0.003
        scale = float(np.clip(scale, 0.0015, 0.008))
        scales.append([scale, scale, 0.001])
    quat = quat_from_z_to_axis(plane[:3])
    return Gaussians(
        means=points.tolist(),
        quats=np.tile(np.asarray([quat], dtype=np.float32), (len(points), 1)).tolist(),
        scales=scales,
        opacities=np.full((len(points),), 0.65, dtype=np.float32).tolist(),
        colors=colors.tolist(),
    )


def write_json(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def main() -> None:
    args = parse_args()
    wp.init()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    image_rgb, depth, tissue_mask, ground_mask, k = load_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tissue_points, tissue_pixels = backproject_mask(depth, tissue_mask, k)
    ground_points, ground_pixels = backproject_mask(depth, ground_mask, k)
    if len(tissue_points) == 0:
        raise RuntimeError("No valid tissue depth points.")
    if len(ground_points) == 0:
        raise RuntimeError("No valid ground depth points.")

    tissue_points_ds, tissue_pixels_ds = downsample_points(tissue_points, tissue_pixels, args.voxel_size)
    ground_points_ds, ground_pixels_ds = downsample_points(ground_points, ground_pixels, args.voxel_size)
    print(f"tissue points: raw={len(tissue_points)} downsampled={len(tissue_points_ds)}")
    print(f"ground points: raw={len(ground_points)} downsampled={len(ground_points_ds)}")

    erosion_size = max(int(args.plane_mask_erode_px), 0)
    if erosion_size > 0:
        kernel_size = erosion_size * 2 + 1
        fit_ground_mask = cv2.erode(
            ground_mask.astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        ).astype(bool)
    else:
        fit_ground_mask = ground_mask
    ground_fit_points, ground_fit_pixels = backproject_mask(depth, fit_ground_mask, k)
    if len(ground_fit_points) < 3:
        ground_fit_points, ground_fit_pixels = ground_points, ground_pixels
    plane, plane_fit_diagnostics = fit_ground_plane(
        ground_points=ground_fit_points,
        ground_pixels=ground_fit_pixels,
        tissue_points=tissue_points_ds,
        k=k,
        subsample=args.plane_fit_subsample,
        huber_k=args.plane_fit_huber_k,
    )
    ground_plane = Ground(plane=tuple(float(v) for v in plane.tolist()))
    write_json(args.output_dir / "ground_plane.json", ground_plane.model_dump_json(indent=4))

    particles, tissue_fill_diagnostics = fill_depth_shaped_particles(
        surface_points=tissue_points_ds,
        mask=tissue_mask,
        plane=plane,
        k=k,
        image_rgb=image_rgb,
        radius=args.particle_radius,
        maximum_height=args.tissue_thickness,
        minimum_height=args.minimum_tissue_thickness,
        smoothing_sigma=args.height_smoothing_sigma,
        max_particles=args.max_particles,
        rng=rng,
    )
    particle_points = np.asarray(particles.means, dtype=np.float32)

    tissue_surface_points = constrain_surface_height(
        tissue_points_ds,
        plane,
        minimum_height=max(args.particle_radius * 0.5, 1e-4),
        maximum_height=args.tissue_thickness,
    )
    gaussian_keep = sample_indices(len(tissue_surface_points), args.max_tissue_gaussians, rng)
    tissue_gaussian_initial = tissue_surface_points[gaussian_keep]
    datapoint = make_datapoint(image_rgb=image_rgb, depth=depth, mask=tissue_mask, k=k)
    gaussians = train_tissue_gaussians(
        initial_points=tissue_gaussian_initial,
        particle_points=particle_points,
        radius=args.particle_radius,
        datapoint=datapoint,
        max_depth=args.max_depth,
        iterations=args.gaussian_iters,
        skip_training=args.skip_gaussian_training,
    )
    gaussians.means = clip_points_to_height(
        np.asarray(gaussians.means, dtype=np.float32),
        plane,
        minimum_height=max(args.particle_radius * 0.5, 1e-4),
        maximum_height=args.tissue_thickness,
    ).tolist()

    tissue_body = Body(
        name="tissue",
        X_WB=np.eye(4, dtype=np.float32).tolist(),
        particles=particles,
        gaussians=gaussians,
    )
    x_wb = SimpleBodyBuilder._convert_to_body_frame(tissue_body.gaussians, tissue_body.particles)
    tissue_body.X_WB = x_wb.tolist()
    write_json(args.output_dir / "tissue.json", tissue_body.model_dump_json(indent=4))

    ground_plane_points, ground_plane_pixels = intersect_mask_rays_with_plane(ground_mask, plane, k)
    ground_plane_points_ds, ground_plane_pixels_ds = downsample_points(
        ground_plane_points, ground_plane_pixels, args.voxel_size
    )
    ground_gaussians = make_ground_gaussians(
        points=ground_plane_points_ds,
        pixels=ground_plane_pixels_ds,
        image_rgb=image_rgb,
        plane=plane,
        max_gaussians=args.max_ground_gaussians,
        rng=rng,
    )
    ground_body = Body(
        name="ground",
        X_WB=np.eye(4, dtype=np.float32).tolist(),
        particles=None,
        gaussians=ground_gaussians,
    )
    # Embed the infinite ground plane into the body JSON for physics (collision, etc.)
    ground_dict = json.loads(ground_body.model_dump_json(indent=4))
    n = plane[:3].astype(np.float32)
    ground_dict["ground_plane"] = {"a": float(n[0]), "b": float(n[1]), "c": float(n[2]), "d": float(plane[3])}
    write_json(args.output_dir / "ground.json", json.dumps(ground_dict, indent=2))

    metadata = {
        "image": str(args.image),
        "depth": str(args.depth),
        "tissue_mask": str(args.tissue_mask),
        "ground_mask": str(args.ground_mask),
        "output_dir": str(args.output_dir),
        "plane": plane.tolist(),
        "tissue_points_raw": int(len(tissue_points)),
        "tissue_points_downsampled": int(len(tissue_points_ds)),
        "ground_points_raw": int(len(ground_points)),
        "ground_points_downsampled": int(len(ground_points_ds)),
        "ground_plane_fit_points": int(len(ground_fit_points)),
        "ground_plane_points_downsampled": int(len(ground_plane_points_ds)),
        "tissue_particles": len(tissue_body.particles.means),
        "tissue_gaussians": len(tissue_body.gaussians.means),
        "ground_gaussians": len(ground_body.gaussians.means),
        "particle_radius": args.particle_radius,
        "maximum_tissue_height": args.tissue_thickness,
        "minimum_tissue_height": args.minimum_tissue_thickness,
        "height_smoothing_sigma": args.height_smoothing_sigma,
        "plane_fit": plane_fit_diagnostics,
        "tissue_fill": tissue_fill_diagnostics,
        "gaussian_iters": 0 if args.skip_gaussian_training else args.gaussian_iters,
        "world_frame": "left rectified OpenCV camera frame",
    }
    write_json(args.output_dir / "build_metadata.json", json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
