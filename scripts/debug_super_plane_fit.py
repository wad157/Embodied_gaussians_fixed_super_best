#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug SuPer first-frame ground plane fitting.")
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
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "super" / "grasp5_native" / "debug" / "plane_fit_debug.ply",
    )
    parser.add_argument("--max-raw-ground", type=int, default=30000)
    parser.add_argument("--max-ray-ground", type=int, default=30000)
    parser.add_argument("--max-tissue", type=int, default=30000)
    parser.add_argument("--plane-resolution", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def backproject_mask(depth: np.ndarray, mask: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(depth) & (depth > 0)
    vs, us = np.nonzero(valid)
    z = depth[vs, us]
    x = (us.astype(np.float32) - k[0, 2]) * z / k[0, 0]
    y = (vs.astype(np.float32) - k[1, 2]) * z / k[1, 1]
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    pixels = np.stack([us, vs], axis=1).astype(np.int32)
    return points, pixels


def sample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= max_points:
        return points
    return points[rng.choice(len(points), size=max_points, replace=False)]


def fit_svd_plane(points: np.ndarray, orient_points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    plane = np.concatenate([normal, np.array([-float(normal @ center)], dtype=np.float32)])
    plane = plane / np.linalg.norm(plane[:3])
    if np.nanmedian(orient_points @ plane[:3] + plane[3]) < 0:
        plane = -plane
    return plane.astype(np.float32)


def intersect_mask_rays_with_plane(mask: np.ndarray, plane: np.ndarray, k: np.ndarray) -> np.ndarray:
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
    valid = np.abs(denom) > 1e-8
    rays = rays[valid]
    t = -plane[3] / denom[valid]
    points = rays[t > 0] * t[t > 0, None]
    return points.astype(np.float32)


def plane_grid(plane: np.ndarray, bounds_points: np.ndarray, resolution: int) -> np.ndarray:
    normal = plane[:3].astype(np.float32)
    normal /= np.linalg.norm(normal)
    center = bounds_points.mean(axis=0)
    center = center - normal * (float(center @ normal) + float(plane[3]))

    axis_a = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if np.linalg.norm(axis_a) < 1e-6:
        axis_a = np.cross(normal, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(normal, axis_a)
    axis_b /= np.linalg.norm(axis_b)

    projected = bounds_points - center
    extent_a = max(0.02, float(np.percentile(np.abs(projected @ axis_a), 99)) * 1.15)
    extent_b = max(0.02, float(np.percentile(np.abs(projected @ axis_b), 99)) * 1.15)
    aa = np.linspace(-extent_a, extent_a, resolution, dtype=np.float32)
    bb = np.linspace(-extent_b, extent_b, resolution, dtype=np.float32)
    grid_a, grid_b = np.meshgrid(aa, bb)
    return (center + grid_a[..., None] * axis_a + grid_b[..., None] * axis_b).reshape(-1, 3)


def add_rows(points: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    colors = np.repeat(np.asarray(color, dtype=np.float32)[None, :], len(points), axis=0)
    return np.concatenate([points.astype(np.float32), colors], axis=1)


def write_ply(path: Path, rows: np.ndarray, comments: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        for comment in comments:
            f.write(f"comment {comment}\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in rows:
            f.write(f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    depth = np.load(args.depth).astype(np.float32)
    tissue_mask = load_mask(args.tissue_mask)
    ground_mask = load_mask(args.ground_mask)
    calib = json.loads(args.calib.read_text(encoding="utf-8"))
    k = np.asarray(calib["K_left_rect"], dtype=np.float32)

    ground_points, _ = backproject_mask(depth, ground_mask, k)
    tissue_points, _ = backproject_mask(depth, tissue_mask, k)
    fit_points = sample(ground_points, min(200000, len(ground_points)), rng)
    plane = fit_svd_plane(fit_points, tissue_points)
    ray_ground_points = intersect_mask_rays_with_plane(ground_mask, plane, k)
    grid_points = plane_grid(plane, ray_ground_points, args.plane_resolution)

    raw_ground_show = sample(ground_points, args.max_raw_ground, rng)
    ray_ground_show = sample(ray_ground_points, args.max_ray_ground, rng)
    tissue_show = sample(tissue_points, args.max_tissue, rng)

    raw_ground_dist = ground_points @ plane[:3] + plane[3]
    tissue_dist = tissue_points @ plane[:3] + plane[3]
    ray_ground_dist = ray_ground_points @ plane[:3] + plane[3]

    rows = np.concatenate(
        [
            add_rows(raw_ground_show, (60, 180, 60)),
            add_rows(ray_ground_show, (0, 255, 255)),
            add_rows(tissue_show, (230, 50, 50)),
            add_rows(grid_points, (50, 90, 255)),
        ],
        axis=0,
    )

    comments = [
        "world frame: left rectified OpenCV camera frame",
        "green=raw ground depth points",
        "cyan=ground mask ray-plane intersections",
        "red=raw tissue depth points",
        "blue=SVD fitted plane grid",
        f"plane={plane.tolist()}",
        f"raw_ground_distance_percentiles={np.percentile(raw_ground_dist, [0, 5, 50, 95, 100]).tolist()}",
        f"ray_ground_distance_percentiles={np.percentile(ray_ground_dist, [0, 5, 50, 95, 100]).tolist()}",
        f"tissue_distance_percentiles={np.percentile(tissue_dist, [0, 5, 50, 95, 100]).tolist()}",
    ]
    write_ply(args.output, rows, comments)

    stats = {
        "output": str(args.output),
        "plane": plane.tolist(),
        "raw_ground_points": int(len(ground_points)),
        "ray_ground_points": int(len(ray_ground_points)),
        "tissue_points": int(len(tissue_points)),
        "raw_ground_distance_percentiles": np.percentile(raw_ground_dist, [0, 5, 50, 95, 100]).tolist(),
        "ray_ground_distance_percentiles": np.percentile(ray_ground_dist, [0, 5, 50, 95, 100]).tolist(),
        "tissue_distance_percentiles": np.percentile(tissue_dist, [0, 5, 50, 95, 100]).tolist(),
    }
    stats_path = args.output.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
