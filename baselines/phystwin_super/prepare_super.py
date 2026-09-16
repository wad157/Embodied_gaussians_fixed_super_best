#!/usr/bin/env python3
"""Build leakage-free PhysTwin observations from legal SUPER training frames."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from common import (
    TissueMasks,
    deproject_pixels,
    farthest_point_indices,
    load_calibration,
    load_rgb,
    nearest_timestamp_index,
    resize_to_depth,
    sample_image,
)
from protocol import (
    COTRACKER_CHECKPOINT,
    COTRACKER_VARIANT,
    DATASETS,
    DEPTH_DOWNSAMPLE,
    INITIALIZATION_FRAME,
    UPSTREAM_COMMIT,
    dataset_spec,
    depth_cache,
    resolve_native,
    resolve_path,
    sha256_file,
    training_frames,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COTRACKER_ROOT = Path("/home/jwshan/.cache/torch/hub/facebookresearch_co-tracker_main")
DEFAULT_COTRACKER_WEIGHT = Path("/home/jwshan/.cache/torch/hub/checkpoints/scaled_offline.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--queries", type=int, default=768)
    parser.add_argument("--max-physics-points", type=int, default=768)
    parser.add_argument("--appearance-points", type=int, default=50000)
    parser.add_argument("--controller-points", type=int, default=30)
    parser.add_argument("--track-chunk", type=int, default=120)
    parser.add_argument("--legal-frame-limit", type=int, default=-1, help="engineering smoke test only")
    parser.add_argument("--cotracker-root", type=Path, default=DEFAULT_COTRACKER_ROOT)
    parser.add_argument("--cotracker-weight", type=Path, default=DEFAULT_COTRACKER_WEIGHT)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initial_queries(mask: np.ndarray, count: int, seed: int) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    y, x = np.nonzero(eroded)
    if len(x) < count:
        raise ValueError(f"Tissue mask has only {len(x)} eligible query pixels")
    rng = np.random.RandomState(seed)
    candidates = rng.choice(len(x), size=min(len(x), count * 20), replace=False)
    points = np.stack((x[candidates], y[candidates]), axis=1).astype(np.float32)
    return points[farthest_point_indices(points, count, seed)]


def track_left(native, depth_root, allowed, masks, predictor, args):
    depth_shape = np.load(depth_root / f"{allowed[0]:06d}-depth.npy", mmap_mode="r").shape
    small_mask = resize_to_depth(masks.get(allowed[0]).astype(np.uint8), depth_shape, nearest=True).astype(bool)
    current_queries = initial_queries(small_mask, args.queries, args.seed)
    all_tracks = np.empty((len(allowed), len(current_queries), 2), dtype=np.float32)
    all_visible = np.zeros((len(allowed), len(current_queries)), dtype=bool)
    start = 0
    chunk_id = 0
    while start < len(allowed):
        end = min(start + args.track_chunk, len(allowed))
        frame_ids = allowed[start:end]
        frames = [resize_to_depth(load_rgb(native, frame), depth_shape) for frame in frame_ids]
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float().to(args.device)
        query = np.concatenate((np.zeros((len(current_queries), 1), np.float32), current_queries), axis=1)
        with torch.inference_mode():
            tracks, visible = predictor(
                video, queries=torch.from_numpy(query)[None].to(args.device), backward_tracking=False
            )
        tracks_np = tracks[0].cpu().numpy().astype(np.float32)
        visible_np = visible[0].cpu().numpy().astype(bool)
        all_tracks[start:end] = tracks_np
        all_visible[start:end] = visible_np
        current_queries = tracks_np[-1].copy()
        chunk_id += 1
        print(f"[CoTracker] chunk={chunk_id} legal_frames={start}:{end}/{len(allowed)}", flush=True)
        del video, tracks, visible
        torch.cuda.empty_cache()
        if end == len(allowed):
            break
        start = end - 1
    return all_tracks, all_visible, depth_shape


def controller_trajectory(repo_root, dataset_key, initial_tissue_points, count, seed, calibration):
    spec = dataset_spec(dataset_key)
    pose_path = resolve_path(repo_root, spec["pose_driver"])
    surface_path = repo_root / "data/super/psm_robot/psm_surface_gaussians.npz"
    with np.load(pose_path, allow_pickle=False) as poses:
        timestamps = np.asarray(poses["timestamps"], dtype=np.float64)
        driver_links = [str(x) for x in poses["link_names"]]
        pose_values = np.asarray(poses["poses_rect_camera_xyz_xyzw"], dtype=np.float32)
    with np.load(surface_path, allow_pickle=False) as surface:
        surface_links = [str(x) for x in surface["link_names"]]
        surface_ids = np.asarray(surface["link_ids"], dtype=np.int64)
        local = np.asarray(surface["means"], dtype=np.float32)
    lookup = {name: index for index, name in enumerate(driver_links)}
    link_ids = np.asarray([lookup[surface_links[index]] for index in surface_ids], dtype=np.int64)
    world_from_camera = np.asarray(calibration["stereo_left"]["X_WC_cv"], dtype=np.float32)
    video_timestamps = np.asarray(calibration["left_timestamps"])

    def world_surface(frame):
        slot = nearest_timestamp_index(timestamps, float(video_timestamps[frame]))
        pose = pose_values[slot]
        rotations = Rotation.from_quat(pose[:, 3:]).as_matrix().astype(np.float32)[link_ids]
        camera = np.einsum("nij,nj->ni", rotations, local) + pose[:, :3][link_ids]
        return camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]

    initial_surface = world_surface(INITIALIZATION_FRAME)
    distance, _ = cKDTree(np.asarray(initial_tissue_points)).query(initial_surface, k=1)
    # The SUPER surface asset also contains the long shaft.  Restrict the FPS
    # pool to the closest contact-capable samples instead of spatially covering
    # the whole robot, which would waste nearly all 30 controller slots.
    pool_count = min(len(initial_surface), max(count * 10, count))
    pool = np.argpartition(distance, pool_count - 1)[:pool_count]
    selected = pool[farthest_point_indices(initial_surface[pool], count, seed)]
    trajectory = np.empty((int(dataset_spec(dataset_key)["frames"]), len(selected), 3), np.float32)
    for frame in range(len(trajectory)):
        trajectory[frame] = world_surface(frame)[selected]
    return trajectory, {
        "pose_driver": str(pose_path),
        "pose_driver_sha256": sha256_file(pose_path),
        "pose_report": str(resolve_path(repo_root, spec["pose_report"])),
        "surface": str(surface_path),
        "surface_sha256": sha256_file(surface_path),
        "points": int(len(selected)),
        "selection": "closest raw-FK surface pool to legal frame-1 tissue, then FPS",
        "initial_distance_to_tissue_m": {
            "min": float(distance[selected].min()),
            "median": float(np.median(distance[selected])),
            "max": float(distance[selected].max()),
        },
    }


def appearance_cloud(native, depth_root, masks, calibration, count, seed):
    frame = INITIALIZATION_FRAME
    depth = np.load(depth_root / f"{frame:06d}-depth.npy").astype(np.float32)
    image = resize_to_depth(load_rgb(native, frame), depth.shape).astype(np.float32) / 255.0
    tissue = resize_to_depth(masks.get(frame).astype(np.uint8), depth.shape, nearest=True).astype(bool)
    valid = tissue & np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(valid)
    rng = np.random.RandomState(seed)
    selected = rng.choice(len(x), size=min(count, len(x)), replace=False)
    uv = np.stack((x[selected], y[selected]), axis=1).astype(np.float32)
    intrinsic = np.asarray(calibration["stereo_left"]["K"], np.float32).copy()
    intrinsic[:2] /= DEPTH_DOWNSAMPLE
    world, world_valid = deproject_pixels(
        uv, depth, intrinsic, calibration["stereo_left"]["X_WC_cv"]
    )
    return world[world_valid], image[y[selected][world_valid], x[selected][world_valid]]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("PhysTwin preprocessing requires CUDA")
    if args.track_chunk < 2:
        raise ValueError("track-chunk must be at least two")
    if not args.cotracker_root.is_dir() or not args.cotracker_weight.is_file():
        raise FileNotFoundError("Missing local CoTracker source or scaled-offline checkpoint")
    set_seed(args.seed)
    native = resolve_native(REPO_ROOT, args.dataset_key, args.dataset)
    spec = dataset_spec(args.dataset_key)
    frame_count = int(spec["frames"])
    formal_allowed = training_frames(args.dataset_key)
    if args.legal_frame_limit == 0:
        raise ValueError("legal-frame-limit must be -1 or positive")
    allowed = formal_allowed if args.legal_frame_limit < 0 else formal_allowed[:args.legal_frame_limit]
    if allowed[0] != INITIALIZATION_FRAME:
        raise AssertionError("First legal SUPER observation must be frame 1")
    depth_root = depth_cache(REPO_ROOT, args.dataset_key)
    actual_depth = sorted(int(path.name[:6]) for path in depth_root.glob("??????-depth.npy"))
    if actual_depth != formal_allowed or not (depth_root / "COMPLETE").is_file():
        raise ValueError("FoundationStereo cache is not the exact legal training schedule")
    calibration = load_calibration(REPO_ROOT, args.dataset_key)
    masks = TissueMasks(native)
    sys.path.insert(0, str(args.cotracker_root.resolve()))
    from cotracker.predictor import CoTrackerPredictor

    predictor = CoTrackerPredictor(
        checkpoint=str(args.cotracker_weight.resolve()), offline=True, v2=False, window_len=60
    ).to(args.device).eval()
    tracks, cotrack_visible, depth_shape = track_left(
        native, depth_root, allowed, masks, predictor, args
    )
    del predictor
    torch.cuda.empty_cache()

    n = tracks.shape[1]
    object_points = np.full((frame_count, n, 3), np.nan, dtype=np.float32)
    object_colors = np.zeros((frame_count, n, 3), dtype=np.float32)
    object_visible = np.zeros((frame_count, n), dtype=bool)
    intrinsic = np.asarray(calibration["stereo_left"]["K"], np.float32).copy()
    intrinsic[:2] /= DEPTH_DOWNSAMPLE
    for local, frame in enumerate(allowed):
        depth = np.load(depth_root / f"{frame:06d}-depth.npy").astype(np.float32)
        image = resize_to_depth(load_rgb(native, frame), depth_shape)
        tissue = resize_to_depth(masks.get(frame).astype(np.uint8), depth_shape, nearest=True).astype(bool)
        world, valid_depth = deproject_pixels(
            tracks[local], depth, intrinsic, calibration["stereo_left"]["X_WC_cv"]
        )
        valid = cotrack_visible[local] & sample_image(tissue, tracks[local]).astype(bool) & valid_depth
        object_points[frame] = world
        object_colors[frame] = sample_image(image, tracks[local]).astype(np.float32) / 255.0
        object_visible[frame] = valid
        if (local + 1) % 100 == 0 or local + 1 == len(allowed):
            print(f"[deproject] {local + 1}/{len(allowed)}", flush=True)
    keep = object_visible[INITIALIZATION_FRAME] & np.isfinite(object_points[INITIALIZATION_FRAME]).all(axis=1)
    if int(keep.sum()) < 32:
        raise RuntimeError(f"Only {int(keep.sum())} persistent particles survive initialization")
    object_points = object_points[:, keep]
    object_colors = object_colors[:, keep]
    object_visible = object_visible[:, keep]
    selected = farthest_point_indices(
        object_points[INITIALIZATION_FRAME], min(args.max_physics_points, object_points.shape[1]), args.seed
    )
    object_points = object_points[:, selected]
    object_colors = object_colors[:, selected]
    object_visible = object_visible[:, selected]
    object_points[0] = object_points[INITIALIZATION_FRAME]
    object_colors[0] = object_colors[INITIALIZATION_FRAME]
    for point in range(object_points.shape[1]):
        last_point = object_points[0, point].copy()
        last_color = object_colors[0, point].copy()
        for frame in range(frame_count):
            if object_visible[frame, point]:
                last_point = object_points[frame, point].copy()
                last_color = object_colors[frame, point].copy()
            else:
                object_points[frame, point] = last_point
                object_colors[frame, point] = last_color
    motions_valid = object_visible[:-1] & object_visible[1:]
    # PhysTwin captures the initial valid-motion count as a CUDA-graph scalar.
    # SUPER withholds frame 0, so bootstrap the 0->1 target as zero motion from
    # the legal frame-1 state.  This neither opens nor estimates frame 0.
    motions_valid[0] = object_visible[INITIALIZATION_FRAME]
    controller, controller_meta = controller_trajectory(
        REPO_ROOT, args.dataset_key, object_points[0], args.controller_points, args.seed, calibration
    )
    appearance_points, appearance_colors = appearance_cloud(
        native, depth_root, masks, calibration, args.appearance_points, args.seed
    )

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "final_data.pkl").open("wb") as stream:
        pickle.dump({
            "object_points": object_points,
            "object_colors": object_colors,
            "object_visibilities": object_visible,
            "object_motions_valid": motions_valid,
            "controller_points": controller,
            "surface_points": np.zeros((0, 3), np.float32),
            "interior_points": np.zeros((0, 3), np.float32),
        }, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(
        args.output_dir / "appearance.npz",
        means_world=appearance_points.astype(np.float32),
        colors_rgb=appearance_colors.astype(np.float32),
    )
    metadata = {
        "schema": "phystwin_super_preprocess_v1",
        "dataset_key": args.dataset_key,
        "seed": args.seed,
        "phystwin_commit": UPSTREAM_COMMIT,
        "tracker": COTRACKER_VARIANT,
        "tracker_checkpoint": COTRACKER_CHECKPOINT,
        "tracker_checkpoint_sha256": sha256_file(args.cotracker_weight),
        "tracker_chunk_legal_frames": args.track_chunk,
        "frame_count": frame_count,
        "future_start": int(spec["future_start"]),
        "initialization_frame": INITIALIZATION_FRAME,
        "full_protocol": args.legal_frame_limit < 0,
        "allowed_observation_frames": allowed,
        "opened_observation_frame_count": len(allowed),
        "reconstruction_holdouts_opened": False,
        "future_observations_opened": False,
        "evaluation_truth_opened": False,
        "depth_cache": str(depth_root),
        "depth_cache_exact_legal_schedule": True,
        "physics_points": int(object_points.shape[1]),
        "appearance_points": int(len(appearance_points)),
        "visibility_fraction_legal_frames": float(object_visible[allowed].mean()),
        "frame_zero_motion_bootstrap": "legal frame-1 state repeated with zero displacement for CUDA-graph denominator",
        "controller": controller_meta,
        "trajectory_source": "PhysTwin native persistent particles",
        "shape_of_motion_used": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
