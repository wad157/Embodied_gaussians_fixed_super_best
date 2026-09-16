#!/usr/bin/env python3
"""Export PhysTwin native particle-LBS tracks and exact SUPER render schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization

from common import load_calibration
from protocol import (
    ADAPTER_VERSION,
    DATASETS,
    HOLDOUT_PHASE,
    INITIALIZATION_FRAME,
    PROTOCOL,
    QUERY_FRAME,
    RENDER_SCALE,
    UPSTREAM_COMMIT,
    dataset_spec,
    observation_allowed,
    project_world_points,
    rendering_frames,
    resolve_native,
    resolve_path,
    sha256_file,
)
from gaussian_splatting.dynamic_utils import get_topk_indices, interpolate_motions, knn_weights


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--physics-dir", type=Path, required=True)
    parser.add_argument("--appearance-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--lbs-neighbours", type=int, default=16)
    parser.add_argument("--lbs-chunk", type=int, default=10000)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--frame-limit", type=int, default=-1, help="engineering smoke test only")
    return parser.parse_args()


def query_inputs(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        frames = np.asarray(archive["frame_indices"], dtype=np.int32)
        point_ids = np.asarray(archive["point_ids"], dtype=np.int32)
        slots = np.flatnonzero(frames == QUERY_FRAME)
        if len(slots) != 1:
            raise ValueError("Frozen SUPER GT has no unique frame-0 query")
        slot = int(slots[0])
        pixels = np.asarray(archive["uv"][slot], dtype=np.float32)
        visible = np.asarray(archive["visible"][slot], dtype=bool)
        valid_3d = np.asarray(archive["valid_3d"][slot], dtype=bool)
    if point_ids.shape != (10,) or not visible.all() or not valid_3d.all():
        raise ValueError("Formal SUPER query requires ten visible, 3D-valid points")
    return frames, point_ids, pixels


def sample_hwc(image: torch.Tensor, pixels: np.ndarray) -> torch.Tensor:
    height, width = image.shape[:2]
    grid = torch.as_tensor(pixels, dtype=image.dtype, device=image.device).clone()
    grid[:, 0] = 2.0 * grid[:, 0] / float(width - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / float(height - 1) - 1.0
    sampled = F.grid_sample(
        image.permute(2, 0, 1)[None], grid.view(1, 1, -1, 2),
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    return sampled[0, :, 0].T


def render_gaussians(means, quats, scales, colors, opacities, calibration, scale, render_depth):
    current = calibration["stereo_left"]
    full_width, full_height = current["resolution_wh"]
    width, height = int(round(full_width * scale)), int(round(full_height * scale))
    intrinsic = np.asarray(current["K"], np.float32).copy()
    intrinsic[0, :] *= width / full_width
    intrinsic[1, :] *= height / full_height
    intrinsic[2, :] = (0, 0, 1)
    rendered, alpha, _ = rasterization(
        means=means, quats=F.normalize(quats, dim=-1), scales=scales,
        colors=colors, opacities=opacities,
        viewmats=torch.as_tensor(current["X_CW_cv"], device=means.device)[None],
        Ks=torch.as_tensor(intrinsic, device=means.device)[None],
        width=width, height=height, packed=False,
        render_mode="RGB+D" if render_depth else "RGB",
        backgrounds=torch.zeros((1, 3), dtype=torch.float32, device=means.device),
        near_plane=0.01, far_plane=2.0,
    )
    return rendered[0], alpha[0]


def query_anchors(pixels, means, quats, scales, colors, opacities, calibration, threshold):
    rendered, alpha = render_gaussians(
        means, quats, scales, colors, opacities, calibration, RENDER_SCALE, True
    )
    scaled_pixels = pixels * RENDER_SCALE
    query_alpha = sample_hwc(alpha, scaled_pixels)[:, 0]
    depth = sample_hwc(rendered[..., 3:4], scaled_pixels)[:, 0] / query_alpha.clamp_min(1.0e-8)
    valid = (query_alpha > threshold) & torch.isfinite(depth) & (depth > 0)
    current = calibration["stereo_left"]
    camera_from_world = torch.as_tensor(current["X_CW_cv"], device=means.device)
    homogeneous = torch.cat((means, torch.ones_like(means[:, :1])), dim=1)
    gaussian_camera = (camera_from_world @ homogeneous.T).T[:, :3]
    z = gaussian_camera[:, 2]
    intrinsic = torch.as_tensor(current["K"], device=means.device)
    projected_h = (intrinsic @ gaussian_camera.T).T
    projected = projected_h[:, :2] / z[:, None].clamp_min(1.0e-8)
    fallback_ids = torch.full((len(pixels),), -1, dtype=torch.long, device=means.device)
    fallback_distance = torch.full((len(pixels),), float("nan"), device=means.device)
    if bool((~valid).any()):
        width, height = current["resolution_wh"]
        visible = (torch.isfinite(projected).all(dim=1) & (z > 0.01)
                   & (projected[:, 0] >= 0) & (projected[:, 0] <= width - 1)
                   & (projected[:, 1] >= 0) & (projected[:, 1] <= height - 1))
        visible_ids = torch.nonzero(visible, as_tuple=False).flatten()
        if len(visible_ids) == 0:
            raise RuntimeError("No query-time visible PhysTwin appearance Gaussian")
        distances = torch.cdist(torch.as_tensor(pixels, device=means.device)[~valid], projected[visible_ids])
        minimum, local = distances.min(dim=1)
        selected = visible_ids[local]
        fallback_ids[~valid] = selected
        fallback_distance[~valid] = minimum
        depth[~valid] = z[selected]
    uv_h = torch.cat((torch.as_tensor(pixels, device=means.device),
                      torch.ones((len(pixels), 1), device=means.device)), dim=1)
    camera_points = (torch.linalg.inv(intrinsic) @ uv_h.T).T * depth[:, None]
    world_from_camera = torch.as_tensor(current["X_WC_cv"], device=means.device)
    anchors = (world_from_camera @ torch.cat((camera_points, torch.ones_like(camera_points[:, :1])), dim=1).T).T[:, :3]
    return anchors, query_alpha, depth, valid, fallback_ids, fallback_distance


def main() -> None:
    args = parse_args()
    if args.capture.exists() or args.render_dir.exists():
        raise FileExistsError("Refusing to overwrite capture or render arrays")
    resolve_native(REPO_ROOT, args.dataset_key, args.dataset)
    spec = dataset_spec(args.dataset_key)
    frame_count = int(spec["frames"])
    ground_truth = resolve_path(REPO_ROOT, spec["ground_truth"])
    physics_path = args.physics_dir / "raw_particle_rollout.npz"
    physics_meta_path = args.physics_dir / "metadata.json"
    gaussian_path = args.appearance_dir / "gaussians.npz"
    appearance_meta_path = args.appearance_dir / "metadata.json"
    for path in (physics_path, physics_meta_path, gaussian_path, appearance_meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    physics_meta = json.loads(physics_meta_path.read_text(encoding="utf-8"))
    appearance_meta = json.loads(appearance_meta_path.read_text(encoding="utf-8"))
    if not physics_meta.get("full_protocol", False) and not args.allow_smoke:
        raise ValueError("Refusing to export smoke-test physics as formal output")
    if appearance_meta.get("evaluation_truth_opened") is not False:
        raise ValueError("Appearance stage does not prove GT isolation")
    with np.load(physics_path, allow_pickle=False) as archive:
        particles_np = np.asarray(archive["particle_positions_world"], np.float32)
    with np.load(gaussian_path, allow_pickle=False) as archive:
        means_np = np.asarray(archive["means_world"], np.float32)
        colors_np = np.asarray(archive["colors_rgb"], np.float32)
        opacities_np = np.asarray(archive["opacities"], np.float32)
        scales_np = np.asarray(archive["scales"], np.float32)
        quats_np = np.asarray(archive["quats"], np.float32)
    if particles_np.shape[0] != frame_count:
        raise ValueError("Physics rollout frame count mismatch")

    # Evaluation queries are deliberately opened only after physics and appearance freeze.
    gt_frames, point_ids, query_pixels = query_inputs(ground_truth)
    gt_lookup = {int(frame): slot for slot, frame in enumerate(gt_frames)}
    calibration = load_calibration(REPO_ROOT, args.dataset_key)
    device = torch.device(args.device)
    means = torch.from_numpy(means_np).to(device)
    colors = torch.from_numpy(colors_np).to(device)
    opacities = torch.from_numpy(opacities_np).to(device)
    scales = torch.from_numpy(scales_np).to(device)
    quats = torch.from_numpy(quats_np).to(device)
    anchors, query_alpha, query_depth, raster_valid, fallback_ids, fallback_distance = query_anchors(
        query_pixels, means, quats, scales, colors, opacities, calibration, args.alpha_threshold
    )
    query_quats = torch.zeros((len(anchors), 4), device=device)
    query_quats[:, 0] = 1
    all_positions = torch.cat((means, anchors), dim=0)
    all_quats = torch.cat((quats, query_quats), dim=0)
    particles = torch.from_numpy(particles_np).to(device)
    relations = get_topk_indices(particles[0], K=args.lbs_neighbours)
    selected_renders = set(rendering_frames(args.dataset_key))
    query_positions = np.full((len(gt_frames), len(anchors), 3), np.nan, np.float32)
    args.render_dir.mkdir(parents=True)
    process_frames = frame_count if args.frame_limit < 0 else min(frame_count, args.frame_limit)
    if process_frames < 1:
        raise ValueError("frame-limit must be -1 or positive")
    with torch.inference_mode():
        for frame in range(process_frames):
            if frame > 0:
                previous = particles[frame - 1]
                motion = particles[frame] - previous
                for start in range(0, len(all_positions), args.lbs_chunk):
                    end = min(start + args.lbs_chunk, len(all_positions))
                    weights = knn_weights(previous, all_positions[start:end], K=args.lbs_neighbours)
                    position, quaternion, _ = interpolate_motions(
                        bones=previous, motions=motion, relations=relations,
                        xyz=all_positions[start:end], quat=all_quats[start:end], weights=weights,
                    )
                    all_positions[start:end] = position
                    all_quats[start:end] = quaternion
            if frame in gt_lookup:
                query_positions[gt_lookup[frame]] = all_positions[len(means):].cpu().numpy()
            if frame in selected_renders:
                rendered, _alpha = render_gaussians(
                    all_positions[:len(means)], all_quats[:len(means)], scales,
                    colors, opacities, calibration, RENDER_SCALE, False,
                )
                np.save(args.render_dir / f"{frame:06d}.npy",
                        rendered[..., :3].clamp(0, 1).cpu().numpy().astype(np.float32))
            if frame % 20 == 0 or frame == process_frames - 1:
                print(f"[LBS/render] {frame}/{process_frames - 1}", flush=True)

    left = calibration["stereo_left"]
    uv = np.empty((len(gt_frames), len(point_ids), 2), np.float32)
    camera_xyz = np.empty((len(gt_frames), len(point_ids), 3), np.float32)
    for index, points in enumerate(query_positions):
        uv[index], _valid, camera_xyz[index] = project_world_points(
            points, left["K"], left["X_WC_cv"], left["resolution_wh"]
        )
    query_index = int(np.flatnonzero(gt_frames == QUERY_FRAME)[0])
    initial_error = np.linalg.norm(uv[query_index] - query_pixels, axis=1)
    if not np.isfinite(initial_error).all() or float(initial_error.max()) > 1.0e-3:
        raise RuntimeError(f"PhysTwin query reprojection failed: {float(initial_error.max())} px")
    args.capture.mkdir(parents=True)
    np.savez_compressed(
        args.capture / "predicted_tracks.npz",
        schema=np.asarray("super_tissue_predicted_tracks_v1"), frame_indices=gt_frames,
        observation_used=np.asarray([observation_allowed(int(frame), args.dataset_key) for frame in gt_frames], bool),
        uv=uv, xyz_camera_m=camera_xyz, xyz_world_m=query_positions,
        query_frame_index=np.asarray(QUERY_FRAME, np.int32), point_ids=point_ids,
        query_pixels_full_resolution=query_pixels,
        query_alpha=query_alpha.cpu().numpy().astype(np.float32),
        query_depth_phystwin_m=query_depth.cpu().numpy().astype(np.float32),
    )
    metadata = {
        "schema": "super_tissue_physics_benchmark_v1", "protocol": PROTOCOL,
        "ground_truth": str(ground_truth), "ground_truth_sha256": sha256_file(ground_truth),
        "trajectory_binding": "PhysTwin native spring-mass particles plus upstream KNN-LBS",
        "trajectory_query_frame": QUERY_FRAME,
        "reconstruction_split": {"ratio": [7, 1],
            "test_rule": f"full video frame_index % 8 == {HOLDOUT_PHASE}",
            "test_end_exclusive": int(spec["future_start"])},
        "future_split": {"train_end_inclusive": int(spec["future_start"]) - 1,
            "test_start_inclusive": int(spec["future_start"]),
            "ground_truth_formal_test_start": int(spec["future_start"])},
        "render_scale": RENDER_SCALE, "render_camera": "stereo_left",
        "render_mask": "all pixels except the frozen SurgicalSAM2 instrument mask",
        "capture_renders": True,
        "baseline": {
            "method": "PhysTwin", "adapter_version": ADAPTER_VERSION,
            "upstream_commit": UPSTREAM_COMMIT, "shape_of_motion_used": False,
            "initialization_frame": INITIALIZATION_FRAME,
            "future_behavior": "native spring-mass open loop with known raw robot control",
            "ground_truth_usage": "frame-0 2D queries after physics/appearance freeze and downstream scoring only",
            "future_observations_used": False,
            "track_report": {
                "decoder": "PhysTwin native particle KNN-LBS (K=16)",
                "query_point_count": int(len(point_ids)),
                "query_raster_coverage_count": int(raster_valid.sum().item()),
                "query_projected_gaussian_fallback_count": int((~raster_valid).sum().item()),
                "fallback_gaussian_ids": fallback_ids.cpu().numpy().tolist(),
                "fallback_pixel_distances": fallback_distance.cpu().numpy().tolist(),
                "initial_query_reprojection_max_px": float(initial_error.max()),
                "query_gt_depth_used": False, "query_gt_3d_used": False,
                "alignment": "none", "lbs_neighbours": args.lbs_neighbours,
            },
        },
    }
    (args.capture / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.capture / "capture_summary.json").write_text(json.dumps({
        "schema": "phystwin_super_capture_v1", "protocol": PROTOCOL,
        "track_frame_count": int(len(gt_frames)), "render_frame_count": len(selected_renders),
        "physics_wall_seconds": float(physics_meta.get("wall_seconds", 0)),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
