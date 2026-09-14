#!/usr/bin/env python3
"""Export native EG PBD query tracks and frozen SUPER capture metadata."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
from gsplat.rendering import rasterization
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_FILE = (
    REPO_ROOT.parent
    / "embodied_gaussians_fixed_super_best_sim/baselines/embodied_gaussians_sim/run_soft.py"
)
sys.path.insert(0, str(REPO_ROOT))

from baselines.embodied_gaussians_super.common import load_calibration  # noqa: E402
from baselines.embodied_gaussians_super.protocol import (  # noqa: E402
    ADAPTER_VERSION,
    DATASETS,
    HOLDOUT_PHASE,
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


def load_shared_runner():
    sys.path.insert(0, str(SHARED_FILE.parent))
    spec = importlib.util.spec_from_file_location("eg_sim_shared_export_runner", SHARED_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SHARED_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    return parser.parse_args()


def sample_hwc(image: torch.Tensor, pixels_uv: np.ndarray) -> torch.Tensor:
    height, width = image.shape[:2]
    pixels = torch.as_tensor(pixels_uv, dtype=image.dtype, device=image.device)
    grid = pixels.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / float(width - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / float(height - 1) - 1.0
    sampled = torch_functional.grid_sample(
        image.permute(2, 0, 1).unsqueeze(0),
        grid.view(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, 0, :].transpose(0, 1)


def query_inputs(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int32)
        point_ids = np.asarray(archive["point_ids"], dtype=np.int32)
        slots = np.flatnonzero(frame_indices == QUERY_FRAME)
        if len(slots) != 1:
            raise ValueError("Frozen SUPER GT has no unique frame-0 query")
        slot = int(slots[0])
        pixels = np.asarray(archive["uv"][slot], dtype=np.float32)
        visible = np.asarray(archive["visible"][slot], dtype=bool)
        valid_3d = np.asarray(archive["valid_3d"][slot], dtype=bool)
    if point_ids.shape != (10,) or not visible.all() or not valid_3d.all():
        raise ValueError("Formal SUPER query must contain ten visible valid points")
    return frame_indices, point_ids, pixels


def main() -> None:
    args = parse_args()
    if not (0.0 < args.alpha_threshold < 1.0):
        raise ValueError("alpha-threshold must be in (0,1)")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("EG model-depth trajectory binding requires CUDA")
    native = resolve_native(REPO_ROOT, args.dataset_key, args.dataset)
    del native
    spec = dataset_spec(args.dataset_key)
    ground_truth = resolve_path(REPO_ROOT, spec["ground_truth"])
    rollout = args.rollout_dir.expanduser().resolve()
    capture = args.capture.expanduser().resolve()
    capture.mkdir(parents=True, exist_ok=True)
    prediction_path = capture / "predicted_tracks.npz"
    metadata_path = capture / "metadata.json"
    if prediction_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite capture: {capture}")
    rollout_metadata = json.loads((rollout / "rollout_metadata.json").read_text())
    if rollout_metadata.get("ground_truth_opened") is not False:
        raise ValueError("Rollout does not prove GT isolation")
    body_path = args.body.expanduser().resolve()
    if sha256_file(body_path) != rollout_metadata["body_sha256"]:
        raise ValueError("Body and rollout hash mismatch")
    gt_frames, point_ids, query_pixels = query_inputs(ground_truth)
    shared = load_shared_runner()
    body = shared.TissueGaussianBody(body_path, device)
    with np.load(rollout / "raw_particle_rollout.npz", allow_pickle=False) as archive:
        raw_frames = np.asarray(archive["frame_indices"], dtype=np.int32)
        particle_positions = np.asarray(archive["particle_positions_world"], dtype=np.float32)
        particle_rotations = np.asarray(archive["particle_rotations_world"], dtype=np.float32)
    raw_lookup = {int(frame): slot for slot, frame in enumerate(raw_frames)}
    if any(int(frame) not in raw_lookup for frame in gt_frames):
        raise ValueError("PBD rollout does not cover the frozen GT schedule")
    query_slot = raw_lookup[QUERY_FRAME]
    parents_all = body.parents.detach().cpu().numpy()
    offsets_all = body.local_offsets.detach().cpu().numpy()
    query_particle_rotations = torch.as_tensor(
        particle_rotations[query_slot, parents_all], device=device
    )
    query_means_np = (
        particle_positions[query_slot, parents_all]
        + np.einsum(
            "gij,gj->gi", particle_rotations[query_slot, parents_all], offsets_all
        )
    ).astype(np.float32)
    query_means = torch.as_tensor(query_means_np, device=device)
    query_quaternions = shared.matrix_to_quaternion(
        query_particle_rotations @ body.local_rotations
    )
    calibration = load_calibration(REPO_ROOT, args.dataset_key)
    left = calibration["stereo_left"]
    intrinsic_np = np.asarray(left["K"], dtype=np.float32)
    camera_from_world_np = np.asarray(left["X_CW_cv"], dtype=np.float32)
    width, height = map(int, left["resolution_wh"])
    rendered, alpha, _ = rasterization(
        means=query_means,
        quats=query_quaternions,
        scales=body.scales,
        colors=torch.ones_like(query_means),
        opacities=body.opacity_logits.sigmoid(),
        viewmats=torch.as_tensor(camera_from_world_np, device=device).unsqueeze(0),
        Ks=torch.as_tensor(intrinsic_np, device=device).unsqueeze(0),
        width=width,
        height=height,
        camera_model="pinhole",
        render_mode="RGB+D",
        backgrounds=torch.zeros((1, 3), dtype=torch.float32, device=device),
        near_plane=0.01,
        far_plane=2.0,
        packed=False,
    )
    query_alpha = sample_hwc(alpha[0], query_pixels)[:, 0]
    depth_premultiplied = sample_hwc(rendered[0, ..., 3:4], query_pixels)[:, 0]
    query_depth = depth_premultiplied / query_alpha.clamp_min(1.0e-8)
    raster_valid = (
        (query_alpha > args.alpha_threshold)
        & torch.isfinite(query_depth)
        & (query_depth > 0.0)
    )
    fallback = ~raster_valid
    query_binding_valid = True
    query_binding_failure_reason = None
    fallback_ids = torch.full((len(point_ids),), -1, dtype=torch.long, device=device)
    fallback_distances = torch.zeros(len(point_ids), device=device)
    camera_from_world = torch.as_tensor(camera_from_world_np, device=device)
    gaussian_camera = query_means @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    gaussian_pixels_h = gaussian_camera @ torch.as_tensor(intrinsic_np, device=device).T
    gaussian_pixels = gaussian_pixels_h[:, :2] / gaussian_pixels_h[:, 2:].clamp_min(1.0e-8)
    if bool(fallback.any()):
        visible = (
            torch.isfinite(gaussian_pixels).all(dim=1)
            & torch.isfinite(gaussian_camera).all(dim=1)
            & (gaussian_camera[:, 2] > 0.01)
            & (gaussian_pixels[:, 0] >= 0.0)
            & (gaussian_pixels[:, 0] <= width - 1)
            & (gaussian_pixels[:, 1] >= 0.0)
            & (gaussian_pixels[:, 1] <= height - 1)
        )
        visible_ids = torch.nonzero(visible, as_tuple=False).flatten()
        if len(visible_ids) == 0:
            if bool(raster_valid.any()):
                raise RuntimeError(
                    "Raster-valid query exists but no visible Gaussian centre exists"
                )
            query_binding_valid = False
            query_binding_failure_reason = "no_query_time_visible_model_gaussian"
            query_depth[:] = torch.nan
        else:
            distances = torch.cdist(
                torch.as_tensor(query_pixels, device=device)[fallback],
                gaussian_pixels[visible_ids],
            )
            minimum, local = distances.min(dim=1)
            selected = visible_ids[local]
            fallback_ids[fallback] = selected
            fallback_distances[fallback] = minimum
            query_depth[fallback] = gaussian_camera[selected, 2]
    if query_binding_valid:
        pixels_tensor = torch.as_tensor(query_pixels, device=device)
        intrinsic = torch.as_tensor(intrinsic_np, device=device)
        query_camera = torch.stack(
            (
                (pixels_tensor[:, 0] - intrinsic[0, 2]) * query_depth / intrinsic[0, 0],
                (pixels_tensor[:, 1] - intrinsic[1, 2]) * query_depth / intrinsic[1, 1],
                query_depth,
            ),
            dim=1,
        )
        world_from_camera = torch.as_tensor(left["X_WC_cv"], device=device)
        query_world = query_camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
        query_world_np = query_world.detach().cpu().numpy().astype(np.float32)
        query_to_gaussian, gaussian_ids = cKDTree(query_means_np).query(
            query_world_np, k=1, workers=-1
        )
        gaussian_ids = np.asarray(gaussian_ids, dtype=np.int64)
        fallback_np = fallback.detach().cpu().numpy()
        fallback_ids_np = fallback_ids.detach().cpu().numpy()
        gaussian_ids[fallback_np] = fallback_ids_np[fallback_np]
        selected_parents = parents_all[gaussian_ids]
        query_parent_positions = particle_positions[query_slot, selected_parents]
        query_parent_rotations = particle_rotations[query_slot, selected_parents]
        local_query = np.einsum(
            "nij,nj->ni",
            np.swapaxes(query_parent_rotations, 1, 2),
            query_world_np - query_parent_positions,
        )
        slots = np.asarray([raw_lookup[int(frame)] for frame in gt_frames], dtype=np.int64)
        selected_positions = particle_positions[slots][:, selected_parents] + np.einsum(
            "tnij,nj->tni", particle_rotations[slots][:, selected_parents], local_query
        )
        uv = np.empty((len(gt_frames), len(point_ids), 2), dtype=np.float32)
        camera_xyz = np.empty((len(gt_frames), len(point_ids), 3), dtype=np.float32)
        for index, points in enumerate(selected_positions):
            uv[index], _valid, camera_xyz[index] = project_world_points(
                points, intrinsic_np, left["X_WC_cv"], left["resolution_wh"]
            )
        query_index = int(np.flatnonzero(gt_frames == QUERY_FRAME)[0])
        initial_error = np.linalg.norm(uv[query_index] - query_pixels, axis=1)
        if not np.isfinite(initial_error).all() or float(initial_error.max()) > 1.0e-3:
            raise RuntimeError(f"PBD query reprojection failed: {float(initial_error.max())} px")
        query_to_gaussian_report = (1000.0 * np.asarray(query_to_gaussian)).tolist()
        initial_error_max = float(initial_error.max())
    else:
        gaussian_ids = np.full(len(point_ids), -1, dtype=np.int64)
        selected_parents = np.full(len(point_ids), -1, dtype=np.int64)
        selected_positions = np.full(
            (len(gt_frames), len(point_ids), 3), np.nan, dtype=np.float32
        )
        uv = np.full((len(gt_frames), len(point_ids), 2), np.nan, dtype=np.float32)
        camera_xyz = np.full(
            (len(gt_frames), len(point_ids), 3), np.nan, dtype=np.float32
        )
        query_to_gaussian_report = [None] * len(point_ids)
        initial_error_max = None
    np.savez_compressed(
        prediction_path,
        schema=np.asarray("super_tissue_predicted_tracks_v1"),
        frame_indices=gt_frames,
        observation_used=np.asarray(
            [observation_allowed(int(frame), args.dataset_key) for frame in gt_frames], dtype=bool
        ),
        uv=uv,
        xyz_camera_m=camera_xyz,
        xyz_world_m=selected_positions.astype(np.float32),
        query_frame_index=np.asarray(QUERY_FRAME, dtype=np.int32),
        point_ids=point_ids,
        query_pixels_full_resolution=query_pixels,
        query_alpha=query_alpha.detach().cpu().numpy().astype(np.float32),
        query_depth_embodied_gaussians_m=query_depth.detach().cpu().numpy().astype(np.float32),
    )
    track_report = {
        "decoder": "native PBD persistent parent-frame query",
        "shape_of_motion_used": False,
        "query_binding_valid": query_binding_valid,
        "query_binding_failure_reason": query_binding_failure_reason,
        "query_point_count": len(point_ids),
        "query_raster_coverage_count": int(raster_valid.sum().item()),
        "query_projected_gaussian_fallback_count": int(fallback.sum().item()),
        "fallback_pixel_distances": (
            fallback_distances[fallback].detach().cpu().numpy().tolist()
            if query_binding_valid
            else [None] * int(fallback.sum().item())
        ),
        "selected_gaussian_ids": gaussian_ids.tolist(),
        "selected_parent_particle_ids": selected_parents.tolist(),
        "query_to_gaussian_distance_mm": query_to_gaussian_report,
        "initial_query_reprojection_max_px": initial_error_max,
        "query_gt_depth_used": False,
        "query_gt_3d_used": False,
        "alignment": "none",
    }
    metadata = {
        "schema": "super_tissue_physics_benchmark_v1",
        "protocol": PROTOCOL,
        "ground_truth": str(ground_truth),
        "ground_truth_sha256": sha256_file(ground_truth),
        "trajectory_binding": "native Embodied Gaussians PBD parent frame",
        "trajectory_query_frame": QUERY_FRAME,
        "reconstruction_split": {
            "ratio": [7, 1],
            "test_rule": f"full video frame_index % 8 == {HOLDOUT_PHASE}",
            "test_end_exclusive": int(spec["future_start"]),
        },
        "future_split": {
            "train_end_inclusive": int(spec["future_start"]) - 1,
            "test_start_inclusive": int(spec["future_start"]),
            "ground_truth_formal_test_start": int(spec["future_start"]),
        },
        "render_scale": RENDER_SCALE,
        "render_camera": "stereo_left",
        "render_mask": "all pixels except the frozen SurgicalSAM2 instrument mask",
        "capture_renders": True,
        "baseline": {
            "method": "Embodied Gaussians EG-Soft paper reconstruction",
            "adapter_version": ADAPTER_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "public_soft_code_available": False,
            "shape_of_motion_used": False,
            "future_behavior": "paper PBD open loop with known raw robot control",
            "ground_truth_usage": "frame-0 2D queries after rollout freeze and downstream scoring only",
            "future_observations_used": False,
            "track_report": track_report,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (capture / "capture_summary.json").write_text(
        json.dumps(
            {
                "schema": "embodied_gaussians_super_capture_v1",
                "protocol": PROTOCOL,
                "track_frame_count": len(gt_frames),
                "render_frame_count": len(rendering_frames(args.dataset_key)),
                "elapsed_s": float(rollout_metadata["total_elapsed_s"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
