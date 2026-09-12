#!/usr/bin/env python3
"""Export EH-SurGS SUPER renders and query-anchored trajectories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional

from arguments import FDMHiddenParams, ModelParams, PipelineParams
from gaussian_renderer import render_flow
from utils.params_utils import merge_hparams

from common import (
    RenderCamera,
    deformed_centers,
    internal_world_from_camera,
    load_calibration,
    load_trained_gaussians,
    scaled_intrinsic,
)
from protocol import (
    ADAPTER_VERSION,
    DATASETS,
    HOLDOUT_PHASE,
    INTERNAL_UNITS_PER_METER,
    PROTOCOL,
    QUERY_FRAME,
    RENDER_SCALE,
    SHAPE_OF_MOTION_COMMIT,
    TRACK_DECODER_VERSION,
    UPSTREAM_COMMIT,
    dataset_spec,
    normalized_time,
    project_world_points,
    rendering_frames,
    resolve_native,
    resolve_path,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "super_unified.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    hidden = FDMHiddenParams(parser)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--configs", default=str(DEFAULT_CONFIG))
    parser.add_argument("--render-scale", type=float, default=RENDER_SCALE)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--tracks-only", action="store_true")
    args = parser.parse_args()
    import mmcv

    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    if not args.model_path or not args.source_path:
        parser.error("--model_path and --source_path must be explicit")
    args.source_path = str(resolve_native(REPO_ROOT, args.dataset_key, Path(args.source_path)))
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        parser.error("Refusing to overwrite export directory: {}".format(args.output_dir))
    if not 0.0 < args.render_scale <= 1.0:
        parser.error("--render-scale must be in (0,1]")
    return args, model, pipeline, hidden


def render(camera, gaussians, pipe, background, learned_mask, override_color=None):
    return render_flow(
        camera,
        gaussians,
        pipe,
        background,
        True,
        override_color=override_color,
        mask=learned_mask,
    )


def make_render_camera(calibration, frame, frames, scale):
    width, height = calibration["resolution_wh"]
    target_wh = (int(round(width * scale)), int(round(height * scale)))
    downsample = float(width) / float(target_wh[0])
    intrinsic = scaled_intrinsic(calibration["K"], downsample)
    return RenderCamera(
        intrinsic,
        calibration["X_WC_opencv"],
        target_wh,
        normalized_time(frame, frames),
    ), intrinsic


def sample_chw(image, pixels_uv):
    height, width = image.shape[-2:]
    pixels = torch.as_tensor(pixels_uv, dtype=image.dtype, device=image.device)
    grid = pixels.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / float(width - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / float(height - 1) - 1.0
    sampled = torch_functional.grid_sample(
        image.unsqueeze(0),
        grid.view(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, 0, :].transpose(0, 1)


def query_inputs(ground_truth):
    with np.load(str(ground_truth), allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int32)
        point_ids = np.asarray(archive["point_ids"], dtype=np.int32)
        slots = np.flatnonzero(frame_indices == QUERY_FRAME)
        if len(slots) != 1:
            raise ValueError("Current SUPER GT has no unique frame-0 query")
        slot = int(slots[0])
        pixels = np.asarray(archive["uv"][slot], dtype=np.float32)
        visible = np.asarray(archive["visible"][slot], dtype=bool)
        valid_3d = np.asarray(archive["valid_3d"][slot], dtype=bool)
    if point_ids.shape != (10,) or not bool(visible.all()) or not bool(valid_3d.all()):
        raise ValueError("Current SUPER query must contain ten visible, valid points")
    return frame_indices, point_ids, pixels


def export_tracks(
    spec,
    calibration,
    gaussians,
    learned_mask,
    pipe,
    background,
    alpha_threshold,
    render_scale,
    output,
):
    frames = int(spec["frames"])
    ground_truth = resolve_path(REPO_ROOT, spec["ground_truth"])
    frame_indices, point_ids, query_pixels_full = query_inputs(ground_truth)
    query_camera, query_intrinsic = make_render_camera(
        calibration, QUERY_FRAME, frames, render_scale
    )
    query_pixels = query_pixels_full * float(render_scale)
    ones = torch.ones(
        (gaussians.get_xyz.shape[0], 3),
        dtype=gaussians.get_xyz.dtype,
        device=gaussians.get_xyz.device,
    )
    query_render = render(
        query_camera, gaussians, pipe, background, learned_mask, override_color=ones
    )
    query_alpha = sample_chw(query_render["render"][0:1], query_pixels)[:, 0]
    query_depth_premultiplied = sample_chw(query_render["depth"], query_pixels)[:, 0]
    query_depth = query_depth_premultiplied / query_alpha.clamp_min(1.0e-8)
    query_valid = (
        (query_alpha > float(alpha_threshold))
        & torch.isfinite(query_depth)
        & (query_depth > 0.0)
    )
    if int(query_valid.sum().item()) != len(point_ids):
        raise RuntimeError("Query alpha/depth coverage is {}/10".format(int(query_valid.sum().item())))
    intrinsic = torch.as_tensor(query_intrinsic, dtype=query_depth.dtype, device=query_depth.device)
    pixels_tensor = torch.as_tensor(query_pixels, dtype=query_depth.dtype, device=query_depth.device)
    camera_xyz = torch.stack(
        (
            (pixels_tensor[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0] * query_depth,
            (pixels_tensor[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1] * query_depth,
            query_depth,
        ),
        dim=1,
    )
    world_from_camera = torch.as_tensor(
        internal_world_from_camera(calibration["X_WC_opencv"]),
        dtype=query_depth.dtype,
        device=query_depth.device,
    )
    query_world = torch.einsum(
        "ij,nj->ni",
        world_from_camera[:3],
        torch_functional.pad(camera_xyz, (0, 1), value=1.0),
    )
    query_centers = deformed_centers(
        gaussians, learned_mask, normalized_time(QUERY_FRAME, frames)
    )
    xyz_span = gaussians.get_xyz.max(dim=0).values - gaussians.get_xyz.min(dim=0).values
    feature_scale = xyz_span.max().clamp_min(1.0e-6)
    positions_world_m = np.full(
        (len(frame_indices), len(point_ids), 3), np.nan, dtype=np.float32
    )
    for local_index, frame in enumerate(frame_indices):
        target = deformed_centers(
            gaussians, learned_mask, normalized_time(int(frame), frames)
        )
        encoded = (target - query_centers) / feature_scale
        displacement_image = render(
            query_camera,
            gaussians,
            pipe,
            background,
            learned_mask,
            override_color=encoded,
        )["render"]
        displacement = (
            sample_chw(displacement_image, query_pixels)
            / query_alpha[:, None].clamp_min(1.0e-8)
            * feature_scale
        )
        decoded = query_world + displacement
        decoded[~query_valid] = torch.nan
        positions_world_m[local_index] = (
            decoded.detach().cpu().numpy().astype(np.float32) / INTERNAL_UNITS_PER_METER
        )
        print("[track] {}/{} frame={}".format(
            local_index + 1, len(frame_indices), int(frame)
        ), flush=True)
    uv = np.empty((len(frame_indices), len(point_ids), 2), dtype=np.float32)
    xyz_camera_m = np.empty((len(frame_indices), len(point_ids), 3), dtype=np.float32)
    valid = np.empty((len(frame_indices), len(point_ids)), dtype=bool)
    for local_index in range(len(frame_indices)):
        uv[local_index], valid[local_index], xyz_camera_m[local_index] = project_world_points(
            positions_world_m[local_index],
            calibration["K"],
            calibration["X_WC_opencv"],
            calibration["resolution_wh"],
        )
    initial_error = np.linalg.norm(
        uv[frame_indices == QUERY_FRAME][0] - query_pixels_full, axis=1
    )
    if not np.all(np.isfinite(initial_error)) or float(initial_error.max()) > 1.0e-3:
        raise RuntimeError("Query anchor reprojection failed: {} px".format(float(initial_error.max())))
    np.savez_compressed(
        str(output / "predicted_tracks.npz"),
        schema=np.asarray("super_tissue_predicted_tracks_v1"),
        frame_indices=frame_indices,
        observation_used=np.zeros(len(frame_indices), dtype=bool),
        uv=uv,
        xyz_camera_m=xyz_camera_m,
        xyz_world_m=positions_world_m,
        query_frame_index=np.asarray(QUERY_FRAME, dtype=np.int32),
        point_ids=point_ids,
        query_pixels_full_resolution=query_pixels_full,
        query_alpha=query_alpha.detach().cpu().numpy().astype(np.float32),
        query_depth_eh_surgs_m=(
            query_depth.detach().cpu().numpy() / INTERNAL_UNITS_PER_METER
        ).astype(np.float32),
    )
    return {
        "decoder_version": TRACK_DECODER_VERSION,
        "query_point_count": len(point_ids),
        "query_valid_count": int(query_valid.sum().item()),
        "initial_query_reprojection_max_px": float(initial_error.max()),
        "feature_scale_internal_units": float(feature_scale.item()),
        "alignment": "none",
    }


def export_renders(
    dataset_key,
    spec,
    calibration,
    gaussians,
    learned_mask,
    pipe,
    background,
    render_scale,
    output,
):
    selected = rendering_frames(dataset_key)
    raw = output / "render_predictions_float32"
    raw.mkdir()
    frames = int(spec["frames"])
    for local_index, frame in enumerate(selected):
        camera, _ = make_render_camera(calibration, frame, frames, render_scale)
        prediction_tensor = render(
            camera, gaussians, pipe, background, learned_mask
        )["render"]
        prediction = (
            prediction_tensor.detach()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        np.save(str(raw / "{:06d}.npy".format(frame)), prediction)
        print("[render] {}/{} frame={}".format(local_index + 1, len(selected), frame), flush=True)
    evaluator = Path(os.environ.get(
        "EVAL_PYTHON", "/Media_HDD/jwshan/conda_envs/eg_codex/bin/python"
    ))
    evaluator_root = evaluator.parents[1]
    evaluator_env = os.environ.copy()
    evaluator_env.pop("PYTHONPATH", None)
    evaluator_env["PYTHONNOUSERSITE"] = "1"
    evaluator_env["PATH"] = str(evaluator.parent) + os.pathsep + "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    evaluator_env["LD_LIBRARY_PATH"] = str(evaluator_root / "lib")
    subprocess.check_call([
        str(evaluator),
        str(REPO_ROOT / "scripts" / "prepare_endogaussian_super_render_metrics.py"),
        "--dataset-key",
        dataset_key,
        "--prediction-dir",
        str(raw),
        "--capture",
        str(output),
        "--render-scale",
        str(render_scale),
        "--delete-prediction-arrays",
    ], env=evaluator_env)
    return len(selected)


def main():
    args, model, pipeline, hidden = parse_args()
    spec = dataset_spec(args.dataset_key)
    ground_truth = resolve_path(REPO_ROOT, spec["ground_truth"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    calibration = load_calibration(REPO_ROOT, args.dataset_key)
    gaussians, learned_mask, iteration = load_trained_gaussians(
        Path(args.model_path), args.iteration, int(args.sh_degree), hidden.extract(args)
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        track_report = export_tracks(
            spec,
            calibration,
            gaussians,
            learned_mask,
            pipeline.extract(args),
            background,
            args.alpha_threshold,
            args.render_scale,
            args.output_dir,
        )
        render_count = 0 if args.tracks_only else export_renders(
            args.dataset_key,
            spec,
            calibration,
            gaussians,
            learned_mask,
            pipeline.extract(args),
            background,
            args.render_scale,
            args.output_dir,
        )
    metadata = {
        "schema": "super_tissue_physics_benchmark_v1",
        "protocol": PROTOCOL,
        "ground_truth": str(ground_truth),
        "ground_truth_sha256": sha256_file(ground_truth),
        "trajectory_binding": "Shape-of-Motion-style query-anchored EH-SurGS displacement field",
        "trajectory_query_frame": QUERY_FRAME,
        "reconstruction_split": {
            "ratio": [7, 1],
            "test_rule": "full video frame_index % 8 == {}".format(HOLDOUT_PHASE),
            "test_end_exclusive": int(spec["future_start"]),
        },
        "future_split": {
            "train_end_inclusive": int(spec["future_start"]) - 1,
            "test_start_inclusive": int(spec["future_start"]),
            "ground_truth_formal_test_start": int(spec["future_start"]),
        },
        "render_scale": args.render_scale,
        "render_camera": "stereo_left",
        "render_mask": "all pixels except the frozen SurgicalSAM2 instrument mask",
        "capture_renders": not args.tracks_only,
        "baseline": {
            "method": "EH-SurGS",
            "adapter_version": ADAPTER_VERSION,
            "eh_surgs_commit": UPSTREAM_COMMIT,
            "shape_of_motion_commit": SHAPE_OF_MOTION_COMMIT,
            "shape_of_motion_usage": "trajectory decoder design only",
            "checkpoint_iteration": iteration,
            "time_normalization": "frame_index / full_sequence_frame_count",
            "future_behavior": "native EH-SurGS temporal deformation field extrapolation",
            "camera_projection": "full non-centered SUPER K in the external camera adapter",
            "training_mask": "all depth-valid pixels outside the frozen SurgicalSAM2 instrument mask",
            "ground_truth_usage": "frame-0 2D query pixels after checkpoint freeze and downstream scoring only",
            "future_observations_used": False,
            "track_report": track_report,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "capture_summary.json").write_text(
        json.dumps({
            "schema": "eh_surgs_super_capture_v1",
            "protocol": PROTOCOL,
            "track_frame_count": len(query_inputs(ground_truth)[0]),
            "render_frame_count": render_count,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
