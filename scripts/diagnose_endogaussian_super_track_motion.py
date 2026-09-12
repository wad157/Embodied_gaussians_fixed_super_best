#!/usr/bin/env python3
"""Diagnose whether exported SUPER tracks reflect an EndoGaussian checkpoint.

This is a read-only diagnostic.  It compares the alpha-composited trajectory
decoder with individual Gaussians near each query ray.  Oracle selections use
the full ground-truth trajectory and are reported only as a diagnostic upper
bound; they must never be used for formal benchmark scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

# When this file is launched directly, Python puts ``scripts/`` before the
# upstream EndoGaussian checkout and would import scripts/utils.py as ``utils``.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if not entry or Path(entry).resolve() != SCRIPT_DIR
]

from arguments import ModelHiddenParams, ModelParams, PipelineParams
from utils.params_utils import merge_hparams

from common import load_trained_gaussians
from protocol import INTERNAL_UNITS_PER_METER, normalized_time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    PipelineParams(parser)
    hidden = ModelHiddenParams(parser)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--candidate-radius-px", type=float, default=8.0)
    parser.add_argument("--candidate-depth-mm", type=float, default=15.0)
    args = parser.parse_args()
    import mmcv

    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    return args, model, hidden


def deform_indices(gaussians, indices: torch.Tensor, time: float) -> torch.Tensor:
    means = gaussians.get_xyz[indices]
    selected = gaussians._deformation_table[indices]
    result = means.clone()
    if bool(selected.any()):
        times = torch.full(
            (int(selected.sum().item()), 1),
            float(time),
            dtype=means.dtype,
            device=means.device,
        )
        deformed, _, _, _ = gaussians._deformation(
            means[selected],
            gaussians._scaling[indices][selected],
            gaussians._rotation[indices][selected],
            gaussians._opacity[indices][selected],
            times,
        )
        result[selected] = deformed
    return result


def world_to_camera(points_world_m: torch.Tensor, world_from_camera: np.ndarray):
    camera_from_world = torch.as_tensor(
        np.linalg.inv(world_from_camera),
        dtype=points_world_m.dtype,
        device=points_world_m.device,
    )
    homogeneous = torch.nn.functional.pad(points_world_m, (0, 1), value=1.0)
    return torch.einsum("ij,nj->ni", camera_from_world[:3], homogeneous)


def project(points_world_m: torch.Tensor, K: np.ndarray, world_from_camera: np.ndarray):
    camera = world_to_camera(points_world_m, world_from_camera)
    intrinsic = torch.as_tensor(K, dtype=camera.dtype, device=camera.device)
    image_h = torch.einsum("ij,nj->ni", intrinsic, camera)
    uv = image_h[:, :2] / image_h[:, 2:].clamp_min(1.0e-10)
    return uv, camera[:, 2]


def percentile_dict(values: torch.Tensor):
    qs = torch.tensor([0.5, 0.9, 0.95, 0.99], device=values.device)
    out = torch.quantile(values, qs).detach().cpu().numpy()
    return {
        "p50": float(out[0]),
        "p90": float(out[1]),
        "p95": float(out[2]),
        "p99": float(out[3]),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
    }


def main():
    args, model, hidden = parse_args()
    gaussians, iteration = load_trained_gaussians(
        Path(args.model_path), args.iteration, int(args.sh_degree), hidden.extract(args)
    )
    gaussians._deformation.eval()

    with np.load(args.ground_truth, allow_pickle=False) as archive:
        frames = np.asarray(archive["frame_indices"], dtype=np.int32)
        gt_uv = np.asarray(archive["uv"], dtype=np.float32)
        gt_world = np.asarray(archive["xyz_world_m"], dtype=np.float32)
        K = np.asarray(archive["K_left_rect"], dtype=np.float64)
        world_from_camera = np.asarray(
            archive["X_world_camera_opencv"], dtype=np.float64
        )
    with np.load(args.capture / "predicted_tracks.npz", allow_pickle=False) as archive:
        pred_uv = np.asarray(archive["uv"], dtype=np.float32)
        pred_world = np.asarray(archive["xyz_world_m"], dtype=np.float32)

    query_slot = int(np.flatnonzero(frames == 0)[0])
    gt_rel_2d = gt_uv - gt_uv[query_slot : query_slot + 1]
    gt_rel_3d = gt_world - gt_world[query_slot : query_slot + 1]
    pred_rel_2d = pred_uv - pred_uv[query_slot : query_slot + 1]
    pred_rel_3d = pred_world - pred_world[query_slot : query_slot + 1]

    device = gaussians.get_xyz.device
    all_indices = torch.arange(gaussians.get_xyz.shape[0], device=device)
    with torch.inference_mode():
        centers0 = deform_indices(
            gaussians, all_indices, normalized_time(0, int(frames[-1]) + 1)
        )
        centers0_m = centers0 / INTERNAL_UNITS_PER_METER
        center_uv, center_depth = project(centers0_m, K, world_from_camera)

        gt0_world = torch.as_tensor(gt_world[query_slot], device=device)
        pred0_world = torch.as_tensor(pred_world[query_slot], device=device)
        nearest_gt = torch.cdist(gt0_world, centers0_m).argmin(dim=1)
        nearest_pred = torch.cdist(pred0_world, centers0_m).argmin(dim=1)

        # Candidate Gaussians are chosen using query-frame geometry only.
        candidate_lists = []
        for point in range(gt_uv.shape[1]):
            screen_distance = torch.linalg.norm(
                center_uv - torch.as_tensor(gt_uv[query_slot, point], device=device),
                dim=1,
            )
            gt_camera = world_to_camera(gt0_world[point : point + 1], world_from_camera)
            depth_distance_mm = torch.abs(center_depth - gt_camera[0, 2]) * 1000.0
            candidates = torch.nonzero(
                (screen_distance <= float(args.candidate_radius_px))
                & (depth_distance_mm <= float(args.candidate_depth_mm))
                & (center_depth > 0),
                as_tuple=False,
            ).flatten()
            candidate_lists.append(candidates)
        candidate_union = torch.unique(
            torch.cat(candidate_lists + [nearest_gt, nearest_pred]), sorted=True
        )
        union_lookup = {int(v): i for i, v in enumerate(candidate_union.cpu().tolist())}
        union_tracks = []
        for frame in frames:
            union_tracks.append(
                deform_indices(
                    gaussians,
                    candidate_union,
                    normalized_time(int(frame), int(frames[-1]) + 1),
                )
                / INTERNAL_UNITS_PER_METER
            )
        union_tracks = torch.stack(union_tracks, dim=0)
        union_displacement = union_tracks - union_tracks[query_slot : query_slot + 1]

        gt_relative = torch.as_tensor(gt_rel_3d, device=device)
        point_reports = []
        oracle_rel_errors = []
        nearest_gt_rel_errors = []
        nearest_pred_rel_errors = []
        for point, candidates in enumerate(candidate_lists):
            local_candidate_indices = torch.as_tensor(
                [union_lookup[int(v)] for v in candidates.cpu().tolist()],
                dtype=torch.long,
                device=device,
            )
            gt_point_relative = gt_relative[:, point]
            if len(local_candidate_indices):
                candidate_errors = torch.linalg.norm(
                    union_displacement[:, local_candidate_indices]
                    - gt_point_relative[:, None, :],
                    dim=-1,
                ).mean(dim=0)
                best_local = int(candidate_errors.argmin().item())
                oracle_gaussian = int(candidates[best_local].item())
                oracle_error_mm = float(candidate_errors[best_local].item() * 1000.0)
            else:
                oracle_gaussian = None
                oracle_error_mm = None

            def one_error(index):
                rel = union_displacement[:, union_lookup[int(index)]]
                return float(
                    torch.linalg.norm(rel - gt_point_relative, dim=-1).mean().item()
                    * 1000.0
                )

            gt_nearest_index = int(nearest_gt[point].item())
            pred_nearest_index = int(nearest_pred[point].item())
            gt_nearest_error = one_error(gt_nearest_index)
            pred_nearest_error = one_error(pred_nearest_index)
            if oracle_error_mm is not None:
                oracle_rel_errors.append(oracle_error_mm)
            nearest_gt_rel_errors.append(gt_nearest_error)
            nearest_pred_rel_errors.append(pred_nearest_error)
            point_reports.append(
                {
                    "point_id": point,
                    "candidate_count": int(len(candidates)),
                    "nearest_to_gt_query_gaussian": gt_nearest_index,
                    "nearest_to_gt_query_distance_mm": float(
                        torch.linalg.norm(centers0_m[gt_nearest_index] - gt0_world[point]).item()
                        * 1000.0
                    ),
                    "nearest_to_gt_relative_motion_error_mm": gt_nearest_error,
                    "nearest_to_model_query_gaussian": pred_nearest_index,
                    "nearest_to_model_query_distance_mm": float(
                        torch.linalg.norm(centers0_m[pred_nearest_index] - pred0_world[point]).item()
                        * 1000.0
                    ),
                    "nearest_to_model_relative_motion_error_mm": pred_nearest_error,
                    "oracle_query_local_gaussian": oracle_gaussian,
                    "oracle_relative_motion_error_mm": oracle_error_mm,
                }
            )

        gt_motion_px = np.linalg.norm(gt_rel_2d, axis=-1)
        peak_slots = np.unique(np.argmax(gt_motion_px, axis=0))
        diagnostic_slots = np.unique(
            np.concatenate(
                [peak_slots, np.asarray([query_slot, len(frames) - 1]), np.flatnonzero(frames >= 1152)[:1]]
            )
        )
        global_deformation = {}
        for slot in diagnostic_slots:
            centers = deform_indices(
                gaussians,
                all_indices,
                normalized_time(int(frames[slot]), int(frames[-1]) + 1),
            )
            displacement_mm = (
                torch.linalg.norm(centers - centers0, dim=1)
                / INTERNAL_UNITS_PER_METER
                * 1000.0
            )
            global_deformation[str(int(frames[slot]))] = percentile_dict(displacement_mm)

    official_rel_error_2d = np.linalg.norm(pred_rel_2d - gt_rel_2d, axis=-1)
    official_rel_error_3d = np.linalg.norm(pred_rel_3d - gt_rel_3d, axis=-1) * 1000.0
    report = {
        "schema": "endogaussian_super_track_motion_diagnostic_v1",
        "warning": "Oracle Gaussian results use all ground-truth frames and are diagnostic only.",
        "dataset_key": args.dataset_key,
        "checkpoint_iteration": iteration,
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "deformable_gaussian_count": int(gaussians._deformation_table.sum().item()),
        "official_export": {
            "predicted_displacement_2d_mean_px": float(
                np.linalg.norm(pred_rel_2d, axis=-1).mean()
            ),
            "ground_truth_displacement_2d_mean_px": float(gt_motion_px.mean()),
            "relative_motion_error_2d_mean_px": float(official_rel_error_2d.mean()),
            "predicted_displacement_3d_mean_mm": float(
                np.linalg.norm(pred_rel_3d, axis=-1).mean() * 1000.0
            ),
            "ground_truth_displacement_3d_mean_mm": float(
                np.linalg.norm(gt_rel_3d, axis=-1).mean() * 1000.0
            ),
            "relative_motion_error_3d_mean_mm": float(official_rel_error_3d.mean()),
        },
        "query_local_single_gaussian": {
            "candidate_rule": {
                "screen_radius_px": float(args.candidate_radius_px),
                "gt_depth_radius_mm": float(args.candidate_depth_mm),
            },
            "nearest_to_gt_query_relative_motion_error_mm_mean": float(
                np.mean(nearest_gt_rel_errors)
            ),
            "nearest_to_model_query_relative_motion_error_mm_mean": float(
                np.mean(nearest_pred_rel_errors)
            ),
            "oracle_relative_motion_error_mm_mean": float(np.mean(oracle_rel_errors))
            if oracle_rel_errors
            else None,
            "points": point_reports,
        },
        "global_gaussian_displacement_mm": global_deformation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
