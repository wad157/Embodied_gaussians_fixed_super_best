#!/usr/bin/env python3
"""Track projected SUPER surface particles with dense AllTracker flow.

AllTracker predicts a dense query-frame-to-video correspondence field.  This
tool samples that field exactly at projected physical surface particles, so the
runtime binding is one material particle per track instead of an overlapping
6 mm range.  Native visibility/confidence is multiplied by a local spatial
consistency score and dynamic tissue-mask membership before any 3D update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
ALLTRACKER = REPO / "third_party/AllTracker"
DEFAULT_VIDEO = REPO / "data/super/grasp5_offline_demo/videos/stereo_left.mp4"
DEFAULT_ASSET = (
    REPO
    / "data/super/grasp5_native/tissue_multiview_v1/"
    "paper_pbd_tissue_v15_denser_mild_paper_constraints_centroid_ellipsoids/"
    "tissue_paper_pbd_centroid_gaussians.npz"
)
DEFAULT_DEPTH = (
    REPO
    / "data/super/grasp5_native/depth_v4_foundation_dense_timestamped/"
    "000000-depth.npy"
)
DEFAULT_TISSUE_MASK = REPO / "data/super/grasp5_native/masks/000000-tissue.png"
DEFAULT_TOOL_MASK = (
    REPO
    / "data/super/grasp5_native/tissue_multiview_v1/masks_left/"
    "tool_dilated/000000-tool-dilated.png"
)
DEFAULT_DYNAMIC_MASKS = (
    REPO
    / "data/super/grasp5_native/visual_force_masks_v1/"
    "tissue_masks_packbits.npy"
)
DEFAULT_CALIBRATION = REPO / "data/super/grasp5_native/calib_rectified.json"
DEFAULT_TABLE_FRAME = REPO / "data/super/table_frame.json"
DEFAULT_CHECKPOINT = ALLTRACKER / "checkpoints/alltracker.pth"
DEFAULT_OUTPUT = REPO / "outputs/grasp5_alltracker_surface_20260829_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--tissue-mask", type=Path, default=DEFAULT_TISSUE_MASK)
    parser.add_argument("--tool-mask", type=Path, default=DEFAULT_TOOL_MASK)
    parser.add_argument(
        "--dynamic-tissue-masks", type=Path, default=DEFAULT_DYNAMIC_MASKS
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--table-frame", type=Path, default=DEFAULT_TABLE_FRAME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--maximum-source-frame", type=int, default=1150)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument(
        "--primary-segment-length",
        type=int,
        default=0,
        help=(
            "Maximum sampled-frame span of one primary AllTracker call. "
            "Zero keeps the original single long-range call. Positive values "
            "continue from the shared boundary-frame query and avoid PyTorch's "
            "32-bit tensor-index limit on long stride-1 videos."
        ),
    )
    parser.add_argument(
        "--reanchor-interval",
        type=int,
        default=32,
        help=(
            "Number of sampled frames between causal dense-flow reanchors. "
            "A value of 32 is about 2.15 s at the default sampled FPS."
        ),
    )
    parser.add_argument(
        "--cross-anchor-agreement-scale-px",
        type=float,
        default=6.0,
        help=(
            "Cauchy scale for disagreement between long-range and causally "
            "reanchored AllTracker estimates."
        ),
    )
    parser.add_argument(
        "--trajectory-source",
        choices=("primary", "reanchored"),
        default="primary",
        help=(
            "Which causal AllTracker trajectory is exported for flow/depth. "
            "The other trajectory remains an independent confidence check."
        ),
    )
    parser.add_argument("--inference-iterations", type=int, default=4)
    parser.add_argument("--maximum-initial-depth-error-mm", type=float, default=3.0)
    parser.add_argument("--native-confidence-threshold", type=float, default=0.10)
    parser.add_argument("--local-neighbours", type=int, default=8)
    parser.add_argument(
        "--query-mode",
        choices=("projected_particles", "tissue_grid"),
        default="projected_particles",
        help=(
            "projected_particles preserves the original sparse material queries; "
            "tissue_grid samples the same dense AllTracker field on a uniform "
            "frame-0 tissue grid and fixes every query to the nearest surface."
        ),
    )
    parser.add_argument(
        "--query-spacing-px",
        type=int,
        default=12,
        help="Full-resolution pixel spacing for tissue_grid queries.",
    )
    parser.add_argument(
        "--maximum-initial-surface-distance-mm",
        type=float,
        default=5.0,
        help=(
            "Maximum lifted-depth to physical-surface distance for tissue_grid "
            "queries. This is an initialization gate, not per-frame rebinding."
        ),
    )
    parser.add_argument(
        "--maximum-queries-per-particle",
        type=int,
        default=0,
        help=(
            "For tissue_grid, retain at most this many image queries per nearest "
            "surface particle, ranked by initial 3D distance; zero keeps all."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def five_number(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("nan")] * 5
    return np.percentile(values, [0, 5, 50, 95, 100]).tolist()


def read_sampled_video(
    path: Path,
    *,
    frame_stride: int,
    maximum_source_frame: int,
    image_size: int,
) -> tuple[torch.Tensor, np.ndarray, dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = min(image_size / source_height, image_size / source_width)
    model_height = max(8, int(source_height * scale) // 8 * 8)
    model_width = max(8, int(source_width * scale) // 8 * 8)
    frames: list[torch.Tensor] = []
    source_indices: list[int] = []
    source_index = 0
    while source_index <= maximum_source_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % frame_stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(
                rgb,
                (model_width, model_height),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
            source_indices.append(source_index)
        source_index += 1
    capture.release()
    if len(frames) < 2:
        raise RuntimeError("AllTracker needs at least two sampled frames")
    metadata: dict[str, float | int] = {
        "source_width": source_width,
        "source_height": source_height,
        "source_fps": source_fps,
        "source_frame_count": source_count,
        "model_width": model_width,
        "model_height": model_height,
        "sampled_fps": source_fps / frame_stride,
        "sampled_frame_count": len(frames),
        "frame_stride": frame_stride,
    }
    return (
        torch.stack(frames, dim=0)[None].float(),
        np.asarray(source_indices, dtype=np.int32),
        metadata,
    )


def project_table_points(
    points_table: np.ndarray,
    intrinsic: np.ndarray,
    x_camera_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.column_stack(
        (points_table, np.ones(len(points_table), dtype=np.float64))
    )
    camera = (x_camera_table @ homogeneous.T).T[:, :3]
    projected = (intrinsic @ camera.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    return pixels, camera[:, 2]


def sample_nearest(image: np.ndarray, pixels_uv: np.ndarray) -> np.ndarray:
    xy = np.rint(pixels_uv).astype(np.int64)
    x = np.clip(xy[:, 0], 0, image.shape[1] - 1)
    y = np.clip(xy[:, 1], 0, image.shape[0] - 1)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < image.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < image.shape[0])
    )
    result = np.zeros(len(pixels_uv), dtype=image.dtype)
    result[inside] = image[y[inside], x[inside]]
    return result


def backproject_to_table(
    pixels_uv: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    x_table_camera: np.ndarray,
) -> np.ndarray:
    """Lift rectified pixels into the fixed SUPER table frame."""

    pixels_uv = np.asarray(pixels_uv, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    homogeneous_camera = np.column_stack(
        (
            (pixels_uv[:, 0] - intrinsic[0, 2])
            * depth_m
            / intrinsic[0, 0],
            (pixels_uv[:, 1] - intrinsic[1, 2])
            * depth_m
            / intrinsic[1, 1],
            depth_m,
            np.ones(len(depth_m), dtype=np.float64),
        )
    )
    return (x_table_camera @ homogeneous_camera.T).T[:, :3]


def tissue_grid_queries(
    *,
    tissue_mask: np.ndarray,
    tool_mask: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    x_table_camera: np.ndarray,
    surface_positions_table: np.ndarray,
    surface_particle_ids: np.ndarray,
    spacing_px: int,
    maximum_surface_distance_m: float,
    maximum_queries_per_particle: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dense, annotation-free image queries with immutable material ids."""

    height, width = tissue_mask.shape
    offset = spacing_px // 2
    x = np.arange(offset, width, spacing_px, dtype=np.int64)
    y = np.arange(offset, height, spacing_px, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(x, y)
    pixels = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(
        np.float64
    )
    sampled_tissue = sample_nearest(tissue_mask, pixels) > 0
    sampled_tool = sample_nearest(tool_mask, pixels) > 0
    sampled_depth = sample_nearest(depth, pixels).astype(np.float64)
    requested = (
        sampled_tissue
        & ~sampled_tool
        & np.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
    )
    pixels = pixels[requested]
    sampled_depth = sampled_depth[requested]
    lifted = backproject_to_table(
        pixels, sampled_depth, intrinsic, x_table_camera
    )
    squared_distance = np.sum(
        (
            lifted[:, None, :]
            - np.asarray(surface_positions_table, dtype=np.float64)[None, :, :]
        )
        ** 2,
        axis=2,
    )
    nearest_slot = np.argmin(squared_distance, axis=1)
    nearest_distance = np.sqrt(
        squared_distance[np.arange(len(lifted)), nearest_slot]
    )
    selected = nearest_distance <= maximum_surface_distance_m
    if maximum_queries_per_particle > 0:
        keep = np.zeros(len(pixels), dtype=bool)
        selected_slots = np.flatnonzero(selected)
        selected_particle_slots = nearest_slot[selected_slots]
        for particle_slot in np.unique(selected_particle_slots):
            local = selected_slots[selected_particle_slots == particle_slot]
            ranked = local[np.argsort(nearest_distance[local])]
            keep[ranked[:maximum_queries_per_particle]] = True
        selected &= keep
    return (
        pixels[selected],
        np.asarray(surface_particle_ids, dtype=np.int32)[nearest_slot[selected]],
        lifted[selected],
    )


def sample_dynamic_mask_membership(
    packed_path: Path,
    source_frame_indices: np.ndarray,
    tracks_original: np.ndarray,
) -> np.ndarray:
    if not packed_path.is_file():
        return np.ones(tracks_original.shape[:2], dtype=bool)
    packed = np.load(packed_path, mmap_mode="r")
    height = packed.shape[1]
    width = packed.shape[2] * 8
    inside = np.zeros(tracks_original.shape[:2], dtype=bool)
    for sampled_index, source_index in enumerate(source_frame_indices):
        xy = np.rint(tracks_original[sampled_index]).astype(np.int64)
        x = np.clip(xy[:, 0], 0, width - 1)
        y = np.clip(xy[:, 1], 0, height - 1)
        byte_values = packed[source_index, y, x // 8]
        bit_offsets = 7 - (x % 8)
        inside[sampled_index] = (
            np.right_shift(byte_values, bit_offsets) & 1
        ).astype(bool)
    return inside


def local_spatial_consistency(
    tracks: np.ndarray,
    rest_positions: np.ndarray,
    neighbour_count: int,
) -> np.ndarray:
    point_count = len(rest_positions)
    if point_count < 2:
        return np.ones(tracks.shape[:2], dtype=np.float32)
    neighbour_count = min(max(1, neighbour_count), point_count - 1)
    distance = np.linalg.norm(
        rest_positions[:, None, :] - rest_positions[None, :, :], axis=2
    )
    neighbours = np.argsort(distance, axis=1)[:, 1 : neighbour_count + 1]
    displacement = tracks - tracks[0:1]
    neighbour_displacement = displacement[:, neighbours]
    local_median = np.median(neighbour_displacement, axis=2)
    residual = np.linalg.norm(displacement - local_median, axis=2)
    neighbour_deviation = np.linalg.norm(
        neighbour_displacement - local_median[:, :, None, :], axis=3
    )
    local_mad = np.median(neighbour_deviation, axis=2)
    # The two-pixel floor keeps genuine sharp deformation near the grasper;
    # the MAD term rejects isolated drift relative to material neighbours.
    scale = np.maximum(2.0, 2.0 + 3.0 * 1.4826 * local_mad)
    consistency = 1.0 / (1.0 + (residual / scale) ** 2)
    consistency[0] = 1.0
    return consistency.astype(np.float32)


def sample_dense_maps_at_points(
    maps: torch.Tensor, points_xy: np.ndarray
) -> np.ndarray:
    """Bilinearly sample ``(T,C,H,W)`` dense maps at material queries."""

    if maps.ndim != 4:
        raise ValueError("Dense maps must be shaped (T,C,H,W)")
    frame_count, _channels, height, width = maps.shape
    points = torch.as_tensor(
        points_xy, dtype=maps.dtype, device=maps.device
    )
    normalized = points.clone()
    normalized[:, 0] = 2.0 * normalized[:, 0] / max(width - 1, 1) - 1.0
    normalized[:, 1] = 2.0 * normalized[:, 1] / max(height - 1, 1) - 1.0
    grid = normalized[None, :, None, :].expand(frame_count, -1, -1, -1)
    sampled = F.grid_sample(
        maps,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[..., 0].permute(0, 2, 1).float().cpu().numpy()


def main() -> None:
    args = parse_args()
    required = (
        args.video,
        args.asset,
        args.depth,
        args.tissue_mask,
        args.tool_mask,
        args.calibration,
        args.table_frame,
        args.checkpoint,
        ALLTRACKER / "nets/alltracker.py",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists; pass --overwrite: {args.output_dir}"
        )
    if (
        args.frame_stride < 1
        or args.window_length < 2
        or args.primary_segment_length < 0
        or args.reanchor_interval < 2
        or args.cross_anchor_agreement_scale_px <= 0.0
        or args.query_spacing_px < 2
        or args.maximum_initial_surface_distance_mm <= 0.0
        or args.maximum_queries_per_particle < 0
    ):
        raise ValueError(
            "Frame stride, window length and reanchor interval must be positive; "
            "primary segment length must be non-negative"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("AllTracker surface tracking requires CUDA")

    rgbs, source_frames, metadata = read_sampled_video(
        args.video,
        frame_stride=args.frame_stride,
        maximum_source_frame=args.maximum_source_frame,
        image_size=args.image_size,
    )
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    table_frame = json.loads(args.table_frame.read_text(encoding="utf-8"))
    intrinsic = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    x_camera_table = np.asarray(
        table_frame["X_camera_table"], dtype=np.float64
    )
    with np.load(args.asset, allow_pickle=False) as loaded:
        rest_positions = loaded["rest_positions_table"].astype(np.float64)
        candidate_mask = loaded["visible_surface_mask"].astype(bool)
        fixed_mask = loaded["fixed_mask"].astype(bool)
    candidate_ids = np.flatnonzero(candidate_mask & ~fixed_mask)
    candidate_pixels, candidate_camera_depth = project_table_points(
        rest_positions[candidate_ids], intrinsic, x_camera_table
    )
    depth = np.load(args.depth, allow_pickle=False)
    tissue_mask = cv2.imread(str(args.tissue_mask), cv2.IMREAD_GRAYSCALE)
    tool_mask = cv2.imread(str(args.tool_mask), cv2.IMREAD_GRAYSCALE)
    if tissue_mask is None or tool_mask is None:
        raise RuntimeError("Could not load frame-0 tissue/tool masks")
    x_table_camera = np.asarray(table_frame["X_table_camera"], dtype=np.float64)
    if args.query_mode == "projected_particles":
        sampled_depth = sample_nearest(depth, candidate_pixels).astype(np.float64)
        sampled_tissue = sample_nearest(tissue_mask, candidate_pixels) > 0
        sampled_tool = sample_nearest(tool_mask, candidate_pixels) > 0
        depth_error_m = np.abs(sampled_depth - candidate_camera_depth)
        selected = (
            np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
            & sampled_tissue
            & ~sampled_tool
            & (
                depth_error_m
                <= args.maximum_initial_depth_error_mm * 1.0e-3
            )
        )
        particle_ids = candidate_ids[selected].astype(np.int32)
        projected_pixels = candidate_pixels[selected]
        query_material_points = rest_positions[particle_ids]
    else:
        (
            projected_pixels,
            particle_ids,
            query_material_points,
        ) = tissue_grid_queries(
            tissue_mask=tissue_mask,
            tool_mask=tool_mask,
            depth=depth,
            intrinsic=intrinsic,
            x_table_camera=x_table_camera,
            surface_positions_table=rest_positions[candidate_ids],
            surface_particle_ids=candidate_ids,
            spacing_px=args.query_spacing_px,
            maximum_surface_distance_m=(
                args.maximum_initial_surface_distance_mm * 1.0e-3
            ),
            maximum_queries_per_particle=(
                args.maximum_queries_per_particle
            ),
        )
    if len(particle_ids) < 100:
        raise RuntimeError(
            f"Only {len(particle_ids)} visible particles passed query gates"
        )

    source_width = int(metadata["source_width"])
    source_height = int(metadata["source_height"])
    model_width = int(metadata["model_width"])
    model_height = int(metadata["model_height"])
    scale_xy = np.asarray(
        [model_width / source_width, model_height / source_height],
        dtype=np.float64,
    )
    query_xy = np.rint(projected_pixels * scale_xy).astype(np.int64)
    query_xy[:, 0] = np.clip(query_xy[:, 0], 0, model_width - 1)
    query_xy[:, 1] = np.clip(query_xy[:, 1], 0, model_height - 1)

    # This repository also has ``scripts/utils.py``.  Some imported packages
    # load it under the top-level name ``utils`` before AllTracker is added to
    # sys.path, while the official code expects its own ``utils/`` namespace.
    loaded_utils = sys.modules.get("utils")
    loaded_utils_path = str(getattr(loaded_utils, "__file__", ""))
    if loaded_utils is not None and str(ALLTRACKER) not in loaded_utils_path:
        for module_name in tuple(sys.modules):
            if module_name == "utils" or module_name.startswith("utils."):
                del sys.modules[module_name]
    sys.path.insert(0, str(ALLTRACKER))
    from nets.alltracker import Net  # noqa: E402

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Net(args.window_length, init_weights=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rgbs = rgbs.to(device, non_blocking=True)
    sampled_frame_count = int(rgbs.shape[1])
    query_model = query_xy.astype(np.float32)
    primary_started = time.perf_counter()
    primary_segment_count = 1
    if (
        args.primary_segment_length > 0
        and args.primary_segment_length < sampled_frame_count - 1
    ):
        primary_tracks_model = np.zeros(
            (sampled_frame_count, len(query_xy), 2), dtype=np.float32
        )
        sampled_visconf = np.zeros(
            (sampled_frame_count, len(query_xy), 2), dtype=np.float32
        )
        primary_tracks_model[0] = query_model
        primary_anchor_model = query_model.copy()
        primary_segment_count = 0
        primary_segment_start = 0
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            while primary_segment_start < sampled_frame_count - 1:
                primary_segment_end = min(
                    sampled_frame_count - 1,
                    primary_segment_start + args.primary_segment_length,
                )
                primary_flows, primary_visconfs, _, _ = model.forward_sliding(
                    rgbs[:, primary_segment_start : primary_segment_end + 1],
                    iters=args.inference_iterations,
                    sw=None,
                    is_training=False,
                )
                primary_flow = sample_dense_maps_at_points(
                    primary_flows[0], primary_anchor_model
                )
                local_visconf = sample_dense_maps_at_points(
                    primary_visconfs[0], primary_anchor_model
                )
                local_tracks = primary_anchor_model[None] + primary_flow
                write_start = 0 if primary_segment_start == 0 else 1
                output_slice = slice(
                    primary_segment_start + write_start,
                    primary_segment_end + 1,
                )
                primary_tracks_model[output_slice] = local_tracks[write_start:]
                sampled_visconf[output_slice] = local_visconf[write_start:]
                primary_anchor_model = local_tracks[-1].astype(np.float32)
                primary_segment_start = primary_segment_end
                primary_segment_count += 1
                del primary_flows, primary_visconfs, primary_flow
    else:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            primary_flows, primary_visconfs, _, _ = model.forward_sliding(
                rgbs,
                iters=args.inference_iterations,
                sw=None,
                is_training=False,
            )
        primary_flow = sample_dense_maps_at_points(
            primary_flows[0], query_model
        )
        sampled_visconf = sample_dense_maps_at_points(
            primary_visconfs[0], query_model
        )
        primary_tracks_model = query_model[None] + primary_flow
        del primary_flows, primary_visconfs, primary_flow
    primary_elapsed = time.perf_counter() - primary_started

    reanchored_tracks_model = np.zeros(
        (sampled_frame_count, len(query_xy), 2), dtype=np.float32
    )
    reanchored_visconf = np.zeros(
        (sampled_frame_count, len(query_xy), 2), dtype=np.float32
    )
    reanchored_tracks_model[0] = query_model
    anchor_model = query_model.copy()
    segment_count = 0
    segment_start = 0
    reanchored_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        while segment_start < sampled_frame_count - 1:
            segment_end = min(
                sampled_frame_count - 1,
                segment_start + args.reanchor_interval,
            )
            flows, visconfs, _, _ = model.forward_sliding(
                rgbs[:, segment_start : segment_end + 1],
                iters=args.inference_iterations,
                sw=None,
                is_training=False,
            )
            local_flow = sample_dense_maps_at_points(flows[0], anchor_model)
            local_visconf = sample_dense_maps_at_points(
                visconfs[0], anchor_model
            )
            local_tracks = anchor_model[None] + local_flow
            write_start = 0 if segment_start == 0 else 1
            output_slice = slice(
                segment_start + write_start, segment_end + 1
            )
            reanchored_tracks_model[output_slice] = local_tracks[write_start:]
            reanchored_visconf[output_slice] = local_visconf[write_start:]
            anchor_model = local_tracks[-1].astype(np.float32)
            segment_start = segment_end
            segment_count += 1
    reanchored_elapsed = time.perf_counter() - reanchored_started
    elapsed = primary_elapsed + reanchored_elapsed
    del rgbs, model, checkpoint, flows, visconfs
    torch.cuda.empty_cache()
    primary_tracks_original = primary_tracks_model / scale_xy[None, None, :]
    reanchored_tracks_original = (
        reanchored_tracks_model / scale_xy[None, None, :]
    )
    tracks_original = (
        reanchored_tracks_original
        if args.trajectory_source == "reanchored"
        else primary_tracks_original
    )
    selected_visconf = (
        reanchored_visconf
        if args.trajectory_source == "reanchored"
        else sampled_visconf
    )
    cross_anchor_disagreement = np.linalg.norm(
        primary_tracks_original - reanchored_tracks_original, axis=2
    )
    cross_anchor_agreement = 1.0 / (
        1.0
        + (
            cross_anchor_disagreement
            / args.cross_anchor_agreement_scale_px
        )
        ** 2
    )
    native_visibility = np.clip(selected_visconf[..., 0], 0.0, 1.0)
    native_confidence = np.clip(selected_visconf[..., 1], 0.0, 1.0)
    native_joint = native_visibility * native_confidence
    spatial_consistency = local_spatial_consistency(
        tracks_original,
        query_material_points,
        args.local_neighbours,
    )
    tracking_confidence = np.clip(
        native_joint * spatial_consistency * cross_anchor_agreement,
        0.0,
        1.0,
    ).astype(np.float32)
    in_bounds = (
        (tracks_original[..., 0] >= 0.0)
        & (tracks_original[..., 0] < source_width)
        & (tracks_original[..., 1] >= 0.0)
        & (tracks_original[..., 1] < source_height)
    )
    visibility = (
        in_bounds & (native_joint >= args.native_confidence_threshold)
    )
    tissue_membership = sample_dynamic_mask_membership(
        args.dynamic_tissue_masks,
        source_frames,
        tracks_original,
    )
    dynamic_valid = visibility & tissue_membership
    tracking_confidence[~dynamic_valid] = 0.0

    # Query material identity is established only at frame 0.  The optional
    # tissue grid improves image-space coverage, while the separate range
    # binding artifact distributes each observation over nearby surface nodes.
    initial_pixels = tracks_original[0].astype(np.float64)
    initial_depth = sample_nearest(depth, initial_pixels).astype(np.float64)
    initial_points_table = backproject_to_table(
        initial_pixels, initial_depth, intrinsic, x_table_camera
    )
    nearest_distance = np.linalg.norm(
        initial_points_table - rest_positions[particle_ids], axis=1
    )
    track_valid = (
        np.isfinite(initial_points_table).all(axis=1)
        & np.isfinite(initial_depth)
        & (initial_depth > 0.0)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "tracks.npz",
        schema=np.asarray(
            "super_alltracker_tissue_grid_tracks_v1"
            if args.query_mode == "tissue_grid"
            else "super_alltracker_surface_particle_tracks_v1"
        ),
        source_frame_indices=source_frames,
        tracks_original_px=tracks_original.astype(np.float32),
        visibility=visibility,
        dynamic_tissue_valid=dynamic_valid,
        tracking_confidence=tracking_confidence,
        native_visibility=native_visibility.astype(np.float32),
        native_confidence=native_confidence.astype(np.float32),
        spatial_consistency=spatial_consistency,
        cross_anchor_agreement=cross_anchor_agreement.astype(np.float32),
        cross_anchor_disagreement_px=(
            cross_anchor_disagreement.astype(np.float32)
        ),
        reanchored_tracks_original_px=(
            reanchored_tracks_original.astype(np.float32)
        ),
        primary_tracks_original_px=(
            primary_tracks_original.astype(np.float32)
        ),
        particle_ids=particle_ids,
    )
    np.savez_compressed(
        args.output_dir / "bindings.npz",
        schema=np.asarray("super_alltracker_direct_particle_bindings_v1"),
        source_frame_indices=source_frames,
        initial_pixels_uv=initial_pixels.astype(np.float32),
        tracker_requested_before_tool_mask=np.ones(len(particle_ids), dtype=bool),
        tool_excluded=np.zeros(len(particle_ids), dtype=bool),
        initial_requested=track_valid,
        track_valid=track_valid,
        particle_ids=particle_ids[:, None],
        particle_weights=np.ones((len(particle_ids), 1), dtype=np.float32),
        support_counts=np.ones(len(particle_ids), dtype=np.int32),
        movable_support_counts=np.ones(len(particle_ids), dtype=np.int32),
        binding_radius_m=np.zeros(len(particle_ids), dtype=np.float32),
        initial_depth_m=initial_depth.astype(np.float32),
        depth_sampling_radius_px=np.zeros(len(particle_ids), dtype=np.int16),
        initial_points_table=initial_points_table.astype(np.float32),
        nearest_surface_distance_m=nearest_distance.astype(np.float32),
    )
    valid_confidence = tracking_confidence[dynamic_valid]
    report = {
        "schema": (
            "super_alltracker_tissue_grid_tracks_v1"
            if args.query_mode == "tissue_grid"
            else "super_alltracker_surface_particle_tracks_v1"
        ),
        "passed": bool(
            len(particle_ids) >= 100
            and track_valid.all()
            and np.isfinite(valid_confidence).all()
        ),
        "method": {
            "tracker": (
                "AllTracker official long-range dense flow; independently "
                "reanchored dense flow calibrates confidence"
            ),
            "query": (
                "uniform frame-0 tissue-mask grid lifted by depth"
                if args.query_mode == "tissue_grid"
                else "frame-0 projected visible physical surface particles"
            ),
            "binding": (
                "one immutable nearest particle in this artifact; runtime uses "
                "a separately built immutable local surface range"
                if args.query_mode == "tissue_grid"
                else "one immutable physical particle per dense-flow query"
            ),
            "range_radius_mm": 0.0,
            "confidence": (
                "native visibility * native confidence * local material-neighbour "
                "flow consistency * long-vs-reanchored agreement * dynamic "
                "tissue membership"
            ),
        },
        "counts": {
            "candidate_visible_surface_particles": int(len(candidate_ids)),
            "selected_particle_tracks": int(len(particle_ids)),
            "sampled_frames": int(len(source_frames)),
            "valid_observations": int(dynamic_valid.sum()),
        },
        "metadata": metadata,
        "settings": {
            "maximum_source_frame": args.maximum_source_frame,
            "image_size": args.image_size,
            "window_length": args.window_length,
            "primary_segment_length_sampled_frames": (
                args.primary_segment_length
            ),
            "primary_segment_count": primary_segment_count,
            "reanchor_interval_sampled_frames": args.reanchor_interval,
            "reanchor_interval_source_frames": (
                args.reanchor_interval * args.frame_stride
            ),
            "causal_segment_count": segment_count,
            "cross_anchor_agreement_scale_px": (
                args.cross_anchor_agreement_scale_px
            ),
            "inference_iterations": args.inference_iterations,
            "maximum_initial_depth_error_mm": (
                args.maximum_initial_depth_error_mm
            ),
            "native_confidence_threshold": args.native_confidence_threshold,
            "local_neighbours": args.local_neighbours,
            "query_mode": args.query_mode,
            "trajectory_source": args.trajectory_source,
            "query_spacing_px": args.query_spacing_px,
            "maximum_initial_surface_distance_mm": (
                args.maximum_initial_surface_distance_mm
            ),
            "maximum_queries_per_particle": (
                args.maximum_queries_per_particle
            ),
        },
        "statistics": {
            "initial_particle_depth_error_mm_min_p05_p50_p95_max": five_number(
                nearest_distance * 1000.0
            ),
            "native_joint_confidence_min_p05_p50_p95_max": five_number(
                native_joint[visibility]
            ),
            "spatial_consistency_min_p05_p50_p95_max": five_number(
                spatial_consistency[visibility]
            ),
            "cross_anchor_disagreement_px_min_p05_p50_p95_max": (
                five_number(cross_anchor_disagreement[visibility])
            ),
            "cross_anchor_agreement_min_p05_p50_p95_max": five_number(
                cross_anchor_agreement[visibility]
            ),
            "final_tracking_confidence_min_p05_p50_p95_max": five_number(
                valid_confidence
            ),
            "valid_fraction_by_frame_min_p05_p50_p95_max": five_number(
                dynamic_valid.mean(axis=1)
            ),
        },
        "runtime": {
            "elapsed_s": elapsed,
            "sampled_frames_per_s": len(source_frames) / max(elapsed, 1.0e-9),
            "primary_long_range_elapsed_s": primary_elapsed,
            "secondary_reanchored_elapsed_s": reanchored_elapsed,
        },
        "inputs": {
            "video": str(args.video.resolve()),
            "asset": str(args.asset.resolve()),
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256(args.checkpoint),
            },
            "alltracker_commit": (
                "e7553135e7b361590dbccd10e2b274b024f41cd6"
            ),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("AllTracker surface-particle gates failed")


if __name__ == "__main__":
    main()
