#!/usr/bin/env python3
"""Bind grasp5 CoTracker tracks to fixed weighted physical-surface ranges.

This is the non-mutating Stage-A diagnostic for the V1 flow-depth observer.
It reads only frame-zero tracks/depth and a frozen tissue asset.  The output
contains particle ids and weights, but never opens or writes a simulator state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_gaussians.physics_simulator.flow_depth_particle_observer import (  # noqa: E402
    FlowDepthRangeBindingSettings,
    FlowDepthTriangleBindingSettings,
    build_fixed_particle_range_bindings,
    build_fixed_particle_triangle_bindings,
)


DEFAULT_TRACKS = (
    REPO_ROOT / "outputs/grasp5_cotracker3_motion_left_20260828_v1/tracks.npz"
)
DEFAULT_DEPTH = (
    REPO_ROOT
    / "data/super/grasp5_native/depth_v4_foundation_dense_timestamped/"
    "000000-depth.npy"
)
DEFAULT_IMAGE = (
    REPO_ROOT / "data/super/grasp5_native/rgb/000000-left.png"
)
DEFAULT_TOOL_MASK = (
    REPO_ROOT
    / "data/super/grasp5_native/tissue_multiview_v1/masks_left/"
    "tool_dilated/000000-tool-dilated.png"
)
DEFAULT_ASSET = (
    REPO_ROOT
    / "data/super/grasp5_native/tissue_multiview_v1/"
    "paper_pbd_tissue_v15_denser_mild_paper_constraints_centroid_ellipsoids/"
    "tissue_paper_pbd_centroid_gaussians.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/grasp5_cotracker3_range_bindings_20260828_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--tool-mask", type=Path, default=DEFAULT_TOOL_MASK)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_native/calib_rectified.json",
    )
    parser.add_argument(
        "--table-frame",
        type=Path,
        default=REPO_ROOT / "data/super/table_frame.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--binding-mode",
        choices=("range", "triangle"),
        default="range",
        help=(
            "Use a Euclidean particle range or one nearest physical surface "
            "triangle with barycentric vertex weights."
        ),
    )
    parser.add_argument("--radius-mm", type=float, default=6.0)
    parser.add_argument("--fallback-radius-mm", type=float, default=8.0)
    parser.add_argument("--sigma-mm", type=float, default=3.0)
    parser.add_argument("--minimum-movable-particles", type=int, default=3)
    parser.add_argument("--triangle-primary-distance-mm", type=float, default=1.0)
    parser.add_argument("--triangle-fallback-distance-mm", type=float, default=2.0)
    parser.add_argument("--triangle-minimum-movable-vertices", type=int, default=3)
    parser.add_argument("--maximum-depth-sampling-radius-px", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def five_number(values: np.ndarray) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return np.percentile(finite, [0, 5, 50, 95, 100]).tolist()


def draw_overlay(
    image_path: Path,
    pixels_uv: np.ndarray,
    valid: np.ndarray,
    tool_excluded: np.ndarray,
    support_counts: np.ndarray,
    output_path: Path,
    binding_label: str = "fixed range",
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    maximum = max(int(support_counts[valid].max(initial=1)), 1)
    normalized = np.clip(support_counts.astype(np.float32) / maximum, 0.0, 1.0)
    colors = cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8)[:, None],
        cv2.COLORMAP_TURBO,
    )[:, 0]
    for track_id, pixel in enumerate(pixels_uv):
        center = tuple(np.rint(pixel).astype(int))
        if tool_excluded[track_id]:
            cv2.drawMarker(
                image,
                center,
                (0, 128, 255),
                cv2.MARKER_TILTED_CROSS,
                9,
                2,
                cv2.LINE_AA,
            )
        elif valid[track_id]:
            color = tuple(int(value) for value in colors[track_id])
            cv2.circle(image, center, 3, color, -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(
                image,
                center,
                (0, 0, 255),
                cv2.MARKER_TILTED_CROSS,
                7,
                1,
                cv2.LINE_AA,
            )
    label = (
        f"V1 {binding_label} bindings | valid {int(valid.sum())}/{len(valid)} | "
        f"tool excluded {int(tool_excluded.sum())} | color = support count"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write {output_path}")


def main() -> None:
    args = parse_args()
    output_asset = args.output_dir / "bindings.npz"
    output_report = args.output_dir / "report.json"
    output_overlay = args.output_dir / "initial_binding_overlay.png"
    collisions = [
        path
        for path in (output_asset, output_report, output_overlay)
        if path.exists()
    ]
    if collisions and not args.overwrite:
        raise FileExistsError(
            "Binding outputs already exist; pass --overwrite:\n- "
            + "\n- ".join(str(path) for path in collisions)
        )
    for path in (
        args.tracks,
        args.depth,
        args.image,
        args.tool_mask,
        args.asset,
        args.calibration,
        args.table_frame,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(args.tracks, allow_pickle=False) as loaded:
        source_frames = loaded["source_frame_indices"].astype(np.int32)
        initial_pixels = loaded["tracks_original_px"][0].astype(np.float64)
        tracker_requested = (
            loaded["visibility"][0].astype(bool)
            & loaded["dynamic_tissue_valid"][0].astype(bool)
        )
    if int(source_frames[0]) != 0:
        raise ValueError("V1 range binding expects the first CoTracker frame to be 0")
    depth = np.load(args.depth, allow_pickle=False)
    tool_mask = cv2.imread(str(args.tool_mask), cv2.IMREAD_GRAYSCALE)
    if tool_mask is None:
        raise FileNotFoundError(args.tool_mask)
    rounded_pixels = np.rint(initial_pixels).astype(np.int64)
    inside_tool_mask = (
        (rounded_pixels[:, 0] >= 0)
        & (rounded_pixels[:, 0] < tool_mask.shape[1])
        & (rounded_pixels[:, 1] >= 0)
        & (rounded_pixels[:, 1] < tool_mask.shape[0])
    )
    tool_excluded = np.zeros(len(initial_pixels), dtype=bool)
    tool_excluded[inside_tool_mask] = (
        tool_mask[
            rounded_pixels[inside_tool_mask, 1],
            rounded_pixels[inside_tool_mask, 0],
        ]
        > 0
    )
    tool_excluded &= tracker_requested
    initial_valid = tracker_requested & ~tool_excluded
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    table_frame = json.loads(args.table_frame.read_text(encoding="utf-8"))
    with np.load(args.asset, allow_pickle=False) as loaded:
        rest_positions = loaded["rest_positions_table"].astype(np.float64)
        surface_faces = loaded["surface_faces"].astype(np.int32)
        surface_mask = loaded["surface_node_mask"].astype(bool)
        fixed_mask = loaded["fixed_mask"].astype(bool)

    common = dict(
        initial_pixels_uv=initial_pixels,
        depth=depth,
        intrinsic=np.asarray(calibration["K_left_rect"], dtype=np.float64),
        x_table_camera=np.asarray(
            table_frame["X_table_camera"], dtype=np.float64
        ),
        rest_positions_table=rest_positions,
        surface_mask=surface_mask,
        fixed_mask=fixed_mask,
        initial_track_valid=initial_valid,
    )
    if args.binding_mode == "triangle":
        settings = FlowDepthTriangleBindingSettings(
            primary_maximum_surface_distance_m=(
                args.triangle_primary_distance_mm / 1000.0
            ),
            fallback_maximum_surface_distance_m=(
                args.triangle_fallback_distance_mm / 1000.0
            ),
            minimum_movable_vertices=args.triangle_minimum_movable_vertices,
            maximum_depth_sampling_radius_px=(
                args.maximum_depth_sampling_radius_px
            ),
        )
        bindings = build_fixed_particle_triangle_bindings(
            **common,
            surface_faces=surface_faces,
            settings=settings,
        )
        primary_distance = settings.primary_maximum_surface_distance_m
        fallback_distance = settings.fallback_maximum_surface_distance_m
        minimum_movable_support = settings.minimum_movable_vertices
    else:
        settings = FlowDepthRangeBindingSettings(
            radius_m=args.radius_mm / 1000.0,
            fallback_radius_m=args.fallback_radius_mm / 1000.0,
            sigma_m=args.sigma_mm / 1000.0,
            minimum_movable_particles=args.minimum_movable_particles,
            maximum_depth_sampling_radius_px=(
                args.maximum_depth_sampling_radius_px
            ),
        )
        bindings = build_fixed_particle_range_bindings(
            **common,
            settings=settings,
        )
        primary_distance = settings.radius_m
        fallback_distance = settings.fallback_radius_m
        minimum_movable_support = settings.minimum_movable_particles
    valid = bindings.track_valid
    valid_counts = bindings.support_counts[valid]
    valid_movable = bindings.movable_support_counts[valid]
    valid_weights = bindings.valid_weight_sums()
    primary = np.isclose(bindings.binding_radius_m, primary_distance)
    fallback = np.isclose(bindings.binding_radius_m, fallback_distance)
    if args.binding_mode == "triangle":
        valid_face_ids = bindings.surface_face_ids[valid]
        valid_projected = bindings.surface_projection_points_table[valid]
        reconstructed = np.einsum(
            "ni,nij->nj",
            bindings.particle_weights[valid].astype(np.float64),
            rest_positions[bindings.particle_ids[valid]],
        )
    else:
        valid_face_ids = np.empty(0, dtype=np.int32)
        valid_projected = np.empty((0, 3), dtype=np.float32)
        reconstructed = np.empty((0, 3), dtype=np.float64)
    gates = {
        "requested_binding_policy_satisfied": bool(
            (
                np.array_equal(valid, initial_valid)
                if args.binding_mode == "range"
                else (
                    np.all(~valid | initial_valid)
                    and np.count_nonzero(valid)
                    >= np.count_nonzero(initial_valid) * 0.99
                    and np.all(
                        bindings.nearest_surface_distance_m[
                            initial_valid & ~valid
                        ]
                        > fallback_distance
                    )
                )
            )
        ),
        "all_tool_mask_tracks_excluded": bool(
            not np.any(valid & tool_excluded)
        ),
        "all_valid_weights_finite": bool(
            np.isfinite(bindings.particle_weights[valid]).all()
        ),
        "all_valid_weights_sum_to_one": bool(
            np.allclose(valid_weights, 1.0, atol=2.0e-6)
        ),
        "all_support_particles_are_surface": bool(
            all(
                np.all(
                    surface_mask[
                        bindings.particle_ids[track_id, : count]
                    ]
                )
                for track_id, count in enumerate(bindings.support_counts)
                if valid[track_id]
            )
        ),
        "minimum_movable_support_satisfied": bool(
            len(valid_movable)
            and np.all(
                valid_movable >= minimum_movable_support
            )
        ),
        "only_primary_or_fallback_radius_used": bool(
            np.all(primary[valid] | fallback[valid])
        ),
        "triangle_face_ids_are_valid": bool(
            args.binding_mode != "triangle"
            or (
                len(valid_face_ids) == int(valid.sum())
                and np.all(valid_face_ids >= 0)
                and np.all(valid_face_ids < len(surface_faces))
            )
        ),
        "triangle_vertices_match_face_ids": bool(
            args.binding_mode != "triangle"
            or np.array_equal(
                bindings.particle_ids[valid], surface_faces[valid_face_ids]
            )
        ),
        "triangle_barycentric_reconstruction_is_exact": bool(
            args.binding_mode != "triangle"
            or np.allclose(
                reconstructed, valid_projected, atol=2.0e-7, rtol=0.0
            )
        ),
        "no_simulator_state_was_loaded_or_written": True,
    }
    report = {
        "schema": (
            "super_tracker_fixed_surface_triangle_bindings_v1"
            if args.binding_mode == "triangle"
            else "super_cotracker_fixed_particle_range_bindings_v1"
        ),
        "passed": bool(all(gates.values())),
        "method": {
            "material_identity": (
                "frame-0 depth point projected once to a fixed physical "
                "surface triangle"
                if args.binding_mode == "triangle"
                else "frame-0 depth point bound once to a fixed physical-"
                "surface particle id/weight range"
            ),
            "weight": (
                "convex barycentric coordinates of the closest point on the triangle"
                if args.binding_mode == "triangle"
                else "exp(-initial_distance^2/(2*sigma^2)), normalized per track"
            ),
            "per_frame_rebinding": False,
            "frame_zero_tool_mask_exclusion": True,
            "simulator_state_mutation": False,
        },
        "settings": {
            "binding_mode": args.binding_mode,
            "radius_mm": (
                args.radius_mm if args.binding_mode == "range" else None
            ),
            "fallback_radius_mm": (
                args.fallback_radius_mm
                if args.binding_mode == "range"
                else None
            ),
            "sigma_mm": (
                args.sigma_mm if args.binding_mode == "range" else None
            ),
            "minimum_movable_particles": (
                args.minimum_movable_particles
                if args.binding_mode == "range"
                else None
            ),
            "triangle_primary_distance_mm": (
                args.triangle_primary_distance_mm
                if args.binding_mode == "triangle"
                else None
            ),
            "triangle_fallback_distance_mm": (
                args.triangle_fallback_distance_mm
                if args.binding_mode == "triangle"
                else None
            ),
            "triangle_minimum_movable_vertices": (
                args.triangle_minimum_movable_vertices
                if args.binding_mode == "triangle"
                else None
            ),
            "maximum_depth_sampling_radius_px": (
                args.maximum_depth_sampling_radius_px
            ),
        },
        "counts": {
            "tracks": int(len(valid)),
            "tracker_requested_before_tool_mask": int(
                tracker_requested.sum()
            ),
            "tool_mask_excluded": int(tool_excluded.sum()),
            "initially_requested": int(initial_valid.sum()),
            "bound": int(valid.sum()),
            "primary_radius": int(np.count_nonzero(primary & valid)),
            "fallback_radius": int(np.count_nonzero(fallback & valid)),
            "rejected": int(np.count_nonzero(initial_valid & ~valid)),
            "surface_particles": int(surface_mask.sum()),
            "surface_faces": int(len(surface_faces)),
            "movable_surface_particles": int(
                np.count_nonzero(surface_mask & ~fixed_mask)
            ),
        },
        "statistics": {
            "support_count_min_p05_p50_p95_max": five_number(valid_counts),
            "movable_support_count_min_p05_p50_p95_max": five_number(
                valid_movable
            ),
            "nearest_surface_distance_mm_min_p05_p50_p95_max": five_number(
                bindings.nearest_surface_distance_m[valid] * 1000.0
            ),
            "initial_depth_mm_min_p05_p50_p95_max": five_number(
                bindings.initial_depth_m[valid] * 1000.0
            ),
            "weight_sum_min_p05_p50_p95_max": five_number(valid_weights),
        },
        "gates": gates,
        "inputs": {
            "tracks": {
                "path": str(args.tracks.resolve()),
                "sha256": sha256(args.tracks),
            },
            "depth": {
                "path": str(args.depth.resolve()),
                "sha256": sha256(args.depth),
            },
            "tool_mask": {
                "path": str(args.tool_mask.resolve()),
                "sha256": sha256(args.tool_mask),
            },
            "asset": {
                "path": str(args.asset.resolve()),
                "sha256": sha256(args.asset),
            },
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": sha256(args.calibration),
            },
            "table_frame": {
                "path": str(args.table_frame.resolve()),
                "sha256": sha256(args.table_frame),
            },
        },
        "outputs": {
            "bindings": str(output_asset.resolve()),
            "overlay": str(output_overlay.resolve()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_asset,
        schema=np.asarray(
            "super_cotracker_fixed_particle_range_bindings_v1"
        ),
        source_frame_indices=source_frames,
        initial_pixels_uv=initial_pixels.astype(np.float32),
        tracker_requested_before_tool_mask=tracker_requested,
        tool_excluded=tool_excluded,
        initial_requested=initial_valid,
        track_valid=bindings.track_valid,
        particle_ids=bindings.particle_ids,
        particle_weights=bindings.particle_weights,
        support_counts=bindings.support_counts,
        movable_support_counts=bindings.movable_support_counts,
        binding_radius_m=bindings.binding_radius_m,
        initial_depth_m=bindings.initial_depth_m,
        depth_sampling_radius_px=bindings.depth_sampling_radius_px,
        initial_points_table=bindings.initial_points_table,
        nearest_surface_distance_m=bindings.nearest_surface_distance_m,
        binding_method=np.asarray(bindings.binding_method),
        surface_face_ids=(
            bindings.surface_face_ids
            if bindings.surface_face_ids is not None
            else np.full(len(bindings.track_valid), -1, dtype=np.int32)
        ),
        surface_projection_points_table=(
            bindings.surface_projection_points_table
            if bindings.surface_projection_points_table is not None
            else np.full(
                (len(bindings.track_valid), 3), np.nan, dtype=np.float32
            )
        ),
    )
    draw_overlay(
        args.image,
        initial_pixels,
        valid,
        tool_excluded,
        bindings.support_counts,
        output_overlay,
        binding_label=(
            "fixed surface-triangle"
            if args.binding_mode == "triangle"
            else "fixed range"
        ),
    )
    output_report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("CoTracker range-binding diagnostic failed")


if __name__ == "__main__":
    main()
