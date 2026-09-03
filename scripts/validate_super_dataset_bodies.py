#!/usr/bin/env python3
"""Validate dataset-local SUPER table coordinates and dense rigid bodies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[1]
BLENDER_FROM_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("grasp1", "grasp3"))
    parser.add_argument("--native-dir", type=Path, default=None)
    parser.add_argument("--offline-dir", type=Path, default=None)
    parser.add_argument("--max-ground-p95-mm", type=float, default=10.0)
    parser.add_argument("--min-mask-projection-fraction", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_points(body: dict, field: str) -> np.ndarray:
    transform = np.asarray(body["X_WB"], dtype=np.float64)
    points = np.asarray(body[field]["means"], dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def projected_mask_fraction(
    points_table: np.ndarray,
    x_table_camera: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
) -> float:
    x_camera_table = np.linalg.inv(x_table_camera)
    camera = points_table @ x_camera_table[:3, :3].T + x_camera_table[:3, 3]
    z = camera[:, 2]
    u = np.rint(camera[:, 0] * K[0, 0] / z + K[0, 2]).astype(np.int64)
    v = np.rint(camera[:, 1] * K[1, 1] / z + K[1, 2]).astype(np.int64)
    inside = (
        (z > 0.0)
        & (u >= 0)
        & (u < mask.shape[1])
        & (v >= 0)
        & (v < mask.shape[0])
    )
    accepted = np.zeros(len(points_table), dtype=bool)
    accepted[inside] = mask[v[inside], u[inside]]
    return float(np.mean(accepted))


def contact_component_sizes(points: np.ndarray, diameter: float) -> list[int]:
    pairs = cKDTree(points).query_pairs(
        diameter + 1.0e-7, output_type="ndarray"
    )
    if len(pairs):
        rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
        columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
            shape=(len(points), len(points)),
        ).tocsr()
    else:
        graph = coo_matrix((len(points), len(points)), dtype=np.uint8).tocsr()
    count, labels = connected_components(graph, directed=False)
    return sorted(map(int, np.bincount(labels, minlength=count)), reverse=True)


def main() -> None:
    args = parse_args()
    native = (args.native_dir or REPO / "data/super" / f"{args.dataset}_native").resolve()
    offline = (
        args.offline_dir or REPO / "data/super" / f"{args.dataset}_offline_demo"
    ).resolve()
    v6 = native / "bodies_v6_dense_camera"
    v7 = native / "bodies_v7_dense_ground_z0"
    v9 = native / "bodies_v9_dense_0p5mm_rigid_tissue"
    output = args.output or v9 / "bodies_validation_report.json"
    required = [
        *(v6 / name for name in ("tissue.json", "ground.json", "ground_plane.json", "build_metadata.json")),
        *(v7 / name for name in ("tissue.json", "ground.json", "ground_plane.json", "build_metadata.json", "table_frame.json", "cameras_table.json")),
        *(v9 / name for name in ("tissue.json", "ground.json", "ground_plane.json", "build_metadata.json", "table_frame.json", "cameras_table.json", "particle_packing_comparison.png", "super_bodies_ground_z0.ply")),
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError("Missing body assets:\n- " + "\n- ".join(missing))

    v6_tissue, v6_ground = read_json(v6 / "tissue.json"), read_json(v6 / "ground.json")
    v7_tissue, v7_ground = read_json(v7 / "tissue.json"), read_json(v7 / "ground.json")
    v9_tissue, v9_ground = read_json(v9 / "tissue.json"), read_json(v9 / "ground.json")
    v7_metadata, v9_metadata = read_json(v7 / "build_metadata.json"), read_json(v9 / "build_metadata.json")
    table = read_json(v9 / "table_frame.json")
    cameras = read_json(v9 / "cameras_table.json")
    calibration = read_json(native / "calib_rectified.json")
    K = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    x_table_camera = np.asarray(table["X_table_camera"], dtype=np.float64)
    rotation = x_table_camera[:3, :3]
    plane = np.asarray(read_json(v9 / "ground_plane.json")["plane"], dtype=np.float64)

    tissue_mask = cv2.imread(str(native / "masks/000000-tissue.png"), cv2.IMREAD_GRAYSCALE) > 0
    ground_mask = cv2.imread(str(native / "masks/000000-ground.png"), cv2.IMREAD_GRAYSCALE) > 0
    v6_tissue_gaussians = world_points(v6_tissue, "gaussians")
    v7_tissue_gaussians = world_points(v7_tissue, "gaussians")
    v9_tissue_gaussians = world_points(v9_tissue, "gaussians")
    v6_ground_gaussians = world_points(v6_ground, "gaussians")
    v7_ground_gaussians = world_points(v7_ground, "gaussians")
    v9_ground_gaussians = world_points(v9_ground, "gaussians")
    expected_v7_tissue = v6_tissue_gaussians @ rotation.T + x_table_camera[:3, 3]
    expected_v7_ground = v6_ground_gaussians @ rotation.T + x_table_camera[:3, 3]

    particles = world_points(v9_tissue, "particles")
    radii = np.asarray(v9_tissue["particles"]["radii"], dtype=np.float64)
    packing = v9_metadata["packing"]
    mass = v9_metadata["mass_preservation"]
    ground_p95 = float(v9_metadata["plane_fit"]["ground_absolute_distance_mm_percentiles"][2])
    left_pose = np.asarray(cameras["stereo_left"]["X_WC"], dtype=np.float64)
    right_pose = np.asarray(cameras["stereo_right"]["X_WC"], dtype=np.float64)
    baseline = float(table["stereo_baseline_m"])
    x_left_right = np.eye(4, dtype=np.float64)
    x_left_right[0, 3] = baseline
    expected_left = x_table_camera @ BLENDER_FROM_OPENCV
    expected_right = x_table_camera @ x_left_right @ BLENDER_FROM_OPENCV
    original_manifest_hash = sha256(offline / "cameras.json")
    recorded_manifest_hash = v7_metadata["coordinate_validation"]["source_camera_manifest_sha256"]
    contact_components = contact_component_sizes(particles, 2.0 * float(radii.max()))

    metrics = {
        "rotation_determinant": float(np.linalg.det(rotation)),
        "rotation_orthogonality_max_error": float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))),
        "ground_gaussian_abs_z_max_m": float(np.max(np.abs(v9_ground_gaussians[:, 2]))),
        "tissue_particle_count": int(len(particles)),
        "tissue_gaussian_count": int(len(v9_tissue_gaussians)),
        "ground_gaussian_count": int(len(v9_ground_gaussians)),
        "particle_radius_m_min_max": [float(radii.min()), float(radii.max())],
        "minimum_particle_sphere_clearance_m": float(np.min(particles[:, 2] - radii)),
        "maximum_particle_center_z_m": float(np.max(particles[:, 2])),
        "v6_to_v7_tissue_gaussian_max_error_m": float(np.max(np.abs(expected_v7_tissue - v7_tissue_gaussians))),
        "v6_to_v7_ground_gaussian_max_error_m": float(np.max(np.abs(expected_v7_ground - v7_ground_gaussians))),
        "v7_to_v9_tissue_gaussian_max_error_m": float(np.max(np.abs(v7_tissue_gaussians - v9_tissue_gaussians))),
        "v7_to_v9_ground_gaussian_max_error_m": float(np.max(np.abs(v7_ground_gaussians - v9_ground_gaussians))),
        "tissue_gaussian_projection_in_manual_mask_fraction": projected_mask_fraction(v9_tissue_gaussians, x_table_camera, K, tissue_mask),
        "ground_gaussian_projection_in_manual_mask_fraction": projected_mask_fraction(v9_ground_gaussians, x_table_camera, K, ground_mask),
        "ground_fit_absolute_residual_p95_mm": ground_p95,
        "packed_nearest_mm_min_p05_p50_p95_max": packing["packed_nearest_mm_min_p05_p50_p95_max"],
        "packed_overlap_mm_min_p05_p50_p95_max": packing["packed_overlap_mm_min_p05_p50_p95_max"],
        "overlapping_nearest_neighbor_fraction": float(packing["overlapping_nearest_neighbor_fraction"]),
        "particle_contact_component_sizes": contact_components,
        "source_mass_g": float(mass["source_sphere_sum_mass_g"]),
        "packed_mass_g": float(mass["packed_sphere_sum_mass_g"]),
        "stereo_camera_center_distance_m": float(np.linalg.norm(right_pose[:3, 3] - left_pose[:3, 3])),
        "offline_camera_manifest_sha256": original_manifest_hash,
        "recorded_pre_alignment_manifest_sha256": recorded_manifest_hash,
    }
    gates = {
        "table_plane_exact_z0": bool(np.array_equal(plane, [0.0, 0.0, 1.0, 0.0])),
        "table_rotation_right_handed": bool(abs(metrics["rotation_determinant"] - 1.0) <= 1.0e-10),
        "table_rotation_orthonormal": bool(metrics["rotation_orthogonality_max_error"] <= 1.0e-10),
        "ground_gaussians_on_z0": bool(metrics["ground_gaussian_abs_z_max_m"] <= 1.0e-7),
        "tissue_uses_half_mm_spheres": bool(np.allclose(radii, 0.0005, atol=1.0e-10)),
        "tissue_does_not_penetrate_ground": bool(metrics["minimum_particle_sphere_clearance_m"] >= -1.0e-6),
        "packing_has_no_nearest_neighbor_holes": bool(metrics["overlapping_nearest_neighbor_fraction"] == 1.0 and packing["packed_overlap_mm_min_p05_p50_p95_max"][2] > 0.0),
        "particle_contact_graph_is_connected": len(contact_components) == 1,
        "packing_preserves_sphere_sum_mass": bool(np.isclose(metrics["source_mass_g"], metrics["packed_mass_g"], rtol=1.0e-12)),
        "v6_to_v7_tissue_equivariant": bool(metrics["v6_to_v7_tissue_gaussian_max_error_m"] <= 1.0e-10),
        "v6_to_v7_ground_equivariant": bool(metrics["v6_to_v7_ground_gaussian_max_error_m"] <= 1.0e-10),
        "v7_to_v9_tissue_gaussians_unchanged": bool(metrics["v7_to_v9_tissue_gaussian_max_error_m"] == 0.0),
        "v7_to_v9_ground_gaussians_unchanged": bool(metrics["v7_to_v9_ground_gaussian_max_error_m"] == 0.0),
        "v7_to_v9_tissue_frame_unchanged": bool(v7_tissue["X_WB"] == v9_tissue["X_WB"]),
        "left_camera_pose_consistent": bool(np.allclose(left_pose, expected_left, atol=1.0e-12)),
        "right_camera_pose_consistent": bool(np.allclose(right_pose, expected_right, atol=1.0e-12)),
        "stereo_baseline_consistent": bool(np.isclose(metrics["stereo_camera_center_distance_m"], baseline, atol=1.0e-12)),
        "tissue_gaussians_project_inside_manual_mask": bool(metrics["tissue_gaussian_projection_in_manual_mask_fraction"] >= args.min_mask_projection_fraction),
        "ground_gaussians_project_inside_manual_mask": bool(metrics["ground_gaussian_projection_in_manual_mask_fraction"] >= args.min_mask_projection_fraction),
        "ground_fit_residual": bool(ground_p95 <= args.max_ground_p95_mm),
        "offline_camera_manifest_unchanged": bool(original_manifest_hash == recorded_manifest_hash),
        "table_frame_copied_exactly": bool(sha256(v7 / "table_frame.json") == sha256(v9 / "table_frame.json")),
        "table_cameras_copied_exactly": bool(sha256(v7 / "cameras_table.json") == sha256(v9 / "cameras_table.json")),
        "preview_assets_present": bool((v9 / "particle_packing_comparison.png").stat().st_size > 0 and (v9 / "super_bodies_ground_z0.ply").stat().st_size > 0),
    }
    report = {
        "dataset": args.dataset,
        "v6_camera_bodies": str(v6),
        "v7_table_bodies": str(v7),
        "v9_dense_rigid_bodies": str(v9),
        "metrics": metrics,
        "thresholds": {
            "max_ground_p95_mm": args.max_ground_p95_mm,
            "min_mask_projection_fraction": args.min_mask_projection_fraction,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics, "gates": gates, "all_gates_passed": report["all_gates_passed"]}, indent=2))
    if not report["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
