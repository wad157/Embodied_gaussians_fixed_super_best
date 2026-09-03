#!/usr/bin/env python3
"""Create dataset-local z=0 table bodies from camera-frame SUPER bodies.

Unlike the historical grasp5 migration, this script never modifies shared
``data/super/table_frame.json`` or an offline camera manifest.  The table frame
and table-space camera manifest are written beside the output bodies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bodies", type=Path, required=True)
    parser.add_argument("--output-bodies", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera-manifest", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_plane(value: np.ndarray) -> np.ndarray:
    plane = np.asarray(value, dtype=np.float64).copy()
    norm = float(np.linalg.norm(plane[:3]))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError(f"Invalid plane: {plane.tolist()}")
    return plane / norm


def transform_plane(plane_source: np.ndarray, x_target_source: np.ndarray) -> np.ndarray:
    return normalize_plane(np.linalg.inv(x_target_source).T @ plane_source)


def table_from_camera(plane_camera: np.ndarray) -> np.ndarray:
    plane = normalize_plane(plane_camera)
    normal = plane[:3]
    target = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(normal, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(normal @ target, -1.0, 1.0))
    if sine < 1.0e-12:
        if cosine < 0.0:
            rotation = Rotation.from_rotvec([np.pi, 0.0, 0.0]).as_matrix()
        else:
            rotation = np.eye(3, dtype=np.float64)
    else:
        rotation = Rotation.from_rotvec(
            cross / sine * np.arctan2(sine, cosine)
        ).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[2, 3] = plane[3]
    aligned = transform_plane(plane, transform)
    if aligned[2] < 0.0:
        aligned = -aligned
    if not np.allclose(aligned, [0.0, 0.0, 1.0, 0.0], atol=1.0e-10):
        raise RuntimeError(f"Ground alignment failed: {aligned.tolist()}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10):
        raise RuntimeError("Table rotation is not right-handed")
    return transform


def baseline_m(calibration: dict) -> tuple[float, str]:
    if "baseline_m" in calibration:
        return float(calibration["baseline_m"]), "baseline_m"
    translation = np.asarray(calibration["T"], dtype=np.float64)
    return float(np.linalg.norm(translation) / 1000.0), "norm(T)_mm_to_m"


def transform_tissue(tissue: dict, x_table_camera: np.ndarray) -> dict:
    output = json.loads(json.dumps(tissue))
    x_camera_body = np.asarray(tissue["X_WB"], dtype=np.float64)
    output["X_WB"] = (x_table_camera @ x_camera_body).tolist()
    return output


def transform_ground(ground: dict, x_table_camera: np.ndarray) -> dict:
    output = json.loads(json.dumps(ground))
    x_camera_body = np.asarray(ground["X_WB"], dtype=np.float64)
    means_body = np.asarray(ground["gaussians"]["means"], dtype=np.float64)
    means_camera = means_body @ x_camera_body[:3, :3].T + x_camera_body[:3, 3]
    means_table = means_camera @ x_table_camera[:3, :3].T + x_table_camera[:3, 3]
    rotations_body = Rotation.from_quat(
        np.asarray(ground["gaussians"]["quats"], dtype=np.float64),
        scalar_first=True,
    ).as_matrix()
    rotations_table = np.einsum(
        "ij,njk->nik",
        x_table_camera[:3, :3] @ x_camera_body[:3, :3],
        rotations_body,
    )
    output["X_WB"] = np.eye(4, dtype=np.float64).tolist()
    output["gaussians"]["means"] = means_table.tolist()
    output["gaussians"]["quats"] = Rotation.from_matrix(
        rotations_table
    ).as_quat(scalar_first=True).tolist()
    output["ground_plane"] = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.0}
    return output


def table_camera_manifest(
    source: dict, x_table_camera: np.ndarray, stereo_baseline_m: float
) -> dict:
    output = json.loads(json.dumps(source))
    opencv_to_blender = np.diag([1.0, -1.0, -1.0, 1.0])
    x_left_right = np.eye(4, dtype=np.float64)
    x_left_right[0, 3] = stereo_baseline_m
    output["stereo_left"]["X_WC"] = (
        x_table_camera @ opencv_to_blender
    ).tolist()
    output["stereo_right"]["X_WC"] = (
        x_table_camera @ x_left_right @ opencv_to_blender
    ).tolist()
    return output


def main() -> None:
    args = parse_args()
    if args.source_bodies.resolve() == args.output_bodies.resolve():
        raise ValueError("Source and output bodies must differ")
    if args.output_bodies.exists() and any(args.output_bodies.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_bodies}")
    paths = {
        name: args.source_bodies / name
        for name in ("tissue.json", "ground.json", "ground_plane.json", "build_metadata.json")
    }
    for path in (*paths.values(), args.calibration, args.camera_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    tissue = read_json(paths["tissue.json"])
    ground = read_json(paths["ground.json"])
    metadata = read_json(paths["build_metadata.json"])
    plane_camera = normalize_plane(read_json(paths["ground_plane.json"])["plane"])
    calibration = read_json(args.calibration)
    source_cameras = read_json(args.camera_manifest)
    stereo_baseline, baseline_source = baseline_m(calibration)
    x_table_camera = table_from_camera(plane_camera)
    x_camera_table = np.linalg.inv(x_table_camera)
    transformed_tissue = transform_tissue(tissue, x_table_camera)
    transformed_ground = transform_ground(ground, x_table_camera)
    cameras_table = table_camera_manifest(
        source_cameras, x_table_camera, stereo_baseline
    )

    ground_world = np.asarray(transformed_ground["gaussians"]["means"], dtype=np.float64)
    if float(np.max(np.abs(ground_world[:, 2]))) > 1.0e-7:
        raise RuntimeError("Transformed ground Gaussians are not on table z=0")
    tissue_local = np.asarray(transformed_tissue["particles"]["means"], dtype=np.float64)
    x_table_body = np.asarray(transformed_tissue["X_WB"], dtype=np.float64)
    tissue_world = tissue_local @ x_table_body[:3, :3].T + x_table_body[:3, 3]
    radii = np.asarray(transformed_tissue["particles"]["radii"], dtype=np.float64)
    minimum_clearance = float(np.min(tissue_world[:, 2] - radii))
    if minimum_clearance < -1.0e-6:
        raise RuntimeError(f"Tissue penetrates table by {-minimum_clearance * 1000.0} mm")

    args.output_bodies.mkdir(parents=True, exist_ok=False)
    write_json(args.output_bodies / "tissue.json", transformed_tissue)
    write_json(args.output_bodies / "ground.json", transformed_ground)
    write_json(args.output_bodies / "ground_plane.json", {"plane": [0.0, 0.0, 1.0, 0.0]})
    table_frame = {
        "convention": "X_A_B maps coordinates from frame B to frame A",
        "source_frame": "left rectified OpenCV camera frame",
        "target_frame": "dataset-local dense-ground-aligned right-handed table frame",
        "X_table_camera": x_table_camera.tolist(),
        "X_camera_table": x_camera_table.tolist(),
        "table_origin_in_camera": x_camera_table[:3, 3].tolist(),
        "source_ground_plane": plane_camera.tolist(),
        "target_ground_plane": [0.0, 0.0, 1.0, 0.0],
        "gravity_table_m_s2": [0.0, 0.0, -9.80665],
        "stereo_baseline_m": stereo_baseline,
        "baseline_source": baseline_source,
        "camera_manifest_pose_convention": "Blender camera-to-world",
        "alignment_policy": "minimal rotation from fitted normal to +Z; no extra yaw; origin shifted only along fitted normal",
    }
    write_json(args.output_bodies / "table_frame.json", table_frame)
    write_json(args.output_bodies / "cameras_table.json", cameras_table)
    rotation = Rotation.from_matrix(x_table_camera[:3, :3])
    output_metadata = {
        **metadata,
        "output_dir": str(args.output_bodies.resolve()),
        "world_frame": "dataset-local dense-ground-aligned right-handed table frame, z-up",
        "plane": [0.0, 0.0, 1.0, 0.0],
        "source_plane_camera": plane_camera.tolist(),
        "X_table_camera": x_table_camera.tolist(),
        "stereo_baseline_m": stereo_baseline,
        "coordinate_policy": "isolated dataset-local table frame; no shared or offline runtime file modified",
        "coordinate_validation": {
            "rotation_determinant": float(np.linalg.det(x_table_camera[:3, :3])),
            "rotation_orthogonality_max_error": float(np.max(np.abs(x_table_camera[:3, :3] @ x_table_camera[:3, :3].T - np.eye(3)))),
            "rotation_angle_deg": float(np.degrees(rotation.magnitude())),
            "ground_gaussian_abs_z_max_m": float(np.max(np.abs(ground_world[:, 2]))),
            "tissue_minimum_sphere_clearance_m": minimum_clearance,
            "source_hashes": {name: sha256(path) for name, path in paths.items()},
            "calibration_sha256": sha256(args.calibration),
            "source_camera_manifest_sha256": sha256(args.camera_manifest),
            "shared_assets_modified": False,
        },
    }
    write_json(args.output_bodies / "build_metadata.json", output_metadata)
    print(json.dumps({
        "output": str(args.output_bodies),
        "source_plane_camera": plane_camera.tolist(),
        "X_table_camera": x_table_camera.tolist(),
        "rotation_angle_deg": output_metadata["coordinate_validation"]["rotation_angle_deg"],
        "ground_gaussian_abs_z_max_m": output_metadata["coordinate_validation"]["ground_gaussian_abs_z_max_m"],
        "tissue_minimum_sphere_clearance_m": minimum_clearance,
        "stereo_baseline_m": stereo_baseline,
    }, indent=2))


if __name__ == "__main__":
    main()
