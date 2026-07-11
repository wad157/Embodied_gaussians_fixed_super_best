#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
X_OPENCV_CAMERA_FROM_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the reconstructed SUPER scene from rectified-left camera coordinates to a z-up table frame."
    )
    parser.add_argument(
        "--source-bodies",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_native/bodies_v4",
    )
    parser.add_argument(
        "--output-bodies",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_native/bodies_v5_table",
    )
    parser.add_argument(
        "--camera-manifest",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_offline_demo/cameras.json",
    )
    parser.add_argument(
        "--camera-frame-backup",
        type=Path,
        default=REPO_ROOT / "data/super/grasp5_offline_demo/cameras_camera_frame.json",
    )
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
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=REPO_ROOT / "examples/embodied_environments/super_embodied",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def world_points(body: dict, field: str) -> np.ndarray:
    X_WB = np.asarray(body["X_WB"], dtype=np.float64)
    points = np.asarray(body[field]["means"], dtype=np.float64)
    return (X_WB[:3, :3] @ points.T + X_WB[:3, 3:4]).T


def make_table_transform(tissue: dict, plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    plane = np.asarray(plane, dtype=np.float64)
    plane /= np.linalg.norm(plane[:3])
    normal = plane[:3]

    if tissue.get("particles") and tissue["particles"].get("means"):
        tissue_center = world_points(tissue, "particles").mean(axis=0)
    else:
        tissue_center = world_points(tissue, "gaussians").mean(axis=0)
    origin_camera = tissue_center - (tissue_center @ normal + plane[3]) * normal

    x_reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis_camera = x_reference - np.dot(x_reference, normal) * normal
    if np.linalg.norm(x_axis_camera) < 1e-6:
        x_reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis_camera = x_reference - np.dot(x_reference, normal) * normal
    x_axis_camera /= np.linalg.norm(x_axis_camera)
    y_axis_camera = np.cross(normal, x_axis_camera)
    y_axis_camera /= np.linalg.norm(y_axis_camera)

    R_camera_table = np.column_stack([x_axis_camera, y_axis_camera, normal])
    if np.linalg.det(R_camera_table) < 0.0:
        y_axis_camera = -y_axis_camera
        R_camera_table = np.column_stack([x_axis_camera, y_axis_camera, normal])

    X_table_camera = np.eye(4, dtype=np.float64)
    X_table_camera[:3, :3] = R_camera_table.T
    X_table_camera[:3, 3] = -R_camera_table.T @ origin_camera
    return X_table_camera, origin_camera


def transform_tissue(tissue: dict, X_table_camera: np.ndarray) -> dict:
    transformed = json.loads(json.dumps(tissue))
    X_camera_body = np.asarray(tissue["X_WB"], dtype=np.float64)
    transformed["X_WB"] = (X_table_camera @ X_camera_body).tolist()
    return transformed


def transform_ground(ground: dict, X_table_camera: np.ndarray) -> dict:
    transformed = json.loads(json.dumps(ground))
    X_camera_body = np.asarray(ground["X_WB"], dtype=np.float64)
    R_camera_body = X_camera_body[:3, :3]
    R_table_camera = X_table_camera[:3, :3]

    means_body = np.asarray(ground["gaussians"]["means"], dtype=np.float64)
    means_camera = (R_camera_body @ means_body.T + X_camera_body[:3, 3:4]).T
    means_table = (
        R_table_camera @ means_camera.T + X_table_camera[:3, 3:4]
    ).T

    quats_body = np.asarray(ground["gaussians"]["quats"], dtype=np.float64)
    rotations_body = R.from_quat(quats_body, scalar_first=True).as_matrix()
    rotations_table = np.einsum(
        "ij,njk->nik", R_table_camera @ R_camera_body, rotations_body
    )
    quats_table = R.from_matrix(rotations_table).as_quat(scalar_first=True)

    transformed["X_WB"] = np.eye(4, dtype=np.float64).tolist()
    transformed["gaussians"]["means"] = means_table.tolist()
    transformed["gaussians"]["quats"] = quats_table.tolist()
    transformed["ground_plane"] = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.0}
    return transformed


def transform_camera_manifest(
    source_manifest: dict, X_table_camera: np.ndarray, baseline_m: float
) -> dict:
    transformed = json.loads(json.dumps(source_manifest))
    left_pose_camera = np.asarray(
        source_manifest["stereo_left"]["X_WC"], dtype=np.float64
    )
    right_pose_camera = np.asarray(
        source_manifest["stereo_right"]["X_WC"], dtype=np.float64
    )

    # The extracted manifest used identity for both cameras. In the rectified
    # left-camera world, the right camera center is +baseline on camera X.
    if np.allclose(left_pose_camera, right_pose_camera, atol=1e-9):
        right_pose_camera = left_pose_camera.copy()
        right_pose_camera[:3, 3] += left_pose_camera[:3, 0] * baseline_m

    left_pose_table_opencv = X_table_camera @ left_pose_camera
    right_pose_table_opencv = X_table_camera @ right_pose_camera
    # FramesBuilder expects camera-to-world poses in Blender camera axes and
    # converts them back to OpenCV internally for rasterization.
    transformed["stereo_left"]["X_WC"] = (
        left_pose_table_opencv @ X_OPENCV_CAMERA_FROM_BLENDER_CAMERA
    ).tolist()
    transformed["stereo_right"]["X_WC"] = (
        right_pose_table_opencv @ X_OPENCV_CAMERA_FROM_BLENDER_CAMERA
    ).tolist()
    return transformed


def main() -> None:
    args = parse_args()
    tissue = read_json(args.source_bodies / "tissue.json")
    ground = read_json(args.source_bodies / "ground.json")
    ground_plane = read_json(args.source_bodies / "ground_plane.json")
    source_plane = np.asarray(ground_plane["plane"], dtype=np.float64)

    X_table_camera, origin_camera = make_table_transform(tissue, source_plane)
    X_camera_table = np.linalg.inv(X_table_camera)
    transformed_tissue = transform_tissue(tissue, X_table_camera)
    transformed_ground = transform_ground(ground, X_table_camera)

    calibration = read_json(args.calibration)
    baseline_m = float(calibration["baseline_m"])
    if args.camera_frame_backup.exists():
        source_camera_manifest = read_json(args.camera_frame_backup)
    else:
        source_camera_manifest = read_json(args.camera_manifest)
        args.camera_frame_backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.camera_manifest, args.camera_frame_backup)
    table_camera_manifest = transform_camera_manifest(
        source_camera_manifest, X_table_camera, baseline_m
    )

    args.output_bodies.mkdir(parents=True, exist_ok=True)
    write_json(args.output_bodies / "tissue.json", transformed_tissue)
    write_json(args.output_bodies / "ground.json", transformed_ground)
    write_json(args.output_bodies / "ground_plane.json", {"plane": [0.0, 0.0, 1.0, 0.0]})

    source_metadata_path = args.source_bodies / "build_metadata.json"
    metadata = read_json(source_metadata_path) if source_metadata_path.exists() else {}
    metadata.update(
        {
            "source_bodies": str(args.source_bodies),
            "output_dir": str(args.output_bodies),
            "source_plane_camera": source_plane.tolist(),
            "plane": [0.0, 0.0, 1.0, 0.0],
            "X_table_camera": X_table_camera.tolist(),
            "world_frame": "right-handed table frame, z-up, ground z=0",
        }
    )
    write_json(args.output_bodies / "build_metadata.json", metadata)
    write_json(args.camera_manifest, table_camera_manifest)

    table_frame = {
        "convention": "X_A_B maps coordinates from frame B to frame A",
        "source_frame": "left rectified OpenCV camera frame",
        "target_frame": "right-handed table frame",
        "X_table_camera": X_table_camera.tolist(),
        "X_camera_table": X_camera_table.tolist(),
        "table_origin_in_camera": origin_camera.tolist(),
        "source_ground_plane": source_plane.tolist(),
        "target_ground_plane": [0.0, 0.0, 1.0, 0.0],
        "gravity_table_m_s2": [0.0, 0.0, -9.80665],
        "stereo_baseline_m": baseline_m,
        "camera_manifest_pose_convention": "Blender camera-to-world",
    }
    write_json(args.table_frame, table_frame)

    objects_dir = args.scene_dir / "objects"
    environment_dir = args.scene_dir / "environment"
    objects_dir.mkdir(parents=True, exist_ok=True)
    environment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output_bodies / "tissue.json", objects_dir / "tissue.json")
    shutil.copy2(args.output_bodies / "ground.json", objects_dir / "ground.json")
    shutil.copy2(
        args.output_bodies / "ground_plane.json",
        environment_dir / "ground_plane.json",
    )

    tissue_world = world_points(transformed_tissue, "particles")
    ground_world = world_points(transformed_ground, "gaussians")
    report = {
        "X_table_camera": X_table_camera.tolist(),
        "tissue_z_m_percentiles": np.percentile(
            tissue_world[:, 2], [0, 5, 50, 95, 100]
        ).tolist(),
        "ground_abs_z_m_percentiles": np.percentile(
            np.abs(ground_world[:, 2]), [50, 95, 100]
        ).tolist(),
        "camera_manifest": str(args.camera_manifest),
        "camera_frame_backup": str(args.camera_frame_backup),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
