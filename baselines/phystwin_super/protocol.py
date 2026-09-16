#!/usr/bin/env python3
"""Frozen inputs and split for the PhysTwin SUPER baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


UPSTREAM_URL = "https://github.com/jianghanxiao/phystwin"
UPSTREAM_COMMIT = "81c718790a37e5e0102eb77af2c6edd34a9db25f"
ADAPTER_VERSION = "phystwin_super_native_particles_v1"
PROTOCOL = "joint_reconstruction_7to1_future_80to20"
COTRACKER_VARIANT = "cotracker3_scaled_offline_chunked"
COTRACKER_CHECKPOINT = "scaled_offline.pth"
INITIALIZATION_FRAME = 1
QUERY_FRAME = 0
HOLDOUT_STRIDE = 8
HOLDOUT_PHASE = 0
RENDER_SCALE = 0.5
DEPTH_DOWNSAMPLE = 3
CAMERAS = ("stereo_left",)
OPENCV_FROM_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])

DATASETS: Mapping[str, Mapping[str, object]] = {
    "grasp5": {
        "frames": 1440,
        "future_start": 1152,
        "native": "data/super/grasp5_native",
        "offline": "data/super/grasp5_offline_demo",
        "camera_file": "data/super/grasp5_offline_demo/cameras.json",
        "ground_truth": "data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz",
        "instrument_masks": "data/super/psm_visual_calibration/raw_paper_lnd_stereo_dense_contact_v4/surgicalsam2_multianchor_parts_dense_contact_v6/stereo_multianchor_part_masks.npz",
        "pose_driver": "data/super/psm_robot/psm_lnd_pose_driver.npz",
        "pose_report": "data/super/psm_robot/psm_lnd_pose_driver_report.json",
    },
    "grasp3": {
        "frames": 2062,
        "future_start": 1649,
        "native": "data/super/grasp3_native",
        "offline": "data/super/grasp3_offline_demo",
        "camera_file": "data/super/grasp3_native/bodies_v9_dense_0p5mm_rigid_tissue/cameras_table.json",
        "ground_truth": "data/super/grasp3_native/evaluation_v1/manual_tissue_tracks_10_v2/ground_truth_2d3d_v1.npz",
        "instrument_masks": "data/super/grasp3_native/instrument_manual_calibration_v1/stereo_annotations/surgicalsam2_multianchor_parts_v4/stereo_multianchor_part_masks.npz",
        "pose_driver": "data/super/grasp3_offline_demo/instruments/psm_lnd_pose_driver.npz",
        "pose_report": "data/super/grasp3_offline_demo/instruments/psm_lnd_pose_driver_report.json",
    },
    "grasp1": {
        "frames": 4205,
        "future_start": 3364,
        "native": "data/super/grasp1_native",
        "offline": "data/super/grasp1_offline_demo",
        "camera_file": "data/super/grasp1_native/bodies_v9_dense_0p5mm_rigid_tissue/cameras_table.json",
        "ground_truth": "data/super/grasp1_native/evaluation_v2/manual_tissue_tracks_10_no_exclusion/ground_truth_2d3d_v1.npz",
        "instrument_masks": "data/super/grasp1_native/instrument_manual_calibration_v1/stereo_annotations/surgicalsam2_multianchor_parts_v4/stereo_multianchor_part_masks.npz",
        "pose_driver": "data/super/grasp1_offline_demo/instruments/psm_lnd_pose_driver.npz",
        "pose_report": "data/super/grasp1_offline_demo/instruments/psm_lnd_pose_driver_report.json",
    },
}


def dataset_spec(key: str) -> Mapping[str, object]:
    try:
        return DATASETS[key.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown SUPER dataset {key}; choose from {sorted(DATASETS)}") from error


def resolve_path(repo_root: Path, value: object) -> Path:
    return (repo_root / str(value)).resolve()


def resolve_native(repo_root: Path, key: str, supplied: Path | None = None) -> Path:
    expected = resolve_path(repo_root, dataset_spec(key)["native"])
    actual = expected if supplied is None else supplied.expanduser().resolve()
    if actual != expected:
        raise ValueError(f"Formal baseline requires {expected}; received {actual}")
    if not actual.is_dir():
        raise FileNotFoundError(actual)
    return actual


def future_start(frame_count: int) -> int:
    """Compatibility entry used by the generic PhysTwin physics runner."""
    return int(frame_count) * 4 // 5


def training_frames(key: str) -> list[int]:
    stop = int(dataset_spec(key)["future_start"])
    return [frame for frame in range(stop) if frame % HOLDOUT_STRIDE != HOLDOUT_PHASE]


def observation_allowed(frame: int, key: str) -> bool:
    return frame in range(int(dataset_spec(key)["future_start"])) and frame % 8 != 0


def reconstruction_holdouts(key: str) -> list[int]:
    return [frame for frame in range(int(dataset_spec(key)["future_start"])) if frame % 8 == 0]


def future_frames(key: str) -> list[int]:
    spec = dataset_spec(key)
    return list(range(int(spec["future_start"]), int(spec["frames"])))


def rendering_frames(key: str) -> list[int]:
    return reconstruction_holdouts(key) + future_frames(key)


def depth_cache(repo_root: Path, key: str) -> Path:
    return resolve_native(repo_root, key) / "endogaussian_foundation_depth_ds3_v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_world_points(points_world, intrinsic, world_from_camera, image_size_wh):
    points = np.asarray(points_world, dtype=np.float64)
    camera_from_world = np.linalg.inv(np.asarray(world_from_camera, dtype=np.float64))
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = (camera_from_world @ homogeneous.T).T[:, :3]
    z = camera[:, 2]
    pixels_h = (np.asarray(intrinsic, dtype=np.float64) @ camera.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pixels_h[:, :2] / z[:, None]
    width, height = map(int, image_size_wh)
    valid = (
        np.isfinite(camera).all(axis=1) & np.isfinite(uv).all(axis=1) & (z > 0)
        & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
        & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
    )
    return uv.astype(np.float32), valid, camera.astype(np.float32)
