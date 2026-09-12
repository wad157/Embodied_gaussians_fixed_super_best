#!/usr/bin/env python3
"""Frozen EH-SurGS definitions for the current SUPER benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np


UPSTREAM_COMMIT = "73fa04e6f5c21cc1685f728eccb1332e81ce620c"
CAMERA_PATCH_SHA256 = "f75e5d151e7f885024e6c75ccc3a1457ebc86389d8277d433cae58175bc310d8"
SHAPE_OF_MOTION_COMMIT = "579753e1c7ba96f60cd7690e5b835627bd1935e9"
TRACK_DECODER_VERSION = "som_query_anchored_displacement_v2"
ADAPTER_VERSION = "eh_surgs_super_v1_noninstrument_mask"
PROTOCOL = "joint_reconstruction_7to1_future_80to20"
QUERY_FRAME = 0
INTERNAL_UNITS_PER_METER = 1000.0
HOLDOUT_STRIDE = 8
HOLDOUT_PHASE = 0
TRAIN_DOWNSAMPLE = 3
RENDER_SCALE = 0.5
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
    },
    "grasp3": {
        "frames": 2062,
        "future_start": 1649,
        "native": "data/super/grasp3_native",
        "offline": "data/super/grasp3_offline_demo",
        "camera_file": "data/super/grasp3_native/bodies_v9_dense_0p5mm_rigid_tissue/cameras_table.json",
        "ground_truth": "data/super/grasp3_native/evaluation_v1/manual_tissue_tracks_10_v2/ground_truth_2d3d_v1.npz",
        "instrument_masks": "data/super/grasp3_native/instrument_manual_calibration_v1/stereo_annotations/surgicalsam2_multianchor_parts_v4/stereo_multianchor_part_masks.npz",
    },
    "grasp1": {
        "frames": 4205,
        "future_start": 3364,
        "native": "data/super/grasp1_native",
        "offline": "data/super/grasp1_offline_demo",
        "camera_file": "data/super/grasp1_native/bodies_v9_dense_0p5mm_rigid_tissue/cameras_table.json",
        "ground_truth": "data/super/grasp1_native/evaluation_v2/manual_tissue_tracks_10_no_exclusion/ground_truth_2d3d_v1.npz",
        "instrument_masks": "data/super/grasp1_native/instrument_manual_calibration_v1/stereo_annotations/surgicalsam2_multianchor_parts_v4/stereo_multianchor_part_masks.npz",
    },
}


def dataset_spec(key: str) -> Mapping[str, object]:
    try:
        return DATASETS[key.lower()]
    except KeyError as error:
        raise ValueError("Unknown SUPER dataset {}; choose from {}".format(key, sorted(DATASETS))) from error


def resolve_path(repo_root: Path, value: object) -> Path:
    return (repo_root / str(value)).resolve()


def resolve_native(repo_root: Path, key: str, supplied: Optional[Path] = None) -> Path:
    expected = resolve_path(repo_root, dataset_spec(key)["native"])
    actual = expected if supplied is None else supplied.expanduser().resolve()
    if actual != expected:
        raise ValueError("Formal baseline requires {}; received {}".format(expected, actual))
    if not actual.is_dir():
        raise FileNotFoundError(actual)
    return actual


def training_frames(key: str) -> List[int]:
    split = int(dataset_spec(key)["future_start"])
    return [frame for frame in range(split) if frame % HOLDOUT_STRIDE != HOLDOUT_PHASE]


def reconstruction_holdouts(key: str) -> List[int]:
    split = int(dataset_spec(key)["future_start"])
    return [frame for frame in range(split) if frame % HOLDOUT_STRIDE == HOLDOUT_PHASE]


def future_frames(key: str) -> List[int]:
    spec = dataset_spec(key)
    return list(range(int(spec["future_start"]), int(spec["frames"])))


def rendering_frames(key: str) -> List[int]:
    return reconstruction_holdouts(key) + future_frames(key)


def normalized_time(frame: int, frame_count: int) -> float:
    """Match the official EH-SurGS/EndoNeRF loader: index divided by N."""
    return float(frame) / float(frame_count)


def depth_cache(repo_root: Path, key: str, downsample: int = TRAIN_DOWNSAMPLE) -> Path:
    native = resolve_path(repo_root, dataset_spec(key)["native"])
    return native / "endogaussian_foundation_depth_ds{}_v1".format(int(downsample))


def load_json(path: Path) -> Dict[str, object]:
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
    homogeneous = np.concatenate(
        (points, np.ones((points.shape[0], 1), dtype=np.float64)), axis=1
    )
    camera = (camera_from_world @ homogeneous.T).T[:, :3]
    z = camera[:, 2]
    pixels_h = (np.asarray(intrinsic, dtype=np.float64) @ camera.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pixels_h[:, :2] / z[:, None]
    width, height = image_size_wh
    valid = (
        np.isfinite(camera).all(axis=1)
        & np.isfinite(uv).all(axis=1)
        & (z > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= width - 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= height - 1.0)
    )
    return uv.astype(np.float32), valid.astype(bool), camera.astype(np.float32)
