#!/usr/bin/env python3
"""Observation-only SUPER helpers for PhysTwin."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from protocol import OPENCV_FROM_OPENGL, dataset_spec, read_json, resolve_path


def nearest_timestamp_index(timestamps: np.ndarray, timestamp: float) -> int:
    insertion = int(np.searchsorted(timestamps, timestamp, side="left"))
    candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(timestamps)]
    return min(candidates, key=lambda i: (abs(float(timestamps[i]) - timestamp), i))


def load_calibration(repo_root: Path, dataset_key: str) -> dict[str, object]:
    spec = dataset_spec(dataset_key)
    offline = resolve_path(repo_root, spec["offline"])
    native = resolve_path(repo_root, spec["native"])
    left = read_json(offline / "videos/stereo_left.json")
    cameras = read_json(resolve_path(repo_root, spec["camera_file"]))
    rectified = read_json(native / "calib_rectified.json")
    world_from_camera = np.asarray(cameras["stereo_left"]["X_WC"], dtype=np.float64) @ OPENCV_FROM_OPENGL
    return {
        "left_timestamps": np.asarray(left["timestamps"], dtype=np.float64),
        "stereo_left": {
            "K": np.asarray(rectified.get("K_left_rect", left["K"]), dtype=np.float32),
            "X_WC_cv": world_from_camera.astype(np.float32),
            "X_CW_cv": np.linalg.inv(world_from_camera).astype(np.float32),
            "resolution_wh": tuple(map(int, left["resolution"])),
            "timestamps": np.asarray(left["timestamps"], dtype=np.float64),
        },
    }


class TissueMasks:
    def __init__(self, native: Path):
        root = native / "visual_force_masks_v1"
        report = read_json(root / "report.json")
        self.width = int(report["resolution_wh"][0])
        self.array = np.load(root / "tissue_masks_packbits.npy", mmap_mode="r")

    def get(self, frame: int) -> np.ndarray:
        packed = np.asarray(self.array[int(frame)])
        return np.unpackbits(packed, axis=1, bitorder="big")[:, : self.width].astype(bool)


def load_rgb(native: Path, frame: int) -> np.ndarray:
    return np.asarray(Image.open(native / "rgb" / f"{frame:06d}-left.png").convert("RGB"))


def resize_to_depth(image: np.ndarray, depth_shape: tuple[int, int], *, nearest=False) -> np.ndarray:
    h, w = depth_shape
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(image, (w, h), interpolation=interpolation)


def sample_image(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    values = np.zeros((len(uv),) + image.shape[2:], dtype=image.dtype)
    values[valid] = image[y[valid], x[valid]]
    return values


def deproject_pixels(uv, depth, intrinsic, world_from_camera):
    h, w = depth.shape
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    z = np.zeros(len(uv), dtype=np.float64)
    z[inside] = depth[y[inside], x[inside]]
    valid = inside & np.isfinite(z) & (z > 0)
    camera = np.stack(((uv[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0] * z,
                       (uv[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1] * z,
                       z, np.ones(len(uv))), axis=1)
    world = (np.asarray(world_from_camera) @ camera.T).T[:, :3]
    valid &= np.isfinite(world).all(axis=1)
    return world.astype(np.float32), valid


def farthest_point_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= count:
        return np.arange(len(points), dtype=np.int64)
    rng = np.random.RandomState(seed)
    chosen = np.empty(count, dtype=np.int64)
    chosen[0] = int(rng.randint(len(points)))
    distance = np.full(len(points), np.inf)
    for index in range(1, count):
        delta = points - points[chosen[index - 1]]
        distance = np.minimum(distance, np.einsum("ij,ij->i", delta, delta))
        chosen[index] = int(np.argmax(distance))
    return chosen
