#!/usr/bin/env python3
"""SUPER stereo I/O shared by the Embodied Gaussians adapter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .protocol import CAMERAS, OPENCV_FROM_OPENGL, dataset_spec, read_json, resolve_path


def nearest_timestamp_index(timestamps: np.ndarray, timestamp: float) -> int:
    insertion = int(np.searchsorted(timestamps, timestamp, side="left"))
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(timestamps)]
    return min(candidates, key=lambda index: (abs(float(timestamps[index]) - timestamp), index))


def load_calibration(repo_root: Path, dataset_key: str) -> dict[str, object]:
    spec = dataset_spec(dataset_key)
    offline = resolve_path(repo_root, spec["offline"])
    native = resolve_path(repo_root, spec["native"])
    left = read_json(offline / "videos/stereo_left.json")
    right = read_json(offline / "videos/stereo_right.json")
    cameras = read_json(resolve_path(repo_root, spec["camera_file"]))
    rectified = read_json(native / "calib_rectified.json")
    result: dict[str, object] = {
        "left_timestamps": np.asarray(left["timestamps"], dtype=np.float64),
        "right_timestamps": np.asarray(right["timestamps"], dtype=np.float64),
        "resolution_wh": tuple(map(int, left["resolution"])),
    }
    for camera, metadata, key in (
        ("stereo_left", left, "K_left_rect"),
        ("stereo_right", right, "K_right_rect"),
    ):
        world_from_camera_gl = np.asarray(cameras[camera]["X_WC"], dtype=np.float64)
        world_from_camera_cv = world_from_camera_gl @ OPENCV_FROM_OPENGL
        result[camera] = {
            "K": np.asarray(rectified.get(key, metadata["K"]), dtype=np.float32),
            "X_WC_gl": world_from_camera_gl.astype(np.float32),
            "X_WC_cv": world_from_camera_cv.astype(np.float32),
            "X_CW_cv": np.linalg.inv(world_from_camera_cv).astype(np.float32),
            "timestamps": np.asarray(metadata["timestamps"], dtype=np.float64),
            "resolution_wh": tuple(map(int, metadata["resolution"])),
        }
    return result


class TissueMasks:
    def __init__(self, native: Path):
        self.data: dict[str, np.ndarray] = {}
        self.width: dict[str, int] = {}
        roots = {
            "stereo_left": native / "visual_force_masks_v1",
            "stereo_right": native / "visual_force_masks_right_v1",
        }
        for camera, root in roots.items():
            report = read_json(root / "report.json")
            self.width[camera] = int(report["resolution_wh"][0])
            self.data[camera] = np.load(root / "tissue_masks_packbits.npy", mmap_mode="r")

    def get(self, camera: str, frame: int) -> np.ndarray:
        packed = np.asarray(self.data[camera][int(frame)])
        return np.unpackbits(packed, axis=1, bitorder="big")[:, : self.width[camera]].astype(bool)


class SuperStereoInputs:
    """Timestamp-aligned left/right RGB and tissue masks; no depth after initialization."""

    def __init__(self, repo_root: Path, dataset_key: str, online_width: int):
        spec = dataset_spec(dataset_key)
        self.native = resolve_path(repo_root, spec["native"])
        self.calibration = load_calibration(repo_root, dataset_key)
        self.online_width = int(online_width)
        self.masks = TissueMasks(self.native)
        self.left_timestamps = np.asarray(self.calibration["left_timestamps"])

    def source_frame(self, camera: str, left_frame: int) -> int:
        if camera == "stereo_left":
            return int(left_frame)
        timestamps = np.asarray(self.calibration[camera]["timestamps"])
        return nearest_timestamp_index(timestamps, float(self.left_timestamps[left_frame]))

    def load_online(self, camera: str, frame: int, device: torch.device):
        source = self.source_frame(camera, frame)
        side = "left" if camera == "stereo_left" else "right"
        image = np.asarray(
            Image.open(self.native / "rgb" / f"{source:06d}-{side}.png").convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        mask = self.masks.get(camera, source)
        source_height, source_width = image.shape[:2]
        target_height = int(round(source_height * self.online_width / source_width))
        size = (self.online_width, target_height)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        image *= mask[..., None]
        intrinsic = np.asarray(self.calibration[camera]["K"], dtype=np.float32).copy()
        intrinsic[0, :] *= size[0] / source_width
        intrinsic[1, :] *= size[1] / source_height
        intrinsic[2, :] = (0.0, 0.0, 1.0)
        return (
            torch.as_tensor(np.ascontiguousarray(image), device=device),
            torch.as_tensor(intrinsic, device=device).unsqueeze(0),
            torch.as_tensor(self.calibration[camera]["X_CW_cv"], device=device).unsqueeze(0),
            size,
        )

    def render_camera_tensors(self, camera: str, scale: float, device: torch.device):
        width, height = self.calibration[camera]["resolution_wh"]
        size = (int(round(width * scale)), int(round(height * scale)))
        intrinsic = np.asarray(self.calibration[camera]["K"], dtype=np.float32).copy()
        intrinsic[0, :] *= size[0] / width
        intrinsic[1, :] *= size[1] / height
        intrinsic[2, :] = (0.0, 0.0, 1.0)
        return (
            torch.as_tensor(intrinsic, device=device).unsqueeze(0),
            torch.as_tensor(self.calibration[camera]["X_CW_cv"], device=device).unsqueeze(0),
            size,
        )
