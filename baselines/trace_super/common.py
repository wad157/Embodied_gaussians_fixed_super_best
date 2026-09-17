#!/usr/bin/env python3
"""RGB-only SUPER data, full-K cameras, and checkpoint helpers for TRACE."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from protocol import (
    CAMERAS,
    INIT_DEPTH_FAR_M,
    INIT_DEPTH_NEAR_M,
    INTERNAL_UNITS_PER_METER,
    OPENCV_FROM_OPENGL,
    dataset_spec,
    normalized_time,
    resolve_path,
    training_frames,
)

from scene.deform_model import DeformModel
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud, focal2fov, getWorld2View2
from utils.system_utils import searchForMaxIteration


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_calibration(repo_root: Path, dataset_key: str) -> Dict[str, dict]:
    spec = dataset_spec(dataset_key)
    offline = resolve_path(repo_root, spec["offline"])
    native = resolve_path(repo_root, spec["native"])
    calibration = read_json(native / "calib_rectified.json")
    camera_table = read_json(resolve_path(repo_root, spec["camera_file"]))
    result = {}
    for name in CAMERAS:
        suffix = "left" if name == "stereo_left" else "right"
        metadata = read_json(offline / "videos" / (name + ".json"))
        world_from_camera = np.asarray(
            camera_table[name]["X_WC"], dtype=np.float64
        ) @ OPENCV_FROM_OPENGL
        result[name] = {
            "K": np.asarray(
                calibration["K_{}_rect".format(suffix)], dtype=np.float64
            ),
            "timestamps": np.asarray(metadata["timestamps"], dtype=np.float64),
            "X_WC_opencv": world_from_camera,
            "resolution_wh": tuple(int(value) for value in metadata["resolution"]),
            "file_suffix": suffix,
        }
    return result


def scaled_intrinsic(intrinsic: np.ndarray, downsample: float) -> np.ndarray:
    result = np.asarray(intrinsic, dtype=np.float64).copy()
    result[0, :] /= float(downsample)
    result[1, :] /= float(downsample)
    result[2, :] = np.asarray([0.0, 0.0, 1.0])
    return result


def internal_world_from_camera(world_from_camera_m: np.ndarray) -> np.ndarray:
    result = np.asarray(world_from_camera_m, dtype=np.float64).copy()
    result[:3, 3] *= INTERNAL_UNITS_PER_METER
    return result


def camera_rt_from_world_from_camera(
    world_from_camera: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    camera_from_world = np.linalg.inv(np.asarray(world_from_camera, dtype=np.float64))
    return camera_from_world[:3, :3].T, camera_from_world[:3, 3]


def projection_matrix_from_k(
    znear: float,
    zfar: float,
    intrinsic: np.ndarray,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build the asymmetric 3DGS perspective matrix from the complete K."""
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError("K must be 3x3")
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    near_fx, near_fy = float(znear) / fx, float(znear) / fy
    left = -(float(width) - cx) * near_fx
    right = cx * near_fx
    bottom = (cy - float(height)) * near_fy
    top = cy * near_fy
    matrix = torch.zeros((4, 4), dtype=torch.float32)
    matrix[0, 0] = 2.0 * znear / (right - left)
    matrix[1, 1] = 2.0 * znear / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix


class FullKCamera:
    """TRACE camera preserving fx, fy, cx and cy without core changes."""

    def __init__(
        self,
        uid: int,
        rotation: np.ndarray,
        translation: np.ndarray,
        intrinsic: np.ndarray,
        image: torch.Tensor,
        image_name: str,
        fid: float,
        data_device: str = "cpu",
        znear: float = 0.01,
        zfar: float = 100.0,
    ):
        self.uid = int(uid)
        self.colmap_id = int(uid)
        self.R = np.asarray(rotation, dtype=np.float64)
        self.T = np.asarray(translation, dtype=np.float64)
        self.image_name = str(image_name)
        self.original_image = image.clamp(0.0, 1.0).to(data_device)
        self.image_height = int(image.shape[1])
        self.image_width = int(image.shape[2])
        self.K = np.asarray(intrinsic, dtype=np.float64)
        self.focal_x = float(self.K[0, 0])
        self.focal_y = float(self.K[1, 1])
        self.principal_x = float(self.K[0, 2])
        self.principal_y = float(self.K[1, 2])
        self.FoVx = focal2fov(self.focal_x, self.image_width)
        self.FoVy = focal2fov(self.focal_y, self.image_height)
        self.znear, self.zfar = float(znear), float(zfar)
        self.fid = torch.tensor([float(fid)], dtype=torch.float32, device=data_device)
        self.depth = None
        self.gt_alpha_mask = None
        world_view = torch.from_numpy(
            getWorld2View2(self.R, self.T).astype(np.float32)
        ).transpose(0, 1)
        projection = projection_matrix_from_k(
            self.znear, self.zfar, self.K, self.image_height, self.image_width
        ).transpose(0, 1)
        self.world_view_transform = world_view.to(data_device)
        self.projection_matrix = projection.to(data_device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load2device(self, data_device: str = "cuda") -> None:
        self.original_image = self.original_image.to(data_device)
        self.world_view_transform = self.world_view_transform.to(data_device)
        self.projection_matrix = self.projection_matrix.to(data_device)
        self.full_proj_transform = self.full_proj_transform.to(data_device)
        self.camera_center = self.camera_center.to(data_device)
        self.fid = self.fid.to(data_device)


class RenderCamera(FullKCamera):
    def __init__(
        self,
        intrinsic: np.ndarray,
        world_from_camera_m: np.ndarray,
        resolution_wh: Tuple[int, int],
        fid: float,
    ):
        width, height = resolution_wh
        rotation, translation = camera_rt_from_world_from_camera(
            internal_world_from_camera(world_from_camera_m)
        )
        empty = torch.zeros((3, int(height), int(width)), dtype=torch.float32)
        super().__init__(
            0,
            rotation,
            translation,
            intrinsic,
            empty,
            "render",
            fid,
            data_device="cuda",
        )


def load_rgb(path: Path, downsample: int) -> torch.Tensor:
    image = Image.open(str(path)).convert("RGB")
    if downsample > 1:
        image = image.resize(
            (image.width // downsample, image.height // downsample),
            resample=Image.Resampling.LANCZOS,
        )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)


class TraceSuperTrainingData:
    """Load legal prefix RGB/calibration, never PSM, masks, depth or GT."""

    def __init__(
        self,
        repo_root: Path,
        dataset_key: str,
        downsample: int,
        init_points: int,
        seed: int,
    ):
        self.repo_root = repo_root.resolve()
        self.dataset_key = dataset_key
        self.native_root = resolve_path(
            self.repo_root, dataset_spec(dataset_key)["native"]
        )
        self.frames = int(dataset_spec(dataset_key)["frames"])
        self.downsample = int(downsample)
        self.init_points = int(init_points)
        self.seed = int(seed)
        if self.downsample < 1 or self.init_points < 1000:
            raise ValueError("downsample must be >=1 and init_points >=1000")
        self.calibration = load_calibration(self.repo_root, dataset_key)
        self.allowed_frames = training_frames(dataset_key)
        self.train_cameras = self._load_training_cameras()
        first_legal = self.allowed_frames[0]
        self.init_cameras = [
            camera
            for camera in self.train_cameras
            if camera.super_frame_index == first_legal
        ]

    def _load_training_cameras(self) -> List[FullKCamera]:
        result = []
        uid = 0
        for frame in self.allowed_frames:
            for camera_name in CAMERAS:
                calibration = self.calibration[camera_name]
                image = load_rgb(
                    self.native_root
                    / "rgb"
                    / "{:06d}-{}.png".format(
                        frame, calibration["file_suffix"]
                    ),
                    self.downsample,
                )
                intrinsic = scaled_intrinsic(
                    calibration["K"], self.downsample
                )
                rotation, translation = camera_rt_from_world_from_camera(
                    internal_world_from_camera(calibration["X_WC_opencv"])
                )
                camera = FullKCamera(
                    uid,
                    rotation,
                    translation,
                    intrinsic,
                    image,
                    "{}_{:06d}".format(camera_name, frame),
                    normalized_time(frame, self.frames),
                    data_device="cpu",
                )
                camera.super_camera_name = camera_name
                camera.super_frame_index = frame
                result.append(camera)
                uid += 1
        expected = len(self.allowed_frames) * len(CAMERAS)
        if len(result) != expected:
            raise AssertionError(
                "Training camera count {} != {}".format(len(result), expected)
            )
        return result

    def initial_point_cloud(self) -> BasicPointCloud:
        """Official-style random initialization in the calibrated left frustum."""
        rng = np.random.RandomState(self.seed)
        camera = self.calibration["stereo_left"]
        intrinsic = camera["K"]
        width, height = camera["resolution_wh"]
        pixels_u = rng.uniform(0.0, float(width - 1), self.init_points)
        pixels_v = rng.uniform(0.0, float(height - 1), self.init_points)
        depths = rng.uniform(
            INIT_DEPTH_NEAR_M, INIT_DEPTH_FAR_M, self.init_points
        )
        camera_points = np.stack(
            (
                (pixels_u - intrinsic[0, 2]) / intrinsic[0, 0] * depths,
                (pixels_v - intrinsic[1, 2]) / intrinsic[1, 1] * depths,
                depths,
            ),
            axis=1,
        )
        homogeneous = np.concatenate(
            (camera_points, np.ones((self.init_points, 1), dtype=np.float64)),
            axis=1,
        )
        points_m = (
            np.asarray(camera["X_WC_opencv"], dtype=np.float64)
            @ homogeneous.T
        ).T[:, :3]
        points = (points_m * INTERNAL_UNITS_PER_METER).astype(np.float32)
        colors = (rng.random_sample((self.init_points, 3)) / 255.0).astype(
            np.float32
        )
        return BasicPointCloud(
            points=points,
            colors=colors,
            normals=np.zeros_like(points, dtype=np.float32),
        )


class TraceSuperScene:
    def __init__(
        self,
        args,
        gaussians,
        load_iteration=None,
        shuffle=True,
        resolution_scales=(1.0,),
        skip_train=False,
        skip_val=False,
        skip_test=False,
    ):
        del shuffle, resolution_scales, skip_train, skip_val, skip_test
        self.model_path = args.model_path
        self.gaussians = gaussians
        self.loaded_iter = None
        data = TraceSuperTrainingData(
            Path(args.super_repo_root),
            args.super_dataset_key,
            args.super_downsample,
            args.super_init_points,
            args.super_seed,
        )
        self.train_cameras = data.train_cameras
        self.init_cameras = data.init_cameras
        self.val_cameras = []
        self.test_cameras = []
        self.cameras_extent = INIT_DEPTH_FAR_M * INTERNAL_UNITS_PER_METER
        if load_iteration is not None:
            self.loaded_iter = (
                searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                if int(load_iteration) == -1
                else int(load_iteration)
            )
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_{}".format(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(
                data.initial_point_cloud(), self.cameras_extent
            )

    def save(self, iteration: int) -> None:
        self.gaussians.save_ply(
            os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_{}".format(iteration),
                "point_cloud.ply",
            )
        )

    def getTrainCameras(self, scale=1.0):
        del scale
        return self.train_cameras

    def getInitCameras(self, scale=1.0):
        del scale
        return self.init_cameras

    def getValCameras(self, scale=1.0):
        del scale
        return self.val_cameras

    def getTestCameras(self, scale=1.0):
        del scale
        return self.test_cameras


def load_trained_model(
    model_path: Path,
    iteration: int,
    sh_degree: int,
    max_time: float,
    fps: int,
    light: bool,
    physics_code: int,
    freegave: bool,
):
    root = model_path / "point_cloud"
    resolved = searchForMaxIteration(str(root)) if iteration < 0 else int(iteration)
    checkpoint = root / "iteration_{}".format(resolved) / "point_cloud.ply"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    gaussians = GaussianModel(int(sh_degree))
    gaussians.load_ply(str(checkpoint))
    deform = DeformModel(
        max_time=float(max_time),
        vel_start_time=0.0,
        light=bool(light),
        physics_code=int(physics_code),
        freegave=bool(freegave),
    )
    deform.load_weights(str(model_path), resolved)
    deform.deform.eval()
    deform.vel.eval()
    if freegave:
        deform.code_field.eval()
    return gaussians, deform, resolved


def classify_motion(gaussians, deform, fps: int, threshold: float = 0.01):
    """Use TRACE's official rendering-time static/moving classification."""
    xyz = gaussians.get_xyz
    zero = torch.zeros((len(xyz), 1), dtype=xyz.dtype, device=xyz.device)
    initial, _, _ = deform.step(xyz, zero, 1.0 / float(fps))
    moving = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
    for sample in range(75):
        time = torch.full_like(zero, float(sample) / 100.0)
        current, _, _ = deform.step(xyz, time, 1.0 / float(fps))
        moving |= (current - initial).norm(dim=1) >= float(threshold)
    return moving


def deformation_at_time(
    gaussians, deform, moving: torch.Tensor, time: float, fps: int
):
    xyz = gaussians.get_xyz
    d_xyz = torch.zeros_like(xyz)
    d_rotation = torch.zeros_like(gaussians.get_rotation)
    d_scaling = torch.zeros_like(gaussians.get_scaling)
    static = ~moving
    if bool(static.any()):
        zero = torch.zeros(
            (int(static.sum().item()), 1), dtype=xyz.dtype, device=xyz.device
        )
        d_xyz[static], d_rotation[static], d_scaling[static] = deform.step(
            xyz[static], zero, 1.0 / float(fps)
        )
    if bool(moving.any()):
        target = torch.full(
            (int(moving.sum().item()), 1),
            float(time),
            dtype=xyz.dtype,
            device=xyz.device,
        )
        d_xyz[moving], d_rotation[moving], d_scaling[moving] = deform.step(
            xyz[moving], target, 1.0 / float(fps)
        )
    return d_xyz, d_rotation, d_scaling
