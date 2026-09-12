#!/usr/bin/env python3
"""SUPER inputs and cameras for the unmodified EndoGaussian implementation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from protocol import (
    INTERNAL_UNITS_PER_METER,
    OPENCV_FROM_OPENGL,
    TRAIN_DOWNSAMPLE,
    dataset_spec,
    depth_cache,
    normalized_time,
    resolve_path,
    training_frames,
)

# Resolved from the pinned upstream checkout by the launcher PYTHONPATH.
from scene.cameras import Camera
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, getWorld2View2
from utils.system_utils import searchForMaxIteration


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_calibration(repo_root: Path, dataset_key: str) -> Dict[str, object]:
    spec = dataset_spec(dataset_key)
    offline = resolve_path(repo_root, spec["offline"])
    native = resolve_path(repo_root, spec["native"])
    left_metadata = read_json(offline / "videos" / "stereo_left.json")
    right_metadata = read_json(offline / "videos" / "stereo_right.json")
    camera_table = read_json(resolve_path(repo_root, spec["camera_file"]))
    calibration = read_json(native / "calib_rectified.json")
    world_from_camera_gl = np.asarray(
        camera_table["stereo_left"]["X_WC"], dtype=np.float64
    )
    return {
        "K": np.asarray(calibration["K_left_rect"], dtype=np.float64),
        "K_right": np.asarray(calibration["K_right_rect"], dtype=np.float64),
        "timestamps": np.asarray(left_metadata["timestamps"], dtype=np.float64),
        "right_timestamps": np.asarray(right_metadata["timestamps"], dtype=np.float64),
        "X_WC_opencv": world_from_camera_gl @ OPENCV_FROM_OPENGL,
        "resolution_wh": tuple(int(value) for value in left_metadata["resolution"]),
    }


class PackedNonInstrumentMasks(object):
    """Match the official EndoGaussian mask meaning: one outside the tool."""

    def __init__(self, path: Path):
        with np.load(str(path), allow_pickle=False) as archive:
            self.shape = tuple(int(value) for value in archive["mask_shape"])
            self.bitorder = str(archive["bitorder"].item())
            quality = np.asarray(archive["quality_valid"], dtype=bool)
            frames = np.asarray(archive["stereo_left_index"], dtype=np.int64)
            self.packed = np.asarray(archive["left_masks_packbits"], dtype=np.uint8)
        self.lookup = {
            int(frame): int(slot)
            for slot, frame in enumerate(frames)
            if bool(quality[slot])
        }

    def get(self, frame: int) -> np.ndarray:
        source = int(frame)
        slot = self.lookup.get(source)
        if slot is None:
            for candidate in (source - 1, source + 1):
                if candidate in self.lookup:
                    slot = self.lookup[candidate]
                    break
        if slot is None:
            raise RuntimeError("No quality-valid instrument mask near frame {}".format(frame))
        instrument = np.unpackbits(
            self.packed[slot],
            bitorder=self.bitorder,
            count=self.shape[0] * self.shape[1],
        ).reshape(self.shape).astype(bool)
        return ~instrument


def scaled_intrinsic(intrinsic: np.ndarray, downsample: int) -> np.ndarray:
    result = np.asarray(intrinsic, dtype=np.float64).copy()
    result[0, :] /= float(downsample)
    result[1, :] /= float(downsample)
    result[2, :] = np.asarray([0.0, 0.0, 1.0])
    return result


def camera_rt_from_world_from_camera(world_from_camera: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    camera_from_world = np.linalg.inv(np.asarray(world_from_camera, dtype=np.float64))
    return camera_from_world[:3, :3].T, camera_from_world[:3, 3]


def projection_from_intrinsic(
    intrinsic: np.ndarray,
    width: int,
    height: int,
    znear: float,
    zfar: float,
) -> torch.Tensor:
    """Build the transposed 3DGS projection for a non-centered OpenCV K."""
    matrix = np.asarray(intrinsic, dtype=np.float64)
    if matrix.shape != (3, 3) or width < 1 or height < 1:
        raise ValueError("Invalid intrinsic or image dimensions")
    if not (0.0 < znear < zfar):
        raise ValueError("Expected 0 < znear < zfar")
    projection = torch.zeros((4, 4), dtype=torch.float32)
    projection[0, 0] = 2.0 * float(matrix[0, 0]) / float(width)
    projection[0, 1] = 2.0 * float(matrix[0, 1]) / float(width)
    projection[1, 1] = 2.0 * float(matrix[1, 1]) / float(height)
    projection[0, 2] = 2.0 * float(matrix[0, 2]) / float(width) - 1.0
    projection[1, 2] = 2.0 * float(matrix[1, 2]) / float(height) - 1.0
    projection[2, 2] = float(zfar) / float(zfar - znear)
    projection[2, 3] = -float(zfar * znear) / float(zfar - znear)
    projection[3, 2] = 1.0
    return projection.transpose(0, 1)


def apply_intrinsic_projection(camera: Camera, intrinsic: np.ndarray) -> Camera:
    """Replace only the centered-FoV camera projection used by upstream."""
    camera.projection_matrix = projection_from_intrinsic(
        intrinsic,
        int(camera.image_width),
        int(camera.image_height),
        float(camera.znear),
        float(camera.zfar),
    )
    camera.full_proj_transform = (
        camera.world_view_transform.unsqueeze(0)
        .bmm(camera.projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    return camera


def internal_world_from_camera(world_from_camera_m: np.ndarray) -> np.ndarray:
    result = np.asarray(world_from_camera_m, dtype=np.float64).copy()
    result[:3, 3] *= INTERNAL_UNITS_PER_METER
    return result


def load_training_arrays(
    native_root: Path,
    depth_root: Path,
    masks: PackedNonInstrumentMasks,
    frame: int,
    downsample: int,
):
    image_path = native_root / "rgb" / "{:06d}-left.png".format(frame)
    image = np.asarray(Image.open(str(image_path)).convert("RGB"), dtype=np.float32) / 255.0
    depth_path = depth_root / "{:06d}-depth.npy".format(frame)
    depth = np.asarray(np.load(str(depth_path)), dtype=np.float32)
    non_instrument = masks.get(frame)
    height, width = image.shape[:2]
    size = (width // downsample, height // downsample)
    if downsample > 1:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        non_instrument = cv2.resize(
            non_instrument.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    if depth.shape != (size[1], size[0]):
        raise ValueError("Depth {} has shape {}, expected {}".format(depth_path, depth.shape, (size[1], size[0])))
    depth = depth * INTERNAL_UNITS_PER_METER
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    valid = non_instrument & valid_depth
    depth = np.where(valid_depth, depth, 0.0).astype(np.float32)
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depth))
    mask_tensor = torch.from_numpy(np.ascontiguousarray(valid[None]))
    return image_tensor, depth_tensor, mask_tensor


class LazyTrainingCameras(Sequence):
    """Random-access views without retaining every full image tensor in RAM."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.allowed_frames)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.data.camera_for_frame(self.data.allowed_frames[index], index)


class SuperTrainingData(object):
    """Read only legal prefix observations; evaluation truth is never opened."""

    def __init__(
        self,
        repo_root: Path,
        dataset_key: str,
        downsample: int = TRAIN_DOWNSAMPLE,
        init_points: int = 30000,
        init_views: int = 64,
        seed: int = 0,
    ):
        self.repo_root = repo_root.resolve()
        self.dataset_key = dataset_key
        self.spec = dataset_spec(dataset_key)
        self.frames = int(self.spec["frames"])
        self.downsample = int(downsample)
        self.init_points = int(init_points)
        self.init_views = int(init_views)
        self.seed = int(seed)
        if self.downsample < 1 or self.init_points < 100 or self.init_views < 1:
            raise ValueError("Invalid downsample/init_points/init_views")
        self.native_root = resolve_path(self.repo_root, self.spec["native"])
        self.depth_root = depth_cache(self.repo_root, dataset_key, self.downsample)
        self.calibration = load_calibration(self.repo_root, dataset_key)
        self.masks = PackedNonInstrumentMasks(
            resolve_path(self.repo_root, self.spec["instrument_masks"])
        )
        self.allowed_frames = training_frames(dataset_key)
        self.train_cameras = LazyTrainingCameras(self)

    def camera_for_frame(self, frame: int, uid: int) -> Camera:
        image, depth, mask = load_training_arrays(
            self.native_root,
            self.depth_root,
            self.masks,
            frame,
            self.downsample,
        )
        intrinsic = scaled_intrinsic(self.calibration["K"], self.downsample)
        width = int(image.shape[2])
        height = int(image.shape[1])
        rotation, translation = camera_rt_from_world_from_camera(
            internal_world_from_camera(self.calibration["X_WC_opencv"])
        )
        camera = Camera(
            colmap_id=uid,
            R=rotation,
            T=translation,
            FoVx=focal2fov(float(intrinsic[0, 0]), width),
            FoVy=focal2fov(float(intrinsic[1, 1]), height),
            image=image,
            depth=depth,
            mask=mask,
            gt_alpha_mask=None,
            image_name="stereo_left_{:06d}".format(frame),
            uid=uid,
            data_device=torch.device("cuda"),
            time=normalized_time(frame, self.frames),
            Znear=0.01,
            Zfar=1000.0,
        )
        apply_intrinsic_projection(camera, intrinsic)
        camera.super_frame_index = frame
        camera.super_intrinsic = intrinsic
        return camera

    def initial_point_cloud(self) -> BasicPointCloud:
        rng = np.random.RandomState(self.seed)
        view_count = min(self.init_views, len(self.allowed_frames))
        slots = np.unique(
            np.linspace(0, len(self.allowed_frames) - 1, view_count).round().astype(np.int64)
        )
        per_view = int(math.ceil(float(self.init_points) / len(slots)))
        all_points = []
        all_colors = []
        intrinsic = scaled_intrinsic(self.calibration["K"], self.downsample)
        world_from_camera = internal_world_from_camera(self.calibration["X_WC_opencv"])
        for slot in slots:
            frame = self.allowed_frames[int(slot)]
            image, depth_tensor, mask_tensor = load_training_arrays(
                self.native_root,
                self.depth_root,
                self.masks,
                frame,
                self.downsample,
            )
            depth = depth_tensor.numpy()
            valid_indices = np.flatnonzero(mask_tensor.numpy().reshape(-1))
            if len(valid_indices) == 0:
                continue
            count = min(per_view, len(valid_indices))
            selected = rng.choice(valid_indices, size=count, replace=False)
            y, x = np.unravel_index(selected, depth.shape)
            z = depth[y, x].astype(np.float64)
            camera_points = np.stack(
                (
                    (x.astype(np.float64) - intrinsic[0, 2]) / intrinsic[0, 0] * z,
                    (y.astype(np.float64) - intrinsic[1, 2]) / intrinsic[1, 1] * z,
                    z,
                ),
                axis=1,
            )
            homogeneous = np.concatenate(
                (camera_points, np.ones((len(camera_points), 1), dtype=np.float64)), axis=1
            )
            world_points = (world_from_camera @ homogeneous.T).T[:, :3]
            colors = image[:, y, x].permute(1, 0).numpy().astype(np.float64)
            all_points.append(world_points)
            all_colors.append(colors)
        if not all_points:
            raise ValueError("No valid FoundationStereo tissue points for initialization")
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        selected = rng.choice(len(points), size=self.init_points, replace=len(points) < self.init_points)
        points = points[selected].astype(np.float32)
        colors = colors[selected].astype(np.float32)
        return BasicPointCloud(
            points=points,
            colors=colors,
            normals=np.zeros_like(points, dtype=np.float32),
        )


class SuperScene(object):
    """Drop-in Scene selected by the external training wrapper."""

    def __init__(
        self,
        args,
        gaussians,
        load_iteration=None,
        shuffle=True,
        resolution_scales=(1.0,),
        load_coarse=False,
    ):
        del shuffle, resolution_scales
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.mode = args.mode
        if load_iteration is not None:
            self.loaded_iter = (
                searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                if int(load_iteration) == -1
                else int(load_iteration)
            )
        data = SuperTrainingData(
            Path(args.super_repo_root),
            args.super_dataset_key,
            downsample=args.super_downsample,
            init_points=args.super_init_points,
            init_views=args.super_init_views,
            seed=args.super_seed,
        )
        self.train_camera = data.train_cameras
        self.test_camera = []
        self.video_camera = []
        point_cloud = data.initial_point_cloud()
        xyz_max = point_cloud.points.max(axis=0)
        xyz_min = point_cloud.points.min(axis=0)
        self.maxtime = 1.0
        self.cameras_extent = float(args.camera_extent)
        self.gaussians._deformation.deformation_net.grid.set_aabb(xyz_max, xyz_min)
        if self.loaded_iter is not None:
            prefix = "coarse_iteration_" if load_coarse else "iteration_"
            model_root = os.path.join(
                self.model_path, "point_cloud", prefix + str(self.loaded_iter)
            )
            self.gaussians.load_ply(os.path.join(model_root, "point_cloud.ply"))
            self.gaussians.load_model(model_root)
        else:
            self.gaussians.create_from_pcd(point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage):
        path = os.path.join(
            self.model_path,
            "point_cloud",
            ("coarse_iteration_" if stage == "coarse" else "iteration_") + str(iteration),
        )
        self.gaussians.save_ply(os.path.join(path, "point_cloud.ply"))
        self.gaussians.save_deformation(path)

    def getTrainCameras(self, scale=1.0):
        del scale
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        del scale
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        del scale
        return self.video_camera


class RenderCamera(object):
    def __init__(self, intrinsic, world_from_camera, resolution_wh, time):
        width, height = resolution_wh
        rotation, translation = camera_rt_from_world_from_camera(
            internal_world_from_camera(world_from_camera)
        )
        self.image_width = int(width)
        self.image_height = int(height)
        self.FoVx = focal2fov(float(intrinsic[0, 0]), self.image_width)
        self.FoVy = focal2fov(float(intrinsic[1, 1]), self.image_height)
        self.znear = 0.01
        self.zfar = 1000.0
        self.time = float(time)
        world_view = getWorld2View2(rotation, translation)
        projection = projection_from_intrinsic(
            intrinsic,
            self.image_width,
            self.image_height,
            self.znear,
            self.zfar,
        )
        self.world_view_transform = torch.from_numpy(world_view).transpose(0, 1)
        self.projection_matrix = projection
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def load_trained_gaussians(model_path: Path, iteration: int, sh_degree: int, hyper):
    from scene.gaussian_model import GaussianModel

    model_root = model_path / "point_cloud"
    resolved_iteration = searchForMaxIteration(str(model_root)) if iteration < 0 else int(iteration)
    checkpoint = model_root / "iteration_{}".format(resolved_iteration)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    gaussians = GaussianModel(sh_degree, hyper)
    gaussians.load_ply(str(checkpoint / "point_cloud.ply"))
    gaussians.load_model(str(checkpoint))
    return gaussians, resolved_iteration


def deformed_centers(gaussians, time: float) -> torch.Tensor:
    means = gaussians.get_xyz
    selected = gaussians._deformation_table
    result = means.clone()
    if bool(selected.any()):
        times = torch.full(
            (int(selected.sum().item()), 1),
            float(time),
            dtype=means.dtype,
            device=means.device,
        )
        deformed, _, _, _ = gaussians._deformation(
            means[selected],
            gaussians._scaling[selected],
            gaussians._rotation[selected],
            gaussians._opacity[selected],
            times,
        )
        result[selected] = deformed
    return result
