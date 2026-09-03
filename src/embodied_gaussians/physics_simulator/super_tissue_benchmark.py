"""Read-only SUPER tissue trajectory and rendering benchmark recorder."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from embodied_gaussians.embodied_simulator.visual_force_masks import (
    PackedStereoInstrumentMasks,
)


BENCHMARK_PROTOCOLS = ("reconstruction_7to1", "future_80to20")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project(points_world: torch.Tensor, x_camera_world: torch.Tensor, intrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    homogeneous = torch.cat(
        (points_world, torch.ones_like(points_world[:, :1])), dim=1
    )
    camera = (x_camera_world @ homogeneous.T).T[:, :3]
    pixels_h = (intrinsic @ camera.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:].clamp_min(1.0e-8)
    return pixels, camera


def _quat_rotate_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by normalized WXYZ quaternions without changing dtype."""
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    imaginary = quaternion[..., 1:]
    uv = torch.linalg.cross(imaginary, vector, dim=-1)
    return vector + 2.0 * (
        quaternion[..., :1] * uv
        + torch.linalg.cross(imaginary, uv, dim=-1)
    )


def _quat_rotate_inverse_wxyz(
    quaternion: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    conjugate = torch.cat(
        (quaternion[..., :1], -quaternion[..., 1:]), dim=-1
    )
    return _quat_rotate_wxyz(conjugate, vector)


def _masked_psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    selected = np.asarray(mask, dtype=bool)
    if not bool(selected.any()):
        return float("nan"), float("nan")
    difference = prediction[selected] - target[selected]
    mse = float(np.mean(difference * difference))
    psnr = float("inf") if mse == 0.0 else float(-10.0 * math.log10(mse))
    return mse, psnr


def _masked_ssim(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """11x11 Gaussian-window SSIM, averaged only over valid non-tool pixels."""
    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(3):
        x = prediction[..., channel].astype(np.float32)
        y = target[..., channel].astype(np.float32)
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        scores.append(numerator / np.maximum(denominator, 1.0e-12))
    score = np.mean(np.stack(scores, axis=-1), axis=-1)
    valid = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    return float(np.mean(score[valid])) if bool(valid.any()) else float("nan")


class SuperTissueBenchmarkRecorder:
    """Capture persistent Gaussian tracks and held-out stereo-left renders.

    This object never writes simulator state.  It only reads the already
    updated Gaussian state after a scheduled physics frame.
    """

    def __init__(
        self,
        output_directory: Path,
        ground_truth: Path,
        *,
        protocol: str,
        instrument_masks: Path,
        render_scale: float = 0.5,
        reconstruction_test_phase: int = 0,
        capture_renders: bool = True,
        future_test_start: int | None = None,
    ) -> None:
        if protocol not in BENCHMARK_PROTOCOLS:
            raise ValueError(f"Unknown benchmark protocol: {protocol}")
        if not (0.0 < render_scale <= 1.0):
            raise ValueError("render scale must lie in (0, 1]")
        if not 0 <= reconstruction_test_phase < 8:
            raise ValueError("reconstruction test phase must lie in 0..7")
        self.output_directory = Path(output_directory).resolve()
        self.ground_truth_path = Path(ground_truth).resolve()
        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise FileExistsError(f"Benchmark output is not empty: {self.output_directory}")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.render_directory = self.output_directory / "render_pairs"
        self.render_directory.mkdir()
        self.protocol = protocol
        self.render_scale = float(render_scale)
        self.reconstruction_test_phase = int(reconstruction_test_phase)
        self.capture_renders = bool(capture_renders)
        self.instrument_masks = PackedStereoInstrumentMasks(instrument_masks)
        with np.load(self.ground_truth_path, allow_pickle=False) as archive:
            self.gt = {name: np.asarray(archive[name]) for name in archive.files}
        self.frame_indices = self.gt["frame_indices"].astype(np.int32)
        self.frame_slot = {int(frame): slot for slot, frame in enumerate(self.frame_indices)}
        frozen_future_start = int(self.gt["future_test_start_frame"].item())
        self.future_test_start = (
            frozen_future_start
            if future_test_start is None
            else int(future_test_start)
        )
        if self.future_test_start < 1:
            raise ValueError("Future test start must be positive")
        self.bound_gaussian_ids: np.ndarray | None = None
        self.bound_local_offsets_m: np.ndarray | None = None
        self.bound_reference_quats_wxyz: np.ndarray | None = None
        self.binding_details: list[dict] = []
        self.track_records: list[dict] = []
        self.render_records: list[dict] = []
        self._captured_frames: set[int] = set()
        self.started = time.perf_counter()
        self.metadata = {
            "schema": "super_tissue_physics_benchmark_v1",
            "protocol": protocol,
            "ground_truth": str(self.ground_truth_path),
            "ground_truth_sha256": _sha256(self.ground_truth_path),
            "trajectory_binding": (
                "query point is initialized from strict stereo 3D at frame 0, "
                "then transported by one uniquely assigned persistent soft "
                "Gaussian using its stored local rigid offset"
            ),
            "trajectory_query_frame": 0,
            "reconstruction_split": {
                "ratio": [7, 1],
                "test_rule": f"full video frame_index % 8 == {self.reconstruction_test_phase}",
            },
            "future_split": {
                "train_end_inclusive": self.future_test_start - 1,
                "test_start_inclusive": self.future_test_start,
                "ground_truth_formal_test_start": frozen_future_start,
            },
            "render_scale": self.render_scale,
            "render_camera": "stereo_left",
            "render_mask": "all pixels except the frozen SurgicalSAM2 instrument mask",
            "capture_renders": self.capture_renders,
        }
        (self.output_directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8"
        )

    def observation_allowed(self, frame_index: int) -> bool:
        if self.protocol == "future_80to20":
            return int(frame_index) < self.future_test_start
        return int(frame_index) % 8 != self.reconstruction_test_phase

    def should_render(self, frame_index: int) -> bool:
        if not self.capture_renders:
            return False
        if self.protocol == "future_80to20":
            return int(frame_index) >= self.future_test_start
        return int(frame_index) % 8 == self.reconstruction_test_phase

    def _bind(self, environment, frames) -> None:
        if 0 not in self.frame_slot:
            raise ValueError("Ground truth must contain query frame 0")
        environment.sim.update_gaussian_transforms()
        soft_ids = environment.sim.gaussian_model.soft_gaussian_ids.long()
        means = environment.sim.gaussian_state.means[soft_ids]
        pixels, _camera = _project(
            means,
            frames.X_CWs_opencv_gpu[0],
            frames.Ks_gpu[0],
        )
        gt_slot = self.frame_slot[0]
        gt_world = torch.as_tensor(
            self.gt["xyz_world_m"][gt_slot], device=means.device, dtype=means.dtype
        )
        gt_uv = torch.as_tensor(
            self.gt["uv"][gt_slot], device=means.device, dtype=means.dtype
        )
        valid_3d = self.gt["valid_3d"][gt_slot].astype(bool)
        if not bool(valid_3d.all()):
            raise ValueError(
                "Every tracking query must have strict stereo 3D at frame 0"
            )
        chosen_local: list[int] = []
        used: set[int] = set()
        self.binding_details = []
        for point_id in range(len(gt_uv)):
            distance = torch.linalg.vector_norm(
                means - gt_world[point_id], dim=1
            )
            order = torch.argsort(distance)
            local_id = next(int(value.item()) for value in order if int(value.item()) not in used)
            used.add(local_id)
            chosen_local.append(local_id)
            self.binding_details.append(
                {
                    "point_id": point_id,
                    "mode": "3d_rigid_offset",
                    "soft_gaussian_local_id": local_id,
                    "gaussian_id": int(soft_ids[local_id].item()),
                    "query_distance_m": float(distance[local_id].item()),
                }
            )
        chosen = torch.as_tensor(chosen_local, device=soft_ids.device)
        chosen_ids = soft_ids[chosen]
        reference_means = environment.sim.gaussian_state.means[chosen_ids]
        reference_quats = environment.sim.gaussian_state.quats[chosen_ids]
        local_offsets = _quat_rotate_inverse_wxyz(
            reference_quats, gt_world - reference_means
        )
        reconstructed_queries = reference_means + _quat_rotate_wxyz(
            reference_quats, local_offsets
        )
        reconstruction_error = torch.linalg.vector_norm(
            reconstructed_queries - gt_world, dim=1
        )
        if float(reconstruction_error.max().item()) > 1.0e-7:
            raise RuntimeError("Persistent query binding failed its rest-pose gate")
        for point_id, detail in enumerate(self.binding_details):
            detail["query_anchor_reconstruction_error_m"] = float(
                reconstruction_error[point_id].item()
            )
            detail["local_offset_m"] = [
                float(value) for value in local_offsets[point_id].tolist()
            ]
        self.bound_gaussian_ids = (
            chosen_ids.detach().cpu().numpy().astype(np.int32)
        )
        self.bound_local_offsets_m = (
            local_offsets.detach().cpu().numpy().astype(np.float32)
        )
        self.bound_reference_quats_wxyz = (
            reference_quats.detach().cpu().numpy().astype(np.float32)
        )

    def capture(self, frame_index: int, environment, frames) -> None:
        frame_index = int(frame_index)
        if frame_index in self._captured_frames:
            return
        capture_track = frame_index in self.frame_slot
        capture_render = self.should_render(frame_index)
        if not capture_track and not capture_render:
            return
        if self.bound_gaussian_ids is None:
            self._bind(environment, frames)
        assert self.bound_gaussian_ids is not None
        environment.sim.update_gaussian_transforms()
        if capture_track:
            assert self.bound_local_offsets_m is not None
            ids = torch.as_tensor(
                self.bound_gaussian_ids,
                device=environment.sim.gaussian_state.means.device,
                dtype=torch.long,
            )
            means = environment.sim.gaussian_state.means[ids]
            quats = environment.sim.gaussian_state.quats[ids]
            local_offsets = torch.as_tensor(
                self.bound_local_offsets_m,
                device=means.device,
                dtype=means.dtype,
            )
            tracked_points = means + _quat_rotate_wxyz(quats, local_offsets)
            pixels, camera = _project(
                tracked_points,
                frames.X_CWs_opencv_gpu[0],
                frames.Ks_gpu[0],
            )
            self.track_records.append(
                {
                    "frame_index": frame_index,
                    "observation_used": self.observation_allowed(frame_index),
                    "uv": pixels.detach().cpu().numpy().astype(np.float32),
                    "xyz_camera_m": camera.detach().cpu().numpy().astype(np.float32),
                    "xyz_world_m": tracked_points.detach().cpu().numpy().astype(np.float32),
                }
            )
        if capture_render:
            self._capture_render(frame_index, environment, frames)
        self._captured_frames.add(frame_index)

    def _capture_render(self, frame_index: int, environment, frames) -> None:
        source_width, source_height = int(frames.width), int(frames.height)
        width = max(1, int(round(source_width * self.render_scale)))
        height = max(1, int(round(source_height * self.render_scale)))
        intrinsic = frames.Ks_gpu[0:1].clone()
        intrinsic[:, 0] *= width / float(source_width)
        intrinsic[:, 1] *= height / float(source_height)
        background = torch.zeros(3, device=intrinsic.device, dtype=torch.float32)
        started = time.perf_counter()
        render, _alpha, _info = environment.sim.render_gaussians(
            environment.sim.gaussian_state,
            frames.X_CWs_opencv_gpu[0:1],
            intrinsic,
            width,
            height,
            background,
        )
        torch.cuda.synchronize(render.device)
        elapsed = time.perf_counter() - started
        prediction = render[0, ..., :3].detach().cpu().numpy().astype(np.float32)
        target_native = frames.colors_gpu[0].flip(-1).detach().cpu().numpy().astype(np.float32)
        target = cv2.resize(target_native, (width, height), interpolation=cv2.INTER_AREA)
        masks = self.instrument_masks.masks_for_camera_frame("stereo_left", frame_index)
        if masks is None:
            raise RuntimeError(f"No quality-valid instrument mask for left frame {frame_index}")
        mask_source_frame = int(self.instrument_masks.last_source_frame_index)
        mask_frame_gap = int(self.instrument_masks.last_frame_gap)
        instrument, _distal = masks
        instrument = cv2.resize(instrument.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        valid = ~instrument
        mse, psnr = _masked_psnr(prediction, target, valid)
        ssim = _masked_ssim(prediction, target, valid)
        masked_prediction = prediction.copy()
        masked_target = target.copy()
        masked_prediction[~valid] = 0.0
        masked_target[~valid] = 0.0
        stem = f"{frame_index:06d}"
        cv2.imwrite(
            str(self.render_directory / f"{stem}-prediction.png"),
            cv2.cvtColor(np.clip(masked_prediction * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(self.render_directory / f"{stem}-target.png"),
            cv2.cvtColor(np.clip(masked_target * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(self.render_directory / f"{stem}-valid.png"), valid.astype(np.uint8) * 255)
        self.render_records.append(
            {
                "frame_index": frame_index,
                "observation_used": self.observation_allowed(frame_index),
                "mse": mse,
                "psnr_db": psnr,
                "ssim": ssim,
                "valid_pixel_count": int(valid.sum()),
                "instrument_mask_source_frame": mask_source_frame,
                "instrument_mask_frame_gap": mask_frame_gap,
                "render_elapsed_s": elapsed,
            }
        )

    def close(self) -> None:
        if self.bound_gaussian_ids is None:
            return
        assert self.bound_local_offsets_m is not None
        assert self.bound_reference_quats_wxyz is not None
        ordered = sorted(self.track_records, key=lambda item: item["frame_index"])
        np.savez_compressed(
            self.output_directory / "predicted_tracks.npz",
            schema=np.asarray("super_tissue_predicted_tracks_v1"),
            frame_indices=np.asarray([item["frame_index"] for item in ordered], dtype=np.int32),
            observation_used=np.asarray([item["observation_used"] for item in ordered], dtype=bool),
            uv=np.stack([item["uv"] for item in ordered]).astype(np.float32),
            xyz_camera_m=np.stack([item["xyz_camera_m"] for item in ordered]).astype(np.float32),
            xyz_world_m=np.stack([item["xyz_world_m"] for item in ordered]).astype(np.float32),
            bound_gaussian_ids=self.bound_gaussian_ids,
            bound_local_offsets_m=self.bound_local_offsets_m,
            bound_reference_quats_wxyz=self.bound_reference_quats_wxyz,
            query_frame_index=np.asarray(0, dtype=np.int32),
        )
        render_metrics = {
            "schema": "super_tissue_render_metrics_without_lpips_v1",
            "protocol": self.protocol,
            "records": self.render_records,
            "mean": {
                key: float(np.mean([record[key] for record in self.render_records])) if self.render_records else None
                for key in ("mse", "psnr_db", "ssim", "render_elapsed_s")
            },
            "note": "LPIPS is computed by the separate final scorer from the saved masked image pairs.",
        }
        (self.output_directory / "render_metrics_partial.json").write_text(
            json.dumps(render_metrics, indent=2) + "\n", encoding="utf-8"
        )
        summary = {
            "schema": "super_tissue_physics_benchmark_capture_v1",
            "protocol": self.protocol,
            "track_frame_count": len(ordered),
            "render_frame_count": len(self.render_records),
            "binding": self.binding_details,
            "elapsed_s": time.perf_counter() - self.started,
        }
        (self.output_directory / "capture_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
