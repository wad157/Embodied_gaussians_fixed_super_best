from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _rotx(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def _rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def _transx(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = distance
    return transform


def _transz(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = distance
    return transform


def _modified_dh(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    return _rotx(alpha) @ _transx(a) @ _rotz(theta) @ _transz(d)


@dataclass(frozen=True)
class PSMLNDKinematics:
    dh_params: tuple[dict, ...]
    T_rectified_camera_psm_base: np.ndarray
    X_table_camera: np.ndarray
    link_names: tuple[str, ...]
    lnd_link_ids: np.ndarray
    T_lndlink_urdf_link: np.ndarray

    @classmethod
    def from_files(
        cls,
        lnd_model_path: Path,
        pose_report_path: Path,
        table_frame_path: Path,
        link_names: list[str],
    ) -> "PSMLNDKinematics":
        lnd_model = json.loads(lnd_model_path.read_text(encoding="utf-8"))
        pose_report = json.loads(pose_report_path.read_text(encoding="utf-8"))
        table_frame = json.loads(table_frame_path.read_text(encoding="utf-8"))
        lnd_link_ids = np.asarray(
            [pose_report["link_mapping"][name] for name in link_names], dtype=np.int64
        )
        link_offsets = np.stack(
            [
                np.asarray(pose_report["T_lndlink_urdf_link"][name], dtype=np.float64)
                for name in link_names
            ]
        )
        return cls(
            dh_params=tuple(lnd_model["dh_params"]),
            T_rectified_camera_psm_base=np.asarray(
                lnd_model["T_rectified_camera_psm_base"], dtype=np.float64
            ),
            X_table_camera=np.asarray(table_frame["X_table_camera"], dtype=np.float64),
            link_names=tuple(link_names),
            lnd_link_ids=lnd_link_ids,
            T_lndlink_urdf_link=link_offsets,
        )

    def lnd_forward_kinematics(self, q7: np.ndarray) -> dict[int, np.ndarray]:
        q7 = np.asarray(q7, dtype=np.float64)
        if q7.shape != (7,):
            raise ValueError(f"Expected q7 shape (7,), got {q7.shape}")
        transforms: dict[int, np.ndarray] = {0: np.eye(4, dtype=np.float64)}
        transform = np.eye(4, dtype=np.float64)
        for index, dh in enumerate(self.dh_params, start=1):
            theta0 = float(dh.get("theta", 0.0))
            d0 = float(dh.get("D", 0.0))
            offset = float(dh.get("offset", 0.0))
            if dh["type"] == "revolute":
                theta = theta0 + float(q7[index - 1]) + offset
                d = d0
            elif dh["type"] == "prismatic":
                theta = theta0
                d = d0 + float(q7[index - 1]) + offset
            else:
                raise ValueError(f"Unknown LND joint type: {dh['type']}")
            transform = transform @ _modified_dh(
                float(dh.get("alpha", 0.0)),
                float(dh.get("A", 0.0)),
                theta,
                d,
            )
            transforms[index] = transform.copy()
        jaw = float(np.clip(q7[6], -1.2, 1.6))
        transforms[7] = transforms[6] @ _rotz(0.5 * jaw)
        transforms[8] = transforms[6] @ _rotz(-0.5 * jaw)
        return transforms

    def visual_matrices_table(self, q7: np.ndarray) -> np.ndarray:
        lnd_transforms = self.lnd_forward_kinematics(q7)
        matrices = []
        for link_index, lnd_link_id in enumerate(self.lnd_link_ids):
            T_rectified_camera_visual = (
                self.T_rectified_camera_psm_base
                @ lnd_transforms[int(lnd_link_id)]
                @ self.T_lndlink_urdf_link[link_index]
            )
            matrices.append(self.X_table_camera @ T_rectified_camera_visual)
        return np.stack(matrices)

    def visual_poses_table(self, q7: np.ndarray) -> np.ndarray:
        matrices = self.visual_matrices_table(q7)
        poses = np.empty((len(matrices), 7), dtype=np.float32)
        poses[:, :3] = matrices[:, :3, 3]
        poses[:, 3:] = Rotation.from_matrix(matrices[:, :3, :3]).as_quat().astype(
            np.float32
        )
        return poses

    def gaussian_world_means(
        self,
        q7: np.ndarray,
        local_means: np.ndarray,
        gaussian_link_ids: np.ndarray,
    ) -> np.ndarray:
        matrices = self.visual_matrices_table(q7)
        result = np.empty_like(local_means, dtype=np.float64)
        for link_index, matrix in enumerate(matrices):
            mask = gaussian_link_ids == link_index
            result[mask] = (
                local_means[mask] @ matrix[:3, :3].T + matrix[:3, 3]
            )
        return result
