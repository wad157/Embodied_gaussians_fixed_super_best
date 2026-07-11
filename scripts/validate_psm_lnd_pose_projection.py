#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from fit_psm_urdf_base_to_lnd import (
    expand_q7_to_full,
    parse_urdf_tree,
    urdf_fk,
)


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct-URDF and LND-driven PSM projections on left video frames."
    )
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 500, 1000, 1400])
    parser.add_argument(
        "--rgb-dir", type=Path, default=REPO / "data/super/grasp5_native/rgb"
    )
    parser.add_argument(
        "--camera-metadata",
        type=Path,
        default=REPO / "data/super/grasp5_offline_demo/videos/stereo_left.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO / "data/super/grasp5_native/calib_rectified.json",
    )
    parser.add_argument(
        "--joints",
        type=Path,
        default=REPO / "data/super/grasp5_native/joints.json",
    )
    parser.add_argument(
        "--urdf", type=Path, default=REPO / "data/super/psm_robot/psm.urdf"
    )
    parser.add_argument(
        "--mimic-map",
        type=Path,
        default=REPO / "data/super/psm_robot/psm_mimic_map.json",
    )
    parser.add_argument(
        "--gaussians",
        type=Path,
        default=REPO / "data/super/psm_robot/psm_surface_gaussians.npz",
    )
    parser.add_argument(
        "--pose-driver",
        type=Path,
        default=REPO / "data/super/psm_robot/psm_lnd_pose_driver.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data/super/psm_robot/lnd_pose_validation",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def transform_link_points(
    means: np.ndarray,
    link_ids: np.ndarray,
    asset_link_names: list[str],
    transforms: dict[str, np.ndarray],
) -> np.ndarray:
    result = np.empty_like(means, dtype=np.float64)
    for link_id, link_name in enumerate(asset_link_names):
        if link_name not in transforms:
            raise KeyError(f"Missing transform for PSM Gaussian link: {link_name}")
        mask = link_ids == link_id
        local = means[mask]
        transform = transforms[link_name]
        result[mask] = (
            transform[:3, :3] @ local.T + transform[:3, 3:4]
        ).T
    return result


def project(points: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
    positive = points[:, 2] > 1e-6
    normalized = points[positive] / points[positive, 2:3]
    uv = (K @ normalized.T).T[:, :2]
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv[inside]


def draw_projection(
    image: np.ndarray,
    uv: np.ndarray,
    color: tuple[int, int, int],
    title: str,
) -> np.ndarray:
    overlay = image.copy()
    for u, v in np.rint(uv).astype(np.int32):
        cv2.circle(overlay, (u, v), 2, color, -1, cv2.LINE_AA)
    result = cv2.addWeighted(overlay, 0.75, image, 0.25, 0.0)
    cv2.rectangle(result, (0, 0), (760, 56), (0, 0, 0), -1)
    cv2.putText(
        result,
        title,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    args = parse_args()
    camera_metadata = read_json(args.camera_metadata)
    calibration = read_json(args.calibration)
    joints_data = read_json(args.joints)
    mimic_map = read_json(args.mimic_map)
    _, urdf_joints, urdf_children = parse_urdf_tree(args.urdf)
    K = np.asarray(calibration["K_left_rect"], dtype=np.float64)
    video_timestamps = np.asarray(camera_metadata["timestamps"], dtype=np.float64)
    joint_timestamps = np.asarray(joints_data["states_timestamps"], dtype=np.float64)

    with np.load(args.gaussians, allow_pickle=False) as asset:
        means = asset["means"].astype(np.float64)
        link_ids = asset["link_ids"].astype(np.int64)
        asset_link_names = asset["link_names"].tolist()
    with np.load(args.pose_driver, allow_pickle=False) as driver:
        driver_link_names = driver["link_names"].tolist()
        poses = driver["poses_rect_camera_xyz_xyzw"].astype(np.float64)
    driver_link_index = {name: i for i, name in enumerate(driver_link_names)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for video_frame in args.frames:
        if not 0 <= video_frame < len(video_timestamps):
            raise IndexError(
                f"Video frame {video_frame} outside [0, {len(video_timestamps) - 1}]"
            )
        image_path = args.rgb_dir / f"{video_frame:06d}-left.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        timestamp = video_timestamps[video_frame]
        state_index = int(np.argmin(np.abs(joint_timestamps - timestamp)))
        timestamp_error_ms = float(
            abs(joint_timestamps[state_index] - timestamp) * 1000.0
        )

        old_transforms = urdf_fk(
            urdf_joints,
            urdf_children,
            expand_q7_to_full(joints_data["states"][state_index]["q"], mimic_map),
        )
        new_transforms = {
            name: pose_to_matrix(poses[state_index, driver_link_index[name]])
            for name in asset_link_names
        }
        old_points = transform_link_points(
            means, link_ids, asset_link_names, old_transforms
        )
        new_points = transform_link_points(
            means, link_ids, asset_link_names, new_transforms
        )
        old_uv = project(old_points, K, width, height)
        new_uv = project(new_points, K, width, height)
        old_image = draw_projection(
            image,
            old_uv,
            (0, 80, 255),
            f"OLD direct URDF | video={video_frame} joint={state_index}",
        )
        new_image = draw_projection(
            image,
            new_uv,
            (255, 255, 0),
            f"STRICT LND+dVRK q=0 | video={video_frame} joint={state_index}",
        )
        comparison = np.concatenate([old_image, new_image], axis=1)
        out_path = args.out_dir / f"frame{video_frame:06d}_left_before_after.png"
        cv2.imwrite(str(out_path), comparison)
        summary.append(
            {
                "video_frame": video_frame,
                "video_timestamp": float(timestamp),
                "joint_state_index": state_index,
                "joint_timestamp": float(joint_timestamps[state_index]),
                "timestamp_error_ms": timestamp_error_ms,
                "old_visible_gaussians": int(len(old_uv)),
                "new_visible_gaussians": int(len(new_uv)),
                "output": str(out_path.relative_to(REPO)),
            }
        )
        print(out_path)

    summary_path = args.out_dir / "projection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
