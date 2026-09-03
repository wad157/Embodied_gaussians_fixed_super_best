#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from fit_psm_urdf_base_to_lnd import (
    expand_q7_to_full,
    parse_urdf_tree,
    umeyama,
    urdf_fk,
)
from build_super_psm_lnd_intermediates import (
    lnd_forward_kinematics,
    parse_lnd_json,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "data/super/psm_robot/psm_lnd_pose_driver.npz"
DEFAULT_REPORT = REPO / "data/super/psm_robot/psm_lnd_pose_driver_report.json"

URDF_TO_LND_LINK = {
    "PSM1_tool_main_link": 3,
    "PSM1_tool_wrist_link": 4,
    "PSM1_tool_wrist_shaft_link": 4,
    "PSM1_tool_wrist_sca_link": 5,
    "PSM1_tool_wrist_sca_shaft_link": 6,
    "PSM1_tool_wrist_sca_ee_link_1": 7,
    "PSM1_tool_wrist_sca_ee_link_2": 8,
}

MODEL_ALIGNMENT_CORRESPONDENCES = [
    ("PSM1_psm_base_link", 0, 3.0),
    ("PSM1_outer_pitch_link", 2, 3.0),
    ("PSM1_tool_main_link", 3, 3.0),
    ("PSM1_tool_wrist_link", 4, 2.0),
    ("PSM1_tool_wrist_sca_link", 5, 2.0),
    ("PSM1_tool_wrist_sca_shaft_link", 6, 2.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an LND-driven pose sequence for the PSM visual links."
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
        "--joints",
        type=Path,
        default=REPO / "data/super/grasp5_native/joints.json",
    )
    parser.add_argument(
        "--lnd-motion",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json",
    )
    parser.add_argument(
        "--lnd-model",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json",
    )
    parser.add_argument("--lnd", type=Path, default=REPO / "data/LND.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def matrix_to_pose_xyzw(matrix: np.ndarray) -> np.ndarray:
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.concatenate([matrix[:3, 3], quat])


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
    )


def main() -> None:
    args = parse_args()
    # Keep explicit per-dataset relative outputs usable.  The report records a
    # repository-relative path below, which requires an absolute normalized
    # path before calling ``relative_to(REPO)``.
    args.output = args.output.resolve()
    args.report = args.report.resolve()
    joints_data = read_json(args.joints)
    motion = read_json(args.lnd_motion)
    model = read_json(args.lnd_model)
    mimic_map = read_json(args.mimic_map)
    lnd = parse_lnd_json(args.lnd)
    _, urdf_joints, urdf_children = parse_urdf_tree(args.urdf)

    states = joints_data["states"]
    if len(states) != len(motion["states"]):
        raise ValueError(
            f"Joint/LND state count mismatch: {len(states)} != {len(motion['states'])}"
        )
    T_rect_base = np.asarray(
        model["T_rectified_camera_psm_base"], dtype=np.float64
    )
    canonical_urdf_world = urdf_fk(
        urdf_joints,
        urdf_children,
        expand_q7_to_full([0.0] * 7, mimic_map),
    )
    X_world_urdf_base = canonical_urdf_world["PSM1_psm_base_link"]
    X_urdf_base_world = np.linalg.inv(X_world_urdf_base)
    canonical_urdf = {
        name: X_urdf_base_world @ transform
        for name, transform in canonical_urdf_world.items()
    }
    canonical_lnd = lnd_forward_kinematics(lnd, np.zeros(7, dtype=np.float64))

    urdf_points = np.asarray(
        [
            canonical_urdf[urdf_link][:3, 3]
            for urdf_link, _lnd_link, _weight in MODEL_ALIGNMENT_CORRESPONDENCES
        ]
    )
    lnd_points = np.asarray(
        [
            canonical_lnd[lnd_link][:3, 3]
            for _urdf_link, lnd_link, _weight in MODEL_ALIGNMENT_CORRESPONDENCES
        ]
    )
    weights = np.asarray(
        [weight for _urdf_link, _lnd_link, weight in MODEL_ALIGNMENT_CORRESPONDENCES]
    )
    rotation, translation = umeyama(urdf_points, lnd_points, weights)
    T_lnd_base_urdf_base = np.eye(4, dtype=np.float64)
    T_lnd_base_urdf_base[:3, :3] = rotation
    T_lnd_base_urdf_base[:3, 3] = translation

    model_alignment_residuals = []
    for urdf_link, lnd_link, weight in MODEL_ALIGNMENT_CORRESPONDENCES:
        predicted = (T_lnd_base_urdf_base @ canonical_urdf[urdf_link])[:3, 3]
        target = canonical_lnd[lnd_link][:3, 3]
        model_alignment_residuals.append(
            {
                "urdf_link": urdf_link,
                "lnd_link": lnd_link,
                "weight": weight,
                "residual_m": float(np.linalg.norm(predicted - target)),
            }
        )

    link_names = list(URDF_TO_LND_LINK)
    T_lndlink_urdf_link: dict[str, np.ndarray] = {}
    for urdf_link, lnd_link in URDF_TO_LND_LINK.items():
        T_lndlink_urdf_link[urdf_link] = (
            np.linalg.inv(canonical_lnd[lnd_link])
            @ T_lnd_base_urdf_base
            @ canonical_urdf[urdf_link]
        )

    poses = np.empty((len(states), len(link_names), 7), dtype=np.float32)
    for state_index, lnd_state in enumerate(motion["states"]):
        lnd_transforms = lnd_state["link_transforms_psm_base"]
        for link_index, urdf_link in enumerate(link_names):
            lnd_link = URDF_TO_LND_LINK[urdf_link]
            T_rect_link = (
                T_rect_base
                @ np.asarray(lnd_transforms[str(lnd_link)], dtype=np.float64)
                @ T_lndlink_urdf_link[urdf_link]
            )
            poses[state_index, link_index] = matrix_to_pose_xyzw(T_rect_link)

    sample_indices = np.unique(np.linspace(0, len(states) - 1, 400).astype(int))
    old_position_errors: dict[str, list[float]] = {name: [] for name in link_names}
    old_orientation_errors: dict[str, list[float]] = {name: [] for name in link_names}
    for state_index in sample_indices:
        old_urdf = urdf_fk(
            urdf_joints,
            urdf_children,
            expand_q7_to_full(states[state_index]["q"], mimic_map),
        )
        for link_index, urdf_link in enumerate(link_names):
            pose = poses[state_index, link_index]
            desired = np.eye(4, dtype=np.float64)
            desired[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
            desired[:3, 3] = pose[:3]
            old_position_errors[urdf_link].append(
                float(np.linalg.norm(old_urdf[urdf_link][:3, 3] - desired[:3, 3]))
            )
            old_orientation_errors[urdf_link].append(
                angular_distance_deg(old_urdf[urdf_link], desired)
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        timestamps=np.asarray(joints_data["states_timestamps"], dtype=np.float64),
        link_names=np.asarray(link_names),
        poses_rect_camera_xyz_xyzw=poses,
    )
    report = {
        "method": (
            "LND modified-DH link poses drive PSM visual bodies; dVRK visual-link "
            "offsets are derived only from both source models at canonical q=0"
        ),
        "uses_dataset_reference_frame": False,
        "uses_image_based_correction": False,
        "num_states": len(states),
        "output": str(args.output.relative_to(REPO)),
        "link_mapping": URDF_TO_LND_LINK,
        "T_lnd_base_urdf_base": T_lnd_base_urdf_base.tolist(),
        "canonical_model_alignment_residuals": model_alignment_residuals,
        "canonical_model_alignment_max_residual_m": max(
            item["residual_m"] for item in model_alignment_residuals
        ),
        "T_lndlink_urdf_link": {
            name: matrix.tolist() for name, matrix in T_lndlink_urdf_link.items()
        },
        "old_direct_urdf_errors": {
            name: {
                "position_mm_p50_p95_max": np.percentile(
                    np.asarray(old_position_errors[name]) * 1000.0, [50, 95, 100]
                ).tolist(),
                "orientation_deg_p50_p95_max": np.percentile(
                    old_orientation_errors[name], [50, 95, 100]
                ).tolist(),
            }
            for name in link_names
        },
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": report["output"],
                "num_states": report["num_states"],
                "links": link_names,
                "uses_dataset_reference_frame": False,
                "uses_image_based_correction": False,
                "canonical_model_alignment_max_residual_m": report[
                    "canonical_model_alignment_max_residual_m"
                ],
                "old_direct_urdf_errors": report["old_direct_urdf_errors"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
