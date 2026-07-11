# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import warp
from scipy.spatial.transform import Rotation

from embodied_gaussians import (
    Body,
    EmbodiedGaussiansBuilder,
    EmbodiedGaussiansEnvironment,
    Ground,
    read_ground,
)

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parents[2]

GROUND_PATH = current_dir / "environment/ground_plane.json"
TISSUE_PATH = current_dir / "objects/tissue.json"
GROUND_BODY_PATH = current_dir / "objects/ground.json"

PSM_URDF_PATH = repo_root / "data/super/psm_robot/psm.urdf"
PSM_MIMIC_MAP_PATH = repo_root / "data/super/psm_robot/psm_mimic_map.json"
PSM_SURFACE_GAUSSIANS_PATH = (
    repo_root / "data/super/psm_robot/psm_surface_gaussians.npz"
)
PSM_LND_POSE_DRIVER_PATH = repo_root / "data/super/psm_robot/psm_lnd_pose_driver.npz"
TABLE_FRAME_PATH = repo_root / "data/super/table_frame.json"
SUPER_DATASET_PATH = repo_root / "data/super/grasp5_offline_demo"
ROBOTS_PATH = SUPER_DATASET_PATH / "robots.json"
PSM_BASE_LINK_NAME = "PSM1_psm_base_link"

# Warp joint_q slot order for PSM URDF.  Must match what Warp produces after
# parsing the articulation tree.  Fixed joints do not get joint_q slots.
PSM_WARP_JOINT_Q_ORDER = [
    "yaw",
    "pitch_2",
    "pitch_1",
    "pitch_4",
    "pitch_3",
    "pitch_5",
    "pitch",
    "insertion",
    "roll",
    "wrist_pitch",
    "wrist_yaw",
    "jaw_mimic_2",
    "jaw_mimic_1",
    "jaw",
]

TISSUE_GAUSSIAN_SCALE = 1.0
GROUND_GAUSSIAN_SCALE = 1.0
TISSUE_PBD_RADIUS_SCALE = 1.0

PSM_ARTICULATION_INDEX = 0


def load_body(path: Path) -> Body:
    with open(path, "r") as f:
        return Body.model_validate(json.load(f))


def urdf_actuated_joint_order(urdf_path: Path) -> list[str]:
    if urdf_path.resolve() != PSM_URDF_PATH.resolve():
        raise ValueError(f"SUPER PSM joint order is only defined for {PSM_URDF_PATH}")
    return list(PSM_WARP_JOINT_Q_ORDER)


def load_mimic_config(path: Path = PSM_MIMIC_MAP_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_table_from_camera(path: Path = TABLE_FRAME_PATH) -> np.ndarray:
    with open(path, "r") as f:
        table_frame = json.load(f)
    return np.asarray(table_frame["X_table_camera"], dtype=np.float32)


def expand_psm_q7_to_joint_dict(q7: list[float] | np.ndarray, mimic_cfg: dict) -> dict[str, float]:
    q7_arr = np.asarray(q7, dtype=np.float32)
    input_names = mimic_cfg["input_joint_names"]
    if len(q7_arr) != len(input_names):
        raise ValueError(f"Expected {len(input_names)} PSM input joints, got {len(q7_arr)}")
    q_by_joint: dict[str, float] = {}
    input_to_urdf = mimic_cfg["input_to_urdf_joint"]
    for input_name, value in zip(input_names, q7_arr):
        q_by_joint[input_to_urdf[input_name]] = float(value)
    for mimic_joint_name, spec in mimic_cfg["mimic"].items():
        source_name = spec["source"]
        q_by_joint[mimic_joint_name] = (
            q_by_joint[source_name] * float(spec.get("multiplier", 1.0))
            + float(spec.get("offset", 0.0))
        )
    return q_by_joint


def expand_psm_q7_to_urdf_order(
    q7: list[float] | np.ndarray,
    mimic_cfg: dict | None = None,
    joint_order: list[str] | None = None,
) -> np.ndarray:
    if mimic_cfg is None:
        mimic_cfg = load_mimic_config()
    if joint_order is None:
        joint_order = urdf_actuated_joint_order(PSM_URDF_PATH)
    q_by_joint = expand_psm_q7_to_joint_dict(q7, mimic_cfg)
    missing = [name for name in joint_order if name not in q_by_joint]
    if missing:
        raise KeyError(f"Missing PSM joint values for URDF joints: {missing}")
    return np.asarray([q_by_joint[name] for name in joint_order], dtype=np.float32)


def load_initial_psm_q_full() -> np.ndarray:
    with open(ROBOTS_PATH, "r") as f:
        robots = json.load(f)
    # super_best robots.json uses key "PSM1"
    q7 = robots["PSM1"]["states"][0]["q"]
    return expand_psm_q7_to_urdf_order(q7)


def add_psm_surface_gaussians(
    builder: EmbodiedGaussiansBuilder,
    robot_body_count: int,
    path: Path = PSM_SURFACE_GAUSSIANS_PATH,
) -> int:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing PSM surface Gaussian asset: {path}. Run "
            "`python scripts/build_psm_surface_gaussians.py` first."
        )
    with np.load(path, allow_pickle=False) as asset:
        required = {
            "means",
            "quats_wxyz",
            "scales",
            "opacities",
            "colors",
            "link_ids",
            "link_names",
        }
        missing = sorted(required.difference(asset.files))
        if missing:
            raise KeyError(f"PSM Gaussian asset is missing arrays: {missing}")
        means = asset["means"].astype(np.float32)
        quats = asset["quats_wxyz"].astype(np.float32)
        scales = asset["scales"].astype(np.float32)
        opacities = asset["opacities"].astype(np.float32)
        colors = asset["colors"].astype(np.float32)
        link_ids = asset["link_ids"].astype(np.int64)
        link_names = asset["link_names"].tolist()

    count = len(means)
    expected_shapes = {
        "quats_wxyz": (count, 4),
        "scales": (count, 3),
        "opacities": (count,),
        "colors": (count, 3),
        "link_ids": (count,),
    }
    arrays = {
        "quats_wxyz": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "link_ids": link_ids,
    }
    if means.shape != (count, 3):
        raise ValueError(f"Invalid PSM Gaussian means shape: {means.shape}")
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"Invalid PSM Gaussian {name} shape: {arrays[name].shape}, "
                f"expected {expected_shape}"
            )
    if not all(np.all(np.isfinite(array)) for array in (means, quats, scales, opacities, colors)):
        raise ValueError("PSM Gaussian asset contains non-finite values")
    if np.any(scales <= 0.0):
        raise ValueError("PSM Gaussian scales must be positive")
    if np.any(link_ids < 0) or np.any(link_ids >= len(link_names)):
        raise ValueError("PSM Gaussian asset contains invalid link ids")

    body_by_name = {
        name: body_id
        for body_id, name in enumerate(builder.body_name[:robot_body_count])
    }
    missing_links = [name for name in link_names if name not in body_by_name]
    if missing_links:
        raise KeyError(f"PSM Gaussian links are absent from the Warp model: {missing_links}")
    body_ids = np.asarray(
        [body_by_name[link_names[link_id]] for link_id in link_ids], dtype=np.int32
    )

    builder.gaussian_means.extend(means.tolist())
    builder.gaussian_quats.extend(quats.tolist())
    builder.gaussian_scales.extend(scales.tolist())
    builder.gaussian_opacities.extend(opacities.tolist())
    builder.gaussian_colors.extend(colors.tolist())
    builder.gaussian_body_ids.extend(body_ids.tolist())
    print(
        "[super_embodied] PSM visual surface gaussians: "
        f"{count} across {len(link_names)} links, "
        f"scale={scales.min() * 1e3:.2f}..{scales.max() * 1e3:.2f}mm"
    )
    return count


def load_psm_lnd_pose_driver(
    X_table_camera: np.ndarray,
    path: Path = PSM_LND_POSE_DRIVER_PATH,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing PSM LND pose driver: {path}. Run "
            "`python scripts/calibrate_psm_lnd_pose_driver.py` first."
        )
    with np.load(path, allow_pickle=False) as asset:
        timestamps = asset["timestamps"].astype(np.float64)
        link_names = asset["link_names"].tolist()
        poses_rect = asset["poses_rect_camera_xyz_xyzw"].astype(np.float64)
    if poses_rect.shape != (len(timestamps), len(link_names), 7):
        raise ValueError(f"Invalid PSM LND pose driver shape: {poses_rect.shape}")

    poses_table = np.empty_like(poses_rect, dtype=np.float32)
    for state_index in range(len(timestamps)):
        for link_index in range(len(link_names)):
            pose = poses_rect[state_index, link_index]
            T_rect_link = np.eye(4, dtype=np.float64)
            T_rect_link[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
            T_rect_link[:3, 3] = pose[:3]
            T_table_link = X_table_camera @ T_rect_link
            poses_table[state_index, link_index, :3] = T_table_link[:3, 3]
            poses_table[state_index, link_index, 3:] = Rotation.from_matrix(
                T_table_link[:3, :3]
            ).as_quat()
    return timestamps, link_names, poses_table


def apply_psm_lnd_pose(
    env: EmbodiedGaussiansEnvironment,
    state_index: int,
    update_gaussians: bool = True,
) -> None:
    poses = env.super_psm_lnd_poses_table  # type: ignore[attr-defined]
    body_ids_by_env = env.super_psm_lnd_body_ids  # type: ignore[attr-defined]
    state_index = max(0, min(int(state_index), len(poses) - 1))
    pose_tensor = torch.as_tensor(poses[state_index], dtype=torch.float32)
    for body_ids in body_ids_by_env:
        for state in (env.sim.state_0, env.sim.state_1):
            body_q = warp.to_torch(state.body_q)
            body_q[body_ids] = pose_tensor.to(body_q.device)
    if update_gaussians:
        env.sim.update_gaussian_transforms()


def build_environment(
    num_envs: int = 1,
    add_gaussians: bool = True,
    device: str = "cuda",
) -> EmbodiedGaussiansEnvironment:
    ground_data = read_ground(GROUND_PATH)
    ground = Ground(plane=ground_data)
    X_table_camera = load_table_from_camera()
    lnd_timestamps, lnd_link_names, lnd_poses_table = load_psm_lnd_pose_driver(
        X_table_camera
    )

    q_start = load_initial_psm_q_full()
    tissue_body = load_body(TISSUE_PATH)
    if tissue_body.gaussians is not None:
        for _i in range(len(tissue_body.gaussians.scales)):
            tissue_body.gaussians.scales[_i] = [
                s * TISSUE_GAUSSIAN_SCALE for s in tissue_body.gaussians.scales[_i]
            ]
    if tissue_body.particles is not None:
        tissue_body.particles.radii = [
            float(r) * TISSUE_PBD_RADIUS_SCALE for r in tissue_body.particles.radii
        ]
    ground_body = load_body(GROUND_BODY_PATH)
    if ground_body.gaussians is not None:
        for _i in range(len(ground_body.gaussians.scales)):
            ground_body.gaussians.scales[_i] = [
                s * GROUND_GAUSSIAN_SCALE for s in ground_body.gaussians.scales[_i]
            ]

    builder = EmbodiedGaussiansBuilder(up_vector=ground.normal())
    builder.add_renderable_articulation_from_urdf(
        urdf_path=PSM_URDF_PATH,
        initial_joints=q_start,
        X_WB=X_table_camera,
        stiffness=500,
        damping=100,
        ignore_inertial_definitions=True,
        ensure_nonstatic_links=True,
        collapse_fixed_joints=False,
        # PSM appearance is loaded from the preprocessed URDF visual meshes below.
        add_gaussians=False,
    )
    robot_body_count = builder.body_count

    psm_gaussian_count = 0
    if add_gaussians:
        psm_gaussian_count = add_psm_surface_gaussians(builder, robot_body_count)

    # ── PSM: no physics collisions ─────────────────────────────────────────
    # PSM is kinematically driven by robots.json q.  Disable all shape
    # collisions (vs ground, vs tissue, vs self) to prevent XPBD blow-up
    # from initial penetrations of the full robot arm through the ground plane.
    psm_shape_ids = []
    for shape_id, body_id in enumerate(builder.shape_body):
        if 0 <= body_id < robot_body_count:
            psm_shape_ids.append(shape_id)
            builder.shape_shape_collision[shape_id] = False
            builder.shape_ground_collision[shape_id] = False
    if any(builder.shape_shape_collision[i] for i in psm_shape_ids):
        raise RuntimeError("Failed to disable PSM shape collisions.")
    if any(builder.shape_ground_collision[i] for i in psm_shape_ids):
        raise RuntimeError("Failed to disable PSM ground collisions.")
    print(
        "[super_embodied] PSM collisions disabled: "
        f"shape_shape=0/{len(psm_shape_ids)}, shape_ground=0/{len(psm_shape_ids)}"
    )

    tissue_body_id = builder.add_rigid_body(
        tissue_body,
        mu=0.05,
        add_gaussians=add_gaussians,
    )

    if add_gaussians:
        builder.add_visual_body(ground_body)

    final_builder = EmbodiedGaussiansBuilder(up_vector=ground.normal())
    psm_base_body_ids: list[int] = []
    psm_lnd_body_ids: list[list[int]] = []
    tissue_body_ids: list[int] = []
    for _ in range(num_envs):
        body_start = final_builder.body_count
        final_builder.add_builder(builder)
        tissue_body_ids.append(body_start + tissue_body_id)
        body_names = final_builder.body_name[body_start : final_builder.body_count]
        body_id_by_name = {
            body_name: body_start + local_body_id
            for local_body_id, body_name in enumerate(body_names)
        }
        missing_lnd_links = [
            name for name in lnd_link_names if name not in body_id_by_name
        ]
        if missing_lnd_links:
            raise KeyError(f"LND-driven PSM links missing from model: {missing_lnd_links}")
        psm_lnd_body_ids.append(
            [body_id_by_name[name] for name in lnd_link_names]
        )
        for local_body_id, body_name in enumerate(body_names):
            if body_name == PSM_BASE_LINK_NAME:
                psm_base_body_ids.append(body_start + local_body_id)

    # Warp's add_builder() copies the child builder's default ground settings,
    # so the dataset plane must be applied after all child builders are added.
    final_builder.set_ground_plane(ground.normal(), ground.offset())

    env = EmbodiedGaussiansEnvironment(final_builder, device=device)

    # Zero gravity for PSM bodies (kinematic drive)
    gravity_factor = warp.to_torch(env.sim.model.gravity_factor).reshape(num_envs, -1)
    gravity_factor[:, :robot_body_count] = 0.0

    q_start_t = torch.from_numpy(q_start).float()
    env.set_robot_q(PSM_ARTICULATION_INDEX, q_start_t)
    env.set_robot_desired_q(PSM_ARTICULATION_INDEX, q_start_t)
    env.super_tissue_body_id = tissue_body_id  # type: ignore[attr-defined]
    env.super_tissue_body_ids = tissue_body_ids  # type: ignore[attr-defined]
    env.super_robot_body_count = robot_body_count  # type: ignore[attr-defined]
    env.super_psm_gaussian_count = psm_gaussian_count  # type: ignore[attr-defined]
    env.super_psm_base_body_ids = psm_base_body_ids  # type: ignore[attr-defined]
    env.super_psm_collisions_enabled = False  # type: ignore[attr-defined]
    env.super_psm_lnd_timestamps = lnd_timestamps  # type: ignore[attr-defined]
    env.super_psm_lnd_link_names = lnd_link_names  # type: ignore[attr-defined]
    env.super_psm_lnd_poses_table = lnd_poses_table  # type: ignore[attr-defined]
    env.super_psm_lnd_body_ids = psm_lnd_body_ids  # type: ignore[attr-defined]
    apply_psm_lnd_pose(env, 0)
    env.stash_state()
    return env
