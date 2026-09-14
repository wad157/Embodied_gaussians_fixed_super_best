#!/usr/bin/env python3
"""Run the paper-reconstructed deformable Embodied Gaussians method on SUPER."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_FILE = (
    REPO_ROOT.parent
    / "embodied_gaussians_fixed_super_best_sim/baselines/embodied_gaussians_sim/run_soft.py"
)
sys.path.insert(0, str(REPO_ROOT))

from baselines.embodied_gaussians_super.common import (  # noqa: E402
    SuperStereoInputs,
    load_calibration,
    nearest_timestamp_index,
)
from baselines.embodied_gaussians_super.protocol import (  # noqa: E402
    ADAPTER_VERSION,
    DATASETS,
    PAPER_PARAMETERS,
    RENDER_SCALE,
    UPSTREAM_COMMIT,
    dataset_spec,
    observation_allowed,
    rendering_frames,
    resolve_native,
    resolve_path,
    sha256_file,
)


def load_shared_runner():
    if not SHARED_FILE.is_file():
        raise FileNotFoundError(f"Missing audited SIM EG implementation: {SHARED_FILE}")
    sys.path.insert(0, str(SHARED_FILE.parent))
    spec = importlib.util.spec_from_file_location("eg_sim_shared_soft_runner", SHARED_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SHARED_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-limit", type=int, default=-1)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


class RawPSMCollisionParticles:
    """Paper sphere collisions driven only by the recorded robot kinematics."""

    def __init__(self, repo_root: Path, dataset_key: str, stereo: SuperStereoInputs, device):
        spec = dataset_spec(dataset_key)
        driver_path = resolve_path(repo_root, spec["pose_driver"])
        surface_path = repo_root / "data/super/psm_robot/psm_surface_gaussians.npz"
        with np.load(driver_path, allow_pickle=False) as archive:
            self.timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
            driver_links = [str(value) for value in archive["link_names"].tolist()]
            self.poses_camera = np.asarray(
                archive["poses_rect_camera_xyz_xyzw"], dtype=np.float32
            )
        with np.load(surface_path, allow_pickle=False) as archive:
            surface_links = [str(value) for value in archive["link_names"].tolist()]
            surface_link_ids = np.asarray(archive["link_ids"], dtype=np.int64)
            self.local = torch.as_tensor(
                np.asarray(archive["means"], dtype=np.float32), device=device
            )
        link_lookup = {name: index for index, name in enumerate(driver_links)}
        self.driver_link_ids = torch.as_tensor(
            [link_lookup[surface_links[index]] for index in surface_link_ids],
            dtype=torch.long,
            device=device,
        )
        calibration = load_calibration(repo_root, dataset_key)
        world_from_camera = np.asarray(
            calibration["stereo_left"]["X_WC_cv"], dtype=np.float32
        )
        self.rotation_world_camera = torch.as_tensor(
            world_from_camera[:3, :3], device=device
        )
        self.translation_world_camera = torch.as_tensor(
            world_from_camera[:3, 3], device=device
        )
        self.video_timestamps = stereo.left_timestamps
        self.device = device
        self.driver_path = driver_path
        self.surface_path = surface_path

    def positions_at(self, frame: int) -> torch.Tensor:
        state = nearest_timestamp_index(
            self.timestamps, float(self.video_timestamps[int(frame)])
        )
        pose = self.poses_camera[state]
        rotations = Rotation.from_quat(pose[:, 3:]).as_matrix().astype(np.float32)
        rotations = torch.as_tensor(rotations, device=self.device)[self.driver_link_ids]
        translations = torch.as_tensor(
            pose[:, :3], dtype=torch.float32, device=self.device
        )[self.driver_link_ids]
        camera_points = torch.einsum("nij,nj->ni", rotations, self.local) + translations
        return (
            torch.einsum("ij,nj->ni", self.rotation_world_camera, camera_points)
            + self.translation_world_camera
        )


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("Seed must be non-negative")
    native = resolve_native(REPO_ROOT, args.dataset_key, args.dataset)
    output = args.output_dir.expanduser().resolve()
    render_dir = args.render_dir.expanduser().resolve()
    if output.exists() or render_dir.exists():
        raise FileExistsError("Refusing to overwrite rollout or render output")
    output.mkdir(parents=True)
    render_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal EG visual-force rollout requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    wp.init()
    shared = load_shared_runner()
    if shared.PAPER_PARAMETERS != PAPER_PARAMETERS:
        raise ValueError("SUPER and audited SIM adapters disagree on paper parameters")
    body_path = args.body.expanduser().resolve()
    body = shared.TissueGaussianBody(body_path, device)
    radius = float(PAPER_PARAMETERS["particle_radius_m"])
    if not np.allclose(body.particle_radii_np, radius, rtol=0.0, atol=1.0e-7):
        raise ValueError("Body does not use the public-code 6 mm particle radius")
    clusters = shared.build_particle_neighbour_clusters(
        body.rest_positions_np,
        particle_mass=float(PAPER_PARAMETERS["particle_mass_kg"]),
    )
    matcher = shared.OrientedShapeMatcher(clusters, device)
    physics = shared.PaperSoftPhysics(
        body,
        matcher,
        shape_stiffness=float(PAPER_PARAMETERS["shape_constraint_projection"]),
        particle_radius=radius,
        device=device,
    )
    stereo = SuperStereoInputs(
        REPO_ROOT, args.dataset_key, int(PAPER_PARAMETERS["online_width"])
    )
    robot = RawPSMCollisionParticles(REPO_ROOT, args.dataset_key, stereo, device)
    formal_frames = int(dataset_spec(args.dataset_key)["frames"])
    frame_count = formal_frames if args.frame_limit < 0 else min(args.frame_limit, formal_frames)
    if args.frame_limit == 0:
        raise ValueError("frame-limit must be -1 or positive")
    positions = np.empty((frame_count, len(body.positions), 3), dtype=np.float32)
    rotations = np.empty((frame_count, len(body.positions), 3, 3), dtype=np.float32)
    selected_renders = set(rendering_frames(args.dataset_key))
    diagnostics = []
    start = time.perf_counter()
    for frame in range(frame_count):
        frame_start = time.perf_counter()
        physics.step(frame, None, robot.positions_at(frame))
        positions[frame] = body.positions.detach().cpu().numpy()
        rotations[frame] = body.rotations.detach().cpu().numpy()
        allowed = observation_allowed(frame, args.dataset_key)
        record: dict[str, object] = {"frame": frame, "observation_allowed": allowed}
        if frame in selected_renders and not args.no_render:
            intrinsic, view_matrix, size = stereo.render_camera_tensors(
                "stereo_left", RENDER_SCALE, device
            )
            rendered, _alpha = shared.render_tissue(
                body, view_matrix, intrinsic, size
            )
            np.save(
                render_dir / f"{frame:06d}.npy",
                rendered[0].detach().clamp(0.0, 1.0).cpu().numpy().astype(np.float32),
            )
        if allowed:
            record.update(shared.compute_visual_force(body, stereo, frame, rng, device))
        else:
            body.external_force.zero_()
        record["elapsed_s"] = time.perf_counter() - frame_start
        diagnostics.append(record)
        print(
            f"[EG-Soft SUPER] {args.dataset_key} frame {frame + 1:04d}/{frame_count:04d} "
            f"observe={allowed} elapsed={record['elapsed_s']:.3f}s",
            flush=True,
        )
    raw_path = output / "raw_particle_rollout.npz"
    np.savez_compressed(
        raw_path,
        frame_indices=np.arange(frame_count, dtype=np.int32),
        timestamps=stereo.left_timestamps[:frame_count],
        particle_positions_world=positions,
        particle_rotations_world=rotations,
    )
    (output / "frame_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema": "embodied_gaussians_super_soft_rollout_v1",
        "formal": args.frame_limit < 0 and not args.no_render,
        "method": "EG-Soft paper reconstruction",
        "adapter_version": ADAPTER_VERSION,
        "dataset_key": args.dataset_key,
        "seed": args.seed,
        "upstream_commit": UPSTREAM_COMMIT,
        "body": str(body_path),
        "body_sha256": sha256_file(body_path),
        "particles": len(body.positions),
        "gaussians": len(body.parents),
        "parameters": PAPER_PARAMETERS,
        "dataset_specific_physical_parameters": {},
        "shape_neighbours": "parameter-free Delaunay rest-state adjacency",
        "shape_parameter_disclosure": "paper omits deformable k_S; Eq. (5) is fully projected (k_S=1)",
        "actuation": "raw LND/FK kinematic link poses plus paper sphere collision only",
        "pose_driver": str(robot.driver_path),
        "pose_driver_sha256": sha256_file(robot.driver_path),
        "robot_surface": str(robot.surface_path),
        "robot_surface_sha256": sha256_file(robot.surface_path),
        "known_future_control_used": True,
        "online_observations": "timestamp-aligned stereo RGB and tissue masks at legal training frames",
        "future_rgb_depth_mask_or_track_observations_used": False,
        "ground_truth_opened": False,
        "shared_equation_implementation": str(SHARED_FILE),
        "shared_equation_implementation_sha256": sha256_file(SHARED_FILE),
        "frame_count": frame_count,
        "total_elapsed_s": time.perf_counter() - start,
        "raw_rollout_sha256": sha256_file(raw_path),
    }
    (output / "rollout_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
