#!/usr/bin/env python3
"""CUDA smoke gate for one real grasp5 flow-depth q/qd state update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    PSM_POSE_DRIVER_PATHS,
    build_environment,
)
from embodied_gaussians import DatasetManager  # noqa: E402
from embodied_gaussians.physics_simulator.flow_depth_particle_observer import (  # noqa: E402
    FlowDepthParticleRangeBindings,
    FlowDepthStateUpdateSettings,
    FlowDepthTrackObservation,
    compute_flow_depth_particle_state_update,
)
from example_embodied_super_offline import (  # noqa: E402
    SuperPlaybackControls,
    build_visual_tissue_residual_mapper,
    grip_control_exclusion_mask,
)


DEFAULT_BINDINGS = (
    REPO_ROOT
    / "outputs/grasp5_cotracker3_range_bindings_20260828_v1/bindings.npz"
)
DEFAULT_OBSERVATIONS = (
    REPO_ROOT
    / "outputs/grasp5_cotracker3_flow_depth_observations_20260828_v1/"
    "observations.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/grasp5_flow_depth_particle_state_update_20260828_v1/report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--position-gain", type=float, default=0.35)
    parser.add_argument("--velocity-gain", type=float, default=0.08)
    parser.add_argument("--maximum-position-correction-mm", type=float, default=0.5)
    parser.add_argument("--maximum-velocity-correction-m-s", type=float, default=0.03)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_bindings(path: Path) -> FlowDepthParticleRangeBindings:
    with np.load(path, allow_pickle=False) as loaded:
        result = FlowDepthParticleRangeBindings(
            track_valid=loaded["track_valid"].astype(bool),
            particle_ids=loaded["particle_ids"].astype(np.int32),
            particle_weights=loaded["particle_weights"].astype(np.float32),
            support_counts=loaded["support_counts"].astype(np.int32),
            movable_support_counts=loaded["movable_support_counts"].astype(np.int32),
            binding_radius_m=loaded["binding_radius_m"].astype(np.float32),
            initial_depth_m=loaded["initial_depth_m"].astype(np.float32),
            depth_sampling_radius_px=loaded["depth_sampling_radius_px"].astype(
                np.int16
            ),
            initial_points_table=loaded["initial_points_table"].astype(np.float32),
            nearest_surface_distance_m=loaded[
                "nearest_surface_distance_m"
            ].astype(np.float32),
        )
    result.validate()
    return result


def load_observation(
    path: Path, pair_index: int
) -> tuple[int, int, FlowDepthTrackObservation]:
    with np.load(path, allow_pickle=False) as loaded:
        pair_count = len(loaded["current_source_frames"])
        if not 0 <= pair_index < pair_count:
            raise IndexError(pair_index)
        current_frame = int(loaded["current_source_frames"][pair_index])
        next_frame = int(loaded["next_source_frames"][pair_index])
        valid = loaded["track_valid"][pair_index].astype(bool)
        result = FlowDepthTrackObservation(
            track_valid=valid,
            confidence=loaded["confidence"][pair_index].astype(np.float32),
            current_depth_m=loaded["current_depth_m"][pair_index].astype(
                np.float32
            ),
            next_depth_m=loaded["next_depth_m"][pair_index].astype(np.float32),
            current_depth_sampling_radius_px=np.where(valid, 0, -1).astype(
                np.int16
            ),
            next_depth_sampling_radius_px=np.where(valid, 0, -1).astype(
                np.int16
            ),
            current_points_table=loaded["current_points_table"][pair_index].astype(
                np.float32
            ),
            next_points_table=loaded["next_points_table"][pair_index].astype(
                np.float32
            ),
            observed_flow_table=loaded["observed_flow_table"][pair_index].astype(
                np.float32
            ),
        )
    result.validate()
    return current_frame, next_frame, result


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output}")
    for path in (args.bindings, args.observations):
        if not path.is_file():
            raise FileNotFoundError(path)
    bindings = load_bindings(args.bindings)
    current_frame, next_frame, observation = load_observation(
        args.observations, args.pair_index
    )
    if current_frame != 0:
        raise ValueError("The first smoke gate currently requires a frame-0 state")

    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("This smoke gate requires CUDA")
    environment = build_environment(
        psm_pose_driver_path=PSM_POSE_DRIVER_PATHS[
            "raw_paper_lnd_sam2_dense_contact_unbounded_xyz"
        ],
        psm_visual_tip_only=False,
        tissue_mode="paper_pbd",
    )
    dataset = DatasetManager(REPO_ROOT / "data/super/grasp5_offline_demo")
    dataset.keep_only_cameras(["stereo_left", "stereo_right"])
    dataset.update_frames(0.0)
    environment.frames = dataset.frames
    mapper = build_visual_tissue_residual_mapper(
        environment,
        iterations=1,
        learning_rate_m=1.0e-5,
        maximum_step_m=1.0e-4,
        previous_residual_carry=0.0,
        temporal_weight=0.10,
        magnitude_weight=0.01,
    )
    controls = SuperPlaybackControls(
        environment,
        dataset,
        fps=30,
        visual_feedback_mode="off",
        visual_residual_mapper=mapper,
        enable_psm_tissue_contact=True,
    )
    controls.reset()
    if len(bindings.track_valid) != len(observation.track_valid):
        raise ValueError("Binding and observation track counts disagree")
    current_positions = (
        wp.to_torch(environment.sim.state_0.particle_q)
        .detach()
        .cpu()
        .numpy()
        .copy()
    )
    while controls.current_frame_index < next_frame:
        if not controls.advance_one_frame():
            raise RuntimeError("Could not advance to the observation frame")
        environment.step(compute_visual_forces=False)
        controls.apply_current_psm_pose()
    predicted_positions_torch = wp.to_torch(
        environment.sim.state_0.particle_q
    ).detach().clone()
    predicted_velocities_torch = wp.to_torch(
        environment.sim.state_0.particle_qd
    ).detach().clone()
    predicted_positions = predicted_positions_torch.cpu().numpy().copy()
    predicted_velocities = predicted_velocities_torch.cpu().numpy().copy()
    inverse_masses = (
        wp.to_torch(environment.sim.model.particle_inv_mass)
        .detach()
        .cpu()
        .numpy()
        .copy()
    )
    exclusion = grip_control_exclusion_mask(mapper, environment)
    observation_dt_s = float(
        controls.playback_timestamps[next_frame]
        - controls.playback_timestamps[current_frame]
    )
    update = compute_flow_depth_particle_state_update(
        bindings=bindings,
        observation=observation,
        current_positions=current_positions,
        predicted_positions=predicted_positions,
        predicted_velocities=predicted_velocities,
        particle_inverse_masses=inverse_masses,
        observation_dt_s=observation_dt_s,
        dynamic_exclusion_mask=exclusion.detach().cpu().numpy(),
        settings=FlowDepthStateUpdateSettings(
            position_gain=args.position_gain,
            velocity_gain=args.velocity_gain,
            maximum_position_correction_m=(
                args.maximum_position_correction_mm / 1000.0
            ),
            maximum_velocity_correction_m_s=(
                args.maximum_velocity_correction_m_s
            ),
        ),
    )
    gaussian_means_before = environment.sim.gaussian_state.means.detach().clone()
    accepted = environment.sim.apply_flow_depth_particle_state_update(
        update,
        mapper=mapper,
    )
    particle_q_after = wp.to_torch(environment.sim.state_0.particle_q).detach()
    particle_qd_after = wp.to_torch(environment.sim.state_0.particle_qd).detach()
    gaussian_means_after = environment.sim.gaussian_state.means.detach()
    metrics = dict(environment.sim.last_flow_depth_particle_update_metrics)
    correction_norm = np.linalg.norm(update.position_correction, axis=1)
    velocity_norm = np.linalg.norm(update.velocity_correction, axis=1)
    observed_norm = np.linalg.norm(
        observation.observed_flow_table[update.track_valid], axis=1
    )
    innovation_norm = np.linalg.norm(
        update.innovation[update.track_valid], axis=1
    )
    fixed = inverse_masses <= 0.0
    gates = {
        "real_flow_depth_tracks_are_usable": bool(update.track_valid.any()),
        "candidate_is_accepted_by_physical_gates": bool(accepted),
        "particle_positions_are_written": bool(
            torch.any(
                torch.linalg.vector_norm(
                    particle_q_after - predicted_positions_torch, dim=1
                )
                > 0.0
            ).item()
        ),
        "particle_velocities_are_written": bool(
            torch.any(
                torch.linalg.vector_norm(
                    particle_qd_after - predicted_velocities_torch, dim=1
                )
                > 0.0
            ).item()
        ),
        "fixed_particles_are_unchanged": bool(
            np.all(update.position_correction[fixed] == 0.0)
            and np.all(update.velocity_correction[fixed] == 0.0)
        ),
        "position_step_cap_is_respected": bool(
            correction_norm.max(initial=0.0)
            <= args.maximum_position_correction_mm / 1000.0 + 1.0e-8
        ),
        "velocity_step_cap_is_respected": bool(
            velocity_norm.max(initial=0.0)
            <= args.maximum_velocity_correction_m_s + 1.0e-8
        ),
        "mode_2_updates_gaussians": bool(
            torch.any(
                torch.linalg.vector_norm(
                    gaussian_means_after - gaussian_means_before, dim=1
                )
                > 0.0
            ).item()
        ),
        "no_new_inverted_tetrahedra": bool(
            metrics["candidate_physical_quality"]["inverted_tetrahedra"]
            <= metrics["baseline_physical_quality"]["inverted_tetrahedra"]
        ),
        "no_new_low_volume_tetrahedra": bool(
            metrics["candidate_physical_quality"][
                "tetrahedra_below_volume_floor"
            ]
            <= metrics["baseline_physical_quality"][
                "tetrahedra_below_volume_floor"
            ]
        ),
    }
    report = {
        "schema": "super_flow_depth_particle_state_update_smoke_v1",
        "passed": bool(all(gates.values())),
        "frames": [current_frame, next_frame],
        "settings": {
            "position_gain": args.position_gain,
            "velocity_gain": args.velocity_gain,
            "maximum_position_correction_mm": (
                args.maximum_position_correction_mm
            ),
            "maximum_velocity_correction_m_s": (
                args.maximum_velocity_correction_m_s
            ),
            "observation_dt_s": observation_dt_s,
        },
        "counts": {
            "valid_tracks": int(update.track_valid.sum()),
            "position_updated_particles": int(np.count_nonzero(correction_norm > 0.0)),
            "velocity_updated_particles": int(np.count_nonzero(velocity_norm > 0.0)),
        },
        "statistics": {
            "observed_flow_mm_p50_p95_max": np.percentile(
                observed_norm * 1000.0, [50, 95, 100]
            ).tolist(),
            "pbd_unexplained_innovation_mm_p50_p95_max": np.percentile(
                innovation_norm * 1000.0, [50, 95, 100]
            ).tolist(),
            "position_correction_mm_p50_p95_max_nonzero": (
                np.percentile(
                    correction_norm[correction_norm > 0.0] * 1000.0,
                    [50, 95, 100],
                ).tolist()
                if np.any(correction_norm > 0.0)
                else None
            ),
            "velocity_correction_m_s_p50_p95_max_nonzero": (
                np.percentile(
                    velocity_norm[velocity_norm > 0.0], [50, 95, 100]
                ).tolist()
                if np.any(velocity_norm > 0.0)
                else None
            ),
        },
        "apply_metrics": metrics,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Flow-depth particle state-update smoke gate failed")


if __name__ == "__main__":
    main()
