#!/usr/bin/env python3
"""One-frame CUDA diagnostic for covariance-aware RGB and q/qd correction."""

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
from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    MultiCameraPackedTissueVisualForceWeights,
)
from embodied_gaussians.physics_simulator.visual_tissue_residual_mapping import (  # noqa: E402
    TetrahedralGaussianVisualResidualMapper,
)
from example_embodied_super_offline import (  # noqa: E402
    RIGHT_VISUAL_FORCE_MASK_DIR,
    VISUAL_FORCE_INSTRUMENT_MASKS,
    VISUAL_FORCE_MASK_DIR,
    SuperPlaybackControls,
    build_visual_tissue_residual_mapper,
    grip_control_exclusion_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=420)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("This diagnostic requires CUDA")

    environment = build_environment(
        psm_pose_driver_path=PSM_POSE_DRIVER_PATHS[
            "raw_paper_lnd_sam2_dense_contact_unbounded_xyz"
        ],
        psm_visual_tip_only=False,
        tissue_mode="paper_pbd",
    )
    dataset = DatasetManager(REPO_ROOT / "data/super/grasp5_offline_demo")
    dataset.keep_only_cameras(["stereo_left", "stereo_right"])
    dataset.set_visual_force_weight_provider(
        MultiCameraPackedTissueVisualForceWeights(
            {
                "stereo_left": VISUAL_FORCE_MASK_DIR,
                "stereo_right": RIGHT_VISUAL_FORCE_MASK_DIR,
            },
            erosion_radius_px=7,
            highlight_weight=0.10,
            instrument_mask_asset=VISUAL_FORCE_INSTRUMENT_MASKS,
            tool_near_radius_px=120.0,
            tool_far_radius_px=360.0,
            tool_falloff_power=2.0,
            tissue_edge_zero_px=24.0,
            tissue_edge_full_px=64.0,
            tool_occlusion_radius_px=6.0,
            image_border_zero_px=48.0,
            image_border_full_px=96.0,
            posterior_full_reach_px=140.0,
            posterior_zero_reach_px=280.0,
        )
    )
    dataset.update_frames(0.0)
    environment.frames = dataset.frames
    exact_mapper = build_visual_tissue_residual_mapper(
        environment,
        iterations=args.iterations,
        learning_rate_m=2.5e-5,
        maximum_step_m=0.00025,
        previous_residual_carry=0.0,
        temporal_weight=0.10,
        magnitude_weight=0.01,
    )
    model = environment.sim.gaussian_model
    legacy_mapper = TetrahedralGaussianVisualResidualMapper.from_visual_face_centroid_bindings(
        rest_positions=exact_mapper.rest_positions,
        tet_indices=exact_mapper.tet_indices,
        fixed_mask=exact_mapper.fixed_mask,
        soft_gaussian_ids=model.soft_gaussian_ids,
        visual_vertex_particle_indices=(
            model.soft_gaussian_visual_vertex_particle_indices
        ),
        visual_vertex_weights=model.soft_gaussian_visual_vertex_weights,
        settings=exact_mapper.settings,
    )
    controls = SuperPlaybackControls(
        environment,
        dataset,
        fps=30,
        visual_force_update_interval=4,
        visual_feedback_mode="residual",
        visual_residual_mapper=exact_mapper,
        visual_residual_gain_profile="full_only",
        enable_psm_tissue_contact=True,
    )
    controls.reset()
    controls.go_to_frame(args.frame)
    environment.step(compute_visual_forces=False)
    controls.apply_current_psm_pose()
    environment.sim.update_gaussian_transforms()
    base_state = environment.sim.clone_embodied_gaussian_rollout_state()
    exclusion = grip_control_exclusion_mask(exact_mapper, environment)

    geometry_rows = {}
    exact_result = None
    for label, mapper in (("legacy_static_covariance", legacy_mapper), ("deformation_covariance", exact_mapper)):
        environment.sim.copy_embodied_gaussian_rollout_state(base_state)
        environment.sim.update_gaussian_transforms()
        result = environment.sim.solve_visual_tissue_residual(
            mapper,
            environment.frames,
            dynamic_exclusion_mask=exclusion,
            observations_are_bgr=True,
        )
        accepted = environment.sim.apply_visual_tissue_residual(
            result,
            mapper=mapper,
            frames=environment.frames,
            observations_are_bgr=True,
        )
        metrics = dict(environment.sim.last_visual_tissue_residual_metrics or {})
        surrogate_gap = abs(
            float(result.final_visual_loss)
            - float(metrics["exact_final_visual_loss"])
        )
        geometry_rows[label] = {
            "accepted": bool(accepted),
            "initial_rgb_loss": float(result.initial_visual_loss),
            "surrogate_final_rgb_loss": float(result.final_visual_loss),
            "runtime_exact_final_rgb_loss": float(metrics["exact_final_visual_loss"]),
            "surrogate_runtime_absolute_gap": surrogate_gap,
            "residual_rms_mm": float(result.rms_residual_m * 1000.0),
        }
        if label == "deformation_covariance":
            exact_result = result
    assert exact_result is not None

    observation_dt = (
        float(controls.playback_timestamps[args.frame] - controls.playback_timestamps[args.frame - 1])
        if args.frame > 0
        else 1.0 / 30.0
    )
    velocity_rows = {}
    for beta in (0.0, 0.10, 0.25, 0.50):
        environment.sim.copy_embodied_gaussian_rollout_state(base_state)
        environment.sim.update_gaussian_transforms()
        accepted = environment.sim.apply_visual_tissue_residual(
            exact_result,
            mapper=exact_mapper,
            frames=environment.frames,
            observations_are_bgr=True,
            observation_dt_s=observation_dt if beta > 0.0 else None,
            velocity_correction_gain=beta,
            maximum_velocity_correction_m_s=0.03,
            dynamic_exclusion_mask=exclusion,
        )
        metrics = dict(environment.sim.last_visual_tissue_residual_metrics or {})
        horizon_losses = {}
        for step in range(1, 6):
            environment.step(compute_visual_forces=False)
            controls.apply_current_psm_pose()
            if step in (1, 3, 5):
                environment.sim.update_gaussian_transforms()
                alignment = environment.sim.evaluate_visual_tissue_alignment(
                    exact_mapper,
                    environment.frames,
                    observations_are_bgr=True,
                )
                horizon_losses[str(step)] = float(alignment.loss)
        velocity_rows[f"beta_{beta:g}"] = {
            "accepted": bool(accepted),
            "velocity_correction_maximum_m_s": float(
                metrics["velocity_correction_maximum_m_s"]
            ),
            "horizon_rgb_losses": horizon_losses,
            "weighted_h135_rgb_loss": float(
                0.10 * horizon_losses["1"]
                + 0.30 * horizon_losses["3"]
                + 0.60 * horizon_losses["5"]
            ),
        }

    legacy_gap = geometry_rows["legacy_static_covariance"][
        "surrogate_runtime_absolute_gap"
    ]
    exact_gap = geometry_rows["deformation_covariance"][
        "surrogate_runtime_absolute_gap"
    ]
    best_velocity = min(
        velocity_rows,
        key=lambda key: velocity_rows[key]["weighted_h135_rgb_loss"],
    )
    report = {
        "schema": "super_visual_state_observer_diagnostic_v1",
        "frame": args.frame,
        "geometry": geometry_rows,
        "velocity": velocity_rows,
        "geometry_surrogate_gap_reduction_fraction": float(
            (legacy_gap - exact_gap) / max(legacy_gap, 1.0e-12)
        ),
        "best_velocity_beta_on_same_image_hold": best_velocity,
        "gates": {
            "deformation_covariance_reduces_surrogate_runtime_gap": bool(
                exact_gap < legacy_gap
            ),
            "deformation_covariance_candidate_is_accepted": bool(
                geometry_rows["deformation_covariance"]["accepted"]
            ),
            "all_velocity_branches_are_accepted": all(
                row["accepted"] for row in velocity_rows.values()
            ),
            "all_values_are_finite": bool(
                all(
                    np.isfinite(value)
                    for row in geometry_rows.values()
                    for key, value in row.items()
                    if key != "accepted"
                )
                and all(
                    np.isfinite(row["weighted_h135_rgb_loss"])
                    for row in velocity_rows.values()
                )
            ),
        },
    }
    report["passed"] = all(report["gates"].values())
    output = json.dumps(report, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
