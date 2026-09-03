#!/usr/bin/env python3
"""CPU gates for AllTracker-primary H1/H3/H5 stiffness admission."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

import example_embodied_super_offline as super_example
from embodied_gaussians.physics_simulator.online_tissue_stiffness import (
    OnlineTissueStiffnessSettings,
    ResidualDrivenPaperStiffnessUpdater,
)


def shadow(
    *,
    trajectory_loss_m: float,
    h1_m: float,
    h3_m: float,
    h5_m: float,
    rgb_loss: float = 0.020,
) -> dict:
    return {
        "visual_loss": rgb_loss,
        "camera_losses": (rgb_loss, rgb_loss),
        "trajectory_tracking_loss_m": trajectory_loss_m,
        "horizon_trajectory_losses_m": {1: h1_m, 3: h3_m, 5: h5_m},
        "horizon_visual_losses": {1: rgb_loss, 3: rgb_loss, 5: rgb_loss},
        "validation_horizons_complete": True,
        "minimum_volume_ratio": 0.90,
        "tetrahedra_below_volume_floor": 0,
        "inverted_tetrahedra": 0,
        "maximum_penetration_m": 0.001,
        "anchor_error_rms_m": 0.0,
        "anchor_error_maximum_m": 0.0,
    }


class Sequence:
    def __init__(self) -> None:
        self.next_source_frames = np.arange(2, 20, 2, dtype=np.int32)

    def pair_index_for_next_frame(self, frame_index: int) -> int | None:
        matches = np.flatnonzero(self.next_source_frames == frame_index)
        return None if not len(matches) else int(matches[0])


def main() -> None:
    super_example.configure_stiffness_admission_policy(
        (1, 3, 5), "direct_alternating_gradient", "relaxed_h135"
    )
    baseline = shadow(
        trajectory_loss_m=0.003000,
        h1_m=0.003000,
        h3_m=0.003000,
        h5_m=0.003000,
    )
    good = shadow(
        trajectory_loss_m=0.002995,
        h1_m=0.003001,
        h3_m=0.002996,
        h5_m=0.002992,
        rgb_loss=0.0201,
    )
    bad_h3 = shadow(
        trajectory_loss_m=0.002995,
        h1_m=0.002999,
        h3_m=0.003001,
        h5_m=0.002990,
    )
    excessive_h3 = shadow(
        trajectory_loss_m=0.002995,
        h1_m=0.002999,
        h3_m=0.003010,
        h5_m=0.002990,
    )
    bad_h35_aggregate = shadow(
        trajectory_loss_m=0.002995,
        h1_m=0.002999,
        h3_m=0.003005,
        h5_m=0.0029995,
    )
    bad_rgb = shadow(
        trajectory_loss_m=0.002990,
        h1_m=0.002999,
        h3_m=0.002995,
        h5_m=0.002985,
        rgb_loss=0.0203,
    )
    numerical_noise = shadow(
        trajectory_loss_m=0.00299995,
        h1_m=0.003000,
        h3_m=0.00299995,
        h5_m=0.00299995,
    )
    good_reasons, improvement, required = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, good, required_long_horizons=(3, 5)
        )
    )
    h3_reasons, _, _ = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, bad_h3, required_long_horizons=(3, 5)
        )
    )
    excessive_h3_reasons, _, _ = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, excessive_h3, required_long_horizons=(3, 5)
        )
    )
    aggregate_reasons, _, _ = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, bad_h35_aggregate, required_long_horizons=(3, 5)
        )
    )
    rgb_reasons, _, _ = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, bad_rgb, required_long_horizons=(3, 5)
        )
    )
    noise_reasons, _, noise_required = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, numerical_noise, required_long_horizons=(3, 5)
        )
    )

    controls = object.__new__(super_example.SuperPlaybackControls)
    controls.playback_timestamps = tuple(float(i) for i in range(20))
    controls.visual_feedback_mode = "trajectory"
    controls.flow_depth_observations = Sequence()
    controls.tissue_benchmark_recorder = SimpleNamespace(
        protocol="future_80to20", observation_allowed=lambda _frame: True
    )
    mapped = controls._stiffness_training_validation_frames(2, (1, 3, 5))

    # Direct online admission keeps H-rollouts disabled, but material evidence
    # must not start before physical contact and must persist across two
    # independent edge-strain observations.  A dropout resets the sequence.
    no_contact = super_example.advance_direct_stiffness_observability(
        previous_count=0,
        previous_frame_index=-1,
        current_frame_index=100,
        contact_count=0,
        strain_active_particles=100,
    )
    first_valid = super_example.advance_direct_stiffness_observability(
        previous_count=no_contact[0],
        previous_frame_index=no_contact[1],
        current_frame_index=102,
        contact_count=1,
        strain_active_particles=24,
    )
    duplicate_frame = super_example.advance_direct_stiffness_observability(
        previous_count=first_valid[0],
        previous_frame_index=first_valid[1],
        current_frame_index=102,
        contact_count=1,
        strain_active_particles=24,
    )
    second_valid = super_example.advance_direct_stiffness_observability(
        previous_count=first_valid[0],
        previous_frame_index=first_valid[1],
        current_frame_index=104,
        contact_count=2,
        strain_active_particles=30,
    )
    strain_dropout = super_example.advance_direct_stiffness_observability(
        previous_count=second_valid[0],
        previous_frame_index=second_valid[1],
        current_frame_index=106,
        contact_count=2,
        strain_active_particles=23,
    )
    after_dropout = super_example.advance_direct_stiffness_observability(
        previous_count=strain_dropout[0],
        previous_frame_index=strain_dropout[1],
        current_frame_index=108,
        contact_count=2,
        strain_active_particles=30,
    )

    alignment_controls = object.__new__(super_example.SuperPlaybackControls)
    alignment_controls.visual_feedback_mode = "trajectory"
    alignment_controls.flow_depth_bindings = SimpleNamespace(
        track_valid=np.asarray([True, True]),
        support_counts=np.asarray([1, 1], dtype=np.int32),
        particle_ids=np.asarray([[0], [1]], dtype=np.int32),
        particle_weights=np.asarray([[1.0], [1.0]], dtype=np.float32),
        initial_points_table=np.zeros((2, 3), dtype=np.float32),
    )
    alignment_observation = SimpleNamespace(
        track_valid=np.asarray([True, False]),
        confidence=np.asarray([1.0, 0.0], dtype=np.float32),
        next_points_table=np.asarray(
            [[0.001, 0.0, 0.0], [np.nan, np.nan, np.nan]],
            dtype=np.float32,
        ),
    )
    alignment_controls.flow_depth_observations = SimpleNamespace(
        pair_index_for_next_frame=lambda frame: 0 if frame == 2 else None,
        observation=lambda _index: alignment_observation,
        current_source_frames=np.asarray([0], dtype=np.int32),
    )
    alignment_controls._flow_depth_reference_range_centers = np.zeros(
        (2, 3), dtype=np.float64
    )
    alignment = alignment_controls._evaluate_alltracker_trajectory_alignment(
        torch.zeros((2, 3), dtype=torch.float32), 2, None
    )

    relaxed_profile_is_valid = bool(
        super_example.STIFFNESS_CANDIDATE_PROFILE
        == "direct_alternating_gradient"
        and "h1_h3_h5" in super_example.stiffness_validation_objective_name()
    )

    # Exercise the actual runtime consumption branch, not only the pure
    # persistence-state helper.
    super_example.configure_stiffness_admission_policy(
        (1, 3, 5), "direct_alternating_gradient", "direct_online"
    )
    node_count = 32
    direct_rest = torch.zeros((node_count, 3), dtype=torch.float32)
    direct_rest[:, 0] = torch.arange(node_count, dtype=torch.float32) * 0.001
    direct_edges = torch.stack(
        (
            torch.arange(node_count - 1, dtype=torch.long),
            torch.arange(1, node_count, dtype=torch.long),
        ),
        dim=1,
    )
    direct_prediction = direct_rest.clone()
    direct_prediction[:, 0] *= 1.04
    direct_corrected = direct_rest.clone()
    direct_corrected[:, 0] *= 1.02
    direct_distance = torch.full((node_count,), 0.01)
    direct_shape = torch.full((node_count,), 0.0005)
    direct_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=direct_rest,
        fixed_mask=torch.zeros(node_count, dtype=torch.bool),
        edges=direct_edges,
        distance_stiffness=direct_distance,
        shape_stiffness=direct_shape,
        settings=OnlineTissueStiffnessSettings(
            log_learning_rate=0.30,
            maximum_log_step=0.10,
            effective_log_step_target=0.08,
            maximum_step_amplification=64.0,
            signal_ema_decay=0.70,
            strain_signal_weight=0.75,
            distance_minimum=0.00001,
            distance_maximum=4.0,
            shape_minimum=0.000001,
            shape_maximum=0.04,
        ),
    )
    direct_controls = object.__new__(super_example.SuperPlaybackControls)
    direct_controls.stiffness_updater = direct_updater
    direct_controls._pending_stiffness_validation = None
    direct_controls._stiffness_trial_count = 0
    direct_controls._last_stiffness_commit_frame_index = -(10**9)
    direct_controls._direct_stiffness_observable_window_count = 0
    direct_controls._direct_stiffness_last_observable_frame_index = -1
    direct_controls._maybe_store_stiffness_history = lambda **_kwargs: None
    installed = super_example.InstalledParticleInnovation(
        corrected_positions=direct_corrected,
        residual=direct_corrected - direct_prediction,
    )
    valid = torch.ones(node_count, dtype=torch.bool)
    excluded = torch.zeros(node_count, dtype=torch.bool)

    def consume_direct(frame: int, contacts: int) -> dict:
        direct_controls.current_frame_index = frame
        return direct_controls._consume_confirmed_visual_stiffness_evidence(
            applied_result=installed,
            quality_valid_mask=valid,
            supervision_valid_mask=valid,
            control_exclusion_mask=excluded,
            stiffness_gate_paused=False,
            stiffness_jaw_speed=0.0,
            stiffness_grip_active=False,
            stiffness_contact_count=contacts,
            accepted_positions=direct_corrected,
            stiffness_metrics=None,
        )

    runtime_no_contact = consume_direct(100, 0)
    runtime_first_window = consume_direct(102, 1)
    runtime_second_window = consume_direct(104, 1)

    gates = {
        "alltracker_h3_h5_improvement_passes": good_reasons == [],
        "primary_improvement_is_in_metres": bool(
            np.isclose(improvement, 6.5e-6)
        ),
        "trajectory_margin_scales_with_current_error": bool(
            np.isclose(required, 3.0e-7)
        ),
        "sub_micrometre_numerical_noise_cannot_commit": bool(
            noise_reasons
            and np.isclose(noise_required, 3.0e-7)
        ),
        "relaxed_h135_accepts_alternating_single_axis_profile": (
            relaxed_profile_is_valid
        ),
        "tiny_h3_regression_is_allowed_when_h5_and_aggregate_improve": (
            h3_reasons == []
        ),
        "excessive_h3_regression_is_rejected": any(
            "alltracker_trajectory_horizon" in reason and "H3" in reason
            for reason in excessive_h3_reasons
        ),
        "h3_h5_weighted_aggregate_must_improve": (
            "alltracker_trajectory_not_improved" in aggregate_reasons
        ),
        "clear_rgb_regression_remains_vetoed": (
            "rgb_auxiliary_regression" in rgb_reasons
        ),
        "horizons_count_tracker_observations": mapped == (4, 8, 12),
        "invalid_zero_weight_tracks_do_not_create_nan": bool(
            alignment is not None
            and np.isfinite(alignment["mean_error_m"])
            and np.isclose(alignment["mean_error_m"], 0.001)
        ),
        "direct_update_requires_contact": bool(
            no_contact[0] == 0
            and not no_contact[2]
            and no_contact[3] == "contact_below_minimum"
        ),
        "direct_update_requires_two_independent_strain_windows": bool(
            first_valid[0] == 1
            and not first_valid[2]
            and duplicate_frame[0] == 1
            and not duplicate_frame[2]
            and second_valid[0] == 2
            and second_valid[2]
        ),
        "direct_strain_dropout_resets_persistence": bool(
            strain_dropout[0] == 0
            and not strain_dropout[2]
            and strain_dropout[3] == "edge_strain_below_minimum"
            and after_dropout[0] == 1
            and not after_dropout[2]
        ),
        "runtime_direct_branch_waits_for_contact_and_second_window": bool(
            runtime_no_contact["status"] == "material_unobservable"
            and runtime_no_contact["ema_history_cleared"] == 1
            and runtime_first_window["status"]
            == "material_observability_warmup"
            and runtime_first_window[
                "observability_consecutive_strain_windows"
            ] == 1
            and runtime_second_window["status"] == "committed"
            and runtime_second_window["observability_ready"]
            and direct_updater.update_count == 1
            and runtime_second_window["maximum_log_step"] <= 0.10
        ),
    }
    report = {"gates": gates, "passed": all(gates.values())}
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
