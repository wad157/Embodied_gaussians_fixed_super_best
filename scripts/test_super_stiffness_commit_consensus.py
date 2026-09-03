#!/usr/bin/env python3
"""CPU gates for unrestricted direct causal stiffness commits."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import example_embodied_super_offline as super_example
import torch


def shadow_metrics(*, improvement: float) -> tuple[dict, dict]:
    baseline = {
        "visual_loss": 0.100000,
        "camera_losses": (0.050000, 0.050000),
        "minimum_volume_ratio": 0.50,
        "inverted_tetrahedra": 0,
        "tetrahedra_below_volume_floor": 0,
        "maximum_penetration_m": 0.0010,
        "anchor_error_rms_m": 0.00010,
        "anchor_error_maximum_m": 0.00010,
        "validation_horizons_complete": True,
        "horizon_visual_losses": {
            horizon: 0.100000
            for horizon in super_example.STIFFNESS_ADMISSION_HORIZONS
        },
    }
    candidate = dict(baseline)
    candidate["visual_loss"] = baseline["visual_loss"] - improvement
    candidate["horizon_visual_losses"] = {
        horizon: (
            baseline["horizon_visual_losses"][horizon]
            - (improvement if horizon in (5, 10) else 0.0)
        )
        for horizon in super_example.STIFFNESS_ADMISSION_HORIZONS
    }
    return baseline, candidate


def record(
    label: str,
    *,
    scope: str,
    distance_scale: float = 0.0,
    shape_scale: float = 0.0,
    improvement: float = 2.0e-6,
    reasons: tuple[str, ...] = (),
) -> dict:
    baseline, shadow = shadow_metrics(improvement=improvement)
    return {
        "variant_label": label,
        "scope": scope,
        "scope_fraction": 0.10 if scope != "global_material_offset" else 1.0,
        "distance_scale": distance_scale,
        "shape_scale": shape_scale,
        "candidate": SimpleNamespace(
            label=label,
            source_mask=torch.tensor(
                [True, True, False, False]
                if scope.startswith("component_")
                else [True, True, True, True]
            ),
        ),
        "shadow": shadow,
        "reasons": list(reasons),
        "improvement": improvement,
        "required_improvement": 1.0e-6,
    }


def make_controls() -> super_example.SuperPlaybackControls:
    controls = object.__new__(super_example.SuperPlaybackControls)
    controls._stiffness_local_phase_family_commits = {}
    controls._stiffness_local_phase_family_last_commit_frame = {}
    controls._stiffness_local_confirmation_state = {}
    controls._stiffness_global_confirmation_key = None
    controls._stiffness_global_confirmation_count = 0
    controls._evaluate_stiffness_history = lambda _candidate: (
        1.0e-4,
        1.0e-4,
        True,
    )
    return controls


def main() -> None:
    original_policy = (
        super_example.STIFFNESS_ADMISSION_HORIZONS,
        super_example.STIFFNESS_CANDIDATE_PROFILE,
        super_example.STIFFNESS_ADMISSION_MODE,
    )
    gates: dict[str, bool] = {}
    try:
        super_example.configure_stiffness_admission_policy(
            (1, 3, 5, 10), "direct_residual_gradient", "causal_fixed_lag"
        )

        local_scale = super_example.SuperPlaybackControls._stiffness_candidate_margin_scale(
            scope="full", scope_fraction=0.10
        )
        global_scale = super_example.SuperPlaybackControls._stiffness_candidate_margin_scale(
            scope="global_material_offset", scope_fraction=0.10
        )
        baseline, candidate = shadow_metrics(improvement=2.0e-6)
        local_reasons, _, local_required = (
            super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
                baseline, candidate, absolute_margin_scale=local_scale
            )
        )
        global_reasons, _, global_required = (
            super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
                baseline, candidate, absolute_margin_scale=global_scale
            )
        )
        gates["local_margin_is_support_normalized_but_global_is_not"] = bool(
            local_scale == 0.10
            and global_scale == 1.0
            and not local_reasons
            and "prediction_gap_not_improved" in global_reasons
            and math.isclose(local_required, 1.0e-6)
            and math.isclose(global_required, 1.0e-5)
        )

        gates["event_driven_policy_has_no_frame_or_count_limits"] = bool(
            super_example.STIFFNESS_CAUSAL_WARMUP_END_FRAME == 0
            and super_example.STIFFNESS_CAUSAL_MAXIMUM_COMMITS is None
            and super_example.STIFFNESS_CAUSAL_MAXIMUM_TRIALS is None
            and super_example.STIFFNESS_CAUSAL_MINIMUM_TRIAL_INTERVAL_FRAMES
            == 0
            and super_example.STIFFNESS_CAUSAL_LOCAL_MAXIMUM_COMMITS_PER_PHASE_FAMILY
            is None
            and super_example.STIFFNESS_CAUSAL_MINIMUM_SAME_FAMILY_COMMIT_INTERVAL_FRAMES
            == 0
        )

        controls = make_controls()
        controls._stiffness_local_phase_family_commits[("lift", "distance")] = 99
        local_distance = record(
            "distance_forward",
            scope="full",
            distance_scale=1.0,
        )
        local_shape = record(
            "shape_forward", scope="full", shape_scale=1.0
        )
        controls._apply_causal_stiffness_commit_policy(
            baseline=baseline,
            records=[local_distance, local_shape],
            proposal_phase="lift",
        )
        next_phase_distance = record(
            "distance_forward",
            scope="full",
            distance_scale=1.0,
        )
        controls._apply_causal_stiffness_commit_policy(
            baseline=baseline,
            records=[next_phase_distance],
            proposal_phase="place",
        )
        gates["local_update_counts_never_veto_safe_candidates"] = bool(
            "local_phase_family_commit_budget_exhausted"
            not in local_distance["reasons"]
            and "local_phase_family_commit_budget_exhausted"
            not in local_shape["reasons"]
            and "local_phase_family_commit_budget_exhausted"
            not in next_phase_distance["reasons"]
        )

        controls = make_controls()
        controls._stiffness_local_phase_family_commits[("press", "distance")] = 1
        controls._stiffness_local_phase_family_last_commit_frame[
            ("press", "distance")
        ] = 400
        early_second = record(
            "distance_forward", scope="full", distance_scale=1.0
        )
        controls._apply_causal_stiffness_commit_policy(
            baseline=baseline,
            records=[early_second],
            proposal_phase="press",
            proposal_frame_index=699,
        )
        late_second = record(
            "distance_forward", scope="full", distance_scale=1.0
        )
        controls._apply_causal_stiffness_commit_policy(
            baseline=baseline,
            records=[late_second],
            proposal_phase="press",
            proposal_frame_index=700,
        )
        gates["same_family_has_no_frame_cooldown"] = bool(
            "local_phase_family_commit_interval_not_reached"
            not in early_second["reasons"]
            and "local_phase_family_commit_interval_not_reached"
            not in late_second["reasons"]
            and "local_phase_family_commit_budget_exhausted"
            not in late_second["reasons"]
        )

        controls = make_controls()
        first_local = record(
            "distance_forward",
            scope="full",
            distance_scale=1.0,
        )
        first_local_policy = controls._apply_causal_stiffness_commit_policy(
            baseline=baseline,
            records=[first_local],
            proposal_phase="press",
        )
        second_local = record(
            "component0_distance_forward",
            scope="component_0",
            distance_scale=1.0,
        )
        second_local_policy = controls._apply_causal_stiffness_commit_policy(
            baseline=baseline,
            records=[second_local],
            proposal_phase="press",
        )
        failed_local = record(
            "distance_forward",
            scope="full",
            distance_scale=1.0,
            reasons=("camera_regression",),
        )
        controls._apply_causal_stiffness_commit_policy(
            baseline=baseline,
            records=[failed_local],
            proposal_phase="press",
        )
        restarted_local = record(
            "distance_forward",
            scope="full",
            distance_scale=1.0,
        )
        restarted_local_policy = controls._apply_causal_stiffness_commit_policy(
            baseline=baseline,
            records=[restarted_local],
            proposal_phase="press",
        )
        same_scope_again = record(
            "distance_forward",
            scope="full",
            distance_scale=1.0,
        )
        same_scope_policy = controls._apply_causal_stiffness_commit_policy(
            baseline=baseline,
            records=[same_scope_again],
            proposal_phase="press",
        )
        gates["two_window_local_confirmation_is_removed"] = bool(
            first_local_policy["confirmation_enabled"] is False
            and second_local_policy["confirmation_enabled"] is False
            and restarted_local_policy["confirmation_enabled"] is False
            and same_scope_policy["confirmation_enabled"] is False
            and not first_local_policy["local_confirmation_state"]
            and not first_local["reasons"]
            and not second_local["reasons"]
            and not restarted_local["reasons"]
            and not same_scope_again["reasons"]
        )

        controls = make_controls()
        first = record(
            "global_shape_0.003",
            scope="global_material_offset",
            improvement=1.2e-5,
        )
        first_policy = controls._apply_causal_stiffness_commit_policy(
            baseline=baseline, records=[first], proposal_phase="press"
        )
        second = record(
            "global_shape_0.003",
            scope="global_material_offset",
            improvement=1.2e-5,
        )
        second_policy = controls._apply_causal_stiffness_commit_policy(
            baseline=baseline, records=[second], proposal_phase="press"
        )
        failed = record(
            "global_shape_0.003",
            scope="global_material_offset",
            improvement=1.2e-5,
            reasons=("penetration_regression",),
        )
        failed_policy = controls._apply_causal_stiffness_commit_policy(
            baseline=baseline, records=[failed], proposal_phase="press"
        )
        after_failure = record(
            "global_shape_0.003",
            scope="global_material_offset",
            improvement=1.2e-5,
        )
        restart_policy = controls._apply_causal_stiffness_commit_policy(
            baseline=baseline,
            records=[after_failure],
            proposal_phase="press",
        )
        gates["two_window_global_confirmation_is_removed"] = bool(
            first_policy["confirmation_enabled"] is False
            and first_policy["global_confirmation_count"] == 0
            and not first["reasons"]
            and second_policy["global_confirmation_count"] == 0
            and not second["reasons"]
            and failed_policy["global_confirmation_count"] == 0
            and restart_policy["global_confirmation_count"] == 0
            and not after_failure["reasons"]
        )

        unsafe_candidate = dict(candidate)
        unsafe_candidate.update(
            inverted_tetrahedra=1,
            tetrahedra_below_volume_floor=1,
            maximum_penetration_m=0.0012,
            anchor_error_rms_m=0.00016,
            anchor_error_maximum_m=0.00016,
        )
        unsafe_reasons, _, _ = (
            super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
                baseline, unsafe_candidate, absolute_margin_scale=0.10
            )
        )
        gates["tet_penetration_and_anchor_remain_hard_vetoes"] = bool(
            "volume_quality_regression" in unsafe_reasons
            and "penetration_regression" in unsafe_reasons
            and "grip_anchor_regression" in unsafe_reasons
        )

        aggregate_candidate = dict(candidate)
        aggregate_candidate["camera_losses"] = (0.030000, 0.070010)
        aggregate_reasons, _, _ = (
            super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
                baseline,
                aggregate_candidate,
                history_safe=False,
                absolute_margin_scale=0.10,
            )
        )
        gates["causal_weighted_rgb_replaces_legacy_camera_history_vetoes"] = bool(
            "camera_regression" not in aggregate_reasons
            and "history_regression" not in aggregate_reasons
        )

        # Exercise the new terminal path itself: one continuous proposal that
        # passes H1/H5/H10 and all hard physics gates must commit immediately.
        controls = object.__new__(super_example.SuperPlaybackControls)
        controls.current_frame_index = 20
        controls.current_timestep = 0.20
        controls._last_state_index = 20
        controls._last_q_full = torch.zeros(1)
        controls._stiffness_trial_count = 1
        controls._current_action_phase = "press"
        controls.stiffness_metrics_recorder = None
        controls.dataset_manager = SimpleNamespace(
            update_frames=lambda _timestamp: None
        )
        controls.environment = SimpleNamespace(
            sim=SimpleNamespace(
                clone_embodied_gaussian_rollout_state=lambda: "live",
                copy_embodied_gaussian_rollout_state=lambda _state: None,
                update_gaussian_transforms=lambda: None,
            )
        )
        controls._synchronize_shadow_transaction = lambda _sim: None
        controls._adopt_validated_stiffness_rollout = lambda _shadow: {
            "validated_rollout_adopted": True,
            "validated_rollout_state_rms_m": 0.0,
            "validated_rollout_state_maximum_m": 0.0,
            "validated_rollout_state_finite": True,
        }
        controls._start_committed_stiffness_evaluation = lambda _pending: None

        material_valid = torch.ones(40, dtype=torch.bool)
        direct_candidate = SimpleNamespace(
            distance_stiffness=torch.full((40,), 0.21),
            shape_stiffness=torch.full((40,), 0.0042),
            distance_log_step=torch.full((40,), 0.05),
            shape_log_step=torch.full((40,), 0.05),
            material_valid_mask=material_valid,
            source_mask=material_valid,
            metrics={"candidate_scope_particles": 40},
        )

        class FakeUpdater:
            def __init__(self) -> None:
                self.distance_stiffness = torch.full((40,), 0.20)
                self.shape_stiffness = torch.full((40,), 0.004)
                self.pending_candidate = direct_candidate

            def restore_verified_stiffness(self, distance, shape) -> None:
                self.distance_stiffness.copy_(distance)
                self.shape_stiffness.copy_(shape)

            def commit(self, proposal) -> dict:
                assert proposal is self.pending_candidate
                self.distance_stiffness.copy_(proposal.distance_stiffness)
                self.shape_stiffness.copy_(proposal.shape_stiffness)
                self.pending_candidate = None
                return {"status": "committed", "update_count": 1}

            def reject(self, reason, proposal) -> dict:
                return {"status": "rejected", "rejection_reason": reason}

        controls.stiffness_updater = FakeUpdater()
        direct_baseline, direct_shadow = shadow_metrics(improvement=0.01)
        direct_baseline.update(
            distance_loss=0.1,
            volume_loss=0.1,
            shape_loss=0.1,
            open_loop_residual_rms_m=0.002,
        )
        direct_shadow.update(
            distance_loss=0.09,
            volume_loss=0.09,
            shape_loss=0.09,
            open_loop_residual_rms_m=0.001,
            particle_positions=torch.zeros((40, 3)),
            rollout_state="candidate",
            rollout_previous_residual=None,
        )
        controls._run_stiffness_prediction_shadow = (
            lambda _pending, *, use_candidate: (
                direct_shadow if use_candidate else direct_baseline
            )
        )
        pending = super_example.PendingStiffnessValidation(
            candidate=direct_candidate,
            rollout_state="proposal",
            frame_index=10,
            grip_active=True,
            history_baseline_rms_m=0.0,
            history_candidate_rms_m=0.0,
            validation_frame_indices=(11, 15, 20),
        )
        controls._pending_stiffness_validation = pending
        direct_metrics = controls._validate_direct_residual_gradient_stiffness(
            pending
        )
        gates["one_direct_safe_proposal_commits_without_extra_wait"] = bool(
            direct_metrics["status"] == "committed"
            and direct_metrics["candidate_search_trial_count"] == 1
            and direct_metrics["discrete_candidate_search_enabled"] is False
            and direct_metrics["two_window_confirmation_enabled"] is False
            and direct_metrics["long_horizon_commit_gate_enabled"] is False
            and direct_metrics["commit_count_limit"] is None
            and direct_metrics["minimum_commit_interval_frames"] == 0
            and controls._pending_stiffness_validation is None
        )
    finally:
        super_example.configure_stiffness_admission_policy(*original_policy)

    report = {"gates": gates, "passed": all(gates.values())}
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
