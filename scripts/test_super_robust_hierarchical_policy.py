#!/usr/bin/env python3
"""CPU-only gates for robust H10/residual-effort material admission."""

from __future__ import annotations

import json

import example_embodied_super_offline as super_example


def shadow(
    *,
    residual_effort: float,
    h1: float = 0.40,
    h3: float = 0.40,
    h5: float = 0.40,
    h10: float = 0.40,
) -> dict:
    return {
        "visual_loss": 0.40,
        "camera_losses": (0.40, 0.40),
        "validation_horizons_complete": True,
        "horizon_visual_losses": {1: h1, 3: h3, 5: h5, 10: h10},
        "open_loop_residual_rms_m": residual_effort,
        "minimum_volume_ratio": 0.90,
        "inverted_tetrahedra": 0,
        "maximum_penetration_m": 0.001,
        "anchor_error_rms_m": 0.0,
        "anchor_error_maximum_m": 0.0,
    }


def main() -> None:
    super_example.configure_stiffness_admission_policy(
        (1, 3, 5, 10),
        "robust_hierarchical_system_id",
        "weighted_window",
    )
    baseline = shadow(residual_effort=1.0e-5, h10=0.50)
    baseline["visual_loss"] = 0.50
    baseline["camera_losses"] = (0.50, 0.50)
    baseline["horizon_visual_losses"] = {
        1: 0.50,
        3: 0.50,
        5: 0.50,
        10: 0.50,
    }
    # H1 may move by bounded noise, while H5/H10 and the weighted aggregate
    # clearly improve. Residual effort ranks otherwise equivalent candidates
    # but must not veto the weak-effort branch.
    good = shadow(
        residual_effort=9.0e-6,
        h1=0.501,
        h3=0.499,
        h5=0.480,
        h10=0.470,
    )
    good["visual_loss"] = 0.480
    good["camera_losses"] = (0.480, 0.480)
    weak_effort = dict(good)
    weak_effort["open_loop_residual_rms_m"] = 1.01e-5
    bad_h10 = dict(good)
    bad_h10["horizon_visual_losses"] = dict(good["horizon_visual_losses"])
    bad_h10["horizon_visual_losses"][10] = 0.50
    bad_volume = dict(good)
    bad_volume["inverted_tetrahedra"] = 1
    bad_penetration = dict(good)
    bad_penetration["maximum_penetration_m"] = 0.0012
    bad_anchor = dict(good)
    bad_anchor["anchor_error_rms_m"] = 6.0e-5
    good_reasons, _, _ = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, good
        )
    )
    effort_reasons, _, _ = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, weak_effort
        )
    )
    h10_reasons, _, _ = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, bad_h10
        )
    )
    volume_reasons, _, _ = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, bad_volume
        )
    )
    penetration_reasons, _, _ = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, bad_penetration
        )
    )
    anchor_reasons, _, _ = (
        super_example.SuperPlaybackControls._stiffness_shadow_rejection_reasons(
            baseline, bad_anchor
        )
    )
    gates = {
        "h1_noise_with_clear_h5_h10_candidate_passes": good_reasons == [],
        "weak_residual_effort_is_not_a_commit_veto": effort_reasons == [],
        "h10_regression_is_rejected": any(
            "long_not_improved:H10" in reason
            for reason in h10_reasons
        ),
        "tetrahedron_gate_remains_hard": (
            "volume_quality_regression" in volume_reasons
        ),
        "penetration_gate_remains_hard": (
            "penetration_regression" in penetration_reasons
        ),
        "anchor_gate_remains_hard": (
            "grip_anchor_regression" in anchor_reasons
        ),
        "residual_effort_breaks_equal_gap_ties": (
            super_example.SuperPlaybackControls._stiffness_candidate_selection_key(
                baseline,
                {
                    "improvement": 0.02,
                    "scope": "full",
                    "variant_label": "better_effort",
                    "shadow": good,
                },
            )
            < super_example.SuperPlaybackControls._stiffness_candidate_selection_key(
                baseline,
                {
                    "improvement": 0.02,
                    "scope": "full",
                    "variant_label": "worse_effort",
                    "shadow": weak_effort,
                },
            )
        ),
        "hard_to_soft_global_jump_is_outside_trust_region": not (
            super_example.stiffness_target_inside_global_trust_region(1.60, 0.10)
        ),
        "hard_adjacent_global_step_is_inside_trust_region": (
            super_example.stiffness_target_inside_global_trust_region(1.60, 0.80)
        ),
        "soft_recovery_step_is_inside_trust_region": (
            super_example.stiffness_target_inside_global_trust_region(0.10, 0.20)
        ),
        "uncommitted_direction_can_move_either_way": (
            super_example.stiffness_target_follows_global_direction(0.40, 0.20, 0)
            and super_example.stiffness_target_follows_global_direction(
                0.40, 0.80, 0
            )
        ),
        "hard_recovery_direction_allows_monotonic_softening": (
            super_example.stiffness_target_follows_global_direction(
                0.80, 0.40, -1
            )
        ),
        "hard_recovery_direction_forbids_bounce": not (
            super_example.stiffness_target_follows_global_direction(
                0.20, 0.40, -1
            )
        ),
        "same_target_is_not_retested": not (
            super_example.stiffness_target_follows_global_direction(
                0.20, 0.20, -1
            )
        ),
        "objective_names_both_2d_rgb_terms": (
            super_example.stiffness_validation_objective_name()
            == "material_isolation_weighted_window_gap_h1_h3_h5_h10_residual_effort_tiebreak_no_residual"
        ),
    }
    report = {"gates": gates, "passed": all(gates.values())}
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
