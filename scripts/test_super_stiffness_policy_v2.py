#!/usr/bin/env python3
"""CPU regression gates for transactional, strain-driven stiffness policy v2."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

import example_embodied_super_offline as super_example
from embodied_gaussians.physics_simulator.online_tissue_stiffness import (
    OnlineTissueStiffnessSettings,
    ResidualDrivenPaperStiffnessUpdater,
)


def make_updater(*, strain_weight: float = 0.80):
    rest = torch.tensor(
        ((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.002, 0.0, 0.0)),
        dtype=torch.float32,
    )
    distance = torch.full((3,), 0.20)
    shape = torch.full((3,), 0.004)
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=torch.zeros(3, dtype=torch.bool),
        edges=torch.tensor(((0, 1), (1, 2)), dtype=torch.long),
        distance_stiffness=distance,
        shape_stiffness=shape,
        settings=OnlineTissueStiffnessSettings(
            maximum_log_step=0.10,
            signal_ema_decay=0.0,
            spatial_smoothing_iterations=0,
            graph_smoothing_iterations=0,
            hardening_bias=0.0,
            strain_signal_weight=strain_weight,
        ),
    )
    return updater, rest, distance, shape


def main() -> None:
    updater, rest, distance, shape = make_updater()
    prediction = rest.clone()
    prediction[:, 2] = -0.001
    prediction[1, 0] += 0.00020
    residual = torch.zeros_like(rest)
    residual[0, 0] = 0.00005
    residual[1, 0] = -0.00010
    evidence = updater.accumulate_evidence(
        physical_prediction=prediction,
        accepted_residual=residual,
    )
    proposal = updater.propose_from_evidence(evidence)
    verified_distance = distance.clone()
    verified_shape = shape.clone()

    distance_only = updater.candidate_variant(
        proposal,
        distance_scale=2.0,
        shape_scale=0.0,
        variant_label="distance_only",
    )
    shape_only = updater.candidate_variant(
        proposal,
        distance_scale=0.0,
        shape_scale=2.0,
        variant_label="shape_only",
    )
    global_distance = updater.global_median_candidate(
        proposal,
        distance_median=0.40,
        variant_label="global_distance_0.4",
    )
    global_shape = updater.global_median_candidate(
        proposal,
        shape_median=0.008,
        variant_label="global_shape_0.008",
    )
    mixed_axis_merge_rejected = False
    try:
        updater.merge_candidate_regions(
            (distance_only, shape_only),
            variant_label="invalid_mixed_axis_merge",
        )
    except ValueError as exc:
        mixed_axis_merge_rejected = "same material axis" in str(exc)

    controls = object.__new__(super_example.SuperPlaybackControls)
    controls.stiffness_updater = updater
    candidate_material = SimpleNamespace(
        distance_stiffness=torch.full((3,), 0.80),
        shape_stiffness=torch.full((3,), 0.016),
    )

    def mutate_shadow(**kwargs):
        updater.install_candidate_for_rollout(kwargs["candidate"])
        return {"mutated": True}

    controls._run_stiffness_rollout_shadow_unisolated = mutate_shadow
    normal_result = (
        super_example.SuperPlaybackControls._run_stiffness_rollout_shadow(
            controls,
            rollout_state="unused",
            commands=[],
            candidate=candidate_material,
            use_candidate=True,
            freeze_grip_state_machine=True,
        )
    )
    normal_restored = torch.equal(distance, verified_distance) and torch.equal(
        shape, verified_shape
    )

    def raising_shadow(**kwargs):
        updater.install_candidate_for_rollout(kwargs["candidate"])
        raise RuntimeError("synthetic shadow failure")

    controls._run_stiffness_rollout_shadow_unisolated = raising_shadow
    raised = False
    try:
        super_example.SuperPlaybackControls._run_stiffness_rollout_shadow(
            controls,
            rollout_state="unused",
            commands=[],
            candidate=candidate_material,
            use_candidate=True,
            freeze_grip_state_machine=True,
        )
    except RuntimeError as exc:
        raised = str(exc) == "synthetic shadow failure"
    exception_restored = torch.equal(
        distance, verified_distance
    ) and torch.equal(shape, verified_shape)

    # Pure rigid translation below the vector-deformation floor has no edge
    # strain and therefore cannot manufacture material evidence.
    rigid_updater, rigid_rest, _, _ = make_updater(strain_weight=1.0)
    rigid_prediction = rigid_rest + torch.tensor((0.0, 0.0, 5.0e-5))
    rigid_residual = torch.full_like(rigid_rest, 2.5e-5)
    rigid_evidence = rigid_updater.accumulate_evidence(
        physical_prediction=rigid_prediction,
        accepted_residual=rigid_residual,
    )

    axis_checks = (
        torch.equal(distance_only.shape_stiffness, verified_shape)
        and torch.equal(shape_only.distance_stiffness, verified_distance)
        and torch.equal(global_distance.shape_stiffness, verified_shape)
        and torch.equal(global_shape.distance_stiffness, verified_distance)
    )
    # Exercise the runtime fail-closed assertion too.
    controls._assert_stiffness_candidate_axis_isolation(
        global_shape,
        verified_distance=verified_distance,
        verified_shape=verified_shape,
        may_change_distance=False,
        may_change_shape=True,
    )
    sparse_local = SimpleNamespace(
        distance_log_step=torch.tensor([0.1, 0.1] + [0.0] * 18),
        shape_log_step=torch.zeros(20),
        material_valid_mask=torch.ones(20, dtype=torch.bool),
        metrics={"candidate_scope": "full"},
    )
    sparse_global = SimpleNamespace(
        distance_log_step=sparse_local.distance_log_step,
        shape_log_step=sparse_local.shape_log_step,
        material_valid_mask=sparse_local.material_valid_mask,
        metrics={"candidate_scope": "global_material_offset"},
    )
    signal_before_defer = updater.signal_ema.detach().clone()
    updates_before_defer = updater.update_count
    rejects_before_defer = updater.rejected_count
    defer_metrics = updater.defer("global_confirmation_pending", proposal)
    defer_preserved_evidence = bool(
        defer_metrics["status"] == "deferred"
        and torch.equal(updater.signal_ema, signal_before_defer)
        and updater.update_count == updates_before_defer
        and updater.rejected_count == rejects_before_defer
        and updater.pending_candidate is None
    )

    gates = {
        "edge_strain_is_primary_and_active": (
            evidence.metrics["strain_signal_weight"] == 0.80
            and evidence.metrics["strain_signal_active_particles"] >= 2
            and bool((evidence.log_step[:2] > 0.0).all().item())
        ),
        "rigid_translation_creates_no_strain_evidence": (
            rigid_evidence.metrics["strain_signal_active_particles"] == 0
            and bool((rigid_evidence.log_step == 0.0).all().item())
        ),
        "local_and_global_candidates_are_axis_independent": axis_checks,
        "global_full_range_target_is_clipped_to_single_0p10_step": bool(
            global_distance.metrics["maximum_log_step"] <= 0.100001
            and global_shape.metrics["maximum_log_step"] <= 0.100001
            and global_distance.metrics["maximum_log_step"] >= 0.099
            and global_shape.metrics["maximum_log_step"] >= 0.099
        ),
        "mixed_axis_component_merge_is_rejected": mixed_axis_merge_rejected,
        "local_margin_uses_realized_material_support": (
            controls._stiffness_candidate_scope_fraction(sparse_local) == 0.10
            and controls._stiffness_candidate_scope_fraction(sparse_global)
            == 1.0
        ),
        "shadow_restores_verified_material_on_return": (
            normal_result == {"mutated": True} and normal_restored
        ),
        "shadow_restores_verified_material_on_exception": (
            raised and exception_restored
        ),
        "global_confirmation_defer_preserves_evidence_and_counts": (
            defer_preserved_evidence
        ),
    }
    report = {"gates": gates, "passed": all(gates.values())}
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
