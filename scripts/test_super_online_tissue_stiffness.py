#!/usr/bin/env python3
"""Deterministic gate for signed, bounded online paper stiffness updates."""

from __future__ import annotations

import json

import torch

from embodied_gaussians.physics_simulator.online_tissue_stiffness import (
    OnlineTissueStiffnessSettings,
    ResidualDrivenPaperStiffnessUpdater,
)


def main() -> None:
    rest = torch.tensor(
        ((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.002, 0.0, 0.0)),
        dtype=torch.float32,
    )
    fixed = torch.tensor((True, False, False))
    edges = torch.tensor(((0, 1), (1, 2)), dtype=torch.long)
    distance = torch.full((3,), 0.4)
    shape = torch.full((3,), 0.008)
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=fixed,
        edges=edges,
        distance_stiffness=distance,
        shape_stiffness=shape,
        settings=OnlineTissueStiffnessSettings(
            log_learning_rate=0.10,
            signal_ema_decay=0.0,
            spatial_smoothing_iterations=0,
            hardening_bias=0.0,
        ),
    )
    prediction = rest.clone()
    prediction[1:, 2] = -0.001
    correction_toward_rest = torch.zeros_like(rest)
    correction_toward_rest[1:, 2] = 0.00030
    hard_metrics = updater.update(
        physical_prediction=prediction,
        accepted_residual=correction_toward_rest,
    )
    hardened_distance = distance.clone()
    hardened_shape = shape.clone()
    correction_farther_from_rest = -correction_toward_rest
    soft_metrics = updater.update(
        physical_prediction=prediction,
        accepted_residual=correction_farther_from_rest,
    )
    updater.reset()
    quality_metrics = updater.update(
        physical_prediction=prediction,
        accepted_residual=correction_toward_rest,
        quality_valid_mask=torch.tensor((False, False, True)),
    )
    quality_gated_distance = distance.clone()
    quality_gated_shape = shape.clone()
    updater.reset()
    candidate_distance = torch.full((3,), 0.4)
    candidate_shape = torch.full((3,), 0.008)
    candidate_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=fixed,
        edges=edges,
        distance_stiffness=candidate_distance,
        shape_stiffness=candidate_shape,
        settings=OnlineTissueStiffnessSettings(
            log_learning_rate=0.10,
            signal_ema_decay=0.0,
            spatial_smoothing_iterations=1,
            spatial_smoothing_blend=0.5,
            hardening_bias=0.0,
        ),
    )
    proposed = candidate_updater.propose(
        physical_prediction=prediction,
        accepted_residual=correction_toward_rest,
        supervision_valid_mask=torch.tensor((False, True, True)),
        control_exclusion_mask=torch.tensor((False, True, False)),
    )
    proposal_does_not_mutate_verified = bool(
        torch.equal(candidate_distance, torch.full((3,), 0.4))
        and torch.equal(candidate_shape, torch.full((3,), 0.008))
    )
    hard_exclusion_blocks_smoothing_and_ema = bool(
        proposed.log_step[1] == 0.0
        and proposed.signal_ema[1] == 0.0
        and candidate_updater.signal_ema[1] == 0.0
        and proposed.log_step[2] > 0.0
    )
    rejected = candidate_updater.reject("synthetic_rejection", proposed)
    rejection_preserves_verified = bool(
        torch.equal(candidate_distance, torch.full((3,), 0.4))
        and torch.equal(candidate_shape, torch.full((3,), 0.008))
        and rejected["status"] == "rejected"
        and candidate_updater.pending_candidate is None
    )
    candidate_updater.signal_ema.fill_(1.0)
    candidate_updater.invalidate_signal_history(
        torch.tensor((False, True, False))
    )
    selective_ema_invalidation_works = bool(
        candidate_updater.signal_ema.tolist() == [1.0, 0.0, 1.0]
    )
    candidate_updater.invalidate_signal_history()
    global_ema_invalidation_works = bool(
        torch.all(candidate_updater.signal_ema == 0.0)
    )

    # Production-response probe: use a spatially uniform signal so the real
    # one-ring smoother preserves its magnitude.  The current GUI baseline is
    # intentionally above the new online floor and must now move both ways.
    production_fixed = torch.zeros(3, dtype=torch.bool)
    production_prediction = rest.clone()
    production_prediction[:, 2] = -0.001
    production_toward_rest = torch.zeros_like(rest)
    production_toward_rest[:, 2] = 0.00030
    production_farther_from_rest = -production_toward_rest
    production_distance = torch.full((3,), 0.20)
    production_shape = torch.full((3,), 0.004)
    production_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=production_distance,
        shape_stiffness=production_shape,
    )
    observable_prediction = rest.clone()
    observable_prediction[1, 0] += 0.00005
    observable_corrected = rest.clone()
    observability_ema_before = production_updater.signal_ema.clone()
    observable_mask = production_updater.edge_strain_observability_mask(
        prediction=observable_prediction,
        corrected=observable_corrected,
        eligible_mask=torch.ones(3, dtype=torch.bool),
    )
    excluded_observable_mask = (
        production_updater.edge_strain_observability_mask(
            prediction=observable_prediction,
            corrected=observable_corrected,
            eligible_mask=torch.tensor((True, False, True)),
        )
    )
    observability_ema_after = production_updater.signal_ema.clone()
    production_softening = production_updater.propose(
        physical_prediction=production_prediction,
        accepted_residual=production_farther_from_rest,
    )
    first_soft_distance = float(
        production_softening.distance_stiffness[0].item()
    )
    first_soft_shape = float(
        production_softening.shape_stiffness[0].item()
    )
    production_updater.reject("first_softening_probe", production_softening)
    production_hardening = production_updater.propose(
        physical_prediction=production_prediction,
        accepted_residual=production_toward_rest,
    )
    first_hard_distance = float(
        production_hardening.distance_stiffness[0].item()
    )
    first_hard_shape = float(
        production_hardening.shape_stiffness[0].item()
    )
    production_updater.reject("first_hardening_probe", production_hardening)

    # Reproduce the immediately preceding production policy in the same probe
    # so the report contains measured before/after response, not hand-derived
    # percentages only.
    previous_distance = torch.full((3,), 0.20)
    previous_shape = torch.full((3,), 0.004)
    previous_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=previous_distance,
        shape_stiffness=previous_shape,
        settings=OnlineTissueStiffnessSettings(
            signal_ema_decay=0.80,
            distance_minimum=0.20,
            shape_minimum=0.004,
        ),
    )
    previous_softening = previous_updater.propose(
        physical_prediction=production_prediction,
        accepted_residual=production_farther_from_rest,
    )
    previous_first_soft_distance = float(
        previous_softening.distance_stiffness[0].item()
    )
    previous_first_soft_shape = float(
        previous_softening.shape_stiffness[0].item()
    )
    previous_updater.reject("previous_softening_probe", previous_softening)
    previous_hardening = previous_updater.propose(
        physical_prediction=production_prediction,
        accepted_residual=production_toward_rest,
    )
    previous_first_hard_distance = float(
        previous_hardening.distance_stiffness[0].item()
    )
    previous_first_hard_shape = float(
        previous_hardening.shape_stiffness[0].item()
    )
    previous_updater.reject("previous_hardening_probe", previous_hardening)

    # Opposite evidence at the two ends of the real one-ring graph must be
    # able to create a soft and a hard region simultaneously.  This keeps the
    # production spatial smoother enabled and exercises the regional use case,
    # rather than only probing uniform global updates.
    regional_distance = torch.full((3,), 0.20)
    regional_shape = torch.full((3,), 0.004)
    regional_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=regional_distance,
        shape_stiffness=regional_shape,
    )
    regional_residual = torch.zeros_like(rest)
    regional_residual[0] = production_farther_from_rest[0]
    regional_residual[2] = production_toward_rest[2]
    regional_soft_floor_update = 0
    regional_hard_ceiling_update = 0
    for update_index in range(1, 41):
        regional_updater.update(
            physical_prediction=production_prediction,
            accepted_residual=regional_residual,
        )
        if regional_soft_floor_update == 0 and torch.isclose(
            regional_distance[0], torch.tensor(0.10)
        ):
            regional_soft_floor_update = update_index
        if regional_hard_ceiling_update == 0 and torch.isclose(
            regional_distance[2], torch.tensor(2.00)
        ):
            regional_hard_ceiling_update = update_index

    reconfigured_distance = torch.tensor((0.08, 0.20, 2.20))
    reconfigured_shape = torch.tensor((0.002, 0.004, 0.025))
    reconfigured_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=reconfigured_distance,
        shape_stiffness=reconfigured_shape,
    )
    reconfigured_updater.signal_ema.fill_(0.5)
    runtime_settings = OnlineTissueStiffnessSettings(
        log_learning_rate=0.12,
        signal_ema_decay=0.60,
        distance_minimum=0.10,
        distance_maximum=2.00,
        shape_minimum=0.003,
        shape_maximum=0.020,
    )
    reconfigured_updater.reconfigure(runtime_settings)

    repeated_distance = torch.full((3,), 0.20)
    repeated_shape = torch.full((3,), 0.004)
    repeated_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=repeated_distance,
        shape_stiffness=repeated_shape,
    )
    repeated_softening: list[dict[str, float]] = []
    distance_updates_to_floor = 0
    shape_updates_to_floor = 0
    for update_index in range(1, 21):
        repeated_updater.update(
            physical_prediction=production_prediction,
            accepted_residual=production_farther_from_rest,
        )
        repeated_softening.append(
            {
                "update": float(update_index),
                "distance": float(repeated_distance[0].item()),
                "shape": float(repeated_shape[0].item()),
            }
        )
        if distance_updates_to_floor == 0 and torch.allclose(
            repeated_distance, torch.full((3,), 0.10)
        ):
            distance_updates_to_floor = update_index
        if shape_updates_to_floor == 0 and torch.allclose(
            repeated_shape, torch.full((3,), 0.003)
        ):
            shape_updates_to_floor = update_index

    repeated_hard_distance = torch.full((3,), 0.20)
    repeated_hard_shape = torch.full((3,), 0.004)
    repeated_hard_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=repeated_hard_distance,
        shape_stiffness=repeated_hard_shape,
    )
    repeated_hardening: list[dict[str, float]] = []
    distance_updates_to_ceiling = 0
    shape_updates_to_ceiling = 0
    for update_index in range(1, 21):
        repeated_hard_updater.update(
            physical_prediction=production_prediction,
            accepted_residual=production_toward_rest,
        )
        repeated_hardening.append(
            {
                "update": float(update_index),
                "distance": float(repeated_hard_distance[0].item()),
                "shape": float(repeated_hard_shape[0].item()),
            }
        )
        if distance_updates_to_ceiling == 0 and torch.allclose(
            repeated_hard_distance, torch.full((3,), 2.00)
        ):
            distance_updates_to_ceiling = update_index
        if shape_updates_to_ceiling == 0 and torch.allclose(
            repeated_hard_shape, torch.full((3,), 0.020)
        ):
            shape_updates_to_ceiling = update_index

    floor_distance = torch.full((3,), 0.10)
    floor_shape = torch.full((3,), 0.003)
    floor_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=fixed,
        edges=edges,
        distance_stiffness=floor_distance,
        shape_stiffness=floor_shape,
        settings=OnlineTissueStiffnessSettings(
            log_learning_rate=0.18,
            signal_ema_decay=0.0,
            spatial_smoothing_iterations=0,
            hardening_bias=0.15,
        ),
    )
    floor_softening = floor_updater.propose(
        physical_prediction=prediction,
        accepted_residual=correction_farther_from_rest,
    )
    lower_bound_saturates = bool(
        torch.equal(
            floor_softening.distance_stiffness,
            torch.full((3,), 0.10),
        )
        and torch.equal(
            floor_softening.shape_stiffness,
            torch.full((3,), 0.003),
        )
        and torch.all(floor_softening.log_step[1:] < 0.0).item()
    )
    floor_updater.reject("lower_bound_probe", floor_softening)
    floor_hardening = floor_updater.propose(
        physical_prediction=prediction,
        accepted_residual=correction_toward_rest,
    )
    lower_bound_can_harden = bool(
        torch.all(floor_hardening.distance_stiffness[1:] > 0.10).item()
        and torch.allclose(
            floor_hardening.shape_stiffness, torch.full((3,), 0.003)
        )
    )
    floor_updater.reject("hardening_probe", floor_hardening)

    # A frozen H=1/3/5 candidate must not block evidence ingestion.  Committing
    # it changes k to the frozen proposal, but keeps the newer EMA for the next
    # proposal instead of rewinding it to the proposal-time snapshot.
    pending_distance = torch.full((3,), 0.20)
    pending_shape = torch.full((3,), 0.004)
    pending_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=pending_distance,
        shape_stiffness=pending_shape,
        settings=OnlineTissueStiffnessSettings(
            signal_ema_decay=0.50,
            spatial_smoothing_iterations=0,
            graph_smoothing_iterations=0,
        ),
    )
    first_evidence = pending_updater.accumulate_evidence(
        physical_prediction=production_prediction,
        accepted_residual=production_toward_rest,
    )
    pending_candidate = pending_updater.propose_from_evidence(first_evidence)
    proposal_ema = pending_candidate.signal_ema.clone()
    pending_updater.accumulate_evidence(
        physical_prediction=production_prediction,
        accepted_residual=production_toward_rest,
    )
    live_pending_ema = pending_updater.signal_ema.clone()
    pending_candidate_stayed_frozen = torch.equal(
        pending_candidate.signal_ema, proposal_ema
    )
    pending_commit_metrics = pending_updater.commit(pending_candidate)
    pending_evidence_survived_commit = torch.equal(
        pending_updater.signal_ema, live_pending_ema
    )

    # Rejection weakens every EMA contributor frozen into the proposal, not
    # only the node active in its last observation.  Evidence observed on a
    # new node during the pending horizon remains untouched.
    local_reject_distance = torch.full((3,), 0.20)
    local_reject_shape = torch.full((3,), 0.004)
    local_reject_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=production_fixed,
        edges=edges,
        distance_stiffness=local_reject_distance,
        shape_stiffness=local_reject_shape,
        settings=OnlineTissueStiffnessSettings(
            signal_ema_decay=0.50,
            rejected_ema_decay=0.50,
            spatial_smoothing_iterations=0,
            graph_smoothing_iterations=0,
        ),
    )
    node_zero_residual = torch.zeros_like(rest)
    node_zero_residual[0] = production_toward_rest[0]
    local_reject_updater.accumulate_evidence(
        physical_prediction=production_prediction,
        accepted_residual=node_zero_residual,
    )
    node_one_residual = torch.zeros_like(rest)
    node_one_residual[1] = production_toward_rest[1]
    local_evidence = local_reject_updater.accumulate_evidence(
        physical_prediction=production_prediction,
        accepted_residual=node_one_residual,
    )
    local_candidate = local_reject_updater.propose_from_evidence(
        local_evidence
    )
    node_two_residual = torch.zeros_like(rest)
    node_two_residual[2] = production_toward_rest[2]
    local_reject_updater.accumulate_evidence(
        physical_prediction=production_prediction,
        accepted_residual=node_two_residual,
    )
    before_local_reject = local_reject_updater.signal_ema.clone()
    local_reject_updater.reject("local_pending_probe", local_candidate)
    after_local_reject = local_reject_updater.signal_ema.clone()

    # Graph regularization acts on k itself, not only on the residual signal.
    # A one-node material spike should have lower log-graph energy while fixed
    # nodes remain exactly untouched.
    graph_fixed = torch.tensor((False, False, False, True))
    graph_rest = torch.tensor(
        (
            (0.000, 0.0, 0.0),
            (0.001, 0.0, 0.0),
            (0.002, 0.0, 0.0),
            (0.003, 0.0, 0.0),
        ),
        dtype=torch.float32,
    )
    graph_edges = torch.tensor(((0, 1), (1, 2), (2, 3)))
    graph_prediction = graph_rest.clone()
    graph_prediction[:, 2] = -0.001
    graph_residual = torch.zeros_like(graph_rest)
    graph_residual[0, 2] = 0.00030

    def graph_candidate(graph_passes: int, graph_blend: float):
        graph_distance = torch.full((4,), 0.20)
        graph_shape = torch.full((4,), 0.004)
        graph_updater = ResidualDrivenPaperStiffnessUpdater(
            rest_positions=graph_rest,
            fixed_mask=graph_fixed,
            edges=graph_edges,
            distance_stiffness=graph_distance,
            shape_stiffness=graph_shape,
            settings=OnlineTissueStiffnessSettings(
                signal_ema_decay=0.0,
                spatial_smoothing_iterations=0,
                graph_smoothing_iterations=graph_passes,
                graph_smoothing_blend=graph_blend,
            ),
        )
        return graph_updater.propose(
            physical_prediction=graph_prediction,
            accepted_residual=graph_residual,
        )

    unsmoothed_graph_candidate = graph_candidate(0, 0.0)
    smoothed_graph_candidate = graph_candidate(2, 0.50)

    # Regression for the production failure seen in the 2026-08-29 direct
    # run: once neighbouring material values are heterogeneous, smoothing the
    # absolute log-k field can move a node far beyond the already-clamped raw
    # gradient.  The final value, not only the input gradient, must stay in the
    # configured per-commit trust region.
    cap_distance = torch.tensor((0.00010, 0.010, 0.010, 0.010))
    cap_shape = torch.full((4,), 0.0005)
    cap_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=graph_rest,
        fixed_mask=torch.zeros(4, dtype=torch.bool),
        edges=graph_edges,
        distance_stiffness=cap_distance,
        shape_stiffness=cap_shape,
        settings=OnlineTissueStiffnessSettings(
            log_learning_rate=0.30,
            maximum_log_step=0.10,
            signal_ema_decay=0.0,
            spatial_smoothing_iterations=0,
            graph_smoothing_iterations=2,
            graph_smoothing_blend=0.50,
            distance_minimum=0.00001,
            distance_maximum=4.0,
            shape_minimum=0.000001,
            shape_maximum=0.04,
        ),
    )
    cap_candidate = cap_updater.propose(
        physical_prediction=graph_prediction,
        accepted_residual=graph_residual,
    )
    cap_actual_distance_step = torch.log(
        cap_candidate.distance_stiffness / cap_distance
    ).abs()
    cap_actual_shape_step = torch.log(
        cap_candidate.shape_stiffness / cap_shape
    ).abs()

    # Two disconnected residual regions must be independently addressable.
    # The stronger component is rank 0; rank 1 changes the other region, and
    # neither candidate is allowed to modify its peer.
    component_rest = torch.tensor(
        tuple((float(index) * 0.001, 0.0, 0.0) for index in range(6)),
        dtype=torch.float32,
    )
    component_edges = torch.tensor(
        ((0, 1), (1, 2), (3, 4), (4, 5)), dtype=torch.long
    )
    component_distance = torch.full((6,), 0.20)
    component_shape = torch.full((6,), 0.004)
    component_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=component_rest,
        fixed_mask=torch.zeros(6, dtype=torch.bool),
        edges=component_edges,
        distance_stiffness=component_distance,
        shape_stiffness=component_shape,
        settings=OnlineTissueStiffnessSettings(
            signal_ema_decay=0.0,
            spatial_smoothing_iterations=0,
            graph_smoothing_iterations=0,
            hardening_bias=0.0,
        ),
    )
    component_prediction = component_rest.clone()
    component_prediction[:, 2] = -0.001
    component_residual = torch.zeros_like(component_rest)
    component_residual[:3, 2] = 0.00030
    component_residual[3:, 2] = 0.00015
    component_proposal = component_updater.propose(
        physical_prediction=component_prediction,
        accepted_residual=component_residual,
    )
    component_zero = component_updater.candidate_variant(
        component_proposal,
        distance_scale=2.0,
        shape_scale=0.0,
        variant_label="component_zero_probe",
        scope="component_0",
    )
    component_one = component_updater.candidate_variant(
        component_proposal,
        distance_scale=2.0,
        shape_scale=0.0,
        variant_label="component_one_probe",
        scope="component_1",
    )
    merged_components = component_updater.merge_candidate_regions(
        (component_zero, component_one),
        variant_label="merged_component_probe",
    )
    component_zero_changed = component_zero.distance_log_step.abs() > 1.0e-10
    component_one_changed = component_one.distance_log_step.abs() > 1.0e-10
    component_updater.reject("component_probe_complete", component_proposal)

    # One proposal now carries two genuinely independent continuous material
    # gradients.  Reducing the square's in-plane stretch asks distance to
    # harden, while the added out-of-plane corner bend asks shape to soften at
    # the same node.  Effective-step normalization must also lift the distance
    # gradient close to the requested 0.08 trust step without exceeding 0.10.
    dual_rest = torch.tensor(
        (
            (0.000, 0.000, 0.0),
            (0.001, 0.000, 0.0),
            (0.000, 0.001, 0.0),
            (0.001, 0.001, 0.0),
        ),
        dtype=torch.float32,
    )
    dual_edges = torch.tensor(
        ((0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)),
        dtype=torch.long,
    )
    dual_prediction = dual_rest.clone()
    dual_prediction[:, 0] *= 1.30
    dual_corrected = dual_prediction.clone()
    dual_corrected[:, 0] = dual_rest[:, 0] * 1.05
    dual_corrected[3, 2] = 0.00050
    dual_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=dual_rest,
        fixed_mask=torch.zeros(4, dtype=torch.bool),
        edges=dual_edges,
        distance_stiffness=torch.full((4,), 0.20),
        shape_stiffness=torch.full((4,), 0.004),
        settings=OnlineTissueStiffnessSettings(
            log_learning_rate=0.10,
            maximum_log_step=0.10,
            effective_log_step_target=0.08,
            maximum_step_amplification=32.0,
            signal_ema_decay=0.0,
            spatial_smoothing_iterations=0,
            graph_smoothing_iterations=0,
            strain_signal_weight=1.0,
            hardening_bias=0.0,
        ),
    )
    dual_evidence = dual_updater.accumulate_evidence(
        physical_prediction=dual_prediction,
        accepted_residual=dual_corrected - dual_prediction,
    )
    dual_candidate = dual_updater.propose_from_evidence(dual_evidence)

    # The aggressive production schedule now applies the same two gradients
    # on alternating successful commits.  Each single-axis candidate must be
    # exactly incapable of changing its peer material family.
    alternating_updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=dual_rest,
        fixed_mask=torch.zeros(4, dtype=torch.bool),
        edges=dual_edges,
        distance_stiffness=torch.full((4,), 0.20),
        shape_stiffness=torch.full((4,), 0.004),
        settings=dual_updater.settings,
    )
    alternating_evidence = alternating_updater.accumulate_evidence(
        physical_prediction=dual_prediction,
        accepted_residual=dual_corrected - dual_prediction,
    )
    distance_axis_candidate = alternating_updater.propose_from_evidence(
        alternating_evidence,
        material_axis="distance",
    )
    distance_axis_shape_unchanged = torch.equal(
        distance_axis_candidate.shape_stiffness,
        alternating_updater.shape_stiffness,
    )
    alternating_updater.commit(distance_axis_candidate)
    distance_after_distance_commit = (
        alternating_updater.distance_stiffness.clone()
    )
    alternating_evidence = alternating_updater.accumulate_evidence(
        physical_prediction=dual_prediction,
        accepted_residual=dual_corrected - dual_prediction,
    )
    shape_axis_candidate = alternating_updater.propose_from_evidence(
        alternating_evidence,
        material_axis="shape",
    )
    shape_axis_distance_unchanged = torch.equal(
        shape_axis_candidate.distance_stiffness,
        distance_after_distance_commit,
    )
    alternating_metrics = alternating_updater.commit(shape_axis_candidate)

    gates = {
        "toward_rest_hardens_distance_without_fabricating_shape": bool(
            torch.all(hardened_distance[1:] > 0.4).item()
            and torch.equal(hardened_shape, torch.full((3,), 0.008))
        ),
        "away_from_rest_softens_distance_after_hardening": bool(
            torch.all(distance[1:] < hardened_distance[1:]).item()
            and torch.equal(shape, hardened_shape)
        ),
        "fixed_node_never_changes": bool(
            torch.isclose(hardened_distance[0], torch.tensor(0.4))
            and torch.isclose(hardened_shape[0], torch.tensor(0.008))
        ),
        "reset_restores_baseline": bool(
            torch.equal(distance, torch.full((3,), 0.4))
            and torch.equal(shape, torch.full((3,), 0.008))
        ),
        "local_bad_tet_masks_only_its_incident_particle": bool(
            quality_gated_distance[1] == 0.4
            and quality_gated_shape[1] == 0.008
            and quality_gated_distance[2] > 0.4
            and quality_gated_shape[2] == 0.008
            and quality_metrics["quality_masked_particles"] == 1
            and quality_metrics["quality_valid_particles"] == 1
        ),
        "large_update_is_bounded": bool(
            hard_metrics["maximum_log_step"] <= 0.18 + 1.0e-8
        ),
        "candidate_does_not_mutate_verified": proposal_does_not_mutate_verified,
        "u_t_exclusion_blocks_smoothing_and_ema": hard_exclusion_blocks_smoothing_and_ema,
        "rejection_preserves_verified": rejection_preserves_verified,
        "selective_ema_invalidation_works": (
            selective_ema_invalidation_works
        ),
        "global_ema_invalidation_works": global_ema_invalidation_works,
        "edge_strain_observability_is_local_and_read_only": bool(
            torch.equal(
                observable_mask,
                torch.tensor((False, True, False)),
            )
            and not bool(excluded_observable_mask.any().item())
            and torch.equal(observability_ema_before, observability_ema_after)
        ),
        "rigid_translation_only_updates_distance": bool(
            first_soft_distance < 0.20
            and first_soft_distance >= 0.10
            and abs(first_soft_shape - 0.004) < 1.0e-8
        ),
        "current_distance_baseline_hardens_more_responsively": bool(
            first_hard_distance > previous_first_hard_distance
            and abs(first_hard_shape - previous_first_hard_shape) < 1.0e-8
        ),
        "previous_policy_reference_is_reproduced": bool(
            abs(previous_first_soft_distance - 0.20) < 1.0e-7
            and abs(previous_first_soft_shape - 0.004) < 1.0e-8
            and previous_first_hard_distance > 0.20
            and previous_first_hard_shape > 0.004
        ),
        "repeated_softening_reaches_distance_floor_only": bool(
            distance_updates_to_floor > 0
            and torch.allclose(
                repeated_distance, torch.full((3,), 0.10)
            )
            and torch.allclose(repeated_shape, torch.full((3,), 0.004))
        ),
        "repeated_hardening_reaches_distance_ceiling_only": bool(
            distance_updates_to_ceiling > 0
            and torch.allclose(
                repeated_hard_distance, torch.full((3,), 2.00)
            )
            and torch.allclose(
                repeated_hard_shape, torch.full((3,), 0.004)
            )
        ),
        "regional_contrast_survives_graph_regularization": bool(
            regional_distance[2] > regional_distance[1]
            and regional_distance[1] > regional_distance[0]
            and torch.allclose(regional_shape, torch.full((3,), 0.004))
            and float(regional_distance[2] / regional_distance[0]) > 1.4
        ),
        "runtime_reconfigure_clips_bounds_and_clears_ema": bool(
            reconfigured_updater.settings == runtime_settings
            and torch.allclose(
                reconfigured_distance, torch.tensor((0.10, 0.20, 2.00))
            )
            and torch.allclose(
                reconfigured_shape, torch.tensor((0.003, 0.004, 0.020))
            )
            and torch.count_nonzero(reconfigured_updater.signal_ema) == 0
        ),
        "new_online_floor_saturates": lower_bound_saturates,
        "new_online_floor_can_still_harden": lower_bound_can_harden,
        "dual_gradients_can_have_opposite_signs_in_one_proposal": bool(
            dual_candidate.distance_gradient_log_step[3] > 0.0
            and dual_candidate.shape_gradient_log_step[3] < 0.0
            and not torch.equal(
                dual_candidate.distance_gradient_log_step,
                dual_candidate.shape_gradient_log_step,
            )
            and dual_candidate.metrics["candidate_variant"]
            == "dual_independent_gradient"
        ),
        "effective_step_target_is_realized_and_bounded": bool(
            dual_candidate.metrics["distance_maximum_log_step"] >= 0.075
            and dual_candidate.metrics["distance_maximum_log_step"] <= 0.10
            and dual_candidate.metrics["shape_maximum_log_step"] >= 0.075
            and dual_candidate.metrics["shape_maximum_log_step"] <= 0.10
        ),
        "zero_hardening_bias_is_the_default": bool(
            OnlineTissueStiffnessSettings().hardening_bias == 0.0
        ),
        "alternating_candidates_are_strictly_axis_isolated": bool(
            distance_axis_shape_unchanged
            and shape_axis_distance_unchanged
            and torch.any(
                distance_axis_candidate.distance_log_step.abs() > 1.0e-10
            ).item()
            and torch.all(
                distance_axis_candidate.shape_log_step == 0.0
            ).item()
            and torch.all(
                shape_axis_candidate.distance_log_step == 0.0
            ).item()
            and torch.any(
                shape_axis_candidate.shape_log_step.abs() > 1.0e-10
            ).item()
            and alternating_metrics["distance_update_count"] == 1
            and alternating_metrics["shape_update_count"] == 1
        ),
        "pending_candidate_keeps_accumulating_fresh_evidence": bool(
            pending_candidate_stayed_frozen
            and torch.all(live_pending_ema > proposal_ema).item()
            and pending_evidence_survived_commit
            and pending_commit_metrics["evidence_updates_while_pending"] == 1
        ),
        "rejection_decays_all_proposal_contributors_only": bool(
            after_local_reject[0] < before_local_reject[0]
            and after_local_reject[1] < before_local_reject[1]
            and after_local_reject[2] == before_local_reject[2]
            and after_local_reject[2] > 0.0
            and local_candidate.metrics["latest_observation_source_particles"]
            == 1
            and local_candidate.metrics["proposal_source_particles"] == 2
        ),
        "stiffness_graph_smoothness_reduces_isolated_spikes": bool(
            smoothed_graph_candidate.metrics[
                "distance_log_graph_energy"
            ]
            < unsmoothed_graph_candidate.metrics[
                "distance_log_graph_energy"
            ]
            and smoothed_graph_candidate.distance_stiffness[1] > 0.20
            and smoothed_graph_candidate.distance_stiffness[3] == 0.20
            and smoothed_graph_candidate.shape_stiffness[3] == 0.004
        ),
        "post_smoothing_realized_step_is_bounded": bool(
            float(cap_actual_distance_step.max()) <= 0.10
            and float(cap_actual_shape_step.max()) <= 0.10
            and cap_candidate.metrics[
                "post_smoothing_pre_cap_maximum_log_step"
            ] > 0.10
            and cap_candidate.metrics[
                "post_smoothing_step_cap_clipped_particles"
            ] > 0
            and cap_candidate.metrics["maximum_log_step"] <= 0.10
        ),
        "spatial_components_receive_independent_material_credit": bool(
            torch.equal(
                component_zero_changed,
                torch.tensor((True, True, True, False, False, False)),
            )
            and torch.equal(
                component_one_changed,
                torch.tensor((False, False, False, True, True, True)),
            )
            and component_zero.metrics["candidate_scope"] == "component_0"
            and component_one.metrics["candidate_scope"] == "component_1"
            and component_zero.metrics["candidate_scope_particles"] == 3
            and component_one.metrics["candidate_scope_particles"] == 3
        ),
        "independently_safe_components_can_merge_without_double_step": bool(
            torch.equal(
                merged_components.distance_log_step,
                torch.where(
                    component_zero.distance_log_step.abs()
                    > component_one.distance_log_step.abs(),
                    component_zero.distance_log_step,
                    component_one.distance_log_step,
                ),
            )
            and torch.all(merged_components.distance_log_step > 0.0).item()
            and torch.equal(
                merged_components.shape_stiffness, component_shape
            )
            and merged_components.metrics["candidate_scope"]
            == "merged_components"
            and merged_components.metrics["merged_candidate_count"] == 2
            and merged_components.metrics["candidate_scope_particles"] == 6
        ),
    }
    report = {
        "stage": "online_visual_residual_paper_stiffness_gate",
        "hardening_metrics": hard_metrics,
        "softening_metrics": soft_metrics,
        "hardened_distance": hardened_distance.tolist(),
        "hardened_shape": hardened_shape.tolist(),
        "quality_gated_distance": quality_gated_distance.tolist(),
        "quality_gated_shape": quality_gated_shape.tolist(),
        "regional_contrast": {
            "distance": regional_distance.tolist(),
            "shape": regional_shape.tolist(),
            "soft_floor_update": regional_soft_floor_update,
            "hard_ceiling_update": regional_hard_ceiling_update,
            "distance_maximum_to_minimum_ratio": float(
                regional_distance.max() / regional_distance.min()
            ),
        },
        "soft_baseline": {
            "initial_distance": 0.20,
            "initial_shape": 0.004,
            "online_distance_floor": 0.10,
            "online_distance_ceiling": 2.00,
            "online_shape_floor": 0.003,
            "first_soft_distance": first_soft_distance,
            "first_soft_shape": first_soft_shape,
            "first_hard_distance": first_hard_distance,
            "first_hard_shape": first_hard_shape,
            "previous_policy": {
                "ema_new_weight": 0.20,
                "first_soft_distance": previous_first_soft_distance,
                "first_soft_shape": previous_first_soft_shape,
                "first_hard_distance": previous_first_hard_distance,
                "first_hard_shape": previous_first_hard_shape,
            },
            "distance_updates_to_floor": distance_updates_to_floor,
            "shape_updates_to_floor": shape_updates_to_floor,
            "repeated_softening": repeated_softening,
            "distance_updates_to_ceiling": distance_updates_to_ceiling,
            "shape_updates_to_ceiling": shape_updates_to_ceiling,
            "repeated_hardening": repeated_hardening,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Online stiffness gate failed")


if __name__ == "__main__":
    main()
