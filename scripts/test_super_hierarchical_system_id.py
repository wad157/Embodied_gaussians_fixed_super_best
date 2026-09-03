#!/usr/bin/env python3
"""Deterministic CPU gates for broad global material recovery."""

from __future__ import annotations

import json

import torch

from embodied_gaussians.physics_simulator.online_tissue_stiffness import (
    OnlineTissueStiffnessSettings,
    ResidualDrivenPaperStiffnessUpdater,
)


def main() -> None:
    distance = torch.tensor((0.8, 1.6, 1.6, 1.6), dtype=torch.float32)
    shape = torch.full((4,), 0.020, dtype=torch.float32)
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=torch.zeros((4, 3)),
        fixed_mask=torch.tensor((False, False, False, True)),
        edges=torch.tensor(((0, 1), (1, 2), (2, 3)), dtype=torch.long),
        distance_stiffness=distance,
        shape_stiffness=shape,
        settings=OnlineTissueStiffnessSettings(
            graph_smoothing_iterations=0,
            distance_minimum=0.10,
            distance_maximum=2.00,
            shape_minimum=0.003,
            shape_maximum=0.020,
        ),
    )
    evidence = updater.accumulate_evidence(
        physical_prediction=torch.tensor(
            ((0.001, 0.0, 0.0),) * 4, dtype=torch.float32
        ),
        accepted_residual=torch.tensor(
            ((-0.0001, 0.0, 0.0),) * 4, dtype=torch.float32
        ),
        quality_valid_mask=torch.ones(4, dtype=torch.bool),
        supervision_valid_mask=torch.tensor((True, False, False, False)),
        control_exclusion_mask=torch.tensor((False, False, True, False)),
    )
    base = updater.propose_from_evidence(evidence)
    before = distance.clone()
    global_candidate = updater.global_median_candidate(
        base,
        distance_median=0.20,
        variant_label="global_distance_0.2",
    )
    changed = global_candidate.distance_stiffness
    gates = {
        "proposal_does_not_mutate_verified": torch.equal(distance, before),
        "global_search_updates_visible_and_occluded_material_nodes": bool(
            torch.allclose(changed[:2], torch.tensor((0.2, 0.4)), atol=1e-6)
        ),
        "global_search_preserves_local_log_contrast": bool(
            torch.isclose(changed[1] / changed[0], torch.tensor(2.0))
        ),
        "contact_control_exclusion_is_hard": bool(changed[2] == before[2]),
        "fixed_node_is_hard_excluded": bool(changed[3] == before[3]),
        "global_step_is_not_limited_by_legacy_local_maximum": bool(
            global_candidate.metrics["maximum_log_step"]
            > updater.settings.maximum_log_step
        ),
    }
    report = {"gates": gates, "passed": all(gates.values())}
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
