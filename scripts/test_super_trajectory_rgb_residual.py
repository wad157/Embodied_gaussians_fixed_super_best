#!/usr/bin/env python3
"""CPU regression gate for the post-trajectory RGB hold constraint."""

from __future__ import annotations

import torch

from embodied_gaussians.physics_simulator.visual_tissue_residual_mapping import (
    TetrahedralGaussianVisualResidualMapper,
    VisualTissueResidualMappingSettings,
)


def build_mapper() -> tuple[TetrahedralGaussianVisualResidualMapper, torch.Tensor]:
    rest = torch.tensor(
        (
            (0.000, 0.000, 0.000),
            (0.004, 0.000, 0.000),
            (0.000, 0.004, 0.000),
            (0.000, 0.000, 0.004),
        ),
        dtype=torch.float32,
    )
    mapper = TetrahedralGaussianVisualResidualMapper.from_visual_face_centroid_bindings(
        rest_positions=rest,
        tet_indices=torch.tensor(((0, 1, 2, 3),), dtype=torch.long),
        fixed_mask=torch.tensor((True, False, False, False)),
        soft_gaussian_ids=torch.tensor((0,), dtype=torch.long),
        visual_vertex_particle_indices=torch.tensor(
            ((0, 1, 2, 0, 1, 2, 0, 1, 2),), dtype=torch.long
        ),
        visual_vertex_weights=torch.tensor(
            ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),),
            dtype=torch.float32,
        ),
        settings=VisualTissueResidualMappingSettings(
            iterations=16,
            learning_rate_m=4.0e-5,
            image_scale=1.0,
            robust_loss_beta=0.0,
            distance_weight=0.001,
            volume_weight=0.001,
            shape_weight=0.0001,
            spatial_weight=0.0001,
            temporal_weight=0.0,
            magnitude_weight=0.0001,
            maximum_residual_m=0.0008,
        ),
    )
    return mapper, rest


def main() -> None:
    mapper, rest = build_mapper()
    base_means = torch.tensor(((0.0013333333, 0.0013333333, 0.0),))

    def render(means: torch.Tensor) -> torch.Tensor:
        return (0.5 + 120.0 * means[0]).reshape(1, 1, 1, 3)

    target = render(base_means + torch.tensor((0.00035, -0.0002, 0.0001))).detach()
    common = dict(
        physical_positions=rest,
        base_gaussian_means=base_means,
        target_colors=target,
        pixel_weights=torch.ones((1, 1, 1)),
        render_colors=render,
    )
    unconstrained = mapper.solve(**common)
    held = mapper.solve(
        **common,
        trajectory_particle_indices=torch.tensor(((0, 1, 2),)),
        trajectory_particle_weights=torch.tensor(((0.2, 0.3, 0.5),)),
        trajectory_confidence=torch.tensor((0.9,)),
        trajectory_valid_mask=torch.tensor((True,)),
        trajectory_hold_weight=0.02,
    )
    ids = torch.tensor((0, 1, 2))
    weights = torch.tensor((0.2, 0.3, 0.5))[:, None]
    unconstrained_shift = torch.sum(unconstrained.residual[ids] * weights, dim=0)
    held_shift = torch.sum(held.residual[ids] * weights, dim=0)
    assert torch.linalg.vector_norm(held_shift) < torch.linalg.vector_norm(
        unconstrained_shift
    )
    assert held.trajectory_hold_track_count == 1
    assert abs(
        held.trajectory_hold_rms_m
        - float(torch.linalg.vector_norm(held_shift).item())
    ) < 1.0e-8
    assert held.final_visual_loss < held.initial_visual_loss
    print("PASS trajectory hold reduces B_j delta-x while RGB loss improves")


if __name__ == "__main__":
    main()
