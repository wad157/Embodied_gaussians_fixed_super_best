#!/usr/bin/env python3
"""CPU regression gates for bounded post-trajectory appearance learning."""

from __future__ import annotations

import torch

from embodied_gaussians.embodied_simulator.gaussians import GaussianState
from embodied_gaussians.embodied_simulator.simulator import (
    EmbodiedGaussiansSimulator,
)
from embodied_gaussians.embodied_simulator.trajectory_appearance import (
    TrajectoryAppearanceSettings,
    refine_trajectory_gaussian_appearance,
)


def toy_visual_loss(
    colors: torch.Tensor, opacities: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    target_color = torch.tensor(
        ((0.58, 0.43, 0.52), (0.46, 0.55, 0.41)), dtype=colors.dtype
    )
    target_opacity = torch.tensor((0.56, 0.44), dtype=opacities.dtype)
    per_gaussian = (colors - target_color).square().mean(dim=-1)
    per_gaussian = per_gaussian + 0.25 * (
        opacities - target_opacity
    ).square()
    camera_losses = torch.stack(
        (per_gaussian.mean(), 1.02 * per_gaussian.mean())
    )
    return camera_losses.mean(), camera_losses


def main() -> None:
    reference_colors = torch.zeros((2, 3), dtype=torch.float32)
    reference_opacities = torch.zeros((2,), dtype=torch.float32)
    settings = TrajectoryAppearanceSettings(
        iterations=3,
        color_learning_rate=0.02,
        opacity_learning_rate=0.005,
        maximum_color_logit_offset=0.08,
        maximum_opacity_logit_offset=0.03,
        color_prior_weight=0.0,
        opacity_prior_weight=0.0,
    )
    colors = reference_colors.clone()
    opacities = reference_opacities.clone()
    first = refine_trajectory_gaussian_appearance(
        current_colors_logits=colors,
        current_opacities_logits=opacities,
        reference_colors_logits=reference_colors,
        reference_opacities_logits=reference_opacities,
        visual_loss=toy_visual_loss,
        settings=settings,
    )
    assert first.accepted
    assert first.accepted_iteration in (1, 2, 3)
    assert first.final_visual_loss < first.initial_visual_loss
    assert first.changed_gaussians == 2

    colors = first.colors_logits
    opacities = first.opacities_logits
    for _ in range(12):
        result = refine_trajectory_gaussian_appearance(
            current_colors_logits=colors,
            current_opacities_logits=opacities,
            reference_colors_logits=reference_colors,
            reference_opacities_logits=reference_opacities,
            visual_loss=toy_visual_loss,
            settings=settings,
        )
        colors = result.colors_logits
        opacities = result.opacities_logits
    assert float(colors.abs().max().item()) <= 0.08 + 1.0e-7
    assert float(opacities.abs().max().item()) <= 0.03 + 1.0e-7

    disabled = refine_trajectory_gaussian_appearance(
        current_colors_logits=colors,
        current_opacities_logits=opacities,
        reference_colors_logits=reference_colors,
        reference_opacities_logits=reference_opacities,
        visual_loss=toy_visual_loss,
        settings=TrajectoryAppearanceSettings(iterations=0),
    )
    assert not disabled.accepted
    assert disabled.completed_iterations == 0
    assert torch.equal(disabled.colors_logits, colors)
    assert torch.equal(disabled.opacities_logits, opacities)
    # The geometry renderer must keep using its frozen reference even after
    # the render appearance is changed by a successful proposal.
    simulator = object.__new__(EmbodiedGaussiansSimulator)
    simulator.gaussian_state = GaussianState(
        means=torch.zeros((2, 3)),
        quats=torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2),
        colors_logits=torch.ones((2, 3)),
        opacities_logits=torch.ones((2,)),
        scale_log=torch.zeros((2, 3)),
    )
    reference_full_colors = torch.zeros((2, 3))
    reference_full_opacities = torch.zeros((2,))
    simulator.set_visual_geometry_appearance_reference(
        reference_full_colors, reference_full_opacities
    )
    geometry_colors, geometry_opacities = simulator._geometry_soft_appearance(
        torch.tensor((0, 1), dtype=torch.long)
    )
    assert torch.equal(geometry_colors, torch.full((2, 3), 0.5))
    assert torch.equal(geometry_opacities, torch.full((2,), 0.5))
    print(
        "PASS bounded color learning and frozen geometry appearance are isolated"
    )


if __name__ == "__main__":
    main()
