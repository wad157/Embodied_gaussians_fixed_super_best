"""Bounded online appearance refinement after a trajectory observation.

The physics/triangle skinning path owns Gaussian position, orientation and
scale.  This module deliberately optimizes only RGB and opacity logits so a
photometric update cannot move a tracked point or detach a Gaussian from its
material frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class TrajectoryAppearanceSettings:
    iterations: int = 3
    color_learning_rate: float = 0.02
    opacity_learning_rate: float = 0.005
    maximum_color_logit_offset: float = 0.20
    maximum_opacity_logit_offset: float = 0.10
    color_prior_weight: float = 2.0e-4
    opacity_prior_weight: float = 2.0e-4
    optimize_opacity: bool = False
    formal_image_scale: float = 0.5
    dssim_weight: float = 0.02
    maximum_camera_relative_regression: float = 5.0e-3
    minimum_visual_improvement: float = 1.0e-8

    def validate(self) -> None:
        if self.iterations < 0:
            raise ValueError("Appearance iterations must be non-negative")
        if self.color_learning_rate <= 0.0:
            raise ValueError("Color learning rate must be positive")
        if self.opacity_learning_rate <= 0.0:
            raise ValueError("Opacity learning rate must be positive")
        if self.maximum_color_logit_offset <= 0.0:
            raise ValueError("Color logit offset cap must be positive")
        if self.maximum_opacity_logit_offset <= 0.0:
            raise ValueError("Opacity logit offset cap must be positive")
        if self.color_prior_weight < 0.0 or self.opacity_prior_weight < 0.0:
            raise ValueError("Appearance prior weights must be non-negative")
        if self.maximum_camera_relative_regression < 0.0:
            raise ValueError("Camera regression tolerance must be non-negative")
        if self.minimum_visual_improvement < 0.0:
            raise ValueError("Minimum visual improvement must be non-negative")
        if not 0.0 < self.formal_image_scale <= 1.0:
            raise ValueError("Formal appearance scale must lie in (0, 1]")
        if self.dssim_weight < 0.0:
            raise ValueError("DSSIM weight must be non-negative")


@dataclass(frozen=True)
class TrajectoryAppearanceResult:
    accepted: bool
    colors_logits: torch.Tensor
    opacities_logits: torch.Tensor
    initial_visual_loss: float
    final_visual_loss: float
    initial_camera_visual_losses: tuple[float, ...]
    final_camera_visual_losses: tuple[float, ...]
    completed_iterations: int
    accepted_iteration: int
    changed_gaussians: int
    maximum_color_logit_offset: float
    maximum_opacity_logit_offset: float


VisualLoss = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def refine_trajectory_gaussian_appearance(
    *,
    current_colors_logits: torch.Tensor,
    current_opacities_logits: torch.Tensor,
    reference_colors_logits: torch.Tensor,
    reference_opacities_logits: torch.Tensor,
    visual_loss: VisualLoss,
    settings: TrajectoryAppearanceSettings,
) -> TrajectoryAppearanceResult:
    """Run a few Adam steps and return the best camera-safe appearance.

    Offsets are clamped relative to a fixed reference, not the previous frame.
    Consequently many accepted video-frame updates cannot accumulate into an
    unbounded color/opacity drift.
    """

    settings.validate()
    if current_colors_logits.shape != reference_colors_logits.shape:
        raise ValueError("Current and reference color logits must match")
    if current_opacities_logits.shape != reference_opacities_logits.shape:
        raise ValueError("Current and reference opacity logits must match")
    if current_colors_logits.shape[:-1] != current_opacities_logits.shape:
        raise ValueError("Color and opacity Gaussian counts must match")

    initial_colors = current_colors_logits.detach().clone()
    initial_opacities = current_opacities_logits.detach().clone()
    reference_colors = reference_colors_logits.detach()
    reference_opacities = reference_opacities_logits.detach()
    with torch.no_grad():
        initial_loss_tensor, initial_camera_tensor = visual_loss(
            initial_colors.sigmoid(), initial_opacities.sigmoid()
        )
    initial_loss = float(initial_loss_tensor.item())
    initial_camera = tuple(
        float(value) for value in initial_camera_tensor.detach().cpu().tolist()
    )
    if settings.iterations == 0:
        return TrajectoryAppearanceResult(
            accepted=False,
            colors_logits=initial_colors,
            opacities_logits=initial_opacities,
            initial_visual_loss=initial_loss,
            final_visual_loss=initial_loss,
            initial_camera_visual_losses=initial_camera,
            final_camera_visual_losses=initial_camera,
            completed_iterations=0,
            accepted_iteration=0,
            changed_gaussians=0,
            maximum_color_logit_offset=float(
                (initial_colors - reference_colors).abs().max().item()
            ),
            maximum_opacity_logit_offset=float(
                (initial_opacities - reference_opacities).abs().max().item()
            ),
        )

    colors = initial_colors.clone().requires_grad_(True)
    opacities = initial_opacities.clone().requires_grad_(
        settings.optimize_opacity
    )
    parameter_groups = [
        {"params": (colors,), "lr": settings.color_learning_rate}
    ]
    if settings.optimize_opacity:
        parameter_groups.append(
            {"params": (opacities,), "lr": settings.opacity_learning_rate}
        )
    optimizer = torch.optim.Adam(parameter_groups)
    best_colors = initial_colors
    best_opacities = initial_opacities
    best_loss = initial_loss
    best_camera = initial_camera
    accepted_iteration = 0
    completed_iterations = 0

    for iteration in range(1, settings.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        rendered_loss, _ = visual_loss(colors.sigmoid(), opacities.sigmoid())
        color_prior = (
            (colors - reference_colors)
            / settings.maximum_color_logit_offset
        ).square().mean()
        opacity_prior = (
            (opacities - reference_opacities)
            / settings.maximum_opacity_logit_offset
        ).square().mean()
        total_loss = (
            rendered_loss
            + settings.color_prior_weight * color_prior
            + settings.opacity_prior_weight * opacity_prior
        )
        if not bool(torch.isfinite(total_loss).item()):
            break
        total_loss.backward()
        if colors.grad is None or (
            settings.optimize_opacity and opacities.grad is None
        ):
            break
        gradients_finite = bool(torch.isfinite(colors.grad).all().item())
        if settings.optimize_opacity:
            assert opacities.grad is not None
            gradients_finite = bool(
                gradients_finite
                and torch.isfinite(opacities.grad).all().item()
            )
        if not gradients_finite:
            break
        optimized_tensors = (
            (colors, opacities)
            if settings.optimize_opacity
            else (colors,)
        )
        torch.nn.utils.clip_grad_norm_(optimized_tensors, max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            colors.clamp_(
                reference_colors - settings.maximum_color_logit_offset,
                reference_colors + settings.maximum_color_logit_offset,
            )
            if settings.optimize_opacity:
                opacities.clamp_(
                    reference_opacities
                    - settings.maximum_opacity_logit_offset,
                    reference_opacities
                    + settings.maximum_opacity_logit_offset,
                )
            else:
                opacities.copy_(initial_opacities)
            colors.clamp_(-12.0, 12.0)
            opacities.clamp_(-12.0, 12.0)
            candidate_loss_tensor, candidate_camera_tensor = visual_loss(
                colors.sigmoid(), opacities.sigmoid()
            )
        completed_iterations = iteration
        candidate_loss = float(candidate_loss_tensor.item())
        candidate_camera = tuple(
            float(value)
            for value in candidate_camera_tensor.detach().cpu().tolist()
        )
        camera_safe = all(
            final
            <= initial
            + max(
                settings.minimum_visual_improvement,
                settings.maximum_camera_relative_regression * initial,
            )
            for initial, final in zip(initial_camera, candidate_camera)
        )
        if (
            camera_safe
            and candidate_loss
            < best_loss - settings.minimum_visual_improvement
        ):
            best_loss = candidate_loss
            best_camera = candidate_camera
            best_colors = colors.detach().clone()
            best_opacities = opacities.detach().clone()
            accepted_iteration = iteration

    accepted = accepted_iteration > 0
    changed = torch.linalg.vector_norm(
        best_colors.sigmoid() - initial_colors.sigmoid(), dim=-1
    ) > 1.0e-7
    changed |= (
        best_opacities.sigmoid() - initial_opacities.sigmoid()
    ).abs() > 1.0e-7
    return TrajectoryAppearanceResult(
        accepted=accepted,
        colors_logits=best_colors,
        opacities_logits=best_opacities,
        initial_visual_loss=initial_loss,
        final_visual_loss=best_loss,
        initial_camera_visual_losses=initial_camera,
        final_camera_visual_losses=best_camera,
        completed_iterations=completed_iterations,
        accepted_iteration=accepted_iteration,
        changed_gaussians=int(torch.count_nonzero(changed).item()),
        maximum_color_logit_offset=float(
            (best_colors - reference_colors).abs().max().item()
        ),
        maximum_opacity_logit_offset=float(
            (best_opacities - reference_opacities).abs().max().item()
        ),
    )
