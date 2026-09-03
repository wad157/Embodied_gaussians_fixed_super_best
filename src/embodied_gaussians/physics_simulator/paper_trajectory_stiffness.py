"""Liang-style online material identification from tracked 3D targets.

The production SUPER solver is implemented in Warp and includes contact and
grip kernels that deliberately disable reverse-mode differentiation.  This
module therefore estimates the gradient of the *actual* XPBD counterfactual
loss with simultaneous central perturbations (SPSA), then applies an ordinary
Adam update to bounded per-particle logits.  Unlike the legacy residual-sign
controller, the direction is determined only by whether a perturbed material
field lowers the replayed trajectory/history objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PaperTrajectoryAdamSettings:
    """Settings for one observation-rate bounded Adam material update."""

    learning_rate: float = 0.10
    maximum_material_log_step: float = 0.10
    beta1: float = 0.90
    beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    parameter_perturbation: float = 0.05
    track_robust_scale_m: float = 0.003
    history_scale_m: float = 0.001
    history_weight: float = 0.10
    distance_smooth_weight: float = 0.001
    shape_smooth_weight: float = 0.001
    minimum_axis_loss_difference: float = 1.0e-7
    distance_minimum: float = 1.0e-5
    distance_maximum: float = 4.0
    shape_minimum: float = 1.0e-6
    shape_maximum: float = 0.04
    # The late SIM system-identification experiments showed that one scalar
    # distance multiplier plus global velocity damping is substantially more
    # identifiable than independent distance/shape values at every particle.
    # Keep the legacy local SPSA path available for old experiments, while the
    # formal SUPER runner opts into this causal low-dimensional mode.
    sim_global_causal_mode: bool = False
    causal_minimum_window_size: int = 2
    causal_window_size: int = 3
    # Robust H3 may span four consecutive physical transitions while using
    # exactly three supervised endpoints.  The missing transition is still
    # replayed, so elapsed time and tool commands are never compressed.
    causal_maximum_window_size: int = 3
    causal_maximum_missing_observations: int = 0
    causal_horizon_weights: tuple[float, float, float] = (1.5, 2.0, 3.0)
    global_maximum_log_offset: float = 0.35
    # H2 is the longest legal reconstruction window between 7:1 held-out
    # frames.  It carries less temporal evidence than H3, so it must remain in
    # a tighter cumulative trust region and be confirmed by a different
    # causal block before it may alter the live material state.
    h2_maximum_log_offset: float = 0.10
    h2_cross_block_cosine_minimum: float = 0.95
    # Reconstruction 7:1 can opt out of H2 entirely once dense, per-frame
    # causal observations are available.  The Future protocol deliberately
    # leaves this enabled so its already-validated v9 path is unchanged.
    h2_updates_enabled: bool = True
    # A reconstruction-only mode consumes one H3 at the end of each disjoint
    # seven-frame training block.  Scheduling is performed by the runtime;
    # this flag is recorded here so the optimizer policy is reproducible.
    reconstruction_nonoverlapping_h3: bool = False
    # When enabled, an H3 proposal must agree with and descend on its last-two
    # transition sub-window.  None keeps the validated Future policy intact.
    h3_tail_cosine_minimum: float | None = None
    velocity_damping_initial_per_second: float = 12.0
    velocity_damping_minimum_per_second: float = 2.0
    velocity_damping_maximum_per_second: float = 30.0
    global_parameter_prior_weight: float = 0.40
    gradient_cosine_minimum: float = 0.95
    region_balance_weight: float = 0.10
    tail_region_weight: float = 0.05
    tail_region_fraction: float = 0.25
    # Optional residual local field applied after a successful global H3
    # update.  Only distance stiffness is locally varied; shape remains fixed
    # to avoid the distance/shape ambiguity seen in earlier SUPER runs.
    local_distance_enabled: bool = False
    local_distance_maximum_log_step: float = 0.008
    local_distance_maximum_log_offset: float = 0.04
    local_distance_minimum_loss_improvement: float = 1.0e-6
    local_distance_smoothing_iterations: int = 2
    local_distance_smoothing_blend: float = 0.35

    def validate(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("Paper Adam learning rate must be positive")
        if self.maximum_material_log_step <= 0.0:
            raise ValueError("Paper Adam material log-step cap must be positive")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("Paper Adam beta values must lie in [0,1)")
        if self.adam_epsilon <= 0.0:
            raise ValueError("Paper Adam epsilon must be positive")
        if self.parameter_perturbation <= 0.0:
            raise ValueError("Paper Adam perturbation must be positive")
        if self.track_robust_scale_m <= 0.0 or self.history_scale_m <= 0.0:
            raise ValueError("Paper stiffness loss scales must be positive")
        if self.history_weight < 0.0:
            raise ValueError("Paper stiffness history weight cannot be negative")
        if self.distance_smooth_weight < 0.0 or self.shape_smooth_weight < 0.0:
            raise ValueError("Paper stiffness smooth weights cannot be negative")
        if self.minimum_axis_loss_difference < 0.0:
            raise ValueError("Minimum loss difference cannot be negative")
        if not 0.0 < self.distance_minimum < self.distance_maximum:
            raise ValueError("Paper distance bounds are invalid")
        if not 0.0 < self.shape_minimum < self.shape_maximum:
            raise ValueError("Paper shape bounds are invalid")
        if not (
            2
            <= self.causal_minimum_window_size
            <= self.causal_window_size
            == 3
        ):
            raise ValueError(
                "SIM-global material ID requires an available H2..H3 window"
            )
        if self.causal_maximum_window_size < self.causal_window_size:
            raise ValueError("Maximum causal window cannot be shorter than H3")
        if self.causal_maximum_missing_observations < 0:
            raise ValueError("Missing-observation allowance cannot be negative")
        if (
            self.causal_maximum_window_size - self.causal_window_size
            < self.causal_maximum_missing_observations
        ):
            raise ValueError("Causal span is too short for the missing observations")
        if (
            len(self.causal_horizon_weights) != self.causal_window_size
            or any(weight <= 0.0 for weight in self.causal_horizon_weights)
        ):
            raise ValueError("Causal horizon weights must contain three positives")
        if self.global_maximum_log_offset <= 0.0:
            raise ValueError("Global material log-offset bound must be positive")
        if not (
            0.0
            < self.h2_maximum_log_offset
            <= self.global_maximum_log_offset
        ):
            raise ValueError("H2 log-offset bound must lie inside the global bound")
        if not -1.0 <= self.h2_cross_block_cosine_minimum <= 1.0:
            raise ValueError("H2 cross-block cosine threshold must lie in [-1,1]")
        if (
            self.h3_tail_cosine_minimum is not None
            and not -1.0 <= self.h3_tail_cosine_minimum <= 1.0
        ):
            raise ValueError("H3 tail cosine threshold must lie in [-1,1]")
        if not (
            0.0
            < self.velocity_damping_minimum_per_second
            <= self.velocity_damping_initial_per_second
            <= self.velocity_damping_maximum_per_second
        ):
            raise ValueError("Velocity-damping bounds do not contain the initial value")
        if self.global_parameter_prior_weight < 0.0:
            raise ValueError("Global parameter prior weight cannot be negative")
        if not -1.0 <= self.gradient_cosine_minimum <= 1.0:
            raise ValueError("Gradient cosine threshold must lie in [-1,1]")
        if min(self.region_balance_weight, self.tail_region_weight) < 0.0:
            raise ValueError("Regional trajectory weights cannot be negative")
        if self.region_balance_weight + self.tail_region_weight > 1.0:
            raise ValueError("Regional trajectory weights cannot exceed one")
        if not 0.0 < self.tail_region_fraction <= 1.0:
            raise ValueError("Tail-region fraction must lie in (0,1]")
        if self.local_distance_maximum_log_step <= 0.0:
            raise ValueError("Local distance log step must be positive")
        if self.local_distance_maximum_log_offset <= 0.0:
            raise ValueError("Local distance log offset must be positive")
        if self.local_distance_minimum_loss_improvement < 0.0:
            raise ValueError("Local distance improvement floor cannot be negative")
        if self.local_distance_smoothing_iterations < 0:
            raise ValueError("Local distance smoothing iterations cannot be negative")
        if not 0.0 <= self.local_distance_smoothing_blend <= 1.0:
            raise ValueError("Local distance smoothing blend must lie in [0,1]")


@dataclass(frozen=True)
class PaperTrajectoryMaterialTrial:
    """One absolute material field used by an isolated XPBD replay."""

    label: str
    axis: str
    sign: int
    distance_stiffness: torch.Tensor
    shape_stiffness: torch.Tensor
    velocity_damping_per_second: float | None = None
    global_log_coefficients: tuple[float, float] | None = None


class PaperTrajectoryAdamOptimizer:
    """Bounded per-node Adam driven by exact counterfactual XPBD losses.

    The simulator supplies four scalar losses per observation: positive and
    negative perturbations for distance and shape.  A deterministic Rademacher
    probe gives an unbiased simultaneous estimate of every active logit's
    gradient while requiring four replays rather than two replays per node.
    """

    def __init__(
        self,
        *,
        distance_stiffness: torch.Tensor,
        shape_stiffness: torch.Tensor,
        fixed_mask: torch.Tensor,
        edges: torch.Tensor,
        settings: PaperTrajectoryAdamSettings | None = None,
    ) -> None:
        self.settings = settings or PaperTrajectoryAdamSettings()
        self.settings.validate()
        self.distance_stiffness = distance_stiffness
        self.shape_stiffness = shape_stiffness
        self.fixed_mask = fixed_mask.detach().to(
            device=distance_stiffness.device, dtype=torch.bool
        )
        self.edges = edges.detach().to(
            device=distance_stiffness.device, dtype=torch.long
        )
        if distance_stiffness.shape != shape_stiffness.shape:
            raise ValueError("Paper Adam material fields disagree")
        if self.fixed_mask.shape != distance_stiffness.shape:
            raise ValueError("Paper Adam fixed mask has the wrong shape")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("Paper Adam edges must have shape [E,2]")
        self.distance_theta = self._material_to_theta(
            distance_stiffness.detach(),
            self.settings.distance_minimum,
            self.settings.distance_maximum,
        )
        self.shape_theta = self._material_to_theta(
            shape_stiffness.detach(),
            self.settings.shape_minimum,
            self.settings.shape_maximum,
        )
        self.distance_first_moment = torch.zeros_like(self.distance_theta)
        self.distance_second_moment = torch.zeros_like(self.distance_theta)
        self.shape_first_moment = torch.zeros_like(self.shape_theta)
        self.shape_second_moment = torch.zeros_like(self.shape_theta)
        self.observation_count = 0
        self.update_count = 0
        self.distance_update_count = 0
        self.shape_update_count = 0
        self.loss_evaluation_count = 0
        self.last_metrics: dict[str, float | int | str] | None = None
        self._distance_probe = torch.zeros_like(self.distance_theta)
        self._shape_probe = torch.zeros_like(self.shape_theta)
        self._active_mask = torch.zeros_like(self.fixed_mask)
        self._global_material_mask = ~self.fixed_mask
        self._global_initial_distance = distance_stiffness.detach().clone()
        self._global_initial_shape = shape_stiffness.detach().clone()
        self.global_log_coefficients = torch.zeros(
            2, dtype=torch.float32, device=distance_stiffness.device
        )
        self.global_first_moment = torch.zeros_like(self.global_log_coefficients)
        self.global_second_moment = torch.zeros_like(self.global_log_coefficients)
        self.global_adam_step = 0
        self.h2_confirmation_gradient: torch.Tensor | None = None
        self.h2_confirmation_block_index: int | None = None
        self.local_distance_log_offsets = torch.zeros_like(
            self._global_initial_distance, dtype=torch.float32
        )
        self.local_distance_update_count = 0
        self._pending_local_distance_offsets: dict[str, torch.Tensor] = {}

    @staticmethod
    def _material_to_theta(
        material: torch.Tensor, minimum: float, maximum: float
    ) -> torch.Tensor:
        unit = ((material.float() - minimum) / (maximum - minimum)).clamp(
            1.0e-6, 1.0 - 1.0e-6
        )
        return torch.log(unit) - torch.log1p(-unit)

    @staticmethod
    def _theta_to_material(
        theta: torch.Tensor, minimum: float, maximum: float
    ) -> torch.Tensor:
        return minimum + (maximum - minimum) * torch.sigmoid(theta)

    def current_material(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.settings.sim_global_causal_mode:
            distance = self._global_initial_distance.detach().clone()
            distance[self._global_material_mask] = torch.clamp(
                self._global_initial_distance[self._global_material_mask]
                * torch.exp(
                    self.global_log_coefficients[0]
                    + self.local_distance_log_offsets[
                        self._global_material_mask
                    ]
                ),
                min=self.settings.distance_minimum,
                max=self.settings.distance_maximum,
            )
            # Shape is intentionally frozen. SIM found that estimating shape
            # from the same image trajectory made distance/shape confounded.
            return distance, self._global_initial_shape.detach().clone()
        return (
            self._theta_to_material(
                self.distance_theta,
                self.settings.distance_minimum,
                self.settings.distance_maximum,
            ),
            self._theta_to_material(
                self.shape_theta,
                self.settings.shape_minimum,
                self.settings.shape_maximum,
            ),
        )

    def reset_from_material(self) -> None:
        """Resynchronise logits and clear Adam state after a scene reset."""
        self.distance_theta.copy_(
            self._material_to_theta(
                self.distance_stiffness.detach(),
                self.settings.distance_minimum,
                self.settings.distance_maximum,
            )
        )
        self.shape_theta.copy_(
            self._material_to_theta(
                self.shape_stiffness.detach(),
                self.settings.shape_minimum,
                self.settings.shape_maximum,
            )
        )
        self.distance_first_moment.zero_()
        self.distance_second_moment.zero_()
        self.shape_first_moment.zero_()
        self.shape_second_moment.zero_()
        self.observation_count = 0
        self.update_count = 0
        self.distance_update_count = 0
        self.shape_update_count = 0
        self.loss_evaluation_count = 0
        self.last_metrics = None
        self._global_initial_distance.copy_(self.distance_stiffness.detach())
        self._global_initial_shape.copy_(self.shape_stiffness.detach())
        self.global_log_coefficients.zero_()
        self.global_first_moment.zero_()
        self.global_second_moment.zero_()
        self.global_adam_step = 0
        self.h2_confirmation_gradient = None
        self.h2_confirmation_block_index = None
        self.local_distance_log_offsets.zero_()
        self.local_distance_update_count = 0
        self._pending_local_distance_offsets.clear()

    def current_velocity_damping_per_second(self) -> float:
        settings = self.settings
        value = settings.velocity_damping_initial_per_second * math.exp(
            float(self.global_log_coefficients[1].item())
        )
        return float(
            min(
                settings.velocity_damping_maximum_per_second,
                max(settings.velocity_damping_minimum_per_second, value),
            )
        )

    def _begin_sim_global_observation(
        self, active: torch.Tensor
    ) -> tuple[PaperTrajectoryMaterialTrial, ...]:
        settings = self.settings
        perturbation = settings.parameter_perturbation
        trials: list[PaperTrajectoryMaterialTrial] = []
        for axis_index, axis in enumerate(("distance", "damping")):
            for sign in (1, -1):
                coefficients = self.global_log_coefficients.detach().clone()
                coefficients[axis_index] += float(sign) * perturbation
                coefficients[0].clamp_(
                    min=-settings.global_maximum_log_offset,
                    max=settings.global_maximum_log_offset,
                )
                damping_minimum_log = math.log(
                    settings.velocity_damping_minimum_per_second
                    / settings.velocity_damping_initial_per_second
                )
                damping_maximum_log = math.log(
                    settings.velocity_damping_maximum_per_second
                    / settings.velocity_damping_initial_per_second
                )
                coefficients[1].clamp_(
                    min=damping_minimum_log, max=damping_maximum_log
                )
                distance = self._global_initial_distance.detach().clone()
                distance[self._global_material_mask] = torch.clamp(
                    self._global_initial_distance[self._global_material_mask]
                    * torch.exp(
                        coefficients[0]
                        + self.local_distance_log_offsets[
                            self._global_material_mask
                        ]
                    ),
                    min=settings.distance_minimum,
                    max=settings.distance_maximum,
                )
                damping = settings.velocity_damping_initial_per_second * math.exp(
                    float(coefficients[1].item())
                )
                trials.append(
                    PaperTrajectoryMaterialTrial(
                        label=f"{axis}_{'plus' if sign > 0 else 'minus'}",
                        axis=axis,
                        sign=sign,
                        distance_stiffness=distance,
                        shape_stiffness=self._global_initial_shape.detach().clone(),
                        velocity_damping_per_second=float(damping),
                        global_log_coefficients=(
                            float(coefficients[0].item()),
                            float(coefficients[1].item()),
                        ),
                    )
                )
        return tuple(trials)

    def prepare_local_distance_direction(
        self,
        raw_signal: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build one smooth, bounded and exactly zero-mean local axis."""

        signal = raw_signal.detach().to(
            device=self.distance_stiffness.device, dtype=torch.float32
        )
        support = support_mask.detach().to(
            device=self.distance_stiffness.device, dtype=torch.bool
        ) & self._global_material_mask
        if signal.shape != self.distance_stiffness.shape:
            raise ValueError("Local distance signal has the wrong shape")
        if support.shape != self.distance_stiffness.shape:
            raise ValueError("Local distance support has the wrong shape")
        current = signal.masked_fill(~support, 0.0)
        source = self.edges[:, 0]
        target = self.edges[:, 1]
        for _ in range(self.settings.local_distance_smoothing_iterations):
            neighbor_hits = torch.zeros_like(current)
            neighbor_hits.index_add_(
                0, source, support[target].to(dtype=current.dtype)
            )
            neighbor_hits.index_add_(
                0, target, support[source].to(dtype=current.dtype)
            )
            expanded = (
                support | (neighbor_hits > 0.0)
            ) & self._global_material_mask
            sums = torch.zeros_like(current)
            counts = torch.zeros_like(current)
            valid_edges = expanded[source] & expanded[target]
            edge_source = source[valid_edges]
            edge_target = target[valid_edges]
            if edge_source.numel():
                sums.index_add_(0, edge_source, current[edge_target])
                sums.index_add_(0, edge_target, current[edge_source])
                ones = torch.ones_like(edge_source, dtype=current.dtype)
                counts.index_add_(0, edge_source, ones)
                counts.index_add_(0, edge_target, ones)
            neighbor_average = sums / counts.clamp_min(1.0)
            current = torch.lerp(
                current,
                neighbor_average,
                self.settings.local_distance_smoothing_blend,
            ).masked_fill(~expanded, 0.0)
            support = expanded
        if int(torch.count_nonzero(support).item()) < 2:
            return torch.zeros_like(current)
        current[support] -= current[support].mean()
        maximum = current.abs().max()
        if float(maximum.item()) <= 1.0e-12:
            return torch.zeros_like(current)
        current /= maximum
        current.masked_fill_(~support, 0.0)
        return current

    def _project_local_distance_offsets(
        self, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Keep the local log field bounded with no global-mean leakage."""

        projected = offsets.detach().clone().to(
            device=self.distance_stiffness.device, dtype=torch.float32
        )
        projected.masked_fill_(~self._global_material_mask, 0.0)
        active = projected[self._global_material_mask]
        active -= active.mean()
        maximum = active.abs().max()
        limit = self.settings.local_distance_maximum_log_offset
        if float(maximum.item()) > limit:
            active *= limit / maximum
        projected[self._global_material_mask] = active
        return projected

    def _limit_local_distance_step(
        self,
        previous: torch.Tensor,
        proposed: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce the trust radius after every zero-mean projection."""

        effective = proposed - previous
        maximum = effective.abs().max()
        limit = self.settings.local_distance_maximum_log_step
        if float(maximum.item()) > limit:
            # Both endpoints already have zero mean and lie inside the
            # cumulative box. Their convex interpolation preserves both.
            proposed = previous + effective * (limit / maximum)
        return proposed

    def begin_local_distance_observation(
        self, direction: torch.Tensor
    ) -> tuple[PaperTrajectoryMaterialTrial, ...]:
        """Create global-only and signed local-distance H3 trials."""

        if not self.settings.local_distance_enabled:
            return ()
        direction = direction.detach().to(
            device=self.distance_stiffness.device, dtype=torch.float32
        )
        if direction.shape != self.distance_stiffness.shape:
            raise ValueError("Local distance direction has the wrong shape")
        if float(direction.abs().max().item()) <= 1.0e-12:
            return ()
        base = self.local_distance_log_offsets.detach().clone()
        step = self.settings.local_distance_maximum_log_step
        offsets = {
            "local_base": base,
            "local_plus": self._limit_local_distance_step(
                base,
                self._project_local_distance_offsets(
                    base + step * direction
                ),
            ),
            "local_minus": self._limit_local_distance_step(
                base,
                self._project_local_distance_offsets(
                    base - step * direction
                ),
            ),
        }
        self._pending_local_distance_offsets = offsets
        trials: list[PaperTrajectoryMaterialTrial] = []
        for label in ("local_base", "local_plus", "local_minus"):
            local = offsets[label]
            distance = self._global_initial_distance.detach().clone()
            distance[self._global_material_mask] = torch.clamp(
                self._global_initial_distance[self._global_material_mask]
                * torch.exp(
                    self.global_log_coefficients[0]
                    + local[self._global_material_mask]
                ),
                min=self.settings.distance_minimum,
                max=self.settings.distance_maximum,
            )
            trials.append(
                PaperTrajectoryMaterialTrial(
                    label=label,
                    axis="local_distance",
                    sign=(1 if label == "local_plus" else -1 if label == "local_minus" else 0),
                    distance_stiffness=distance,
                    shape_stiffness=self._global_initial_shape.detach().clone(),
                    velocity_damping_per_second=(
                        self.current_velocity_damping_per_second()
                    ),
                    global_log_coefficients=(
                        float(self.global_log_coefficients[0].item()),
                        float(self.global_log_coefficients[1].item()),
                    ),
                )
            )
        self.loss_evaluation_count += len(trials)
        return tuple(trials)

    def finish_local_distance_observation(
        self, losses: dict[str, float]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
        """Accept a signed local field only when it beats global-only H3."""

        required = {"local_base", "local_plus", "local_minus"}
        if set(losses) != required or set(self._pending_local_distance_offsets) != required:
            raise ValueError("Local distance H3 requires base/plus/minus trials")
        if not all(math.isfinite(float(value)) for value in losses.values()):
            raise ValueError("Local distance H3 received a non-finite loss")
        base_loss = float(losses["local_base"])
        label = min(("local_plus", "local_minus"), key=lambda key: losses[key])
        candidate_loss = float(losses[label])
        improvement = base_loss - candidate_loss
        accepted = bool(
            improvement
            > self.settings.local_distance_minimum_loss_improvement
        )
        previous = self.local_distance_log_offsets.detach().clone()
        if accepted:
            self.local_distance_log_offsets.copy_(
                self._pending_local_distance_offsets[label]
            )
            self.local_distance_update_count += 1
        actual_step = self.local_distance_log_offsets - previous
        self._pending_local_distance_offsets.clear()
        distance, shape = self.current_material()
        active = self.local_distance_log_offsets[self._global_material_mask]
        metrics: dict[str, float | int | str] = {
            "local_distance_status": (
                "accepted" if accepted else "no_h3_improvement"
            ),
            "local_distance_selected_trial": label,
            "local_distance_base_loss": base_loss,
            "local_distance_candidate_loss": candidate_loss,
            "local_distance_loss_improvement": float(improvement),
            "local_distance_update_count": int(
                self.local_distance_update_count
            ),
            "local_distance_step_maximum": float(
                actual_step.abs().max().item()
            ),
            "local_distance_log_minimum": float(active.min().item()),
            "local_distance_log_mean": float(active.mean().item()),
            "local_distance_log_maximum": float(active.max().item()),
            "local_distance_global_mean_preserved": int(
                abs(float(active.mean().item())) <= 1.0e-7
            ),
        }
        return distance.detach(), shape.detach(), metrics

    def _probe(self, *, axis_salt: int, active_mask: torch.Tensor) -> torch.Tensor:
        """Return a deterministic frame-varying Rademacher graph probe."""
        particle_ids = torch.arange(
            len(active_mask), dtype=torch.int64, device=active_mask.device
        )
        # Integer hashing avoids a CUDA RNG dependency and makes a completed
        # evaluation exactly reproducible from its observation index.
        hashed = (
            particle_ids * 1103515245
            + (self.observation_count + 1) * 2654435761
            + int(axis_salt) * 2246822519
        )
        probe = torch.where(
            torch.bitwise_and(hashed, 1) == 0,
            torch.ones_like(self.distance_theta),
            -torch.ones_like(self.distance_theta),
        )
        return probe.masked_fill(~active_mask, 0.0)

    def _activate_observation(self, active_mask: torch.Tensor) -> torch.Tensor:
        active = active_mask.detach().to(
            device=self.distance_theta.device, dtype=torch.bool
        ) & ~self.fixed_mask
        if active.shape != self.fixed_mask.shape:
            raise ValueError("Paper Adam active mask has the wrong shape")
        self.observation_count += 1
        self._active_mask = active
        return active

    def record_causal_warmup_observation(
        self,
        active_mask: torch.Tensor,
        *,
        horizon_count: int,
        status: str = "causal_window_warmup",
    ) -> dict[str, float | int | str]:
        """Record an H1 observation without running any counterfactual.

        A single transition cannot pass the configured H2..H3 identification
        protocol.  Constructing four material trials is cheap, but replaying
        them is both wasteful and unsafe: a nominal no-op must not perturb the
        live contact state while the causal window is warming up.
        """
        if not self.settings.sim_global_causal_mode:
            raise RuntimeError("Causal warmup is available only in SIM-global mode")
        active = self._activate_observation(active_mask)
        distance, shape = self.current_material()
        weights = self.settings.causal_horizon_weights[: int(horizon_count)]
        metrics: dict[str, float | int | str] = {
            "status": str(status),
            "observation_count": self.observation_count,
            "update_count": self.update_count,
            "distance_update_count": self.distance_update_count,
            "shape_update_count": self.shape_update_count,
            "loss_evaluation_count": self.loss_evaluation_count,
            "active_particles": int(torch.count_nonzero(active).item()),
            "optimizer": "sim_global_causal_adam",
            "gradient_estimator": "not_evaluated_during_h1_warmup",
            "causal_window_size": int(horizon_count),
            "causal_minimum_window_size": int(
                self.settings.causal_minimum_window_size
            ),
            "available_horizon_weights": ",".join(
                f"{value:g}" for value in weights
            ),
            "warmup_counterfactual_replays": 0,
            "gradient_cosine": -1.0,
            "gradient_direction_consistent": 0,
            "descent_direction": 0,
            "distance_log_step": 0.0,
            "damping_log_step": 0.0,
            "global_distance_log_offset": float(
                self.global_log_coefficients[0].item()
            ),
            "global_damping_log_offset": float(
                self.global_log_coefficients[1].item()
            ),
            "velocity_damping_per_second": (
                self.current_velocity_damping_per_second()
            ),
            "shape_frozen": 1,
            "distance_minimum": float(distance.min().item()),
            "distance_median": float(distance.median().item()),
            "distance_maximum": float(distance.max().item()),
            "shape_minimum": float(shape.min().item()),
            "shape_median": float(shape.median().item()),
            "shape_maximum": float(shape.max().item()),
        }
        self.last_metrics = metrics
        return metrics

    def begin_observation(
        self, active_mask: torch.Tensor
    ) -> tuple[PaperTrajectoryMaterialTrial, ...]:
        """Create independent central perturbations for distance and shape."""
        active = self._activate_observation(active_mask)
        if self.settings.sim_global_causal_mode:
            self.loss_evaluation_count += 4
            return self._begin_sim_global_observation(active)
        self._distance_probe = self._probe(axis_salt=17, active_mask=active)
        self._shape_probe = self._probe(axis_salt=43, active_mask=active)
        perturbation = self.settings.parameter_perturbation
        trials: list[PaperTrajectoryMaterialTrial] = []
        for axis, theta, probe in (
            ("distance", self.distance_theta, self._distance_probe),
            ("shape", self.shape_theta, self._shape_probe),
        ):
            for sign in (1, -1):
                perturbed = theta + float(sign) * perturbation * probe
                # Preserve the non-probed family and every inactive/fixed node
                # bit-for-bit. Sigmoid/logit round-tripping may otherwise add
                # a one-ULP second-axis perturbation to a nominally independent
                # counterfactual.
                distance = self.distance_stiffness.detach().clone()
                shape = self.shape_stiffness.detach().clone()
                if axis == "distance":
                    perturbed_material = self._theta_to_material(
                        perturbed,
                        self.settings.distance_minimum,
                        self.settings.distance_maximum,
                    )
                    distance[active] = perturbed_material[active]
                else:
                    perturbed_material = self._theta_to_material(
                        perturbed,
                        self.settings.shape_minimum,
                        self.settings.shape_maximum,
                    )
                    shape[active] = perturbed_material[active]
                trials.append(
                    PaperTrajectoryMaterialTrial(
                        label=f"{axis}_{'plus' if sign > 0 else 'minus'}",
                        axis=axis,
                        sign=sign,
                        distance_stiffness=distance,
                        shape_stiffness=shape,
                    )
                )
        self.loss_evaluation_count += len(trials)
        return tuple(trials)

    def graph_smooth_loss(
        self, distance: torch.Tensor, shape: torch.Tensor
    ) -> tuple[float, float]:
        if not self.edges.numel():
            return 0.0, 0.0
        source = self.edges[:, 0]
        target = self.edges[:, 1]
        distance_log = torch.log(distance.clamp_min(1.0e-12))
        shape_log = torch.log(shape.clamp_min(1.0e-12))
        distance_loss = torch.mean(
            (distance_log[source] - distance_log[target]).square()
        )
        shape_loss = torch.mean(
            (shape_log[source] - shape_log[target]).square()
        )
        return float(distance_loss.item()), float(shape_loss.item())

    def total_loss(
        self,
        *,
        track_loss: float,
        history_loss: float,
        distance_smooth_loss: float,
        shape_smooth_loss: float,
        parameter_prior_loss: float = 0.0,
    ) -> float:
        settings = self.settings
        return float(
            track_loss
            + settings.history_weight * history_loss
            + settings.distance_smooth_weight * distance_smooth_loss
            + settings.shape_smooth_weight * shape_smooth_loss
            + parameter_prior_loss
        )

    def global_parameter_prior_loss(
        self, trial: PaperTrajectoryMaterialTrial
    ) -> float:
        coefficients = trial.global_log_coefficients
        if coefficients is None:
            return 0.0
        # Match SIM's three-coefficient global objective exactly.  Coupling is
        # frozen, but its zero entry remains in the mean; omitting it made the
        # SUPER prior three times too strong relative to the trajectory loss
        # and repeatedly pulled distance/damping back through their optima.
        return float(
            self.settings.global_parameter_prior_weight
            * (coefficients[0] ** 2 + coefficients[1] ** 2)
            / 3.0
        )

    @staticmethod
    def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
        denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(
            second
        )
        if float(denominator.item()) <= 1.0e-12:
            return -1.0
        return float(torch.dot(first, second).item() / denominator.item())

    def _finish_sim_global_observation(
        self, losses: dict[str, dict[str, float]]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
        settings = self.settings
        required = {
            "distance_plus",
            "distance_minus",
            "damping_plus",
            "damping_minus",
        }
        if set(losses) != required:
            raise ValueError("SIM-global Adam requires distance+damping trials")
        total = {label: float(values["total_loss"]) for label, values in losses.items()}
        if not all(math.isfinite(value) for value in total.values()):
            raise ValueError("SIM-global Adam received a non-finite loss")
        perturbation = settings.parameter_perturbation
        # The authoritative gradient is a central finite difference through
        # one continuous Warp rollout over the causal H1/H2/H3 window.
        warp_gradient = torch.tensor(
            [
                (total["distance_plus"] - total["distance_minus"])
                / (2.0 * perturbation),
                (total["damping_plus"] - total["damping_minus"])
                / (2.0 * perturbation),
            ],
            dtype=torch.float32,
            device=self.global_log_coefficients.device,
        )
        surrogate_total = {
            label: float(values.get("surrogate_total_loss", float("nan")))
            for label, values in losses.items()
        }
        surrogate_gradient = torch.tensor(
            [
                (
                    surrogate_total["distance_plus"]
                    - surrogate_total["distance_minus"]
                )
                / (2.0 * perturbation),
                (
                    surrogate_total["damping_plus"]
                    - surrogate_total["damping_minus"]
                )
                / (2.0 * perturbation),
            ],
            dtype=torch.float32,
            device=self.global_log_coefficients.device,
        )
        horizon_count = min(
            (
                len(values.get("horizon_total_losses", ()))
                for values in losses.values()
            ),
            default=0,
        )
        causal_block_indices = {
            int(values.get("causal_block_index", self.observation_count))
            for values in losses.values()
        }
        if len(causal_block_indices) != 1:
            raise ValueError("SIM-global trials disagree on the causal block")
        causal_block_index = causal_block_indices.pop()
        gradient_cosine = -1.0
        direction_consistent = False
        if (
            horizon_count >= settings.causal_minimum_window_size
            and all(math.isfinite(value) for value in surrogate_total.values())
        ):
            gradient_cosine = self._cosine(
                surrogate_gradient, warp_gradient
            )
            direction_consistent = bool(
                math.isfinite(gradient_cosine)
                and gradient_cosine >= settings.gradient_cosine_minimum
            )
        sensitive = bool(
            torch.isfinite(warp_gradient).all().item()
            and torch.linalg.vector_norm(warp_gradient).item()
            >= settings.minimum_axis_loss_difference
        )
        h3_tail_required = bool(
            horizon_count >= settings.causal_window_size
            and settings.h3_tail_cosine_minimum is not None
        )
        h3_tail_gradient = torch.zeros_like(warp_gradient)
        h3_tail_gradient_cosine = -1.0
        h3_tail_direction_consistent = not h3_tail_required
        if h3_tail_required:
            tail_totals: dict[str, float] = {}
            tail_weights = settings.causal_horizon_weights[
                horizon_count - 2 : horizon_count
            ]
            full_weights = settings.causal_horizon_weights[:horizon_count]
            for label, values in losses.items():
                horizons = tuple(
                    float(value)
                    for value in values.get("horizon_total_losses", ())
                )
                if len(horizons) < horizon_count:
                    raise ValueError("H3 tail validation requires three losses")
                full_track = float(
                    values.get(
                        "track_loss",
                        sum(
                            weight * value
                            for weight, value in zip(full_weights, horizons)
                        )
                        / sum(full_weights),
                    )
                )
                shared_loss = total[label] - full_track
                tail_track = sum(
                    weight * value
                    for weight, value in zip(tail_weights, horizons[-2:])
                ) / sum(tail_weights)
                tail_totals[label] = tail_track + shared_loss
            h3_tail_gradient = torch.tensor(
                [
                    (
                        tail_totals["distance_plus"]
                        - tail_totals["distance_minus"]
                    )
                    / (2.0 * perturbation),
                    (
                        tail_totals["damping_plus"]
                        - tail_totals["damping_minus"]
                    )
                    / (2.0 * perturbation),
                ],
                dtype=torch.float32,
                device=self.global_log_coefficients.device,
            )
            h3_tail_gradient_cosine = self._cosine(
                warp_gradient, h3_tail_gradient
            )
            h3_tail_direction_consistent = bool(
                torch.isfinite(h3_tail_gradient).all().item()
                and torch.linalg.vector_norm(h3_tail_gradient).item()
                >= settings.minimum_axis_loss_difference
                and math.isfinite(h3_tail_gradient_cosine)
                and h3_tail_gradient_cosine
                >= float(settings.h3_tail_cosine_minimum)
            )
        status = "causal_window_warmup"
        effective_step = torch.zeros_like(warp_gradient)
        directional_derivative = float("nan")
        h3_tail_directional_derivative = float("nan")
        predicted_loss_decrease = float("nan")
        descent_direction = False
        available_weights = settings.causal_horizon_weights[:horizon_count]
        step_scale = (
            sum(available_weights) / sum(settings.causal_horizon_weights)
            if available_weights
            else 0.0
        )
        effective_step_cap = settings.maximum_material_log_step * step_scale
        h2_cross_block_cosine = -1.0
        h2_cross_block_confirmed = horizon_count >= settings.causal_window_size
        h2_confirmation_source_block = self.h2_confirmation_block_index
        if horizon_count >= settings.causal_minimum_window_size:
            status = "gradient_misaligned"
        if h3_tail_required and not h3_tail_direction_consistent:
            status = "h3_tail_gradient_misaligned"
        if (
            horizon_count == settings.causal_minimum_window_size
            and not settings.h2_updates_enabled
        ):
            status = "h2_updates_disabled"
            h2_cross_block_confirmed = False
            self.h2_confirmation_gradient = None
            self.h2_confirmation_block_index = None
        elif horizon_count == settings.causal_minimum_window_size:
            if sensitive and direction_consistent:
                previous_gradient = self.h2_confirmation_gradient
                previous_block = self.h2_confirmation_block_index
                if previous_gradient is None or previous_block is None:
                    self.h2_confirmation_gradient = warp_gradient.detach().clone()
                    self.h2_confirmation_block_index = causal_block_index
                    status = "h2_awaiting_cross_block_confirmation"
                elif previous_block == causal_block_index:
                    status = "h2_awaiting_independent_block"
                else:
                    h2_cross_block_cosine = self._cosine(
                        previous_gradient, warp_gradient
                    )
                    h2_cross_block_confirmed = bool(
                        math.isfinite(h2_cross_block_cosine)
                        and h2_cross_block_cosine
                        >= settings.h2_cross_block_cosine_minimum
                    )
                    if not h2_cross_block_confirmed:
                        self.h2_confirmation_gradient = (
                            warp_gradient.detach().clone()
                        )
                        self.h2_confirmation_block_index = causal_block_index
                        status = "h2_cross_block_misaligned"
            else:
                # Confirmation must come from adjacent independently valid
                # blocks; do not retain stale evidence across a failed block.
                self.h2_confirmation_gradient = None
                self.h2_confirmation_block_index = None
        if (
            sensitive
            and direction_consistent
            and h2_cross_block_confirmed
            and h3_tail_direction_consistent
        ):
            next_adam_step = self.global_adam_step + 1
            first = settings.beta1 * self.global_first_moment + (
                1.0 - settings.beta1
            ) * warp_gradient
            second = settings.beta2 * self.global_second_moment + (
                1.0 - settings.beta2
            ) * warp_gradient.square()
            first_hat = first / (1.0 - settings.beta1 ** next_adam_step)
            second_hat = second / (1.0 - settings.beta2 ** next_adam_step)
            raw_step = -settings.learning_rate * first_hat / (
                torch.sqrt(second_hat) + settings.adam_epsilon
            )
            raw_step.clamp_(min=-effective_step_cap, max=effective_step_cap)
            proposed = self.global_log_coefficients + raw_step
            cumulative_offset_limit = (
                settings.h2_maximum_log_offset
                if horizon_count == 2
                else settings.global_maximum_log_offset
            )
            damping_minimum_log = math.log(
                settings.velocity_damping_minimum_per_second
                / settings.velocity_damping_initial_per_second
            )
            damping_maximum_log = math.log(
                settings.velocity_damping_maximum_per_second
                / settings.velocity_damping_initial_per_second
            )
            lower_bounds = torch.tensor(
                [
                    -cumulative_offset_limit,
                    max(damping_minimum_log, -cumulative_offset_limit),
                ],
                dtype=proposed.dtype,
                device=proposed.device,
            )
            upper_bounds = torch.tensor(
                [
                    cumulative_offset_limit,
                    min(damping_maximum_log, cumulative_offset_limit),
                ],
                dtype=proposed.dtype,
                device=proposed.device,
            )
            if horizon_count == 2:
                # An earlier H3 update may legally leave a coefficient outside
                # H2's tighter trust region.  H2 may then move it inward by at
                # most its ordinary step cap, but must neither move it farther
                # outward nor snap it to the H2 boundary in one large jump.
                below = self.global_log_coefficients < lower_bounds
                above = self.global_log_coefficients > upper_bounds
                proposed = torch.where(
                    below,
                    torch.maximum(
                        proposed, self.global_log_coefficients
                    ),
                    proposed,
                )
                proposed = torch.where(
                    above,
                    torch.minimum(
                        proposed, self.global_log_coefficients
                    ),
                    proposed,
                )
                inside = ~(below | above)
                proposed = torch.where(
                    inside,
                    torch.maximum(
                        lower_bounds, torch.minimum(proposed, upper_bounds)
                    ),
                    proposed,
                )
            else:
                proposed = torch.maximum(
                    lower_bounds, torch.minimum(proposed, upper_bounds)
                )
            candidate_step = proposed - self.global_log_coefficients
            directional_derivative = float(
                torch.dot(warp_gradient, candidate_step).item()
            )
            if h3_tail_required:
                h3_tail_directional_derivative = float(
                    torch.dot(h3_tail_gradient, candidate_step).item()
                )
            predicted_loss_decrease = -directional_derivative
            descent_direction = bool(
                math.isfinite(directional_derivative)
                and directional_derivative < 0.0
                and (
                    not h3_tail_required
                    or (
                        math.isfinite(h3_tail_directional_derivative)
                        and h3_tail_directional_derivative < 0.0
                    )
                )
            )
            if descent_direction:
                effective_step = candidate_step
                self.global_log_coefficients.copy_(proposed)
                self.global_first_moment.copy_(first)
                self.global_second_moment.copy_(second)
                self.global_adam_step = next_adam_step
                status = "sim_global_updated"
                self.h2_confirmation_gradient = None
                self.h2_confirmation_block_index = None
            else:
                status = (
                    "non_descent_h3_tail_step"
                    if h3_tail_required
                    and math.isfinite(h3_tail_directional_derivative)
                    and h3_tail_directional_derivative >= 0.0
                    else "non_descent_adam_step"
                )
                if horizon_count == settings.causal_minimum_window_size:
                    self.h2_confirmation_gradient = (
                        warp_gradient.detach().clone()
                    )
                    self.h2_confirmation_block_index = causal_block_index
        distance, shape = self.current_material()
        changed = bool(torch.any(effective_step.abs() > 1.0e-12).item())
        self.update_count += int(changed)
        self.distance_update_count += int(abs(float(effective_step[0].item())) > 1.0e-12)
        metrics: dict[str, float | int | str] = {
            "status": status,
            "observation_count": self.observation_count,
            "update_count": self.update_count,
            "distance_update_count": self.distance_update_count,
            "shape_update_count": 0,
            "loss_evaluation_count": self.loss_evaluation_count,
            "active_particles": int(torch.count_nonzero(self._active_mask).item()),
            "gradient_estimator": (
                "warp_open_loop_central_fd_global_distance_damping"
            ),
            "gradient_consistency_reference": (
                "independent_restarted_one_step_warp_fd"
            ),
            "optimizer": "sim_global_causal_adam",
            "causal_window_size": int(horizon_count),
            "causal_block_index": int(causal_block_index),
            "causal_minimum_window_size": int(
                settings.causal_minimum_window_size
            ),
            "causal_horizon_weights": ",".join(
                f"{value:g}" for value in settings.causal_horizon_weights
            ),
            "available_horizon_weights": ",".join(
                f"{value:g}" for value in available_weights
            ),
            "available_horizon_step_scale": float(step_scale),
            "configured_maximum_material_log_step": float(
                settings.maximum_material_log_step
            ),
            "effective_maximum_material_log_step": float(
                effective_step_cap
            ),
            "active_cumulative_log_offset_limit": float(
                settings.h2_maximum_log_offset
                if horizon_count == 2
                else settings.global_maximum_log_offset
            ),
            "h2_updates_enabled": int(settings.h2_updates_enabled),
            "h2_cross_block_cosine": float(h2_cross_block_cosine),
            "h2_cross_block_cosine_minimum": float(
                settings.h2_cross_block_cosine_minimum
            ),
            "h2_cross_block_confirmed": int(h2_cross_block_confirmed),
            "h2_confirmation_source_block": int(
                -1
                if h2_confirmation_source_block is None
                else h2_confirmation_source_block
            ),
            "gradient_cosine": float(gradient_cosine),
            "gradient_cosine_minimum": float(settings.gradient_cosine_minimum),
            "gradient_direction_consistent": int(direction_consistent),
            "h3_tail_validation_enabled": int(h3_tail_required),
            "h3_tail_cosine_minimum": float(
                -1.0
                if settings.h3_tail_cosine_minimum is None
                else settings.h3_tail_cosine_minimum
            ),
            "h3_tail_gradient_cosine": float(h3_tail_gradient_cosine),
            "h3_tail_direction_consistent": int(
                h3_tail_direction_consistent
            ),
            "h3_tail_distance_gradient": float(
                h3_tail_gradient[0].item()
            ),
            "h3_tail_damping_gradient": float(
                h3_tail_gradient[1].item()
            ),
            "distance_gradient": float(warp_gradient[0].item()),
            "damping_gradient": float(warp_gradient[1].item()),
            "surrogate_distance_gradient": float(
                surrogate_gradient[0].item()
            ),
            "surrogate_damping_gradient": float(
                surrogate_gradient[1].item()
            ),
            "directional_derivative": float(directional_derivative),
            "h3_tail_directional_derivative": float(
                h3_tail_directional_derivative
            ),
            "predicted_loss_decrease": float(predicted_loss_decrease),
            "descent_direction": int(descent_direction),
            "distance_log_step": float(effective_step[0].item()),
            "damping_log_step": float(effective_step[1].item()),
            "global_distance_log_offset": float(
                self.global_log_coefficients[0].item()
            ),
            "global_damping_log_offset": float(
                self.global_log_coefficients[1].item()
            ),
            "velocity_damping_per_second": (
                self.current_velocity_damping_per_second()
            ),
            "shape_frozen": 1,
            "distance_minimum": float(distance.min().item()),
            "distance_median": float(distance.median().item()),
            "distance_maximum": float(distance.max().item()),
            "shape_minimum": float(shape.min().item()),
            "shape_median": float(shape.median().item()),
            "shape_maximum": float(shape.max().item()),
            "track_loss": float(
                sum(values["track_loss"] for values in losses.values()) / 4.0
            ),
            "history_loss": float(
                sum(values["history_loss"] for values in losses.values()) / 4.0
            ),
            "total_loss": float(sum(total.values()) / 4.0),
        }
        self.last_metrics = metrics
        return distance.detach(), shape.detach(), metrics

    def _adam_axis(
        self,
        *,
        theta: torch.Tensor,
        first_moment: torch.Tensor,
        second_moment: torch.Tensor,
        gradient: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        settings = self.settings
        first_moment.mul_(settings.beta1).add_(
            gradient, alpha=1.0 - settings.beta1
        )
        second_moment.mul_(settings.beta2).addcmul_(
            gradient, gradient, value=1.0 - settings.beta2
        )
        bias1 = 1.0 - settings.beta1 ** self.observation_count
        bias2 = 1.0 - settings.beta2 ** self.observation_count
        step = settings.learning_rate * (first_moment / bias1) / (
            torch.sqrt(second_moment / bias2) + settings.adam_epsilon
        )
        updated = theta.clone()
        updated[active_mask] -= step[active_mask]
        return updated

    def _limit_material_log_step(
        self,
        *,
        previous: torch.Tensor,
        proposed: torch.Tensor,
        active_mask: torch.Tensor,
        minimum: float,
        maximum: float,
    ) -> torch.Tensor:
        """Apply the trust bound to the realized material, not raw logits."""
        maximum_step = self.settings.maximum_material_log_step
        realized = torch.log(
            proposed.clamp_min(1.0e-12) / previous.clamp_min(1.0e-12)
        ).clamp(min=-maximum_step, max=maximum_step)
        limited = previous.clone()
        limited[active_mask] = (
            previous[active_mask] * torch.exp(realized[active_mask])
        ).clamp(min=minimum, max=maximum)
        return limited

    def finish_observation(
        self,
        losses: dict[str, dict[str, float]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
        """Apply one Adam step from the four counterfactual measurements."""
        if self.settings.sim_global_causal_mode:
            return self._finish_sim_global_observation(losses)
        required = {
            "distance_plus",
            "distance_minus",
            "shape_plus",
            "shape_minus",
        }
        if set(losses) != required:
            raise ValueError("Paper Adam requires four labelled trial losses")
        total = {label: float(values["total_loss"]) for label, values in losses.items()}
        if not all(math.isfinite(value) for value in total.values()):
            raise ValueError("Paper Adam received a non-finite loss")
        perturbation = self.settings.parameter_perturbation
        distance_difference = total["distance_plus"] - total["distance_minus"]
        shape_difference = total["shape_plus"] - total["shape_minus"]
        distance_sensitive = (
            abs(distance_difference)
            >= self.settings.minimum_axis_loss_difference
        )
        shape_sensitive = (
            abs(shape_difference)
            >= self.settings.minimum_axis_loss_difference
        )
        distance_gradient = (
            distance_difference / (2.0 * perturbation) * self._distance_probe
            if distance_sensitive
            else torch.zeros_like(self.distance_theta)
        )
        shape_gradient = (
            shape_difference / (2.0 * perturbation) * self._shape_probe
            if shape_sensitive
            else torch.zeros_like(self.shape_theta)
        )
        old_distance, old_shape = self.current_material()
        with torch.no_grad():
            proposed_distance_theta = self._adam_axis(
                theta=self.distance_theta,
                first_moment=self.distance_first_moment,
                second_moment=self.distance_second_moment,
                gradient=distance_gradient,
                active_mask=self._active_mask,
            )
            proposed_shape_theta = self._adam_axis(
                theta=self.shape_theta,
                first_moment=self.shape_first_moment,
                second_moment=self.shape_second_moment,
                gradient=shape_gradient,
                active_mask=self._active_mask,
            )
            proposed_distance = self._theta_to_material(
                proposed_distance_theta,
                self.settings.distance_minimum,
                self.settings.distance_maximum,
            )
            proposed_shape = self._theta_to_material(
                proposed_shape_theta,
                self.settings.shape_minimum,
                self.settings.shape_maximum,
            )
            limited_distance = self._limit_material_log_step(
                previous=old_distance,
                proposed=proposed_distance,
                active_mask=self._active_mask,
                minimum=self.settings.distance_minimum,
                maximum=self.settings.distance_maximum,
            )
            limited_shape = self._limit_material_log_step(
                previous=old_shape,
                proposed=proposed_shape,
                active_mask=self._active_mask,
                minimum=self.settings.shape_minimum,
                maximum=self.settings.shape_maximum,
            )
            # Keep inactive logits bit-for-bit.  The material/logit round trip
            # is only needed where this observation carries material evidence.
            self.distance_theta = proposed_distance_theta
            self.shape_theta = proposed_shape_theta
            limited_distance_theta = self._material_to_theta(
                limited_distance,
                self.settings.distance_minimum,
                self.settings.distance_maximum,
            )
            limited_shape_theta = self._material_to_theta(
                limited_shape,
                self.settings.shape_minimum,
                self.settings.shape_maximum,
            )
            self.distance_theta[self._active_mask] = limited_distance_theta[
                self._active_mask
            ]
            self.shape_theta[self._active_mask] = limited_shape_theta[
                self._active_mask
            ]
        distance, shape = self.current_material()
        distance_log_step = torch.log(
            distance / old_distance.clamp_min(1.0e-12)
        )
        shape_log_step = torch.log(shape / old_shape.clamp_min(1.0e-12))
        # Adam momentum can legitimately produce a step when the current
        # central difference falls below the numerical sensitivity threshold.
        # Treat the realized step—not only the current raw gradient—as the
        # commit decision.  Otherwise theta would move internally while the
        # simulator kept the old material, causing a delayed cumulative jump.
        changed_distance = bool(
            torch.any(distance_log_step.abs() > 1.0e-12).item()
        )
        changed_shape = bool(torch.any(shape_log_step.abs() > 1.0e-12).item())
        changed = changed_distance or changed_shape
        self.update_count += int(changed)
        self.distance_update_count += int(changed_distance)
        self.shape_update_count += int(changed_shape)
        distance_smooth, shape_smooth = self.graph_smooth_loss(distance, shape)
        metrics: dict[str, float | int | str] = {
            "status": "adam_updated" if changed else "insensitive_observation",
            "observation_count": self.observation_count,
            "update_count": self.update_count,
            "distance_update_count": self.distance_update_count,
            "shape_update_count": self.shape_update_count,
            "loss_evaluation_count": self.loss_evaluation_count,
            "active_particles": int(torch.count_nonzero(self._active_mask).item()),
            "gradient_estimator": "central_spsa_exact_xpbd_counterfactual",
            "optimizer": "adam_bounded_sigmoid_logits",
            "material_log_step_cap_enabled": 1,
            "maximum_material_log_step": float(
                self.settings.maximum_material_log_step
            ),
            "distance_axis_loss_difference": float(distance_difference),
            "shape_axis_loss_difference": float(shape_difference),
            "distance_axis_sensitive": int(distance_sensitive),
            "shape_axis_sensitive": int(shape_sensitive),
            "distance_gradient_rms": float(
                torch.sqrt(torch.mean(distance_gradient.square())).item()
            ),
            "shape_gradient_rms": float(
                torch.sqrt(torch.mean(shape_gradient.square())).item()
            ),
            "distance_maximum_log_step": float(distance_log_step.abs().max().item()),
            "shape_maximum_log_step": float(shape_log_step.abs().max().item()),
            "distance_log_graph_energy": distance_smooth,
            "shape_log_graph_energy": shape_smooth,
            "distance_minimum": float(distance.min().item()),
            "distance_median": float(distance.median().item()),
            "distance_maximum": float(distance.max().item()),
            "shape_minimum": float(shape.min().item()),
            "shape_median": float(shape.median().item()),
            "shape_maximum": float(shape.max().item()),
            "track_loss": float(
                sum(values["track_loss"] for values in losses.values()) / 4.0
            ),
            "history_loss": float(
                sum(values["history_loss"] for values in losses.values()) / 4.0
            ),
            "total_loss": float(sum(total.values()) / 4.0),
        }
        self.last_metrics = metrics
        return distance.detach(), shape.detach(), metrics


def confidence_weighted_huber_track_loss(
    *,
    predicted_points: torch.Tensor,
    target_points: torch.Tensor,
    confidence: torch.Tensor,
    valid_mask: torch.Tensor,
    robust_scale_m: float,
    region_ids: torch.Tensor | None = None,
    region_balance_weight: float = 0.0,
    tail_region_weight: float = 0.0,
    tail_region_fraction: float = 0.25,
) -> tuple[float, dict[str, float | int]]:
    """Confidence-weighted robust 3D point loss used by every material trial."""
    valid = valid_mask.bool() & torch.isfinite(predicted_points).all(dim=1)
    valid &= torch.isfinite(target_points).all(dim=1) & (confidence > 0.0)
    if not bool(valid.any().item()):
        raise ValueError("No valid trajectory target is available")
    errors = torch.linalg.vector_norm(
        predicted_points[valid] - target_points[valid], dim=1
    )
    normalized = errors / float(robust_scale_m)
    rho = torch.where(
        normalized <= 1.0,
        0.5 * normalized.square(),
        normalized - 0.5,
    )
    weights = confidence[valid]
    weight_sum = weights.sum().clamp_min(1.0e-12)
    point_mean = torch.sum(weights * rho) / weight_sum
    region_mean = point_mean
    tail_mean = point_mean
    valid_region_count = 0
    if region_ids is not None and (region_balance_weight > 0.0 or tail_region_weight > 0.0):
        regions = region_ids.to(device=valid.device, dtype=torch.long)[valid]
        region_losses: list[torch.Tensor] = []
        for region in torch.unique(regions, sorted=True):
            selected = regions == region
            selected_weight = weights[selected].sum().clamp_min(1.0e-12)
            region_losses.append(
                torch.sum(weights[selected] * rho[selected]) / selected_weight
            )
        if region_losses:
            stacked = torch.stack(region_losses)
            valid_region_count = len(region_losses)
            region_mean = stacked.mean()
            tail_count = max(
                1, int(math.ceil(tail_region_fraction * len(region_losses)))
            )
            tail_mean = torch.topk(stacked, k=tail_count).values.mean()
    point_weight = 1.0 - region_balance_weight - tail_region_weight
    loss = (
        point_weight * point_mean
        + region_balance_weight * region_mean
        + tail_region_weight * tail_mean
    )
    rmse = torch.sqrt(torch.sum(weights * errors.square()) / weight_sum)
    mean = torch.sum(weights * errors) / weight_sum
    return float(loss.item()), {
        "valid_tracks": int(torch.count_nonzero(valid).item()),
        "confidence_sum": float(weight_sum.item()),
        "mean_error_m": float(mean.item()),
        "rmse_m": float(rmse.item()),
        "point_mean_loss": float(point_mean.item()),
        "region_mean_loss": float(region_mean.item()),
        "tail_region_loss": float(tail_mean.item()),
        "valid_regions": int(valid_region_count),
    }
