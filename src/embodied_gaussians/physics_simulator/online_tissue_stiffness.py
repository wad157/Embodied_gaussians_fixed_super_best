"""Bounded online paper-PBD stiffness adaptation from accepted visual residuals."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class OnlineTissueStiffnessSettings:
    """Aggressive but bounded adaptation used by the SUPER realtime GUI.

    The signed signal compares the physical deformation with the accepted
    visual correction.  A correction back toward rest means the prediction
    over-deformed and therefore hardens the local distance/shape constraints;
    a correction farther from rest softens them.  This is the realtime
    closed-loop approximation; it does not differentiate through the full
    multi-frame XPBD rollout used by the much slower paper optimizer.
    """

    log_learning_rate: float = 0.18
    maximum_log_step: float = 0.18
    residual_full_scale_m: float = 0.00020
    deformation_full_scale_m: float = 0.00075
    minimum_residual_m: float = 0.00002
    minimum_deformation_m: float = 0.00010
    # A positive constant here previously made ambiguous residuals harden the
    # material even when their signed cosine was neutral.  Keep the knob for
    # old configuration compatibility, but make the unbiased estimator the
    # production default.
    hardening_bias: float = 0.0
    # Keep temporal smoothing, but give a newly verified visual signal enough
    # weight to be visible within a few accepted updates.  This implements
    # e_t = 0.70 e_{t-1} + 0.30 signal_t.
    signal_ema_decay: float = 0.70
    # A rejected material proposal should weaken the evidence that generated
    # it, but must not erase newer observations accumulated while the H=1/3/5
    # shadow was pending.
    rejected_ema_decay: float = 0.85
    spatial_smoothing_iterations: int = 1
    spatial_smoothing_blend: float = 0.35
    # This is deliberately separate from signal smoothing.  It applies a
    # local graph-proximal pass to the candidate stiffness field itself.
    graph_smoothing_iterations: int = 2
    graph_smoothing_blend: float = 0.20
    # Distance and shape now have independent evidence fields.  This gain is
    # therefore an axis-specific learning-rate multiplier, not a copy of the
    # distance step.  Keep it neutral by default.
    shape_update_gain: float = 1.00
    # A high trust-region ceiling is ineffective when the residual-derived
    # gradient only realizes a 0.02--0.04 log step.  When non-zero, rescale
    # each independent axis so its strongest supported node reaches this
    # target, while preserving every node's sign/relative magnitude and the
    # absolute ``maximum_log_step`` safety cap.
    effective_log_step_target: float = 0.0
    maximum_step_amplification: float = 4.0
    # Edge strain is invariant to a free tissue patch's rigid translation and
    # rotation.  It is therefore the primary local material cue; the legacy
    # deformation/residual dot product remains as a 20% fallback where valid
    # graph strain cannot be measured.
    # The restored scheme uses the visual residual/deformation vector signal.
    # Edge strain remains an opt-in research diagnostic, not a default material
    # update input.
    strain_signal_weight: float = 0.0
    minimum_edge_strain: float = 0.001
    edge_strain_full_scale: float = 0.025
    # Distance stiffness carries most of the regional stretch contrast.  Keep
    # its per-commit log step unchanged, but allow verified regions to separate
    # over repeated, independently validated observations.
    # The standalone updater keeps the conservative historical defaults.
    # The SUPER formal runner explicitly opts into wider recovery bounds; the
    # bounds are only a search box and every commit still passes causal RGB
    # prediction plus tetrahedron/contact safety gates.
    distance_minimum: float = 0.10
    distance_maximum: float = 2.00
    shape_minimum: float = 0.003
    # Liang et al. optimize shape stiffness in [0, 0.02]. Keep a conservative
    # positive floor below the current 0.004 scene baseline, but do not let
    # online feedback harden beyond the paper's upper bound.
    shape_maximum: float = 0.020

    def validate(self) -> None:
        if self.log_learning_rate <= 0.0 or self.maximum_log_step <= 0.0:
            raise ValueError("Online stiffness step sizes must be positive")
        if min(
            self.residual_full_scale_m,
            self.deformation_full_scale_m,
            self.minimum_residual_m,
            self.minimum_deformation_m,
        ) <= 0.0:
            raise ValueError("Online stiffness metric scales must be positive")
        if self.minimum_residual_m > self.residual_full_scale_m:
            raise ValueError(
                "Minimum residual cannot exceed residual full scale"
            )
        if self.minimum_deformation_m > self.deformation_full_scale_m:
            raise ValueError(
                "Minimum deformation cannot exceed deformation full scale"
            )
        if not -1.0 <= self.hardening_bias <= 1.0:
            raise ValueError("Online stiffness hardening bias must lie in [-1,1]")
        if not 0.0 <= self.signal_ema_decay < 1.0:
            raise ValueError("Online stiffness EMA decay must lie in [0,1)")
        if not 0.0 <= self.rejected_ema_decay <= 1.0:
            raise ValueError("Rejected stiffness EMA decay must lie in [0,1]")
        if self.spatial_smoothing_iterations < 0:
            raise ValueError("Stiffness smoothing iterations cannot be negative")
        if not 0.0 <= self.spatial_smoothing_blend <= 1.0:
            raise ValueError("Stiffness smoothing blend must lie in [0,1]")
        if self.graph_smoothing_iterations < 0:
            raise ValueError("Graph smoothing iterations cannot be negative")
        if not 0.0 <= self.graph_smoothing_blend <= 1.0:
            raise ValueError("Graph smoothing blend must lie in [0,1]")
        if self.shape_update_gain <= 0.0:
            raise ValueError("Shape stiffness update gain must be positive")
        if not 0.0 <= self.effective_log_step_target <= self.maximum_log_step:
            raise ValueError(
                "Effective stiffness step target must lie in [0, maximum]"
            )
        if self.maximum_step_amplification < 1.0:
            raise ValueError("Maximum stiffness step amplification must be >= 1")
        if not 0.0 <= self.strain_signal_weight <= 1.0:
            raise ValueError("Strain signal weight must lie in [0,1]")
        if not 0.0 < self.minimum_edge_strain <= self.edge_strain_full_scale:
            raise ValueError("Edge-strain scales are invalid")
        if not 0.0 < self.distance_minimum <= self.distance_maximum:
            raise ValueError("Distance stiffness bounds are invalid")
        if not 0.0 < self.shape_minimum <= self.shape_maximum:
            raise ValueError("Shape stiffness bounds are invalid")


@dataclass
class PaperStiffnessEvidence:
    """One accepted 2D residual incorporated into the material evidence."""

    signal_ema: torch.Tensor
    log_step: torch.Tensor
    distance_signal_ema: torch.Tensor
    shape_signal_ema: torch.Tensor
    distance_gradient_log_step: torch.Tensor
    shape_gradient_log_step: torch.Tensor
    eligible_mask: torch.Tensor
    # Nodes on which a material hypothesis is physically legal.  Unlike
    # ``eligible_mask`` this does not require a non-zero image gradient: a
    # global material parameter is shared by visible and temporarily occluded
    # nodes, while fixed/contact-controlled/invalid nodes remain hard excluded.
    material_valid_mask: torch.Tensor
    # Particles activated by this exact observation.  This is diagnostic only;
    # a proposal may also contain older, still-valid EMA contributors.
    source_mask: torch.Tensor
    metrics: dict[str, float | int | str]


@dataclass
class PaperStiffnessCandidate:
    """Tentative material state that has not yet reached the XPBD projector."""

    distance_stiffness: torch.Tensor
    shape_stiffness: torch.Tensor
    signal_ema: torch.Tensor
    # ``log_step`` is the frozen evidence direction.  Distance and shape use
    # separate realized steps so material-isolation search can independently
    # test either family and can reverse the heuristic direction.
    log_step: torch.Tensor
    distance_signal_ema: torch.Tensor
    shape_signal_ema: torch.Tensor
    distance_gradient_log_step: torch.Tensor
    shape_gradient_log_step: torch.Tensor
    distance_log_step: torch.Tensor
    shape_log_step: torch.Tensor
    eligible_mask: torch.Tensor
    material_valid_mask: torch.Tensor
    # Frozen proposal contributors, not merely the particles active in the
    # last image.  Rejection must decay every EMA value that actually produced
    # this candidate while preserving evidence arriving later during pending.
    source_mask: torch.Tensor
    metrics: dict[str, float | int | str]


class ResidualDrivenPaperStiffnessUpdater:
    """Update per-particle paper distance/shape stiffness in log space."""

    def __init__(
        self,
        *,
        rest_positions: torch.Tensor,
        fixed_mask: torch.Tensor,
        edges: torch.Tensor,
        distance_stiffness: torch.Tensor,
        shape_stiffness: torch.Tensor,
        settings: OnlineTissueStiffnessSettings | None = None,
    ) -> None:
        self.settings = settings or OnlineTissueStiffnessSettings()
        self.settings.validate()
        self.rest_positions = rest_positions.detach().to(dtype=torch.float32)
        self.fixed_mask = fixed_mask.detach().to(
            device=self.rest_positions.device, dtype=torch.bool
        )
        self.edges = edges.detach().to(
            device=self.rest_positions.device, dtype=torch.long
        )
        self.distance_stiffness = distance_stiffness
        self.shape_stiffness = shape_stiffness
        particle_count = len(self.rest_positions)
        expected = (particle_count,)
        if self.fixed_mask.shape != expected:
            raise ValueError("Fixed mask does not match stiffness particles")
        if self.distance_stiffness.shape != expected:
            raise ValueError("Distance stiffness does not match particles")
        if self.shape_stiffness.shape != expected:
            raise ValueError("Shape stiffness does not match particles")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("Stiffness graph edges must have shape [E,2]")
        if self.edges.numel() and (
            int(self.edges.min()) < 0 or int(self.edges.max()) >= particle_count
        ):
            raise ValueError("Stiffness graph edge is out of range")
        self.rest_edge_lengths = torch.linalg.vector_norm(
            self.rest_positions[self.edges[:, 1]]
            - self.rest_positions[self.edges[:, 0]],
            dim=1,
        ).clamp_min(1.0e-8)
        # Candidate scopes are tiny (about 1.5k particles), so a deterministic
        # CPU adjacency avoids an iterative GPU connected-component kernel and
        # is built only once.  It contains topology only, never observations.
        self._adjacency: tuple[tuple[int, ...], ...] = self._build_adjacency(
            particle_count
        )
        self.initial_distance = self.distance_stiffness.detach().clone()
        self.initial_shape = self.shape_stiffness.detach().clone()
        self.distance_signal_ema = torch.zeros(
            particle_count,
            dtype=torch.float32,
            device=self.rest_positions.device,
        )
        self.shape_signal_ema = torch.zeros_like(self.distance_signal_ema)
        # Backward-compatible public handle used by the GUI and older gates.
        # It intentionally aliases the distance/stretch evidence, while the
        # independent shape evidence is exposed explicitly above.
        self.signal_ema = self.distance_signal_ema
        self.update_count = 0
        self.distance_update_count = 0
        self.shape_update_count = 0
        self.candidate_count = 0
        self.rejected_count = 0
        self.rejection_reason_counts: dict[str, int] = {}
        self.evidence_count = 0
        self.pending_candidate: PaperStiffnessCandidate | None = None
        self.last_metrics: dict[str, float | int | str] | None = None

    def _build_adjacency(
        self, particle_count: int
    ) -> tuple[tuple[int, ...], ...]:
        neighbors: list[set[int]] = [set() for _ in range(particle_count)]
        for source, target in self.edges.detach().cpu().tolist():
            neighbors[int(source)].add(int(target))
            neighbors[int(target)].add(int(source))
        return tuple(tuple(sorted(values)) for values in neighbors)

    def _candidate_scope_mask(
        self,
        base_log_step: torch.Tensor,
        eligible_mask: torch.Tensor,
        scope: str,
    ) -> torch.Tensor:
        """Select a reproducible spatial trust region from proposal evidence.

        ``component_0`` and ``component_1`` are the two strongest connected
        same-sign EMA regions above 10% of the proposal's peak magnitude.  The
        ranking uses integrated |log-step|, then the smallest particle index,
        so CUDA scheduling cannot change it.  This is internal PBD topology;
        no point cloud, depth, manual track, or observed 3D point is used.
        """
        if scope == "full":
            return (
                eligible_mask & (base_log_step.detach().abs() > 1.0e-10)
            ).detach().clone()
        if not scope.startswith("component_"):
            raise ValueError(f"Unknown stiffness candidate scope: {scope}")
        try:
            component_rank = int(scope.removeprefix("component_"))
        except ValueError as exc:
            raise ValueError(
                f"Invalid stiffness candidate scope: {scope}"
            ) from exc
        if component_rank < 0:
            raise ValueError(f"Invalid stiffness candidate scope: {scope}")

        step = base_log_step.detach()
        magnitude = step.abs()
        maximum = float(magnitude.max().item()) if magnitude.numel() else 0.0
        result = torch.zeros_like(eligible_mask, dtype=torch.bool)
        if maximum <= 1.0e-10:
            return result
        support = (
            eligible_mask
            & (magnitude >= max(1.0e-10, 0.10 * maximum))
        )
        support_cpu = support.detach().cpu().tolist()
        positive_cpu = (step > 0.0).detach().cpu().tolist()
        magnitude_cpu = magnitude.detach().cpu().tolist()
        visited = [False] * len(support_cpu)
        components: list[tuple[float, int, list[int]]] = []
        for seed in range(len(support_cpu)):
            if not support_cpu[seed] or visited[seed]:
                continue
            sign = positive_cpu[seed]
            stack = [seed]
            visited[seed] = True
            nodes: list[int] = []
            while stack:
                node = stack.pop()
                nodes.append(node)
                for neighbor in self._adjacency[node]:
                    if (
                        support_cpu[neighbor]
                        and not visited[neighbor]
                        and positive_cpu[neighbor] == sign
                    ):
                        visited[neighbor] = True
                        stack.append(neighbor)
            score = float(sum(magnitude_cpu[node] for node in nodes))
            components.append((score, min(nodes), nodes))
        components.sort(key=lambda item: (-item[0], item[1]))
        if component_rank >= len(components):
            return result
        selected = torch.as_tensor(
            components[component_rank][2],
            dtype=torch.long,
            device=result.device,
        )
        result[selected] = True
        return result

    def reset(self) -> None:
        with torch.no_grad():
            self.distance_stiffness.copy_(self.initial_distance)
            self.shape_stiffness.copy_(self.initial_shape)
            self.distance_signal_ema.zero_()
            self.shape_signal_ema.zero_()
        self.update_count = 0
        self.distance_update_count = 0
        self.shape_update_count = 0
        self.candidate_count = 0
        self.rejected_count = 0
        self.rejection_reason_counts.clear()
        self.evidence_count = 0
        self.pending_candidate = None
        self.last_metrics = None

    def reconfigure(self, settings: OnlineTissueStiffnessSettings) -> None:
        """Install runtime tuning without mixing old EMA into the new policy.

        The caller must finish or reject an outstanding candidate first.  The
        verified material field is preserved, except that values outside newly
        selected absolute bounds are clipped.  Reset baselines are unchanged.
        """
        settings.validate()
        if self.pending_candidate is not None:
            raise RuntimeError(
                "Cannot reconfigure online stiffness with a pending candidate"
            )
        with torch.no_grad():
            self.distance_stiffness.clamp_(
                settings.distance_minimum,
                settings.distance_maximum,
            )
            self.shape_stiffness.clamp_(
                settings.shape_minimum,
                settings.shape_maximum,
            )
            self.distance_signal_ema.zero_()
            self.shape_signal_ema.zero_()
        self.settings = settings
        self.last_metrics = None

    def invalidate_signal_history(
        self, mask: torch.Tensor | None = None
    ) -> None:
        """Clear EMA evidence invalidated by a control/contact transition."""
        with torch.no_grad():
            if mask is None:
                self.distance_signal_ema.zero_()
                self.shape_signal_ema.zero_()
                return
            invalid = self._validated_mask(
                mask, default=False, label="EMA invalidation"
            )
            self.distance_signal_ema[invalid] = 0.0
            self.shape_signal_ema[invalid] = 0.0

    def _smooth(self, signal: torch.Tensor) -> torch.Tensor:
        if self.settings.spatial_smoothing_iterations == 0:
            return signal
        source = self.edges[:, 0]
        target = self.edges[:, 1]
        ones = torch.ones_like(source, dtype=signal.dtype)
        counts = torch.zeros_like(signal)
        counts.index_add_(0, source, ones)
        counts.index_add_(0, target, ones)
        current = signal
        for _ in range(self.settings.spatial_smoothing_iterations):
            sums = torch.zeros_like(current)
            sums.index_add_(0, source, current[target])
            sums.index_add_(0, target, current[source])
            neighbor_average = sums / counts.clamp_min(1.0)
            current = torch.lerp(
                current,
                neighbor_average,
                self.settings.spatial_smoothing_blend,
            )
        return current

    def _edge_strain_signal(
        self,
        prediction: torch.Tensor,
        corrected: torch.Tensor,
        eligible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return rigid-motion-invariant node evidence from accepted RGB state.

        Only edges whose two endpoints are valid visual/material nodes may
        contribute.  This prevents a direct/support/contact-controlled node
        from injecting an apparent strain into a neighboring free node.
        Positive evidence means the physical prediction stretched/compressed
        an edge more than the accepted visual state and therefore hardens it;
        negative evidence softens it.
        """
        zeros = torch.zeros(
            len(self.rest_positions),
            dtype=prediction.dtype,
            device=prediction.device,
        )
        if not self.edges.numel():
            return zeros, zeros.clone()
        source = self.edges[:, 0]
        target = self.edges[:, 1]
        valid_edges = eligible_mask[source] & eligible_mask[target]
        if not bool(valid_edges.any().item()):
            return zeros, zeros.clone()
        source = source[valid_edges]
        target = target[valid_edges]
        rest_lengths = self.rest_edge_lengths[valid_edges]
        predicted_lengths = torch.linalg.vector_norm(
            prediction[target] - prediction[source], dim=1
        )
        corrected_lengths = torch.linalg.vector_norm(
            corrected[target] - corrected[source], dim=1
        )
        predicted_strain = (predicted_lengths - rest_lengths) / rest_lengths
        corrected_strain = (corrected_lengths - rest_lengths) / rest_lengths
        evidence_scale = torch.maximum(
            predicted_strain.abs(), corrected_strain.abs()
        )
        active_edges = (
            predicted_strain.abs() >= self.settings.minimum_edge_strain
        )
        denominator = torch.maximum(
            evidence_scale,
            torch.full_like(evidence_scale, self.settings.minimum_edge_strain),
        )
        edge_signal = torch.where(
            active_edges,
            (predicted_strain.abs() - corrected_strain.abs()) / denominator,
            torch.zeros_like(evidence_scale),
        ).clamp(-1.0, 1.0)
        confidence = torch.where(
            active_edges,
            torch.clamp(
                evidence_scale / self.settings.edge_strain_full_scale,
                max=1.0,
            ),
            torch.zeros_like(evidence_scale),
        )
        weighted_signal = edge_signal * confidence
        node_sum = zeros.clone()
        node_weight = zeros.clone()
        node_sum.index_add_(0, source, weighted_signal)
        node_sum.index_add_(0, target, weighted_signal)
        node_weight.index_add_(0, source, confidence)
        node_weight.index_add_(0, target, confidence)
        return node_sum / node_weight.clamp_min(1.0e-8), node_weight

    def edge_strain_observability_mask(
        self,
        *,
        prediction: torch.Tensor,
        corrected: torch.Tensor,
        eligible_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return nodes carrying current, local edge-strain evidence.

        This is a read-only observability query for the trajectory-loss Adam
        path.  It deliberately does not update the legacy residual-gradient
        EMA and therefore cannot leak a second material controller into Adam.
        """
        eligible = self._validated_mask(
            eligible_mask,
            default=False,
            label="edge-strain observability",
        ) & ~self.fixed_mask
        _, strain_weight = self._edge_strain_signal(
            prediction,
            corrected,
            eligible,
        )
        residual_norm = torch.linalg.vector_norm(
            corrected - prediction, dim=1
        )
        return (
            eligible
            & (strain_weight > 0.0)
            & (residual_norm >= self.settings.minimum_residual_m)
        )

    def _rigid_aligned_shape_signal(
        self,
        prediction: torch.Tensor,
        corrected: torch.Tensor,
        residual_norm: torch.Tensor,
        eligible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a shape-matching gradient independent of edge strain.

        Distance stiffness is identified from edge-length strain.  Shape
        stiffness instead needs evidence about non-rigid form.  For the
        currently observable material nodes, fit the best rigid transform of
        the rest state separately to the physical prediction and to the
        accepted visual state.  The signed reduction in per-node non-rigid
        error is the continuous shape gradient: positive hardens shape,
        negative softens it.  Pure translation/rotation therefore cannot be
        mistaken for a shape-stiffness error.
        """
        zeros = torch.zeros(
            len(self.rest_positions),
            dtype=prediction.dtype,
            device=prediction.device,
        )
        indices = torch.nonzero(eligible_mask, as_tuple=False).flatten()
        if indices.numel() < 3:
            return zeros, zeros.clone()
        rest = self.rest_positions[indices].to(dtype=prediction.dtype)

        def aligned_error(positions: torch.Tensor) -> torch.Tensor:
            observed = positions[indices]
            rest_center = rest.mean(dim=0, keepdim=True)
            observed_center = observed.mean(dim=0, keepdim=True)
            rest_centered = rest - rest_center
            observed_centered = observed - observed_center
            covariance = rest_centered.transpose(0, 1) @ observed_centered
            u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
            rotation = u @ vh
            if float(torch.linalg.det(rotation).item()) < 0.0:
                correction = torch.eye(
                    3, dtype=rotation.dtype, device=rotation.device
                )
                correction[-1, -1] = -1.0
                rotation = u @ correction @ vh
            aligned = rest_centered @ rotation + observed_center
            return torch.linalg.vector_norm(observed - aligned, dim=1)

        predicted_error = aligned_error(prediction)
        corrected_error = aligned_error(corrected)
        evidence_scale = torch.maximum(predicted_error, corrected_error)
        active = (
            (residual_norm[indices] >= self.settings.minimum_residual_m)
            & (evidence_scale >= self.settings.minimum_deformation_m)
        )
        signed = torch.where(
            active,
            (predicted_error - corrected_error)
            / evidence_scale.clamp_min(self.settings.minimum_deformation_m),
            torch.zeros_like(evidence_scale),
        ).clamp(-1.0, 1.0)
        confidence = torch.where(
            active,
            torch.clamp(
                evidence_scale / self.settings.deformation_full_scale_m,
                max=1.0,
            )
            * torch.clamp(
                residual_norm[indices] / self.settings.residual_full_scale_m,
                max=1.0,
            ),
            torch.zeros_like(evidence_scale),
        )
        signal = zeros.clone()
        weight = zeros.clone()
        signal[indices] = signed * confidence
        weight[indices] = confidence
        return signal, weight

    def _effective_log_step(
        self,
        signal_ema: torch.Tensor,
        eligible_mask: torch.Tensor,
        *,
        axis_gain: float,
    ) -> tuple[torch.Tensor, float, float]:
        """Convert one axis' EMA gradient to a bounded effective log step."""
        settings = self.settings
        raw = settings.log_learning_rate * float(axis_gain) * signal_ema
        raw = raw.masked_fill(~eligible_mask, 0.0)
        raw_maximum = float(raw.abs().max().item()) if raw.numel() else 0.0
        amplification = 1.0
        if settings.effective_log_step_target > 0.0 and raw_maximum > 1.0e-10:
            amplification = min(
                settings.maximum_step_amplification,
                max(1.0, settings.effective_log_step_target / raw_maximum),
            )
        result = torch.clamp(
            raw * amplification,
            min=-settings.maximum_log_step,
            max=settings.maximum_log_step,
        )
        result[~eligible_mask] = 0.0
        return result, float(amplification), raw_maximum

    @staticmethod
    def _stronger_signed_field(
        distance_field: torch.Tensor,
        shape_field: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility view retaining the stronger signed axis per node."""
        return torch.where(
            distance_field.abs() >= shape_field.abs(),
            distance_field,
            shape_field,
        )

    def _expand_mask(
        self,
        mask: torch.Tensor,
        allowed: torch.Tensor,
        layers: int,
    ) -> torch.Tensor:
        """Expand a material seed only through allowed physical graph nodes."""
        current = mask & allowed
        if layers <= 0 or not self.edges.numel():
            return current
        source = self.edges[:, 0]
        target = self.edges[:, 1]
        for _ in range(layers):
            expanded = current.clone()
            expanded[source] |= current[target] & allowed[source]
            expanded[target] |= current[source] & allowed[target]
            current = expanded & allowed
        return current

    def _smooth_stiffness_field(
        self,
        *,
        verified: torch.Tensor,
        raw_candidate: torch.Tensor,
        seed_mask: torch.Tensor,
        eligible_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a bounded local graph proximal step in log-stiffness space."""
        settings = self.settings
        if (
            settings.graph_smoothing_iterations == 0
            or settings.graph_smoothing_blend == 0.0
            or not self.edges.numel()
            or not bool(seed_mask.any().item())
        ):
            return raw_candidate
        allowed = eligible_mask & ~self.fixed_mask
        support = self._expand_mask(
            seed_mask,
            allowed,
            settings.graph_smoothing_iterations,
        )
        source = self.edges[:, 0]
        target = self.edges[:, 1]
        valid_edges = support[source] & support[target]
        edge_source = source[valid_edges]
        edge_target = target[valid_edges]
        if not edge_source.numel():
            return raw_candidate
        current = torch.log(raw_candidate.clamp_min(1.0e-12))
        verified_log = torch.log(verified.clamp_min(1.0e-12))
        ones = torch.ones_like(edge_source, dtype=current.dtype)
        counts = torch.zeros_like(current)
        counts.index_add_(0, edge_source, ones)
        counts.index_add_(0, edge_target, ones)
        active = support & (counts > 0.0)
        for _ in range(settings.graph_smoothing_iterations):
            sums = torch.zeros_like(current)
            sums.index_add_(0, edge_source, current[edge_target])
            sums.index_add_(0, edge_target, current[edge_source])
            neighbor_average = sums / counts.clamp_min(1.0)
            next_value = current.clone()
            next_value[active] = torch.lerp(
                current[active],
                neighbor_average[active],
                settings.graph_smoothing_blend,
            )
            next_value[~allowed] = verified_log[~allowed]
            current = next_value
        smoothed = torch.exp(current)
        # exp(log(k)) can differ by one float32 ULP.  Restore hard-excluded and
        # fixed values by copy so graph regularization is exactly incapable of
        # touching them, not merely numerically close.
        smoothed[~allowed] = verified[~allowed]
        return smoothed

    def _enforce_realized_log_step_cap(
        self,
        *,
        verified: torch.Tensor,
        candidate: torch.Tensor,
        minimum: float,
        maximum: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bound the *final* material change after graph regularization.

        The evidence gradient is already clamped before graph smoothing, but
        smoothing operates on the absolute log-stiffness field.  Once repeated
        commits have made neighbouring values heterogeneous, that proximal
        pass can move a node farther than ``maximum_log_step`` relative to its
        currently verified value.  Enforce the trust region a second time on
        the value that will actually be committed.

        Returns the bounded candidate, its realized log step, and the
        pre-cap realized step for diagnostics and regression tests.
        """
        settings = self.settings
        verified_safe = verified.detach().clamp(minimum, maximum)
        candidate_safe = candidate.detach().clamp(minimum, maximum)
        pre_cap_step = torch.log(
            candidate_safe / verified_safe.clamp_min(1.0e-12)
        )
        # exp/log round-tripping in float32 can turn an exact 0.10 clamp into
        # 0.10000004 in the actually stored material ratio.  Stay a few ULPs
        # inside the user-facing cap so the realized value is strictly <= it.
        numerical_margin = 8.0 * torch.finfo(verified_safe.dtype).eps
        realized_cap = max(
            float(settings.maximum_log_step) - numerical_margin,
            0.0,
        )
        bounded_step = pre_cap_step.clamp(
            min=-realized_cap,
            max=realized_cap,
        )
        bounded_candidate = torch.clamp(
            verified_safe * torch.exp(bounded_step),
            minimum,
            maximum,
        )
        realized_step = torch.log(
            bounded_candidate / verified_safe.clamp_min(1.0e-12)
        )
        return bounded_candidate, realized_step, pre_cap_step

    def _graph_log_energy(self, values: torch.Tensor) -> float:
        if not self.edges.numel():
            return 0.0
        log_values = torch.log(values.clamp_min(1.0e-12))
        difference = (
            log_values[self.edges[:, 1]] - log_values[self.edges[:, 0]]
        )
        return float(torch.mean(difference * difference).item())

    def _validated_mask(
        self,
        mask: torch.Tensor | None,
        *,
        default: bool,
        label: str,
    ) -> torch.Tensor:
        if mask is None:
            return torch.full_like(self.fixed_mask, default)
        result = mask.detach().to(
            device=self.rest_positions.device, dtype=torch.bool
        )
        if result.shape != self.fixed_mask.shape:
            raise ValueError(f"Stiffness {label} mask has the wrong shape")
        return result

    def accumulate_evidence(
        self,
        *,
        physical_prediction: torch.Tensor,
        accepted_residual: torch.Tensor,
        quality_valid_mask: torch.Tensor | None = None,
        supervision_valid_mask: torch.Tensor | None = None,
        control_exclusion_mask: torch.Tensor | None = None,
        globally_paused: bool = False,
    ) -> PaperStiffnessEvidence:
        """Accumulate one accepted 2D residual, even while a candidate waits.

        ``control_exclusion_mask`` is a hard target mask as well as a source
        mask: spatial smoothing and EMA history cannot leak an update back into
        a direct/support ``u_t`` neighborhood.
        """
        settings = self.settings
        prediction = physical_prediction.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        residual = accepted_residual.detach().to(
            device=self.rest_positions.device, dtype=torch.float32
        )
        if prediction.shape != self.rest_positions.shape:
            raise ValueError("Stiffness update prediction has the wrong shape")
        if residual.shape != self.rest_positions.shape:
            raise ValueError("Stiffness update residual has the wrong shape")
        quality_valid = self._validated_mask(
            quality_valid_mask,
            default=True,
            label="quality-valid",
        )
        supervision_valid = self._validated_mask(
            supervision_valid_mask,
            default=True,
            label="supervision-valid",
        )
        control_excluded = self._validated_mask(
            control_exclusion_mask,
            default=False,
            label="control-exclusion",
        )
        material_valid = quality_valid & ~control_excluded & ~self.fixed_mask
        eligible = material_valid & supervision_valid
        if globally_paused:
            material_valid.zero_()
            eligible.zero_()
        # A bad tetrahedron or direct positional-control region invalidates
        # old evidence immediately, even if the new material candidate is
        # later rejected.  Otherwise stale EMA can reappear as soon as the
        # node becomes eligible again.
        with torch.no_grad():
            invalid_history = (
                ~quality_valid | control_excluded | self.fixed_mask
            )
            self.distance_signal_ema[invalid_history] = 0.0
            self.shape_signal_ema[invalid_history] = 0.0
        deformation = prediction - self.rest_positions
        corrected = prediction + residual
        deformation_norm = torch.linalg.vector_norm(deformation, dim=1)
        residual_norm = torch.linalg.vector_norm(residual, dim=1)
        vector_active = (
            (residual_norm >= settings.minimum_residual_m)
            & (deformation_norm >= settings.minimum_deformation_m)
            & eligible
        )
        cosine = -torch.sum(deformation * residual, dim=1) / (
            deformation_norm * residual_norm
        ).clamp_min(1.0e-12)
        signed_direction = torch.clamp(
            cosine + settings.hardening_bias, min=-1.0, max=1.0
        )
        residual_strength = torch.clamp(
            residual_norm / settings.residual_full_scale_m, max=1.0
        )
        deformation_strength = torch.clamp(
            deformation_norm / settings.deformation_full_scale_m, max=1.0
        )
        vector_signal = torch.where(
            vector_active,
            signed_direction * residual_strength * deformation_strength,
            torch.zeros_like(signed_direction),
        )
        strain_signal, strain_weight = self._edge_strain_signal(
            prediction, corrected, eligible
        )
        strain_active = (
            (strain_weight > 0.0)
            & (residual_norm >= settings.minimum_residual_m)
            & eligible
        )
        distance_signal = torch.where(
            strain_active,
            settings.strain_signal_weight * strain_signal
            + (1.0 - settings.strain_signal_weight) * vector_signal,
            vector_signal,
        )
        shape_signal, shape_weight = self._rigid_aligned_shape_signal(
            prediction,
            corrected,
            residual_norm,
            eligible,
        )
        shape_active = (shape_weight > 0.0) & eligible
        material_active = vector_active | strain_active | shape_active
        distance_signal = self._smooth(distance_signal)
        shape_signal = self._smooth(shape_signal)
        distance_signal[~eligible] = 0.0
        shape_signal[~eligible] = 0.0
        # Evidence is independent of candidate verification.  A frozen
        # candidate may wait five video frames, while this accumulator keeps
        # ingesting every accepted 2D residual in those frames.
        with torch.no_grad():
            self.distance_signal_ema.mul_(settings.signal_ema_decay)
            self.distance_signal_ema.add_(
                distance_signal,
                alpha=1.0 - settings.signal_ema_decay,
            )
            self.shape_signal_ema.mul_(settings.signal_ema_decay)
            self.shape_signal_ema.add_(
                shape_signal,
                alpha=1.0 - settings.signal_ema_decay,
            )
            self.distance_signal_ema[invalid_history] = 0.0
            self.shape_signal_ema[invalid_history] = 0.0
            if globally_paused:
                self.distance_signal_ema.zero_()
                self.shape_signal_ema.zero_()
        self.evidence_count += 1
        distance_log_step, distance_amplification, distance_raw_maximum = (
            self._effective_log_step(
                self.distance_signal_ema,
                eligible,
                axis_gain=1.0,
            )
        )
        shape_log_step, shape_amplification, shape_raw_maximum = (
            self._effective_log_step(
                self.shape_signal_ema,
                eligible,
                axis_gain=settings.shape_update_gain,
            )
        )
        log_step = self._stronger_signed_field(
            distance_log_step, shape_log_step
        )
        signal_ema = self._stronger_signed_field(
            self.distance_signal_ema, self.shape_signal_ema
        )
        active_count = int(torch.count_nonzero(material_active).item())
        vector_active_count = int(torch.count_nonzero(vector_active).item())
        strain_active_count = int(torch.count_nonzero(strain_active).item())
        shape_active_count = int(torch.count_nonzero(shape_active).item())
        hardening_count = int(
            torch.count_nonzero(
                (distance_log_step > 0.0) | (shape_log_step > 0.0)
            ).item()
        )
        softening_count = int(
            torch.count_nonzero(
                (distance_log_step < 0.0) | (shape_log_step < 0.0)
            ).item()
        )
        quality_valid_count = int(
            torch.count_nonzero(quality_valid & ~self.fixed_mask).item()
        )
        quality_masked_count = int(
            torch.count_nonzero(~quality_valid & ~self.fixed_mask).item()
        )
        supervision_masked_count = int(
            torch.count_nonzero(~supervision_valid & ~self.fixed_mask).item()
        )
        control_excluded_count = int(
            torch.count_nonzero(control_excluded & ~self.fixed_mask).item()
        )
        ema_active_count = int(
            torch.count_nonzero(
                (self.distance_signal_ema.abs() > 1.0e-8)
                | (self.shape_signal_ema.abs() > 1.0e-8)
            ).item()
        )
        metrics: dict[str, float | int | str] = {
            "status": "evidence",
            "evidence_count": self.evidence_count,
            "candidate_count": self.candidate_count,
            "update_count": self.update_count,
            "rejected_count": self.rejected_count,
            "active_particles": active_count,
            "vector_signal_active_particles": vector_active_count,
            "strain_signal_active_particles": strain_active_count,
            "shape_signal_active_particles": shape_active_count,
            "gradient_axes": "independent_distance_strain_and_rigid_shape",
            "strain_signal_weight": float(settings.strain_signal_weight),
            "minimum_edge_strain": float(settings.minimum_edge_strain),
            "edge_strain_full_scale": float(settings.edge_strain_full_scale),
            "hardening_particles": hardening_count,
            "softening_particles": softening_count,
            "quality_valid_particles": quality_valid_count,
            "quality_masked_particles": quality_masked_count,
            "supervision_masked_particles": supervision_masked_count,
            "control_excluded_particles": control_excluded_count,
            "ema_active_particles": ema_active_count,
            "globally_paused": int(globally_paused),
            "maximum_log_step": float(log_step.abs().max().item()),
            "mean_absolute_log_step": float(log_step.abs().mean().item()),
            "effective_log_step_target": float(
                settings.effective_log_step_target
            ),
            "maximum_step_amplification": float(
                settings.maximum_step_amplification
            ),
            "distance_raw_maximum_log_step": distance_raw_maximum,
            "shape_raw_maximum_log_step": shape_raw_maximum,
            "distance_step_amplification": distance_amplification,
            "shape_step_amplification": shape_amplification,
            "distance_gradient_maximum_log_step": float(
                distance_log_step.abs().max().item()
            ),
            "shape_gradient_maximum_log_step": float(
                shape_log_step.abs().max().item()
            ),
            "distance_edge_roughness": self._edge_absolute_roughness(
                self.distance_stiffness
            ),
            "shape_edge_roughness": self._edge_absolute_roughness(
                self.shape_stiffness
            ),
            "distance_log_graph_energy": self._graph_log_energy(
                self.distance_stiffness
            ),
            "shape_log_graph_energy": self._graph_log_energy(
                self.shape_stiffness
            ),
            "distance_minimum": float(self.distance_stiffness.min().item()),
            "distance_median": float(self.distance_stiffness.median().item()),
            "distance_maximum": float(self.distance_stiffness.max().item()),
            "shape_minimum": float(self.shape_stiffness.min().item()),
            "shape_median": float(self.shape_stiffness.median().item()),
            "shape_maximum": float(self.shape_stiffness.max().item()),
        }
        evidence = PaperStiffnessEvidence(
            signal_ema=signal_ema.detach().clone(),
            log_step=log_step.detach().clone(),
            distance_signal_ema=self.distance_signal_ema.detach().clone(),
            shape_signal_ema=self.shape_signal_ema.detach().clone(),
            distance_gradient_log_step=distance_log_step.detach().clone(),
            shape_gradient_log_step=shape_log_step.detach().clone(),
            eligible_mask=eligible.detach().clone(),
            material_valid_mask=material_valid.detach().clone(),
            source_mask=material_active.detach().clone(),
            metrics=metrics,
        )
        # Do not replace the user-facing pending-candidate diagnostics while
        # fresh frames continue to feed the evidence accumulator.  The live
        # EMA above is still updated and will seed the *next* proposal after
        # the frozen candidate is accepted or rejected.
        if self.pending_candidate is None:
            self.last_metrics = metrics
        return evidence

    def _edge_absolute_roughness(self, values: torch.Tensor) -> float:
        if not self.edges.numel():
            return 0.0
        return float(
            torch.mean(
                torch.abs(
                    values[self.edges[:, 1]] - values[self.edges[:, 0]]
                )
            ).item()
        )

    def _candidate_from_direction(
        self,
        *,
        base_log_step: torch.Tensor,
        signal_ema: torch.Tensor,
        distance_signal_ema: torch.Tensor,
        shape_signal_ema: torch.Tensor,
        base_distance_log_step: torch.Tensor,
        base_shape_log_step: torch.Tensor,
        eligible_mask: torch.Tensor,
        material_valid_mask: torch.Tensor,
        source_mask: torch.Tensor,
        base_metrics: dict[str, float | int | str],
        distance_scale: float,
        shape_scale: float,
        variant_label: str,
        scope: str = "full",
    ) -> PaperStiffnessCandidate:
        """Build one signed, independently parameterized material candidate."""
        settings = self.settings
        scope_mask = self._candidate_scope_mask(
            base_log_step,
            eligible_mask,
            scope,
        )
        distance_log_step = torch.clamp(
            base_distance_log_step.masked_fill(~scope_mask, 0.0)
            * float(distance_scale),
            min=-settings.maximum_log_step,
            max=settings.maximum_log_step,
        )
        shape_log_step = torch.clamp(
            base_shape_log_step.masked_fill(~scope_mask, 0.0)
            * float(shape_scale),
            min=-settings.maximum_log_step,
            max=settings.maximum_log_step,
        )
        distance_log_step[~eligible_mask] = 0.0
        shape_log_step[~eligible_mask] = 0.0
        distance_seed = distance_log_step.abs() > 1.0e-10
        shape_seed = shape_log_step.abs() > 1.0e-10
        # ``evidence.source_mask`` describes only the newest observation.  The
        # candidate, however, is built from the full EMA and can therefore
        # change many older contributors.  Freezing only the newest source
        # meant a rejected candidate left most of the bad EMA untouched; when
        # the newest frame had no active node, rejection decayed nothing and
        # the identical stale proposal was retried.  Track the exact proposal
        # contributors instead.  Evidence accumulated after proposal is not in
        # this frozen mask and consequently survives rejection as intended.
        proposal_source_mask = distance_seed | shape_seed
        proposal_source_count = int(
            torch.count_nonzero(proposal_source_mask).item()
        )
        parent_source_count = int(
            base_metrics.get(
                "proposal_parent_source_particles",
                base_metrics.get(
                    "proposal_source_particles", proposal_source_count
                ),
            )
        )
        with torch.no_grad():
            raw_distance = torch.clamp(
                self.distance_stiffness.detach()
                * torch.exp(distance_log_step),
                settings.distance_minimum,
                settings.distance_maximum,
            )
            raw_shape = torch.clamp(
                self.shape_stiffness.detach() * torch.exp(shape_log_step),
                settings.shape_minimum,
                settings.shape_maximum,
            )
            candidate_distance = self._smooth_stiffness_field(
                verified=self.distance_stiffness.detach(),
                raw_candidate=raw_distance,
                seed_mask=distance_seed,
                eligible_mask=eligible_mask,
            ).clamp(settings.distance_minimum, settings.distance_maximum)
            candidate_shape = self._smooth_stiffness_field(
                verified=self.shape_stiffness.detach(),
                raw_candidate=raw_shape,
                seed_mask=shape_seed,
                eligible_mask=eligible_mask,
            ).clamp(settings.shape_minimum, settings.shape_maximum)
            (
                candidate_distance,
                realized_distance_step,
                pre_cap_distance_step,
            ) = self._enforce_realized_log_step_cap(
                verified=self.distance_stiffness.detach(),
                candidate=candidate_distance,
                minimum=settings.distance_minimum,
                maximum=settings.distance_maximum,
            )
            (
                candidate_shape,
                realized_shape_step,
                pre_cap_shape_step,
            ) = self._enforce_realized_log_step_cap(
                verified=self.shape_stiffness.detach(),
                candidate=candidate_shape,
                minimum=settings.shape_minimum,
                maximum=settings.shape_maximum,
            )
        combined_step = torch.maximum(
            realized_distance_step.abs(), realized_shape_step.abs()
        )
        metrics = dict(base_metrics)
        metrics.update(
            status="candidate",
            candidate_variant=variant_label,
            distance_step_scale=float(distance_scale),
            shape_step_scale=float(shape_scale),
            maximum_log_step=float(combined_step.max().item()),
            mean_absolute_log_step=float(combined_step.mean().item()),
            distance_maximum_log_step=float(
                realized_distance_step.abs().max().item()
            ),
            shape_maximum_log_step=float(
                realized_shape_step.abs().max().item()
            ),
            post_smoothing_pre_cap_maximum_log_step=float(
                torch.maximum(
                    pre_cap_distance_step.abs(), pre_cap_shape_step.abs()
                ).max().item()
            ),
            post_smoothing_step_cap=float(settings.maximum_log_step),
            post_smoothing_step_cap_clipped_particles=int(
                torch.count_nonzero(
                    (pre_cap_distance_step.abs() > settings.maximum_log_step)
                    | (pre_cap_shape_step.abs() > settings.maximum_log_step)
                ).item()
            ),
            distance_hardening_particles=int(
                torch.count_nonzero(realized_distance_step > 1.0e-10).item()
            ),
            distance_softening_particles=int(
                torch.count_nonzero(realized_distance_step < -1.0e-10).item()
            ),
            shape_hardening_particles=int(
                torch.count_nonzero(realized_shape_step > 1.0e-10).item()
            ),
            shape_softening_particles=int(
                torch.count_nonzero(realized_shape_step < -1.0e-10).item()
            ),
            distance_edge_roughness=self._edge_absolute_roughness(
                candidate_distance
            ),
            shape_edge_roughness=self._edge_absolute_roughness(candidate_shape),
            distance_log_graph_energy=self._graph_log_energy(
                candidate_distance
            ),
            shape_log_graph_energy=self._graph_log_energy(candidate_shape),
            graph_smoothing_iterations=settings.graph_smoothing_iterations,
            graph_smoothing_blend=settings.graph_smoothing_blend,
            candidate_scope=scope,
            candidate_scope_particles=int(
                torch.count_nonzero(scope_mask).item()
            ),
            latest_observation_source_particles=int(
                torch.count_nonzero(source_mask).item()
            ),
            proposal_source_particles=proposal_source_count,
            proposal_parent_source_particles=max(
                proposal_source_count, parent_source_count
            ),
            distance_minimum=float(candidate_distance.min().item()),
            distance_median=float(candidate_distance.median().item()),
            distance_maximum=float(candidate_distance.max().item()),
            shape_minimum=float(candidate_shape.min().item()),
            shape_median=float(candidate_shape.median().item()),
            shape_maximum=float(candidate_shape.max().item()),
        )
        return PaperStiffnessCandidate(
            distance_stiffness=candidate_distance.detach().clone(),
            shape_stiffness=candidate_shape.detach().clone(),
            signal_ema=signal_ema.detach().clone(),
            log_step=base_log_step.detach().clone(),
            distance_signal_ema=distance_signal_ema.detach().clone(),
            shape_signal_ema=shape_signal_ema.detach().clone(),
            distance_gradient_log_step=base_distance_log_step.detach().clone(),
            shape_gradient_log_step=base_shape_log_step.detach().clone(),
            distance_log_step=realized_distance_step.detach().clone(),
            shape_log_step=realized_shape_step.detach().clone(),
            eligible_mask=eligible_mask.detach().clone(),
            material_valid_mask=material_valid_mask.detach().clone(),
            source_mask=proposal_source_mask.detach().clone(),
            metrics=metrics,
        )

    def propose_from_evidence(
        self,
        evidence: PaperStiffnessEvidence,
        *,
        material_axis: str = "joint",
    ) -> PaperStiffnessCandidate:
        """Freeze accumulated evidence into a joint or single-axis proposal."""
        if self.pending_candidate is not None:
            raise RuntimeError(
                "A stiffness candidate is already awaiting verification"
            )
        if material_axis == "joint":
            base_log_step = evidence.log_step
            distance_scale = 1.0
            shape_scale = 1.0
            variant_label = "dual_independent_gradient"
        elif material_axis == "distance":
            base_log_step = evidence.distance_gradient_log_step
            distance_scale = 1.0
            shape_scale = 0.0
            variant_label = "distance_continuous_gradient"
        elif material_axis == "shape":
            base_log_step = evidence.shape_gradient_log_step
            distance_scale = 0.0
            shape_scale = 1.0
            variant_label = "shape_continuous_gradient"
        else:
            raise ValueError(f"Unknown stiffness material axis: {material_axis}")
        self.candidate_count += 1
        metrics = dict(evidence.metrics)
        metrics["candidate_count"] = self.candidate_count
        metrics["evidence_count_at_proposal"] = self.evidence_count
        metrics["selected_material_axis"] = material_axis
        candidate = self._candidate_from_direction(
            base_log_step=base_log_step,
            signal_ema=evidence.signal_ema,
            distance_signal_ema=evidence.distance_signal_ema,
            shape_signal_ema=evidence.shape_signal_ema,
            base_distance_log_step=evidence.distance_gradient_log_step,
            base_shape_log_step=evidence.shape_gradient_log_step,
            eligible_mask=evidence.eligible_mask,
            material_valid_mask=evidence.material_valid_mask,
            source_mask=evidence.source_mask,
            base_metrics=metrics,
            distance_scale=distance_scale,
            shape_scale=shape_scale,
            variant_label=variant_label,
            scope="full",
        )
        self.pending_candidate = candidate
        self.last_metrics = candidate.metrics
        return candidate

    def candidate_variant(
        self,
        candidate: PaperStiffnessCandidate,
        *,
        distance_scale: float,
        shape_scale: float,
        variant_label: str,
        scope: str = "full",
    ) -> PaperStiffnessCandidate:
        """Create a signed distance/shape variant without changing verified k."""
        return self._candidate_from_direction(
            base_log_step=candidate.log_step,
            signal_ema=candidate.signal_ema,
            distance_signal_ema=getattr(
                candidate, "distance_signal_ema", candidate.signal_ema
            ),
            shape_signal_ema=getattr(
                candidate, "shape_signal_ema", candidate.signal_ema
            ),
            base_distance_log_step=getattr(
                candidate, "distance_gradient_log_step", candidate.log_step
            ),
            base_shape_log_step=getattr(
                candidate,
                "shape_gradient_log_step",
                candidate.log_step * self.settings.shape_update_gain,
            ),
            eligible_mask=candidate.eligible_mask,
            material_valid_mask=getattr(
                candidate, "material_valid_mask", candidate.eligible_mask
            ),
            source_mask=candidate.source_mask,
            base_metrics=candidate.metrics,
            distance_scale=distance_scale,
            shape_scale=shape_scale,
            variant_label=variant_label,
            scope=scope,
        )

    def global_median_candidate(
        self,
        candidate: PaperStiffnessCandidate,
        *,
        distance_median: float | None = None,
        shape_median: float | None = None,
        variant_label: str,
    ) -> PaperStiffnessCandidate:
        """Shift the global log-material offset while preserving local contrast.

        The legacy updater can only modify nodes with a visible residual.  Its
        median therefore stays near the initialization and cannot recover from
        a deliberately very soft or very hard starting point.  This candidate
        implements the global level of a hierarchical system identifier:

        ``log(k_i') = log(k_i) + log(k_target / median(k))``.

        The same offset is applied to every physically legal material node, so
        existing local log-space deviations are retained.  The candidate is
        still tentative and must pass the exact same residual-off multi-horizon
        shadow rollout and physical safety gates as every local proposal.
        """
        if distance_median is None and shape_median is None:
            raise ValueError("A global material candidate needs a target")
        settings = self.settings
        material_mask = candidate.material_valid_mask & ~self.fixed_mask
        if not bool(material_mask.any().item()):
            material_mask = candidate.eligible_mask & ~self.fixed_mask
        with torch.no_grad():
            distance = self.distance_stiffness.detach().clone()
            shape = self.shape_stiffness.detach().clone()
            if distance_median is not None:
                target = float(distance_median)
                if not settings.distance_minimum <= target <= settings.distance_maximum:
                    raise ValueError("Global distance target is outside bounds")
                current = float(distance[material_mask].median().item())
                distance_step = float(
                    max(
                        -settings.maximum_log_step,
                        min(
                            settings.maximum_log_step,
                            math.log(target / max(current, 1.0e-12)),
                        ),
                    )
                )
                distance[material_mask] = torch.clamp(
                    distance[material_mask] * math.exp(distance_step),
                    settings.distance_minimum,
                    settings.distance_maximum,
                )
            if shape_median is not None:
                target = float(shape_median)
                if not settings.shape_minimum <= target <= settings.shape_maximum:
                    raise ValueError("Global shape target is outside bounds")
                current = float(shape[material_mask].median().item())
                shape_step = float(
                    max(
                        -settings.maximum_log_step,
                        min(
                            settings.maximum_log_step,
                            math.log(target / max(current, 1.0e-12)),
                        ),
                    )
                )
                shape[material_mask] = torch.clamp(
                    shape[material_mask] * math.exp(shape_step),
                    settings.shape_minimum,
                    settings.shape_maximum,
                )
            distance_step = torch.log(
                distance / self.distance_stiffness.detach().clamp_min(1.0e-12)
            )
            shape_step = torch.log(
                shape / self.shape_stiffness.detach().clamp_min(1.0e-12)
            )
        source_mask = candidate.source_mask.detach().clone()
        combined_step = torch.maximum(distance_step.abs(), shape_step.abs())
        metrics = dict(candidate.metrics)
        metrics.update(
            status="candidate",
            candidate_variant=variant_label,
            candidate_scope="global_material_offset",
            candidate_scope_particles=int(
                torch.count_nonzero(material_mask).item()
            ),
            global_distance_target=(
                "unchanged" if distance_median is None else float(distance_median)
            ),
            global_shape_target=(
                "unchanged" if shape_median is None else float(shape_median)
            ),
            maximum_log_step=float(combined_step.max().item()),
            mean_absolute_log_step=float(combined_step.mean().item()),
            distance_maximum_log_step=float(distance_step.abs().max().item()),
            shape_maximum_log_step=float(shape_step.abs().max().item()),
            distance_hardening_particles=int(
                torch.count_nonzero(distance_step > 1.0e-10).item()
            ),
            distance_softening_particles=int(
                torch.count_nonzero(distance_step < -1.0e-10).item()
            ),
            shape_hardening_particles=int(
                torch.count_nonzero(shape_step > 1.0e-10).item()
            ),
            shape_softening_particles=int(
                torch.count_nonzero(shape_step < -1.0e-10).item()
            ),
            distance_edge_roughness=self._edge_absolute_roughness(distance),
            shape_edge_roughness=self._edge_absolute_roughness(shape),
            distance_log_graph_energy=self._graph_log_energy(distance),
            shape_log_graph_energy=self._graph_log_energy(shape),
            distance_minimum=float(distance.min().item()),
            distance_median=float(distance.median().item()),
            distance_maximum=float(distance.max().item()),
            shape_minimum=float(shape.min().item()),
            shape_median=float(shape.median().item()),
            shape_maximum=float(shape.max().item()),
        )
        return PaperStiffnessCandidate(
            distance_stiffness=distance,
            shape_stiffness=shape,
            signal_ema=candidate.signal_ema.detach().clone(),
            log_step=candidate.log_step.detach().clone(),
            distance_signal_ema=candidate.distance_signal_ema.detach().clone(),
            shape_signal_ema=candidate.shape_signal_ema.detach().clone(),
            distance_gradient_log_step=(
                candidate.distance_gradient_log_step.detach().clone()
            ),
            shape_gradient_log_step=(
                candidate.shape_gradient_log_step.detach().clone()
            ),
            distance_log_step=distance_step.detach().clone(),
            shape_log_step=shape_step.detach().clone(),
            eligible_mask=candidate.eligible_mask.detach().clone(),
            material_valid_mask=material_mask.detach().clone(),
            source_mask=source_mask,
            metrics=metrics,
        )

    def merge_candidate_regions(
        self,
        candidates: tuple[PaperStiffnessCandidate, ...],
        *,
        variant_label: str,
    ) -> PaperStiffnessCandidate:
        """Merge independently proposed local regions into one candidate.

        All inputs must change the same single material family.  Each node
        takes the largest-magnitude realized log step among the inputs, so
        overlapping graph-smoothing fringes are never added twice.  The merged
        field is still tentative and must pass the same residual-off rollout,
        image, volume, penetration, anchor, and history gates.
        """
        if len(candidates) < 2:
            raise ValueError("At least two stiffness regions are required")
        reference = candidates[0]
        material_axes: set[str] = set()
        for candidate in candidates[1:]:
            if not torch.equal(candidate.eligible_mask, reference.eligible_mask):
                raise ValueError("Merged candidates use different eligibility")
            if not torch.equal(
                candidate.material_valid_mask, reference.material_valid_mask
            ):
                raise ValueError("Merged candidates use different material masks")
            if not torch.equal(candidate.log_step, reference.log_step):
                raise ValueError("Merged candidates use different evidence")
        for candidate in candidates:
            changes_distance = bool(
                torch.any(candidate.distance_log_step.abs() > 1.0e-10).item()
            )
            changes_shape = bool(
                torch.any(candidate.shape_log_step.abs() > 1.0e-10).item()
            )
            if changes_distance == changes_shape:
                raise ValueError(
                    "Merged candidates must each change exactly one material axis"
                )
            material_axes.add("distance" if changes_distance else "shape")
        if len(material_axes) != 1:
            raise ValueError(
                "Merged candidates must use the same material axis"
            )
        merged_material_axis = next(iter(material_axes))

        distance_step = torch.zeros_like(reference.distance_log_step)
        shape_step = torch.zeros_like(reference.shape_log_step)
        source_mask = torch.zeros_like(reference.source_mask)
        component_scopes: list[str] = []
        for candidate in candidates:
            use_distance = (
                candidate.distance_log_step.abs() > distance_step.abs()
            )
            use_shape = candidate.shape_log_step.abs() > shape_step.abs()
            distance_step = torch.where(
                use_distance, candidate.distance_log_step, distance_step
            )
            shape_step = torch.where(
                use_shape, candidate.shape_log_step, shape_step
            )
            source_mask |= candidate.source_mask
            component_scopes.append(
                str(candidate.metrics.get("candidate_scope", "unknown"))
            )

        settings = self.settings
        with torch.no_grad():
            distance = torch.clamp(
                self.distance_stiffness.detach() * torch.exp(distance_step),
                settings.distance_minimum,
                settings.distance_maximum,
            )
            shape = torch.clamp(
                self.shape_stiffness.detach() * torch.exp(shape_step),
                settings.shape_minimum,
                settings.shape_maximum,
            )
        combined_step = torch.maximum(distance_step.abs(), shape_step.abs())
        metrics = dict(reference.metrics)
        metrics.update(
            status="candidate",
            candidate_variant=variant_label,
            candidate_scope="merged_components",
            merged_candidate_scopes=tuple(component_scopes),
            merged_candidate_count=len(candidates),
            merged_material_axis=merged_material_axis,
            candidate_scope_particles=int(
                torch.count_nonzero(source_mask).item()
            ),
            proposal_source_particles=int(
                torch.count_nonzero(source_mask).item()
            ),
            maximum_log_step=float(combined_step.max().item()),
            mean_absolute_log_step=float(combined_step.mean().item()),
            distance_maximum_log_step=float(distance_step.abs().max().item()),
            shape_maximum_log_step=float(shape_step.abs().max().item()),
            distance_hardening_particles=int(
                torch.count_nonzero(distance_step > 1.0e-10).item()
            ),
            distance_softening_particles=int(
                torch.count_nonzero(distance_step < -1.0e-10).item()
            ),
            shape_hardening_particles=int(
                torch.count_nonzero(shape_step > 1.0e-10).item()
            ),
            shape_softening_particles=int(
                torch.count_nonzero(shape_step < -1.0e-10).item()
            ),
            distance_edge_roughness=self._edge_absolute_roughness(distance),
            shape_edge_roughness=self._edge_absolute_roughness(shape),
            distance_log_graph_energy=self._graph_log_energy(distance),
            shape_log_graph_energy=self._graph_log_energy(shape),
            distance_minimum=float(distance.min().item()),
            distance_median=float(distance.median().item()),
            distance_maximum=float(distance.max().item()),
            shape_minimum=float(shape.min().item()),
            shape_median=float(shape.median().item()),
            shape_maximum=float(shape.max().item()),
        )
        return PaperStiffnessCandidate(
            distance_stiffness=distance.detach().clone(),
            shape_stiffness=shape.detach().clone(),
            signal_ema=reference.signal_ema.detach().clone(),
            log_step=reference.log_step.detach().clone(),
            distance_signal_ema=reference.distance_signal_ema.detach().clone(),
            shape_signal_ema=reference.shape_signal_ema.detach().clone(),
            distance_gradient_log_step=(
                reference.distance_gradient_log_step.detach().clone()
            ),
            shape_gradient_log_step=(
                reference.shape_gradient_log_step.detach().clone()
            ),
            distance_log_step=distance_step.detach().clone(),
            shape_log_step=shape_step.detach().clone(),
            eligible_mask=reference.eligible_mask.detach().clone(),
            material_valid_mask=reference.material_valid_mask.detach().clone(),
            source_mask=source_mask.detach().clone(),
            metrics=metrics,
        )

    def propose(
        self,
        *,
        physical_prediction: torch.Tensor,
        accepted_residual: torch.Tensor,
        quality_valid_mask: torch.Tensor | None = None,
        supervision_valid_mask: torch.Tensor | None = None,
        control_exclusion_mask: torch.Tensor | None = None,
        globally_paused: bool = False,
    ) -> PaperStiffnessCandidate:
        """Compatibility helper: accumulate one observation, then propose."""
        evidence = self.accumulate_evidence(
            physical_prediction=physical_prediction,
            accepted_residual=accepted_residual,
            quality_valid_mask=quality_valid_mask,
            supervision_valid_mask=supervision_valid_mask,
            control_exclusion_mask=control_exclusion_mask,
            globally_paused=globally_paused,
        )
        return self.propose_from_evidence(evidence)

    def install_candidate_for_rollout(
        self, candidate: PaperStiffnessCandidate | None = None
    ) -> None:
        """Temporarily expose a candidate to XPBD for an isolated rollout."""
        candidate = candidate or self.pending_candidate
        if candidate is None:
            raise RuntimeError("No stiffness candidate is available")
        with torch.no_grad():
            self.distance_stiffness.copy_(candidate.distance_stiffness)
            self.shape_stiffness.copy_(candidate.shape_stiffness)

    def restore_verified_stiffness(
        self,
        distance: torch.Tensor,
        shape: torch.Tensor,
    ) -> None:
        """Restore explicitly saved verified arrays after a shadow rollout."""
        with torch.no_grad():
            self.distance_stiffness.copy_(distance)
            self.shape_stiffness.copy_(shape)

    def absolute_adam_candidate(
        self,
        *,
        distance_stiffness: torch.Tensor,
        shape_stiffness: torch.Tensor,
        active_mask: torch.Tensor,
        optimizer_metrics: dict[str, float | int | str],
    ) -> PaperStiffnessCandidate:
        """Wrap one bounded Adam result in the normal material transaction.

        The Liang-style optimizer works in absolute sigmoid-bounded logits and
        intentionally has no legacy fixed log-step proposal.  This adapter
        preserves one source of truth for material installation, counters, and
        diagnostics without reapplying graph smoothing or a trust-region cap.
        """
        if self.pending_candidate is not None:
            raise RuntimeError(
                "A stiffness candidate is already awaiting verification"
            )
        distance = distance_stiffness.detach().to(
            device=self.distance_stiffness.device,
            dtype=self.distance_stiffness.dtype,
        )
        shape = shape_stiffness.detach().to(
            device=self.shape_stiffness.device,
            dtype=self.shape_stiffness.dtype,
        )
        active = active_mask.detach().to(
            device=self.fixed_mask.device, dtype=torch.bool
        ) & ~self.fixed_mask
        if distance.shape != self.distance_stiffness.shape:
            raise ValueError("Adam distance field has the wrong shape")
        if shape.shape != self.shape_stiffness.shape:
            raise ValueError("Adam shape field has the wrong shape")
        if active.shape != self.fixed_mask.shape:
            raise ValueError("Adam active mask has the wrong shape")
        if not torch.isfinite(distance).all() or not torch.isfinite(shape).all():
            raise ValueError("Adam material field contains non-finite values")
        settings = self.settings
        tolerance = 4.0 * torch.finfo(distance.dtype).eps
        if (
            float(distance.min().item()) < settings.distance_minimum - tolerance
            or float(distance.max().item()) > settings.distance_maximum + tolerance
            or float(shape.min().item()) < settings.shape_minimum - tolerance
            or float(shape.max().item()) > settings.shape_maximum + tolerance
        ):
            raise ValueError("Adam material field lies outside configured bounds")
        distance_log_step = torch.log(
            distance / self.distance_stiffness.detach().clamp_min(1.0e-12)
        )
        shape_log_step = torch.log(
            shape / self.shape_stiffness.detach().clamp_min(1.0e-12)
        )
        distance_log_step[~active] = 0.0
        shape_log_step[~active] = 0.0
        # Fixed/inactive nodes are exactly preserved even if sigmoid
        # round-tripping differs by one float32 ULP.
        distance = distance.clone()
        shape = shape.clone()
        distance[~active] = self.distance_stiffness.detach()[~active]
        shape[~active] = self.shape_stiffness.detach()[~active]
        source_mask = (
            (distance_log_step.abs() > 1.0e-12)
            | (shape_log_step.abs() > 1.0e-12)
        )
        combined_step = torch.maximum(
            distance_log_step.abs(), shape_log_step.abs()
        )
        self.candidate_count += 1
        metrics = dict(optimizer_metrics)
        metrics.update(
            status="candidate",
            candidate_count=self.candidate_count,
            candidate_variant="paper_track_history_smooth_adam",
            candidate_scope="all_material_valid_particles",
            candidate_scope_particles=int(torch.count_nonzero(active).item()),
            proposal_source_particles=int(
                torch.count_nonzero(source_mask).item()
            ),
            maximum_log_step=float(combined_step.max().item()),
            mean_absolute_log_step=float(combined_step.mean().item()),
            distance_maximum_log_step=float(
                distance_log_step.abs().max().item()
            ),
            shape_maximum_log_step=float(shape_log_step.abs().max().item()),
            distance_minimum=float(distance.min().item()),
            distance_median=float(distance.median().item()),
            distance_maximum=float(distance.max().item()),
            shape_minimum=float(shape.min().item()),
            shape_median=float(shape.median().item()),
            shape_maximum=float(shape.max().item()),
            distance_log_graph_energy=self._graph_log_energy(distance),
            shape_log_graph_energy=self._graph_log_energy(shape),
            graph_smoothing_iterations=0,
            graph_smoothing_blend=0.0,
            bounded_parameterization="sigmoid",
            fixed_log_step_cap_enabled=0,
        )
        zeros = torch.zeros_like(distance_log_step)
        candidate = PaperStiffnessCandidate(
            distance_stiffness=distance,
            shape_stiffness=shape,
            signal_ema=zeros.clone(),
            log_step=torch.where(
                distance_log_step.abs() >= shape_log_step.abs(),
                distance_log_step,
                shape_log_step,
            ),
            distance_signal_ema=zeros.clone(),
            shape_signal_ema=zeros.clone(),
            distance_gradient_log_step=distance_log_step.detach().clone(),
            shape_gradient_log_step=shape_log_step.detach().clone(),
            distance_log_step=distance_log_step.detach().clone(),
            shape_log_step=shape_log_step.detach().clone(),
            eligible_mask=active.detach().clone(),
            material_valid_mask=active.detach().clone(),
            source_mask=source_mask.detach().clone(),
            metrics=metrics,
        )
        self.pending_candidate = candidate
        self.last_metrics = metrics
        return candidate

    def commit(
        self, candidate: PaperStiffnessCandidate | None = None
    ) -> dict[str, float | int | str]:
        candidate = candidate or self.pending_candidate
        if candidate is None or candidate is not self.pending_candidate:
            raise RuntimeError("Cannot commit an unknown stiffness candidate")
        self.install_candidate_for_rollout(candidate)
        # Do not restore the proposal-time EMA snapshot here. New accepted
        # residuals may have accumulated while this proposal waited for its
        # H1/H5/H10 causal validation window.
        distance_changed = bool(
            torch.any(candidate.distance_log_step.abs() > 1.0e-10).item()
        )
        shape_changed = bool(
            torch.any(candidate.shape_log_step.abs() > 1.0e-10).item()
        )
        self.update_count += 1
        self.distance_update_count += int(distance_changed)
        self.shape_update_count += int(shape_changed)
        metrics = dict(candidate.metrics)
        metrics.update(
            status="committed",
            update_count=self.update_count,
            distance_update_count=self.distance_update_count,
            shape_update_count=self.shape_update_count,
            distance_axis_changed=int(distance_changed),
            shape_axis_changed=int(shape_changed),
            rejected_count=self.rejected_count,
            evidence_count=self.evidence_count,
            evidence_updates_while_pending=max(
                0,
                self.evidence_count
                - int(metrics.get("evidence_count_at_proposal", 0)),
            ),
        )
        self.pending_candidate = None
        self.last_metrics = metrics
        return metrics

    def reject(
        self,
        reason: str,
        candidate: PaperStiffnessCandidate | None = None,
    ) -> dict[str, float | int | str]:
        candidate = candidate or self.pending_candidate
        if candidate is None or candidate is not self.pending_candidate:
            raise RuntimeError("Cannot reject an unknown stiffness candidate")
        # Decay only the evidence region that generated the rejected proposal.
        # Other nodes may contain newer observations gathered during the
        # pending horizon and must survive.
        with torch.no_grad():
            self.distance_signal_ema[candidate.source_mask] *= (
                self.settings.rejected_ema_decay
            )
            self.shape_signal_ema[candidate.source_mask] *= (
                self.settings.rejected_ema_decay
            )
        self.rejected_count += 1
        for rejection_reason in str(reason).split("+"):
            self.rejection_reason_counts[rejection_reason] = (
                self.rejection_reason_counts.get(rejection_reason, 0) + 1
            )
        metrics = dict(candidate.metrics)
        metrics.update(
            status="rejected",
            rejection_reason=str(reason),
            update_count=self.update_count,
            rejected_count=self.rejected_count,
            evidence_count=self.evidence_count,
            evidence_updates_while_pending=max(
                0,
                self.evidence_count
                - int(metrics.get("evidence_count_at_proposal", 0)),
            ),
        )
        self.pending_candidate = None
        self.last_metrics = metrics
        return metrics

    def defer(
        self,
        reason: str,
        candidate: PaperStiffnessCandidate | None = None,
    ) -> dict[str, float | int | str]:
        """Close a safe proposal window without decaying its evidence.

        This is intentionally narrower than ``reject``.  It is used when an
        otherwise admitted global candidate is waiting for a second
        independent proposal window.  No material is installed, no update or
        rejection counter changes, and the accumulated strain/RGB evidence is
        preserved so consensus remains reachable.
        """
        candidate = candidate or self.pending_candidate
        if candidate is None or candidate is not self.pending_candidate:
            raise RuntimeError("Cannot defer an unknown stiffness candidate")
        metrics = dict(candidate.metrics)
        metrics.update(
            status="deferred",
            deferral_reason=str(reason),
            update_count=self.update_count,
            rejected_count=self.rejected_count,
            evidence_count=self.evidence_count,
            evidence_updates_while_pending=max(
                0,
                self.evidence_count
                - int(metrics.get("evidence_count_at_proposal", 0)),
            ),
        )
        self.pending_candidate = None
        self.last_metrics = metrics
        return metrics

    def update(
        self,
        *,
        physical_prediction: torch.Tensor,
        accepted_residual: torch.Tensor,
        quality_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, float | int | str]:
        """Legacy immediate update retained for isolated unit tests only."""
        candidate = self.propose(
            physical_prediction=physical_prediction,
            accepted_residual=accepted_residual,
            quality_valid_mask=quality_valid_mask,
        )
        return self.commit(candidate)
