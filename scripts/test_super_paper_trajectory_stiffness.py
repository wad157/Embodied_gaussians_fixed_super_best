#!/usr/bin/env python3
"""CPU regression tests for Liang-style trajectory Adam material updates."""

from __future__ import annotations

import torch

from embodied_gaussians.physics_simulator.paper_trajectory_stiffness import (
    PaperTrajectoryAdamOptimizer,
    PaperTrajectoryAdamSettings,
    confidence_weighted_huber_track_loss,
)
from embodied_gaussians.physics_simulator.online_tissue_stiffness import (
    OnlineTissueStiffnessSettings,
    ResidualDrivenPaperStiffnessUpdater,
)


def make_optimizer(
    *,
    minimum_difference: float = 1.0e-7,
    learning_rate: float = 0.02,
    maximum_material_log_step: float = 0.10,
):
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    fixed = torch.tensor([True, False, False, False, False, False])
    edges = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
    )
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=fixed,
        edges=edges,
        settings=PaperTrajectoryAdamSettings(
            learning_rate=learning_rate,
            maximum_material_log_step=maximum_material_log_step,
            parameter_perturbation=0.05,
            minimum_axis_loss_difference=minimum_difference,
            distance_minimum=1.0e-5,
            distance_maximum=4.0,
            shape_minimum=1.0e-6,
            shape_maximum=0.04,
        ),
    )
    return optimizer, distance, shape


def measured(total: float) -> dict[str, float]:
    return {
        "track_loss": total,
        "history_loss": 0.0,
        "distance_smooth_loss": 0.0,
        "shape_smooth_loss": 0.0,
        "total_loss": total,
    }


def test_independent_trials_and_bounds() -> None:
    optimizer, distance, shape = make_optimizer()
    active = torch.tensor([False, True, True, False, True, False])
    trials = {trial.label: trial for trial in optimizer.begin_observation(active)}
    assert set(trials) == {
        "distance_plus",
        "distance_minus",
        "shape_plus",
        "shape_minus",
    }
    assert torch.equal(trials["distance_plus"].shape_stiffness, shape)
    assert torch.equal(trials["shape_plus"].distance_stiffness, distance)
    for trial in trials.values():
        assert float(trial.distance_stiffness.min()) >= 1.0e-5
        assert float(trial.distance_stiffness.max()) <= 4.0
        assert float(trial.shape_stiffness.min()) >= 1.0e-6
        assert float(trial.shape_stiffness.max()) <= 0.04
        assert trial.distance_stiffness[0] == distance[0]
        assert trial.shape_stiffness[0] == shape[0]


def test_adam_follows_measured_counterfactual_gradient() -> None:
    optimizer, _, _ = make_optimizer()
    active = torch.tensor([False, True, True, True, True, True])
    optimizer.begin_observation(active)
    old_distance_theta = optimizer.distance_theta.clone()
    old_shape_theta = optimizer.shape_theta.clone()
    distance, shape, metrics = optimizer.finish_observation(
        {
            "distance_plus": measured(1.2),
            "distance_minus": measured(0.8),
            "shape_plus": measured(0.7),
            "shape_minus": measured(1.1),
        }
    )
    distance_delta = optimizer.distance_theta - old_distance_theta
    shape_delta = optimizer.shape_theta - old_shape_theta
    assert torch.sum(distance_delta * optimizer._distance_probe) < 0.0
    assert torch.sum(shape_delta * optimizer._shape_probe) > 0.0
    assert metrics["status"] == "adam_updated"
    assert metrics["distance_update_count"] == 1
    assert metrics["shape_update_count"] == 1
    assert torch.isfinite(distance).all() and torch.isfinite(shape).all()


def test_insensitive_axis_does_not_random_walk() -> None:
    optimizer, distance, shape = make_optimizer(minimum_difference=1.0e-3)
    active = torch.tensor([False, True, True, True, True, True])
    optimizer.begin_observation(active)
    new_distance, new_shape, metrics = optimizer.finish_observation(
        {
            "distance_plus": measured(1.00001),
            "distance_minus": measured(1.0),
            "shape_plus": measured(0.99999),
            "shape_minus": measured(1.0),
        }
    )
    assert metrics["status"] == "insensitive_observation"
    assert torch.allclose(new_distance, distance, atol=1.0e-8, rtol=0.0)
    assert torch.allclose(new_shape, shape, atol=1.0e-8, rtol=0.0)


def test_adam_momentum_step_is_reported_for_immediate_commit() -> None:
    optimizer, distance, shape = make_optimizer(minimum_difference=1.0e-3)
    active = torch.tensor([False, True, True, True, True, True])
    optimizer.begin_observation(active)
    first_distance, first_shape, _ = optimizer.finish_observation(
        {
            "distance_plus": measured(1.2),
            "distance_minus": measured(0.8),
            "shape_plus": measured(0.7),
            "shape_minus": measured(1.1),
        }
    )
    distance.copy_(first_distance)
    shape.copy_(first_shape)
    optimizer.begin_observation(active)
    second_distance, second_shape, metrics = optimizer.finish_observation(
        {
            "distance_plus": measured(1.00001),
            "distance_minus": measured(1.0),
            "shape_plus": measured(0.99999),
            "shape_minus": measured(1.0),
        }
    )
    assert metrics["distance_axis_sensitive"] == 0
    assert metrics["shape_axis_sensitive"] == 0
    assert metrics["status"] == "adam_updated"
    assert not torch.equal(second_distance, first_distance)
    assert not torch.equal(second_shape, first_shape)


def test_realized_material_step_is_capped_after_adam_momentum() -> None:
    optimizer, _, _ = make_optimizer(
        learning_rate=0.50,
        maximum_material_log_step=0.03,
    )
    active = torch.tensor([False, True, True, True, True, True])
    optimizer.begin_observation(active)
    old_distance, old_shape = optimizer.current_material()
    distance, shape, metrics = optimizer.finish_observation(
        {
            "distance_plus": measured(2.0),
            "distance_minus": measured(0.1),
            "shape_plus": measured(0.1),
            "shape_minus": measured(2.0),
        }
    )
    distance_step = torch.log(distance / old_distance)
    shape_step = torch.log(shape / old_shape)
    assert float(distance_step[active].abs().max()) <= 0.030001
    assert float(shape_step[active].abs().max()) <= 0.030001
    assert metrics["material_log_step_cap_enabled"] == 1
    assert metrics["maximum_material_log_step"] == 0.03


def test_confidence_weighted_huber_track_loss() -> None:
    predicted = torch.tensor([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]])
    target = torch.zeros_like(predicted)
    confidence = torch.tensor([1.0, 0.5])
    valid = torch.tensor([True, True])
    loss, metrics = confidence_weighted_huber_track_loss(
        predicted_points=predicted,
        target_points=target,
        confidence=confidence,
        valid_mask=valid,
        robust_scale_m=0.003,
    )
    assert abs(loss - (0.5 * 0.5 / 1.5)) < 1.0e-7
    assert metrics["valid_tracks"] == 2
    assert abs(metrics["confidence_sum"] - 1.5) < 1.0e-7


def test_absolute_adam_candidate_has_no_legacy_log_cap() -> None:
    optimizer, distance, shape = make_optimizer()
    active = torch.tensor([False, True, True, True, True, True])
    optimizer.begin_observation(active)
    new_distance, new_shape, optimizer_metrics = optimizer.finish_observation(
        {
            "distance_plus": measured(1.2),
            "distance_minus": measured(0.8),
            "shape_plus": measured(0.7),
            "shape_minus": measured(1.1),
        }
    )
    rest = torch.stack(
        [torch.arange(6, dtype=torch.float32), torch.zeros(6), torch.zeros(6)],
        dim=1,
    )
    updater = ResidualDrivenPaperStiffnessUpdater(
        rest_positions=rest,
        fixed_mask=torch.tensor([True, False, False, False, False, False]),
        edges=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        distance_stiffness=distance,
        shape_stiffness=shape,
        settings=OnlineTissueStiffnessSettings(
            maximum_log_step=0.001,
            distance_minimum=1.0e-5,
            distance_maximum=4.0,
            shape_minimum=1.0e-6,
            shape_maximum=0.04,
        ),
    )
    candidate = updater.absolute_adam_candidate(
        distance_stiffness=new_distance,
        shape_stiffness=new_shape,
        active_mask=active,
        optimizer_metrics=optimizer_metrics,
    )
    assert float(candidate.metrics["maximum_log_step"]) > 0.001
    assert candidate.metrics["fixed_log_step_cap_enabled"] == 0
    committed = updater.commit(candidate)
    assert committed["status"] == "committed"
    assert updater.update_count == 1


def test_sim_global_causal_updates_distance_and_damping_only() -> None:
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    fixed = torch.tensor([True, False, False, False, False, False])
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=fixed,
        edges=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        settings=PaperTrajectoryAdamSettings(
            sim_global_causal_mode=True,
            learning_rate=0.05,
            parameter_perturbation=0.05,
            minimum_axis_loss_difference=1.0e-9,
        ),
    )
    active = torch.tensor([False, True, True, False, False, False])
    trials = {trial.label: trial for trial in optimizer.begin_observation(active)}
    assert set(trials) == {
        "distance_plus",
        "distance_minus",
        "damping_plus",
        "damping_minus",
    }
    assert torch.equal(trials["distance_plus"].shape_stiffness, shape)
    # Global material identification modifies every dynamic material node,
    # not only the two particles that made this observation observable.
    assert trials["distance_plus"].distance_stiffness[5] != distance[5]

    def global_loss(total: float, horizons: tuple[float, float, float]):
        return {
            **measured(total),
            "surrogate_total_loss": total,
            "horizon_total_losses": horizons,
        }

    new_distance, new_shape, metrics = optimizer.finish_observation(
        {
            "distance_plus": global_loss(1.2, (1.1, 1.2, 1.3)),
            "distance_minus": global_loss(0.8, (0.9, 0.8, 0.7)),
            "damping_plus": global_loss(0.7, (0.8, 0.7, 0.6)),
            "damping_minus": global_loss(1.1, (1.0, 1.1, 1.2)),
        }
    )
    assert metrics["status"] == "sim_global_updated"
    assert metrics["gradient_direction_consistent"] == 1
    assert metrics["shape_frozen"] == 1
    assert torch.equal(new_shape, shape)
    assert not torch.equal(new_distance[~fixed], distance[~fixed])
    assert 2.0 <= optimizer.current_velocity_damping_per_second() <= 30.0


def test_sim_global_rejects_open_loop_one_step_gradient_disagreement() -> None:
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=torch.tensor([True, False, False, False, False, False]),
        edges=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        settings=PaperTrajectoryAdamSettings(
            sim_global_causal_mode=True,
            minimum_axis_loss_difference=1.0e-9,
        ),
    )
    optimizer.begin_observation(torch.tensor([False, True, True, True, True, True]))

    def loss(total: float, surrogate: float) -> dict[str, object]:
        return {
            **measured(total),
            "surrogate_total_loss": surrogate,
            "horizon_total_losses": (total, total, total),
        }

    new_distance, new_shape, metrics = optimizer.finish_observation(
        {
            "distance_plus": loss(1.2, 0.8),
            "distance_minus": loss(0.8, 1.2),
            "damping_plus": loss(0.7, 1.1),
            "damping_minus": loss(1.1, 0.7),
        }
    )
    assert metrics["status"] == "gradient_misaligned"
    assert metrics["update_count"] == 0
    assert torch.equal(new_distance, distance)
    assert torch.equal(new_shape, shape)


def test_sim_global_h2_scales_the_material_step_cap() -> None:
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=torch.tensor([True, False, False, False, False, False]),
        edges=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        settings=PaperTrajectoryAdamSettings(
            sim_global_causal_mode=True,
            learning_rate=0.50,
            maximum_material_log_step=0.02,
            minimum_axis_loss_difference=1.0e-9,
        ),
    )
    optimizer.begin_observation(torch.tensor([False, True, True, True, True, True]))

    def h2_loss(total: float, block: int) -> dict[str, object]:
        return {
            **measured(total),
            "surrogate_total_loss": total,
            "horizon_total_losses": (total, total),
            "causal_block_index": block,
        }

    _, _, pending = optimizer.finish_observation(
        {
            "distance_plus": h2_loss(1.2, 10),
            "distance_minus": h2_loss(0.8, 10),
            "damping_plus": h2_loss(0.7, 10),
            "damping_minus": h2_loss(1.1, 10),
        }
    )
    assert pending["status"] == "h2_awaiting_cross_block_confirmation"
    optimizer.begin_observation(
        torch.tensor([False, True, True, True, True, True])
    )
    _, _, metrics = optimizer.finish_observation(
        {
            "distance_plus": h2_loss(1.2, 11),
            "distance_minus": h2_loss(0.8, 11),
            "damping_plus": h2_loss(0.7, 11),
            "damping_minus": h2_loss(1.1, 11),
        }
    )
    expected_cap = 0.02 * (1.5 + 2.0) / (1.5 + 2.0 + 3.0)
    assert metrics["status"] == "sim_global_updated"
    assert metrics["causal_window_size"] == 2
    assert abs(metrics["effective_maximum_material_log_step"] - expected_cap) < 1.0e-9
    assert abs(metrics["distance_log_step"]) <= expected_cap + 1.0e-7
    assert abs(metrics["damping_log_step"]) <= expected_cap + 1.0e-7
    assert metrics["descent_direction"] == 1
    assert metrics["h2_cross_block_confirmed"] == 1
    assert metrics["h2_confirmation_source_block"] == 10


def test_sim_global_rejects_adam_momentum_ascent() -> None:
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=torch.tensor([True, False, False, False, False, False]),
        edges=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        settings=PaperTrajectoryAdamSettings(
            sim_global_causal_mode=True,
            minimum_axis_loss_difference=1.0e-9,
        ),
    )
    optimizer.begin_observation(torch.tensor([False, True, True, True, True, True]))

    def h2_loss(total: float, block: int) -> dict[str, object]:
        return {
            **measured(total),
            "surrogate_total_loss": total,
            "horizon_total_losses": (total, total),
            "causal_block_index": block,
        }

    _, _, pending = optimizer.finish_observation(
        {
            "distance_plus": h2_loss(1.2, 20),
            "distance_minus": h2_loss(0.8, 20),
            "damping_plus": h2_loss(1.2, 20),
            "damping_minus": h2_loss(0.8, 20),
        }
    )
    assert pending["status"] == "h2_awaiting_cross_block_confirmation"
    optimizer.global_first_moment.fill_(-1.0)
    optimizer.begin_observation(
        torch.tensor([False, True, True, True, True, True])
    )
    new_distance, new_shape, metrics = optimizer.finish_observation(
        {
            "distance_plus": h2_loss(1.2, 21),
            "distance_minus": h2_loss(0.8, 21),
            "damping_plus": h2_loss(1.2, 21),
            "damping_minus": h2_loss(0.8, 21),
        }
    )
    assert metrics["gradient_direction_consistent"] == 1
    assert metrics["descent_direction"] == 0
    assert metrics["status"] == "non_descent_adam_step"
    assert metrics["directional_derivative"] > 0.0
    assert optimizer.update_count == 0
    assert torch.equal(new_distance, distance)
    assert torch.equal(new_shape, shape)


def test_sim_global_h2_cumulative_offsets_are_bounded() -> None:
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=torch.tensor([True, False, False, False, False, False]),
        edges=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        settings=PaperTrajectoryAdamSettings(
            sim_global_causal_mode=True,
            learning_rate=0.50,
            maximum_material_log_step=0.02,
            h2_maximum_log_offset=0.10,
            minimum_axis_loss_difference=1.0e-9,
        ),
    )
    active = torch.tensor([False, True, True, True, True, True])

    def losses(block: int) -> dict[str, dict[str, object]]:
        def value(total: float) -> dict[str, object]:
            return {
                **measured(total),
                "surrogate_total_loss": total,
                "horizon_total_losses": (total, total),
                "causal_block_index": block,
            }

        return {
            "distance_plus": value(1.2),
            "distance_minus": value(0.8),
            "damping_plus": value(1.2),
            "damping_minus": value(0.8),
        }

    for block in range(30):
        optimizer.begin_observation(active)
        optimizer.finish_observation(losses(block))
    assert float(optimizer.global_log_coefficients.abs().max().item()) <= 0.100001


def test_sim_global_h2_rejects_cross_block_gradient_reversal() -> None:
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=torch.tensor([True, False, False, False, False, False]),
        edges=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        settings=PaperTrajectoryAdamSettings(
            sim_global_causal_mode=True,
            minimum_axis_loss_difference=1.0e-9,
        ),
    )
    active = torch.tensor([False, True, True, True, True, True])

    def losses(block: int, reverse: bool) -> dict[str, dict[str, object]]:
        plus, minus = ((0.8, 1.2) if reverse else (1.2, 0.8))

        def value(total: float) -> dict[str, object]:
            return {
                **measured(total),
                "surrogate_total_loss": total,
                "horizon_total_losses": (total, total),
                "causal_block_index": block,
            }

        return {
            "distance_plus": value(plus),
            "distance_minus": value(minus),
            "damping_plus": value(plus),
            "damping_minus": value(minus),
        }

    optimizer.begin_observation(active)
    _, _, first = optimizer.finish_observation(losses(30, False))
    optimizer.begin_observation(active)
    new_distance, new_shape, second = optimizer.finish_observation(
        losses(31, True)
    )
    assert first["status"] == "h2_awaiting_cross_block_confirmation"
    assert second["status"] == "h2_cross_block_misaligned"
    assert second["h2_cross_block_confirmed"] == 0
    assert second["h2_cross_block_cosine"] < 0.0
    assert optimizer.update_count == 0
    assert torch.equal(new_distance, distance)
    assert torch.equal(new_shape, shape)


def test_h2_does_not_snap_h3_coefficient_to_tighter_bound() -> None:
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    settings = PaperTrajectoryAdamSettings(
        sim_global_causal_mode=True,
        learning_rate=0.50,
        maximum_material_log_step=0.02,
        h2_maximum_log_offset=0.10,
        minimum_axis_loss_difference=1.0e-9,
    )
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=torch.tensor([True, False, False, False, False, False]),
        edges=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        settings=settings,
    )
    optimizer.global_log_coefficients.fill_(0.20)
    active = torch.tensor([False, True, True, True, True, True])

    def losses(block: int) -> dict[str, dict[str, object]]:
        def value(total: float) -> dict[str, object]:
            return {
                **measured(total),
                "surrogate_total_loss": total,
                "horizon_total_losses": (total, total),
                "causal_block_index": block,
            }

        # Positive gradient asks Adam to move both coefficients inward.
        return {
            "distance_plus": value(1.2),
            "distance_minus": value(0.8),
            "damping_plus": value(1.2),
            "damping_minus": value(0.8),
        }

    optimizer.begin_observation(active)
    optimizer.finish_observation(losses(40))
    before = optimizer.global_log_coefficients.detach().clone()
    optimizer.begin_observation(active)
    _, _, metrics = optimizer.finish_observation(losses(41))
    realized = (optimizer.global_log_coefficients - before).abs()
    expected_cap = 0.02 * (1.5 + 2.0) / (1.5 + 2.0 + 3.0)
    assert metrics["status"] == "sim_global_updated"
    assert float(realized.max().item()) <= expected_cap + 1.0e-7
    assert float(optimizer.global_log_coefficients.min().item()) > 0.10


def test_reconstruction_policy_disables_h2_updates() -> None:
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=torch.tensor([True, False, False, False, False, False]),
        edges=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        settings=PaperTrajectoryAdamSettings(
            sim_global_causal_mode=True,
            h2_updates_enabled=False,
            reconstruction_nonoverlapping_h3=True,
            h3_tail_cosine_minimum=0.90,
            minimum_axis_loss_difference=1.0e-9,
        ),
    )
    optimizer.begin_observation(
        torch.tensor([False, True, True, True, True, True])
    )

    def value(total: float) -> dict[str, object]:
        return {
            **measured(total),
            "surrogate_total_loss": total,
            "horizon_total_losses": (total, total),
        }

    new_distance, new_shape, metrics = optimizer.finish_observation(
        {
            "distance_plus": value(1.2),
            "distance_minus": value(0.8),
            "damping_plus": value(1.2),
            "damping_minus": value(0.8),
        }
    )
    assert metrics["status"] == "h2_updates_disabled"
    assert metrics["h2_updates_enabled"] == 0
    assert optimizer.update_count == 0
    assert torch.equal(new_distance, distance)
    assert torch.equal(new_shape, shape)


def test_reconstruction_h3_requires_tail_h2_gradient_agreement() -> None:
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=torch.tensor([True, False, False, False, False, False]),
        edges=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        settings=PaperTrajectoryAdamSettings(
            sim_global_causal_mode=True,
            h2_updates_enabled=False,
            reconstruction_nonoverlapping_h3=True,
            h3_tail_cosine_minimum=0.90,
            minimum_axis_loss_difference=1.0e-9,
        ),
    )
    optimizer.begin_observation(
        torch.tensor([False, True, True, True, True, True])
    )
    weights = (1.5, 2.0, 3.0)

    def value(horizons: tuple[float, float, float]) -> dict[str, object]:
        total = sum(w * v for w, v in zip(weights, horizons)) / sum(weights)
        return {
            **measured(total),
            "track_loss": total,
            "surrogate_total_loss": total,
            "horizon_total_losses": horizons,
        }

    new_distance, new_shape, metrics = optimizer.finish_observation(
        {
            "distance_plus": value((10.0, 0.0, 0.0)),
            "distance_minus": value((0.0, 1.0, 1.0)),
            "damping_plus": value((10.0, 0.0, 0.0)),
            "damping_minus": value((0.0, 1.0, 1.0)),
        }
    )
    assert metrics["gradient_direction_consistent"] == 1
    assert metrics["h3_tail_validation_enabled"] == 1
    assert metrics["h3_tail_direction_consistent"] == 0
    assert metrics["h3_tail_gradient_cosine"] < 0.0
    assert metrics["status"] == "h3_tail_gradient_misaligned"
    assert optimizer.update_count == 0
    assert torch.equal(new_distance, distance)
    assert torch.equal(new_shape, shape)


def test_reconstruction_h3_commits_when_full_and_tail_agree() -> None:
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=torch.tensor([True, False, False, False, False, False]),
        edges=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
        ),
        settings=PaperTrajectoryAdamSettings(
            sim_global_causal_mode=True,
            h2_updates_enabled=False,
            reconstruction_nonoverlapping_h3=True,
            h3_tail_cosine_minimum=0.90,
            global_maximum_log_offset=0.15,
            minimum_axis_loss_difference=1.0e-9,
        ),
    )
    optimizer.begin_observation(
        torch.tensor([False, True, True, True, True, True])
    )

    def value(total: float) -> dict[str, object]:
        return {
            **measured(total),
            "track_loss": total,
            "surrogate_total_loss": total,
            "horizon_total_losses": (total, total, total),
        }

    _, _, metrics = optimizer.finish_observation(
        {
            "distance_plus": value(1.2),
            "distance_minus": value(0.8),
            "damping_plus": value(1.2),
            "damping_minus": value(0.8),
        }
    )
    assert metrics["status"] == "sim_global_updated"
    assert metrics["h3_tail_direction_consistent"] == 1
    assert metrics["h3_tail_directional_derivative"] < 0.0
    assert metrics["active_cumulative_log_offset_limit"] == 0.15
    assert optimizer.update_count == 1


def test_sim_global_h1_warmup_runs_no_counterfactuals() -> None:
    optimizer, _, _ = make_optimizer()
    # This test needs the global mode but reuses the compact fixture tensors.
    optimizer.settings = PaperTrajectoryAdamSettings(sim_global_causal_mode=True)
    metrics = optimizer.record_causal_warmup_observation(
        torch.tensor([False, True, True, True, True, True]),
        horizon_count=1,
    )
    assert metrics["status"] == "causal_window_warmup"
    assert metrics["warmup_counterfactual_replays"] == 0
    assert metrics["loss_evaluation_count"] == 0
    assert optimizer.update_count == 0


def test_distribution_robust_track_loss_reports_regions() -> None:
    predicted = torch.tensor(
        [[0.001, 0.0, 0.0], [0.002, 0.0, 0.0], [0.010, 0.0, 0.0]]
    )
    target = torch.zeros_like(predicted)
    loss, metrics = confidence_weighted_huber_track_loss(
        predicted_points=predicted,
        target_points=target,
        confidence=torch.ones(3),
        valid_mask=torch.ones(3, dtype=torch.bool),
        robust_scale_m=0.003,
        region_ids=torch.tensor([0, 0, 1]),
        region_balance_weight=0.10,
        tail_region_weight=0.05,
        tail_region_fraction=0.25,
    )
    assert loss > metrics["point_mean_loss"]
    assert metrics["valid_regions"] == 2


def test_local_distance_field_is_zero_mean_bounded_and_optional() -> None:
    distance = torch.full((6,), 0.01, dtype=torch.float32)
    shape = torch.full((6,), 0.0005, dtype=torch.float32)
    fixed = torch.tensor([True, False, False, False, False, False])
    edges = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
    )
    optimizer = PaperTrajectoryAdamOptimizer(
        distance_stiffness=distance,
        shape_stiffness=shape,
        fixed_mask=fixed,
        edges=edges,
        settings=PaperTrajectoryAdamSettings(
            sim_global_causal_mode=True,
            local_distance_enabled=True,
            local_distance_maximum_log_step=0.008,
            local_distance_maximum_log_offset=0.04,
            local_distance_minimum_loss_improvement=1.0e-6,
        ),
    )
    raw = torch.tensor([0.0, 1.0, 0.5, -0.5, -1.0, 0.0])
    support = torch.tensor([False, True, True, True, True, False])
    direction = optimizer.prepare_local_distance_direction(raw, support)
    trials = {
        trial.label: trial
        for trial in optimizer.begin_local_distance_observation(direction)
    }
    assert set(trials) == {"local_base", "local_plus", "local_minus"}
    before_global = optimizer.global_log_coefficients.clone()
    _, _, metrics = optimizer.finish_local_distance_observation(
        {"local_base": 1.0, "local_plus": 0.8, "local_minus": 1.1}
    )
    active = optimizer.local_distance_log_offsets[~fixed]
    assert metrics["local_distance_status"] == "accepted"
    assert metrics["local_distance_global_mean_preserved"] == 1
    assert abs(float(active.mean())) <= 1.0e-7
    assert float(active.abs().max()) <= 0.04 + 1.0e-7
    assert metrics["local_distance_step_maximum"] <= 0.008 + 1.0e-7
    assert torch.equal(optimizer.global_log_coefficients, before_global)
    assert optimizer.local_distance_log_offsets[0] == 0.0
    accepted = optimizer.local_distance_log_offsets.clone()
    optimizer.begin_local_distance_observation(direction)
    _, _, rejected = optimizer.finish_local_distance_observation(
        {
            "local_base": 1.0,
            "local_plus": 1.0002,
            "local_minus": 1.0001,
        }
    )
    assert rejected["local_distance_status"] == "no_h3_improvement"
    assert torch.equal(optimizer.local_distance_log_offsets, accepted)


def main() -> None:
    tests = (
        test_independent_trials_and_bounds,
        test_adam_follows_measured_counterfactual_gradient,
        test_insensitive_axis_does_not_random_walk,
        test_adam_momentum_step_is_reported_for_immediate_commit,
        test_realized_material_step_is_capped_after_adam_momentum,
        test_confidence_weighted_huber_track_loss,
        test_absolute_adam_candidate_has_no_legacy_log_cap,
        test_sim_global_causal_updates_distance_and_damping_only,
        test_sim_global_rejects_open_loop_one_step_gradient_disagreement,
        test_sim_global_h2_scales_the_material_step_cap,
        test_sim_global_rejects_adam_momentum_ascent,
        test_sim_global_h2_cumulative_offsets_are_bounded,
        test_sim_global_h2_rejects_cross_block_gradient_reversal,
        test_h2_does_not_snap_h3_coefficient_to_tighter_bound,
        test_reconstruction_policy_disables_h2_updates,
        test_reconstruction_h3_requires_tail_h2_gradient_agreement,
        test_reconstruction_h3_commits_when_full_and_tail_agree,
        test_sim_global_h1_warmup_runs_no_counterfactuals,
        test_distribution_robust_track_loss_reports_regions,
        test_local_distance_field_is_zero_mean_bounded_and_optional,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} paper trajectory stiffness tests")


if __name__ == "__main__":
    main()
