#!/usr/bin/env python3
"""Headless regression gates for causal stiffness and future handoff."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

import example_embodied_super_offline as super_example


def safe_physical_metrics() -> dict:
    return {
        "minimum_volume_ratio": 0.50,
        "volume_ratio_p01": 0.80,
        "volume_weighted_mean_ratio": 1.00,
        "tetrahedra_below_volume_floor": 0,
        "inverted_tetrahedra": 0,
        "maximum_penetration_m": 0.001,
        "anchor_error_maximum_m": 0.0001,
    }


def test_causal_prediction_policy() -> None:
    controls = object.__new__(super_example.SuperPlaybackControls)
    controls.current_frame_index = 15
    controls.tissue_benchmark_recorder = None
    pending = super_example.PendingStiffnessValidation(
        candidate=SimpleNamespace(),
        rollout_state="proposal",
        frame_index=10,
        grip_active=True,
        history_baseline_rms_m=0.0,
        history_candidate_rms_m=0.0,
        previous_residual=torch.tensor((0.002,)),
        commands=[
            super_example.StiffnessToolCommand(index, index / 30.0, index)
            for index in range(11, 16)
        ],
        validation_frame_indices=(11, 13, 15),
    )
    call: dict = {}

    def fake_shadow(**kwargs):
        call.update(kwargs)
        return {
            "visual_loss": 0.50,
            "camera_losses": (0.40, 0.60),
            "trajectory_visual_loss": 0.30,
            "trajectory_camera_losses": (0.20, 0.40),
            "validation_visual_losses": {11: 0.40, 13: 0.30, 15: 0.20},
            "validation_camera_losses": {
                11: (0.30, 0.50),
                13: (0.20, 0.40),
                15: (0.10, 0.30),
            },
        }

    controls._run_stiffness_rollout_shadow = fake_shadow
    result = controls._run_stiffness_prediction_shadow(
        pending, use_candidate=True
    )
    assert call["freeze_grip_state_machine"] is False
    assert call["replay_visual_residuals"] is True
    assert torch.equal(call["initial_previous_residual"], pending.previous_residual)
    assert call["replay_after_frame_index"] == 10
    assert call["replay_frame_indices"] == (11, 12, 13, 14, 15)
    assert call["validation_frame_indices"] == (11, 13, 15)
    assert np.isclose(result["visual_loss"], 0.30)
    assert np.allclose(result["camera_losses"], (0.20, 0.40))


class FakeStateSimulator:
    def __init__(self) -> None:
        self.current_state = "live_next"
        self.state_0 = SimpleNamespace(
            particle_q=wp.array(
                np.zeros((2, 3), dtype=np.float32),
                dtype=wp.vec3,
                device="cpu",
            )
        )

    def clone_embodied_gaussian_rollout_state(self):
        return self.current_state

    def copy_embodied_gaussian_rollout_state(self, state) -> None:
        self.current_state = state

    def update_gaussian_transforms(self) -> None:
        pass

    def triangle_skin_contact_metrics(self):
        return {}


def test_last_training_residual_reaches_future_frame_one() -> None:
    simulator = FakeStateSimulator()

    def step(*, compute_visual_forces=False) -> None:
        assert compute_visual_forces is False
        if simulator.current_state == "corrected":
            simulator.current_state = "corrected_next"

    controls = object.__new__(super_example.SuperPlaybackControls)
    controls.environment = SimpleNamespace(sim=simulator, step=step)
    controls.current_frame_index = 1152
    controls.current_timestep = 38.4
    controls._last_state_index = 1152
    controls._last_q_full = torch.zeros(1)
    controls._benchmark_observation_gap_frames = 1
    controls._last_visual_open_loop_prediction_frame_index = -1
    controls._visual_open_loop_prediction_count = 0
    controls._previous_visual_residual = None
    controls.stiffness_metrics_recorder = None
    controls.tissue_benchmark_recorder = SimpleNamespace(
        protocol="future_80to20"
    )
    controls.apply_current_psm_pose = lambda: None
    controls._current_visual_open_loop_physical_metrics = (
        lambda: safe_physical_metrics()
    )
    controls._pending_visual_residual_validation = (
        super_example.PendingVisualResidualValidation(
            result=SimpleNamespace(residual=torch.ones((2, 3)) * 1.0e-4),
            candidate_state="corrected",
            frame_index=1151,
            selected_gain=0.5,
            commands=[
                super_example.StiffnessToolCommand(
                    1152, 38.4, 1152, "lift"
                )
            ],
        )
    )
    metrics = controls._adopt_pending_visual_residual_open_loop_prediction()
    assert metrics is not None and metrics["accepted"] is True
    assert metrics["uses_future_rgb"] is False
    assert metrics["mode"] == "pending_training_residual_state"
    assert simulator.current_state == "corrected_next"
    assert controls._pending_visual_residual_validation is None
    assert torch.allclose(
        controls._previous_visual_residual, torch.ones((2, 3)) * 1.0e-4
    )


def test_future_boundary_preserves_only_one_gap() -> None:
    class Benchmark:
        protocol = "future_80to20"

        @staticmethod
        def observation_allowed(_frame_index: int) -> bool:
            return False

    controls = object.__new__(super_example.SuperPlaybackControls)
    controls.tissue_benchmark_recorder = Benchmark()
    controls._benchmark_observation_gap_frames = 0
    controls._pending_visual_residual_validation = SimpleNamespace()
    controls._pending_stiffness_validation = None
    controls._active_stiffness_evaluations = []
    controls.stiffness_metrics_recorder = None
    controls.prepare_benchmark_frame(1152)
    assert controls._pending_visual_residual_validation is not None
    controls.prepare_benchmark_frame(1153)
    assert controls._pending_visual_residual_validation is None


def test_reconstruction_holdout_never_uses_physical_only_handoff() -> None:
    controls = object.__new__(super_example.SuperPlaybackControls)
    controls.visual_feedback_mode = "residual"
    controls.visual_residual_gain_profile = "cross_frame_hold"
    controls.tissue_benchmark_recorder = SimpleNamespace(
        protocol="reconstruction_7to1"
    )
    controls._benchmark_observation_enabled = False
    controls._benchmark_observation_gap_frames = 1
    controls._last_visual_open_loop_prediction_frame_index = -1
    controls.current_frame_index = 10
    controls._previous_visual_residual = torch.ones((2, 3))
    controls._pending_visual_residual_validation = SimpleNamespace()
    controls._adopt_pending_visual_residual_open_loop_prediction = lambda: (_ for _ in ()).throw(
        AssertionError("reconstruction must wait for training RGB")
    )
    assert controls.apply_one_frame_visual_open_loop_prediction() is None


def test_ranked_reconstruction_uses_only_confirmed_residual() -> None:
    simulator = FakeStateSimulator()
    simulator.triangle_skin_contact_projector = None
    simulator.material_projector = None
    mapper = SimpleNamespace(
        fixed_mask=torch.zeros(2, dtype=torch.bool),
        _clip_vectors=lambda vectors, _maximum: vectors,
        physical_quality_metrics=lambda _positions: safe_physical_metrics(),
    )
    pending = SimpleNamespace(marker="unconfirmed")
    controls = object.__new__(super_example.SuperPlaybackControls)
    controls.visual_feedback_mode = "residual"
    controls.visual_residual_gain_profile = "cross_frame_ranked_hold"
    controls.visual_residual_mapper = mapper
    controls.environment = SimpleNamespace(
        sim=simulator,
        physics_settings=SimpleNamespace(paper_distance_stiffness=0.20),
    )
    controls.tissue_benchmark_recorder = SimpleNamespace(
        protocol="reconstruction_7to1"
    )
    controls._benchmark_observation_enabled = False
    controls._benchmark_observation_gap_frames = 1
    controls._last_visual_open_loop_prediction_frame_index = -1
    controls._visual_open_loop_prediction_count = 0
    controls.current_frame_index = 8
    controls.current_timestep = 8.0 / 30.0
    controls._current_action_phase = "lift"
    controls._previous_visual_residual = torch.full((2, 3), 5.0e-5)
    controls._pending_visual_residual_validation = pending
    controls.stiffness_metrics_recorder = None
    metrics = controls.apply_one_frame_visual_open_loop_prediction()
    assert metrics is not None and metrics["accepted"] is True
    assert metrics["uses_current_rgb"] is False
    assert controls._pending_visual_residual_validation is pending
    assert torch.count_nonzero(wp.to_torch(simulator.state_0.particle_q)) > 0


def test_ranked_cross_frame_keeps_alternative_gain() -> None:
    simulator = FakeStateSimulator()
    simulator.current_state = "live_next"

    def step(*, compute_visual_forces=False) -> None:
        assert compute_visual_forces is False
        simulator.current_state = {
            "full": "full_next",
            "half": "half_next",
        }.get(simulator.current_state, simulator.current_state)

    controls = object.__new__(super_example.SuperPlaybackControls)
    controls.environment = SimpleNamespace(
        sim=simulator, step=step, frames=object()
    )
    controls.dataset_manager = SimpleNamespace(update_frames=lambda _time: None)
    controls.visual_residual_mapper = object()
    controls.current_frame_index = 11
    controls.current_timestep = 11.0 / 30.0
    controls._last_state_index = 11
    controls._benchmark_observation_enabled = True
    controls.apply_current_psm_pose = lambda: None
    controls._apply_stiffness_tool_command = lambda _command: None
    controls._synchronize_shadow_transaction = lambda _sim: None
    controls._previous_visual_residual = None
    controls.stiffness_metrics_recorder = None
    controls._current_action_phase = "lift"

    baseline_metrics = {
        **safe_physical_metrics(),
        "visual_loss": 0.50,
        "camera_losses": (0.50, 0.50),
    }

    def current_metrics() -> dict:
        if simulator.current_state == "full_next":
            return {
                **baseline_metrics,
                "visual_loss": 0.51,
                "camera_losses": (0.51, 0.51),
            }
        if simulator.current_state == "half_next":
            return {
                **baseline_metrics,
                "visual_loss": 0.45,
                "camera_losses": (0.45, 0.45),
            }
        return dict(baseline_metrics)

    controls._current_visual_cross_frame_metrics = current_metrics
    full_result = SimpleNamespace(residual=torch.ones((2, 3)))
    half_result = SimpleNamespace(residual=torch.full((2, 3), 0.5))
    pending = super_example.PendingVisualResidualValidation(
        result=full_result,
        candidate_state="full",
        frame_index=10,
        selected_gain=1.0,
        commands=[
            super_example.StiffnessToolCommand(11, 11.0 / 30.0, 11)
        ],
        candidate_branches=(
            super_example.VisualResidualCandidateBranch(
                full_result, "full", 1.0, 0.40, {1: 0.01}
            ),
            super_example.VisualResidualCandidateBranch(
                half_result, "half", 0.5, 0.42, {1: 0.005}
            ),
        ),
    )
    controls._pending_visual_residual_validation = pending
    metrics, confirmed = controls._validate_pending_visual_residual_cross_frame()
    assert metrics is not None and metrics["accepted"] is True
    assert metrics["candidate_count"] == 2
    assert np.isclose(metrics["selected_gain"], 0.5)
    assert simulator.current_state == "half_next"
    assert confirmed is pending and confirmed.result is half_result
    assert torch.allclose(controls._previous_visual_residual, half_result.residual)


def test_causal_rollout_adoption_is_bounded() -> None:
    simulator = FakeStateSimulator()
    simulator.current_state = "live"
    controls = object.__new__(super_example.SuperPlaybackControls)
    controls.environment = SimpleNamespace(sim=simulator)
    controls._previous_visual_residual = None
    candidate = {
        "particle_positions": torch.full((2, 3), 5.0e-5),
        "rollout_state": "causal-verified",
        "rollout_previous_residual": torch.tensor((0.003,)),
    }
    accepted = controls._adopt_validated_stiffness_rollout(candidate)
    assert accepted["validated_rollout_adopted"] is True
    assert simulator.current_state == "causal-verified"
    rejected = controls._adopt_validated_stiffness_rollout(
        {**candidate, "particle_positions": torch.full((2, 3), 2.0e-3)}
    )
    assert rejected["validated_rollout_adopted"] is False


def main() -> None:
    wp.init()
    original_horizons = super_example.STIFFNESS_ADMISSION_HORIZONS
    original_profile = super_example.STIFFNESS_CANDIDATE_PROFILE
    original_mode = super_example.STIFFNESS_ADMISSION_MODE
    try:
        mismatch_rejected = False
        try:
            super_example.configure_stiffness_admission_policy(
                (1, 3, 5), "bidirectional_8", "causal_fixed_lag"
            )
        except ValueError:
            mismatch_rejected = True
        assert mismatch_rejected
        super_example.configure_stiffness_admission_policy(
            (1, 3, 5), "direct_residual_gradient", "causal_fixed_lag"
        )
        assert super_example.STIFFNESS_CANDIDATE_VARIANTS == ()
        assert "training_rgb_residual_replay" in (
            super_example.stiffness_validation_objective_name()
        )
        test_causal_prediction_policy()
        test_last_training_residual_reaches_future_frame_one()
        test_future_boundary_preserves_only_one_gap()
        test_reconstruction_holdout_never_uses_physical_only_handoff()
        test_ranked_reconstruction_uses_only_confirmed_residual()
        test_ranked_cross_frame_keeps_alternative_gain()
        test_causal_rollout_adoption_is_bounded()
    finally:
        super_example.configure_stiffness_admission_policy(
            original_horizons, original_profile, original_mode
        )
    print("causal fixed-lag stiffness and visual handoff gates: PASS")


if __name__ == "__main__":
    main()
