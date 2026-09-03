#!/usr/bin/env python3
"""CPU gates for leakage-free H10 validation across 7:1 held-out frames."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

import example_embodied_super_offline as super_example


def main() -> None:
    super_example.configure_stiffness_admission_policy(
        (1, 3, 5, 10),
        "robust_hierarchical_system_id",
        "strict_all",
    )
    controls = object.__new__(super_example.SuperPlaybackControls)
    recorder = SimpleNamespace(
        protocol="reconstruction_7to1",
        observation_allowed=lambda frame_index: int(frame_index) % 8 != 0,
    )
    controls.tissue_benchmark_recorder = recorder
    controls.playback_timestamps = tuple(range(100))
    target_frames = controls._stiffness_training_validation_frames(7)

    buffered = tuple(
        SimpleNamespace(supervision_valid=value, frame_index=index)
        for index, value in enumerate((True, False, True, True))
    )
    three_of_four = super_example.select_recent_supervised_causal_window(
        buffered,
        required_supervised=3,
        maximum_span=4,
        maximum_missing=1,
    )
    two_dropouts = tuple(
        SimpleNamespace(supervision_valid=value, frame_index=index)
        for index, value in enumerate((True, False, False, True))
    )
    rejected_two_dropouts = (
        super_example.select_recent_supervised_causal_window(
            two_dropouts,
            required_supervised=2,
            maximum_span=4,
            maximum_missing=1,
        )
    )
    four_valid = tuple(
        SimpleNamespace(supervision_valid=True, frame_index=index)
        for index in range(4)
    )
    latest_three = super_example.select_recent_supervised_causal_window(
        four_valid,
        required_supervised=3,
        maximum_span=4,
        maximum_missing=1,
    )

    pending = super_example.PendingStiffnessValidation(
        candidate=SimpleNamespace(metrics={"status": "candidate"}),
        rollout_state="proposal",
        frame_index=7,
        grip_active=False,
        history_baseline_rms_m=0.0,
        history_candidate_rms_m=0.0,
        previous_residual=torch.zeros(1),
        validation_frame_indices=target_frames,
    )
    controls._pending_stiffness_validation = pending
    controls._active_stiffness_evaluations = []
    controls._record_incomplete_stiffness_evaluations = lambda reason: None
    controls.stiffness_updater = SimpleNamespace(
        reject=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("7:1 held-out frame must not reject pending material")
        )
    )
    controls.prepare_benchmark_frame(8)
    pending_survives = controls._pending_stiffness_validation is pending

    shadow_call: dict = {}

    def fake_shadow(**kwargs):
        shadow_call.update(kwargs)
        frames = kwargs["validation_frame_indices"]
        return {
            "visual_loss": 0.5,
            "camera_losses": (0.5, 0.5),
            "validation_visual_losses": {
                frame: 0.5 - 0.01 * slot
                for slot, frame in enumerate(frames, start=1)
            },
            "validation_camera_losses": {
                frame: (0.5 - 0.01 * slot, 0.5 - 0.01 * slot)
                for slot, frame in enumerate(frames, start=1)
            },
        }

    controls._run_stiffness_rollout_shadow = fake_shadow
    shadow = controls._run_stiffness_prediction_shadow(
        pending,
        use_candidate=True,
    )

    controls.current_frame_index = 18
    controls._last_stiffness_validation_metrics = None
    pending.commands = [
        super_example.StiffnessToolCommand(frame, float(frame), frame)
        for frame in range(8, 19)
    ]
    waiting = controls._validate_pending_stiffness(
        gate_paused=False,
        gate_reason="",
        grip_active=False,
    )

    gates = {
        "nonoverlapping_h3_ends_before_held_out_frame": (
            super_example.is_reconstruction_h3_block_end(7, 0)
            and super_example.is_reconstruction_h3_block_end(15, 0)
            and not super_example.is_reconstruction_h3_block_end(6, 0)
            and all(frame % 8 != 0 for frame in (4, 5, 6, 7))
        ),
        "three_of_four_keeps_physical_gap": (
            len(three_of_four) == 4
            and [item.frame_index for item in three_of_four] == [0, 1, 2, 3]
            and sum(item.supervision_valid for item in three_of_four) == 3
        ),
        "more_than_one_gap_is_rejected": rejected_two_dropouts == (),
        "four_valid_observations_use_latest_h3": (
            [item.frame_index for item in latest_three] == [1, 2, 3]
        ),
        "horizons_skip_held_out_frames": target_frames == (9, 11, 13, 19),
        "held_out_frame_preserves_pending_candidate": pending_survives,
        "shadow_reads_training_frames_only": (
            shadow_call.get("validation_frame_indices") == target_frames
            and all(frame % 8 != 0 for frame in target_frames)
        ),
        "observed_horizons_retain_h13510_labels": (
            shadow["horizon_visual_losses"]
            == {1: 0.49, 3: 0.48, 5: 0.47, 10: 0.46}
        ),
        "validator_waits_for_tenth_training_frame": (
            waiting is not None
            and waiting["status"] == "pending_horizon"
            and waiting["required_prediction_horizon_frames"] == 12
            and controls._pending_stiffness_validation is pending
        ),
    }
    report = {"gates": gates, "passed": all(gates.values())}
    print(json.dumps(report, indent=2))
    super_example.configure_stiffness_admission_policy(
        (1, 3, 5), "bidirectional_8", "strict_all"
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
