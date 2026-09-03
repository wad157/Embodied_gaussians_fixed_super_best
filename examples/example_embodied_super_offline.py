# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field, replace
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trio
import warp as wp

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent

# 优先导入当前仓库源码，避免运行到系统里旧安装的 embodied_gaussians。
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(current_dir))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    PSM_ARTICULATION_INDEX,
    PSM_LND_POSE_DRIVER_PATH,
    PSM_POSE_DRIVER_PATHS,
    apply_psm_lnd_pose,
    build_environment,
    expand_psm_q7_to_urdf_order,
    load_mimic_config,
    set_psm_tissue_collisions,
    urdf_actuated_joint_order,
)
from embodied_gaussians import DatasetManager, EmbodiedGaussiansEnvironment  # noqa: E402
from embodied_gaussians.embodied_simulator.visual_force_masks import (  # noqa: E402
    MultiCameraPackedTissueVisualForceWeights,
)
from embodied_gaussians.embodied_simulator.trajectory_appearance import (  # noqa: E402
    TrajectoryAppearanceSettings,
)
from embodied_gaussians.physics_simulator.visual_tissue_residual_mapping import (  # noqa: E402
    TetrahedralGaussianVisualResidualMapper,
    VisualTissueResidualMappingSettings,
    strict_one_tetrahedron_ring_mask,
)
from embodied_gaussians.physics_simulator.flow_depth_particle_observer import (  # noqa: E402
    FlowDepthObservationSequence,
    FlowDepthParticleRangeBindings,
    FlowDepthStateUpdateSettings,
    compute_flow_depth_particle_state_update,
    fixed_range_centers,
    load_fixed_particle_range_bindings,
    load_flow_depth_observation_sequence,
)
from embodied_gaussians.physics_simulator.online_tissue_stiffness import (  # noqa: E402
    OnlineTissueStiffnessSettings,
    PaperStiffnessCandidate,
    ResidualDrivenPaperStiffnessUpdater,
)
from embodied_gaussians.physics_simulator.sim_particle_graph_stiffness import (  # noqa: E402
    OnlineTissueStiffnessSettings as SimParticleGraphSettings,
    PaperStiffnessCandidate as SimParticleGraphCandidate,
    ResidualDrivenPaperStiffnessUpdater as SimParticleGraphUpdater,
)
from embodied_gaussians.physics_simulator.paper_trajectory_stiffness import (  # noqa: E402
    PaperTrajectoryAdamOptimizer,
    PaperTrajectoryAdamSettings,
    PaperTrajectoryMaterialTrial,
    confidence_weighted_huber_track_loss,
)
from embodied_gaussians.physics_simulator.stiffness_evaluation import (  # noqa: E402
    StiffnessActionPhaseClassifier,
    StiffnessMetricsRecorder,
    parse_stiffness_evaluation_horizons,
)
from embodied_gaussians.physics_simulator.super_tissue_benchmark import (  # noqa: E402
    BENCHMARK_PROTOCOLS,
    SuperTissueBenchmarkRecorder,
)
import embodied_gaussians as embodied_gaussians_package  # noqa: E402


LOADED_EMBODIED_GAUSSIANS_SOURCE = Path(
    embodied_gaussians_package.__file__
).resolve().parent
EXPECTED_EMBODIED_GAUSSIANS_SOURCE = (
    repo_root / "src/embodied_gaussians"
).resolve()
# A residual can still be visually acceptable around a tet that was already
# compressed by contact. Mask only particles incident on such a tet instead of
# allowing one global minimum to disable stiffness learning everywhere.
STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO = 0.001
# Preliminary validation gates.  They are deliberately centralized so a
# frozen replay can calibrate them instead of scattering magic values through
# the runtime loop.
STIFFNESS_VISUAL_GRADIENT_RELATIVE_FLOOR = 1.0e-4
# Keep proposal admission nearly open. Fast jaw motion is still excluded from
# history snapshots below, but it no longer suppresses a new candidate by
# itself; the next-image shadow rollout decides whether that candidate is real.
STIFFNESS_MAXIMUM_JAW_SPEED_RAD_S: float | None = None
STIFFNESS_TRANSITION_COOLDOWN_UPDATES = 1
# Align material-learning admission with the configured grip capture envelope.
STIFFNESS_MAXIMUM_PENETRATION_M = 0.0028


def is_reconstruction_h3_block_end(
    destination_frame: int, test_phase: int
) -> bool:
    """Whether H3 ends at the last legal frame of one 7:1 train block."""

    if not 0 <= int(test_phase) < 8:
        raise ValueError("7:1 reconstruction phase must lie in [0, 7]")
    return (int(destination_frame) + 1) % 8 == int(test_phase)


def select_recent_supervised_causal_window(
    observations: tuple,
    *,
    required_supervised: int,
    maximum_span: int,
    maximum_missing: int,
) -> tuple:
    """Select the latest Hn supervision while preserving physical gaps.

    Returned observations are a consecutive slice of the physical transition
    buffer. A supervision dropout remains in the slice and is replayed with
    zero visual loss; it is never deleted from physical time.
    """

    valid_indices = [
        index
        for index, observation in enumerate(observations)
        if bool(getattr(observation, "supervision_valid", True))
    ]
    if len(valid_indices) < int(required_supervised):
        return ()
    start = valid_indices[-int(required_supervised)]
    selected = tuple(observations[start:])
    missing = sum(
        not bool(getattr(observation, "supervision_valid", True))
        for observation in selected
    )
    if len(selected) > int(maximum_span) or missing > int(maximum_missing):
        return ()
    return selected
# The cumulative H=5 shadow exposed 20 commits at a 1e-6 floor, but most of
# their apparent gains were only optimizer/render noise and the 211-frame mean
# regressed by 0.0523%. Ten micro-loss is about 0.05% of the observed loss and
# retains the loose contact gate while requiring material evidence above that
# measured noise floor.
STIFFNESS_PREDICTION_ABSOLUTE_MARGIN = 1.0e-5
STIFFNESS_PREDICTION_RELATIVE_MARGIN = 0.0
STIFFNESS_CAMERA_ABSOLUTE_REGRESSION = 1.0e-6
STIFFNESS_CAMERA_RELATIVE_REGRESSION = 5.0e-3
STIFFNESS_MINIMUM_VOLUME_ABSOLUTE_DROP = 0.01
STIFFNESS_MINIMUM_VOLUME_RELATIVE_DROP = 0.02
STIFFNESS_PENETRATION_TOLERANCE_M = 0.0001
STIFFNESS_ANCHOR_ERROR_TOLERANCE_M = 0.00005
# A delayed residual has already passed the solver's local no-flip projection.
# Cross-frame validation therefore rejects true unsafe collapse and robust
# distribution/total-volume degradation, rather than requiring the single
# worst tetrahedron to remain bitwise stationary after another physics step.
VISUAL_CROSS_FRAME_MINIMUM_VOLUME_RATIO = 0.30
VISUAL_CROSS_FRAME_P01_ABSOLUTE_DROP = 0.05
VISUAL_CROSS_FRAME_P01_RELATIVE_DROP = 0.10
VISUAL_CROSS_FRAME_WEIGHTED_MEAN_ABSOLUTE_DROP = 0.01
VISUAL_CROSS_FRAME_WEIGHTED_MEAN_RELATIVE_DROP = 0.02
# A confirmed image correction is partly erased by a stiff material during the
# first frame without RGB. Carry it exactly once as a causal observer
# prediction, scaled by the installed distance stiffness. It is never replayed
# deeper into an open-loop future and never reads the withheld frame.
VISUAL_ONE_FRAME_CARRY_MINIMUM_GAIN = 0.25
VISUAL_ONE_FRAME_CARRY_MAXIMUM_GAIN = 1.00
VISUAL_ONE_FRAME_CARRY_DISTANCE_MINIMUM = 0.10
VISUAL_ONE_FRAME_CARRY_DISTANCE_MAXIMUM = 1.60
VISUAL_ONE_FRAME_CARRY_MAXIMUM_CORRECTION_M = 0.00010
VISUAL_ONE_FRAME_CARRY_MAXIMUM_BACKTRACKS = 6
STIFFNESS_MAXIMUM_PENDING_ROLLOUT_STEPS: int | None = None
STIFFNESS_MAXIMUM_PREDICTION_HORIZON_FRAMES = 10
# Material admission compares old/new material with visual residual strictly
# disabled and the persistent grip state machine frozen.  The defaults retain
# the written H=1/3/5 protocol; CLI ablations can add H=10 without changing the
# residual gate or using any held-out 3D observation.
STIFFNESS_ADMISSION_HORIZONS = (1, 3, 5)
STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES = 5
# The residual/deformation heuristic only proposes a spatial direction.  The
# material-isolation shadow decides the actual sign and independently tests
# distance and shape, so a wrong heuristic sign cannot block learning.  The
# larger independent probes make their effect observable without multiplying
# the old four-way line-search cost excessively.
STIFFNESS_CANDIDATE_PROFILES = {
    # The causal online path consumes the continuous residual-gradient update
    # produced by PaperOnlineStiffnessUpdater directly.  The empty profile is
    # intentional: there are no signed/scaled variants and no shadow contest.
    "direct_residual_gradient": (),
    # The direct-online path alternates one material family per successful
    # commit.  This removes distance/shape credit ambiguity without restoring
    # discrete shadows or any frame/count admission limit.
    "direct_alternating_gradient": (),
    "bidirectional_8": (
        ("distance_forward", 2.0, 0.0, "full"),
        ("distance_reverse", -2.0, 0.0, "full"),
        ("shape_forward", 0.0, 2.0, "full"),
        ("shape_reverse", 0.0, -2.0, "full"),
        ("joint_forward", 1.0, 1.0, "full"),
        ("joint_reverse", -1.0, -1.0, "full"),
        ("joint_forward_large", 2.0, 2.0, "full"),
        ("joint_reverse_large", -2.0, -2.0, "full"),
    ),
    # Causal fixed-lag material identification mirrors the successful sim
    # protocol: every counterfactual changes exactly one material family, and
    # local regions are judged independently.  There are deliberately no
    # joint distance/shape probes, so a winning shadow has unambiguous credit.
    "causal_fixed_lag_12": (
        # Event-driven trust steps: the 0.8 and 1.0 branches map a configured
        # 0.10 cap to the requested 0.08/0.10 update range.
        ("distance_forward_0.8", 0.8, 0.0, "full"),
        ("distance_reverse_0.8", -0.8, 0.0, "full"),
        ("distance_forward", 1.0, 0.0, "full"),
        ("distance_reverse", -1.0, 0.0, "full"),
        ("shape_forward_0.8", 0.0, 0.8, "full"),
        ("shape_reverse_0.8", 0.0, -0.8, "full"),
        ("shape_forward", 0.0, 1.0, "full"),
        ("shape_reverse", 0.0, -1.0, "full"),
        ("component0_distance_forward", 1.0, 0.0, "component_0"),
        ("component0_distance_reverse", -1.0, 0.0, "component_0"),
        ("component0_shape_forward", 0.0, 1.0, "component_0"),
        ("component0_shape_reverse", 0.0, -1.0, "component_0"),
    ),
    # The independent probes above cannot express opposite-signed distance
    # and shape updates in one material field.  These four extra quadrants
    # let the residual-off shadow test that missing family directly.
    "cross_signed_12": (
        ("distance_forward", 2.0, 0.0, "full"),
        ("distance_reverse", -2.0, 0.0, "full"),
        ("shape_forward", 0.0, 2.0, "full"),
        ("shape_reverse", 0.0, -2.0, "full"),
        ("joint_forward", 1.0, 1.0, "full"),
        ("joint_reverse", -1.0, -1.0, "full"),
        ("joint_forward_large", 2.0, 2.0, "full"),
        ("joint_reverse_large", -2.0, -2.0, "full"),
        ("distance_forward_shape_reverse", 1.0, -1.0, "full"),
        ("distance_reverse_shape_forward", -1.0, 1.0, "full"),
        ("distance_forward_shape_reverse_large", 2.0, -2.0, "full"),
        ("distance_reverse_shape_forward_large", -2.0, 2.0, "full"),
    ),
    # Global candidates can hide a bad region behind a larger good region in
    # the mean image loss.  This profile gives the two strongest disconnected
    # same-sign evidence components independent distance/shape credit while
    # retaining four full-field probes as a fallback.
    "spatial_components_12": (
        ("distance_forward", 2.0, 0.0, "full"),
        ("distance_reverse", -2.0, 0.0, "full"),
        ("shape_forward", 0.0, 2.0, "full"),
        ("shape_reverse", 0.0, -2.0, "full"),
        ("component0_distance_forward", 2.0, 0.0, "component_0"),
        ("component0_distance_reverse", -2.0, 0.0, "component_0"),
        ("component0_shape_forward", 0.0, 2.0, "component_0"),
        ("component0_shape_reverse", 0.0, -2.0, "component_0"),
        ("component1_distance_forward", 2.0, 0.0, "component_1"),
        ("component1_distance_reverse", -2.0, 0.0, "component_1"),
        ("component1_shape_forward", 0.0, 2.0, "component_1"),
        ("component1_shape_reverse", 0.0, -2.0, "component_1"),
    ),
    # Hierarchical moving-horizon system identification.  Four local probes
    # retain regional adaptation; a separate runtime-generated absolute grid
    # searches the global distance/shape offsets over their full legal range.
    # This is intentionally not a larger version of the legacy local step: it
    # can recover when every node starts an order of magnitude too soft/hard.
    "hierarchical_system_id": (
        ("local_distance_forward", 2.0, 0.0, "full"),
        ("local_distance_reverse", -2.0, 0.0, "full"),
        ("local_shape_forward", 0.0, 2.0, "full"),
        ("local_shape_reverse", 0.0, -2.0, "full"),
    ),
    # Robust successor: the runtime adds only adjacent global grid targets,
    # validates H=1/3/5/10 and uses 2D-derived residual effort as a tie-break,
    # locks each global material family after trustworthy calibration, and stops
    # material commits after a finite identification budget.  Visual residual
    # state estimation continues for the whole observed interval.
    "robust_hierarchical_system_id": (
        ("local_distance_forward", 2.0, 0.0, "full"),
        ("local_distance_reverse", -2.0, 0.0, "full"),
        ("local_shape_forward", 0.0, 2.0, "full"),
        ("local_shape_reverse", 0.0, -2.0, "full"),
    ),
}
# Same probes as spatial_components_12, plus runtime-generated composites of
# independently safe component-0/component-1 winners.  Distance and shape are
# merged separately: a regional merge is never allowed to become a hidden
# dual-axis material probe.  Keep a distinct profile name so completed
# ablations remain exactly reproducible.
STIFFNESS_CANDIDATE_PROFILES["spatial_components_merge_12"] = (
    STIFFNESS_CANDIDATE_PROFILES["spatial_components_12"]
)
STIFFNESS_CANDIDATE_PROFILE = "bidirectional_8"
STIFFNESS_CANDIDATE_VARIANTS = STIFFNESS_CANDIDATE_PROFILES[
    STIFFNESS_CANDIDATE_PROFILE
]
# Coarse absolute medians span the full online range and include the original
# scene baseline (0.20/0.004).  Successive accepted windows naturally provide
# coordinate descent: one window may fix distance, a later one shape.
STIFFNESS_GLOBAL_DISTANCE_MEDIANS = (
    0.025,
    0.05,
    0.10,
    0.20,
    0.40,
    0.80,
    1.60,
    3.20,
    4.00,
)
STIFFNESS_GLOBAL_SHAPE_MEDIANS = (
    0.001,
    0.002,
    0.003,
    0.004,
    0.008,
    0.012,
    0.020,
    0.030,
    0.040,
)
STIFFNESS_ADMISSION_MODES = (
    "strict_all",
    "weighted_window",
    "causal_fixed_lag",
    "relaxed_h135",
    "direct_online",
    "paper_trajectory_adam",
    "sim_particle_graph_lm",
)
STIFFNESS_ADMISSION_MODE = "strict_all"
STIFFNESS_WINDOW_HORIZON_ABSOLUTE_REGRESSION = 1.0e-6
STIFFNESS_WINDOW_HORIZON_RELATIVE_REGRESSION = 5.0e-3
# H1/H3 may fluctuate within the bounded window above.  H5/H10 are the
# material-identification horizons and must each beat the baseline beyond the
# measured one-micro-loss shadow jitter, while the weighted aggregate retains
# the stronger 1e-5 improvement requirement below.
STIFFNESS_LONG_HORIZON_ABSOLUTE_MARGIN = 2.0e-6
STIFFNESS_LONG_HORIZON_RELATIVE_MARGIN = 0.0
# AllTracker admission measures confidence-weighted 3D endpoint error in
# metres.  Candidate and baseline start from the identical accepted state, so
# their difference is far less noisy than the absolute stereo-depth error.
# Keep the sign/margin gate close to numerical resolution.  In relaxed H1/H3/H5
# mode H3 may regress by a tiny bounded amount, H5 must improve, and their 3:5
# horizon-weighted aggregate must improve.  RGB remains a bounded auxiliary
# veto, not the primary ranking signal, in trajectory-feedback runs.
STIFFNESS_TRAJECTORY_MINIMUM_LOCAL_TRACKS = 3
# A material update has to beat both a small absolute floor and a fraction of
# the current tracking error.  The old 10/20 nm thresholds were below stereo
# depth and rollout jitter, so numerically different but practically identical
# candidates could commit.  These floors remain permissive (0.05/0.10 um),
# while the 0.05% relative term scales with the actually observed motion.
STIFFNESS_TRAJECTORY_HORIZON_ABSOLUTE_MARGIN_M = 2.5e-8
STIFFNESS_TRAJECTORY_HORIZON_RELATIVE_MARGIN = 1.0e-4
STIFFNESS_TRAJECTORY_AGGREGATE_ABSOLUTE_MARGIN_M = 5.0e-8
STIFFNESS_TRAJECTORY_AGGREGATE_RELATIVE_MARGIN = 1.0e-4
STIFFNESS_TRAJECTORY_H1_ABSOLUTE_REGRESSION_M = 5.0e-6
STIFFNESS_TRAJECTORY_H1_RELATIVE_REGRESSION = 5.0e-3
STIFFNESS_TRAJECTORY_H3_ABSOLUTE_REGRESSION_M = 5.0e-7
STIFFNESS_TRAJECTORY_H3_RELATIVE_REGRESSION = 2.0e-3
STIFFNESS_TRAJECTORY_RGB_ABSOLUTE_REGRESSION = 2.0e-5
STIFFNESS_TRAJECTORY_RGB_RELATIVE_REGRESSION = 1.0e-2
# A physically correct candidate should usually need less subsequent visual
# state correction.  This 2D RGB-derived value now ranks candidates only after
# the causal multi-horizon gap gate; it is no longer a hard commit veto because
# the residual solve itself is noisy and previously rejected every otherwise
# useful reconstruction candidate.
STIFFNESS_RESIDUAL_EFFORT_ABSOLUTE_MARGIN_M = 2.0e-7
STIFFNESS_RESIDUAL_EFFORT_RELATIVE_MARGIN = 0.0
# Broad absolute values remain searchable over multiple trustworthy commits,
# but one counterfactual may move at most one octave.  This prevents 1.60->0.10
# jumps from a state whose deformation history was produced by the old field.
STIFFNESS_GLOBAL_TRUST_REGION_RATIO = 2.0
# An extreme initialization needs several trustworthy octave-sized moves
# (1.60->0.80->0.40->0.20, or 0.003->0.004). Once the first global move picks
# a direction, later global moves in that material family may not reverse it.
STIFFNESS_ROBUST_MAXIMUM_GLOBAL_COMMITS_PER_FAMILY = 3
# Material identification is a calibration stage, not an endless image-fitting
# controller.  Residual state estimation remains active after this budget.
STIFFNESS_ROBUST_MAXIMUM_COMMITS = 12
# Event-driven material identification has no frame-number cooldown and no
# update/trial budget. A proposal becomes eligible whenever the material is
# observable; one continuous residual-gradient proposal is verified at
# H1/H5/H10 and immediately committed when its RGB and physical gates pass.
STIFFNESS_CAUSAL_WARMUP_END_FRAME = 0
STIFFNESS_CAUSAL_MAXIMUM_COMMITS: int | None = None
STIFFNESS_CAUSAL_MAXIMUM_TRIALS: int | None = None
STIFFNESS_CAUSAL_MINIMUM_TRIAL_INTERVAL_FRAMES = 0
STIFFNESS_CAUSAL_LOCAL_MAXIMUM_COMMITS_PER_PHASE_FAMILY: int | None = None
STIFFNESS_CAUSAL_MINIMUM_SAME_FAMILY_COMMIT_INTERVAL_FRAMES = 0
STIFFNESS_EVENT_DRIVEN_MINIMUM_ACTIVE_PARTICLES = 32
STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES = 24
STIFFNESS_EVENT_DRIVEN_MINIMUM_EMA_PARTICLES = 24
STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT = 1
STIFFNESS_EVENT_DRIVEN_MINIMUM_SCOPE_PARTICLES = 32
STIFFNESS_EVENT_DRIVEN_MINIMUM_SUPPORT_IOU = 0.50
# Direct online material updates do not use H-rollout admission, but one noisy
# flow/depth observation is not enough to identify material.  Require contact
# plus edge-strain support in two independent accepted observation windows.
STIFFNESS_DIRECT_MINIMUM_OBSERVABLE_WINDOWS = 2

VISUAL_RESIDUAL_GAIN_PROFILES = {
    "full_only": (1.0,),
    # A residual is an observer correction, not an impulse.  Compare several
    # gains by the same residual-off H=1/3/5 hold and keep only the best branch.
    "multiscale_hold": (1.0, 0.5, 0.25),
    # Same 2D RGB solve and H1/3/5 hold with less observer authority. These
    # profiles are selected on a non-formal 7:1 validation phase before use.
    "conservative_multiscale_hold": (0.5, 0.25, 0.125),
    "micro_multiscale_hold": (0.25, 0.125, 0.0625),
    # Same H=1/3/5 same-image persistence search as multiscale_hold, followed
    # by a causal validation on the next observable training image.  The
    # corrected branch is advanced as a shadow while the live branch remains
    # uncorrected; only a cross-frame improvement is allowed to enter live
    # state or become material-system-identification evidence.
    "cross_frame_hold": (1.0, 0.5, 0.25),
    # Keep every same-image-safe gain alive until the next observable RGB
    # frame.  The legacy cross_frame_hold commits to one gain before that
    # causal evidence exists, so a full-gain overfit can be rejected one frame
    # later even though its half/quarter branch would have generalized.  This
    # profile ranks the surviving branches on the later image and also permits
    # bounded H=1 solver noise when H=3/H=5 and the weighted hold improve.
    "cross_frame_ranked_hold": (1.0, 0.5, 0.25),
}
TRAJECTORY_FEEDBACK_MODES = {"trajectory", "trajectory_residual"}


def configure_stiffness_admission_policy(
    horizons: tuple[int, ...],
    candidate_profile: str,
    admission_mode: str = "strict_all",
) -> None:
    """Configure a reproducible material-search ablation before runtime."""
    global STIFFNESS_ADMISSION_HORIZONS
    global STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES
    global STIFFNESS_MAXIMUM_PREDICTION_HORIZON_FRAMES
    global STIFFNESS_CANDIDATE_PROFILE
    global STIFFNESS_CANDIDATE_VARIANTS
    global STIFFNESS_ADMISSION_MODE
    normalized = tuple(sorted(int(value) for value in horizons))
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("Stiffness admission horizons must be positive")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Stiffness admission horizons must be unique")
    if candidate_profile not in STIFFNESS_CANDIDATE_PROFILES:
        raise ValueError(
            f"Unknown stiffness candidate profile: {candidate_profile}"
        )
    if admission_mode not in STIFFNESS_ADMISSION_MODES:
        raise ValueError(f"Unknown stiffness admission mode: {admission_mode}")
    if (
        admission_mode in {"causal_fixed_lag", "relaxed_h135"}
        and candidate_profile
        not in {"direct_residual_gradient", "direct_alternating_gradient"}
    ):
        raise ValueError(
            f"{admission_mode} admission requires a single continuous-gradient "
            "proposal, optionally alternating distance/shape axes"
        )
    if admission_mode == "relaxed_h135" and normalized != (1, 3, 5):
        raise ValueError(
            "relaxed_h135 admission requires exactly H1/H3/H5"
        )
    if (
        admission_mode == "direct_online"
        and candidate_profile
        not in {"direct_residual_gradient", "direct_alternating_gradient"}
    ):
        raise ValueError(
            "direct_online admission requires a direct continuous-gradient "
            "profile"
        )
    if (
        admission_mode in {"paper_trajectory_adam", "sim_particle_graph_lm"}
        and candidate_profile != "direct_residual_gradient"
    ):
        raise ValueError(
            f"{admission_mode} uses one continuous optimizer result and requires "
            "the direct_residual_gradient placeholder profile"
        )
    STIFFNESS_ADMISSION_HORIZONS = normalized
    STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES = max(normalized)
    STIFFNESS_MAXIMUM_PREDICTION_HORIZON_FRAMES = max(
        10, STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES
    )
    STIFFNESS_CANDIDATE_PROFILE = candidate_profile
    STIFFNESS_CANDIDATE_VARIANTS = STIFFNESS_CANDIDATE_PROFILES[
        candidate_profile
    ]
    STIFFNESS_ADMISSION_MODE = admission_mode


def stiffness_validation_objective_name() -> str:
    if STIFFNESS_ADMISSION_MODE == "sim_particle_graph_lm":
        return (
            "sim_particle_graph_lm_per_particle_distance_weak_shape_"
            "cauchy_relative_h1_h3_h5_warp_directional_gate"
        )
    if STIFFNESS_ADMISSION_MODE == "paper_trajectory_adam":
        return (
            "contact_edge_strain_observable_alltracker_depth_triangle_track_"
            "plus_4_of_20_history_plus_graph_smooth_bounded_adam"
        )
    if STIFFNESS_ADMISSION_MODE == "direct_online":
        if STIFFNESS_CANDIDATE_PROFILE == "direct_alternating_gradient":
            return (
                "contact_two_strain_windows_then_immediate_"
                "alternating_axis_gradient_no_shadow"
            )
        return (
            "contact_two_strain_windows_then_immediate_"
            "dual_continuous_gradient_no_shadow"
        )
    suffix = "_".join(f"h{value}" for value in STIFFNESS_ADMISSION_HORIZONS)
    if STIFFNESS_ADMISSION_MODE in {"causal_fixed_lag", "relaxed_h135"}:
        reduction = "causal_replayed_cumulative"
    elif STIFFNESS_ADMISSION_MODE == "weighted_window":
        reduction = "weighted_window"
    else:
        reduction = "mean"
    residual_effort = (
        "_residual_effort_tiebreak"
        if (
            STIFFNESS_CANDIDATE_PROFILE
            == "robust_hierarchical_system_id"
        )
        else ""
    )
    if STIFFNESS_ADMISSION_MODE in {"causal_fixed_lag", "relaxed_h135"}:
        policy = (
            "_relaxed_h1_noise_h3_bounded_h5_improve_h35_weighted"
            if STIFFNESS_ADMISSION_MODE == "relaxed_h135"
            else ""
        )
        return (
            f"{reduction}_gap_{suffix}{policy}{residual_effort}"
            "_training_rgb_residual_replay"
        )
    return (
        f"material_isolation_{reduction}_gap_{suffix}"
        f"{residual_effort}_no_residual"
    )


def advance_direct_stiffness_observability(
    *,
    previous_count: int,
    previous_frame_index: int,
    current_frame_index: int,
    contact_count: int,
    strain_active_particles: int,
) -> tuple[int, int, bool, str]:
    """Advance the direct-update contact/strain persistence state.

    The two windows are consecutive accepted material-evidence observations,
    not necessarily adjacent raw video frames.  A loss of contact or edge
    strain resets the sequence.  Re-entering the same frame cannot create a
    second independent confirmation or a duplicate commit.
    """
    if int(contact_count) < STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT:
        return 0, -1, False, "contact_below_minimum"
    if (
        int(strain_active_particles)
        < STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES
    ):
        return 0, -1, False, "edge_strain_below_minimum"
    if int(current_frame_index) <= int(previous_frame_index):
        return (
            int(previous_count),
            int(previous_frame_index),
            False,
            "non_independent_observation_frame",
        )
    count = min(
        max(int(previous_count), 0) + 1,
        STIFFNESS_DIRECT_MINIMUM_OBSERVABLE_WINDOWS,
    )
    ready = count >= STIFFNESS_DIRECT_MINIMUM_OBSERVABLE_WINDOWS
    return (
        count,
        int(current_frame_index),
        ready,
        "" if ready else "edge_strain_confirmation_pending",
    )


def stiffness_adopt_validated_rollout_enabled() -> bool:
    """Only a causal replay shadow is eligible to advance live state."""
    return bool(
        STIFFNESS_ADOPT_VALIDATED_ROLLOUT
        or STIFFNESS_ADMISSION_MODE == "causal_fixed_lag"
    )


def stiffness_target_inside_global_trust_region(
    current: float, target: float
) -> bool:
    if current <= 0.0 or target <= 0.0:
        raise ValueError("Stiffness trust-region values must be positive")
    ratio = STIFFNESS_GLOBAL_TRUST_REGION_RATIO
    return current / ratio <= target <= current * ratio


def stiffness_target_follows_global_direction(
    current: float, target: float, direction: int
) -> bool:
    """Return whether a global target preserves an already chosen direction."""
    if direction not in (-1, 0, 1):
        raise ValueError("Global stiffness direction must be -1, 0, or 1")
    if current <= 0.0 or target <= 0.0:
        raise ValueError("Global stiffness direction values must be positive")
    if np.isclose(current, target, rtol=1.0e-6, atol=1.0e-9):
        return False
    target_direction = 1 if target > current else -1
    return direction == 0 or target_direction == direction
# Material-isolation shadows are counterfactual experiments.  Per the written
# protocol, a passed gate commits k_verified only; copying the residual-off H=5
# particle state into the live residual-on trajectory would break isolation.
STIFFNESS_ADOPT_VALIDATED_ROLLOUT = False
STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M = 0.00025
STIFFNESS_MAXIMUM_ADOPTED_STATE_MAXIMUM_M = 0.001
STIFFNESS_HISTORY_MAXIMUM_SNAPSHOTS = 4
STIFFNESS_HISTORY_RELATIVE_TOLERANCE = 0.10
STIFFNESS_HISTORY_ABSOLUTE_TOLERANCE_M = 1.0e-5
STIFFNESS_HISTORY_MAXIMUM_PARTICLE_SPEED_M_S = 0.02
STIFFNESS_HISTORY_MAXIMUM_JAW_SPEED_RAD_S = 0.10

# A confirmed image innovation updates both q and qd through a bounded
# alpha-beta observer.  Every candidate must still survive these no-vision
# physics horizons and the later training-RGB gate before entering live state.
VISUAL_RESIDUAL_VELOCITY_CORRECTION_GAIN = 0.50
VISUAL_RESIDUAL_MAXIMUM_VELOCITY_CORRECTION_M_S = 0.03
VISUAL_RESIDUAL_PERSISTENCE_HORIZONS = (1, 3, 5)


@dataclass(frozen=True)
class StiffnessToolCommand:
    frame_index: int
    timestep: float
    state_index: int
    phase: str = "idle"


@dataclass
class StiffnessHistorySnapshot:
    rollout_state: object
    accepted_positions: torch.Tensor
    frame_index: int


@dataclass(frozen=True)
class PaperAdamCausalObservation:
    """One transition in a supervised H3 or physical 3-of-4 window."""

    source_state: object
    source_frame: int
    commands: tuple[StiffnessToolCommand, ...]
    destination_frame: int
    target_points: torch.Tensor
    confidence: torch.Tensor
    track_valid: torch.Tensor
    supervision_valid: bool = True
    material_active_mask: torch.Tensor | None = None
    local_distance_signal: torch.Tensor | None = None


@dataclass(frozen=True)
class SimParticleGraphTransition:
    """One causal SUPER transition replayed by the SIM Warp direction gate."""

    source_state: object
    source_frame: int
    commands: tuple[StiffnessToolCommand, ...]
    destination_frame: int
    corrected_positions: torch.Tensor
    eligible_mask: torch.Tensor
    material_active_mask: torch.Tensor
    control_exclusion_mask: torch.Tensor
    track_target_positions: torch.Tensor
    track_valid_mask: torch.Tensor


@dataclass
class PendingStiffnessValidation:
    candidate: PaperStiffnessCandidate
    rollout_state: object
    frame_index: int
    grip_active: bool
    history_baseline_rms_m: float
    history_candidate_rms_m: float
    # Freeze the semantic phase at proposal time.  Validation may finish ten
    # observable frames later, after the live classifier has changed phase.
    proposal_phase: str = "idle"
    previous_residual: torch.Tensor | None = None
    commands: list[StiffnessToolCommand] = field(default_factory=list)
    # Frozen at proposal time. For reconstruction_7to1 these are the frame
    # indices of the 1st/3rd/5th/10th future *training observations*, not raw
    # video offsets. Held-out RGB is therefore never used for admission.
    validation_frame_indices: tuple[int, ...] = ()
    # After the H1/H5/H10 search selects one candidate, only that winner is
    # replayed to H30/H60.  This preserves long-horizon evidence without
    # multiplying every broad/local search branch by sixty frames.
    long_validation_candidate: PaperStiffnessCandidate | None = None
    long_validation_frame_indices: tuple[int, ...] = ()
    short_selection_record: dict | None = None


@dataclass
class PendingVisualResidualValidation:
    """A visual state correction awaiting a later training RGB image."""

    result: object
    candidate_state: object
    frame_index: int
    selected_gain: float
    commands: list[StiffnessToolCommand] = field(default_factory=list)
    quality_valid_mask: torch.Tensor | None = None
    supervision_valid_mask: torch.Tensor | None = None
    control_exclusion_mask: torch.Tensor | None = None
    stiffness_gate_paused: bool = True
    stiffness_jaw_speed_rad_s: float = float("inf")
    stiffness_grip_active: bool = False
    stiffness_contact_count: int = 0
    candidate_branches: tuple["VisualResidualCandidateBranch", ...] = ()


@dataclass(frozen=True)
class VisualResidualCandidateBranch:
    """One gain branch awaiting causal selection on a later training RGB."""

    result: object
    candidate_state: object
    selected_gain: float
    persistence_score: float
    persistence_improvements: dict[int, float | None]


@dataclass(frozen=True)
class InstalledParticleInnovation:
    """Minimal accepted-state view consumed by the stiffness estimator."""

    corrected_positions: torch.Tensor
    residual: torch.Tensor


@dataclass
class CommittedStiffnessEvaluation:
    """One committed candidate awaiting exact multi-horizon evaluation."""

    evaluation_id: int
    candidate: PaperStiffnessCandidate
    rollout_state: object
    start_frame_index: int
    start_phase: str
    commands: list[StiffnessToolCommand]
    pending_horizons: set[int]
    previous_residual: torch.Tensor | None = None


@dataclass(frozen=True)
class ContactGripGuiSettings:
    """Editable contact/grip policy copied from the live runtime."""

    sample_spacing_m: float
    contact_spread_layers: int
    contact_margin_m: float
    query_distance_m: float
    ccd_velocity_scale: float
    contact_relaxation: float
    contact_max_correction_m: float
    top_barrier_max_correction_m: float
    contact_iterations: int
    post_contact_material_iterations: int
    final_barrier_max_correction_m: float
    contact_min_volume_ratio: float
    contact_substep_stride: int
    contact_projection_velocity_scale: float
    material_projection_velocity_scale: float
    particle_velocity_damping_per_second: float
    top_barrier_lateral_tolerance_m: float
    top_barrier_contact_patch_radius_m: float
    top_barrier_clearance_m: float
    jaw_friction_coefficient: float
    jaw_contact_distal_length_m: float
    top_barrier_distal_length_m: float
    top_barrier_tip_allowance_m: float
    persistent_grip_enabled: bool
    grip_minimum_contact_samples_per_jaw: int
    grip_maximum_jaw_patch_separation_m: float
    grip_activation_steps: int
    grip_maximum_capture_penetration_m: float
    grip_minimum_capture_volume_ratio: float
    grip_closed_angle_max_rad: float
    grip_release_angle_min_rad: float
    grip_release_angle_delta_rad: float
    grip_wide_open_angle_rad: float
    grip_angle_motion_epsilon_rad: float
    grip_compliance_m_per_n: float
    grip_relaxation: float
    grip_maximum_correction_m: float
    grip_transfer_layers: int
    grip_minimum_volume_ratio: float
    grip_support_radius_m: float
    grip_support_generations: int

    @classmethod
    def from_runtime(cls, physics, projector) -> ContactGripGuiSettings:
        return cls(
            sample_spacing_m=float(projector.sample_spacing_m),
            contact_spread_layers=int(projector.spread_layers),
            contact_margin_m=float(physics.triangle_skin_contact_margin_m),
            query_distance_m=float(physics.triangle_skin_query_distance_m),
            ccd_velocity_scale=float(
                physics.triangle_skin_ccd_velocity_scale
            ),
            contact_relaxation=float(
                physics.triangle_skin_contact_relaxation
            ),
            contact_max_correction_m=float(
                physics.triangle_skin_contact_max_correction_m
            ),
            top_barrier_max_correction_m=float(
                physics.triangle_skin_top_barrier_max_correction_m
            ),
            contact_iterations=int(
                physics.triangle_skin_contact_iterations
            ),
            post_contact_material_iterations=int(
                physics.triangle_skin_post_contact_material_iterations
            ),
            final_barrier_max_correction_m=float(
                physics.triangle_skin_final_barrier_max_correction_m
            ),
            contact_min_volume_ratio=float(
                physics.triangle_skin_contact_min_volume_ratio
            ),
            contact_substep_stride=int(
                physics.triangle_skin_contact_substep_stride
            ),
            contact_projection_velocity_scale=float(
                physics.contact_projection_velocity_scale
            ),
            material_projection_velocity_scale=float(
                physics.material_projection_velocity_scale
            ),
            particle_velocity_damping_per_second=float(
                physics.particle_velocity_damping_per_second
            ),
            top_barrier_lateral_tolerance_m=float(
                projector.top_barrier_lateral_tolerance_m
            ),
            top_barrier_contact_patch_radius_m=float(
                projector.top_barrier_contact_patch_radius_m
            ),
            top_barrier_clearance_m=float(
                projector.top_barrier_clearance_m
            ),
            jaw_friction_coefficient=float(
                projector.jaw_friction_coefficient
            ),
            jaw_contact_distal_length_m=float(
                projector.jaw_contact_distal_length_m
            ),
            top_barrier_distal_length_m=float(
                projector.top_barrier_distal_length_m
            ),
            top_barrier_tip_allowance_m=float(
                projector.top_barrier_tip_allowance_m
            ),
            persistent_grip_enabled=bool(
                projector.persistent_grip_enabled
            ),
            grip_minimum_contact_samples_per_jaw=int(
                projector.persistent_grip_minimum_contact_samples_per_jaw
            ),
            grip_maximum_jaw_patch_separation_m=float(
                projector.persistent_grip_maximum_jaw_patch_separation_m
            ),
            grip_activation_steps=int(
                projector.persistent_grip_activation_steps
            ),
            grip_maximum_capture_penetration_m=float(
                projector.persistent_grip_maximum_capture_penetration_m
            ),
            grip_minimum_capture_volume_ratio=float(
                projector.persistent_grip_minimum_capture_volume_ratio
            ),
            grip_closed_angle_max_rad=float(
                projector.persistent_grip_closed_angle_max_rad
            ),
            grip_release_angle_min_rad=float(
                projector.persistent_grip_release_angle_min_rad
            ),
            grip_release_angle_delta_rad=float(
                projector.persistent_grip_release_angle_delta_rad
            ),
            grip_wide_open_angle_rad=float(
                projector.persistent_grip_wide_open_angle_rad
            ),
            grip_angle_motion_epsilon_rad=float(
                projector.persistent_grip_angle_motion_epsilon_rad
            ),
            grip_compliance_m_per_n=float(
                projector.persistent_grip_compliance_m_per_n
            ),
            grip_relaxation=float(
                projector.persistent_grip_relaxation
            ),
            grip_maximum_correction_m=float(
                projector.persistent_grip_maximum_correction_m
            ),
            grip_transfer_layers=int(
                projector.persistent_grip_transfer_layers
            ),
            grip_minimum_volume_ratio=float(
                projector.persistent_grip_minimum_volume_ratio
            ),
            grip_support_radius_m=float(
                projector.persistent_grip_support_radius_m
            ),
            grip_support_generations=int(
                projector.persistent_grip_support_generations
            ),
        )

    def validate(self) -> None:
        scalar_values = (
            self.sample_spacing_m,
            self.contact_margin_m,
            self.query_distance_m,
            self.ccd_velocity_scale,
            self.contact_relaxation,
            self.contact_max_correction_m,
            self.top_barrier_max_correction_m,
            self.final_barrier_max_correction_m,
            self.contact_min_volume_ratio,
            self.contact_projection_velocity_scale,
            self.material_projection_velocity_scale,
            self.particle_velocity_damping_per_second,
            self.top_barrier_lateral_tolerance_m,
            self.top_barrier_contact_patch_radius_m,
            self.top_barrier_clearance_m,
            self.jaw_friction_coefficient,
            self.jaw_contact_distal_length_m,
            self.top_barrier_distal_length_m,
            self.top_barrier_tip_allowance_m,
            self.grip_maximum_jaw_patch_separation_m,
            self.grip_maximum_capture_penetration_m,
            self.grip_minimum_capture_volume_ratio,
            self.grip_closed_angle_max_rad,
            self.grip_release_angle_min_rad,
            self.grip_release_angle_delta_rad,
            self.grip_wide_open_angle_rad,
            self.grip_angle_motion_epsilon_rad,
            self.grip_compliance_m_per_n,
            self.grip_relaxation,
            self.grip_maximum_correction_m,
            self.grip_minimum_volume_ratio,
            self.grip_support_radius_m,
        )
        if not all(np.isfinite(value) for value in scalar_values):
            raise ValueError("contact/grip settings must be finite")
        if self.sample_spacing_m <= 0.0:
            raise ValueError("contact sample spacing must be positive")
        if self.contact_spread_layers < 0:
            raise ValueError("contact spread layers cannot be negative")
        if self.contact_margin_m < 0.0:
            raise ValueError("contact margin cannot be negative")
        if self.query_distance_m < self.contact_margin_m:
            raise ValueError("query distance must be at least the contact margin")
        if self.ccd_velocity_scale < 0.0:
            raise ValueError("CCD velocity scale cannot be negative")
        if not 0.0 < self.contact_relaxation <= 1.0:
            raise ValueError("contact relaxation must lie in (0, 1]")
        if min(
            self.contact_max_correction_m,
            self.top_barrier_max_correction_m,
            self.final_barrier_max_correction_m,
        ) < 0.0:
            raise ValueError("contact correction caps cannot be negative")
        if self.contact_iterations < 1:
            raise ValueError("contact iterations must be positive")
        if self.post_contact_material_iterations < 0:
            raise ValueError("post-contact material iterations cannot be negative")
        if self.contact_substep_stride < 1:
            raise ValueError("contact substep stride must be positive")
        if not 0.0 <= self.contact_min_volume_ratio <= 1.0:
            raise ValueError("contact minimum J must lie in [0, 1]")
        if not 0.0 <= self.contact_projection_velocity_scale <= 1.0:
            raise ValueError("contact velocity transfer must lie in [0, 1]")
        if not 0.0 <= self.material_projection_velocity_scale <= 1.0:
            raise ValueError("material velocity transfer must lie in [0, 1]")
        if self.particle_velocity_damping_per_second < 0.0:
            raise ValueError("particle velocity damping cannot be negative")
        if self.top_barrier_lateral_tolerance_m <= 0.0:
            raise ValueError("top barrier lateral tolerance must be positive")
        if min(
            self.top_barrier_contact_patch_radius_m,
            self.top_barrier_clearance_m,
            self.jaw_friction_coefficient,
            self.jaw_contact_distal_length_m,
            self.top_barrier_distal_length_m,
            self.top_barrier_tip_allowance_m,
        ) < 0.0:
            raise ValueError("contact geometry/friction values cannot be negative")
        barrier_distal = (
            self.top_barrier_distal_length_m
            if self.top_barrier_distal_length_m > 0.0
            else self.jaw_contact_distal_length_m
        )
        if barrier_distal > self.jaw_contact_distal_length_m:
            raise ValueError("top barrier length cannot exceed jaw contact length")
        if barrier_distal > 0.0 and self.top_barrier_tip_allowance_m >= barrier_distal:
            raise ValueError("tip entry allowance must be shorter than top barrier")
        if self.grip_minimum_contact_samples_per_jaw < 1:
            raise ValueError("grip samples per jaw must be positive")
        if self.grip_maximum_jaw_patch_separation_m <= 0.0:
            raise ValueError("grip patch separation must be positive")
        if self.grip_activation_steps < 1:
            raise ValueError("grip activation steps must be positive")
        if self.grip_maximum_capture_penetration_m <= 0.0:
            raise ValueError("grip capture penetration must be positive")
        if not 0.0 < self.grip_minimum_capture_volume_ratio <= 1.0:
            raise ValueError("grip capture minimum J must lie in (0, 1]")
        if not (
            self.grip_closed_angle_max_rad
            < self.grip_release_angle_min_rad
            <= self.grip_wide_open_angle_rad
        ):
            raise ValueError("grip angles must satisfy closed < release <= wide-open")
        if self.grip_release_angle_delta_rad <= 0.0:
            raise ValueError("grip release angle delta must be positive")
        if self.grip_angle_motion_epsilon_rad <= 0.0:
            raise ValueError("grip motion epsilon must be positive")
        if self.grip_compliance_m_per_n < 0.0:
            raise ValueError("grip compliance cannot be negative")
        if not 0.0 < self.grip_relaxation <= 1.0:
            raise ValueError("grip relaxation must lie in (0, 1]")
        if self.grip_maximum_correction_m < 0.0:
            raise ValueError("grip correction cap cannot be negative")
        if self.grip_transfer_layers < 1:
            raise ValueError("grip transfer layers must be positive")
        if not 0.0 < self.grip_minimum_volume_ratio <= 1.0:
            raise ValueError("grip minimum J must lie in (0, 1]")
        if self.grip_support_radius_m < 0.0:
            raise ValueError("grip support radius cannot be negative")
        if self.grip_support_generations < 1:
            raise ValueError("grip support generations must be positive")
if (
    LOADED_EMBODIED_GAUSSIANS_SOURCE
    != EXPECTED_EMBODIED_GAUSSIANS_SOURCE
):
    raise RuntimeError(
        "Loaded embodied_gaussians from the wrong checkout: "
        f"{LOADED_EMBODIED_GAUSSIANS_SOURCE}; expected "
        f"{EXPECTED_EMBODIED_GAUSSIANS_SOURCE}"
    )


VISUAL_FORCE_MASK_DIR = (
    repo_root / "data/super/grasp5_native/visual_force_masks_v1"
)
RIGHT_VISUAL_FORCE_MASK_DIR = (
    repo_root / "data/super/grasp5_native/visual_force_masks_right_v1"
)
VISUAL_FORCE_INSTRUMENT_MASKS = (
    repo_root
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_dense_contact_v4/"
    "surgicalsam2_multianchor_parts_dense_contact_v6/"
    "stereo_multianchor_part_masks.npz"
)


def stiffness_local_quality_mask(
    mapper: TetrahedralGaussianVisualResidualMapper,
    physical_prediction: torch.Tensor,
    corrected_positions: torch.Tensor,
    minimum_volume_ratio: float,
) -> torch.Tensor:
    """Return particles whose incident tets are valid before and after residual."""

    def volume_ratios(positions: torch.Tensor) -> torch.Tensor:
        points = positions[mapper.tet_indices]
        signed_six_volume = torch.linalg.det(
            torch.stack(
                (
                    points[:, 1] - points[:, 0],
                    points[:, 2] - points[:, 0],
                    points[:, 3] - points[:, 0],
                ),
                dim=-1,
            )
        )
        return signed_six_volume / (6.0 * mapper.rest_volumes)

    initial_ratio = volume_ratios(physical_prediction)
    corrected_ratio = volume_ratios(corrected_positions)
    valid_tets = (
        torch.isfinite(initial_ratio)
        & torch.isfinite(corrected_ratio)
        & (initial_ratio >= minimum_volume_ratio)
        & (corrected_ratio >= minimum_volume_ratio)
    )
    valid_particles = torch.ones_like(mapper.fixed_mask)
    invalid_particle_ids = mapper.tet_indices[~valid_tets].reshape(-1)
    if invalid_particle_ids.numel():
        valid_particles[invalid_particle_ids] = False
    valid_particles[mapper.fixed_mask] = False
    return valid_particles


def stiffness_visual_supervision_mask(
    mapper: TetrahedralGaussianVisualResidualMapper,
    visual_gradient_norm: torch.Tensor,
) -> torch.Tensor:
    """Select nodes with a non-negligible masked RGB data gradient."""
    gradient = visual_gradient_norm.detach().to(
        device=mapper.rest_positions.device, dtype=torch.float32
    )
    if gradient.shape != mapper.fixed_mask.shape:
        raise ValueError("Visual supervision gradient has the wrong shape")
    finite = torch.isfinite(gradient)
    dynamic = finite & ~mapper.fixed_mask
    maximum = gradient[dynamic].max() if bool(dynamic.any().item()) else None
    if maximum is None or float(maximum.item()) <= 0.0:
        return torch.zeros_like(mapper.fixed_mask)
    threshold = max(
        float(maximum.item()) * STIFFNESS_VISUAL_GRADIENT_RELATIVE_FLOOR,
        torch.finfo(gradient.dtype).tiny,
    )
    return dynamic & (gradient >= threshold)


def grip_control_exclusion_mask(
    mapper: TetrahedralGaussianVisualResidualMapper,
    environment: EmbodiedGaussiansEnvironment,
) -> torch.Tensor:
    """Mask active direct/support controls and exactly one small tet ring."""
    projector = environment.sim.triangle_skin_contact_projector
    excluded = torch.zeros_like(mapper.fixed_mask)
    if projector is None:
        return excluded
    if not bool(projector.persistent_grip_state.numpy()[0]):
        return excluded
    grip_body = wp.to_torch(projector.persistent_grip_particle_body).to(
        device=excluded.device
    )
    excluded |= grip_body >= 0
    excluded = strict_one_tetrahedron_ring_mask(
        excluded, mapper.tet_indices
    )
    excluded[mapper.fixed_mask] = True
    return excluded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 SUPER grasp5 离线回放 demo。")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "SUPER 离线数据目录。目录内需要包含 robots.json、cameras.json、"
            "videos/stereo_left.mp4、videos/stereo_right.mp4 和对应 json。"
        ),
    )
    parser.add_argument("--fps", type=int, default=30, help="离线回放速度。默认按视频约 30 FPS 播放。")
    parser.add_argument(
        "--monitor-psm-base-q",
        action="store_true",
        help="打印 PSM 基座 body_q，诊断机械臂基座是否真的在物理仿真中漂移。",
    )
    parser.add_argument(
        "--monitor-tissue-q",
        action="store_true",
        help="打印 tissue 刚体 body_q，诊断组织是否被物理或视觉力拉走。",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.5,
        help="body_q 监控打印间隔，单位秒。默认 0.5 秒。",
    )
    parser.add_argument(
        "--visual-feedback-mode",
        choices=("trajectory", "trajectory_residual", "residual", "force", "off"),
        default="residual",
        help=(
            "视觉反馈方式：trajectory=CoTracker+depth固定范围轨迹修正；"
            "trajectory_residual=轨迹/刚度更新后再做一次受轨迹保持约束的RGB微残差；"
            "residual=旧RGB物理节点残差实时回写（默认）；"
            "force=旧 Gaussian 位移转粒子力；off=关闭视觉反馈。"
        ),
    )
    parser.add_argument(
        "--flow-depth-bindings",
        type=Path,
        default=(
            repo_root
            / "outputs/grasp5_cotracker3_range_bindings_20260828_v1/bindings.npz"
        ),
        help="trajectory 模式使用的固定轨迹到物理表面范围绑定。",
    )
    parser.add_argument(
        "--flow-depth-observations",
        type=Path,
        default=(
            repo_root
            / "outputs/grasp5_cotracker3_causal_sparse_depth_observations_20260828_v2/observations.npz"
        ),
        help="trajectory 模式使用的因果预计算3D flow观测。",
    )
    parser.add_argument(
        "--flow-depth-position-gain",
        type=float,
        default=0.60,
        help="联合绝对位置/flow innovation位置增益，默认0.60。",
    )
    parser.add_argument(
        "--flow-depth-velocity-gain",
        type=float,
        default=0.30,
        help="直接观测速度innovation增益，默认0.30。",
    )
    parser.add_argument(
        "--flow-depth-absolute-position-weight",
        type=float,
        default=0.80,
        help="绝对下一时刻3D位置残差在联合观测中的权重，默认0.80。",
    )
    parser.add_argument(
        "--flow-depth-solver-regularization",
        type=float,
        default=0.02,
        help="重叠轨迹联合最小二乘正则，默认0.02。",
    )
    parser.add_argument(
        "--flow-depth-solver-iterations",
        type=int,
        default=16,
        help="重叠轨迹联合CG求解迭代次数，默认16。",
    )
    parser.add_argument(
        "--flow-depth-robust-residual-mm",
        type=float,
        default=20.0,
        help="轨迹残差稳健降权尺度，默认20 mm，与最大有效3D flow同量级。",
    )
    parser.add_argument(
        "--flow-depth-maximum-position-correction-mm",
        type=float,
        default=20.0,
        help="单次轨迹视觉粒子修正上限，默认20 mm；轨迹模式不执行全局回溯。",
    )
    parser.add_argument(
        "--flow-depth-maximum-velocity-correction-m-s",
        type=float,
        default=0.25,
        help="单次轨迹视觉速度修正上限，默认0.25 m/s。",
    )
    parser.add_argument(
        "--flow-depth-local-material-relaxation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "只软化轨迹绑定粒子的严格一层四面体邻域及其直接关联四面体，"
            "降低下一帧XPBD拉回；远区仍保持初始材料刚度。"
        ),
    )
    parser.add_argument(
        "--flow-depth-local-distance-stiffness",
        type=float,
        default=0.001,
        help="局部轨迹邻域的distance stiffness，默认0.001。",
    )
    parser.add_argument(
        "--flow-depth-local-volume-stiffness",
        type=float,
        default=1.0e3,
        help="轨迹粒子直接关联四面体的volume stiffness，默认1e3。",
    )
    parser.add_argument(
        "--flow-depth-local-shape-stiffness",
        type=float,
        default=0.00005,
        help="局部轨迹邻域的shape stiffness，默认5e-5。",
    )
    parser.add_argument(
        "--visual-force-iterations",
        type=int,
        default=1,
        help=(
            "tissue visual forces 每次更新的图像优化迭代数，默认 1；"
            "PSM 始终不参与。"
        ),
    )
    parser.add_argument(
        "--visual-residual-iterations",
        type=int,
        default=8,
        help="每次实时视觉残差更新的 Adam 迭代数，默认 8。",
    )
    parser.add_argument(
        "--visual-residual-learning-rate-m",
        type=float,
        default=1.0e-5,
        help="实时视觉残差节点学习率，单位 m，默认保守值 1.0e-5。",
    )
    parser.add_argument(
        "--visual-residual-maximum-step-m",
        type=float,
        default=1.0e-4,
        help="每个观测对单个物理节点的最大增量，单位 m，默认 1.0e-4（0.10 mm）。",
    )
    parser.add_argument(
        "--visual-residual-previous-carry",
        type=float,
        default=0.0,
        help=(
            "上一帧已应用 residual 作为下一帧增量正则目标的比例；"
            "默认 0，避免重复写入产生长期漂移。"
        ),
    )
    parser.add_argument(
        "--visual-residual-temporal-weight",
        type=float,
        default=0.10,
        help="视觉增量时间/收缩正则权重，默认 0.10。",
    )
    parser.add_argument(
        "--visual-residual-magnitude-weight",
        type=float,
        default=0.01,
        help="视觉增量幅值正则权重，默认 0.01。",
    )
    parser.add_argument(
        "--visual-residual-gain-profile",
        choices=tuple(VISUAL_RESIDUAL_GAIN_PROFILES),
        default="full_only",
        help=(
            "residual 回写增益：full_only 保留旧行为；multiscale_hold 会把 "
            "1/0.5/0.25 三种修正分别做无视觉 H=1/3/5 保持，选择持续误差最小"
            "且每个 horizon 均改善的一支；conservative/micro 使用更小的三档"
            "增益，供独立 7:1 验证相位抑制跨帧漂移。"
        ),
    )
    parser.add_argument(
        "--trajectory-rgb-residual-track-weight",
        type=float,
        default=0.02,
        help=(
            "轨迹后RGB微残差对 sum_j c_j||B_j delta x||^2 的权重，"
            "默认0.02；只在trajectory_residual模式生效。"
        ),
    )
    parser.add_argument(
        "--trajectory-rgb-residual-position-gain",
        type=float,
        default=1.0,
        help=(
            "轨迹后RGB微残差的位置回写增益，范围(0,1]，默认1.0；"
            "只缩放RGB微修正，不缩放AllTracker/depth轨迹修正。"
        ),
    )
    parser.add_argument(
        "--trajectory-rgb-residual-velocity-gain",
        type=float,
        default=0.15,
        help="轨迹后RGB微残差的速度观测增益，默认0.15。",
    )
    parser.add_argument(
        "--trajectory-rgb-residual-maximum-velocity-m-s",
        type=float,
        default=0.015,
        help="轨迹后RGB微残差的单粒子速度修正上限，默认0.015m/s。",
    )
    parser.add_argument(
        "--trajectory-gaussian-appearance-iterations",
        type=int,
        default=0,
        help=(
            "轨迹与RGB位置修正后，对组织高斯颜色/透明度执行的外观Adam步数；"
            "建议2–3，默认0保持旧实验不变。"
        ),
    )
    parser.add_argument(
        "--trajectory-gaussian-color-learning-rate",
        type=float,
        default=0.02,
        help="轨迹后组织高斯颜色logit学习率，默认0.02。",
    )
    parser.add_argument(
        "--trajectory-gaussian-opacity-learning-rate",
        type=float,
        default=0.005,
        help="轨迹后组织高斯透明度logit学习率，默认0.005。",
    )
    parser.add_argument(
        "--trajectory-gaussian-maximum-color-logit-offset",
        type=float,
        default=0.20,
        help="颜色相对初始高斯的累计logit偏移上限，默认0.20。",
    )
    parser.add_argument(
        "--trajectory-gaussian-maximum-opacity-logit-offset",
        type=float,
        default=0.10,
        help="透明度相对初始高斯的累计logit偏移上限，默认0.10。",
    )
    parser.add_argument(
        "--trajectory-gaussian-optimize-opacity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "是否允许轨迹后外观块学习透明度；默认关闭，避免轮廓/可见性"
            "变化干扰后续几何。"
        ),
    )
    parser.add_argument(
        "--trajectory-gaussian-appearance-image-scale",
        type=float,
        default=0.5,
        help="外观优化分辨率，默认0.5，与正式渲染评分一致。",
    )
    parser.add_argument(
        "--trajectory-gaussian-appearance-dssim-weight",
        type=float,
        default=0.02,
        help="正式掩膜DSSIM在外观损失中的权重，默认0.02。",
    )
    parser.add_argument(
        "--visual-force-update-interval",
        "--visual-feedback-update-interval",
        dest="visual_force_update_interval",
        type=int,
        default=3,
        help=(
            "每隔多少个物理步重新计算一次双目视觉力，默认 3；"
            "中间物理步复用最近一次力。"
        ),
    )
    parser.add_argument(
        "--online-stiffness-update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "accepted residual 先生成 paper distance/shape 候选，等待 H=1/3/5 "
            "验证后才提交；legacy 模式使用 residual-off 材料隔离，causal "
            "fixed-lag 模式重放训练期 residual/抓持演化；volume 刚度固定。"
        ),
    )
    parser.add_argument(
        "--stiffness-log-learning-rate",
        type=float,
        default=0.10,
        help="在线刚度 log-space 学习率，事件驱动默认 0.10。",
    )
    parser.add_argument(
        "--stiffness-maximum-log-step",
        type=float,
        default=0.10,
        help=(
            "单次候选的绝对 log-step 上限，默认 0.10；"
            "0.8/1.0 分支对应约 0.08/0.10 的 trust step。"
        ),
    )
    parser.add_argument(
        "--stiffness-effective-log-step-target",
        type=float,
        default=0.0,
        help=(
            "distance/shape 独立连续梯度在有效区域内期望达到的最大 log-step；"
            "0 表示不放大，正式双梯度实验使用 0.08，仍受 maximum-log-step 限制。"
        ),
    )
    parser.add_argument(
        "--stiffness-maximum-step-amplification",
        type=float,
        default=4.0,
        help="弱连续梯度达到有效步长目标时允许的最大放大倍数，默认 4。",
    )
    parser.add_argument(
        "--stiffness-strain-signal-weight",
        type=float,
        default=0.0,
        help=(
            "局部材料证据中边应变信号的权重，范围 0..1。默认 0 保留旧结果；"
            "causal fixed-lag 推荐 0.5，使 sim 的局部应变证据与视觉位移证据等权。"
        ),
    )
    parser.add_argument(
        "--stiffness-hardening-bias",
        type=float,
        default=0.0,
        help=(
            "残差方向的常数硬化偏置；新策略固定使用无偏默认值 0，"
            "保留参数仅用于旧实验复现。"
        ),
    )
    parser.add_argument(
        "--stiffness-minimum-residual-mm",
        type=float,
        default=0.02,
        help="进入刚度梯度的最小已接受视觉位移，默认0.02mm。",
    )
    parser.add_argument(
        "--stiffness-minimum-deformation-mm",
        type=float,
        default=0.10,
        help="进入刚度梯度的最小物理形变，默认0.10mm。",
    )
    parser.add_argument(
        "--stiffness-distance-minimum",
        type=float,
        default=0.10,
        help="在线 distance stiffness 搜索下界，默认 0.10；正式极值恢复实验显式放宽。",
    )
    parser.add_argument(
        "--stiffness-distance-maximum",
        type=float,
        default=2.0,
        help="在线 distance stiffness 搜索上界，默认 2.0；正式极值恢复实验显式放宽。",
    )
    parser.add_argument(
        "--stiffness-shape-minimum",
        type=float,
        default=0.003,
        help="在线 shape stiffness 搜索下界，默认 0.003；正式极值恢复实验显式放宽。",
    )
    parser.add_argument(
        "--stiffness-shape-maximum",
        type=float,
        default=0.020,
        help="在线 shape stiffness 搜索上界，默认 0.020；正式极值恢复实验显式放宽。",
    )
    parser.add_argument(
        "--paper-stiffness-adam-learning-rate",
        type=float,
        default=0.10,
        help="论文式轨迹/历史损失对有界材料logit的Adam学习率，默认0.10。",
    )
    parser.add_argument(
        "--paper-stiffness-perturbation",
        type=float,
        default=0.05,
        help="实际XPBD中心反事实梯度的logit扰动，默认0.05。",
    )
    parser.add_argument(
        "--paper-stiffness-track-robust-scale-mm",
        type=float,
        default=3.0,
        help="AllTracker+深度3D Huber轨迹损失尺度，默认3mm。",
    )
    parser.add_argument(
        "--paper-stiffness-history-scale-mm",
        type=float,
        default=1.0,
        help="4/20零控制历史静力RMS归一化尺度，默认1mm。",
    )
    parser.add_argument(
        "--paper-stiffness-history-weight",
        type=float,
        default=0.10,
        help="论文式历史静力一致性损失权重，默认0.10。",
    )
    parser.add_argument(
        "--paper-stiffness-distance-smooth-weight",
        type=float,
        default=0.001,
        help="distance log刚度图平滑损失权重，默认0.001。",
    )
    parser.add_argument(
        "--paper-stiffness-shape-smooth-weight",
        type=float,
        default=0.001,
        help="shape log刚度图平滑损失权重，默认0.001。",
    )
    parser.add_argument(
        "--paper-stiffness-minimum-axis-loss-difference",
        type=float,
        default=1.0e-7,
        help="中心差分低于该数值噪声时该材料轴Adam梯度置零。",
    )
    parser.add_argument(
        "--paper-stiffness-sim-global-causal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "采用SIM验证过的三观测全局distance+damping系统辨识；"
            "shape与已知夹持耦合保持固定。"
        ),
    )
    parser.add_argument(
        "--paper-stiffness-three-of-four-h3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "允许H3跨最多4个连续物理转移并忽略其中1个无监督端点；"
            "缺失转移仍完整执行XPBD和器械命令。"
        ),
    )
    parser.add_argument(
        "--paper-stiffness-local-distance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="在成功的全局H3之后验证一个零均值局部distance刚度场。",
    )
    parser.add_argument(
        "--paper-stiffness-local-distance-log-step",
        type=float,
        default=0.008,
        help="局部distance场每次H3验证的最大log步长，默认0.008。",
    )
    parser.add_argument(
        "--paper-stiffness-local-distance-log-offset",
        type=float,
        default=0.04,
        help="零均值局部distance场累计log范围，默认±0.04。",
    )
    parser.add_argument(
        "--paper-stiffness-local-distance-minimum-improvement",
        type=float,
        default=1.0e-6,
        help="局部候选相对全局H3基线所需的最小损失下降。",
    )
    parser.add_argument(
        "--stiffness-rejected-ema-keep-ratio",
        type=float,
        default=0.85,
        help=(
            "候选拒绝后保留其冻结证据的比例，默认 0.85；"
            "较小值更快忘记已经验证失败的局部方向。"
        ),
    )
    parser.add_argument(
        "--stiffness-admission-horizons",
        type=str,
        default="1,3,5",
        help=(
            "候选提交前必须逐点改善的视频帧 horizon。默认 1,3,5；"
            "长时域消融使用 1,3,5,10。"
        ),
    )
    parser.add_argument(
        "--stiffness-admission-mode",
        choices=STIFFNESS_ADMISSION_MODES,
        default="strict_all",
        help=(
            "strict_all 要求每个 horizon 严格改善；weighted_window 使用长 horizon "
            "权重更高的滑动窗口总损失，并仅允许噪声量级的单点退化；"
            "relaxed_h135 对单个joint连续梯度仅做H1/H3/H5短验证，"
            "取消接触/粒子数量门、候选比赛、次数与间隔限制；"
            "direct_online 对每次已接受视觉创新立即提交连续梯度，"
            "不等待 shadow horizon、不设提交次数或帧间隔；"
            "paper_trajectory_adam 每个有效视频观测用真实XPBD反事实的"
            "轨迹+4/20历史+图平滑损失更新有界Adam参数。"
        ),
    )
    parser.add_argument(
        "--stiffness-candidate-profile",
        choices=tuple(STIFFNESS_CANDIDATE_PROFILES),
        default="bidirectional_8",
        help=(
            "材料候选集合：bidirectional_8 为当前正负/独立八候选；"
            "cross_signed_12 额外测试 distance/shape 反向组合；"
            "spatial_components_12 将最强的两个图连通区域独立验证；"
            "spatial_components_merge_12 还会合并两个独立通过的区域；"
            "direct_residual_gradient 直接验证连续残差梯度提案，不做候选竞争；"
            "direct_alternating_gradient 按成功提交次数交替更新 distance/shape；"
            "causal_fixed_lag_12 仅保留给旧实验复现；"
            "hierarchical_system_id 同时搜索全局材料量级和局部区域差异；"
            "robust_hierarchical_system_id 再加入 H10、residual-effort 排序、"
            "全局 trust-region/单调分级搜索和有限校准预算。"
        ),
    )
    parser.add_argument(
        "--stiffness-evaluation-output",
        type=Path,
        default=None,
        help=(
            "可选评测输出目录。指定后，对每个已提交刚度运行 H=1/3/5/10 "
            "材料隔离与端到端开放环，并写 events.jsonl/summary.json；"
            "目录必须没有同名结果文件。默认关闭，正常 GUI 不承担多影子开销。"
        ),
    )
    parser.add_argument(
        "--stiffness-evaluation-horizons",
        type=str,
        default="1,3,5,10",
        help="开放环视频帧 horizon，逗号分隔，默认 1,3,5,10。",
    )
    parser.add_argument(
        "--evaluation-headless",
        action="store_true",
        help=(
            "不启动 GUI，按固定视频帧和每帧物理步数跑可重复评测；"
            "必须同时指定刚度评测或 tissue benchmark 输出目录。"
        ),
    )
    parser.add_argument(
        "--evaluation-start-frame",
        type=int,
        default=350,
        help="Headless 评测起始视频帧，默认 350。",
    )
    parser.add_argument(
        "--evaluation-frame-count",
        type=int,
        default=0,
        help="Headless 评测帧数；0 表示从起始帧跑到末尾。",
    )
    parser.add_argument(
        "--evaluation-physics-steps-per-frame",
        type=int,
        default=3,
        help="Headless 评测每个视频帧固定执行的物理步数，默认 3。",
    )
    parser.add_argument(
        "--tissue-benchmark-output",
        type=Path,
        default=None,
        help="可选的独立 SUPER 2D/3D 轨迹与渲染评测输出目录。",
    )
    parser.add_argument(
        "--tissue-benchmark-ground-truth",
        type=Path,
        default=(
            repo_root
            / "data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
        ),
    )
    parser.add_argument(
        "--tissue-benchmark-protocol",
        choices=BENCHMARK_PROTOCOLS,
        default="future_80to20",
    )
    parser.add_argument(
        "--tissue-benchmark-render-scale",
        type=float,
        default=0.5,
        help="评测渲染分辨率比例，默认 0.5（960x540）。",
    )
    parser.add_argument(
        "--tissue-benchmark-reconstruction-test-phase",
        type=int,
        choices=range(8),
        default=0,
        help="7:1 重建留出相位，正式测试为 0；参数调优验证可使用其他相位。",
    )
    parser.add_argument(
        "--tissue-benchmark-track-only",
        action="store_true",
        help="只记录持久点轨迹、不渲染，用于不触碰正式测试相位的快速参数验证。",
    )
    parser.add_argument(
        "--tissue-benchmark-future-test-start-frame",
        type=int,
        default=None,
        help=(
            "覆盖开放环起始帧，仅用于正式 1152 之前的前缀验证；"
            "默认使用冻结真值中的正式 80/20 边界。"
        ),
    )
    parser.add_argument(
        "--psm-tissue-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "启用器械与组织接触（默认开启）；使用 "
            "--no-psm-tissue-contact 可关闭。"
        ),
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="stereo_left,stereo_right",
        help="逗号分隔的离线相机名。默认同时启用左右目。",
    )
    parser.add_argument(
        "--camera-go-zoom",
        type=float,
        default=0.9,
        help=(
            "Go To Camera 视角缩放系数。1.0 表示保持纵横比并完整包含相机画面；"
            "默认 0.9，在完整画面外额外保留约 10%% 边距。"
        ),
    )
    parser.add_argument(
        "--psm-pose-driver",
        choices=tuple(PSM_POSE_DRIVER_PATHS),
        default="depth_then_visual",
        help=(
            "PSM逐帧位姿来源（默认 depth_then_visual）："
            "raw_kinematics=原始bag/q7/标定/hand-eye/LND纯机器人运动学；"
            "raw_p420006=相同原始运动学主干加P420006器械CAD；"
            "raw_p420006_stereo_visual=在raw_p420006上使用10对原始双目"
            "人工标注得到的受约束视觉矫正；"
            "raw_p420006_sam2_online=仅首对人工提示、完整1631对双目"
            "SurgicalSAM2在线视觉矫正；"
            "raw_paper_lnd_sam2_online=重新标注首对、采用论文原始LND "
            "CAD/FK并扩展到完整1631对双目的在线视觉矫正；"
            "raw_paper_lnd_first_stereo_static_q5=只在原始首对上按论文"
            "方法联合左右目优化固定SE(3)与q5零位，后续完全使用机器人学；"
            "raw_paper_lnd_first_stereo_se3_fixed_q5=只在原始首对上联合"
            "左右目优化固定XYZ与三维旋转（可修正倾转和自旋），q5及其余"
            "q7逐时刻严格保留原始值，后续完全使用机器人学；"
            "raw_paper_lnd_sam2_multianchor_closedjaw=左右目8组锚点、"
            "黑杆/银色末端分对象SurgicalSAM2，并严格保留原始q7夹爪"
            "闭合状态的论文LND双目矫正；"
            "raw_paper_lnd_sam2_dense_contact_closedjaw=在上一版基础上"
            "增加500、540、700、747、900、1100双目接触段锚点，"
            "并严格保留原始q7夹爪闭合状态；"
            "raw_paper_lnd_sam2_dense_contact_se3_only=沿用接触段加密"
            "人工双目锚点，对全部1631对仅优化时变XYZ与三维旋转，"
            "q1-q7逐时刻严格保留原始值；"
            "raw_paper_lnd_sam2_dense_contact_unbounded_xyz=使用22对双目"
            "锚点、增强人工尖端约束，对XYZ取消硬边界，仍严格保留原始"
            "q1-q7；"
            "strict=现有LND；"
            "registered_lnd=固定几何注册后的LND先验；paper=纯论文图像跟踪；"
            "paper_robust=此前的论文增强版；hybrid=LND先验加论文图像残差；"
            "corrected=深度优化前的完整三部件时序矫正结果；"
            "depth_then_visual=在 corrected 基础上完成多轮双目深度微调和视觉矫正。"
        ),
    )
    parser.add_argument(
        "--psm-visual-mode",
        choices=("tip", "full"),
        default="full",
        help=(
            "PSM可视范围：full=显示长杆、腕部和夹爪（默认）；"
            "tip=只显示腕部和夹爪。组织接触始终只使用两片夹爪。"
        ),
    )
    parser.add_argument(
        "--tissue-mode",
        choices=("paper_pbd", "paper_soft", "adaptive_soft", "rigid_v9"),
        default="paper_pbd",
        help=(
            "组织运行模式：paper_pbd=论文式 distance/volume/shape-matching "
            "XPBD 基线并支持 RGB residual/在线刚度（默认）；paper_soft=旧 "
            "Neo-Hookean XPBD 固定参数回退；两者均保留 Gaussian、视觉力和 "
            "triangle-skin 接触，paper_soft 不启用在线刚度；"
            "adaptive_soft=旧软体实验；rigid_v9=旧刚体组织回退。"
        ),
    )
    parser.add_argument(
        "--paper-distance-stiffness-initial",
        type=float,
        default=None,
        help=(
            "覆盖 paper_pbd 的初始 distance stiffness；默认沿用场景值 0.20。"
            "用于故意极软/极硬初始化的可恢复性实验。"
        ),
    )
    parser.add_argument(
        "--paper-volume-stiffness-initial",
        type=float,
        default=None,
        help=(
            "覆盖 paper_pbd 的初始 volume stiffness；默认沿用场景值1e10。"
            "用于测试视觉校正后的材料拉回强度。"
        ),
    )
    parser.add_argument(
        "--paper-shape-stiffness-initial",
        type=float,
        default=None,
        help=(
            "覆盖 paper_pbd 的初始 shape stiffness；默认沿用场景值 0.004。"
            "用于故意极软/极硬初始化的可恢复性实验。"
        ),
    )
    parser.add_argument(
        "--calibrated-profile",
        type=Path,
        default=None,
        help=(
            "可选的离线组织标定 profile。默认不加载；新组织先以冻结基础"
            "参数进入 GUI，刚度优化留到后续阶段。"
        ),
    )
    parser.add_argument(
        "--psm-roll-offset-deg",
        type=float,
        default=None,
        help=(
            "手动给 PSM roll 关节增加一个角度偏移，单位 degree。"
            "roll 是器械沿长杆轴线的自旋，用于临时对齐视频中的腕部/夹爪朝向；"
            "raw运动学/P420006系列默认0 deg，其他模式默认-27 deg；只在运行时"
            "生效，不修改输入数据。"
        ),
    )
    parser.add_argument(
        "--psm-camera-translation-mm",
        type=float,
        nargs=3,
        metavar=("RIGHT", "DOWN", "FAR"),
        default=(0.0, 0.0, 0.0),
        help="人工相机坐标平移修正，单位 mm，顺序为右、下、远。",
    )
    parser.add_argument(
        "--psm-world-translation-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help=(
            "固定的 PSM 世界坐标平移修正，单位 mm，顺序为 X、Y、Z；"
            "只从命令行载入，不在 GUI 中实时修改。"
        ),
    )
    return parser.parse_args()


def load_calibrated_tissue_profile(
    environment, profile_path: Path | None
) -> bool:
    """Apply frozen scalar and smooth regional material settings to the GUI env."""
    if profile_path is None:
        return False
    if not profile_path.is_file() or getattr(environment, "super_tissue_mode", "") != "paper_soft":
        return False
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    material = profile["material"]
    E = float(material["young_modulus_pa"]); nu = float(material["poisson_ratio"])
    region_path = profile_path.parent.parent / "stage_f_local_stiffness" / "region_profile.npz"
    region = np.load(region_path)
    weights = np.asarray(region["tet_region_weights"], dtype=np.float32)
    multipliers = profile["regional_stiffness"]
    E_tet = E * (weights @ np.asarray([multipliers["primary_multiplier"], multipliers["transition_multiplier"], multipliers["far_multiplier"]], dtype=np.float32))
    handle = environment.super_tissue_soft_handle
    expected_tets = handle.tet_end - handle.tet_start
    if len(weights) != expected_tets:
        raise ValueError(
            "Calibrated stiffness profile does not match the selected tissue "
            f"asset: profile has {len(weights)} tetrahedra, asset has "
            f"{expected_tets}. Re-optimize it for the new asset first."
        )
    materials = wp.to_torch(environment.sim.model.tet_materials)
    mu = torch.as_tensor(E_tet / (2.0 * (1.0 + nu)), device=materials.device, dtype=materials.dtype)
    lam = torch.as_tensor(E_tet * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), device=materials.device, dtype=materials.dtype)
    materials[handle.tet_start:handle.tet_end, 0] = mu
    materials[handle.tet_start:handle.tet_end, 1] = lam
    materials[handle.tet_start:handle.tet_end, 2] = 0.0
    environment.physics_settings.particle_velocity_damping_per_second = float(material["velocity_damping_per_second"])
    environment.super_tissue_young_modulus_pa = float(E)
    environment.super_tissue_calibrated_profile_path = str(profile_path)
    environment.super_tissue_calibrated_profile = profile
    if hasattr(environment.sim, "_physics_step_cache"):
        delattr(environment.sim, "_physics_step_cache")
    print(f"[example_embodied_super_offline] loaded calibrated tissue profile: {profile_path}; E={E:g} Pa; regional stiffness=ON; visual residual=OFF")
    return True


def configure_initial_paper_stiffness(
    environment,
    *,
    tissue_mode: str,
    distance_stiffness: float | None,
    volume_stiffness: float | None,
    shape_stiffness: float | None,
) -> tuple[float, float, float]:
    """Install a reproducible paper-PBD initialization for every ablation arm."""
    settings = environment.physics_settings
    distance = (
        float(settings.paper_distance_stiffness)
        if distance_stiffness is None
        else float(distance_stiffness)
    )
    shape = (
        float(settings.paper_shape_stiffness)
        if shape_stiffness is None
        else float(shape_stiffness)
    )
    volume = (
        float(settings.paper_volume_stiffness)
        if volume_stiffness is None
        else float(volume_stiffness)
    )
    if distance <= 0.0 or volume <= 0.0 or shape <= 0.0:
        raise ValueError("Initial paper stiffness values must be positive")
    if (
        distance_stiffness is not None
        or volume_stiffness is not None
        or shape_stiffness is not None
    ) and tissue_mode != "paper_pbd":
        raise ValueError("Initial paper stiffness overrides require paper_pbd")
    settings.paper_distance_stiffness = distance
    settings.paper_volume_stiffness = volume
    settings.paper_shape_stiffness = shape
    projector = environment.sim.material_projector
    if projector is not None:
        projector.configure_constraint_model(
            settings.tetrahedral_constraint_model,
            distance,
            volume,
            shape,
            preserve_spatial_stiffness=False,
        )
    print(
        "[example_embodied_super_offline] initial paper stiffness: "
        f"distance={distance:g}, volume={volume:g}, shape={shape:g}, "
        "source="
        f"{'CLI_OVERRIDE' if distance_stiffness is not None or volume_stiffness is not None or shape_stiffness is not None else 'scene_default'}"
    )
    return distance, volume, shape


def configure_flow_depth_local_material_relaxation(
    environment,
    *,
    bindings: FlowDepthParticleRangeBindings,
    mapper: TetrahedralGaussianVisualResidualMapper,
    distance_stiffness: float,
    volume_stiffness: float,
    shape_stiffness: float,
) -> dict[str, int | float]:
    """Make only the directly observed material patch compliant.

    Distance and shape constraints average their per-particle stiffness, so
    the particle field is expanded by exactly one tetrahedron ring.  Volume
    constraints are relaxed only on tetrahedra incident to an observed seed;
    using the expanded mask here would silently turn one ring into two.
    """
    if min(distance_stiffness, volume_stiffness, shape_stiffness) <= 0.0:
        raise ValueError("Local paper stiffness values must be positive")
    projector = environment.sim.material_projector
    if projector is None:
        raise ValueError("Local material relaxation requires paper XPBD")
    tet_indices = mapper.tet_indices.long()
    particle_count = len(mapper.rest_positions)
    seed_mask = torch.zeros(
        particle_count, dtype=torch.bool, device=tet_indices.device
    )
    valid_track_ids = np.flatnonzero(bindings.track_valid)
    seed_ids: list[np.ndarray] = []
    for track_id in valid_track_ids:
        count = int(bindings.support_counts[track_id])
        if count:
            seed_ids.append(bindings.particle_ids[track_id, :count])
    if not seed_ids:
        raise ValueError("Local material relaxation has no valid particles")
    unique_seed_ids = np.unique(np.concatenate(seed_ids)).astype(np.int64)
    if unique_seed_ids[0] < 0 or unique_seed_ids[-1] >= particle_count:
        raise ValueError("Flow-depth binding particle id is out of range")
    seed_mask[
        torch.as_tensor(unique_seed_ids, device=tet_indices.device)
    ] = True
    incident_tet_mask = seed_mask[tet_indices].any(dim=1)
    relaxed_particle_mask = strict_one_tetrahedron_ring_mask(
        seed_mask, tet_indices
    )

    distance_field = wp.to_torch(projector.paper_distance_stiffness)
    volume_field = wp.to_torch(projector.paper_volume_stiffness)
    shape_field = wp.to_torch(projector.paper_shape_stiffness)
    with torch.no_grad():
        distance_field[relaxed_particle_mask] = float(distance_stiffness)
        volume_field[incident_tet_mask] = float(volume_stiffness)
        shape_field[relaxed_particle_mask] = float(shape_stiffness)
    environment.physics_settings.preserve_spatial_paper_stiffness = True
    summary: dict[str, int | float] = {
        "seed_particles": int(seed_mask.sum().item()),
        "relaxed_particles": int(relaxed_particle_mask.sum().item()),
        "particle_count": int(particle_count),
        "relaxed_tets": int(incident_tet_mask.sum().item()),
        "tet_count": int(len(tet_indices)),
        "distance_stiffness": float(distance_stiffness),
        "volume_stiffness": float(volume_stiffness),
        "shape_stiffness": float(shape_stiffness),
    }
    environment.super_flow_depth_local_material_relaxation = summary
    print(
        "[example_embodied_super_offline] flow-depth local material: ON; "
        f"seed_particles={summary['seed_particles']}, "
        f"one_ring_particles={summary['relaxed_particles']}/"
        f"{summary['particle_count']}, "
        f"incident_tets={summary['relaxed_tets']}/{summary['tet_count']}, "
        f"local(distance/volume/shape)={distance_stiffness:g}/"
        f"{volume_stiffness:g}/{shape_stiffness:g}; remote=initial stiffness"
    )
    return summary


def resolve_dataset_path(dataset_arg: Path | None) -> Path:
    default_dataset = repo_root / "data/super/grasp5_offline_demo"
    dataset_from_env = os.environ.get("EMBODIED_GAUSSIANS_SUPER_DATASET")
    dataset_path = dataset_arg
    if dataset_path is None and dataset_from_env:
        dataset_path = Path(dataset_from_env).expanduser()
    if dataset_path is None:
        dataset_path = default_dataset
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    return dataset_path


def build_visual_tissue_residual_mapper(
    environment: EmbodiedGaussiansEnvironment,
    *,
    iterations: int,
    learning_rate_m: float,
    maximum_step_m: float,
    previous_residual_carry: float,
    temporal_weight: float,
    magnitude_weight: float,
) -> TetrahedralGaussianVisualResidualMapper:
    """Construct the online particle residual map from runtime mode-2 bindings."""
    simulator = environment.sim
    model = simulator.gaussian_model
    if model.num_soft_gaussians == 0:
        raise ValueError("Visual residual mode requires soft tissue Gaussians")
    binding_modes = model.soft_gaussian_binding_modes.long()
    if not bool(torch.all(binding_modes == 2).item()):
        modes = torch.unique(binding_modes).detach().cpu().tolist()
        raise ValueError(
            "Visual residual mode currently requires triangular-face-centroid "
            f"Gaussian bindings (mode 2); found modes {modes}"
        )
    settings = VisualTissueResidualMappingSettings(
        iterations=iterations,
        learning_rate_m=learning_rate_m,
        image_scale=0.25,
        photometric_affine_calibration=True,
        temporal_weight=temporal_weight,
        magnitude_weight=magnitude_weight,
        previous_residual_carry=previous_residual_carry,
        maximum_residual_m=maximum_step_m,
        minimum_volume_ratio=0.30,
    )
    mapper = (
        TetrahedralGaussianVisualResidualMapper.from_visual_face_centroid_bindings(
            rest_positions=wp.to_torch(simulator.model.particle_q).detach().clone(),
            tet_indices=wp.to_torch(simulator.model.tet_indices)
            .long()
            .reshape(-1, 4),
            fixed_mask=(
                wp.to_torch(simulator.model.particle_inv_mass).detach() == 0.0
            ),
            soft_gaussian_ids=model.soft_gaussian_ids,
            visual_vertex_particle_indices=(
                model.soft_gaussian_visual_vertex_particle_indices
            ),
            visual_vertex_weights=model.soft_gaussian_visual_vertex_weights,
            visual_vertex_rest_offsets=(
                model.soft_gaussian_visual_vertex_rest_offsets
            ),
            visual_vertex_rest_physical_frames=(
                model.soft_gaussian_visual_vertex_rest_physical_frames
            ),
            rest_visual_face_poses=(
                model.soft_gaussian_rest_visual_face_poses
            ),
            rest_gaussian_quats=model.quats[model.soft_gaussian_ids.long()],
            rest_gaussian_scales=model.scales[model.soft_gaussian_ids.long()],
            settings=settings,
        )
    )
    print(
        "[example_embodied_super_offline] realtime visual residual mapper: "
        f"particles={len(mapper.rest_positions)}, tets={len(mapper.tet_indices)}, "
        f"soft_gaussians={len(mapper.soft_gaussian_ids)}, "
        f"iterations={settings.iterations}, lr={settings.learning_rate_m:g}m, "
        f"image_scale={settings.image_scale:g}, "
        f"max_step={settings.maximum_residual_m * 1e3:.3f}mm, "
        f"previous_carry={settings.previous_residual_carry:g}, "
        f"temporal={settings.temporal_weight:g}, "
        f"magnitude={settings.magnitude_weight:g}"
    )
    return mapper


class SuperPlaybackControls:
    """SUPER 离线回放控制器。

    它做三件事：
    1. 按当前时间戳从 DatasetManager 取视频帧；
    2. 按同一个时间戳从当前 pose source 取 q7，展开成完整 PSM q_full。
    3. 在所选 pose driver 上叠加 GUI 的自旋、夹爪和整体平移增量。

    原 PushT demo 可以直接把 q 塞给 Panda。SUPER 不行，因为 PSM 的完整
    URDF 有 mimic 关节，所以这里多了一步 q7 -> q_full。
    """

    def __init__(
        self,
        environment: EmbodiedGaussiansEnvironment,
        dataset_manager: DatasetManager,
        fps: int,
        monitor_psm_base_q: bool = False,
        monitor_tissue_q: bool = False,
        monitor_interval: float = 0.5,
        psm_roll_offset_deg: float = 0.0,
        psm_camera_translation_mm=(0.0, 0.0, 0.0),
        psm_world_translation_mm=(0.0, 0.0, 0.0),
        visual_force_update_interval: int = 4,
        visual_feedback_mode: str = "residual",
        visual_residual_mapper: TetrahedralGaussianVisualResidualMapper
        | None = None,
        flow_depth_bindings: FlowDepthParticleRangeBindings | None = None,
        flow_depth_observations: FlowDepthObservationSequence | None = None,
        flow_depth_settings: FlowDepthStateUpdateSettings | None = None,
        visual_residual_gain_profile: str = "full_only",
        trajectory_rgb_residual_track_weight: float = 0.02,
        trajectory_rgb_residual_position_gain: float = 1.0,
        trajectory_rgb_residual_velocity_gain: float = 0.15,
        trajectory_rgb_residual_maximum_velocity_m_s: float = 0.015,
        trajectory_appearance_settings: TrajectoryAppearanceSettings
        | None = None,
        stiffness_updater: (
            ResidualDrivenPaperStiffnessUpdater | SimParticleGraphUpdater | None
        ) = None,
        paper_trajectory_stiffness_optimizer: (
            PaperTrajectoryAdamOptimizer | None
        ) = None,
        enable_psm_tissue_contact: bool = True,
        stiffness_evaluation_output: Path | None = None,
        stiffness_evaluation_horizons: tuple[int, ...] = (1, 3, 5, 10),
        tissue_benchmark_recorder: SuperTissueBenchmarkRecorder | None = None,
    ):
        self.playing = False
        self.environment = environment
        self.dataset_manager = dataset_manager
        self.fps = fps
        offline_cameras = self.dataset_manager.offline_cameras
        if offline_cameras is None or len(offline_cameras) == 0:
            raise RuntimeError("SUPER exact-timestamp playback requires a camera")
        self.playback_camera_name, playback_camera = next(
            iter(offline_cameras.items())
        )
        self.playback_timestamps = np.asarray(
            playback_camera.timestamps, dtype=np.float64
        )
        if len(self.playback_timestamps) == 0:
            raise RuntimeError("SUPER playback camera has no timestamps")
        if np.any(np.diff(self.playback_timestamps) <= 0.0):
            raise ValueError("SUPER playback camera timestamps must increase")
        self.current_frame_index = 0
        self.current_timestep = float(self.playback_timestamps[0])
        self.first_state = (
            environment.sim.clone_embodied_gaussian_rollout_state()
        )
        self._tissue_rest_positions = wp.to_torch(
            self.first_state.embodied_state.physics_state.particle_q
        ).clone()
        self._tissue_max_displacement_m = 0.0
        self.monitor_psm_base_q = monitor_psm_base_q
        self.monitor_tissue_q = monitor_tissue_q
        self.monitor_interval = monitor_interval
        self.default_psm_roll_offset_deg = float(psm_roll_offset_deg)
        self.psm_roll_offset_deg = self.default_psm_roll_offset_deg
        self.psm_wrist_rod_offset_deg = 0.0
        self.psm_wrist_rod_offset_enabled = bool(
            environment.super_psm_lnd_kinematics.manual_offset_conventions.get(
                "paper_wrist_pitch"
            )
        )
        self.psm_jaw_offset_deg = 0.0
        self._last_base_monitor_time = -float("inf")
        self._last_tissue_monitor_time = -float("inf")
        self._initial_base_q: np.ndarray | None = None
        self._initial_tissue_q: np.ndarray | None = None
        self._warned_missing_base_id = False
        self._warned_missing_tissue_id = False
        self.default_psm_camera_translation_mm = np.asarray(
            psm_camera_translation_mm, dtype=np.float64
        ).copy()
        if self.default_psm_camera_translation_mm.shape != (3,):
            raise ValueError("PSM camera translation must contain three values")
        self.psm_manual_camera_translation_mm = (
            self.default_psm_camera_translation_mm.copy()
        )
        self.psm_world_translation_mm = np.asarray(
            psm_world_translation_mm, dtype=np.float64
        ).copy()
        if self.psm_world_translation_mm.shape != (3,):
            raise ValueError("PSM world translation must contain three values")
        self.default_psm_tissue_contact_enabled = bool(
            enable_psm_tissue_contact
        )
        self.psm_tissue_collisions_enabled = False
        self.visual_force_update_interval = int(visual_force_update_interval)
        if self.visual_force_update_interval < 1:
            raise ValueError("Visual-force update interval must be at least one")
        self._visual_force_step = self.visual_force_update_interval - 1
        self.visual_feedback_mode = str(visual_feedback_mode)
        if self.visual_feedback_mode not in {
            "trajectory",
            "trajectory_residual",
            "residual",
            "force",
            "off",
        }:
            raise ValueError("Unknown visual feedback mode")
        if (
            self.visual_feedback_mode == "residual"
            and visual_residual_mapper is None
        ):
            raise ValueError("Residual feedback mode requires a residual mapper")
        self.visual_residual_mapper = visual_residual_mapper
        if (flow_depth_bindings is None) != (flow_depth_observations is None):
            raise ValueError(
                "Flow-depth trajectory mode requires both bindings and observations"
            )
        if self.visual_feedback_mode in TRAJECTORY_FEEDBACK_MODES:
            if flow_depth_bindings is None or flow_depth_observations is None:
                raise ValueError(
                    "Trajectory feedback mode requires flow-depth assets"
                )
            if visual_residual_mapper is None:
                raise ValueError(
                    "Trajectory feedback mode requires the physical safety mapper"
                )
            if len(flow_depth_bindings.track_valid) != (
                flow_depth_observations.track_valid.shape[1]
            ):
                raise ValueError(
                    "Flow-depth bindings and observations disagree on track count"
                )
        self.flow_depth_bindings = flow_depth_bindings
        self.flow_depth_observations = flow_depth_observations
        self.flow_depth_settings = (
            flow_depth_settings or FlowDepthStateUpdateSettings()
        )
        self.flow_depth_settings.validate()
        self._trajectory_rgb_particle_indices: torch.Tensor | None = None
        self._trajectory_rgb_particle_weights: torch.Tensor | None = None
        if self.visual_feedback_mode == "trajectory_residual":
            assert flow_depth_bindings is not None
            assert visual_residual_mapper is not None
            residual_device = visual_residual_mapper.rest_positions.device
            self._trajectory_rgb_particle_indices = torch.as_tensor(
                flow_depth_bindings.particle_ids,
                device=residual_device,
                dtype=torch.long,
            )
            self._trajectory_rgb_particle_weights = torch.as_tensor(
                flow_depth_bindings.particle_weights,
                device=residual_device,
                dtype=torch.float32,
            )
        self._flow_depth_source_states: dict[int, np.ndarray] = {}
        self._flow_depth_reference_range_centers: np.ndarray | None = None
        self._last_flow_depth_runtime_frame_index = -1
        self._flow_depth_update_count = 0
        self._defer_flow_depth_update_until_frame_end = False
        self._headless_physics_iteration_limit: int | None = None
        if visual_residual_gain_profile not in VISUAL_RESIDUAL_GAIN_PROFILES:
            raise ValueError("Unknown visual residual gain profile")
        self.visual_residual_gain_profile = visual_residual_gain_profile
        self.visual_residual_gain_candidates = VISUAL_RESIDUAL_GAIN_PROFILES[
            visual_residual_gain_profile
        ]
        self.trajectory_rgb_residual_track_weight = float(
            trajectory_rgb_residual_track_weight
        )
        self.trajectory_rgb_residual_position_gain = float(
            trajectory_rgb_residual_position_gain
        )
        self.trajectory_rgb_residual_velocity_gain = float(
            trajectory_rgb_residual_velocity_gain
        )
        self.trajectory_rgb_residual_maximum_velocity_m_s = float(
            trajectory_rgb_residual_maximum_velocity_m_s
        )
        if self.trajectory_rgb_residual_track_weight < 0.0:
            raise ValueError("Trajectory RGB track weight must be non-negative")
        if not 0.0 < self.trajectory_rgb_residual_position_gain <= 1.0:
            raise ValueError(
                "Trajectory RGB position gain must lie in (0, 1]"
            )
        if self.trajectory_rgb_residual_velocity_gain < 0.0:
            raise ValueError("Trajectory RGB velocity gain must be non-negative")
        if self.trajectory_rgb_residual_maximum_velocity_m_s <= 0.0:
            raise ValueError("Trajectory RGB velocity cap must be positive")
        self.trajectory_appearance_settings = (
            trajectory_appearance_settings or TrajectoryAppearanceSettings(
                iterations=0
            )
        )
        self.trajectory_appearance_settings.validate()
        self._trajectory_appearance_reference_colors_logits = (
            self.first_state.embodied_state.gaussian_state.colors_logits
            .detach()
            .clone()
        )
        self._trajectory_appearance_reference_opacities_logits = (
            self.first_state.embodied_state.gaussian_state.opacities_logits
            .detach()
            .clone()
        )
        if self.trajectory_appearance_settings.iterations > 0:
            self.environment.sim.set_visual_geometry_appearance_reference(
                self._trajectory_appearance_reference_colors_logits,
                self._trajectory_appearance_reference_opacities_logits,
            )
        self._trajectory_appearance_solve_count = 0
        self._trajectory_appearance_accept_count = 0
        self._pending_trajectory_appearance_result = None
        self._pending_trajectory_appearance_frame_index = -1
        self.stiffness_updater = stiffness_updater
        self.paper_trajectory_stiffness_optimizer = (
            paper_trajectory_stiffness_optimizer
        )
        if paper_trajectory_stiffness_optimizer is not None and (
            stiffness_updater is None
            or self.visual_feedback_mode not in TRAJECTORY_FEEDBACK_MODES
            or STIFFNESS_ADMISSION_MODE != "paper_trajectory_adam"
        ):
            raise ValueError(
                "Paper trajectory Adam requires trajectory feedback, an online "
                "material updater, and paper_trajectory_adam admission"
            )
        self._paper_adam_source_rollout_states: dict[int, object] = {}
        self._paper_adam_source_commands: dict[
            int, list[StiffnessToolCommand]
        ] = {}
        paper_causal_maximum_window = 3
        if paper_trajectory_stiffness_optimizer is not None:
            paper_causal_maximum_window = (
                paper_trajectory_stiffness_optimizer.settings
                .causal_maximum_window_size
            )
        self._paper_adam_causal_window: deque[
            PaperAdamCausalObservation
        ] = deque(maxlen=paper_causal_maximum_window)
        self._paper_adam_causal_block_index = 0
        self._paper_adam_track_region_ids: torch.Tensor | None = None
        self._sim_graph_source_velocities: dict[int, torch.Tensor] = {}
        self._sim_graph_transitions: deque[
            SimParticleGraphTransition
        ] = deque(maxlen=5)
        self.stiffness_evaluation_horizons = tuple(
            sorted(set(int(value) for value in stiffness_evaluation_horizons))
        )
        if not self.stiffness_evaluation_horizons or any(
            value <= 0 for value in self.stiffness_evaluation_horizons
        ):
            raise ValueError("Stiffness evaluation horizons must be positive")
        if (
            stiffness_evaluation_output is not None
            and visual_residual_mapper is None
        ):
            raise ValueError(
                "Trajectory evaluation requires a visual residual mapper"
            )
        self._action_phase_classifier = StiffnessActionPhaseClassifier()
        self._current_action_phase = "idle"
        self._active_stiffness_evaluations: list[
            CommittedStiffnessEvaluation
        ] = []
        self._next_stiffness_evaluation_id = 1
        self._stiffness_evaluation_epoch = 0
        evaluation_projector = (
            self.environment.sim.triangle_skin_contact_projector
        )
        self.stiffness_metrics_recorder = (
            StiffnessMetricsRecorder(
                stiffness_evaluation_output,
                metadata={
                    "stiffness_admission_profile": "formal",
                    "dataset": str(self.dataset_manager.path),
                    "horizons": self.stiffness_evaluation_horizons,
                    "protocols": ("material_isolation", "end_to_end"),
                    "visual_feedback_mode": self.visual_feedback_mode,
                    "visual_residual_gain_profile": (
                        self.visual_residual_gain_profile
                    ),
                    "visual_residual_gain_candidates": (
                        self.visual_residual_gain_candidates
                    ),
                    "trajectory_rgb_residual_enabled": (
                        self.visual_feedback_mode == "trajectory_residual"
                    ),
                    "trajectory_rgb_residual_track_weight": (
                        self.trajectory_rgb_residual_track_weight
                    ),
                    "trajectory_rgb_residual_position_gain": (
                        self.trajectory_rgb_residual_position_gain
                    ),
                    "trajectory_rgb_residual_velocity_gain": (
                        self.trajectory_rgb_residual_velocity_gain
                    ),
                    "trajectory_rgb_residual_maximum_velocity_m_s": (
                        self.trajectory_rgb_residual_maximum_velocity_m_s
                    ),
                    "trajectory_rgb_residual_is_material_evidence": False,
                    "trajectory_gaussian_appearance": {
                        "iterations": (
                            self.trajectory_appearance_settings.iterations
                        ),
                        "color_learning_rate": (
                            self.trajectory_appearance_settings
                            .color_learning_rate
                        ),
                        "opacity_learning_rate": (
                            self.trajectory_appearance_settings
                            .opacity_learning_rate
                        ),
                        "maximum_color_logit_offset": (
                            self.trajectory_appearance_settings
                            .maximum_color_logit_offset
                        ),
                        "maximum_opacity_logit_offset": (
                            self.trajectory_appearance_settings
                            .maximum_opacity_logit_offset
                        ),
                        "optimized_properties": (
                            ("rgb", "opacity")
                            if self.trajectory_appearance_settings.optimize_opacity
                            else ("rgb",)
                        ),
                        "position_rotation_scale_owner": "physics_skinning",
                        "geometry_appearance": "fixed_initial_reference",
                        "admission": "next_legal_training_frame_validation",
                        "formal_image_scale": (
                            self.trajectory_appearance_settings.formal_image_scale
                        ),
                        "dssim_weight": (
                            self.trajectory_appearance_settings.dssim_weight
                        ),
                        "future_test_feedback": False,
                    },
                    "visual_residual_later_rgb_ranks_all_safe_gains": bool(
                        self.visual_residual_gain_profile
                        == "cross_frame_ranked_hold"
                    ),
                    "visual_residual_h1_noise_tolerant": bool(
                        self.visual_residual_gain_profile
                        == "cross_frame_ranked_hold"
                    ),
                    "visual_residual_velocity_correction_gain": (
                        VISUAL_RESIDUAL_VELOCITY_CORRECTION_GAIN
                    ),
                    "visual_residual_maximum_velocity_correction_m_s": (
                        VISUAL_RESIDUAL_MAXIMUM_VELOCITY_CORRECTION_M_S
                    ),
                    "visual_residual_deformation_covariance_enabled": bool(
                        visual_residual_mapper.deformation_covariance_enabled
                    ),
                    "visual_residual_photometric_affine_calibration": bool(
                        visual_residual_mapper.settings.photometric_affine_calibration
                    ),
                    "visual_residual_photometric_gain_bounds": (
                        visual_residual_mapper.settings.photometric_gain_minimum,
                        visual_residual_mapper.settings.photometric_gain_maximum,
                    ),
                    "visual_residual_photometric_bias_limit": (
                        visual_residual_mapper.settings.photometric_bias_limit
                    ),
                    "visual_cross_frame_minimum_volume_ratio": (
                        VISUAL_CROSS_FRAME_MINIMUM_VOLUME_RATIO
                    ),
                    "visual_cross_frame_p01_absolute_drop": (
                        VISUAL_CROSS_FRAME_P01_ABSOLUTE_DROP
                    ),
                    "visual_cross_frame_p01_relative_drop": (
                        VISUAL_CROSS_FRAME_P01_RELATIVE_DROP
                    ),
                    "visual_cross_frame_weighted_mean_absolute_drop": (
                        VISUAL_CROSS_FRAME_WEIGHTED_MEAN_ABSOLUTE_DROP
                    ),
                    "visual_cross_frame_weighted_mean_relative_drop": (
                        VISUAL_CROSS_FRAME_WEIGHTED_MEAN_RELATIVE_DROP
                    ),
                    "visual_one_frame_carry_gain_bounds": (
                        VISUAL_ONE_FRAME_CARRY_MINIMUM_GAIN,
                        VISUAL_ONE_FRAME_CARRY_MAXIMUM_GAIN,
                    ),
                    "visual_one_frame_carry_distance_bounds": (
                        VISUAL_ONE_FRAME_CARRY_DISTANCE_MINIMUM,
                        VISUAL_ONE_FRAME_CARRY_DISTANCE_MAXIMUM,
                    ),
                    "visual_one_frame_carry_maximum_correction_m": (
                        VISUAL_ONE_FRAME_CARRY_MAXIMUM_CORRECTION_M
                    ),
                    "visual_one_frame_carry_uses_withheld_rgb": False,
                    "online_stiffness_update": stiffness_updater is not None,
                    "stiffness_maximum_penetration_m": (
                        STIFFNESS_MAXIMUM_PENETRATION_M
                    ),
                    "stiffness_maximum_jaw_speed_rad_s": (
                        STIFFNESS_MAXIMUM_JAW_SPEED_RAD_S
                    ),
                    "stiffness_transition_cooldown_updates": (
                        STIFFNESS_TRANSITION_COOLDOWN_UPDATES
                    ),
                    "stiffness_prediction_absolute_margin": (
                        STIFFNESS_PREDICTION_ABSOLUTE_MARGIN
                    ),
                    "stiffness_prediction_relative_margin": (
                        STIFFNESS_PREDICTION_RELATIVE_MARGIN
                    ),
                    "stiffness_camera_absolute_regression": (
                        STIFFNESS_CAMERA_ABSOLUTE_REGRESSION
                    ),
                    "stiffness_camera_relative_regression": (
                        STIFFNESS_CAMERA_RELATIVE_REGRESSION
                    ),
                    "stiffness_minimum_volume_absolute_drop": (
                        STIFFNESS_MINIMUM_VOLUME_ABSOLUTE_DROP
                    ),
                    "stiffness_minimum_volume_relative_drop": (
                        STIFFNESS_MINIMUM_VOLUME_RELATIVE_DROP
                    ),
                    "stiffness_penetration_tolerance_m": (
                        STIFFNESS_PENETRATION_TOLERANCE_M
                    ),
                    "stiffness_anchor_error_tolerance_m": (
                        STIFFNESS_ANCHOR_ERROR_TOLERANCE_M
                    ),
                    "stiffness_history_relative_tolerance": (
                        STIFFNESS_HISTORY_RELATIVE_TOLERANCE
                    ),
                    "stiffness_history_absolute_tolerance_m": (
                        STIFFNESS_HISTORY_ABSOLUTE_TOLERANCE_M
                    ),
                    "stiffness_local_minimum_volume_ratio": (
                        STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO
                    ),
                    "stiffness_commit_validation_horizon_frames": (
                        STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES
                    ),
                    "stiffness_admission_horizons": (
                        STIFFNESS_ADMISSION_HORIZONS
                    ),
                    "stiffness_candidate_profile": (
                        STIFFNESS_CANDIDATE_PROFILE
                    ),
                    "stiffness_admission_mode": STIFFNESS_ADMISSION_MODE,
                    "paper_trajectory_adam_enabled": bool(
                        paper_trajectory_stiffness_optimizer is not None
                    ),
                    "paper_trajectory_adam_settings": (
                        None
                        if paper_trajectory_stiffness_optimizer is None
                        else {
                            key: value
                            for key, value in vars(
                                paper_trajectory_stiffness_optimizer.settings
                            ).items()
                        }
                    ),
                    "stiffness_residual_effort_required": False,
                    "stiffness_residual_effort_policy": (
                        "candidate_tiebreak_not_commit_veto"
                    ),
                    "stiffness_residual_effort_absolute_margin_m": (
                        STIFFNESS_RESIDUAL_EFFORT_ABSOLUTE_MARGIN_M
                    ),
                    "stiffness_long_horizon_absolute_margin": (
                        STIFFNESS_LONG_HORIZON_ABSOLUTE_MARGIN
                    ),
                    "stiffness_long_horizon_relative_margin": (
                        STIFFNESS_LONG_HORIZON_RELATIVE_MARGIN
                    ),
                    "stiffness_global_trust_region_ratio": (
                        STIFFNESS_GLOBAL_TRUST_REGION_RATIO
                    ),
                    "stiffness_maximum_global_commits_per_family": (
                        STIFFNESS_ROBUST_MAXIMUM_GLOBAL_COMMITS_PER_FAMILY
                    ),
                    "stiffness_robust_maximum_commits": (
                        STIFFNESS_ROBUST_MAXIMUM_COMMITS
                    ),
                    "stiffness_causal_maximum_commits": (
                        STIFFNESS_CAUSAL_MAXIMUM_COMMITS
                    ),
                    "stiffness_causal_maximum_trials": (
                        STIFFNESS_CAUSAL_MAXIMUM_TRIALS
                    ),
                    "stiffness_causal_warmup_end_frame": (
                        STIFFNESS_CAUSAL_WARMUP_END_FRAME
                    ),
                    "stiffness_causal_minimum_trial_interval_frames": (
                        STIFFNESS_CAUSAL_MINIMUM_TRIAL_INTERVAL_FRAMES
                    ),
                    "stiffness_causal_local_maximum_commits_per_phase_family": (
                        STIFFNESS_CAUSAL_LOCAL_MAXIMUM_COMMITS_PER_PHASE_FAMILY
                    ),
                    "stiffness_causal_two_window_confirmation_enabled": False,
                    "stiffness_direct_two_window_strain_observability_enabled": (
                        STIFFNESS_ADMISSION_MODE == "direct_online"
                    ),
                    "stiffness_direct_minimum_observable_windows": (
                        STIFFNESS_DIRECT_MINIMUM_OBSERVABLE_WINDOWS
                    ),
                    "stiffness_causal_minimum_same_family_commit_interval_frames": (
                        STIFFNESS_CAUSAL_MINIMUM_SAME_FAMILY_COMMIT_INTERVAL_FRAMES
                    ),
                    "stiffness_causal_discrete_candidate_search_enabled": False,
                    "stiffness_event_driven_minimum_active_particles": (
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_ACTIVE_PARTICLES
                    ),
                    "stiffness_event_driven_minimum_strain_particles": (
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES
                    ),
                    "stiffness_event_driven_minimum_ema_particles": (
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_EMA_PARTICLES
                    ),
                    "stiffness_event_driven_minimum_contact_count": (
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT
                    ),
                    "stiffness_event_driven_minimum_scope_particles": (
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_SCOPE_PARTICLES
                    ),
                    "stiffness_event_driven_support_iou": (
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_SUPPORT_IOU
                    ),
                    "stiffness_causal_long_horizon_commit_gate_enabled": False,
                    "stiffness_causal_candidate_search_mode": (
                        "single_continuous_proposal"
                    ),
                    "stiffness_causal_local_margin_policy": (
                        "realized_material_support_fraction"
                    ),
                    "stiffness_causal_global_margin_policy": "full_margin",
                    "stiffness_admission_visual_residual_enabled": (
                        STIFFNESS_ADMISSION_MODE == "causal_fixed_lag"
                    ),
                    "stiffness_admission_grip_state_machine_frozen": (
                        STIFFNESS_ADMISSION_MODE != "causal_fixed_lag"
                    ),
                    "stiffness_online_observation_source": (
                        "cotracker_2d_plus_stereo_depth_3d_innovation"
                        if self.visual_feedback_mode in TRAJECTORY_FEEDBACK_MODES
                        else "stereo_2d_rgb_residual_only"
                    ),
                    "stiffness_online_uses_point_cloud": (
                        self.visual_feedback_mode in TRAJECTORY_FEEDBACK_MODES
                    ),
                    "stiffness_online_uses_observed_3d_points": (
                        self.visual_feedback_mode in TRAJECTORY_FEEDBACK_MODES
                    ),
                    "trajectory_stiffness_shadow_rgb_writeback": False,
                    "trajectory_stiffness_commit_adopts_shadow_state": False,
                    "trajectory_stiffness_primary_objective": (
                        "confidence_weighted_alltracker_depth_3d_epe"
                        if self.visual_feedback_mode in TRAJECTORY_FEEDBACK_MODES
                        else None
                    ),
                    "trajectory_stiffness_local_track_minimum": (
                        STIFFNESS_TRAJECTORY_MINIMUM_LOCAL_TRACKS
                    ),
                    "trajectory_stiffness_horizon_margin_m": (
                        STIFFNESS_TRAJECTORY_HORIZON_ABSOLUTE_MARGIN_M
                    ),
                    "trajectory_stiffness_horizon_relative_margin": (
                        STIFFNESS_TRAJECTORY_HORIZON_RELATIVE_MARGIN
                    ),
                    "trajectory_stiffness_aggregate_margin_m": (
                        STIFFNESS_TRAJECTORY_AGGREGATE_ABSOLUTE_MARGIN_M
                    ),
                    "trajectory_stiffness_aggregate_relative_margin": (
                        STIFFNESS_TRAJECTORY_AGGREGATE_RELATIVE_MARGIN
                    ),
                    "trajectory_stiffness_rgb_auxiliary_relative_regression": (
                        STIFFNESS_TRAJECTORY_RGB_RELATIVE_REGRESSION
                    ),
                    "stiffness_evidence_accumulates_while_pending": True,
                    "stiffness_validation_visual_objective": (
                        stiffness_validation_objective_name()
                    ),
                    "visual_residual_persistence_horizons_physics_steps": (
                        VISUAL_RESIDUAL_PERSISTENCE_HORIZONS
                    ),
                    "stiffness_candidate_variants": tuple(
                        {
                            "label": label,
                            "distance_scale": distance_scale,
                            "shape_scale": shape_scale,
                            "scope": scope,
                        }
                        for label, distance_scale, shape_scale, scope in (
                            STIFFNESS_CANDIDATE_VARIANTS
                        )
                    ),
                    "stiffness_global_distance_medians": (
                        STIFFNESS_GLOBAL_DISTANCE_MEDIANS
                        if STIFFNESS_CANDIDATE_PROFILE
                        in {
                            "causal_fixed_lag_12",
                            "hierarchical_system_id",
                            "robust_hierarchical_system_id",
                        }
                        else ()
                    ),
                    "stiffness_global_shape_medians": (
                        STIFFNESS_GLOBAL_SHAPE_MEDIANS
                        if STIFFNESS_CANDIDATE_PROFILE
                        in {
                            "causal_fixed_lag_12",
                            "hierarchical_system_id",
                            "robust_hierarchical_system_id",
                        }
                        else ()
                    ),
                    "stiffness_adopt_validated_rollout": (
                        stiffness_adopt_validated_rollout_enabled()
                    ),
                    "stiffness_maximum_adopted_state_rms_m": (
                        STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M
                    ),
                    "stiffness_maximum_adopted_state_maximum_m": (
                        STIFFNESS_MAXIMUM_ADOPTED_STATE_MAXIMUM_M
                    ),
                    "stiffness_log_learning_rate": (
                        None
                        if stiffness_updater is None
                        else stiffness_updater.settings.log_learning_rate
                    ),
                    "stiffness_graph_smoothing_iterations": (
                        None
                        if stiffness_updater is None
                        else stiffness_updater.settings.graph_smoothing_iterations
                    ),
                    "stiffness_graph_smoothing_blend": (
                        None
                        if stiffness_updater is None
                        else stiffness_updater.settings.graph_smoothing_blend
                    ),
                    "stiffness_strain_signal_weight": (
                        None
                        if stiffness_updater is None
                        else stiffness_updater.settings.strain_signal_weight
                    ),
                    "stiffness_minimum_edge_strain": (
                        None
                        if stiffness_updater is None
                        else stiffness_updater.settings.minimum_edge_strain
                    ),
                    "stiffness_edge_strain_full_scale": (
                        None
                        if stiffness_updater is None
                        else stiffness_updater.settings.edge_strain_full_scale
                    ),
                    "tip_entry_allowance_m": (
                        None
                        if evaluation_projector is None
                        else evaluation_projector.top_barrier_tip_allowance_m
                    ),
                    "grip_maximum_capture_penetration_m": (
                        None
                        if evaluation_projector is None
                        else (
                            evaluation_projector
                            .persistent_grip_maximum_capture_penetration_m
                        )
                    ),
                    "experiment_mode": (
                        "fixed_pbd"
                        if self.visual_feedback_mode == "off"
                        else (
                            "residual_online_stiffness"
                            if stiffness_updater is not None
                            else "residual_only"
                        )
                    ),
                },
            )
            if stiffness_evaluation_output is not None
            else None
        )
        self.tissue_benchmark_recorder = tissue_benchmark_recorder
        self._benchmark_observation_enabled = True
        self._benchmark_observation_gap_frames = 0
        self._last_visual_open_loop_prediction_frame_index = -1
        self._visual_open_loop_prediction_count = 0
        if self.stiffness_metrics_recorder is not None:
            print(
                "[stiffness evaluation] ON; output="
                f"{self.stiffness_metrics_recorder.output_directory}; "
                f"horizons={self.stiffness_evaluation_horizons}; "
                "protocols=material_isolation,end_to_end"
            )
        self._startup_stiffness_settings = (
            stiffness_updater.settings if stiffness_updater is not None else None
        )
        self._stiffness_gui_draft = self._startup_stiffness_settings
        self._stiffness_gui_message = "Pause playback before applying changes."
        self._pending_stiffness_validation: (
            PendingStiffnessValidation | None
        ) = None
        self._pending_visual_residual_validation: (
            PendingVisualResidualValidation | None
        ) = None
        self._last_cross_frame_visual_validation: dict | None = None
        self._stiffness_history: deque[StiffnessHistorySnapshot] = deque(
            maxlen=(
                20
                if paper_trajectory_stiffness_optimizer is not None
                else STIFFNESS_HISTORY_MAXIMUM_SNAPSHOTS
            )
        )
        self._last_stiffness_gate_timestep: float | None = None
        self._last_stiffness_gate_jaw_angle: float | None = None
        self._stiffness_transition_cooldown = 0
        self._last_stiffness_grip_active = False
        self._last_stiffness_contact_count: int | None = None
        self._direct_stiffness_observable_window_count = 0
        self._direct_stiffness_last_observable_frame_index = -1
        self._last_stiffness_validation_metrics: dict | None = None
        self._stiffness_global_distance_locked = False
        self._stiffness_global_shape_locked = False
        self._stiffness_global_distance_direction = 0
        self._stiffness_global_shape_direction = 0
        self._stiffness_global_distance_commits = 0
        self._stiffness_global_shape_commits = 0
        self._stiffness_local_phase_family_commits: dict[
            tuple[str, str], int
        ] = {}
        self._stiffness_local_phase_family_last_commit_frame: dict[
            tuple[str, str], int
        ] = {}
        self._stiffness_local_confirmation_state: dict[
            tuple[str, str], tuple[int, int, str, torch.Tensor]
        ] = {}
        self._stiffness_global_confirmation_key: tuple[str, str] | None = None
        self._stiffness_global_confirmation_count = 0
        self._last_stiffness_commit_frame_index = -10**9
        self._last_stiffness_trial_frame_index = -10**9
        self._stiffness_trial_count = 0
        self._previous_visual_residual: torch.Tensor | None = None
        # Exact result that was actually installed by the multiscale observer.
        # This can differ from the raw solve result when gain=0.5/0.25.  Every
        # downstream consumer (temporal carry, material evidence and history)
        # must use this scaled result or the state estimator and system-ID
        # branches disagree about the displacement that entered the simulator.
        self._last_applied_visual_residual_result = None
        self._visual_residual_solve_count = 0
        self._last_visual_residual_frame_index = -1
        self._last_visual_residual_accepted: bool | None = None
        self._last_trajectory_rgb_residual_frame_index = -1
        self._trajectory_rgb_residual_solve_count = 0
        self._trajectory_appearance_solve_count = 0
        self._trajectory_appearance_accept_count = 0
        self._pending_trajectory_appearance_result = None
        self._pending_trajectory_appearance_frame_index = -1
        self._last_trajectory_observation_frame_index = -1
        self._physics_iteration_count = 0
        self._contact_ui_metrics: dict | None = None
        self._last_contact_ui_refresh_time = -float("inf")
        contact_projector = (
            self.environment.sim.triangle_skin_contact_projector
        )
        self._startup_contact_grip_settings = (
            ContactGripGuiSettings.from_runtime(
                self.environment.physics_settings,
                contact_projector,
            )
            if contact_projector is not None
            else None
        )
        self._contact_grip_gui_draft = (
            self._startup_contact_grip_settings
        )
        self._contact_grip_gui_message = (
            "Pause playback before applying contact/grip changes."
        )

        # mimic_cfg records how 7 active joints derive mimic joints.
        # joint_order records the joint order expected by the simulator.
        self.mimic_cfg = load_mimic_config(
            environment.super_psm_mimic_map_path
        )
        self.joint_order = urdf_actuated_joint_order(
            environment.super_psm_urdf_path
        )
        # PSM is fully driven by offline q, needs re-anchoring after each physics step.
        # _last_q_full stores the most recent q from go_to_timestep,
        # so run_physics can pull PSM back to the correct pose after each step.
        self._last_q_full: torch.Tensor = self.q_full_at(self.current_timestep)
        self._last_state_index = self.state_index_at(self.current_timestep)

    def state_index_at(self, timestep: float) -> int:
        timestamps = self.environment.super_psm_lnd_timestamps
        state_index = int(
            np.searchsorted(timestamps, timestep, side="right") - 1
        )
        return max(0, min(state_index, len(timestamps) - 1))

    def commanded_psm_q7_at(self, timestep: float) -> np.ndarray:
        """Return the recorded q7 command plus explicit manual calibration."""
        state_index = self.state_index_at(timestep)
        q7 = np.asarray(
            self.environment.super_psm_q7_states[state_index],
            dtype=np.float64,
        ).copy()
        return q7 + self.manual_psm_joint_offsets()

    def q_full_at(self, timestep: float) -> torch.Tensor:
        # Articulation, visible LND jaws, and collision bodies all receive the
        # recorded q7 directly.  No artificial opening or resistance limit is
        # inserted into the jaw kinematics.
        q7 = self.commanded_psm_q7_at(timestep)
        q_full = expand_psm_q7_to_urdf_order(
            q7,
            mimic_cfg=self.mimic_cfg,
            joint_order=self.joint_order,
        )
        return torch.from_numpy(q_full).float()

    def manual_psm_joint_offsets(self) -> np.ndarray:
        offsets = np.zeros(7, dtype=np.float64)
        offsets[3] = np.deg2rad(self.psm_roll_offset_deg)
        if self.psm_wrist_rod_offset_enabled:
            offsets[4] = np.deg2rad(self.psm_wrist_rod_offset_deg)
        offsets[6] = np.deg2rad(self.psm_jaw_offset_deg)
        return offsets

    def apply_current_psm_pose(self) -> None:
        self._last_q_full = self.q_full_at(self.current_timestep)
        self.environment.set_robot_q(
            PSM_ARTICULATION_INDEX, self._last_q_full
        )
        self.environment.set_robot_desired_q(
            PSM_ARTICULATION_INDEX, self._last_q_full
        )
        apply_psm_lnd_pose(
            self.environment,
            self._last_state_index,
            joint_offsets=self.manual_psm_joint_offsets(),
            translation_offset=self.manual_psm_translation_world(),
        )

    def manual_psm_translation_world(self) -> np.ndarray:
        fixed_world = self.psm_world_translation_mm / 1000.0
        if self.environment.frames is None:
            return fixed_world.copy()
        X_CW = (
            self.environment.frames.X_CWs_opencv_gpu[0]
            .detach()
            .cpu()
            .numpy()
        )
        translation_camera = self.psm_manual_camera_translation_mm / 1000.0
        return fixed_world + X_CW[:3, :3].T @ translation_camera

    def reset(self):
        self._record_incomplete_stiffness_evaluations("reset")
        self._active_stiffness_evaluations.clear()
        self._stiffness_evaluation_epoch += 1
        self._action_phase_classifier.reset()
        self._current_action_phase = "idle"
        set_psm_tissue_collisions(self.environment, False)
        self.psm_tissue_collisions_enabled = False
        self.psm_manual_camera_translation_mm[:] = (
            self.default_psm_camera_translation_mm
        )
        self.current_frame_index = 0
        self.current_timestep = float(self.playback_timestamps[0])
        self.playing = False
        self._last_state_index = self.state_index_at(self.current_timestep)
        self._visual_force_step = self.visual_force_update_interval - 1
        self.environment.sim.copy_embodied_gaussian_rollout_state(
            self.first_state
        )
        self.environment.sim.clear_soft_visual_force_cache()
        self.environment.sim.clear_visual_tissue_residual_metrics()
        if self.stiffness_updater is not None:
            self.stiffness_updater.reset()
        if self.paper_trajectory_stiffness_optimizer is not None:
            self.paper_trajectory_stiffness_optimizer.reset_from_material()
        self._pending_stiffness_validation = None
        self._pending_visual_residual_validation = None
        self._last_cross_frame_visual_validation = None
        self._stiffness_history.clear()
        self._last_stiffness_gate_timestep = None
        self._last_stiffness_gate_jaw_angle = None
        self._stiffness_transition_cooldown = 0
        self._last_stiffness_grip_active = False
        self._last_stiffness_contact_count = None
        self._direct_stiffness_observable_window_count = 0
        self._direct_stiffness_last_observable_frame_index = -1
        self._last_stiffness_validation_metrics = None
        self._stiffness_global_distance_locked = False
        self._stiffness_global_shape_locked = False
        self._stiffness_global_distance_direction = 0
        self._stiffness_global_shape_direction = 0
        self._stiffness_global_distance_commits = 0
        self._stiffness_global_shape_commits = 0
        self._stiffness_local_phase_family_commits.clear()
        self._stiffness_local_phase_family_last_commit_frame.clear()
        self._stiffness_local_confirmation_state.clear()
        self._stiffness_global_confirmation_key = None
        self._stiffness_global_confirmation_count = 0
        self._last_stiffness_commit_frame_index = -10**9
        self._last_stiffness_trial_frame_index = -10**9
        self._stiffness_trial_count = 0
        self._previous_visual_residual = None
        self._last_applied_visual_residual_result = None
        self._visual_residual_solve_count = 0
        self._last_visual_residual_frame_index = -1
        self._last_visual_residual_accepted = None
        self._last_trajectory_rgb_residual_frame_index = -1
        self._trajectory_rgb_residual_solve_count = 0
        self._trajectory_appearance_solve_count = 0
        self._trajectory_appearance_accept_count = 0
        self._pending_trajectory_appearance_result = None
        self._pending_trajectory_appearance_frame_index = -1
        self._flow_depth_source_states.clear()
        self._paper_adam_source_rollout_states.clear()
        self._paper_adam_source_commands.clear()
        self._paper_adam_causal_window.clear()
        self._paper_adam_causal_block_index = 0
        self._paper_adam_track_region_ids = None
        self._sim_graph_source_velocities.clear()
        self._sim_graph_transitions.clear()
        if (
            self.paper_trajectory_stiffness_optimizer is not None
            and self.paper_trajectory_stiffness_optimizer.settings.sim_global_causal_mode
        ):
            self.environment.physics_settings.particle_velocity_damping_per_second = (
                self.paper_trajectory_stiffness_optimizer.settings
                .velocity_damping_initial_per_second
            )
        self._flow_depth_reference_range_centers = None
        self._last_flow_depth_runtime_frame_index = -1
        self._flow_depth_update_count = 0
        self._last_trajectory_observation_frame_index = -1
        self._benchmark_observation_gap_frames = 0
        self._last_visual_open_loop_prediction_frame_index = -1
        self._visual_open_loop_prediction_count = 0
        self._physics_iteration_count = 0
        self._tissue_max_displacement_m = 0.0
        self.environment.sim.eval_ik()
        self.go_to_frame(0)
        self.environment.sim.sync_kinematic_body_interpolation()
        if self.default_psm_tissue_contact_enabled:
            self.psm_tissue_collisions_enabled = set_psm_tissue_collisions(
                self.environment, True
            )
        self._initial_base_q = self.current_psm_base_q()
        self._initial_tissue_q = self.current_tissue_q()
        self._last_base_monitor_time = -float("inf")
        self._last_tissue_monitor_time = -float("inf")
        self.maybe_print_psm_base_q(force=True)
        self.maybe_print_tissue_q(force=True)
        if self.stiffness_metrics_recorder is not None:
            self.stiffness_metrics_recorder.record(
                event="reset",
                frame_index=self.current_frame_index,
                timestamp_s=self.current_timestep,
                phase=self._current_action_phase,
                details={"epoch": self._stiffness_evaluation_epoch},
                force_summary=True,
            )

    def go_to_frame(self, frame_index: int):
        self.current_frame_index = max(
            0, min(int(frame_index), len(self.playback_timestamps) - 1)
        )
        self.go_to_timestep(
            float(self.playback_timestamps[self.current_frame_index])
        )

    def advance_one_frame(self) -> bool:
        next_frame_index = self.current_frame_index + 1
        if next_frame_index >= len(self.playback_timestamps):
            self.playing = False
            return False
        self.go_to_frame(next_frame_index)
        if self.current_frame_index == len(self.playback_timestamps) - 1:
            self.playing = False
        return True

    def go_to_timestep(self, timestep: float):
        self.current_timestep = timestep
        self._last_state_index = self.state_index_at(timestep)

        # Both the articulation and all LND visual/collision bodies use the
        # same raw q7 angle. apply_psm_pose_driver_joint_offsets() rotates the
        # two jaws about the shared local-z hinge by +/-q7/2.
        self.apply_current_psm_pose()
        self.dataset_manager.update_frames(timestep)

    def psm_base_body_id(self) -> int | None:
        """返回第 0 个环境里的 PSM 基座 body id。

        body_q 是按 body id 索引的。SUPER 场景构建时已经把
        PSM1_psm_base_link 的 id 存到 environment.super_psm_base_body_ids。
        """
        body_ids = getattr(self.environment, "super_psm_base_body_ids", [])
        if not body_ids:
            return None
        return int(body_ids[0])

    def current_psm_base_q(self) -> np.ndarray | None:
        """读取当前 PSM 基座 body_q。

        返回值是 7 维数组：[x, y, z, qx, qy, qz, qw]。前三个数是基座在
        world 坐标系的位置，后四个数是姿态四元数。
        """
        body_id = self.psm_base_body_id()
        if body_id is None:
            return None
        body_q = wp.to_torch(self.environment.sim.state_0.body_q)
        return body_q[body_id].detach().cpu().numpy().copy()

    def tissue_body_id(self) -> int | None:
        """返回第 0 个环境里的 tissue body id。"""
        body_ids = getattr(self.environment, "super_tissue_body_ids", [])
        if body_ids:
            return int(body_ids[0])
        body_id = getattr(self.environment, "super_tissue_body_id", None)
        if body_id is None:
            return None
        return int(body_id)

    def current_tissue_q(self) -> np.ndarray | None:
        """读取当前 tissue 刚体 body_q。

        返回值是 7 维数组：[x, y, z, qx, qy, qz, qw]。
        """
        body_id = self.tissue_body_id()
        if body_id is None:
            return None
        body_q = wp.to_torch(self.environment.sim.state_0.body_q)
        return body_q[body_id].detach().cpu().numpy().copy()

    def format_pose_delta(self, q: np.ndarray, q0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        p = q[:3]
        quat = q[3:]
        p0 = q0[:3]
        quat0 = q0[3:]
        dpos = float(np.linalg.norm(p - p0))

        quat_norm = np.linalg.norm(quat)
        quat0_norm = np.linalg.norm(quat0)
        if quat_norm > 0.0 and quat0_norm > 0.0:
            quat_dot = float(abs(np.dot(quat / quat_norm, quat0 / quat0_norm)))
            dangle_deg = float(np.degrees(2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0))))
        else:
            dangle_deg = float("nan")
        return p, quat, dpos, dangle_deg

    def maybe_print_psm_base_q(self, force: bool = False):
        if not self.monitor_psm_base_q:
            return

        current_time = self.environment.time()
        if not force and current_time - self._last_base_monitor_time < self.monitor_interval:
            return
        self._last_base_monitor_time = current_time

        body_id = self.psm_base_body_id()
        q = self.current_psm_base_q()
        if body_id is None or q is None:
            if not self._warned_missing_base_id:
                print("[PSM base monitor] 没找到 PSM1_psm_base_link 的 body id，无法监控基座。")
                self._warned_missing_base_id = True
            return

        if self._initial_base_q is None:
            self._initial_base_q = q.copy()

        p, quat, dpos, dangle_deg = self.format_pose_delta(q, self._initial_base_q)

        print(
            "[PSM base monitor] "
            f"sim_t={current_time:.4f}s body_id={body_id} "
            f"p=({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}) "
            f"q_xyzw=({quat[0]:+.6f}, {quat[1]:+.6f}, {quat[2]:+.6f}, {quat[3]:+.6f}) "
            f"dpos_from_reset={dpos:.9f}m dangle_from_reset={dangle_deg:.6f}deg"
        )

    def maybe_print_tissue_q(self, force: bool = False):
        if not self.monitor_tissue_q:
            return

        current_time = self.environment.time()
        if not force and current_time - self._last_tissue_monitor_time < self.monitor_interval:
            return
        self._last_tissue_monitor_time = current_time

        body_id = self.tissue_body_id()
        q = self.current_tissue_q()
        if body_id is None or q is None:
            if not self._warned_missing_tissue_id:
                print("[tissue monitor] 没找到 tissue 的 body id，无法监控组织。")
                self._warned_missing_tissue_id = True
            return

        if self._initial_tissue_q is None:
            self._initial_tissue_q = q.copy()

        p, quat, dpos, dangle_deg = self.format_pose_delta(q, self._initial_tissue_q)

        print(
            "[tissue monitor] "
            f"sim_t={current_time:.4f}s body_id={body_id} "
            f"p=({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}) "
            f"q_xyzw=({quat[0]:+.6f}, {quat[1]:+.6f}, {quat[2]:+.6f}, {quat[3]:+.6f}) "
            f"dpos_from_reset={dpos:.9f}m dangle_from_reset={dangle_deg:.6f}deg"
        )

    def _apply_stiffness_gui_draft(self) -> None:
        updater = self.stiffness_updater
        draft = self._stiffness_gui_draft
        if updater is None or draft is None:
            self._stiffness_gui_message = "Online stiffness is not enabled."
            return
        if self.playing:
            self._stiffness_gui_message = "Pause playback, then press Apply again."
            return
        if (
            self._pending_stiffness_validation is not None
            or updater.pending_candidate is not None
        ):
            self._stiffness_gui_message = (
                "A candidate is still pending; keep paused for one physics tick."
            )
            return
        try:
            updater.reconfigure(draft)
        except (RuntimeError, ValueError) as error:
            self._stiffness_gui_message = f"Settings rejected: {error}"
            return
        # History scores and temporal/contact evidence were collected under a
        # different update policy.  Keep verified material and particle state,
        # but do not compare new candidates against stale tuning evidence.
        self._stiffness_history.clear()
        self._last_stiffness_validation_metrics = None
        self._previous_visual_residual = None
        self._stiffness_transition_cooldown = (
            STIFFNESS_TRANSITION_COOLDOWN_UPDATES
        )
        self._stiffness_gui_message = (
            "Applied: verified k preserved/clipped; EMA and history cleared."
        )

    def _apply_contact_grip_gui_draft(self) -> None:
        draft = self._contact_grip_gui_draft
        sim = self.environment.sim
        projector = sim.triangle_skin_contact_projector
        if draft is None or projector is None:
            self._contact_grip_gui_message = "Triangle-skin contact is not configured."
            return
        if self.playing:
            self._contact_grip_gui_message = (
                "Pause playback, then press Apply again."
            )
            return
        if (
            self._pending_stiffness_validation is not None
            or (
                self.stiffness_updater is not None
                and self.stiffness_updater.pending_candidate is not None
            )
        ):
            self._contact_grip_gui_message = (
                "A stiffness candidate is pending; keep paused for one physics tick."
            )
            return
        try:
            draft.validate()
            # Rebuilding is intentional: sample masks, grip transfer tets and
            # support neighborhoods depend on these settings.  Constructing
            # the replacement first also keeps the live projector intact if
            # validation or allocation fails.
            sim.configure_triangle_skin_contacts(
                projector.tool_shape_ids,
                sample_spacing_m=draft.sample_spacing_m,
                spread_layers=draft.contact_spread_layers,
                top_support_lateral_radius_m=(
                    projector.top_support_lateral_radius_m
                ),
                top_support_depth_m=projector.top_support_depth_m,
                top_support_weight_scale=projector.top_support_weight_scale,
                top_pressure_shoulder_lateral_radius_m=(
                    projector.top_pressure_shoulder_lateral_radius_m
                ),
                top_pressure_shoulder_depth_m=(
                    projector.top_pressure_shoulder_depth_m
                ),
                top_pressure_shoulder_upward_scale=(
                    projector.top_pressure_shoulder_upward_scale
                ),
                top_pressure_shoulder_outward_scale=(
                    projector.top_pressure_shoulder_outward_scale
                ),
                top_pressure_shoulder_bias_direction_world=(
                    projector.top_pressure_shoulder_bias_direction_world
                ),
                top_pressure_shoulder_bias_start_m=(
                    projector.top_pressure_shoulder_bias_start_m
                ),
                top_barrier_lateral_tolerance_m=(
                    draft.top_barrier_lateral_tolerance_m
                ),
                top_barrier_contact_patch_radius_m=(
                    draft.top_barrier_contact_patch_radius_m
                ),
                top_barrier_clearance_m=draft.top_barrier_clearance_m,
                top_barrier_shape_ids=projector.top_barrier_shape_ids,
                jaw_contact_shape_ids=projector.jaw_contact_shape_ids,
                jaw_contact_distal_length_m=(
                    draft.jaw_contact_distal_length_m
                ),
                top_barrier_distal_length_m=(
                    draft.top_barrier_distal_length_m
                ),
                top_barrier_tip_allowance_m=(
                    draft.top_barrier_tip_allowance_m
                ),
                jaw_friction_coefficient=draft.jaw_friction_coefficient,
                persistent_grip_enabled=draft.persistent_grip_enabled,
                persistent_grip_minimum_contact_samples_per_jaw=(
                    draft.grip_minimum_contact_samples_per_jaw
                ),
                persistent_grip_nearest_surface_particles=(
                    projector.persistent_grip_nearest_surface_particles
                ),
                persistent_grip_maximum_jaw_patch_separation_m=(
                    draft.grip_maximum_jaw_patch_separation_m
                ),
                persistent_grip_activation_steps=(
                    draft.grip_activation_steps
                ),
                persistent_grip_maximum_capture_penetration_m=(
                    draft.grip_maximum_capture_penetration_m
                ),
                persistent_grip_minimum_capture_volume_ratio=(
                    draft.grip_minimum_capture_volume_ratio
                ),
                persistent_grip_closed_angle_max_rad=(
                    draft.grip_closed_angle_max_rad
                ),
                persistent_grip_release_angle_min_rad=(
                    draft.grip_release_angle_min_rad
                ),
                persistent_grip_release_angle_delta_rad=(
                    draft.grip_release_angle_delta_rad
                ),
                persistent_grip_wide_open_angle_rad=(
                    draft.grip_wide_open_angle_rad
                ),
                persistent_grip_angle_motion_epsilon_rad=(
                    draft.grip_angle_motion_epsilon_rad
                ),
                persistent_grip_compliance_m_per_n=(
                    draft.grip_compliance_m_per_n
                ),
                persistent_grip_relaxation=draft.grip_relaxation,
                persistent_grip_maximum_correction_m=(
                    draft.grip_maximum_correction_m
                ),
                persistent_grip_transfer_layers=(
                    draft.grip_transfer_layers
                ),
                persistent_grip_minimum_volume_ratio=(
                    draft.grip_minimum_volume_ratio
                ),
                persistent_grip_support_radius_m=(
                    draft.grip_support_radius_m
                ),
                persistent_grip_support_generations=(
                    draft.grip_support_generations
                ),
            )
        except (RuntimeError, ValueError) as error:
            self._contact_grip_gui_message = f"Settings rejected: {error}"
            return

        physics = self.environment.physics_settings
        physics.triangle_skin_contact_margin_m = draft.contact_margin_m
        physics.triangle_skin_query_distance_m = draft.query_distance_m
        physics.triangle_skin_ccd_velocity_scale = draft.ccd_velocity_scale
        physics.triangle_skin_contact_relaxation = draft.contact_relaxation
        physics.triangle_skin_contact_max_correction_m = (
            draft.contact_max_correction_m
        )
        physics.triangle_skin_top_barrier_max_correction_m = (
            draft.top_barrier_max_correction_m
        )
        physics.triangle_skin_contact_iterations = draft.contact_iterations
        physics.triangle_skin_post_contact_material_iterations = (
            draft.post_contact_material_iterations
        )
        physics.triangle_skin_final_barrier_max_correction_m = (
            draft.final_barrier_max_correction_m
        )
        physics.triangle_skin_contact_min_volume_ratio = (
            draft.contact_min_volume_ratio
        )
        physics.triangle_skin_contact_substep_stride = (
            draft.contact_substep_stride
        )
        physics.contact_projection_velocity_scale = (
            draft.contact_projection_velocity_scale
        )
        physics.material_projection_velocity_scale = (
            draft.material_projection_velocity_scale
        )
        physics.particle_velocity_damping_per_second = (
            draft.particle_velocity_damping_per_second
        )

        # Contact semantics changed: preserve tissue/material state, but drop
        # the old grip anchors and any learning evidence gathered around them.
        if self.stiffness_updater is not None:
            self.stiffness_updater.invalidate_signal_history()
        self._stiffness_history.clear()
        self._last_stiffness_gate_timestep = None
        self._last_stiffness_gate_jaw_angle = None
        self._last_stiffness_grip_active = False
        self._last_stiffness_contact_count = None
        self._direct_stiffness_observable_window_count = 0
        self._direct_stiffness_last_observable_frame_index = -1
        self._last_stiffness_validation_metrics = None
        self._previous_visual_residual = None
        self._stiffness_transition_cooldown = (
            STIFFNESS_TRANSITION_COOLDOWN_UPDATES
        )
        self._contact_ui_metrics = sim.triangle_skin_contact_metrics()
        self._last_contact_ui_refresh_time = self.environment.time()
        self._contact_grip_gui_message = (
            "Applied: tissue pose kept; contact cache and old grip anchors cleared."
        )

    def _draw_manual_pose_panel(self, imgui) -> None:
        if not imgui.collapsing_header("PSM manual alignment"):
            return
        joint_changed = False
        changed, value = imgui.slider_float(
            "PSM roll offset deg",
            float(self.psm_roll_offset_deg),
            -180.0,
            180.0,
        )
        if changed:
            self.psm_roll_offset_deg = float(value)
            joint_changed = True
        if self.psm_wrist_rod_offset_enabled:
            changed, value = imgui.slider_float(
                "PSM wrist pitch offset deg",
                float(self.psm_wrist_rod_offset_deg),
                -60.0,
                60.0,
            )
            if changed:
                self.psm_wrist_rod_offset_deg = float(value)
                joint_changed = True
        changed, value = imgui.slider_float(
            "PSM jaw offset deg",
            float(self.psm_jaw_offset_deg),
            -30.0,
            30.0,
        )
        if changed:
            self.psm_jaw_offset_deg = float(value)
            joint_changed = True
        if joint_changed:
            self.go_to_timestep(self.current_timestep)

        translation_changed = False
        labels_and_ranges = (
            ("Manual image X mm (+right)", 0, -5.0, 5.0),
            ("Manual image Y mm (+down)", 1, -5.0, 5.0),
            ("Manual camera Z mm (+far)", 2, -30.0, 30.0),
        )
        for label, index, minimum, maximum in labels_and_ranges:
            changed, value = imgui.slider_float(
                label,
                float(self.psm_manual_camera_translation_mm[index]),
                minimum,
                maximum,
            )
            if changed:
                self.psm_manual_camera_translation_mm[index] = value
                translation_changed = True
        if translation_changed:
            self.go_to_timestep(self.current_timestep)
        if imgui.button("Reset manual position"):
            self.psm_manual_camera_translation_mm[:] = (
                self.default_psm_camera_translation_mm
            )
            self.go_to_timestep(self.current_timestep)
        imgui.same_line()
        if imgui.button("Reset all pose offsets"):
            self.psm_roll_offset_deg = self.default_psm_roll_offset_deg
            self.psm_wrist_rod_offset_deg = 0.0
            self.psm_jaw_offset_deg = 0.0
            self.psm_manual_camera_translation_mm[:] = (
                self.default_psm_camera_translation_mm
            )
            self.go_to_timestep(self.current_timestep)
        total_mm = self.manual_psm_translation_world() * 1000.0
        q7 = self.commanded_psm_q7_at(self.current_timestep)
        imgui.text(
            "World XYZ / q7: "
            f"({total_mm[0]:+.3f}, {total_mm[1]:+.3f}, "
            f"{total_mm[2]:+.3f}) mm / {q7[6]:+.3f} rad"
        )

    @staticmethod
    def _stiffness_tooltip(imgui, text: str) -> None:
        if imgui.is_item_hovered():
            imgui.set_tooltip(text)

    def _draw_stiffness_panel(self, imgui) -> None:
        updater = self.stiffness_updater
        if updater is None:
            imgui.text("Online stiffness: OFF")
            return

        settings = updater.settings
        metrics = updater.last_metrics
        if metrics is None:
            imgui.text("Online stiffness: waiting for accepted visual evidence")
            imgui.text(
                "Allowed dist / shape: "
                f"{settings.distance_minimum:.2f}..{settings.distance_maximum:.2f} / "
                f"{settings.shape_minimum:.3f}..{settings.shape_maximum:.3f}"
            )
        else:
            imgui.text(
                "Stiffness: "
                f"{metrics.get('status', 'unknown')} | "
                f"commit {int(metrics.get('update_count', 0))} | "
                f"reject {int(metrics.get('rejected_count', 0))}"
            )
            imgui.text(
                "Distance min / median / max: "
                f"{float(metrics['distance_minimum']):.4f} / "
                f"{float(metrics['distance_median']):.4f} / "
                f"{float(metrics['distance_maximum']):.4f}"
            )
            imgui.text(
                "Shape min / median / max: "
                f"{float(metrics['shape_minimum']):.5f} / "
                f"{float(metrics['shape_median']):.5f} / "
                f"{float(metrics['shape_maximum']):.5f}"
            )
            imgui.text(
                "Evidence active / harden / soften: "
                f"{int(metrics['active_particles'])} / "
                f"{int(metrics['hardening_particles'])} / "
                f"{int(metrics['softening_particles'])}"
            )
            validation = self._last_stiffness_validation_metrics
            if validation is not None and validation.get("rejection_reason"):
                imgui.text(
                    "Latest rejection: "
                    f"{validation['rejection_reason']}"
                )
        recorder = getattr(self, "stiffness_metrics_recorder", None)
        if recorder is not None:
            imgui.text(
                "Open-loop eval: ON | active "
                f"{len(self._active_stiffness_evaluations)} | events "
                f"{recorder.event_count}"
            )

        if imgui.collapsing_header("Online stiffness tuning (pause to apply)"):
            draft = self._stiffness_gui_draft or settings
            changed, value = imgui.slider_float(
                "Distance lower bound",
                float(draft.distance_minimum),
                0.01,
                1.00,
                "%.3f",
            )
            self._stiffness_tooltip(
                imgui, "Lower = softer regions and more local stretch."
            )
            if changed:
                draft = replace(draft, distance_minimum=float(value))
            changed, value = imgui.slider_float(
                "Distance upper bound",
                float(draft.distance_maximum),
                0.10,
                10.00,
                "%.2f",
            )
            self._stiffness_tooltip(
                imgui, "Higher = harder regions; may resist the gripper more."
            )
            if changed:
                draft = replace(draft, distance_maximum=float(value))
            changed, value = imgui.slider_float(
                "Shape lower bound",
                float(draft.shape_minimum),
                0.0001,
                0.030,
                "%.4f",
            )
            self._stiffness_tooltip(
                imgui,
                "Higher raises the softest allowed shape stiffness; "
                "it must not exceed the selected upper bound.",
            )
            if changed:
                draft = replace(draft, shape_minimum=float(value))
            changed, value = imgui.slider_float(
                "Shape upper bound",
                float(draft.shape_maximum),
                0.001,
                0.100,
                "%.4f",
            )
            self._stiffness_tooltip(
                imgui,
                "Above 0.020 is an intentionally wide experimental range; "
                "high values can spread a local impulse farther.",
            )
            if changed:
                draft = replace(draft, shape_maximum=float(value))
            changed, value = imgui.slider_float(
                "Learning rate (log space)",
                float(draft.log_learning_rate),
                0.02,
                0.30,
                "%.3f",
            )
            self._stiffness_tooltip(
                imgui, "Higher = each accepted image changes stiffness faster."
            )
            if changed:
                draft = replace(draft, log_learning_rate=float(value))
            ema_new_weight = 1.0 - draft.signal_ema_decay
            changed, value = imgui.slider_float(
                "EMA new-evidence weight",
                float(ema_new_weight),
                0.05,
                0.60,
                "%.2f",
            )
            self._stiffness_tooltip(
                imgui, "Higher = reacts faster but follows image noise more."
            )
            if changed:
                draft = replace(draft, signal_ema_decay=1.0 - float(value))
            changed, value = imgui.slider_float(
                "Hardening bias",
                float(draft.hardening_bias),
                -0.20,
                0.40,
                "%.2f",
            )
            self._stiffness_tooltip(
                imgui, "Higher = ambiguous evidence is more likely to harden."
            )
            if changed:
                draft = replace(draft, hardening_bias=float(value))
            changed, value = imgui.slider_float(
                "Shape update gain",
                float(draft.shape_update_gain),
                0.50,
                2.50,
                "%.2f",
            )
            self._stiffness_tooltip(
                imgui, "Multiplier applied to the distance log step for shape."
            )
            if changed:
                draft = replace(draft, shape_update_gain=float(value))

            if imgui.collapsing_header("Advanced evidence and smoothing"):
                changed, value = imgui.slider_float(
                    "Maximum log step",
                    float(draft.maximum_log_step),
                    0.03,
                    0.30,
                    "%.3f",
                )
                self._stiffness_tooltip(
                    imgui, "Absolute per-commit safety cap before bounds."
                )
                if changed:
                    draft = replace(draft, maximum_log_step=float(value))
                changed, value = imgui.slider_float(
                    "Minimum residual (mm)",
                    float(draft.minimum_residual_m * 1.0e3),
                    0.005,
                    0.100,
                    "%.3f mm",
                )
                if changed:
                    draft = replace(
                        draft, minimum_residual_m=float(value) * 1.0e-3
                    )
                changed, value = imgui.slider_float(
                    "Residual full scale (mm)",
                    float(draft.residual_full_scale_m * 1.0e3),
                    0.05,
                    0.60,
                    "%.3f mm",
                )
                if changed:
                    draft = replace(
                        draft, residual_full_scale_m=float(value) * 1.0e-3
                    )
                changed, value = imgui.slider_float(
                    "Minimum deformation (mm)",
                    float(draft.minimum_deformation_m * 1.0e3),
                    0.02,
                    0.50,
                    "%.3f mm",
                )
                if changed:
                    draft = replace(
                        draft, minimum_deformation_m=float(value) * 1.0e-3
                    )
                changed, value = imgui.slider_float(
                    "Deformation full scale (mm)",
                    float(draft.deformation_full_scale_m * 1.0e3),
                    0.20,
                    2.00,
                    "%.3f mm",
                )
                if changed:
                    draft = replace(
                        draft, deformation_full_scale_m=float(value) * 1.0e-3
                    )
                changed, value = imgui.slider_int(
                    "Spatial smoothing passes",
                    int(draft.spatial_smoothing_iterations),
                    0,
                    3,
                )
                if changed:
                    draft = replace(
                        draft, spatial_smoothing_iterations=int(value)
                    )
                changed, value = imgui.slider_float(
                    "Neighbor smoothing blend",
                    float(draft.spatial_smoothing_blend),
                    0.0,
                    0.75,
                    "%.2f",
                )
                if changed:
                    draft = replace(
                        draft, spatial_smoothing_blend=float(value)
                    )
                self._stiffness_tooltip(
                    imgui,
                    "Smooths the signed evidence before it proposes a material change.",
                )
                changed, value = imgui.slider_int(
                    "Stiffness graph smoothing passes",
                    int(draft.graph_smoothing_iterations),
                    0,
                    6,
                )
                if changed:
                    draft = replace(
                        draft, graph_smoothing_iterations=int(value)
                    )
                self._stiffness_tooltip(
                    imgui,
                    "Smooths the candidate stiffness field itself over the local mesh graph.",
                )
                changed, value = imgui.slider_float(
                    "Stiffness graph smoothing blend",
                    float(draft.graph_smoothing_blend),
                    0.0,
                    0.90,
                    "%.2f",
                )
                if changed:
                    draft = replace(
                        draft, graph_smoothing_blend=float(value)
                    )
                self._stiffness_tooltip(
                    imgui,
                    "Higher discourages isolated hard/soft spikes; too high can erase real regional boundaries.",
                )
                changed, value = imgui.slider_float(
                    "Rejected EMA keep ratio",
                    float(draft.rejected_ema_decay),
                    0.0,
                    0.90,
                    "%.2f",
                )
                if changed:
                    draft = replace(draft, rejected_ema_decay=float(value))

            self._stiffness_gui_draft = draft
            if imgui.button("Apply stiffness settings"):
                self._apply_stiffness_gui_draft()
            imgui.same_line()
            if imgui.button("Restore startup values"):
                self._stiffness_gui_draft = self._startup_stiffness_settings
                self._stiffness_gui_message = (
                    "Startup values loaded as draft; press Apply."
                )
            imgui.text(self._stiffness_gui_message)

        if metrics is not None and imgui.collapsing_header(
            "Stiffness diagnostics"
        ):
            imgui.text(
                "Candidate / commit / reject: "
                f"{int(metrics.get('candidate_count', 0))} / "
                f"{int(metrics.get('update_count', 0))} / "
                f"{int(metrics.get('rejected_count', 0))}"
            )
            imgui.text(
                "Tet valid / masked; visual masked; u_t excluded: "
                f"{int(metrics['quality_valid_particles'])} / "
                f"{int(metrics['quality_masked_particles'])}; "
                f"{int(metrics.get('supervision_masked_particles', 0))}; "
                f"{int(metrics.get('control_excluded_particles', 0))}"
            )
            imgui.text(
                "EMA active / mean |log step|: "
                f"{int(metrics.get('ema_active_particles', 0))} / "
                f"{float(metrics.get('mean_absolute_log_step', 0.0)):.6f}"
            )
            imgui.text(
                "Edge roughness distance / shape: "
                f"{float(metrics.get('distance_edge_roughness', 0.0)):.6f} / "
                f"{float(metrics.get('shape_edge_roughness', 0.0)):.6f}"
            )
            validation = self._last_stiffness_validation_metrics
            if validation is not None:
                imgui.text(
                    "Prediction verified -> candidate / horizon: "
                    f"{float(validation.get('baseline_visual_loss', 0.0)):.6f} -> "
                    f"{float(validation.get('candidate_visual_loss', 0.0)):.6f} / "
                    f"{int(validation.get('prediction_horizon_frames', 0))} frames"
                )

    def _draw_material_panel(self, imgui, paper_mode: bool) -> None:
        tissue_young_modulus_pa = getattr(
            self.environment, "super_tissue_young_modulus_pa", None
        )
        if tissue_young_modulus_pa is None:
            return
        physics = self.environment.physics_settings
        if getattr(
            self.environment, "super_tissue_constraint_model", None
        ) == "paper":
            baseline = self.environment.super_tissue_paper_stiffness
            imgui.text(
                "Reset material dist / volume / shape: "
                f"{baseline['distance']:g} / {baseline['volume']:g} / "
                f"{baseline['shape']:g}"
            )
        else:
            imgui.text(
                "Tissue material / damping: "
                f"E={tissue_young_modulus_pa / 1e3:.2f} kPa / "
                f"{physics.particle_velocity_damping_per_second:.1f}/s"
            )
        imgui.text(
            "Tissue max displacement from Reset: "
            f"{self._tissue_max_displacement_m * 1e3:.2f} mm"
        )
        if paper_mode:
            self._draw_stiffness_panel(imgui)

        if not imgui.collapsing_header("Material and contact setup details"):
            return
        imgui.text(
            "Dynamics gravity / damping: "
            f"{getattr(self.environment, 'super_tissue_gravity_m_s2', 0.0):.1f} m/s^2 / "
            f"{physics.particle_velocity_damping_per_second:.1f}/s"
        )
        imgui.text(
            "Contact stride / iterations / post-material passes: "
            f"{physics.triangle_skin_contact_substep_stride} / "
            f"{physics.triangle_skin_contact_iterations} / "
            f"{physics.triangle_skin_post_contact_material_iterations}"
        )
        imgui.text(
            "Contact correction / final barrier: "
            f"{physics.triangle_skin_contact_max_correction_m * 1e3:.3f} / "
            f"{physics.triangle_skin_final_barrier_max_correction_m * 1e3:.3f} mm"
        )
        imgui.text(
            "Velocity transfer contact / material: "
            f"{physics.contact_projection_velocity_scale:.2f} / "
            f"{physics.material_projection_velocity_scale:.2f}"
        )
        metrics = self._contact_ui_metrics
        if metrics is not None:
            imgui.text(
                "Jaw distal / tip entry / support radius x depth: "
                f"{metrics.get('jaw_contact_distal_length_m', 0.0) * 1e3:.1f} / "
                f"{metrics.get('top_barrier_tip_allowance_m', 0.0) * 1e3:.1f} / "
                f"{metrics.get('top_support_lateral_radius_m', 0.0) * 1e3:.1f} x "
                f"{metrics.get('top_support_depth_m', 0.0) * 1e3:.1f} mm"
            )
        gap = getattr(
            self.environment, "super_psm_tissue_kinematic_guard_gap_m", None
        )
        if gap is not None:
            imgui.text(
                "Kinematic guard gap / last retraction: "
                f"{gap * 1e3:+.3f} / "
                f"{getattr(self.environment, 'super_psm_tissue_kinematic_guard_offset_m', 0.0) * 1e3:.3f} mm"
            )

    def _draw_contact_grip_tuning_panel(self, imgui) -> None:
        draft = self._contact_grip_gui_draft
        if draft is None:
            return
        if not imgui.collapsing_header(
            "Tip entry and grip (pause to apply)"
        ):
            return

        # The hidden top-barrier distal length follows a larger tip allowance
        # up to the jaw-contact distal length.  This gives the visible control
        # useful range while preserving tip < barrier <= jaw geometry.
        maximum_tip_entry_mm = max(
            0.0,
            draft.jaw_contact_distal_length_m * 1.0e3 - 0.01,
        )
        changed, value = imgui.slider_float(
            "Tip entry allowance (mm)",
            float(draft.top_barrier_tip_allowance_m * 1.0e3),
            0.0,
            maximum_tip_entry_mm,
            "%.2f mm",
        )
        self._stiffness_tooltip(
            imgui,
            "Higher lets the distal tip pass farther below the top barrier; "
            "the hidden barrier length expands safely up to the jaw length.",
        )
        if changed:
            tip_allowance_m = float(value) * 1.0e-3
            required_barrier_distal_m = min(
                draft.jaw_contact_distal_length_m,
                tip_allowance_m + 0.00001,
            )
            draft = replace(
                draft,
                top_barrier_tip_allowance_m=tip_allowance_m,
                top_barrier_distal_length_m=max(
                    draft.top_barrier_distal_length_m,
                    required_barrier_distal_m,
                ),
            )

        changed, value = imgui.slider_float(
            "Capture max penetration (mm)",
            float(draft.grip_maximum_capture_penetration_m * 1.0e3),
            0.10,
            10.00,
            "%.2f mm",
        )
        self._stiffness_tooltip(
            imgui,
            "Higher permits deeper overlap when persistent grip is captured.",
        )
        if changed:
            draft = replace(
                draft,
                grip_maximum_capture_penetration_m=(
                    float(value) * 1.0e-3
                ),
            )

        changed, value = imgui.slider_float(
            "Grip support radius (mm)",
            float(draft.grip_support_radius_m * 1.0e3),
            0.0,
            30.0,
            "%.2f mm",
        )
        self._stiffness_tooltip(
            imgui,
            "Higher carries a wider surface neighborhood with the four "
            "direct jaw anchors.",
        )
        if changed:
            draft = replace(
                draft,
                grip_support_radius_m=float(value) * 1.0e-3,
            )

        if imgui.collapsing_header("Particle jump stabilization"):
            changed, value = imgui.slider_float(
                "Grip correction cap (mm/substep)",
                float(draft.grip_maximum_correction_m * 1.0e3),
                0.005,
                2.000,
                "%.3f mm",
            )
            self._stiffness_tooltip(
                imgui,
                "Lower is the first control to try for isolated jumping "
                "particles; higher makes anchors chase the jaw faster.",
            )
            if changed:
                draft = replace(
                    draft,
                    grip_maximum_correction_m=(
                        float(value) * 1.0e-3
                    ),
                )

            changed, value = imgui.slider_float(
                "Grip compliance (m/N)",
                float(draft.grip_compliance_m_per_n),
                0.0,
                2.0,
                "%.3f",
            )
            self._stiffness_tooltip(
                imgui,
                "Higher makes the jaw attachment softer and reduces "
                "snapping; too high can make the grasp lag or slip.",
            )
            if changed:
                draft = replace(
                    draft,
                    grip_compliance_m_per_n=float(value),
                )

            changed, value = imgui.slider_float(
                "Material/grip velocity transfer",
                float(draft.material_projection_velocity_scale),
                0.0,
                1.0,
                "%.3f",
            )
            self._stiffness_tooltip(
                imgui,
                "Lower converts less material and persistent-grip position "
                "correction into next-step velocity.",
            )
            if changed:
                draft = replace(
                    draft,
                    material_projection_velocity_scale=float(value),
                )

            changed, value = imgui.slider_float(
                "Particle velocity damping (/s)",
                float(draft.particle_velocity_damping_per_second),
                0.0,
                100.0,
                "%.1f /s",
            )
            self._stiffness_tooltip(
                imgui,
                "Higher suppresses velocity oscillation globally; too high "
                "looks viscous and slows recovery.",
            )
            if changed:
                draft = replace(
                    draft,
                    particle_velocity_damping_per_second=float(value),
                )

        self._contact_grip_gui_draft = draft
        if imgui.button("Apply entry/grip settings"):
            self._apply_contact_grip_gui_draft()
        imgui.same_line()
        if imgui.button("Restore startup entry/grip"):
            startup = self._startup_contact_grip_settings
            if startup is not None:
                self._contact_grip_gui_draft = replace(
                    draft,
                    top_barrier_tip_allowance_m=(
                        startup.top_barrier_tip_allowance_m
                    ),
                    top_barrier_distal_length_m=(
                        startup.top_barrier_distal_length_m
                    ),
                    grip_maximum_capture_penetration_m=(
                        startup.grip_maximum_capture_penetration_m
                    ),
                    grip_support_radius_m=(
                        startup.grip_support_radius_m
                    ),
                    grip_maximum_correction_m=(
                        startup.grip_maximum_correction_m
                    ),
                    grip_compliance_m_per_n=(
                        startup.grip_compliance_m_per_n
                    ),
                    material_projection_velocity_scale=(
                        startup.material_projection_velocity_scale
                    ),
                    particle_velocity_damping_per_second=(
                        startup.particle_velocity_damping_per_second
                    ),
                )
            self._contact_grip_gui_message = (
                "Startup entry/grip values loaded as draft; press Apply."
            )
        imgui.text(self._contact_grip_gui_message)
    def _draw_contact_runtime_panel(self, imgui) -> None:
        self._draw_contact_grip_tuning_panel(imgui)
        metrics = self._contact_ui_metrics
        if metrics is None:
            imgui.text("Contact runtime: waiting for metrics")
            return
        jaw_shape_ids = metrics["jaw_contact_shape_ids"]
        contact_counts = metrics["contact_count_by_shape"]
        jaw_counts = tuple(contact_counts[index] for index in jaw_shape_ids)
        required = metrics["persistent_grip_minimum_contact_samples_per_jaw"]
        patch_separation_m = metrics["persistent_grip_jaw_patch_separation_m"]
        maximum_separation_m = metrics[
            "persistent_grip_maximum_jaw_patch_separation_m"
        ]
        if metrics["persistent_grip_active"]:
            verdict = "GRASPED"
        elif not metrics["persistent_grip_capture_allowed"]:
            verdict = "WAITING FOR JAW CLOSURE"
        elif min(jaw_counts, default=0) < required:
            verdict = "WAITING FOR TWO-SIDED CONTACT"
        elif patch_separation_m > maximum_separation_m:
            verdict = "CONTACT PATCHES TOO FAR APART"
        else:
            verdict = "CONTACT SUSTAINING"
        imgui.text(
            "Grip: "
            f"{verdict} | state={metrics['persistent_grip_q7_motion_state']} | "
            f"attached={metrics['persistent_grip_particle_count']}"
        )
        imgui.text(
            "Contact left/right | gap | penetration: "
            f"{jaw_counts} | {metrics['minimum_signed_distance_m'] * 1e3:+.3f} | "
            f"{metrics['maximum_penetration_m'] * 1e3:.3f} mm"
        )
        unsafe = int(metrics.get("material_safety_unsafe_tetrahedra", 0))
        local_unsafe = int(metrics.get("contact_local_unsafe_tetrahedra", 0))
        if unsafe or local_unsafe:
            imgui.text(
                "SAFETY WARNING global/local unsafe tets: "
                f"{unsafe} / {local_unsafe}"
            )

        if not imgui.collapsing_header("Contact diagnostics"):
            return
        imgui.text(
            "q7 / capture allowed / active: "
            f"{metrics['persistent_grip_q7_angle_rad']:+.3f} rad / "
            f"{bool(metrics['persistent_grip_capture_allowed'])} / "
            f"{bool(metrics['persistent_grip_active'])}"
        )
        imgui.text(
            "Selected / sustain / patch separation: "
            f"{metrics['persistent_grip_selected_particle_count']}/4 / "
            f"{metrics['persistent_grip_activation_counter']}/"
            f"{metrics['persistent_grip_activation_steps']} / "
            f"{patch_separation_m * 1e3:.2f} mm"
        )
        imgui.text(
            "Contact candidates / barrier contacts / spread layers: "
            f"{metrics['contact_count']} / "
            f"{metrics['top_barrier_contact_count']} / "
            f"{metrics['contact_spread_layers']}"
        )
        jaw_safe = metrics.get("persistent_grip_jaw_safe_scales", (1.0, 1.0))
        imgui.text(
            "Material / direct / jaw A-B safe scales: "
            f"{metrics.get('material_safety_step_scale', 1.0):.4f} / "
            f"{metrics.get('persistent_grip_direct_safe_scale', 1.0):.4f} / "
            f"{jaw_safe[0]:.4f}-{jaw_safe[1]:.4f}"
        )

    def _draw_visual_feedback_panel(self, imgui) -> None:
        imgui.text(
            f"Visual feedback {self.visual_feedback_mode}: "
            f"{'ACTIVE' if self.playing else 'PAUSED'}"
        )
        if self.visual_feedback_mode in {"residual", "trajectory_residual"}:
            metrics = self.environment.sim.last_visual_tissue_residual_metrics
            if metrics is None:
                imgui.text("Residual: waiting for first solve")
                return
            imgui.text(
                "Residual accepted / loss reduction / max: "
                f"{'YES' if metrics['accepted'] else 'NO'} / "
                f"{float(metrics['visual_loss_reduction_fraction']) * 100.0:+.2f}% / "
                f"{float(metrics['maximum_residual_m']) * 1e3:.3f} mm"
            )
            if not metrics["accepted"]:
                imgui.text(f"Residual rejection: {metrics['rejection_reason']}")
            if not imgui.collapsing_header("Visual residual diagnostics"):
                return
            imgui.text(
                "Solve time / count / RMS: "
                f"{float(metrics.get('solve_elapsed_s', 0.0)) * 1e3:.1f} ms / "
                f"{int(metrics.get('solve_count', 0))} / "
                f"{float(metrics['rms_residual_m']) * 1e3:.3f} mm"
            )
            imgui.text(
                "Visual loss initial -> exact final: "
                f"{float(metrics['initial_visual_loss']):.6f} -> "
                f"{float(metrics.get('exact_final_visual_loss', metrics['final_visual_loss'])):.6f}"
            )
            persistence_improvements = metrics.get(
                "persistence_visual_improvements"
            )
            if persistence_improvements is not None:
                imgui.text(
                    "No-vision physics hold H1/H3/H5 improvements: "
                    + " / ".join(
                        "missing"
                        if persistence_improvements.get(horizon) is None
                        else f"{float(persistence_improvements[horizon]):+.6f}"
                        for horizon in VISUAL_RESIDUAL_PERSISTENCE_HORIZONS
                    )
                )
            for camera_index, values in enumerate(
                zip(
                    metrics.get("initial_camera_visual_losses", ()),
                    metrics.get("exact_final_camera_visual_losses", ()),
                    metrics.get("camera_active_pixel_counts", ()),
                    metrics.get("camera_mask_coverage_fractions", ()),
                )
            ):
                initial, final, active_pixels, coverage = values
                imgui.text(
                    f"Camera {camera_index}: {float(initial):.6f} -> "
                    f"{float(final):.6f}; px={int(active_pixels)}; "
                    f"mask={float(coverage) * 100.0:.1f}%"
                )
            imgui.text(
                "Minimum J initial -> final / safety frozen / grip excluded / backtracks: "
                f"{float(metrics['initial_minimum_volume_ratio']):.3f} -> "
                f"{float(metrics['minimum_volume_ratio']):.3f} / "
                f"{int(metrics['locally_frozen_particles'])} / "
                f"{int(metrics.get('dynamically_excluded_particles', 0))} / "
                f"{int(metrics['backtrack_count'])}"
            )
        elif self.visual_feedback_mode == "force":
            metrics = self.environment.sim.last_soft_visual_force_metrics
            if metrics is None:
                imgui.text("Visual force: waiting for first solve")
                return
            imgui.text(
                "Visual force applied / requested / active particles: "
                f"{metrics['applied_force_budget_n']:.5f} / "
                f"{metrics['force_budget_before_n']:.5f} N / "
                f"{metrics['active_particles']}"
            )
            if imgui.collapsing_header("Visual force diagnostics"):
                imgui.text(
                    "Target max / acceleration max: "
                    f"{metrics['target_delta_max_m'] * 1e3:.3f} mm / "
                    f"{metrics.get('maximum_particle_acceleration_m_s2', 0.0):.3f} m/s^2"
                )

    def draw(self):
        # marsoom/pyglet 需要真实显示环境；放到 draw 里导入，可以让
        # `python examples/example_embodied_super_offline.py --help` 这类非 GUI
        # 操作在无显示环境里也能正常运行。
        from marsoom import imgui

        imgui.set_next_window_size_constraints((560, 360), (900, 700))
        imgui.begin("SUPER Playback")
        imgui.text(
            f"Frame: {self.current_frame_index + 1}/{len(self.playback_timestamps)}"
        )
        imgui.text(
            f"{self.playback_camera_name} timestamp: {self.current_timestep:.6f}s"
        )
        imgui.text(f"Playing: {self.playing}")
        _, self.fps = imgui.slider_int("Playback FPS", self.fps, 1, 120)
        self._draw_manual_pose_panel(imgui)
        paper_mode = getattr(
            self.environment, "super_tissue_mode", None
        ) in {"paper_pbd", "paper_soft"}
        collision_changed, collision_enabled = imgui.checkbox(
            "PSM-tissue collision",
            self.psm_tissue_collisions_enabled,
        )
        if collision_changed:
            self.psm_tissue_collisions_enabled = set_psm_tissue_collisions(
                self.environment, collision_enabled
            )
        current_sim_time = self.environment.time()
        if (
            current_sim_time - self._last_contact_ui_refresh_time >= 0.25
        ):
            if (
                self.environment.sim.triangle_skin_contact_projector
                is not None
            ):
                self._contact_ui_metrics = (
                    self.environment.sim.triangle_skin_contact_metrics()
                )
            else:
                self._contact_ui_metrics = None
            soft_handle = getattr(
                self.environment, "super_tissue_soft_handle", None
            )
            if soft_handle is not None:
                current_positions = wp.to_torch(
                    self.environment.sim.state_0.particle_q
                )[soft_handle.particle_start : soft_handle.particle_end]
                rest_positions = self._tissue_rest_positions[
                    soft_handle.particle_start : soft_handle.particle_end
                ]
                self._tissue_max_displacement_m = float(
                    torch.linalg.vector_norm(
                        current_positions - rest_positions, dim=1
                    )
                    .max()
                    .item()
                )
            self._last_contact_ui_refresh_time = current_sim_time
        imgui.text(
            "Contact status: "
            f"{'ON' if self.psm_tissue_collisions_enabled else 'OFF'}"
        )
        self._draw_material_panel(imgui, paper_mode)
        self._draw_contact_runtime_panel(imgui)
        self._draw_visual_feedback_panel(imgui)
        if imgui.button("Play"):
            if self.current_frame_index >= len(self.playback_timestamps) - 1:
                self.go_to_frame(0)
            self._visual_force_step = self.visual_force_update_interval - 1
            self._previous_visual_residual = None
            self.playing = True
        imgui.same_line()
        if imgui.button("Pause"):
            self.playing = False
        imgui.same_line()
        if imgui.button("Reset"):
            self.reset()
        imgui.end()

    def _current_tool_height_m(self) -> float | None:
        body_ids_by_environment = getattr(
            self.environment, "super_psm_lnd_body_ids", None
        )
        if not body_ids_by_environment:
            return None
        body_ids = tuple(int(value) for value in body_ids_by_environment[0])
        if not body_ids:
            return None
        body_q = wp.to_torch(self.environment.sim.state_0.body_q)
        selected = body_q[
            torch.as_tensor(body_ids, device=body_q.device, dtype=torch.long)
        ]
        return float(selected[:, 2].mean().item())

    def _update_action_phase(self, contact_metrics: dict | None = None) -> str:
        if contact_metrics is None:
            contact_metrics = (
                self.environment.sim.triangle_skin_contact_metrics()
            )
        self._current_action_phase = self._action_phase_classifier.classify(
            contact_metrics,
            self._current_tool_height_m(),
        )
        return self._current_action_phase

    @staticmethod
    def _selected_contact_metrics(
        contact_metrics: dict | None,
    ) -> dict[str, object]:
        if contact_metrics is None:
            return {}
        keys = (
            "contact_count",
            "top_barrier_contact_count",
            "minimum_signed_distance_m",
            "maximum_penetration_m",
            "material_safety_step_scale",
            "material_safety_unsafe_tetrahedra",
            "contact_local_unsafe_tetrahedra",
            "persistent_grip_active",
            "persistent_grip_activation_counter",
            "persistent_grip_particle_count",
            "persistent_grip_direct_particle_count",
            "persistent_grip_support_particle_count",
            "persistent_grip_anchor_error_rms_m",
            "persistent_grip_anchor_error_maximum_m",
            "persistent_grip_q7_angle_rad",
            "persistent_grip_q7_motion_state",
            "persistent_grip_inversion_safe_scale",
            "persistent_grip_direct_safe_scale",
        )
        return {
            key: contact_metrics[key]
            for key in keys
            if key in contact_metrics
        }

    def _current_physical_evaluation_metrics(
        self,
        contact_metrics: dict | None = None,
        visual_metrics: dict | None = None,
    ) -> dict[str, object]:
        physical: dict[str, object] = {}
        mapper = self.visual_residual_mapper
        if mapper is not None:
            positions = wp.to_torch(
                self.environment.sim.state_0.particle_q
            )
            physical.update(mapper.physical_quality_metrics(positions))
        physical.update(self._selected_contact_metrics(contact_metrics))
        if visual_metrics is not None:
            for key in (
                "dynamically_excluded_particles",
                "locally_frozen_particles",
                "local_volume_projection_passes",
                "backtrack_count",
                "initial_minimum_volume_ratio",
                "minimum_volume_ratio",
                "newly_inverted_tetrahedra",
            ):
                if key in visual_metrics:
                    physical[f"residual_{key}"] = visual_metrics[key]
        return physical

    def _current_material_evaluation_metrics(
        self,
        stiffness_metrics: dict | None = None,
    ) -> dict[str, object]:
        updater = self.stiffness_updater
        if updater is None:
            return {}
        metrics = dict(stiffness_metrics or updater.last_metrics or {})
        settings = updater.settings
        metrics.update(
            configured_distance_minimum=settings.distance_minimum,
            configured_distance_maximum=settings.distance_maximum,
            configured_shape_minimum=settings.shape_minimum,
            configured_shape_maximum=settings.shape_maximum,
            configured_log_learning_rate=settings.log_learning_rate,
            configured_maximum_log_step=settings.maximum_log_step,
            configured_ema_new_weight=1.0 - settings.signal_ema_decay,
            configured_spatial_smoothing_iterations=(
                settings.spatial_smoothing_iterations
            ),
            configured_spatial_smoothing_blend=(
                settings.spatial_smoothing_blend
            ),
            configured_graph_smoothing_iterations=(
                settings.graph_smoothing_iterations
            ),
            configured_graph_smoothing_blend=(
                settings.graph_smoothing_blend
            ),
        )
        return metrics

    def _record_visual_stiffness_metrics(
        self,
        *,
        result,
        visual_metrics: dict,
        stiffness_metrics: dict | None,
        contact_metrics: dict | None,
        gate_paused: bool,
        gate_reason: str,
    ) -> None:
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        image = {
            "accepted": bool(visual_metrics["accepted"]),
            "rejection_reason": visual_metrics["rejection_reason"],
            "left_right_loss_before": visual_metrics[
                "initial_camera_visual_losses"
            ],
            "left_right_loss_after": visual_metrics[
                "exact_final_camera_visual_losses"
            ],
            "mean_loss_before": visual_metrics["initial_visual_loss"],
            "mean_loss_after": visual_metrics["exact_final_visual_loss"],
            "loss_reduction_fraction": visual_metrics[
                "visual_loss_reduction_fraction"
            ],
            "residual_maximum_m": result.maximum_residual_m,
            "residual_rms_m": result.rms_residual_m,
            "active_pixel_counts": visual_metrics[
                "camera_active_pixel_counts"
            ],
            "mask_coverage_fractions": visual_metrics[
                "camera_mask_coverage_fractions"
            ],
            "solve_elapsed_s": visual_metrics.get("solve_elapsed_s", 0.0),
            "persistence_gate_evaluated": visual_metrics.get(
                "persistence_gate_evaluated", False
            ),
            "persistence_gate_passed": visual_metrics.get(
                "persistence_gate_passed", False
            ),
            "persistence_horizons_physics_steps": visual_metrics.get(
                "persistence_horizons_physics_steps", ()
            ),
            "persistence_baseline_visual_losses": visual_metrics.get(
                "persistence_baseline_visual_losses", {}
            ),
            "persistence_candidate_visual_losses": visual_metrics.get(
                "persistence_candidate_visual_losses", {}
            ),
            "persistence_visual_improvements": visual_metrics.get(
                "persistence_visual_improvements", {}
            ),
            "velocity_correction_enabled": visual_metrics.get(
                "velocity_correction_enabled", False
            ),
            "velocity_correction_gain": visual_metrics.get(
                "velocity_correction_gain", 0.0
            ),
            "velocity_observation_dt_s": visual_metrics.get(
                "velocity_observation_dt_s"
            ),
            "velocity_correction_maximum_m_s": visual_metrics.get(
                "velocity_correction_maximum_m_s", 0.0
            ),
            "velocity_correction_rms_m_s": visual_metrics.get(
                "velocity_correction_rms_m_s", 0.0
            ),
            "velocity_updated_particles": visual_metrics.get(
                "velocity_updated_particles", 0
            ),
            "stage": visual_metrics.get("stage", "visual_residual"),
            "material_evidence_used": visual_metrics.get(
                "material_evidence_used", True
            ),
            "trajectory_hold_weight": visual_metrics.get(
                "trajectory_hold_weight", 0.0
            ),
            "trajectory_hold_track_count": visual_metrics.get(
                "trajectory_hold_track_count", 0
            ),
            "trajectory_hold_rms_m": visual_metrics.get(
                "trajectory_hold_rms_m", 0.0
            ),
            "trajectory_hold_maximum_m": visual_metrics.get(
                "trajectory_hold_maximum_m", 0.0
            ),
        }
        recorder.record(
            event="visual_update",
            frame_index=self.current_frame_index,
            timestamp_s=self.current_timestep,
            phase=self._current_action_phase,
            image=image,
            physical=self._current_physical_evaluation_metrics(
                contact_metrics, visual_metrics
            ),
            material=self._current_material_evaluation_metrics(
                stiffness_metrics
            ),
            details={
                "epoch": self._stiffness_evaluation_epoch,
                "stiffness_gate_paused": gate_paused,
                "stiffness_gate_reason": gate_reason,
            },
        )

    def _record_trajectory_observation(
        self,
        *,
        alignment,
        contact_metrics: dict | None,
    ) -> None:
        """Record the same pre-feedback image loss for every A/B/C mode.

        This observation is evaluated after the physics step and before the
        current frame's residual is applied.  It is therefore a fair
        prediction error for fixed PBD, residual-only and online-stiffness
        runs; post-residual fitting loss remains in the separate visual_update
        event.
        """
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        recorder.record(
            event="trajectory_observation",
            frame_index=self.current_frame_index,
            timestamp_s=self.current_timestep,
            phase=self._current_action_phase,
            image={
                "prediction_loss": alignment.loss,
                "left_right_prediction_losses": alignment.camera_losses,
                "camera_weight_sums": alignment.camera_weight_sums,
                "active_pixel_counts": (
                    alignment.camera_active_pixel_counts
                ),
                "mask_coverage_fractions": (
                    alignment.camera_mask_coverage_fractions
                ),
            },
            physical=self._current_physical_evaluation_metrics(
                contact_metrics
            ),
            material=self._current_material_evaluation_metrics(),
            details={
                "epoch": self._stiffness_evaluation_epoch,
                "pre_feedback": True,
            },
        )

    def _record_incomplete_stiffness_evaluations(self, reason: str) -> None:
        recorder = getattr(self, "stiffness_metrics_recorder", None)
        if recorder is None:
            return
        for evaluation in getattr(
            self, "_active_stiffness_evaluations", ()
        ):
            recorder.record(
                event="open_loop_incomplete",
                frame_index=self.current_frame_index,
                timestamp_s=self.current_timestep,
                phase=self._current_action_phase,
                prediction={
                    "evaluation_id": evaluation.evaluation_id,
                    "start_frame_index": evaluation.start_frame_index,
                    "remaining_horizons": sorted(
                        evaluation.pending_horizons
                    ),
                },
                details={"reason": reason},
                force_summary=True,
            )

    def close(self) -> None:
        # cross_frame_hold keeps the uncorrected branch live until validation,
        # so dropping an unfinished shadow here cannot mutate simulator state.
        self._pending_visual_residual_validation = None
        self._record_incomplete_stiffness_evaluations("runtime_closed")
        self._active_stiffness_evaluations.clear()
        if self.stiffness_metrics_recorder is not None:
            self.stiffness_metrics_recorder.close()
        if self.tissue_benchmark_recorder is not None:
            self.tissue_benchmark_recorder.close()

    def prepare_benchmark_frame(self, frame_index: int) -> None:
        recorder = getattr(self, "tissue_benchmark_recorder", None)
        self._benchmark_observation_enabled = (
            True if recorder is None else recorder.observation_allowed(frame_index)
        )
        if self._benchmark_observation_enabled:
            self._benchmark_observation_gap_frames = 0
            return
        self._benchmark_observation_gap_frames = int(
            getattr(self, "_benchmark_observation_gap_frames", 0)
        ) + 1
        if recorder is not None and recorder.protocol == "reconstruction_7to1":
            # Keep a material candidate alive across an interleaved test frame.
            # Its frozen validation_frame_indices contain training frames only;
            # the held-out tool command still advances physics but its RGB is
            # never evaluated or replayed. Post-commit diagnostic rollouts are
            # discarded here because they are not part of material admission.
            self._record_incomplete_stiffness_evaluations(
                "benchmark_observation_withheld"
            )
            self._active_stiffness_evaluations.clear()
            return
        if recorder is not None and recorder.protocol == "future_80to20":
            # Appearance proposals are admitted only by the next observed
            # training frame.  Once the future boundary is crossed there is
            # no legal observation left to validate them, so discard the tail
            # proposal explicitly instead of retaining unused visual state.
            self._pending_trajectory_appearance_result = None
            self._pending_trajectory_appearance_frame_index = -1
        # Preserve the last training-frame proposal across exactly the first
        # future frame.  It will be propagated with tool/physics commands and
        # judged only by hard physical safety -- never by future RGB.  Deeper
        # gaps cannot retain an unconfirmed visual branch.
        preserve_first_future_proposal = bool(
            recorder is not None
            and recorder.protocol == "future_80to20"
            and self._benchmark_observation_gap_frames == 1
            and self._pending_visual_residual_validation is not None
        )
        if not preserve_first_future_proposal:
            self._pending_visual_residual_validation = None
        # A held-out frame must not complete a material candidate that began
        # on a training frame.  Cancelling only optimizer bookkeeping leaves
        # the already accepted physical state and committed material intact.
        if self._pending_stiffness_validation is not None:
            assert self.stiffness_updater is not None
            pending = self._pending_stiffness_validation
            if STIFFNESS_ADMISSION_MODE == "causal_fixed_lag":
                self._clear_stiffness_global_confirmation()
                self._clear_stiffness_local_confirmations()
            metrics = self.stiffness_updater.reject(
                "benchmark_observation_withheld", pending.candidate
            )
            metrics.update(validation_status="cancelled")
            self._record_terminal_stiffness_validation(
                pending, metrics, "benchmark_observation_withheld"
            )
            self._pending_stiffness_validation = None
        self._record_incomplete_stiffness_evaluations(
            "benchmark_observation_withheld"
        )
        self._active_stiffness_evaluations.clear()

    @staticmethod
    def _one_frame_visual_prediction_gain(distance_median: float) -> float:
        """Scale missing-frame state prediction by known material relaxation."""
        if not np.isfinite(distance_median):
            return VISUAL_ONE_FRAME_CARRY_MINIMUM_GAIN
        interpolation = np.clip(
            (
                float(distance_median)
                - VISUAL_ONE_FRAME_CARRY_DISTANCE_MINIMUM
            )
            / (
                VISUAL_ONE_FRAME_CARRY_DISTANCE_MAXIMUM
                - VISUAL_ONE_FRAME_CARRY_DISTANCE_MINIMUM
            ),
            0.0,
            1.0,
        )
        return float(
            VISUAL_ONE_FRAME_CARRY_MINIMUM_GAIN
            + interpolation
            * (
                VISUAL_ONE_FRAME_CARRY_MAXIMUM_GAIN
                - VISUAL_ONE_FRAME_CARRY_MINIMUM_GAIN
            )
        )

    @staticmethod
    def _one_frame_visual_prediction_rejection_reasons(
        baseline: dict,
        candidate: dict,
    ) -> list[str]:
        """Check a no-RGB observer prediction using physical safety only."""
        reasons: list[str] = []
        baseline_inverted = int(baseline.get("inverted_tetrahedra", 0))
        candidate_inverted = int(candidate.get("inverted_tetrahedra", 0))
        if candidate_inverted > baseline_inverted:
            reasons.append("open_loop_new_tetrahedron_inversion")
        baseline_minimum = float(baseline.get("minimum_volume_ratio", 0.0))
        candidate_minimum = float(candidate.get("minimum_volume_ratio", 0.0))
        allowed_minimum = (
            VISUAL_CROSS_FRAME_MINIMUM_VOLUME_RATIO
            if baseline_minimum >= VISUAL_CROSS_FRAME_MINIMUM_VOLUME_RATIO
            else baseline_minimum * 0.98
        )
        if (
            not np.isfinite(candidate_minimum)
            or candidate_minimum < allowed_minimum
        ):
            reasons.append("open_loop_minimum_volume_unsafe")
        baseline_below_floor = int(
            baseline.get("tetrahedra_below_volume_floor", 0)
        )
        candidate_below_floor = int(
            candidate.get("tetrahedra_below_volume_floor", 0)
        )
        if candidate_below_floor > baseline_below_floor:
            reasons.append("open_loop_added_low_volume_tetrahedra")
        baseline_p01 = float(
            baseline.get("volume_ratio_p01", baseline_minimum)
        )
        candidate_p01 = float(
            candidate.get("volume_ratio_p01", candidate_minimum)
        )
        allowed_p01 = baseline_p01 - max(
            VISUAL_CROSS_FRAME_P01_ABSOLUTE_DROP,
            VISUAL_CROSS_FRAME_P01_RELATIVE_DROP * abs(baseline_p01),
        )
        if not np.isfinite(candidate_p01) or candidate_p01 < allowed_p01:
            reasons.append("open_loop_volume_distribution_regressed")
        baseline_weighted_mean = float(
            baseline.get("volume_weighted_mean_ratio", 1.0)
        )
        candidate_weighted_mean = float(
            candidate.get("volume_weighted_mean_ratio", 1.0)
        )
        allowed_weighted_mean = baseline_weighted_mean - max(
            VISUAL_CROSS_FRAME_WEIGHTED_MEAN_ABSOLUTE_DROP,
            VISUAL_CROSS_FRAME_WEIGHTED_MEAN_RELATIVE_DROP
            * abs(baseline_weighted_mean),
        )
        if (
            not np.isfinite(candidate_weighted_mean)
            or candidate_weighted_mean < allowed_weighted_mean
        ):
            reasons.append("open_loop_total_volume_regressed")
        baseline_penetration = float(
            baseline.get("maximum_penetration_m", 0.0)
        )
        candidate_penetration = float(
            candidate.get("maximum_penetration_m", 0.0)
        )
        if candidate_penetration > max(
            STIFFNESS_MAXIMUM_PENETRATION_M,
            baseline_penetration + STIFFNESS_PENETRATION_TOLERANCE_M,
        ):
            reasons.append("open_loop_penetration_regressed")
        baseline_anchor = float(
            baseline.get("anchor_error_maximum_m", 0.0)
        )
        candidate_anchor = float(
            candidate.get("anchor_error_maximum_m", 0.0)
        )
        if candidate_anchor > baseline_anchor + 2.0e-4:
            reasons.append("open_loop_anchor_regressed")
        return reasons

    def apply_one_frame_visual_open_loop_prediction(self) -> dict | None:
        """Carry the last confirmed residual across one missing RGB frame.

        This is a state-observer prediction, not feedback: the current image is
        explicitly unavailable, no render/loss is evaluated, velocities remain
        unchanged, and the correction is never repeated after gap frame one.
        """
        benchmark_recorder = getattr(self, "tissue_benchmark_recorder", None)
        benchmark_protocol = (
            None if benchmark_recorder is None else benchmark_recorder.protocol
        )
        ranked_cross_frame = bool(
            self.visual_residual_gain_profile == "cross_frame_ranked_hold"
        )
        if (
            self.visual_feedback_mode != "residual"
            or self.visual_residual_gain_profile
            not in {"cross_frame_hold", "cross_frame_ranked_hold"}
            or benchmark_recorder is None
            # Legacy behavior remains frozen for the equivalence/control run.
            # The ranked challenger restores the confirmed-residual predictor
            # on isolated reconstruction holdouts as originally designed.
            or (
                benchmark_protocol != "future_80to20"
                and not (
                    ranked_cross_frame
                    and benchmark_protocol == "reconstruction_7to1"
                )
            )
            or self._benchmark_observation_enabled
            or self._benchmark_observation_gap_frames != 1
            or self._last_visual_open_loop_prediction_frame_index
            == int(self.current_frame_index)
        ):
            return None
        if (
            benchmark_protocol == "future_80to20"
            and self._pending_visual_residual_validation is not None
        ):
            return self._adopt_pending_visual_residual_open_loop_prediction()
        # On an interleaved reconstruction holdout the newest gain branches
        # must remain pending for the next available training RGB.  Predict
        # this one missing frame only from the last *confirmed* residual.
        if self._previous_visual_residual is None:
            return None
        assert self.visual_residual_mapper is not None
        sim = self.environment.sim
        positions = wp.to_torch(sim.state_0.particle_q)
        original_positions = positions.detach().clone()
        baseline = self.visual_residual_mapper.physical_quality_metrics(
            original_positions
        )
        baseline_contact = sim.triangle_skin_contact_metrics() or {}
        baseline.update(
            maximum_penetration_m=float(
                baseline_contact.get("maximum_penetration_m", 0.0)
            ),
            anchor_error_maximum_m=float(
                baseline_contact.get(
                    "persistent_grip_anchor_error_maximum_m", 0.0
                )
            ),
        )
        projector = sim.material_projector
        if projector is None:
            distance_median = float(
                self.environment.physics_settings.paper_distance_stiffness
            )
        else:
            distance_median = float(
                wp.to_torch(projector.paper_distance_stiffness)
                .detach()
                .median()
                .item()
            )
        gain = self._one_frame_visual_prediction_gain(distance_median)
        correction = self.visual_residual_mapper._clip_vectors(
            self._previous_visual_residual.detach() * gain,
            VISUAL_ONE_FRAME_CARRY_MAXIMUM_CORRECTION_M,
        )
        dynamic_exclusion = grip_control_exclusion_mask(
            self.visual_residual_mapper,
            self.environment,
        )
        exclusion = self.visual_residual_mapper.fixed_mask | dynamic_exclusion
        correction = correction.masked_fill(exclusion[:, None], 0.0)
        accepted = False
        rejection_reasons: list[str] = []
        candidate = baseline
        installed_correction = torch.zeros_like(correction)
        backtracks = 0
        for backtracks in range(
            VISUAL_ONE_FRAME_CARRY_MAXIMUM_BACKTRACKS + 1
        ):
            trial_correction = correction * (0.5**backtracks)
            with torch.no_grad():
                positions.copy_(original_positions + trial_correction)
            sim.update_gaussian_transforms()
            candidate = self.visual_residual_mapper.physical_quality_metrics(
                positions
            )
            candidate_contact = sim.triangle_skin_contact_metrics() or {}
            candidate.update(
                maximum_penetration_m=float(
                    candidate_contact.get("maximum_penetration_m", 0.0)
                ),
                anchor_error_maximum_m=float(
                    candidate_contact.get(
                        "persistent_grip_anchor_error_maximum_m", 0.0
                    )
                ),
            )
            rejection_reasons = (
                self._one_frame_visual_prediction_rejection_reasons(
                    baseline, candidate
                )
            )
            if not rejection_reasons:
                accepted = True
                installed_correction = trial_correction
                break
        if not accepted:
            with torch.no_grad():
                positions.copy_(original_positions)
            sim.update_gaussian_transforms()
        self._last_visual_open_loop_prediction_frame_index = int(
            self.current_frame_index
        )
        self._visual_open_loop_prediction_count += 1
        metrics = {
            "accepted": accepted,
            "frame_index": int(self.current_frame_index),
            "observation_gap_frames": int(
                self._benchmark_observation_gap_frames
            ),
            "distance_stiffness_median": distance_median,
            "gain": gain,
            "backtracks": backtracks,
            "maximum_correction_m": float(
                torch.linalg.vector_norm(
                    installed_correction, dim=1
                ).max().item()
            ),
            "rms_correction_m": float(
                torch.sqrt(torch.mean(installed_correction.square())).item()
            ),
            "rejection_reasons": tuple(rejection_reasons),
            "baseline_physical_quality": baseline,
            "candidate_physical_quality": candidate,
            "uses_current_rgb": False,
            "uses_depth_or_point_cloud": False,
            "uses_manual_ground_truth": False,
        }
        if self.stiffness_metrics_recorder is not None:
            self.stiffness_metrics_recorder.record(
                event="visual_open_loop_prediction",
                frame_index=int(self.current_frame_index),
                timestamp_s=float(self.current_timestep),
                phase=self._current_action_phase,
                image=metrics,
                force_summary=True,
            )
        return metrics

    def _current_visual_open_loop_physical_metrics(self) -> dict:
        """Measure hard safety without rendering or reading an image."""
        assert self.visual_residual_mapper is not None
        sim = self.environment.sim
        positions = wp.to_torch(sim.state_0.particle_q)
        metrics = self.visual_residual_mapper.physical_quality_metrics(
            positions
        )
        contact = sim.triangle_skin_contact_metrics() or {}
        metrics.update(
            maximum_penetration_m=float(
                contact.get("maximum_penetration_m", 0.0)
            ),
            anchor_error_maximum_m=float(
                contact.get(
                    "persistent_grip_anchor_error_maximum_m", 0.0
                )
            ),
        )
        return metrics

    def _adopt_pending_visual_residual_open_loop_prediction(
        self,
    ) -> dict | None:
        """Propagate the final training correction into future frame one.

        The correction already passed same-image H1/H3/H5 persistence.  This
        method advances it using the exact recorded tool commands, compares
        only tetrahedron/contact/anchor safety to the live branch, and either
        installs or discards it.  No future observation is loaded or scored.
        """
        pending = self._pending_visual_residual_validation
        if pending is None or not pending.commands:
            return None
        sim = self.environment.sim
        self._synchronize_shadow_transaction(sim)
        live_state = sim.clone_embodied_gaussian_rollout_state()
        live_frame_index = int(self.current_frame_index)
        live_timestep = float(self.current_timestep)
        live_state_index = int(self._last_state_index)
        live_q_full = self._last_q_full.detach().clone()
        baseline = self._current_visual_open_loop_physical_metrics()
        candidate_state = None
        try:
            sim.copy_embodied_gaussian_rollout_state(pending.candidate_state)
            for command in pending.commands:
                self._apply_stiffness_tool_command(command)
                self.environment.step(compute_visual_forces=False)
                self.apply_current_psm_pose()
            sim.update_gaussian_transforms()
            candidate = self._current_visual_open_loop_physical_metrics()
            self._synchronize_shadow_transaction(sim)
            candidate_state = sim.clone_embodied_gaussian_rollout_state()
        finally:
            self._synchronize_shadow_transaction(sim)
            sim.copy_embodied_gaussian_rollout_state(live_state)
            self.current_frame_index = live_frame_index
            self.current_timestep = live_timestep
            self._last_state_index = live_state_index
            self._last_q_full = live_q_full
            self.apply_current_psm_pose()
            sim.update_gaussian_transforms()
            self._synchronize_shadow_transaction(sim)
        reasons = self._one_frame_visual_prediction_rejection_reasons(
            baseline, candidate
        )
        accepted = not reasons
        if accepted:
            assert candidate_state is not None
            sim.copy_embodied_gaussian_rollout_state(candidate_state)
            self.current_frame_index = live_frame_index
            self.current_timestep = live_timestep
            self._last_state_index = live_state_index
            self._last_q_full = live_q_full
            self.apply_current_psm_pose()
            sim.update_gaussian_transforms()
            self._previous_visual_residual = (
                pending.result.residual.detach().clone()
            )
        else:
            self._previous_visual_residual = None
        self._pending_visual_residual_validation = None
        self._last_visual_open_loop_prediction_frame_index = live_frame_index
        self._visual_open_loop_prediction_count += 1
        metrics = {
            "accepted": accepted,
            "mode": "pending_training_residual_state",
            "start_frame_index": int(pending.frame_index),
            "frame_index": live_frame_index,
            "observation_gap_frames": int(
                self._benchmark_observation_gap_frames
            ),
            "selected_gain": float(pending.selected_gain),
            "rejection_reasons": tuple(reasons),
            "baseline_physical_quality": baseline,
            "candidate_physical_quality": candidate,
            "uses_current_rgb": False,
            "uses_future_rgb": False,
            "uses_depth_or_point_cloud": False,
            "uses_manual_ground_truth": False,
        }
        if self.stiffness_metrics_recorder is not None:
            self.stiffness_metrics_recorder.record(
                event="visual_open_loop_prediction",
                frame_index=live_frame_index,
                timestamp_s=live_timestep,
                phase=self._current_action_phase,
                image=metrics,
                force_summary=True,
            )
        return metrics

    def _stiffness_training_validation_frames(
        self,
        start_frame_index: int,
        horizons: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """Map material horizons to observable training-frame indices.

        A trajectory-feedback material shadow needs an actual AllTracker/depth
        destination at every checkpoint.  In that mode H1/H3/H5 count causal
        tracked observations (and, for 7:1 reconstruction, only training
        observations), rather than raw video frames that may have no flow
        sample at all.
        """
        horizons = tuple(horizons or STIFFNESS_ADMISSION_HORIZONS)
        playback_timestamps = getattr(self, "playback_timestamps", ())
        # Lightweight policy/evaluation harnesses intentionally construct the
        # controller without loading a dataset.  In that case the requested
        # horizons themselves define the finite synthetic frame extent; a real
        # playback always supplies its exact timestamp count here.
        frame_limit = len(playback_timestamps)
        if frame_limit <= 0:
            frame_limit = int(start_frame_index) + max(horizons, default=0) + 1
        recorder = getattr(self, "tissue_benchmark_recorder", None)
        trajectory_sequence = (
            getattr(self, "flow_depth_observations", None)
            if getattr(self, "visual_feedback_mode", None) == "trajectory"
            else None
        )
        if recorder is None and trajectory_sequence is None:
            return tuple(
                int(start_frame_index) + horizon
                for horizon in horizons
                if int(start_frame_index) + horizon
                < frame_limit
            )
        if (
            trajectory_sequence is None
            and recorder is not None
            and recorder.protocol != "reconstruction_7to1"
        ):
            # Future prediction may identify material only from its 0..1151
            # training prefix.  Never create a validation target that would
            # read withheld RGB after that boundary.
            return tuple(
                int(start_frame_index) + horizon
                for horizon in horizons
                if int(start_frame_index) + horizon
                < frame_limit
                and recorder.observation_allowed(
                    int(start_frame_index) + horizon
                )
            )
        required = set(horizons)
        targets: dict[int, int] = {}
        observed_count = 0
        frame_index = int(start_frame_index) + 1
        while frame_index < frame_limit and len(targets) < len(required):
            training_observation = bool(
                recorder is None or recorder.observation_allowed(frame_index)
            )
            trajectory_observation = bool(
                trajectory_sequence is None
                or trajectory_sequence.pair_index_for_next_frame(frame_index)
                is not None
            )
            if training_observation and trajectory_observation:
                observed_count += 1
                if observed_count in required:
                    targets[observed_count] = frame_index
            frame_index += 1
        return tuple(
            targets[horizon]
            for horizon in horizons
            if horizon in targets
        )

    def _pending_stiffness_validation_frames(
        self,
        pending: PendingStiffnessValidation,
    ) -> tuple[int, ...]:
        if pending.validation_frame_indices:
            return tuple(int(value) for value in pending.validation_frame_indices)
        return self._stiffness_training_validation_frames(pending.frame_index)

    def _stiffness_training_replay_frames(
        self,
        pending: PendingStiffnessValidation,
        target_frames: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """Return only observable training frames inside the fixed lag."""
        targets = tuple(
            target_frames or self._pending_stiffness_validation_frames(pending)
        )
        if not targets:
            return ()
        recorder = getattr(self, "tissue_benchmark_recorder", None)
        return tuple(
            frame_index
            for frame_index in range(
                int(pending.frame_index) + 1, int(targets[-1]) + 1
            )
            if recorder is None or recorder.observation_allowed(frame_index)
        )

    def _current_stiffness_tool_command(self) -> StiffnessToolCommand:
        return StiffnessToolCommand(
            frame_index=int(self.current_frame_index),
            timestep=float(self.current_timestep),
            state_index=int(self._last_state_index),
            phase=self._current_action_phase,
        )

    @staticmethod
    def _synchronize_shadow_transaction(sim) -> None:
        """Fence Warp/Torch streams before cloning or restoring a shadow.

        Rollout state is copied by Warp while material tensors and visual
        residuals are copied by Torch.  Without an explicit device fence, a
        rejected CUDA shadow can finish writing after the live snapshot has
        nominally been restored, producing the observed no-commit drift.
        """
        model_device = getattr(getattr(sim, "model", None), "device", None)
        if model_device is not None:
            wp.synchronize_device(model_device)
        if torch.cuda.is_available() and str(model_device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(str(model_device)))

    def _apply_stiffness_tool_command(
        self, command: StiffnessToolCommand
    ) -> None:
        self.current_frame_index = int(command.frame_index)
        self.current_timestep = float(command.timestep)
        self._last_state_index = int(command.state_index)
        self.apply_current_psm_pose()

    def _stiffness_global_gate(
        self, contact_metrics: dict | None
    ) -> tuple[bool, str, float, bool]:
        if contact_metrics is None:
            return True, "missing_contact_metrics", float("inf"), False
        jaw_angle = float(
            contact_metrics.get("persistent_grip_q7_angle_rad", 0.0)
        )
        timestamp = float(
            contact_metrics.get("persistent_grip_q7_timestamp_s", 0.0)
        )
        jaw_speed = 0.0
        if (
            self._last_stiffness_gate_timestep is not None
            and timestamp > self._last_stiffness_gate_timestep
            and self._last_stiffness_gate_jaw_angle is not None
        ):
            jaw_speed = abs(
                jaw_angle - self._last_stiffness_gate_jaw_angle
            ) / (timestamp - self._last_stiffness_gate_timestep)
        grip_active = bool(
            contact_metrics.get("persistent_grip_active", False)
        )
        transition_reasons: list[str] = []
        if grip_active != self._last_stiffness_grip_active:
            transition_reasons.append("capture_or_release")
        if (
            STIFFNESS_MAXIMUM_JAW_SPEED_RAD_S is not None
            and jaw_speed > STIFFNESS_MAXIMUM_JAW_SPEED_RAD_S
        ):
            transition_reasons.append("rapid_q7")
        if (
            float(contact_metrics.get("maximum_penetration_m", 0.0))
            > STIFFNESS_MAXIMUM_PENETRATION_M
        ):
            transition_reasons.append("excessive_penetration")
        contact_count = int(contact_metrics.get("contact_count", 0))
        previous_contact_count = self._last_stiffness_contact_count
        if (
            previous_contact_count is not None
            and min(contact_count, previous_contact_count) > 0
            and abs(contact_count - previous_contact_count) >= 8
            and max(contact_count, previous_contact_count)
            > 2 * min(contact_count, previous_contact_count)
        ):
            transition_reasons.append("contact_patch_switch")
        if transition_reasons:
            self._stiffness_transition_cooldown = max(
                self._stiffness_transition_cooldown,
                STIFFNESS_TRANSITION_COOLDOWN_UPDATES,
            )
            if (
                self.stiffness_updater is not None
                and (
                    "capture_or_release" in transition_reasons
                    or "contact_patch_switch" in transition_reasons
                    or "excessive_penetration" in transition_reasons
                )
            ):
                self.stiffness_updater.invalidate_signal_history()
        paused = self._stiffness_transition_cooldown > 0
        reason = (
            "+".join(transition_reasons)
            if transition_reasons
            else ("transition_cooldown" if paused else "")
        )
        if self._stiffness_transition_cooldown > 0:
            self._stiffness_transition_cooldown -= 1
        self._last_stiffness_gate_timestep = timestamp
        self._last_stiffness_gate_jaw_angle = jaw_angle
        self._last_stiffness_grip_active = grip_active
        self._last_stiffness_contact_count = contact_count
        return paused, reason, jaw_speed, grip_active

    def _mass_weighted_particle_rms(
        self, reference: torch.Tensor, current: torch.Tensor
    ) -> float:
        inverse_mass = wp.to_torch(
            self.environment.sim.model.particle_inv_mass
        ).detach()
        reference = reference.to(device=inverse_mass.device)
        current = current.to(device=inverse_mass.device)
        dynamic = inverse_mass > 0.0
        if not bool(dynamic.any().item()):
            return 0.0
        mass = 1.0 / inverse_mass[dynamic]
        squared = torch.sum((current[dynamic] - reference[dynamic]) ** 2, dim=1)
        return float(torch.sqrt(torch.sum(mass * squared) / mass.sum()).item())

    def _run_history_relaxation(
        self,
        snapshot: StiffnessHistorySnapshot,
        candidate: PaperStiffnessCandidate | None,
        verified_distance: torch.Tensor,
        verified_shape: torch.Tensor,
    ) -> float:
        sim = self.environment.sim
        sim.copy_embodied_gaussian_rollout_state(snapshot.rollout_state)
        particle_qd = wp.to_torch(sim.state_0.particle_qd)
        with torch.no_grad():
            particle_qd.zero_()
        assert self.stiffness_updater is not None
        # A historical snapshot carries the stiffness that was verified when
        # it was captured.  Both sides of today's regression test must instead
        # start from today's verified material; otherwise the comparison mixes
        # material age with the candidate's effect.
        self.stiffness_updater.restore_verified_stiffness(
            verified_distance, verified_shape
        )
        if candidate is not None:
            self.stiffness_updater.install_candidate_for_rollout(candidate)
        projector = sim.triangle_skin_contact_projector
        if projector is not None:
            projector.freeze_persistent_grip_state_machine()
        sim.sync_kinematic_body_interpolation()
        self.environment.step(compute_visual_forces=False)
        if projector is not None:
            projector.freeze_persistent_grip_state_machine()
        current = wp.to_torch(sim.state_0.particle_q).detach()
        return self._mass_weighted_particle_rms(
            snapshot.accepted_positions, current
        )

    def _evaluate_stiffness_history(
        self, candidate: PaperStiffnessCandidate
    ) -> tuple[float, float, bool]:
        if not self._stiffness_history:
            return 0.0, 0.0, True
        sim = self.environment.sim
        self._synchronize_shadow_transaction(sim)
        live = sim.clone_embodied_gaussian_rollout_state()
        assert self.stiffness_updater is not None
        verified_distance = (
            self.stiffness_updater.distance_stiffness.detach().clone()
        )
        verified_shape = (
            self.stiffness_updater.shape_stiffness.detach().clone()
        )
        baseline_values: list[float] = []
        candidate_values: list[float] = []
        try:
            for snapshot in self._stiffness_history:
                baseline_values.append(
                    self._run_history_relaxation(
                        snapshot,
                        None,
                        verified_distance,
                        verified_shape,
                    )
                )
                candidate_values.append(
                    self._run_history_relaxation(
                        snapshot,
                        candidate,
                        verified_distance,
                        verified_shape,
                    )
                )
        finally:
            self._synchronize_shadow_transaction(sim)
            sim.copy_embodied_gaussian_rollout_state(live)
            self.stiffness_updater.restore_verified_stiffness(
                verified_distance, verified_shape
            )
            sim.update_gaussian_transforms()
            self._synchronize_shadow_transaction(sim)
        baseline = float(np.mean(baseline_values))
        candidate_loss = float(np.mean(candidate_values))
        allowed = (
            baseline * (1.0 + STIFFNESS_HISTORY_RELATIVE_TOLERANCE)
            + STIFFNESS_HISTORY_ABSOLUTE_TOLERANCE_M
        )
        return baseline, candidate_loss, candidate_loss <= allowed

    def _maybe_store_stiffness_history(
        self,
        *,
        accepted_positions: torch.Tensor,
        jaw_speed_rad_s: float,
        gate_paused: bool,
    ) -> None:
        if gate_paused or jaw_speed_rad_s > STIFFNESS_HISTORY_MAXIMUM_JAW_SPEED_RAD_S:
            return
        speeds = torch.linalg.vector_norm(
            wp.to_torch(self.environment.sim.state_0.particle_qd), dim=1
        )
        if float(speeds.max().item()) > STIFFNESS_HISTORY_MAXIMUM_PARTICLE_SPEED_M_S:
            return
        self._stiffness_history.append(
            StiffnessHistorySnapshot(
                rollout_state=(
                    self.environment.sim.clone_embodied_gaussian_rollout_state()
                ),
                accepted_positions=accepted_positions.detach().clone(),
                frame_index=int(self.current_frame_index),
            )
        )

    def _run_stiffness_rollout_shadow(
        self,
        *,
        rollout_state: object,
        commands: list[StiffnessToolCommand],
        candidate: PaperStiffnessCandidate,
        use_candidate: bool,
        freeze_grip_state_machine: bool,
        measure_open_loop_residual: bool = False,
        replay_visual_residuals: bool = False,
        initial_previous_residual: torch.Tensor | None = None,
        replay_after_frame_index: int | None = None,
        replay_frame_indices: tuple[int, ...] = (),
        validation_frame_indices: tuple[int, ...] = (),
    ) -> dict:
        """Run one material counterfactual as a stiffness transaction.

        Simulator snapshots include material arrays, but candidate construction
        happens between consecutive shadow calls.  Leaving the just-tested
        candidate installed until the next call therefore contaminates the
        next candidate before that next snapshot is restored.  Preserve and
        restore the verified tensors at this boundary, including exceptions.
        """
        assert self.stiffness_updater is not None
        sim = getattr(getattr(self, "environment", None), "sim", None)
        if sim is not None:
            self._synchronize_shadow_transaction(sim)
        verified_distance = (
            self.stiffness_updater.distance_stiffness.detach().clone()
        )
        verified_shape = (
            self.stiffness_updater.shape_stiffness.detach().clone()
        )
        try:
            return self._run_stiffness_rollout_shadow_unisolated(
                rollout_state=rollout_state,
                commands=commands,
                candidate=candidate,
                use_candidate=use_candidate,
                freeze_grip_state_machine=freeze_grip_state_machine,
                measure_open_loop_residual=measure_open_loop_residual,
                replay_visual_residuals=replay_visual_residuals,
                initial_previous_residual=initial_previous_residual,
                replay_after_frame_index=replay_after_frame_index,
                replay_frame_indices=replay_frame_indices,
                validation_frame_indices=validation_frame_indices,
            )
        finally:
            if sim is not None:
                self._synchronize_shadow_transaction(sim)
            self.stiffness_updater.restore_verified_stiffness(
                verified_distance, verified_shape
            )
            if sim is not None:
                self._synchronize_shadow_transaction(sim)

    def _alltracker_candidate_track_support(
        self,
        candidate: PaperStiffnessCandidate,
    ) -> np.ndarray | None:
        """Return tracks directly bound to particles changed by a proposal."""

        bindings = self.flow_depth_bindings
        if bindings is None:
            return None
        changed = (
            (candidate.distance_log_step.detach().abs() > 1.0e-12)
            | (candidate.shape_log_step.detach().abs() > 1.0e-12)
        ).cpu().numpy()
        supported = np.zeros(len(bindings.track_valid), dtype=bool)
        for track_id in np.flatnonzero(bindings.track_valid):
            count = int(bindings.support_counts[track_id])
            particle_ids = bindings.particle_ids[track_id, :count]
            if (
                count > 0
                and np.all(particle_ids >= 0)
                and np.all(particle_ids < len(changed))
            ):
                supported[track_id] = bool(np.any(changed[particle_ids]))
        return supported

    def _evaluate_alltracker_trajectory_alignment(
        self,
        particle_positions: torch.Tensor,
        frame_index: int,
        candidate_track_support: np.ndarray | None,
    ) -> dict | None:
        """Measure fixed-material range centres against AllTracker+depth 3D."""

        bindings = self.flow_depth_bindings
        sequence = self.flow_depth_observations
        reference_centers = self._flow_depth_reference_range_centers
        if (
            self.visual_feedback_mode not in TRAJECTORY_FEEDBACK_MODES
            or bindings is None
            or sequence is None
            or reference_centers is None
        ):
            return None
        pair_index = sequence.pair_index_for_next_frame(int(frame_index))
        if pair_index is None:
            return None
        observation = sequence.observation(pair_index)
        predicted_centers = fixed_range_centers(
            particle_positions.detach().cpu().numpy(), bindings
        )
        observed_centers = reference_centers + (
            observation.next_points_table - bindings.initial_points_table
        )
        valid = (
            bindings.track_valid
            & observation.track_valid
            & np.isfinite(predicted_centers).all(axis=1)
            & np.isfinite(observed_centers).all(axis=1)
            & (observation.confidence > 0.0)
        )
        local_valid = (
            valid & candidate_track_support
            if candidate_track_support is not None
            else np.zeros_like(valid)
        )
        use_local = bool(
            np.count_nonzero(local_valid)
            >= STIFFNESS_TRAJECTORY_MINIMUM_LOCAL_TRACKS
        )
        objective_mask = local_valid if use_local else valid
        if not np.any(objective_mask):
            return None
        errors = np.linalg.vector_norm(
            predicted_centers - observed_centers, axis=1
        )
        weights = np.where(
            objective_mask,
            observation.confidence.astype(np.float64),
            0.0,
        )
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            return None
        selected_errors = errors[objective_mask]
        selected_weights = weights[objective_mask]
        return {
            # Do not multiply invalid tracks by zero: IEEE 0*NaN is still NaN
            # and previously made every real AllTracker objective silently
            # fall back to RGB admission.
            "mean_error_m": float(
                np.sum(selected_weights * selected_errors) / weight_sum
            ),
            "rmse_m": float(
                np.sqrt(
                    np.sum(
                        selected_weights * selected_errors * selected_errors
                    )
                    / weight_sum
                )
            ),
            "median_error_m": float(np.median(selected_errors)),
            "p90_error_m": float(np.quantile(selected_errors, 0.90)),
            "valid_tracks": int(np.count_nonzero(objective_mask)),
            "global_valid_tracks": int(np.count_nonzero(valid)),
            "local_valid_tracks": int(np.count_nonzero(local_valid)),
            "confidence_sum": float(selected_weights.sum()),
            "scope": "candidate_local" if use_local else "global_fallback",
            "source_frame": int(sequence.current_source_frames[pair_index]),
            "destination_frame": int(frame_index),
        }

    def _run_stiffness_rollout_shadow_unisolated(
        self,
        *,
        rollout_state: object,
        commands: list[StiffnessToolCommand],
        candidate: PaperStiffnessCandidate,
        use_candidate: bool,
        freeze_grip_state_machine: bool,
        measure_open_loop_residual: bool = False,
        replay_visual_residuals: bool = False,
        initial_previous_residual: torch.Tensor | None = None,
        replay_after_frame_index: int | None = None,
        replay_frame_indices: tuple[int, ...] = (),
        validation_frame_indices: tuple[int, ...] = (),
    ) -> dict:
        assert self.stiffness_updater is not None
        assert self.visual_residual_mapper is not None
        sim = self.environment.sim
        sim.copy_embodied_gaussian_rollout_state(rollout_state)
        if use_candidate:
            self.stiffness_updater.install_candidate_for_rollout(
                candidate
            )
        projector = sim.triangle_skin_contact_projector
        target_frame_index = (
            commands[-1].frame_index if commands else self.current_frame_index
        )
        previous_residual = (
            None
            if initial_previous_residual is None
            else initial_previous_residual.detach().clone()
        )
        previous_replay_timestep = (
            float(self.playback_timestamps[replay_after_frame_index])
            if replay_after_frame_index is not None
            and 0 <= replay_after_frame_index < len(self.playback_timestamps)
            else None
        )
        validation_frames = set(int(value) for value in validation_frame_indices)
        replay_frames = set(int(value) for value in replay_frame_indices)
        validation_visual_losses: dict[int, float] = {}
        validation_camera_losses: dict[int, tuple[float, ...]] = {}
        validation_trajectory_metrics: dict[int, dict] = {}
        candidate_track_support = self._alltracker_candidate_track_support(
            candidate
        )
        # Causal fixed-lag admission retains every pre-residual training-image
        # loss; legacy material-isolation admission uses only the explicit
        # validation checkpoints and calls with residual replay disabled.
        trajectory_visual_losses: list[float] = []
        trajectory_camera_losses: list[tuple[float, ...]] = []
        for command_index, command in enumerate(commands):
            self._apply_stiffness_tool_command(command)
            if projector is not None and freeze_grip_state_machine:
                projector.freeze_persistent_grip_state_machine()
            self.environment.step(compute_visual_forces=False)
            self.apply_current_psm_pose()
            if projector is not None and freeze_grip_state_machine:
                projector.freeze_persistent_grip_state_machine()
            next_frame_index = (
                commands[command_index + 1].frame_index
                if command_index + 1 < len(commands)
                else None
            )
            frame_complete = next_frame_index != command.frame_index
            replay_this_frame = bool(
                replay_visual_residuals
                and frame_complete
                and command.frame_index < target_frame_index
                and (
                    replay_after_frame_index is None
                    or command.frame_index > replay_after_frame_index
                )
                and (
                    not replay_frames
                    or command.frame_index in replay_frames
                )
            )
            validate_this_frame = bool(
                frame_complete and command.frame_index in validation_frames
            )
            if replay_this_frame or validate_this_frame:
                self.dataset_manager.update_frames(command.timestep)
                sim.update_gaussian_transforms()
            if validate_this_frame:
                assert self.environment.frames is not None
                checkpoint_alignment = (
                    sim.evaluate_visual_tissue_alignment(
                        self.visual_residual_mapper,
                        self.environment.frames,
                        observations_are_bgr=True,
                    )
                )
                validation_visual_losses[int(command.frame_index)] = float(
                    checkpoint_alignment.loss
                )
                validation_camera_losses[int(command.frame_index)] = tuple(
                    checkpoint_alignment.camera_losses
                )
                trajectory_alignment = (
                    self._evaluate_alltracker_trajectory_alignment(
                        wp.to_torch(sim.state_0.particle_q).detach(),
                        int(command.frame_index),
                        candidate_track_support,
                    )
                )
                if trajectory_alignment is not None:
                    validation_trajectory_metrics[int(command.frame_index)] = (
                        trajectory_alignment
                    )
            if replay_this_frame:
                control_exclusion_mask = grip_control_exclusion_mask(
                    self.visual_residual_mapper,
                    self.environment,
                )
                residual = sim.solve_visual_tissue_residual(
                    self.visual_residual_mapper,
                    self.environment.frames,
                    previous_residual=previous_residual,
                    dynamic_exclusion_mask=control_exclusion_mask,
                    observations_are_bgr=True,
                )
                trajectory_visual_losses.append(
                    float(residual.initial_visual_loss)
                )
                trajectory_camera_losses.append(
                    tuple(residual.initial_camera_visual_losses)
                )
                if previous_replay_timestep is not None:
                    replay_observation_dt_s = float(
                        command.timestep - previous_replay_timestep
                    )
                elif command.frame_index > 0:
                    replay_observation_dt_s = float(
                        self.playback_timestamps[command.frame_index]
                        - self.playback_timestamps[command.frame_index - 1]
                    )
                else:
                    replay_observation_dt_s = 1.0 / float(self.fps)
                accepted = sim.apply_visual_tissue_residual(
                    residual,
                    mapper=self.visual_residual_mapper,
                    frames=self.environment.frames,
                    observations_are_bgr=True,
                    observation_dt_s=max(replay_observation_dt_s, 1.0e-6),
                    velocity_correction_gain=(
                        VISUAL_RESIDUAL_VELOCITY_CORRECTION_GAIN
                    ),
                    maximum_velocity_correction_m_s=(
                        VISUAL_RESIDUAL_MAXIMUM_VELOCITY_CORRECTION_M_S
                    ),
                    dynamic_exclusion_mask=control_exclusion_mask,
                )
                previous_replay_timestep = float(command.timestep)
                previous_residual = (
                    residual.residual.detach().clone() if accepted else None
                )
        if commands:
            self.dataset_manager.update_frames(commands[-1].timestep)
        sim.update_gaussian_transforms()
        assert self.environment.frames is not None
        alignment = sim.evaluate_visual_tissue_alignment(
            self.visual_residual_mapper,
            self.environment.frames,
            observations_are_bgr=True,
        )
        if replay_visual_residuals:
            trajectory_visual_losses.append(float(alignment.loss))
            trajectory_camera_losses.append(tuple(alignment.camera_losses))
        trajectory_visual_loss = (
            float(np.mean(trajectory_visual_losses))
            if trajectory_visual_losses
            else float(alignment.loss)
        )
        trajectory_camera_loss = (
            tuple(
                float(np.mean(values))
                for values in zip(*trajectory_camera_losses)
            )
            if trajectory_camera_losses
            else tuple(alignment.camera_losses)
        )
        physical = self.visual_residual_mapper.physical_quality_metrics(
            wp.to_torch(sim.state_0.particle_q)
        )
        contact = sim.triangle_skin_contact_metrics() or {}
        open_loop_residual = None
        if measure_open_loop_residual:
            control_exclusion_mask = grip_control_exclusion_mask(
                self.visual_residual_mapper,
                self.environment,
            )
            open_loop_residual = sim.solve_visual_tissue_residual(
                self.visual_residual_mapper,
                self.environment.frames,
                previous_residual=None,
                dynamic_exclusion_mask=control_exclusion_mask,
                observations_are_bgr=True,
            )
        particle_positions = (
            wp.to_torch(sim.state_0.particle_q).detach().clone()
        )
        final_rollout_state = sim.clone_embodied_gaussian_rollout_state()
        return {
            "visual_loss": alignment.loss,
            "camera_losses": alignment.camera_losses,
            "trajectory_visual_loss": trajectory_visual_loss,
            "trajectory_camera_losses": trajectory_camera_loss,
            "validation_visual_losses": validation_visual_losses,
            "validation_camera_losses": validation_camera_losses,
            "validation_trajectory_metrics": validation_trajectory_metrics,
            "camera_weight_sums": alignment.camera_weight_sums,
            "camera_active_pixel_counts": (
                alignment.camera_active_pixel_counts
            ),
            "camera_mask_coverage_fractions": (
                alignment.camera_mask_coverage_fractions
            ),
            **physical,
            "maximum_penetration_m": float(
                contact.get("maximum_penetration_m", 0.0)
            ),
            "anchor_error_rms_m": float(
                contact.get("persistent_grip_anchor_error_rms_m", 0.0)
            ),
            "anchor_error_maximum_m": float(
                contact.get(
                    "persistent_grip_anchor_error_maximum_m", 0.0
                )
            ),
            "grip_active": bool(
                contact.get("persistent_grip_active", False)
            ),
            "open_loop_residual_rms_m": (
                None
                if open_loop_residual is None
                else open_loop_residual.rms_residual_m
            ),
            "open_loop_residual_maximum_m": (
                None
                if open_loop_residual is None
                else open_loop_residual.maximum_residual_m
            ),
            "particle_positions": particle_positions,
            "rollout_state": final_rollout_state,
            "rollout_previous_residual": (
                None
                if previous_residual is None
                else previous_residual.detach().clone()
            ),
        }

    def _run_stiffness_prediction_shadow(
        self,
        pending: PendingStiffnessValidation,
        *,
        use_candidate: bool,
        horizons: tuple[int, ...] | None = None,
        target_frames: tuple[int, ...] | None = None,
    ) -> dict:
        horizons = tuple(horizons or STIFFNESS_ADMISSION_HORIZONS)
        target_frames = tuple(
            target_frames or self._pending_stiffness_validation_frames(pending)
        )
        if len(target_frames) != len(horizons):
            return {
                "visual_loss": float("inf"),
                "camera_losses": (),
                "endpoint_visual_loss": float("inf"),
                "endpoint_camera_losses": (),
                "validation_horizons_complete": False,
                "horizon_visual_losses": {
                    horizon: None for horizon in horizons
                },
                "horizon_camera_losses": {
                    horizon: None for horizon in horizons
                },
                "trajectory_tracking_loss_m": None,
                "horizon_trajectory_losses_m": {
                    horizon: None for horizon in horizons
                },
                "horizon_trajectory_metrics": {
                    horizon: None for horizon in horizons
                },
                "validation_horizons": horizons,
                "minimum_volume_ratio": 0.0,
                "inverted_tetrahedra": 0,
                "tetrahedra_below_volume_floor": 0,
                "maximum_penetration_m": 0.0,
                "anchor_error_rms_m": 0.0,
                "anchor_error_maximum_m": 0.0,
                "open_loop_residual_rms_m": None,
                "particle_positions": None,
                "rollout_state": None,
                "rollout_previous_residual": None,
            }
        commands = [
            command
            for command in pending.commands
            if command.frame_index <= target_frames[-1]
        ]
        causal_fixed_lag = STIFFNESS_ADMISSION_MODE in {
            "causal_fixed_lag",
            "relaxed_h135",
        }
        replay_rgb_residuals = bool(
            causal_fixed_lag
            and getattr(self, "visual_feedback_mode", "residual") == "residual"
        )
        result = self._run_stiffness_rollout_shadow(
            rollout_state=pending.rollout_state,
            commands=commands,
            candidate=pending.candidate,
            use_candidate=use_candidate,
            # The causal fixed-lag policy replays the exact training-time
            # observer and grip transitions in both branches, as in sim.  The
            # legacy policies retain their residual-off isolation semantics.
            freeze_grip_state_machine=not causal_fixed_lag,
            measure_open_loop_residual=(
                causal_fixed_lag
                or STIFFNESS_CANDIDATE_PROFILE
                == "robust_hierarchical_system_id"
            ),
            replay_visual_residuals=replay_rgb_residuals,
            initial_previous_residual=(
                pending.previous_residual if replay_rgb_residuals else None
            ),
            replay_after_frame_index=(
                pending.frame_index if replay_rgb_residuals else None
            ),
            replay_frame_indices=(
                self._stiffness_training_replay_frames(
                    pending, target_frames
                )
                if replay_rgb_residuals
                else ()
            ),
            validation_frame_indices=target_frames,
        )
        checkpoint_losses = result["validation_visual_losses"]
        checkpoint_camera_losses = result["validation_camera_losses"]
        checkpoint_trajectory_metrics = result.get(
            "validation_trajectory_metrics", {}
        )
        trajectory_expected = (
            getattr(self, "visual_feedback_mode", None) == "trajectory"
        )
        complete = all(frame in checkpoint_losses for frame in target_frames)
        trajectory_complete = bool(
            not trajectory_expected
            or all(
                frame in checkpoint_trajectory_metrics
                for frame in target_frames
            )
        )
        complete = bool(complete and trajectory_complete)
        horizon_losses = {
            horizon: checkpoint_losses.get(frame)
            for horizon, frame in zip(horizons, target_frames)
        }
        horizon_camera_losses = {
            horizon: checkpoint_camera_losses.get(frame)
            for horizon, frame in zip(horizons, target_frames)
        }
        horizon_trajectory_metrics = {
            horizon: checkpoint_trajectory_metrics.get(frame)
            for horizon, frame in zip(horizons, target_frames)
        }
        horizon_trajectory_losses_m = {
            horizon: (
                None
                if horizon_trajectory_metrics[horizon] is None
                else float(
                    horizon_trajectory_metrics[horizon]["mean_error_m"]
                )
            )
            for horizon in horizons
        }
        result["endpoint_visual_loss"] = result["visual_loss"]
        result["endpoint_camera_losses"] = result["camera_losses"]
        result["validation_horizons_complete"] = complete
        result["horizon_visual_losses"] = horizon_losses
        result["horizon_camera_losses"] = horizon_camera_losses
        result["horizon_trajectory_losses_m"] = (
            horizon_trajectory_losses_m
        )
        result["horizon_trajectory_metrics"] = horizon_trajectory_metrics
        result["trajectory_horizons_complete"] = trajectory_complete
        result["trajectory_tracking_loss_m"] = None
        result["validation_horizons"] = horizons
        if trajectory_complete and trajectory_expected:
            trajectory_weights = np.asarray(horizons, dtype=np.float64)
            trajectory_weights /= trajectory_weights.sum()
            result["trajectory_tracking_loss_m"] = float(
                np.average(
                    [horizon_trajectory_losses_m[h] for h in horizons],
                    weights=trajectory_weights,
                )
            )
            result["stiffness_admission_objective"] = (
                "alltracker_depth_3d_epe_candidate_local_horizon_weighted"
            )
        else:
            result["stiffness_admission_objective"] = (
                "training_rgb_robust_equal_camera"
            )
        if complete:
            if causal_fixed_lag:
                # This is the sim fixed-lag objective: every loss is measured
                # before applying that frame's residual, then the same causal
                # residual solve advances both material branches.  It captures
                # how much observer effort each material needs over the window.
                result["visual_loss"] = result["trajectory_visual_loss"]
                result["camera_losses"] = result[
                    "trajectory_camera_losses"
                ]
            elif STIFFNESS_ADMISSION_MODE == "weighted_window":
                horizon_weights = np.asarray(
                    horizons, dtype=np.float64
                )
                horizon_weights /= horizon_weights.sum()
            else:
                horizon_weights = np.full(
                    len(horizons),
                    1.0 / len(horizons),
                    dtype=np.float64,
                )
            if not causal_fixed_lag:
                result["visual_loss"] = float(
                    np.average(
                        [
                            horizon_losses[h]
                            for h in horizons
                        ],
                        weights=horizon_weights,
                    )
                )
                result["camera_losses"] = tuple(
                    float(np.average(values, weights=horizon_weights))
                    for values in zip(
                        *[
                            horizon_camera_losses[h]
                            for h in horizons
                        ]
                    )
                )
        else:
            # Missing exact-frame evidence is never admissible. Keep finite
            # endpoint values for diagnostics and let the rejection helper add
            # an explicit incomplete-horizon reason.
            result["visual_loss"] = result["endpoint_visual_loss"]
            result["camera_losses"] = result["endpoint_camera_losses"]
        return result

    def _run_visual_residual_persistence_shadow(
        self, rollout_state: object
    ) -> dict:
        """Relax one state for H=1/3/5 physics steps without visual feedback."""
        assert self.visual_residual_mapper is not None
        assert self.environment.frames is not None
        sim = self.environment.sim
        sim.copy_embodied_gaussian_rollout_state(rollout_state)
        projector = sim.triangle_skin_contact_projector
        visual_losses: dict[int, float] = {}
        camera_losses: dict[int, tuple[float, ...]] = {}
        for physics_step in range(
            1, max(VISUAL_RESIDUAL_PERSISTENCE_HORIZONS) + 1
        ):
            if projector is not None:
                projector.freeze_persistent_grip_state_machine()
            self.environment.step(compute_visual_forces=False)
            self.apply_current_psm_pose()
            if projector is not None:
                projector.freeze_persistent_grip_state_machine()
            if physics_step not in VISUAL_RESIDUAL_PERSISTENCE_HORIZONS:
                continue
            sim.update_gaussian_transforms()
            alignment = sim.evaluate_visual_tissue_alignment(
                self.visual_residual_mapper,
                self.environment.frames,
                observations_are_bgr=True,
            )
            visual_losses[physics_step] = float(alignment.loss)
            camera_losses[physics_step] = tuple(alignment.camera_losses)
        return {
            "visual_losses": visual_losses,
            "camera_losses": camera_losses,
        }

    @staticmethod
    def _visual_residual_persistence_rejection_reasons(
        baseline: dict,
        candidate: dict,
        *,
        allow_h1_noise: bool = False,
    ) -> tuple[list[str], dict[int, float | None]]:
        """Check whether a correction survives short no-vision relaxation.

        The legacy profiles retain strict improvement at every horizon.  The
        ranked cross-frame profile treats H=1 as a noisy solver transient, but
        still requires explicit H=3/H=5 improvement and a positive long-term
        weighted objective before a branch may reach the later-RGB gate.
        """
        baseline_losses = baseline.get("visual_losses", {})
        candidate_losses = candidate.get("visual_losses", {})
        improvements: dict[int, float | None] = {}
        failed_horizons: list[int] = []
        for horizon in VISUAL_RESIDUAL_PERSISTENCE_HORIZONS:
            baseline_loss = baseline_losses.get(horizon)
            candidate_loss = candidate_losses.get(horizon)
            if baseline_loss is None or candidate_loss is None:
                improvements[horizon] = None
                failed_horizons.append(horizon)
                continue
            improvement = float(baseline_loss - candidate_loss)
            improvements[horizon] = improvement
            invalid = not np.isfinite(baseline_loss) or not np.isfinite(
                candidate_loss
            )
            if allow_h1_noise and horizon == 1:
                h1_noise_tolerance = max(
                    1.0e-7, 5.0e-3 * abs(float(baseline_loss))
                )
                failed = invalid or improvement < -h1_noise_tolerance
            else:
                required_improvement = (
                    max(1.0e-7, 1.0e-5 * abs(float(baseline_loss)))
                    if allow_h1_noise
                    else 0.0
                )
                failed = invalid or improvement <= required_improvement
            if failed:
                failed_horizons.append(horizon)
        reasons = []
        if failed_horizons:
            reasons.append(
                "open_loop_persistence_not_improved:H"
                + ",H".join(str(value) for value in failed_horizons)
            )
        if allow_h1_noise and not failed_horizons:
            weights = {1: 0.10, 3: 0.30, 5: 0.60}
            weighted_improvement = sum(
                weights[horizon] * float(improvements[horizon])
                for horizon in VISUAL_RESIDUAL_PERSISTENCE_HORIZONS
            )
            if weighted_improvement <= 1.0e-7:
                reasons.append("open_loop_persistence_weighted_not_improved")
        return reasons, improvements

    def _apply_visual_tissue_residual_with_persistence_gate(
        self,
        result,
        *,
        observation_dt_s: float | None = None,
        dynamic_exclusion_mask: torch.Tensor | None = None,
    ) -> bool:
        """Select a residual observer gain by H=1/3/5 no-vision relaxation.

        ``full_only`` exactly retains the original behavior.  The new
        ``multiscale_hold`` profile treats the visual solve as an innovation,
        not a command: full/half/quarter corrections are each installed in a
        clone, allowed to relax under physics with vision disabled, and the
        lowest-loss branch that improves every horizon is written back.
        """
        assert self.visual_residual_mapper is not None
        assert self.environment.frames is not None
        sim = self.environment.sim
        velocity_gain = (
            VISUAL_RESIDUAL_VELOCITY_CORRECTION_GAIN
            if observation_dt_s is not None
            else 0.0
        )
        self._last_applied_visual_residual_result = None
        self._synchronize_shadow_transaction(sim)
        uncorrected_state = sim.clone_embodied_gaussian_rollout_state()
        gains = tuple(
            float(value)
            for value in getattr(
                self, "visual_residual_gain_candidates", (1.0,)
            )
        )
        uncorrected_positions = None
        if any(gain != 1.0 for gain in gains):
            uncorrected_positions = (
                wp.to_torch(sim.state_0.particle_q).detach().clone()
            )
        baseline = None
        branch_records: list[dict] = []
        try:
            baseline = self._run_visual_residual_persistence_shadow(
                uncorrected_state
            )
            for gain in gains:
                self._synchronize_shadow_transaction(sim)
                sim.copy_embodied_gaussian_rollout_state(uncorrected_state)
                sim.update_gaussian_transforms()
                scaled_result = result
                if gain != 1.0:
                    assert uncorrected_positions is not None
                    scaled_result = replace(
                        result,
                        residual=result.residual * gain,
                        corrected_positions=(
                            uncorrected_positions + result.residual * gain
                        ),
                        maximum_residual_m=(
                            float(result.maximum_residual_m) * gain
                        ),
                        rms_residual_m=float(result.rms_residual_m) * gain,
                    )
                immediate_accepted = sim.apply_visual_tissue_residual(
                    scaled_result,
                    mapper=self.visual_residual_mapper,
                    frames=self.environment.frames,
                    observations_are_bgr=True,
                    observation_dt_s=observation_dt_s,
                    velocity_correction_gain=velocity_gain,
                    maximum_velocity_correction_m_s=(
                        VISUAL_RESIDUAL_MAXIMUM_VELOCITY_CORRECTION_M_S
                    ),
                    dynamic_exclusion_mask=dynamic_exclusion_mask,
                )
                immediate_metrics = dict(
                    sim.last_visual_tissue_residual_metrics or {}
                )
                if not immediate_accepted:
                    branch_records.append(
                        {
                            "gain": gain,
                            "result": scaled_result,
                            "state": None,
                            "shadow": None,
                            "metrics": immediate_metrics,
                            "reasons": [
                                immediate_metrics.get(
                                    "rejection_reason", "immediate_rejection"
                                )
                            ],
                            "improvements": {},
                            "score": float("inf"),
                        }
                    )
                    continue
                corrected_state = sim.clone_embodied_gaussian_rollout_state()
                candidate = self._run_visual_residual_persistence_shadow(
                    corrected_state
                )
                ranked_cross_frame = bool(
                    getattr(
                        self, "visual_residual_gain_profile", "full_only"
                    )
                    == "cross_frame_ranked_hold"
                )
                reasons, improvements = (
                    self._visual_residual_persistence_rejection_reasons(
                        baseline,
                        candidate,
                        allow_h1_noise=ranked_cross_frame,
                    )
                )
                score = float(
                    np.mean(
                        [
                            candidate["visual_losses"][horizon]
                            for horizon in VISUAL_RESIDUAL_PERSISTENCE_HORIZONS
                        ]
                    )
                )
                branch_records.append(
                    {
                        "gain": gain,
                        "result": scaled_result,
                        "state": corrected_state,
                        "shadow": candidate,
                        "metrics": immediate_metrics,
                        "reasons": reasons,
                        "improvements": improvements,
                        "score": score,
                    }
                )
        finally:
            self._synchronize_shadow_transaction(sim)
            sim.copy_embodied_gaussian_rollout_state(uncorrected_state)
            sim.update_gaussian_transforms()
            self._synchronize_shadow_transaction(sim)

        safe_records = [record for record in branch_records if not record["reasons"]]
        selected = min(
            safe_records or branch_records,
            key=lambda record: (record["score"], -record["gain"]),
        )
        accepted = bool(safe_records)
        defer_to_next_training_image = bool(
            getattr(self, "visual_residual_gain_profile", "full_only")
            in {"cross_frame_hold", "cross_frame_ranked_hold"}
        )
        if accepted and selected["state"] is not None:
            self._last_applied_visual_residual_result = selected["result"]
            if defer_to_next_training_image:
                if self._pending_visual_residual_validation is not None:
                    raise RuntimeError(
                        "A cross-frame visual residual is already pending"
                    )
                self._pending_visual_residual_validation = (
                    PendingVisualResidualValidation(
                        result=selected["result"],
                        candidate_state=selected["state"],
                        frame_index=int(self.current_frame_index),
                        selected_gain=float(selected["gain"]),
                        candidate_branches=(
                            tuple(
                                VisualResidualCandidateBranch(
                                    result=record["result"],
                                    candidate_state=record["state"],
                                    selected_gain=float(record["gain"]),
                                    persistence_score=float(record["score"]),
                                    persistence_improvements=dict(
                                        record["improvements"]
                                    ),
                                )
                                for record in sorted(
                                    safe_records,
                                    key=lambda record: (
                                        record["score"],
                                        -record["gain"],
                                    ),
                                )
                            )
                            if getattr(
                                self,
                                "visual_residual_gain_profile",
                                "full_only",
                            )
                            == "cross_frame_ranked_hold"
                            else ()
                        ),
                    )
                )
                # The finally block above restored the uncorrected branch.
                # Keep it live until a later observable RGB frame selects
                # between the two causally advanced states.
            else:
                sim.copy_embodied_gaussian_rollout_state(selected["state"])
                sim.update_gaussian_transforms()
        candidate = selected["shadow"]
        reasons = [] if accepted else list(selected["reasons"])
        improvements = selected["improvements"]
        immediate_metrics = dict(selected["metrics"])
        immediate_metrics.update(
            accepted=accepted,
            rejection_reason="" if accepted else "+".join(reasons),
            persistence_gate_evaluated=True,
            persistence_gate_passed=accepted,
            persistence_horizons_physics_steps=(
                VISUAL_RESIDUAL_PERSISTENCE_HORIZONS
            ),
            persistence_baseline_visual_losses=baseline["visual_losses"],
            persistence_candidate_visual_losses=(
                {} if candidate is None else candidate["visual_losses"]
            ),
            persistence_visual_improvements=improvements,
            persistence_visual_residual_enabled=False,
            persistence_grip_state_machine_frozen=True,
            persistence_gain_profile=getattr(
                self, "visual_residual_gain_profile", "full_only"
            ),
            persistence_selected_gain=float(selected["gain"]),
            cross_frame_validation_required=defer_to_next_training_image,
            cross_frame_validation_pending=bool(
                accepted and defer_to_next_training_image
            ),
            cross_frame_candidate_count=(
                len(safe_records)
                if (
                    accepted
                    and getattr(
                        self, "visual_residual_gain_profile", "full_only"
                    )
                    == "cross_frame_ranked_hold"
                )
                else (1 if accepted and defer_to_next_training_image else 0)
            ),
            persistence_gain_search=tuple(
                {
                    "gain": float(record["gain"]),
                    "passed": not record["reasons"],
                    "score": (
                        None
                        if not np.isfinite(record["score"])
                        else float(record["score"])
                    ),
                    "reasons": tuple(str(value) for value in record["reasons"]),
                    "improvements": record["improvements"],
                }
                for record in branch_records
            ),
        )
        sim.last_visual_tissue_residual_metrics = immediate_metrics
        return accepted

    @staticmethod
    def _cross_frame_visual_rejection_reasons(
        baseline: dict, candidate: dict
    ) -> tuple[list[str], float]:
        """Gate a delayed visual correction using a later training image."""
        baseline_loss = float(baseline["visual_loss"])
        candidate_loss = float(candidate["visual_loss"])
        improvement = baseline_loss - candidate_loss
        reasons: list[str] = []
        required_improvement = max(1.0e-7, 1.0e-5 * abs(baseline_loss))
        if (
            not np.isfinite(baseline_loss)
            or not np.isfinite(candidate_loss)
            or improvement <= required_improvement
        ):
            reasons.append("next_training_image_not_improved")
        for camera_index, (baseline_camera, candidate_camera) in enumerate(
            zip(baseline.get("camera_losses", ()), candidate.get("camera_losses", ()))
        ):
            tolerance = max(1.0e-6, 5.0e-3 * abs(float(baseline_camera)))
            if (
                not np.isfinite(baseline_camera)
                or not np.isfinite(candidate_camera)
                or float(candidate_camera) > float(baseline_camera) + tolerance
            ):
                reasons.append(f"next_training_camera_{camera_index}_regressed")
        baseline_inverted = int(baseline.get("inverted_tetrahedra", 0))
        candidate_inverted = int(candidate.get("inverted_tetrahedra", 0))
        if candidate_inverted > baseline_inverted:
            reasons.append("cross_frame_new_tetrahedron_inversion")
        baseline_ratio = float(baseline.get("minimum_volume_ratio", 0.0))
        candidate_ratio = float(candidate.get("minimum_volume_ratio", 0.0))
        # An absolute 0.30 floor starved every visual/material observation once
        # contact had already compressed one baseline tet to about 0.01.  Keep
        # the absolute floor while the baseline is healthy; otherwise require
        # non-catastrophic relative preservation and no additional low-volume
        # tetrahedra below.  New inversions remain an unconditional veto.
        allowed_minimum_ratio = (
            VISUAL_CROSS_FRAME_MINIMUM_VOLUME_RATIO
            if baseline_ratio >= VISUAL_CROSS_FRAME_MINIMUM_VOLUME_RATIO
            else max(0.0, baseline_ratio * 0.98)
        )
        if (
            not np.isfinite(candidate_ratio)
            or candidate_ratio < allowed_minimum_ratio
        ):
            reasons.append("cross_frame_minimum_volume_unsafe")
        baseline_below_floor = int(
            baseline.get("tetrahedra_below_volume_floor", 0)
        )
        candidate_below_floor = int(
            candidate.get("tetrahedra_below_volume_floor", 0)
        )
        if candidate_below_floor > baseline_below_floor:
            reasons.append("cross_frame_added_low_volume_tetrahedra")
        baseline_p01 = float(
            baseline.get("volume_ratio_p01", baseline_ratio)
        )
        candidate_p01 = float(
            candidate.get("volume_ratio_p01", candidate_ratio)
        )
        p01_floor = (
            VISUAL_CROSS_FRAME_MINIMUM_VOLUME_RATIO
            if baseline_p01 >= VISUAL_CROSS_FRAME_MINIMUM_VOLUME_RATIO
            else max(0.0, baseline_p01 * 0.90)
        )
        allowed_p01 = max(
            p01_floor,
            baseline_p01 - VISUAL_CROSS_FRAME_P01_ABSOLUTE_DROP,
            baseline_p01
            * (1.0 - VISUAL_CROSS_FRAME_P01_RELATIVE_DROP),
        )
        if not np.isfinite(candidate_p01) or candidate_p01 < allowed_p01:
            reasons.append("cross_frame_volume_distribution_regressed")
        baseline_weighted_mean = float(
            baseline.get("volume_weighted_mean_ratio", 1.0)
        )
        candidate_weighted_mean = float(
            candidate.get("volume_weighted_mean_ratio", 1.0)
        )
        allowed_weighted_mean = baseline_weighted_mean - max(
            VISUAL_CROSS_FRAME_WEIGHTED_MEAN_ABSOLUTE_DROP,
            VISUAL_CROSS_FRAME_WEIGHTED_MEAN_RELATIVE_DROP
            * abs(baseline_weighted_mean),
        )
        if (
            not np.isfinite(candidate_weighted_mean)
            or candidate_weighted_mean < allowed_weighted_mean
        ):
            reasons.append("cross_frame_total_volume_regressed")
        baseline_penetration = float(
            baseline.get("maximum_penetration_m", 0.0)
        )
        candidate_penetration = float(
            candidate.get("maximum_penetration_m", 0.0)
        )
        if candidate_penetration > max(
            STIFFNESS_MAXIMUM_PENETRATION_M,
            baseline_penetration + 1.0e-4,
        ):
            reasons.append("cross_frame_penetration_regressed")
        baseline_anchor = float(baseline.get("anchor_error_maximum_m", 0.0))
        candidate_anchor = float(candidate.get("anchor_error_maximum_m", 0.0))
        if candidate_anchor > baseline_anchor + 2.0e-4:
            reasons.append("cross_frame_anchor_regressed")
        return reasons, float(improvement)

    def _current_visual_cross_frame_metrics(self) -> dict:
        assert self.visual_residual_mapper is not None
        assert self.environment.frames is not None
        sim = self.environment.sim
        sim.update_gaussian_transforms()
        alignment = sim.evaluate_visual_tissue_alignment(
            self.visual_residual_mapper,
            self.environment.frames,
            observations_are_bgr=True,
        )
        positions = wp.to_torch(sim.state_0.particle_q)
        physical = self.visual_residual_mapper.physical_quality_metrics(
            positions
        )
        contact = sim.triangle_skin_contact_metrics() or {}
        return {
            "visual_loss": float(alignment.loss),
            "camera_losses": tuple(alignment.camera_losses),
            **physical,
            "maximum_penetration_m": float(
                contact.get("maximum_penetration_m", 0.0)
            ),
            "anchor_error_maximum_m": float(
                contact.get(
                    "persistent_grip_anchor_error_maximum_m", 0.0
                )
            ),
        }

    def _validate_pending_visual_residual_cross_frame(
        self,
    ) -> tuple[dict | None, PendingVisualResidualValidation | None]:
        """Select baseline/corrected state on the next observable RGB frame.

        The live simulator advances the uncorrected branch.  This method
        advances the corrected proposal through the exact same tool commands
        in a shadow, compares both against the current *training* RGB image,
        then installs only the winning state.  No held-out image, depth, point
        cloud or manual trajectory is accessed.
        """
        pending = self._pending_visual_residual_validation
        if pending is None:
            return None, None
        if int(self.current_frame_index) <= int(pending.frame_index):
            return None, None
        if not self._benchmark_observation_enabled:
            return None, None
        if not pending.commands:
            return None, None
        assert self.visual_residual_mapper is not None
        assert self.environment.frames is not None
        sim = self.environment.sim
        self._synchronize_shadow_transaction(sim)
        live_state = sim.clone_embodied_gaussian_rollout_state()
        live_frame_index = int(self.current_frame_index)
        live_timestep = float(self.current_timestep)
        live_state_index = int(self._last_state_index)
        baseline = self._current_visual_cross_frame_metrics()
        branches = pending.candidate_branches or (
            VisualResidualCandidateBranch(
                result=pending.result,
                candidate_state=pending.candidate_state,
                selected_gain=float(pending.selected_gain),
                persistence_score=float("inf"),
                persistence_improvements={},
            ),
        )
        branch_records: list[dict] = []
        try:
            for branch in branches:
                self._synchronize_shadow_transaction(sim)
                sim.copy_embodied_gaussian_rollout_state(
                    branch.candidate_state
                )
                for command in pending.commands:
                    self._apply_stiffness_tool_command(command)
                    self.environment.step(compute_visual_forces=False)
                    self.apply_current_psm_pose()
                self.dataset_manager.update_frames(live_timestep)
                self.current_frame_index = live_frame_index
                self.current_timestep = live_timestep
                self._last_state_index = live_state_index
                self.apply_current_psm_pose()
                candidate = self._current_visual_cross_frame_metrics()
                self._synchronize_shadow_transaction(sim)
                candidate_state = (
                    sim.clone_embodied_gaussian_rollout_state()
                )
                reasons, improvement = (
                    self._cross_frame_visual_rejection_reasons(
                        baseline, candidate
                    )
                )
                branch_records.append(
                    {
                        "branch": branch,
                        "candidate": candidate,
                        "candidate_state": candidate_state,
                        "reasons": reasons,
                        "improvement": float(improvement),
                    }
                )
        finally:
            self._synchronize_shadow_transaction(sim)
            sim.copy_embodied_gaussian_rollout_state(live_state)
            self.current_frame_index = live_frame_index
            self.current_timestep = live_timestep
            self._last_state_index = live_state_index
            self.dataset_manager.update_frames(live_timestep)
            self.apply_current_psm_pose()
            sim.update_gaussian_transforms()
            self._synchronize_shadow_transaction(sim)

        def branch_rank(record: dict) -> tuple[float, float, float]:
            persistence_score = float(
                record["branch"].persistence_score
            )
            if not np.isfinite(persistence_score):
                persistence_score = float("inf")
            return (
                float(record["candidate"]["visual_loss"]),
                persistence_score,
                -float(record["branch"].selected_gain),
            )

        safe_records = [
            record for record in branch_records if not record["reasons"]
        ]
        selected_record = min(
            safe_records or branch_records,
            key=branch_rank,
        )
        selected_branch = selected_record["branch"]
        candidate = selected_record["candidate"]
        candidate_state = selected_record["candidate_state"]
        reasons = list(selected_record["reasons"])
        improvement = float(selected_record["improvement"])
        accepted = bool(safe_records)
        pending.result = selected_branch.result
        pending.candidate_state = selected_branch.candidate_state
        pending.selected_gain = float(selected_branch.selected_gain)
        if pending.quality_valid_mask is not None:
            pending.quality_valid_mask = stiffness_local_quality_mask(
                self.visual_residual_mapper,
                (
                    pending.result.corrected_positions
                    - pending.result.residual
                ),
                pending.result.corrected_positions,
                STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO,
            ).detach().clone()
        if accepted:
            sim.copy_embodied_gaussian_rollout_state(candidate_state)
            self.current_frame_index = live_frame_index
            self.current_timestep = live_timestep
            self._last_state_index = live_state_index
            self.apply_current_psm_pose()
            sim.update_gaussian_transforms()
            self._previous_visual_residual = (
                pending.result.residual.detach().clone()
            )
        else:
            self._previous_visual_residual = None
        metrics = {
            "accepted": accepted,
            "start_frame_index": int(pending.frame_index),
            "validation_frame_index": live_frame_index,
            "frame_gap": live_frame_index - int(pending.frame_index),
            "selected_gain": float(pending.selected_gain),
            "baseline_visual_loss": baseline["visual_loss"],
            "candidate_visual_loss": candidate["visual_loss"],
            "visual_improvement": improvement,
            "baseline_camera_losses": baseline["camera_losses"],
            "candidate_camera_losses": candidate["camera_losses"],
            "baseline_physical_quality": {
                key: value
                for key, value in baseline.items()
                if key not in {"visual_loss", "camera_losses"}
            },
            "candidate_physical_quality": {
                key: value
                for key, value in candidate.items()
                if key not in {"visual_loss", "camera_losses"}
            },
            "rejection_reasons": tuple(reasons),
            "candidate_count": len(branch_records),
            "candidate_search": tuple(
                {
                    "gain": float(record["branch"].selected_gain),
                    "persistence_score": (
                        None
                        if not np.isfinite(
                            float(record["branch"].persistence_score)
                        )
                        else float(record["branch"].persistence_score)
                    ),
                    "visual_loss": float(
                        record["candidate"]["visual_loss"]
                    ),
                    "visual_improvement": float(record["improvement"]),
                    "accepted": not record["reasons"],
                    "rejection_reasons": tuple(record["reasons"]),
                }
                for record in branch_records
            ),
            "uses_training_rgb_only": True,
            "uses_depth_or_point_cloud": False,
            "uses_manual_ground_truth": False,
        }
        self._pending_visual_residual_validation = None
        self._last_cross_frame_visual_validation = metrics
        if self.stiffness_metrics_recorder is not None:
            self.stiffness_metrics_recorder.record(
                event="visual_cross_frame_validation",
                frame_index=live_frame_index,
                timestamp_s=live_timestep,
                phase=self._current_action_phase,
                image=metrics,
                force_summary=True,
            )
        return metrics, pending if accepted else None

    def _consume_confirmed_visual_stiffness_evidence(
        self,
        *,
        applied_result,
        quality_valid_mask: torch.Tensor,
        supervision_valid_mask: torch.Tensor,
        control_exclusion_mask: torch.Tensor,
        stiffness_gate_paused: bool,
        stiffness_jaw_speed: float,
        stiffness_grip_active: bool,
        stiffness_contact_count: int,
        accepted_positions: torch.Tensor,
        stiffness_metrics: dict | None,
    ) -> dict | None:
        """Feed only a genuinely installed visual innovation to system ID."""
        if self.stiffness_updater is None:
            return stiffness_metrics
        if (
            STIFFNESS_ADMISSION_MODE == "direct_online"
            and stiffness_gate_paused
        ):
            self._direct_stiffness_observable_window_count = 0
            self._direct_stiffness_last_observable_frame_index = -1
            self.stiffness_updater.invalidate_signal_history()
        if (
            STIFFNESS_ADMISSION_MODE == "causal_fixed_lag"
            and int(self.current_frame_index)
            < STIFFNESS_CAUSAL_WARMUP_END_FRAME
        ):
            # Do not accumulate early residual/strain evidence.  The warmup is
            # a clean identification boundary, not merely a commit cooldown.
            # Keep the ordinary live material summary available during the
            # identification warmup.  The realtime logger and diagnostics use
            # these fields even though evidence accumulation is intentionally
            # disabled before the causal boundary.
            stiffness_metrics = dict(
                stiffness_metrics
                or getattr(self.stiffness_updater, "last_metrics", None)
                or {}
            )
            distance_stiffness = getattr(
                self.stiffness_updater, "distance_stiffness", None
            )
            shape_stiffness = getattr(
                self.stiffness_updater, "shape_stiffness", None
            )
            if distance_stiffness is not None:
                stiffness_metrics.update(
                    distance_minimum=float(distance_stiffness.min().item()),
                    distance_median=float(distance_stiffness.median().item()),
                    distance_maximum=float(distance_stiffness.max().item()),
                )
            if shape_stiffness is not None:
                stiffness_metrics.update(
                    shape_minimum=float(shape_stiffness.min().item()),
                    shape_median=float(shape_stiffness.median().item()),
                    shape_maximum=float(shape_stiffness.max().item()),
                )
            stiffness_metrics.update({
                "status": "calibration_warmup",
                "calibration_warmup_end_frame": (
                    STIFFNESS_CAUSAL_WARMUP_END_FRAME
                ),
                "calibration_warmup_frames_remaining": (
                    STIFFNESS_CAUSAL_WARMUP_END_FRAME
                    - int(self.current_frame_index)
                ),
                "calibration_trial_count": self._stiffness_trial_count,
                "evidence_count": self.stiffness_updater.evidence_count,
                "visual_residual_continues": True,
            })
            stiffness_metrics.setdefault("quality_masked_particles", 0)
            self._maybe_store_stiffness_history(
                accepted_positions=accepted_positions,
                jaw_speed_rad_s=stiffness_jaw_speed,
                gate_paused=stiffness_gate_paused,
            )
            return stiffness_metrics
        physical_prediction = (
            applied_result.corrected_positions - applied_result.residual
        )
        if not stiffness_gate_paused:
            evidence = self.stiffness_updater.accumulate_evidence(
                physical_prediction=physical_prediction,
                accepted_residual=applied_result.residual,
                quality_valid_mask=quality_valid_mask,
                supervision_valid_mask=supervision_valid_mask,
                control_exclusion_mask=control_exclusion_mask,
            )
            direct_online = STIFFNESS_ADMISSION_MODE == "direct_online"
            robust_budget_exhausted = bool(
                STIFFNESS_CANDIDATE_PROFILE
                == "robust_hierarchical_system_id"
                and self.stiffness_updater.update_count
                >= STIFFNESS_ROBUST_MAXIMUM_COMMITS
            )
            causal_fixed_lag = STIFFNESS_ADMISSION_MODE in {
                "causal_fixed_lag",
                "relaxed_h135",
            }
            # Both causal H1/H3/H5 policies identify material only after the
            # tool is in contact and edge strain is spatially observable.  The
            # previous relaxed path admitted early no-contact proposals whose
            # tiny shadow differences were numerical noise.
            strict_observability = STIFFNESS_ADMISSION_MODE in {
                "causal_fixed_lag",
                "relaxed_h135",
            }
            if direct_online:
                if self._pending_stiffness_validation is not None:
                    raise RuntimeError(
                        "direct_online cannot have a pending stiffness shadow"
                    )
                strain_active_particles = int(
                    evidence.metrics["strain_signal_active_particles"]
                )
                (
                    self._direct_stiffness_observable_window_count,
                    self._direct_stiffness_last_observable_frame_index,
                    direct_observability_ready,
                    direct_observability_reason,
                ) = advance_direct_stiffness_observability(
                    previous_count=(
                        self._direct_stiffness_observable_window_count
                    ),
                    previous_frame_index=(
                        self._direct_stiffness_last_observable_frame_index
                    ),
                    current_frame_index=int(self.current_frame_index),
                    contact_count=int(stiffness_contact_count),
                    strain_active_particles=strain_active_particles,
                )
                direct_observability_metrics = {
                    "observability_count_gates_enabled": True,
                    "observability_contact_required": True,
                    "observability_contact_count": int(
                        stiffness_contact_count
                    ),
                    "observability_minimum_contact_count": (
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT
                    ),
                    "observability_strain_active_particles": (
                        strain_active_particles
                    ),
                    "observability_minimum_strain_particles": (
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES
                    ),
                    "observability_consecutive_strain_windows": (
                        self._direct_stiffness_observable_window_count
                    ),
                    "observability_required_consecutive_strain_windows": (
                        STIFFNESS_DIRECT_MINIMUM_OBSERVABLE_WINDOWS
                    ),
                    "observability_last_independent_frame_index": (
                        self._direct_stiffness_last_observable_frame_index
                    ),
                    "observability_ready": bool(
                        direct_observability_ready
                    ),
                    "observability_reason": direct_observability_reason,
                    "two_window_strain_observability_enabled": True,
                }
                if not direct_observability_ready:
                    # Contact/strain dropouts invalidate the EMA that would
                    # otherwise leak pre-contact evidence into the next valid
                    # pair of windows.  The first valid window is retained so
                    # the second independent window can confirm persistence.
                    history_invalid = direct_observability_reason in {
                        "contact_below_minimum",
                        "edge_strain_below_minimum",
                    }
                    if history_invalid:
                        self.stiffness_updater.invalidate_signal_history()
                    stiffness_metrics = dict(evidence.metrics)
                    stiffness_metrics.update(
                        status=(
                            "material_observability_warmup"
                            if self._direct_stiffness_observable_window_count
                            else "material_unobservable"
                        ),
                        validation_status="not_admitted",
                        validation_stage="contact_strain_observability",
                        admission_mode="direct_online",
                        candidate_search_trial_count=0,
                        discrete_candidate_search_enabled=False,
                        shadow_validation_enabled=False,
                        two_window_confirmation_enabled=False,
                        long_horizon_commit_gate_enabled=False,
                        commit_count_limit=None,
                        minimum_commit_interval_frames=0,
                        ema_history_cleared=int(history_invalid),
                        local_minimum_volume_ratio=(
                            STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO
                        ),
                        causal_trial_count=self._stiffness_trial_count,
                        visual_residual_continues=True,
                        **direct_observability_metrics,
                    )
                else:
                    alternating_axis = (
                        STIFFNESS_CANDIDATE_PROFILE
                        == "direct_alternating_gradient"
                    )
                    if alternating_axis:
                        selected_material_axis = (
                            "distance"
                            if self.stiffness_updater.update_count % 2 == 0
                            else "shape"
                        )
                    else:
                        selected_material_axis = "joint"
                    candidate = self.stiffness_updater.propose_from_evidence(
                        evidence,
                        material_axis=selected_material_axis,
                    )
                    candidate_step = torch.maximum(
                        candidate.distance_log_step.abs(),
                        candidate.shape_log_step.abs(),
                    )
                    self._stiffness_trial_count += 1
                    if float(candidate_step.max().item()) <= 1.0e-10:
                        stiffness_metrics = self.stiffness_updater.reject(
                            "no_active_material_signal", candidate
                        )
                    else:
                        stiffness_metrics = self.stiffness_updater.commit(
                            candidate
                        )
                        self._last_stiffness_commit_frame_index = int(
                            self.current_frame_index
                        )
                    stiffness_metrics.update(
                        validation_status=stiffness_metrics["status"],
                        validation_stage="immediate_online_commit",
                        admission_mode="direct_online",
                        candidate_search_mode=(
                            "successful_commit_alternating_single_axis_gradient"
                            if alternating_axis
                            else "dual_independent_continuous_gradient"
                        ),
                        selected_material_axis=selected_material_axis,
                        candidate_search_trial_count=1,
                        discrete_candidate_search_enabled=False,
                        shadow_validation_enabled=False,
                        two_window_confirmation_enabled=False,
                        long_horizon_commit_gate_enabled=False,
                        commit_count_limit=None,
                        minimum_commit_interval_frames=0,
                        local_minimum_volume_ratio=(
                            STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO
                        ),
                        causal_trial_count=self._stiffness_trial_count,
                        visual_residual_continues=True,
                        **direct_observability_metrics,
                    )
            elif (
                self._pending_stiffness_validation is None
                and robust_budget_exhausted
            ):
                stiffness_metrics = dict(evidence.metrics)
                stiffness_metrics.update(
                    status="calibration_locked",
                    calibration_commit_budget=(
                        STIFFNESS_ROBUST_MAXIMUM_COMMITS
                    ),
                    calibration_trial_budget=None,
                    calibration_trial_count=(
                        self._stiffness_trial_count
                    ),
                    visual_residual_continues=True,
                )
            elif (
                self._pending_stiffness_validation is None
                and strict_observability
                and (
                    int(evidence.metrics["active_particles"])
                    < STIFFNESS_EVENT_DRIVEN_MINIMUM_ACTIVE_PARTICLES
                    or int(evidence.metrics["strain_signal_active_particles"])
                    < STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES
                    or int(evidence.metrics["ema_active_particles"])
                    < STIFFNESS_EVENT_DRIVEN_MINIMUM_EMA_PARTICLES
                    or int(stiffness_contact_count)
                    < STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT
                )
            ):
                stiffness_metrics = dict(evidence.metrics)
                stiffness_metrics.update(
                    status="material_unobservable",
                    observability_minimum_active_particles=(
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_ACTIVE_PARTICLES
                    ),
                    observability_minimum_strain_particles=(
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES
                    ),
                    observability_minimum_ema_particles=(
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_EMA_PARTICLES
                    ),
                    observability_contact_count=int(
                        stiffness_contact_count
                    ),
                    observability_minimum_contact_count=(
                        STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT
                    ),
                    visual_residual_continues=True,
                )
            elif self._pending_stiffness_validation is None:
                alternating_axis = (
                    STIFFNESS_CANDIDATE_PROFILE
                    == "direct_alternating_gradient"
                )
                selected_material_axis = (
                    "distance"
                    if alternating_axis and self._stiffness_trial_count % 2 == 0
                    else "shape"
                    if alternating_axis
                    else "joint"
                )
                candidate = self.stiffness_updater.propose_from_evidence(
                    evidence,
                    material_axis=selected_material_axis,
                )
                candidate.metrics.update(
                    selected_material_axis=selected_material_axis,
                    candidate_search_mode=(
                        "trial_alternating_single_axis_gradient"
                        if alternating_axis
                        else "single_continuous_proposal"
                    ),
                )
                candidate_step = torch.maximum(
                    candidate.distance_log_step.abs(),
                    candidate.shape_log_step.abs(),
                )
                if float(candidate_step.max().item()) == 0.0:
                    stiffness_metrics = self.stiffness_updater.reject(
                        "no_active_material_signal", candidate
                    )
                else:
                    if causal_fixed_lag:
                        self._last_stiffness_trial_frame_index = int(
                            self.current_frame_index
                        )
                        self._stiffness_trial_count += 1
                    validation_frames = (
                        self._stiffness_training_validation_frames(
                            int(self.current_frame_index),
                            STIFFNESS_ADMISSION_HORIZONS,
                        )
                    )
                    if causal_fixed_lag:
                        # Causal admission uses only H1/H5/H10 below. Avoid the
                        # legacy rest-history proxy, which biased against valid
                        # hardening updates.
                        history_baseline = 0.0
                        history_candidate = 0.0
                        history_safe = True
                    else:
                        (
                            history_baseline,
                            history_candidate,
                            history_safe,
                        ) = self._evaluate_stiffness_history(candidate)
                    candidate.metrics.update(
                        history_baseline_rms_m=history_baseline,
                        history_candidate_rms_m=history_candidate,
                    )
                    if len(validation_frames) != len(
                        STIFFNESS_ADMISSION_HORIZONS
                    ):
                        stiffness_metrics = self.stiffness_updater.reject(
                            "future_training_horizons_unavailable", candidate
                        )
                    elif not history_safe and not causal_fixed_lag:
                        stiffness_metrics = self.stiffness_updater.reject(
                            "history_regression", candidate
                        )
                    else:
                        self._pending_stiffness_validation = (
                            PendingStiffnessValidation(
                                candidate=candidate,
                                rollout_state=(
                                    self.environment.sim
                                    .clone_embodied_gaussian_rollout_state()
                                ),
                                frame_index=int(self.current_frame_index),
                                grip_active=stiffness_grip_active,
                                history_baseline_rms_m=history_baseline,
                                history_candidate_rms_m=history_candidate,
                                proposal_phase=str(
                                    self._current_action_phase
                                ),
                                previous_residual=(
                                    applied_result.residual.detach().clone()
                                ),
                                validation_frame_indices=(
                                    validation_frames
                                ),
                            )
                        )
                        stiffness_metrics = candidate.metrics
            else:
                stiffness_metrics = dict(
                    stiffness_metrics
                    or self._pending_stiffness_validation.candidate.metrics
                )
                stiffness_metrics.update(
                    evidence_accumulated_while_pending=1,
                    evidence_count=evidence.metrics["evidence_count"],
                    latest_evidence_active_particles=(
                        evidence.metrics["active_particles"]
                    ),
                    latest_evidence_ema_active_particles=(
                        evidence.metrics["ema_active_particles"]
                    ),
                    latest_evidence_maximum_log_step=(
                        evidence.metrics["maximum_log_step"]
                    ),
                )
        self._maybe_store_stiffness_history(
            accepted_positions=accepted_positions,
            jaw_speed_rad_s=stiffness_jaw_speed,
            gate_paused=stiffness_gate_paused,
        )
        return stiffness_metrics

    @staticmethod
    def _stiffness_field_summary(
        values: torch.Tensor | None,
    ) -> dict[str, float]:
        if values is None or values.numel() == 0:
            return {}
        values = values.detach().float()
        return {
            "minimum": float(values.min().item()),
            "median": float(values.median().item()),
            "maximum": float(values.max().item()),
            "mean": float(values.mean().item()),
        }

    def _start_committed_stiffness_evaluation(
        self,
        pending: PendingStiffnessValidation,
    ) -> None:
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        evaluation = CommittedStiffnessEvaluation(
            evaluation_id=self._next_stiffness_evaluation_id,
            candidate=pending.candidate,
            rollout_state=pending.rollout_state,
            start_frame_index=pending.frame_index,
            start_phase=(
                pending.commands[0].phase
                if pending.commands
                else self._current_action_phase
            ),
            commands=list(pending.commands),
            pending_horizons=set(self.stiffness_evaluation_horizons),
            previous_residual=(
                None
                if pending.previous_residual is None
                else pending.previous_residual.detach().clone()
            ),
        )
        self._next_stiffness_evaluation_id += 1
        self._active_stiffness_evaluations.append(evaluation)
        recorder.record(
            event="open_loop_started",
            frame_index=self.current_frame_index,
            timestamp_s=self.current_timestep,
            phase=self._current_action_phase,
            material={
                "evaluation_id": evaluation.evaluation_id,
                "candidate_metrics": pending.candidate.metrics,
            },
            prediction={
                "start_frame_index": pending.frame_index,
                "horizons": sorted(evaluation.pending_horizons),
                "protocols": ("material_isolation", "end_to_end"),
            },
            details={"epoch": self._stiffness_evaluation_epoch},
            force_summary=True,
        )

    @staticmethod
    def _serializable_shadow_metrics(result: dict) -> dict:
        return {
            key: value
            for key, value in result.items()
            if key
            not in {
                "particle_positions",
                "rollout_state",
                "rollout_previous_residual",
            }
        }

    def _evaluation_phase(
        self,
        evaluation: CommittedStiffnessEvaluation,
        commands: list[StiffnessToolCommand],
    ) -> str:
        if commands:
            return commands[-1].phase
        return evaluation.start_phase

    def _record_unavailable_stiffness_horizon(
        self,
        evaluation: CommittedStiffnessEvaluation,
        horizon: int,
        reason: str,
    ) -> None:
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        recorder.record(
            event="open_loop_unavailable",
            frame_index=self.current_frame_index,
            timestamp_s=self.current_timestep,
            phase=self._current_action_phase,
            prediction={
                "evaluation_id": evaluation.evaluation_id,
                "start_frame_index": evaluation.start_frame_index,
                "horizon_frames": horizon,
            },
            details={"reason": reason},
            force_summary=True,
        )

    def _evaluate_committed_stiffness_horizon(
        self,
        evaluation: CommittedStiffnessEvaluation,
        horizon: int,
    ) -> None:
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        target_frame_index = evaluation.start_frame_index + horizon
        if target_frame_index >= len(self.playback_timestamps):
            self._record_unavailable_stiffness_horizon(
                evaluation, horizon, "target_after_dataset_end"
            )
            return
        commands = [
            command
            for command in evaluation.commands
            if command.frame_index <= target_frame_index
        ]
        if not commands:
            self._record_unavailable_stiffness_horizon(
                evaluation, horizon, "no_recorded_tool_commands"
            )
            return

        sim = self.environment.sim
        live = sim.clone_embodied_gaussian_rollout_state()
        controller_state = (
            self.current_frame_index,
            self.current_timestep,
            self._last_state_index,
            self._last_q_full.detach().clone(),
        )
        observation_timestep = self.current_timestep
        target_timestep = float(
            self.playback_timestamps[target_frame_index]
        )
        reference_positions = wp.to_torch(
            evaluation.rollout_state.embodied_state.physics_state.particle_q
        ).detach()
        old_distance = (
            evaluation.rollout_state.auxiliary_state
            .paper_distance_stiffness
        )
        old_shape = (
            evaluation.rollout_state.auxiliary_state
            .paper_shape_stiffness
        )
        material_summary = {
            "evaluation_id": evaluation.evaluation_id,
            "old_distance": self._stiffness_field_summary(old_distance),
            "new_distance": self._stiffness_field_summary(
                evaluation.candidate.distance_stiffness
            ),
            "old_shape": self._stiffness_field_summary(old_shape),
            "new_shape": self._stiffness_field_summary(
                evaluation.candidate.shape_stiffness
            ),
        }
        if old_distance is not None:
            new_distance = evaluation.candidate.distance_stiffness.to(
                device=old_distance.device, dtype=old_distance.dtype
            )
            material_summary["distance_change_rms"] = float(
                torch.sqrt(
                    torch.mean((new_distance - old_distance) ** 2)
                ).item()
            )
        if old_shape is not None:
            new_shape = evaluation.candidate.shape_stiffness.to(
                device=old_shape.device, dtype=old_shape.dtype
            )
            material_summary["shape_change_rms"] = float(
                torch.sqrt(
                    torch.mean((new_shape - old_shape) ** 2)
                ).item()
            )
        phase = self._evaluation_phase(evaluation, commands)
        try:
            self.dataset_manager.update_frames(target_timestep)
            for protocol, freeze_grip in (
                ("material_isolation", True),
                ("end_to_end", False),
            ):
                replay_rgb_residuals = bool(
                    not freeze_grip
                    and getattr(self, "visual_feedback_mode", "residual")
                    == "residual"
                )
                baseline = self._run_stiffness_rollout_shadow(
                    rollout_state=evaluation.rollout_state,
                    commands=commands,
                    candidate=evaluation.candidate,
                    use_candidate=False,
                    freeze_grip_state_machine=freeze_grip,
                    measure_open_loop_residual=True,
                    replay_visual_residuals=replay_rgb_residuals,
                    initial_previous_residual=(
                        evaluation.previous_residual
                        if replay_rgb_residuals
                        else None
                    ),
                    replay_after_frame_index=evaluation.start_frame_index,
                )
                candidate = self._run_stiffness_rollout_shadow(
                    rollout_state=evaluation.rollout_state,
                    commands=commands,
                    candidate=evaluation.candidate,
                    use_candidate=True,
                    freeze_grip_state_machine=freeze_grip,
                    measure_open_loop_residual=True,
                    replay_visual_residuals=replay_rgb_residuals,
                    initial_previous_residual=(
                        evaluation.previous_residual
                        if replay_rgb_residuals
                        else None
                    ),
                    replay_after_frame_index=evaluation.start_frame_index,
                )
                baseline_positions = baseline["particle_positions"]
                candidate_positions = candidate["particle_positions"]
                prediction = {
                    "evaluation_id": evaluation.evaluation_id,
                    "protocol": protocol,
                    "start_frame_index": evaluation.start_frame_index,
                    "target_frame_index": target_frame_index,
                    "horizon_frames": horizon,
                    "rollout_steps": len(commands),
                    "baseline_gap": baseline["visual_loss"],
                    "candidate_gap": candidate["visual_loss"],
                    "gap_improvement": (
                        baseline["visual_loss"]
                        - candidate["visual_loss"]
                    ),
                    "baseline_camera_gaps": baseline["camera_losses"],
                    "candidate_camera_gaps": candidate["camera_losses"],
                    "baseline_displacement_rms_m": (
                        self._mass_weighted_particle_rms(
                            reference_positions, baseline_positions
                        )
                    ),
                    "candidate_displacement_rms_m": (
                        self._mass_weighted_particle_rms(
                            reference_positions, candidate_positions
                        )
                    ),
                    "baseline_candidate_particle_rms_m": (
                        self._mass_weighted_particle_rms(
                            baseline_positions, candidate_positions
                        )
                    ),
                    "baseline_open_loop_residual_rms_m": baseline[
                        "open_loop_residual_rms_m"
                    ],
                    "candidate_open_loop_residual_rms_m": candidate[
                        "open_loop_residual_rms_m"
                    ],
                }
                recorder.record(
                    event="open_loop_prediction",
                    frame_index=target_frame_index,
                    timestamp_s=target_timestep,
                    phase=phase,
                    image={
                        "baseline_gap": baseline["visual_loss"],
                        "candidate_gap": candidate["visual_loss"],
                        "baseline_camera_gaps": baseline[
                            "camera_losses"
                        ],
                        "candidate_camera_gaps": candidate[
                            "camera_losses"
                        ],
                    },
                    physical={
                        "baseline": self._serializable_shadow_metrics(
                            baseline
                        ),
                        "candidate": self._serializable_shadow_metrics(
                            candidate
                        ),
                    },
                    material=material_summary,
                    prediction=prediction,
                    details={"epoch": self._stiffness_evaluation_epoch},
                    force_summary=True,
                )
        finally:
            sim.copy_embodied_gaussian_rollout_state(live)
            (
                self.current_frame_index,
                self.current_timestep,
                self._last_state_index,
                self._last_q_full,
            ) = controller_state
            self.dataset_manager.update_frames(observation_timestep)
            sim.update_gaussian_transforms()

    def _advance_committed_stiffness_evaluations(self) -> None:
        if self.stiffness_metrics_recorder is None:
            return
        completed: list[CommittedStiffnessEvaluation] = []
        for evaluation in list(self._active_stiffness_evaluations):
            for horizon in sorted(evaluation.pending_horizons):
                target = evaluation.start_frame_index + horizon
                if target >= len(self.playback_timestamps):
                    self._evaluate_committed_stiffness_horizon(
                        evaluation, horizon
                    )
                    evaluation.pending_horizons.remove(horizon)
                elif target <= self.current_frame_index:
                    self._evaluate_committed_stiffness_horizon(
                        evaluation, horizon
                    )
                    evaluation.pending_horizons.remove(horizon)
            if not evaluation.pending_horizons:
                completed.append(evaluation)
        for evaluation in completed:
            self._active_stiffness_evaluations.remove(evaluation)
            self.stiffness_metrics_recorder.record(
                event="open_loop_completed",
                frame_index=self.current_frame_index,
                timestamp_s=self.current_timestep,
                phase=self._current_action_phase,
                prediction={"evaluation_id": evaluation.evaluation_id},
                force_summary=True,
            )

    def _record_terminal_stiffness_validation(
        self,
        pending: PendingStiffnessValidation,
        metrics: dict,
        reason: str,
    ) -> None:
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        recorder.record(
            event="stiffness_validation",
            frame_index=self.current_frame_index,
            timestamp_s=self.current_timestep,
            phase=self._current_action_phase,
            physical=self._current_physical_evaluation_metrics(),
            material=self._current_material_evaluation_metrics(metrics),
            prediction={
                "start_frame_index": pending.frame_index,
                "horizon_frames": (
                    self.current_frame_index - pending.frame_index
                ),
                "rollout_steps": len(pending.commands),
                "status": metrics.get("validation_status", metrics["status"]),
            },
            details={"reason": reason},
            force_summary=True,
        )

    def _stiffness_candidate_variant(
        self,
        candidate: PaperStiffnessCandidate,
        *,
        distance_scale: float,
        shape_scale: float,
        variant_label: str,
        scope: str = "full",
    ) -> PaperStiffnessCandidate:
        """Build a signed, independently scaled material candidate."""
        assert self.stiffness_updater is not None
        return self.stiffness_updater.candidate_variant(
            candidate,
            distance_scale=distance_scale,
            shape_scale=shape_scale,
            variant_label=variant_label,
            scope=scope,
        )

    def _scaled_stiffness_candidate(
        self,
        candidate: PaperStiffnessCandidate,
        scale: float,
    ) -> PaperStiffnessCandidate:
        """Compatibility wrapper for a same-sign joint candidate."""
        if scale <= 0.0:
            raise ValueError("Stiffness candidate scale must be positive")
        return self._stiffness_candidate_variant(
            candidate,
            distance_scale=float(scale),
            shape_scale=float(scale),
            variant_label=f"joint_forward_x{float(scale):g}",
            scope="full",
        )

    @staticmethod
    def _stiffness_shadow_rejection_reasons(
        baseline: dict,
        candidate: dict,
        *,
        history_safe: bool = True,
        absolute_margin_scale: float = 1.0,
        admission_horizons: tuple[int, ...] | None = None,
        required_long_horizons: tuple[int, ...] | None = None,
    ) -> tuple[list[str], float, float]:
        admission_horizons = tuple(
            admission_horizons or STIFFNESS_ADMISSION_HORIZONS
        )
        if not 0.0 < absolute_margin_scale <= 1.0:
            raise ValueError("Stiffness margin scale must lie in (0,1]")
        baseline_trajectory_loss = baseline.get(
            "trajectory_tracking_loss_m"
        )
        candidate_trajectory_loss = candidate.get(
            "trajectory_tracking_loss_m"
        )
        trajectory_primary = bool(
            baseline_trajectory_loss is not None
            and candidate_trajectory_loss is not None
            and np.isfinite(float(baseline_trajectory_loss))
            and np.isfinite(float(candidate_trajectory_loss))
        )
        if trajectory_primary:
            baseline_aggregate_loss = float(baseline_trajectory_loss)
            candidate_aggregate_loss = float(candidate_trajectory_loss)
            # relaxed_h135 identifies material from the meaningful H3/H5
            # persistence window.  Do not let bounded H1 solver noise turn the
            # aggregate gate back into an implicit H1 veto.
            if STIFFNESS_ADMISSION_MODE == "relaxed_h135":
                baseline_h35 = baseline.get("horizon_trajectory_losses_m")
                candidate_h35 = candidate.get("horizon_trajectory_losses_m")
                h35_complete = bool(
                    baseline_h35 is not None
                    and candidate_h35 is not None
                    and all(
                        baseline_h35.get(horizon) is not None
                        and candidate_h35.get(horizon) is not None
                        and np.isfinite(float(baseline_h35[horizon]))
                        and np.isfinite(float(candidate_h35[horizon]))
                        for horizon in (3, 5)
                    )
                )
                if h35_complete:
                    baseline_aggregate_loss = float(
                        np.average(
                            [baseline_h35[3], baseline_h35[5]],
                            weights=[3.0, 5.0],
                        )
                    )
                    candidate_aggregate_loss = float(
                        np.average(
                            [candidate_h35[3], candidate_h35[5]],
                            weights=[3.0, 5.0],
                        )
                    )
            improvement = baseline_aggregate_loss - candidate_aggregate_loss
            required_improvement = max(
                STIFFNESS_TRAJECTORY_AGGREGATE_ABSOLUTE_MARGIN_M
                * absolute_margin_scale,
                STIFFNESS_TRAJECTORY_AGGREGATE_RELATIVE_MARGIN
                * abs(baseline_aggregate_loss),
            )
        else:
            improvement = baseline["visual_loss"] - candidate["visual_loss"]
            required_improvement = max(
                STIFFNESS_PREDICTION_ABSOLUTE_MARGIN
                * absolute_margin_scale,
                STIFFNESS_PREDICTION_RELATIVE_MARGIN
                * baseline["visual_loss"],
            )
        rgb_auxiliary_safe = bool(
            not trajectory_primary
            or candidate["visual_loss"]
            <= baseline["visual_loss"]
            + max(
                STIFFNESS_TRAJECTORY_RGB_ABSOLUTE_REGRESSION,
                STIFFNESS_TRAJECTORY_RGB_RELATIVE_REGRESSION
                * abs(float(baseline["visual_loss"])),
            )
        )
        camera_safe = all(
            candidate_loss
            <= baseline_loss
            + max(
                STIFFNESS_CAMERA_ABSOLUTE_REGRESSION,
                STIFFNESS_CAMERA_RELATIVE_REGRESSION * baseline_loss,
            )
            for baseline_loss, candidate_loss in zip(
                baseline["camera_losses"], candidate["camera_losses"]
            )
        )
        allowed_minimum_volume = max(
            baseline["minimum_volume_ratio"]
            - STIFFNESS_MINIMUM_VOLUME_ABSOLUTE_DROP,
            baseline["minimum_volume_ratio"]
            * (1.0 - STIFFNESS_MINIMUM_VOLUME_RELATIVE_DROP),
        )
        volume_safe = (
            candidate["inverted_tetrahedra"]
            <= baseline["inverted_tetrahedra"]
            and candidate["minimum_volume_ratio"]
            >= allowed_minimum_volume
            and int(candidate.get("tetrahedra_below_volume_floor", 0))
            <= int(baseline.get("tetrahedra_below_volume_floor", 0))
        )
        penetration_safe = (
            candidate["maximum_penetration_m"]
            <= baseline["maximum_penetration_m"]
            + STIFFNESS_PENETRATION_TOLERANCE_M
        )
        anchor_safe = (
            candidate["anchor_error_rms_m"]
            <= baseline["anchor_error_rms_m"]
            + STIFFNESS_ANCHOR_ERROR_TOLERANCE_M
            and candidate["anchor_error_maximum_m"]
            <= baseline["anchor_error_maximum_m"]
                + STIFFNESS_ANCHOR_ERROR_TOLERANCE_M
        )
        reasons = []
        horizons_complete = bool(
            baseline.get("validation_horizons_complete", True)
            and candidate.get("validation_horizons_complete", True)
        )
        if not horizons_complete:
            reasons.append("material_isolation_horizons_incomplete")
        else:
            baseline_horizons = baseline.get(
                "horizon_trajectory_losses_m"
                if trajectory_primary
                else "horizon_visual_losses"
            )
            candidate_horizons = candidate.get(
                "horizon_trajectory_losses_m"
                if trajectory_primary
                else "horizon_visual_losses"
            )
            if baseline_horizons is not None or candidate_horizons is not None:
                failed_horizons = []
                long_horizons = tuple(
                    required_long_horizons
                    or tuple(
                        horizon
                        for horizon in (5, 10)
                        if horizon in admission_horizons
                    )
                )
                if not long_horizons:
                    long_horizons = (max(admission_horizons),)
                for horizon in admission_horizons:
                    if (
                        baseline_horizons is None
                        or candidate_horizons is None
                        or baseline_horizons.get(horizon) is None
                        or candidate_horizons.get(horizon) is None
                    ):
                        failed_horizons.append(horizon)
                        continue
                    baseline_horizon = float(baseline_horizons[horizon])
                    candidate_horizon = float(candidate_horizons[horizon])
                    if STIFFNESS_ADMISSION_MODE == "strict_all":
                        failed = candidate_horizon >= baseline_horizon
                    elif (
                        trajectory_primary
                        and STIFFNESS_ADMISSION_MODE == "relaxed_h135"
                        and horizon == 3
                    ):
                        allowed = max(
                            STIFFNESS_TRAJECTORY_H3_ABSOLUTE_REGRESSION_M,
                            STIFFNESS_TRAJECTORY_H3_RELATIVE_REGRESSION
                            * abs(baseline_horizon),
                        )
                        failed = (
                            candidate_horizon > baseline_horizon + allowed
                        )
                    elif horizon in long_horizons:
                        long_required = (
                            max(
                                STIFFNESS_TRAJECTORY_HORIZON_ABSOLUTE_MARGIN_M
                                * absolute_margin_scale,
                                STIFFNESS_TRAJECTORY_HORIZON_RELATIVE_MARGIN
                                * abs(baseline_horizon),
                            )
                            if trajectory_primary
                            else max(
                                STIFFNESS_LONG_HORIZON_ABSOLUTE_MARGIN
                                * absolute_margin_scale,
                                STIFFNESS_LONG_HORIZON_RELATIVE_MARGIN
                                * baseline_horizon,
                            )
                        )
                        failed = (
                            baseline_horizon - candidate_horizon
                            < long_required
                        )
                    else:
                        allowed = (
                            max(
                                STIFFNESS_TRAJECTORY_H1_ABSOLUTE_REGRESSION_M,
                                STIFFNESS_TRAJECTORY_H1_RELATIVE_REGRESSION
                                * baseline_horizon,
                            )
                            if trajectory_primary
                            else max(
                                STIFFNESS_WINDOW_HORIZON_ABSOLUTE_REGRESSION,
                                STIFFNESS_WINDOW_HORIZON_RELATIVE_REGRESSION
                                * baseline_horizon,
                            )
                        )
                        failed = candidate_horizon > baseline_horizon + allowed
                    if failed:
                        failed_horizons.append(horizon)
                if failed_horizons:
                    reasons.append(
                        (
                            "alltracker_trajectory_horizon_"
                            if trajectory_primary
                            else "material_isolation_horizon_"
                        )
                        + (
                            "not_improved:H"
                            if STIFFNESS_ADMISSION_MODE == "strict_all"
                            else "short_regressed_or_long_not_improved:H"
                        )
                        + ",H".join(str(value) for value in failed_horizons)
                    )
        if improvement < required_improvement:
            reasons.append(
                "alltracker_trajectory_not_improved"
                if trajectory_primary
                else "prediction_gap_not_improved"
            )
        if not rgb_auxiliary_safe:
            reasons.append("rgb_auxiliary_regression")
        # The causal objective is already the weighted multi-camera RGB gap.
        # A per-camera veto duplicated that objective and routinely discarded
        # a clear aggregate improvement because one view moved by numerical
        # noise.  Retain it only for legacy admission modes.
        if (
            not camera_safe
            and STIFFNESS_ADMISSION_MODE
            not in {"causal_fixed_lag", "relaxed_h135"}
        ):
            reasons.append("camera_regression")
        if not volume_safe:
            reasons.append("volume_quality_regression")
        if not penetration_safe:
            reasons.append("penetration_regression")
        if not anchor_safe:
            reasons.append("grip_anchor_regression")
        # H5/H10 provide the causal material check. The legacy rest-history
        # RMS was correlated with stiffness itself and could reject a better
        # RGB trajectory simply for being less relaxed. Keep it for old modes.
        if (
            not history_safe
            and STIFFNESS_ADMISSION_MODE
            not in {"causal_fixed_lag", "relaxed_h135"}
        ):
            reasons.append("history_regression")
        return reasons, improvement, required_improvement

    @staticmethod
    def _stiffness_residual_effort_improvement(
        baseline: dict, candidate: dict
    ) -> float | None:
        baseline_effort = baseline.get("open_loop_residual_rms_m")
        candidate_effort = candidate.get("open_loop_residual_rms_m")
        if baseline_effort is None or candidate_effort is None:
            return None
        return float(baseline_effort) - float(candidate_effort)

    @staticmethod
    def _stiffness_candidate_selection_key(
        baseline: dict, record: dict
    ) -> tuple[float | int | str, ...]:
        """Rank admitted candidates by gap, then residual effort as tie-break."""
        effort_improvement = (
            SuperPlaybackControls._stiffness_residual_effort_improvement(
                baseline, record["shadow"]
            )
        )
        if effort_improvement is None:
            effort_improvement = float("-inf")
        return (
            -float(record["improvement"]),
            -float(effort_improvement),
            0 if record["scope"] == "global_material_offset" else 1,
            str(record["variant_label"]),
        )

    @staticmethod
    def _stiffness_record_material_axis(record: dict) -> str:
        """Return the one material family changed by a search record."""
        label = str(record["variant_label"])
        if label.startswith("global_distance_"):
            return "distance"
        if label.startswith("global_shape_"):
            return "shape"
        distance_active = abs(float(record["distance_scale"])) > 1.0e-12
        shape_active = abs(float(record["shape_scale"])) > 1.0e-12
        if distance_active == shape_active:
            raise RuntimeError(
                "Causal stiffness candidate must change exactly one material "
                f"family: {label}"
            )
        return "distance" if distance_active else "shape"

    @staticmethod
    def _stiffness_candidate_margin_scale(
        *, scope: str, scope_fraction: float
    ) -> float:
        """Use support-normalized evidence locally, never for global offsets."""
        if (
            STIFFNESS_ADMISSION_MODE
            in {"causal_fixed_lag", "relaxed_h135"}
            and scope != "global_material_offset"
        ):
            return float(np.clip(scope_fraction, 0.10, 1.0))
        return 1.0

    def _clear_stiffness_global_confirmation(self) -> None:
        self._stiffness_global_confirmation_key = None
        self._stiffness_global_confirmation_count = 0

    def _clear_stiffness_local_confirmations(self) -> None:
        state = getattr(self, "_stiffness_local_confirmation_state", None)
        if state is None:
            self._stiffness_local_confirmation_state = {}
        else:
            state.clear()

    def _apply_causal_stiffness_commit_policy(
        self,
        *,
        baseline: dict,
        records: list[dict],
        proposal_phase: str,
        proposal_frame_index: int | None = None,
    ) -> dict[str, object]:
        """Compatibility hook; causal admission no longer waits for consensus."""
        return {
            "proposal_phase": proposal_phase,
            "confirmation_enabled": False,
            "local_confirmation_state": {},
            "global_confirmation_key": None,
            "global_confirmation_count": 0,
        }

    @staticmethod
    def _stiffness_candidate_scope_fraction(
        candidate: PaperStiffnessCandidate,
    ) -> float:
        """Fraction of legal material actually changed by this candidate.

        A ``full`` local probe means all *currently evidenced* particles, not
        the whole tissue.  Normalizing it by proposal evidence therefore made
        a 20-node update pay the same full-image margin as a 1300-node global
        offset.  Use realized changed nodes over physically legal material;
        explicit global-offset candidates retain the full margin.
        """
        if candidate.metrics.get("candidate_scope") == "global_material_offset":
            return 1.0
        changed = (
            (candidate.distance_log_step.abs() > 1.0e-10)
            | (candidate.shape_log_step.abs() > 1.0e-10)
        )
        changed_particles = int(torch.count_nonzero(changed).item())
        legal_particles = int(
            torch.count_nonzero(candidate.material_valid_mask).item()
        )
        if changed_particles <= 0 or legal_particles <= 0:
            return 1.0
        return float(
            np.clip(changed_particles / legal_particles, 0.10, 1.0)
        )

    @staticmethod
    def _assert_stiffness_candidate_axis_isolation(
        candidate: PaperStiffnessCandidate,
        *,
        verified_distance: torch.Tensor,
        verified_shape: torch.Tensor,
        may_change_distance: bool,
        may_change_shape: bool,
    ) -> None:
        """Fail closed if a nominally single-axis probe changes its peer."""
        if not may_change_distance and not torch.equal(
            candidate.distance_stiffness, verified_distance
        ):
            raise RuntimeError(
                "Distance stiffness changed inside a shape-only candidate"
            )
        if not may_change_shape and not torch.equal(
            candidate.shape_stiffness, verified_shape
        ):
            raise RuntimeError(
                "Shape stiffness changed inside a distance-only candidate"
            )

    def _adopt_validated_stiffness_rollout(self, candidate: dict) -> dict:
        """Install a bounded, already-observed fixed-lag candidate state."""
        sim = self.environment.sim
        current_positions = wp.to_torch(sim.state_0.particle_q).detach()
        candidate_positions = candidate["particle_positions"].to(
            device=current_positions.device,
            dtype=current_positions.dtype,
        )
        displacement = torch.linalg.vector_norm(
            candidate_positions - current_positions, dim=1
        )
        state_rms_m = float(
            torch.sqrt(torch.mean(displacement * displacement)).item()
        )
        state_maximum_m = float(displacement.max().item())
        finite = bool(torch.isfinite(displacement).all().item())
        allowed = bool(
            stiffness_adopt_validated_rollout_enabled()
            and finite
            and state_rms_m <= STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M
            and state_maximum_m
            <= STIFFNESS_MAXIMUM_ADOPTED_STATE_MAXIMUM_M
        )
        if allowed:
            sim.copy_embodied_gaussian_rollout_state(
                candidate["rollout_state"]
            )
            sim.update_gaussian_transforms()
            rollout_previous_residual = candidate[
                "rollout_previous_residual"
            ]
            self._previous_visual_residual = (
                None
                if rollout_previous_residual is None
                else rollout_previous_residual.detach().clone()
            )
        return {
            "validated_rollout_adopted": allowed,
            "validated_rollout_state_rms_m": state_rms_m,
            "validated_rollout_state_maximum_m": state_maximum_m,
            "validated_rollout_state_finite": finite,
            "maximum_adopted_state_rms_m": (
                STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M
            ),
            "maximum_adopted_state_maximum_m": (
                STIFFNESS_MAXIMUM_ADOPTED_STATE_MAXIMUM_M
            ),
        }

    def _validate_direct_residual_gradient_stiffness(
        self,
        pending: PendingStiffnessValidation,
    ) -> dict:
        """Validate one continuous updater proposal without candidate search.

        The residual/edge-strain gradient already determines sign, magnitude,
        material family and spatial support.  We replay exactly that proposal
        against the verified baseline at the configured short horizons. Passing RGB and hard
        physical gates commits immediately: there is no variant contest,
        second-window confirmation, H30/H60 gate, count budget, or cooldown.
        Trajectory mode uses confidence-weighted AllTracker+depth 3D EPE as
        the primary H3/H5 objective and keeps RGB only as a bounded veto.
        """
        updater = self.stiffness_updater
        assert updater is not None
        candidate_proposal = pending.candidate
        alternating_axis = (
            STIFFNESS_CANDIDATE_PROFILE == "direct_alternating_gradient"
        )
        candidate_proposal.metrics.update(
            candidate_variant=(
                "direct_alternating_gradient"
                if alternating_axis
                else "direct_residual_gradient"
            ),
            candidate_scope="direct_evidence",
            candidate_search_mode=(
                "trial_alternating_single_axis_gradient"
                if alternating_axis
                else "single_continuous_proposal"
            ),
        )
        scope_fraction = self._stiffness_candidate_scope_fraction(
            candidate_proposal
        )
        self._synchronize_shadow_transaction(self.environment.sim)
        live = self.environment.sim.clone_embodied_gaussian_rollout_state()
        controller_state = (
            self.current_frame_index,
            self.current_timestep,
            self._last_state_index,
            self._last_q_full.detach().clone(),
        )
        verified_distance = updater.distance_stiffness.detach().clone()
        verified_shape = updater.shape_stiffness.detach().clone()
        try:
            baseline = self._run_stiffness_prediction_shadow(
                pending, use_candidate=False
            )
            updater.restore_verified_stiffness(
                verified_distance, verified_shape
            )
            direct = self._run_stiffness_prediction_shadow(
                pending, use_candidate=True
            )
        finally:
            self._synchronize_shadow_transaction(self.environment.sim)
            self.environment.sim.copy_embodied_gaussian_rollout_state(live)
            updater.restore_verified_stiffness(
                verified_distance, verified_shape
            )
            (
                self.current_frame_index,
                self.current_timestep,
                self._last_state_index,
                self._last_q_full,
            ) = controller_state
            self.dataset_manager.update_frames(self.current_timestep)
            self.environment.sim.update_gaussian_transforms()
            self._synchronize_shadow_transaction(self.environment.sim)

        relaxed_h135 = STIFFNESS_ADMISSION_MODE == "relaxed_h135"
        reasons, improvement, required_improvement = (
            self._stiffness_shadow_rejection_reasons(
                baseline,
                direct,
                history_safe=True,
                absolute_margin_scale=(
                    self._stiffness_candidate_margin_scale(
                        scope="direct_evidence",
                        scope_fraction=scope_fraction,
                    )
                ),
                required_long_horizons=(3, 5) if relaxed_h135 else None,
            )
        )
        if (
            not relaxed_h135
            and (
                int(
                    candidate_proposal.metrics.get(
                        "candidate_scope_particles", 0
                    )
                )
                < STIFFNESS_EVENT_DRIVEN_MINIMUM_SCOPE_PARTICLES
            )
        ):
            reasons.append("material_support_too_small")

        if reasons:
            metrics = updater.reject(
                "+".join(reasons), candidate_proposal
            )
        else:
            metrics = updater.commit(candidate_proposal)
            # Diagnostic only; this value is never consulted for admission.
            self._last_stiffness_commit_frame_index = int(
                self.current_frame_index
            )

        rollout_adoption_metrics = {
            "validated_rollout_adopted": False,
            "validated_rollout_state_rms_m": 0.0,
            "validated_rollout_state_maximum_m": 0.0,
            "validated_rollout_state_finite": True,
            "maximum_adopted_state_rms_m": (
                STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M
            ),
            "maximum_adopted_state_maximum_m": (
                STIFFNESS_MAXIMUM_ADOPTED_STATE_MAXIMUM_M
            ),
        }
        if (
            metrics["status"] == "committed"
            and self.visual_feedback_mode == "residual"
        ):
            rollout_adoption_metrics = (
                self._adopt_validated_stiffness_rollout(direct)
            )

        horizon_improvements = {
            horizon: (
                baseline["horizon_visual_losses"][horizon]
                - direct["horizon_visual_losses"][horizon]
            )
            if baseline["horizon_visual_losses"][horizon] is not None
            and direct["horizon_visual_losses"][horizon] is not None
            else None
            for horizon in STIFFNESS_ADMISSION_HORIZONS
        }
        trajectory_horizon_improvements_m = {
            horizon: (
                baseline["horizon_trajectory_losses_m"][horizon]
                - direct["horizon_trajectory_losses_m"][horizon]
            )
            if baseline["horizon_trajectory_losses_m"].get(horizon) is not None
            and direct["horizon_trajectory_losses_m"].get(horizon) is not None
            else None
            for horizon in STIFFNESS_ADMISSION_HORIZONS
        }
        residual_effort_improvement = (
            self._stiffness_residual_effort_improvement(baseline, direct)
        )
        metrics.update(
            validation_status=metrics["status"],
            validation_stage=(
                (
                    "relaxed_alternating_axis_h1_h3_h5"
                    if alternating_axis
                    else "relaxed_joint_h1_h3_h5"
                )
                if relaxed_h135
                else "direct_h1_h5_h10"
            ),
            candidate_search_mode=(
                "trial_alternating_single_axis_gradient"
                if alternating_axis
                else "single_continuous_proposal"
            ),
            candidate_search_trial_count=1,
            discrete_candidate_search_enabled=False,
            two_window_confirmation_enabled=False,
            long_horizon_commit_gate_enabled=False,
            commit_count_limit=None,
            minimum_commit_interval_frames=0,
            observability_count_gates_enabled=True,
            observability_contact_required=True,
            required_improvement_horizons=(5,) if relaxed_h135 else (5, 10),
            bounded_regression_horizons=(1, 3) if relaxed_h135 else (1,),
            aggregate_improvement_horizons=(
                (3, 5) if relaxed_h135 else (5, 10)
            ),
            prediction_horizon_frames=(
                self.current_frame_index - pending.frame_index
            ),
            prediction_rollout_steps=len(pending.commands),
            validation_horizons=STIFFNESS_ADMISSION_HORIZONS,
            validation_visual_objective=(
                direct.get(
                    "stiffness_admission_objective",
                    stiffness_validation_objective_name(),
                )
            ),
            baseline_visual_loss=baseline["visual_loss"],
            candidate_visual_loss=direct["visual_loss"],
            baseline_horizon_visual_losses=baseline[
                "horizon_visual_losses"
            ],
            candidate_horizon_visual_losses=direct[
                "horizon_visual_losses"
            ],
            horizon_visual_improvements=horizon_improvements,
            baseline_trajectory_tracking_loss_m=baseline.get(
                "trajectory_tracking_loss_m"
            ),
            candidate_trajectory_tracking_loss_m=direct.get(
                "trajectory_tracking_loss_m"
            ),
            baseline_horizon_trajectory_losses_m=baseline.get(
                "horizon_trajectory_losses_m"
            ),
            candidate_horizon_trajectory_losses_m=direct.get(
                "horizon_trajectory_losses_m"
            ),
            horizon_trajectory_improvements_m=(
                trajectory_horizon_improvements_m
            ),
            horizon_trajectory_metrics=direct.get(
                "horizon_trajectory_metrics"
            ),
            prediction_improvement=float(improvement),
            required_prediction_improvement=float(required_improvement),
            baseline_camera_losses=baseline["camera_losses"],
            candidate_camera_losses=direct["camera_losses"],
            baseline_open_loop_residual_rms_m=baseline.get(
                "open_loop_residual_rms_m"
            ),
            candidate_open_loop_residual_rms_m=direct.get(
                "open_loop_residual_rms_m"
            ),
            residual_effort_improvement_m=residual_effort_improvement,
            residual_effort_policy="diagnostic_only_no_candidate_ranking",
            baseline_minimum_volume_ratio=baseline[
                "minimum_volume_ratio"
            ],
            candidate_minimum_volume_ratio=direct[
                "minimum_volume_ratio"
            ],
            baseline_tetrahedra_below_volume_floor=int(
                baseline.get("tetrahedra_below_volume_floor", 0)
            ),
            candidate_tetrahedra_below_volume_floor=int(
                direct.get("tetrahedra_below_volume_floor", 0)
            ),
            baseline_maximum_penetration_m=baseline[
                "maximum_penetration_m"
            ],
            candidate_maximum_penetration_m=direct[
                "maximum_penetration_m"
            ],
            baseline_anchor_error_rms_m=baseline["anchor_error_rms_m"],
            candidate_anchor_error_rms_m=direct["anchor_error_rms_m"],
            baseline_anchor_error_maximum_m=baseline[
                "anchor_error_maximum_m"
            ],
            candidate_anchor_error_maximum_m=direct[
                "anchor_error_maximum_m"
            ],
            baseline_distance_loss=baseline["distance_loss"],
            candidate_distance_loss=direct["distance_loss"],
            baseline_volume_loss=baseline["volume_loss"],
            candidate_volume_loss=direct["volume_loss"],
            baseline_shape_loss=baseline["shape_loss"],
            candidate_shape_loss=direct["shape_loss"],
            history_baseline_rms_m=0.0,
            history_candidate_rms_m=0.0,
            selected_candidate_variant="direct_residual_gradient",
            selected_distance_scale=1.0,
            selected_shape_scale=1.0,
            selected_candidate_scope="direct_evidence",
            selected_candidate_scope_fraction=scope_fraction,
            causal_trial_count=self._stiffness_trial_count,
            **rollout_adoption_metrics,
        )
        if metrics["status"] == "committed":
            self._start_committed_stiffness_evaluation(pending)
        if self.stiffness_metrics_recorder is not None:
            self.stiffness_metrics_recorder.record(
                event="stiffness_validation",
                frame_index=self.current_frame_index,
                timestamp_s=self.current_timestep,
                phase=self._current_action_phase,
                physical={
                    "baseline": self._serializable_shadow_metrics(baseline),
                    "candidate": self._serializable_shadow_metrics(direct),
                },
                material=self._current_material_evaluation_metrics(metrics),
                prediction={
                    "horizon_frames": metrics[
                        "prediction_horizon_frames"
                    ],
                    "horizons": STIFFNESS_ADMISSION_HORIZONS,
                    "baseline_horizon_gaps": baseline[
                        "horizon_visual_losses"
                    ],
                    "candidate_horizon_gaps": direct[
                        "horizon_visual_losses"
                    ],
                    "objective": direct.get(
                        "stiffness_admission_objective",
                        stiffness_validation_objective_name(),
                    ),
                    "baseline_horizon_trajectory_losses_m": baseline.get(
                        "horizon_trajectory_losses_m"
                    ),
                    "candidate_horizon_trajectory_losses_m": direct.get(
                        "horizon_trajectory_losses_m"
                    ),
                    "horizon_trajectory_improvements_m": (
                        trajectory_horizon_improvements_m
                    ),
                    "gap_improvement": improvement,
                    "required_improvement": required_improvement,
                    "status": metrics["status"],
                },
                details={
                    "validation_stage": "direct_h1_h5_h10",
                    "candidate_search_mode": (
                        "single_continuous_proposal"
                    ),
                    "rejection_reasons": reasons,
                },
                force_summary=True,
            )
        self._pending_stiffness_validation = None
        self._last_stiffness_validation_metrics = metrics
        return metrics

    def _validate_pending_stiffness_long_horizon(
        self,
        pending: PendingStiffnessValidation,
    ) -> dict:
        """Removed compatibility entry point; long horizons cannot qualify."""
        raise RuntimeError(
            "H30/H60 stiffness commit qualification has been removed"
        )
        updater = self.stiffness_updater
        assert updater is not None
        assert pending.long_validation_candidate is not None
        pending.candidate = pending.long_validation_candidate
        updater.pending_candidate = pending.candidate
        horizons = STIFFNESS_EVENT_DRIVEN_LONG_HORIZONS
        target_frames = tuple(pending.long_validation_frame_indices)
        selected = dict(pending.short_selection_record or {})
        scope = str(
            selected.get(
                "scope",
                pending.candidate.metrics.get("selected_candidate_scope", "full"),
            )
        )
        scope_fraction = float(
            selected.get(
                "scope_fraction",
                pending.candidate.metrics.get(
                    "selected_candidate_scope_fraction", 1.0
                ),
            )
        )
        self._synchronize_shadow_transaction(self.environment.sim)
        live = self.environment.sim.clone_embodied_gaussian_rollout_state()
        controller_state = (
            self.current_frame_index,
            self.current_timestep,
            self._last_state_index,
            self._last_q_full.detach().clone(),
        )
        verified_distance = updater.distance_stiffness.detach().clone()
        verified_shape = updater.shape_stiffness.detach().clone()
        try:
            baseline = self._run_stiffness_prediction_shadow(
                pending,
                use_candidate=False,
                horizons=horizons,
                target_frames=target_frames,
            )
            candidate = self._run_stiffness_prediction_shadow(
                pending,
                use_candidate=True,
                horizons=horizons,
                target_frames=target_frames,
            )
        finally:
            self._synchronize_shadow_transaction(self.environment.sim)
            self.environment.sim.copy_embodied_gaussian_rollout_state(live)
            updater.restore_verified_stiffness(
                verified_distance, verified_shape
            )
            (
                self.current_frame_index,
                self.current_timestep,
                self._last_state_index,
                self._last_q_full,
            ) = controller_state
            self.dataset_manager.update_frames(self.current_timestep)
            self.environment.sim.update_gaussian_transforms()
            self._synchronize_shadow_transaction(self.environment.sim)
        reasons, improvement, required_improvement = (
            self._stiffness_shadow_rejection_reasons(
                baseline,
                candidate,
                history_safe=True,
                absolute_margin_scale=(
                    self._stiffness_candidate_margin_scale(
                        scope=scope,
                        scope_fraction=scope_fraction,
                    )
                ),
                admission_horizons=horizons,
                required_long_horizons=horizons,
            )
        )
        if reasons:
            metrics = updater.reject(
                "+".join(reasons), pending.candidate
            )
        else:
            metrics = updater.commit(pending.candidate)
            self._last_stiffness_commit_frame_index = int(
                self.current_frame_index
            )
            variant_label = str(
                selected.get(
                    "variant_label",
                    pending.candidate.metrics.get(
                        "selected_candidate_variant", "unknown"
                    ),
                )
            )
            if scope != "global_material_offset":
                axis = self._stiffness_record_material_axis(
                    {
                        "variant_label": variant_label,
                        "distance_scale": float(
                            selected.get("distance_scale", 0.0)
                        ),
                        "shape_scale": float(
                            selected.get("shape_scale", 0.0)
                        ),
                    }
                )
                key = ("event_driven", axis)
                self._stiffness_local_phase_family_commits[key] = int(
                    self._stiffness_local_phase_family_commits.get(key, 0)
                ) + 1
                self._stiffness_local_phase_family_last_commit_frame[key] = (
                    int(self.current_frame_index)
                )
            elif variant_label.startswith("global_distance_"):
                self._stiffness_global_distance_commits += 1
            elif variant_label.startswith("global_shape_"):
                self._stiffness_global_shape_commits += 1
            self._clear_stiffness_global_confirmation()
            self._clear_stiffness_local_confirmations()
        rollout_adoption_metrics = {
            "validated_rollout_adopted": False,
            "validated_rollout_state_rms_m": 0.0,
            "validated_rollout_state_maximum_m": 0.0,
            "validated_rollout_state_finite": True,
        }
        if metrics["status"] == "committed":
            rollout_adoption_metrics = (
                self._adopt_validated_stiffness_rollout(candidate)
            )
        metrics.update(
            validation_status=metrics["status"],
            validation_stage="long_horizon_winner",
            prediction_horizon_frames=(
                self.current_frame_index - pending.frame_index
            ),
            prediction_rollout_steps=len(pending.commands),
            validation_horizons=horizons,
            baseline_horizon_visual_losses=baseline[
                "horizon_visual_losses"
            ],
            candidate_horizon_visual_losses=candidate[
                "horizon_visual_losses"
            ],
            horizon_visual_improvements={
                horizon: (
                    baseline["horizon_visual_losses"][horizon]
                    - candidate["horizon_visual_losses"][horizon]
                )
                for horizon in horizons
            },
            prediction_improvement=float(improvement),
            required_prediction_improvement=float(required_improvement),
            baseline_minimum_volume_ratio=baseline[
                "minimum_volume_ratio"
            ],
            candidate_minimum_volume_ratio=candidate[
                "minimum_volume_ratio"
            ],
            baseline_tetrahedra_below_volume_floor=int(
                baseline.get("tetrahedra_below_volume_floor", 0)
            ),
            candidate_tetrahedra_below_volume_floor=int(
                candidate.get("tetrahedra_below_volume_floor", 0)
            ),
            baseline_maximum_penetration_m=baseline[
                "maximum_penetration_m"
            ],
            candidate_maximum_penetration_m=candidate[
                "maximum_penetration_m"
            ],
            baseline_anchor_error_rms_m=baseline["anchor_error_rms_m"],
            candidate_anchor_error_rms_m=candidate["anchor_error_rms_m"],
            baseline_anchor_error_maximum_m=baseline[
                "anchor_error_maximum_m"
            ],
            candidate_anchor_error_maximum_m=candidate[
                "anchor_error_maximum_m"
            ],
            **rollout_adoption_metrics,
        )
        if metrics["status"] == "committed":
            self._start_committed_stiffness_evaluation(pending)
        if self.stiffness_metrics_recorder is not None:
            self.stiffness_metrics_recorder.record(
                event="stiffness_validation",
                frame_index=self.current_frame_index,
                timestamp_s=self.current_timestep,
                phase=self._current_action_phase,
                physical={
                    "baseline": self._serializable_shadow_metrics(baseline),
                    "candidate": self._serializable_shadow_metrics(candidate),
                },
                material=self._current_material_evaluation_metrics(metrics),
                prediction={
                    "horizon_frames": metrics[
                        "prediction_horizon_frames"
                    ],
                    "horizons": horizons,
                    "baseline_horizon_gaps": baseline[
                        "horizon_visual_losses"
                    ],
                    "candidate_horizon_gaps": candidate[
                        "horizon_visual_losses"
                    ],
                    "gap_improvement": improvement,
                    "required_improvement": required_improvement,
                    "status": metrics["status"],
                },
                details={
                    "validation_stage": "long_horizon_winner",
                    "rejection_reasons": reasons,
                },
                force_summary=True,
            )
        self._pending_stiffness_validation = None
        self._last_stiffness_validation_metrics = metrics
        return metrics

    def _validate_pending_stiffness(
        self,
        *,
        gate_paused: bool,
        gate_reason: str,
        grip_active: bool,
    ) -> dict | None:
        pending = self._pending_stiffness_validation
        updater = self.stiffness_updater
        if pending is None or updater is None:
            return None
        # H30/H60 is no longer a commit-qualification stage. Every pending
        # proposal is decided at the configured short causal horizons.
        long_validation_stage = False
        validation_horizons = STIFFNESS_ADMISSION_HORIZONS
        horizon_frames = self.current_frame_index - pending.frame_index
        validation_frames = self._pending_stiffness_validation_frames(pending)
        required_frame_index = (
            validation_frames[-1]
            if len(validation_frames) == len(validation_horizons)
            else pending.frame_index + STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES
        )
        required_frame_span = required_frame_index - pending.frame_index
        maximum_frame_span = max(
            STIFFNESS_MAXIMUM_PREDICTION_HORIZON_FRAMES,
            required_frame_span + 1,
        )
        if (
            (
                STIFFNESS_MAXIMUM_PENDING_ROLLOUT_STEPS is not None
                and len(pending.commands)
                > STIFFNESS_MAXIMUM_PENDING_ROLLOUT_STEPS
            )
            or horizon_frames
            > maximum_frame_span
        ):
            if STIFFNESS_ADMISSION_MODE == "causal_fixed_lag":
                self._clear_stiffness_global_confirmation()
                self._clear_stiffness_local_confirmations()
            metrics = updater.reject(
                "validation_rollout_too_long", pending.candidate
            )
            metrics.update(
                validation_status="expired",
                prediction_horizon_frames=horizon_frames,
                prediction_rollout_steps=len(pending.commands),
            )
            self._record_terminal_stiffness_validation(
                pending, metrics, "validation_rollout_too_long"
            )
            self._pending_stiffness_validation = None
            self._last_stiffness_validation_metrics = metrics
            return metrics
        return self._validate_pending_stiffness_continuation(
            gate_paused=gate_paused,
            gate_reason=gate_reason,
            grip_active=grip_active,
        )

    def _flow_depth_observation_allowed(self, frame_index: int) -> bool:
        recorder = getattr(self, "tissue_benchmark_recorder", None)
        return bool(
            recorder is None or recorder.observation_allowed(int(frame_index))
        )

    def capture_flow_depth_source_state(self, frame_index: int) -> None:
        """Cache q+ only when a future precomputed pair names this source."""

        sequence = self.flow_depth_observations
        if sequence is None:
            return
        frame_index = int(frame_index)
        if not np.any(sequence.current_source_frames == frame_index):
            return
        positions = wp.to_torch(
            self.environment.sim.state_0.particle_q
        ).detach().cpu().numpy().copy()
        self._flow_depth_source_states[frame_index] = positions
        if (
            self.paper_trajectory_stiffness_optimizer is not None
            or STIFFNESS_ADMISSION_MODE == "sim_particle_graph_lm"
        ):
            self._synchronize_shadow_transaction(self.environment.sim)
            self._paper_adam_source_rollout_states[frame_index] = (
                self.environment.sim.clone_embodied_gaussian_rollout_state()
            )
            self._paper_adam_source_commands[frame_index] = []
            if STIFFNESS_ADMISSION_MODE == "sim_particle_graph_lm":
                self._sim_graph_source_velocities[frame_index] = (
                    wp.to_torch(self.environment.sim.state_0.particle_qd)
                    .detach()
                    .clone()
                )
        first_source_frame = int(sequence.current_source_frames.min())
        if (
            frame_index == first_source_frame
            and self._flow_depth_reference_range_centers is None
        ):
            assert self.flow_depth_bindings is not None
            self._flow_depth_reference_range_centers = fixed_range_centers(
                positions, self.flow_depth_bindings
            )
        live_sources = set(
            int(value)
            for value in sequence.current_source_frames[
                sequence.next_source_frames > frame_index
            ]
        )
        self._flow_depth_source_states = {
            source: state
            for source, state in self._flow_depth_source_states.items()
            if source in live_sources or source == frame_index
        }
        self._paper_adam_source_rollout_states = {
            source: state
            for source, state in self._paper_adam_source_rollout_states.items()
            if source in live_sources or source == frame_index
        }
        self._paper_adam_source_commands = {
            source: commands
            for source, commands in self._paper_adam_source_commands.items()
            if source in live_sources or source == frame_index
        }
        self._sim_graph_source_velocities = {
            source: velocity
            for source, velocity in self._sim_graph_source_velocities.items()
            if source in live_sources or source == frame_index
        }

    def _record_flow_depth_runtime_metrics(
        self,
        *,
        source_frame: int,
        destination_frame: int,
        status: str,
        update_metrics: dict | None = None,
        stiffness_metrics: dict | None = None,
    ) -> None:
        recorder = self.stiffness_metrics_recorder
        if recorder is None:
            return
        update_metrics = update_metrics or {}
        recorder.record(
            event="flow_depth_state_update",
            frame_index=destination_frame,
            timestamp_s=self.current_timestep,
            phase=self._current_action_phase,
            image={
                "status": status,
                "source_frame": source_frame,
                "destination_frame": destination_frame,
                "accepted": bool(update_metrics.get("accepted", False)),
                "valid_tracks": int(update_metrics.get("valid_tracks", 0)),
                "updated_particles": int(
                    update_metrics.get("updated_particles", 0)
                ),
                "maximum_position_correction_m": float(
                    update_metrics.get("maximum_position_correction_m", 0.0)
                ),
                "maximum_velocity_correction_m_s": float(
                    update_metrics.get(
                        "maximum_velocity_correction_m_s", 0.0
                    )
                ),
                "accepted_scale": float(
                    update_metrics.get("accepted_scale", 0.0)
                ),
                "backtrack_count": int(
                    update_metrics.get("backtrack_count", 0)
                ),
                "rejection_reason": str(
                    update_metrics.get("rejection_reason", "")
                ),
            },
            material=stiffness_metrics,
            force_summary=True,
        )

    def _paper_adam_history_snapshots(
        self,
    ) -> tuple[StiffnessHistorySnapshot, ...]:
        """Uniformly sample at most four states from the closest 20 frames."""
        snapshots = tuple(self._stiffness_history)
        if len(snapshots) <= 4:
            return snapshots
        indices = np.linspace(0, len(snapshots) - 1, num=4)
        selected = tuple(
            snapshots[int(round(value))] for value in indices
        )
        # Rounding is unique for len>=5, but retain an explicit invariant so a
        # future sampling change cannot silently double-weight one frame.
        if len({snapshot.frame_index for snapshot in selected}) != 4:
            raise RuntimeError("Paper 4-of-20 history sampling is not unique")
        return selected

    def _append_paper_adam_history_snapshot(
        self,
        *,
        frame_index: int,
        accepted_positions: torch.Tensor,
    ) -> None:
        """Store one accepted visual state without duplicating material ID."""
        sim = self.environment.sim
        self._synchronize_shadow_transaction(sim)
        self._stiffness_history.append(
            StiffnessHistorySnapshot(
                rollout_state=sim.clone_embodied_gaussian_rollout_state(),
                accepted_positions=accepted_positions.detach().clone(),
                frame_index=int(frame_index),
            )
        )

    def _paper_adam_track_target(
        self, frame_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fixed-triangle 3D targets, confidence and validity."""
        assert self.flow_depth_bindings is not None
        assert self.flow_depth_observations is not None
        if self._flow_depth_reference_range_centers is None:
            raise RuntimeError("Paper Adam track reference is unavailable")
        pair_index = self.flow_depth_observations.pair_index_for_next_frame(
            int(frame_index)
        )
        if pair_index is None:
            raise RuntimeError("Paper Adam destination has no observation")
        observation = self.flow_depth_observations.observation(pair_index)
        targets = self._flow_depth_reference_range_centers + (
            observation.next_points_table
            - self.flow_depth_bindings.initial_points_table
        )
        device = self.stiffness_updater.distance_stiffness.device
        dtype = self.stiffness_updater.distance_stiffness.dtype
        target_tensor = torch.as_tensor(targets, device=device, dtype=dtype)
        confidence_tensor = torch.as_tensor(
            observation.confidence, device=device, dtype=dtype
        )
        valid_tensor = torch.as_tensor(
            self.flow_depth_bindings.track_valid
            & observation.track_valid,
            device=device,
            dtype=torch.bool,
        )
        # A 2D tracker sample is not an effective 3D material observation when
        # either depth lookup/back-projection is missing or its confidence is
        # zero.  Keep this definition here so an invalid observation is
        # rejected before any of the four expensive XPBD counterfactuals.
        valid_tensor &= torch.isfinite(target_tensor).all(dim=1)
        valid_tensor &= torch.isfinite(confidence_tensor)
        valid_tensor &= confidence_tensor > 0.0
        return (
            target_tensor,
            confidence_tensor,
            valid_tensor,
        )

    def _paper_adam_predicted_track_points(
        self, particle_positions: torch.Tensor
    ) -> torch.Tensor:
        """Apply the frozen triangle barycentric matrix B to particles."""
        assert self.flow_depth_bindings is not None
        centres = fixed_range_centers(
            particle_positions.detach().cpu().numpy(),
            self.flow_depth_bindings,
        )
        return torch.as_tensor(
            centres,
            device=particle_positions.device,
            dtype=particle_positions.dtype,
        )

    def _paper_adam_fixed_track_regions(self, device: torch.device) -> torch.Tensor:
        """Partition fixed surface tracks into 4x3 spatial regions.

        This mirrors the SIM distribution-robust objective: the loss cannot be
        dominated by whichever image patch happens to contain most tracks.
        Region IDs depend only on immutable initial triangle centres.
        """
        if self._paper_adam_track_region_ids is None:
            if self._flow_depth_reference_range_centers is None:
                raise RuntimeError("Paper Adam track reference is unavailable")
            centers = np.asarray(
                self._flow_depth_reference_range_centers, dtype=np.float64
            )
            xy = centers[:, :2]
            minimum = np.nanmin(xy, axis=0)
            span = np.maximum(np.nanmax(xy, axis=0) - minimum, 1.0e-12)
            normalized = np.clip((xy - minimum) / span, 0.0, 1.0 - 1.0e-12)
            x_bin = np.floor(4.0 * normalized[:, 0]).astype(np.int64)
            y_bin = np.floor(3.0 * normalized[:, 1]).astype(np.int64)
            self._paper_adam_track_region_ids = torch.as_tensor(
                y_bin * 4 + x_bin, dtype=torch.long
            )
        return self._paper_adam_track_region_ids.to(device=device)

    def _paper_adam_history_loss(
        self,
        *,
        trial: PaperTrajectoryMaterialTrial,
        snapshots: tuple[StiffnessHistorySnapshot, ...],
    ) -> tuple[float, tuple[float, ...]]:
        """Evaluate the paper's zero-control equilibrium consistency term."""
        optimizer = self.paper_trajectory_stiffness_optimizer
        updater = self.stiffness_updater
        assert optimizer is not None and updater is not None
        if not snapshots:
            return 0.0, ()
        sim = self.environment.sim
        rms_values: list[float] = []
        for snapshot in snapshots:
            sim.copy_embodied_gaussian_rollout_state(snapshot.rollout_state)
            updater.restore_verified_stiffness(
                trial.distance_stiffness, trial.shape_stiffness
            )
            particle_qd = wp.to_torch(sim.state_0.particle_qd)
            with torch.no_grad():
                particle_qd.zero_()
            projector = sim.triangle_skin_contact_projector
            if projector is not None:
                projector.freeze_persistent_grip_state_machine()
            sim.sync_kinematic_body_interpolation()
            self.environment.step(compute_visual_forces=False)
            if projector is not None:
                projector.freeze_persistent_grip_state_machine()
            current = wp.to_torch(sim.state_0.particle_q).detach()
            rms_values.append(
                self._mass_weighted_particle_rms(
                    snapshot.accepted_positions, current
                )
            )
        scale = optimizer.settings.history_scale_m
        normalized = tuple((value / scale) ** 2 for value in rms_values)
        return float(np.mean(normalized)), tuple(rms_values)

    def _paper_adam_trial_loss(
        self,
        *,
        source_state: object,
        commands: tuple[StiffnessToolCommand, ...],
        destination_frame: int,
        trial: PaperTrajectoryMaterialTrial,
        target_points: torch.Tensor,
        confidence: torch.Tensor,
        track_valid: torch.Tensor,
        history_snapshots: tuple[StiffnessHistorySnapshot, ...],
    ) -> dict[str, float | int | tuple[float, ...]]:
        """Replay one absolute material field and evaluate all three losses."""
        optimizer = self.paper_trajectory_stiffness_optimizer
        updater = self.stiffness_updater
        assert optimizer is not None and updater is not None
        sim = self.environment.sim
        sim.copy_embodied_gaussian_rollout_state(source_state)
        updater.restore_verified_stiffness(
            trial.distance_stiffness, trial.shape_stiffness
        )
        if trial.velocity_damping_per_second is not None:
            self.environment.physics_settings.particle_velocity_damping_per_second = (
                float(trial.velocity_damping_per_second)
            )
        projector = sim.triangle_skin_contact_projector
        for command in commands:
            if int(command.frame_index) > int(destination_frame):
                break
            self._apply_stiffness_tool_command(command)
            if projector is not None:
                # The source snapshot already contains the observed capture
                # state.  Material differentiation must not branch on a new
                # discrete grasp decision under +/- perturbations.
                projector.freeze_persistent_grip_state_machine()
            self.environment.step(compute_visual_forces=False)
            self.apply_current_psm_pose()
            if projector is not None:
                projector.freeze_persistent_grip_state_machine()
        predicted_positions = wp.to_torch(sim.state_0.particle_q).detach()
        predicted_points = self._paper_adam_predicted_track_points(
            predicted_positions
        )
        track_loss, track_metrics = confidence_weighted_huber_track_loss(
            predicted_points=predicted_points,
            target_points=target_points,
            confidence=confidence,
            valid_mask=track_valid,
            robust_scale_m=optimizer.settings.track_robust_scale_m,
            region_ids=self._paper_adam_fixed_track_regions(
                predicted_points.device
            ),
            region_balance_weight=(
                optimizer.settings.region_balance_weight
                if optimizer.settings.sim_global_causal_mode
                else 0.0
            ),
            tail_region_weight=(
                optimizer.settings.tail_region_weight
                if optimizer.settings.sim_global_causal_mode
                else 0.0
            ),
            tail_region_fraction=optimizer.settings.tail_region_fraction,
        )
        history_loss, history_rms = self._paper_adam_history_loss(
            trial=trial, snapshots=history_snapshots
        )
        distance_smooth, shape_smooth = optimizer.graph_smooth_loss(
            trial.distance_stiffness, trial.shape_stiffness
        )
        total_loss = optimizer.total_loss(
            track_loss=track_loss,
            history_loss=history_loss,
            distance_smooth_loss=distance_smooth,
            shape_smooth_loss=shape_smooth,
            parameter_prior_loss=optimizer.global_parameter_prior_loss(trial),
        )
        return {
            "track_loss": float(track_loss),
            "history_loss": float(history_loss),
            "distance_smooth_loss": float(distance_smooth),
            "shape_smooth_loss": float(shape_smooth),
            "total_loss": float(total_loss),
            "valid_tracks": int(track_metrics["valid_tracks"]),
            "confidence_sum": float(track_metrics["confidence_sum"]),
            "track_mean_error_m": float(track_metrics["mean_error_m"]),
            "track_rmse_m": float(track_metrics["rmse_m"]),
            "history_snapshot_count": len(history_snapshots),
            "history_rms_m": history_rms,
            "track_point_mean_loss": float(track_metrics["point_mean_loss"]),
            "track_region_mean_loss": float(track_metrics["region_mean_loss"]),
            "track_tail_region_loss": float(track_metrics["tail_region_loss"]),
            "track_valid_regions": int(track_metrics["valid_regions"]),
        }

    def _paper_adam_causal_open_loop_track_losses(
        self,
        *,
        trial: PaperTrajectoryMaterialTrial,
        observations: tuple[PaperAdamCausalObservation, ...],
    ) -> tuple[dict[str, float | int], ...]:
        """Replay H1/H2/H3 continuously from the earliest observed state.

        The previous implementation restored the visually corrected source
        state before every horizon.  That measures three unrelated one-step
        fits and cannot identify a parameter that must survive the future
        open-loop interval.  SIM instead restores once, advances through each
        consecutive transition, and scores every intermediate target.
        """
        if not observations:
            return ()
        optimizer = self.paper_trajectory_stiffness_optimizer
        updater = self.stiffness_updater
        assert optimizer is not None and updater is not None
        sim = self.environment.sim
        sim.copy_embodied_gaussian_rollout_state(observations[0].source_state)
        updater.restore_verified_stiffness(
            trial.distance_stiffness, trial.shape_stiffness
        )
        if trial.velocity_damping_per_second is not None:
            self.environment.physics_settings.particle_velocity_damping_per_second = (
                float(trial.velocity_damping_per_second)
            )
        projector = sim.triangle_skin_contact_projector
        values: list[dict[str, float | int]] = []
        previous_destination: int | None = None
        for observation in observations:
            if (
                previous_destination is not None
                and int(observation.source_frame) != previous_destination
            ):
                raise RuntimeError(
                    "SIM-global causal observations must be consecutive"
                )
            for command in observation.commands:
                if int(command.frame_index) > int(observation.destination_frame):
                    break
                self._apply_stiffness_tool_command(command)
                if projector is not None:
                    projector.freeze_persistent_grip_state_machine()
                self.environment.step(compute_visual_forces=False)
                self.apply_current_psm_pose()
                if projector is not None:
                    projector.freeze_persistent_grip_state_machine()
            previous_destination = int(observation.destination_frame)
            if not observation.supervision_valid:
                # Keep the real XPBD evolution and tool motion across a visual
                # dropout, but contribute no target loss for that endpoint.
                continue
            positions = wp.to_torch(sim.state_0.particle_q).detach()
            predicted_points = self._paper_adam_predicted_track_points(
                positions
            )
            track_loss, track_metrics = confidence_weighted_huber_track_loss(
                predicted_points=predicted_points,
                target_points=observation.target_points,
                confidence=observation.confidence,
                valid_mask=observation.track_valid,
                robust_scale_m=optimizer.settings.track_robust_scale_m,
                region_ids=self._paper_adam_fixed_track_regions(
                    predicted_points.device
                ),
                region_balance_weight=optimizer.settings.region_balance_weight,
                tail_region_weight=optimizer.settings.tail_region_weight,
                tail_region_fraction=optimizer.settings.tail_region_fraction,
            )
            values.append(
                {
                    "track_loss": float(track_loss),
                    "valid_tracks": int(track_metrics["valid_tracks"]),
                    "track_mean_error_m": float(
                        track_metrics["mean_error_m"]
                    ),
                    "track_rmse_m": float(track_metrics["rmse_m"]),
                    "track_point_mean_loss": float(
                        track_metrics["point_mean_loss"]
                    ),
                    "track_region_mean_loss": float(
                        track_metrics["region_mean_loss"]
                    ),
                    "track_tail_region_loss": float(
                        track_metrics["tail_region_loss"]
                    ),
                    "track_valid_regions": int(
                        track_metrics["valid_regions"]
                    ),
                }
            )
        return tuple(values)

    def _run_paper_trajectory_adam_update(
        self,
        *,
        source_frame: int,
        destination_frame: int,
        active_mask: torch.Tensor,
        accepted_positions: torch.Tensor,
        observation_supervision_valid: bool = True,
        local_distance_signal: torch.Tensor | None = None,
    ) -> dict[str, float | int | str]:
        """Run four exact counterfactuals and commit one bounded Adam step."""
        optimizer = self.paper_trajectory_stiffness_optimizer
        updater = self.stiffness_updater
        assert optimizer is not None and updater is not None
        source_state = self._paper_adam_source_rollout_states.get(source_frame)
        commands = tuple(self._paper_adam_source_commands.get(source_frame, ()))
        if source_state is None or not commands:
            metrics: dict[str, float | int | str] = {
                "status": "paper_adam_missing_source_rollout",
                "source_frame": int(source_frame),
                "destination_frame": int(destination_frame),
                "command_count": len(commands),
            }
            optimizer.last_metrics = metrics
            return metrics
        target_points, confidence, track_valid = self._paper_adam_track_target(
            destination_frame
        )
        observation_supervision_valid = bool(observation_supervision_valid)
        if not observation_supervision_valid:
            track_valid = torch.zeros_like(track_valid)
        effective_tracks = int(torch.count_nonzero(track_valid).item())
        effective_particles = int(torch.count_nonzero(active_mask).item())
        if observation_supervision_valid and (
            effective_tracks == 0 or effective_particles == 0
        ):
            status = (
                "paper_adam_no_valid_3d_target"
                if effective_tracks == 0
                else "paper_adam_no_active_material_particle"
            )
            metrics = {
                "status": status,
                "source_frame": int(source_frame),
                "destination_frame": int(destination_frame),
                "command_count": len(commands),
                "valid_tracks": effective_tracks,
                "active_particles": effective_particles,
                "observation_frequency": (
                    "every_material_observable_video_observation"
                ),
                "future_observation_used": 0,
            }
            optimizer.last_metrics = metrics
            updater.last_metrics = dict(metrics)
            # The accepted corrected state is still a valid historical state;
            # only its absent 3D target prevents a material-gradient update.
            self._append_paper_adam_history_snapshot(
                frame_index=destination_frame,
                accepted_positions=accepted_positions,
            )
            return metrics
        if optimizer.settings.sim_global_causal_mode:
            if (
                self._paper_adam_causal_window
                and int(self._paper_adam_causal_window[-1].destination_frame)
                != int(source_frame)
            ):
                # A missing material-observable transition breaks an open-loop
                # H1/H2/H3 chain.  SIM clears its causal history on the same
                # discontinuity instead of silently joining unrelated states.
                self._paper_adam_causal_window.clear()
                self._paper_adam_causal_block_index += 1
            self._paper_adam_causal_window.append(
                PaperAdamCausalObservation(
                    source_state=source_state,
                    source_frame=int(source_frame),
                    commands=commands,
                    destination_frame=int(destination_frame),
                    target_points=target_points.detach().clone(),
                    confidence=confidence.detach().clone(),
                    track_valid=track_valid.detach().clone(),
                    supervision_valid=observation_supervision_valid,
                    material_active_mask=active_mask.detach().clone(),
                    local_distance_signal=(
                        None
                        if local_distance_signal is None
                        else local_distance_signal.detach().clone()
                    ),
                )
            )
            buffered_observations = tuple(self._paper_adam_causal_window)
            # SIM-global uses the causal trajectory window plus a parameter
            # prior. The static history term is optional and disabled by the
            # formal runner to avoid confounding damping with rest relaxation.
            history_snapshots = (
                self._paper_adam_history_snapshots()
                if optimizer.settings.history_weight > 0.0
                else ()
            )
        else:
            causal_observations = (
                PaperAdamCausalObservation(
                    source_state=source_state,
                    source_frame=int(source_frame),
                    commands=commands,
                    destination_frame=int(destination_frame),
                    target_points=target_points,
                    confidence=confidence,
                    track_valid=track_valid,
                    supervision_valid=observation_supervision_valid,
                    material_active_mask=active_mask,
                    local_distance_signal=local_distance_signal,
                ),
            )
            history_snapshots = self._paper_adam_history_snapshots()
        reconstruction_h3_only = bool(
            optimizer.settings.sim_global_causal_mode
            and optimizer.settings.reconstruction_nonoverlapping_h3
        )
        reconstruction_h3_block_end = True
        if reconstruction_h3_only:
            recorder = self.tissue_benchmark_recorder
            if recorder is None or recorder.protocol != "reconstruction_7to1":
                raise RuntimeError(
                    "Non-overlapping reconstruction H3 requires the 7:1 protocol"
                )
            reconstruction_h3_block_end = is_reconstruction_h3_block_end(
                destination_frame,
                recorder.reconstruction_test_phase,
            )
        if optimizer.settings.sim_global_causal_mode:
            buffered_supervised = sum(
                observation.supervision_valid
                for observation in buffered_observations
            )
            selected_supervised = (
                optimizer.settings.causal_window_size
                if reconstruction_h3_only or buffered_supervised >= 3
                else optimizer.settings.causal_minimum_window_size
            )
            causal_observations = select_recent_supervised_causal_window(
                buffered_observations,
                required_supervised=selected_supervised,
                maximum_span=(
                    optimizer.settings.causal_maximum_window_size
                ),
                maximum_missing=(
                    optimizer.settings.causal_maximum_missing_observations
                ),
            )
        supervised_horizon_count = sum(
            observation.supervision_valid
            for observation in causal_observations
        )
        causal_missing_count = (
            len(causal_observations) - supervised_horizon_count
        )
        window_active_mask = active_mask
        if optimizer.settings.sim_global_causal_mode and causal_observations:
            window_active_mask = torch.zeros_like(active_mask)
            for observation in causal_observations:
                if (
                    observation.supervision_valid
                    and observation.material_active_mask is not None
                ):
                    window_active_mask |= observation.material_active_mask
        required_causal_window = (
            optimizer.settings.causal_window_size
            if reconstruction_h3_only
            else optimizer.settings.causal_minimum_window_size
        )
        if optimizer.settings.sim_global_causal_mode and (
            supervised_horizon_count < required_causal_window
            or not reconstruction_h3_block_end
            or (
                not observation_supervision_valid
                and not reconstruction_h3_only
            )
        ):
            # H1/H2 warm the dense reconstruction buffer.  Only the last,
            # disjoint H3 wholly contained in each seven-frame training block
            # may run material counterfactuals; no held-out frame is consumed.
            warmup_status = (
                "reconstruction_h3_waiting_for_block_end"
                if reconstruction_h3_only
                and supervised_horizon_count >= required_causal_window
                else (
                    "causal_gap_buffered"
                    if not observation_supervision_valid
                    else "causal_window_warmup"
                )
            )
            optimizer_metrics = optimizer.record_causal_warmup_observation(
                window_active_mask,
                horizon_count=supervised_horizon_count,
                status=warmup_status,
            )
            updater.evidence_count += 1
            optimizer_metrics.update(
                source_frame=int(source_frame),
                destination_frame=int(destination_frame),
                history_window_size=len(self._stiffness_history),
                history_snapshot_frames="",
                command_count=len(commands),
                trial_losses="{}",
                admission_mode="paper_trajectory_adam",
                observation_frequency=(
                    "every_material_observable_video_observation"
                ),
                triangle_correspondence="fixed_barycentric",
                future_observation_used=0,
                causal_block_index=int(
                    self._paper_adam_causal_block_index
                ),
                reconstruction_nonoverlapping_h3=int(
                    reconstruction_h3_only
                ),
                reconstruction_h3_block_end=int(
                    reconstruction_h3_block_end
                ),
                causal_physical_span=len(causal_observations),
                causal_supervised_observations=supervised_horizon_count,
                causal_missing_observations=causal_missing_count,
                causal_three_of_four_used=int(
                    supervised_horizon_count == 3
                    and len(causal_observations) == 4
                ),
            )
            optimizer.last_metrics = dict(optimizer_metrics)
            updater.last_metrics = dict(optimizer_metrics)
            result = dict(optimizer_metrics)
            result.update(
                candidate_count=updater.candidate_count,
                evidence_count=updater.evidence_count,
                rejected_count=updater.rejected_count,
            )
            self._append_paper_adam_history_snapshot(
                frame_index=destination_frame,
                accepted_positions=accepted_positions,
            )
            return result
        sim = self.environment.sim
        self._synchronize_shadow_transaction(sim)
        live_state = sim.clone_embodied_gaussian_rollout_state()
        live_frame_index = int(self.current_frame_index)
        live_timestep = float(self.current_timestep)
        live_state_index = int(self._last_state_index)
        live_q_full = self._last_q_full.detach().clone()
        verified_distance = updater.distance_stiffness.detach().clone()
        verified_shape = updater.shape_stiffness.detach().clone()
        verified_velocity_damping = float(
            self.environment.physics_settings
            .particle_velocity_damping_per_second
        )
        trial_losses: dict[str, dict] = {}
        verbose_trial_losses: dict[str, dict] = {}
        try:
            for trial in optimizer.begin_observation(window_active_mask):
                if optimizer.settings.sim_global_causal_mode:
                    # Independent restarted one-step replays are a compact
                    # surrogate gradient.  The authoritative objective below
                    # is a separate continuous Warp rollout.  A commit is
                    # legal only when the two 2-D parameter directions agree,
                    # mirroring SIM's Torch/Warp cosine gate.
                    surrogate_horizon_values = [
                        self._paper_adam_trial_loss(
                            source_state=observation.source_state,
                            commands=observation.commands,
                            destination_frame=(
                                observation.destination_frame
                            ),
                            trial=trial,
                            target_points=observation.target_points,
                            confidence=observation.confidence,
                            track_valid=observation.track_valid,
                            history_snapshots=(),
                        )
                        for observation in causal_observations
                        if observation.supervision_valid
                    ]
                    open_loop_horizon_values = (
                        self._paper_adam_causal_open_loop_track_losses(
                            trial=trial,
                            observations=causal_observations,
                        )
                    )
                    weights = optimizer.settings.causal_horizon_weights[
                        : len(open_loop_horizon_values)
                    ]
                    weighted_track_loss = float(
                        np.average(
                            [
                                value["track_loss"]
                                for value in open_loop_horizon_values
                            ],
                            weights=weights,
                        )
                    )
                    surrogate_track_loss = float(
                        np.average(
                            [
                                value["track_loss"]
                                for value in surrogate_horizon_values
                            ],
                            weights=weights,
                        )
                    )
                    if history_snapshots:
                        history_loss, history_rms = (
                            self._paper_adam_history_loss(
                                trial=trial, snapshots=history_snapshots
                            )
                        )
                    else:
                        history_loss, history_rms = 0.0, ()
                    distance_smooth, shape_smooth = (
                        optimizer.graph_smooth_loss(
                            trial.distance_stiffness,
                            trial.shape_stiffness,
                        )
                    )
                    values = {
                        "causal_block_index": int(
                            self._paper_adam_causal_block_index
                        ),
                        "track_loss": weighted_track_loss,
                        "history_loss": float(history_loss),
                        "distance_smooth_loss": float(distance_smooth),
                        "shape_smooth_loss": float(shape_smooth),
                        "total_loss": optimizer.total_loss(
                            track_loss=weighted_track_loss,
                            history_loss=history_loss,
                            distance_smooth_loss=distance_smooth,
                            shape_smooth_loss=shape_smooth,
                            parameter_prior_loss=(
                                optimizer.global_parameter_prior_loss(trial)
                            ),
                        ),
                        "surrogate_total_loss": optimizer.total_loss(
                            track_loss=surrogate_track_loss,
                            history_loss=history_loss,
                            distance_smooth_loss=distance_smooth,
                            shape_smooth_loss=shape_smooth,
                            parameter_prior_loss=(
                                optimizer.global_parameter_prior_loss(trial)
                            ),
                        ),
                        "horizon_total_losses": tuple(
                            float(value["track_loss"])
                            for value in open_loop_horizon_values
                        ),
                        "surrogate_horizon_total_losses": tuple(
                            float(value["track_loss"])
                            for value in surrogate_horizon_values
                        ),
                        "horizon_frames": tuple(
                            observation.destination_frame
                            for observation in causal_observations
                            if observation.supervision_valid
                        ),
                        "horizon_track_metrics": open_loop_horizon_values,
                        "surrogate_horizon_track_metrics": (
                            surrogate_horizon_values
                        ),
                        "history_snapshot_count": len(history_snapshots),
                        "history_rms_m": history_rms,
                    }
                else:
                    values = self._paper_adam_trial_loss(
                        source_state=source_state,
                        commands=commands,
                        destination_frame=destination_frame,
                        trial=trial,
                        target_points=target_points,
                        confidence=confidence,
                        track_valid=track_valid,
                        history_snapshots=history_snapshots,
                    )
                verbose_trial_losses[trial.label] = dict(values)
                trial_losses[trial.label] = dict(values)
            distance, shape, optimizer_metrics = (
                optimizer.finish_observation(trial_losses)
            )
            if (
                optimizer_metrics.get("status") == "sim_global_updated"
                and optimizer.settings.local_distance_enabled
            ):
                supervised_observations = tuple(
                    observation
                    for observation in causal_observations
                    if observation.supervision_valid
                    and observation.local_distance_signal is not None
                    and observation.material_active_mask is not None
                )
                local_raw = torch.zeros_like(active_mask, dtype=torch.float32)
                local_support = torch.zeros_like(active_mask, dtype=torch.bool)
                local_weights = optimizer.settings.causal_horizon_weights[
                    : len(supervised_observations)
                ]
                for weight, observation in zip(
                    local_weights, supervised_observations
                ):
                    assert observation.local_distance_signal is not None
                    assert observation.material_active_mask is not None
                    local_raw += (
                        float(weight) * observation.local_distance_signal
                    )
                    local_support |= (
                        observation.material_active_mask
                        & (observation.local_distance_signal.abs() > 0.0)
                    )
                if local_weights:
                    local_raw /= float(sum(local_weights))
                local_direction = optimizer.prepare_local_distance_direction(
                    local_raw, local_support
                )
                local_trials = optimizer.begin_local_distance_observation(
                    local_direction
                )
                if local_trials:
                    local_losses: dict[str, float] = {}
                    for local_trial in local_trials:
                        local_horizon_values = (
                            self._paper_adam_causal_open_loop_track_losses(
                                trial=local_trial,
                                observations=causal_observations,
                            )
                        )
                        weights = (
                            optimizer.settings.causal_horizon_weights[
                                : len(local_horizon_values)
                            ]
                        )
                        local_track_loss = float(
                            np.average(
                                [
                                    value["track_loss"]
                                    for value in local_horizon_values
                                ],
                                weights=weights,
                            )
                        )
                        distance_smooth, shape_smooth = (
                            optimizer.graph_smooth_loss(
                                local_trial.distance_stiffness,
                                local_trial.shape_stiffness,
                            )
                        )
                        local_total_loss = optimizer.total_loss(
                            track_loss=local_track_loss,
                            history_loss=0.0,
                            distance_smooth_loss=distance_smooth,
                            shape_smooth_loss=shape_smooth,
                            parameter_prior_loss=(
                                optimizer.global_parameter_prior_loss(
                                    local_trial
                                )
                            ),
                        )
                        local_losses[local_trial.label] = local_total_loss
                        verbose_trial_losses[local_trial.label] = {
                            "track_loss": local_track_loss,
                            "distance_smooth_loss": distance_smooth,
                            "shape_smooth_loss": shape_smooth,
                            "total_loss": local_total_loss,
                            "horizon_total_losses": tuple(
                                float(value["track_loss"])
                                for value in local_horizon_values
                            ),
                        }
                    distance, shape, local_metrics = (
                        optimizer.finish_local_distance_observation(
                            local_losses
                        )
                    )
                    optimizer_metrics.update(local_metrics)
                else:
                    optimizer_metrics.update(
                        local_distance_status="no_observable_direction",
                        local_distance_update_count=(
                            optimizer.local_distance_update_count
                        ),
                        local_distance_global_mean_preserved=1,
                    )
            elif optimizer.settings.local_distance_enabled:
                optimizer_metrics.update(
                    local_distance_status="global_h3_not_updated",
                    local_distance_update_count=(
                        optimizer.local_distance_update_count
                    ),
                    local_distance_global_mean_preserved=1,
                )
        finally:
            self._synchronize_shadow_transaction(sim)
            sim.copy_embodied_gaussian_rollout_state(live_state)
            updater.restore_verified_stiffness(
                verified_distance, verified_shape
            )
            self.environment.physics_settings.particle_velocity_damping_per_second = (
                verified_velocity_damping
            )
            self.current_frame_index = live_frame_index
            self.current_timestep = live_timestep
            self._last_state_index = live_state_index
            self._last_q_full = live_q_full
            self.dataset_manager.update_frames(live_timestep)
            self.apply_current_psm_pose()
            sim.update_gaussian_transforms()
            self._synchronize_shadow_transaction(sim)
        updater.evidence_count += 1
        optimizer_metrics.update(
            source_frame=int(source_frame),
            destination_frame=int(destination_frame),
            history_window_size=len(self._stiffness_history),
            history_snapshot_frames=",".join(
                str(snapshot.frame_index) for snapshot in history_snapshots
            ),
            command_count=len(commands),
            trial_losses=json.dumps(verbose_trial_losses, sort_keys=True),
            admission_mode="paper_trajectory_adam",
            observation_frequency=(
                "every_material_observable_video_observation"
            ),
            triangle_correspondence="fixed_barycentric",
            future_observation_used=0,
            causal_physical_span=len(causal_observations),
            causal_supervised_observations=supervised_horizon_count,
            causal_missing_observations=causal_missing_count,
            causal_three_of_four_used=int(
                supervised_horizon_count == 3
                and len(causal_observations) == 4
            ),
        )
        optimizer_updated = optimizer_metrics["status"] in {
            "adam_updated",
            "sim_global_updated",
        }
        if not optimizer_updated:
            updater.last_metrics = dict(optimizer_metrics)
            result = dict(optimizer_metrics)
            result.update(
                candidate_count=updater.candidate_count,
                evidence_count=updater.evidence_count,
                rejected_count=updater.rejected_count,
            )
        else:
            candidate_active_mask = (
                ~updater.fixed_mask
                if optimizer.settings.sim_global_causal_mode
                else active_mask
            )
            candidate = updater.absolute_adam_candidate(
                distance_stiffness=distance,
                shape_stiffness=shape,
                active_mask=candidate_active_mask,
                optimizer_metrics=optimizer_metrics,
            )
            result = updater.commit(candidate)
            if optimizer.settings.sim_global_causal_mode:
                self.environment.physics_settings.particle_velocity_damping_per_second = (
                    optimizer.current_velocity_damping_per_second()
                )
            result.update(
                validation_status="committed",
                validation_stage=(
                    "sim_global_available_h2_h3_cosine_descent_adam"
                    if optimizer.settings.sim_global_causal_mode
                    else "paper_loss_adam_immediate"
                ),
                shadow_validation_enabled=False,
                fixed_log_step_cap_enabled=(
                    optimizer.settings.sim_global_causal_mode
                ),
            )
            self._last_stiffness_commit_frame_index = int(destination_frame)
        # H_t contains previous accepted observations only.  Add the current
        # accepted state after its gradient/commit transaction completes.
        self._append_paper_adam_history_snapshot(
            frame_index=destination_frame,
            accepted_positions=accepted_positions,
        )
        return result

    def _sim_graph_warp_objective(
        self, coefficients: torch.Tensor
    ) -> float:
        """Replay the SIM H1/H3/H5 relative-track objective in current Warp."""
        updater = self.stiffness_updater
        if not isinstance(updater, SimParticleGraphUpdater):
            raise RuntimeError("SIM particle-graph objective has wrong updater")
        transitions = tuple(self._sim_graph_transitions)
        if len(transitions) < 5:
            return float("inf")
        coefficients = updater.normalize_local_coefficients(
            coefficients.detach().to(
                device=updater.rest_positions.device, dtype=torch.float32
            )
        )
        distance = updater.local_distance_from_coefficients(coefficients)
        shape = updater.local_shape_from_coefficients(coefficients)
        sim = self.environment.sim
        self._synchronize_shadow_transaction(sim)
        live_state = sim.clone_embodied_gaussian_rollout_state()
        live_frame_index = int(self.current_frame_index)
        live_timestep = float(self.current_timestep)
        live_state_index = int(self._last_state_index)
        live_q_full = self._last_q_full.detach().clone()
        live_distance = updater.distance_stiffness.detach().clone()
        live_shape = updater.shape_stiffness.detach().clone()
        position_losses: list[tuple[float, torch.Tensor]] = []
        strain_losses: list[tuple[float, torch.Tensor]] = []
        edge_source, edge_target = updater.edges[:, 0], updater.edges[:, 1]
        try:
            sim.copy_embodied_gaussian_rollout_state(transitions[0].source_state)
            updater.restore_verified_stiffness(distance, shape)
            projector = sim.triangle_skin_contact_projector
            for horizon, transition in enumerate(transitions, start=1):
                for command in transition.commands:
                    if int(command.frame_index) > int(
                        transition.destination_frame
                    ):
                        break
                    self._apply_stiffness_tool_command(command)
                    if projector is not None:
                        projector.freeze_persistent_grip_state_machine()
                    self.environment.step(compute_visual_forces=False)
                    self.apply_current_psm_pose()
                    if projector is not None:
                        projector.freeze_persistent_grip_state_machine()
                horizon_slot = {1: 0, 3: 1, 5: 2}.get(horizon)
                if horizon_slot is None:
                    continue
                horizon_weight = updater.settings.local_horizon_weights[
                    horizon_slot
                ]
                positions = wp.to_torch(sim.state_0.particle_q).detach().clone()
                active = (
                    transition.eligible_mask
                    & transition.material_active_mask
                )
                predicted_tracks = updater.fixed_track_centers(positions)
                support = torch.sum(
                    updater.track_particle_weights
                    * active[updater.track_particle_ids].to(torch.float32),
                    dim=1,
                )
                track_active = transition.track_valid_mask & (support >= 0.50)
                if bool(track_active.any().item()):
                    (
                        node_loss,
                        relative_loss,
                        _relative_active,
                        _active_pair_count,
                    ) = updater.robust_relative_track_losses(
                        predicted_tracks,
                        transition.track_target_positions,
                        track_active,
                    )
                    distribution_loss, *_ = (
                        updater.global_position_distribution_loss(
                            node_loss,
                            track_active,
                            updater.track_region_basis,
                        )
                    )
                    relative_weight = updater.settings.global_relative_track_weight
                    position_losses.append(
                        (
                            horizon_weight,
                            (1.0 - relative_weight) * distribution_loss
                            + relative_weight * relative_loss,
                        )
                    )
                if updater.edges.numel():
                    edge_active = (
                        (active[edge_source] | active[edge_target])
                        & ~transition.control_exclusion_mask[edge_source]
                        & ~transition.control_exclusion_mask[edge_target]
                    )
                    if bool(edge_active.any().item()):
                        current_length = torch.linalg.vector_norm(
                            positions[edge_target] - positions[edge_source], dim=1
                        )
                        target_length = torch.linalg.vector_norm(
                            transition.corrected_positions[edge_target]
                            - transition.corrected_positions[edge_source],
                            dim=1,
                        )
                        strain_error = (
                            (current_length - target_length)
                            / updater.rest_edge_lengths
                        ) / updater.settings.edge_strain_full_scale
                        robust_strain = torch.sqrt(
                            strain_error.square() + 1.0e-4
                        ) - 0.01
                        strain_losses.append(
                            (horizon_weight, robust_strain[edge_active].mean())
                        )
            if not position_losses and not strain_losses:
                return float("inf")
            position_loss = (
                sum(weight * loss for weight, loss in position_losses)
                / sum(weight for weight, _ in position_losses)
                if position_losses
                else coefficients.new_zeros(())
            )
            strain_loss = (
                sum(weight * loss for weight, loss in strain_losses)
                / sum(weight for weight, _ in strain_losses)
                if strain_losses
                else coefficients.new_zeros(())
            )
            prior, _coefficient_prior, _spatial_prior = (
                updater.local_regularization_loss(coefficients)
            )
            blend = updater.settings.strain_signal_weight
            return float(
                ((1.0 - blend) * position_loss + blend * strain_loss + prior)
                .detach()
                .item()
            )
        finally:
            self._synchronize_shadow_transaction(sim)
            sim.copy_embodied_gaussian_rollout_state(live_state)
            updater.restore_verified_stiffness(live_distance, live_shape)
            self.current_frame_index = live_frame_index
            self.current_timestep = live_timestep
            self._last_state_index = live_state_index
            self._last_q_full = live_q_full
            self.dataset_manager.update_frames(live_timestep)
            self.apply_current_psm_pose()
            sim.update_gaussian_transforms()
            self._synchronize_shadow_transaction(sim)

    def _sim_graph_directional_derivatives(
        self, autograd_gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate two high-dimensional Torch directions in real Warp."""
        updater = self.stiffness_updater
        if not isinstance(updater, SimParticleGraphUpdater):
            raise RuntimeError("SIM particle-graph direction gate has wrong updater")
        center = updater.normalize_local_coefficients(
            updater.low_dimensional_log_coefficients.detach().clone()
        )
        gradient = updater.project_local_gradient(
            autograd_gradient.detach().to(center)
        )
        if len(self._sim_graph_transitions) < 5:
            missing = torch.full(
                (updater.settings.graph_directional_probe_count,),
                float("nan"),
                device=center.device,
                dtype=center.dtype,
            )
            return missing, missing.clone()
        primary = gradient / gradient.abs().max().clamp_min(1.0e-12)
        directions = [primary]
        indices = torch.arange(
            len(center) - 1, device=center.device, dtype=center.dtype
        )
        frequencies = (0.017, 0.031, 0.047, 0.071)
        for probe_index in range(
            1, updater.settings.graph_directional_probe_count
        ):
            direction = torch.zeros_like(center)
            direction[0] = 1.0 if probe_index % 2 else -1.0
            frequency = frequencies[(probe_index - 1) % len(frequencies)]
            direction[1:] = torch.sin(
                indices * frequency + float(probe_index) * 0.73
            )
            direction = updater.project_local_gradient(direction)
            direction -= (
                torch.dot(direction, primary)
                / torch.dot(primary, primary).clamp_min(1.0e-12)
            ) * primary
            direction = updater.project_local_gradient(direction)
            direction /= direction.abs().max().clamp_min(1.0e-12)
            directions.append(direction)
        epsilon = updater.settings.graph_directional_log_epsilon
        predicted = torch.empty(
            len(directions), device=center.device, dtype=center.dtype
        )
        measured = torch.empty_like(predicted)
        for index, direction in enumerate(directions):
            plus = updater.normalize_local_coefficients(
                center + epsilon * direction
            )
            minus = updater.normalize_local_coefficients(
                center - epsilon * direction
            )
            predicted[index] = torch.dot(gradient, direction)
            measured[index] = (
                self._sim_graph_warp_objective(plus)
                - self._sim_graph_warp_objective(minus)
            ) / (2.0 * epsilon)
        return predicted, measured

    def _run_sim_particle_graph_update(
        self,
        *,
        source_frame: int,
        destination_frame: int,
        predicted_positions: torch.Tensor,
        corrected_positions: torch.Tensor,
        accepted_residual: torch.Tensor,
        quality_valid_mask: torch.Tensor,
        supervision_valid_mask: torch.Tensor,
        control_exclusion_mask: torch.Tensor,
        contact_metrics: dict,
        gate_paused: bool,
        gate_reason: str,
    ) -> dict[str, float | int | str]:
        """Run the migrated SIM particle-graph LM transaction."""
        updater = self.stiffness_updater
        if not isinstance(updater, SimParticleGraphUpdater):
            raise RuntimeError("SIM particle-graph admission has wrong updater")
        source_state = self._paper_adam_source_rollout_states.get(source_frame)
        source_positions = self._flow_depth_source_states.get(source_frame)
        source_velocities = self._sim_graph_source_velocities.get(source_frame)
        commands = tuple(self._paper_adam_source_commands.get(source_frame, ()))
        if (
            source_state is None
            or source_positions is None
            or source_velocities is None
            or not commands
        ):
            metrics = {
                "status": "sim_graph_missing_source_rollout",
                "source_frame": int(source_frame),
                "destination_frame": int(destination_frame),
                "command_count": len(commands),
            }
            updater.last_metrics = dict(metrics)
            return metrics
        if gate_paused:
            metrics = {
                "status": "sim_graph_global_gate_paused",
                "rejection_reason": str(gate_reason),
                "source_frame": int(source_frame),
                "destination_frame": int(destination_frame),
                "candidate_count": int(updater.candidate_count),
                "update_count": int(updater.update_count),
                "rejected_count": int(updater.rejected_count),
                "future_observation_used": 0,
            }
            updater.last_metrics = dict(metrics)
            return metrics
        _edge_signal, edge_support = updater._edge_strain_signal(
            predicted_positions, corrected_positions
        )
        strain_observable = (
            (edge_support > 0.0)
            & quality_valid_mask
            & supervision_valid_mask
            & ~control_exclusion_mask
            & ~updater.fixed_mask
        )
        strain_particle_count = int(
            torch.count_nonzero(strain_observable).item()
        )
        contact_count = int(contact_metrics.get("contact_count", 0))
        persistent_grip_active = bool(
            contact_metrics.get("persistent_grip_active", False)
        )
        effective_contact = contact_count >= 1 or persistent_grip_active
        if (
            not effective_contact
            or strain_particle_count
            < STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES
        ):
            # A five-observation GraphLM window must be entirely material
            # observable. Camera/depth motion before tool loading cannot
            # identify stiffness and previously caused 13/22 Future commits.
            updater.invalidate_signal_history()
            self._sim_graph_transitions.clear()
            metrics = {
                "status": "sim_graph_material_unobservable",
                "source_frame": int(source_frame),
                "destination_frame": int(destination_frame),
                "observability_contact_count": contact_count,
                "observability_persistent_grip_active": int(
                    persistent_grip_active
                ),
                "observability_edge_strain_particles": strain_particle_count,
                "observability_minimum_edge_strain_particles": int(
                    STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES
                ),
                "future_observation_used": 0,
            }
            updater.last_metrics = dict(metrics)
            return metrics
        target_points, _confidence, track_valid = self._paper_adam_track_target(
            destination_frame
        )
        projector = self.environment.sim.triangle_skin_contact_projector
        direct_control_mask = torch.zeros_like(control_exclusion_mask)
        if projector is not None and bool(
            projector.persistent_grip_state.numpy()[0]
        ):
            grip_body = wp.to_torch(
                projector.persistent_grip_particle_body
            ).to(device=control_exclusion_mask.device)
            direct_control_mask = grip_body >= 0
        source_positions_tensor = torch.as_tensor(
            source_positions,
            device=predicted_positions.device,
            dtype=predicted_positions.dtype,
        )
        coupling_displacements = torch.zeros_like(source_positions_tensor)
        coupling_displacements[direct_control_mask] = (
            corrected_positions[direct_control_mask]
            - source_positions_tensor[direct_control_mask]
        )
        candidate = updater.propose(
            physical_prediction=predicted_positions,
            accepted_residual=accepted_residual,
            quality_valid_mask=quality_valid_mask,
            supervision_valid_mask=supervision_valid_mask,
            control_exclusion_mask=control_exclusion_mask,
            control_frozen_mask=direct_control_mask,
            control_coupling_base_positions=source_positions_tensor,
            control_coupling_displacements=coupling_displacements,
            track_target_positions=target_points,
            track_valid_mask=track_valid,
            rollout_start_positions=source_positions_tensor,
            physical_velocities=source_velocities,
            frame_index=destination_frame,
            globally_paused=gate_paused,
        )
        if (
            self._sim_graph_transitions
            and self._sim_graph_transitions[-1].destination_frame
            != int(source_frame)
        ):
            self._sim_graph_transitions.clear()
        self._sim_graph_transitions.append(
            SimParticleGraphTransition(
                source_state=source_state,
                source_frame=int(source_frame),
                commands=commands,
                destination_frame=int(destination_frame),
                corrected_positions=corrected_positions.detach().clone(),
                eligible_mask=candidate.eligible_mask.detach().clone(),
                material_active_mask=candidate.source_mask.detach().clone(),
                control_exclusion_mask=control_exclusion_mask.detach().clone(),
                track_target_positions=target_points.detach().clone(),
                track_valid_mask=track_valid.detach().clone(),
            )
        )
        maximum = updater.candidate_parameter_step_maximum(candidate)
        if maximum <= 1.0e-12:
            reason = gate_reason if gate_paused else "graph_warmup_or_zero_step"
            return updater.reject(reason, candidate)
        gradient = candidate.autograd_parameter_gradient
        if gradient is None:
            return updater.reject("missing_graph_autograd_gradient", candidate)
        predicted_directional, measured_directional = (
            self._sim_graph_directional_derivatives(gradient)
        )
        allowed, _cosine = updater.apply_particle_graph_directional_gate(
            candidate, predicted_directional, measured_directional
        )
        if not allowed:
            return updater.reject(
                "particle_graph_directional_cosine_below_"
                f"{updater.settings.graph_directional_cosine_minimum:.2f}",
                candidate,
            )
        updater.replace_candidate_step_with_particle_graph_lm(candidate)
        metrics = updater.commit(candidate)
        metrics.update(
            validation_status="causal_particle_graph_lm_commit",
            admission_mode="sim_particle_graph_lm",
            update_policy=(
                "one_global+one_local_log_distance_per_particle;"
                "weak_tied_shape;fixed_volume+damping;"
                "alltracker_cauchy_relative_h1_h3_h5;"
                "observation_weighted_graph_lm;two_warp_directional_probes"
            ),
            source_frame=int(source_frame),
            destination_frame=int(destination_frame),
            future_observation_used=0,
            observability_contact_count=contact_count,
            observability_persistent_grip_active=int(
                persistent_grip_active
            ),
            observability_edge_strain_particles=strain_particle_count,
        )
        self._last_stiffness_commit_frame_index = int(destination_frame)
        return metrics

    def _process_flow_depth_trajectory_frame(self) -> dict | None:
        """Apply at most one causal track-flow observation on this video frame."""

        if self.visual_feedback_mode not in TRAJECTORY_FEEDBACK_MODES:
            return None
        frame_index = int(self.current_frame_index)
        if frame_index <= self._last_flow_depth_runtime_frame_index:
            return None
        self._last_flow_depth_runtime_frame_index = frame_index
        assert self.flow_depth_bindings is not None
        assert self.flow_depth_observations is not None
        assert self.visual_residual_mapper is not None

        contact_metrics = self.environment.sim.triangle_skin_contact_metrics() or {}
        stiffness_metrics = None
        stiffness_gate_paused = True
        stiffness_gate_reason = "stiffness_off"
        stiffness_jaw_speed = float("inf")
        stiffness_grip_active = False
        if self.stiffness_updater is not None:
            if STIFFNESS_ADMISSION_MODE in {
                "direct_online",
                "relaxed_h135",
                "paper_trajectory_adam",
            }:
                # The accepted trajectory innovation is already locally
                # projected against inversions.  Do not let contact-patch
                # transitions clear its material evidence or impose a frame
                # cooldown in the direct/relaxed ablations.  relaxed_h135
                # still validates its material candidate at H1/H3/H5.
                stiffness_gate_paused = False
                stiffness_gate_reason = ""
                stiffness_jaw_speed = 0.0
                stiffness_grip_active = bool(
                    contact_metrics.get("persistent_grip_active", False)
                )
            else:
                (
                    stiffness_gate_paused,
                    stiffness_gate_reason,
                    stiffness_jaw_speed,
                    stiffness_grip_active,
                ) = self._stiffness_global_gate(contact_metrics)
            stiffness_metrics = self._validate_pending_stiffness(
                gate_paused=stiffness_gate_paused,
                gate_reason=stiffness_gate_reason,
                grip_active=stiffness_grip_active,
            )
            self._advance_committed_stiffness_evaluations()

        pair_index = self.flow_depth_observations.pair_index_for_next_frame(
            frame_index
        )
        if pair_index is None:
            return stiffness_metrics
        source_frame = int(
            self.flow_depth_observations.current_source_frames[pair_index]
        )
        if not (
            self._benchmark_observation_enabled
            and self._flow_depth_observation_allowed(source_frame)
            and self._flow_depth_observation_allowed(frame_index)
        ):
            self._record_flow_depth_runtime_metrics(
                source_frame=source_frame,
                destination_frame=frame_index,
                status="withheld_by_causal_protocol",
                stiffness_metrics=stiffness_metrics,
            )
            return stiffness_metrics
        current_positions = self._flow_depth_source_states.get(source_frame)
        if current_positions is None:
            self._record_flow_depth_runtime_metrics(
                source_frame=source_frame,
                destination_frame=frame_index,
                status="missing_source_state",
                stiffness_metrics=stiffness_metrics,
            )
            return stiffness_metrics

        sim = self.environment.sim
        predicted_positions_torch = wp.to_torch(
            sim.state_0.particle_q
        ).detach().clone()
        predicted_velocities_torch = wp.to_torch(
            sim.state_0.particle_qd
        ).detach().clone()
        inverse_masses = wp.to_torch(
            sim.model.particle_inv_mass
        ).detach().cpu().numpy().copy()
        control_exclusion_mask = grip_control_exclusion_mask(
            self.visual_residual_mapper, self.environment
        )
        observation_dt_s = float(
            self.playback_timestamps[frame_index]
            - self.playback_timestamps[source_frame]
        )
        update = compute_flow_depth_particle_state_update(
            bindings=self.flow_depth_bindings,
            observation=self.flow_depth_observations.observation(pair_index),
            current_positions=current_positions,
            predicted_positions=predicted_positions_torch.cpu().numpy(),
            predicted_velocities=predicted_velocities_torch.cpu().numpy(),
            particle_inverse_masses=inverse_masses,
            observation_dt_s=observation_dt_s,
            reference_range_centers=(
                self._flow_depth_reference_range_centers
            ),
            dynamic_exclusion_mask=control_exclusion_mask.detach().cpu().numpy(),
            settings=self.flow_depth_settings,
        )
        accepted = sim.apply_flow_depth_particle_state_update(
            update,
            mapper=self.visual_residual_mapper,
            maximum_penetration_m=None,
            maximum_backtracks=0,
            maximum_local_inversion_projection_passes=16,
            enforce_inversion_gate=False,
            enforce_low_volume_gate=False,
            enforce_anchor_gate=False,
            enforce_penetration_gate=False,
        )
        update_metrics = dict(sim.last_flow_depth_particle_update_metrics)
        self._flow_depth_update_count += 1
        update_metrics["update_count"] = self._flow_depth_update_count
        update_metrics["source_frame"] = source_frame
        update_metrics["destination_frame"] = frame_index

        if accepted and self.stiffness_updater is not None:
            corrected_positions = wp.to_torch(
                sim.state_0.particle_q
            ).detach().clone()
            accepted_residual = corrected_positions - predicted_positions_torch
            installed = InstalledParticleInnovation(
                corrected_positions=corrected_positions,
                residual=accepted_residual,
            )
            quality_valid_mask = stiffness_local_quality_mask(
                self.visual_residual_mapper,
                predicted_positions_torch,
                corrected_positions,
                STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO,
            )
            confidence_support = torch.as_tensor(
                update.particle_confidence_support,
                device=corrected_positions.device,
                dtype=corrected_positions.dtype,
            )
            supervision_valid_mask = (
                (confidence_support > 0.0)
                & (
                    torch.linalg.vector_norm(accepted_residual, dim=1)
                    > 0.0
                )
            )
            if STIFFNESS_ADMISSION_MODE == "sim_particle_graph_lm":
                stiffness_metrics = self._run_sim_particle_graph_update(
                    source_frame=source_frame,
                    destination_frame=frame_index,
                    predicted_positions=predicted_positions_torch,
                    corrected_positions=corrected_positions,
                    accepted_residual=accepted_residual,
                    quality_valid_mask=quality_valid_mask,
                    supervision_valid_mask=supervision_valid_mask,
                    control_exclusion_mask=control_exclusion_mask,
                    contact_metrics=contact_metrics,
                    gate_paused=stiffness_gate_paused,
                    gate_reason=stiffness_gate_reason,
                )
            elif self.paper_trajectory_stiffness_optimizer is not None:
                material_valid_mask = (
                    quality_valid_mask
                    & supervision_valid_mask
                    & ~control_exclusion_mask
                    & ~self.visual_residual_mapper.fixed_mask
                )
                edge_strain_observable_mask = (
                    self.stiffness_updater.edge_strain_observability_mask(
                        prediction=predicted_positions_torch,
                        corrected=corrected_positions,
                        eligible_mask=material_valid_mask,
                    )
                )
                local_distance_signal, _local_distance_support = (
                    self.stiffness_updater._edge_strain_signal(
                        predicted_positions_torch,
                        corrected_positions,
                        material_valid_mask,
                    )
                )
                edge_strain_observable_particles = int(
                    torch.count_nonzero(
                        edge_strain_observable_mask
                    ).item()
                )
                material_contact_count = int(
                    contact_metrics.get("contact_count", 0)
                )
                material_observable = bool(
                    material_contact_count
                    >= STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT
                    and edge_strain_observable_particles
                    >= STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES
                )
                update_metrics["material_supervision_particles"] = int(
                    torch.count_nonzero(supervision_valid_mask).item()
                )
                update_metrics["material_active_particles"] = int(
                    torch.count_nonzero(material_valid_mask).item()
                )
                update_metrics["material_contact_count"] = (
                    material_contact_count
                )
                update_metrics["material_edge_strain_particles"] = (
                    edge_strain_observable_particles
                )
                update_metrics["material_observable"] = material_observable
                if material_observable:
                    stiffness_metrics = self._run_paper_trajectory_adam_update(
                        source_frame=source_frame,
                        destination_frame=frame_index,
                        active_mask=material_valid_mask,
                        accepted_positions=corrected_positions,
                        observation_supervision_valid=True,
                        local_distance_signal=local_distance_signal,
                    )
                    stiffness_metrics.update(
                        observability_contact_count=(
                            material_contact_count
                        ),
                        observability_minimum_contact_count=(
                            STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT
                        ),
                        observability_edge_strain_particles=(
                            edge_strain_observable_particles
                        ),
                        observability_minimum_edge_strain_particles=(
                            STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES
                        ),
                        observability_ready=True,
                    )
                elif (
                    self.paper_trajectory_stiffness_optimizer.settings
                    .causal_maximum_missing_observations
                    > 0
                ):
                    # Preserve one physical transition as an unsupervised gap.
                    # Its commands are replayed by a later 3-of-4 H3 window,
                    # but it contributes no visual target or material signal.
                    stiffness_metrics = self._run_paper_trajectory_adam_update(
                        source_frame=source_frame,
                        destination_frame=frame_index,
                        active_mask=material_valid_mask,
                        accepted_positions=corrected_positions,
                        observation_supervision_valid=False,
                        local_distance_signal=torch.zeros_like(
                            local_distance_signal
                        ),
                    )
                    stiffness_metrics.update(
                        observability_contact_count=material_contact_count,
                        observability_minimum_contact_count=(
                            STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT
                        ),
                        observability_edge_strain_particles=(
                            edge_strain_observable_particles
                        ),
                        observability_minimum_edge_strain_particles=(
                            STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES
                        ),
                        observability_ready=False,
                        causal_unsupervised_gap=True,
                    )
                else:
                    stiffness_metrics = {
                        "status": "material_unobservable",
                        "source_frame": int(source_frame),
                        "destination_frame": int(frame_index),
                        "active_particles": int(
                            torch.count_nonzero(material_valid_mask).item()
                        ),
                        "observability_contact_count": (
                            material_contact_count
                        ),
                        "observability_minimum_contact_count": (
                            STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT
                        ),
                        "observability_edge_strain_particles": (
                            edge_strain_observable_particles
                        ),
                        "observability_minimum_edge_strain_particles": (
                            STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES
                        ),
                        "observation_frequency": (
                            "every_material_observable_video_observation"
                        ),
                        "future_observation_used": 0,
                    }
                    self.paper_trajectory_stiffness_optimizer.last_metrics = (
                        dict(stiffness_metrics)
                    )
                    self.stiffness_updater.last_metrics = dict(
                        stiffness_metrics
                    )
                    self._append_paper_adam_history_snapshot(
                        frame_index=frame_index,
                        accepted_positions=corrected_positions,
                    )
            else:
                stiffness_metrics = (
                    self._consume_confirmed_visual_stiffness_evidence(
                        applied_result=installed,
                        quality_valid_mask=quality_valid_mask,
                        supervision_valid_mask=supervision_valid_mask,
                        control_exclusion_mask=control_exclusion_mask,
                        stiffness_gate_paused=stiffness_gate_paused,
                        stiffness_jaw_speed=stiffness_jaw_speed,
                        stiffness_grip_active=stiffness_grip_active,
                        stiffness_contact_count=int(
                            contact_metrics.get("contact_count", 0)
                        ),
                        accepted_positions=corrected_positions,
                        stiffness_metrics=stiffness_metrics,
                    )
                )
        self._record_flow_depth_runtime_metrics(
            source_frame=source_frame,
            destination_frame=frame_index,
            status="accepted" if accepted else "rejected",
            update_metrics=update_metrics,
            stiffness_metrics=stiffness_metrics,
        )
        if (
            self._flow_depth_update_count <= 3
            or self._flow_depth_update_count % 30 == 1
            or not accepted
        ):
            print(
                "[flow-depth trajectory] "
                f"frame={source_frame}->{frame_index}, accepted={accepted}, "
                f"tracks={int(update.track_valid.sum())}, "
                f"particles={update_metrics['updated_particles']}, "
                "requested_max="
                f"{float(update_metrics.get('requested_maximum_position_correction_m', 0.0)) * 1e3:.3f}mm, "
                f"max={float(update_metrics['maximum_position_correction_m']) * 1e3:.3f}mm, "
                f"scale={float(update_metrics['accepted_scale']):g}, "
                "local_suppressed="
                f"{int(update_metrics.get('locally_suppressed_particles', 0))}, "
                "reason="
                f"{str(update_metrics.get('rejection_reason', '')) or 'none'}, "
                f"stiffness={str((stiffness_metrics or {}).get('status', 'off'))}"
            )
        return stiffness_metrics

    def _trajectory_appearance_formal_inputs(
        self, frame_index: int
    ) -> tuple[tuple[int, ...], torch.Tensor]:
        """Build the exact left-camera non-instrument benchmark mask."""

        frames = self.environment.frames
        assert frames is not None
        camera_name = (
            "stereo_left"
            if "stereo_left" in frames.names
            else frames.names[0]
        )
        camera_index = frames.names.index(camera_name)
        recorder = self.tissue_benchmark_recorder
        if recorder is None:
            if frames.loss_weights_gpu is None:
                weights = torch.ones(
                    (1, frames.height, frames.width),
                    device=frames.colors_gpu.device,
                    dtype=torch.float32,
                )
            else:
                weights = frames.loss_weights_gpu[camera_index : camera_index + 1]
            return (camera_index,), weights
        masks = recorder.instrument_masks.masks_for_camera_frame(
            camera_name, int(frame_index)
        )
        if masks is None:
            raise RuntimeError(
                f"No formal appearance instrument mask for frame {frame_index}"
            )
        instrument, _distal = masks
        weights = torch.as_tensor(
            ~instrument,
            device=frames.colors_gpu.device,
            dtype=torch.float32,
        )[None]
        if weights.shape[1:] != (frames.height, frames.width):
            weights = torch.nn.functional.interpolate(
                weights[:, None],
                size=(frames.height, frames.width),
                mode="nearest",
            )[:, 0]
        return (camera_index,), weights.contiguous()

    def _process_trajectory_rgb_residual_frame(self) -> dict | None:
        """Apply one small RGB correction after trajectory/material updates.

        The correction is an image-state refinement only.  Its innovation is
        never passed to the material updater, and the fixed barycentric
        AllTracker bindings penalize any displacement of already corrected
        trajectory centres.  Benchmark observation gating makes the method
        causal for both reconstruction holdouts and the 80/20 future split.
        """

        if self.visual_feedback_mode != "trajectory_residual":
            return None
        frame_index = int(self.current_frame_index)
        if frame_index <= self._last_trajectory_rgb_residual_frame_index:
            return None
        self._last_trajectory_rgb_residual_frame_index = frame_index
        if (
            not self.playing
            or self.environment.frames is None
            or not self._benchmark_observation_enabled
            or not self._flow_depth_observation_allowed(frame_index)
            or self._flow_depth_update_count == 0
        ):
            return None
        mapper = self.visual_residual_mapper
        bindings = self.flow_depth_bindings
        sequence = self.flow_depth_observations
        indices = self._trajectory_rgb_particle_indices
        weights = self._trajectory_rgb_particle_weights
        assert mapper is not None
        assert bindings is not None and sequence is not None
        assert indices is not None and weights is not None

        available_pairs = np.flatnonzero(
            sequence.next_source_frames <= frame_index
        )
        if not len(available_pairs):
            return None
        pair_index = int(available_pairs[-1])
        confidence_np = sequence.confidence[pair_index]
        valid_np = (
            bindings.track_valid
            & sequence.track_valid[pair_index]
            & np.isfinite(confidence_np)
            & (confidence_np > 0.0)
        )
        if not np.any(valid_np):
            return None
        device = mapper.rest_positions.device
        confidence = torch.as_tensor(
            confidence_np, device=device, dtype=torch.float32
        )
        valid = torch.as_tensor(valid_np, device=device, dtype=torch.bool)
        control_exclusion = grip_control_exclusion_mask(
            mapper, self.environment
        )
        sim = self.environment.sim
        sim.update_gaussian_transforms()
        solve_start = time.perf_counter()
        result = sim.solve_visual_tissue_residual(
            mapper,
            self.environment.frames,
            previous_residual=self._previous_visual_residual,
            dynamic_exclusion_mask=control_exclusion,
            observations_are_bgr=True,
            trajectory_particle_indices=indices,
            trajectory_particle_weights=weights,
            trajectory_confidence=confidence,
            trajectory_valid_mask=valid,
            trajectory_hold_weight=self.trajectory_rgb_residual_track_weight,
        )
        position_gain = self.trajectory_rgb_residual_position_gain
        if position_gain < 1.0:
            uncorrected_positions = (
                result.corrected_positions - result.residual
            )
            scaled_residual = result.residual * position_gain
            corrected_positions = uncorrected_positions + scaled_residual
            if mapper.deformation_covariance_enabled:
                corrected_gaussian_means, _ = (
                    mapper.deformed_gaussian_geometry(corrected_positions)
                )
            else:
                corrected_gaussian_means = mapper.corrected_gaussian_means(
                    sim.gaussian_state.means.detach()[
                        mapper.soft_gaussian_ids
                    ],
                    scaled_residual,
                )
            scaled_quality = mapper.physical_quality_metrics(
                corrected_positions
            )
            result = replace(
                result,
                residual=scaled_residual,
                corrected_positions=corrected_positions,
                corrected_gaussian_means=corrected_gaussian_means,
                trajectory_hold_rms_m=(
                    result.trajectory_hold_rms_m * position_gain
                ),
                trajectory_hold_maximum_m=(
                    result.trajectory_hold_maximum_m * position_gain
                ),
                maximum_residual_m=(
                    result.maximum_residual_m * position_gain
                ),
                rms_residual_m=result.rms_residual_m * position_gain,
                minimum_volume_ratio=float(
                    scaled_quality["minimum_volume_ratio"]
                ),
                inverted_tetrahedra=int(
                    scaled_quality["inverted_tetrahedra"]
                ),
                newly_inverted_tetrahedra=max(
                    0,
                    int(scaled_quality["inverted_tetrahedra"])
                    - int(result.initial_inverted_tetrahedra),
                ),
            )
        previous_frame = int(self._last_visual_residual_frame_index)
        if 0 <= previous_frame < frame_index:
            observation_dt_s = float(
                self.playback_timestamps[frame_index]
                - self.playback_timestamps[previous_frame]
            )
        elif frame_index > 0:
            observation_dt_s = float(
                self.playback_timestamps[frame_index]
                - self.playback_timestamps[frame_index - 1]
            )
        else:
            observation_dt_s = 1.0 / float(self.fps)
        accepted = sim.apply_visual_tissue_residual(
            result,
            mapper=mapper,
            frames=self.environment.frames,
            observations_are_bgr=True,
            observation_dt_s=max(observation_dt_s, 1.0e-6),
            velocity_correction_gain=(
                self.trajectory_rgb_residual_velocity_gain
            ),
            maximum_velocity_correction_m_s=(
                self.trajectory_rgb_residual_maximum_velocity_m_s
            ),
            dynamic_exclusion_mask=control_exclusion,
        )
        appearance_result = None
        appearance_elapsed_s = 0.0
        appearance_commit_accepted = False
        appearance_cross_frame_improvement = None
        if self.trajectory_appearance_settings.iterations > 0:
            appearance_start = time.perf_counter()
            camera_indices, formal_weights = (
                self._trajectory_appearance_formal_inputs(frame_index)
            )
            evaluation_settings = replace(
                self.trajectory_appearance_settings, iterations=0
            )
            # A proposal trained on frame t is never installed immediately.
            # At the next legal observed frame, compare it with the currently
            # installed render appearance using the formal held-out-style loss.
            pending = self._pending_trajectory_appearance_result
            if (
                pending is not None
                and frame_index
                > self._pending_trajectory_appearance_frame_index
            ):
                baseline_appearance = (
                    sim.refine_trajectory_gaussian_appearance(
                        mapper,
                        self.environment.frames,
                        reference_colors_logits=(
                            self._trajectory_appearance_reference_colors_logits
                        ),
                        reference_opacities_logits=(
                            self._trajectory_appearance_reference_opacities_logits
                        ),
                        settings=evaluation_settings,
                        observations_are_bgr=True,
                        camera_indices=camera_indices,
                        loss_weights_override=formal_weights,
                        image_scale=(
                            self.trajectory_appearance_settings
                            .formal_image_scale
                        ),
                        dssim_weight=(
                            self.trajectory_appearance_settings.dssim_weight
                        ),
                        apply_result=False,
                    )
                )
                ids = mapper.soft_gaussian_ids
                current_colors = (
                    sim.gaussian_state.colors_logits.detach()[ids].clone()
                )
                current_opacities = (
                    sim.gaussian_state.opacities_logits.detach()[ids].clone()
                )
                try:
                    with torch.no_grad():
                        sim.gaussian_state.colors_logits.index_copy_(
                            0, ids, pending.colors_logits
                        )
                        sim.gaussian_state.opacities_logits.index_copy_(
                            0, ids, pending.opacities_logits
                        )
                    pending_appearance = (
                        sim.refine_trajectory_gaussian_appearance(
                            mapper,
                            self.environment.frames,
                            reference_colors_logits=(
                                self._trajectory_appearance_reference_colors_logits
                            ),
                            reference_opacities_logits=(
                                self._trajectory_appearance_reference_opacities_logits
                            ),
                            settings=evaluation_settings,
                            observations_are_bgr=True,
                            camera_indices=camera_indices,
                            loss_weights_override=formal_weights,
                            image_scale=(
                                self.trajectory_appearance_settings
                                .formal_image_scale
                            ),
                            dssim_weight=(
                                self.trajectory_appearance_settings.dssim_weight
                            ),
                            apply_result=False,
                        )
                    )
                finally:
                    with torch.no_grad():
                        sim.gaussian_state.colors_logits.index_copy_(
                            0, ids, current_colors
                        )
                        sim.gaussian_state.opacities_logits.index_copy_(
                            0, ids, current_opacities
                        )
                appearance_cross_frame_improvement = (
                    baseline_appearance.initial_visual_loss
                    - pending_appearance.initial_visual_loss
                )
                camera_safe = all(
                    candidate <= baseline
                    for baseline, candidate in zip(
                        baseline_appearance.initial_camera_visual_losses,
                        pending_appearance.initial_camera_visual_losses,
                    )
                )
                appearance_commit_accepted = bool(
                    camera_safe
                    and appearance_cross_frame_improvement
                    > self.trajectory_appearance_settings.minimum_visual_improvement
                )
                if appearance_commit_accepted:
                    with torch.no_grad():
                        sim.gaussian_state.colors_logits.index_copy_(
                            0, ids, pending.colors_logits
                        )
                        sim.gaussian_state.opacities_logits.index_copy_(
                            0, ids, pending.opacities_logits
                        )
                    self._trajectory_appearance_accept_count += 1
                self._pending_trajectory_appearance_result = None
                self._pending_trajectory_appearance_frame_index = -1
            appearance_result = sim.refine_trajectory_gaussian_appearance(
                mapper,
                self.environment.frames,
                reference_colors_logits=(
                    self._trajectory_appearance_reference_colors_logits
                ),
                reference_opacities_logits=(
                    self._trajectory_appearance_reference_opacities_logits
                ),
                settings=self.trajectory_appearance_settings,
                observations_are_bgr=True,
                camera_indices=camera_indices,
                loss_weights_override=formal_weights,
                image_scale=(
                    self.trajectory_appearance_settings.formal_image_scale
                ),
                dssim_weight=self.trajectory_appearance_settings.dssim_weight,
                apply_result=False,
            )
            appearance_elapsed_s = time.perf_counter() - appearance_start
            self._trajectory_appearance_solve_count += 1
            if appearance_result.accepted:
                self._pending_trajectory_appearance_result = appearance_result
                self._pending_trajectory_appearance_frame_index = frame_index
        self._last_visual_residual_frame_index = frame_index
        self._trajectory_rgb_residual_solve_count += 1
        self._previous_visual_residual = (
            result.residual.detach().clone() if accepted else None
        )
        # Appearance gsplat runs on Torch CUDA streams while XPBD runs on Warp
        # streams.  Fence both paths even when appearance is disabled so the
        # presence of an appearance pass cannot change when the next physics
        # frame observes shared Gaussian/particle buffers.
        self._synchronize_shadow_transaction(sim)
        metrics = dict(sim.last_visual_tissue_residual_metrics or {})
        metrics.update(
            solve_elapsed_s=time.perf_counter() - solve_start,
            solve_count=self._trajectory_rgb_residual_solve_count,
            stage="post_trajectory_rgb_micro_residual",
            material_evidence_used=False,
            trajectory_hold_weight=(
                self.trajectory_rgb_residual_track_weight
            ),
            trajectory_rgb_position_gain=position_gain,
            trajectory_hold_track_count=(
                result.trajectory_hold_track_count
            ),
            trajectory_hold_rms_m=result.trajectory_hold_rms_m,
            trajectory_hold_maximum_m=(
                result.trajectory_hold_maximum_m
            ),
            gaussian_appearance_enabled=(appearance_result is not None),
            gaussian_appearance_accepted=(
                False
                if appearance_result is None
                else appearance_commit_accepted
            ),
            gaussian_appearance_proposal_accepted=(
                False
                if appearance_result is None
                else appearance_result.accepted
            ),
            gaussian_appearance_cross_frame_improvement=(
                appearance_cross_frame_improvement
            ),
            gaussian_appearance_iterations=(
                0
                if appearance_result is None
                else appearance_result.completed_iterations
            ),
            gaussian_appearance_accepted_iteration=(
                0
                if appearance_result is None
                else appearance_result.accepted_iteration
            ),
            gaussian_appearance_initial_visual_loss=(
                None
                if appearance_result is None
                else appearance_result.initial_visual_loss
            ),
            gaussian_appearance_final_visual_loss=(
                None
                if appearance_result is None
                else appearance_result.final_visual_loss
            ),
            gaussian_appearance_changed_gaussians=(
                0
                if appearance_result is None
                else appearance_result.changed_gaussians
            ),
            gaussian_appearance_color_logit_offset_maximum=(
                0.0
                if appearance_result is None
                else appearance_result.maximum_color_logit_offset
            ),
            gaussian_appearance_opacity_logit_offset_maximum=(
                0.0
                if appearance_result is None
                else appearance_result.maximum_opacity_logit_offset
            ),
            gaussian_appearance_solve_elapsed_s=appearance_elapsed_s,
            gaussian_appearance_solve_count=(
                self._trajectory_appearance_solve_count
            ),
            gaussian_appearance_accept_count=(
                self._trajectory_appearance_accept_count
            ),
        )
        sim.last_visual_tissue_residual_metrics = metrics
        contact = sim.triangle_skin_contact_metrics() or {}
        self._record_visual_stiffness_metrics(
            result=result,
            visual_metrics=metrics,
            stiffness_metrics=(
                None
                if self.stiffness_updater is None
                else self.stiffness_updater.last_metrics
            ),
            contact_metrics=contact,
            gate_paused=True,
            gate_reason="post_trajectory_rgb_not_material_evidence",
        )
        if (
            self._trajectory_rgb_residual_solve_count <= 3
            or self._trajectory_rgb_residual_solve_count % 30 == 0
            or not accepted
        ):
            print(
                "[trajectory RGB micro residual] "
                f"frame={frame_index}, accepted={accepted}, "
                f"loss={result.initial_visual_loss:.6f}->"
                f"{float(metrics.get('exact_final_visual_loss', result.final_visual_loss)):.6f}, "
                f"max={result.maximum_residual_m * 1e3:.3f}mm, "
                "track_hold_rms/max="
                f"{result.trajectory_hold_rms_m * 1e3:.4f}/"
                f"{result.trajectory_hold_maximum_m * 1e3:.4f}mm, "
                f"tracks={result.trajectory_hold_track_count}, "
                "appearance="
                + (
                    "off, "
                    if appearance_result is None
                    else (
                        f"proposal={appearance_result.accepted}, "
                        f"commit={appearance_commit_accepted}, "
                        f"loss={appearance_result.initial_visual_loss:.6f}->"
                        f"{appearance_result.final_visual_loss:.6f}, "
                        f"iter={appearance_result.accepted_iteration}/"
                        f"{appearance_result.completed_iterations}, "
                    )
                )
                + f"elapsed={metrics['solve_elapsed_s'] * 1e3:.1f}ms"
            )
        return metrics

    def _validate_pending_stiffness_continuation(
        self,
        *,
        gate_paused: bool,
        gate_reason: str,
        grip_active: bool,
    ) -> dict | None:
        """Continue short-horizon validation after the expiry pre-check."""

        pending = self._pending_stiffness_validation
        updater = self.stiffness_updater
        if pending is None or updater is None:
            return None
        validation_horizons = STIFFNESS_ADMISSION_HORIZONS
        horizon_frames = self.current_frame_index - pending.frame_index
        validation_frames = self._pending_stiffness_validation_frames(pending)
        required_frame_index = (
            validation_frames[-1]
            if len(validation_frames) == len(validation_horizons)
            else pending.frame_index
            + STIFFNESS_COMMIT_VALIDATION_HORIZON_FRAMES
        )
        required_frame_span = required_frame_index - pending.frame_index
        if self.current_frame_index <= pending.frame_index:
            return None
        if (
            STIFFNESS_ADMISSION_MODE
            not in {"causal_fixed_lag", "relaxed_h135"}
            and (gate_paused or grip_active != pending.grip_active)
        ):
            reason = gate_reason or "grip_state_changed"
            if STIFFNESS_ADMISSION_MODE == "causal_fixed_lag":
                self._clear_stiffness_global_confirmation()
                self._clear_stiffness_local_confirmations()
            metrics = updater.reject(reason, pending.candidate)
            metrics.update(
                validation_status="rejected_before_rollout",
                prediction_horizon_frames=(
                    self.current_frame_index - pending.frame_index
                ),
            )
            self._record_terminal_stiffness_validation(
                pending, metrics, reason
            )
            self._pending_stiffness_validation = None
            self._last_stiffness_validation_metrics = metrics
            return metrics
        if self.current_frame_index < required_frame_index:
            metrics = dict(pending.candidate.metrics)
            metrics.update(
                status="pending_horizon",
                validation_status="pending_horizon",
                prediction_horizon_frames=horizon_frames,
                prediction_rollout_steps=len(pending.commands),
                required_prediction_horizon_frames=(
                    required_frame_span
                ),
                validation_frame_indices=validation_frames,
                validation_horizons=validation_horizons,
                validation_horizon_axis=(
                    "observable_alltracker_training_frames"
                    if getattr(self, "visual_feedback_mode", None)
                    == "trajectory"
                    else (
                        "observable_training_frames"
                        if getattr(self, "tissue_benchmark_recorder", None)
                        is not None
                        and getattr(
                            self, "tissue_benchmark_recorder"
                        ).protocol
                        == "reconstruction_7to1"
                        else "video_frames"
                    )
                ),
            )
            self._last_stiffness_validation_metrics = metrics
            return metrics
        if STIFFNESS_ADMISSION_MODE in {
            "causal_fixed_lag",
            "relaxed_h135",
        }:
            return self._validate_direct_residual_gradient_stiffness(pending)
        self._synchronize_shadow_transaction(self.environment.sim)
        live = self.environment.sim.clone_embodied_gaussian_rollout_state()
        controller_state = (
            self.current_frame_index,
            self.current_timestep,
            self._last_state_index,
            self._last_q_full.detach().clone(),
        )
        base_candidate = pending.candidate
        verified_distance = updater.distance_stiffness.detach().clone()
        verified_shape = updater.shape_stiffness.detach().clone()
        candidate_search_records: list[dict] = []
        try:
            baseline = self._run_stiffness_prediction_shadow(
                pending, use_candidate=False
            )
            for (
                variant_label,
                distance_scale,
                shape_scale,
                scope,
            ) in STIFFNESS_CANDIDATE_VARIANTS:
                updater.restore_verified_stiffness(
                    verified_distance, verified_shape
                )
                variant = self._stiffness_candidate_variant(
                    base_candidate,
                    distance_scale=distance_scale,
                    shape_scale=shape_scale,
                    variant_label=variant_label,
                    scope=scope,
                )
                self._assert_stiffness_candidate_axis_isolation(
                    variant,
                    verified_distance=verified_distance,
                    verified_shape=verified_shape,
                    may_change_distance=distance_scale != 0.0,
                    may_change_shape=shape_scale != 0.0,
                )
                pending.candidate = variant
                shadow = self._run_stiffness_prediction_shadow(
                    pending, use_candidate=True
                )
                scope_fraction = self._stiffness_candidate_scope_fraction(
                    variant
                )
                variant_reasons, variant_improvement, required_improvement = (
                    self._stiffness_shadow_rejection_reasons(
                        baseline,
                        shadow,
                        absolute_margin_scale=(
                            self._stiffness_candidate_margin_scale(
                                scope=scope,
                                scope_fraction=scope_fraction,
                            )
                        ),
                    )
                )
                if (
                    int(variant.metrics["candidate_scope_particles"])
                    < STIFFNESS_EVENT_DRIVEN_MINIMUM_SCOPE_PARTICLES
                ):
                    variant_reasons.append(
                        "material_support_too_small"
                    )
                candidate_search_records.append(
                    {
                        "variant_label": variant_label,
                        "distance_scale": float(distance_scale),
                        "shape_scale": float(shape_scale),
                        "scope": scope,
                        "scope_fraction": scope_fraction,
                        "candidate": variant,
                        "shadow": shadow,
                        "reasons": variant_reasons,
                        "improvement": variant_improvement,
                        "required_improvement": required_improvement,
                    }
                )
            if STIFFNESS_CANDIDATE_PROFILE in {
                "causal_fixed_lag_12",
                "hierarchical_system_id",
                "robust_hierarchical_system_id",
            }:
                global_targets = [
                    (
                        f"global_distance_{value:g}",
                        float(value),
                        None,
                    )
                    for value in STIFFNESS_GLOBAL_DISTANCE_MEDIANS
                ] + [
                    (
                        f"global_shape_{value:g}",
                        None,
                        float(value),
                    )
                    for value in STIFFNESS_GLOBAL_SHAPE_MEDIANS
                ]
                material_settings = self.stiffness_updater.settings
                global_targets = [
                    (label, distance_target, shape_target)
                    for label, distance_target, shape_target in global_targets
                    if (
                        distance_target is not None
                        and material_settings.distance_minimum
                        <= distance_target
                        <= material_settings.distance_maximum
                    )
                    or (
                        shape_target is not None
                        and material_settings.shape_minimum
                        <= shape_target
                        <= material_settings.shape_maximum
                    )
                ]
                if (
                    STIFFNESS_CANDIDATE_PROFILE
                    in {
                        "causal_fixed_lag_12",
                        "robust_hierarchical_system_id",
                    }
                ):
                    # Shadow calls are transactional, but make this read-side
                    # dependency explicit too: trust-region filtering must
                    # always start from the verified global medians.
                    updater.restore_verified_stiffness(
                        verified_distance, verified_shape
                    )
                    material_mask = base_candidate.material_valid_mask
                    current_distance = float(
                        self.stiffness_updater.distance_stiffness[
                            material_mask
                        ].median().item()
                    )
                    current_shape = float(
                        self.stiffness_updater.shape_stiffness[
                            material_mask
                        ].median().item()
                    )
                    if STIFFNESS_ADMISSION_MODE == "causal_fixed_lag":
                        # Four broad event-driven probes replace a long list
                        # of duplicate absolute targets.  The updater clips
                        # each global move to the configured 0.10 log trust
                        # step, while repeated safe commits can traverse the
                        # full material bounds in either direction.
                        step = float(material_settings.maximum_log_step)
                        global_targets = [
                            (
                                "global_distance_soften",
                                max(
                                    material_settings.distance_minimum,
                                    current_distance * float(np.exp(-step)),
                                ),
                                None,
                            ),
                            (
                                "global_distance_harden",
                                min(
                                    material_settings.distance_maximum,
                                    current_distance * float(np.exp(step)),
                                ),
                                None,
                            ),
                            (
                                "global_shape_soften",
                                None,
                                max(
                                    material_settings.shape_minimum,
                                    current_shape * float(np.exp(-step)),
                                ),
                            ),
                            (
                                "global_shape_harden",
                                None,
                                min(
                                    material_settings.shape_maximum,
                                    current_shape * float(np.exp(step)),
                                ),
                            ),
                        ]
                    else:
                        global_targets = [
                            (label, distance_target, shape_target)
                            for label, distance_target, shape_target in global_targets
                            if (
                                distance_target is not None
                                and not self._stiffness_global_distance_locked
                                and self._stiffness_global_distance_commits
                                < STIFFNESS_ROBUST_MAXIMUM_GLOBAL_COMMITS_PER_FAMILY
                                and stiffness_target_inside_global_trust_region(
                                    current_distance, distance_target
                                )
                                and stiffness_target_follows_global_direction(
                                    current_distance,
                                    distance_target,
                                    self._stiffness_global_distance_direction,
                                )
                            )
                            or (
                                shape_target is not None
                                and not self._stiffness_global_shape_locked
                                and self._stiffness_global_shape_commits
                                < STIFFNESS_ROBUST_MAXIMUM_GLOBAL_COMMITS_PER_FAMILY
                                and stiffness_target_inside_global_trust_region(
                                    current_shape, shape_target
                                )
                                and stiffness_target_follows_global_direction(
                                    current_shape,
                                    shape_target,
                                    self._stiffness_global_shape_direction,
                                )
                            )
                        ]
                for variant_label, distance_target, shape_target in global_targets:
                    updater.restore_verified_stiffness(
                        verified_distance, verified_shape
                    )
                    variant = self.stiffness_updater.global_median_candidate(
                        base_candidate,
                        distance_median=distance_target,
                        shape_median=shape_target,
                        variant_label=variant_label,
                    )
                    self._assert_stiffness_candidate_axis_isolation(
                        variant,
                        verified_distance=verified_distance,
                        verified_shape=verified_shape,
                        may_change_distance=distance_target is not None,
                        may_change_shape=shape_target is not None,
                    )
                    if float(variant.metrics["maximum_log_step"]) <= 1.0e-8:
                        continue
                    pending.candidate = variant
                    shadow = self._run_stiffness_prediction_shadow(
                        pending, use_candidate=True
                    )
                    variant_reasons, variant_improvement, required_improvement = (
                        self._stiffness_shadow_rejection_reasons(
                            baseline,
                            shadow,
                            absolute_margin_scale=1.0,
                        )
                    )
                    candidate_search_records.append(
                        {
                            "variant_label": variant_label,
                            "distance_scale": 0.0,
                            "shape_scale": 0.0,
                            "scope": "global_material_offset",
                            "scope_fraction": 1.0,
                            "candidate": variant,
                            "shadow": shadow,
                            "reasons": variant_reasons,
                            "improvement": variant_improvement,
                            "required_improvement": required_improvement,
                        }
                    )
            if STIFFNESS_CANDIDATE_PROFILE == "spatial_components_merge_12":
                for material_axis in ("distance", "shape"):
                    component_winners: list[dict] = []
                    peer_axis = (
                        "shape" if material_axis == "distance" else "distance"
                    )
                    for component_scope in ("component_0", "component_1"):
                        safe_records = [
                            record
                            for record in candidate_search_records
                            if record["scope"] == component_scope
                            and not record["reasons"]
                            and record[f"{material_axis}_scale"] != 0.0
                            and record[f"{peer_axis}_scale"] == 0.0
                        ]
                        if safe_records:
                            component_winners.append(
                                max(
                                    safe_records,
                                    key=lambda record: (
                                        record["improvement"]
                                        / record["scope_fraction"],
                                        record["improvement"],
                                    ),
                                )
                            )
                    if len(component_winners) != 2:
                        continue
                    updater.restore_verified_stiffness(
                        verified_distance, verified_shape
                    )
                    merged_label = (
                        f"merged_{material_axis}_component0_component1"
                    )
                    merged = self.stiffness_updater.merge_candidate_regions(
                        tuple(
                            record["candidate"]
                            for record in component_winners
                        ),
                        variant_label=merged_label,
                    )
                    self._assert_stiffness_candidate_axis_isolation(
                        merged,
                        verified_distance=verified_distance,
                        verified_shape=verified_shape,
                        may_change_distance=material_axis == "distance",
                        may_change_shape=material_axis == "shape",
                    )
                    pending.candidate = merged
                    merged_shadow = self._run_stiffness_prediction_shadow(
                        pending, use_candidate=True
                    )
                    merged_fraction = (
                        self._stiffness_candidate_scope_fraction(merged)
                    )
                    (
                        merged_reasons,
                        merged_improvement,
                        merged_required,
                    ) = self._stiffness_shadow_rejection_reasons(
                        baseline,
                        merged_shadow,
                        absolute_margin_scale=(
                            self._stiffness_candidate_margin_scale(
                                scope="merged_components",
                                scope_fraction=merged_fraction,
                            )
                        ),
                    )
                    candidate_search_records.append(
                        {
                            "variant_label": merged_label,
                            "distance_scale": (
                                1.0 if material_axis == "distance" else 0.0
                            ),
                            "shape_scale": (
                                1.0 if material_axis == "shape" else 0.0
                            ),
                            "scope": "merged_components",
                            "scope_fraction": merged_fraction,
                            "candidate": merged,
                            "shadow": merged_shadow,
                            "reasons": merged_reasons,
                            "improvement": merged_improvement,
                            "required_improvement": merged_required,
                            "merged_material_axis": material_axis,
                            "merged_from": tuple(
                                record["variant_label"]
                                for record in component_winners
                            ),
                        }
                    )
        finally:
            pending.candidate = base_candidate
            self._synchronize_shadow_transaction(self.environment.sim)
            self.environment.sim.copy_embodied_gaussian_rollout_state(live)
            updater.restore_verified_stiffness(
                verified_distance, verified_shape
            )
            (
                self.current_frame_index,
                self.current_timestep,
                self._last_state_index,
                self._last_q_full,
            ) = controller_state
            self.dataset_manager.update_frames(self.current_timestep)
            self.environment.sim.update_gaussian_transforms()
            self._synchronize_shadow_transaction(self.environment.sim)

        proposal_phase = str(
            getattr(pending, "proposal_phase", "idle") or "idle"
        )
        causal_commit_policy = self._apply_causal_stiffness_commit_policy(
            baseline=baseline,
            records=candidate_search_records,
            proposal_phase=proposal_phase,
            proposal_frame_index=int(pending.frame_index),
        )

        # Prefer the visually best physically safe signed/material-family
        # variant, then check its historical relaxation. If history rejects
        # it, fall back to the next best safe candidate.
        selected_record = None
        for record in sorted(
            (
                record
                for record in candidate_search_records
                if not record["reasons"]
            ),
            key=lambda record: self._stiffness_candidate_selection_key(
                baseline, record
            ),
        ):
            if STIFFNESS_ADMISSION_MODE == "causal_fixed_lag":
                # The event was observable when proposed and the replay uses
                # the recorded tool commands.  H30/H60, not a rest-state RMS,
                # is the causal long-term material test.
                history_baseline = pending.history_baseline_rms_m
                history_candidate = pending.history_candidate_rms_m
                history_safe = True
            elif "history_safe" in record:
                history_baseline = record["history_baseline_rms_m"]
                history_candidate = record["history_candidate_rms_m"]
                history_safe = bool(record["history_safe"])
            else:
                history_baseline, history_candidate, history_safe = (
                    self._evaluate_stiffness_history(record["candidate"])
                )
                record["history_baseline_rms_m"] = history_baseline
                record["history_candidate_rms_m"] = history_candidate
                record["history_safe"] = bool(history_safe)
            if history_safe:
                selected_record = record
                break
            record["reasons"] = [*record["reasons"], "history_regression"]

        if selected_record is None:
            # No candidate passed every safety/admission check.  Preserve the
            # closest measured alternative for truthful diagnostics and for
            # spatially targeted reject decay; profiles are not required to
            # contain the legacy joint-forward label.
            # A global target whose *only* remaining condition is the second
            # consensus window is deferred without evidence decay.  Prefer it
            # over a genuinely failed alternative so consensus is reachable.
            confirmation_pending_reasons = {
                "local_confirmation_pending",
                "global_confirmation_pending",
            }
            confirmation_pending_records = [
                record
                for record in candidate_search_records
                if len(record["reasons"]) == 1
                and record["reasons"][0] in confirmation_pending_reasons
            ]
            selected_record = min(
                confirmation_pending_records or candidate_search_records,
                key=lambda record: self._stiffness_candidate_selection_key(
                    baseline, record
                ),
            )
            history_baseline = pending.history_baseline_rms_m
            history_candidate = pending.history_candidate_rms_m
            history_safe = bool(
                STIFFNESS_ADMISSION_MODE == "causal_fixed_lag"
                or history_candidate
                <= history_baseline
                * (1.0 + STIFFNESS_HISTORY_RELATIVE_TOLERANCE)
                + STIFFNESS_HISTORY_ABSOLUTE_TOLERANCE_M
            )
            if (
                not history_safe
                and "history_regression" not in selected_record["reasons"]
            ):
                selected_record["reasons"].append("history_regression")
        elif STIFFNESS_ADMISSION_MODE != "causal_fixed_lag":
            history_baseline = selected_record["history_baseline_rms_m"]
            history_candidate = selected_record["history_candidate_rms_m"]

        pending.candidate = selected_record["candidate"]
        updater.pending_candidate = pending.candidate
        pending.history_baseline_rms_m = float(history_baseline)
        pending.history_candidate_rms_m = float(history_candidate)
        pending.candidate.metrics.update(
            selected_candidate_variant=selected_record["variant_label"],
            selected_distance_scale=selected_record["distance_scale"],
            selected_shape_scale=selected_record["shape_scale"],
            selected_candidate_scope=selected_record["scope"],
            selected_candidate_scope_fraction=selected_record[
                "scope_fraction"
            ],
            stiffness_proposal_phase=proposal_phase,
            selected_local_phase_family=selected_record.get(
                "local_phase_family"
            ),
            selected_local_phase_commit_count=selected_record.get(
                "local_phase_commit_count"
            ),
            selected_local_phase_family_last_commit_frame=selected_record.get(
                "local_phase_family_last_commit_frame"
            ),
            selected_local_phase_family_commit_interval_frames=(
                selected_record.get(
                    "local_phase_family_commit_interval_frames"
                )
            ),
            selected_local_confirmation_direction=selected_record.get(
                "local_confirmation_direction"
            ),
            selected_local_confirmation_count=selected_record.get(
                "local_confirmation_count"
            ),
            selected_local_confirmation_support_iou=selected_record.get(
                "local_confirmation_support_iou"
            ),
            selected_local_confirmation_scope=selected_record.get(
                "local_confirmation_scope"
            ),
            selected_global_confirmation_count=selected_record.get(
                "global_confirmation_count"
            ),
        )
        candidate = selected_record["shadow"]
        improvement = float(selected_record["improvement"])
        residual_effort_improvement = (
            self._stiffness_residual_effort_improvement(
                baseline, candidate
            )
        )
        required_improvement = float(
            selected_record["required_improvement"]
        )
        reasons = list(selected_record["reasons"])
        if (
            not reasons
            and STIFFNESS_ADMISSION_MODE == "causal_fixed_lag"
        ):
            # The full search ends at H10.  Freeze only its winner and replay
            # that one candidate (plus baseline) to H30/H60 before any live
            # material write.  New evidence may continue accumulating while
            # this independent long window is pending.
            pending.long_validation_candidate = pending.candidate
            pending.short_selection_record = {
                key: selected_record[key]
                for key in (
                    "variant_label",
                    "distance_scale",
                    "shape_scale",
                    "scope",
                    "scope_fraction",
                    "improvement",
                    "required_improvement",
                )
            }
            pending.candidate.metrics.update(
                status="pending_long_horizon",
                validation_status="pending_long_horizon",
                short_validation_horizons=STIFFNESS_ADMISSION_HORIZONS,
                short_baseline_horizon_visual_losses=baseline[
                    "horizon_visual_losses"
                ],
                short_candidate_horizon_visual_losses=candidate[
                    "horizon_visual_losses"
                ],
                short_prediction_improvement=improvement,
                short_required_prediction_improvement=(
                    required_improvement
                ),
                long_validation_horizons=(
                    STIFFNESS_EVENT_DRIVEN_LONG_HORIZONS
                ),
                long_validation_frame_indices=(
                    pending.long_validation_frame_indices
                ),
            )
            metrics = dict(pending.candidate.metrics)
            updater.last_metrics = metrics
            self._last_stiffness_validation_metrics = metrics
            return metrics
        if len(reasons) == 1 and reasons[0] in {
            "local_confirmation_pending",
            "global_confirmation_pending",
        }:
            metrics = updater.defer(
                reasons[0], pending.candidate
            )
        elif reasons:
            metrics = updater.reject("+".join(reasons), pending.candidate)
        else:
            previous_distance_median = float(
                updater.distance_stiffness.median().item()
            )
            previous_shape_median = float(
                updater.shape_stiffness.median().item()
            )
            metrics = updater.commit(pending.candidate)
            if STIFFNESS_ADMISSION_MODE == "causal_fixed_lag":
                self._last_stiffness_commit_frame_index = int(
                    self.current_frame_index
                )
                if selected_record["scope"] != "global_material_offset":
                    selected_axis = self._stiffness_record_material_axis(
                        selected_record
                    )
                    phase_key = (proposal_phase, selected_axis)
                    phase_commits = self._stiffness_local_phase_family_commits
                    phase_commits[phase_key] = int(
                        phase_commits.get(phase_key, 0)
                    ) + 1
                    self._stiffness_local_phase_family_last_commit_frame[
                        phase_key
                    ] = int(self.current_frame_index)
                # Any commit changes the verified material baseline.  A
                # global confirmation collected before that change is stale
                # and must earn two fresh consecutive windows.
                self._clear_stiffness_global_confirmation()
                self._clear_stiffness_local_confirmations()
            if (
                STIFFNESS_CANDIDATE_PROFILE
                in {
                    "causal_fixed_lag_12",
                    "robust_hierarchical_system_id",
                }
            ):
                selected_label = str(selected_record["variant_label"])
                if selected_label.startswith("global_distance_"):
                    new_distance_median = float(
                        updater.distance_stiffness.median().item()
                    )
                    move = (
                        1
                        if new_distance_median > previous_distance_median
                        else -1
                    )
                    if (
                        STIFFNESS_ADMISSION_MODE != "causal_fixed_lag"
                        and self._stiffness_global_distance_direction == 0
                    ):
                        self._stiffness_global_distance_direction = move
                    self._stiffness_global_distance_commits += 1
                    self._stiffness_global_distance_locked = bool(
                        STIFFNESS_ADMISSION_MODE != "causal_fixed_lag"
                        and self._stiffness_global_distance_commits
                        >= STIFFNESS_ROBUST_MAXIMUM_GLOBAL_COMMITS_PER_FAMILY
                    )
                elif selected_label.startswith("global_shape_"):
                    new_shape_median = float(
                        updater.shape_stiffness.median().item()
                    )
                    move = (
                        1 if new_shape_median > previous_shape_median else -1
                    )
                    if (
                        STIFFNESS_ADMISSION_MODE != "causal_fixed_lag"
                        and self._stiffness_global_shape_direction == 0
                    ):
                        self._stiffness_global_shape_direction = move
                    self._stiffness_global_shape_commits += 1
                    self._stiffness_global_shape_locked = bool(
                        STIFFNESS_ADMISSION_MODE != "causal_fixed_lag"
                        and self._stiffness_global_shape_commits
                        >= STIFFNESS_ROBUST_MAXIMUM_GLOBAL_COMMITS_PER_FAMILY
                    )
        rollout_adoption_metrics = {
            "validated_rollout_adopted": False,
            "validated_rollout_state_rms_m": 0.0,
            "validated_rollout_state_maximum_m": 0.0,
            "validated_rollout_state_finite": True,
            "maximum_adopted_state_rms_m": (
                STIFFNESS_MAXIMUM_ADOPTED_STATE_RMS_M
            ),
            "maximum_adopted_state_maximum_m": (
                STIFFNESS_MAXIMUM_ADOPTED_STATE_MAXIMUM_M
            ),
        }
        if metrics["status"] == "committed":
            rollout_adoption_metrics = (
                self._adopt_validated_stiffness_rollout(candidate)
            )
        metrics.update(
            validation_status=metrics["status"],
            prediction_horizon_frames=(
                self.current_frame_index - pending.frame_index
            ),
            prediction_rollout_steps=len(pending.commands),
            baseline_visual_loss=baseline["visual_loss"],
            candidate_visual_loss=candidate["visual_loss"],
            baseline_open_loop_residual_rms_m=baseline.get(
                "open_loop_residual_rms_m"
            ),
            candidate_open_loop_residual_rms_m=candidate.get(
                "open_loop_residual_rms_m"
            ),
            residual_effort_improvement_m=(
                residual_effort_improvement
            ),
            residual_effort_policy="candidate_tiebreak_not_commit_veto",
            global_distance_locked=self._stiffness_global_distance_locked,
            global_shape_locked=self._stiffness_global_shape_locked,
            global_distance_direction=(
                self._stiffness_global_distance_direction
            ),
            global_shape_direction=self._stiffness_global_shape_direction,
            global_distance_commits=self._stiffness_global_distance_commits,
            global_shape_commits=self._stiffness_global_shape_commits,
            causal_trial_count=self._stiffness_trial_count,
            causal_maximum_trials=STIFFNESS_CAUSAL_MAXIMUM_TRIALS,
            causal_warmup_end_frame=STIFFNESS_CAUSAL_WARMUP_END_FRAME,
            causal_minimum_trial_interval_frames=(
                STIFFNESS_CAUSAL_MINIMUM_TRIAL_INTERVAL_FRAMES
            ),
            causal_local_maximum_commits_per_phase_family=(
                STIFFNESS_CAUSAL_LOCAL_MAXIMUM_COMMITS_PER_PHASE_FAMILY
            ),
            causal_local_confirmation_windows=(
                0
            ),
            causal_local_confirmation_state={
                f"{phase}:{family}": {
                    "direction": state[0],
                    "count": state[1],
                }
                for (phase, family), state in sorted(
                    causal_commit_policy["local_confirmation_state"].items()
                )
            },
            causal_local_phase_family_commits={
                f"{phase}:{family}": count
                for (phase, family), count in sorted(
                    self._stiffness_local_phase_family_commits.items()
                )
            },
            causal_local_phase_family_last_commit_frames={
                f"{phase}:{family}": frame
                for (phase, family), frame in sorted(
                    self._stiffness_local_phase_family_last_commit_frame.items()
                )
            },
            causal_minimum_same_family_commit_interval_frames=(
                STIFFNESS_CAUSAL_MINIMUM_SAME_FAMILY_COMMIT_INTERVAL_FRAMES
            ),
            causal_global_confirmation_windows=(
                0
            ),
            causal_global_confirmation_key=(
                causal_commit_policy["global_confirmation_key"]
            ),
            causal_global_confirmation_count=(
                causal_commit_policy["global_confirmation_count"]
            ),
            stiffness_proposal_phase=proposal_phase,
            validation_visual_objective=(
                stiffness_validation_objective_name()
            ),
            validation_horizons=STIFFNESS_ADMISSION_HORIZONS,
            validation_visual_residual_enabled=(
                STIFFNESS_ADMISSION_MODE == "causal_fixed_lag"
            ),
            validation_grip_state_machine_frozen=(
                STIFFNESS_ADMISSION_MODE != "causal_fixed_lag"
            ),
            baseline_horizon_visual_losses=baseline[
                "horizon_visual_losses"
            ],
            candidate_horizon_visual_losses=candidate[
                "horizon_visual_losses"
            ],
            horizon_visual_improvements={
                horizon: (
                    baseline["horizon_visual_losses"][horizon]
                    - candidate["horizon_visual_losses"][horizon]
                )
                if baseline["horizon_visual_losses"][horizon] is not None
                and candidate["horizon_visual_losses"][horizon] is not None
                else None
                for horizon in STIFFNESS_ADMISSION_HORIZONS
            },
            baseline_endpoint_visual_loss=baseline[
                "endpoint_visual_loss"
            ],
            candidate_endpoint_visual_loss=candidate[
                "endpoint_visual_loss"
            ],
            prediction_improvement=improvement,
            required_prediction_improvement=required_improvement,
            baseline_camera_losses=baseline["camera_losses"],
            candidate_camera_losses=candidate["camera_losses"],
            baseline_minimum_volume_ratio=baseline["minimum_volume_ratio"],
            candidate_minimum_volume_ratio=candidate["minimum_volume_ratio"],
            baseline_maximum_penetration_m=baseline["maximum_penetration_m"],
            candidate_maximum_penetration_m=candidate["maximum_penetration_m"],
            baseline_anchor_error_rms_m=baseline["anchor_error_rms_m"],
            candidate_anchor_error_rms_m=candidate["anchor_error_rms_m"],
            baseline_anchor_error_maximum_m=(
                baseline["anchor_error_maximum_m"]
            ),
            candidate_anchor_error_maximum_m=(
                candidate["anchor_error_maximum_m"]
            ),
            baseline_distance_loss=baseline["distance_loss"],
            candidate_distance_loss=candidate["distance_loss"],
            baseline_volume_loss=baseline["volume_loss"],
            candidate_volume_loss=candidate["volume_loss"],
            baseline_shape_loss=baseline["shape_loss"],
            candidate_shape_loss=candidate["shape_loss"],
            history_baseline_rms_m=history_baseline,
            history_candidate_rms_m=history_candidate,
            selected_candidate_variant=selected_record["variant_label"],
            selected_distance_scale=selected_record["distance_scale"],
            selected_shape_scale=selected_record["shape_scale"],
            selected_candidate_scope=selected_record["scope"],
            selected_candidate_scope_fraction=selected_record[
                "scope_fraction"
            ],
            candidate_search_trial_count=len(candidate_search_records),
            **rollout_adoption_metrics,
        )
        if metrics["status"] == "committed":
            self._start_committed_stiffness_evaluation(pending)
        if self.stiffness_metrics_recorder is not None:
            self.stiffness_metrics_recorder.record(
                event="stiffness_validation",
                frame_index=self.current_frame_index,
                timestamp_s=self.current_timestep,
                phase=self._current_action_phase,
                physical={
                    "baseline": self._serializable_shadow_metrics(baseline),
                    "candidate": self._serializable_shadow_metrics(candidate),
                },
                material=self._current_material_evaluation_metrics(metrics),
                prediction={
                    "horizon_frames": metrics[
                        "prediction_horizon_frames"
                    ],
                    "baseline_gap": baseline["visual_loss"],
                    "candidate_gap": candidate["visual_loss"],
                    "gap_improvement": improvement,
                    "required_improvement": required_improvement,
                    "horizons": STIFFNESS_ADMISSION_HORIZONS,
                    "baseline_horizon_gaps": baseline[
                        "horizon_visual_losses"
                    ],
                    "candidate_horizon_gaps": candidate[
                        "horizon_visual_losses"
                    ],
                    "visual_residual_enabled": (
                        STIFFNESS_ADMISSION_MODE == "causal_fixed_lag"
                    ),
                    "grip_state_machine_frozen": (
                        STIFFNESS_ADMISSION_MODE != "causal_fixed_lag"
                    ),
                    "status": metrics["status"],
                },
                details={
                    "rejection_reasons": reasons,
                    "candidate_search_trials": [
                        {
                            "variant_label": record["variant_label"],
                            "distance_scale": record["distance_scale"],
                            "shape_scale": record["shape_scale"],
                            "scope": record["scope"],
                            "scope_fraction": record["scope_fraction"],
                            "proposal_phase": record.get("proposal_phase"),
                            "local_phase_family": record.get(
                                "local_phase_family"
                            ),
                            "local_phase_commit_count": record.get(
                                "local_phase_commit_count"
                            ),
                            "local_phase_family_last_commit_frame": record.get(
                                "local_phase_family_last_commit_frame"
                            ),
                            "local_phase_family_commit_interval_frames": (
                                record.get(
                                    "local_phase_family_commit_interval_frames"
                                )
                            ),
                            "local_confirmation_direction": record.get(
                                "local_confirmation_direction"
                            ),
                            "local_confirmation_count": record.get(
                                "local_confirmation_count"
                            ),
                            "global_confirmation_key": record.get(
                                "global_confirmation_key"
                            ),
                            "global_confirmation_count": record.get(
                                "global_confirmation_count"
                            ),
                            "merged_from": record.get("merged_from", ()),
                            "gap_improvement": record["improvement"],
                            "residual_effort_improvement_m": (
                                self._stiffness_residual_effort_improvement(
                                    baseline, record["shadow"]
                                )
                            ),
                            "horizon_gap_improvements": {
                                horizon: (
                                    baseline["horizon_visual_losses"][horizon]
                                    - record["shadow"][
                                        "horizon_visual_losses"
                                    ][horizon]
                                )
                                if baseline["horizon_visual_losses"][horizon]
                                is not None
                                and record["shadow"][
                                    "horizon_visual_losses"
                                ][horizon]
                                is not None
                                else None
                                for horizon in STIFFNESS_ADMISSION_HORIZONS
                            },
                            "rejection_reasons": record["reasons"],
                        }
                        for record in candidate_search_records
                    ],
                },
                force_summary=True,
            )
        self._pending_stiffness_validation = None
        self._last_stiffness_validation_metrics = metrics
        return metrics

    async def run_physics(self):
        dt = self.environment.dt()
        while True:
            if (
                self._headless_physics_iteration_limit is not None
                and self._physics_iteration_count
                >= self._headless_physics_iteration_limit
            ):
                await trio.sleep(dt)
                continue
            # XPBD eval_ik restores articulation FK, so reapply the strict
            # timestamp-aligned LND pose after every physics step.
            terminal_evaluation_step = bool(
                self._active_stiffness_evaluations
                and not self.playing
                and self.current_frame_index
                == len(self.playback_timestamps) - 1
            )
            if (
                self._active_stiffness_evaluations
                and not self.playing
                and not terminal_evaluation_step
            ):
                self._record_incomplete_stiffness_evaluations(
                    "playback_paused"
                )
                self._active_stiffness_evaluations.clear()
            command = None
            if (self.playing or terminal_evaluation_step) and (
                self._pending_stiffness_validation is not None
                or self._pending_visual_residual_validation is not None
                or self._active_stiffness_evaluations
            ):
                command = self._current_stiffness_tool_command()
            if self._pending_visual_residual_validation is not None:
                if not self.playing:
                    # The live branch is deliberately the uncorrected branch;
                    # cancelling a pending shadow is therefore a no-op on
                    # physics state.
                    self._pending_visual_residual_validation = None
                    self._previous_visual_residual = None
                else:
                    assert command is not None
                    self._pending_visual_residual_validation.commands.append(
                        command
                    )
            if self._pending_stiffness_validation is not None:
                if not self.playing:
                    assert self.stiffness_updater is not None
                    pending = self._pending_stiffness_validation
                    if STIFFNESS_ADMISSION_MODE == "causal_fixed_lag":
                        self._clear_stiffness_global_confirmation()
                        self._clear_stiffness_local_confirmations()
                    metrics = self.stiffness_updater.reject(
                        "playback_paused_before_validation",
                        pending.candidate,
                    )
                    metrics.update(validation_status="cancelled")
                    self._record_terminal_stiffness_validation(
                        pending, metrics, "playback_paused_before_validation"
                    )
                    self._pending_stiffness_validation = None
                    self._last_stiffness_validation_metrics = metrics
                else:
                    assert command is not None
                    self._pending_stiffness_validation.commands.append(command)
            if command is not None:
                for evaluation in self._active_stiffness_evaluations:
                    evaluation.commands.append(command)
            if (
                self.playing
                and (
                    self.paper_trajectory_stiffness_optimizer is not None
                    or STIFFNESS_ADMISSION_MODE == "sim_particle_graph_lm"
                )
                and self._paper_adam_source_commands
            ):
                paper_command = command or self._current_stiffness_tool_command()
                for source_commands in self._paper_adam_source_commands.values():
                    source_commands.append(paper_command)
            self.environment.step(compute_visual_forces=False)
            self._physics_iteration_count += 1
            self.apply_current_psm_pose()
            evaluation_contact_metrics = None
            if self.stiffness_metrics_recorder is not None:
                evaluation_contact_metrics = (
                    self.environment.sim.triangle_skin_contact_metrics()
                )
                self._update_action_phase(evaluation_contact_metrics)
                if (
                    self.playing
                    and self.environment.frames is not None
                    and self.visual_residual_mapper is not None
                    and self.current_frame_index
                    > self._last_trajectory_observation_frame_index
                ):
                    self.environment.sim.update_gaussian_transforms()
                    alignment = (
                        self.environment.sim.evaluate_visual_tissue_alignment(
                            self.visual_residual_mapper,
                            self.environment.frames,
                            observations_are_bgr=True,
                        )
                    )
                    self._record_trajectory_observation(
                        alignment=alignment,
                        contact_metrics=evaluation_contact_metrics,
                    )
                    self._last_trajectory_observation_frame_index = int(
                        self.current_frame_index
                    )
            if terminal_evaluation_step:
                self._advance_committed_stiffness_evaluations()
                self._record_incomplete_stiffness_evaluations(
                    "dataset_finished"
                )
                self._active_stiffness_evaluations.clear()
            self._visual_force_step += 1
            if (
                self.playing
                and self.environment.frames is not None
                and self.visual_feedback_mode in TRAJECTORY_FEEDBACK_MODES
                and not self._defer_flow_depth_update_until_frame_end
            ):
                self._process_flow_depth_trajectory_frame()
                self._process_trajectory_rgb_residual_frame()
            visual_feedback_enabled = (
                self.playing
                and self.environment.frames is not None
                and self.visual_feedback_mode != "off"
                and self._benchmark_observation_enabled
            )
            update_visual_feedback = (
                self._visual_force_step % self.visual_force_update_interval == 0
            )
            if (
                visual_feedback_enabled
                and update_visual_feedback
                and self.visual_feedback_mode == "residual"
            ):
                assert self.visual_residual_mapper is not None
                solve_start = time.perf_counter()
                (
                    cross_frame_metrics,
                    confirmed_visual_pending,
                ) = self._validate_pending_visual_residual_cross_frame()
                # A shadow rollout restores the complete live state, but read
                # contact metrics again so the following gate is explicitly
                # tied to the selected branch.
                contact_metrics = (
                    self.environment.sim.triangle_skin_contact_metrics()
                    or evaluation_contact_metrics
                    or {}
                )
                (
                    stiffness_gate_paused,
                    stiffness_gate_reason,
                    stiffness_jaw_speed,
                    stiffness_grip_active,
                ) = self._stiffness_global_gate(contact_metrics)
                stiffness_metrics = self._validate_pending_stiffness(
                    gate_paused=stiffness_gate_paused,
                    gate_reason=stiffness_gate_reason,
                    grip_active=stiffness_grip_active,
                )
                self._advance_committed_stiffness_evaluations()
                if (
                    confirmed_visual_pending is not None
                    and self.stiffness_updater is not None
                ):
                    assert (
                        confirmed_visual_pending.quality_valid_mask is not None
                        and confirmed_visual_pending.supervision_valid_mask
                        is not None
                        and confirmed_visual_pending.control_exclusion_mask
                        is not None
                    )
                    stiffness_metrics = (
                        self._consume_confirmed_visual_stiffness_evidence(
                            applied_result=confirmed_visual_pending.result,
                            quality_valid_mask=(
                                confirmed_visual_pending.quality_valid_mask
                            ),
                            supervision_valid_mask=(
                                confirmed_visual_pending
                                .supervision_valid_mask
                            ),
                            control_exclusion_mask=(
                                confirmed_visual_pending
                                .control_exclusion_mask
                            ),
                            stiffness_gate_paused=(
                                confirmed_visual_pending
                                .stiffness_gate_paused
                            ),
                            stiffness_jaw_speed=(
                                confirmed_visual_pending
                                .stiffness_jaw_speed_rad_s
                            ),
                            stiffness_grip_active=(
                                confirmed_visual_pending
                                .stiffness_grip_active
                            ),
                            stiffness_contact_count=(
                                confirmed_visual_pending
                                .stiffness_contact_count
                            ),
                            accepted_positions=wp.to_torch(
                                self.environment.sim.state_0.particle_q
                            ).detach().clone(),
                            stiffness_metrics=stiffness_metrics,
                        )
                    )
                # One image timestamp may span several physics iterations.
                # Re-solving the same observation would repeatedly learn from
                # stale evidence and can immediately recreate an expired
                # candidate.  Material learning advances only with images.
                if (
                    self.current_frame_index
                    <= self._last_visual_residual_frame_index
                ):
                    await trio.sleep(dt)
                    continue
                control_exclusion_mask = grip_control_exclusion_mask(
                    self.visual_residual_mapper,
                    self.environment,
                )
                result = self.environment.sim.solve_visual_tissue_residual(
                    self.visual_residual_mapper,
                    self.environment.frames,
                    previous_residual=self._previous_visual_residual,
                    dynamic_exclusion_mask=control_exclusion_mask,
                    observations_are_bgr=True,
                )
                previous_observation_frame = int(
                    self._last_visual_residual_frame_index
                )
                if 0 <= previous_observation_frame < self.current_frame_index:
                    visual_observation_dt_s = float(
                        self.playback_timestamps[self.current_frame_index]
                        - self.playback_timestamps[previous_observation_frame]
                    )
                elif self.current_frame_index > 0:
                    visual_observation_dt_s = float(
                        self.playback_timestamps[self.current_frame_index]
                        - self.playback_timestamps[self.current_frame_index - 1]
                    )
                else:
                    visual_observation_dt_s = 1.0 / float(self.fps)
                accepted = (
                    self._apply_visual_tissue_residual_with_persistence_gate(
                        result,
                        observation_dt_s=visual_observation_dt_s,
                        dynamic_exclusion_mask=control_exclusion_mask,
                    )
                )
                applied_result = (
                    self._last_applied_visual_residual_result
                    if accepted
                    else None
                )
                self._last_visual_residual_frame_index = int(
                    self.current_frame_index
                )
                if accepted and self.stiffness_updater is not None:
                    assert applied_result is not None
                    physical_prediction = (
                        applied_result.corrected_positions
                        - applied_result.residual
                    )
                    quality_valid_mask = stiffness_local_quality_mask(
                        self.visual_residual_mapper,
                        physical_prediction,
                        applied_result.corrected_positions,
                        STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO,
                    )
                    supervision_valid_mask = (
                        stiffness_visual_supervision_mask(
                            self.visual_residual_mapper,
                            result.visual_gradient_norm,
                        )
                    )
                    if self.visual_residual_gain_profile in {
                        "cross_frame_hold",
                        "cross_frame_ranked_hold",
                    }:
                        pending_visual = (
                            self._pending_visual_residual_validation
                        )
                        assert pending_visual is not None
                        pending_visual.quality_valid_mask = (
                            quality_valid_mask.detach().clone()
                        )
                        pending_visual.supervision_valid_mask = (
                            supervision_valid_mask.detach().clone()
                        )
                        pending_visual.control_exclusion_mask = (
                            control_exclusion_mask.detach().clone()
                        )
                        pending_visual.stiffness_gate_paused = bool(
                            stiffness_gate_paused
                        )
                        pending_visual.stiffness_jaw_speed_rad_s = float(
                            stiffness_jaw_speed
                        )
                        pending_visual.stiffness_grip_active = bool(
                            stiffness_grip_active
                        )
                        pending_visual.stiffness_contact_count = int(
                            contact_metrics.get("contact_count", 0)
                        )
                    else:
                        stiffness_metrics = (
                            self._consume_confirmed_visual_stiffness_evidence(
                                applied_result=applied_result,
                                quality_valid_mask=quality_valid_mask,
                                supervision_valid_mask=(
                                    supervision_valid_mask
                                ),
                                control_exclusion_mask=(
                                    control_exclusion_mask
                                ),
                                stiffness_gate_paused=(
                                    stiffness_gate_paused
                                ),
                                stiffness_jaw_speed=stiffness_jaw_speed,
                                stiffness_grip_active=(
                                    stiffness_grip_active
                                ),
                                stiffness_contact_count=int(
                                    contact_metrics.get("contact_count", 0)
                                ),
                                accepted_positions=(
                                    applied_result.corrected_positions
                                ),
                                stiffness_metrics=stiffness_metrics,
                            )
                        )
                self._visual_residual_solve_count += 1
                metrics = (
                    self.environment.sim.last_visual_tissue_residual_metrics
                )
                assert metrics is not None
                metrics["solve_elapsed_s"] = time.perf_counter() - solve_start
                metrics["solve_count"] = self._visual_residual_solve_count
                if self.visual_residual_gain_profile not in {
                    "cross_frame_hold",
                    "cross_frame_ranked_hold",
                }:
                    self._previous_visual_residual = (
                        applied_result.residual.detach().clone()
                        if applied_result is not None
                        else None
                    )
                if cross_frame_metrics is not None:
                    metrics["previous_cross_frame_validation"] = (
                        cross_frame_metrics
                    )
                self._record_visual_stiffness_metrics(
                    result=result,
                    visual_metrics=metrics,
                    stiffness_metrics=stiffness_metrics,
                    contact_metrics=contact_metrics,
                    gate_paused=stiffness_gate_paused,
                    gate_reason=stiffness_gate_reason,
                )
                acceptance_changed = (
                    self._last_visual_residual_accepted is None
                    or accepted != self._last_visual_residual_accepted
                )
                self._last_visual_residual_accepted = accepted
                if acceptance_changed or self._visual_residual_solve_count % 30 == 0:
                    stiffness_summary = (
                        "stiffness_med="
                        f"{stiffness_metrics['distance_median']:.3f}/"
                        f"{stiffness_metrics['shape_median']:.4f}, "
                        if stiffness_metrics is not None
                        else ""
                    )
                    print(
                        "[visual residual realtime] "
                        f"accepted={accepted}, "
                        f"loss={result.initial_visual_loss:.6f}->"
                        f"{float(metrics['exact_final_visual_loss']):.6f}, "
                        f"max={result.maximum_residual_m * 1e3:.3f}mm, "
                        "min_volume="
                        f"{result.initial_minimum_volume_ratio:.3f}->"
                        f"{result.minimum_volume_ratio:.3f}, "
                        "grip_excluded="
                        f"{result.dynamically_excluded_particles}, "
                        f"local_frozen={result.locally_frozen_particles}, "
                        f"backtracks={result.backtrack_count}, "
                        "stiffness_quality_masked="
                        f"{int(stiffness_metrics['quality_masked_particles']) if stiffness_metrics is not None else 0}, "
                        "stiffness_status="
                        f"{str(stiffness_metrics.get('status', 'paused')) if stiffness_metrics is not None else ('paused' if stiffness_gate_paused else 'off')}, "
                        "hold_H1/3/5="
                        + "/".join(
                            "NA"
                            if metrics.get(
                                "persistence_visual_improvements", {}
                            ).get(horizon)
                            is None
                            else (
                                f"{float(metrics['persistence_visual_improvements'][horizon]):+.2e}"
                            )
                            for horizon in VISUAL_RESIDUAL_PERSISTENCE_HORIZONS
                        )
                        + ", "
                        f"{stiffness_summary}"
                        f"elapsed={metrics['solve_elapsed_s'] * 1e3:.1f}ms"
                    )
            elif (
                visual_feedback_enabled
                and update_visual_feedback
                and self.visual_feedback_mode == "force"
            ):
                self.environment.sim.compute_visual_forces(
                    self.environment.visual_forces_settings,
                    self.environment.frames,
                    self.environment.physics_settings.dt
                    / self.environment.physics_settings.substeps,
                )
            elif visual_feedback_enabled and self.visual_feedback_mode == "force":
                self.environment.sim.reapply_last_soft_visual_forces()
            self.maybe_print_psm_base_q()
            self.maybe_print_tissue_q()
            await trio.sleep(dt)

    async def run(self):
        async with trio.open_nursery() as nursery:
            nursery.start_soon(self.run_physics)
            while True:
                if self.playing:
                    self.capture_flow_depth_source_state(
                        self.current_frame_index
                    )
                    self.advance_one_frame()
                await trio.sleep(1 / self.fps)


async def run_headless_trajectory_evaluation(
    playback_controls: SuperPlaybackControls,
    *,
    start_frame: int,
    frame_count: int,
    physics_steps_per_frame: int,
) -> None:
    """Run an exact frame/physics-step schedule without constructing a GUI."""
    total_frames = len(playback_controls.playback_timestamps)
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(
            f"Evaluation start frame {start_frame} is outside 0..{total_frames - 1}"
        )
    if frame_count < 0:
        raise ValueError("Evaluation frame count cannot be negative")
    if physics_steps_per_frame < 1:
        raise ValueError("Evaluation physics steps per frame must be positive")
    end_frame_exclusive = (
        total_frames
        if frame_count == 0
        else min(total_frames, start_frame + frame_count)
    )
    playback_controls.go_to_frame(start_frame)
    playback_controls.environment.sim.sync_kinematic_body_interpolation()
    playback_controls.playing = True
    playback_controls._defer_flow_depth_update_until_frame_end = True
    playback_controls._headless_physics_iteration_limit = (
        playback_controls._physics_iteration_count
    )
    print(
        "[trajectory evaluation] headless schedule: "
        f"frames={start_frame}..{end_frame_exclusive - 1}, "
        f"physics_steps_per_frame={physics_steps_per_frame}, "
        f"mode={playback_controls.visual_feedback_mode}, "
        "online_stiffness="
        f"{playback_controls.stiffness_updater is not None}"
    )
    dt = max(float(playback_controls.environment.dt()), 1.0e-4)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(playback_controls.run_physics)
        for frame_index in range(start_frame, end_frame_exclusive):
            playback_controls.prepare_benchmark_frame(frame_index)
            if playback_controls.current_frame_index != frame_index:
                playback_controls.go_to_frame(frame_index)
            target_physics_iterations = (
                playback_controls._physics_iteration_count
                + physics_steps_per_frame
            )
            playback_controls._headless_physics_iteration_limit = (
                target_physics_iterations
            )
            with trio.fail_after(900.0):
                while (
                    playback_controls._physics_iteration_count
                    < target_physics_iterations
                    or (
                        playback_controls.stiffness_metrics_recorder is not None
                        and playback_controls._last_trajectory_observation_frame_index
                        < frame_index
                    )
                ):
                    await trio.sleep(dt)
            if playback_controls.visual_feedback_mode in TRAJECTORY_FEEDBACK_MODES:
                playback_controls._process_flow_depth_trajectory_frame()
                playback_controls._process_trajectory_rgb_residual_frame()
            if playback_controls.tissue_benchmark_recorder is not None:
                playback_controls.apply_one_frame_visual_open_loop_prediction()
                playback_controls.tissue_benchmark_recorder.capture(
                    frame_index,
                    playback_controls.environment,
                    playback_controls.environment.frames,
                )
            playback_controls.capture_flow_depth_source_state(frame_index)
        playback_controls.playing = False
        playback_controls._headless_physics_iteration_limit = (
            playback_controls._physics_iteration_count
        )
        # Give run_physics one checkpoint to cancel a pending candidate and
        # persist incomplete open-loop horizons before shutdown.
        await trio.sleep(dt * 1.1)
        nursery.cancel_scope.cancel()
    print(
        "[trajectory evaluation] complete: "
        f"observed_frames={start_frame}..{end_frame_exclusive - 1}, "
        f"physics_iterations={playback_controls._physics_iteration_count}, "
        "flow_depth_updates="
        f"{playback_controls._flow_depth_update_count}"
    )
    stiffness_updater = playback_controls.stiffness_updater
    if stiffness_updater is not None:
        print(
            "[trajectory evaluation] stiffness summary: "
            f"evidence={stiffness_updater.evidence_count}, "
            f"candidates={stiffness_updater.candidate_count}, "
            f"commits={stiffness_updater.update_count}, "
            f"distance_commits={stiffness_updater.distance_update_count}, "
            f"shape_commits={stiffness_updater.shape_update_count}, "
            f"rejections={stiffness_updater.rejected_count}, "
            "rejection_reasons="
            f"{dict(sorted(stiffness_updater.rejection_reason_counts.items()))}, "
            "distance(min/median/max)="
            f"{float(stiffness_updater.distance_stiffness.min().item()):g}/"
            f"{float(stiffness_updater.distance_stiffness.median().item()):g}/"
            f"{float(stiffness_updater.distance_stiffness.max().item()):g}, "
            "shape(min/median/max)="
            f"{float(stiffness_updater.shape_stiffness.min().item()):g}/"
            f"{float(stiffness_updater.shape_stiffness.median().item()):g}/"
            f"{float(stiffness_updater.shape_stiffness.max().item()):g}"
        )


async def main(
    dataset_path: Path,
    fps: int,
    monitor_psm_base_q: bool,
    monitor_tissue_q: bool,
    monitor_interval: float,
    visual_feedback_mode: str,
    flow_depth_bindings_path: Path,
    flow_depth_observations_path: Path,
    flow_depth_position_gain: float,
    flow_depth_velocity_gain: float,
    flow_depth_absolute_position_weight: float,
    flow_depth_solver_regularization: float,
    flow_depth_solver_iterations: int,
    flow_depth_robust_residual_mm: float,
    flow_depth_maximum_position_correction_mm: float,
    flow_depth_maximum_velocity_correction_m_s: float,
    flow_depth_local_material_relaxation: bool,
    flow_depth_local_distance_stiffness: float,
    flow_depth_local_volume_stiffness: float,
    flow_depth_local_shape_stiffness: float,
    visual_force_iterations: int,
    visual_residual_iterations: int,
    visual_residual_learning_rate_m: float,
    visual_residual_maximum_step_m: float,
    visual_residual_previous_carry: float,
    visual_residual_temporal_weight: float,
    visual_residual_magnitude_weight: float,
    visual_residual_gain_profile: str,
    trajectory_rgb_residual_track_weight: float,
    trajectory_rgb_residual_position_gain: float,
    trajectory_rgb_residual_velocity_gain: float,
    trajectory_rgb_residual_maximum_velocity_m_s: float,
    trajectory_gaussian_appearance_iterations: int,
    trajectory_gaussian_color_learning_rate: float,
    trajectory_gaussian_opacity_learning_rate: float,
    trajectory_gaussian_maximum_color_logit_offset: float,
    trajectory_gaussian_maximum_opacity_logit_offset: float,
    trajectory_gaussian_optimize_opacity: bool,
    trajectory_gaussian_appearance_image_scale: float,
    trajectory_gaussian_appearance_dssim_weight: float,
    visual_force_update_interval: int,
    online_stiffness_update: bool,
    stiffness_log_learning_rate: float,
    stiffness_maximum_log_step: float,
    stiffness_effective_log_step_target: float,
    stiffness_maximum_step_amplification: float,
    stiffness_strain_signal_weight: float,
    stiffness_hardening_bias: float,
    stiffness_minimum_residual_mm: float,
    stiffness_minimum_deformation_mm: float,
    stiffness_rejected_ema_keep_ratio: float,
    stiffness_distance_minimum: float,
    stiffness_distance_maximum: float,
    stiffness_shape_minimum: float,
    stiffness_shape_maximum: float,
    paper_stiffness_adam_learning_rate: float,
    paper_stiffness_perturbation: float,
    paper_stiffness_track_robust_scale_mm: float,
    paper_stiffness_history_scale_mm: float,
    paper_stiffness_history_weight: float,
    paper_stiffness_distance_smooth_weight: float,
    paper_stiffness_shape_smooth_weight: float,
    paper_stiffness_minimum_axis_loss_difference: float,
    paper_stiffness_sim_global_causal: bool,
    paper_stiffness_three_of_four_h3: bool,
    paper_stiffness_local_distance: bool,
    paper_stiffness_local_distance_log_step: float,
    paper_stiffness_local_distance_log_offset: float,
    paper_stiffness_local_distance_minimum_improvement: float,
    stiffness_evaluation_output: Path | None,
    stiffness_evaluation_horizons: tuple[int, ...],
    evaluation_headless: bool,
    evaluation_start_frame: int,
    evaluation_frame_count: int,
    evaluation_physics_steps_per_frame: int,
    tissue_benchmark_output: Path | None,
    tissue_benchmark_ground_truth: Path,
    tissue_benchmark_protocol: str,
    tissue_benchmark_render_scale: float,
    tissue_benchmark_reconstruction_test_phase: int,
    tissue_benchmark_track_only: bool,
    tissue_benchmark_future_test_start_frame: int | None,
    camera_names: list[str],
    camera_go_zoom: float,
    psm_roll_offset_deg: float,
    psm_camera_translation_mm: tuple[float, float, float],
    psm_world_translation_mm: tuple[float, float, float],
    psm_pose_driver: str,
    psm_visual_mode: str,
    tissue_mode: str,
    paper_distance_stiffness_initial: float | None,
    paper_volume_stiffness_initial: float | None,
    paper_shape_stiffness_initial: float | None,
    psm_tissue_contact: bool,
    calibrated_profile: Path | None,
):
    pose_driver_path = PSM_POSE_DRIVER_PATHS[psm_pose_driver]
    environment = build_environment(
        psm_pose_driver_path=pose_driver_path,
        psm_visual_tip_only=psm_visual_mode == "tip",
        tissue_mode=tissue_mode,
    )
    load_calibrated_tissue_profile(environment, calibrated_profile)
    configure_initial_paper_stiffness(
        environment,
        tissue_mode=tissue_mode,
        distance_stiffness=paper_distance_stiffness_initial,
        volume_stiffness=paper_volume_stiffness_initial,
        shape_stiffness=paper_shape_stiffness_initial,
    )
    print(
        "[example_embodied_super_offline] psm_pose_driver="
        f"{psm_pose_driver}: {pose_driver_path}; visual_mode={psm_visual_mode}; "
        f"tissue_mode={tissue_mode}"
    )
    print(
        "[example_embodied_super_offline] frozen manual correction="
        f"roll={psm_roll_offset_deg:+.3f} deg, "
        f"camera_translation_mm={list(psm_camera_translation_mm)}, "
        f"world_translation_mm={list(psm_world_translation_mm)}, "
        "jaw_mode=raw_q7_full_closure_with_3_to_5_particle_gate"
    )
    print(
        "[example_embodied_super_offline] embodied_gaussians_source="
        f"{LOADED_EMBODIED_GAUSSIANS_SOURCE}"
    )
    environment.visual_forces_settings.iterations = (
        visual_force_iterations if visual_feedback_mode == "force" else 0
    )
    environment.super_tissue_residual_mapping_enabled = (
        visual_feedback_mode == "residual"
    )
    print(
        "[example_embodied_super_offline] visual_feedback_mode="
        f"{visual_feedback_mode}; force_iterations={visual_force_iterations}; "
        f"residual_iterations={visual_residual_iterations}; "
        f"update_interval={visual_force_update_interval}; PSM feedback disabled"
    )
    if visual_feedback_mode == "residual":
        hold_policy = (
            "H1 bounded noise; H3 tiny bounded regression; H5 and the "
            "H3/H5 weighted hold must improve; "
            "later training RGB ranks all safe gains"
            if visual_residual_gain_profile == "cross_frame_ranked_hold"
            else "all horizons must improve"
        )
        print(
            "[example_embodied_super_offline] residual writeback gate="
            f"no-vision physics hold H=1/3/5; {hold_policy}; "
            f"gain_profile={visual_residual_gain_profile}; gains="
            f"{VISUAL_RESIDUAL_GAIN_PROFILES[visual_residual_gain_profile]}"
        )
    visual_settings = environment.visual_forces_settings
    print(
        "[example_embodied_super_offline] tissue_visual_force_limits="
        f"lr_means={visual_settings.lr_means}, kp={visual_settings.kp}, "
        f"normalized={visual_settings.normalize_forces_by_gaussian_count}, "
        f"rigid_max_force={visual_settings.max_force}N, "
        f"soft_max_total_force={visual_settings.soft_max_total_force}N, "
        "soft_max_particle_acceleration="
        f"{visual_settings.soft_max_particle_acceleration}m/s^2, "
        f"soft_spread_layers={visual_settings.soft_force_spread_layers}, "
        f"max_moment={visual_settings.max_moment}Nm"
    )

    dataset_manager = DatasetManager(dataset_path)
    if hasattr(dataset_manager, 'keep_only_cameras'):
        dataset_manager.keep_only_cameras(camera_names)
    visual_force_weights = MultiCameraPackedTissueVisualForceWeights(
        {
            "stereo_left": VISUAL_FORCE_MASK_DIR,
            "stereo_right": RIGHT_VISUAL_FORCE_MASK_DIR,
        },
        erosion_radius_px=7,
        highlight_weight=0.10,
        instrument_mask_asset=VISUAL_FORCE_INSTRUMENT_MASKS,
        tool_near_radius_px=120.0,
        tool_far_radius_px=360.0,
        tool_falloff_power=2.0,
        tissue_edge_zero_px=24.0,
        tissue_edge_full_px=64.0,
        tool_occlusion_radius_px=12.0,
        image_border_zero_px=48.0,
        image_border_full_px=96.0,
        posterior_full_reach_px=140.0,
        posterior_zero_reach_px=280.0,
    )
    print(
        "[example_embodied_super_offline] visual-force pixel field="
        "distal/jaw SDF near/full=120px, far/zero=360px; "
        "far-transition power=2; "
        "tissue edge zero/full=24px/64px; tool occlusion=12px; "
        "image border zero/full=48px/96px; "
        "posterior(-image-y) full/zero reach=140px/280px; "
        "q7/cache/Gaussian gates unchanged"
    )
    dataset_manager.set_visual_force_weight_provider(visual_force_weights)
    dataset_manager.update_frames(0.0)
    print(f"[example_embodied_super_offline] enabled_cameras={getattr(dataset_manager.frames, 'names', 'N/A')}")
    environment.frames = dataset_manager.frames

    flow_depth_bindings = None
    flow_depth_observations = None
    flow_depth_settings = None
    if visual_feedback_mode in TRAJECTORY_FEEDBACK_MODES:
        flow_depth_bindings = load_fixed_particle_range_bindings(
            flow_depth_bindings_path
        )
        flow_depth_observations = load_flow_depth_observation_sequence(
            flow_depth_observations_path
        )
        flow_depth_settings = FlowDepthStateUpdateSettings(
            position_gain=flow_depth_position_gain,
            velocity_gain=flow_depth_velocity_gain,
            absolute_position_weight=flow_depth_absolute_position_weight,
            solver_regularization=flow_depth_solver_regularization,
            solver_iterations=flow_depth_solver_iterations,
            robust_residual_scale_m=(
                flow_depth_robust_residual_mm * 1.0e-3
            ),
            maximum_position_correction_m=(
                flow_depth_maximum_position_correction_mm * 1.0e-3
            ),
            maximum_velocity_correction_m_s=(
                flow_depth_maximum_velocity_correction_m_s
            ),
        )
        flow_depth_settings.validate()
        print(
            "[example_embodied_super_offline] flow-depth trajectory: ON; "
            f"pairs={len(flow_depth_observations.current_source_frames)}, "
            f"tracks={flow_depth_observations.track_valid.shape[1]}, "
            f"position_gain={flow_depth_position_gain:g}, "
            f"velocity_gain={flow_depth_velocity_gain:g}, "
            "absolute_position_weight="
            f"{flow_depth_absolute_position_weight:g}, "
            f"joint_solver_iterations={flow_depth_solver_iterations}, "
            "joint_solver_regularization="
            f"{flow_depth_solver_regularization:g}, "
            f"robust_residual={flow_depth_robust_residual_mm:g}mm, "
            "maximum_position_correction="
            f"{flow_depth_maximum_position_correction_mm:g}mm, "
            "maximum_velocity_correction="
            f"{flow_depth_maximum_velocity_correction_m_s:g}m/s"
        )

    visual_residual_mapper = None
    stiffness_updater = None
    paper_trajectory_stiffness_optimizer = None
    if (
        visual_feedback_mode in ({"residual"} | TRAJECTORY_FEEDBACK_MODES)
        or stiffness_evaluation_output is not None
    ):
        visual_residual_mapper = build_visual_tissue_residual_mapper(
            environment,
            iterations=visual_residual_iterations,
            learning_rate_m=visual_residual_learning_rate_m,
            maximum_step_m=visual_residual_maximum_step_m,
            previous_residual_carry=visual_residual_previous_carry,
            temporal_weight=visual_residual_temporal_weight,
            magnitude_weight=visual_residual_magnitude_weight,
        )
        if visual_feedback_mode == "trajectory_residual":
            print(
                "[example_embodied_super_offline] post-trajectory RGB residual: ON; "
                "frequency=once_per_observable_video_frame, "
                f"track_hold_weight={trajectory_rgb_residual_track_weight:g}, "
                f"position_gain={trajectory_rgb_residual_position_gain:g}, "
                f"velocity_gain={trajectory_rgb_residual_velocity_gain:g}, "
                "maximum_velocity_correction="
                f"{trajectory_rgb_residual_maximum_velocity_m_s:g}m/s, "
                "material_evidence=OFF, future_test_feedback=OFF"
            )
            if trajectory_gaussian_appearance_iterations > 0:
                print(
                    "[example_embodied_super_offline] post-trajectory "
                    "Gaussian appearance: ON; properties="
                    + (
                        "rgb,opacity, "
                        if trajectory_gaussian_optimize_opacity
                        else "rgb(color-only), "
                    )
                    + f"iterations={trajectory_gaussian_appearance_iterations}, "
                    "lr_color/opacity="
                    f"{trajectory_gaussian_color_learning_rate:g}/"
                    f"{trajectory_gaussian_opacity_learning_rate:g}, "
                    "cumulative_logit_caps="
                    f"{trajectory_gaussian_maximum_color_logit_offset:g}/"
                    f"{trajectory_gaussian_maximum_opacity_logit_offset:g}, "
                    "geometry_appearance=fixed_initial, "
                    "commit=next_legal_training_frame_validation, "
                    "formal_scale/DSSIM="
                    f"{trajectory_gaussian_appearance_image_scale:g}/"
                    f"{trajectory_gaussian_appearance_dssim_weight:g}, "
                    "future_test_feedback=OFF"
                )
    if flow_depth_local_material_relaxation:
        if visual_feedback_mode not in TRAJECTORY_FEEDBACK_MODES:
            raise ValueError(
                "Local flow-depth material relaxation requires trajectory mode"
            )
        if flow_depth_bindings is None or visual_residual_mapper is None:
            raise ValueError("Trajectory material relaxation is not initialized")
        configure_flow_depth_local_material_relaxation(
            environment,
            bindings=flow_depth_bindings,
            mapper=visual_residual_mapper,
            distance_stiffness=flow_depth_local_distance_stiffness,
            volume_stiffness=flow_depth_local_volume_stiffness,
            shape_stiffness=flow_depth_local_shape_stiffness,
        )
    if visual_feedback_mode in ({"residual"} | TRAJECTORY_FEEDBACK_MODES):
        environment.sim.clear_soft_visual_force_cache()
        if online_stiffness_update:
            if tissue_mode != "paper_pbd":
                raise ValueError(
                    "Online paper stiffness update requires --tissue-mode paper_pbd"
                )
            material_projector = environment.sim.material_projector
            if material_projector is None:
                raise ValueError(
                    "Online stiffness update requires the material projector"
                )
            # The projector is constructed before the scene-specific paper
            # settings are installed.  Seed its per-particle arrays from the
            # actual v15 baseline before enabling preservation; otherwise the
            # constructor defaults (0.2/0.005) would be frozen instead of the
            # configured 0.20/0.004 values printed by the GUI.
            physics_settings = environment.physics_settings
            material_projector.configure_constraint_model(
                physics_settings.tetrahedral_constraint_model,
                physics_settings.paper_distance_stiffness,
                physics_settings.paper_volume_stiffness,
                physics_settings.paper_shape_stiffness,
                preserve_spatial_stiffness=(
                    flow_depth_local_material_relaxation
                ),
            )
            updater_arguments = dict(
                rest_positions=visual_residual_mapper.rest_positions,
                fixed_mask=visual_residual_mapper.fixed_mask,
                edges=visual_residual_mapper.edges,
                distance_stiffness=wp.to_torch(
                    material_projector.paper_distance_stiffness
                ),
                shape_stiffness=wp.to_torch(
                    material_projector.paper_shape_stiffness
                ),
            )
            if STIFFNESS_ADMISSION_MODE == "sim_particle_graph_lm":
                if visual_feedback_mode not in TRAJECTORY_FEEDBACK_MODES:
                    raise ValueError(
                        "sim_particle_graph_lm requires AllTracker trajectory mode"
                    )
                assert flow_depth_bindings is not None
                stiffness_updater = SimParticleGraphUpdater(
                    **updater_arguments,
                    inverse_mass=wp.to_torch(
                        environment.sim.model.particle_inv_mass
                    ),
                    tet_indices=visual_residual_mapper.tet_indices,
                    track_particle_ids=torch.as_tensor(
                        flow_depth_bindings.particle_ids,
                        device=visual_residual_mapper.rest_positions.device,
                        dtype=torch.long,
                    ),
                    track_particle_weights=torch.as_tensor(
                        flow_depth_bindings.particle_weights,
                        device=visual_residual_mapper.rest_positions.device,
                        dtype=torch.float32,
                    ),
                    track_valid_mask=torch.as_tensor(
                        flow_depth_bindings.track_valid,
                        device=visual_residual_mapper.rest_positions.device,
                        dtype=torch.bool,
                    ),
                    initial_velocity_damping_per_second=float(
                        environment.physics_settings
                        .particle_velocity_damping_per_second
                    ),
                    settings=SimParticleGraphSettings(
                        update_mode="differentiable_particle_graph_lm",
                        log_learning_rate=stiffness_log_learning_rate,
                        maximum_log_step=stiffness_maximum_log_step,
                        signal_ema_decay=0.90,
                        spatial_smoothing_iterations=3,
                        spatial_smoothing_blend=0.35,
                        strain_signal_weight=0.20,
                        autograd_unroll_steps=5,
                        autograd_region_count=12,
                        autograd_frame_dt=(
                            2.0 / 30.0
                            if tissue_benchmark_protocol == "future_80to20"
                            else 1.0 / 30.0
                        ),
                        causal_frame_stride=(
                            2
                            if tissue_benchmark_protocol == "future_80to20"
                            else 1
                        ),
                        distance_minimum=stiffness_distance_minimum,
                        distance_maximum=stiffness_distance_maximum,
                        shape_minimum=stiffness_shape_minimum,
                        shape_maximum=stiffness_shape_maximum,
                    ),
                )
            else:
                stiffness_updater = ResidualDrivenPaperStiffnessUpdater(
                    **updater_arguments,
                    settings=OnlineTissueStiffnessSettings(
                    log_learning_rate=stiffness_log_learning_rate,
                    maximum_log_step=stiffness_maximum_log_step,
                    effective_log_step_target=(
                        stiffness_effective_log_step_target
                    ),
                    maximum_step_amplification=(
                        stiffness_maximum_step_amplification
                    ),
                    strain_signal_weight=stiffness_strain_signal_weight,
                    hardening_bias=stiffness_hardening_bias,
                    minimum_residual_m=(
                        stiffness_minimum_residual_mm * 1.0e-3
                    ),
                    minimum_deformation_m=(
                        stiffness_minimum_deformation_mm * 1.0e-3
                    ),
                    rejected_ema_decay=stiffness_rejected_ema_keep_ratio,
                    distance_minimum=stiffness_distance_minimum,
                    distance_maximum=stiffness_distance_maximum,
                    shape_minimum=stiffness_shape_minimum,
                    shape_maximum=stiffness_shape_maximum,
                    ),
                )
            if STIFFNESS_ADMISSION_MODE == "paper_trajectory_adam":
                if visual_feedback_mode not in TRAJECTORY_FEEDBACK_MODES:
                    raise ValueError(
                        "paper_trajectory_adam requires AllTracker trajectory mode"
                    )
                paper_trajectory_stiffness_optimizer = (
                    PaperTrajectoryAdamOptimizer(
                        distance_stiffness=(
                            stiffness_updater.distance_stiffness
                        ),
                        shape_stiffness=stiffness_updater.shape_stiffness,
                        fixed_mask=visual_residual_mapper.fixed_mask,
                        edges=visual_residual_mapper.edges,
                        settings=PaperTrajectoryAdamSettings(
                            learning_rate=(
                                paper_stiffness_adam_learning_rate
                            ),
                            maximum_material_log_step=(
                                stiffness_maximum_log_step
                            ),
                            parameter_perturbation=(
                                paper_stiffness_perturbation
                            ),
                            track_robust_scale_m=(
                                paper_stiffness_track_robust_scale_mm
                                * 1.0e-3
                            ),
                            history_scale_m=(
                                paper_stiffness_history_scale_mm * 1.0e-3
                            ),
                            history_weight=paper_stiffness_history_weight,
                            distance_smooth_weight=(
                                paper_stiffness_distance_smooth_weight
                            ),
                            shape_smooth_weight=(
                                paper_stiffness_shape_smooth_weight
                            ),
                            minimum_axis_loss_difference=(
                                paper_stiffness_minimum_axis_loss_difference
                            ),
                            sim_global_causal_mode=(
                                paper_stiffness_sim_global_causal
                            ),
                            causal_maximum_window_size=(
                                4
                                if paper_stiffness_three_of_four_h3
                                else 3
                            ),
                            causal_maximum_missing_observations=(
                                1
                                if paper_stiffness_three_of_four_h3
                                else 0
                            ),
                            local_distance_enabled=(
                                paper_stiffness_local_distance
                            ),
                            local_distance_maximum_log_step=(
                                paper_stiffness_local_distance_log_step
                            ),
                            local_distance_maximum_log_offset=(
                                paper_stiffness_local_distance_log_offset
                            ),
                            local_distance_minimum_loss_improvement=(
                                paper_stiffness_local_distance_minimum_improvement
                            ),
                            h2_updates_enabled=(
                                tissue_benchmark_protocol
                                != "reconstruction_7to1"
                            ),
                            reconstruction_nonoverlapping_h3=(
                                tissue_benchmark_protocol
                                == "reconstruction_7to1"
                            ),
                            h3_tail_cosine_minimum=(
                                0.90
                                if tissue_benchmark_protocol
                                == "reconstruction_7to1"
                                else None
                            ),
                            global_maximum_log_offset=(
                                0.15
                                if tissue_benchmark_protocol
                                == "reconstruction_7to1"
                                else 0.35
                            ),
                            velocity_damping_initial_per_second=float(
                                environment.physics_settings
                                .particle_velocity_damping_per_second
                            ),
                            distance_minimum=stiffness_distance_minimum,
                            distance_maximum=stiffness_distance_maximum,
                            shape_minimum=stiffness_shape_minimum,
                            shape_maximum=stiffness_shape_maximum,
                        ),
                    )
                )
            environment.physics_settings.preserve_spatial_paper_stiffness = True
            environment.super_tissue_stiffness_optimization_enabled = True
            environment.super_tissue_fixed_material_parameters = False
            print(
                "[example_embodied_super_offline] online paper stiffness: ON; "
                f"log_lr={stiffness_log_learning_rate:g}, "
                f"max_log_step={stiffness_maximum_log_step:g}, "
                "effective_log_step_target="
                f"{stiffness_effective_log_step_target:g}, "
                "gradient_axes=distance_edge_strain+shape_rigid_aligned, "
                f"hardening_bias={stiffness_hardening_bias:g}, "
                "rejected_ema_keep="
                f"{stiffness_rejected_ema_keep_ratio:g}, "
                f"admission_horizons={STIFFNESS_ADMISSION_HORIZONS}, "
                f"candidate_profile={STIFFNESS_CANDIDATE_PROFILE}, "
                f"admission_mode={STIFFNESS_ADMISSION_MODE}, "
                "causal_schedule=event_driven,no_frame_or_count_limits,"
                "observability="
                f"{(f'finite_confident_AllTracker_depth_3D_target,contact_count>={STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT},edge_strain_particles>={STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES}' if STIFFNESS_ADMISSION_MODE == 'paper_trajectory_adam' else (f'contact_count>={STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT},strain_particles>={STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES},consecutive_windows>={STIFFNESS_DIRECT_MINIMUM_OBSERVABLE_WINDOWS}' if STIFFNESS_ADMISSION_MODE == 'direct_online' else f'active/strain/ema>={STIFFNESS_EVENT_DRIVEN_MINIMUM_ACTIVE_PARTICLES}/{STIFFNESS_EVENT_DRIVEN_MINIMUM_STRAIN_PARTICLES}/{STIFFNESS_EVENT_DRIVEN_MINIMUM_EMA_PARTICLES},contact_count>={STIFFNESS_EVENT_DRIVEN_MINIMUM_CONTACT_COUNT}'))},"
                "admission="
                f"{('per_material_observable_video_frame_bounded_adam' if STIFFNESS_ADMISSION_MODE == 'paper_trajectory_adam' else ('after_observability_immediate_' + ('alternating_axis' if STIFFNESS_CANDIDATE_PROFILE == 'direct_alternating_gradient' else 'dual') + '_gradient_no_shadow' if STIFFNESS_ADMISSION_MODE == 'direct_online' else 'direct_gradient+short_horizons+hard_physics'))},"
                "no_candidate_confirmation,no_candidate_contest,no_long_gate; "
                "distance_bounds="
                f"{stiffness_distance_minimum:g}..{stiffness_distance_maximum:g}, "
                "shape_bounds="
                f"{stiffness_shape_minimum:g}..{stiffness_shape_maximum:g}, "
                "ema=new0.30/history0.70, volume=FIXED, spatial_smoothing=1, "
                "edge_strain_signal="
                f"{stiffness_updater.settings.strain_signal_weight:g}, "
                "minimum_residual/deformation="
                f"{stiffness_minimum_residual_mm:g}/"
                f"{stiffness_minimum_deformation_mm:g}mm, "
                "stiffness_graph_smoothing=2x0.20, "
                "local_tet_quality_gate="
                f"{STIFFNESS_UPDATE_LOCAL_MINIMUM_VOLUME_RATIO:g}"
            )
            if paper_trajectory_stiffness_optimizer is not None:
                paper_settings = (
                    paper_trajectory_stiffness_optimizer.settings
                )
                print(
                    "[example_embodied_super_offline] Liang trajectory Adam: ON; "
                    "frequency=every_contact+edge_strain_observable_video_observation, "
                    "loss=confidence_weighted_3d_huber+4_of_20_history+"
                    "independent_log_graph_smooth, "
                    "gradient=central_SPSA_exact_XPBD_replay, "
                    f"adam_lr={paper_settings.learning_rate:g}, "
                    "realized_material_log_step_cap="
                    f"{paper_settings.maximum_material_log_step:g}, "
                    f"perturbation={paper_settings.parameter_perturbation:g}, "
                    "track/history_scale_mm="
                    f"{paper_settings.track_robust_scale_m * 1e3:g}/"
                    f"{paper_settings.history_scale_m * 1e3:g}, "
                    f"history_weight={paper_settings.history_weight:g}, "
                    "smooth_distance/shape="
                    f"{paper_settings.distance_smooth_weight:g}/"
                    f"{paper_settings.shape_smooth_weight:g}, "
                    "fixed_direction_step=OFF, bounds=sigmoid"
                )

    playback_controls = SuperPlaybackControls(
        environment,
        dataset_manager,
        fps,
        monitor_psm_base_q=monitor_psm_base_q,
        monitor_tissue_q=monitor_tissue_q,
        monitor_interval=monitor_interval,
        psm_roll_offset_deg=psm_roll_offset_deg,
        psm_camera_translation_mm=psm_camera_translation_mm,
        psm_world_translation_mm=psm_world_translation_mm,
        visual_force_update_interval=visual_force_update_interval,
        visual_feedback_mode=visual_feedback_mode,
        visual_residual_mapper=visual_residual_mapper,
        flow_depth_bindings=flow_depth_bindings,
        flow_depth_observations=flow_depth_observations,
        flow_depth_settings=flow_depth_settings,
        visual_residual_gain_profile=visual_residual_gain_profile,
        trajectory_rgb_residual_track_weight=(
            trajectory_rgb_residual_track_weight
        ),
        trajectory_rgb_residual_position_gain=(
            trajectory_rgb_residual_position_gain
        ),
        trajectory_rgb_residual_velocity_gain=(
            trajectory_rgb_residual_velocity_gain
        ),
        trajectory_rgb_residual_maximum_velocity_m_s=(
            trajectory_rgb_residual_maximum_velocity_m_s
        ),
        trajectory_appearance_settings=TrajectoryAppearanceSettings(
            iterations=trajectory_gaussian_appearance_iterations,
            color_learning_rate=trajectory_gaussian_color_learning_rate,
            opacity_learning_rate=(
                trajectory_gaussian_opacity_learning_rate
            ),
            maximum_color_logit_offset=(
                trajectory_gaussian_maximum_color_logit_offset
            ),
            maximum_opacity_logit_offset=(
                trajectory_gaussian_maximum_opacity_logit_offset
            ),
            optimize_opacity=trajectory_gaussian_optimize_opacity,
            formal_image_scale=trajectory_gaussian_appearance_image_scale,
            dssim_weight=trajectory_gaussian_appearance_dssim_weight,
        ),
        stiffness_updater=stiffness_updater,
        paper_trajectory_stiffness_optimizer=(
            paper_trajectory_stiffness_optimizer
        ),
        enable_psm_tissue_contact=psm_tissue_contact,
        stiffness_evaluation_output=stiffness_evaluation_output,
        stiffness_evaluation_horizons=stiffness_evaluation_horizons,
        tissue_benchmark_recorder=(
            SuperTissueBenchmarkRecorder(
                tissue_benchmark_output,
                tissue_benchmark_ground_truth,
                protocol=tissue_benchmark_protocol,
                instrument_masks=VISUAL_FORCE_INSTRUMENT_MASKS,
                render_scale=tissue_benchmark_render_scale,
                reconstruction_test_phase=(
                    tissue_benchmark_reconstruction_test_phase
                ),
                capture_renders=not tissue_benchmark_track_only,
                future_test_start=(
                    tissue_benchmark_future_test_start_frame
                ),
            )
            if tissue_benchmark_output is not None
            else None
        ),
    )
    playback_controls.reset()

    if evaluation_headless:
        if stiffness_evaluation_output is None and tissue_benchmark_output is None:
            raise ValueError(
                "--evaluation-headless requires a stiffness or tissue benchmark output"
            )
        try:
            await run_headless_trajectory_evaluation(
                playback_controls,
                start_frame=evaluation_start_frame,
                frame_count=evaluation_frame_count,
                physics_steps_per_frame=(
                    evaluation_physics_steps_per_frame
                ),
            )
        finally:
            playback_controls.close()
        return

    from embodied_gaussians.vis import EmbodiedGUI

    visualizer = EmbodiedGUI()
    visualizer.set_environment(environment)
    visualizer.viewer_3d.camera_go_zoom = camera_go_zoom
    # SUPER visual-force previews should show the actual target/force scale.
    # The generic viewer defaults (25x pose and 10x soft-force gain) make a
    # millimetre target look as if Gaussians have flown centimetres away.
    visualizer.viewer_3d.settings.visual_forces_pose_scale = 1.0
    visualizer.viewer_3d.settings.visual_forces_scale = 8.0
    # Display-only gain: make the post-clamp particle arrows readable without
    # changing the visual force sent to the physics solver.
    visualizer.viewer_3d.settings.soft_force_display_gain = 60.0
    visualizer.viewer_3d.settings.soft_force_line_width = 2.0
    visualizer.viewer_3d.settings.draw_visual_forces_gaussians_meshes = False
    visualizer.viewer_3d.settings.draw_visual_forces_gaussians_outlines = False
    visualizer.viewer_3d.settings.draw_visual_forces_render = False
    # The SUPER command explicitly opting into visual forces should show the
    # actual post-clamp particle arrows without another expensive gsplat pass.
    visualizer.viewer_3d.settings.draw_visual_forces = (
        visual_feedback_mode == "force" and visual_force_iterations > 0
    )
    # 默认自由视角改成侧视角：从世界 -X 方向看向手术区域，
    # 方便一进 demo 就检查 tissue 高度、z=0 ground plane、相机视锥和 PSM 的相对位置。
    # 这只影响启动后的 3D viewer 初始视角；点击 Go To Camera 仍然会切到真实离线相机视角。
    visualizer.viewer_3d.configure_side_view(
        position=(-0.18, -0.02, 0.053),
        front=(0.995, 0.0, -0.100),
        up=(0.0, 0.0, 1.0),
        activate=True,
    )
    visualizer.callbacks_render.append(playback_controls.draw)

    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(playback_controls.run)
            await visualizer.run()
            nursery.cancel_scope.cancel()
    finally:
        playback_controls.close()


if __name__ == "__main__":
    args = parse_args()
    configure_stiffness_admission_policy(
        parse_stiffness_evaluation_horizons(
            args.stiffness_admission_horizons
        ),
        args.stiffness_candidate_profile,
        args.stiffness_admission_mode,
    )
    dataset_path = resolve_dataset_path(args.dataset)
    print(f"[example_embodied_super_offline] 使用数据集目录: {dataset_path}")
    camera_names = [name.strip() for name in args.cameras.split(",") if name.strip()]
    stiffness_evaluation_horizons = parse_stiffness_evaluation_horizons(
        args.stiffness_evaluation_horizons
    )
    psm_roll_offset_deg = args.psm_roll_offset_deg
    if psm_roll_offset_deg is None:
        psm_roll_offset_deg = (
            0.0
            if args.psm_pose_driver
            in {
                "raw_kinematics",
                "raw_p420006",
                "raw_p420006_stereo_visual",
                "raw_p420006_sam2_online",
                "raw_paper_lnd_sam2_online",
                "raw_paper_lnd_first_stereo_static_q5",
                "raw_paper_lnd_first_stereo_se3_fixed_q5",
                "raw_paper_lnd_sam2_multianchor_closedjaw",
                "raw_paper_lnd_sam2_dense_contact_closedjaw",
                "raw_paper_lnd_sam2_dense_contact_se3_only",
                "raw_paper_lnd_sam2_dense_contact_unbounded_xyz",
            }
            else -27.0
        )
    wp.config.quiet = True
    wp.init()
    trio.run(
        main,
        dataset_path,
        args.fps,
        args.monitor_psm_base_q,
        args.monitor_tissue_q,
        args.monitor_interval,
        args.visual_feedback_mode,
        args.flow_depth_bindings,
        args.flow_depth_observations,
        args.flow_depth_position_gain,
        args.flow_depth_velocity_gain,
        args.flow_depth_absolute_position_weight,
        args.flow_depth_solver_regularization,
        args.flow_depth_solver_iterations,
        args.flow_depth_robust_residual_mm,
        args.flow_depth_maximum_position_correction_mm,
        args.flow_depth_maximum_velocity_correction_m_s,
        args.flow_depth_local_material_relaxation,
        args.flow_depth_local_distance_stiffness,
        args.flow_depth_local_volume_stiffness,
        args.flow_depth_local_shape_stiffness,
        args.visual_force_iterations,
        args.visual_residual_iterations,
        args.visual_residual_learning_rate_m,
        args.visual_residual_maximum_step_m,
        args.visual_residual_previous_carry,
        args.visual_residual_temporal_weight,
        args.visual_residual_magnitude_weight,
        args.visual_residual_gain_profile,
        args.trajectory_rgb_residual_track_weight,
        args.trajectory_rgb_residual_position_gain,
        args.trajectory_rgb_residual_velocity_gain,
        args.trajectory_rgb_residual_maximum_velocity_m_s,
        args.trajectory_gaussian_appearance_iterations,
        args.trajectory_gaussian_color_learning_rate,
        args.trajectory_gaussian_opacity_learning_rate,
        args.trajectory_gaussian_maximum_color_logit_offset,
        args.trajectory_gaussian_maximum_opacity_logit_offset,
        args.trajectory_gaussian_optimize_opacity,
        args.trajectory_gaussian_appearance_image_scale,
        args.trajectory_gaussian_appearance_dssim_weight,
        args.visual_force_update_interval,
        args.online_stiffness_update,
        args.stiffness_log_learning_rate,
        args.stiffness_maximum_log_step,
        args.stiffness_effective_log_step_target,
        args.stiffness_maximum_step_amplification,
        args.stiffness_strain_signal_weight,
        args.stiffness_hardening_bias,
        args.stiffness_minimum_residual_mm,
        args.stiffness_minimum_deformation_mm,
        args.stiffness_rejected_ema_keep_ratio,
        args.stiffness_distance_minimum,
        args.stiffness_distance_maximum,
        args.stiffness_shape_minimum,
        args.stiffness_shape_maximum,
        args.paper_stiffness_adam_learning_rate,
        args.paper_stiffness_perturbation,
        args.paper_stiffness_track_robust_scale_mm,
        args.paper_stiffness_history_scale_mm,
        args.paper_stiffness_history_weight,
        args.paper_stiffness_distance_smooth_weight,
        args.paper_stiffness_shape_smooth_weight,
        args.paper_stiffness_minimum_axis_loss_difference,
        args.paper_stiffness_sim_global_causal,
        args.paper_stiffness_three_of_four_h3,
        args.paper_stiffness_local_distance,
        args.paper_stiffness_local_distance_log_step,
        args.paper_stiffness_local_distance_log_offset,
        args.paper_stiffness_local_distance_minimum_improvement,
        args.stiffness_evaluation_output,
        stiffness_evaluation_horizons,
        args.evaluation_headless,
        args.evaluation_start_frame,
        args.evaluation_frame_count,
        args.evaluation_physics_steps_per_frame,
        args.tissue_benchmark_output,
        args.tissue_benchmark_ground_truth,
        args.tissue_benchmark_protocol,
        args.tissue_benchmark_render_scale,
        args.tissue_benchmark_reconstruction_test_phase,
        args.tissue_benchmark_track_only,
        args.tissue_benchmark_future_test_start_frame,
        camera_names,
        args.camera_go_zoom,
        psm_roll_offset_deg,
        tuple(args.psm_camera_translation_mm),
        tuple(args.psm_world_translation_mm),
        args.psm_pose_driver,
        args.psm_visual_mode,
        args.tissue_mode,
        args.paper_distance_stiffness_initial,
        args.paper_volume_stiffness_initial,
        args.paper_shape_stiffness_initial,
        args.psm_tissue_contact,
        args.calibrated_profile,
    )
