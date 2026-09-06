#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_alltracker_rgb_sim_h2h3_descent_full_metrics_20260831_v4}"
TRAJECTORY_BASELINE_ROOT="${2:-}"
EXECUTION_MODE="${3:-parallel}"
TRAJECTORY_RGB_RESIDUAL_ENABLED="${4:-${TRAJECTORY_RGB_RESIDUAL_ENABLED:-1}}"
BASELINE_ROOT="${BASELINE_ROOT:-${ROOT_DIR}/outputs/super_paper_trajectory_adam_three_way_full_metrics_20260830_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
RECONSTRUCTION_BINDINGS="${RECONSTRUCTION_BINDINGS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_d1_f1_20260831_v2/bindings.npz}"
RECONSTRUCTION_OBSERVATIONS="${RECONSTRUCTION_OBSERVATIONS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_observations_f1_20260831_v2/observations.npz}"
FUTURE_BINDINGS="${FUTURE_BINDINGS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_d1_f2_20260829_v1/bindings.npz}"
FUTURE_OBSERVATIONS="${FUTURE_OBSERVATIONS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_observations_20260829_v1/observations.npz}"
RERUN_RECONSTRUCTION_TRAJECTORY="${RERUN_RECONSTRUCTION_TRAJECTORY:-1}"
RERUN_FUTURE_TRAJECTORY="${RERUN_FUTURE_TRAJECTORY:-0}"
RERUN_PURE_PBD="${RERUN_PURE_PBD:-0}"
RUN_ONLY_PURE_PBD="${RUN_ONLY_PURE_PBD:-0}"
GPU_RECONSTRUCTION="${GPU_RECONSTRUCTION:-0}"
GPU_FUTURE="${GPU_FUTURE:-1}"
TRAJECTORY_RGB_POSITION_GAIN="${TRAJECTORY_RGB_POSITION_GAIN:-1.0}"
TRAJECTORY_RGB_VELOCITY_GAIN="${TRAJECTORY_RGB_VELOCITY_GAIN:-0.15}"
PAPER_DISTANCE_STIFFNESS_INITIAL="${PAPER_DISTANCE_STIFFNESS_INITIAL:-0.01}"
PAPER_VOLUME_STIFFNESS_INITIAL="${PAPER_VOLUME_STIFFNESS_INITIAL:-100000}"
PAPER_SHAPE_STIFFNESS_INITIAL="${PAPER_SHAPE_STIFFNESS_INITIAL:-0.0005}"
TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS="${TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS:-0}"
TRAJECTORY_GAUSSIAN_COLOR_LR="${TRAJECTORY_GAUSSIAN_COLOR_LR:-0.02}"
TRAJECTORY_GAUSSIAN_OPACITY_LR="${TRAJECTORY_GAUSSIAN_OPACITY_LR:-0.005}"
TRAJECTORY_GAUSSIAN_COLOR_LOGIT_CAP="${TRAJECTORY_GAUSSIAN_COLOR_LOGIT_CAP:-0.20}"
TRAJECTORY_GAUSSIAN_OPACITY_LOGIT_CAP="${TRAJECTORY_GAUSSIAN_OPACITY_LOGIT_CAP:-0.10}"
TRAJECTORY_GAUSSIAN_APPEARANCE_IMAGE_SCALE="${TRAJECTORY_GAUSSIAN_APPEARANCE_IMAGE_SCALE:-0.5}"
TRAJECTORY_GAUSSIAN_APPEARANCE_DSSIM_WEIGHT="${TRAJECTORY_GAUSSIAN_APPEARANCE_DSSIM_WEIGHT:-0.02}"
PAPER_STIFFNESS_THREE_OF_FOUR_H3="${PAPER_STIFFNESS_THREE_OF_FOUR_H3:-0}"
PAPER_STIFFNESS_MAXIMUM_LOG_STEP="${PAPER_STIFFNESS_MAXIMUM_LOG_STEP:-0.02}"
PAPER_STIFFNESS_RECONSTRUCTION_GLOBAL_LOG_OFFSET="${PAPER_STIFFNESS_RECONSTRUCTION_GLOBAL_LOG_OFFSET:-0.15}"
PAPER_STIFFNESS_FUTURE_GLOBAL_LOG_OFFSET="${PAPER_STIFFNESS_FUTURE_GLOBAL_LOG_OFFSET:-0.35}"
PAPER_STIFFNESS_LOCAL_DISTANCE="${PAPER_STIFFNESS_LOCAL_DISTANCE:-0}"
PAPER_STIFFNESS_LOCAL_DISTANCE_LOG_STEP="${PAPER_STIFFNESS_LOCAL_DISTANCE_LOG_STEP:-0.008}"
PAPER_STIFFNESS_LOCAL_DISTANCE_LOG_OFFSET="${PAPER_STIFFNESS_LOCAL_DISTANCE_LOG_OFFSET:-0.04}"
PAPER_STIFFNESS_LOCAL_DISTANCE_MINIMUM_IMPROVEMENT="${PAPER_STIFFNESS_LOCAL_DISTANCE_MINIMUM_IMPROVEMENT:-0.000001}"
PAPER_STIFFNESS_LOCAL_SHAPE="${PAPER_STIFFNESS_LOCAL_SHAPE:-0}"
PAPER_STIFFNESS_LOCAL_SHAPE_LOG_STEP="${PAPER_STIFFNESS_LOCAL_SHAPE_LOG_STEP:-0.008}"
PAPER_STIFFNESS_LOCAL_SHAPE_LOG_OFFSET="${PAPER_STIFFNESS_LOCAL_SHAPE_LOG_OFFSET:-0.04}"
PAPER_STIFFNESS_LOCAL_CROSS_WINDOW="${PAPER_STIFFNESS_LOCAL_CROSS_WINDOW:-0}"
PAPER_STIFFNESS_LOCAL_TAIL_RELATIVE_TOLERANCE="${PAPER_STIFFNESS_LOCAL_TAIL_RELATIVE_TOLERANCE:-0.001}"
PAPER_STIFFNESS_DISTANCE_SMOOTH_WEIGHT="${PAPER_STIFFNESS_DISTANCE_SMOOTH_WEIGHT:-0.0}"
PAPER_STIFFNESS_SHAPE_SMOOTH_WEIGHT="${PAPER_STIFFNESS_SHAPE_SMOOTH_WEIGHT:-0.0}"

for path in \
    "${ENV_PREFIX}/bin/python" \
    "${GROUND_TRUTH}" \
    "${RECONSTRUCTION_BINDINGS}" \
    "${RECONSTRUCTION_OBSERVATIONS}" \
    "${FUTURE_BINDINGS}" \
    "${FUTURE_OBSERVATIONS}"
do
    if [[ ! -e "${path}" ]]; then
        printf 'Missing required input: %s\n' "${path}" >&2
        exit 2
    fi
done
if [[ "${RERUN_PURE_PBD}" != "1" && ! -e "${BASELINE_ROOT}/pure_pbd" ]]; then
    printf 'Missing required Pure PBD baseline: %s\n' \
        "${BASELINE_ROOT}/pure_pbd" >&2
    exit 2
fi
if [[ "${RUN_ONLY_PURE_PBD}" == "1" && "${RERUN_PURE_PBD}" != "1" ]]; then
    printf 'RUN_ONLY_PURE_PBD requires RERUN_PURE_PBD=1\n' >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused output root: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi
if [[ "${EXECUTION_MODE}" != "parallel" && "${EXECUTION_MODE}" != "serial" ]]; then
    printf 'Execution mode must be parallel or serial: %s\n' \
        "${EXECUTION_MODE}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ "${RERUN_PURE_PBD}" != "1" ]]; then
    ln -s "${BASELINE_ROOT}/pure_pbd" "${OUTPUT_ROOT}/pure_pbd"
fi
if [[ -n "${TRAJECTORY_BASELINE_ROOT}" ]]; then
    mkdir -p "${OUTPUT_ROOT}/pbd_alltracker_depth_rgb_residual"
    for protocol in reconstruction_7to1 future_80to20; do
        path="${TRAJECTORY_BASELINE_ROOT}/pbd_alltracker_depth_rgb_residual/${protocol}/evaluation_results.json"
        if [[ ! -f "${path}" ]]; then
            printf 'Missing frozen trajectory baseline: %s\n' "${path}" >&2
            exit 2
        fi
        rerun_flag="${RERUN_FUTURE_TRAJECTORY}"
        if [[ "${protocol}" == "reconstruction_7to1" ]]; then
            rerun_flag="${RERUN_RECONSTRUCTION_TRAJECTORY}"
        fi
        if [[ "${rerun_flag}" != "1" ]]; then
            ln -s \
                "${TRAJECTORY_BASELINE_ROOT}/pbd_alltracker_depth_rgb_residual/${protocol}" \
                "${OUTPUT_ROOT}/pbd_alltracker_depth_rgb_residual/${protocol}"
        fi
    done
fi
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples:${ROOT_DIR}/scripts"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

sha256sum \
    examples/example_embodied_super_offline.py \
    src/embodied_gaussians/embodied_simulator/simulator.py \
    src/embodied_gaussians/physics_simulator/flow_depth_particle_observer.py \
    src/embodied_gaussians/physics_simulator/visual_tissue_residual_mapping.py \
    src/embodied_gaussians/physics_simulator/online_tissue_stiffness.py \
    src/embodied_gaussians/physics_simulator/paper_trajectory_stiffness.py \
    src/embodied_gaussians/embodied_simulator/trajectory_appearance.py \
    scripts/score_super_tissue_evaluation.py \
    scripts/summarize_super_tissue_method_comparison.py \
    scripts/summarize_super_compact_tracking_rendering.py \
    scripts/run_super_alltracker_rgb_observable_adam_full_metrics.sh \
    "${RECONSTRUCTION_BINDINGS}" "${RECONSTRUCTION_OBSERVATIONS}" \
    "${FUTURE_BINDINGS}" "${FUTURE_OBSERVATIONS}" \
    > "${OUTPUT_ROOT}/CODE_AND_INPUT_SHA256.txt"
printf 'trajectory_rgb_residual_position_gain=%s\n' \
    "${TRAJECTORY_RGB_POSITION_GAIN}" \
    > "${OUTPUT_ROOT}/RUN_CONFIGURATION.txt"
printf 'trajectory_rgb_residual_enabled=%s\n' \
    "${TRAJECTORY_RGB_RESIDUAL_ENABLED}" \
    >> "${OUTPUT_ROOT}/RUN_CONFIGURATION.txt"
printf 'trajectory_rgb_residual_velocity_gain=%s\n' \
    "${TRAJECTORY_RGB_VELOCITY_GAIN}" \
    >> "${OUTPUT_ROOT}/RUN_CONFIGURATION.txt"
printf '%s\n' \
    "paper_distance_stiffness_initial=${PAPER_DISTANCE_STIFFNESS_INITIAL}" \
    "paper_volume_stiffness_initial=${PAPER_VOLUME_STIFFNESS_INITIAL}" \
    "paper_shape_stiffness_initial=${PAPER_SHAPE_STIFFNESS_INITIAL}" \
    "trajectory_gaussian_appearance_iterations=${TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS}" \
    "trajectory_gaussian_color_lr=${TRAJECTORY_GAUSSIAN_COLOR_LR}" \
    "trajectory_gaussian_opacity_lr=${TRAJECTORY_GAUSSIAN_OPACITY_LR}" \
    "trajectory_gaussian_color_logit_cap=${TRAJECTORY_GAUSSIAN_COLOR_LOGIT_CAP}" \
    "trajectory_gaussian_opacity_logit_cap=${TRAJECTORY_GAUSSIAN_OPACITY_LOGIT_CAP}" \
    'trajectory_gaussian_optimized_properties=rgb_color_only' \
    'trajectory_gaussian_geometry_owner=physics_skinning' \
    'trajectory_gaussian_geometry_appearance=fixed_initial_reference' \
    "trajectory_gaussian_formal_image_scale=${TRAJECTORY_GAUSSIAN_APPEARANCE_IMAGE_SCALE}" \
    "trajectory_gaussian_dssim_weight=${TRAJECTORY_GAUSSIAN_APPEARANCE_DSSIM_WEIGHT}" \
    'trajectory_gaussian_loss=formal_left_noninstrument_mse_plus_dssim' \
    'trajectory_gaussian_acceptance=next_legal_training_frame_validation' \
    'trajectory_gaussian_future_feedback=OFF' \
    >> "${OUTPUT_ROOT}/RUN_CONFIGURATION.txt"
printf '%s\n' \
    'reconstruction_observations=stride1' \
    'reconstruction_stiffness=one_nonoverlapping_H3_per_7to1_training_block' \
    'reconstruction_h2_updates=OFF' \
    'reconstruction_h3_tail_H2_cosine_minimum=0.90' \
    'reconstruction_h3_cumulative_log_offset=0.15' \
    'future_observations=stride2_v9_frozen_policy' \
    'future_stiffness=available_H2_to_H3_v9' \
    'stiffness_horizon_weights=1.5,2,3' \
    'stiffness_h2_step_scale=3.5/6.5' \
    'stiffness_h2_cross_block_cosine_minimum=0.95' \
    'stiffness_h2_cumulative_log_offset=0.10' \
    'stiffness_direction_gate=cosine>=0.95_and_gradient_dot_step<0' \
    "paper_stiffness_maximum_log_step=${PAPER_STIFFNESS_MAXIMUM_LOG_STEP}" \
    "paper_stiffness_reconstruction_global_log_offset=${PAPER_STIFFNESS_RECONSTRUCTION_GLOBAL_LOG_OFFSET}" \
    "paper_stiffness_future_global_log_offset=${PAPER_STIFFNESS_FUTURE_GLOBAL_LOG_OFFSET}" \
    "paper_stiffness_three_of_four_h3=${PAPER_STIFFNESS_THREE_OF_FOUR_H3}" \
    "paper_stiffness_local_distance=${PAPER_STIFFNESS_LOCAL_DISTANCE}" \
    "paper_stiffness_local_distance_log_step=${PAPER_STIFFNESS_LOCAL_DISTANCE_LOG_STEP}" \
    "paper_stiffness_local_distance_log_offset=${PAPER_STIFFNESS_LOCAL_DISTANCE_LOG_OFFSET}" \
    "paper_stiffness_local_distance_minimum_improvement=${PAPER_STIFFNESS_LOCAL_DISTANCE_MINIMUM_IMPROVEMENT}" \
    "paper_stiffness_local_shape=${PAPER_STIFFNESS_LOCAL_SHAPE}" \
    "paper_stiffness_local_shape_log_step=${PAPER_STIFFNESS_LOCAL_SHAPE_LOG_STEP}" \
    "paper_stiffness_local_shape_log_offset=${PAPER_STIFFNESS_LOCAL_SHAPE_LOG_OFFSET}" \
    "paper_stiffness_local_cross_window=${PAPER_STIFFNESS_LOCAL_CROSS_WINDOW}" \
    "paper_stiffness_local_tail_relative_tolerance=${PAPER_STIFFNESS_LOCAL_TAIL_RELATIVE_TOLERANCE}" \
    "paper_stiffness_distance_smooth_weight=${PAPER_STIFFNESS_DISTANCE_SMOOTH_WEIGHT}" \
    "paper_stiffness_shape_smooth_weight=${PAPER_STIFFNESS_SHAPE_SMOOTH_WEIGHT}" \
    >> "${OUTPUT_ROOT}/RUN_CONFIGURATION.txt"
printf 'trajectory_baseline_root=%s\n' \
    "${TRAJECTORY_BASELINE_ROOT:-rerun}" \
    >> "${OUTPUT_ROOT}/RUN_CONFIGURATION.txt"
printf 'execution_mode=%s\n' "${EXECUTION_MODE}" \
    >> "${OUTPUT_ROOT}/RUN_CONFIGURATION.txt"
printf 'rerun_pure_pbd=%s\nrun_only_pure_pbd=%s\n' \
    "${RERUN_PURE_PBD}" "${RUN_ONLY_PURE_PBD}" \
    >> "${OUTPUT_ROOT}/RUN_CONFIGURATION.txt"

"${ENV_PREFIX}/bin/python" scripts/audit_super_evaluation_protocol.py \
    --report "${OUTPUT_ROOT}/PROTOCOL_AUDIT.json" \
    > "${OUTPUT_ROOT}/protocol_audit.log" 2>&1

run_case() {
    local gpu_id="$1" protocol="$2" method="$3"
    local result_dir="${OUTPUT_ROOT}/${method}/${protocol}"
    local bindings="${FUTURE_BINDINGS}"
    local observations="${FUTURE_OBSERVATIONS}"
    local online_flag="--no-online-stiffness-update"
    local method_args=()
    local trajectory_rgb_flag="--trajectory-rgb-residual-enabled"
    local visual_feedback_mode="trajectory_residual"

    if [[ "${TRAJECTORY_RGB_RESIDUAL_ENABLED}" == "0" ]]; then
        trajectory_rgb_flag="--no-trajectory-rgb-residual-enabled"
    fi
    if [[ "${method}" == "pure_pbd" ]]; then
        visual_feedback_mode="off"
    fi

    if [[ "${protocol}" == "reconstruction_7to1" ]]; then
        bindings="${RECONSTRUCTION_BINDINGS}"
        observations="${RECONSTRUCTION_OBSERVATIONS}"
    fi

    if [[ "${method}" == "pbd_alltracker_depth_rgb_residual_online_stiffness" ]]; then
        online_flag="--online-stiffness-update"
        method_args+=(
            --stiffness-admission-mode paper_trajectory_adam
            --stiffness-candidate-profile direct_residual_gradient
            --stiffness-distance-minimum 0.00001
            --stiffness-distance-maximum 4.0
            --stiffness-shape-minimum 0.000001
            --stiffness-shape-maximum 0.04
            --stiffness-maximum-log-step "${PAPER_STIFFNESS_MAXIMUM_LOG_STEP}"
            --paper-stiffness-adam-learning-rate 0.03
            --paper-stiffness-perturbation 0.005
            --paper-stiffness-sim-global-causal
            --paper-stiffness-track-robust-scale-mm 0.2
            --paper-stiffness-history-scale-mm 1.0
            --paper-stiffness-history-weight 0.0
            --paper-stiffness-distance-smooth-weight "${PAPER_STIFFNESS_DISTANCE_SMOOTH_WEIGHT}"
            --paper-stiffness-shape-smooth-weight "${PAPER_STIFFNESS_SHAPE_SMOOTH_WEIGHT}"
            --paper-stiffness-minimum-axis-loss-difference 1e-7
            --paper-stiffness-reconstruction-global-log-offset "${PAPER_STIFFNESS_RECONSTRUCTION_GLOBAL_LOG_OFFSET}"
            --paper-stiffness-future-global-log-offset "${PAPER_STIFFNESS_FUTURE_GLOBAL_LOG_OFFSET}"
        )
        if [[ "${PAPER_STIFFNESS_THREE_OF_FOUR_H3}" == "1" ]]; then
            method_args+=(--paper-stiffness-three-of-four-h3)
        fi
        if [[ "${PAPER_STIFFNESS_LOCAL_DISTANCE}" == "1" ]]; then
            method_args+=(
                --paper-stiffness-local-distance
                --paper-stiffness-local-distance-log-step "${PAPER_STIFFNESS_LOCAL_DISTANCE_LOG_STEP}"
                --paper-stiffness-local-distance-log-offset "${PAPER_STIFFNESS_LOCAL_DISTANCE_LOG_OFFSET}"
                --paper-stiffness-local-distance-minimum-improvement "${PAPER_STIFFNESS_LOCAL_DISTANCE_MINIMUM_IMPROVEMENT}"
            )
        fi
        if [[ "${PAPER_STIFFNESS_LOCAL_SHAPE}" == "1" ]]; then
            method_args+=(
                --paper-stiffness-local-shape
                --paper-stiffness-local-shape-log-step "${PAPER_STIFFNESS_LOCAL_SHAPE_LOG_STEP}"
                --paper-stiffness-local-shape-log-offset "${PAPER_STIFFNESS_LOCAL_SHAPE_LOG_OFFSET}"
            )
        fi
        if [[ "${PAPER_STIFFNESS_LOCAL_CROSS_WINDOW}" == "1" ]]; then
            method_args+=(--paper-stiffness-local-cross-window-confirmation)
        fi
        method_args+=(
            --paper-stiffness-local-tail-relative-tolerance "${PAPER_STIFFNESS_LOCAL_TAIL_RELATIVE_TOLERANCE}"
        )
    fi

    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_alltracker_rgb_observable_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol "${protocol}" \
        --tissue-benchmark-render-scale 0.5 \
        --tissue-benchmark-reconstruction-test-phase 0 \
        --paper-distance-stiffness-initial "${PAPER_DISTANCE_STIFFNESS_INITIAL}" \
        --paper-volume-stiffness-initial "${PAPER_VOLUME_STIFFNESS_INITIAL}" \
        --paper-shape-stiffness-initial "${PAPER_SHAPE_STIFFNESS_INITIAL}" \
        --visual-feedback-mode "${visual_feedback_mode}" \
        --flow-depth-bindings "${bindings}" \
        --flow-depth-observations "${observations}" \
        --flow-depth-position-gain 1.0 \
        --flow-depth-velocity-gain 1.0 \
        --flow-depth-absolute-position-weight 1.0 \
        --flow-depth-solver-regularization 0.005 \
        --flow-depth-solver-iterations 16 \
        --flow-depth-robust-residual-mm 5.0 \
        --flow-depth-maximum-position-correction-mm 5.0 \
        --flow-depth-maximum-velocity-correction-m-s 0.10 \
        --visual-residual-iterations 8 \
        --visual-residual-learning-rate-m 1e-5 \
        --visual-residual-maximum-step-m 0.00025 \
        --visual-residual-previous-carry 0.0 \
        --visual-residual-temporal-weight 0.10 \
        --visual-residual-magnitude-weight 0.01 \
        "${trajectory_rgb_flag}" \
        --trajectory-rgb-residual-track-weight 0.02 \
        --trajectory-rgb-residual-position-gain "${TRAJECTORY_RGB_POSITION_GAIN}" \
        --trajectory-rgb-residual-velocity-gain "${TRAJECTORY_RGB_VELOCITY_GAIN}" \
        --trajectory-rgb-residual-maximum-velocity-m-s 0.015 \
        --trajectory-gaussian-appearance-iterations "${TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS}" \
        --trajectory-gaussian-color-learning-rate "${TRAJECTORY_GAUSSIAN_COLOR_LR}" \
        --trajectory-gaussian-opacity-learning-rate "${TRAJECTORY_GAUSSIAN_OPACITY_LR}" \
        --trajectory-gaussian-maximum-color-logit-offset "${TRAJECTORY_GAUSSIAN_COLOR_LOGIT_CAP}" \
        --trajectory-gaussian-maximum-opacity-logit-offset "${TRAJECTORY_GAUSSIAN_OPACITY_LOGIT_CAP}" \
        --no-trajectory-gaussian-optimize-opacity \
        --trajectory-gaussian-appearance-image-scale "${TRAJECTORY_GAUSSIAN_APPEARANCE_IMAGE_SCALE}" \
        --trajectory-gaussian-appearance-dssim-weight "${TRAJECTORY_GAUSSIAN_APPEARANCE_DSSIM_WEIGHT}" \
        "${online_flag}" \
        --stiffness-evaluation-output "${result_dir}/diagnostics" \
        "${method_args[@]}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --lpips-device cuda:0
}

run_protocol_lane() {
    local gpu_id="$1" protocol="$2"
    if [[ "${RERUN_PURE_PBD}" == "1" ]]; then
        run_case "${gpu_id}" "${protocol}" pure_pbd
    fi
    if [[ "${RUN_ONLY_PURE_PBD}" == "1" ]]; then
        return
    fi
    if [[ ! -e "${OUTPUT_ROOT}/pbd_alltracker_depth_rgb_residual/${protocol}" ]]; then
        run_case "${gpu_id}" "${protocol}" \
            pbd_alltracker_depth_rgb_residual
    fi
    run_case "${gpu_id}" "${protocol}" \
        pbd_alltracker_depth_rgb_residual_online_stiffness
}

set +e
if [[ "${EXECUTION_MODE}" == "parallel" ]]; then
    run_protocol_lane "${GPU_RECONSTRUCTION}" reconstruction_7to1 \
        > "${OUTPUT_ROOT}/reconstruction_gpu${GPU_RECONSTRUCTION}.log" 2>&1 &
    reconstruction_pid=$!
    run_protocol_lane "${GPU_FUTURE}" future_80to20 \
        > "${OUTPUT_ROOT}/future_gpu${GPU_FUTURE}.log" 2>&1 &
    future_pid=$!
    wait "${reconstruction_pid}"; reconstruction_status=$?
    wait "${future_pid}"; future_status=$?
else
    run_protocol_lane "${GPU_RECONSTRUCTION}" reconstruction_7to1 \
        > "${OUTPUT_ROOT}/reconstruction_gpu${GPU_RECONSTRUCTION}.log" 2>&1
    reconstruction_status=$?
    if [[ "${reconstruction_status}" -eq 0 ]]; then
        run_protocol_lane "${GPU_FUTURE}" future_80to20 \
            > "${OUTPUT_ROOT}/future_gpu${GPU_FUTURE}.log" 2>&1
        future_status=$?
    else
        future_status=125
    fi
fi
set -e
printf 'reconstruction_status=%s\nfuture_status=%s\n' \
    "${reconstruction_status}" "${future_status}" \
    > "${OUTPUT_ROOT}/status.txt"
if [[ "${reconstruction_status}" -ne 0 || "${future_status}" -ne 0 ]]; then
    exit 3
fi

if [[ "${RUN_ONLY_PURE_PBD}" == "1" ]]; then
    printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
    printf 'Pure PBD matched-initialization evaluation complete: %s\n' \
        "${OUTPUT_ROOT}"
    exit 0
fi

"${ENV_PREFIX}/bin/python" scripts/summarize_super_tissue_method_comparison.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
"${ENV_PREFIX}/bin/python" scripts/summarize_super_compact_tracking_rendering.py \
    --root "${OUTPUT_ROOT}"
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
printf 'AllTracker trajectory + RGB residual + observable Adam evaluation complete: %s\n' \
    "${OUTPUT_ROOT}"
