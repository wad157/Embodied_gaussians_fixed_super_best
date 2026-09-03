#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_sim_particle_graph_lm_third_group_20260902_v1}"
BASELINE_ROOT="${BASELINE_ROOT:-${ROOT_DIR}/outputs/super_trajectory_causal_color_only_full_metrics_20260901_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
RECONSTRUCTION_BINDINGS="${RECONSTRUCTION_BINDINGS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_d1_f1_20260831_v2/bindings.npz}"
RECONSTRUCTION_OBSERVATIONS="${RECONSTRUCTION_OBSERVATIONS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_observations_f1_20260831_v2/observations.npz}"
FUTURE_BINDINGS="${FUTURE_BINDINGS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_d1_f2_20260829_v1/bindings.npz}"
FUTURE_OBSERVATIONS="${FUTURE_OBSERVATIONS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_observations_20260829_v1/observations.npz}"
GPU_RECONSTRUCTION="${GPU_RECONSTRUCTION:-0}"
GPU_FUTURE="${GPU_FUTURE:-1}"
TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS="${TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS:-3}"
STIFFNESS_LOG_LEARNING_RATE="${STIFFNESS_LOG_LEARNING_RATE:-0.03}"
STIFFNESS_MAXIMUM_LOG_STEP="${STIFFNESS_MAXIMUM_LOG_STEP:-0.02}"
METHOD="pbd_alltracker_depth_rgb_residual_online_stiffness"

for path in \
    "${ENV_PREFIX}/bin/python" "${GROUND_TRUTH}" \
    "${RECONSTRUCTION_BINDINGS}" "${RECONSTRUCTION_OBSERVATIONS}" \
    "${FUTURE_BINDINGS}" "${FUTURE_OBSERVATIONS}" \
    "${BASELINE_ROOT}/pure_pbd" \
    "${BASELINE_ROOT}/pbd_alltracker_depth_rgb_residual"
do
    if [[ ! -e "${path}" ]]; then
        printf 'Missing required input: %s\n' "${path}" >&2
        exit 2
    fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused output root: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}/pbd_alltracker_depth_rgb_residual"
ln -s "${BASELINE_ROOT}/pure_pbd" "${OUTPUT_ROOT}/pure_pbd"
for protocol in reconstruction_7to1 future_80to20; do
    ln -s \
        "${BASELINE_ROOT}/pbd_alltracker_depth_rgb_residual/${protocol}" \
        "${OUTPUT_ROOT}/pbd_alltracker_depth_rgb_residual/${protocol}"
done
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples:${ROOT_DIR}/scripts"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

sha256sum \
    examples/example_embodied_super_offline.py \
    src/embodied_gaussians/physics_simulator/sim_particle_graph_stiffness.py \
    src/embodied_gaussians/physics_simulator/flow_depth_particle_observer.py \
    src/embodied_gaussians/physics_simulator/visual_tissue_residual_mapping.py \
    src/embodied_gaussians/embodied_simulator/trajectory_appearance.py \
    scripts/score_super_tissue_evaluation.py \
    scripts/run_super_sim_particle_graph_lm_third_group.sh \
    "${RECONSTRUCTION_BINDINGS}" "${RECONSTRUCTION_OBSERVATIONS}" \
    "${FUTURE_BINDINGS}" "${FUTURE_OBSERVATIONS}" \
    > "${OUTPUT_ROOT}/CODE_AND_INPUT_SHA256.txt"
printf '%s\n' \
    "baseline_root=${BASELINE_ROOT}" \
    'groups_run=third_only' \
    'stiffness_method=SIM_differentiable_particle_graph_lm' \
    'material_field=one_global_mean_plus_one_zero_mean_log_distance_per_particle' \
    'shape_log_coupling=0.15' \
    'volume_and_damping=fixed' \
    'objective=AllTracker_Cauchy_relative_H1_H3_H5_plus_edge_strain' \
    'graph_lm=damping_0.20_spatial_hessian_0.25_CG8' \
    'direction_gate=two_actual_Warp_FD_directions_cosine_at_least_0.90' \
    'update_interval_frames=10' \
    "log_learning_rate=${STIFFNESS_LOG_LEARNING_RATE}" \
    "maximum_log_step=${STIFFNESS_MAXIMUM_LOG_STEP}" \
    "trajectory_gaussian_appearance_iterations=${TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS}" \
    'gaussian_scale_update=physical_mode2_surface_deformation_covariance' \
    'future_feedback=disabled_after_80_percent_boundary' \
    > "${OUTPUT_ROOT}/RUN_CONFIGURATION.txt"

"${ENV_PREFIX}/bin/python" scripts/audit_super_evaluation_protocol.py \
    --report "${OUTPUT_ROOT}/PROTOCOL_AUDIT.json" \
    > "${OUTPUT_ROOT}/protocol_audit.log" 2>&1

run_case() {
    local gpu_id="$1" protocol="$2" bindings="$3" observations="$4"
    local result_dir="${OUTPUT_ROOT}/${METHOD}/${protocol}"
    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_super_graph_lm_gpu${gpu_id}" \
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
        --paper-distance-stiffness-initial 0.01 \
        --paper-volume-stiffness-initial 100000 \
        --paper-shape-stiffness-initial 0.0005 \
        --visual-feedback-mode trajectory_residual \
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
        --trajectory-rgb-residual-track-weight 0.02 \
        --trajectory-rgb-residual-position-gain 1.0 \
        --trajectory-rgb-residual-velocity-gain 0.15 \
        --trajectory-rgb-residual-maximum-velocity-m-s 0.015 \
        --trajectory-gaussian-appearance-iterations "${TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS}" \
        --trajectory-gaussian-color-learning-rate 0.02 \
        --trajectory-gaussian-opacity-learning-rate 0.005 \
        --trajectory-gaussian-maximum-color-logit-offset 0.20 \
        --trajectory-gaussian-maximum-opacity-logit-offset 0.10 \
        --no-trajectory-gaussian-optimize-opacity \
        --trajectory-gaussian-appearance-image-scale 0.5 \
        --trajectory-gaussian-appearance-dssim-weight 0.02 \
        --online-stiffness-update \
        --stiffness-admission-mode sim_particle_graph_lm \
        --stiffness-candidate-profile direct_residual_gradient \
        --stiffness-log-learning-rate "${STIFFNESS_LOG_LEARNING_RATE}" \
        --stiffness-maximum-log-step "${STIFFNESS_MAXIMUM_LOG_STEP}" \
        --stiffness-distance-minimum 0.00001 \
        --stiffness-distance-maximum 4.0 \
        --stiffness-shape-minimum 0.000001 \
        --stiffness-shape-maximum 0.04 \
        --stiffness-evaluation-output "${result_dir}/diagnostics"
    local simulation_status=$?
    if [[ "${simulation_status}" -ne 0 ]]; then
        return "${simulation_status}"
    fi
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --lpips-device cuda:0
}

set +e
run_case "${GPU_RECONSTRUCTION}" reconstruction_7to1 \
    "${RECONSTRUCTION_BINDINGS}" "${RECONSTRUCTION_OBSERVATIONS}" \
    > "${OUTPUT_ROOT}/reconstruction_gpu${GPU_RECONSTRUCTION}.log" 2>&1 &
reconstruction_pid=$!
run_case "${GPU_FUTURE}" future_80to20 \
    "${FUTURE_BINDINGS}" "${FUTURE_OBSERVATIONS}" \
    > "${OUTPUT_ROOT}/future_gpu${GPU_FUTURE}.log" 2>&1 &
future_pid=$!
wait "${reconstruction_pid}"; reconstruction_status=$?
wait "${future_pid}"; future_status=$?
set -e
printf 'reconstruction_status=%s\nfuture_status=%s\n' \
    "${reconstruction_status}" "${future_status}" \
    > "${OUTPUT_ROOT}/status.txt"
if [[ "${reconstruction_status}" -ne 0 || "${future_status}" -ne 0 ]]; then
    exit 3
fi
"${ENV_PREFIX}/bin/python" scripts/summarize_super_tissue_method_comparison.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
"${ENV_PREFIX}/bin/python" scripts/summarize_super_compact_tracking_rendering.py \
    --root "${OUTPUT_ROOT}"
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
printf 'SUPER SIM particle-graph LM third-group evaluation complete: %s\n' \
    "${OUTPUT_ROOT}"
