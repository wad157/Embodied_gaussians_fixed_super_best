#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_stiffness_dual_gradient_full_evaluation_20260826_v1}"
BASELINE_ROOT="${BASELINE_ROOT:-${ROOT_DIR}/outputs/super_stiffness_consensus_full_evaluation_20260825_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
GPU_RECONSTRUCTION="${GPU_RECONSTRUCTION:-0}"
GPU_FUTURE="${GPU_FUTURE:-1}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    printf 'Missing Python environment: %s\n' "${ENV_PREFIX}" >&2
    exit 2
fi
if [[ ! -f "${GROUND_TRUTH}" ]]; then
    printf 'Missing frozen 2D/3D ground truth: %s\n' "${GROUND_TRUTH}" >&2
    exit 2
fi
for method in pure_pbd pbd_visual_residual; do
    for protocol in reconstruction_7to1 future_80to20; do
        baseline_result="${BASELINE_ROOT}/${method}/${protocol}/evaluation_results.json"
        if [[ ! -f "${baseline_result}" ]]; then
            printf 'Missing frozen baseline result: %s\n' "${baseline_result}" >&2
            exit 2
        fi
    done
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused output root: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

sha256sum \
    examples/example_embodied_super_offline.py \
    src/embodied_gaussians/embodied_simulator/builder.py \
    src/embodied_gaussians/embodied_simulator/simulator.py \
    src/embodied_gaussians/embodied_simulator/warp.py \
    src/embodied_gaussians/physics_simulator/online_tissue_stiffness.py \
    src/embodied_gaussians/physics_simulator/visual_tissue_residual_mapping.py \
    scripts/run_super_stiffness_temporal_schedule_full_evaluation.sh \
    > "${OUTPUT_ROOT}/CODE_SHA256.txt"
printf '%s\n' "${BASELINE_ROOT}" > "${OUTPUT_ROOT}/BASELINE_REFERENCE.txt"

"${ENV_PREFIX}/bin/python" scripts/audit_super_evaluation_protocol.py \
    --report "${OUTPUT_ROOT}/PROTOCOL_AUDIT.json" \
    > "${OUTPUT_ROOT}/protocol_audit.log" 2>&1

run_case() {
    local gpu_id="$1"
    local protocol="$2"
    local result_dir="${OUTPUT_ROOT}/pbd_visual_residual_online_stiffness/${protocol}"
    local future_args=()
    if [[ "${protocol}" == "future_80to20" ]]; then
        future_args+=(
            --tissue-benchmark-future-test-start-frame 1152
        )
    fi

    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_stiffness_temporal_full_gpu${gpu_id}" \
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
        --paper-distance-stiffness-initial 0.20 \
        --paper-shape-stiffness-initial 0.004 \
        --visual-feedback-mode residual \
        --visual-feedback-update-interval 3 \
        --visual-residual-gain-profile cross_frame_ranked_hold \
        --visual-residual-iterations 8 \
        --visual-residual-learning-rate-m 2e-5 \
        --visual-residual-maximum-step-m 2e-4 \
        --visual-residual-previous-carry 0 \
        --visual-residual-temporal-weight 0.10 \
        --visual-residual-magnitude-weight 0.01 \
        --online-stiffness-update \
        --stiffness-log-learning-rate 0.10 \
        --stiffness-maximum-log-step 0.10 \
        --stiffness-effective-log-step-target 0.08 \
        --stiffness-maximum-step-amplification 32.0 \
        --stiffness-strain-signal-weight 0.80 \
        --stiffness-distance-minimum 0.025 \
        --stiffness-distance-maximum 4.0 \
        --stiffness-shape-minimum 0.001 \
        --stiffness-shape-maximum 0.040 \
        --stiffness-candidate-profile direct_residual_gradient \
        --stiffness-admission-mode causal_fixed_lag \
        --stiffness-admission-horizons 1,5,10 \
        --stiffness-evaluation-output "${result_dir}/diagnostics" \
        --stiffness-evaluation-horizons 1,3,5,10 \
        "${future_args[@]}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --lpips-device cuda:0
}

run_case "${GPU_RECONSTRUCTION}" reconstruction_7to1 \
    > "${OUTPUT_ROOT}/reconstruction_gpu${GPU_RECONSTRUCTION}.log" 2>&1 &
reconstruction_pid=$!
run_case "${GPU_FUTURE}" future_80to20 \
    > "${OUTPUT_ROOT}/future_gpu${GPU_FUTURE}.log" 2>&1 &
future_pid=$!

set +e
wait "${reconstruction_pid}"
reconstruction_status=$?
wait "${future_pid}"
future_status=$?
set -e
printf 'reconstruction_status=%s\nfuture_status=%s\n' \
    "${reconstruction_status}" "${future_status}" \
    > "${OUTPUT_ROOT}/status.txt"
if [[ "${reconstruction_status}" -ne 0 || "${future_status}" -ne 0 ]]; then
    exit 3
fi

ln -s "${BASELINE_ROOT}/pure_pbd" "${OUTPUT_ROOT}/pure_pbd"
ln -s "${BASELINE_ROOT}/pbd_visual_residual" \
    "${OUTPUT_ROOT}/pbd_visual_residual"
"${ENV_PREFIX}/bin/python" scripts/summarize_super_tissue_method_comparison.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/method_comparison.log" 2>&1
"${ENV_PREFIX}/bin/python" scripts/summarize_super_dynamic_covariance_evaluation.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/detailed_summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
printf 'Formal temporal-schedule stiffness evaluation complete: %s\n' \
    "${OUTPUT_ROOT}"
