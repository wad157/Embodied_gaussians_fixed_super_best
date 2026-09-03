#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_ranked_causal_extreme_selection_20260824_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
FRAME_COUNT=480
FUTURE_VALIDATION_START=384
RECONSTRUCTION_PHASE=2

if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused output root: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

run_case() {
    local gpu_id="$1"
    local initialization="$2"
    local distance_initial="$3"
    local shape_initial="$4"
    local protocol="$5"
    local method="$6"
    local online_flag="--no-online-stiffness-update"
    local future_args=()
    local score_args=()
    local result_dir="${OUTPUT_ROOT}/${initialization}/${method}/${protocol}"
    local extra_args=(
        --visual-residual-gain-profile cross_frame_ranked_hold
        --stiffness-evaluation-output "${result_dir}/diagnostics"
    )
    if [[ "${protocol}" == "future_80to20" ]]; then
        future_args+=(
            --tissue-benchmark-future-test-start-frame \
                "${FUTURE_VALIDATION_START}"
        )
        score_args+=(--allow-prefix-validation)
    fi
    if [[ "${method}" == "ranked_residual_online_causal" ]]; then
        online_flag="--online-stiffness-update"
        extra_args+=(
            --stiffness-log-learning-rate 0.12
            --stiffness-maximum-log-step 0.12
            --stiffness-strain-signal-weight 0.50
            --stiffness-distance-minimum 0.025
            --stiffness-distance-maximum 4.0
            --stiffness-shape-minimum 0.001
            --stiffness-shape-maximum 0.040
            --stiffness-candidate-profile direct_residual_gradient
            --stiffness-admission-mode causal_fixed_lag
            --stiffness-admission-horizons 1,3,5,10
            --stiffness-evaluation-horizons 1,3,5,10
        )
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_ranked_causal_extreme_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count "${FRAME_COUNT}" \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol "${protocol}" \
        --tissue-benchmark-reconstruction-test-phase \
            "${RECONSTRUCTION_PHASE}" \
        --tissue-benchmark-track-only \
        --paper-distance-stiffness-initial "${distance_initial}" \
        --paper-shape-stiffness-initial "${shape_initial}" \
        --visual-feedback-mode residual \
        --visual-residual-iterations 8 \
        --visual-residual-learning-rate-m 2e-5 \
        --visual-residual-maximum-step-m 2e-4 \
        --visual-residual-previous-carry 0 \
        --visual-residual-temporal-weight 0.10 \
        --visual-residual-magnitude-weight 0.01 \
        "${online_flag}" \
        "${future_args[@]}" \
        "${extra_args[@]}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --skip-lpips \
        "${score_args[@]}"
}

run_initialization() {
    local gpu_id="$1"
    local initialization="$2"
    local distance_initial="$3"
    local shape_initial="$4"
    for protocol in reconstruction_7to1 future_80to20; do
        run_case "${gpu_id}" "${initialization}" \
            "${distance_initial}" "${shape_initial}" "${protocol}" \
            ranked_residual
        run_case "${gpu_id}" "${initialization}" \
            "${distance_initial}" "${shape_initial}" "${protocol}" \
            ranked_residual_online_causal
    done
}

run_initialization 0 extreme_soft 0.025 0.001 \
    > "${OUTPUT_ROOT}/extreme_soft_gpu0.log" 2>&1 &
soft_pid=$!
run_initialization 1 extreme_hard 4.0 0.040 \
    > "${OUTPUT_ROOT}/extreme_hard_gpu1.log" 2>&1 &
hard_pid=$!

set +e
wait "${soft_pid}"
soft_status=$?
wait "${hard_pid}"
hard_status=$?
set -e
printf 'extreme_soft_status=%s\nextreme_hard_status=%s\n' \
    "${soft_status}" "${hard_status}" > "${OUTPUT_ROOT}/status.txt"
if [[ "${soft_status}" -ne 0 || "${hard_status}" -ne 0 ]]; then
    exit 3
fi
"${ENV_PREFIX}/bin/python" \
    scripts/summarize_super_ranked_causal_extreme_selection.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
