#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_stiffness_future_merged_components_ablation_20260822}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused merged-components output: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

run_case() {
    local name="$1"
    local gpu_id="$2"
    local log_lr="$3"
    local maximum_log_step="$4"
    local rejected_ema_keep="$5"
    local result_dir="${OUTPUT_ROOT}/${name}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_stiffness_merge_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol future_80to20 \
        --tissue-benchmark-track-only \
        --visual-feedback-mode residual \
        --visual-residual-iterations 8 \
        --visual-residual-learning-rate-m 2e-5 \
        --visual-residual-maximum-step-m 2e-4 \
        --visual-residual-previous-carry 0 \
        --visual-residual-temporal-weight 0.10 \
        --visual-residual-magnitude-weight 0.01 \
        --online-stiffness-update \
        --stiffness-log-learning-rate "${log_lr}" \
        --stiffness-maximum-log-step "${maximum_log_step}" \
        --stiffness-rejected-ema-keep-ratio "${rejected_ema_keep}" \
        --stiffness-admission-horizons 1,3,5 \
        --stiffness-candidate-profile spatial_components_merge_12 \
        --stiffness-evaluation-output "${result_dir}/stiffness_diagnostics" \
        --stiffness-evaluation-horizons 1,3,5,10
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --skip-lpips
}

run_case merged_balanced 0 0.18 0.18 0.85 \
    > "${OUTPUT_ROOT}/merged_balanced.log" 2>&1 &
pid_balanced=$!
run_case merged_strong 1 0.24 0.24 0.95 \
    > "${OUTPUT_ROOT}/merged_strong.log" 2>&1 &
pid_strong=$!

set +e
wait "${pid_balanced}"
status_balanced=$?
wait "${pid_strong}"
status_strong=$?
set -e
printf 'merged_balanced_status=%s\nmerged_strong_status=%s\n' \
    "${status_balanced}" "${status_strong}" > "${OUTPUT_ROOT}/status.txt"
if [[ "${status_balanced}" -ne 0 || "${status_strong}" -ne 0 ]]; then
    exit 3
fi

"${ENV_PREFIX}/bin/python" \
    scripts/summarize_super_stiffness_future_merged_components_ablation.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
