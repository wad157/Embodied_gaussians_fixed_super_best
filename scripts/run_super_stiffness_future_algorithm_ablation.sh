#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_stiffness_future_algorithm_ablation_20260821}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused algorithm-ablation output: %s\n' "${OUTPUT_ROOT}" >&2
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
    local profile="$3"
    local result_dir="${OUTPUT_ROOT}/${name}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_stiffness_algorithm_gpu${gpu_id}" \
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
        --stiffness-log-learning-rate 0.18 \
        --stiffness-admission-horizons 1,3,5,10 \
        --stiffness-candidate-profile "${profile}" \
        --stiffness-evaluation-output "${result_dir}/stiffness_diagnostics" \
        --stiffness-evaluation-horizons 1,3,5,10
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --skip-lpips
}

run_case h10_bidirectional_8 0 bidirectional_8 \
    > "${OUTPUT_ROOT}/h10_bidirectional_8.log" 2>&1 &
pid_h10=$!
run_case h10_cross_signed_12 1 cross_signed_12 \
    > "${OUTPUT_ROOT}/h10_cross_signed_12.log" 2>&1 &
pid_cross=$!

set +e
wait "${pid_h10}"
status_h10=$?
wait "${pid_cross}"
status_cross=$?
set -e
printf 'h10_bidirectional_8_status=%s\nh10_cross_signed_12_status=%s\n' \
    "${status_h10}" "${status_cross}" > "${OUTPUT_ROOT}/status.txt"
if [[ "${status_h10}" -ne 0 || "${status_cross}" -ne 0 ]]; then
    exit 3
fi

"${ENV_PREFIX}/bin/python" \
    scripts/summarize_super_stiffness_future_algorithm_ablation.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
