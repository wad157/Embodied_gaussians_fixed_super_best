#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_visual_ranked_validation_20260824_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
RECONSTRUCTION_PHASE=2
# Selection data stops far before the frozen formal future boundary at 1152.
FUTURE_FRAME_COUNT=480
FUTURE_VALIDATION_START=384

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
    local method="$2"
    local protocol="$3"
    local frame_count=1440
    local future_args=()
    local score_args=()
    local result_dir="${OUTPUT_ROOT}/${method}/${protocol}"
    if [[ "${protocol}" == "future_80to20" ]]; then
        frame_count="${FUTURE_FRAME_COUNT}"
        future_args+=(
            --tissue-benchmark-future-test-start-frame \
                "${FUTURE_VALIDATION_START}"
        )
        score_args+=(--allow-prefix-validation)
    fi
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_visual_ranked_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count "${frame_count}" \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol "${protocol}" \
        --tissue-benchmark-reconstruction-test-phase \
            "${RECONSTRUCTION_PHASE}" \
        --tissue-benchmark-track-only \
        --paper-distance-stiffness-initial 0.20 \
        --paper-shape-stiffness-initial 0.004 \
        --visual-feedback-mode residual \
        --visual-residual-iterations 8 \
        --visual-residual-learning-rate-m 2e-5 \
        --visual-residual-maximum-step-m 2e-4 \
        --visual-residual-previous-carry 0 \
        --visual-residual-temporal-weight 0.10 \
        --visual-residual-magnitude-weight 0.01 \
        --visual-residual-gain-profile "${method}" \
        --stiffness-evaluation-output "${result_dir}/visual_diagnostics" \
        --no-online-stiffness-update \
        "${future_args[@]}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --skip-lpips \
        "${score_args[@]}"
}

(
    run_case 0 cross_frame_hold reconstruction_7to1
    run_case 0 cross_frame_ranked_hold reconstruction_7to1
) > "${OUTPUT_ROOT}/reconstruction_gpu0.log" 2>&1 &
reconstruction_pid=$!
(
    run_case 1 cross_frame_hold future_80to20
    run_case 1 cross_frame_ranked_hold future_80to20
) > "${OUTPUT_ROOT}/future_gpu1.log" 2>&1 &
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
"${ENV_PREFIX}/bin/python" \
    scripts/summarize_super_visual_ranked_validation.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
