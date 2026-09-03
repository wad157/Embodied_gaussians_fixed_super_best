#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_tissue_evaluation_residual_online_fixed_20260821}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
PURE_PBD_ROOT="${ROOT_DIR}/outputs/super_tissue_evaluation_formal_20260821/pure_pbd"
REPAIRED_RESIDUAL_ROOT="${ROOT_DIR}/outputs/super_tissue_evaluation_residual_fixed_20260821"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused repaired-online output: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi
for required in \
    "${GROUND_TRUTH}" \
    "${PURE_PBD_ROOT}/reconstruction_7to1/evaluation_results.json" \
    "${PURE_PBD_ROOT}/future_80to20/evaluation_results.json" \
    "${REPAIRED_RESIDUAL_ROOT}/reconstruction_7to1/evaluation_results.json" \
    "${REPAIRED_RESIDUAL_ROOT}/future_80to20/evaluation_results.json"; do
    if [[ ! -f "${required}" ]]; then
        printf 'Missing required comparison input: %s\n' "${required}" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

run_protocol() {
    local protocol="$1"
    local gpu_id="$2"
    local result_dir="${OUTPUT_ROOT}/${protocol}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_eval_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol "${protocol}" \
        --tissue-benchmark-reconstruction-test-phase 0 \
        --tissue-benchmark-render-scale 0.5 \
        --visual-feedback-mode residual \
        --visual-residual-iterations 8 \
        --visual-residual-learning-rate-m 2e-5 \
        --visual-residual-maximum-step-m 2e-4 \
        --visual-residual-previous-carry 0 \
        --visual-residual-temporal-weight 0.10 \
        --visual-residual-magnitude-weight 0.01 \
        --online-stiffness-update \
        --stiffness-log-learning-rate 0.18 \
        --stiffness-evaluation-output "${result_dir}/stiffness_diagnostics" \
        --stiffness-evaluation-horizons 1,3,5,10
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --lpips-device cuda:0
}

run_protocol reconstruction_7to1 0 > "${OUTPUT_ROOT}/reconstruction.log" 2>&1 &
reconstruction_pid=$!
run_protocol future_80to20 1 > "${OUTPUT_ROOT}/future.log" 2>&1 &
future_pid=$!

set +e
wait "${reconstruction_pid}"
reconstruction_status=$?
wait "${future_pid}"
future_status=$?
set -e
printf 'reconstruction_status=%s\nfuture_status=%s\n' \
    "${reconstruction_status}" "${future_status}" > "${OUTPUT_ROOT}/status.txt"
if [[ "${reconstruction_status}" -ne 0 || "${future_status}" -ne 0 ]]; then
    exit 3
fi

"${ENV_PREFIX}/bin/python" scripts/summarize_super_repaired_online_comparison.py \
    --output-root "${OUTPUT_ROOT}" \
    --pure-pbd-root "${PURE_PBD_ROOT}" \
    --repaired-residual-root "${REPAIRED_RESIDUAL_ROOT}" \
    > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
