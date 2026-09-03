#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_residual_future_full_ablation_20260821}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
PURE_PBD="${ROOT_DIR}/outputs/super_tissue_evaluation_formal_20260821/pure_pbd/future_80to20"
RESIDUAL_0P20="${ROOT_DIR}/outputs/super_tissue_evaluation_residual_fixed_20260821/future_80to20"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused full-ablation output: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi
for required in "${GROUND_TRUTH}" "${PURE_PBD}/evaluation_results.json" "${RESIDUAL_0P20}/evaluation_results.json"; do
    if [[ ! -f "${required}" ]]; then
        printf 'Missing required input: %s\n' "${required}" >&2
        exit 2
    fi
done
mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

run_case() {
    local name="$1"
    local gpu_id="$2"
    local learning_rate_m="$3"
    local maximum_step_m="$4"
    local result_dir="${OUTPUT_ROOT}/${name}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_eval_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol future_80to20 \
        --tissue-benchmark-render-scale 0.5 \
        --visual-feedback-mode residual \
        --no-online-stiffness-update \
        --visual-residual-iterations 8 \
        --visual-residual-learning-rate-m "${learning_rate_m}" \
        --visual-residual-maximum-step-m "${maximum_step_m}" \
        --visual-residual-previous-carry 0 \
        --visual-residual-temporal-weight 0.10 \
        --visual-residual-magnitude-weight 0.01
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --lpips-device cuda:0
}

run_case residual_step_0p30mm 0 3e-5 3e-4 > "${OUTPUT_ROOT}/gpu0.log" 2>&1 &
gpu0_pid=$!
run_case residual_step_0p40mm 1 4e-5 4e-4 > "${OUTPUT_ROOT}/gpu1.log" 2>&1 &
gpu1_pid=$!

set +e
wait "${gpu0_pid}"
gpu0_status=$?
wait "${gpu1_pid}"
gpu1_status=$?
set -e
printf 'gpu0_status=%s\ngpu1_status=%s\n' "${gpu0_status}" "${gpu1_status}" \
    > "${OUTPUT_ROOT}/status.txt"
if [[ "${gpu0_status}" -ne 0 || "${gpu1_status}" -ne 0 ]]; then
    exit 3
fi
"${ENV_PREFIX}/bin/python" scripts/summarize_super_residual_future_full_ablation.py \
    --output-root "${OUTPUT_ROOT}" \
    --pure-pbd "${PURE_PBD}" \
    --residual-0p20 "${RESIDUAL_0P20}" \
    > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
