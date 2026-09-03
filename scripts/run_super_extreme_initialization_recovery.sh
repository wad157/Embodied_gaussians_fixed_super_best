#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_extreme_initialization_recovery_20260822}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused extreme-initialization output: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

run_case() {
    local initialization="$1"
    local method="$2"
    local gpu_id="$3"
    local distance_initial="$4"
    local shape_initial="$5"
    local visual_mode="off"
    local online_flag="--no-online-stiffness-update"
    local result_dir="${OUTPUT_ROOT}/${initialization}/${method}"
    local extra_args=()

    if [[ "${method}" == "residual_only" ]]; then
        visual_mode="residual"
        extra_args+=(--visual-residual-gain-profile multiscale_hold)
    elif [[ "${method}" == "residual_online_hierarchical" ]]; then
        visual_mode="residual"
        online_flag="--online-stiffness-update"
        extra_args+=(
            --visual-residual-gain-profile multiscale_hold
            --stiffness-candidate-profile hierarchical_system_id
            --stiffness-admission-mode weighted_window
            --stiffness-admission-horizons 1,3,5
            --stiffness-evaluation-output "${result_dir}/stiffness_diagnostics"
            --stiffness-evaluation-horizons 1,3,5,10
        )
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_extreme_recovery_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol future_80to20 \
        --tissue-benchmark-track-only \
        --paper-distance-stiffness-initial "${distance_initial}" \
        --paper-shape-stiffness-initial "${shape_initial}" \
        --visual-feedback-mode "${visual_mode}" \
        --visual-residual-iterations 8 \
        --visual-residual-learning-rate-m 2e-5 \
        --visual-residual-maximum-step-m 2e-4 \
        --visual-residual-previous-carry 0 \
        --visual-residual-temporal-weight 0.10 \
        --visual-residual-magnitude-weight 0.01 \
        "${online_flag}" \
        "${extra_args[@]}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --skip-lpips
}

run_initialization() {
    local name="$1"
    local gpu_id="$2"
    local distance="$3"
    local shape="$4"
    local group_log="${OUTPUT_ROOT}/${name}.log"
    {
        run_case "${name}" pure_pbd "${gpu_id}" "${distance}" "${shape}"
        run_case "${name}" residual_only "${gpu_id}" "${distance}" "${shape}"
        run_case "${name}" residual_online_hierarchical "${gpu_id}" "${distance}" "${shape}"
    } > "${group_log}" 2>&1
}

# Deliberately wrong initializations at opposite ends of the legal search range.
run_initialization extreme_soft 0 0.10 0.003 &
soft_pid=$!
run_initialization extreme_hard 1 1.60 0.020 &
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
    scripts/summarize_super_extreme_initialization_recovery.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
