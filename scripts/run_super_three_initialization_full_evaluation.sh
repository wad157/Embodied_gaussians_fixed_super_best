#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point: keep one protocol-corrected implementation for
# the complete three-method, two-task matrix.
COMPAT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${COMPAT_ROOT_DIR}/scripts/run_super_cross_frame_full_evaluation.sh" "$@"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_three_initialization_full_evaluation_20260822_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    printf 'Missing Python environment: %s\n' "${ENV_PREFIX}" >&2
    exit 2
fi
if [[ ! -f "${GROUND_TRUTH}" ]]; then
    printf 'Missing frozen 2D/3D ground truth: %s\n' "${GROUND_TRUTH}" >&2
    exit 2
fi
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
    local initialization="$1"
    local method="$2"
    local protocol="$3"
    local gpu_id="$4"
    local distance_initial="$5"
    local shape_initial="$6"
    local result_dir="${OUTPUT_ROOT}/${initialization}/${method}/${protocol}"
    local visual_mode="off"
    local online_flag="--no-online-stiffness-update"
    local extra_args=()

    if [[ "${method}" == "residual_only" ]]; then
        visual_mode="residual"
        extra_args+=(--visual-residual-gain-profile multiscale_hold)
    elif [[ "${method}" == "residual_online_robust" ]]; then
        visual_mode="residual"
        online_flag="--online-stiffness-update"
        extra_args+=(
            --visual-residual-gain-profile multiscale_hold
            --stiffness-candidate-profile robust_hierarchical_system_id
            --stiffness-admission-mode strict_all
            --stiffness-admission-horizons 1,3,5,10
            --stiffness-evaluation-output "${result_dir}/stiffness_diagnostics"
            --stiffness-evaluation-horizons 1,3,5,10
        )
    fi

    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_three_init_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol "${protocol}" \
        --tissue-benchmark-render-scale 0.5 \
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
        --lpips-device cuda:0
}

run_pair() {
    local initialization="$1"
    local method="$2"
    local gpu_id="$3"
    local distance="$4"
    local shape="$5"
    run_case "${initialization}" "${method}" reconstruction_7to1 \
        "${gpu_id}" "${distance}" "${shape}"
    run_case "${initialization}" "${method}" future_80to20 \
        "${gpu_id}" "${distance}" "${shape}"
}

run_lane_zero() {
    run_pair extreme_soft pure_pbd 0 0.10 0.003
    run_pair extreme_soft residual_only 0 0.10 0.003
    run_pair extreme_soft residual_online_robust 0 0.10 0.003
    run_pair moderate pure_pbd 0 0.20 0.004
    run_pair moderate residual_only 0 0.20 0.004
}

run_lane_one() {
    run_pair extreme_hard pure_pbd 1 1.60 0.020
    run_pair extreme_hard residual_only 1 1.60 0.020
    run_pair extreme_hard residual_online_robust 1 1.60 0.020
    run_pair moderate residual_online_robust 1 0.20 0.004
}

run_lane_zero > "${OUTPUT_ROOT}/gpu0.log" 2>&1 &
gpu0_pid=$!
run_lane_one > "${OUTPUT_ROOT}/gpu1.log" 2>&1 &
gpu1_pid=$!

set +e
wait "${gpu0_pid}"
gpu0_status=$?
wait "${gpu1_pid}"
gpu1_status=$?
set -e
printf 'gpu0_status=%s\ngpu1_status=%s\n' \
    "${gpu0_status}" "${gpu1_status}" > "${OUTPUT_ROOT}/status.txt"
if [[ "${gpu0_status}" -ne 0 || "${gpu1_status}" -ne 0 ]]; then
    exit 3
fi

"${ENV_PREFIX}/bin/python" \
    scripts/summarize_super_three_initialization_full_evaluation.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
