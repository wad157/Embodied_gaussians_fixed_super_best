#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_one_frame_carry_pareto_20260823_v1}"
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
    local gpu_id="$1"
    local method="$2"
    local protocol="$3"
    local result_dir="${OUTPUT_ROOT}/extreme_hard/${method}/${protocol}"
    local online_flag="--no-online-stiffness-update"
    local diagnostics_name="observer_diagnostics"
    local extra_args=()

    if [[ "${method}" == "residual_online_robust" ]]; then
        online_flag="--online-stiffness-update"
        diagnostics_name="stiffness_diagnostics"
        extra_args+=(
            --stiffness-candidate-profile robust_hierarchical_system_id
            --stiffness-admission-mode strict_all
            --stiffness-admission-horizons 1,3,5,10
        )
    fi

    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_one_frame_carry_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol "${protocol}" \
        --tissue-benchmark-render-scale 0.5 \
        --paper-distance-stiffness-initial 1.60 \
        --paper-shape-stiffness-initial 0.020 \
        --visual-feedback-mode residual \
        --visual-residual-iterations 8 \
        --visual-residual-learning-rate-m 2e-5 \
        --visual-residual-maximum-step-m 2e-4 \
        --visual-residual-previous-carry 0 \
        --visual-residual-temporal-weight 0.10 \
        --visual-residual-magnitude-weight 0.01 \
        --visual-residual-gain-profile cross_frame_hold \
        "${online_flag}" \
        --stiffness-evaluation-output "${result_dir}/${diagnostics_name}" \
        --stiffness-evaluation-horizons 1,3,5,10 \
        "${extra_args[@]}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --lpips-device cuda:0
}

run_case 0 residual_only reconstruction_7to1 \
    > "${OUTPUT_ROOT}/gpu0_reconstruction.log" 2>&1 &
gpu0_pid=$!
run_case 1 residual_online_robust future_80to20 \
    > "${OUTPUT_ROOT}/gpu1_future.log" 2>&1 &
gpu1_pid=$!

set +e
wait "${gpu0_pid}"
gpu0_status=$?
wait "${gpu1_pid}"
gpu1_status=$?
set -e
printf 'gpu0_reconstruction_status=%s\ngpu1_future_status=%s\n' \
    "${gpu0_status}" "${gpu1_status}" > "${OUTPUT_ROOT}/status.txt"
if [[ "${gpu0_status}" -ne 0 || "${gpu1_status}" -ne 0 ]]; then
    exit 3
fi
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
