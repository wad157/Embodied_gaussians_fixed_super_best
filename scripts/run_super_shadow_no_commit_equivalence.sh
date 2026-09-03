#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_shadow_no_commit_equivalence_20260824_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
FRAME_COUNT="${FRAME_COUNT:-160}"

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
    local method="$1"
    local online_flag="--no-online-stiffness-update"
    local extra_args=()
    local result_dir="${OUTPUT_ROOT}/${method}"
    if [[ "${method}" == "online_rejected" || "${method}" == "online_deferred" ]]; then
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
            --stiffness-admission-horizons 1,3,5
            --stiffness-evaluation-output "${result_dir}/stiffness"
        )
        if [[ "${method}" == "online_deferred" ]]; then
            # Keep the updater initialization/evidence path active, but move
            # the first validation beyond this short diagnostic.  Comparing
            # this branch with online_rejected isolates shadow execution from
            # the preserve-spatial-stiffness setup itself.
            extra_args+=(--stiffness-admission-horizons 50)
        fi
    fi
    CUDA_VISIBLE_DEVICES=0 \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_shadow_equivalence_gpu0" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count "${FRAME_COUNT}" \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol reconstruction_7to1 \
        --tissue-benchmark-reconstruction-test-phase 2 \
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
        --visual-residual-gain-profile cross_frame_hold \
        "${online_flag}" \
        "${extra_args[@]}"
}

run_case residual_a
run_case residual_b
run_case online_deferred
run_case online_rejected
"${ENV_PREFIX}/bin/python" scripts/compare_super_shadow_no_commit_equivalence.py \
    --root "${OUTPUT_ROOT}"
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
