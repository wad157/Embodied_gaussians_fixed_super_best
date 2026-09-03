#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_causal_fixed_lag_validation_20260824_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
RECONSTRUCTION_PHASE=2
FUTURE_PREFIX_START=921
FUTURE_PREFIX_COUNT=1152

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

"${ENV_PREFIX}/bin/python" scripts/audit_super_evaluation_protocol.py \
    --report "${OUTPUT_ROOT}/PROTOCOL_AUDIT.json" \
    > "${OUTPUT_ROOT}/protocol_audit.log" 2>&1

run_case() {
    local method="$1"
    local protocol="$2"
    local gpu_id="$3"
    local frame_count=1440
    local future_args=()
    local visual_mode="off"
    local online_flag="--no-online-stiffness-update"
    local extra_args=()
    local score_args=()
    local result_dir="${OUTPUT_ROOT}/${method}/${protocol}"

    if [[ "${protocol}" == "future_80to20" ]]; then
        frame_count="${FUTURE_PREFIX_COUNT}"
        future_args+=(
            --tissue-benchmark-future-test-start-frame "${FUTURE_PREFIX_START}"
        )
        score_args+=(--allow-prefix-validation)
    fi
    if [[ "${method}" == "visual_residual" ]]; then
        visual_mode="residual"
        extra_args+=(--visual-residual-gain-profile cross_frame_hold)
    elif [[ "${method}" == "visual_residual_online_causal" ]]; then
        visual_mode="residual"
        online_flag="--online-stiffness-update"
        extra_args+=(
            --visual-residual-gain-profile cross_frame_hold
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
            --stiffness-evaluation-output "${result_dir}/stiffness_diagnostics"
            --stiffness-evaluation-horizons 1,3,5,10
        )
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_causal_validation_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count "${frame_count}" \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol "${protocol}" \
        --tissue-benchmark-reconstruction-test-phase "${RECONSTRUCTION_PHASE}" \
        --tissue-benchmark-track-only \
        --paper-distance-stiffness-initial 0.20 \
        --paper-shape-stiffness-initial 0.004 \
        --visual-feedback-mode "${visual_mode}" \
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

run_protocol() {
    local protocol="$1"
    local gpu_id="$2"
    run_case pure_pbd "${protocol}" "${gpu_id}"
    run_case visual_residual "${protocol}" "${gpu_id}"
    run_case visual_residual_online_causal "${protocol}" "${gpu_id}"
}

run_protocol reconstruction_7to1 0 \
    > "${OUTPUT_ROOT}/reconstruction_gpu0.log" 2>&1 &
reconstruction_pid=$!
run_protocol future_80to20 1 \
    > "${OUTPUT_ROOT}/future_gpu1.log" 2>&1 &
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

"${ENV_PREFIX}/bin/python" scripts/summarize_super_causal_fixed_lag_validation.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
