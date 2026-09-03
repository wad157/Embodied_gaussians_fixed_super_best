#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_dynamic_covariance_qd_three_way_20260824_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
GPU_RECONSTRUCTION="${GPU_RECONSTRUCTION:-0}"
GPU_FUTURE="${GPU_FUTURE:-1}"

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

sha256sum \
    examples/example_embodied_super_offline.py \
    src/embodied_gaussians/embodied_simulator/builder.py \
    src/embodied_gaussians/embodied_simulator/simulator.py \
    src/embodied_gaussians/embodied_simulator/warp.py \
    src/embodied_gaussians/physics_simulator/online_tissue_stiffness.py \
    src/embodied_gaussians/physics_simulator/visual_tissue_residual_mapping.py \
    scripts/run_super_dynamic_covariance_three_way_evaluation.sh \
    > "${OUTPUT_ROOT}/CODE_SHA256.txt"

"${ENV_PREFIX}/bin/python" scripts/audit_super_evaluation_protocol.py \
    --report "${OUTPUT_ROOT}/PROTOCOL_AUDIT.json" \
    > "${OUTPUT_ROOT}/protocol_audit.log" 2>&1

run_case() {
    local gpu_id="$1"
    local protocol="$2"
    local method="$3"
    local result_dir="${OUTPUT_ROOT}/${method}/${protocol}"
    local visual_mode="off"
    local online_flag="--no-online-stiffness-update"
    local diagnostics_args=()
    local method_args=()

    if [[ "${method}" == "pbd_visual_residual" ]]; then
        visual_mode="residual"
        diagnostics_args+=(
            --stiffness-evaluation-output "${result_dir}/diagnostics"
        )
        method_args+=(
            --visual-residual-gain-profile cross_frame_ranked_hold
        )
    elif [[ "${method}" == "pbd_visual_residual_online_stiffness" ]]; then
        visual_mode="residual"
        online_flag="--online-stiffness-update"
        diagnostics_args+=(
            --stiffness-evaluation-output "${result_dir}/diagnostics"
            --stiffness-evaluation-horizons 1,3,5,10
        )
        method_args+=(
            --visual-residual-gain-profile cross_frame_ranked_hold
            --stiffness-log-learning-rate 0.12
            --stiffness-maximum-log-step 0.12
            --stiffness-strain-signal-weight 0.80
            --stiffness-distance-minimum 0.025
            --stiffness-distance-maximum 4.0
            --stiffness-shape-minimum 0.001
            --stiffness-shape-maximum 0.040
            --stiffness-candidate-profile direct_residual_gradient
            --stiffness-admission-mode causal_fixed_lag
            --stiffness-admission-horizons 1,3,5,10
        )
    elif [[ "${method}" != "pure_pbd" ]]; then
        printf 'Unknown evaluation method: %s\n' "${method}" >&2
        return 2
    fi

    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_dynamic_qd_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol "${protocol}" \
        --tissue-benchmark-render-scale 0.5 \
        --tissue-benchmark-reconstruction-test-phase 0 \
        --paper-distance-stiffness-initial 0.20 \
        --paper-shape-stiffness-initial 0.004 \
        --visual-feedback-mode "${visual_mode}" \
        --visual-feedback-update-interval 3 \
        --visual-residual-iterations 8 \
        --visual-residual-learning-rate-m 2e-5 \
        --visual-residual-maximum-step-m 2e-4 \
        --visual-residual-previous-carry 0 \
        --visual-residual-temporal-weight 0.10 \
        --visual-residual-magnitude-weight 0.01 \
        "${online_flag}" \
        "${diagnostics_args[@]}" \
        "${method_args[@]}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --lpips-device cuda:0
}

run_protocol_lane() {
    local gpu_id="$1"
    local protocol="$2"
    for method in \
        pure_pbd \
        pbd_visual_residual \
        pbd_visual_residual_online_stiffness
    do
        run_case "${gpu_id}" "${protocol}" "${method}"
    done
}

run_protocol_lane "${GPU_RECONSTRUCTION}" reconstruction_7to1 \
    > "${OUTPUT_ROOT}/reconstruction_gpu${GPU_RECONSTRUCTION}.log" 2>&1 &
reconstruction_pid=$!
run_protocol_lane "${GPU_FUTURE}" future_80to20 \
    > "${OUTPUT_ROOT}/future_gpu${GPU_FUTURE}.log" 2>&1 &
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

"${ENV_PREFIX}/bin/python" scripts/summarize_super_tissue_method_comparison.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/method_comparison.log" 2>&1
"${ENV_PREFIX}/bin/python" scripts/summarize_super_dynamic_covariance_evaluation.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/detailed_summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
printf 'Formal dynamic-covariance q/qd three-way evaluation complete: %s\n' \
    "${OUTPUT_ROOT}"
