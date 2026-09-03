#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_alltracker_triangle_direct_signal_stiffness_full_metrics_20260829_v1}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${ROOT_DIR}/outputs/super_alltracker_triangle_d1_f2_three_way_full_metrics_20260829_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
BINDINGS="${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_d1_f2_20260829_v1/bindings.npz"
OBSERVATIONS="${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_observations_20260829_v1/observations.npz"
GPU_RECONSTRUCTION="${GPU_RECONSTRUCTION:-0}"
GPU_FUTURE="${GPU_FUTURE:-1}"

for path in \
    "${ENV_PREFIX}/bin/python" \
    "${GROUND_TRUTH}" \
    "${BINDINGS}" \
    "${OBSERVATIONS}" \
    "${REFERENCE_ROOT}/pbd_alltracker_depth/reconstruction_7to1/evaluation_results.json" \
    "${REFERENCE_ROOT}/pbd_alltracker_depth/future_80to20/evaluation_results.json"
do
    if [[ ! -e "${path}" ]]; then
        printf 'Missing required input: %s\n' "${path}" >&2
        exit 2
    fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused output root: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples:${ROOT_DIR}/scripts"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

sha256sum \
    examples/example_embodied_super_offline.py \
    src/embodied_gaussians/physics_simulator/flow_depth_particle_observer.py \
    src/embodied_gaussians/physics_simulator/online_tissue_stiffness.py \
    scripts/score_super_tissue_evaluation.py \
    scripts/run_super_alltracker_triangle_direct_signal_stiffness_full_metrics.sh \
    "${BINDINGS}" "${OBSERVATIONS}" \
    > "${OUTPUT_ROOT}/CODE_AND_INPUT_SHA256.txt"

"${ENV_PREFIX}/bin/python" scripts/audit_super_evaluation_protocol.py \
    --report "${OUTPUT_ROOT}/PROTOCOL_AUDIT.json" \
    > "${OUTPUT_ROOT}/protocol_audit.log" 2>&1

run_case() {
    local gpu_id="$1" protocol="$2"
    local result_dir="${OUTPUT_ROOT}/direct_signal_online_stiffness/${protocol}"
    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_alltracker_triangle_direct_gpu${gpu_id}" \
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
        --paper-distance-stiffness-initial 0.01 \
        --paper-volume-stiffness-initial 100000 \
        --paper-shape-stiffness-initial 0.0005 \
        --visual-feedback-mode trajectory \
        --flow-depth-bindings "${BINDINGS}" \
        --flow-depth-observations "${OBSERVATIONS}" \
        --flow-depth-position-gain 1.0 \
        --flow-depth-velocity-gain 1.0 \
        --flow-depth-absolute-position-weight 1.0 \
        --flow-depth-solver-regularization 0.005 \
        --flow-depth-solver-iterations 16 \
        --flow-depth-robust-residual-mm 5.0 \
        --flow-depth-maximum-position-correction-mm 5.0 \
        --flow-depth-maximum-velocity-correction-m-s 0.10 \
        --online-stiffness-update \
        --stiffness-evaluation-output "${result_dir}/diagnostics" \
        --stiffness-admission-mode direct_online \
        --stiffness-candidate-profile direct_alternating_gradient \
        --stiffness-log-learning-rate 0.30 \
        --stiffness-maximum-log-step 0.10 \
        --stiffness-effective-log-step-target 0.08 \
        --stiffness-maximum-step-amplification 64 \
        --stiffness-strain-signal-weight 0.75 \
        --stiffness-hardening-bias 0.0 \
        --stiffness-minimum-residual-mm 0.001 \
        --stiffness-minimum-deformation-mm 0.005 \
        --stiffness-rejected-ema-keep-ratio 0.25 \
        --stiffness-distance-minimum 0.00001 \
        --stiffness-distance-maximum 4.0 \
        --stiffness-shape-minimum 0.000001 \
        --stiffness-shape-maximum 0.04

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --lpips-device cuda:0
}

run_case "${GPU_RECONSTRUCTION}" reconstruction_7to1 \
    > "${OUTPUT_ROOT}/reconstruction_gpu${GPU_RECONSTRUCTION}.log" 2>&1 &
reconstruction_pid=$!
run_case "${GPU_FUTURE}" future_80to20 \
    > "${OUTPUT_ROOT}/future_gpu${GPU_FUTURE}.log" 2>&1 &
future_pid=$!

set +e
wait "${reconstruction_pid}"; reconstruction_status=$?
wait "${future_pid}"; future_status=$?
set -e
printf 'reconstruction_status=%s\nfuture_status=%s\n' \
    "${reconstruction_status}" "${future_status}" > "${OUTPUT_ROOT}/status.txt"
if [[ "${reconstruction_status}" -ne 0 || "${future_status}" -ne 0 ]]; then
    exit 3
fi
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
