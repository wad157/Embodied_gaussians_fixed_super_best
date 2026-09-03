#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_paper_trajectory_adam_three_way_full_metrics_20260830_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
BINDINGS="${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_d1_f2_20260829_v1/bindings.npz"
OBSERVATIONS="${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_observations_20260829_v1/observations.npz"
GPU_RECONSTRUCTION="${GPU_RECONSTRUCTION:-0}"
GPU_FUTURE="${GPU_FUTURE:-1}"

for path in "${ENV_PREFIX}/bin/python" "${GROUND_TRUTH}" "${BINDINGS}" "${OBSERVATIONS}"; do
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
    src/embodied_gaussians/physics_simulator/paper_trajectory_stiffness.py \
    scripts/score_super_tissue_evaluation.py \
    scripts/summarize_super_tissue_method_comparison.py \
    scripts/run_super_paper_trajectory_adam_three_way_full_metrics.sh \
    "${BINDINGS}" "${OBSERVATIONS}" \
    > "${OUTPUT_ROOT}/CODE_AND_INPUT_SHA256.txt"

"${ENV_PREFIX}/bin/python" scripts/audit_super_evaluation_protocol.py \
    --report "${OUTPUT_ROOT}/PROTOCOL_AUDIT.json" \
    > "${OUTPUT_ROOT}/protocol_audit.log" 2>&1

run_case() {
    local gpu_id="$1" protocol="$2" method="$3"
    local result_dir="${OUTPUT_ROOT}/${method}/${protocol}"
    local visual_mode="off"
    local online_flag="--no-online-stiffness-update"
    local method_args=()

    case "${method}" in
        pure_pbd)
            ;;
        pbd_alltracker_depth)
            visual_mode="trajectory"
            method_args+=(
                --stiffness-evaluation-output "${result_dir}/diagnostics"
            )
            ;;
        pbd_alltracker_depth_online_stiffness)
            visual_mode="trajectory"
            online_flag="--online-stiffness-update"
            method_args+=(
                --stiffness-evaluation-output "${result_dir}/diagnostics"
                --stiffness-admission-mode paper_trajectory_adam
                --stiffness-candidate-profile direct_residual_gradient
                --stiffness-distance-minimum 0.00001
                --stiffness-distance-maximum 4.0
                --stiffness-shape-minimum 0.000001
                --stiffness-shape-maximum 0.04
                --paper-stiffness-adam-learning-rate 0.10
                --paper-stiffness-perturbation 0.05
                --paper-stiffness-track-robust-scale-mm 3.0
                --paper-stiffness-history-scale-mm 1.0
                --paper-stiffness-history-weight 0.10
                --paper-stiffness-distance-smooth-weight 0.001
                --paper-stiffness-shape-smooth-weight 0.001
                --paper-stiffness-minimum-axis-loss-difference 1e-7
            )
            ;;
        *)
            printf 'Unknown method: %s\n' "${method}" >&2
            return 2
            ;;
    esac

    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_alltracker_full_metrics_gpu${gpu_id}" \
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
        --visual-feedback-mode "${visual_mode}" \
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
        "${online_flag}" \
        "${method_args[@]}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --lpips-device cuda:0
}

run_protocol_lane() {
    local gpu_id="$1" protocol="$2"
    for method in \
        pure_pbd \
        pbd_alltracker_depth \
        pbd_alltracker_depth_online_stiffness
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
wait "${reconstruction_pid}"; reconstruction_status=$?
wait "${future_pid}"; future_status=$?
set -e
printf 'reconstruction_status=%s\nfuture_status=%s\n' \
    "${reconstruction_status}" "${future_status}" \
    > "${OUTPUT_ROOT}/status.txt"
if [[ "${reconstruction_status}" -ne 0 || "${future_status}" -ne 0 ]]; then
    exit 3
fi

"${ENV_PREFIX}/bin/python" scripts/summarize_super_tissue_method_comparison.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
printf 'Paper trajectory Adam three-way full evaluation complete: %s\n' \
    "${OUTPUT_ROOT}"
