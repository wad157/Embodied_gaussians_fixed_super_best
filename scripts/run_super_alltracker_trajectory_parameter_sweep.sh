#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_alltracker_trajectory_parameter_sweep_20260829_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
BINDINGS="${ROOT_DIR}/outputs/grasp5_alltracker_surface_crossanchor_20260829_v3/bindings.npz"
OBSERVATIONS="${ROOT_DIR}/outputs/grasp5_alltracker_crossanchor_flow_depth_observations_20260829_v3/observations.npz"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused output root: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples:${ROOT_DIR}/scripts"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

run_case() {
    local gpu_id="$1" name="$2" protocol="$3"
    local position_gain="$4" velocity_gain="$5" absolute_weight="$6"
    local regularization="$7" robust_mm="$8" position_cap_mm="$9"
    local velocity_cap="${10}"
    local result_dir="${OUTPUT_ROOT}/${name}/${protocol}"
    local split_args=(--tissue-benchmark-reconstruction-test-phase 4)
    if [[ "${protocol}" == "future_80to20" ]]; then
        split_args=(--tissue-benchmark-future-test-start-frame 921)
    fi
    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_alltracker_sweep_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1152 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol "${protocol}" \
        --tissue-benchmark-track-only \
        "${split_args[@]}" \
        --paper-distance-stiffness-initial 0.01 \
        --paper-volume-stiffness-initial 100000 \
        --paper-shape-stiffness-initial 0.0005 \
        --visual-feedback-mode trajectory \
        --flow-depth-bindings "${BINDINGS}" \
        --flow-depth-observations "${OBSERVATIONS}" \
        --flow-depth-position-gain "${position_gain}" \
        --flow-depth-velocity-gain "${velocity_gain}" \
        --flow-depth-absolute-position-weight "${absolute_weight}" \
        --flow-depth-solver-regularization "${regularization}" \
        --flow-depth-solver-iterations 16 \
        --flow-depth-robust-residual-mm "${robust_mm}" \
        --flow-depth-maximum-position-correction-mm "${position_cap_mm}" \
        --flow-depth-maximum-velocity-correction-m-s "${velocity_cap}" \
        --no-online-stiffness-update
    "${ENV_PREFIX}/bin/python" scripts/score_super_track_validation.py \
        --ground-truth "${GROUND_TRUTH}" --capture "${result_dir}" \
        > "${result_dir}/score.log" 2>&1
}

run_config() {
    local gpu_id="$1" name="$2" position_gain="$3" velocity_gain="$4"
    local absolute_weight="$5" regularization="$6" robust_mm="$7"
    local position_cap_mm="$8" velocity_cap="$9"
    for protocol in reconstruction_7to1 future_80to20; do
        run_case "${gpu_id}" "${name}" "${protocol}" \
            "${position_gain}" "${velocity_gain}" "${absolute_weight}" \
            "${regularization}" "${robust_mm}" "${position_cap_mm}" \
            "${velocity_cap}"
    done
}

(
    run_config 0 current 1.0 0.75 1.0 0.001 20.0 20.0 0.25
    run_config 0 bounded3 1.0 0.75 1.0 0.001 3.0 3.0 0.10
    run_config 0 confidence 1.0 0.75 1.0 0.020 5.0 5.0 0.10
    run_config 0 velocity_high 1.0 1.0 1.0 0.005 5.0 5.0 0.10
) > "${OUTPUT_ROOT}/gpu0.log" 2>&1 &
gpu0_pid=$!
(
    run_config 1 bounded5 1.0 0.75 1.0 0.001 5.0 5.0 0.10
    run_config 1 velocity_low 1.0 0.30 1.0 0.005 5.0 5.0 0.10
    run_config 1 blend85 1.0 0.50 0.85 0.005 5.0 5.0 0.10
    run_config 1 position075 0.75 0.50 1.0 0.005 5.0 5.0 0.10
) > "${OUTPUT_ROOT}/gpu1.log" 2>&1 &
gpu1_pid=$!

set +e
wait "${gpu0_pid}"; gpu0_status=$?
wait "${gpu1_pid}"; gpu1_status=$?
set -e
printf 'gpu0_status=%s\ngpu1_status=%s\n' \
    "${gpu0_status}" "${gpu1_status}" > "${OUTPUT_ROOT}/status.txt"
if [[ "${gpu0_status}" -ne 0 || "${gpu1_status}" -ne 0 ]]; then
    exit 3
fi
"${ENV_PREFIX}/bin/python" scripts/summarize_super_alltracker_trajectory_sweep.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
