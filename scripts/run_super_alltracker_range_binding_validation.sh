#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_alltracker_dense_best_validation_20260829_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
BINDINGS="${BINDINGS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_range_20260829_v1/bindings.npz}"
OBSERVATIONS="${OBSERVATIONS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_observations_20260829_v1/observations.npz}"

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
    local gpu_id="$1" protocol="$2"
    local result_dir="${OUTPUT_ROOT}/${protocol}"
    local split_args=(--tissue-benchmark-reconstruction-test-phase 4)
    if [[ "${protocol}" == "future_80to20" ]]; then
        split_args=(--tissue-benchmark-future-test-start-frame 921)
    fi
    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_alltracker_range_gpu${gpu_id}" \
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
        --flow-depth-position-gain 1.0 \
        --flow-depth-velocity-gain 1.0 \
        --flow-depth-absolute-position-weight 1.0 \
        --flow-depth-solver-regularization 0.005 \
        --flow-depth-solver-iterations 16 \
        --flow-depth-robust-residual-mm 5.0 \
        --flow-depth-maximum-position-correction-mm 5.0 \
        --flow-depth-maximum-velocity-correction-m-s 0.10 \
        --no-online-stiffness-update
    "${ENV_PREFIX}/bin/python" scripts/score_super_track_validation.py \
        --ground-truth "${GROUND_TRUTH}" --capture "${result_dir}" \
        > "${result_dir}/score.log" 2>&1
}

run_case 0 reconstruction_7to1 > "${OUTPUT_ROOT}/gpu0.log" 2>&1 &
gpu0_pid=$!
run_case 1 future_80to20 > "${OUTPUT_ROOT}/gpu1.log" 2>&1 &
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
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
