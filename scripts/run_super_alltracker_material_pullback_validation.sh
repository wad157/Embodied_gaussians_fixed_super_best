#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_alltracker_material_pullback_full_20260829_v2}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
BINDINGS="${ROOT_DIR}/outputs/grasp5_alltracker_surface_crossanchor_20260829_v3/bindings.npz"
OBSERVATIONS="${ROOT_DIR}/outputs/grasp5_alltracker_crossanchor_flow_depth_observations_20260829_v3/observations.npz"

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
    shift 2
    local result_dir="${OUTPUT_ROOT}/${method}"
    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_alltracker_material_pullback_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol reconstruction_7to1 \
        --tissue-benchmark-reconstruction-test-phase 4 \
        --tissue-benchmark-track-only \
        --visual-feedback-mode trajectory \
        --flow-depth-bindings "${BINDINGS}" \
        --flow-depth-observations "${OBSERVATIONS}" \
        --flow-depth-position-gain 1.0 \
        --flow-depth-velocity-gain 0.75 \
        --flow-depth-absolute-position-weight 1.0 \
        --flow-depth-solver-regularization 0.001 \
        --flow-depth-solver-iterations 16 \
        --flow-depth-robust-residual-mm 20.0 \
        --flow-depth-maximum-position-correction-mm 20.0 \
        --flow-depth-maximum-velocity-correction-m-s 0.25 \
        --no-online-stiffness-update \
        "$@"
}

# A: all material constraints are made another order of magnitude softer.
run_case 0 global_extra_soft \
    --paper-distance-stiffness-initial 0.001 \
    --paper-volume-stiffness-initial 1e3 \
    --paper-shape-stiffness-initial 0.00005 \
    > "${OUTPUT_ROOT}/global_extra_soft.log" 2>&1 &
global_pid=$!

# B: retain the prior stable global baseline and relax only observed support.
run_case 1 local_observed_patch_soft \
    --paper-distance-stiffness-initial 0.01 \
    --paper-volume-stiffness-initial 1e5 \
    --paper-shape-stiffness-initial 0.0005 \
    --flow-depth-local-material-relaxation \
    --flow-depth-local-distance-stiffness 0.001 \
    --flow-depth-local-volume-stiffness 1e3 \
    --flow-depth-local-shape-stiffness 0.00005 \
    > "${OUTPUT_ROOT}/local_observed_patch_soft.log" 2>&1 &
local_pid=$!

set +e
wait "${global_pid}"
global_status=$?
wait "${local_pid}"
local_status=$?
set -e
printf 'global_extra_soft_status=%s\nlocal_observed_patch_soft_status=%s\n' \
    "${global_status}" "${local_status}" > "${OUTPUT_ROOT}/status.txt"
if [[ "${global_status}" -ne 0 || "${local_status}" -ne 0 ]]; then
    exit 3
fi
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
