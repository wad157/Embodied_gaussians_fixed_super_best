#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_alltracker_soft_material_frame_end_20260829_v1}"
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
    local visual_mode="off"
    if [[ "${method}" == "soft_pbd_alltracker_frame_end" ]]; then
        visual_mode="trajectory"
    fi
    local result_dir="${OUTPUT_ROOT}/${method}"
    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_alltracker_soft_frame_end_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 800 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol reconstruction_7to1 \
        --tissue-benchmark-reconstruction-test-phase 4 \
        --tissue-benchmark-track-only \
        --paper-distance-stiffness-initial 0.01 \
        --paper-volume-stiffness-initial 1e5 \
        --paper-shape-stiffness-initial 0.0005 \
        --visual-feedback-mode "${visual_mode}" \
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
        --no-online-stiffness-update
}

run_case 0 soft_pure_pbd > "${OUTPUT_ROOT}/soft_pure_pbd.log" 2>&1 &
pure_pid=$!
run_case 1 soft_pbd_alltracker_frame_end \
    > "${OUTPUT_ROOT}/soft_pbd_alltracker_frame_end.log" 2>&1 &
visual_pid=$!

set +e
wait "${pure_pid}"
pure_status=$?
wait "${visual_pid}"
visual_status=$?
set -e
printf 'soft_pure_pbd_status=%s\nsoft_pbd_alltracker_frame_end_status=%s\n' \
    "${pure_status}" "${visual_status}" > "${OUTPUT_ROOT}/status.txt"
if [[ "${pure_status}" -ne 0 || "${visual_status}" -ne 0 ]]; then
    exit 3
fi

"${ENV_PREFIX}/bin/python" scripts/score_super_partial_track_validation.py \
    --ground-truth "${GROUND_TRUTH}" \
    --pure-pbd "${OUTPUT_ROOT}/soft_pure_pbd" \
    --trajectory "${OUTPUT_ROOT}/soft_pbd_alltracker_frame_end" \
    --output "${OUTPUT_ROOT}/PARTIAL_TRACK_COMPARISON.json" \
    > "${OUTPUT_ROOT}/score.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
