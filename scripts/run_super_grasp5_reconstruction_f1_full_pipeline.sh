#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
ASSET_ROOT="${1:-${ROOT_DIR}/outputs/grasp5_alltracker_full_f1_0_1439_20260906_v2}"
EVALUATION_ROOT="${2:-${ROOT_DIR}/outputs/super_grasp5_reconstruction_f1_future_f2_soft_strong_h3_v1}"
GPU_TRACKING="${GPU_TRACKING:-${GPU_ID:-0}}"
GPU_RECONSTRUCTION="${GPU_RECONSTRUCTION:-0}"
GPU_FUTURE="${GPU_FUTURE:-1}"

TRACK_ROOT="${ASSET_ROOT}/tracks"
BINDING_ROOT="${ASSET_ROOT}/bindings"
OBSERVATION_ROOT="${ASSET_ROOT}/observations"
NATIVE_ROOT="${ROOT_DIR}/data/super/grasp5_native"
OFFLINE_ROOT="${ROOT_DIR}/data/super/grasp5_offline_demo"
DEPTH_ROOT="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/stereo_depth_v1"
TISSUE_ASSET="${NATIVE_ROOT}/tissue_multiview_v1/paper_pbd_tissue_v15_denser_mild_paper_constraints_centroid_ellipsoids/tissue_paper_pbd_centroid_gaussians.npz"
TOOL_MASK="${NATIVE_ROOT}/tissue_multiview_v1/masks_left/tool_dilated/000000-tool-dilated.png"
TABLE_FRAME="${ROOT_DIR}/data/super/table_frame.json"
CALIBRATION="${NATIVE_ROOT}/calib_rectified.json"
VALIDATOR="${ROOT_DIR}/scripts/validate_super_grasp5_reconstruction_f1_full.py"
RUNNER="${ROOT_DIR}/scripts/run_super_alltracker_rgb_observable_adam_full_metrics.sh"
FUTURE_BINDINGS="${FUTURE_BINDINGS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_d1_f2_20260829_v1/bindings.npz}"
FUTURE_OBSERVATIONS="${FUTURE_OBSERVATIONS:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_observations_20260829_v1/observations.npz}"
FUTURE_F2_VALIDATOR="${ROOT_DIR}/scripts/validate_super_grasp5_future_f2.py"

for required_path in \
    "${ENV_PREFIX}/bin/python" \
    "${OFFLINE_ROOT}/videos/stereo_left.mp4" \
    "${DEPTH_ROOT}/000000-depth.npy" \
    "${DEPTH_ROOT}/001439-depth.npy" \
    "${DEPTH_ROOT}/001439-confidence.npz" \
    "${NATIVE_ROOT}/masks/000000-tissue.png" \
    "${NATIVE_ROOT}/visual_force_masks_v1/tissue_masks_packbits.npy" \
    "${TISSUE_ASSET}" \
    "${TOOL_MASK}" \
    "${TABLE_FRAME}" \
    "${CALIBRATION}" \
    "${FUTURE_BINDINGS}" \
    "${FUTURE_OBSERVATIONS}" \
    "${ROOT_DIR}/third_party/AllTracker/checkpoints/alltracker.pth"; do
    if [[ ! -e "${required_path}" ]]; then
        printf 'Missing required input: %s\n' "${required_path}" >&2
        exit 2
    fi
done
if [[ -e "${ASSET_ROOT}" || -e "${EVALUATION_ROOT}" ]]; then
    printf 'Refusing reused asset or evaluation root.\n' >&2
    exit 2
fi

mkdir -p "${ASSET_ROOT}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples:${ROOT_DIR}/scripts"

CUDA_VISIBLE_DEVICES="${GPU_TRACKING}" "${ENV_PREFIX}/bin/python" \
    scripts/track_super_alltracker_surface_particles.py \
    --device cuda:0 \
    --video "${OFFLINE_ROOT}/videos/stereo_left.mp4" \
    --asset "${TISSUE_ASSET}" \
    --depth "${DEPTH_ROOT}/000000-depth.npy" \
    --tissue-mask "${NATIVE_ROOT}/masks/000000-tissue.png" \
    --tool-mask "${TOOL_MASK}" \
    --dynamic-tissue-masks "${NATIVE_ROOT}/visual_force_masks_v1/tissue_masks_packbits.npy" \
    --calibration "${CALIBRATION}" \
    --table-frame "${TABLE_FRAME}" \
    --output-dir "${TRACK_ROOT}" \
    --frame-stride 1 \
    --maximum-source-frame 1439 \
    --primary-segment-length 900 \
    --reanchor-interval 64 \
    --query-mode tissue_grid \
    --query-spacing-px 20 \
    --maximum-queries-per-particle 3 \
    > "${ASSET_ROOT}/01_alltracker_f1_full.log" 2>&1

"${ENV_PREFIX}/bin/python" scripts/build_super_cotracker_range_bindings.py \
    --tracks "${TRACK_ROOT}/tracks.npz" \
    --depth "${DEPTH_ROOT}/000000-depth.npy" \
    --image "${NATIVE_ROOT}/rgb/000000-left.png" \
    --tool-mask "${TOOL_MASK}" \
    --asset "${TISSUE_ASSET}" \
    --calibration "${CALIBRATION}" \
    --table-frame "${TABLE_FRAME}" \
    --output-dir "${BINDING_ROOT}" \
    --binding-mode triangle \
    --triangle-primary-distance-mm 1.0 \
    --triangle-fallback-distance-mm 2.0 \
    --triangle-minimum-movable-vertices 3 \
    --maximum-depth-sampling-radius-px 2 \
    > "${ASSET_ROOT}/02_bindings.log" 2>&1

"${ENV_PREFIX}/bin/python" \
    scripts/build_super_cotracker_causal_sparse_depth_observations.py \
    --tracks "${TRACK_ROOT}/tracks.npz" \
    --bindings "${BINDING_ROOT}/bindings.npz" \
    --depth-dir "${DEPTH_ROOT}" \
    --calibration "${CALIBRATION}" \
    --table-frame "${TABLE_FRAME}" \
    --output-dir "${OBSERVATION_ROOT}" \
    --training-end-frame 1439 \
    --depth-hold-decay-frames 10.0 \
    --maximum-depth-hold-frames 12 \
    --maximum-depth-change-mm 15.0 \
    --maximum-observed-flow-mm 20.0 \
    --minimum-tracking-confidence 0.05 \
    > "${ASSET_ROOT}/03_observations.log" 2>&1

"${ENV_PREFIX}/bin/python" "${VALIDATOR}" \
    --asset-root "${ASSET_ROOT}" \
    --output "${ASSET_ROOT}/asset_validation.json" \
    > "${ASSET_ROOT}/04_asset_validation.log" 2>&1
printf 'complete\n' > "${ASSET_ROOT}/COMPLETE"

env \
    RECONSTRUCTION_BINDINGS="${BINDING_ROOT}/bindings.npz" \
    RECONSTRUCTION_OBSERVATIONS="${OBSERVATION_ROOT}/observations.npz" \
    FUTURE_BINDINGS="${FUTURE_BINDINGS}" \
    FUTURE_OBSERVATIONS="${FUTURE_OBSERVATIONS}" \
    RECONSTRUCTION_OBSERVATION_SCHEDULE=stride1_full_0_1439 \
    FUTURE_OBSERVATION_SCHEDULE=stride2_train_0_1150_future_open_loop_1152_1439 \
    PAPER_DISTANCE_STIFFNESS_INITIAL=0.01 \
    PAPER_VOLUME_STIFFNESS_INITIAL=100000 \
    PAPER_SHAPE_STIFFNESS_INITIAL=0.0005 \
    PAPER_STIFFNESS_MAXIMUM_LOG_STEP=0.08 \
    PAPER_STIFFNESS_RECONSTRUCTION_GLOBAL_LOG_OFFSET=0.60 \
    PAPER_STIFFNESS_FUTURE_GLOBAL_LOG_OFFSET=0.80 \
    PAPER_STIFFNESS_THREE_OF_FOUR_H3=0 \
    PAPER_STIFFNESS_LOCAL_DISTANCE=0 \
    PAPER_STIFFNESS_LOCAL_SHAPE=0 \
    RERUN_PURE_PBD=1 \
    RERUN_RECONSTRUCTION_TRAJECTORY=1 \
    RERUN_FUTURE_TRAJECTORY=1 \
    RUN_RECONSTRUCTION=1 \
    RUN_FUTURE=1 \
    GPU_RECONSTRUCTION="${GPU_RECONSTRUCTION}" \
    GPU_FUTURE="${GPU_FUTURE}" \
    bash "${RUNNER}" "${EVALUATION_ROOT}" "" parallel 0

"${ENV_PREFIX}/bin/python" "${VALIDATOR}" \
    --asset-root "${ASSET_ROOT}" \
    --evaluation-root "${EVALUATION_ROOT}" \
    --output "${EVALUATION_ROOT}/full_reconstruction_validation.json" \
    > "${EVALUATION_ROOT}/full_reconstruction_validation.log" 2>&1
"${ENV_PREFIX}/bin/python" "${FUTURE_F2_VALIDATOR}" \
    --bindings "${FUTURE_BINDINGS}" \
    --observations "${FUTURE_OBSERVATIONS}" \
    --evaluation-root "${EVALUATION_ROOT}" \
    --output "${EVALUATION_ROOT}/future_f2_validation.json" \
    > "${EVALUATION_ROOT}/future_f2_validation.log" 2>&1
printf 'complete\n' > "${EVALUATION_ROOT}/RECONSTRUCTION_F1_FUTURE_F2_VALIDATED"
