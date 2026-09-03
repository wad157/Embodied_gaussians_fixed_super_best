#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
TRACK_ROOT="${1:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_20260829_v1}"
BINDING_ROOT="${2:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_d1_f2_20260829_v1}"
OBSERVATION_ROOT="${3:-${ROOT_DIR}/outputs/grasp5_alltracker_tissue_grid20_q3_triangle_observations_20260829_v1}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples:${ROOT_DIR}/scripts"

if [[ ! -f "${BINDING_ROOT}/bindings.npz" || ! -f "${BINDING_ROOT}/report.json" ]]; then
    "${ENV_PREFIX}/bin/python" scripts/build_super_cotracker_range_bindings.py \
        --tracks "${TRACK_ROOT}/tracks.npz" \
        --output-dir "${BINDING_ROOT}" \
        --binding-mode triangle \
        --triangle-primary-distance-mm 1.0 \
        --triangle-fallback-distance-mm 2.0 \
        --triangle-minimum-movable-vertices 3 \
        --maximum-depth-sampling-radius-px 2
fi

if [[ ! -f "${OBSERVATION_ROOT}/observations.npz" || ! -f "${OBSERVATION_ROOT}/report.json" ]]; then
    "${ENV_PREFIX}/bin/python" \
        scripts/build_super_cotracker_causal_sparse_depth_observations.py \
        --tracks "${TRACK_ROOT}/tracks.npz" \
        --bindings "${BINDING_ROOT}/bindings.npz" \
        --output-dir "${OBSERVATION_ROOT}" \
        --training-end-frame 1151 \
        --depth-hold-decay-frames 10.0 \
        --maximum-depth-hold-frames 12 \
        --maximum-depth-change-mm 15.0 \
        --maximum-observed-flow-mm 20.0 \
        --minimum-tracking-confidence 0.05
fi

printf 'complete\n' > "${OBSERVATION_ROOT}/COMPLETE"
