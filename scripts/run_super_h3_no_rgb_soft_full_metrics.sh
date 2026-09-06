#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_h3_no_post_rgb_soft_strong_h3_20260904_v1}"
RUNNER="${ROOT_DIR}/scripts/run_super_alltracker_rgb_observable_adam_full_metrics.sh"

# "Soft" is the original intermediate setting between the explicitly named
# extreme-soft and moderate endpoints used by the surrounding sweep.
env \
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
    GPU_RECONSTRUCTION=0 \
    GPU_FUTURE=1 \
    bash "${RUNNER}" "${OUTPUT_ROOT}" "" parallel 0
