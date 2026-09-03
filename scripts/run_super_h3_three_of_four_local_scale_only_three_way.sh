#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_h3_three_of_four_local_scale_only_20260903_v1}"
BASELINE="${BASELINE:-${ROOT_DIR}/outputs/super_old_h3_reproduction_scale_only_20260903_v1}"

# Freeze identical Pure-PBD and trajectory+RGB captures. Recompute only the
# third group with a dropout-tolerant H3 and a small zero-mean local distance
# field after each successful global H3 transaction.
export BASELINE_ROOT="${BASELINE}"
export RERUN_RECONSTRUCTION_TRAJECTORY=0
export RERUN_FUTURE_TRAJECTORY=0
export TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS=0
export TRAJECTORY_RGB_POSITION_GAIN=1.0
export TRAJECTORY_RGB_VELOCITY_GAIN=0.15
export PAPER_STIFFNESS_THREE_OF_FOUR_H3=1
export PAPER_STIFFNESS_LOCAL_DISTANCE=1
export PAPER_STIFFNESS_LOCAL_DISTANCE_LOG_STEP=0.008
export PAPER_STIFFNESS_LOCAL_DISTANCE_LOG_OFFSET=0.04
export PAPER_STIFFNESS_LOCAL_DISTANCE_MINIMUM_IMPROVEMENT=0.000001

exec bash "${ROOT_DIR}/scripts/run_super_alltracker_rgb_observable_adam_full_metrics.sh" \
    "${OUTPUT_ROOT}" "${BASELINE}" parallel
