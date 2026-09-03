#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_old_h3_reproduction_scale_only_20260903_v1}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${ROOT_DIR}/outputs/super_alltracker_rgb_recon_nonoverlap_h3_full_metrics_20260831_v10}"

# Reproduce the previously best H3 material-identification run without
# touching the current GraphLM implementation. Freeze the exact Pure-PBD and
# trajectory+RGB captures from the reference run and recompute only the old
# paper_trajectory_adam stiffness group on both formal protocols.
export BASELINE_ROOT="${REFERENCE_ROOT}"
export RERUN_RECONSTRUCTION_TRAJECTORY=0
export RERUN_FUTURE_TRAJECTORY=0
export TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS=0
export TRAJECTORY_RGB_POSITION_GAIN=1.0
export TRAJECTORY_RGB_VELOCITY_GAIN=0.15
# This wrapper is the frozen, selected H3 baseline.  Set the experimental
# extensions explicitly instead of inheriting a caller's shell environment.
export PAPER_STIFFNESS_THREE_OF_FOUR_H3=0
export PAPER_STIFFNESS_LOCAL_DISTANCE=0

exec bash "${ROOT_DIR}/scripts/run_super_alltracker_rgb_observable_adam_full_metrics.sh" \
    "${OUTPUT_ROOT}" "${REFERENCE_ROOT}" parallel
