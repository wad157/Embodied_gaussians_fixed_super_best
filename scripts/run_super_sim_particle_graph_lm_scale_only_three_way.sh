#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_sim_particle_graph_lm_scale_only_three_way_20260902_v1}"

# Freeze the completed no-appearance-learning first and second groups. Their
# mode-2 tissue Gaussians already transport covariance/scales from the bound
# visual triangle deformation. Only the GraphLM third group is recomputed.
export BASELINE_ROOT="${BASELINE_ROOT:-${ROOT_DIR}/outputs/super_alltracker_rgb_recon_nonoverlap_h3_full_metrics_20260831_v10}"
export TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS=0
# The final post-projection trust-region guard now makes this a real 0.03
# maximum, not the larger 0.062--0.108 effective steps seen before the fix.
export STIFFNESS_MAXIMUM_LOG_STEP="${STIFFNESS_MAXIMUM_LOG_STEP:-0.03}"

exec bash "${ROOT_DIR}/scripts/run_super_sim_particle_graph_lm_third_group.sh" \
    "${OUTPUT_ROOT}"
