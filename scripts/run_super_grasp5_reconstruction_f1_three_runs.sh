#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
CAMPAIGN_ROOT="${1:-${ROOT_DIR}/outputs/super_grasp5_reconstruction_full_f1_legacy_plus_two_20260910_v2}"
ASSET_ROOT="${RECONSTRUCTION_ASSET_ROOT:-${ROOT_DIR}/outputs/grasp5_alltracker_full_f1_0_1439_20260906_v2}"
LEGACY_RUN="${LEGACY_RUN:-${ROOT_DIR}/outputs/super_grasp5_reconstruction_future_full_f1_soft_strong_h3_20260906_v2}"
RUNNER="${ROOT_DIR}/scripts/run_super_alltracker_rgb_observable_adam_full_metrics.sh"
VALIDATOR="${ROOT_DIR}/scripts/validate_super_grasp5_reconstruction_f1_full.py"
SUMMARIZER="${ROOT_DIR}/scripts/summarize_super_reconstruction_f1_three_runs.py"

if [[ -e "${CAMPAIGN_ROOT}" ]]; then
    printf 'Refusing reused campaign root: %s\n' "${CAMPAIGN_ROOT}" >&2
    exit 2
fi
for required_path in \
    "${ENV_PREFIX}/bin/python" \
    "${ASSET_ROOT}/COMPLETE" \
    "${ASSET_ROOT}/asset_validation.json" \
    "${ASSET_ROOT}/bindings/bindings.npz" \
    "${ASSET_ROOT}/observations/observations.npz" \
    "${LEGACY_RUN}/COMPLETE" \
    "${LEGACY_RUN}/full_reconstruction_validation.json" \
    "${RUNNER}" "${VALIDATOR}" "${SUMMARIZER}"; do
    if [[ ! -e "${required_path}" ]]; then
        printf 'Missing required input: %s\n' "${required_path}" >&2
        exit 2
    fi
done

mkdir -p "${CAMPAIGN_ROOT}"
cd "${ROOT_DIR}"
printf '%s\n' \
    'schema=super_grasp5_reconstruction_full_f1_legacy_plus_two_v1' \
    "asset_root=${ASSET_ROOT}" \
    "legacy_run=${LEGACY_RUN}" \
    'protocol=reconstruction_7to1' \
    'observation_schedule=stride1_full_0_1439' \
    'new_run_count=2' \
    'aggregate_run_count=3' \
    'method_schedule=pure_then_parallel_trajectory_and_h3' \
    > "${CAMPAIGN_ROOT}/CAMPAIGN_CONFIGURATION.txt"

run_one() {
    local run_index="$1"
    local run_root="${CAMPAIGN_ROOT}/run${run_index}"
    printf 'Starting new full-f1 Reconstruction repetition %s/2\n' "${run_index}"
    env \
        RECONSTRUCTION_BINDINGS="${ASSET_ROOT}/bindings/bindings.npz" \
        RECONSTRUCTION_OBSERVATIONS="${ASSET_ROOT}/observations/observations.npz" \
        RECONSTRUCTION_OBSERVATION_SCHEDULE=stride1_full_0_1439 \
        FUTURE_OBSERVATION_SCHEDULE=stride2_train_0_1150_future_open_loop_1152_1439 \
        PAPER_DISTANCE_STIFFNESS_INITIAL=0.01 \
        PAPER_VOLUME_STIFFNESS_INITIAL=100000 \
        PAPER_SHAPE_STIFFNESS_INITIAL=0.0005 \
        FLOW_DEPTH_CONFIDENCE_AUTHORITY_POWER=0.0 \
        FLOW_DEPTH_VELOCITY_GAIN=1.0 \
        TRAJECTORY_RGB_RESIDUAL_ENABLED=0 \
        TRAJECTORY_GAUSSIAN_APPEARANCE_ITERATIONS=0 \
        PAPER_STIFFNESS_MAXIMUM_LOG_STEP=0.08 \
        PAPER_STIFFNESS_RECONSTRUCTION_GLOBAL_LOG_OFFSET=0.60 \
        PAPER_STIFFNESS_FUTURE_GLOBAL_LOG_OFFSET=0.80 \
        PAPER_STIFFNESS_THREE_OF_FOUR_H3=0 \
        PAPER_STIFFNESS_LOCAL_DISTANCE=0 \
        PAPER_STIFFNESS_LOCAL_SHAPE=0 \
        RERUN_PURE_PBD=1 \
        RERUN_RECONSTRUCTION_TRAJECTORY=1 \
        RERUN_FUTURE_TRAJECTORY=0 \
        RUN_RECONSTRUCTION=1 \
        RUN_FUTURE=0 \
        RUN_TRAJECTORY_METHOD=1 \
        RUN_ONLINE_STIFFNESS_METHOD=1 \
        PARALLEL_BC_METHODS=1 \
        GPU_RECONSTRUCTION=0 \
        GPU_ONLINE_STIFFNESS=1 \
        bash "${RUNNER}" "${run_root}" "" parallel 0

    "${ENV_PREFIX}/bin/python" "${VALIDATOR}" \
        --asset-root "${ASSET_ROOT}" \
        --evaluation-root "${run_root}" \
        --output "${run_root}/full_reconstruction_validation.json"
    printf 'validated\n' > "${run_root}/FULL_F1_VALIDATED"
}

for run_index in 1 2; do
    run_one "${run_index}" \
        > "${CAMPAIGN_ROOT}/run${run_index}.log" 2>&1
done

"${ENV_PREFIX}/bin/python" "${SUMMARIZER}" \
    --run "legacy=${LEGACY_RUN}" \
    --run "new1=${CAMPAIGN_ROOT}/run1" \
    --run "new2=${CAMPAIGN_ROOT}/run2" \
    --allow-mixed-code-and-config \
    --output-json "${CAMPAIGN_ROOT}/average_results.json" \
    --output-markdown "${CAMPAIGN_ROOT}/average_results.md"
printf 'complete\n' > "${CAMPAIGN_ROOT}/COMPLETE"
printf 'Three-run full-f1 Reconstruction evaluation complete: %s\n' \
    "${CAMPAIGN_ROOT}"
