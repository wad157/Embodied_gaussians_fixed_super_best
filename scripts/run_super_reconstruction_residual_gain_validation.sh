#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_reconstruction_residual_gain_validation_20260822_v1}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
VALIDATION_PHASE=2

if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing reused output root: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

# Frozen annotations are sampled every ten frames, so not every modulo-eight
# phase necessarily contains GT. Fail before launching any 1440-frame run.
"${ENV_PREFIX}/bin/python" -c \
    'import numpy as np,sys; z=np.load(sys.argv[1],allow_pickle=False); phase=int(sys.argv[2]); frames=np.asarray(z["frame_indices"]); count=int(np.count_nonzero(frames % 8 == phase)); print(f"validation_phase={phase}, scored_gt_frames={count}"); raise SystemExit(0 if count else 2)' \
    "${GROUND_TRUTH}" "${VALIDATION_PHASE}" \
    > "${OUTPUT_ROOT}/phase_preflight.txt"

run_case() {
    local initialization="$1"
    local method="$2"
    local gpu_id="$3"
    local distance_initial="$4"
    local shape_initial="$5"
    local output="${OUTPUT_ROOT}/${initialization}/${method}"
    local visual_mode="residual"
    local gain_profile="multiscale_hold"

    if [[ "${method}" == "pure_pbd" ]]; then
        visual_mode="off"
    elif [[ "${method}" == "residual_conservative" ]]; then
        gain_profile="conservative_multiscale_hold"
    elif [[ "${method}" == "residual_micro" ]]; then
        gain_profile="micro_multiscale_hold"
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_recon_gain_gpu${gpu_id}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${output}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol reconstruction_7to1 \
        --tissue-benchmark-reconstruction-test-phase "${VALIDATION_PHASE}" \
        --tissue-benchmark-track-only \
        --paper-distance-stiffness-initial "${distance_initial}" \
        --paper-shape-stiffness-initial "${shape_initial}" \
        --visual-feedback-mode "${visual_mode}" \
        --visual-residual-iterations 8 \
        --visual-residual-learning-rate-m 2e-5 \
        --visual-residual-maximum-step-m 2e-4 \
        --visual-residual-previous-carry 0 \
        --visual-residual-temporal-weight 0.10 \
        --visual-residual-magnitude-weight 0.01 \
        --visual-residual-gain-profile "${gain_profile}" \
        --no-online-stiffness-update

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${output}" \
        --skip-lpips
}

run_lane_zero() {
    run_case extreme_soft pure_pbd 0 0.10 0.003
    run_case extreme_soft residual_current 0 0.10 0.003
    run_case extreme_soft residual_conservative 0 0.10 0.003
    run_case extreme_soft residual_micro 0 0.10 0.003
    run_case moderate pure_pbd 0 0.20 0.004
    run_case moderate residual_current 0 0.20 0.004
}

run_lane_one() {
    run_case extreme_hard pure_pbd 1 1.60 0.020
    run_case extreme_hard residual_current 1 1.60 0.020
    run_case extreme_hard residual_conservative 1 1.60 0.020
    run_case extreme_hard residual_micro 1 1.60 0.020
    run_case moderate residual_conservative 1 0.20 0.004
    run_case moderate residual_micro 1 0.20 0.004
}

run_lane_zero > "${OUTPUT_ROOT}/gpu0.log" 2>&1 &
gpu0_pid=$!
run_lane_one > "${OUTPUT_ROOT}/gpu1.log" 2>&1 &
gpu1_pid=$!
set +e
wait "${gpu0_pid}"
gpu0_status=$?
wait "${gpu1_pid}"
gpu1_status=$?
set -e
printf 'gpu0_status=%s\ngpu1_status=%s\n' \
    "${gpu0_status}" "${gpu1_status}" > "${OUTPUT_ROOT}/status.txt"
if [[ "${gpu0_status}" -ne 0 || "${gpu1_status}" -ne 0 ]]; then
    exit 3
fi
"${ENV_PREFIX}/bin/python" \
    scripts/summarize_super_reconstruction_residual_gain_validation.py \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/summary.log" 2>&1
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
