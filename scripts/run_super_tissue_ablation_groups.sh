#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
EVAL_ROOT="${1:-${ROOT_DIR}/outputs/super_tissue_evaluation_formal_20260821}"
GROUND_TRUTH="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
GPU_PURE_PBD="${GPU_PURE_PBD:-0}"
GPU_RESIDUAL="${GPU_RESIDUAL:-1}"
RUN_PURE_PBD="${RUN_PURE_PBD:-1}"
RUN_RESIDUAL="${RUN_RESIDUAL:-1}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    printf 'Missing Python environment: %s\n' "${ENV_PREFIX}" >&2
    exit 2
fi
if [[ ! -f "${GROUND_TRUTH}" ]]; then
    printf 'Missing formal ground truth: %s\n' "${GROUND_TRUTH}" >&2
    exit 2
fi
for run_flag in "${RUN_PURE_PBD}" "${RUN_RESIDUAL}"; do
    if [[ "${run_flag}" != 0 && "${run_flag}" != 1 ]]; then
        printf 'RUN_PURE_PBD and RUN_RESIDUAL must be 0 or 1.\n' >&2
        exit 2
    fi
done
validate_method_target() {
    local method="$1"
    local run_flag="$2"
    local target="${EVAL_ROOT}/${method}"
    if [[ "${run_flag}" == 1 ]]; then
        if [[ -e "${target}" ]]; then
            printf 'Refusing reused ablation directory: %s\n' "${target}" >&2
            exit 2
        fi
        return
    fi
    for protocol in reconstruction_7to1 future_80to20; do
        if [[ ! -f "${target}/${protocol}/evaluation_results.json" ]]; then
            printf 'Cannot skip incomplete method: %s/%s\n' "${target}" "${protocol}" >&2
            exit 2
        fi
    done
}
validate_method_target pure_pbd "${RUN_PURE_PBD}"
validate_method_target pbd_visual_residual "${RUN_RESIDUAL}"
for protocol in reconstruction_7to1 future_80to20; do
    if [[ ! -f "${EVAL_ROOT}/${protocol}/evaluation_results.json" ]]; then
        printf 'Missing online-stiffness formal result: %s/%s\n' "${EVAL_ROOT}" "${protocol}" >&2
        exit 2
    fi
done

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

run_protocol() {
    local method="$1"
    local protocol="$2"
    local visual_mode="$3"
    local gpu_id="$4"
    local extension_cache="$5"
    local result_dir="${EVAL_ROOT}/${method}/${protocol}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" TORCH_EXTENSIONS_DIR="${extension_cache}" \
        "${ENV_PREFIX}/bin/python" examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol "${protocol}" \
        --tissue-benchmark-render-scale 0.5 \
        --visual-feedback-mode "${visual_mode}" \
        --no-online-stiffness-update

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${ENV_PREFIX}/bin/python" \
        scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --lpips-device cuda:0
}

run_group() {
    local method="$1"
    local visual_mode="$2"
    local gpu_id="$3"
    local extension_cache="$4"
    run_protocol "${method}" reconstruction_7to1 "${visual_mode}" "${gpu_id}" "${extension_cache}"
    run_protocol "${method}" future_80to20 "${visual_mode}" "${gpu_id}" "${extension_cache}"
}

pure_pid=""
residual_pid=""
if [[ "${RUN_PURE_PBD}" == 1 ]]; then
    run_group pure_pbd off "${GPU_PURE_PBD}" \
        /Media_HDD/jwshan/tmp/torch_extensions_eval_gpu0 \
        > "${EVAL_ROOT}/pure_pbd_ablation.log" 2>&1 &
    pure_pid=$!
fi
if [[ "${RUN_RESIDUAL}" == 1 ]]; then
    run_group pbd_visual_residual residual "${GPU_RESIDUAL}" \
        /Media_HDD/jwshan/tmp/torch_extensions_eval_gpu1 \
        > "${EVAL_ROOT}/pbd_visual_residual_ablation.log" 2>&1 &
    residual_pid=$!
fi

set +e
pure_status=0
residual_status=0
if [[ -n "${pure_pid}" ]]; then
    wait "${pure_pid}"
    pure_status=$?
fi
if [[ -n "${residual_pid}" ]]; then
    wait "${residual_pid}"
    residual_status=$?
fi
set -e
printf 'pure_pbd_status=%s\npbd_visual_residual_status=%s\n' \
    "${pure_status}" "${residual_status}" \
    > "${EVAL_ROOT}/ablation_status.txt"
if [[ "${pure_status}" -ne 0 || "${residual_status}" -ne 0 ]]; then
    exit 3
fi

"${ENV_PREFIX}/bin/python" scripts/summarize_super_tissue_method_comparison.py \
    --root "${EVAL_ROOT}" \
    > "${EVAL_ROOT}/method_comparison.log" 2>&1
printf 'complete\n' > "${EVAL_ROOT}/ABLATION_COMPLETE"
printf 'SUPER tissue ablation complete: %s\n' "${EVAL_ROOT}"
