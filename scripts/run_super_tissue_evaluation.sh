#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/super_tissue_evaluation_v1}"
GT_ROOT="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10"
ANNOTATIONS="${GT_ROOT}/annotations.json"
DEPTH_DIR="${GT_ROOT}/stereo_depth_v1"
GROUND_TRUTH="${GT_ROOT}/ground_truth_2d3d_v1.npz"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    printf 'Missing Python environment: %s\n' "${ENV_PREFIX}" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    printf 'Refusing non-empty/reused evaluation root: %s\n' "${OUTPUT_ROOT}" >&2
    exit 2
fi

cd "${ROOT_DIR}"
"${ENV_PREFIX}/bin/python" scripts/annotate_super_tissue_gt_tracks.py --validate-only

if [[ ! -f "${DEPTH_DIR}/depth_generation_summary.json" ]]; then
    if [[ -d "${DEPTH_DIR}" ]] && find "${DEPTH_DIR}" -mindepth 1 -print -quit | grep -q .; then
        printf 'Incomplete stereo depth directory exists; inspect it before retrying: %s\n' "${DEPTH_DIR}" >&2
        exit 2
    fi
    FRAMES="$("${ENV_PREFIX}/bin/python" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(map(str,d["sampling"]["scheduled_frames"])))' "${ANNOTATIONS}")"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ENV_PREFIX}/bin/python" \
        scripts/generate_super_depth_foundation_timestamped.py \
        --frames "${FRAMES}" \
        --output-dir "${DEPTH_DIR}" \
        --device cuda:0
fi

"${ENV_PREFIX}/bin/python" scripts/prepare_super_tissue_evaluation_gt.py
mkdir -p "${OUTPUT_ROOT}"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/examples"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/Media_HDD/jwshan/tmp/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

run_protocol() {
    local protocol="$1"
    local result_dir="${OUTPUT_ROOT}/${protocol}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ENV_PREFIX}/bin/python" \
        examples/example_embodied_super_offline.py \
        --evaluation-headless \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 1440 \
        --evaluation-physics-steps-per-frame 3 \
        --tissue-benchmark-output "${result_dir}" \
        --tissue-benchmark-ground-truth "${GROUND_TRUTH}" \
        --tissue-benchmark-protocol "${protocol}" \
        --tissue-benchmark-render-scale 0.5 \
        --visual-feedback-mode residual \
        --online-stiffness-update
    "${ENV_PREFIX}/bin/python" scripts/score_super_tissue_evaluation.py \
        --ground-truth "${GROUND_TRUTH}" \
        --capture "${result_dir}" \
        --lpips-device cuda:0
}

run_protocol reconstruction_7to1
run_protocol future_80to20

printf 'SUPER evaluation complete: %s\n' "${OUTPUT_ROOT}"
