#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHYSTWIN_PY="${ROOT_DIR}/scripts/run_phystwin_super_python.sh"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
GPU_ID="${PHYSTWIN_GPU_ID:-0}"

if [[ $# -ne 2 ]]; then
    echo "Usage: bash scripts/resume_phystwin_super_after_physics.sh <grasp5|grasp3|grasp1> <OUTPUT_ROOT>" >&2
    exit 2
fi
DATASET_KEY="$1"
OUTPUT_ROOT="$(realpath -m "$2")"
case "${DATASET_KEY}" in
    grasp5)
        NATIVE="${ROOT_DIR}/data/super/grasp5_native"
        GT="${ROOT_DIR}/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
        ;;
    grasp3)
        NATIVE="${ROOT_DIR}/data/super/grasp3_native"
        GT="${NATIVE}/evaluation_v1/manual_tissue_tracks_10_v2/ground_truth_2d3d_v1.npz"
        ;;
    grasp1)
        NATIVE="${ROOT_DIR}/data/super/grasp1_native"
        GT="${NATIVE}/evaluation_v2/manual_tissue_tracks_10_no_exclusion/ground_truth_2d3d_v1.npz"
        ;;
    *) echo "Unknown dataset: ${DATASET_KEY}" >&2; exit 2 ;;
esac

PREPROCESS="${OUTPUT_ROOT}/preprocess"
APPEARANCE="${OUTPUT_ROOT}/appearance"
PHYSICS="${OUTPUT_ROOT}/physics"
CAPTURE="${OUTPUT_ROOT}/capture"
RENDERS="${OUTPUT_ROOT}/render_predictions"
LOG="${OUTPUT_ROOT}/run.log"
for required in \
    "${PREPROCESS}/metadata.json" \
    "${APPEARANCE}/gaussians.npz" \
    "${APPEARANCE}/metadata.json" \
    "${PHYSICS}/raw_particle_rollout.npz" \
    "${PHYSICS}/metadata.json" \
    "${GT}"; do
    [[ -f "${required}" ]] || { echo "Missing completed artifact: ${required}" >&2; exit 1; }
done
"${EVAL_PY}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["full_protocol"] is True' \
    "${PHYSICS}/metadata.json"
if [[ -e "${CAPTURE}" || -e "${RENDERS}" ]]; then
    echo "Refusing to overwrite existing export: ${OUTPUT_ROOT}" >&2
    exit 1
fi
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "status=failed\nexit_code=%s\n" "$code" >"${OUTPUT_ROOT}/status.txt"; fi' EXIT

printf 'status=exporting\ndataset=%s\ngpu=%s\nprotocol=joint_reconstruction_7to1_future_80to20\n' \
    "${DATASET_KEY}" "${GPU_ID}" >"${OUTPUT_ROOT}/status.txt"
CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${PHYSTWIN_PY}" \
    "${ROOT_DIR}/baselines/phystwin_super/export_super.py" \
    --dataset-key "${DATASET_KEY}" --dataset "${NATIVE}" \
    --physics-dir "${PHYSICS}" --appearance-dir "${APPEARANCE}" \
    --capture "${CAPTURE}" --render-dir "${RENDERS}" --device cuda:0 \
    >>"${LOG}" 2>&1

printf 'status=preparing_render_metrics\ndataset=%s\ngpu=%s\nprotocol=joint_reconstruction_7to1_future_80to20\n' \
    "${DATASET_KEY}" "${GPU_ID}" >"${OUTPUT_ROOT}/status.txt"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${ROOT_DIR}/scripts" "${EVAL_PY}" \
    "${ROOT_DIR}/scripts/prepare_endogaussian_super_render_metrics.py" \
    --dataset-key "${DATASET_KEY}" --prediction-dir "${RENDERS}" \
    --capture "${CAPTURE}" --render-scale 0.5 --delete-prediction-arrays >>"${LOG}" 2>&1

printf 'status=scoring\ndataset=%s\ngpu=%s\nprotocol=joint_reconstruction_7to1_future_80to20\n' \
    "${DATASET_KEY}" "${GPU_ID}" >"${OUTPUT_ROOT}/status.txt"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${ROOT_DIR}/scripts" "${EVAL_PY}" \
    "${ROOT_DIR}/scripts/score_super_joint_80to20_evaluation.py" \
    --ground-truth "${GT}" --capture "${CAPTURE}" --lpips-device cuda:0 >>"${LOG}" 2>&1

printf 'status=complete\ndataset=%s\ngpu=%s\nprotocol=joint_reconstruction_7to1_future_80to20\n' \
    "${DATASET_KEY}" "${GPU_ID}" >"${OUTPUT_ROOT}/status.txt"
find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"${OUTPUT_ROOT}/SHA256SUMS"
echo "Complete: ${OUTPUT_ROOT}"
