#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHYSTWIN_PY="${ROOT_DIR}/scripts/run_phystwin_super_python.sh"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
GPU_ID="${PHYSTWIN_GPU_ID:-0}"

if [[ $# -ne 4 ]]; then
    echo "Usage: bash scripts/run_phystwin_super_baseline_once.sh <grasp5|grasp3|grasp1> <repeat_01|repeat_02|repeat_03> <seed> <OUTPUT_ROOT>" >&2
    exit 2
fi
DATASET_KEY="$1"
REPEAT_ID="$2"
SEED="$3"
OUTPUT_ROOT="$(realpath -m "$4")"
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
if [[ ! "${SEED}" =~ ^[0-9]+$ ]] || [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "Seed and PHYSTWIN_GPU_ID must be non-negative integers" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2
    exit 1
fi
for required in "${PHYSTWIN_PY}" "${EVAL_PY}" "${GT}"; do
    [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done

PREPROCESS="${OUTPUT_ROOT}/preprocess"
APPEARANCE="${OUTPUT_ROOT}/appearance"
PHYSICS="${OUTPUT_ROOT}/physics"
CAPTURE="${OUTPUT_ROOT}/capture"
RENDERS="${OUTPUT_ROOT}/render_predictions"
LOG="${OUTPUT_ROOT}/run.log"
mkdir -p "${OUTPUT_ROOT}"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "status=failed\nexit_code=%s\n" "$code" >"${OUTPUT_ROOT}/status.txt"; fi' EXIT

write_status() {
    printf '%s\n' "status=$1" "dataset=${DATASET_KEY}" "repeat=${REPEAT_ID}" \
        "seed=${SEED}" "gpu=${GPU_ID}" 'method=PhysTwin_native_particles_KNN_LBS' \
        'protocol=joint_reconstruction_7to1_future_80to20' >"${OUTPUT_ROOT}/status.txt"
}

write_status auditing
"${EVAL_PY}" "${ROOT_DIR}/scripts/audit_phystwin_super_protocol.py" \
    --dataset-key "${DATASET_KEY}" --native "${NATIVE}" \
    --output "${OUTPUT_ROOT}/protocol_audit.json" >"${LOG}" 2>&1

write_status preprocessing
CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${PHYSTWIN_PY}" \
    "${ROOT_DIR}/baselines/phystwin_super/prepare_super.py" \
    --dataset-key "${DATASET_KEY}" --dataset "${NATIVE}" --output-dir "${PREPROCESS}" \
    --seed "${SEED}" --device cuda:0 >>"${LOG}" 2>&1

write_status training_appearance
CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${PHYSTWIN_PY}" \
    "${ROOT_DIR}/baselines/phystwin_super/train_appearance.py" \
    --dataset-key "${DATASET_KEY}" --dataset "${NATIVE}" \
    --preprocess-dir "${PREPROCESS}" --output-dir "${APPEARANCE}" \
    --seed "${SEED}" --device cuda:0 --iterations 1000 >>"${LOG}" 2>&1

write_status optimizing_physics
CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${PHYSTWIN_PY}" \
    "${ROOT_DIR}/baselines/phystwin_super/run_physics.py" \
    --dataset-key "${DATASET_KEY}" --preprocess-dir "${PREPROCESS}" \
    --output-dir "${PHYSICS}" --seed "${SEED}" --device cuda:0 \
    --cma-iterations 20 --adam-iterations 200 --checkpoint-interval 20 --substeps 667 \
    >>"${LOG}" 2>&1

write_status exporting
CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${PHYSTWIN_PY}" \
    "${ROOT_DIR}/baselines/phystwin_super/export_super.py" \
    --dataset-key "${DATASET_KEY}" --dataset "${NATIVE}" \
    --physics-dir "${PHYSICS}" --appearance-dir "${APPEARANCE}" \
    --capture "${CAPTURE}" --render-dir "${RENDERS}" --device cuda:0 \
    >>"${LOG}" 2>&1

write_status preparing_render_metrics
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${ROOT_DIR}/scripts" "${EVAL_PY}" \
    "${ROOT_DIR}/scripts/prepare_endogaussian_super_render_metrics.py" \
    --dataset-key "${DATASET_KEY}" --prediction-dir "${RENDERS}" \
    --capture "${CAPTURE}" --render-scale 0.5 --delete-prediction-arrays >>"${LOG}" 2>&1

write_status scoring
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${ROOT_DIR}/scripts" "${EVAL_PY}" \
    "${ROOT_DIR}/scripts/score_super_joint_80to20_evaluation.py" \
    --ground-truth "${GT}" --capture "${CAPTURE}" --lpips-device cuda:0 >>"${LOG}" 2>&1

write_status complete
find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"${OUTPUT_ROOT}/SHA256SUMS"
echo "Complete: ${OUTPUT_ROOT}"
