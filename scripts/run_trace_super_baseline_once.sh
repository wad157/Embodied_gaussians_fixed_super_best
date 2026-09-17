#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_PY="$ROOT_DIR/scripts/run_trace_super_python.sh"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
GPU_ID="${TRACE_SUPER_GPU_ID:-0}"

if [[ $# -ne 4 ]]; then
    echo "Usage: bash scripts/run_trace_super_baseline_once.sh <grasp5|grasp3|grasp1> <repeat_01|repeat_02|repeat_03> <seed> <output>" >&2
    exit 2
fi
DATASET_KEY="$1"
REPEAT_ID="$2"
SEED="$3"
OUTPUT_ROOT="$(realpath -m "$4")"
case "$DATASET_KEY" in
    grasp5)
        NATIVE="$ROOT_DIR/data/super/grasp5_native"
        GT="$ROOT_DIR/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz"
        ;;
    grasp3)
        NATIVE="$ROOT_DIR/data/super/grasp3_native"
        GT="$NATIVE/evaluation_v1/manual_tissue_tracks_10_v2/ground_truth_2d3d_v1.npz"
        ;;
    grasp1)
        NATIVE="$ROOT_DIR/data/super/grasp1_native"
        GT="$NATIVE/evaluation_v2/manual_tissue_tracks_10_no_exclusion/ground_truth_2d3d_v1.npz"
        ;;
    *) echo "ERROR: unknown SUPER dataset: $DATASET_KEY" >&2; exit 2 ;;
esac
if [[ ! "$SEED" =~ ^[0-9]+$ ]] || [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: seed and TRACE_SUPER_GPU_ID must be non-negative integers" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: refusing to overwrite TRACE output: $OUTPUT_ROOT" >&2
    exit 1
fi

MODEL_PATH="$OUTPUT_ROOT/model"
CAPTURE="$OUTPUT_ROOT/capture"
LOG="$OUTPUT_ROOT/run.log"
mkdir -p "$OUTPUT_ROOT"

write_status() {
    local stage="$1"
    printf '%s\n' \
        "status=$stage" \
        "dataset=$DATASET_KEY" \
        "repeat=$REPEAT_ID" \
        "seed=$SEED" \
        "gpu=$GPU_ID" \
        'method=trace_pinned_rgb_only_full_k_no_psm' \
        'protocol=joint_reconstruction_7to1_future_80to20' \
        >"$OUTPUT_ROOT/status.txt"
}

write_status auditing
"$EVAL_PY" "$ROOT_DIR/scripts/audit_trace_super_protocol.py" \
    --dataset-key "$DATASET_KEY" \
    --native "$NATIVE" \
    --output "$OUTPUT_ROOT/protocol_audit.json" \
    >"$LOG" 2>&1

write_status training
CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$TRACE_PY" \
    "$ROOT_DIR/baselines/trace_super/train_super.py" \
    --dataset-key "$DATASET_KEY" \
    --seed "$SEED" \
    --source_path "$NATIVE" \
    --model_path "$MODEL_PATH" \
    --iterations 40000 \
    --super-downsample 3 \
    --super-init-points 50000 \
    >>"$LOG" 2>&1

write_status exporting
CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$TRACE_PY" \
    "$ROOT_DIR/baselines/trace_super/export_super.py" \
    --dataset-key "$DATASET_KEY" \
    --source-path "$NATIVE" \
    --model-path "$MODEL_PATH" \
    --output-dir "$CAPTURE" \
    --render-scale 0.5 \
    >>"$LOG" 2>&1

write_status scoring
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$ROOT_DIR/scripts" "$EVAL_PY" \
    "$ROOT_DIR/scripts/score_super_joint_80to20_evaluation.py" \
    --ground-truth "$GT" \
    --capture "$CAPTURE" \
    --lpips-device cuda:0 \
    >>"$LOG" 2>&1

write_status complete
find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$OUTPUT_ROOT/SHA256SUMS"
echo "Complete: $OUTPUT_ROOT"
