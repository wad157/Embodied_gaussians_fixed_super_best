#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EH_PY="$ROOT_DIR/scripts/run_eh_surgs_super_python.sh"
EVAL_ENV="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
EVAL_PY="$EVAL_ENV/bin/python"
GPU_ID="${EH_SURGS_GPU_ID:-0}"
CONFIG="${EH_SURGS_CONFIG:-$ROOT_DIR/baselines/eh_surgs_super/configs/super_unified.py}"

if [[ $# -ne 4 ]]; then
    echo "Usage: bash scripts/run_eh_surgs_super_baseline_once.sh <grasp5|grasp3|grasp1> <repeat_id> <seed> <output>" >&2
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
    *) echo "Unknown dataset: $DATASET_KEY" >&2; exit 2 ;;
esac
if [[ ! "$SEED" =~ ^[0-9]+$ ]] || [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Seed and EH_SURGS_GPU_ID must be non-negative integers" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite $OUTPUT_ROOT" >&2
    exit 1
fi

MODEL_PATH="$OUTPUT_ROOT/model"
CAPTURE="$OUTPUT_ROOT/capture"
LOG="$OUTPUT_ROOT/run.log"
mkdir -p "$OUTPUT_ROOT"
printf '%s\n' \
    'status=preparing_depth' \
    "dataset=$DATASET_KEY" "repeat=$REPEAT_ID" "seed=$SEED" "gpu=$GPU_ID" \
    'method=EH-SurGS_super_v1_noninstrument_mask' \
    'protocol=joint_reconstruction_7to1_future_80to20' \
    >"$OUTPUT_ROOT/status.txt"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$EVAL_PY" \
    "$ROOT_DIR/scripts/prepare_endogaussian_super_depth.py" \
    --dataset-key "$DATASET_KEY" --device cuda:0 >"$LOG" 2>&1

printf 'status=auditing\n' >"$OUTPUT_ROOT/status.txt"
"$EVAL_PY" "$ROOT_DIR/scripts/audit_eh_surgs_super_protocol.py" \
    --dataset-key "$DATASET_KEY" --native "$NATIVE" \
    --output "$OUTPUT_ROOT/protocol_audit.json" >>"$LOG" 2>&1

printf 'status=training\n' >"$OUTPUT_ROOT/status.txt"
CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$EH_PY" \
    "$ROOT_DIR/baselines/eh_surgs_super/train_super.py" \
    --dataset-key "$DATASET_KEY" --seed "$SEED" \
    --source_path "$NATIVE" --model_path "$MODEL_PATH" \
    --configs "$CONFIG" >>"$LOG" 2>&1

printf 'status=exporting\n' >"$OUTPUT_ROOT/status.txt"
CUDA_VISIBLE_DEVICES="$GPU_ID" EVAL_PYTHON="$EVAL_PY" bash "$EH_PY" \
    "$ROOT_DIR/baselines/eh_surgs_super/export_super.py" \
    --dataset-key "$DATASET_KEY" --source_path "$NATIVE" \
    --model_path "$MODEL_PATH" --output-dir "$CAPTURE" \
    --configs "$CONFIG" >>"$LOG" 2>&1

printf 'status=scoring\n' >"$OUTPUT_ROOT/status.txt"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$ROOT_DIR/scripts" "$EVAL_PY" \
    "$ROOT_DIR/scripts/score_super_joint_80to20_evaluation.py" \
    --ground-truth "$GT" --capture "$CAPTURE" --lpips-device cuda:0 \
    >>"$LOG" 2>&1

printf '%s\n' \
    'status=complete' \
    "dataset=$DATASET_KEY" "repeat=$REPEAT_ID" "seed=$SEED" "gpu=$GPU_ID" \
    'method=EH-SurGS_super_v1_noninstrument_mask' \
    'protocol=joint_reconstruction_7to1_future_80to20' \
    >"$OUTPUT_ROOT/status.txt"
find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/SHA256SUMS"
echo "Complete: $OUTPUT_ROOT"
