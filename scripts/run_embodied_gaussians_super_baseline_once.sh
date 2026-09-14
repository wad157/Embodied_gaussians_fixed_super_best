#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EG_PY="${EG_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
GPU_ID="${EG_SUPER_GPU_ID:-1}"

if [[ $# -ne 4 ]]; then
    echo "Usage: bash scripts/run_embodied_gaussians_super_baseline_once.sh <grasp5|grasp3|grasp1> <repeat_01|repeat_02|repeat_03> <seed> <output>" >&2
    exit 2
fi
DATASET_KEY="$1"
REPEAT_ID="$2"
SEED="$3"
OUTPUT_ROOT="$(realpath -m "$4")"
case "$DATASET_KEY" in
    grasp5) NATIVE="$ROOT_DIR/data/super/grasp5_native"; GT="$ROOT_DIR/data/super/evaluation_v1/manual_tissue_tracks_10/ground_truth_2d3d_v1.npz" ;;
    grasp3) NATIVE="$ROOT_DIR/data/super/grasp3_native"; GT="$NATIVE/evaluation_v1/manual_tissue_tracks_10_v2/ground_truth_2d3d_v1.npz" ;;
    grasp1) NATIVE="$ROOT_DIR/data/super/grasp1_native"; GT="$NATIVE/evaluation_v2/manual_tissue_tracks_10_no_exclusion/ground_truth_2d3d_v1.npz" ;;
    *) echo "Unknown dataset: $DATASET_KEY" >&2; exit 2 ;;
esac
if [[ ! "$SEED" =~ ^[0-9]+$ ]] || [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Seed and EG_SUPER_GPU_ID must be non-negative integers" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite $OUTPUT_ROOT" >&2
    exit 1
fi
export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp/embodied_gaussians_super}"
mkdir -p "$TMPDIR" "$OUTPUT_ROOT/initialization"
DEPTH="$NATIVE/embodied_gaussians_initial_stereo_depth_v1"
BODY="$OUTPUT_ROOT/initialization/body.json"
ARTIFACTS="$OUTPUT_ROOT/artifacts"
CAPTURE="$OUTPUT_ROOT/capture"
RENDER_ARRAYS="$CAPTURE/render_predictions_float32"
LOG="$OUTPUT_ROOT/run.log"

printf '%s\n' \
    'status=preparing_initial_stereo_depth' \
    "dataset=$DATASET_KEY" "repeat=$REPEAT_ID" "seed=$SEED" "gpu=$GPU_ID" \
    'method=embodied_gaussians_super_paper_soft_v1' \
    'protocol=joint_reconstruction_7to1_future_80to20' >"$OUTPUT_ROOT/status.txt"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$EG_PY" \
    "$ROOT_DIR/baselines/embodied_gaussians_super/prepare_initial_depth.py" \
    --dataset-key "$DATASET_KEY" --device cuda:0 >"$LOG" 2>&1

printf 'status=initializing\n' >"$OUTPUT_ROOT/status.txt"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$EG_PY" \
    "$ROOT_DIR/baselines/embodied_gaussians_super/initialize.py" \
    --dataset-key "$DATASET_KEY" --dataset "$NATIVE" --depth-dir "$DEPTH" \
    --seed "$SEED" --output "$BODY" >>"$LOG" 2>&1

printf 'status=auditing\n' >"$OUTPUT_ROOT/status.txt"
"$EG_PY" "$ROOT_DIR/scripts/audit_embodied_gaussians_super_protocol.py" \
    --dataset-key "$DATASET_KEY" --body "$BODY" --depth-dir "$DEPTH" \
    --output "$OUTPUT_ROOT/protocol_audit.json" >>"$LOG" 2>&1

printf 'status=running_paper_soft_rollout\n' >"$OUTPUT_ROOT/status.txt"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$EG_PY" \
    "$ROOT_DIR/baselines/embodied_gaussians_super/run_soft.py" \
    --dataset-key "$DATASET_KEY" --dataset "$NATIVE" --body "$BODY" \
    --output-dir "$ARTIFACTS" --render-dir "$RENDER_ARRAYS" --seed "$SEED" \
    >>"$LOG" 2>&1

printf 'status=exporting_native_pbd_tracks\n' >"$OUTPUT_ROOT/status.txt"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$EG_PY" \
    "$ROOT_DIR/baselines/embodied_gaussians_super/export_super.py" \
    --dataset-key "$DATASET_KEY" --dataset "$NATIVE" --body "$BODY" \
    --rollout-dir "$ARTIFACTS" --capture "$CAPTURE" >>"$LOG" 2>&1

printf 'status=preparing_render_metrics\n' >"$OUTPUT_ROOT/status.txt"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$EG_PY" \
    "$ROOT_DIR/scripts/prepare_endogaussian_super_render_metrics.py" \
    --dataset-key "$DATASET_KEY" --prediction-dir "$RENDER_ARRAYS" \
    --capture "$CAPTURE" --render-scale 0.5 --delete-prediction-arrays \
    >>"$LOG" 2>&1

printf 'status=scoring\n' >"$OUTPUT_ROOT/status.txt"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$ROOT_DIR/scripts" "$EG_PY" \
    "$ROOT_DIR/scripts/score_super_joint_80to20_evaluation.py" \
    --ground-truth "$GT" --capture "$CAPTURE" --lpips-device cuda:0 \
    >>"$LOG" 2>&1

printf '%s\n' \
    'status=complete' \
    "dataset=$DATASET_KEY" "repeat=$REPEAT_ID" "seed=$SEED" "gpu=$GPU_ID" \
    'method=embodied_gaussians_super_paper_soft_v1' \
    'protocol=joint_reconstruction_7to1_future_80to20' >"$OUTPUT_ROOT/status.txt"
find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$OUTPUT_ROOT/SHA256SUMS"
echo "Complete: $OUTPUT_ROOT"
