#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN="$(realpath -m "${1:-$ROOT_DIR/outputs/eh_surgs_super_joint_v1}")"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"

if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/run_eh_surgs_super_three_datasets_three_repeats_two_gpus.sh [campaign]" >&2
    exit 2
fi
if [[ ! -f "$CAMPAIGN/grasp5/repeat_01/status.txt" ]] || \
   ! grep -q '^status=complete$' "$CAMPAIGN/grasp5/repeat_01/status.txt" || \
   ! grep -q '^seed=0$' "$CAMPAIGN/grasp5/repeat_01/status.txt"; then
    echo "ERROR: grasp5 repeat_01 seed 0 is not a completed reusable run" >&2
    exit 1
fi
if [[ -e "$CAMPAIGN/summary" ]]; then
    echo "ERROR: summary already exists; refusing to overwrite $CAMPAIGN" >&2
    exit 1
fi
mkdir -p "$CAMPAIGN/logs"
printf '%s\n' \
    'status=running' \
    'method=EH-SurGS_super_v1_noninstrument_mask' \
    'datasets=grasp5,grasp3,grasp1' \
    'seeds=0,1,2' \
    'repeat_count=3' \
    'grasp5_repeat_01=reused_completed_seed0' \
    'aggregation=arithmetic_mean_population_std_no_best_selection' \
    'gpus=0,1' >"$CAMPAIGN/campaign_status.txt"

run_one() {
    local dataset="$1" repeat="$2" seed="$3" gpu="$4"
    local output="$CAMPAIGN/$dataset/$repeat"
    if [[ -f "$output/status.txt" ]] && grep -q '^status=complete$' "$output/status.txt"; then
        return
    fi
    if [[ -e "$output" ]]; then
        echo "ERROR: refusing to overwrite incomplete output: $output" >&2
        return 1
    fi
    printf 'start=%s/%s seed=%s gpu=%s\n' "$dataset" "$repeat" "$seed" "$gpu" \
        >>"$CAMPAIGN/logs/progress.log"
    EH_SURGS_GPU_ID="$gpu" bash "$ROOT_DIR/scripts/run_eh_surgs_super_baseline_once.sh" \
        "$dataset" "$repeat" "$seed" "$output"
    printf 'complete=%s/%s seed=%s gpu=%s\n' "$dataset" "$repeat" "$seed" "$gpu" \
        >>"$CAMPAIGN/logs/progress.log"
}

gpu_zero() {
    run_one grasp5 repeat_02 1 0
    run_one grasp3 repeat_01 0 0
    run_one grasp1 repeat_01 0 0
    run_one grasp3 repeat_03 2 0
}

gpu_one() {
    run_one grasp5 repeat_03 2 1
    run_one grasp3 repeat_02 1 1
    run_one grasp1 repeat_02 1 1
    run_one grasp1 repeat_03 2 1
}

gpu_zero & pid_zero=$!
gpu_one & pid_one=$!
batch_status=0
wait "$pid_zero" || batch_status=1
wait "$pid_one" || batch_status=1
if (( batch_status != 0 )); then
    echo "ERROR: at least one GPU worker failed; outputs are preserved and no summary is written" >&2
    exit 1
fi

"$EVAL_PY" "$ROOT_DIR/scripts/summarize_eh_surgs_super_baseline.py" \
    --campaign "$CAMPAIGN"
printf '%s\n' \
    'status=complete' \
    'method=EH-SurGS_super_v1_noninstrument_mask' \
    'datasets=grasp5,grasp3,grasp1' \
    'seeds=0,1,2' \
    'repeat_count=3' \
    'grasp5_repeat_01=reused_completed_seed0' \
    'aggregation=arithmetic_mean_population_std_no_best_selection' \
    'gpus=0,1' >"$CAMPAIGN/campaign_status.txt"
find "$CAMPAIGN/summary" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$CAMPAIGN/summary/SHA256SUMS"
echo "Complete: $CAMPAIGN/summary/summary.md"
