#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN="$(realpath -m "${1:-$ROOT_DIR/outputs/trace_super_joint_v1}")"
RUNNER="$ROOT_DIR/scripts/run_trace_super_baseline_once.sh"
SUMMARY_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
MAX_ATTEMPTS="${TRACE_SUPER_MAX_ATTEMPTS:-3}"

mkdir -p "$CAMPAIGN/logs"

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

is_complete() {
    local output="$1"
    [[ -f "$output/status.txt" ]] && grep -q '^status=complete$' "$output/status.txt"
}

archive_incomplete() {
    local output="$1"
    local attempt="$2"
    if [[ -e "$output" || -L "$output" ]]; then
        local archived="${output}.failed_attempt_${attempt}_$(date '+%Y%m%d_%H%M%S')"
        mv "$output" "$archived"
        echo "[$(timestamp)] archived incomplete output: $archived"
    fi
}

run_one() {
    local gpu="$1"
    local dataset="$2"
    local repeat="$3"
    local seed="$4"
    local output="$5"
    local attempt=1
    if is_complete "$output"; then
        echo "[$(timestamp)] skip complete $dataset/$repeat"
        return 0
    fi
    while (( attempt <= MAX_ATTEMPTS )); do
        archive_incomplete "$output" "$attempt"
        echo "[$(timestamp)] start gpu=$gpu $dataset/$repeat seed=$seed attempt=$attempt"
        if TRACE_SUPER_GPU_ID="$gpu" bash "$RUNNER" \
            "$dataset" "$repeat" "$seed" "$output"; then
            if is_complete "$output"; then
                echo "[$(timestamp)] complete gpu=$gpu $dataset/$repeat"
                return 0
            fi
        fi
        echo "[$(timestamp)] failed gpu=$gpu $dataset/$repeat attempt=$attempt"
        attempt=$((attempt + 1))
        sleep 15
    done
    echo "[$(timestamp)] exhausted retries gpu=$gpu $dataset/$repeat" >&2
    return 1
}

run_queue() {
    local gpu="$1"
    shift
    local failed=0
    while (( $# >= 4 )); do
        run_one "$gpu" "$1" "$2" "$3" "$4" || failed=1
        shift 4
    done
    return "$failed"
}

printf 'status=running\nstarted=%s\n' "$(timestamp)" >"$CAMPAIGN/scheduler_status.txt"

run_queue 0 \
    grasp5 repeat_01 0 "$CAMPAIGN/grasp5/repeat_01" \
    grasp3 repeat_01 0 "$CAMPAIGN/grasp3/repeat_01" \
    grasp1 repeat_01 0 "$CAMPAIGN/grasp1/repeat_01" \
    grasp1 repeat_03 2 "$CAMPAIGN/grasp1/repeat_03" \
    >"$CAMPAIGN/logs/gpu0_queue.log" 2>&1 &
QUEUE_A=$!

run_queue 1 \
    grasp5 repeat_02 1 "$CAMPAIGN/grasp5/repeat_02" \
    grasp5 repeat_03 2 "$CAMPAIGN/grasp5/repeat_03" \
    grasp3 repeat_02 1 "$CAMPAIGN/grasp3/repeat_02" \
    grasp3 repeat_03 2 "$CAMPAIGN/grasp3/repeat_03" \
    grasp1 repeat_02 1 "$CAMPAIGN/grasp1/repeat_02" \
    >"$CAMPAIGN/logs/gpu1_queue.log" 2>&1 &
QUEUE_B=$!

printf 'queue_pids=%s,%s\n' "$QUEUE_A" "$QUEUE_B" >>"$CAMPAIGN/scheduler_status.txt"

FAILED=0
wait "$QUEUE_A" || FAILED=1
wait "$QUEUE_B" || FAILED=1

if (( FAILED == 0 )); then
    if "$SUMMARY_PY" "$ROOT_DIR/scripts/summarize_trace_super_baseline.py" \
        --campaign "$CAMPAIGN" >"$CAMPAIGN/logs/summary.log" 2>&1; then
        printf 'status=complete\nfinished=%s\n' "$(timestamp)" \
            >"$CAMPAIGN/scheduler_status.txt"
        touch "$CAMPAIGN/COMPLETE"
        exit 0
    fi
fi
printf 'status=failed\nfinished=%s\n' "$(timestamp)" \
    >"$CAMPAIGN/scheduler_status.txt"
exit 1
