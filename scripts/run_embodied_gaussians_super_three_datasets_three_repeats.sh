#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN="$(realpath -m "${1:-$ROOT_DIR/outputs/embodied_gaussians_super_joint_v1}")"
GPU_ID="${EG_SUPER_GPU_ID:-1}"
PYTHON="${EG_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"

if [[ -e "$CAMPAIGN/summary" ]]; then
    echo "Refusing to overwrite completed summary: $CAMPAIGN/summary" >&2
    exit 1
fi
mkdir -p "$CAMPAIGN/logs"
printf '%s\n' \
    'status=running' 'method=embodied_gaussians_super_paper_soft_v1' \
    'datasets=grasp5,grasp3,grasp1' 'seeds=0,1,2' 'repeat_count=3' \
    "gpu=$GPU_ID" 'aggregation=arithmetic_mean_population_std_no_best_selection' \
    >"$CAMPAIGN/campaign_status.txt"

for dataset in grasp5 grasp3 grasp1; do
    for seed in 0 1 2; do
        repeat="repeat_0$((seed + 1))"
        output="$CAMPAIGN/$dataset/$repeat"
        if [[ -f "$output/status.txt" ]] && grep -q '^status=complete$' "$output/status.txt"; then
            echo "Reuse complete $dataset/$repeat" >>"$CAMPAIGN/logs/progress.log"
            continue
        fi
        if [[ -e "$output" ]]; then
            echo "Incomplete output requires inspection: $output" >&2
            exit 1
        fi
        echo "Start $dataset/$repeat seed=$seed gpu=$GPU_ID" | tee -a "$CAMPAIGN/logs/progress.log"
        EG_SUPER_GPU_ID="$GPU_ID" bash "$ROOT_DIR/scripts/run_embodied_gaussians_super_baseline_once.sh" \
            "$dataset" "$repeat" "$seed" "$output" \
            >>"$CAMPAIGN/logs/${dataset}_${repeat}.log" 2>&1
        echo "Complete $dataset/$repeat seed=$seed gpu=$GPU_ID" | tee -a "$CAMPAIGN/logs/progress.log"
    done
done

"$PYTHON" "$ROOT_DIR/scripts/summarize_embodied_gaussians_super_baseline.py" --campaign "$CAMPAIGN"
printf '%s\n' \
    'status=complete' 'method=embodied_gaussians_super_paper_soft_v1' \
    'datasets=grasp5,grasp3,grasp1' 'seeds=0,1,2' 'repeat_count=3' \
    "gpu=$GPU_ID" 'aggregation=arithmetic_mean_population_std_no_best_selection' \
    >"$CAMPAIGN/campaign_status.txt"
(
    cd "$CAMPAIGN/summary"
    sha256sum summary.csv summary.json summary.md >SHA256SUMS
)
echo "Complete: $CAMPAIGN/summary/summary.md"
