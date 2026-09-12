#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONCE="${ROOT_DIR}/scripts/run_endogaussian_super_baseline_once.sh"
CAMPAIGN="${1:-${ROOT_DIR}/outputs/endogaussian_super_joint_v1}"
GPU_ID="${ENDOGAUSSIAN_GPU_ID:-0}"

mkdir -p "${CAMPAIGN}/logs"
for dataset in grasp5 grasp3 grasp1; do
    for seed in 0 1 2; do
        repeat="repeat_0$((seed + 1))"
        output="${CAMPAIGN}/${dataset}/${repeat}"
        if [[ -f "${output}/status.txt" ]] && grep -q '^status=complete$' "${output}/status.txt"; then
            echo "Skip complete ${dataset}/${repeat}"
            continue
        fi
        if [[ -e "${output}" ]]; then
            echo "Incomplete output requires inspection before resume: ${output}" >&2
            exit 2
        fi
        echo "Start ${dataset}/${repeat} on GPU ${GPU_ID}"
        ENDOGAUSSIAN_GPU_ID="${GPU_ID}" bash "${ONCE}" \
            "${dataset}" "${repeat}" "${seed}" "${output}" \
            > "${CAMPAIGN}/logs/${dataset}_${repeat}.log" 2>&1
    done
done

"/Media_HDD/jwshan/conda_envs/eg_codex/bin/python" \
    "${ROOT_DIR}/scripts/summarize_endogaussian_super_baseline.py" \
    --campaign "${CAMPAIGN}" \
    --output "${CAMPAIGN}/summary"
touch "${CAMPAIGN}/COMPLETE"
echo "Complete: ${CAMPAIGN}"
