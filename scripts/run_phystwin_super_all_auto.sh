#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN="${PHYSTWIN_SUPER_CAMPAIGN:-${ROOT_DIR}/outputs/phystwin_super_joint_v1}"
RUNNER="${ROOT_DIR}/scripts/run_phystwin_super_baseline_once.sh"
EVAL_PY="${EVAL_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"

status_of() {
    local path="$1"
    [[ -f "${path}/status.txt" ]] || { echo missing; return; }
    sed -n 's/^status=//p' "${path}/status.txt" | head -1
}

wait_existing() {
    local path="$1"
    while true; do
        case "$(status_of "${path}")" in
            complete) return 0 ;;
            failed) echo "Existing run failed: ${path}" >&2; return 1 ;;
            *) sleep 60 ;;
        esac
    done
}

run_one() {
    local gpu="$1" dataset="$2" repeat="$3" seed="$4"
    local output="${CAMPAIGN}/${dataset}/${repeat}"
    if [[ -e "${output}" ]]; then
        wait_existing "${output}"
    else
        PHYSTWIN_GPU_ID="${gpu}" bash "${RUNNER}" "${dataset}" "${repeat}" "${seed}" "${output}"
    fi
}

worker_zero() {
    run_one 0 grasp5 repeat_01 0
    run_one 0 grasp1 repeat_01 0
    run_one 0 grasp1 repeat_03 2
    run_one 0 grasp3 repeat_02 1
}

worker_one() {
    run_one 1 grasp5 repeat_02 1
    run_one 1 grasp1 repeat_02 1
    run_one 1 grasp3 repeat_01 0
    run_one 1 grasp3 repeat_03 2
    run_one 1 grasp5 repeat_03 2
}

if [[ "${1:-}" == "worker-zero" ]]; then
    worker_zero
    exit
elif [[ "${1:-}" == "worker-one" ]]; then
    worker_one
    exit
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/run_phystwin_super_all_auto.sh" >&2
    exit 2
fi

mkdir -p "${CAMPAIGN}"
printf 'status=running\nprotocol=joint_reconstruction_7to1_future_80to20\n' >"${CAMPAIGN}/campaign_status.txt"
bash "$0" worker-zero &
PID_ZERO=$!
bash "$0" worker-one &
PID_ONE=$!
wait "${PID_ZERO}"
wait "${PID_ONE}"
"${EVAL_PY}" "${ROOT_DIR}/scripts/summarize_phystwin_super_baseline.py" \
    --campaign "${CAMPAIGN}" --output "${CAMPAIGN}/summary"
printf 'status=complete\nprotocol=joint_reconstruction_7to1_future_80to20\n' >"${CAMPAIGN}/campaign_status.txt"
echo "PhysTwin SUPER campaign complete: ${CAMPAIGN}"
