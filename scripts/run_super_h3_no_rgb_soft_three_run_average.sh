#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ONE="${ROOT_DIR}/scripts/run_super_h3_no_rgb_soft_full_metrics.sh"
SUMMARIZER="${ROOT_DIR}/scripts/summarize_super_soft_three_runs.py"
RUN1="${ROOT_DIR}/outputs/super_h3_no_post_rgb_soft_strong_h3_20260904_v1"
RUN2="${ROOT_DIR}/outputs/super_h3_no_post_rgb_soft_strong_h3_20260904_repeat2_v1"
RUN3="${ROOT_DIR}/outputs/super_h3_no_post_rgb_soft_strong_h3_20260904_repeat3_v1"
SUMMARY_ROOT="${ROOT_DIR}/outputs/super_h3_no_post_rgb_soft_strong_h3_three_run_average_20260904_v1"

if [[ ! -f "${RUN1}/COMPLETE" ]]; then
    printf 'Run 1 is not complete: %s\n' "${RUN1}" >&2
    exit 2
fi
if [[ -e "${RUN2}" || -e "${RUN3}" || -e "${SUMMARY_ROOT}" ]]; then
    printf 'Refusing reused repeat/summary output root.\n' >&2
    exit 2
fi

mkdir -p "${SUMMARY_ROOT}"
printf 'run1=%s\nrun2=%s\nrun3=%s\n' \
    "${RUN1}" "${RUN2}" "${RUN3}" > "${SUMMARY_ROOT}/RUNS.txt"

bash "${RUN_ONE}" "${RUN2}" > "${SUMMARY_ROOT}/repeat2.log" 2>&1
bash "${RUN_ONE}" "${RUN3}" > "${SUMMARY_ROOT}/repeat3.log" 2>&1

"/Media_HDD/jwshan/conda_envs/eg_codex/bin/python" "${SUMMARIZER}" \
    --run "run1=${RUN1}" \
    --run "run2=${RUN2}" \
    --run "run3=${RUN3}" \
    --output-json "${SUMMARY_ROOT}/average_results.json" \
    --output-markdown "${SUMMARY_ROOT}/average_results.md"
printf 'complete\n' > "${SUMMARY_ROOT}/COMPLETE"
