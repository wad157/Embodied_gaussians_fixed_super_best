#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENDOGAUSSIAN_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/endogaussian_baseline}"
UPSTREAM_ROOT="${ENDOGAUSSIAN_ROOT:-${ROOT_DIR}/baselines/EndoGaussian}"
LOCAL_SOURCE="${ENDOGAUSSIAN_LOCAL_SOURCE:-${ROOT_DIR}/../embodied_gaussians_fixed_super_best_sim/baselines/EndoGaussian}"
UPSTREAM_COMMIT="8d12793838a1595b299df0696c8149c07329e980"

if [[ ! -d "${UPSTREAM_ROOT}/.git" ]]; then
    mkdir -p "$(dirname "${UPSTREAM_ROOT}")"
    if [[ -d "${LOCAL_SOURCE}/.git" ]]; then
        cp -a --reflink=auto "${LOCAL_SOURCE}" "${UPSTREAM_ROOT}"
    else
        git clone https://github.com/CUHK-AIM-Group/EndoGaussian.git "${UPSTREAM_ROOT}"
        git -C "${UPSTREAM_ROOT}" checkout --detach "${UPSTREAM_COMMIT}"
        git -C "${UPSTREAM_ROOT}" submodule update --init --recursive
    fi
fi
actual_commit="$(git -C "${UPSTREAM_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${UPSTREAM_COMMIT}" ]]; then
    echo "EndoGaussian checkout mismatch: ${actual_commit}" >&2
    exit 1
fi

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
    echo "The baseline environment must be created at ${ENV_ROOT}." >&2
    echo "Run the setup from the SIM baseline or set ENDOGAUSSIAN_ENV_PREFIX." >&2
    exit 1
fi

bash "${ROOT_DIR}/scripts/run_endogaussian_super_python.sh" -c \
    'import torch,diff_gaussian_rasterization,simple_knn; print(torch.__version__, torch.version.cuda); print(diff_gaussian_rasterization.__file__)'
echo "EndoGaussian SUPER baseline ready: ${UPSTREAM_ROOT}"
