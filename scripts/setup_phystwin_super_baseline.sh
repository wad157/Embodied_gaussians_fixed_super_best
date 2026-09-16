#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${PHYSTWIN_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/phystwin_sim}"
UPSTREAM_ROOT="${PHYSTWIN_ROOT:-${ROOT_DIR}/baselines/PhysTwin}"
LOCAL_SOURCE="${PHYSTWIN_LOCAL_SOURCE:-${ROOT_DIR}/../embodied_gaussians_fixed_super_best_sim/baselines/PhysTwin}"
UPSTREAM_COMMIT="81c718790a37e5e0102eb77af2c6edd34a9db25f"
PATCH="${ROOT_DIR}/baselines/phystwin_super/patches/headless_imports.patch"
EXPECTED_PATCH_SHA256="4ad745d0baf8410672b95817f529bb4d9ac0cfbfe45bc6cb1626d0d08df693b8"

if [[ ! -d "${UPSTREAM_ROOT}/.git" ]]; then
    mkdir -p "$(dirname "${UPSTREAM_ROOT}")"
    if [[ -d "${LOCAL_SOURCE}/.git" ]]; then
        cp -a --reflink=auto "${LOCAL_SOURCE}" "${UPSTREAM_ROOT}"
    else
        git clone --recursive https://github.com/jianghanxiao/phystwin.git "${UPSTREAM_ROOT}"
    fi
fi
if [[ "$(git -C "${UPSTREAM_ROOT}" rev-parse HEAD)" != "${UPSTREAM_COMMIT}" ]]; then
    git -C "${UPSTREAM_ROOT}" fetch origin
    git -C "${UPSTREAM_ROOT}" checkout --detach "${UPSTREAM_COMMIT}"
    git -C "${UPSTREAM_ROOT}" submodule update --init --recursive
fi
if [[ -z "$(git -C "${UPSTREAM_ROOT}" diff -- qqtt/__init__.py qqtt/utils/__init__.py)" ]]; then
    git -C "${UPSTREAM_ROOT}" apply "${PATCH}"
fi
actual_patch="$(git -C "${UPSTREAM_ROOT}" diff -- qqtt/__init__.py qqtt/utils/__init__.py | sha256sum | awk '{print $1}')"
if [[ "${actual_patch}" != "${EXPECTED_PATCH_SHA256}" ]]; then
    echo "Unexpected PhysTwin headless patch: ${actual_patch}" >&2
    exit 1
fi
if [[ -n "$(git -C "${UPSTREAM_ROOT}" diff --name-only -- . ':(exclude)qqtt/__init__.py' ':(exclude)qqtt/utils/__init__.py')" ]]; then
    echo "PhysTwin contains unauthorized tracked source changes" >&2
    exit 1
fi
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
    echo "Missing shared PhysTwin environment: ${ENV_ROOT}" >&2
    echo "Run the SIM PhysTwin setup first or set PHYSTWIN_ENV_PREFIX." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${PHYSTWIN_GPU_ID:-0}" bash "${ROOT_DIR}/scripts/run_phystwin_super_python.sh" -c \
    'import cma,gsplat,open3d,torch,warp; from qqtt.model.diff_simulator import SpringMassSystemWarp; print(torch.__version__,torch.version.cuda); print("warp",warp.__version__,"gsplat",gsplat.__version__)'
echo "PhysTwin SUPER baseline ready: ${UPSTREAM_ROOT}"
