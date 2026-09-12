#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENDOGAUSSIAN_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/endogaussian_baseline}"
UPSTREAM_ROOT="${ENDOGAUSSIAN_ROOT:-${ROOT_DIR}/baselines/EndoGaussian}"
ADAPTER_ROOT="${ROOT_DIR}/baselines/endogaussian_super"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_endogaussian_super_python.sh SCRIPT [ARGS...]" >&2
    exit 2
fi
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
    echo "Missing EndoGaussian environment: ${ENV_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${UPSTREAM_ROOT}/gaussian_renderer/__init__.py" ]]; then
    echo "Missing pinned EndoGaussian checkout: ${UPSTREAM_ROOT}" >&2
    exit 1
fi

unset CC CXX CPP CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH CPLUS_INCLUDE_PATH LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDA_HOME="${ENV_ROOT}"
export CUDA_PATH="${ENV_ROOT}"
export PATH="${ENV_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${UPSTREAM_ROOT}:${ADAPTER_ROOT}"
export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

exec "${ENV_ROOT}/bin/python" "$@"
