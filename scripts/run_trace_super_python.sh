#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${TRACE_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/freegave}"
UPSTREAM_ROOT="${TRACE_ROOT:-/Media_HDD/jwshan/wad/embodied_gaussians_fixed_super_best_sim/baselines/TRACE}"
ADAPTER_ROOT="$ROOT_DIR/baselines/trace_super"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_trace_super_python.sh <python-script> [args...]" >&2
    exit 2
fi
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    echo "ERROR: TRACE environment not found: $ENV_ROOT" >&2
    exit 1
fi
if [[ ! -f "$UPSTREAM_ROOT/train_gui.py" ]]; then
    echo "ERROR: pinned TRACE source not found: $UPSTREAM_ROOT" >&2
    exit 1
fi

unset CC CXX CPP CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH CPLUS_INCLUDE_PATH LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDA_HOME="$ENV_ROOT"
export CUDA_PATH="$ENV_ROOT"
export PATH="$ENV_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ADAPTER_ROOT:$UPSTREAM_ROOT"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"

exec "$ENV_ROOT/bin/python" "$@"
