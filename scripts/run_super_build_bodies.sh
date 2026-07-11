#!/usr/bin/env bash
set -euo pipefail

ROOT="/Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best"
ENV_PREFIX="/Media_HDD/jwshan/conda_envs/eg_codex"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Python not found: ${ENV_PREFIX}/bin/python" >&2
  exit 1
fi

export ENV_PREFIX
export CONDA_PREFIX="${ENV_PREFIX}"
export CUDA_HOME="${ENV_PREFIX}"
export CUDA_PATH="${ENV_PREFIX}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/eg_torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

include_paths=(
  "${ENV_PREFIX}/targets/x86_64-linux/include"
  "${ENV_PREFIX}/include"
)

library_paths=(
  "${ENV_PREFIX}/targets/x86_64-linux/lib"
  "${ENV_PREFIX}/x86_64-conda-linux-gnu/lib"
  "${ENV_PREFIX}/usr/lib/x86_64-linux-gnu"
  "${ENV_PREFIX}/lib"
)

join_existing_paths() {
  local joined=""
  local path
  for path in "$@"; do
    if [[ -d "${path}" ]]; then
      if [[ -z "${joined}" ]]; then
        joined="${path}"
      else
        joined="${joined}:${path}"
      fi
    fi
  done
  printf '%s' "${joined}"
}

export CPATH="$(join_existing_paths "${include_paths[@]}")"
export CPLUS_INCLUDE_PATH="${CPATH}"
export LIBRARY_PATH="$(join_existing_paths "${library_paths[@]}")"
export LD_LIBRARY_PATH="${LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${ROOT}"

mode="${1:-build}"
if [[ $# -gt 0 ]]; then
  shift
fi
case "${mode}" in
  prewarm)
    exec "${ENV_PREFIX}/bin/python" scripts/prewarm_gsplat.py "$@"
    ;;
  build)
    exec "${ENV_PREFIX}/bin/python" scripts/build_super_bodies_from_first_frame.py \
      --particle-radius 0.0015 \
      --gaussian-iters 600 \
      --output-dir data/super/grasp5_native/bodies \
      "$@"
    ;;
  quick)
    exec "${ENV_PREFIX}/bin/python" scripts/build_super_bodies_from_first_frame.py \
      --particle-radius 0.0015 \
      --skip-gaussian-training \
      --output-dir data/super/grasp5_native/bodies \
      "$@"
    ;;
  *)
    echo "Usage: $0 {prewarm|build|quick}" >&2
    exit 2
    ;;
esac
