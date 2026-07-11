#!/usr/bin/env bash
set -euo pipefail

# 在已经存在的虚拟显示器上运行 SUPER grasp5 离线回放。
# 默认 DISPLAY=:12，和 start_display_browser.sh 保持一致。
# 用法：
#   bash scripts/run_demo_on_display.sh
#   bash scripts/run_demo_on_display.sh /绝对/或/相对/数据集目录
#   bash scripts/run_demo_on_display.sh --monitor-psm-base-q
#   bash scripts/run_demo_on_display.sh /数据集目录 --monitor-psm-base-q --monitor-interval 0.2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
DISPLAY_NUM="${DISPLAY_NUM:-12}"
DATASET=""

if [[ $# -gt 0 && "$1" != --* ]]; then
    DATASET="$1"
    shift
fi

export CONDA_PREFIX="$ENV_PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/usr/bin:$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$CONDA_PREFIX/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export DISPLAY=":${DISPLAY_NUM}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/Media_HDD/jwshan/tmp/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

cd "$ROOT_DIR"

cmd=("$CONDA_PREFIX/bin/python" examples/example_embodied_super_offline.py --fps 30)
if [[ -n "$DATASET" ]]; then
    cmd+=(--dataset "$DATASET")
fi
cmd+=("$@")

exec "${cmd[@]}"
