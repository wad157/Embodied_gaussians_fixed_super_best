#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"

detect_thinlinc_display() {
    local socket display vendor
    for socket in /tmp/.X11-unix/X*; do
        [[ -S "$socket" && -O "$socket" ]] || continue
        display=":${socket##*/X}"
        vendor="$(
            xdpyinfo -display "$display" 2>/dev/null \
                | sed -n 's/^vendor string:[[:space:]]*//p' \
                | head -n 1
        )"
        if [[ "$vendor" == *"ThinLinc"* ]]; then
            printf '%s\n' "$display"
            return 0
        fi
    done
    return 1
}

if [[ -z "${DISPLAY:-}" ]] || ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    DISPLAY="$(detect_thinlinc_display || true)"
    if [[ -z "$DISPLAY" ]]; then
        echo "ERROR: No usable ThinLinc DISPLAY was found." >&2
        echo "Run this command inside the ThinLinc desktop terminal." >&2
        exit 1
    fi
    export DISPLAY
    echo "[tissue_gt_annotator] detected ThinLinc DISPLAY=$DISPLAY"
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "ERROR: Python not found: $ENV_PREFIX/bin/python" >&2
    exit 1
fi

cd "$ROOT_DIR"
exec "$ENV_PREFIX/bin/python" scripts/annotate_super_tissue_gt_tracks.py "$@"
