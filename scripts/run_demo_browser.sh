#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
DISPLAY_NUM="${DISPLAY_NUM:-99}"
VNC_PORT="${VNC_PORT:-5901}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
GEOMETRY="${GEOMETRY:-1280x720x24}"
LOG_DIR="${LOG_DIR:-/Media_HDD/jwshan/tmp/eg_demo_logs}"
DATASET="${1:-}"

mkdir -p "$LOG_DIR"

export CONDA_PREFIX="$ENV_PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/usr/bin:$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$CONDA_PREFIX/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$CONDA_PREFIX/bin/python" ]]; then
    echo "Missing conda Python at $CONDA_PREFIX/bin/python" >&2
    exit 1
fi

cleanup_old() {
    pkill -u "$USER" -f "Xvfb :${DISPLAY_NUM}( |$)" 2>/dev/null || true
    pkill -u "$USER" -f "x11vnc .*:${DISPLAY_NUM}" 2>/dev/null || true
    pkill -u "$USER" -f "websockify .*${NOVNC_PORT} .*${VNC_PORT}" 2>/dev/null || true
    pkill -u "$USER" -f "example_embodied_pusht_offline.py" 2>/dev/null || true
    rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}" 2>/dev/null || true
}

wait_for_port() {
    local port="$1"
    for _ in $(seq 1 40); do
        if "$CONDA_PREFIX/bin/python" - "$port" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(0.2)
try:
    sock.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
else:
    sys.exit(0)
finally:
    sock.close()
PY
        then
            return 0
        fi
        sleep 0.25
    done
    return 1
}

cleanup_old

echo "[1/4] Starting Xvfb on :${DISPLAY_NUM}"
Xvfb ":${DISPLAY_NUM}" -screen 0 "$GEOMETRY" +extension GLX +render -noreset \
    >"$LOG_DIR/xvfb.log" 2>&1 &
XVFB_PID=$!
sleep 1
if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "Xvfb failed. Log:" >&2
    cat "$LOG_DIR/xvfb.log" >&2
    exit 1
fi

echo "[2/4] Starting x11vnc on 127.0.0.1:${VNC_PORT}"
x11vnc -display ":${DISPLAY_NUM}" -localhost -nopw -forever -shared -rfbport "$VNC_PORT" \
    >"$LOG_DIR/x11vnc.log" 2>&1 &
X11VNC_PID=$!
if ! wait_for_port "$VNC_PORT"; then
    echo "x11vnc failed. Log:" >&2
    cat "$LOG_DIR/x11vnc.log" >&2
    exit 1
fi

echo "[3/4] Starting noVNC on 127.0.0.1:${NOVNC_PORT}"
websockify --web "$CONDA_PREFIX/usr/share/novnc" "$NOVNC_PORT" "127.0.0.1:${VNC_PORT}" \
    >"$LOG_DIR/novnc.log" 2>&1 &
NOVNC_PID=$!
if ! wait_for_port "$NOVNC_PORT"; then
    echo "noVNC/websockify failed. Log:" >&2
    cat "$LOG_DIR/novnc.log" >&2
    exit 1
fi

echo "[4/4] Starting demo on DISPLAY=:${DISPLAY_NUM}"
cd "$ROOT_DIR"
export DISPLAY=":${DISPLAY_NUM}"

echo
echo "Server is ready."
echo "Open this from your local browser after creating the SSH tunnel:"
echo "  http://127.0.0.1:16080/vnc.html?host=127.0.0.1&port=16080&autoconnect=true"
echo
echo "Logs:"
echo "  $LOG_DIR/xvfb.log"
echo "  $LOG_DIR/x11vnc.log"
echo "  $LOG_DIR/novnc.log"
echo

if [[ -n "$DATASET" ]]; then
    exec "$CONDA_PREFIX/bin/python" examples/example_embodied_pusht_offline.py --dataset "$DATASET"
else
    exec "$CONDA_PREFIX/bin/python" examples/example_embodied_pusht_offline.py
fi
