#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
SERVER_IP="${SERVER_IP:-192.168.0.6}"
LISTEN_ADDR="${LISTEN_ADDR:-0.0.0.0}"
DISPLAY_NUM="${DISPLAY_NUM:-12}"
VNC_PORT="${VNC_PORT:-5912}"
NOVNC_PORT="${NOVNC_PORT:-6082}"
GEOMETRY="${GEOMETRY:-1280x720x24}"

export CONDA_PREFIX="$ENV_PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/usr/bin:$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$CONDA_PREFIX/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

pkill -u "$USER" -f "Xvfb :${DISPLAY_NUM}( |$)" 2>/dev/null || true
pkill -u "$USER" -f "x11vnc .*${VNC_PORT}" 2>/dev/null || true
pkill -u "$USER" -f "websockify.*${NOVNC_PORT}" 2>/dev/null || true
rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}" 2>/dev/null || true

setsid -f Xvfb ":${DISPLAY_NUM}" -screen 0 "$GEOMETRY" +extension GLX +render -noreset \
    >"/tmp/xvfb-${DISPLAY_NUM}.log" 2>&1 < /dev/null
sleep 1

DISPLAY=":${DISPLAY_NUM}" xsetroot -cursor_name left_ptr 2>/dev/null || true

setsid -f x11vnc -display ":${DISPLAY_NUM}" -rfbport "$VNC_PORT" -forever -shared -nopw -listen 0.0.0.0 -cursor arrow \
    >"/tmp/x11vnc-${DISPLAY_NUM}.log" 2>&1 < /dev/null
sleep 1

setsid -f "$CONDA_PREFIX/bin/python" -m websockify --web "$CONDA_PREFIX/usr/share/novnc" \
    "${LISTEN_ADDR}:${NOVNC_PORT}" "127.0.0.1:${VNC_PORT}" \
    >"/tmp/novnc-${NOVNC_PORT}.log" 2>&1 < /dev/null
sleep 1

echo "Display stack started:"
echo "  X display: :${DISPLAY_NUM}"
echo "  VNC:       ${SERVER_IP}:${VNC_PORT}"
echo "  noVNC:     http://${SERVER_IP}:${NOVNC_PORT}/vnc.html"
echo
echo "Local tunnel example:"
echo "  ssh -N -L 20000:${SERVER_IP}:${NOVNC_PORT} -p 221 jwshan@137.189.101.233"
echo "  http://localhost:20000/vnc.html"
echo
echo "Logs:"
echo "  /tmp/xvfb-${DISPLAY_NUM}.log"
echo "  /tmp/x11vnc-${DISPLAY_NUM}.log"
echo "  /tmp/novnc-${NOVNC_PORT}.log"
