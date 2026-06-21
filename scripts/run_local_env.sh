#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-scourrl-hud:dev}"
NAME="${NAME:-scourrl-hud-env}"
HUD_PORT="${HUD_PORT:-8765}"
NOVNC_PORT="${NOVNC_PORT:-8080}"

docker build -f Dockerfile.hud -t "$IMAGE" .
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  -p "${HUD_PORT}:8765" \
  -p "${NOVNC_PORT}:8080" \
  "$IMAGE" >/dev/null

echo "HUD control channel: tcp://127.0.0.1:${HUD_PORT}"
echo "Browser viewer:      http://127.0.0.1:${NOVNC_PORT}/vnc.html"
echo
echo "Waiting for the environment..."
for _ in $(seq 1 60); do
  if uv run hud client info --url "tcp://127.0.0.1:${HUD_PORT}" >/dev/null 2>&1; then
    uv run hud client info --url "tcp://127.0.0.1:${HUD_PORT}"
    exit 0
  fi
  sleep 2
done

echo "Environment did not become ready within 120s. Logs:" >&2
docker logs "$NAME" >&2
exit 1
