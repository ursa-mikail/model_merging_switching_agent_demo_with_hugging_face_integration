#!/usr/bin/env bash
# up.sh — build and start the Splice stack (backend + frontend).
#
# Frees up the ports we need first (in case a previous run, or some other
# process on the machine, is still squatting on them), then builds and
# starts both containers via docker compose.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BACKEND_PORT="${BACKEND_PORT:-8080}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# Prefer `docker compose` (v2 plugin); fall back to the legacy `docker-compose`.
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "Error: neither 'docker compose' nor 'docker-compose' was found on PATH." >&2
  echo "Install Docker Desktop (or the Docker Engine + compose plugin) and try again." >&2
  exit 1
fi

free_port() {
  local port="$1"
  local pids=""

  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser "${port}/tcp" 2>/dev/null || true)"
  fi

  if [ -n "$pids" ]; then
    echo "Port $port is in use by PID(s): $pids — freeing it up..."
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
}

echo "==> Checking for port conflicts on $BACKEND_PORT and $FRONTEND_PORT..."
free_port "$BACKEND_PORT"
free_port "$FRONTEND_PORT"

echo "==> Building images (this can take a minute the first time)..."
BACKEND_PORT="$BACKEND_PORT" FRONTEND_PORT="$FRONTEND_PORT" "${COMPOSE[@]}" build

echo "==> Starting containers..."
BACKEND_PORT="$BACKEND_PORT" FRONTEND_PORT="$FRONTEND_PORT" "${COMPOSE[@]}" up -d

echo "==> Waiting for the backend to report healthy..."
attempts=0
until curl -fs "http://localhost:${BACKEND_PORT}/api/health" >/dev/null 2>&1; do
  attempts=$((attempts + 1))
  if [ "$attempts" -gt 30 ]; then
    echo "Backend did not become healthy in time. Check logs with:" >&2
    echo "  ${COMPOSE[*]} logs backend" >&2
    exit 1
  fi
  sleep 1
done

cat <<EOF

✅ Splice is up.

   Frontend:  http://localhost:${FRONTEND_PORT}
   Backend:   http://localhost:${BACKEND_PORT}/api/health

   Logs:      ${COMPOSE[*]} logs -f
   Stop:      ./scripts/down.sh
   Reset all: ./scripts/clean.sh
EOF
