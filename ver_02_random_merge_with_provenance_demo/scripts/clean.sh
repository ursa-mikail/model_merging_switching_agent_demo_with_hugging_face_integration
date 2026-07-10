#!/usr/bin/env bash
# clean.sh — full teardown: stops containers, removes the built images and
# the model_data volume (deleting every downloaded/merged model), and frees
# up the ports. Use this when you want a completely fresh start.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BACKEND_PORT="${BACKEND_PORT:-8080}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "Error: neither 'docker compose' nor 'docker-compose' was found on PATH." >&2
  exit 1
fi

read -r -p "This will delete all downloaded/merged models and built images. Continue? [y/N] " confirm
case "$confirm" in
  [yY][eE][sS]|[yY]) ;;
  *) echo "Aborted."; exit 0 ;;
esac

echo "==> Stopping and removing containers, network, and the model_data volume..."
"${COMPOSE[@]}" down --volumes --remove-orphans

echo "==> Removing built images..."
docker image rm -f hf-mergekit-demo-backend:latest hf-mergekit-demo-frontend:latest 2>/dev/null || true

free_port() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser "${port}/tcp" 2>/dev/null || true)"
  fi
  if [ -n "$pids" ]; then
    echo "Freeing port $port (PID(s): $pids)..."
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
}

echo "==> Freeing ports $BACKEND_PORT and $FRONTEND_PORT..."
free_port "$BACKEND_PORT"
free_port "$FRONTEND_PORT"

echo "✅ Clean. Everything (including downloaded models) has been removed."
