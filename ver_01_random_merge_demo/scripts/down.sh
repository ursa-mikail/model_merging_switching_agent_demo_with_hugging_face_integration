#!/usr/bin/env bash
# down.sh — stop the Splice stack without deleting downloaded models.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "Error: neither 'docker compose' nor 'docker-compose' was found on PATH." >&2
  exit 1
fi

echo "==> Stopping containers (downloaded models are preserved in the model_data volume)..."
"${COMPOSE[@]}" down

echo "✅ Stopped. Run ./scripts/up.sh to start again — previously downloaded models will still be there."
