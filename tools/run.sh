#!/usr/bin/env bash
# One-command launcher for local June development.
#
# Starts everything June needs and tears it down on Ctrl-C:
#   1. Ollama (started if not already running; only for the local gemma provider)
#   2. The default Gemma model (pulled on first run)
#   3. The API (uvicorn, june_api)
#   4. The web dev server (SvelteKit)
#
# For health checks WITHOUT starting anything, use tools/dev.sh instead.
#
# Run from anywhere: ./tools/run.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PROVIDER="${MODEL_PROVIDER:-gemma}"
GEMMA_TAG="${GEMMA_MODEL:-gemma4:e2b}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
OLLAMA_HOST="${OLLAMA_URL%/v1}"
API_HOST="${JUNE_API_HOST:-127.0.0.1}"
API_PORT="${JUNE_API_PORT:-8000}"
VENV="${JUNE_VENV:-packages/brain/.venv}"
PYTHON_BIN="$REPO_ROOT/$VENV/bin/python"

PIDS=()
STARTED_OLLAMA=""

cleanup() {
  echo
  echo "==> Stopping June..."
  for pid in "${PIDS[@]:-}"; do
    [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null || true
  done
  # Only stop Ollama if this script started it.
  [ -n "$STARTED_OLLAMA" ] && kill "$STARTED_OLLAMA" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ ! -x "$PYTHON_BIN" ]; then
  echo "error: no venv python at $PYTHON_BIN — run ./tools/bootstrap.sh first." >&2
  exit 1
fi

# --- Ollama (local provider only) -------------------------------------------
if [ "$PROVIDER" = "gemma" ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "error: ollama not installed — 'brew install ollama' (macOS), then re-run." >&2
    exit 1
  fi
  if ! curl -sSf "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    echo "==> Starting ollama serve..."
    ollama serve >/tmp/june-ollama.log 2>&1 &
    STARTED_OLLAMA=$!
    for _ in $(seq 1 30); do
      curl -sSf "$OLLAMA_HOST/api/tags" >/dev/null 2>&1 && break
      sleep 0.5
    done
  fi
  if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$GEMMA_TAG"; then
    echo "==> Pulling $GEMMA_TAG (first run only, this can take a few minutes)..."
    ollama pull "$GEMMA_TAG"
  fi
  echo "==> Ollama ready ($GEMMA_TAG)"
fi

# --- API --------------------------------------------------------------------
echo "==> Starting API on http://$API_HOST:$API_PORT"
( cd packages/api && exec "$PYTHON_BIN" -m june_api ) &
PIDS+=("$!")

# --- Web --------------------------------------------------------------------
echo "==> Starting web dev server..."
pnpm --filter @june/web dev &
PIDS+=("$!")

echo
echo "==> June is up.  Web: http://localhost:5173    API: http://$API_HOST:$API_PORT"
echo "    Press Ctrl-C to stop everything."
wait
