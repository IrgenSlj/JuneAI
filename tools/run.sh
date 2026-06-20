#!/usr/bin/env bash
# One-command launcher for local June development.
#
# Starts everything June needs and tears it down on Ctrl-C:
#   1. Ollama (started if not already running; only for the local gemma provider)
#   2. The default Gemma model (pulled on first run)
#   3. The API (uvicorn, june_api)
#   4. The web dev server (SvelteKit)
#
# For health checks WITHOUT starting anything, use tools/preflight.sh instead.
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
WEB_PORT="${JUNE_WEB_PORT:-5173}"
VENV="${JUNE_VENV:-packages/brain/.venv}"
PYTHON_BIN="$REPO_ROOT/$VENV/bin/python"

PIDS=()
STARTED_OLLAMA=""
DIED=0

cleanup() {
  local sig="${1:-EXIT}"
  echo
  echo "==> Stopping June..."
  for pid in "${PIDS[@]:-}"; do
    [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null || true
  done
  # Only stop Ollama if this script started it.
  [ -n "$STARTED_OLLAMA" ] && kill "$STARTED_OLLAMA" 2>/dev/null || true
  # Only re-raise the signal on non-EXIT so we don't loop.
  if [ "$sig" != "EXIT" ]; then
    trap - "$sig"
    kill -s "$sig" "$$" 2>/dev/null || true
  fi
}
trap 'cleanup INT' INT
trap 'cleanup TERM' TERM
trap 'cleanup EXIT' EXIT

if [ ! -x "$PYTHON_BIN" ]; then
  echo "error: no venv python at $PYTHON_BIN — run ./tools/bootstrap.sh first." >&2
  exit 1
fi

# --- Port conflict helper ----------------------------------------------------
port_in_use() {
  local host="$1" port="$2"
  nc -z "$host" "$port" 2>/dev/null
}

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
if port_in_use "$API_HOST" "$API_PORT"; then
  echo "error: port $API_PORT is already in use (a prior June or another process)." >&2
  echo "       Kill it with: kill \$(lsof -ti :$API_PORT)" >&2
  exit 1
fi
echo "==> Starting API on http://$API_HOST:$API_PORT"
( cd packages/api && exec "$PYTHON_BIN" -m june_api ) &
API_PID=$!
PIDS+=("$API_PID")

# Wait for the API to respond (up to 15 s).
for _ in $(seq 1 30); do
  if curl -sSf "http://$API_HOST:$API_PORT/healthz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "error: API process exited unexpectedly — check stderr above." >&2
    DIED=1
    break
  fi
  sleep 0.5
done
if [ "$DIED" = 1 ]; then
  exit 1
fi
echo "==> API ready"

# --- Web --------------------------------------------------------------------
echo "==> Starting web dev server on http://localhost:$WEB_PORT ..."
pnpm --filter @june/web dev --port "$WEB_PORT" &
PIDS+=("$!")

echo
echo "==> June is up.  Web: http://localhost:$WEB_PORT    API: http://$API_HOST:$API_PORT"
echo "    Press Ctrl-C to stop everything."
wait
