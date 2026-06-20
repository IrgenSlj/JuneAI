#!/usr/bin/env bash
# Developer health check for June v2 (does NOT start the app).
#
# To actually launch Ollama + API + web together, use ./tools/run.sh.
#
# Runs three checks so you know the local setup is healthy before you
# start coding:
#   1. Ollama is running and Gemma 4 is pulled (only when MODEL_PROVIDER=gemma,
#      unless JUNE_SKIP_MODEL_CHECK=1 is set).
#   2. tools/bootstrap.sh has installed the Python and JS workspaces.
#   3. Backend tests pass through tools/check.sh.
#
# Run from the repo root: ./tools/preflight.sh

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
SKIP_MODEL_CHECK="${JUNE_SKIP_MODEL_CHECK:-}"

echo "==> June preflight check"
echo "    provider : $PROVIDER"

if [ "$SKIP_MODEL_CHECK" = "1" ] || [ "$SKIP_MODEL_CHECK" = "true" ]; then
  echo "    model    : skipped via JUNE_SKIP_MODEL_CHECK=$SKIP_MODEL_CHECK"
elif [ "$PROVIDER" = "gemma" ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "    ollama   : NOT INSTALLED — install with 'brew install ollama' (macOS) and re-run."
    exit 1
  fi
  if ! curl -sSf "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    echo "    ollama   : NOT RUNNING at $OLLAMA_HOST — start it with 'ollama serve' and re-run."
    exit 1
  fi
  if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$GEMMA_TAG"; then
    echo "    gemma    : NOT PULLED — run 'ollama pull $GEMMA_TAG' and re-run."
    exit 1
  fi
  echo "    ollama   : reachable, $GEMMA_TAG pulled"
fi

if [ "$PROVIDER" = "gemini" ]; then
  if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "    gemini   : GEMINI_API_KEY is unset — add it to .env and re-run."
    exit 1
  fi
  echo "    gemini   : API key present"
fi

if command -v cargo >/dev/null 2>&1; then
  echo "    rust     : $(rustc --version 2>/dev/null | awk '{print $1, $2}')"
else
  echo "    rust     : NOT FOUND — needed for the desktop shell only."
  echo "               Install with 'curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh' before working on desktop."
  echo "               See docs/setup/desktop.md."
fi

./tools/bootstrap.sh

echo "==> Running backend tests"
JUNE_CHECK_FRONTEND=0 JUNE_CHECK_CODEGEN=0 ./tools/check.sh

echo "==> Ready"
