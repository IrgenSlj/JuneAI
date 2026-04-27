#!/usr/bin/env bash
# One-command developer startup for June v2.
#
# Runs three checks so you know the local setup is healthy before you
# start coding:
#   1. Ollama is running and Gemma 4 is pulled (only when MODEL_PROVIDER=gemma).
#   2. Python 3.10+ venv exists at packages/brain/.venv.
#   3. Unit tests pass.
#
# Run from the repo root: ./tools/dev.sh

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
GEMMA_TAG="${GEMMA_MODEL:-gemma4:e4b}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
OLLAMA_HOST="${OLLAMA_URL%/v1}"

echo "==> June dev check"
echo "    provider : $PROVIDER"

if [ "$PROVIDER" = "gemma" ]; then
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
  echo "               Install with 'curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh' when you start Phase 1."
  echo "               See docs/setup/desktop.md."
fi

VENV="packages/brain/.venv"
if [ ! -d "$VENV" ]; then
  echo "    venv     : creating at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -e "packages/brain[dev]"
else
  echo "    venv     : $VENV"
fi

echo "==> Running brain tests"
"$VENV/bin/python" -m pytest packages/brain/tests -q

echo "==> Ready"
