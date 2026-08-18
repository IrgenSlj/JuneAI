#!/usr/bin/env bash
# Bootstrap the June workspace from a fresh clone.
#
# This installs the Python workspace into packages/brain/.venv and runs
# pnpm install when node_modules is missing. It does not verify Ollama/Gemini;
# use tools/preflight.sh for provider checks.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${JUNE_VENV:-packages/brain/.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: Python not found: $PYTHON_BIN" >&2
  echo "       install Python 3.10+ or set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 13):
    raise SystemExit(
        f"error: Python 3.13+ is required; found {sys.version.split()[0]}"
    )
PY

echo "==> Python bootstrap"
echo "    python : $("$PYTHON_BIN" --version 2>&1)"
echo "    venv   : $VENV"

if [ ! -d "$VENV" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install -q --upgrade pip

echo "==> Installing Python workspace packages"
"$VENV/bin/python" -m pip install -q -e "packages/brain[dev]"
"$VENV/bin/python" -m pip install -q -e "packages/api[dev]"
"$VENV/bin/python" -m pip install -q \
  -e "skills/calendar" \
  -e "skills/files" \
  -e "skills/research"

if [ "${JUNE_SKIP_PNPM_INSTALL:-}" = "1" ] || [ "${JUNE_SKIP_PNPM_INSTALL:-}" = "true" ]; then
  echo "==> pnpm install skipped via JUNE_SKIP_PNPM_INSTALL=${JUNE_SKIP_PNPM_INSTALL}"
elif [ -d node_modules ]; then
  echo "==> pnpm install skipped; node_modules exists"
else
  if ! command -v pnpm >/dev/null 2>&1; then
    echo "error: pnpm is required for frontend dependencies." >&2
    echo "       install Node.js 20+ and enable pnpm, then re-run this script." >&2
    exit 1
  fi
  echo "==> Installing frontend workspace packages"
  pnpm install
fi

echo "==> Bootstrap complete"
