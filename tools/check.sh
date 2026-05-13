#!/usr/bin/env bash
# Run the project checks that are currently enforced for contributors and CI.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${JUNE_VENV:-packages/brain/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/$VENV/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: no usable python at $PYTHON_BIN" >&2
  echo "       run ./tools/bootstrap.sh first, or set PYTHON_BIN." >&2
  exit 1
fi

export JUNE_DATA_DIR="${JUNE_DATA_DIR:-$REPO_ROOT/.tmp/june-test-data}"
export JUNE_SKIP_MODEL_CHECK="${JUNE_SKIP_MODEL_CHECK:-1}"
export JUNE_SKILLS_DISABLED="${JUNE_SKILLS_DISABLED:-1}"

echo "==> Backend tests"
"$PYTHON_BIN" -m pytest packages/brain/tests packages/api/tests -q

if [ "${JUNE_CHECK_FRONTEND:-1}" = "1" ]; then
  if ! command -v pnpm >/dev/null 2>&1; then
    echo "error: pnpm is required for frontend checks." >&2
    exit 1
  fi
  echo "==> Frontend checks"
  pnpm check
else
  echo "==> Frontend checks skipped via JUNE_CHECK_FRONTEND=0"
fi

if [ "${JUNE_CHECK_CODEGEN:-1}" = "1" ]; then
  echo "==> OpenAPI codegen drift check"
  PYTHON_BIN="$PYTHON_BIN" ./tools/codegen.sh
  git diff --exit-code packages/api/openapi.json packages/ui/src/api/types.ts
else
  echo "==> OpenAPI codegen skipped via JUNE_CHECK_CODEGEN=0"
fi

echo "==> Ruff/mypy gates are tracked in Phase 1.3 and are not enforced by check.sh yet"
echo "==> Checks complete"
