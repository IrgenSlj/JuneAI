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

# Playwright e2e smokes are OFF by default so the everyday gate stays fast; they
# boot a dev server and a headless browser. Opt in with JUNE_E2E=1 (and CI).
if [ "${JUNE_E2E:-0}" = "1" ]; then
  echo "==> Playwright e2e smokes"
  pnpm --filter @june/web test:e2e
else
  echo "==> Playwright e2e skipped (set JUNE_E2E=1 to run)"
fi

if [ "${JUNE_CHECK_CODEGEN:-1}" = "1" ]; then
  echo "==> OpenAPI codegen drift check"
  PYTHON_BIN="$PYTHON_BIN" ./tools/codegen.sh
  git diff --exit-code packages/api/openapi.json packages/ui/src/api/types.ts
else
  echo "==> OpenAPI codegen skipped via JUNE_CHECK_CODEGEN=0"
fi

echo "==> Ruff lint"
"$PYTHON_BIN" -m ruff check .

if [ "${JUNE_CHECK_MYPY:-1}" = "1" ]; then
  # Narrow mypy gate: the codebase is not fully annotated, so a full strict run
  # is noisy (~270 errors, mostly missing generics). These two codes, however,
  # are the real-bug classes — operator misuse (e.g. str / "x") and missing
  # required call args. They must stay at zero. Intentionally NOT gating
  # arg-type/valid-type/attr-defined: those still carry false positives
  # (LangGraph ToolNode state typing; list-named store methods) until a focused
  # typing-cleanup pass. mypy has no "only these codes" flag, so we grep.
  echo "==> Mypy real-bug gate (operator, call-arg)"
  mypy_out="$("$PYTHON_BIN" -m mypy packages/brain/src packages/api/src --no-error-summary 2>&1 || true)"
  if echo "$mypy_out" | grep -E '\[(operator|call-arg)\]'; then
    echo "error: mypy found real-bug-class errors (operator/call-arg); see above" >&2
    exit 1
  fi
else
  echo "==> Mypy gate skipped via JUNE_CHECK_MYPY=0"
fi

echo "==> Checks complete"
