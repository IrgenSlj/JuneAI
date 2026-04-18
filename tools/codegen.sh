#!/usr/bin/env bash
# codegen.sh — single source of truth for frontend API types.
#
# 1. Dumps the FastAPI OpenAPI spec to packages/api/openapi.json
# 2. Runs openapi-typescript to emit packages/ui/src/api/types.ts
#
# Why this exists: the Next.js app, Tauri desktop shell, and any other
# shell all call the same Python API. Rather than hand-writing TS types
# that drift from Pydantic, we generate them on demand so a backend
# schema change surfaces as a frontend type error on the next build.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OPENAPI_JSON="$REPO_ROOT/packages/api/openapi.json"
TS_OUT="$REPO_ROOT/packages/ui/src/api/types.ts"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/packages/brain/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: no usable python at $PYTHON_BIN" >&2
  echo "       run ./tools/dev.sh once to bootstrap the venv." >&2
  exit 1
fi

echo "[codegen] dumping OpenAPI spec from june_api.app"
"$PYTHON_BIN" - <<'PY' > "$OPENAPI_JSON"
import json
from june_api.app import app
print(json.dumps(app.openapi(), indent=2))
PY
echo "[codegen] wrote $OPENAPI_JSON"

mkdir -p "$(dirname "$TS_OUT")"

if command -v pnpm >/dev/null 2>&1; then
  RUNNER="pnpm dlx openapi-typescript@^7"
elif command -v npx >/dev/null 2>&1; then
  RUNNER="npx --yes openapi-typescript@^7"
else
  echo "error: neither pnpm nor npx on PATH; install Node.js to generate TS types." >&2
  exit 1
fi

echo "[codegen] generating TS types via $RUNNER"
$RUNNER "$OPENAPI_JSON" --output "$TS_OUT"
echo "[codegen] wrote $TS_OUT"
