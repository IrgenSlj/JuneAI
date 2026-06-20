#!/usr/bin/env bash
# DEPRECATED: dev.sh was a check-only preflight, not a runner — the name misled.
# It is now tools/preflight.sh. This shim warns once and forwards so existing
# muscle memory and scripts keep working for one release.
#
# To actually launch Ollama + API + web together, use ./tools/run.sh.

echo "warning: tools/dev.sh is deprecated; use ./tools/preflight.sh (forwarding now)." >&2
exec "$(cd "$(dirname "$0")" && pwd)/preflight.sh" "$@"
