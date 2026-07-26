#!/usr/bin/env python
"""HISTORICAL — one-shot migration: ChromaDB -> sqlite-vec (ADR 0019).

Kept only because ADR 0019 references it by path and ADRs are append-only.
Chroma has not been a June dependency since that ADR landed, so this script is
useful only to someone opening a data directory written before the switch. It is
not part of any current workflow and nothing in check.sh exercises it.

Re-embeds every semantic fact from the authoritative SQLite shadow table into
the new vec0 index, then archives the legacy ``chroma`` directory to
``chroma.bak``. Idempotent and safe to re-run: facts already indexed are
skipped, and nothing is deleted (the shadow rows are the source of truth).

Normally this runs automatically on first start of the new version (the data
dir manifest migration archives chroma; the API startup backfills vectors).
This script is for power users who want to force a full backfill in one go.

Usage:
    packages/brain/.venv/bin/python tools/migrate_chroma_to_sqlitevec.py
"""

from __future__ import annotations

import sys


def main() -> int:
    from june_brain.memory.sqlite import _get_connection, db_path
    from june_brain.memory.vec_backfill import archive_chroma_dir, backfill_vectors
    from june_brain.memory.vector import _get_default_embedder

    conn = _get_connection(db_path())
    summary = backfill_vectors(conn, _get_default_embedder())
    archived = archive_chroma_dir()

    print(
        f"vec backfill: {summary['embedded']}/{summary['total']} embedded, "
        f"{summary['remaining']} remaining"
    )
    print(f"chroma archived: {archived}")
    if summary["remaining"]:
        print(
            "Some facts could not be embedded (is Ollama running with the "
            "embedding model pulled?). Re-run once it is available.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
