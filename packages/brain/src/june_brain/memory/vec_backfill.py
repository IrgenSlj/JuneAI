"""Migration helpers: ChromaDB -> sqlite-vec (ADR 0019).

The semantic_facts shadow table is the authoritative copy of every fact, so the
migration does not read ChromaDB at all — it re-embeds the shadow rows into the
new vec0 index (the old Chroma embeddings used a different model/dimension and
are not reusable) and archives the chroma directory to ``chroma.bak``.

Everything here is idempotent and degradation-safe (invariant 6): if the local
embedder is unavailable, the backfill embeds what it can (nothing) and returns —
the shadow rows are untouched and a later run completes the work. New writes are
indexed lazily on upsert regardless, so recall is never blocked on the backfill.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import vec_index
from .sqlite import _current_memory_dir, _get_connection, db_path

logger = logging.getLogger(__name__)

_BATCH = 64


def archive_chroma_dir() -> bool:
    """Rename ``<memory>/chroma`` to ``chroma.bak`` if present. Returns whether moved."""
    memory_dir = Path(_current_memory_dir())
    chroma = memory_dir / "chroma"
    backup = memory_dir / "chroma.bak"
    if not chroma.exists() or backup.exists():
        return False
    try:
        chroma.rename(backup)
        logger.info("archived legacy ChromaDB store to %s", backup)
        return True
    except OSError as exc:
        logger.warning("could not archive chroma dir (%s)", exc)
        return False


def backfill_vectors(conn: Any, embedder: Any, *, batch: int = _BATCH) -> dict[str, int]:
    """Re-embed shadow rows missing from the vec0 index. Idempotent.

    Returns ``{"total", "embedded", "remaining"}`` where ``total`` is the count
    of shadow rows lacking a vector at the start. Stops early (and leaves the
    rest for a later run) if the embedder becomes unavailable.
    """
    pending = _rows_missing_vectors(conn)
    total = len(pending)
    embedded = 0
    for start in range(0, total, batch):
        chunk = pending[start : start + batch]
        texts = [text for _fid, text in chunk]
        try:
            vectors = embedder.embed(texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("backfill stopped — embedder unavailable (%s)", exc)
            break
        if vectors is None or len(vectors) != len(chunk):
            logger.warning("backfill stopped — embedder returned no vectors")
            break
        for (fact_id, _text), vec in zip(chunk, vectors):
            if vec_index.upsert(conn, fact_id, vec):
                embedded += 1
    remaining = total - embedded
    if total:
        logger.info("vec backfill: %s/%s embedded (%s remaining)", embedded, total, remaining)
    return {"total": total, "embedded": embedded, "remaining": remaining}


def _rows_missing_vectors(conn: Any) -> list[tuple[str, str]]:
    """(fact_id, text) for shadow rows that have no vec0 entry yet."""
    try:
        if vec_index.is_available() is False:
            return []
        if _vec_table_present(conn):
            rows = conn.execute(
                "SELECT s.fact_id, s.text FROM semantic_facts s "
                "WHERE s.fact_id NOT IN (SELECT fact_id FROM semantic_vectors)"
            ).fetchall()
        else:
            rows = conn.execute("SELECT fact_id, text FROM semantic_facts").fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("backfill scan failed (%s)", exc)
        return []
    return [(str(r[0]), str(r[1])) for r in rows]


def _vec_table_present(conn: Any) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='semantic_vectors'"
    ).fetchone()
    return row is not None


def maybe_backfill_on_start() -> dict[str, int]:
    """Best-effort startup backfill: re-embed any shadow rows missing a vector.

    No-op when nothing is missing or the embedder is unavailable. Never raises —
    a failed backfill must not block API startup.
    """
    try:
        from .vector import _get_default_embedder

        conn = _get_connection(db_path())
        shadow_count = conn.execute("SELECT count(*) FROM semantic_facts").fetchone()[0]
        if not shadow_count:
            return {"total": 0, "embedded": 0, "remaining": 0}
        if vec_index.count(conn) >= shadow_count:
            return {"total": 0, "embedded": 0, "remaining": 0}
        return backfill_vectors(conn, _get_default_embedder())
    except Exception as exc:  # noqa: BLE001
        logger.warning("startup vec backfill skipped (%s)", exc)
        return {"total": 0, "embedded": 0, "remaining": 0}
