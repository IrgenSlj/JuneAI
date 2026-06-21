"""sqlite-vec vector index — a vec0 virtual table for semantic recall (ADR 0019).

Replaces ChromaDB's ANN index. Vectors live in the *same* SQLite database as
facts, structure, and the graph, so the whole data directory is one coherent,
copyable file. The ``semantic_facts`` shadow table stays the authoritative
text/metadata store; this module holds only the embeddings and serves KNN.

Everything here is best-effort and degrades gracefully (invariant 6): if the
sqlite-vec extension cannot load on this platform, ``load_extension`` returns
False once, the table is never created, and callers fall back to the SQLite
keyword scan. Distance is cosine to match the prior Chroma behavior.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

_VEC_TABLE = "semantic_vectors"
_META_TABLE = "semantic_vectors_meta"

# Tri-state availability: None = not yet probed, True/False = result cached so
# the "unavailable" warning is logged at most once per process.
_available: bool | None = None


def serialize(vec: Sequence[float]) -> bytes:
    """Pack a float vector into the little-endian float32 blob vec0 expects."""
    return struct.pack(f"{len(vec)}f", *vec)


def is_available() -> bool | None:
    """Return the cached availability tri-state (None until first probe)."""
    return _available


def load_extension(conn: Any) -> bool:
    """Best-effort load of the sqlite-vec extension into ``conn``.

    Returns True if the extension is loaded and usable. On any failure the
    result is cached so we degrade to keyword recall and warn only once.
    """
    global _available
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        _available = True
        return True
    except Exception as exc:  # noqa: BLE001
        if _available is not False:
            logger.warning(
                "sqlite-vec unavailable (%s); semantic recall degrades to the "
                "SQLite keyword scan until it loads.",
                exc,
            )
        _available = False
        return False


def _table_dim(conn: Any) -> int | None:
    """Return the dimension the vec0 table was created with, or None if absent."""
    if not _meta_exists(conn):
        return None
    row = conn.execute(f"SELECT dim FROM {_META_TABLE} WHERE rowid = 1").fetchone()
    return int(row[0]) if row else None


def _meta_exists(conn: Any) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (_META_TABLE,),
    ).fetchone()
    return row is not None


def _vec_table_exists(conn: Any) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name=?",
        (_VEC_TABLE,),
    ).fetchone()
    return row is not None


def ensure_table(conn: Any, dim: int) -> bool:
    """Ensure the vec0 table exists with dimension ``dim``. Returns usability.

    If a table exists at a different dimension (the embedding model changed),
    it is dropped and recreated empty — the caller re-embeds from the
    authoritative shadow rows, so this never loses data it cannot rebuild.
    """
    if _available is False or dim <= 0:
        return False
    if not _vec_table_exists(conn):
        return _create_table(conn, dim)
    existing = _table_dim(conn)
    if existing is not None and existing != dim:
        logger.info(
            "embedding dimension changed (%s -> %s); rebuilding vec index",
            existing,
            dim,
        )
        conn.execute(f"DROP TABLE IF EXISTS {_VEC_TABLE}")
        conn.execute(f"DELETE FROM {_META_TABLE}")
        conn.commit()
        return _create_table(conn, dim)
    return True


def _create_table(conn: Any, dim: int) -> bool:
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {_VEC_TABLE} USING vec0("
            f"fact_id TEXT PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)"
        )
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_META_TABLE} (rowid INTEGER PRIMARY KEY, dim INTEGER NOT NULL)"
        )
        conn.execute(
            f"INSERT OR REPLACE INTO {_META_TABLE} (rowid, dim) VALUES (1, ?)",
            (dim,),
        )
        conn.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not create vec index (%s); keyword recall only", exc)
        return False


def upsert(conn: Any, fact_id: str, vec: Sequence[float]) -> bool:
    """Insert or replace the embedding for ``fact_id``. Returns success."""
    if not ensure_table(conn, len(vec)):
        return False
    try:
        conn.execute(f"DELETE FROM {_VEC_TABLE} WHERE fact_id = ?", (fact_id,))
        conn.execute(
            f"INSERT INTO {_VEC_TABLE} (fact_id, embedding) VALUES (?, ?)",
            (fact_id, serialize(vec)),
        )
        conn.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("vec upsert failed for %s (%s)", fact_id, exc)
        return False


def search(conn: Any, query_vec: Sequence[float], k: int) -> list[tuple[str, float]]:
    """Return up to ``k`` (fact_id, distance) pairs nearest to ``query_vec``.

    Empty list when the index is unavailable or empty — the caller then relies
    on the keyword scan.
    """
    if _available is False or not _vec_table_exists(conn):
        return []
    try:
        rows = conn.execute(
            f"SELECT fact_id, distance FROM {_VEC_TABLE} "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (serialize(query_vec), max(1, k)),
        ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("vec search failed (%s)", exc)
        return []


def delete(conn: Any, fact_id: str) -> None:
    """Remove ``fact_id`` from the index (no-op if the index is absent)."""
    if _available is False or not _vec_table_exists(conn):
        return
    try:
        conn.execute(f"DELETE FROM {_VEC_TABLE} WHERE fact_id = ?", (fact_id,))
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("vec delete failed for %s (%s)", fact_id, exc)


def count(conn: Any) -> int:
    """Return the number of indexed vectors (0 when the index is absent)."""
    if _available is False or not _vec_table_exists(conn):
        return 0
    try:
        row = conn.execute(f"SELECT count(*) FROM {_VEC_TABLE}").fetchone()
        return int(row[0]) if row else 0
    except Exception:  # noqa: BLE001
        return 0


def reset_availability() -> None:
    """Clear the cached availability probe (tests use this between connections)."""
    global _available
    _available = None
