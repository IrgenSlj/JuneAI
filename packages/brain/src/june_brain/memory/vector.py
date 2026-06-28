"""Semantic memory — sqlite-vec index with Ollama-served embeddings (ADR 0019).

The vector index now lives in the *same* SQLite database as everything else: a
``vec0`` virtual table (see ``vec_index``) holds the embeddings, and the query
embedding comes from a local Ollama model via ``EmbeddingService`` (default
``nomic-embed-text``). This replaces ChromaDB + the in-process
sentence-transformer, so the whole data directory is one coherent, copyable
file and the install drops the heavy torch/transformers tree.

Per ADR 0004, this is the "find me memories like this" layer. SQLite answers
"what goals do I have"; the vector store answers "what did I say that *feels*
like this new message."

The ``semantic_facts`` SQLite table remains the authoritative shadow copy of
each fact (text, source, metadata). The vec0 index is rebuildable from it, so
deletes and listing work even if the index is dropped, and the memory browser
has a single authoritative view. Graceful degradation (invariant 6): when no
local embedder is reachable, ``upsert`` still writes the shadow and ``search``
returns ``[]`` — structured/keyword recall keeps working.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from . import vec_index
from .embeddings import EmbeddingService, default_embed_model
from .sqlite import _get_connection, db_path

logger = logging.getLogger(__name__)

# Module-level default embedder. Reset between temp data dirs in tests.
_default_embedder: EmbeddingService | None = None


def _db_path() -> str:
    return db_path()


def _get_default_embedder() -> EmbeddingService:
    """Lazy shared EmbeddingService bound to the current memory connection."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = EmbeddingService(
            _get_connection(_db_path()), model=default_embed_model()
        )
    return _default_embedder


def reset_singletons() -> None:
    """Drop in-process singletons. Tests use this between temp dirs."""
    global _default_embedder
    _default_embedder = None
    vec_index.reset_availability()


class VectorStore:
    """Semantic recall backed by a sqlite-vec vec0 index + a local embedder."""

    def __init__(self, user_id: str, embedder: Any | None = None) -> None:
        self.user_id = user_id
        self._injected_embedder = embedder

    def _embedder(self) -> Any:
        if self._injected_embedder is not None:
            return self._injected_embedder
        return _get_default_embedder()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        fact_id: str | None = None,
        source: str = "conversation",
    ) -> dict[str, Any]:
        """Store a fact. Returns the stored record (id + metadata)."""
        text = text.strip()
        if not text:
            raise ValueError("vector upsert requires non-empty text")
        fact_id = fact_id or uuid.uuid4().hex
        metadata = {k: _coerce_metadata_value(v) for k, v in (metadata or {}).items()}
        metadata.setdefault("source", source)
        # The shadow row is authoritative and is written first; the vec0 index is
        # a rebuildable convenience, so an embedding failure must not lose data.
        self._write_shadow(fact_id, text, source, metadata)
        self._index(fact_id, text)
        return {"fact_id": fact_id, "text": text, "metadata": metadata, "source": source}

    def _index(self, fact_id: str, text: str) -> None:
        try:
            vec = self._embedder().embed_one(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding skipped (unavailable) for user=%s: %s", self.user_id, exc)
            return
        if vec is None:
            return
        vec_index.upsert(_get_connection(_db_path()), fact_id, vec)

    def _write_shadow(
        self,
        fact_id: str,
        text: str,
        source: str,
        metadata: dict[str, Any],
    ) -> None:
        now = datetime.now().isoformat()
        _get_connection(_db_path()).execute(
            """INSERT INTO semantic_facts (user_id, fact_id, text, source, metadata, created_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(user_id, fact_id) DO UPDATE SET
                 text=excluded.text, source=excluded.source,
                 metadata=excluded.metadata, created_at=excluded.created_at""",
            (self.user_id, fact_id, text, source, json.dumps(metadata), now),
        )
        _get_connection(_db_path()).commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k most semantically similar facts for ``query``."""
        query = query.strip()
        if not query:
            return []
        try:
            query_vec = self._embedder().embed_one(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("vector search embed failed for user=%s: %s", self.user_id, exc)
            return []
        if query_vec is None:
            return []

        conn = _get_connection(_db_path())
        # Over-fetch then filter to this user via the authoritative shadow rows.
        # The data dir is single-user in practice, so this filters nothing while
        # staying correct if multiple users share a database in tests.
        candidates = vec_index.search(conn, query_vec, max(k * 5, k + 20))
        if not candidates:
            return []

        # One batched lookup of the shadow rows (keyed by fact_id), instead of a
        # SELECT per candidate. We then walk the candidates in distance order.
        ids = [fact_id for fact_id, _distance in candidates]
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            "SELECT fact_id, text, metadata FROM semantic_facts "
            f"WHERE user_id=? AND fact_id IN ({placeholders})",
            (self.user_id, *ids),
        ).fetchall()
        by_id = {row["fact_id"]: row for row in rows}

        hits: list[dict[str, Any]] = []
        for fact_id, distance in candidates:
            row = by_id.get(fact_id)
            if row is None:
                continue
            hits.append(
                {
                    "fact_id": fact_id,
                    "text": row["text"],
                    "metadata": _loads(row["metadata"]),
                    "distance": float(distance),
                }
            )
            if len(hits) >= k:
                break
        return hits

    def list_facts(
        self, limit: int = 50, source_prefix: str = ""
    ) -> list[dict[str, Any]]:
        """List facts from the shadow table (authoritative view).

        ``source_prefix`` filters by source tag — pass ``"skill:daily:"`` to
        get only writes from the daily skill, ``"skill:"`` for any skill,
        or empty for everything.
        """
        if source_prefix:
            rows = _get_connection(_db_path()).execute(
                "SELECT fact_id, text, source, metadata, created_at FROM semantic_facts "
                "WHERE user_id=? AND source LIKE ? ORDER BY created_at DESC LIMIT ?",
                (self.user_id, f"{source_prefix}%", limit),
            ).fetchall()
        else:
            rows = _get_connection(_db_path()).execute(
                "SELECT fact_id, text, source, metadata, created_at FROM semantic_facts "
                "WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (self.user_id, limit),
            ).fetchall()
        return [
            {
                "fact_id": r["fact_id"],
                "text": r["text"],
                "source": r["source"],
                "metadata": _loads(r["metadata"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def get(self, fact_id: str) -> dict[str, Any] | None:
        row = _get_connection(_db_path()).execute(
            "SELECT fact_id, text, source, metadata, created_at FROM semantic_facts "
            "WHERE user_id=? AND fact_id=?",
            (self.user_id, fact_id),
        ).fetchone()
        if not row:
            return None
        return {
            "fact_id": row["fact_id"],
            "text": row["text"],
            "source": row["source"],
            "metadata": _loads(row["metadata"]),
            "created_at": row["created_at"],
        }

    def fact_ids_for_ref(self, ref: str) -> list[str]:
        """Return semantic fact IDs whose metadata points at a structured ref."""
        rows = _get_connection(_db_path()).execute(
            "SELECT fact_id, metadata FROM semantic_facts WHERE user_id=?",
            (self.user_id,),
        ).fetchall()
        return [
            r["fact_id"]
            for r in rows
            if str(_loads(r["metadata"]).get("ref", "")) == ref
        ]

    def fact_ids_for_ref_prefix(self, ref_prefix: str) -> list[str]:
        """Return semantic fact IDs whose structured ref starts with a prefix."""
        rows = _get_connection(_db_path()).execute(
            "SELECT fact_id, metadata FROM semantic_facts WHERE user_id=?",
            (self.user_id,),
        ).fetchall()
        return [
            r["fact_id"]
            for r in rows
            if str(_loads(r["metadata"]).get("ref", "")).startswith(ref_prefix)
        ]

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------

    def delete(self, fact_id: str) -> None:
        """Remove a fact from both the vec0 index and the shadow table."""
        vec_index.delete(_get_connection(_db_path()), fact_id)
        _get_connection(_db_path()).execute(
            "DELETE FROM semantic_facts WHERE user_id=? AND fact_id=?",
            (self.user_id, fact_id),
        )
        _get_connection(_db_path()).commit()

    # ------------------------------------------------------------------
    # Reversible forget — forgetting is conservative and reversible (vision /
    # CLAUDE.md). Snapshot the fact into the trash table BEFORE deleting it from
    # the live store, so recall never sees forgotten data yet the user can undo.
    # ------------------------------------------------------------------

    def forget(self, fact_id: str) -> bool:
        """Archive a fact to the trash, then delete it from the live store.

        Returns False when the fact doesn't exist (nothing to forget).
        """
        record = self.get(fact_id)
        if record is None:
            return False
        conn = _get_connection(_db_path())
        conn.execute(
            """INSERT INTO forgotten_facts
                 (user_id, fact_id, text, source, metadata, created_at, forgotten_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(user_id, fact_id) DO UPDATE SET
                 text=excluded.text, source=excluded.source,
                 metadata=excluded.metadata, created_at=excluded.created_at,
                 forgotten_at=excluded.forgotten_at""",
            (
                self.user_id,
                record["fact_id"],
                record["text"],
                record["source"],
                json.dumps(record["metadata"]),
                record["created_at"],
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        self.delete(fact_id)
        return True

    def list_forgotten(self, limit: int = 50) -> list[dict[str, Any]]:
        """List trashed facts, most recently forgotten first."""
        rows = _get_connection(_db_path()).execute(
            "SELECT fact_id, text, source, metadata, created_at, forgotten_at "
            "FROM forgotten_facts WHERE user_id=? ORDER BY forgotten_at DESC LIMIT ?",
            (self.user_id, limit),
        ).fetchall()
        return [
            {
                "fact_id": r["fact_id"],
                "text": r["text"],
                "source": r["source"],
                "metadata": _loads(r["metadata"]),
                "created_at": r["created_at"],
                "forgotten_at": r["forgotten_at"],
            }
            for r in rows
        ]

    def restore(self, fact_id: str) -> dict[str, Any] | None:
        """Bring a trashed fact back to the live store with its original id.

        Returns the restored record, or None when no trashed fact matches.
        """
        row = _get_connection(_db_path()).execute(
            "SELECT fact_id, text, source, metadata, created_at "
            "FROM forgotten_facts WHERE user_id=? AND fact_id=?",
            (self.user_id, fact_id),
        ).fetchone()
        if not row:
            return None
        restored = self.upsert(
            row["text"],
            metadata=_loads(row["metadata"]),
            fact_id=row["fact_id"],
            source=row["source"],
        )
        _get_connection(_db_path()).execute(
            "DELETE FROM forgotten_facts WHERE user_id=? AND fact_id=?",
            (self.user_id, fact_id),
        )
        _get_connection(_db_path()).commit()
        return restored

    def delete_by_ref(self, ref: str) -> int:
        """Remove every semantic paraphrase attached to a structured ref."""
        fact_ids = self.fact_ids_for_ref(ref)
        for fact_id in fact_ids:
            self.delete(fact_id)
        return len(fact_ids)

    def delete_by_ref_prefix(self, ref_prefix: str) -> int:
        """Remove every semantic paraphrase attached to refs with a prefix."""
        fact_ids = self.fact_ids_for_ref_prefix(ref_prefix)
        for fact_id in fact_ids:
            self.delete(fact_id)
        return len(fact_ids)


def _coerce_metadata_value(value: Any) -> Any:
    """Keep metadata values JSON-friendly and stable in the shadow row.

    Scalars pass through; sequences become comma-joined strings and dicts become
    JSON, matching the prior representation so existing rows round-trip.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
