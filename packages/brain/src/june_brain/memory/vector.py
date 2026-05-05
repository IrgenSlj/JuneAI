"""Semantic memory — ChromaDB with a local sentence-transformer embedder.

ChromaDB runs in embedded mode (no server process) and persists to
``MEMORY_DIR/chroma``. Embeddings are generated locally with
``all-MiniLM-L6-v2`` (≈90 MB, downloaded once and cached). The model
loads lazily so startup stays fast — first recall/extract pays the
model-load cost; subsequent operations don't.

Per ADR 0004, this is the "find me memories like this" layer. SQLite
answers "what goals do I have"; the vector store answers "what did I
say that *feels* like this new message."

Note: we keep a shadow copy of each fact in SQLite's ``semantic_facts``
table. That way, deletes and listing work even if ChromaDB is rebuilt
from scratch, and the memory browser has a single authoritative view.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .sqlite import _current_memory_dir, _get_connection

logger = logging.getLogger(__name__)

# Module-level singletons. The ChromaDB client is expensive to create
# and safe to share across users; collections scope by user_id via the
# collection name.
_client: Any = None
_embedding_function: Any = None


def _db_path() -> str:
    return str(Path(_current_memory_dir()) / "june.db")


def _chroma_dir() -> Path:
    path = Path(_current_memory_dir()) / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_client() -> Any:
    """Lazy-load the ChromaDB persistent client."""
    global _client
    if _client is None:
        import chromadb  # heavy import — deferred
        from chromadb.config import Settings

        _client = chromadb.PersistentClient(
            path=str(_chroma_dir()),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_embedding_function() -> Any:
    """Lazy-load the sentence-transformer embedder."""
    global _embedding_function
    if _embedding_function is None:
        from chromadb.utils import embedding_functions  # heavy import — deferred

        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )
    return _embedding_function


def _collection_name(user_id: str) -> str:
    """Chroma collection names must be 3-63 chars, alnum/underscore/hyphen.

    We namespace per user, slugged, with a prefix so the collection can
    be identified in mixed environments.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in user_id.strip().lower())
    safe = safe.strip("-_") or "default"
    return f"june-{safe}"[:63]


def reset_singletons() -> None:
    """Drop in-process singletons. Tests use this between temp dirs."""
    global _client, _embedding_function
    _client = None
    _embedding_function = None


class VectorStore:
    """Semantic recall backed by ChromaDB + a local embedder, per user."""

    def __init__(self, user_id: str, embedding_function: Any | None = None) -> None:
        self.user_id = user_id
        self._injected_embedder = embedding_function

    @property
    def _collection(self) -> Any:
        embedder = self._injected_embedder or _get_embedding_function()
        return _get_client().get_or_create_collection(
            name=_collection_name(self.user_id),
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"},
        )

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
        metadata = {k: _flatten_for_chroma(v) for k, v in (metadata or {}).items()}
        metadata.setdefault("source", source)
        self._collection.upsert(
            ids=[fact_id],
            documents=[text],
            metadatas=[metadata],
        )
        self._write_shadow(fact_id, text, source, metadata)
        return {"fact_id": fact_id, "text": text, "metadata": metadata, "source": source}

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
            result = self._collection.query(query_texts=[query], n_results=max(1, k))
        except Exception as exc:  # noqa: BLE001
            logger.warning("vector search failed for user=%s: %s", self.user_id, exc)
            return []

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        hits = []
        for fact_id, text, meta, dist in zip(ids, docs, metas, dists):
            hits.append(
                {
                    "fact_id": fact_id,
                    "text": text,
                    "metadata": meta or {},
                    "distance": float(dist) if dist is not None else None,
                }
            )
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

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------

    def delete(self, fact_id: str) -> None:
        """Remove a fact from both Chroma and the shadow table."""
        try:
            self._collection.delete(ids=[fact_id])
        except Exception as exc:  # noqa: BLE001
            logger.warning("vector delete failed for user=%s id=%s: %s", self.user_id, fact_id, exc)
        _get_connection(_db_path()).execute(
            "DELETE FROM semantic_facts WHERE user_id=? AND fact_id=?",
            (self.user_id, fact_id),
        )
        _get_connection(_db_path()).commit()


def _flatten_for_chroma(value: Any) -> Any:
    """Chroma metadata values must be str/int/float/bool. Coerce everything else."""
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
