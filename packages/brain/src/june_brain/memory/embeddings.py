"""Embedding service — Ollama-served embeddings with a SQLite hash cache (ADR 0019).

Replaces the in-process sentence-transformer. Embeddings come from a local
Ollama model (default ``nomic-embed-text``) via the provider registry, and are
cached in the ``embedding_cache`` table keyed by ``(model, sha256(text))`` so
re-embedding the same text is free.

Graceful degradation (invariant 6): if no local embedding provider is
reachable, or the model is not pulled, ``embed`` returns ``None`` and the
VectorStore falls back to the SQLite keyword scan. Nothing here ever triggers a
model download — pulling the embedding model is the job of ``run.sh`` / the
managed-Ollama path (S8), so an uncached model degrades rather than egressing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
_EMBED_PROVIDER_ROLES = ("local-fast", "local-deep")


def default_embed_model() -> str:
    """The embedding model name, overridable via ``JUNE_EMBED_MODEL``."""
    return os.environ.get("JUNE_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL


def _serialize(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"{dim}f", blob))


class EmbeddingService:
    """Cache-backed embedder over a local Ollama model.

    ``conn`` is the SQLite connection holding the ``embedding_cache`` table.
    ``provider_embed`` is an async callable ``(texts, model) -> vectors`` used
    instead of the registry — tests inject a deterministic one so they never
    need a running Ollama.
    """

    def __init__(
        self,
        conn: Any,
        *,
        model: str | None = None,
        provider_embed: Any | None = None,
    ) -> None:
        self._conn = conn
        self.model = model or default_embed_model()
        self._provider_embed = provider_embed

    # -- public ---------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Return one vector per input text, or None if embedding is unavailable.

        Cache hits are served from SQLite; misses are embedded in one provider
        call and written back. If any miss cannot be embedded (no provider /
        model not pulled), the whole call returns None so the caller degrades
        consistently rather than recalling on a partial index.
        """
        items = list(texts)
        if not items:
            return []
        result: list[list[float] | None] = [None] * len(items)
        miss_idx: list[int] = []
        miss_text: list[str] = []
        for i, text in enumerate(items):
            cached = self._cache_get(text)
            if cached is not None:
                result[i] = cached
            else:
                miss_idx.append(i)
                miss_text.append(text)

        if miss_text:
            vectors = self._embed_uncached(miss_text)
            if vectors is None or len(vectors) != len(miss_text):
                return None
            for j, vec in enumerate(vectors):
                vec_list = [float(x) for x in vec]
                result[miss_idx[j]] = vec_list
                self._cache_put(miss_text[j], vec_list)

        return [v for v in result if v is not None]

    def embed_one(self, text: str) -> list[float] | None:
        """Embed a single text; None when embedding is unavailable."""
        vectors = self.embed([text])
        if not vectors:
            return None
        return vectors[0]

    # -- cache ----------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        return sha256(f"{self.model}\x00{text}".encode()).hexdigest()

    def _cache_get(self, text: str) -> list[float] | None:
        try:
            row = self._conn.execute(
                "SELECT vector, dim FROM embedding_cache WHERE model=? AND text_hash=?",
                (self.model, self._cache_key(text)),
            ).fetchone()
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        blob, dim = row[0], int(row[1])
        try:
            return _deserialize(blob, dim)
        except Exception:  # noqa: BLE001
            return None

    def _cache_put(self, text: str, vec: list[float]) -> None:
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (model, text_hash, dim, vector, created_at) "
                "VALUES (?,?,?,?,?)",
                (
                    self.model,
                    self._cache_key(text),
                    len(vec),
                    _serialize(vec),
                    datetime.now().isoformat(),
                ),
            )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.debug("embedding cache write skipped (%s)", exc)

    # -- provider -------------------------------------------------------

    def _embed_uncached(self, texts: list[str]) -> list[list[float]] | None:
        embed_call = self._provider_embed or self._registry_embed
        try:
            return _run_async_blocking(lambda: embed_call(texts, self.model))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "embedding provider unavailable (%s); semantic recall degrades "
                "to the SQLite keyword scan.",
                exc,
            )
            return None

    async def _registry_embed(self, texts: list[str], model: str) -> list[list[float]]:
        from ..providers.registry import get_registry

        registry = get_registry()
        errors: list[str] = []
        for role in _EMBED_PROVIDER_ROLES:
            try:
                provider = registry.get(role)
            except KeyError as exc:
                errors.append(f"{role}: {exc}")
                continue
            embed = getattr(provider, "embed", None)
            if embed is None:
                errors.append(f"{role}: provider has no embed()")
                continue
            return await embed(texts, model=model)
        raise RuntimeError(
            "no local embedding provider available: " + ("; ".join(errors) or "none configured")
        )


def _run_async_blocking(coro_factory: Any) -> Any:
    """Run an async embed call from the synchronous VectorStore path."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro_factory())).result()
