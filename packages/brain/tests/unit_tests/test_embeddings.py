"""Unit tests for the embedding service (ADR 0019, S2.2).

Covers the cache and degradation paths with a deterministic fake provider
and an in-memory cache table, so nothing here needs a running Ollama.
"""

from __future__ import annotations

import sqlite3

import pytest
from june_brain.memory.embeddings import EmbeddingService, default_embed_model

_EMBED_CACHE_DDL = """
CREATE TABLE embedding_cache (
    model       TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (model, text_hash)
);
"""


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute(_EMBED_CACHE_DDL)
    c.commit()
    yield c
    c.close()


class _FakeProvider:
    """Async embed that maps text -> a deterministic 3-dim vector; counts calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.embedded: list[str] = []

    async def embed(self, texts, model):
        self.calls += 1
        self.embedded.extend(texts)
        return [[float(len(t)), 1.0, 2.0] for t in texts]


def _service(conn, provider, model="test-embed"):
    return EmbeddingService(conn, model=model, provider_embed=provider.embed)


def test_embed_returns_one_vector_per_text(conn):
    p = _FakeProvider()
    svc = _service(conn, p)
    vecs = svc.embed(["hello", "hi"])
    assert vecs == [[5.0, 1.0, 2.0], [2.0, 1.0, 2.0]]
    assert p.calls == 1


def test_cache_hit_skips_provider(conn):
    p = _FakeProvider()
    svc = _service(conn, p)
    svc.embed(["alpha", "beta"])
    assert p.calls == 1
    # Second call for the same texts is served entirely from the cache.
    again = svc.embed(["alpha", "beta"])
    assert again == [[5.0, 1.0, 2.0], [4.0, 1.0, 2.0]]
    assert p.calls == 1
    assert p.embedded == ["alpha", "beta"]  # never re-embedded


def test_partial_cache_only_embeds_misses(conn):
    p = _FakeProvider()
    svc = _service(conn, p)
    svc.embed(["alpha"])
    assert p.calls == 1
    # "alpha" cached, "gamma" is a miss -> provider sees only "gamma".
    out = svc.embed(["alpha", "gamma"])
    assert out == [[5.0, 1.0, 2.0], [5.0, 1.0, 2.0]]
    assert p.calls == 2
    assert p.embedded == ["alpha", "gamma"]


def test_empty_input_returns_empty(conn):
    p = _FakeProvider()
    assert _service(conn, p).embed([]) == []
    assert p.calls == 0


def test_provider_failure_degrades_to_none(conn):
    class _Boom:
        async def embed(self, texts, model):
            raise RuntimeError("ollama down")

    svc = EmbeddingService(conn, model="test-embed", provider_embed=_Boom().embed)
    assert svc.embed(["anything"]) is None


def test_embed_one(conn):
    p = _FakeProvider()
    svc = _service(conn, p)
    assert svc.embed_one("hey") == [3.0, 1.0, 2.0]
    assert svc.embed_one("hey") == [3.0, 1.0, 2.0]
    assert p.calls == 1


def test_default_model_is_nomic(monkeypatch):
    monkeypatch.delenv("JUNE_EMBED_MODEL", raising=False)
    assert default_embed_model() == "nomic-embed-text"
    monkeypatch.setenv("JUNE_EMBED_MODEL", "custom-embed")
    assert default_embed_model() == "custom-embed"
