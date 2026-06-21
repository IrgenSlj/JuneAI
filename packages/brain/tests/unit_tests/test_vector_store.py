"""Unit tests for the VectorStore.

We use a deterministic fake embedder so the tests never hit Ollama (and
never download an embedding model). The production embedder is the
EmbeddingService over a local Ollama model; tests inject an object with
the same ``embed`` / ``embed_one`` surface. A separate integration smoke
test exercises the real embedder.
"""

from __future__ import annotations

import hashlib

import pytest
from june_brain.memory import VectorStore
from june_brain.memory import vector as vector_module


class _HashEmbedder:
    """Deterministic 64-dim embedder: hash → vector.

    Good enough for tests: identical text yields identical vectors, and the
    vec0 index returns the matching document when the query equals the stored
    text. Implements the EmbeddingService surface (``embed`` / ``embed_one``).
    """

    @staticmethod
    def _vec(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) * 2.0 - 1.0 for i in range(64)]

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        return self._vec(text)


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("june_brain.memory.MEMORY_DIR", str(tmp_path))
    vector_module.reset_singletons()
    yield tmp_path
    vector_module.reset_singletons()


@pytest.fixture
def store(memory_dir):
    from june_brain.memory import Memory
    Memory("test_user")  # ensure semantic_facts table exists
    return VectorStore("test_user", embedder=_HashEmbedder())


def test_upsert_writes_to_shadow_and_chroma(store):
    record = store.upsert("I love ramen", source="test", metadata={"kind": "fact"})
    assert record["fact_id"]
    listed = store.list_facts()
    assert len(listed) == 1
    assert listed[0]["text"] == "I love ramen"
    assert listed[0]["metadata"]["kind"] == "fact"


def test_search_returns_upserted_fact(store):
    store.upsert("I love ramen")
    store.upsert("I prefer espresso over drip coffee")
    hits = store.search("I love ramen", k=2)
    assert hits
    assert hits[0]["text"] == "I love ramen"


def test_delete_removes_from_both_stores(store):
    record = store.upsert("temporary fact")
    store.delete(record["fact_id"])
    assert store.get(record["fact_id"]) is None
    assert store.list_facts() == []


def test_upsert_requires_text(store):
    with pytest.raises(ValueError):
        store.upsert("   ")
