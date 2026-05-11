"""Unit tests for the VectorStore.

We use a deterministic fake embedding function so the tests never hit
sentence-transformers (and never download the 90 MB model). The
production embedder lives behind the same chromadb API, so covering the
Chroma surface with a stub embedder is enough here. A separate
integration smoke test exercises the real embedder.
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

from june_brain.memory import VectorStore, vector as vector_module


from chromadb.api.types import EmbeddingFunction


class _HashEmbedder(EmbeddingFunction):
    """Deterministic 64-dim embedder: hash → unit vector.

    Good enough for tests: identical text yields identical vectors,
    different text yields different vectors. Chroma queries return the
    matching document when query text equals the stored text.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input):  # chroma protocol
        # chromadb 1.5+ sometimes passes a single string to embed_query
        # which defaults to __call__, so accept both shapes.
        if isinstance(input, str):
            texts = [input]
            single = True
        else:
            texts = list(input)
            single = False
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [(digest[i % len(digest)] / 255.0) * 2.0 - 1.0 for i in range(64)]
            vectors.append(vec)
        return vectors[0] if single else vectors

    @staticmethod
    def name():
        return "test-hash-embedder"

    @staticmethod
    def build_from_config(_config):
        return _HashEmbedder()

    def get_config(self):
        return {}


@pytest.fixture
def memory_dir(tmp_path):
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        vector_module.reset_singletons()
        yield tmp_path
        vector_module.reset_singletons()


@pytest.fixture
def store(memory_dir):
    from june_brain.memory import Memory
    Memory("test_user")  # ensure semantic_facts table exists
    return VectorStore("test_user", embedding_function=_HashEmbedder())


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


def test_collection_name_uses_hash_to_avoid_slug_collisions():
    name_a = vector_module._collection_name("User@example.com")
    name_b = vector_module._collection_name("User example com")

    assert name_a != name_b
    assert name_a.startswith("june-user-example-com-")
    assert name_b.startswith("june-user-example-com-")
    assert len(name_a) <= 63
    assert len(name_b) <= 63
