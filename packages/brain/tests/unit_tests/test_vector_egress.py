"""Tests for VectorStore egress-safety and graceful degradation (ADR 0019).

Embeddings now come from a local Ollama model (localhost) rather than an
in-process sentence-transformer, so there is no HF Hub fetch to guard. The
enduring invariant is degradation: when no embedder is reachable, the store
must keep the authoritative SQLite shadow working — upsert writes, search
returns [], delete removes — and must never raise. A None-returning embedder
stands in for "embedding model not pulled / Ollama down"; nothing here ever
triggers a download.
"""

from __future__ import annotations

import pytest
from june_brain.memory import Memory, VectorStore
from june_brain.memory import vector as vector_module


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    monkeypatch.setattr("june_brain.memory.MEMORY_DIR", str(tmp_path))
    vector_module.reset_singletons()
    yield
    vector_module.reset_singletons()


class _UnavailableEmbedder:
    """Stands in for an unreachable embedder (model not pulled / Ollama down)."""

    def embed(self, texts):
        return None

    def embed_one(self, text):
        return None


def test_upsert_degrades_when_embedder_unavailable():
    Memory("degtest")
    store = VectorStore("degtest", embedder=_UnavailableEmbedder())

    # upsert must NOT raise; the fact must land in the SQLite shadow.
    record = store.upsert("some fact")
    assert record["fact_id"]

    facts = store.list_facts()
    assert len(facts) == 1
    assert facts[0]["text"] == "some fact"

    fetched = store.get(record["fact_id"])
    assert fetched is not None and fetched["text"] == "some fact"


def test_search_returns_empty_when_embedder_unavailable():
    Memory("degtest")
    store = VectorStore("degtest", embedder=_UnavailableEmbedder())
    store.upsert("a fact that cannot be embedded")

    # Semantic recall is disabled, but the call degrades to [] (never raises).
    assert store.search("anything") == []


def test_delete_degrades_shadow_removed_without_index():
    Memory("deltest")
    store = VectorStore("deltest", embedder=_UnavailableEmbedder())

    record = store.upsert("fact to delete")
    fact_id = record["fact_id"]
    assert store.get(fact_id) is not None

    # delete must NOT raise even though no vector was ever indexed.
    store.delete(fact_id)
    assert store.get(fact_id) is None


def test_embedder_failure_is_caught_on_upsert_and_search():
    class _Boom:
        def embed(self, texts):
            raise RuntimeError("ollama exploded")

        def embed_one(self, text):
            raise RuntimeError("ollama exploded")

    Memory("boomtest")
    store = VectorStore("boomtest", embedder=_Boom())

    # A raising embedder must not bubble up — shadow write still succeeds.
    record = store.upsert("resilient fact")
    assert store.get(record["fact_id"]) is not None
    assert store.search("resilient fact") == []
