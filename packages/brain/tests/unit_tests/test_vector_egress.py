"""Tests for vector.py egress-safety behaviour.

These tests never load the real sentence-transformer model. All heavy
callables are monkeypatched at the module level so CI stays offline.
"""

from __future__ import annotations

import pytest
from june_brain.memory import vector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    """Isolate singleton state and data dir around every test."""
    monkeypatch.setattr("june_brain.memory.MEMORY_DIR", str(tmp_path))
    vector.reset_singletons()
    yield
    vector.reset_singletons()


# ---------------------------------------------------------------------------
# Test 1: local-only + not cached -> no egress, no build
# ---------------------------------------------------------------------------


def test_local_only_not_cached_returns_none(monkeypatch):
    monkeypatch.setattr(vector, "_model_is_cached", lambda: False)
    monkeypatch.setattr(vector, "_egress_blocked", lambda: True)

    calls = []

    def _must_not_be_called(local_files_only):
        calls.append(local_files_only)
        raise AssertionError("_build_embedder must not be called in local-only mode")

    monkeypatch.setattr(vector, "_build_embedder", _must_not_be_called)

    result = vector._get_embedding_function()
    assert result is None
    assert calls == []


# ---------------------------------------------------------------------------
# Test 2: cached -> loaded with local_files_only=True
# ---------------------------------------------------------------------------


def test_cached_loads_with_local_files_only_true(monkeypatch):
    monkeypatch.setattr(vector, "_model_is_cached", lambda: True)

    calls = []
    sentinel = object()

    def fake_build(local_files_only):
        calls.append(local_files_only)
        return sentinel

    monkeypatch.setattr(vector, "_build_embedder", fake_build)

    result = vector._get_embedding_function()
    assert result is not None
    assert result is sentinel
    assert calls == [True]


# ---------------------------------------------------------------------------
# Test 3: not cached + egress allowed -> loaded with local_files_only=False
# ---------------------------------------------------------------------------


def test_not_cached_egress_allowed_builds_with_local_files_only_false(monkeypatch):
    monkeypatch.setattr(vector, "_model_is_cached", lambda: False)
    monkeypatch.setattr(vector, "_egress_blocked", lambda: False)

    calls = []
    sentinel = object()

    def fake_build(local_files_only):
        calls.append(local_files_only)
        return sentinel

    monkeypatch.setattr(vector, "_build_embedder", fake_build)

    result = vector._get_embedding_function()
    assert result is not None
    assert result is sentinel
    assert calls == [False]


# ---------------------------------------------------------------------------
# Test 4: load failure degrades to None
# ---------------------------------------------------------------------------


def test_load_failure_degrades_to_none(monkeypatch):
    monkeypatch.setattr(vector, "_model_is_cached", lambda: True)

    def failing_build(local_files_only):
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(vector, "_build_embedder", failing_build)

    result = vector._get_embedding_function()
    assert result is None


# ---------------------------------------------------------------------------
# Test 5: graceful degradation — upsert/search survive no embedder
# ---------------------------------------------------------------------------


def test_graceful_degradation_no_embedder(tmp_path, monkeypatch):
    monkeypatch.setattr("june_brain.memory.MEMORY_DIR", str(tmp_path))
    vector.reset_singletons()

    # Ensure the shadow table exists.
    from june_brain.memory import Memory
    Memory("degtest")

    build_calls = []

    def _must_not_build(local_files_only):
        build_calls.append(local_files_only)
        raise AssertionError("_build_embedder must not be called")

    monkeypatch.setattr(vector, "_build_embedder", _must_not_build)
    monkeypatch.setattr(vector, "_get_embedding_function", lambda: None)

    store = vector.VectorStore("degtest")

    # upsert must NOT raise; fact must land in the SQLite shadow.
    record = store.upsert("some fact")
    assert record["fact_id"]

    # Shadow table reflects the write.
    facts = store.list_facts()
    assert len(facts) == 1
    assert facts[0]["text"] == "some fact"

    fetched = store.get(record["fact_id"])
    assert fetched is not None
    assert fetched["text"] == "some fact"

    # search must return [] gracefully (not raise).
    hits = store.search("anything")
    assert hits == []

    assert build_calls == []


# ---------------------------------------------------------------------------
# Test 6: delete in local-only degrades — shadow row is still removed
# ---------------------------------------------------------------------------


def test_delete_local_only_degrades_shadow_removed(tmp_path, monkeypatch):
    monkeypatch.setattr("june_brain.memory.MEMORY_DIR", str(tmp_path))
    vector.reset_singletons()

    from june_brain.memory import Memory
    Memory("deltest")

    monkeypatch.setattr(vector, "_get_embedding_function", lambda: None)

    store = vector.VectorStore("deltest")

    record = store.upsert("fact to delete")
    fact_id = record["fact_id"]

    # Shadow was written.
    assert store.get(fact_id) is not None

    # delete must NOT raise even though the vector index is unavailable.
    store.delete(fact_id)

    # Shadow row must be gone.
    assert store.get(fact_id) is None


# ---------------------------------------------------------------------------
# Test 7: the embedder load quiets transformers' console noise
# ---------------------------------------------------------------------------


def test_quiet_transformers_sets_error_verbosity():
    # Best-effort and version-tolerant: it must never raise.
    vector._quiet_transformers()
    from transformers.utils import logging as hf_logging

    assert hf_logging.get_verbosity() >= hf_logging.ERROR
