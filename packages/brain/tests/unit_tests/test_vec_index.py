"""Unit tests for the sqlite-vec vector index (ADR 0019, S2.1).

Isolated from the VectorStore: exercises the vec0 helper directly on an
in-memory connection with deterministic vectors, so it proves the index
mechanics (load, create, upsert, KNN, delete, dim-change rebuild, degrade)
without Ollama or the rest of the memory stack.
"""

from __future__ import annotations

import sqlite3

import pytest
from june_brain.memory import vec_index


@pytest.fixture
def conn():
    vec_index.reset_availability()
    c = sqlite3.connect(":memory:")
    loaded = vec_index.load_extension(c)
    if not loaded:
        pytest.skip("sqlite-vec extension cannot load on this platform")
    yield c
    c.close()
    vec_index.reset_availability()


def test_load_extension_reports_available(conn):
    assert vec_index.is_available() is True


def test_upsert_and_search_orders_by_distance(conn):
    vec_index.upsert(conn, "a", [1.0, 0.0, 0.0, 0.0])
    vec_index.upsert(conn, "b", [0.0, 1.0, 0.0, 0.0])
    vec_index.upsert(conn, "c", [0.9, 0.1, 0.0, 0.0])
    assert vec_index.count(conn) == 3

    hits = vec_index.search(conn, [1.0, 0.0, 0.0, 0.0], k=3)
    ids = [fid for fid, _dist in hits]
    assert ids[0] == "a"  # exact match is nearest
    assert ids[1] == "c"  # then the near-parallel vector
    assert ids[2] == "b"
    assert hits[0][1] == pytest.approx(0.0, abs=1e-5)  # cosine distance to itself


def test_upsert_replaces_existing_id(conn):
    vec_index.upsert(conn, "x", [1.0, 0.0, 0.0, 0.0])
    vec_index.upsert(conn, "x", [0.0, 1.0, 0.0, 0.0])
    assert vec_index.count(conn) == 1
    hits = vec_index.search(conn, [0.0, 1.0, 0.0, 0.0], k=1)
    assert hits[0][0] == "x"
    assert hits[0][1] == pytest.approx(0.0, abs=1e-5)


def test_delete_removes_from_index(conn):
    vec_index.upsert(conn, "a", [1.0, 0.0, 0.0, 0.0])
    vec_index.upsert(conn, "b", [0.0, 1.0, 0.0, 0.0])
    vec_index.delete(conn, "a")
    assert vec_index.count(conn) == 1
    ids = [fid for fid, _ in vec_index.search(conn, [1.0, 0.0, 0.0, 0.0], k=5)]
    assert "a" not in ids


def test_dimension_change_rebuilds_index(conn):
    vec_index.upsert(conn, "a", [1.0, 0.0, 0.0, 0.0])  # dim 4
    assert vec_index.count(conn) == 1
    # A 3-dim vector means the embedding model changed; the index rebuilds
    # empty (the caller re-embeds from the authoritative shadow rows).
    assert vec_index.upsert(conn, "b", [1.0, 0.0, 0.0]) is True
    assert vec_index.count(conn) == 1
    hits = vec_index.search(conn, [1.0, 0.0, 0.0], k=5)
    assert [fid for fid, _ in hits] == ["b"]


def test_search_empty_when_index_absent(conn):
    # No upserts yet -> the vec0 table was never created.
    assert vec_index.search(conn, [1.0, 0.0, 0.0, 0.0], k=3) == []
    assert vec_index.count(conn) == 0
