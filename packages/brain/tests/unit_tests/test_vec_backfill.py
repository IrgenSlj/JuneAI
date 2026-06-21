"""Tests for the ChromaDB -> sqlite-vec migration (ADR 0019, S2.4)."""

from __future__ import annotations

import hashlib

import pytest
from june_brain.memory import vec_index
from june_brain.memory.sqlite import Memory, _get_connection, db_path
from june_brain.memory.vector import VectorStore


class _HashEmbedder:
    @staticmethod
    def _vec(text: str) -> list[float]:
        d = hashlib.sha256(text.encode("utf-8")).digest()
        return [(d[i % len(d)] / 255.0) * 2 - 1 for i in range(64)]

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        return self._vec(text)


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("june_brain.memory.MEMORY_DIR", str(tmp_path))
    import june_brain.memory.vector as vector_module

    vector_module.reset_singletons()
    yield tmp_path
    vector_module.reset_singletons()


# ---------------------------------------------------------------------------
# Manifest migration: chroma -> chroma.bak
# ---------------------------------------------------------------------------


def test_manifest_v1_to_v2_archives_chroma(tmp_path, monkeypatch):
    monkeypatch.setenv("JUNE_DATA_DIR", str(tmp_path))
    from june_brain.datadir import layout, manifest

    layout.ensure_layout()
    chroma = layout.memory_dir() / "chroma"
    chroma.mkdir(parents=True, exist_ok=True)
    (chroma / "index.bin").write_text("legacy")

    # Write a v1 manifest, then load_or_init should migrate to current version.
    old = manifest.Manifest(
        schema_version=1,
        created_at="2026-01-01T00:00:00+00:00",
        june_version="0.2.0",
        contents=[],
    )
    manifest.write_manifest(old)

    migrated, fresh = manifest.load_or_init()
    assert fresh is False
    assert migrated.schema_version == manifest.SCHEMA_VERSION
    assert not chroma.exists()
    assert (layout.memory_dir() / "chroma.bak").exists()


# ---------------------------------------------------------------------------
# Vector backfill from shadow rows
# ---------------------------------------------------------------------------


def test_backfill_embeds_missing_shadow_rows(memory_dir):
    from june_brain.memory import vec_backfill

    Memory("alex")
    # Write facts with an unavailable embedder so they land in the shadow only.
    store = VectorStore("alex", embedder=_Unavailable())
    r1 = store.upsert("first fact")
    r2 = store.upsert("second fact")

    conn = _get_connection(db_path())
    assert vec_index.count(conn) == 0  # nothing indexed yet

    summary = vec_backfill.backfill_vectors(conn, _HashEmbedder())
    assert summary == {"total": 2, "embedded": 2, "remaining": 0}
    assert vec_index.count(conn) == 2

    # The backfilled vectors are searchable.
    hits = VectorStore("alex", embedder=_HashEmbedder()).search("first fact", k=1)
    assert hits and hits[0]["fact_id"] in {r1["fact_id"], r2["fact_id"]}


def test_backfill_is_idempotent(memory_dir):
    from june_brain.memory import vec_backfill

    Memory("alex")
    VectorStore("alex", embedder=_HashEmbedder()).upsert("already indexed")
    conn = _get_connection(db_path())
    assert vec_index.count(conn) == 1

    # Nothing missing -> backfill embeds zero.
    summary = vec_backfill.backfill_vectors(conn, _HashEmbedder())
    assert summary == {"total": 0, "embedded": 0, "remaining": 0}


def test_backfill_degrades_when_embedder_unavailable(memory_dir):
    from june_brain.memory import vec_backfill

    Memory("alex")
    VectorStore("alex", embedder=_Unavailable()).upsert("orphan fact")
    conn = _get_connection(db_path())

    summary = vec_backfill.backfill_vectors(conn, _Unavailable())
    assert summary["embedded"] == 0
    assert summary["total"] == 1  # still pending, nothing lost


class _Unavailable:
    def embed(self, texts):
        return None

    def embed_one(self, text):
        return None
