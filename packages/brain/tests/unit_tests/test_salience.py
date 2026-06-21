"""Tests for C.4 salience-scored recall."""

from __future__ import annotations

import hashlib as _hashlib
import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Pure math tests
# ---------------------------------------------------------------------------

def test_salience_ordering_older_relevant_beats_newer_irrelevant():
    """A relevant-but-older memory outranks a similar-but-newer-but-less-relevant one."""
    from june_brain.memory.salience import SalienceWeights, salience

    weights = SalienceWeights()  # defaults: rel=0.6, rec=0.25, freq=0.15

    # Memory A: highly relevant (0.9), last accessed 48 hours ago, 1 access
    score_a = salience(0.9, 48.0, 1, weights)

    # Memory B: less relevant (0.4), last accessed 1 hour ago, 5 accesses
    score_b = salience(0.4, 1.0, 5, weights)

    assert score_a > score_b, (
        f"Relevant-but-older ({score_a:.4f}) should beat similar-but-newer ({score_b:.4f})"
    )


def test_salience_recency_decay():
    """Recency decreases monotonically with age."""
    from june_brain.memory.salience import SalienceWeights, salience

    weights = SalienceWeights()
    score_recent = salience(0.5, 1.0, 0, weights)
    score_old = salience(0.5, 200.0, 0, weights)
    assert score_recent > score_old


def test_salience_frequency_increases_score():
    """Higher access count raises score."""
    from june_brain.memory.salience import SalienceWeights, salience

    weights = SalienceWeights()
    score_low = salience(0.5, 10.0, 0, weights)
    score_high = salience(0.5, 10.0, 20, weights)
    assert score_high > score_low


def test_relevance_from_distance_clamps():
    """relevance_from_distance clamps to [0, 1]."""
    from june_brain.memory.salience import relevance_from_distance

    assert relevance_from_distance(0.0) == 1.0
    assert relevance_from_distance(1.0) == 0.0
    assert relevance_from_distance(1.5) == 0.0
    assert relevance_from_distance(-0.1) == 1.0
    assert relevance_from_distance(None) == 0.0


def test_relevance_from_distance_midpoint():
    """distance=0.3 → relevance=0.7."""
    from june_brain.memory.salience import relevance_from_distance

    assert abs(relevance_from_distance(0.3) - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# Weights config-driven
# ---------------------------------------------------------------------------

def test_salience_weights_from_env_defaults():
    """from_env() returns the documented defaults when no env vars are set."""
    env_clean = {k: v for k, v in os.environ.items()
                 if not k.startswith("JUNE_SALIENCE_")}
    with patch.dict(os.environ, env_clean, clear=True):
        from june_brain.memory.salience import SalienceWeights
        w = SalienceWeights.from_env()
    assert w.rel == pytest.approx(0.6)
    assert w.rec == pytest.approx(0.25)
    assert w.freq == pytest.approx(0.15)


def test_salience_weights_from_env_override():
    """JUNE_SALIENCE_REL/REC/FREQ override the defaults."""
    env_override = {"JUNE_SALIENCE_REL": "0.8", "JUNE_SALIENCE_REC": "0.1",
                    "JUNE_SALIENCE_FREQ": "0.1"}
    with patch.dict(os.environ, env_override):
        from june_brain.memory.salience import SalienceWeights
        w = SalienceWeights.from_env()
    assert w.rel == pytest.approx(0.8)
    assert w.rec == pytest.approx(0.1)
    assert w.freq == pytest.approx(0.1)


def test_env_weights_change_ranking():
    """Setting high relevance weight preserves the dominant memory."""
    from june_brain.memory.salience import SalienceWeights, salience

    w_rel_heavy = SalienceWeights(rel=0.95, rec=0.03, freq=0.02)
    # Very relevant, older
    score_a = salience(0.95, 100.0, 0, w_rel_heavy)
    # Less relevant, fresh
    score_b = salience(0.3, 0.0, 50, w_rel_heavy)
    assert score_a > score_b


# ---------------------------------------------------------------------------
# Migration idempotency
# ---------------------------------------------------------------------------

def test_migration_idempotent(tmp_path):
    """Running migration 5 twice does not raise and leaves columns present."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        from june_brain.memory.migration import MIGRATIONS, _migration_005
        from june_brain.memory.sqlite import _get_connection, db_path
        from june_brain.memory.vector import reset_singletons

        reset_singletons()

        conn = _get_connection(db_path())
        MIGRATIONS.ensure(conn)

        # Running the migration function directly a second time must not raise.
        _migration_005(conn)

        # Columns must exist.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(semantic_facts)")}
        assert "access_count" in cols
        assert "last_accessed" in cols
        reset_singletons()


# ---------------------------------------------------------------------------
# Access bookkeeping via recall path
# ---------------------------------------------------------------------------

class _HashEmbedder:
    """Deterministic 64-dim embedder used across memory tests.

    Implements the EmbeddingService surface (``embed`` / ``embed_one``).
    """

    @staticmethod
    def _vec(text: str) -> list[float]:
        digest = _hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) * 2.0 - 1.0 for i in range(64)]

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        return self._vec(text)


@pytest.fixture
def memory_dir(tmp_path):
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        from june_brain.memory import vector as vector_module
        vector_module.reset_singletons()
        yield tmp_path
        vector_module.reset_singletons()


def _make_store(user_id: str):
    from june_brain.memory import Memory, VectorStore
    Memory(user_id)  # ensures schema + migrations run
    return VectorStore(user_id, embedder=_HashEmbedder())


def test_access_bookkeeping_after_recall(memory_dir):
    """After recall returns a vector fact, its access_count is incremented and last_accessed set."""
    from june_brain.memory.manager import MemoryManager
    from june_brain.memory.sqlite import _get_connection, db_path

    store = _make_store("u1")
    record = store.upsert("Paris is in France", source="test")
    fact_id = record["fact_id"]

    mm = MemoryManager("u1", vector=store)
    hits = mm.recall("Where is Paris?", k=5)

    # At least the stored fact should come back.
    assert any(h["ref"] == fact_id for h in hits), "Stored fact not recalled"

    conn = _get_connection(db_path())
    row = conn.execute(
        "SELECT access_count, last_accessed FROM semantic_facts WHERE user_id=? AND fact_id=?",
        ("u1", fact_id),
    ).fetchone()
    assert row is not None
    assert int(row["access_count"]) >= 1
    assert row["last_accessed"] != ""


def test_access_count_increments_on_each_recall(memory_dir):
    """access_count increases by 1 on each recall that returns the fact."""
    from june_brain.memory.manager import MemoryManager
    from june_brain.memory.sqlite import _get_connection, db_path

    store = _make_store("u2")
    record = store.upsert("I run marathons", source="test")
    fact_id = record["fact_id"]

    mm = MemoryManager("u2", vector=store)
    mm.recall("running", k=5)
    mm.recall("running", k=5)

    conn = _get_connection(db_path())
    row = conn.execute(
        "SELECT access_count FROM semantic_facts WHERE user_id=? AND fact_id=?",
        ("u2", fact_id),
    ).fetchone()
    assert int(row["access_count"]) >= 2


def test_salience_ordering_via_recall(memory_dir):
    """Via the recall path (_salience_rerank), a highly relevant fact outranks a barely relevant one.

    We call _salience_rerank directly with controlled distances so the
    test does not depend on ChromaDB's internal distance calculation.
    """
    from june_brain.memory import Memory, VectorStore
    from june_brain.memory.recall import _salience_rerank

    Memory("u3")
    store = VectorStore("u3", embedder=_HashEmbedder())

    # Upsert two facts so shadow rows exist with access_count=0.
    r1 = store.upsert("highly relevant fact", source="test")
    r2 = store.upsert("barely relevant fact", source="test")

    # Simulate raw vector hits: r2 listed first but with higher distance (less relevant).
    raw_hits = [
        {"fact_id": r2["fact_id"], "text": "barely relevant fact",
         "metadata": {"kind": "fact"}, "distance": 0.80},   # less relevant, listed first
        {"fact_id": r1["fact_id"], "text": "highly relevant fact",
         "metadata": {"kind": "fact"}, "distance": 0.05},   # very close, listed second
    ]

    result = _salience_rerank("u3", raw_hits, k=5)

    assert result, "No results from _salience_rerank"
    # After salience re-ranking, the highly relevant fact (distance=0.05) should rank first
    # even though it was listed second in the raw input.
    assert result[0]["ref"] == r1["fact_id"], (
        f"Highly-relevant fact (distance=0.05) should rank first, got {result[0]['ref']!r} "
        f"instead of {r1['fact_id']!r}"
    )


def test_salience_rerank_stores_distance_like_score(memory_dir):
    """The stored score must be distance-like (lower = more salient).

    recall()'s final rank sort is ASCENDING by score and its feedback multipliers
    assume lower=better. If _salience_rerank stored raw salience (higher=better),
    that ascending sort would surface the LEAST salient memory first. This guards
    the end-to-end ordering that the _salience_rerank-only test cannot see.
    """
    from june_brain.memory import Memory, VectorStore
    from june_brain.memory.recall import _salience_rerank

    Memory("u4")
    store = VectorStore("u4", embedder=_HashEmbedder())
    r_hi = store.upsert("very relevant fact", source="test")
    r_lo = store.upsert("weakly relevant fact", source="test")

    raw_hits = [
        {"fact_id": r_lo["fact_id"], "text": "weakly relevant fact",
         "metadata": {"kind": "fact"}, "distance": 0.90},
        {"fact_id": r_hi["fact_id"], "text": "very relevant fact",
         "metadata": {"kind": "fact"}, "distance": 0.05},
    ]
    result = _salience_rerank("u4", raw_hits, k=5)
    by_ref = {h["ref"]: h["score"] for h in result}

    # More salient fact must carry the LOWER stored score.
    assert by_ref[r_hi["fact_id"]] < by_ref[r_lo["fact_id"]]
    # And the ascending sort recall() applies puts the most salient first.
    ordered = sorted(result, key=lambda h: h["score"])
    assert ordered[0]["ref"] == r_hi["fact_id"]
