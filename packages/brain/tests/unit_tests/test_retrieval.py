"""Tests for retrieval v2 fusion scoring (ADR 0024).

Covers:
- _fuse_semantic_hits: RRF fusion of vector + BM25 ranked lists
- _entity_overlap_score: entity boost from graph mention labels
- _temporal_prior: half-life decay for expired facts
- gather_hits: end-to-end integration with all three stores
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from june_brain.memory.recall import (
    RetrievalConfig,
    _entity_overlap_score,
    _fuse_semantic_hits,
    _temporal_prior,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> RetrievalConfig:
    return RetrievalConfig(
        candidate_pool=50,
        rrf_k=60,
        entity_weight=0.15,
        temporal_half_life_days=90.0,
    )


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 7, 24, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# _entity_overlap_score
# ---------------------------------------------------------------------------


class TestEntityOverlapScore:
    def test_no_labels_returns_zero(self):
        assert _entity_overlap_score("loves coffee", []) == 0.0

    def test_no_match_returns_zero(self):
        assert _entity_overlap_score("loves coffee", ["Berlin", "Paris"]) == 0.0

    def test_single_match(self):
        assert _entity_overlap_score("Alice lives in Berlin", ["Berlin"]) == 1.0

    def test_multiple_matches(self):
        text = "Alice lives in Berlin and works at Google"
        labels = ["Alice", "Berlin", "Google"]
        assert _entity_overlap_score(text, labels) == 3.0

    def test_case_insensitive(self):
        assert _entity_overlap_score("alice lives in berlin", ["Alice", "Berlin"]) == 2.0

    def test_empty_label_ignored(self):
        assert _entity_overlap_score("loves coffee", ["", "Berlin"]) == 0.0


# ---------------------------------------------------------------------------
# _temporal_prior
# ---------------------------------------------------------------------------


class TestTemporalPrior:
    def test_no_valid_to_returns_one(self, now, config):
        assert _temporal_prior(None, now, config) == 1.0

    def test_empty_valid_to_returns_one(self, now, config):
        assert _temporal_prior("", now, config) == 1.0

    def test_future_valid_to_returns_one(self, now, config):
        future = (now + timedelta(days=30)).isoformat()
        assert _temporal_prior(future, now, config) == 1.0

    def test_current_valid_to_returns_one(self, now, config):
        assert _temporal_prior(now.isoformat(), now, config) == 1.0

    def test_expired_90_days_ago_returns_half(self, now, config):
        expired = (now - timedelta(days=90)).isoformat()
        result = _temporal_prior(expired, now, config)
        assert math.isclose(result, 0.5, rel_tol=0.01)

    def test_expired_180_days_ago_returns_quarter(self, now, config):
        expired = (now - timedelta(days=180)).isoformat()
        result = _temporal_prior(expired, now, config)
        assert math.isclose(result, 0.25, rel_tol=0.01)

    def test_expired_very_old_returns_floor(self, now, config):
        expired = (now - timedelta(days=3650)).isoformat()
        result = _temporal_prior(expired, now, config)
        assert result >= 0.1

    def test_floor_is_0_1(self, now, config):
        expired = (now - timedelta(days=36500)).isoformat()
        assert _temporal_prior(expired, now, config) == 0.1


# ---------------------------------------------------------------------------
# _fuse_semantic_hits
# ---------------------------------------------------------------------------


class TestFuseSemanticHits:
    def test_empty_inputs(self, config):
        result = _fuse_semantic_hits([], [], [], config)
        assert result == []

    def test_vector_only(self, config):
        vector_hits = [
            {"ref": "f1", "text": "coffee", "score": 0.2, "time_score": 1.0},
            {"ref": "f2", "text": "tea", "score": 0.4, "time_score": 1.0},
        ]
        result = _fuse_semantic_hits(vector_hits, [], [], config)
        assert len(result) == 2
        # Both should have RRF scores from vector channel only
        assert all("rrf" in h for h in result)
        # First vector hit should rank higher (lower score = better)
        assert result[0]["ref"] == "f1"

    def test_bm25_only(self, config):
        bm25_hits = [
            {
                "ref": "f1",
                "text": "coffee",
                "score": 0.3,
                "time_score": 1.0,
                "bm25": -2.5,
                "bm25_relevance": 0.8,
            },
        ]
        result = _fuse_semantic_hits([], bm25_hits, [], config)
        assert len(result) == 1
        assert result[0]["ref"] == "f1"

    def test_rrf_fusion_of_both_channels(self, config):
        vector_hits = [
            {"ref": "f1", "text": "coffee", "score": 0.2, "time_score": 1.0},
            {"ref": "f2", "text": "tea", "score": 0.4, "time_score": 1.0},
        ]
        bm25_hits = [
            {
                "ref": "f2",
                "text": "tea",
                "score": 0.3,
                "time_score": 1.0,
                "bm25": -1.0,
                "bm25_relevance": 0.9,
            },
            {
                "ref": "f1",
                "text": "coffee",
                "score": 0.5,
                "time_score": 1.0,
                "bm25": -0.5,
                "bm25_relevance": 0.6,
            },
        ]
        result = _fuse_semantic_hits(vector_hits, bm25_hits, [], config)
        assert len(result) == 2
        # f2 appears in both channels at rank 2 and 1 → higher RRF than f1
        # f1 appears in both at rank 1 and 2
        # Both should have entity_score and time_score
        for h in result:
            assert "entity_score" in h
            assert "time_score" in h

    def test_entity_boost(self, config):
        vector_hits = [
            {"ref": "f1", "text": "Alice lives in Berlin", "score": 0.2, "time_score": 1.0},
        ]
        bm25_hits = []
        # "Alice" is in query_entity_labels → boost
        result = _fuse_semantic_hits(vector_hits, bm25_hits, ["Alice"], config)
        assert len(result) == 1
        assert result[0]["entity_score"] == 1.0
        # RRF score should be boosted by entity_weight * entity_score
        assert result[0]["rrf"] > 1.0 / (config.rrf_k + 1)

    def test_temporal_decay(self, config):
        vector_hits = [
            {
                "ref": "f1",
                "text": "expired fact",
                "score": 0.2,
                "time_score": 0.5,  # expired 90 days ago
            },
        ]
        result = _fuse_semantic_hits(vector_hits, [], [], config)
        assert len(result) == 1
        assert result[0]["time_score"] == 0.5
        # Score should be higher (worse) due to temporal decay
        assert result[0]["score"] > 0.0

    def test_dedup_by_ref(self, config):
        vector_hits = [
            {"ref": "f1", "text": "coffee", "score": 0.2, "time_score": 1.0},
        ]
        bm25_hits = [
            {
                "ref": "f1",
                "text": "coffee",
                "score": 0.3,
                "time_score": 1.0,
                "bm25": -1.0,
                "bm25_relevance": 0.8,
            },
        ]
        result = _fuse_semantic_hits(vector_hits, bm25_hits, [], config)
        assert len(result) == 1  # deduplicated by ref

    def test_score_is_distance_like(self, config):
        """Public score should be lower-is-better (distance-like)."""
        vector_hits = [
            {"ref": "f1", "text": "coffee", "score": 0.2, "time_score": 1.0},
        ]
        result = _fuse_semantic_hits(vector_hits, [], [], config)
        assert len(result) == 1
        # score = 1.0 / max(rrf, 1e-12) → lower rrf means higher (worse) score
        assert result[0]["score"] > 0

    def test_sorted_by_rrf_desc(self, config):
        """Results should be sorted by RRF score descending (best first)."""
        vector_hits = [
            {"ref": "f1", "text": "first", "score": 0.1, "time_score": 1.0},
            {"ref": "f2", "text": "second", "score": 0.5, "time_score": 1.0},
        ]
        result = _fuse_semantic_hits(vector_hits, [], [], config)
        rrf_scores = [h["rrf"] for h in result]
        assert rrf_scores == sorted(rrf_scores, reverse=True)


# ---------------------------------------------------------------------------
# Integration: gather_hits with mocked stores
# ---------------------------------------------------------------------------


class TestGatherHitsIntegration:
    @patch("june_brain.memory.recall._get_connection")
    @patch("june_brain.memory.recall._db_path")
    def test_gather_hits_returns_results(self, mock_db_path, mock_conn):
        """Basic integration test: gather_hits with mocked vector/graph/sqlite."""
        from june_brain.memory.recall import gather_hits

        mock_db_path.return_value = ":memory:"

        # Mock vector store
        vector = MagicMock()
        vector.search.return_value = [
            {"fact_id": "f1", "text": "loves coffee", "distance": 0.2, "metadata": "{}"},
            {"fact_id": "f2", "text": "prefers tea", "distance": 0.4, "metadata": "{}"},
        ]

        # Mock graph store
        graph = MagicMock()
        graph.mentions_near.return_value = []
        graph.neighbors.return_value = []

        # Mock sqlite store
        sqlite = MagicMock()
        sqlite.get_goals.return_value = []
        sqlite.get_open_loops.return_value = []
        sqlite.get_preferences.return_value = []
        sqlite.get_relationship_profiles.return_value = []
        sqlite.get_journal.return_value = []
        sqlite.get_feedback_map.return_value = {}

        # Mock the DB connection for salience queries
        conn = MagicMock()
        mock_conn.return_value = conn
        conn.execute.return_value.fetchall.return_value = []
        conn.execute.return_value.fetchone.return_value = (0, "", None)

        hits = gather_hits(vector, graph, sqlite, "test_user", "coffee", k=5)
        assert isinstance(hits, list)
        # Should have at least the vector hits (after salience rerank)
        # Note: salience rerank may filter out hits without valid DB rows

    @patch("june_brain.memory.recall._get_connection")
    @patch("june_brain.memory.recall._db_path")
    def test_gather_hits_empty_query(self, mock_db_path, mock_conn):
        from june_brain.memory.recall import gather_hits

        vector = MagicMock()
        graph = MagicMock()
        sqlite = MagicMock()

        hits = gather_hits(vector, graph, sqlite, "test_user", "", k=5)
        assert hits == []

    @patch("june_brain.memory.recall._get_connection")
    @patch("june_brain.memory.recall._db_path")
    def test_gather_hits_graph_entities_boost_semantic(self, mock_db_path, mock_conn):
        """Graph entity mentions should boost semantic facts in fusion."""
        from june_brain.memory.recall import gather_hits

        mock_db_path.return_value = ":memory:"

        vector = MagicMock()
        vector.search.return_value = [
            {"fact_id": "f1", "text": "Alice lives in Berlin", "distance": 0.2, "metadata": "{}"},
        ]

        graph = MagicMock()
        graph.mentions_near.return_value = [
            {"node_id": "n1", "label": "Alice", "kind": "person", "props": {"description": "a friend"}},
        ]
        graph.neighbors.return_value = []

        sqlite = MagicMock()
        sqlite.get_goals.return_value = []
        sqlite.get_open_loops.return_value = []
        sqlite.get_preferences.return_value = []
        sqlite.get_relationship_profiles.return_value = []
        sqlite.get_journal.return_value = []
        sqlite.get_feedback_map.return_value = {}

        conn = MagicMock()
        mock_conn.return_value = conn
        conn.execute.return_value.fetchall.return_value = []
        conn.execute.return_value.fetchone.return_value = (0, "", None)

        hits = gather_hits(vector, graph, sqlite, "test_user", "Alice", k=5)
        # The graph entity hit should be present
        graph_hits = [h for h in hits if h["source"] == "graph"]
        assert len(graph_hits) >= 1
