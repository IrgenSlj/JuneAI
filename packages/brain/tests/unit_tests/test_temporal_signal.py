"""Tests for the temporal signal in recall — ADR 0024 four-signal fusion.

The temporal signal has two components:
- time_score: penalizes expired facts (valid_to)
- valid_from_score: boosts recently-created facts (valid_from)

These are distinct signals: time_score is "when did it stop being true",
valid_from_score is "when did it become true".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from june_brain.memory.recall import (
    RetrievalConfig,
    _temporal_prior,
    _valid_from_prior,
)


class TestTemporalPrior:
    """Tests for _temporal_prior (valid_to expiry)."""

    def test_no_valid_to_returns_one(self) -> None:
        now = datetime.now(UTC)
        config = RetrievalConfig()
        assert _temporal_prior(None, now, config) == 1.0
        assert _temporal_prior("", now, config) == 1.0

    def test_future_valid_to_returns_one(self) -> None:
        now = datetime.now(UTC)
        future = (now + timedelta(days=30)).isoformat()
        config = RetrievalConfig()
        assert _temporal_prior(future, now, config) == 1.0

    def test_past_valid_to_decays(self) -> None:
        now = datetime.now(UTC)
        past = (now - timedelta(days=90)).isoformat()
        config = RetrievalConfig(temporal_half_life_days=90.0)
        score = _temporal_prior(past, now, config)
        # At 90 days with 90-day half-life, score should be ~0.5
        assert 0.4 < score < 0.6

    def test_very_old_fact_has_floor(self) -> None:
        now = datetime.now(UTC)
        very_old = (now - timedelta(days=3650)).isoformat()
        config = RetrievalConfig(temporal_half_life_days=90.0)
        score = _temporal_prior(very_old, now, config)
        assert score >= 0.1  # floor

    def test_recent_past_has_higher_score(self) -> None:
        now = datetime.now(UTC)
        recent = (now - timedelta(days=10)).isoformat()
        old = (now - timedelta(days=100)).isoformat()
        config = RetrievalConfig(temporal_half_life_days=90.0)
        assert _temporal_prior(recent, now, config) > _temporal_prior(old, now, config)


class TestValidFromPrior:
    """Tests for _valid_from_prior (valid_from recency)."""

    def test_no_valid_from_returns_one(self) -> None:
        now = datetime.now(UTC)
        config = RetrievalConfig()
        assert _valid_from_prior(None, now, config) == 1.0
        assert _valid_from_prior("", now, config) == 1.0

    def test_invalid_valid_from_returns_one(self) -> None:
        now = datetime.now(UTC)
        config = RetrievalConfig()
        assert _valid_from_prior("not-a-date", now, config) == 1.0

    def test_recent_valid_from_has_boost(self) -> None:
        now = datetime.now(UTC)
        recent = (now - timedelta(days=10)).isoformat()
        config = RetrievalConfig(temporal_half_life_days=90.0)
        score = _valid_from_prior(recent, now, config)
        # Recent facts should have score > 0.5 (boost)
        assert score > 0.5

    def test_old_valid_from_decays(self) -> None:
        now = datetime.now(UTC)
        old = (now - timedelta(days=180)).isoformat()
        config = RetrievalConfig(temporal_half_life_days=90.0)
        score = _valid_from_prior(old, now, config)
        # Old facts should have score closer to 0.5 (no boost)
        assert 0.5 <= score < 0.7

    def test_very_old_valid_from_has_floor(self) -> None:
        now = datetime.now(UTC)
        very_old = (now - timedelta(days=3650)).isoformat()
        config = RetrievalConfig(temporal_half_life_days=90.0)
        score = _valid_from_prior(very_old, now, config)
        assert score >= 0.5  # floor

    def test_recent_beats_old(self) -> None:
        now = datetime.now(UTC)
        recent = (now - timedelta(days=5)).isoformat()
        old = (now - timedelta(days=300)).isoformat()
        config = RetrievalConfig(temporal_half_life_days=90.0)
        assert _valid_from_prior(recent, now, config) > _valid_from_prior(old, now, config)


class TestFusionScore:
    """Tests that the four-signal fusion includes valid_from_score."""

    def test_fusion_includes_valid_from_score(self) -> None:
        """Verify that _fuse_semantic_hits uses valid_from_score in the fusion."""
        from june_brain.memory.recall import _fuse_semantic_hits

        config = RetrievalConfig()
        vector_hits = [
            {
                "ref": "fact1",
                "text": "test fact",
                "score": 0.3,
                "time_score": 1.0,
                "valid_from_score": 1.2,  # boost
            }
        ]
        bm25_hits = []

        result = _fuse_semantic_hits(vector_hits, bm25_hits, [], config)
        assert len(result) == 1
        assert result[0]["valid_from_score"] == 1.2
        # The fused score should reflect the valid_from boost
        assert result[0]["rrf"] > 0

    def test_valid_from_boost_increases_fusion_score(self) -> None:
        """A fact with a recent valid_from should score higher than one without."""
        from june_brain.memory.recall import _fuse_semantic_hits

        config = RetrievalConfig()
        # Same RRF score, but different valid_from scores
        hits_with_boost = [
            {
                "ref": "fact1",
                "text": "test fact",
                "score": 0.3,
                "time_score": 1.0,
                "valid_from_score": 1.3,  # recent valid_from
            }
        ]
        hits_without_boost = [
            {
                "ref": "fact2",
                "text": "test fact",
                "score": 0.3,
                "time_score": 1.0,
                "valid_from_score": 1.0,  # no boost
            }
        ]

        result_with = _fuse_semantic_hits(hits_with_boost, [], [], config)
        result_without = _fuse_semantic_hits(hits_without_boost, [], [], config)

        assert result_with[0]["rrf"] > result_without[0]["rrf"]
