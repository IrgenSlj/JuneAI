"""Salience scoring for memory recall.

Formula (C.4):
    salience = weights.rel * relevance
             + weights.rec * recency
             + weights.freq * frequency

where:
    relevance = clamp(1 - cosine_distance, 0, 1)
    recency   = exp(-LAMBDA * hours_since_last_access)
    frequency = log1p(access_count) / log1p(MAX_ACCESS)

Weights are read from env at import time via SalienceWeights.from_env().
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

# Recency decay constant.  0.0058/hr gives a half-life of ~5 days:
#   exp(-0.0058 * 24 * 5) ≈ 0.5
LAMBDA: float = float(os.environ.get("JUNE_SALIENCE_LAMBDA", "0.0058"))

# Frequency normaliser — access counts are clamped to this before log.
MAX_ACCESS: int = int(os.environ.get("JUNE_SALIENCE_MAX_ACCESS", "50"))


@dataclass
class SalienceWeights:
    rel: float = 0.6
    rec: float = 0.25
    freq: float = 0.15

    @classmethod
    def from_env(cls) -> SalienceWeights:
        rel = float(os.environ.get("JUNE_SALIENCE_REL", "0.6"))
        rec = float(os.environ.get("JUNE_SALIENCE_REC", "0.25"))
        freq = float(os.environ.get("JUNE_SALIENCE_FREQ", "0.15"))
        return cls(rel=rel, rec=rec, freq=freq)

    @classmethod
    def load(cls) -> SalienceWeights:
        """Resolve weights with precedence env > config store > default (S5.4).

        Runtime-tunable: the config store is written by the settings surface, so
        re-tuning takes effect on the next recall without a restart. An explicit
        ``JUNE_SALIENCE_*`` env var still wins (deploy/test override).
        """
        try:
            from june_brain.config_store import get_salience_weights

            cfg = get_salience_weights()
        except Exception:  # noqa: BLE001 — config is a convenience layer
            cfg = {}

        def pick(env_key: str, cfg_key: str, default: float) -> float:
            if env_key in os.environ:
                return float(os.environ[env_key])
            if cfg_key in cfg:
                return cfg[cfg_key]
            return default

        return cls(
            rel=pick("JUNE_SALIENCE_REL", "rel", 0.6),
            rec=pick("JUNE_SALIENCE_REC", "rec", 0.25),
            freq=pick("JUNE_SALIENCE_FREQ", "freq", 0.15),
        )


def relevance_from_distance(distance: float | None) -> float:
    """Convert a cosine distance (lower = closer) to a 0..1 relevance score."""
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - distance))


def salience(
    relevance: float,
    hours_since_access: float,
    access_count: int,
    weights: SalienceWeights,
) -> float:
    """Return the salience score for one memory candidate.

    Args:
        relevance:          0..1 — semantic similarity to the query.
        hours_since_access: hours elapsed since the memory was last accessed
                            (0 = just accessed; never-accessed row uses 0).
        access_count:       number of times this memory has been recalled.
        weights:            SalienceWeights controlling the blend.
    """
    recency = math.exp(-LAMBDA * max(0.0, hours_since_access))
    frequency = math.log1p(access_count) / math.log1p(MAX_ACCESS)
    return weights.rel * relevance + weights.rec * recency + weights.freq * frequency
