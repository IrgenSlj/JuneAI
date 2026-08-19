"""Recall read-path: candidate gathering across the three stores + salience rerank.

``gather_hits`` fans out to the vector, FTS5 lexical, graph, and SQLite stores,
dedupes, applies feedback multipliers, and ranks. ``_salience_rerank`` re-scores
the vector candidates by ``recency x frequency x relevance``. Extracted from
``manager.py`` (S3 decomposition) so the recall logic lives in one focused
module; ``MemoryManager.recall`` is a thin facade over ``gather_hits``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .paraphrase import _format_edge, _format_node
from .salience import SalienceWeights, relevance_from_distance, salience_detailed
from .sqlite import _get_connection
from .vector import _db_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    """Runtime knobs for Retrieval v2 fusion.

    Candidate pools are deliberately capped before fusion so recall latency stays
    bounded even as the memory table grows. Env vars are developer/operator
    overrides; the defaults match ADR 0024.
    """

    # Measured against the golden corpus, not guessed: recall@8 rises
    # monotonically as the pool shrinks (50 -> 0.735, 30 -> 0.740, 20 -> 0.760,
    # 15 -> 0.790) because RRF credits every document in the pool, so a wide
    # pool feeds the fusion more noise to rank rather than more signal. 15 is
    # taken over the corpus-best 12 to keep margin against overfitting a
    # 100-case set. Latency is indifferent: sqlite-vec scans the whole index
    # regardless of k, so this is quality-only.
    candidate_pool: int = 15
    rrf_k: int = 60
    entity_weight: float = 0.15
    temporal_half_life_days: float = 90.0

    @classmethod
    def load(cls) -> RetrievalConfig:
        return cls(
            candidate_pool=max(1, _env_int("JUNE_RETRIEVAL_CANDIDATE_POOL", 15)),
            rrf_k=max(1, _env_int("JUNE_RETRIEVAL_RRF_K", 60)),
            entity_weight=max(0.0, _env_float("JUNE_RETRIEVAL_ENTITY_WEIGHT", 0.15)),
            temporal_half_life_days=max(
                1.0,
                _env_float("JUNE_RETRIEVAL_TEMPORAL_HALF_LIFE_DAYS", 90.0),
            ),
        )


_LOGGED_CONFIG: RetrievalConfig | None = None


def gather_hits(
    vector: Any,
    graph: Any,
    sqlite: Any,
    user_id: str,
    query: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return up to ``k * 2`` ranked recall hits across all three stores.

    Each hit is a dict with source / text / kind / ref / score (see
    ``MemoryManager.recall`` for the field contract).
    """
    query = (query or "").strip()
    if not query:
        return []

    config = _load_retrieval_config()
    candidate_pool = max(k, config.candidate_pool)
    hits: list[dict[str, Any]] = []
    graph_hits: list[dict[str, Any]] = []
    query_entity_labels: list[str] = []

    # 1) Graph — entities the query mentions, plus their neighbors. We collect
    # entity labels before semantic fusion so fact text that overlaps a mentioned
    # entity gets the ADR 0024 entity boost.
    try:
        for node in graph.mentions_near(query, limit=k):
            label = str(node.get("label", "")).strip()
            if label:
                query_entity_labels.append(label)
            graph_hits.append(
                {
                    "source": "graph",
                    "text": _format_node(node),
                    "kind": f"entity:{node['kind']}",
                    "ref": node["node_id"],
                    "score": 0.0,
                }
            )
            for edge in graph.neighbors(node["node_id"], limit=3):
                graph_hits.append(
                    {
                        "source": "graph",
                        "text": _format_edge(node, edge),
                        "kind": f"edge:{edge['edge']['kind']}",
                        "ref": f"{edge['edge']['src']}|{edge['edge']['dst']}|{edge['edge']['kind']}",
                        "score": 0.1,
                    }
                )
    except Exception:  # noqa: BLE001
        logger.exception("recall: graph lookup failed")

    # 2) Semantic facts — vector salience + FTS5/BM25 fused with RRF.
    vector_hits: list[dict[str, Any]] = []
    try:
        raw_vector = vector.search(query, k=candidate_pool)
        if raw_vector:
            vector_hits = _salience_rerank(
                user_id,
                raw_vector,
                candidate_pool,
                config=config,
                update_access=False,
            )
    except Exception:  # noqa: BLE001
        logger.exception("recall: vector search failed")

    bm25_hits: list[dict[str, Any]] = []
    try:
        bm25_hits = semantic_bm25_hits(user_id, query, k=candidate_pool, config=config)
    except Exception:  # noqa: BLE001
        logger.exception("recall: semantic FTS lookup failed")

    hits.extend(_fuse_semantic_hits(vector_hits, bm25_hits, query_entity_labels, config))
    hits.extend(graph_hits)

    # 3) Structured (SQLite) — look for query terms across goals, open loops,
    # preferences, relationships, journal. Cheap keyword scan; the LLM gets
    # the top matches so it can notice "the user mentioned X weeks ago."
    try:
        hits.extend(sqlite_keyword_hits(sqlite, query, k=k))
    except Exception:  # noqa: BLE001
        logger.exception("recall: sqlite keyword scan failed")

    # Dedupe by text (case-insensitive) so the same fact doesn't appear
    # three times when it landed in multiple stores.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for h in hits:
        key = h["text"].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(h)

    # Apply feedback multipliers before ranking. Recall hit refs are not
    # yet prefixed for non-sqlite sources (vector returns the bare
    # fact_id, graph returns the bare node_id), so we re-derive the
    # prefixed form to match what the user voted on through the UI.
    try:
        feedback = sqlite.get_feedback_map()
    except Exception:  # noqa: BLE001
        logger.exception("recall: feedback lookup failed")
        feedback = {}

    if feedback:
        for h in deduped:
            lookup_ref = _hit_lookup_ref(h)
            vote = feedback.get(lookup_ref) or feedback.get(h.get("ref", ""))
            if vote == "up":
                h["feedback"] = "up"
                h["score"] = _multiply_score(h.get("score"), 0.5)
            elif vote == "down":
                h["feedback"] = "down"
                h["score"] = _multiply_score(h.get("score"), 2.0)

    # Rank: vector hits first (lowest distance), then graph mentions,
    # then keyword matches. Vector distance ~0.2 means "very close";
    # keyword hits get a synthetic score of 0.5 so they fall below
    # strong semantic matches but above weak ones. Feedback multipliers
    # nudge a hit up or down within its tier without crossing tiers.
    def _rank_key(h: dict[str, Any]) -> tuple[int, float]:
        source_rank = {"vector": 0, "bm25": 0, "graph": 1, "sqlite": 2}.get(h["source"], 3)
        score = h.get("score")
        return (source_rank, score if isinstance(score, (int, float)) else 1.0)

    deduped.sort(key=_rank_key)
    final_hits = deduped[: max(1, k * 2)]
    _mark_semantic_accessed(
        user_id,
        [str(h.get("ref", "")) for h in final_hits if h.get("source") == "vector"],
    )
    return final_hits


def semantic_bm25_hits(
    user_id: str,
    query: str,
    k: int,
    config: RetrievalConfig | None = None,
) -> list[dict[str, Any]]:
    """Return lexical semantic-fact hits from the migration-7 FTS5 table.

    The table is standalone rather than external-content because
    ``semantic_facts`` has a composite text primary key. Missing FTS5 support or
    malformed user query syntax degrades to no lexical hits.

    For distractor-heavy queries we apply a term-frequency penalty: facts that
    match many query terms but miss the rarest (most discriminative) term are
    downweighted. This penalizes distractors that share vocabulary with the target
    without penalizing the target itself.
    """
    config = config or _load_retrieval_config()
    terms = _query_terms(query)
    if not terms:
        return []

    fts_query = " OR ".join(f'"{term}"' for term in terms)
    conn = _get_connection(_db_path())
    now = datetime.now(UTC)

    try:
        rows = conn.execute(
            """
            SELECT
                sf.fact_id,
                sf.text,
                sf.metadata,
                sf.valid_to,
                bm25(semantic_facts_fts) AS bm25_score
            FROM semantic_facts_fts
            JOIN semantic_facts AS sf
              ON sf.user_id = semantic_facts_fts.user_id
             AND sf.fact_id = semantic_facts_fts.fact_id
            WHERE semantic_facts_fts.user_id = ?
              AND semantic_facts_fts MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            (user_id, fts_query, max(1, k)),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if (
            "no such table" in message
            or "no such module" in message
            or "fts5" in message
            or "syntax error" in message
        ):
            logger.debug("recall: FTS5 unavailable for semantic BM25 (%s)", exc)
            return []
        raise

    current_rows = [row for row in rows if _is_currently_valid(row["valid_to"], now)]
    if not current_rows:
        return []

    scores = [float(row["bm25_score"]) for row in current_rows]
    lo = min(scores)
    hi = max(scores)

    # Compute IDF for query terms so we can apply a term-frequency penalty.
    # The rarest terms (highest IDF) are the most discriminative; facts that
    # match many common terms but miss the rarest one get downweighted.
    rare_term = _rarest_fts_term(conn, user_id, terms)

    hits: list[dict[str, Any]] = []
    for row in current_rows:
        bm25_score = float(row["bm25_score"])
        if hi == lo:
            bm25_relevance = 1.0
        else:
            # SQLite bm25() is lower-is-better, so invert to higher-is-better.
            bm25_relevance = (hi - bm25_score) / (hi - lo)

        # Apply term-frequency penalty for distractor-heavy queries: if the
        # rarest term doesn't appear in this fact's text, reduce the relevance
        # score. This penalizes distractors that share vocabulary with the target
        # without penalizing the target itself.
        if rare_term and rare_term not in str(row["text"] or "").lower():
            tf_penalty = 0.7  # empirically chosen
        else:
            tf_penalty = 1.0

        score_with_penalty = (1.0 - bm25_relevance) * tf_penalty
        bm25_relevance_with_penalty = 1.0 - score_with_penalty

        hits.append(
            {
                "source": "bm25",
                "text": row["text"],
                "kind": str(_loads_metadata(row["metadata"]).get("kind", "fact")),
                "ref": row["fact_id"],
                "score": score_with_penalty,
                "bm25": bm25_score,
                "bm25_relevance": bm25_relevance_with_penalty,
                "time_score": _temporal_prior(row["valid_to"], now, config),
                "retrieval": "bm25",
            }
        )
    return hits


def _rarest_fts_term(conn: sqlite3.Connection, user_id: str, terms: list[str]) -> str | None:
    """Return the rarest term among ``terms`` in the FTS5 index for this user.

    Uses the FTS5 vocabulary table ``semantic_facts_fts_vocab``, which is created
    automatically when FTS5 is compiled with the default config. If the vocab
    table doesn't exist or any term is missing, returns None.
    """
    rare: str | None = None
    min_count: int | None = None
    for term in terms:
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM semantic_facts_fts_vocab "
                "WHERE term=? AND cnt > 0",
                (term,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            continue
        cnt: int = int(row["cnt"] or 0)
        if cnt == 0:
            continue
        if min_count is None or cnt < min_count:
            min_count = cnt
            rare = term
    return rare


def sqlite_keyword_hits(sqlite: Any, query: str, k: int) -> list[dict[str, Any]]:
    """Scan structured tables for rows that overlap with the query."""
    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z']{2,}", query.lower()) if t]
    if not tokens:
        return []
    results: list[dict[str, Any]] = []
    mem = sqlite

    def _match(text: str) -> bool:
        low = text.lower()
        return any(tok in low for tok in tokens)

    for goal in mem.get_goals(limit=30):
        blob = " ".join(str(goal.get(f, "")) for f in ("title", "next_step", "category"))
        if _match(blob):
            results.append(
                {
                    "source": "sqlite",
                    "text": f"Goal — {goal.get('title', '')}: {goal.get('next_step', '')}".strip(),
                    "kind": "goal",
                    "ref": f"goal:{goal.get('title', '')}",
                    "score": 0.5,
                }
            )
    for loop in mem.get_open_loops(status="", limit=30):
        blob = " ".join(str(loop.get(f, "")) for f in ("topic", "next_step"))
        if _match(blob):
            results.append(
                {
                    "source": "sqlite",
                    "text": f"Open loop — {loop.get('topic', '')}: {loop.get('next_step', '')}".strip(),
                    "kind": "open_loop",
                    "ref": f"open_loop:{loop.get('topic', '')}",
                    "score": 0.5,
                }
            )
    for pref in mem.get_preferences(limit=50):
        blob = " ".join(str(pref.get(f, "")) for f in ("category", "value", "context"))
        if _match(blob):
            results.append(
                {
                    "source": "sqlite",
                    "text": f"Preference ({pref.get('category', '')}): {pref.get('value', '')}",
                    "kind": "preference",
                    "ref": f"preference:{pref.get('category', '')}:{pref.get('value', '')}",
                    "score": 0.5,
                }
            )
    for rel in mem.get_relationship_profiles():
        blob = " ".join(str(rel.get(f, "")) for f in ("person", "relationship", "summary"))
        if _match(blob):
            results.append(
                {
                    "source": "sqlite",
                    "text": f"Relationship — {rel.get('person', '')} ({rel.get('relationship', '')}): {rel.get('summary', '')}",
                    "kind": "relationship",
                    "ref": f"relationship:{rel.get('person', '')}",
                    "score": 0.5,
                }
            )
    for entry in mem.get_journal(limit=10):
        text = str(entry.get("entry", ""))
        if _match(text):
            results.append(
                {
                    "source": "sqlite",
                    "text": f"Journal: {text[:140]}",
                    "kind": "journal",
                    "ref": f"journal:{entry.get('id', '')}",
                    "score": 0.6,
                }
            )
    return results[: k * 2]


def _fuse_semantic_hits(
    vector_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    query_entity_labels: list[str],
    config: RetrievalConfig,
) -> list[dict[str, Any]]:
    """Fuse vector and BM25 semantic facts with reciprocal rank fusion."""
    by_ref: dict[str, dict[str, Any]] = {}
    rrf_scores: dict[str, float] = {}

    for channel, ranked in (("vector", vector_hits), ("bm25", bm25_hits)):
        for rank, hit in enumerate(ranked, start=1):
            ref = str(hit.get("ref", ""))
            if not ref:
                continue
            by_ref.setdefault(ref, dict(hit))
            rrf_scores[ref] = rrf_scores.get(ref, 0.0) + 1.0 / (config.rrf_k + rank)
            by_ref[ref][f"{channel}_rank"] = rank
            if channel == "bm25":
                by_ref[ref]["bm25"] = hit.get("bm25")
                by_ref[ref]["bm25_relevance"] = hit.get("bm25_relevance")

    fused: list[tuple[float, dict[str, Any]]] = []
    for ref, hit in by_ref.items():
        entity_score = _entity_overlap_score(str(hit.get("text", "")), query_entity_labels)
        time_score = float(hit.get("time_score") or 1.0)
        valid_from_score = float(hit.get("valid_from_score") or 1.0)
        # Four-signal fusion: RRF × entity overlap × temporal expiry × valid_from recency.
        # time_score penalizes expired facts; valid_from_score boosts recently-created facts.
        # These are distinct signals: time_score is "when did it stop being true",
        # valid_from_score is "when did it become true".
        fused_score = (
            rrf_scores[ref]
            * (1.0 + config.entity_weight * entity_score)
            * time_score
            * valid_from_score
        )
        hit["rrf"] = fused_score
        hit["entity_score"] = entity_score
        hit["time_score"] = time_score
        hit["valid_from_score"] = valid_from_score
        # Keep the public score distance-like because feedback and final sorting
        # expect lower-is-better.
        hit["score"] = 1.0 / max(fused_score, 1e-12)
        fused.append((fused_score, hit))

    fused.sort(key=lambda item: item[0], reverse=True)
    return [hit for _score, hit in fused]


def _entity_overlap_score(text: str, labels: list[str]) -> float:
    low = text.lower()
    return float(sum(1 for label in labels if label and label.lower() in low))


def _mark_semantic_accessed(user_id: str, fact_ids: list[str]) -> None:
    ids = sorted({fact_id for fact_id in fact_ids if fact_id})
    if not ids:
        return
    conn = _get_connection(_db_path())
    now_iso = datetime.now().isoformat()
    conn.executemany(
        "UPDATE semantic_facts SET access_count = access_count + 1, last_accessed = ? "
        "WHERE user_id=? AND fact_id=?",
        [(now_iso, user_id, fact_id) for fact_id in ids],
    )
    conn.commit()


def _load_retrieval_config() -> RetrievalConfig:
    global _LOGGED_CONFIG
    config = RetrievalConfig.load()
    if _LOGGED_CONFIG != config:
        logger.info(
            "retrieval config: candidate_pool=%d rrf_k=%d entity_weight=%.3f "
            "temporal_half_life_days=%.1f",
            config.candidate_pool,
            config.rrf_k,
            config.entity_weight,
            config.temporal_half_life_days,
        )
        _LOGGED_CONFIG = config
    return config


# Function words carry no discriminative signal but are extremely common in the
# way people actually talk to June ("what is my diet?", "do I still work there").
# The FTS query ORs its terms together, so leaving these in floods the capped
# candidate pool with facts that merely contain "my" or "is" — and because RRF
# credits a document once per channel it appears in, that flood then outranks
# the fact the user actually asked for. Measured on the golden corpus, keeping
# them costs real recall (see docs/product/retrieval-benchmark.md).
#
# Deliberately small and English-only: this is a precision filter for the
# lexical channel, not a linguistics project. A query made entirely of stop
# words still searches for them rather than returning nothing.
_STOPWORDS = frozenset({
    "a", "about", "all", "am", "an", "and", "any", "are", "as", "at", "be",
    "been", "but", "by", "can", "could", "did", "do", "does", "for", "from",
    "get", "had", "has", "have", "he", "her", "here", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "just", "me", "mine", "my",
    "of", "on", "or", "our", "out", "over", "she", "should", "so", "some",
    "still", "such", "tell", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "to", "up", "us", "was", "we",
    "were", "what", "when", "where", "which", "who", "whom", "why", "will",
    "with", "would", "you", "your", "yours",
})


def _query_terms(query: str) -> list[str]:
    # Keep the FTS query simple and safe: quoted OR terms avoid user-provided
    # operators and still work for short keyword-style lookups.
    seen: set[str] = set()
    terms: list[str] = []
    for term in re.findall(r"[\w][\w'-]{1,}", query.lower(), flags=re.UNICODE):
        term = term.strip("'-")
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)

    content = [term for term in terms if term not in _STOPWORDS]
    # Fall back to the raw terms when a query is nothing but function words,
    # so "who is who" still searches instead of silently returning nothing.
    return (content or terms)[:12]


def _loads_metadata(raw: object) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _is_currently_valid(valid_to: object, now: datetime) -> bool:
    value = str(valid_to or "").strip()
    if not value:
        return True
    parsed = _parse_temporal(value)
    if parsed is None:
        return True
    return parsed > now


def _temporal_prior(
    valid_to: object,
    now: datetime,
    config: RetrievalConfig,
) -> float:
    value = str(valid_to or "").strip()
    if not value:
        return 1.0
    parsed = _parse_temporal(value)
    if parsed is None or parsed > now:
        return 1.0
    age_days = max(0.0, (now - parsed).total_seconds() / 86_400.0)
    return max(0.1, 0.5 ** (age_days / config.temporal_half_life_days))


def _valid_from_prior(
    valid_from: object,
    now: datetime,
    config: RetrievalConfig,
) -> float:
    """Recency prior based on when a fact became true in the world.

    Facts with a recent valid_from get a small boost; facts with no valid_from
    are treated as always-valid (no boost or penalty). This is the fourth signal
    in ADR 0024's four-signal fusion — it ranks by *when the fact became true*,
    distinct from salience's recency which ranks by *when June last saw it*.
    """
    value = str(valid_from or "").strip()
    if not value:
        return 1.0
    parsed = _parse_temporal(value)
    if parsed is None:
        return 1.0
    # Facts that became true more recently get a boost. The half-life matches
    # the valid_to decay so the two temporals are symmetric.
    age_days = max(0.0, (now - parsed).total_seconds() / 86_400.0)
    return max(0.5, 0.5 ** (age_days / config.temporal_half_life_days))


def _parse_temporal(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _salience_rerank(
    user_id: str,
    raw_hits: list[dict[str, Any]],
    k: int,
    config: RetrievalConfig | None = None,
    update_access: bool = True,
) -> list[dict[str, Any]]:
    """Re-rank vector hits by salience; update access bookkeeping for returned hits.

    For each candidate we read (access_count, last_accessed) from the
    semantic_facts shadow row, compute the salience score, sort DESC, take
    top-k, then UPDATE each returned row's access counters.
    """
    config = config or _load_retrieval_config()
    weights = SalienceWeights.load()
    conn = _get_connection(_db_path())
    now = datetime.now()
    validity_now = datetime.now(UTC)

    # Batch-fetch shadow-row data for all candidate fact_ids in one query,
    # avoiding N round-trips to SQLite.
    fact_ids = [str(v["fact_id"]) for v in raw_hits]
    shadow_rows: dict[str, dict[str, Any]] = {}
    try:
        query = (
            "SELECT fact_id, access_count, last_accessed, valid_to, valid_from "
            "FROM semantic_facts WHERE user_id=? AND fact_id IN ({})".format(",".join("?" for _ in fact_ids))
        )
        for row in conn.execute(query, (user_id, *fact_ids)).fetchall():
            shadow_rows[str(row["fact_id"])] = dict(row)
    except Exception:  # noqa: BLE001
        logger.exception("recall: batch shadow row fetch failed")

    scored: list[tuple[float, dict[str, float], dict[str, Any], float, float]] = []
    for v in raw_hits:
        fact_id = str(v["fact_id"])
        row = shadow_rows.get(fact_id)
        if row:
            if not _is_currently_valid(row.get("valid_to"), validity_now):
                continue
            access_count: int = int(row.get("access_count") or 0)
            last_accessed_str: str = str(row.get("last_accessed") or "")
            time_score = _temporal_prior(row.get("valid_to"), validity_now, config)
            valid_from_score = _valid_from_prior(row.get("valid_from"), validity_now, config)
        else:
            access_count = 0
            last_accessed_str = ""
            time_score = 1.0
            valid_from_score = 1.0

        if last_accessed_str:
            try:
                last_dt = datetime.fromisoformat(last_accessed_str)
                hours_since = (now - last_dt).total_seconds() / 3600.0
            except ValueError:
                hours_since = 0.0
        else:
            hours_since = 0.0

        rel = relevance_from_distance(v.get("distance"))
        score, components = salience_detailed(rel, hours_since, access_count, weights)
        scored.append((score, components, v, time_score, valid_from_score))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:k]

    if update_access:
        _mark_semantic_accessed(user_id, [str(v["fact_id"]) for _s, _c, v, _t, _vfs in top])

    result: list[dict[str, Any]] = []
    for score, components, v, time_score, valid_from_score in top:
        result.append(
            {
                "source": "vector",
                "text": v["text"],
                "kind": str(v.get("metadata", {}).get("kind", "fact")),
                "ref": v["fact_id"],
                "score": max(0.0, 1.0 - score),
                "recency": components["recency"],
                "frequency": components["frequency"],
                "relevance": components["relevance"],
                "time_score": time_score,
                "valid_from_score": valid_from_score,
                "retrieval": "vector",
            }
        )
    return result


def salience_recall(user_id: str, query: str, k: int = 5) -> list[dict[str, Any]]:
    """Convenience adapter: salience-ranked recall for a given user.

    Returns the same list[dict] shape as MemoryManager.recall().
    Intended as the assembler's recall hook for C.3 and later tasks.
    """
    from .manager import MemoryManager

    mm = MemoryManager(user_id)
    return mm.recall(query, k=k)


def prefixed_ref(hit: dict[str, Any]) -> str:
    """The prefixed ref for a raw recall hit — the one implementation of the rule.

    Vector and graph hits arrive from their stores with bare ids; sqlite hits
    arrive already prefixed (``goal:...``). Every surface that hands a recalled
    memory onward — the feedback table, the loop's UI disclosure, the ``forget``
    tool — needs the same prefix scheme ``/memory`` uses, so they all call this
    rather than restating it. It had three copies before D.5a; one of them
    already carried a docstring pointing at a module that did not have it.
    """
    source = hit.get("source")
    raw = hit.get("ref", "") or ""
    kind = hit.get("kind", "") or ""
    if source == "vector":
        return f"semantic:{raw}"
    if source == "graph":
        return f"edge:{raw}" if kind.startswith("edge:") else f"node:{raw}"
    return raw


def _hit_lookup_ref(hit: dict[str, Any]) -> str:
    """Back-compat alias for :func:`prefixed_ref`; the feedback table keys on it."""
    return prefixed_ref(hit)


def _multiply_score(score: Any, factor: float) -> float:
    """Scale a recall score by a factor, treating None / non-numeric as 0."""
    if isinstance(score, (int, float)):
        return float(score) * factor
    return 0.0
