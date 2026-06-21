"""Recall read-path: candidate gathering across the three stores + salience rerank.

``gather_hits`` fans out to the vector, graph, and SQLite stores, dedupes,
applies feedback multipliers, and ranks. ``_salience_rerank`` re-scores the
vector candidates by ``recency x frequency x relevance`` and updates access
bookkeeping. Extracted from ``manager.py`` (S3 decomposition) so the recall
logic lives in one focused module; ``MemoryManager.recall`` is a thin facade
over ``gather_hits``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from .paraphrase import _format_edge, _format_node
from .salience import SalienceWeights, relevance_from_distance, salience
from .sqlite import _get_connection
from .vector import _db_path

logger = logging.getLogger(__name__)


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

    hits: list[dict[str, Any]] = []

    # 1) Semantic (vector) — re-ranked by salience (recency × frequency × relevance).
    try:
        raw_vector = vector.search(query, k=k)
        if raw_vector:
            hits.extend(_salience_rerank(user_id, raw_vector, k))
    except Exception:  # noqa: BLE001
        logger.exception("recall: vector search failed")

    # 2) Graph — entities the query mentions, plus their neighbors.
    try:
        for node in graph.mentions_near(query, limit=k):
            hits.append(
                {
                    "source": "graph",
                    "text": _format_node(node),
                    "kind": f"entity:{node['kind']}",
                    "ref": node["node_id"],
                    "score": 0.0,
                }
            )
            for edge in graph.neighbors(node["node_id"], limit=3):
                hits.append(
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
        source_rank = {"vector": 0, "graph": 1, "sqlite": 2}.get(h["source"], 3)
        score = h.get("score")
        return (source_rank, score if isinstance(score, (int, float)) else 1.0)

    deduped.sort(key=_rank_key)
    return deduped[: max(1, k * 2)]


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


def _salience_rerank(
    user_id: str,
    raw_hits: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Re-rank vector hits by salience; update access bookkeeping for returned hits.

    For each candidate we read (access_count, last_accessed) from the
    semantic_facts shadow row, compute the salience score, sort DESC, take
    top-k, then UPDATE each returned row's access counters.
    """
    weights = SalienceWeights.load()
    conn = _get_connection(_db_path())
    now = datetime.now()

    scored: list[tuple[float, dict[str, Any]]] = []
    for v in raw_hits:
        fact_id = v["fact_id"]
        row = conn.execute(
            "SELECT access_count, last_accessed FROM semantic_facts "
            "WHERE user_id=? AND fact_id=?",
            (user_id, fact_id),
        ).fetchone()
        if row:
            access_count: int = int(row["access_count"] or 0)
            last_accessed_str: str = str(row["last_accessed"] or "")
        else:
            access_count = 0
            last_accessed_str = ""

        if last_accessed_str:
            try:
                last_dt = datetime.fromisoformat(last_accessed_str)
                hours_since = (now - last_dt).total_seconds() / 3600.0
            except ValueError:
                hours_since = 0.0
        else:
            hours_since = 0.0

        rel = relevance_from_distance(v.get("distance"))
        score = salience(rel, hours_since, access_count, weights)
        scored.append((score, v))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:k]

    now_iso = now.isoformat()
    for _score, v in top:
        conn.execute(
            "UPDATE semantic_facts SET access_count = access_count + 1, last_accessed = ? "
            "WHERE user_id=? AND fact_id=?",
            (now_iso, user_id, v["fact_id"]),
        )
    conn.commit()

    result: list[dict[str, Any]] = []
    for score, v in top:
        result.append(
            {
                "source": "vector",
                "text": v["text"],
                "kind": str(v.get("metadata", {}).get("kind", "fact")),
                "ref": v["fact_id"],
                # Store distance-like (lower = more salient): recall()'s final rank
                # sort is ascending and the feedback multipliers assume lower=better,
                # so a raw (higher=better) salience here would invert the ordering.
                "score": max(0.0, 1.0 - score),
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


def _hit_lookup_ref(hit: dict[str, Any]) -> str:
    """Build the prefixed ref the feedback table is keyed by, given a raw recall hit.

    Vector and graph hits arrive with bare ids; sqlite hits arrive
    already prefixed (``goal:...``). Mirrors graph._normalize_recall_hit.
    """
    source = hit.get("source")
    raw = hit.get("ref", "") or ""
    kind = hit.get("kind", "") or ""
    if source == "vector":
        return f"semantic:{raw}"
    if source == "graph":
        return f"edge:{raw}" if kind.startswith("edge:") else f"node:{raw}"
    return raw


def _multiply_score(score: Any, factor: float) -> float:
    """Scale a recall score by a factor, treating None / non-numeric as 0."""
    if isinstance(score, (int, float)):
        return float(score) * factor
    return 0.0
