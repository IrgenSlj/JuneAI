"""MemoryManager composes the three memory stores behind one interface.

Per ADR 0004, every June turn runs a recall/extract loop:

1. **Recall** — fan out to all three stores with the incoming message,
   merge and rank the hits, inject them into the system prompt.
2. **Generate** — the LLM speaks with that context in hand.
3. **Extract** — a small LLM pass pulls durable facts, entities, and
   relationships from the exchange and writes them back to the
   appropriate store(s).

Consumers (agent, API, tools) only ever touch ``MemoryManager``. The
three backing stores are implementation details.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from .graph import KnowledgeGraph, _slug
from .sqlite import Memory
from .vector import VectorStore

logger = logging.getLogger(__name__)

_EXTRACTOR_PROMPT_PATH = Path(__file__).parent / "extractor_prompt.txt"


class MemoryManager:
    """Single entry point for all memory reads/writes for a user."""

    def __init__(
        self,
        user_id: str,
        *,
        vector: VectorStore | None = None,
        graph: KnowledgeGraph | None = None,
        sqlite: Memory | None = None,
    ) -> None:
        self.user_id = user_id
        self.sqlite = sqlite or Memory(user_id)
        self.vector = vector if vector is not None else VectorStore(user_id)
        self.graph = graph if graph is not None else KnowledgeGraph(user_id)

    # ------------------------------------------------------------------
    # Recall — read path
    # ------------------------------------------------------------------

    def recall(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return up to ``k`` relevant memories across all three stores.

        Each hit is a dict with:
          - source: one of "vector", "graph", "sqlite"
          - text:   human-readable snippet to inject
          - kind:   sub-type for UI / filtering
          - ref:    stable identifier the caller can use to delete
          - score:  loose relevance score (lower = more relevant for
                    distance-based sources, higher = more relevant for
                    keyword hits).
        """
        query = (query or "").strip()
        if not query:
            return []

        hits: list[dict[str, Any]] = []

        # 1) Semantic (vector) — widest net, ranked by cosine distance.
        try:
            for v in self.vector.search(query, k=k):
                hits.append(
                    {
                        "source": "vector",
                        "text": v["text"],
                        "kind": str(v.get("metadata", {}).get("kind", "fact")),
                        "ref": v["fact_id"],
                        "score": v.get("distance"),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("recall: vector search failed: %s", exc)

        # 2) Graph — entities the query mentions, plus their neighbors.
        try:
            for node in self.graph.mentions_near(query, limit=k):
                hits.append(
                    {
                        "source": "graph",
                        "text": _format_node(node),
                        "kind": f"entity:{node['kind']}",
                        "ref": node["node_id"],
                        "score": 0.0,
                    }
                )
                for edge in self.graph.neighbors(node["node_id"], limit=3):
                    hits.append(
                        {
                            "source": "graph",
                            "text": _format_edge(node, edge),
                            "kind": f"edge:{edge['edge']['kind']}",
                            "ref": f"{edge['edge']['src']}|{edge['edge']['dst']}|{edge['edge']['kind']}",
                            "score": 0.1,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("recall: graph lookup failed: %s", exc)

        # 3) Structured (SQLite) — look for query terms across goals, open loops,
        # preferences, relationships, journal. Cheap keyword scan; the LLM gets
        # the top matches so it can notice "the user mentioned X weeks ago."
        try:
            hits.extend(self._sqlite_keyword_hits(query, k=k))
        except Exception as exc:  # noqa: BLE001
            logger.warning("recall: sqlite keyword scan failed: %s", exc)

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

        # Rank: vector hits first (lowest distance), then graph mentions,
        # then keyword matches. Vector distance ~0.2 means "very close";
        # keyword hits get a synthetic score of 0.5 so they fall below
        # strong semantic matches but above weak ones.
        def _rank_key(h: dict[str, Any]) -> tuple[int, float]:
            source_rank = {"vector": 0, "graph": 1, "sqlite": 2}.get(h["source"], 3)
            score = h.get("score")
            return (source_rank, score if isinstance(score, (int, float)) else 1.0)

        deduped.sort(key=_rank_key)
        return deduped[: max(1, k * 2)]

    def _sqlite_keyword_hits(self, query: str, k: int) -> list[dict[str, Any]]:
        """Scan structured tables for rows that overlap with the query."""
        tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z']{2,}", query.lower()) if t]
        if not tokens:
            return []
        results: list[dict[str, Any]] = []
        mem = self.sqlite

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
                        "ref": f"journal:{entry.get('timestamp', '')}",
                        "score": 0.6,
                    }
                )
        return results[: k * 2]

    def format_for_prompt(self, hits: list[dict[str, Any]]) -> str:
        """Render recall hits as a compact block for the system prompt."""
        if not hits:
            return ""
        lines = ["Relevant memories (most relevant first):"]
        for h in hits:
            prefix = {
                "vector": "• (semantic)",
                "graph": "• (graph)",
                "sqlite": "• (structured)",
            }.get(h["source"], "•")
            lines.append(f"{prefix} {h['text']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Extract — write path
    # ------------------------------------------------------------------

    def extract(
        self,
        exchange: dict[str, Any],
        llm_call: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        """Pull durable facts/entities/relations from one exchange; write them.

        ``exchange`` expects at minimum ``{"user": str, "assistant": str}``.
        ``llm_call`` takes a prompt string and returns the raw model text —
        injected so tests can run the full extraction logic without any
        real model. Production callers pass a thin wrapper around the
        configured chat model.
        """
        user_text = str(exchange.get("user", "")).strip()
        assistant_text = str(exchange.get("assistant", "")).strip()
        if not user_text and not assistant_text:
            return {"facts": 0, "entities": 0, "relations": 0}

        if llm_call is None:
            llm_call = _default_extractor_llm

        prompt = self._render_extractor_prompt(user_text, assistant_text)
        try:
            raw = llm_call(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory.extract: llm_call failed: %s", exc)
            return {"facts": 0, "entities": 0, "relations": 0, "error": str(exc)}

        payload = _parse_json_block(raw) or {}
        facts = payload.get("facts") or []
        entities = payload.get("entities") or []
        relations = payload.get("relations") or []

        fact_count = 0
        for fact in facts:
            if not isinstance(fact, str):
                continue
            text = fact.strip()
            if not text:
                continue
            try:
                self.vector.upsert(text=text, source="extraction", metadata={"kind": "fact"})
                fact_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory.extract: vector upsert failed: %s", exc)

        entity_count = 0
        name_to_node_id: dict[str, str] = {"user": f"person:{_slug(self.user_id)}"}
        # Ensure the user node exists so relations anchored on "user" resolve.
        try:
            self.graph.add_node(
                label=self.user_id,
                kind="person",
                node_id=name_to_node_id["user"],
                props={"is_self": True},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory.extract: failed to ensure user node: %s", exc)

        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            kind = str(entity.get("kind", "entity")).strip() or "entity"
            props: dict[str, Any] = {}
            if entity.get("description"):
                props["description"] = str(entity["description"])
            try:
                node = self.graph.add_node(label=name, kind=kind, props=props)
                name_to_node_id[name.lower()] = node["node_id"]
                entity_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory.extract: graph add_node failed for %s: %s", name, exc)

        relation_count = 0
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            src_name = str(relation.get("src", "")).strip()
            dst_name = str(relation.get("dst", "")).strip()
            kind = str(relation.get("kind", "related_to")).strip() or "related_to"
            if not src_name or not dst_name:
                continue
            src_id = self._resolve_node_id(src_name, name_to_node_id)
            dst_id = self._resolve_node_id(dst_name, name_to_node_id)
            try:
                self.graph.add_edge(src=src_id, dst=dst_id, kind=kind)
                relation_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory.extract: graph add_edge failed: %s", exc)

        return {
            "facts": fact_count,
            "entities": entity_count,
            "relations": relation_count,
        }

    def _resolve_node_id(self, name: str, cache: dict[str, str]) -> str:
        """Return a stable node_id for a name the extractor emitted."""
        low = name.lower()
        if low in cache:
            return cache[low]
        # Unknown name — create a bare node so the edge has both ends.
        node = self.graph.add_node(label=name, kind="entity")
        cache[low] = node["node_id"]
        return node["node_id"]

    def _render_extractor_prompt(self, user_text: str, assistant_text: str) -> str:
        template = _EXTRACTOR_PROMPT_PATH.read_text(encoding="utf-8")
        return template.replace("{user_text}", user_text).replace(
            "{assistant_text}", assistant_text
        )

    # ------------------------------------------------------------------
    # Delete — propagate to all stores
    # ------------------------------------------------------------------

    def forget(self, ref: str) -> bool:
        """Remove a fact by its ``ref`` (as returned from recall hits).

        Refs are prefixed with the source so we know which store to hit:
          - "semantic:<fact_id>" → vector + shadow table
          - "node:<node_id>"     → graph node (and all its edges)
          - "edge:<src>|<dst>|<kind>" → single graph edge
          - "goal:<title>"       → SQLite goals row
          - "open_loop:<topic>"  → SQLite open_loops row
          - "calendar:<title>" or "calendar:<title>|<date>|<time>"
                                 → SQLite calendar_items row
          - "journal:<id>"       → SQLite journal row
          - "body_metric:<date>" → SQLite body_metrics row
          - anything else falls through to the vector store by raw id
            (so the memory-browser UI can pass a fact_id directly).
        """
        ref = ref.strip()
        if not ref:
            return False
        if ref.startswith("semantic:"):
            fact_id = ref.removeprefix("semantic:")
            if not self.vector.get(fact_id):
                return False
            self.vector.delete(fact_id)
            return True
        if ref.startswith("node:"):
            node_id = ref.removeprefix("node:")
            if not self.graph.get_node(node_id):
                return False
            self.graph.remove_node(node_id)
            return True
        if ref.startswith("edge:"):
            body = ref.removeprefix("edge:")
            parts = body.split("|", 2)
            if len(parts) == 3:
                self.graph.remove_edge(parts[0], parts[1], parts[2])
                return True
            return False
        if ref.startswith("goal:"):
            title = ref.removeprefix("goal:")
            return self.sqlite.delete_goal(title)
        if ref.startswith("open_loop:"):
            topic = ref.removeprefix("open_loop:")
            return self.sqlite.delete_open_loop(topic)
        if ref.startswith("calendar:"):
            body = ref.removeprefix("calendar:")
            parts = body.split("|", 2)
            title = parts[0]
            date = parts[1] if len(parts) > 1 else ""
            time = parts[2] if len(parts) > 2 else ""
            return self.sqlite.delete_calendar_item(title, date, time)
        if ref.startswith("journal:"):
            entry_id = ref.removeprefix("journal:")
            try:
                return self.sqlite.delete_journal_entry(int(entry_id))
            except ValueError:
                return False
        if ref.startswith("body_metric:"):
            date = ref.removeprefix("body_metric:")
            return self.sqlite.delete_body_metric(date)
        # Fall-through: treat as a vector fact_id.
        if self.vector.get(ref):
            self.vector.delete(ref)
            return True
        return False

    # ------------------------------------------------------------------
    # Update — patch a structured row by ref
    # ------------------------------------------------------------------

    def update(self, ref: str, fields: dict[str, str]) -> dict | None:
        """Patch a structured-row memory by ``ref``.

        Returns the updated row dict (with the *new* fields), or ``None`` if
        the row was not found. Only the SQLite ref kinds are supported here:
        semantic facts and graph nodes have their own edit paths (re-upsert
        and add_node respectively).
        """
        ref = ref.strip()
        if not ref:
            return None
        if ref.startswith("goal:"):
            old_title = ref.removeprefix("goal:")
            return self.sqlite.update_goal(old_title, **fields)
        if ref.startswith("open_loop:"):
            old_topic = ref.removeprefix("open_loop:")
            return self.sqlite.update_open_loop(old_topic, **fields)
        if ref.startswith("calendar:"):
            body = ref.removeprefix("calendar:")
            parts = body.split("|", 2)
            old_title = parts[0]
            old_date = parts[1] if len(parts) > 1 else ""
            old_time = parts[2] if len(parts) > 2 else ""
            return self.sqlite.update_calendar_item(old_title, old_date, old_time, **fields)
        return None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _format_node(node: dict[str, Any]) -> str:
    desc = node.get("props", {}).get("description", "")
    label = node.get("label", "")
    kind = node.get("kind", "entity")
    if desc:
        return f"{label} ({kind}) — {desc}"
    return f"{label} ({kind})"


def _format_edge(source_node: dict[str, Any], edge_hit: dict[str, Any]) -> str:
    other = edge_hit.get("node", {})
    edge = edge_hit.get("edge", {})
    kind = str(edge.get("kind", "related_to")).replace("_", " ")
    other_label = other.get("label", "")
    return f"{source_node.get('label', '')} {kind} {other_label}".strip()


def _parse_json_block(raw: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of ``raw`` (tolerates fences)."""
    text = raw.strip()
    if not text:
        return None
    # Strip fenced code blocks.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    # Find the outermost object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _default_extractor_llm(prompt: str) -> str:
    """Invoke the configured runtime LLM for extraction.

    Kept lazy so tests that inject their own ``llm_call`` don't need the
    models package, and so importing ``MemoryManager`` doesn't eagerly
    resolve the runtime config (which would require an API key for the
    Gemini preset).
    """
    from ..config import resolve_runtime_config
    from ..models import build_chat_model
    from langchain_core.messages import HumanMessage

    runtime = resolve_runtime_config()
    llm = build_chat_model(runtime)
    response = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)
