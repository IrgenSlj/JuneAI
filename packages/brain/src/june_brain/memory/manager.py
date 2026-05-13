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
from hashlib import sha256
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

        # Apply feedback multipliers before ranking. Recall hit refs are not
        # yet prefixed for non-sqlite sources (vector returns the bare
        # fact_id, graph returns the bare node_id), so we re-derive the
        # prefixed form to match what the user voted on through the UI.
        try:
            feedback = self.sqlite.get_feedback_map()
        except Exception as exc:  # noqa: BLE001
            logger.warning("recall: feedback lookup failed: %s", exc)
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

    # ------------------------------------------------------------------
    # Feedback pass-through (B.4)
    # ------------------------------------------------------------------

    def set_feedback(self, ref: str, vote: str) -> dict | None:
        """Record an up/down vote on a memory by ref."""
        return self.sqlite.set_feedback(ref, vote)

    def clear_feedback(self, ref: str) -> bool:
        return self.sqlite.clear_feedback(ref)

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
                        "ref": f"journal:{entry.get('id', '')}",
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
    # Write — single entry point for all memory writes
    # ------------------------------------------------------------------

    def write(self, payload: dict[str, Any], source: str = "manual") -> dict[str, Any]:
        """Persist a memory across whichever stores apply.

        ``payload`` is ``{"kind": str, "fields": dict}``. ``kind`` selects
        the handler:

          - ``"fact"``        → vector store (text + metadata)
          - ``"entity"``      → graph node
          - ``"relation"``    → graph edge
          - ``"goal"`` / ``"open_loop"`` / ``"calendar"`` /
            ``"journal"`` / ``"body_metric"`` / ``"mood"`` →
              SQLite row + paraphrase upserted to the vector store so
              the structured row also feeds recall

        ``source`` becomes the vector-store ``source`` tag, e.g.
        ``"extraction"`` for facts pulled from chat or
        ``"skill:daily:save_journal_entry"`` for skill writes. The tag
        lets the UI explain where a recalled memory came from and lets
        future feature work filter by writer.

        Returns ``{"written": bool, "kind": str, "ref": str|None,
        "stores": list[str]}``. ``ref`` uses the same prefix scheme as
        ``/memory`` so callers can hand it to ``forget`` or render it as
        a deep link.
        """
        kind = str(payload.get("kind", "")).strip()
        fields = payload.get("fields") or {}
        if not isinstance(fields, dict):
            return {"written": False, "kind": kind, "ref": None, "stores": []}

        handler = _WRITE_HANDLERS.get(kind)
        if handler is None:
            return {"written": False, "kind": kind, "ref": None, "stores": []}
        try:
            return handler(self, fields, source)
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory.write: %s handler failed: %s", kind, exc)
            return {"written": False, "kind": kind, "ref": None, "stores": [], "error": str(exc)}

    # --- write handlers -------------------------------------------------

    def _write_fact(self, fields: dict[str, Any], source: str) -> dict[str, Any]:
        text = str(fields.get("text", "")).strip()
        if not text:
            return {"written": False, "kind": "fact", "ref": None, "stores": []}
        metadata = dict(fields.get("metadata") or {})
        metadata.setdefault("kind", "fact")
        record = self.vector.upsert(text=text, source=source, metadata=metadata)
        return {
            "written": True,
            "kind": "fact",
            "ref": f"semantic:{record['fact_id']}",
            "stores": ["vector"],
        }

    def _write_entity(self, fields: dict[str, Any], source: str) -> dict[str, Any]:
        label = str(fields.get("label", "")).strip()
        if not label:
            return {"written": False, "kind": "entity", "ref": None, "stores": []}
        kind = str(fields.get("kind", "entity")).strip() or "entity"
        props = dict(fields.get("props") or {})
        node_id = fields.get("node_id")
        node = self.graph.add_node(
            label=label,
            kind=kind,
            props=props,
            **({"node_id": node_id} if node_id else {}),
        )
        return {
            "written": True,
            "kind": "entity",
            "ref": f"node:{node['node_id']}",
            "stores": ["graph"],
            "node_id": node["node_id"],
        }

    def _write_relation(self, fields: dict[str, Any], source: str) -> dict[str, Any]:
        src = str(fields.get("src", "")).strip()
        dst = str(fields.get("dst", "")).strip()
        kind = str(fields.get("kind", "related_to")).strip() or "related_to"
        if not src or not dst:
            return {"written": False, "kind": "relation", "ref": None, "stores": []}
        props = dict(fields.get("props") or {})
        self.graph.add_edge(src=src, dst=dst, kind=kind, props=props)
        return {
            "written": True,
            "kind": "relation",
            "ref": f"edge:{src}|{dst}|{kind}",
            "stores": ["graph"],
        }

    def _write_structured(
        self,
        kind: str,
        fields: dict[str, Any],
        source: str,
        *,
        save: Callable[[], dict[str, Any]],
        ref_for: Callable[[dict[str, Any]], str],
        paraphrase: Callable[[dict[str, Any]], str],
    ) -> dict[str, Any]:
        """Common path: persist the structured row, then paraphrase to vector.

        The paraphrased fact carries ``kind`` and ``ref`` in metadata so
        recall can attribute the hit and the UI can render a back-link.
        Vector upsert failures don't fail the write — the structured row
        is the source of truth; the paraphrase is the recall convenience.
        """
        row = save()
        ref = ref_for(row)
        text = paraphrase(row).strip()
        stores = ["sqlite"]
        if self._sync_structured_vector(kind, ref, text, source):
            stores.append("vector")
        return {"written": True, "kind": kind, "ref": ref, "stores": stores}

    def _sync_structured_vector(
        self,
        kind: str,
        ref: str,
        text: str,
        source: str,
    ) -> bool:
        """Replace the vector paraphrase for one structured memory ref."""
        text = text.strip()
        try:
            self.vector.delete_by_ref(ref)
            if not text:
                return False
            self.vector.upsert(
                text=text,
                source=source,
                metadata={"kind": kind, "ref": ref},
                fact_id=_vector_fact_id(ref),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory.write: vector paraphrase sync failed for %s: %s", ref, exc)
            return False

    def _delete_structured_vector(self, ref: str) -> int:
        try:
            return self.vector.delete_by_ref(ref)
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory.forget: vector paraphrase delete failed for %s: %s", ref, exc)
            return 0

    def _delete_structured_vector_prefix(self, ref_prefix: str) -> int:
        try:
            return self.vector.delete_by_ref_prefix(ref_prefix)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "memory.forget: vector paraphrase prefix delete failed for %s: %s",
                ref_prefix,
                exc,
            )
            return 0

    def _write_goal(self, fields: dict[str, Any], source: str) -> dict[str, Any]:
        title = str(fields.get("title", "")).strip()
        if not title:
            return {"written": False, "kind": "goal", "ref": None, "stores": []}
        return self._write_structured(
            "goal",
            fields,
            source,
            save=lambda: self.sqlite.save_goal(
                title=title,
                category=str(fields.get("category", "personal")),
                target_date=str(fields.get("target_date", "")),
                next_step=str(fields.get("next_step", "")),
                status=str(fields.get("status", "active")),
            ),
            ref_for=lambda row: f"goal:{row.get('title', '')}",
            paraphrase=lambda row: _paraphrase_goal(row),
        )

    def _write_open_loop(self, fields: dict[str, Any], source: str) -> dict[str, Any]:
        topic = str(fields.get("topic", "")).strip()
        if not topic:
            return {"written": False, "kind": "open_loop", "ref": None, "stores": []}
        return self._write_structured(
            "open_loop",
            fields,
            source,
            save=lambda: self.sqlite.save_open_loop(
                topic=topic,
                next_step=str(fields.get("next_step", "")),
                due_date=str(fields.get("due_date", "")),
                status=str(fields.get("status", "open")),
            ),
            ref_for=lambda row: f"open_loop:{row.get('topic', '')}",
            paraphrase=lambda row: _paraphrase_open_loop(row),
        )

    def _write_calendar(self, fields: dict[str, Any], source: str) -> dict[str, Any]:
        title = str(fields.get("title", "")).strip()
        date = str(fields.get("date", "")).strip()
        if not title:
            return {"written": False, "kind": "calendar", "ref": None, "stores": []}
        return self._write_structured(
            "calendar",
            fields,
            source,
            save=lambda: self.sqlite.save_calendar_item(
                title=title,
                date=date,
                time=str(fields.get("time", "")),
                details=str(fields.get("details", "")),
                status=str(fields.get("status", "planned")),
                source=str(fields.get("source", "conversation")),
            ),
            ref_for=lambda row: f"calendar:{row.get('title', '')}|{row.get('date', '')}|{row.get('time', '')}",
            paraphrase=lambda row: _paraphrase_calendar(row),
        )

    def _write_journal(self, fields: dict[str, Any], source: str) -> dict[str, Any]:
        entry = str(fields.get("entry", "")).strip()
        if not entry:
            return {"written": False, "kind": "journal", "ref": None, "stores": []}
        # save_journal returns {entry, timestamp} but not the auto-id; fetch
        # the most recent to get it for the ref.
        self.sqlite.save_journal(entry)
        latest = self.sqlite.get_journal(limit=1)
        if not latest:
            return {"written": False, "kind": "journal", "ref": None, "stores": ["sqlite"]}
        row = latest[0]
        ref = f"journal:{row.get('id', '')}"
        text = _paraphrase_journal(row)
        stores = ["sqlite"]
        if self._sync_structured_vector("journal", ref, text, source):
            stores.append("vector")
        return {"written": True, "kind": "journal", "ref": ref, "stores": stores}

    def _write_body_metric(self, fields: dict[str, Any], source: str) -> dict[str, Any]:
        return self._write_structured(
            "body_metric",
            fields,
            source,
            save=lambda: self.sqlite.log_body_metrics(
                weight_kg=float(fields.get("weight_kg") or 0),
                sleep_hours=float(fields.get("sleep_hours") or 0),
                sleep_quality=int(fields.get("sleep_quality") or 0),
                energy=int(fields.get("energy") or 0),
                stress=int(fields.get("stress") or 0),
                soreness=int(fields.get("soreness") or 0),
                resting_hr=int(fields.get("resting_hr") or 0),
                steps=int(fields.get("steps") or 0),
                notes=str(fields.get("notes", "")),
            ),
            ref_for=lambda row: f"body_metric:{row.get('date', '')}",
            paraphrase=lambda row: _paraphrase_body_metric(row),
        )

    def _write_mood(self, fields: dict[str, Any], source: str) -> dict[str, Any]:
        mood = str(fields.get("mood", "")).strip()
        if not mood:
            return {"written": False, "kind": "mood", "ref": None, "stores": []}
        note = str(fields.get("note", ""))
        # Mood rows are append-only; the timestamp returned by log_mood is
        # the stable identifier for any future ref-based lookup.
        row = self.sqlite.log_mood(mood, note)
        ref = f"mood:{row.get('timestamp', '')}"
        text = _paraphrase_mood(row)
        stores = ["sqlite"]
        if self._sync_structured_vector("mood", ref, text, source):
            stores.append("vector")
        return {"written": True, "kind": "mood", "ref": ref, "stores": stores}

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
            result = self.write(
                {"kind": "fact", "fields": {"text": fact}},
                source="extraction",
            )
            if result.get("written"):
                fact_count += 1

        entity_count = 0
        name_to_node_id: dict[str, str] = {"user": f"person:{_slug(self.user_id)}"}
        # Ensure the user node exists so relations anchored on "user" resolve.
        self.write(
            {
                "kind": "entity",
                "fields": {
                    "label": self.user_id,
                    "kind": "person",
                    "node_id": name_to_node_id["user"],
                    "props": {"is_self": True},
                },
            },
            source="extraction",
        )

        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            props: dict[str, Any] = {}
            if entity.get("description"):
                props["description"] = str(entity["description"])
            result = self.write(
                {
                    "kind": "entity",
                    "fields": {
                        "label": name,
                        "kind": str(entity.get("kind", "entity")).strip() or "entity",
                        "props": props,
                    },
                },
                source="extraction",
            )
            if result.get("written"):
                name_to_node_id[name.lower()] = result.get("node_id", "")
                entity_count += 1

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
            result = self.write(
                {
                    "kind": "relation",
                    "fields": {"src": src_id, "dst": dst_id, "kind": kind},
                },
                source="extraction",
            )
            if result.get("written"):
                relation_count += 1

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
            removed_sqlite = self.sqlite.delete_goal(title)
            removed_vector = self._delete_structured_vector(ref)
            return removed_sqlite or removed_vector > 0
        if ref.startswith("open_loop:"):
            topic = ref.removeprefix("open_loop:")
            removed_sqlite = self.sqlite.delete_open_loop(topic)
            removed_vector = self._delete_structured_vector(ref)
            return removed_sqlite or removed_vector > 0
        if ref.startswith("calendar:"):
            body = ref.removeprefix("calendar:")
            parts = body.split("|", 2)
            title = parts[0]
            date = parts[1] if len(parts) > 1 else ""
            time = parts[2] if len(parts) > 2 else ""
            removed_sqlite = self.sqlite.delete_calendar_item(title, date, time)
            if len(parts) > 1:
                removed_vector = self._delete_structured_vector(ref)
            else:
                removed_vector = self._delete_structured_vector(ref)
                removed_vector += self._delete_structured_vector_prefix(f"{ref}|")
            return removed_sqlite or removed_vector > 0
        if ref.startswith("journal:"):
            entry_id = ref.removeprefix("journal:")
            try:
                removed_sqlite = self.sqlite.delete_journal_entry(int(entry_id))
            except ValueError:
                return False
            removed_vector = self._delete_structured_vector(ref)
            return removed_sqlite or removed_vector > 0
        if ref.startswith("body_metric:"):
            date = ref.removeprefix("body_metric:")
            removed_sqlite = self.sqlite.delete_body_metric(date)
            removed_vector = self._delete_structured_vector(ref)
            return removed_sqlite or removed_vector > 0
        # Fall-through: treat as a vector fact_id.
        if self.vector.get(ref):
            self.vector.delete(ref)
            return True
        return False

    # ------------------------------------------------------------------
    # Update — patch a structured row by ref
    # ------------------------------------------------------------------

    def update(
        self,
        ref: str,
        fields: dict[str, str],
        source: str = "manual",
    ) -> dict | None:
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
            row = self.sqlite.update_goal(old_title, **fields)
            if row is not None:
                new_ref = f"goal:{row.get('title', '')}"
                if new_ref != ref:
                    self._delete_structured_vector(ref)
                self._sync_structured_vector(
                    "goal",
                    new_ref,
                    _paraphrase_goal(row),
                    source,
                )
            return row
        if ref.startswith("open_loop:"):
            old_topic = ref.removeprefix("open_loop:")
            row = self.sqlite.update_open_loop(old_topic, **fields)
            if row is not None:
                new_ref = f"open_loop:{row.get('topic', '')}"
                if new_ref != ref:
                    self._delete_structured_vector(ref)
                self._sync_structured_vector(
                    "open_loop",
                    new_ref,
                    _paraphrase_open_loop(row),
                    source,
                )
            return row
        if ref.startswith("calendar:"):
            body = ref.removeprefix("calendar:")
            parts = body.split("|", 2)
            old_title = parts[0]
            old_date = parts[1] if len(parts) > 1 else ""
            old_time = parts[2] if len(parts) > 2 else ""
            row = self.sqlite.update_calendar_item(old_title, old_date, old_time, **fields)
            if row is not None:
                new_ref = (
                    f"calendar:{row.get('title', '')}|"
                    f"{row.get('date', '')}|{row.get('time', '')}"
                )
                if new_ref != ref:
                    self._delete_structured_vector(ref)
                self._sync_structured_vector(
                    "calendar",
                    new_ref,
                    _paraphrase_calendar(row),
                    source,
                )
            return row
        return None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _vector_fact_id(ref: str) -> str:
    digest = sha256(ref.encode("utf-8")).hexdigest()[:32]
    return f"structured-{digest}"


def _paraphrase_goal(row: dict[str, Any]) -> str:
    title = str(row.get("title", "")).strip()
    if not title:
        return ""
    next_step = str(row.get("next_step", "")).strip()
    target = str(row.get("target_date", "")).strip()
    parts = [f"Goal: {title}."]
    if next_step:
        parts.append(f"Next step: {next_step}.")
    if target:
        parts.append(f"Target date: {target}.")
    return " ".join(parts)


def _paraphrase_open_loop(row: dict[str, Any]) -> str:
    topic = str(row.get("topic", "")).strip()
    if not topic:
        return ""
    next_step = str(row.get("next_step", "")).strip()
    due = str(row.get("due_date", "")).strip()
    parts = [f"Open loop: {topic}."]
    if next_step:
        parts.append(f"Next step: {next_step}.")
    if due:
        parts.append(f"Due {due}.")
    return " ".join(parts)


def _paraphrase_calendar(row: dict[str, Any]) -> str:
    title = str(row.get("title", "")).strip()
    if not title:
        return ""
    date = str(row.get("date", "")).strip()
    time = str(row.get("time", "")).strip()
    details = str(row.get("details", "")).strip()
    parts = [f"Calendar item: {title}."]
    if date and time:
        parts.append(f"On {date} at {time}.")
    elif date:
        parts.append(f"On {date}.")
    if details:
        parts.append(details if details.endswith(".") else f"{details}.")
    return " ".join(parts)


def _paraphrase_journal(row: dict[str, Any]) -> str:
    entry = str(row.get("entry", "")).strip()
    if not entry:
        return ""
    return f"Journal entry: {entry}"


def _paraphrase_body_metric(row: dict[str, Any]) -> str:
    date = str(row.get("date", "")).strip()
    weight = row.get("weight_kg") or 0
    sleep = row.get("sleep_hours") or 0
    energy = row.get("energy") or 0
    stress = row.get("stress") or 0
    parts = []
    if weight:
        parts.append(f"weight {weight}kg")
    if sleep:
        parts.append(f"slept {sleep}h")
    if energy:
        parts.append(f"energy {energy}/5")
    if stress:
        parts.append(f"stress {stress}/5")
    if not parts:
        return ""
    head = f"Body check on {date}" if date else "Body check"
    return f"{head}: {', '.join(parts)}."


def _paraphrase_mood(row: dict[str, Any]) -> str:
    mood = str(row.get("mood", "")).strip()
    if not mood:
        return ""
    note = str(row.get("note", "")).strip()
    return f"Mood: {mood}. {note}".strip() if note else f"Mood: {mood}."


_WRITE_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "fact": MemoryManager._write_fact,
    "entity": MemoryManager._write_entity,
    "relation": MemoryManager._write_relation,
    "goal": MemoryManager._write_goal,
    "open_loop": MemoryManager._write_open_loop,
    "calendar": MemoryManager._write_calendar,
    "journal": MemoryManager._write_journal,
    "body_metric": MemoryManager._write_body_metric,
    "mood": MemoryManager._write_mood,
}


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
    from langchain_core.messages import HumanMessage

    from ..config import resolve_runtime_config
    from ..models import build_chat_model

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
