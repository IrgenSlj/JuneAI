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

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Any

from .graph import KnowledgeGraph, _slug
from .paraphrase import (
    _paraphrase_body_metric,
    _paraphrase_calendar,
    _paraphrase_goal,
    _paraphrase_journal,
    _paraphrase_mood,
    _paraphrase_open_loop,
)
from .recall import gather_hits, sqlite_keyword_hits
from .sqlite import Memory
from .vector import VectorStore

logger = logging.getLogger(__name__)

_EXTRACTOR_PROMPT_PATH = Path(__file__).parent / "extractor_prompt.txt"
_LOCAL_EXTRACTOR_ROLES = ("local-fast", "local-deep")
_LOCAL_EXTRACTOR_MAX_TOKENS = 2048


class _LocalExtractorUnavailable(RuntimeError):
    """Raised when memory extraction cannot run without crossing the local boundary."""


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
        return gather_hits(self.vector, self.graph, self.sqlite, self.user_id, query, k)

    # ------------------------------------------------------------------
    # Feedback pass-through (B.4)
    # ------------------------------------------------------------------

    def set_feedback(self, ref: str, vote: str) -> dict | None:
        """Record an up/down vote on a memory by ref."""
        return self.sqlite.set_feedback(ref, vote)

    def clear_feedback(self, ref: str) -> bool:
        return self.sqlite.clear_feedback(ref)

    def _sqlite_keyword_hits(self, query: str, k: int) -> list[dict[str, Any]]:
        """Scan structured tables for rows that overlap with the query.

        Retained as a method so the keyword path stays a documented recall
        fallback surface; delegates to ``recall.sqlite_keyword_hits``.
        """
        return sqlite_keyword_hits(self.sqlite, query, k)

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
            logger.exception("memory.write: %s handler failed", kind)
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
        except Exception:  # noqa: BLE001
            logger.exception("memory.write: vector paraphrase sync failed for %s", ref)
            return False

    def _delete_structured_vector(self, ref: str) -> int:
        try:
            return self.vector.delete_by_ref(ref)
        except Exception:  # noqa: BLE001
            logger.exception("memory.forget: vector paraphrase delete failed for %s", ref)
            return 0

    def _delete_structured_vector_prefix(self, ref_prefix: str) -> int:
        try:
            return self.vector.delete_by_ref_prefix(ref_prefix)
        except Exception:  # noqa: BLE001
            logger.exception(
                "memory.forget: vector paraphrase prefix delete failed for %s",
                ref_prefix,
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
        real model. Without an injected callable, extraction uses a local-only
        provider from the registry and skips gracefully if none is available.
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
        except _LocalExtractorUnavailable as exc:
            logger.info("memory.extract: %s", exc)
            return {"facts": 0, "entities": 0, "relations": 0, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("memory.extract: llm_call failed")
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
    """Invoke a local-only provider for memory extraction.

    Kept lazy so tests that inject their own ``llm_call`` don't need the
    provider package. This path intentionally ignores the active chat runtime:
    post-chat extraction must not make a silent cloud call when the visible
    turn used Gemini.
    """
    from ..providers import GenerateRequest, Message

    provider = _local_extractor_provider()

    async def _generate() -> str:
        result = await provider.generate(
            GenerateRequest(
                messages=[Message(role="user", content=prompt)],
                max_tokens=_LOCAL_EXTRACTOR_MAX_TOKENS,
                temperature=0.0,
                response_format="json",
            )
        )
        return result.text

    return str(_run_async_blocking(_generate))


def _local_extractor_provider() -> Any:
    """Return an available local provider, or raise with a skip-safe error."""
    from ..providers.registry import get_registry

    registry = get_registry()
    errors: list[str] = []
    for role in _LOCAL_EXTRACTOR_ROLES:
        try:
            provider = registry.get(role)
        except KeyError as exc:
            errors.append(f"{role}: {exc}")
            continue

        tier = str(getattr(provider, "tier", ""))
        if not tier.startswith("local-"):
            errors.append(f"{role}: resolved to non-local tier {tier!r}")
            continue

        try:
            health = _run_async_blocking(provider.health)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{role}: health check failed: {exc}")
            continue

        detail = str(getattr(health, "detail", "") or "").strip()
        if not getattr(health, "reachable", False):
            suffix = f" ({detail})" if detail else ""
            errors.append(f"{role}: not reachable{suffix}")
            continue
        if not getattr(health, "loaded", False):
            suffix = f" ({detail})" if detail else ""
            errors.append(f"{role}: model not available locally{suffix}")
            continue
        return provider

    reason = "; ".join(errors) if errors else "no local provider roles configured"
    raise _LocalExtractorUnavailable(
        f"Local memory extractor unavailable; skipping extraction. {reason}"
    )


def _run_async_blocking(coro_factory: Callable[[], Awaitable[Any]]) -> Any:
    """Run async provider calls from the synchronous MemoryManager API."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro_factory())).result()  # type: ignore[arg-type]
