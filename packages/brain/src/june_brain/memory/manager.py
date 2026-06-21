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
from pathlib import Path
from typing import Any

from . import writers
from .graph import KnowledgeGraph, _slug
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

        handler = writers.WRITE_HANDLERS.get(kind)
        if handler is None:
            return {"written": False, "kind": kind, "ref": None, "stores": []}
        try:
            return handler(self, fields, source)
        except Exception as exc:  # noqa: BLE001
            logger.exception("memory.write: %s handler failed", kind)
            return {"written": False, "kind": kind, "ref": None, "stores": [], "error": str(exc)}

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
        return writers.forget(self, ref)

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
        return writers.update(self, ref, fields, source)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

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
