"""Post-turn fact extraction: prompt render, local-only provider call, JSON parse, and the typed-write fan-out for MemoryManager (S3 decomposition)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .graph import _slug

logger = logging.getLogger(__name__)

_EXTRACTOR_PROMPT_PATH = Path(__file__).parent / "extractor_prompt.txt"
_LOCAL_EXTRACTOR_ROLES = ("local-fast", "local-deep")
_LOCAL_EXTRACTOR_MAX_TOKENS = 2048


class _LocalExtractorUnavailable(RuntimeError):
    """Raised when memory extraction cannot run without crossing the local boundary."""


def extract(mgr, exchange, llm_call: Callable[[str], str] | None = None) -> dict[str, Any]:
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

    prompt = render_extractor_prompt(user_text, assistant_text)
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
        result = mgr.write(
            {"kind": "fact", "fields": {"text": fact}},
            source="extraction",
        )
        if result.get("written"):
            fact_count += 1

    entity_count = 0
    name_to_node_id: dict[str, str] = {"user": f"person:{_slug(mgr.user_id)}"}
    # Ensure the user node exists so relations anchored on "user" resolve.
    mgr.write(
        {
            "kind": "entity",
            "fields": {
                "label": mgr.user_id,
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
        result = mgr.write(
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
        src_id = resolve_node_id(mgr, src_name, name_to_node_id)
        dst_id = resolve_node_id(mgr, dst_name, name_to_node_id)
        result = mgr.write(
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


def resolve_node_id(mgr, name: str, cache: dict[str, str]) -> str:
    """Return a stable node_id for a name the extractor emitted."""
    low = name.lower()
    if low in cache:
        return cache[low]
    # Unknown name — create a bare node so the edge has both ends.
    node = mgr.graph.add_node(label=name, kind="entity")
    cache[low] = node["node_id"]
    return node["node_id"]


def render_extractor_prompt(user_text: str, assistant_text: str) -> str:
    template = _EXTRACTOR_PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{user_text}", user_text).replace(
        "{assistant_text}", assistant_text
    )


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
