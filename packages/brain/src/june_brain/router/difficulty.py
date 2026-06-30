"""Difficulty classifier — tags each request and picks a provider tier.

Routing is the privacy/cost chokepoint, so the primary classifier is a single
constrained call to ``local-fast`` (language-agnostic by construction), cached
and time-boxed, with the English-regex heuristic as the degradation path
(invariant 6). Roles returned by ``tier_for_difficulty`` are the brain's stable
role vocabulary: "local-fast", "local-deep", "cloud-capable". This module never
emits "cloud-capable" automatically; creative work defaults to "local-deep".
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

Difficulty = Literal["trivial", "standard", "hard", "creative"]
ClassifierSource = Literal["model", "heuristic", "cache"]

_LABELS: set[str] = {"trivial", "standard", "hard", "creative"}

_CREATIVE_CUES = re.compile(
    r"\b(write|draft|story|poem|imagine|brainstorm)\b", re.IGNORECASE
)
_HARD_CUES = re.compile(
    r"\b(then|after that|plan|compare|step by step)\b", re.IGNORECASE
)
_MULTI_QUESTION = re.compile(r"\?.*\?", re.DOTALL)

# Short acknowledgements / greetings across the owner's languages (EN/NL/EL).
# The heuristic is only the fallback; the model classifier handles the long
# tail of languages. Most of these are also caught by the <=4-word rule.
_GREETINGS: set[str] = {
    # English
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "yes", "no",
    "bye", "good morning", "good evening", "good night",
    # Dutch
    "hoi", "hallo", "dank je", "dankjewel", "bedankt", "ja", "nee", "doei",
    "goedemorgen", "goedenavond", "goedenacht",
    # Greek
    "γεια", "γεια σου", "γεια σας", "καλημέρα", "καλησπέρα", "καληνύχτα",
    "ευχαριστώ", "ναι", "όχι", "αντίο",
}


@dataclass(frozen=True)
class DifficultyResult:
    """A classification plus where it came from (for provenance)."""

    label: Difficulty
    source: ClassifierSource
    # Populated only when source == "model"; None for cache/heuristic paths
    # (no LLM call happened on those paths so reporting metadata would be dishonest).
    model_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


# Bounded LRU cache keyed on the normalized message. Only successful model
# classifications are cached, so a transient model failure can later succeed.
_CACHE_MAX = 256
_CACHE: OrderedDict[str, Difficulty] = OrderedDict()


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _cache_get(key: str) -> Difficulty | None:
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return None


def _cache_put(key: str, label: Difficulty) -> None:
    _CACHE[key] = label
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def reset_cache() -> None:
    """Clear the classifier cache (tests / settings changes)."""
    _CACHE.clear()


def heuristic_difficulty(text: str) -> Difficulty:
    stripped = text.strip()
    word_count = len(stripped.split())

    if word_count <= 4 or stripped.lower() in _GREETINGS:
        return "trivial"

    if (
        word_count > 40
        or _MULTI_QUESTION.search(stripped)
        or _HARD_CUES.search(stripped)
    ):
        return "hard"

    if _CREATIVE_CUES.search(stripped):
        return "creative"

    return "standard"


def _parse_label(raw: str) -> Difficulty | None:
    """Extract a difficulty label from model output (plain word or JSON)."""
    s = raw.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            s = str(obj.get("difficulty") or obj.get("label") or "")
    except (json.JSONDecodeError, TypeError):
        pass
    cleaned = s.strip().lower().strip(".!,\"' ")
    if cleaned in _LABELS:
        return cleaned  # type: ignore[return-value]
    for token in cleaned.replace("\n", " ").split():
        token = token.strip(".!,\"'")
        if token in _LABELS:
            return token  # type: ignore[return-value]
    return None


async def _model_classify(
    text: str, registry, role: str
) -> tuple[Difficulty | None, Any]:
    """Call the model classifier and return (label, GenerateResult).

    Returns (None, result) when the model responds but the output cannot be
    parsed as a valid label; the caller degrades to the heuristic in that case.
    """
    if registry is None:
        from june_brain.providers.registry import get_registry

        registry = get_registry()
    provider = registry.get(role)

    from june_brain.providers.base import GenerateRequest, Message

    req = GenerateRequest(
        messages=[
            Message(
                role="system",
                content=(
                    "You are a difficulty classifier. Classify the user's request "
                    "as exactly one of: trivial, standard, hard, creative. "
                    'Reply with JSON: {"difficulty": "<label>"}. No other text.'
                ),
            ),
            Message(role="user", content=f"Classify this request: {text}"),
        ],
        max_tokens=16,
        temperature=0.0,
        response_format="json",
    )
    result = await provider.generate(req)
    return _parse_label(result.text), result


async def classify_difficulty_detailed(
    text: str,
    *,
    registry=None,
    role: str = "local-fast",
    timeout: float = 0.3,
    use_cache: bool = True,
) -> DifficultyResult:
    """Classify ``text``, preferring the model with the heuristic as fallback.

    Cached by normalized message; the model call is bounded by ``timeout`` so a
    slow local model never stalls a turn — it degrades to the heuristic.
    """
    key = _normalize(text)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return DifficultyResult(cached, "cache")

    try:
        label, gen_result = await asyncio.wait_for(
            _model_classify(text, registry, role), timeout=timeout
        )
        if label is not None:
            if use_cache:
                _cache_put(key, label)
            return DifficultyResult(
                label=label,
                source="model",
                model_id=gen_result.model_id if gen_result is not None else None,
                input_tokens=gen_result.input_tokens if gen_result is not None else None,
                output_tokens=gen_result.output_tokens if gen_result is not None else None,
                latency_ms=gen_result.latency_ms if gen_result is not None else None,
            )
    except Exception:  # noqa: BLE001 — timeout or any failure degrades to heuristic
        pass

    return DifficultyResult(heuristic_difficulty(text), "heuristic")


async def classify_difficulty(
    text: str,
    *,
    registry=None,
    role: str = "local-fast",
) -> Difficulty:
    """Back-compatible label-only classification (see ``*_detailed``)."""
    result = await classify_difficulty_detailed(text, registry=registry, role=role)
    return result.label


def tier_for_difficulty(d: Difficulty) -> str:
    if d in ("trivial", "standard"):
        return "local-fast"
    return "local-deep"
