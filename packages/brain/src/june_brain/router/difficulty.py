"""Difficulty classifier — tags each request and picks a provider tier.

Roles returned by tier_for_difficulty are the brain's stable role vocabulary:
"local-fast", "local-deep", "cloud-capable".  This module never emits
"cloud-capable" automatically; creative work defaults to "local-deep".
"""

from __future__ import annotations

import re
from typing import Literal

Difficulty = Literal["trivial", "standard", "hard", "creative"]

_CREATIVE_CUES = re.compile(
    r"\b(write|draft|story|poem|imagine|brainstorm)\b", re.IGNORECASE
)
_HARD_CUES = re.compile(
    r"\b(then|after that|plan|compare|step by step)\b", re.IGNORECASE
)
_MULTI_QUESTION = re.compile(r"\?.*\?", re.DOTALL)


def heuristic_difficulty(text: str) -> Difficulty:
    stripped = text.strip()
    word_count = len(stripped.split())

    if word_count <= 4 or stripped.lower() in {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "yes",
        "no",
        "bye",
        "good morning",
        "good evening",
        "good night",
    }:
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


async def classify_difficulty(
    text: str,
    *,
    registry=None,
    role: str = "local-fast",
) -> Difficulty:
    _LABELS: set[str] = {"trivial", "standard", "hard", "creative"}

    try:
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
                        "You are a difficulty classifier. "
                        "Reply with exactly one word — one of: trivial, standard, hard, creative. "
                        "No explanation, no punctuation, just the single label."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Classify this request: {text}",
                ),
            ],
            max_tokens=8,
            temperature=0.0,
        )
        result = await provider.generate(req)
        label = result.text.strip().lower().rstrip(".")
        if label in _LABELS:
            return label  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        pass

    return heuristic_difficulty(text)


def tier_for_difficulty(d: Difficulty) -> str:
    if d in ("trivial", "standard"):
        return "local-fast"
    return "local-deep"
