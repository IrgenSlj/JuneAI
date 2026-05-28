"""Capability probe — measures local-model quality across four dimensions.

Optimistic default: all "good", checked_at="".  June runs full-quality
operations until a probe shows weakness; C.3/C.5 fallbacks then downgrade.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

logger = logging.getLogger(__name__)

Verdict = Literal["good", "weak", "poor"]

_NEEDLE_VALUE = "blue-42"
_FILLER = (
    "The sky is often described as a vast expanse. "
    "Scientists study atmospheric phenomena. "
    "Weather patterns vary across regions. "
    "Climate research continues worldwide. "
    "Many factors influence local conditions. "
    "Researchers publish findings regularly. "
    "Temperature records are kept meticulously. "
)
_LONG_CONTEXT_PASSAGE = (
    _FILLER
    + f"The secret code is {_NEEDLE_VALUE}. "
    + _FILLER
)

_SUMMARIZATION_PASSAGE = (
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. "
    "She was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes "
    "in two different sciences — physics and chemistry."
)
_SUMMARIZATION_KEY_TERMS = {"curie", "nobel", "radioactiv"}

_RELEVANCE_ITEMS = ["the French Revolution", "a recipe for pasta", "quantum entanglement"]
_RELEVANCE_QUERY = "physics"
_RELEVANCE_CORRECT = "quantum entanglement"

_STRUCTURED_KEYS = {"name", "age", "city"}


@dataclass
class CapabilityProfile:
    summarization: Verdict = "good"
    structured_output: Verdict = "good"
    long_context: Verdict = "good"
    relevance_scoring: Verdict = "good"
    checked_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CapabilityProfile:
        return cls(
            summarization=d.get("summarization", "good"),
            structured_output=d.get("structured_output", "good"),
            long_context=d.get("long_context", "good"),
            relevance_scoring=d.get("relevance_scoring", "good"),
            checked_at=d.get("checked_at", ""),
        )


def _score_summarization(text: str) -> Verdict:
    lower = text.lower()
    matched = sum(1 for t in _SUMMARIZATION_KEY_TERMS if t in lower)
    words = len(text.split())
    if matched >= 2 and 5 <= words <= 80:
        return "good"
    if matched >= 1:
        return "weak"
    return "poor"


def _score_structured_output(text: str) -> Verdict:
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and _STRUCTURED_KEYS.issubset(data.keys()):
            return "good"
        return "weak"
    except (json.JSONDecodeError, ValueError):
        return "poor"


def _score_long_context(text: str) -> Verdict:
    if _NEEDLE_VALUE in text:
        return "good"
    if _NEEDLE_VALUE.split("-")[1] in text:
        return "weak"
    return "poor"


def _score_relevance(text: str) -> Verdict:
    lower = text.lower()
    if "quantum" in lower:
        return "good"
    if "entanglement" in lower or "physics" in lower:
        return "weak"
    return "poor"


async def _run_task(provider, req) -> str:
    result = await provider.generate(req)
    return result.text


async def run_probe(*, registry=None, role: str = "local-fast") -> CapabilityProfile:
    if registry is None:
        from june_brain.providers.registry import get_registry

        registry = get_registry()

    provider = registry.get(role)

    from june_brain.providers.base import GenerateRequest, Message

    now = datetime.now(UTC).isoformat()

    # --- summarization ---
    sum_verdict: Verdict = "poor"
    try:
        req = GenerateRequest(
            messages=[
                Message(role="system", content="Summarize in one sentence."),
                Message(role="user", content=_SUMMARIZATION_PASSAGE),
            ],
            max_tokens=64,
            temperature=0.0,
        )
        text = await _run_task(provider, req)
        sum_verdict = _score_summarization(text)
    except Exception:  # noqa: BLE001
        pass

    # --- structured output ---
    struct_verdict: Verdict = "poor"
    try:
        req = GenerateRequest(
            messages=[
                Message(
                    role="system",
                    content=(
                        "Reply with valid JSON only. "
                        'The JSON must have exactly these keys: "name", "age", "city". '
                        "Use any plausible values."
                    ),
                ),
                Message(role="user", content="Give me a JSON object."),
            ],
            max_tokens=64,
            temperature=0.0,
            response_format="json",
        )
        text = await _run_task(provider, req)
        struct_verdict = _score_structured_output(text)
    except Exception:  # noqa: BLE001
        pass

    # --- long context ---
    lc_verdict: Verdict = "poor"
    try:
        req = GenerateRequest(
            messages=[
                Message(
                    role="system",
                    content="Answer with only the secret code value — nothing else.",
                ),
                Message(
                    role="user",
                    content=(
                        f"{_LONG_CONTEXT_PASSAGE}\n\nWhat is the secret code?"
                    ),
                ),
            ],
            max_tokens=16,
            temperature=0.0,
        )
        text = await _run_task(provider, req)
        lc_verdict = _score_long_context(text)
    except Exception:  # noqa: BLE001
        pass

    # --- relevance scoring ---
    rel_verdict: Verdict = "poor"
    try:
        items_str = ", ".join(f'"{x}"' for x in _RELEVANCE_ITEMS)
        req = GenerateRequest(
            messages=[
                Message(
                    role="system",
                    content="Reply with only the single most relevant item — no explanation.",
                ),
                Message(
                    role="user",
                    content=(
                        f"Items: {items_str}\n"
                        f'Query: "{_RELEVANCE_QUERY}"\n'
                        "Which item is most relevant to the query?"
                    ),
                ),
            ],
            max_tokens=16,
            temperature=0.0,
        )
        text = await _run_task(provider, req)
        rel_verdict = _score_relevance(text)
    except Exception:  # noqa: BLE001
        pass

    profile = CapabilityProfile(
        summarization=sum_verdict,
        structured_output=struct_verdict,
        long_context=lc_verdict,
        relevance_scoring=rel_verdict,
        checked_at=now,
    )

    _persist_profile(profile)
    return profile


def _persist_profile(profile: CapabilityProfile) -> None:
    try:
        from june_brain.datadir.layout import config_dir

        cfg = config_dir()
        cfg.mkdir(parents=True, exist_ok=True)
        path = cfg / "capability_profile.json"
        path.write_text(json.dumps(profile.to_dict(), indent=2))
    except Exception:  # noqa: BLE001
        pass


# Module-level cache — None means never probed; access via get_capability_profile().
_cached_profile: CapabilityProfile | None = None


def get_capability_profile() -> CapabilityProfile:
    if _cached_profile is None:
        return CapabilityProfile()
    return _cached_profile


async def refresh_capability_profile(
    *, registry=None, role: str = "local-fast"
) -> CapabilityProfile:
    global _cached_profile
    profile = await run_probe(registry=registry, role=role)
    _cached_profile = profile
    return profile


def reset_capability_cache() -> None:
    global _cached_profile
    _cached_profile = None
