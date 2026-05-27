"""Quick Capture: turn a messy thought into typed candidates (P3).

Rules first — deterministic, instant, fully offline — and only when the rules
find nothing do we fall back to a LOCAL model. Classification is private by
construction: the fallback always uses local Gemma and never the cloud,
regardless of the user's privacy dial. If the local model is unavailable the
capture degrades to a plain note rather than reaching out.

The flow records everything through the event ledger (ADR 0014): a capture is
saved, classified, and turned into candidate action intents. Actually executing
those intents (writing memory, creating tasks/events) is approval/commit work
(P4) — this module proposes, it does not commit.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .events import EventLedger
from .operating_layer import (
    ActionIntent,
    ActionKind,
    ActionRisk,
    CaptureItem,
    CaptureKind,
    EventKind,
    LedgerEvent,
)

logger = logging.getLogger(__name__)

# --- Rule vocabulary --------------------------------------------------------

_PROMISE_RE = re.compile(
    r"\b(i\s+promised|i\s+told\s+\w+\s+(i'?d|i\s+would)|promise\s+to|i'?ll\s+make\s+sure)\b",
    re.IGNORECASE,
)
_FEELING_RE = re.compile(
    r"\b(anxious|anxiety|stressed|overwhelmed|worried|scared|afraid|sad|down|"
    r"depressed|lonely|stuck|frustrated|angry|happy|excited|grateful|calm|tired|exhausted)\b",
    re.IGNORECASE,
)
_TASK_RE = re.compile(
    r"\b(need\s+to|have\s+to|must|todo|to-do|remember\s+to|call|email|finish|"
    r"buy|book|send|fix|write|schedule|pick\s+up|follow\s+up)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|this\s+(morning|afternoon|evening|week|weekend)|"
    r"next\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|"
    r"at\s+\d{1,2}(:\d{2})?\s*(am|pm)?)\b",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(decided\s+to|i'?ll\s+go\s+with|going\s+with|we'?ll\s+ship|let'?s\s+ship|chose\s+to)\b",
    re.IGNORECASE,
)
_IDEA_RE = re.compile(r"\b(idea:|what\s+if|maybe\s+(we|i)\s+could|it\s+would\s+be\s+cool)\b", re.IGNORECASE)


def rule_classify(text: str) -> tuple[CaptureKind, ...]:
    """Classify with deterministic rules. May return multiple kinds."""
    t = text.strip()
    if not t:
        return ()
    kinds: list[CaptureKind] = []
    if _PROMISE_RE.search(t):
        kinds.append(CaptureKind.PROMISE)
    if _FEELING_RE.search(t):
        kinds.append(CaptureKind.FEELING)
    if _TASK_RE.search(t):
        kinds.append(CaptureKind.TASK)
    if _DATE_RE.search(t):
        kinds.append(CaptureKind.EVENT)
    if _DECISION_RE.search(t):
        kinds.append(CaptureKind.DECISION)
    if _IDEA_RE.search(t):
        kinds.append(CaptureKind.IDEA)
    if not kinds and t.endswith("?"):
        kinds.append(CaptureKind.QUESTION)
    # De-dup while preserving order.
    seen: dict[CaptureKind, None] = {}
    for k in kinds:
        seen.setdefault(k, None)
    return tuple(seen)


def llm_classify(text: str) -> tuple[CaptureKind, ...]:
    """Best-effort fallback using a LOCAL model only. Never cloud.

    Returns () on any failure so the caller can degrade to a note. Kept as a
    module function so it is easy to stub in tests.
    """
    try:
        from .config import resolve_runtime_config
        from .models import build_chat_model

        runtime = resolve_runtime_config("gemma")  # force local, never cloud
        if not runtime.is_local:
            return ()
        llm = build_chat_model(runtime)
        labels = ", ".join(k.value for k in CaptureKind)
        prompt = (
            "Classify the note into one or more of these labels: "
            f"{labels}. Reply with only a comma-separated list of labels.\n\nNote: {text}"
        )
        reply = llm.invoke(prompt)
        raw = getattr(reply, "content", "") or ""
        out: list[CaptureKind] = []
        for token in str(raw).replace("\n", ",").split(","):
            try:
                out.append(CaptureKind(token.strip().lower()))
            except ValueError:
                continue
        return tuple(dict.fromkeys(out))
    except Exception:  # noqa: BLE001 — classification is best-effort
        logger.debug("local llm_classify fallback failed", exc_info=True)
        return ()


def classify(text: str) -> tuple[CaptureKind, ...]:
    """Rules first; local-model fallback only when rules find nothing."""
    kinds = rule_classify(text)
    if kinds:
        return kinds
    kinds = llm_classify(text)
    return kinds or (CaptureKind.NOTE,)


# --- Capture kind -> candidate action ---------------------------------------

_KIND_TO_ACTION: dict[CaptureKind, tuple[ActionKind, ActionRisk]] = {
    CaptureKind.TASK: (ActionKind.CREATE_TASK, ActionRisk.LOW),
    CaptureKind.EVENT: (ActionKind.CREATE_CALENDAR_EVENT, ActionRisk.EXTERNAL),
    CaptureKind.PROMISE: (ActionKind.SAVE_MEMORY, ActionRisk.LOW),
    CaptureKind.FEELING: (ActionKind.SAVE_MEMORY, ActionRisk.LOW),
    CaptureKind.MEMORY: (ActionKind.SAVE_MEMORY, ActionRisk.LOW),
    CaptureKind.DECISION: (ActionKind.SAVE_MEMORY, ActionRisk.LOW),
    CaptureKind.IDEA: (ActionKind.SAVE_MEMORY, ActionRisk.LOW),
    CaptureKind.NOTE: (ActionKind.SAVE_MEMORY, ActionRisk.LOW),
    # QUESTION produces no write candidate — it is answered, not stored.
}

_SUPPORT_LINE = (
    "That sounds heavy. One small step is often enough to start — "
    "want to note it down or talk it through?"
)


def process_capture(
    text: str,
    user_id: str = "default",
    source: str = "chat",
    *,
    ledger: EventLedger | None = None,
) -> dict[str, Any]:
    """Classify a capture, persist it, and propose candidate intents.

    Records capture_received + capture_classified events and one
    action_intent_created event per candidate. Returns the capture, the
    candidate intents, and an optional supportive message for feelings.
    """
    ledger = ledger or EventLedger()
    kinds = classify(text)

    item = CaptureItem(text=text, user_id=user_id, source=source, kinds=kinds)
    ledger.save_capture(item)
    ledger.append(LedgerEvent(kind=EventKind.CAPTURE_RECEIVED, user_id=user_id, source=source,
                              payload={"capture_id": item.id}))
    ledger.append(LedgerEvent(kind=EventKind.CAPTURE_CLASSIFIED, user_id=user_id, source=source,
                              payload={"capture_id": item.id, "kinds": [k.value for k in kinds]}))

    candidates: list[ActionIntent] = []
    for kind in kinds:
        mapping = _KIND_TO_ACTION.get(kind)
        if mapping is None:
            continue
        action_kind, risk = mapping
        intent = ActionIntent(
            kind=action_kind,
            title=text[:80],
            summary=text,
            user_id=user_id,
            risk=risk,
            source_capture_id=item.id,
            payload={"capture_kind": kind.value},
        )
        ledger.save_intent(intent)
        ledger.append(LedgerEvent(kind=EventKind.ACTION_INTENT_CREATED, user_id=user_id,
                                  source=source, payload={"intent_id": intent.id}))
        candidates.append(intent)

    message = _SUPPORT_LINE if CaptureKind.FEELING in kinds else ""
    return {
        "capture": item.to_dict(),
        "candidates": [c.to_dict() for c in candidates],
        "message": message,
    }
