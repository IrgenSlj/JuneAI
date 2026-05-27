"""Tests for Quick Capture classification + the capture flow (P3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from june_brain.capture import classify, process_capture, rule_classify
from june_brain.operating_layer import ActionKind, CaptureKind

# --- Classifier (pure rules — the golden workflows) -------------------------

def test_rules_detect_task_and_event() -> None:
    kinds = rule_classify("Tomorrow I need to call Sam and finish the deck")
    assert CaptureKind.TASK in kinds
    assert CaptureKind.EVENT in kinds  # "tomorrow"


def test_rules_detect_promise() -> None:
    kinds = rule_classify("I promised Lisa I would send the file Friday")
    assert CaptureKind.PROMISE in kinds


def test_rules_detect_feeling() -> None:
    kinds = rule_classify("I am anxious about money and feel stuck")
    assert CaptureKind.FEELING in kinds


def test_question_falls_back_to_question_kind() -> None:
    assert CaptureKind.QUESTION in rule_classify("what's the capital of France?")


def test_classify_defaults_to_note_when_rules_and_local_model_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ambiguous text: no rule matches. Stub the local fallback to silent so the
    # test never touches a model, and assert we degrade to NOTE (never cloud).
    import june_brain.capture as cap

    monkeypatch.setattr(cap, "llm_classify", lambda _text: ())
    assert classify("xyzzy") == (CaptureKind.NOTE,)


# --- Capture flow (through the ledger) --------------------------------------

@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())


def test_process_capture_creates_task_candidate_and_persists(isolated) -> None:
    from june_brain.events import EventLedger

    result = process_capture("Tomorrow call Sam", user_id="u")
    assert CaptureKind.TASK.value in result["capture"]["kinds"]
    kinds = {c["kind"] for c in result["candidates"]}
    assert ActionKind.CREATE_TASK.value in kinds

    # Capture and at least the two capture events were recorded in the ledger.
    assert len(EventLedger().recent_captures("u")) == 1
    assert len(EventLedger().list_events("u")) >= 2


def test_process_capture_calendar_candidate_requires_approval(isolated) -> None:
    result = process_capture("Dinner with Mara tomorrow at 7pm", user_id="u")
    calendar = [c for c in result["candidates"] if c["kind"] == ActionKind.CREATE_CALENDAR_EVENT.value]
    assert calendar and calendar[0]["requires_approval"] is True
    assert calendar[0]["can_commit"] is False


def test_process_capture_feeling_returns_supportive_message(isolated) -> None:
    result = process_capture("I feel overwhelmed and anxious", user_id="u")
    assert CaptureKind.FEELING.value in result["capture"]["kinds"]
    assert result["message"]  # a supportive, non-clinical line


def test_process_capture_never_calls_cloud(isolated, monkeypatch: pytest.MonkeyPatch) -> None:
    # If the fallback ever ran, it must use the local builder — fail loudly if a
    # cloud path is taken. Rules cover this input, so the fallback shouldn't run.
    import june_brain.capture as cap

    def _boom(_text):
        raise AssertionError("llm fallback should not run when rules match")

    monkeypatch.setattr(cap, "llm_classify", _boom)
    result = process_capture("I need to email the team today", user_id="u")
    assert CaptureKind.TASK.value in result["capture"]["kinds"]
