"""Tests for the local startup greeting (no model call, never cloud)."""

from __future__ import annotations

import pytest


def _patch_stores(monkeypatch, *, chat, facts):
    import june_brain.memory as mem_pkg

    class FakeMem:
        def __init__(self, _uid): ...
        def load_chat(self):
            return chat

    class FakeVec:
        def __init__(self, _uid): ...
        def list_facts(self, limit=50):
            return facts

    monkeypatch.setattr(mem_pkg, "Memory", FakeMem)
    monkeypatch.setattr(mem_pkg, "VectorStore", FakeVec)


def test_greeting_new_user_uses_name_and_no_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stores(monkeypatch, chat=[], facts=[])
    from june_brain.greeting import build_greeting

    out = build_greeting("u", "Alex")
    assert "Alex" in out["greeting"]
    assert out["has_context"] is False


def test_greeting_references_most_recent_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stores(
        monkeypatch,
        chat=[{"role": "user"}],
        facts=[
            {"text": "old note", "created_at": "2026-01-01"},
            {"text": "loves espresso", "created_at": "2026-05-27"},
        ],
    )
    from june_brain.greeting import build_greeting

    out = build_greeting("u", "Alex")
    assert "espresso" in out["greeting"]  # newest fact, not the older one
    assert out["has_context"] is True


def test_greeting_never_fails_when_memory_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import june_brain.memory as mem_pkg

    class Boom:
        def __init__(self, _uid):
            raise RuntimeError("store down")

    monkeypatch.setattr(mem_pkg, "Memory", Boom)
    monkeypatch.setattr(mem_pkg, "VectorStore", Boom)
    from june_brain.greeting import build_greeting

    out = build_greeting("u", "Alex")
    assert "Alex" in out["greeting"]
    assert out["has_context"] is False
