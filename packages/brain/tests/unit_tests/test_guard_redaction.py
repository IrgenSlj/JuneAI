"""Secret redaction in traces (ADR 0021, S6.4)."""

from __future__ import annotations

from june_brain.guard.redaction import REDACTED, redact_secrets


def test_redacts_configured_env_secret(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "supersecretkey-1234567890")
    text = "calling gemini with key supersecretkey-1234567890 now"
    out = redact_secrets(text)
    assert "supersecretkey-1234567890" not in out
    assert REDACTED in out


def test_redacts_openai_style_key():
    out = redact_secrets("here is sk-ABCDEFGH1234567890xyz token")
    assert "sk-ABCDEFGH1234567890xyz" not in out
    assert REDACTED in out


def test_redacts_google_key():
    out = redact_secrets("key=AIzaSyA1234567890ABCDEFGHIJKLMNOPQRST end")
    assert "AIzaSy" not in out
    assert REDACTED in out


def test_redacts_telegram_token():
    out = redact_secrets("token 123456789:AAEabcdefghijklmnopqrstuvwxyz0123456 ok")
    assert "AAEabcdefghij" not in out
    assert REDACTED in out


def test_redacts_bearer_token():
    out = redact_secrets("Authorization: Bearer abcdef1234567890XYZ")
    assert "abcdef1234567890XYZ" not in out


def test_leaves_ordinary_prose_untouched():
    text = "The user asked about their goals and recent journal entries."
    assert redact_secrets(text) == text


def test_empty_is_safe():
    assert redact_secrets("") == ""


def test_persisted_trace_does_not_contain_secret(tmp_path, monkeypatch):
    """A turn that uses a key must not leave it in the persisted trace file."""
    monkeypatch.setenv("GEMINI_API_KEY", "leak-me-9876543210abcdef")
    monkeypatch.setattr("june_brain.loop.trace.TRACE_MAX", 100, raising=False)

    from june_brain.loop import trace as trace_mod

    monkeypatch.setattr(trace_mod.TraceStore, "_dir", lambda self: tmp_path)

    t = trace_mod.TurnTrace(turn_id="redact-test", user_id="u")
    t.record("prompt", "prompt assembled", detail="system: use key leak-me-9876543210abcdef")
    t.record("tool_result", "web_search", detail="auth=leak-me-9876543210abcdef")

    assert trace_mod.TraceStore().write(t) is True

    written = (tmp_path / "redact-test.json").read_text(encoding="utf-8")
    assert "leak-me-9876543210abcdef" not in written
    assert REDACTED in written
