"""Central wiring of the Trust Ledger into egress + guard seams (ADR 0022, slice 3).

These are the bypass / tamper / redaction integration tests: they prove egress
and consequential actions route through the central ledger append, that a secret
never lands in the stored payload, and that the ledger appears in the export.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from june_brain.loop.interface import SessionState, ToolCall
from june_brain.loop.wiring import make_dispatch_fn
from june_brain.providers.provenance import CloudCallEvent, record_cloud_call
from june_brain.trust import LedgerReader, reset_for_tests


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    reset_for_tests()
    yield
    reset_for_tests()


class _Tool:
    def __init__(self, name: str, result: str = "ok") -> None:
        self.name = name
        self.args: dict = {}
        self._result = result

    def invoke(self, arguments=None):
        return self._result


def _run(tools, tool_calls, *, session=None):
    import june_brain.loop.agent_helpers as helpers_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        helpers_mod, "_select_tools_for_runtime", lambda rt: tools, raising=False
    )
    monkeypatch.setattr(
        "june_brain.config.resolve_runtime_config", lambda: object(), raising=False
    )
    try:
        dispatch = make_dispatch_fn([], [])
        session = session or SessionState(user_id="u", messages=[])
        return asyncio.run(dispatch(tool_calls, session))
    finally:
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# Egress — one entry per cloud call, written centrally (bypass guarantee)
# ---------------------------------------------------------------------------


def test_cloud_call_records_exactly_one_egress_entry() -> None:
    # A start/end pair (as every GeminiProvider call emits) must yield ONE entry.
    record_cloud_call(CloudCallEvent(model_id="gemini-x", phase="start", payload_summary="3 messages, gemini-x"))
    record_cloud_call(CloudCallEvent(model_id="gemini-x", phase="end", payload_summary="3 messages, gemini-x"))
    reader = LedgerReader()
    egress = [e for e in reader.page(limit=100) if e.kind == "egress"]
    assert len(egress) == 1
    assert egress[0].actor == "june"
    assert egress[0].payload["model_id"] == "gemini-x"
    assert reader.verify_chain().ok is True


def test_no_cloud_call_means_no_egress_entry() -> None:
    # Local-only acceptance: with no cloud call, there are no egress entries.
    reader = LedgerReader()
    assert [e for e in reader.page(limit=100) if e.kind == "egress"] == []


def test_egress_payload_redacts_configured_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "AIzaSyABCDEF1234567890abcdefghij"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    record_cloud_call(
        CloudCallEvent(model_id="gemini-x", phase="start", payload_summary=f"leaked {secret}")
    )
    reader = LedgerReader()
    row = reader._conn.execute("SELECT payload FROM trust_ledger WHERE kind='egress'").fetchone()
    assert secret not in row["payload"]
    assert "[REDACTED]" in row["payload"]


# ---------------------------------------------------------------------------
# Action — a gated action that executes (approved/allow-listed) is recorded
# ---------------------------------------------------------------------------


def test_approved_network_write_records_an_action_entry() -> None:
    session = SessionState(user_id="u", messages=[], approved_tools={"send_telegram_message"})
    obs = _run(
        [_Tool("send_telegram_message", "sent")],
        [ToolCall(name="send_telegram_message", args={"text": "hi"})],
        session=session,
    )
    assert "sent" in obs[0].content  # it actually ran
    reader = LedgerReader()
    actions = [e for e in reader.page(limit=100) if e.kind == "action"]
    assert len(actions) == 1
    assert actions[0].payload["tool"] == "send_telegram_message"
    assert actions[0].payload["action_class"] == "write_network"
    assert reader.verify_chain().ok is True


def test_local_write_records_no_action_entry() -> None:
    # Local, non-gated writes are not consequential actions — no ledger entry.
    _run(
        [_Tool("save_journal_entry", "saved")],
        [ToolCall(name="save_journal_entry", args={"entry": "hi"})],
    )
    reader = LedgerReader()
    assert [e for e in reader.page(limit=100) if e.kind == "action"] == []


def test_blocked_network_write_records_no_action_entry() -> None:
    # Blocked (not approved) → the tool never runs → no action entry.
    _run(
        [_Tool("send_telegram_message", "sent")],
        [ToolCall(name="send_telegram_message", args={"text": "hi"})],
    )
    reader = LedgerReader()
    assert [e for e in reader.page(limit=100) if e.kind == "action"] == []


# ---------------------------------------------------------------------------
# Export includes the full ledger (verifiable offline)
# ---------------------------------------------------------------------------


def test_export_includes_trust_ledger() -> None:
    record_cloud_call(CloudCallEvent(model_id="gemini-x", phase="start", payload_summary="s"))
    from june_brain.memory.export import export_memory

    archive = export_memory("u")
    tables = archive["stores"]["sqlite"]
    assert "trust_ledger" in tables
    assert len(tables["trust_ledger"]) == 1
    assert tables["trust_ledger"][0]["kind"] == "egress"
