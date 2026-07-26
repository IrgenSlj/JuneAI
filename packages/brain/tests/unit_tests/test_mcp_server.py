"""June's MCP memory server: consent, read-only surface, and the audit trail.

These tests pin the three properties ADR 0030 calls load-bearing. Each is a
security property, not a convenience: a regression here does not degrade the
feature, it breaks the claim the feature exists to demonstrate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from june_brain.mcp.consent import ConsentStore, Grant
from june_brain.mcp.server import build_server


class _FakeVector:
    def __init__(self) -> None:
        self.facts = [
            {"fact_id": "f1", "text": "Ana's dog is called Biscuit.", "source": "golden"},
            {"fact_id": "f2", "text": "Ana runs on Tuesdays.", "source": "golden"},
        ]

    def list_facts(self, limit: int = 50, source_prefix: str = "") -> list[dict[str, Any]]:
        return self.facts[:limit]

    def get(self, fact_id: str) -> dict[str, Any] | None:
        return next((f for f in self.facts if f["fact_id"] == fact_id), None)


class _FakeManager:
    def __init__(self) -> None:
        self.vector = _FakeVector()
        self.recall_calls: list[str] = []

    def recall(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        self.recall_calls.append(query)
        return [
            {
                "text": "Ana's dog is called Biscuit.",
                "ref": "semantic:f1",
                "kind": "fact",
                "source": "vector",
                "score": 0.1,
            }
        ][:k]


class _FakeLedger:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def append(self, *, kind: str, actor: str, payload: dict[str, Any]) -> Any:
        self.entries.append({"kind": kind, "actor": actor, "payload": payload})
        return None


@pytest.fixture
def consent(tmp_path: Path) -> ConsentStore:
    return ConsentStore(path=tmp_path / "mcp_grants.json")


@pytest.fixture
def wired(consent: ConsentStore):
    manager, ledger = _FakeManager(), _FakeLedger()
    server = build_server(
        manager=manager, consent=consent, ledger=ledger, client="test-client"
    )
    return server, manager, ledger, consent


def _call(server: Any, tool: str, **args: Any) -> Any:
    return server._tools[tool].fn(**args)


# -- consent ------------------------------------------------------------


def test_every_tool_is_denied_without_a_grant(wired) -> None:
    server, _manager, _ledger, _consent = wired
    for tool, args in (
        ("search_memory", {"query": "dog"}),
        ("get_memory", {"ref": "semantic:f1"}),
        ("list_recent", {}),
    ):
        with pytest.raises(Exception, match="has not granted"):
            _call(server, tool, **args)


def test_a_grant_is_scoped_to_one_tool(wired) -> None:
    server, _manager, _ledger, consent = wired
    consent.grant("test-client", "search_memory")

    assert _call(server, "search_memory", query="dog")["count"] == 1
    # The other two remain denied — a grant is not a key to the whole memory.
    with pytest.raises(Exception, match="has not granted"):
        _call(server, "list_recent")


def test_revocation_applies_to_the_very_next_call(wired) -> None:
    server, _manager, _ledger, consent = wired
    consent.grant("test-client", "search_memory")
    assert _call(server, "search_memory", query="dog")["count"] == 1

    consent.revoke("test-client", "search_memory")
    with pytest.raises(Exception, match="has not granted"):
        _call(server, "search_memory", query="dog")


def test_a_grant_for_another_client_does_not_transfer(wired) -> None:
    server, _manager, _ledger, consent = wired
    consent.grant("some-other-client", "search_memory")
    with pytest.raises(Exception, match="has not granted"):
        _call(server, "search_memory", query="dog")


def test_unreadable_grant_file_denies_everything(tmp_path: Path) -> None:
    # Unreadable consent is not consent: corruption must fail closed.
    bad = tmp_path / "mcp_grants.json"
    bad.write_text("{not json", encoding="utf-8")
    store = ConsentStore(path=bad)
    assert store.is_allowed("test-client", "search_memory") is False
    assert store.list_grants() == []


def test_expired_grants_stop_working() -> None:
    old = Grant(client="c", tool="search_memory", granted_at=0.0)
    assert old.is_expired(now=100 * 86400) is True
    assert old.is_expired(now=10 * 86400) is False


def test_ungrantable_tool_names_are_rejected(consent: ConsentStore) -> None:
    with pytest.raises(ValueError):
        consent.grant("c", "forget_everything")
    assert consent.is_allowed("c", "forget_everything") is False


# -- read-only surface --------------------------------------------------


def test_the_server_exposes_only_the_three_read_tools(wired) -> None:
    server, _manager, _ledger, _consent = wired
    assert set(server._tools) == {"search_memory", "get_memory", "list_recent"}


def test_no_tool_name_suggests_a_write_path(wired) -> None:
    server, _manager, _ledger, _consent = wired
    forbidden = ("write", "remember", "forget", "delete", "update", "purge")
    for name in server._tools:
        assert not any(word in name for word in forbidden)


def test_get_memory_only_addresses_the_semantic_namespace(wired) -> None:
    server, _manager, _ledger, consent = wired
    consent.grant("test-client", "get_memory")
    # A structured-row ref must not resolve through this surface.
    assert _call(server, "get_memory", ref="goal:123")["found"] is False
    assert _call(server, "get_memory", ref="semantic:f1")["found"] is True


# -- the audit trail ----------------------------------------------------


def test_a_successful_read_is_ledgered(wired) -> None:
    server, _manager, ledger, consent = wired
    consent.grant("test-client", "search_memory")
    _call(server, "search_memory", query="dog")

    entries = [e for e in ledger.entries if e["payload"].get("allowed")]
    assert len(entries) == 1
    assert entries[0]["kind"] == "mcp_access"
    assert entries[0]["payload"]["client"] == "test-client"
    assert entries[0]["payload"]["returned"] == 1


def test_a_denied_read_is_also_ledgered(wired) -> None:
    server, _manager, ledger, _consent = wired
    with pytest.raises(Exception):
        _call(server, "search_memory", query="dog")

    assert len(ledger.entries) == 1
    assert ledger.entries[0]["payload"]["allowed"] is False
    assert ledger.entries[0]["payload"]["reason"] == "no grant"


def test_the_ledger_records_shape_not_content(wired) -> None:
    """The audit trail must not become a second copy of the memory."""
    server, _manager, ledger, consent = wired
    consent.grant("test-client", "search_memory")
    _call(server, "search_memory", query="what is my dog called")

    for entry in ledger.entries:
        flat = str(entry["payload"])
        assert "Biscuit" not in flat
        assert "what is my dog called" not in flat
    # The shape is still there: how long the query was, how much came back.
    assert ledger.entries[-1]["payload"]["query_length"] == len("what is my dog called")


def test_use_is_recorded_against_the_grant(wired) -> None:
    server, _manager, _ledger, consent = wired
    consent.grant("test-client", "list_recent")
    _call(server, "list_recent", limit=2)

    grant = next(g for g in consent.list_grants() if g.tool == "list_recent")
    assert grant.uses == 1
    assert grant.last_used is not None


def test_results_are_capped_regardless_of_requested_limit(wired) -> None:
    server, _manager, _ledger, consent = wired
    consent.grant("test-client", "list_recent")
    out = _call(server, "list_recent", limit=10_000)
    assert out["count"] <= 25
