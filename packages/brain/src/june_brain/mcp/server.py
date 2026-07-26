"""June as an MCP memory server — read-only, consent-gated, ledgered (ADR 0030).

This lets an assistant the user already runs (Claude Desktop, Cursor, or any
other MCP client) read June's memory without installing June. It is the only
surface where June's central claim becomes immediately checkable by a stranger:
connect a client, ask it something, then watch the Trust screen show exactly
which memories it read and under whose grant.

Three properties are load-bearing and none of them is optional:

* **Read-only.** No tool here writes, updates, or forgets. A memory store any
  connected agent can write is a poisoning vector, and June's differentiator is
  being able to say where a memory came from.
* **Denied by default.** A client sees nothing until the user grants a specific
  tool, and a revoke lands on the next call.
* **Every access on the record.** Reads are ledgered, not just refusals. For a
  memory system exfiltration is the threat, so logging only writes would leave
  the one thing worth auditing unaudited.

Run it with the ``june-mcp`` console script; it speaks JSON-RPC over stdio like
every other MCP server in this repo, reusing ``skills.server.MCPStdioServer``
rather than taking an SDK dependency.
"""

from __future__ import annotations

import os
from typing import Any

from ..memory.manager import MemoryManager
from ..skills.server import MCPStdioServer
from ..trust.ledger import LedgerWriter, get_writer
from .consent import ConsentStore

SERVER_NAME = "june-memory"
SERVER_VERSION = "0.1.0"

# How the calling client identifies itself. MCP has no authenticated client
# identity, so this is a declared name, not a proven one — which is exactly why
# it is paired with an explicit user grant and a ledger entry rather than
# trusted on its own.
CLIENT_ENV_VAR = "JUNE_MCP_CLIENT"
DEFAULT_CLIENT = "unknown-client"

MAX_RESULTS = 25


def _client_id() -> str:
    return (os.environ.get(CLIENT_ENV_VAR) or DEFAULT_CLIENT).strip() or DEFAULT_CLIENT


def _user_id() -> str:
    return os.environ.get("JUNE_USER_ID", "default")


class _Denied(Exception):
    """Raised when no live grant covers the call."""


def build_server(
    *,
    manager: MemoryManager | None = None,
    consent: ConsentStore | None = None,
    ledger: LedgerWriter | None = None,
    client: str | None = None,
) -> MCPStdioServer:
    """Wire the three read tools. Dependencies are injectable for tests."""
    server = MCPStdioServer(name=SERVER_NAME, version=SERVER_VERSION)
    store = consent or ConsentStore()
    who = client or _client_id()
    mgr = manager if manager is not None else MemoryManager(_user_id())
    book = ledger if ledger is not None else get_writer()

    def _record(tool: str, *, allowed: bool, detail: dict[str, Any]) -> None:
        """Append the access to the ledger.

        The actor stays ``june`` because June's own process performed the read;
        the third party is named in the payload. The payload deliberately
        carries the *shape* of the access — which tool, whose grant, how many
        facts came back — and never the fact text, so the ledger stays an audit
        trail rather than a second copy of the memory.
        """
        try:
            book.append(
                kind="mcp_access",
                actor="june",
                payload={"client": who, "tool": tool, "allowed": allowed, **detail},
            )
        except Exception:  # noqa: BLE001 — an audit failure must not leak data
            # Failing closed is the only safe direction: if the access cannot be
            # recorded, the access does not happen.
            raise

    def _guard(tool: str, detail: dict[str, Any]) -> None:
        if not store.is_allowed(who, tool):
            _record(tool, allowed=False, detail={**detail, "reason": "no grant"})
            raise _Denied(
                f"June has not granted {who!r} access to {tool!r}. "
                "Grant it from June's Trust screen."
            )

    def _hit_to_payload(hit: dict[str, Any]) -> dict[str, Any]:
        """Recall hits and stored fact rows have different shapes; normalise."""
        ref = hit.get("ref") or (
            f"semantic:{hit['fact_id']}" if hit.get("fact_id") else ""
        )
        return {
            "text": hit.get("text", ""),
            "ref": ref,
            "kind": hit.get("kind") or "fact",
            "source": hit.get("source", ""),
            "score": hit.get("score"),
        }

    @server.tool(
        name="search_memory",
        description=(
            "Search June's long-term memory. Returns facts ranked by salience "
            "(recency x frequency x relevance) across semantic, lexical, entity "
            "and temporal signals."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "limit": {
                    "type": "integer",
                    "description": f"Max results (1-{MAX_RESULTS}).",
                },
            },
            "required": ["query"],
        },
    )
    def search_memory(query: str = "", limit: int = 8) -> dict[str, Any]:
        query = (query or "").strip()
        k = max(1, min(int(limit or 8), MAX_RESULTS))
        _guard("search_memory", {"query_length": len(query)})
        if not query:
            return {"results": [], "count": 0}
        hits = mgr.recall(query, k=k)
        results = [_hit_to_payload(h) for h in hits]
        _record(
            "search_memory",
            allowed=True,
            detail={"query_length": len(query), "returned": len(results)},
        )
        store.record_use(who, "search_memory")
        return {"results": results, "count": len(results)}

    @server.tool(
        name="get_memory",
        description="Fetch one remembered fact by its reference.",
        input_schema={
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "The fact reference."}
            },
            "required": ["ref"],
        },
    )
    def get_memory(ref: str = "") -> dict[str, Any]:
        ref = (ref or "").strip()
        _guard("get_memory", {"ref": ref})
        found = None
        if ref:
            # Go through the facade's stores, never raw SQL: ADR 0019's
            # single-facade rule holds at this boundary too. Only the
            # ``semantic:`` namespace is addressable here — structured rows and
            # graph nodes are not part of the read-only contract.
            fact_id = ref.split("semantic:", 1)[1] if ref.startswith("semantic:") else ""
            record = mgr.vector.get(fact_id) if fact_id else None
            if record:
                found = _hit_to_payload(record)
        _record("get_memory", allowed=True, detail={"ref": ref, "found": bool(found)})
        store.record_use(who, "get_memory")
        return {"result": found, "found": bool(found)}

    @server.tool(
        name="list_recent",
        description="List the memories June has touched most recently.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": f"How many to return (1-{MAX_RESULTS}).",
                }
            },
        },
    )
    def list_recent(limit: int = 10) -> dict[str, Any]:
        k = max(1, min(int(limit or 10), MAX_RESULTS))
        _guard("list_recent", {"limit": k})
        items = [_hit_to_payload(i) for i in mgr.vector.list_facts(limit=k)]
        _record("list_recent", allowed=True, detail={"returned": len(items)})
        store.record_use(who, "list_recent")
        return {"results": items, "count": len(items)}

    return server


def main() -> None:
    """Console-script entry point (``june-mcp``)."""
    build_server().run()


if __name__ == "__main__":
    main()
