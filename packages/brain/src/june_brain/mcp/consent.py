"""Consent for third-party MCP clients reading June's memory (ADR 0030).

Access is **denied by default**. A client is granted specific tools, one at a
time, and any grant can be revoked with effect on the very next call — there is
no cached decision that outlives the user's choice.

Grants live in a mode-0600 JSON file under the data directory's ``config/``
rather than in ``june.db``. That separation is deliberate: a grant is a statement
about *who may read the memory*, not a memory, and keeping it out of the store it
governs means a compromised or restored database cannot quietly re-authorise a
client the user revoked.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..datadir.layout import config_dir

GRANTS_FILENAME = "mcp_grants.json"

# Tools a client can be granted. Enumerated by hand rather than derived from the
# server's registry: a protocol boundary is a security boundary, and deriving
# the list would make every future tool externally grantable by accident.
GRANTABLE_TOOLS = ("search_memory", "get_memory", "list_recent")

# A grant that is never exercised expires. An abandoned integration should not
# leave a standing key to the user's memory.
DEFAULT_TTL_DAYS = 90


def grants_path() -> Path:
    return config_dir() / GRANTS_FILENAME


@dataclass
class Grant:
    """One client's permission to call one tool."""

    client: str
    tool: str
    granted_at: float
    last_used: float | None = None
    uses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "client": self.client,
            "tool": self.tool,
            "granted_at": self.granted_at,
            "last_used": self.last_used,
            "uses": self.uses,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Grant:
        return cls(
            client=str(raw.get("client", "")),
            tool=str(raw.get("tool", "")),
            granted_at=float(raw.get("granted_at") or 0.0),
            last_used=(float(raw["last_used"]) if raw.get("last_used") else None),
            uses=int(raw.get("uses") or 0),
        )

    def is_expired(self, *, now: float, ttl_days: float = DEFAULT_TTL_DAYS) -> bool:
        reference = self.last_used or self.granted_at
        return (now - reference) > (ttl_days * 86400)


@dataclass
class ConsentStore:
    """Read/write access to the grant file.

    Every method re-reads the file rather than caching. Revocation has to take
    effect on the next call even when the revoke happened in another process
    (the UI) from the one being checked (the MCP server).
    """

    path: Path = field(default_factory=grants_path)

    # -- reads ----------------------------------------------------------

    def _load(self) -> list[Grant]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A missing or corrupt grant file means nobody is authorised. This
            # is the safe direction to fail: unreadable consent is not consent.
            return []
        if not isinstance(raw, list):
            return []
        return [Grant.from_dict(item) for item in raw if isinstance(item, dict)]

    def list_grants(self, *, now: float | None = None) -> list[Grant]:
        """Live, unexpired grants."""
        now = time.time() if now is None else now
        return [g for g in self._load() if not g.is_expired(now=now)]

    def is_allowed(self, client: str, tool: str, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if tool not in GRANTABLE_TOOLS:
            return False
        return any(
            g.client == client and g.tool == tool and not g.is_expired(now=now)
            for g in self._load()
        )

    # -- writes ---------------------------------------------------------

    def _save(self, grants: list[Grant]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps([g.to_dict() for g in grants], indent=2), encoding="utf-8"
        )
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def grant(self, client: str, tool: str, *, now: float | None = None) -> Grant:
        """Authorise one client for one tool. Idempotent."""
        if tool not in GRANTABLE_TOOLS:
            raise ValueError(f"{tool!r} is not a grantable tool")
        if not client.strip():
            raise ValueError("client identity is required")
        now = time.time() if now is None else now
        grants = self._load()
        for existing in grants:
            if existing.client == client and existing.tool == tool:
                existing.granted_at = now
                self._save(grants)
                return existing
        fresh = Grant(client=client, tool=tool, granted_at=now)
        grants.append(fresh)
        self._save(grants)
        return fresh

    def revoke(self, client: str, tool: str | None = None) -> int:
        """Remove a grant, or every grant for a client when ``tool`` is None."""
        grants = self._load()
        kept = [
            g
            for g in grants
            if not (g.client == client and (tool is None or g.tool == tool))
        ]
        removed = len(grants) - len(kept)
        if removed:
            self._save(kept)
        return removed

    def record_use(self, client: str, tool: str, *, now: float | None = None) -> None:
        """Stamp a successful call so the UI can show what a grant has done."""
        now = time.time() if now is None else now
        grants = self._load()
        for g in grants:
            if g.client == client and g.tool == tool:
                g.last_used = now
                g.uses += 1
                self._save(grants)
                return
