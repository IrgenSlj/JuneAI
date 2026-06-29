"""Persistence for Silence Model decisions + the batched-surfacing digest (ADR 0023).

Every decision the policy makes (now / batch / suppress) is recorded here, so the
"why June stayed quiet" surface is real and inspectable, and so a future v2 has a
training log. Each decision is also written to the Trust Ledger (ADR 0022) as an
``action`` by ``june`` — June's restraint is itself auditable.

Batching is the key mechanism for the fourth inversion: a `batch` decision is held
in a digest and surfaced only when the user next shows up. The drain is a plain
function the session-open / "user shows up" path calls — there is no scheduled
job, no timer, no background loop (ADR 0016). That boundary is asserted by a test
that scans this package for any scheduler/clock machinery.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ..memory.sqlite import _get_connection, db_path
from .policy import SurfacingCandidate, SurfacingDecision

# How many recent decisions of a kind to look back over when counting dismissals,
# so the "dismissed similar N times recently" signal stays recent and bounded.
_DISMISSAL_LOOKBACK = 20

VALID_OUTCOMES = frozenset({"engaged", "dismissed", "expired"})

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS surfacing_decisions (
    id           TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    kind         TEXT NOT NULL,
    action       TEXT NOT NULL,
    reason       TEXT NOT NULL,
    features     TEXT NOT NULL DEFAULT '{}',
    ts           TEXT NOT NULL,
    outcome      TEXT,
    surfaced_at  TEXT,
    verdict      TEXT
);
CREATE INDEX IF NOT EXISTS idx_surfacing_ts ON surfacing_decisions(ts);
CREATE INDEX IF NOT EXISTS idx_surfacing_kind ON surfacing_decisions(kind, ts);
CREATE INDEX IF NOT EXISTS idx_surfacing_digest ON surfacing_decisions(action, surfaced_at);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SurfacingRecord:
    """One stored decision row."""

    id: str
    candidate_id: str
    kind: str
    action: str
    reason: str
    features: dict[str, Any]
    ts: str
    outcome: str | None = None
    surfaced_at: str | None = None
    verdict: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "action": self.action,
            "reason": self.reason,
            "features": self.features,
            "ts": self.ts,
            "outcome": self.outcome,
            "surfaced_at": self.surfaced_at,
            "verdict": self.verdict,
        }


def _row_to_record(row: sqlite3.Row) -> SurfacingRecord:
    try:
        features = json.loads(row["features"]) if row["features"] else {}
    except (TypeError, ValueError):
        features = {}
    return SurfacingRecord(
        id=str(row["id"]),
        candidate_id=str(row["candidate_id"]),
        kind=str(row["kind"]),
        action=str(row["action"]),
        reason=str(row["reason"]),
        features=features,
        ts=str(row["ts"]),
        outcome=row["outcome"],
        surfaced_at=row["surfaced_at"],
        verdict=row["verdict"],
    )


class SurfacingStore:
    """Read/write interface to the surfacing_decisions table + the held digest."""

    def __init__(self) -> None:
        self._db_path = db_path()
        conn = _get_connection(self._db_path)
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    @property
    def _conn(self) -> sqlite3.Connection:
        return _get_connection(self._db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        candidate: SurfacingCandidate,
        decision: SurfacingDecision,
        *,
        ts: str | None = None,
    ) -> SurfacingRecord:
        """Persist a decision and mirror it into the Trust Ledger (best-effort)."""
        rec_id = uuid4().hex
        when = ts or _now()
        self._conn.execute(
            "INSERT INTO surfacing_decisions "
            "(id, candidate_id, kind, action, reason, features, ts, outcome, surfaced_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                rec_id,
                candidate.candidate_id,
                candidate.kind,
                decision.action,
                decision.reason,
                json.dumps(decision.features),
                when,
                None,
                None,
            ),
        )
        self._conn.commit()
        self._ledger_record(candidate, decision)
        return SurfacingRecord(
            id=rec_id,
            candidate_id=candidate.candidate_id,
            kind=candidate.kind,
            action=decision.action,
            reason=decision.reason,
            features=decision.features,
            ts=when,
        )

    @staticmethod
    def _ledger_record(candidate: SurfacingCandidate, decision: SurfacingDecision) -> None:
        try:
            from june_brain.trust import get_writer

            get_writer().append(
                kind="action",
                actor="june",
                payload={
                    "event": "surfacing_decision",
                    "candidate_id": candidate.candidate_id,
                    "candidate_kind": candidate.kind,
                    "decision": decision.action,
                    "reason": decision.reason,
                },
            )
        except Exception:  # noqa: BLE001 - the ledger is best-effort at the call site
            import logging

            logging.getLogger(__name__).debug(
                "trust-ledger surfacing append failed", exc_info=True
            )

    def latest_for_candidate(self, candidate_id: str) -> SurfacingRecord | None:
        """Most recent row (ts DESC) for this candidate_id, or None."""
        row = self._conn.execute(
            "SELECT * FROM surfacing_decisions WHERE candidate_id=? ORDER BY ts DESC LIMIT 1",
            (candidate_id,),
        ).fetchone()
        return _row_to_record(row) if row else None

    def record_if_new(
        self,
        candidate: SurfacingCandidate,
        decision: SurfacingDecision,
        *,
        ts: str | None = None,
    ) -> SurfacingRecord | None:
        """Persist a decision only if the action has changed since the last run.

        Idempotent: if the latest row for this candidate already has the same
        action, return None (no duplicate). If different (a real state
        transition, e.g. batch -> now as a deadline nears), record and return
        the new row. If there is no prior row, record and return it.
        """
        latest = self.latest_for_candidate(candidate.candidate_id)
        if latest is not None and latest.action == decision.action:
            return None
        return self.record(candidate, decision, ts=ts)

    def set_outcome(self, decision_id: str, outcome: str) -> bool:
        """Record what became of a surfaced decision (engaged | dismissed | expired)."""
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"invalid surfacing outcome {outcome!r}")
        cur = self._conn.execute(
            "UPDATE surfacing_decisions SET outcome=? WHERE id=?",
            (outcome, decision_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def record_feedback(self, decision_id: str, verdict: str) -> SurfacingRecord | None:
        """Record the user's judgment of a decision; map it to the dismissal signal.

        ``verdict`` is the raw v2 training signal (stored verbatim). It also moves
        ``dismissals_for_similar``, but honestly w.r.t. what June actually did:
        the user "wants these" when they approve a surfacing OR reject a
        suppression, and "doesn't want these" when they reject a surfacing OR
        approve a suppression. Wanting → outcome=engaged; not wanting →
        outcome=dismissed (which raises the suppress signal for similar items).
        Returns the updated record, or None if the id is unknown.
        """
        if verdict not in ("good", "bad"):
            raise ValueError(f"invalid surfacing verdict {verdict!r}")
        rec = self.get(decision_id)
        if rec is None:
            return None
        surfaced = rec.action in ("now", "batch")
        user_wants = (surfaced and verdict == "good") or (not surfaced and verdict == "bad")
        outcome = "engaged" if user_wants else "dismissed"
        self._conn.execute(
            "UPDATE surfacing_decisions SET verdict=?, outcome=? WHERE id=?",
            (verdict, outcome, decision_id),
        )
        self._conn.commit()
        return self.get(decision_id)

    # ------------------------------------------------------------------
    # Signals feeding the policy
    # ------------------------------------------------------------------

    def dismissals_for_similar(self, kind: str, *, lookback: int = _DISMISSAL_LOOKBACK) -> int:
        """Recent dismissals of the same kind — the policy's suppress signal."""
        rows = self._conn.execute(
            "SELECT outcome FROM surfacing_decisions WHERE kind=? ORDER BY ts DESC LIMIT ?",
            (kind, max(1, int(lookback))),
        ).fetchall()
        return sum(1 for r in rows if r["outcome"] == "dismissed")

    # ------------------------------------------------------------------
    # The batched digest — drained only at event boundaries (no timer)
    # ------------------------------------------------------------------

    def held_digest(self, *, limit: int = 50) -> list[SurfacingRecord]:
        """Read-only peek at batched items not yet surfaced (newest-first)."""
        rows = self._conn.execute(
            "SELECT * FROM surfacing_decisions "
            "WHERE action='batch' AND surfaced_at IS NULL ORDER BY ts DESC LIMIT ?",
            (max(1, min(int(limit), 200)),),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def held_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM surfacing_decisions "
            "WHERE action='batch' AND surfaced_at IS NULL"
        ).fetchone()
        return int(row["n"]) if row else 0

    def drain_digest(self, *, ts: str | None = None) -> list[SurfacingRecord]:
        """Return all held batch items and mark them surfaced.

        This is the event-boundary drain: call it from the session-open / "user
        shows up" path. It is a plain function — never wire it to a scheduler or
        timer (ADR 0016). Calling it twice in a row returns the second time empty,
        because the first call marked everything surfaced.
        """
        when = ts or _now()
        held = self.held_digest(limit=200)
        if held:
            self._conn.execute(
                "UPDATE surfacing_decisions SET surfaced_at=? "
                "WHERE action='batch' AND surfaced_at IS NULL",
                (when,),
            )
            self._conn.commit()
        return held

    # ------------------------------------------------------------------
    # Read (recent decisions — the "why June stayed quiet" surface)
    # ------------------------------------------------------------------

    def page(self, *, cursor: str | None = None, limit: int = 50) -> list[SurfacingRecord]:
        """Recent decisions, newest-first. ``cursor`` is an exclusive ts upper bound."""
        limit = max(1, min(int(limit), 200))
        if cursor:
            rows = self._conn.execute(
                "SELECT * FROM surfacing_decisions WHERE ts < ? ORDER BY ts DESC LIMIT ?",
                (cursor, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM surfacing_decisions ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def get(self, decision_id: str) -> SurfacingRecord | None:
        row = self._conn.execute(
            "SELECT * FROM surfacing_decisions WHERE id=?", (decision_id,)
        ).fetchone()
        return _row_to_record(row) if row else None


# ---------------------------------------------------------------------------
# Process-wide handle, keyed on the active db_path (mirrors trust.get_writer).
# ---------------------------------------------------------------------------
_store: SurfacingStore | None = None
_store_lock = threading.Lock()


def get_store() -> SurfacingStore:
    global _store
    current = db_path()
    if _store is None or _store._db_path != current:
        with _store_lock:
            if _store is None or _store._db_path != current:
                _store = SurfacingStore()
    return _store


def reset_for_tests() -> None:
    """Drop the cached store so fixtures that swap MEMORY_DIR get a fresh db path."""
    global _store
    with _store_lock:
        _store = None
