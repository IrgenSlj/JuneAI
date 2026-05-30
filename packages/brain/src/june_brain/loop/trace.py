"""Per-turn harness trace — the glass-box record of one turn.

A :class:`TurnTrace` is an ordered list of :class:`TraceEvent`s capturing
everything the loop did for a single turn: the rendered prompt sent to the
model (the "LLM factory" input), each loop iteration's intermediate output,
tool calls and their full results, the model's reasoning, and compaction.

Each event carries a short ``summary`` (the one-line the activity terminal
shows collapsed) and a full ``detail`` (revealed on expand). Traces persist
to ``<datadir>/traces/<turn_id>.json`` so a past turn can be reopened.

Privacy: trace files hold the assembled prompt — recalled memories, character,
and history in the clear. They are local-only and live inside the data dir that
local-only mode governs. Writing is always best-effort: a trace-write failure
must never break a turn (see :meth:`TraceStore.write`).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

# Event kinds a trace can record. Superset of StreamEvent types — a trace is the
# durable record, StreamEvents are the live wire.
TraceKind = str  # one of: prompt, iteration, recall, tool_call, tool_result,
#                  reasoning, compaction, provenance, done, error


@dataclass
class TraceEvent:
    """One recorded step in a turn."""

    seq: int
    ts: float  # epoch seconds
    kind: TraceKind
    summary: str  # short, collapsed-line text
    detail: str = ""  # full content, revealed on expand

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TurnTrace:
    """The full record of one harness turn."""

    turn_id: str
    user_id: str
    started_at: float = field(default_factory=time.time)
    events: list[TraceEvent] = field(default_factory=list)

    def record(self, kind: TraceKind, summary: str, detail: str = "") -> TraceEvent:
        """Append an event and return it. seq is assigned in order."""
        event = TraceEvent(
            seq=len(self.events),
            ts=time.time(),
            kind=kind,
            summary=summary,
            detail=detail,
        )
        self.events.append(event)
        return event

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "user_id": self.user_id,
            "started_at": self.started_at,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TurnTrace:
        return cls(
            turn_id=str(data.get("turn_id", "")),
            user_id=str(data.get("user_id", "")),
            started_at=float(data.get("started_at", 0.0)),
            events=[
                TraceEvent(
                    seq=int(e.get("seq", i)),
                    ts=float(e.get("ts", 0.0)),
                    kind=str(e.get("kind", "")),
                    summary=str(e.get("summary", "")),
                    detail=str(e.get("detail", "")),
                )
                for i, e in enumerate(data.get("events", []))
            ],
        )


class TraceStore:
    """Reads and writes turn traces under ``<datadir>/traces/``.

    All writes are best-effort: failures are swallowed so the chat turn that
    produced the trace can never be broken by a disk or serialization error.
    """

    def _dir(self):
        from june_brain.datadir import layout  # noqa: PLC0415

        return layout.traces_dir()

    def write(self, trace: TurnTrace) -> bool:
        """Persist a trace. Returns True on success, False on any failure."""
        try:
            d = self._dir()
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{trace.turn_id}.json"
            path.write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))
            return True
        except Exception:
            # Trace persistence is a debug convenience, never load-bearing.
            return False

    def read(self, turn_id: str) -> TurnTrace | None:
        try:
            path = self._dir() / f"{turn_id}.json"
            if not path.exists():
                return None
            return TurnTrace.from_dict(json.loads(path.read_text()))
        except Exception:
            return None

    def list_recent(self, limit: int = 50) -> list[dict]:
        """List recent traces newest-first, as lightweight summaries.

        Each entry: {turn_id, started_at, event_count}. Reading every file's
        full body would be wasteful, so we parse minimally.
        """
        try:
            d = self._dir()
            if not d.exists():
                return []
            files = sorted(
                d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )[:limit]
            out: list[dict] = []
            for p in files:
                try:
                    data = json.loads(p.read_text())
                    out.append(
                        {
                            "turn_id": str(data.get("turn_id", p.stem)),
                            "started_at": float(data.get("started_at", p.stat().st_mtime)),
                            "event_count": len(data.get("events", [])),
                        }
                    )
                except Exception:
                    continue
            return out
        except Exception:
            return []
