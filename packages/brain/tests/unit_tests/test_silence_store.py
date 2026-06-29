"""Tests for Silence Model storage, batching + event-boundary drain (ADR 0023, slice 6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from june_brain.silence import (
    SurfacingCandidate,
    SurfacingContext,
    SurfacingStore,
    decide,
)
from june_brain.trust import LedgerReader


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite
    import june_brain.silence as silence_pkg
    import june_brain.trust as trust_pkg

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    trust_pkg.reset_for_tests()
    silence_pkg.reset_for_tests()
    yield
    trust_pkg.reset_for_tests()
    silence_pkg.reset_for_tests()


def _decide_batch() -> tuple[SurfacingCandidate, object]:
    cand = SurfacingCandidate(candidate_id="c1", kind="promise_nudge", salience=0.4)
    ctx = SurfacingContext(now="2026-06-29T12:00:00+00:00", presence_state="absent")
    return cand, decide(cand, ctx)  # absent + low salience -> batch


def test_record_round_trips() -> None:
    store = SurfacingStore()
    cand, decision = _decide_batch()
    rec = store.record(cand, decision)
    assert rec.action == "batch"
    fetched = store.get(rec.id)
    assert fetched is not None
    assert fetched.candidate_id == "c1"
    assert fetched.reason == decision.reason
    assert fetched.features["presence_state"] == "absent"


def test_record_writes_a_ledger_entry() -> None:
    store = SurfacingStore()
    cand, decision = _decide_batch()
    store.record(cand, decision)
    ledger = [e for e in LedgerReader().page(limit=50) if e.kind == "action"]
    assert len(ledger) == 1
    assert ledger[0].actor == "june"
    assert ledger[0].payload["event"] == "surfacing_decision"
    assert ledger[0].payload["decision"] == "batch"
    assert LedgerReader().verify_chain().ok is True


def test_dismissals_for_similar_counts_dismissed_outcomes() -> None:
    store = SurfacingStore()
    cand, decision = _decide_batch()
    ids = [store.record(cand, decision).id for _ in range(3)]
    assert store.dismissals_for_similar("promise_nudge") == 0
    store.set_outcome(ids[0], "dismissed")
    store.set_outcome(ids[1], "dismissed")
    store.set_outcome(ids[2], "engaged")
    assert store.dismissals_for_similar("promise_nudge") == 2
    assert store.dismissals_for_similar("deadline") == 0  # different kind


def test_set_outcome_rejects_invalid() -> None:
    store = SurfacingStore()
    cand, decision = _decide_batch()
    rec = store.record(cand, decision)
    with pytest.raises(ValueError):
        store.set_outcome(rec.id, "bogus")


# ---------------------------------------------------------------------------
# Batching + drain — drains ONLY when the function is called (no timer)
# ---------------------------------------------------------------------------


def test_batch_items_accumulate_and_drain_only_on_call() -> None:
    store = SurfacingStore()
    cand, decision = _decide_batch()
    store.record(cand, decision)
    store.record(SurfacingCandidate(candidate_id="c2", kind="promise_nudge", salience=0.4), decision)

    # Held until explicitly drained — nothing surfaces on its own.
    assert store.held_count() == 2
    assert len(store.held_digest()) == 2

    drained = store.drain_digest()
    assert len(drained) == 2
    # A second drain returns nothing: the first marked them surfaced.
    assert store.drain_digest() == []
    assert store.held_count() == 0


def test_now_and_suppress_are_not_in_the_digest() -> None:
    store = SurfacingStore()
    # A "now" decision (urgent deadline) is not batched.
    now_cand = SurfacingCandidate(
        candidate_id="d1", kind="deadline", deadline_at="2026-06-29T12:30:00+00:00"
    )
    now_ctx = SurfacingContext(now="2026-06-29T12:00:00+00:00", presence_state="present-idle")
    store.record(now_cand, decide(now_cand, now_ctx))
    assert store.held_count() == 0


def test_page_newest_first_with_cursor() -> None:
    store = SurfacingStore()
    cand, decision = _decide_batch()
    recs = [store.record(cand, decision, ts=f"2026-06-29T12:00:0{i}+00:00") for i in range(5)]
    page = store.page(limit=2)
    assert [r.id for r in page] == [recs[4].id, recs[3].id]
    nxt = store.page(cursor=page[-1].ts, limit=2)
    assert [r.id for r in nxt] == [recs[2].id, recs[1].id]


# ---------------------------------------------------------------------------
# Invariant: no timer / scheduler anywhere in the silence package (ADR 0016)
# ---------------------------------------------------------------------------


def test_silence_package_has_no_timer_or_scheduler() -> None:
    """No timer/scheduler machinery in the silence package (ADR 0016).

    Scans NAME tokens only, so the prose in docstrings that *describes* the
    no-timer rule does not trip the check — only real code identifiers do.
    """
    import tokenize

    import june_brain.silence as silence_pkg

    pkg_dir = Path(silence_pkg.__file__).parent
    banned = {"sleep", "Timer", "cron", "scheduler", "apscheduler"}
    offenders = []
    for py in pkg_dir.rglob("*.py"):
        names: set[str] = set()
        with open(py, "rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type == tokenize.NAME:
                    names.add(tok.string)
        hit = names & banned
        if hit:
            offenders.append(f"{py.name}:{sorted(hit)}")
    assert offenders == [], f"silence must not schedule or wake on a timer: {offenders}"
