"""Tests for the Silence Model v1 rules (ADR 0023, slice 5)."""

from __future__ import annotations

from pathlib import Path

from june_brain.silence import (
    SurfacingCandidate,
    SurfacingContext,
    decide,
)

NOW = "2026-06-29T12:00:00+00:00"


def _candidate(**kw) -> SurfacingCandidate:
    base = {"candidate_id": "c1", "kind": "promise_nudge", "salience": 0.5}
    base.update(kw)
    return SurfacingCandidate(**base)


def _ctx(**kw) -> SurfacingContext:
    base = {"now": NOW}
    base.update(kw)
    return SurfacingContext(**base)


# ---------------------------------------------------------------------------
# Each rule branch → expected action + a reason naming the deciding feature
# ---------------------------------------------------------------------------


def test_deadline_within_urgent_window_surfaces_now() -> None:
    cand = _candidate(kind="deadline", deadline_at="2026-06-29T14:00:00+00:00")  # +2h
    d = decide(cand, _ctx(presence_state="absent"))
    assert d.action == "now"
    assert "deadline" in d.reason
    assert d.features["deadline_delta_hours"] == 2.0


def test_overdue_deadline_surfaces_now() -> None:
    cand = _candidate(kind="deadline", deadline_at="2026-06-29T09:00:00+00:00")  # -3h
    d = decide(cand, _ctx())
    assert d.action == "now"
    assert "passed" in d.reason


def test_deadline_urgency_trumps_dismissals() -> None:
    cand = _candidate(kind="deadline", deadline_at="2026-06-29T13:00:00+00:00")  # +1h
    d = decide(cand, _ctx(dismissals_for_similar=5))
    assert d.action == "now"  # urgent deadline wins over the dismissal signal


def test_dismissed_twice_suppresses() -> None:
    d = decide(_candidate(salience=0.9), _ctx(dismissals_for_similar=2, presence_state="present-idle"))
    assert d.action == "suppress"
    assert "dismissed" in d.reason
    assert "2" in d.reason


def test_high_salience_idle_no_dismissal_surfaces_now() -> None:
    d = decide(_candidate(salience=0.8), _ctx(presence_state="present-idle", dismissals_for_similar=0))
    assert d.action == "now"
    assert "salience" in d.reason


def test_high_salience_but_one_dismissal_does_not_surface_now() -> None:
    # Below the suppress threshold, but not a clean history → not "now".
    d = decide(_candidate(salience=0.8), _ctx(presence_state="present-idle", dismissals_for_similar=1))
    assert d.action == "batch"


def test_mid_task_batches() -> None:
    d = decide(_candidate(salience=0.9), _ctx(presence_state="present-active"))
    assert d.action == "batch"
    assert "mid-task" in d.reason


def test_active_thread_open_batches_even_when_idle_flag() -> None:
    d = decide(_candidate(salience=0.9), _ctx(presence_state="present-idle", active_thread_open=True))
    assert d.action == "batch"
    assert "mid-task" in d.reason


def test_absent_batches() -> None:
    d = decide(_candidate(salience=0.5), _ctx(presence_state="absent"))
    assert d.action == "batch"
    assert "away" in d.reason


def test_default_non_urgent_low_salience_batches() -> None:
    d = decide(_candidate(salience=0.3), _ctx(presence_state="present-idle"))
    assert d.action == "batch"
    assert "low urgency" in d.reason


def test_features_capture_all_inputs() -> None:
    d = decide(
        _candidate(salience=0.42, kind="contradiction"),
        _ctx(presence_state="present-idle", dismissals_for_similar=1, local_time_bucket="evening"),
    )
    f = d.features
    assert f["salience"] == 0.42
    assert f["kind"] == "contradiction"
    assert f["dismissals_for_similar"] == 1
    assert f["presence_state"] == "present-idle"
    assert f["local_time_bucket"] == "evening"


def test_malformed_deadline_is_ignored_not_raised() -> None:
    d = decide(_candidate(deadline_at="not-a-date"), _ctx(presence_state="absent"))
    assert d.action == "batch"  # treated as no deadline
    assert d.features["deadline_delta_hours"] is None


def test_far_future_deadline_is_not_urgent() -> None:
    cand = _candidate(kind="deadline", deadline_at="2026-07-15T12:00:00+00:00")  # weeks away
    d = decide(cand, _ctx(presence_state="absent"))
    assert d.action == "batch"


# ---------------------------------------------------------------------------
# Invariant: the direct-reply path has NO dependency on the silence policy
# ---------------------------------------------------------------------------


def test_loop_package_does_not_import_silence() -> None:
    """June always answers when spoken to; the reply path must not route through decide.

    Structural guard: no module under june_brain/loop references the silence
    package. If a future change wires silence into the reply path, this fails.
    """
    import june_brain.loop as loop_pkg

    loop_dir = Path(loop_pkg.__file__).parent
    offenders = []
    for py in loop_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "silence" in text:
            offenders.append(py.name)
    assert offenders == [], f"loop modules must not reference silence: {offenders}"
