"""Temporal context — passive, read-time time-awareness for the assembler (D.1).

An LLM has no clock. Without this block June cannot reliably answer "what day is
it", reason about "this evening" / "tomorrow", or notice that a deadline is
near. This folds the current wall-clock time into the assembled context at read
time only — no process, no timer (ADR 0016): the value is computed when, and
only when, a turn is actually assembled.

Pure builder: ``build_temporal_block`` takes the time as data (no clock inside),
so it is fully deterministic and testable; the assembler injects the clock. This
mirrors the Silence Model discipline (ADR 0023): read-time producers feed a pure
builder, and no clock lives inside the builder.
"""

from __future__ import annotations

from datetime import datetime

# Time-of-day buckets by local hour. Kept coarse and human — this is a hint for
# June's phrasing ("good evening"), not a precise schedule.
_MORNING = range(5, 12)
_AFTERNOON = range(12, 17)
_EVENING = range(17, 21)


def _time_of_day(hour: int) -> str:
    if hour in _MORNING:
        return "morning"
    if hour in _AFTERNOON:
        return "afternoon"
    if hour in _EVENING:
        return "evening"
    return "night"


def build_temporal_block(now: datetime) -> str:
    """Return a one-line temporal-context system block for the given *local* time.

    Formatted explicitly (not via platform-specific strftime directives like
    ``%-d``) so it renders identically on every OS.
    """
    day_name = now.strftime("%A")
    month_name = now.strftime("%B")
    hour12 = now.hour % 12 or 12
    ampm = "AM" if now.hour < 12 else "PM"
    tod = _time_of_day(now.hour)
    stamp = f"{day_name}, {now.day} {month_name} {now.year}, {hour12}:{now.minute:02d} {ampm}"
    return f"[Temporal context] It is currently {stamp} ({tod}), the user's local time."
