"""The tool alias and parameter-normalization table (Phase 8.4).

391 lines that rewrite the name and the arguments of every tool call a local
model emits, and until now none of it was tested. The 2026-07-27 audit flagged
it as the largest untested surface in the repo.

It is a *correctness* risk rather than a gate bypass — resolution runs before
the guard, so the action gate always classifies the canonical name, and there is
a test below pinning that ordering. What a bug here does instead is quietly send
the wrong tool the wrong arguments: a journal entry dropped, a goal saved with
an empty title, a reminder that never gets a date.

The table was clean when these tests were written — no duplicate aliases, no
shadowing, idempotent, every required argument surviving normalization. That is
what makes this a regression net: it records a state worth keeping.
"""

from __future__ import annotations

from collections import defaultdict

import pytest
from june_brain.tool_aliases import TOOL_ALIASES, resolve_tool_call

ALIAS_PAIRS = [
    (alias, canonical)
    for canonical, spec in TOOL_ALIASES.items()
    for alias in spec.aliases
]


# -- table integrity ----------------------------------------------------


def test_no_alias_is_claimed_by_two_tools() -> None:
    """A duplicate would make resolution depend on dict ordering.

    It would still resolve — to whichever entry happened to come first — so the
    symptom is a tool call silently going somewhere else after an unrelated edit.
    """
    owners: dict[str, list[str]] = defaultdict(list)
    for alias, canonical in ALIAS_PAIRS:
        owners[alias].append(canonical)
    clashes = {a: c for a, c in owners.items() if len(c) > 1}
    assert clashes == {}, f"aliases claimed twice: {clashes}"


def test_no_alias_shadows_a_canonical_tool_name() -> None:
    """An alias equal to another tool's real name would hijack that tool."""
    shadowed = {a: c for a, c in ALIAS_PAIRS if a in TOOL_ALIASES}
    assert shadowed == {}, f"aliases shadowing real tool names: {shadowed}"


def test_no_tool_lists_itself_as_its_own_alias() -> None:
    assert [c for _a, c in ALIAS_PAIRS if _a == c] == []


@pytest.mark.parametrize("canonical", sorted(TOOL_ALIASES))
def test_resolution_is_idempotent(canonical: str) -> None:
    """Resolving twice must equal resolving once.

    The loop breaks after the first match, but a normalizer that reshaped its
    own output would drift on a second pass — and the dispatch path is not
    obliged to call this exactly once forever.
    """
    once = resolve_tool_call(canonical, {"title": "x", "entry": "y"})
    twice = resolve_tool_call(*once)
    assert once == twice


# -- resolution ---------------------------------------------------------


@pytest.mark.parametrize("alias,canonical", ALIAS_PAIRS, ids=[a for a, _ in ALIAS_PAIRS])
def test_every_declared_alias_reaches_its_tool(alias: str, canonical: str) -> None:
    name, _args = resolve_tool_call(alias, {"title": "T", "date": "2026-08-01"})
    assert name == canonical


@pytest.mark.parametrize("canonical", sorted(TOOL_ALIASES))
def test_a_canonical_name_stays_canonical(canonical: str) -> None:
    name, _args = resolve_tool_call(canonical, {"title": "T"})
    assert name == canonical


def test_an_unknown_tool_passes_through_untouched() -> None:
    """The table must not be a filter. A tool it has never heard of is not its business."""
    args = {"anything": "at all", "nested": {"x": 1}}
    name, out = resolve_tool_call("some_skill_tool", args)
    assert name == "some_skill_tool"
    assert out == args


def test_consequential_tool_names_are_never_rewritten() -> None:
    """Nothing in this table may redirect a gated action somewhere else."""
    for name in ("send_telegram_message", "web_search", "fetch_url", "run_shell"):
        resolved, _ = resolve_tool_call(name, {"text": "hello"})
        assert resolved == name


@pytest.mark.parametrize("args", [None, {}], ids=["none", "empty"])
def test_missing_arguments_do_not_crash(args) -> None:  # type: ignore[no-untyped-def]
    for name in ("save_goal", "unknown_tool", "save_journal_entry"):
        resolved, out = resolve_tool_call(name, args)
        assert isinstance(resolved, str)
        assert isinstance(out, dict)


def test_the_caller_s_dict_is_not_mutated() -> None:
    """Dispatch reuses the args it passed in; a shared mutation would surprise it."""
    original = {"goal": "run a marathon"}
    snapshot = dict(original)
    resolve_tool_call("save_goal", original)
    assert original == snapshot


# -- parameter mapping --------------------------------------------------


def test_a_model_s_alternate_key_names_are_accepted() -> None:
    """The entire reason this table exists: small local models rename things."""
    _, out = resolve_tool_call("save_reminder", {"event": "Dentist", "when": "2026-08-03"})
    assert out["title"] == "Dentist"
    assert out["date"] == "2026-08-03"


def test_the_canonical_key_wins_over_an_alternate() -> None:
    """When the model supplies both, the one the tool actually declares is right."""
    _, out = resolve_tool_call("save_calendar_item", {"title": "Canonical", "event": "Alternate"})
    assert out["title"] == "Canonical"


def test_an_empty_canonical_value_falls_back_to_the_alternate() -> None:
    """An empty string is the model failing to fill a field, not a deliberate blank."""
    _, out = resolve_tool_call("save_calendar_item", {"title": "", "event": "Real event"})
    assert out["title"] == "Real event"


def test_an_unfilled_field_normalizes_to_an_empty_string() -> None:
    """The tool's schema declares defaults for these; the table must not send None."""
    _, out = resolve_tool_call("save_calendar_item", {"event": "x", "when": "2026-08-03"})
    assert out["details"] == ""
    assert out["time"] == ""


@pytest.mark.parametrize(
    "canonical,supplied,field,expected",
    [
        ("save_calendar_item", {"event": "Dentist", "when": "2026-08-03"}, "title", "Dentist"),
        ("save_calendar_item", {"event": "Dentist", "when": "2026-08-03"}, "date", "2026-08-03"),
        ("save_birthday", {"name": "Sam", "day": "2026-09-01"}, "title", "Sam"),
        ("save_trip", {"title": "Lisbon", "date": "2026-10-02", "note": "flights booked"}, "details", "flights booked"),
    ],
)
def test_alternate_keys_across_the_table(
    canonical: str, supplied: dict, field: str, expected: str
) -> None:
    _, out = resolve_tool_call(canonical, supplied)
    assert out[field] == expected


# -- what the table no longer does --------------------------------------


def test_a_retired_tool_name_is_not_rewritten_into_a_live_one() -> None:
    """`save_journal_entry` used to reroute a dated JSON blob to the calendar.

    That normalizer went with the tool (ADR 0032). What matters now is that the
    name passes through untouched rather than being quietly turned into a call
    the model did not make: the merge denylist is what stops a retired name, and
    it can only stop a name the table did not already rewrite.
    """
    from june_brain.tools import RETIRED_TOOL_NAMES

    entry = '{"title": "Dentist", "date": "2026-08-03"}'
    name, out = resolve_tool_call("save_journal_entry", {"entry": entry})
    assert name == "save_journal_entry"
    assert name in RETIRED_TOOL_NAMES
    assert out == {"entry": entry}


def test_no_alias_resolves_to_a_retired_tool() -> None:
    """An alias pointing at a deleted tool is a call that can only fail."""
    from june_brain.tools import RETIRED_TOOL_NAMES

    leaked = sorted(set(TOOL_ALIASES) & RETIRED_TOOL_NAMES)
    assert leaked == [], f"the alias table still routes to retired tools: {leaked}"


# -- the ordering the guard depends on ----------------------------------


def test_the_action_gate_classifies_the_resolved_name() -> None:
    """Aliasing runs before dispatch, so the guard sees the canonical name.

    This is what makes the table a correctness risk rather than a security one.
    If the order ever inverted, a model could reach a gated tool under a
    read-sounding alias and the gate would classify the wrong name.
    """
    from june_brain.guard.actions import classify_action

    for alias, canonical in ALIAS_PAIRS:
        resolved, _ = resolve_tool_call(alias, {})
        assert classify_action(resolved) == classify_action(canonical), (
            f"{alias} resolves to {resolved}, classified differently from {canonical}"
        )


def test_no_alias_reaches_a_networked_or_executing_tool() -> None:
    """Nothing in this table should quietly widen what a call can do.

    Not load-bearing — the gate would still catch it, because it classifies the
    resolved name. It is asserted anyway so that adding such an alias is a
    deliberate act with a failing test in front of it.
    """
    from june_brain.guard.actions import classify_action

    escalating = {
        canonical
        for _alias, canonical in ALIAS_PAIRS
        if classify_action(canonical) in ("write_network", "execute")
    }
    assert escalating == set()


def test_the_normalizer_does_not_drop_arguments_it_does_not_know_about() -> None:
    """A repair pass that discards valid input is worse than no repair pass.

    `_normalize_save_calendar_item` rebuilt a fixed four-key dict, so every
    other argument the model supplied vanished. Measured on real calls (D.5d):
    `status` and `source` were dropped every time. `source` is the provenance
    tag the memory browser uses to say where a saved item came from, so the
    normalizer was quietly degrading the record it exists to improve.
    """
    supplied = {
        "title": "Dentist",
        "date": "2026-09-03",
        "time": "09:00",
        "status": "planned",
        "source": "conversation",
    }
    _, out = resolve_tool_call("save_calendar_item", dict(supplied))

    for key, value in supplied.items():
        assert out.get(key) == value, f"{key} was dropped or altered"


def test_the_normalizer_still_folds_alternate_spellings() -> None:
    """And leaves only one spelling of each field behind."""
    name, out = resolve_tool_call(
        "save_reminder",
        {"event": "Dentist", "when": "2026-09-03", "at": "09:00", "note": "bring xrays"},
    )

    assert name == "save_calendar_item"
    assert out["title"] == "Dentist"
    assert out["date"] == "2026-09-03"
    assert out["time"] == "09:00"
    assert out["details"] == "bring xrays"
    for alt in ("event", "when", "at", "note"):
        assert alt not in out, f"{alt} survived alongside its canonical form"
