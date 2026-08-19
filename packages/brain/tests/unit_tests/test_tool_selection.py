"""Scoring for the tool-selection benchmark (D.5d).

The math is tested without a model so the harness's numbers mean what the
report says they mean — a benchmark whose scoring is itself unverified is
worse than no benchmark, because it produces a number people quote.
"""

from __future__ import annotations

from june_brain.experiments.tool_selection import (
    CORPUS,
    SelectionCase,
    SelectionReport,
    SelectionResult,
    render_report,
)


def _r(expected, called, raw=None, all_called=None):
    return SelectionResult(
        case=SelectionCase("u", expected),
        called=called,
        raw_called=raw or called,
        all_called=tuple(all_called if all_called is not None else ([called] if called else [])),
    )


def test_the_corpus_covers_every_gemma_tool_and_abstention() -> None:
    """A benchmark that skips a tool cannot notice that tool regressing."""
    from june_brain.tools import JUNE_TOOLS_GEMMA

    expected = {c.expected for c in CORPUS if c.expected is not None}
    advertised = {t.name for t in JUNE_TOOLS_GEMMA}

    unknown = sorted(expected - advertised)
    assert unknown == [], f"the corpus expects tools the model is not offered: {unknown}"
    assert any(c.expected is None for c in CORPUS), "no abstention cases"


def test_a_correct_call_is_none_of_the_failure_modes() -> None:
    r = _r("remember", "remember")
    assert r.correct
    assert not r.wrong_tool and not r.missed and not r.spurious


def test_the_three_failure_modes_are_distinguished() -> None:
    wrong = _r("remember", "forget")
    missed = _r("remember", None)
    spurious = _r(None, "remember")

    assert wrong.wrong_tool and not wrong.missed and not wrong.spurious
    assert missed.missed and not missed.wrong_tool and not missed.spurious
    assert spurious.spurious and not spurious.wrong_tool and not spurious.missed


def test_correct_abstention_counts_as_correct() -> None:
    assert _r(None, None).correct


def test_alias_fired_only_when_the_name_was_rewritten() -> None:
    assert _r("save_calendar_item", "save_calendar_item", raw="save_reminder").alias_fired
    assert not _r("remember", "remember").alias_fired


def test_summary_separates_tool_turns_from_quiet_turns() -> None:
    """One accuracy number hides the trade the corpus exists to expose."""
    report = SelectionReport()
    for r in (_r("remember", "remember"), _r("forget", "forget"),
              _r("list_promises", None), _r(None, "remember"), _r(None, None)):
        report.add(r)
    s = report.summary()

    assert s["n"] == 5.0
    assert s["accuracy"] == 3 / 5
    assert s["reached_accuracy"] == 3 / 5
    assert s["tool_turn_accuracy"] == 2 / 3
    assert s["abstention_accuracy"] == 1 / 2
    assert s["missed"] == 1.0
    assert s["spurious"] == 1.0
    assert s["wrong_tool"] == 0.0


def test_a_two_step_answer_counts_as_reaching_the_tool() -> None:
    """list_promises then update_promise is right in two steps, not wrong once."""
    r = _r("update_promise", "list_promises", all_called=["list_promises", "update_promise"])
    assert not r.correct
    assert r.reached


def test_reaching_is_not_credit_for_calling_something_else() -> None:
    r = _r("update_promise", "list_promises", all_called=["list_promises"])
    assert not r.reached


def test_a_quiet_turn_is_only_reached_when_nothing_ran() -> None:
    assert _r(None, None).reached
    assert not _r(None, "remember").reached


def test_an_empty_report_does_not_divide_by_zero() -> None:
    s = SelectionReport().summary()
    assert s["n"] == 0.0
    assert s["accuracy"] == 0.0


def test_confusions_list_only_the_failures() -> None:
    report = SelectionReport()
    report.add(_r("remember", "remember"))
    report.add(_r("remember", "forget"))
    report.add(_r(None, "remember"))

    assert report.confusions() == [("remember", "forget"), ("-none-", "remember")]


def test_render_report_states_every_metric() -> None:
    report = SelectionReport()
    report.add(_r("remember", "forget"))
    text = render_report(report)

    for label in ("first-call accuracy", "reached-tool accuracy", "tool turns correct", "quiet turns correct",
                  "wrong tool", "missed", "spurious", "alias table fired"):
        assert label in text
    assert "remember -> forget" in text
