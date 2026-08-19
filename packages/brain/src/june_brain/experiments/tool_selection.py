"""Tool-selection accuracy — does the model reach for the right tool, or any tool?

D.5a cut `JUNE_TOOLS_GEMMA` from 24 tools to 5 on the argument that a small
model picking between near-synonyms is where wrong calls come from, and that
`tool_aliases.py` existed to paper over those wrong calls. D.5d is where that
argument gets checked instead of asserted.

Two failure modes matter and they pull in opposite directions, so a single
accuracy number hides the trade:

- **Wrong tool.** The model acted, on the wrong capability. This is what a
  crowded, overlapping tool surface produces.
- **Spurious call.** The model called a tool when the turn needed a plain
  answer. This is what an *instructive* prompt produces — telling a 2B model
  about tools makes it want to use them — and it gets worse, not better, as the
  tool list shrinks and each remaining tool looks more applicable.

The corpus carries `None` cases for exactly that reason: a surface that scores
100% on tool turns and calls `remember` on "hello" is not more reliable, it is
differently unreliable, and the Glass Box shows the user every spurious call.

Pure and dependency-free so the scoring is unit-tested without a live model;
``tools/tool_selection_harness.py`` drives real turns against Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SelectionCase:
    """One utterance and the tool it should reach for (``None`` = answer directly)."""

    utterance: str
    expected: str | None
    note: str = ""


# The corpus. Utterances are written the way a user actually types — lowercase,
# elliptical, sometimes burying the request mid-sentence — because a benchmark
# of well-formed commands measures the benchmark author, not the model.
CORPUS: tuple[SelectionCase, ...] = (
    # -- remember ---------------------------------------------------------
    SelectionCase("remember that my sister is called Mira", "remember"),
    SelectionCase("please keep in mind that I'm vegetarian", "remember"),
    SelectionCase("note that I work best in the early morning", "remember"),
    SelectionCase("my daughter's birthday is on the 3rd of March, don't forget that", "remember"),
    SelectionCase("i want you to remember I get anxious about flying", "remember"),
    # -- forget -----------------------------------------------------------
    SelectionCase("forget what I told you about my old address", "forget"),
    SelectionCase("please delete the note about my job at Acme", "forget"),
    SelectionCase("drop the thing you saved about my coffee order", "forget"),
    SelectionCase("i'd rather you didn't keep the memory about my ex", "forget"),
    # -- list_promises ----------------------------------------------------
    SelectionCase("what are you working on for me?", "list_promises"),
    SelectionCase("what's still outstanding?", "list_promises"),
    SelectionCase("show me my open promises", "list_promises"),
    SelectionCase("remind me what you're carrying right now", "list_promises"),
    # -- update_promise ---------------------------------------------------
    SelectionCase("the passport renewal is done", "update_promise"),
    SelectionCase("cancel the flight booking you were tracking", "update_promise"),
    SelectionCase("put the tax return on hold for now", "update_promise"),
    SelectionCase("i finished the thing about the dentist", "update_promise"),
    # -- no tool ----------------------------------------------------------
    SelectionCase("hello", None, "greeting"),
    SelectionCase("how are you today?", None, "small talk"),
    SelectionCase("what can you actually do?", None, "capability question"),
    SelectionCase("thanks, that's helpful", None, "acknowledgement"),
    SelectionCase("what's the difference between a promise and a reminder?", None, "asks about a concept, does not invoke it"),
    SelectionCase("i had a long day", None, "shares feeling; the floor says do not volunteer observations"),
    SelectionCase("tell me a joke", None, "unrelated request"),
)


@dataclass
class SelectionResult:
    """What the model did on one case.

    ``called`` is the *first* tool of the turn and ``all_called`` is every tool
    it reached in the same turn. Both are scored, because a model that answers
    "the passport renewal is done" by calling `list_promises` and then
    `update_promise` has done the right thing in two steps, and a first-call
    metric alone would record that as a failure.
    """

    case: SelectionCase
    called: str | None
    raw_called: str | None = None
    all_called: tuple[str, ...] = ()

    @property
    def correct(self) -> bool:
        return self.called == self.case.expected

    @property
    def reached(self) -> bool:
        """The right tool ran at some point in the turn (or nothing ran, correctly)."""
        if self.case.expected is None:
            return not self.all_called and self.called is None
        return self.case.expected in self.all_called or self.called == self.case.expected

    @property
    def wrong_tool(self) -> bool:
        """Acted, but on the wrong capability."""
        return self.case.expected is not None and self.called is not None and not self.correct

    @property
    def missed(self) -> bool:
        """Should have acted and did not."""
        return self.case.expected is not None and self.called is None

    @property
    def spurious(self) -> bool:
        """Called a tool on a turn that needed a plain answer."""
        return self.case.expected is None and self.called is not None

    @property
    def alias_fired(self) -> bool:
        """The alias table rewrote the name the model emitted."""
        return self.raw_called is not None and self.raw_called != self.called


@dataclass
class SelectionReport:
    """Aggregate over a whole corpus run."""

    results: list[SelectionResult] = field(default_factory=list)

    def add(self, result: SelectionResult) -> None:
        self.results.append(result)

    def summary(self) -> dict[str, float]:
        total = len(self.results)
        if total == 0:
            return {
                "n": 0.0, "accuracy": 0.0, "reached_accuracy": 0.0,
                "tool_turn_accuracy": 0.0,
                "abstention_accuracy": 0.0, "wrong_tool": 0.0,
                "missed": 0.0, "spurious": 0.0, "alias_fired": 0.0,
            }
        tool_turns = [r for r in self.results if r.case.expected is not None]
        quiet_turns = [r for r in self.results if r.case.expected is None]
        return {
            "n": float(total),
            "accuracy": sum(r.correct for r in self.results) / total,
            "reached_accuracy": sum(r.reached for r in self.results) / total,
            "tool_turn_accuracy": (
                sum(r.correct for r in tool_turns) / len(tool_turns) if tool_turns else 0.0
            ),
            "abstention_accuracy": (
                sum(r.correct for r in quiet_turns) / len(quiet_turns) if quiet_turns else 0.0
            ),
            "wrong_tool": float(sum(r.wrong_tool for r in self.results)),
            "missed": float(sum(r.missed for r in self.results)),
            "spurious": float(sum(r.spurious for r in self.results)),
            "alias_fired": float(sum(r.alias_fired for r in self.results)),
        }

    def confusions(self) -> list[tuple[str, str]]:
        """(expected, called) pairs the model got wrong, for reading the failure shape."""
        return [
            (r.case.expected or "-none-", r.called or "-none-")
            for r in self.results
            if not r.correct
        ]


def render_report(report: SelectionReport) -> str:
    """Human-readable summary, in the shape the reliability harness prints."""
    s = report.summary()
    lines = [
        f"tool-selection accuracy over n={int(s['n'])} cases",
        "",
        f"  first-call accuracy   {s['accuracy']:.1%}",
        f"  reached-tool accuracy {s['reached_accuracy']:.1%}",
        f"  tool turns correct    {s['tool_turn_accuracy']:.1%}",
        f"  quiet turns correct   {s['abstention_accuracy']:.1%}",
        "",
        f"  wrong tool            {int(s['wrong_tool'])}",
        f"  missed (no call)      {int(s['missed'])}",
        f"  spurious call         {int(s['spurious'])}",
        f"  alias table fired     {int(s['alias_fired'])}",
    ]
    confusions = report.confusions()
    if confusions:
        lines.append("")
        lines.append("  failures (expected -> called):")
        for expected, called in confusions:
            lines.append(f"    {expected} -> {called}")
    return "\n".join(lines)
