"""PinnedState — the anchor that survives compaction.

Intentionally dependency-free beyond stdlib so loop/interface.py can import
it without a cycle (interface imports PinnedState; pinned_state imports nothing
from loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

_LIST_CAP = 8


def _dedup_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


@dataclass
class PinnedState:
    """Small structured anchor (~100-300 tokens) that compaction merges into."""

    user_goal: str | None = None
    constraints: list[str] = field(default_factory=list)
    confirmed_facts: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    last_tool_outcomes: list[str] = field(default_factory=list)
    updated_at: str = ""

    def is_empty(self) -> bool:
        return (
            not self.user_goal
            and not self.constraints
            and not self.confirmed_facts
            and not self.open_questions
            and not self.last_tool_outcomes
        )

    def merge(
        self,
        *,
        user_goal: str | None = None,
        constraints: list[str] | None = None,
        confirmed_facts: list[str] | None = None,
        open_questions: list[str] | None = None,
        last_tool_outcomes: list[str] | None = None,
    ) -> None:
        """Update in place — never blank an existing field."""
        if user_goal:
            self.user_goal = user_goal

        if constraints:
            merged = _dedup_ordered(self.constraints + constraints)
            self.constraints = merged[-_LIST_CAP:]

        if confirmed_facts:
            merged = _dedup_ordered(self.confirmed_facts + confirmed_facts)
            self.confirmed_facts = merged[-_LIST_CAP:]

        if open_questions:
            merged = _dedup_ordered(self.open_questions + open_questions)
            self.open_questions = merged[-_LIST_CAP:]

        if last_tool_outcomes:
            merged = _dedup_ordered(self.last_tool_outcomes + last_tool_outcomes)
            self.last_tool_outcomes = merged[-_LIST_CAP:]

        self.updated_at = datetime.now(UTC).isoformat()

    def to_block(self) -> str:
        """Render a compact text block for the assembler; returns '' when empty."""
        if self.is_empty():
            return ""
        lines: list[str] = []
        if self.user_goal:
            lines.append(f"Goal: {self.user_goal}")
        if self.constraints:
            lines.append("Constraints:")
            for c in self.constraints:
                lines.append(f"  - {c}")
        if self.confirmed_facts:
            lines.append("Confirmed:")
            for f in self.confirmed_facts:
                lines.append(f"  - {f}")
        if self.open_questions:
            lines.append("Open questions:")
            for q in self.open_questions:
                lines.append(f"  - {q}")
        if self.last_tool_outcomes:
            lines.append("Last tool outcomes:")
            for o in self.last_tool_outcomes:
                lines.append(f"  - {o}")
        return "\n".join(lines)
