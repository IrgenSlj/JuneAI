"""The bundled skills' declared capability contracts (Phase 6.0).

Declaring scopes is what makes the enforcement in `guard/actions.py` do
anything: a skill with no contract has nothing to violate. So the contracts
have to be right, and "right" means they cover every tool the skill actually
advertises — no more, and no less.

Getting this wrong in either direction is a real failure. Too narrow and a
working tool starts being blocked in someone's install; too wide and the
contract stops meaning anything. This test is the thing that notices.
"""

from __future__ import annotations

import pytest
from june_brain.guard.actions import classify_action
from june_brain.skills.manifest import DEFAULT_MANIFEST

# The tools each bundled skill advertises. Kept here rather than discovered by
# spawning the skills: this file must fail when someone adds a tool without
# revisiting the contract, and a runtime discovery would just silently agree.
BUNDLED_TOOLS: dict[str, tuple[str, ...]] = {
    "calendar": ("save_calendar_item", "list_calendar_items", "update_calendar_item_status"),
    "health": (
        "log_body_metrics",
        "log_water",
        "log_workout_session",
        "log_habit_completion",
        "get_today_summary",
    ),
    "research": ("web_search", "fetch_url"),
    "files": ("read_pdf", "read_webpage", "list_directory", "read_file", "search_files"),
    "daily": ("log_mood", "save_journal_entry", "track_goal", "save_open_loop"),
    "telegram": ("send_telegram_message", "get_telegram_bot_status"),
}


def test_every_bundled_skill_declares_a_contract() -> None:
    """An undeclared skill is ungoverned, which is the state this slice ends."""
    for key, entry in DEFAULT_MANIFEST.entries.items():
        assert entry.declared_scopes, f"{key} declares no scopes"


def test_the_manifest_and_this_file_describe_the_same_skills() -> None:
    assert set(BUNDLED_TOOLS) == set(DEFAULT_MANIFEST.entries)


@pytest.mark.parametrize("key", sorted(BUNDLED_TOOLS))
def test_a_contract_covers_every_tool_the_skill_advertises(key: str) -> None:
    """Too narrow: a real tool would be blocked in a user's install."""
    entry = DEFAULT_MANIFEST.entries[key]
    declared = frozenset(entry.declared_scopes)

    for tool in BUNDLED_TOOLS[key]:
        derived = classify_action(tool)
        assert derived in declared, (
            f"{key}.{tool} classifies as {derived!r}, which {key} did not declare "
            f"({sorted(declared)}). Either the contract is too narrow or the tool "
            f"is misnamed for what it does."
        )


@pytest.mark.parametrize("key", sorted(BUNDLED_TOOLS))
def test_a_contract_claims_nothing_the_skill_does_not_use(key: str) -> None:
    """Too wide: an over-declared scope is permission granted for nothing."""
    entry = DEFAULT_MANIFEST.entries[key]
    used = {classify_action(t) for t in BUNDLED_TOOLS[key]}
    unused = frozenset(entry.declared_scopes) - used
    assert not unused, f"{key} declares {sorted(unused)} but advertises no such tool"


def test_only_the_telegram_skill_may_send_data_off_the_device() -> None:
    """A capability worth asserting by name rather than leaving to a set diff."""
    senders = {
        key
        for key, entry in DEFAULT_MANIFEST.entries.items()
        if "write_network" in entry.declared_scopes
    }
    assert senders == {"telegram"}


def test_no_bundled_skill_may_execute_code() -> None:
    for key, entry in DEFAULT_MANIFEST.entries.items():
        assert "execute" not in entry.declared_scopes, key
