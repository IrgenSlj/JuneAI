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
    "research": ("web_search", "fetch_url"),
    "files": ("read_pdf", "read_webpage", "list_directory", "read_file", "search_files"),
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


# ---------------------------------------------------------------------------
# The manifest on disk outlives the code that wrote it.
# ---------------------------------------------------------------------------


def _load(tmp_path, body: str):
    from june_brain.skills.manifest import load_manifest

    target = tmp_path / "skills.toml"
    target.write_text(body)
    return load_manifest(target)


_RESEARCH_NO_SCOPES = (
    "[skill.research]\n"
    "enabled = true\n"
    'command = "python"\n'
    'args = ["-m", "june_skill_research"]\n'
    'description = "Web search."\n'
)


def test_a_manifest_predating_declared_scopes_still_gets_the_contract(tmp_path) -> None:
    """The bug this catches disarmed the contract check on every upgraded install.

    `declared_scopes` was added after the manifest format shipped. Every other
    field falls back to DEFAULT_MANIFEST when the file omits it; this one did
    not, so it loaded empty — and `exceeds_declared_scopes` treats empty as "no
    contract to violate". calendar, files and research all ran ungoverned.
    """
    manifest = _load(tmp_path, _RESEARCH_NO_SCOPES)

    assert manifest.entries["research"].declared_scopes == ["read_network"]


def test_an_explicit_empty_contract_is_respected(tmp_path) -> None:
    """Omitting the field means "not stated"; writing [] means "stated as none"."""
    manifest = _load(tmp_path, _RESEARCH_NO_SCOPES + "declared_scopes = []\n")

    assert manifest.entries["research"].declared_scopes == []


def test_a_retired_skill_in_a_persisted_manifest_is_not_spawned(tmp_path) -> None:
    """D.5c deleted skills/health and skills/daily; installs kept spawning them."""
    from june_brain.skills.manifest import RETIRED_SKILL_KEYS

    manifest = _load(
        tmp_path,
        "[skill.health]\n"
        "enabled = true\n"
        'command = "python"\n'
        'args = ["-m", "june_skill_health"]\n'
        "\n" + _RESEARCH_NO_SCOPES,
    )

    assert "health" in RETIRED_SKILL_KEYS
    assert "health" not in manifest.entries
    assert "research" in manifest.entries


def test_no_retired_skill_is_also_a_default_skill() -> None:
    """A key in both sets would be dropped on load and re-added right after."""
    from june_brain.skills.manifest import DEFAULT_MANIFEST, RETIRED_SKILL_KEYS

    overlap = sorted(RETIRED_SKILL_KEYS & set(DEFAULT_MANIFEST.entries))
    assert overlap == [], f"{overlap} is both retired and bundled"
