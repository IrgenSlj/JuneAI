"""Regression tests for daily orchestration default-schedule creation.

These cover the crash cluster fixed alongside this file: create_default_schedules
used to raise TypeError twice — once on ``MEMORY_DIR / "june.db"`` (str, not Path)
and once on ``Schedule(...)`` without the required ``id``. The setup flow swallowed
the exception, so no default schedules were ever created and no error surfaced.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point june.db at a tmp dir and reset the per-thread connection cache."""
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    return tmp_path


def test_create_default_schedules_creates_three(isolated_db: Path) -> None:
    from june_brain.orchestration import create_default_schedules

    created = create_default_schedules("alice")
    names = {s["name"] for s in created}
    assert names == {"Morning briefing", "Evening review", "Weekly review"}
    # Every created schedule got a backfilled id.
    assert all(s["id"] for s in created)


def test_create_default_schedules_is_idempotent(isolated_db: Path) -> None:
    from june_brain.orchestration import create_default_schedules

    first = create_default_schedules("alice")
    assert len(first) == 3
    second = create_default_schedules("alice")
    assert second == []
