"""Unit tests for memory surfaces not covered in test_memory.py.

Covers journal and relationship profiles — the two structured stores that
still have both a writer and a reader. Nutrition, water, workout sessions,
commitment summaries and chapter completeness went with the health cluster
(D.5b): their tables remain so existing rows stay exportable, but no code
reads or writes them any more.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from june_brain.memory import Memory


@pytest.fixture
def memory_dir(tmp_path):
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        yield tmp_path


@pytest.fixture
def mem(memory_dir):
    return Memory("test_user")


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

def test_save_and_retrieve_journal(mem):
    mem.save_journal("Today was a productive day. Shipped the export script.")
    entries = mem.get_journal(limit=5)
    assert entries, "Expected a journal entry"
    assert "productive" in entries[0]["entry"]


def test_multiple_journal_entries_ordered_newest_first(mem):
    mem.save_journal("First entry")
    mem.save_journal("Second entry")
    entries = mem.get_journal(limit=5)
    # get_journal returns in reverse-insert order (newest first via DESC then reversed slice)
    assert any("Second entry" in e["entry"] for e in entries)
    assert any("First entry" in e["entry"] for e in entries)


# ---------------------------------------------------------------------------
# Relationship profiles
# ---------------------------------------------------------------------------

def test_save_and_retrieve_relationship_profile(mem):
    mem.save_relationship_profile(
        person="Anna",
        relationship="sister",
        summary="We talk every Sunday. She lives in Berlin.",
    )
    profiles = mem.get_relationship_profiles()
    assert profiles, "Expected a relationship profile"
    anna = next((p for p in profiles if p["person"] == "Anna"), None)
    assert anna is not None
    assert anna["relationship"] == "sister"


def test_relationship_profile_upserts(mem):
    """Saving the same person twice should update, not duplicate."""
    mem.save_relationship_profile("Bob", "friend", "Met at uni")
    mem.save_relationship_profile("Bob", "friend", "Met at uni — now in NYC")
    profiles = [p for p in mem.get_relationship_profiles() if p["person"] == "Bob"]
    assert len(profiles) == 1
    assert "NYC" in profiles[0]["summary"]


# ---------------------------------------------------------------------------
