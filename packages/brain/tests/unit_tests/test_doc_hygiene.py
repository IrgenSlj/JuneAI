"""Tests for the tools/check_doc_hygiene.py stale-token checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[4] / "tools" / "check_doc_hygiene.py"
_spec = importlib.util.spec_from_file_location("check_doc_hygiene", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
check_doc_hygiene = importlib.util.module_from_spec(_spec)
sys.modules["check_doc_hygiene"] = check_doc_hygiene
_spec.loader.exec_module(check_doc_hygiene)

scan_text = check_doc_hygiene.scan_text


def test_banned_token_is_flagged() -> None:
    text = "We used ChromaDB for vectors before switching to sqlite-vec."
    hits = scan_text(text)
    assert len(hits) == 1
    lineno, token, reason = hits[0]
    assert lineno == 1
    assert token == "chromadb"
    assert "sqlite-vec" in reason


def test_clean_text_passes() -> None:
    text = "June uses sqlite-vec for embeddings and a hand-written loop engine."
    assert scan_text(text) == []


# ---------------------------------------------------------------------------
# June's voice: the name, or "it" — never "she"
# ---------------------------------------------------------------------------

scan_voice = check_doc_hygiene.scan_voice


def test_feminine_pronouns_are_flagged() -> None:
    hits = scan_voice("June is alpha software. She runs as a web app.")
    assert len(hits) == 1
    assert hits[0][1] == "She"


def test_pronouns_inside_words_are_not_flagged() -> None:
    """A substring check would fire on 'other', 'where', 'gather', 'here'."""
    text = "Gather the other files here, wherever they are, and thereafter ship them."
    assert scan_voice(text) == []


def test_the_approved_voice_passes() -> None:
    text = (
        "June is alpha software. It runs as a web app, and June remembers what "
        "matters to you. Every action it takes is visible."
    )
    assert scan_voice(text) == []


def test_every_voice_surface_is_clean() -> None:
    """The live assertion: the product surfaces themselves, not a sample."""
    offenders = []
    for path in check_doc_hygiene._voice_paths():
        for lineno, word, _reason in scan_voice(path.read_text(encoding="utf-8")):
            rel = path.relative_to(check_doc_hygiene.REPO_ROOT)
            offenders.append(f"{rel}:{lineno} '{word}'")
    assert offenders == [], "\n".join(offenders)


def test_the_voice_scope_is_not_silently_empty() -> None:
    """A typo'd path would make the check pass by scanning nothing."""
    assert len(check_doc_hygiene._voice_paths()) >= 10
