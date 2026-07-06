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
