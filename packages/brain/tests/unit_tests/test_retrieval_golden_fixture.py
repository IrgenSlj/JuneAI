"""Integrity of the golden retrieval corpus.

The benchmark itself needs Ollama and takes minutes, so it does not run in the
gate. These checks are the part that must never rot: if a case points at a fact
id that no longer exists, the benchmark silently scores it as a miss and the
whole measurement quietly becomes wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/retrieval_golden"

KNOWN_CATEGORIES = {
    "direct",
    "paraphrase",
    "lexical_rare",
    "entity_linked",
    "temporal_supersession",
    "negation_contrast",
    "multi_hop",
    "distractor_heavy",
}


@pytest.fixture(scope="module")
def facts() -> dict[str, dict]:
    doc = json.loads((FIXTURES / "facts.json").read_text(encoding="utf-8"))
    return {f["id"]: f for f in doc["facts"]}


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    doc = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
    return doc["cases"]


def test_fact_ids_are_unique(facts: dict[str, dict]) -> None:
    raw = json.loads((FIXTURES / "facts.json").read_text(encoding="utf-8"))["facts"]
    assert len(raw) == len(facts), "duplicate fact id in facts.json"


def test_every_fact_has_text(facts: dict[str, dict]) -> None:
    for fact_id, fact in facts.items():
        assert fact.get("text", "").strip(), f"{fact_id} has no text"


def test_case_ids_are_unique(cases: list[dict]) -> None:
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case id in cases.json"


def test_corpus_is_the_promised_size(cases: list[dict]) -> None:
    # The release gate is stated as a 100-case corpus; keep that honest.
    assert len(cases) == 100


def test_every_expected_id_resolves(cases: list[dict], facts: dict[str, dict]) -> None:
    for case in cases:
        assert case["expected"], f"{case['id']} expects nothing"
        for fid in case["expected"]:
            assert fid in facts, f"{case['id']} expects unknown fact {fid}"


def test_every_outranks_id_resolves(cases: list[dict], facts: dict[str, dict]) -> None:
    for case in cases:
        for fid in case.get("outranks", []):
            assert fid in facts, f"{case['id']} outranks unknown fact {fid}"
            assert fid not in case["expected"], (
                f"{case['id']} lists {fid} as both expected and outranked"
            )


def test_categories_are_known(cases: list[dict]) -> None:
    for case in cases:
        assert case["category"] in KNOWN_CATEGORIES, (
            f"{case['id']} has unknown category {case['category']}"
        )


def test_every_category_is_exercised(cases: list[dict]) -> None:
    seen = {c["category"] for c in cases}
    assert seen == KNOWN_CATEGORIES, f"unexercised categories: {KNOWN_CATEGORIES - seen}"


def test_supersession_pairs_are_coherent(facts: dict[str, dict]) -> None:
    """A fact that supersedes another must exist, and the stale one must expire."""
    for fact_id, fact in facts.items():
        successor = fact.get("superseded_by")
        if not successor:
            continue
        assert successor in facts, f"{fact_id} superseded by unknown {successor}"
        assert fact.get("valid_to"), f"{fact_id} is superseded but never expires"


def test_supersession_cases_point_at_current_facts(
    cases: list[dict], facts: dict[str, dict]
) -> None:
    """The expected answer for a supersession case must be the live fact."""
    for case in cases:
        if case["category"] != "temporal_supersession":
            continue
        for fid in case["expected"]:
            assert not facts[fid].get("valid_to"), (
                f"{case['id']} expects {fid}, which is itself expired"
            )
        for fid in case.get("outranks", []):
            assert facts[fid].get("valid_to"), (
                f"{case['id']} outranks {fid}, which is not an expired fact"
            )
