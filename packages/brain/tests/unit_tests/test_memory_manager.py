"""Unit tests for MemoryManager: the recall/extract loop."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from june_brain.memory import (
    KnowledgeGraph,
    Memory,
    MemoryManager,
    VectorStore,
    vector as vector_module,
)
from june_brain.memory.manager import _parse_json_block

# Reuse the stub embedder from test_vector_store so both test files
# exercise the same deterministic behavior.
from .test_vector_store import _HashEmbedder


@pytest.fixture
def memory_dir(tmp_path):
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        vector_module.reset_singletons()
        yield tmp_path
        vector_module.reset_singletons()


@pytest.fixture
def manager(memory_dir):
    user_id = "test_user"
    Memory(user_id)  # create tables
    return MemoryManager(
        user_id,
        vector=VectorStore(user_id, embedding_function=_HashEmbedder()),
        graph=KnowledgeGraph(user_id),
        sqlite=Memory(user_id),
    )


def test_recall_returns_vector_hits(manager):
    manager.vector.upsert("User loves ramen")
    manager.vector.upsert("User plays tennis on Sundays")
    hits = manager.recall("what food does the user love", k=5)
    assert any("ramen" in h["text"].lower() for h in hits)
    assert all("source" in h for h in hits)


def test_recall_includes_graph_mentions(manager):
    manager.graph.add_node("Ana", kind="person", props={"description": "user's sister"})
    hits = manager.recall("I saw Ana today")
    labels = [h["text"] for h in hits if h["source"] == "graph"]
    assert any("Ana" in text for text in labels)


def test_recall_includes_sqlite_keyword_hits(manager):
    manager.sqlite.save_goal(
        title="Run a marathon", category="health", next_step="train 4x/week"
    )
    hits = manager.recall("how's my marathon training")
    assert any(h["source"] == "sqlite" and "marathon" in h["text"].lower() for h in hits)


def test_recall_dedupes_across_stores(manager):
    # Same exact text lands in two stores — only one should surface.
    manager.vector.upsert("User lives in Lisbon")
    manager.sqlite.save_goal(title="Move to Lisbon", next_step="find apartment")
    hits = manager.recall("where does the user live")
    texts = [h["text"].lower() for h in hits]
    assert len(texts) == len(set(texts))


def test_extract_writes_facts_entities_relations(manager):
    def fake_llm(_prompt: str) -> str:
        return json.dumps(
            {
                "facts": ["User lives in Lisbon", "User is training for a marathon"],
                "entities": [
                    {"name": "Lisbon", "kind": "place"},
                    {"name": "Marco", "kind": "person", "description": "running coach"},
                ],
                "relations": [
                    {"src": "user", "dst": "Lisbon", "kind": "lives_in"},
                    {"src": "user", "dst": "Marco", "kind": "coached_by"},
                ],
            }
        )

    result = manager.extract(
        {"user": "I moved to Lisbon and Marco is coaching my marathon", "assistant": "Nice!"},
        llm_call=fake_llm,
    )
    assert result == {"facts": 2, "entities": 2, "relations": 2}
    # Vector store received the facts
    facts = manager.vector.list_facts()
    assert len(facts) == 2
    # Graph has three nodes: user + Lisbon + Marco
    nodes = manager.graph.find_nodes(limit=10)
    labels = {n["label"] for n in nodes}
    assert {"Lisbon", "Marco"}.issubset(labels)
    # Edges anchored on user node
    user_node = next(n for n in nodes if n["props"].get("is_self"))
    out_edges = manager.graph.neighbors(user_node["node_id"], direction="out")
    kinds = {e["edge"]["kind"] for e in out_edges}
    assert kinds == {"lives_in", "coached_by"}


def test_extract_handles_broken_json(manager):
    def bad_llm(_prompt: str) -> str:
        return "not json at all"

    result = manager.extract({"user": "hi", "assistant": "hello"}, llm_call=bad_llm)
    assert result == {"facts": 0, "entities": 0, "relations": 0}


def test_extract_skips_when_exchange_empty(manager):
    result = manager.extract({"user": "", "assistant": ""}, llm_call=lambda _: "{}")
    assert result == {"facts": 0, "entities": 0, "relations": 0}


def test_two_turn_recall_surfaces_extracted_fact(manager):
    """Week 4 exit criterion: if the user mentions something on turn 1,
    ``recall`` on turn 2 should surface it. This models the "what did I
    tell you about X" test without booting an LLM."""

    def extractor(_prompt: str) -> str:
        return json.dumps(
            {
                "facts": ["User loves ramen"],
                "entities": [{"name": "ramen", "kind": "concept"}],
                "relations": [{"src": "user", "dst": "ramen", "kind": "likes"}],
            }
        )

    # Turn 1 — user mentions ramen; extract runs after the response.
    manager.extract(
        {"user": "I love ramen, it's my favorite food.", "assistant": "Good to know!"},
        llm_call=extractor,
    )

    # Turn 2 — user asks a related question; recall fans out.
    hits = manager.recall("what food do I love", k=5)
    assert any("ramen" in h["text"].lower() for h in hits)


def test_forget_removes_vector_fact(manager):
    record = manager.vector.upsert("ephemeral fact")
    assert manager.forget(f"semantic:{record['fact_id']}") is True
    assert manager.vector.get(record["fact_id"]) is None


def test_forget_removes_graph_node(manager):
    node = manager.graph.add_node("Ana", kind="person")
    assert manager.forget(f"node:{node['node_id']}") is True
    assert manager.graph.get_node(node["node_id"]) is None


def test_forget_excludes_fact_from_future_recall(manager):
    """Week 4 exit criterion: deleting a fact removes it from recall."""
    record = manager.vector.upsert("User loves ramen")
    hits_before = manager.recall("what food do I love", k=5)
    assert any("ramen" in h["text"].lower() for h in hits_before)

    assert manager.forget(f"semantic:{record['fact_id']}") is True

    hits_after = manager.recall("what food do I love", k=5)
    assert not any("ramen" in h["text"].lower() for h in hits_after)


def test_forget_removes_goal_row(manager):
    manager.sqlite.save_goal("learn rust")
    assert manager.forget("goal:learn rust") is True
    assert all(g["title"].lower() != "learn rust" for g in manager.sqlite.get_goals(limit=20))


def test_forget_removes_open_loop_row(manager):
    manager.sqlite.save_open_loop("call mom")
    assert manager.forget("open_loop:call mom") is True
    assert all(
        loop["topic"].lower() != "call mom"
        for loop in manager.sqlite.get_open_loops(status="", limit=20)
    )


def test_forget_removes_calendar_item(manager):
    manager.sqlite.save_calendar_item("dentist", date="2026-06-01", time="10:00")
    assert manager.forget("calendar:dentist|2026-06-01|10:00") is True
    assert all(
        item["title"].lower() != "dentist"
        for item in manager.sqlite.get_calendar_items(limit=20)
    )


def test_forget_calendar_falls_back_to_title_only(manager):
    """Older clients may pass calendar:<title> without date/time."""
    manager.sqlite.save_calendar_item("yoga", date="2026-06-02", time="07:00")
    assert manager.forget("calendar:yoga") is True
    assert all(
        item["title"].lower() != "yoga"
        for item in manager.sqlite.get_calendar_items(limit=20)
    )


def test_forget_unknown_goal_returns_false(manager):
    assert manager.forget("goal:nonexistent") is False


def test_update_goal_in_place(manager):
    manager.sqlite.save_goal("learn rust", category="career", next_step="install rustup")
    updated = manager.update("goal:learn rust", {"next_step": "read the book"})
    assert updated is not None
    assert updated["next_step"] == "read the book"
    assert updated["category"] == "career"  # untouched fields preserved


def test_update_goal_renames_via_pk_change(manager):
    manager.sqlite.save_goal("old name", category="health")
    updated = manager.update("goal:old name", {"title": "new name"})
    assert updated is not None
    assert updated["title"] == "new name"
    titles = [g["title"] for g in manager.sqlite.get_goals(limit=20)]
    assert "new name" in titles
    assert "old name" not in titles


def test_update_calendar_item_reschedule(manager):
    manager.sqlite.save_calendar_item("dentist", date="2026-06-01", time="10:00")
    updated = manager.update(
        "calendar:dentist|2026-06-01|10:00",
        {"date": "2026-06-08", "time": "11:00"},
    )
    assert updated is not None
    assert updated["date"] == "2026-06-08"
    assert updated["time"] == "11:00"
    items = manager.sqlite.get_calendar_items(limit=20)
    assert any(
        i["title"].lower() == "dentist" and i["date"] == "2026-06-08"
        for i in items
    )
    assert not any(
        i["title"].lower() == "dentist" and i["date"] == "2026-06-01"
        for i in items
    )


def test_update_unknown_returns_none(manager):
    assert manager.update("goal:nonexistent", {"next_step": "x"}) is None


def test_update_rejects_non_sqlite_refs(manager):
    """Vector and graph edits go through their own paths; update() ignores them."""
    record = manager.vector.upsert("a fact")
    assert manager.update(f"semantic:{record['fact_id']}", {"text": "new"}) is None


def test_forget_removes_journal_entry(manager):
    manager.sqlite.save_journal("today felt slow")
    entries = manager.sqlite.get_journal(limit=10)
    assert len(entries) == 1
    entry_id = entries[0]["id"]
    assert manager.forget(f"journal:{entry_id}") is True
    assert manager.sqlite.get_journal(limit=10) == []


def test_forget_journal_with_garbage_id_returns_false(manager):
    assert manager.forget("journal:not-a-number") is False


def test_forget_removes_body_metric(manager):
    manager.sqlite.log_body_metrics(weight_kg=80, sleep_hours=7)
    rows = manager.sqlite.get_body_metrics(days=10)
    assert len(rows) == 1
    date = rows[0]["date"]
    assert manager.forget(f"body_metric:{date}") is True
    assert manager.sqlite.get_body_metrics(days=10) == []


def test_parse_json_block_strips_code_fence():
    raw = "```json\n{\"facts\": []}\n```"
    assert _parse_json_block(raw) == {"facts": []}
