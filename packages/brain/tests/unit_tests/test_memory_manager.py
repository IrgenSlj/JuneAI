"""Unit tests for MemoryManager: the recall/extract loop."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from june_brain.memory import (
    KnowledgeGraph,
    Memory,
    MemoryManager,
    VectorStore,
)
from june_brain.memory import (
    vector as vector_module,
)
from june_brain.memory.extractor import _parse_json_block
from june_brain.providers import reset_cloud_call_recorder, set_cloud_call_recorder
from june_brain.providers.registry import ProviderRegistry

# Reuse the stub embedder from test_vector_store so both test files
# exercise the same deterministic behavior.
from .test_vector_store import _HashEmbedder


def _fts5_available() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


_FTS5_AVAILABLE = _fts5_available()


class _NoEmbedder:
    def embed(self, texts):
        return [None for _text in texts]

    def embed_one(self, text):
        return None


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
        vector=VectorStore(user_id, embedder=_HashEmbedder()),
        graph=KnowledgeGraph(user_id),
        sqlite=Memory(user_id),
    )


class _FakeLocalExtractor:
    model_id = "gemma4:e2b"
    tier = "local-fast"

    def __init__(
        self,
        text: str,
        *,
        reachable: bool = True,
        loaded: bool = True,
        detail: str = "",
    ) -> None:
        self.text = text
        self.reachable = reachable
        self.loaded = loaded
        self.detail = detail
        self.requests = []

    async def health(self):
        return SimpleNamespace(
            reachable=self.reachable,
            loaded=self.loaded,
            detail=self.detail,
        )

    async def generate(self, req):
        self.requests.append(req)
        return SimpleNamespace(text=self.text)


def _registry_with(role: str, provider) -> ProviderRegistry:
    registry = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    registry.register(role, provider)
    return registry


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


@pytest.mark.skipif(not _FTS5_AVAILABLE, reason="fts5 module unavailable in this sqlite build")
def test_recall_uses_semantic_bm25_when_vector_unavailable(memory_dir):
    user_id = "test_user"
    Memory(user_id)
    manager = MemoryManager(
        user_id,
        vector=VectorStore(user_id, embedder=_NoEmbedder()),
        graph=KnowledgeGraph(user_id),
        sqlite=Memory(user_id),
    )

    manager.vector.upsert("User keeps the Delft blue notebook for tax notes")

    hits = manager.recall("Delft notebook", k=5)

    assert any(
        h["source"] == "vector"
        and h.get("retrieval") == "bm25"
        and "delft blue notebook" in h["text"].lower()
        for h in hits
    )


@pytest.mark.skipif(not _FTS5_AVAILABLE, reason="fts5 module unavailable in this sqlite build")
def test_recall_excludes_superseded_semantic_fact(manager):
    from june_brain.memory.sqlite import _get_connection, db_path

    record = manager.vector.upsert("User stores receipts in the Delft blue notebook")
    conn = _get_connection(db_path())
    conn.execute(
        "UPDATE semantic_facts SET valid_to = ? WHERE user_id = ? AND fact_id = ?",
        ("2026-01-01T00:00:00+00:00", manager.user_id, record["fact_id"]),
    )
    conn.commit()

    hits = manager.recall("Delft blue notebook", k=5)

    assert not any(h.get("ref") == record["fact_id"] for h in hits)


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


def test_default_extract_uses_local_provider_when_runtime_is_gemini(manager, monkeypatch):
    provider = _FakeLocalExtractor(
        json.dumps({"facts": ["User prefers tea"], "entities": [], "relations": []})
    )
    registry = _registry_with("local-fast", provider)
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("june_brain.providers.registry.get_registry", lambda: registry)

    cloud_events = []
    set_cloud_call_recorder(cloud_events.append)
    try:
        result = manager.extract(
            {"user": "I prefer tea.", "assistant": "I'll remember that."}
        )
    finally:
        reset_cloud_call_recorder()

    assert result == {"facts": 1, "entities": 0, "relations": 0}
    # Extraction goes through the provider registry only — no cloud fallback.
    assert cloud_events == []
    assert len(provider.requests) == 1
    assert provider.requests[0].response_format == "json"


def test_default_extract_skips_when_local_extractor_unavailable(manager, monkeypatch):
    provider = _FakeLocalExtractor("{}", reachable=False, detail="ollama down")
    registry = _registry_with("local-fast", provider)
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("june_brain.providers.registry.get_registry", lambda: registry)

    result = manager.extract(
        {"user": "Please remember this.", "assistant": "Done."}
    )

    assert {k: result[k] for k in ("facts", "entities", "relations")} == {
        "facts": 0,
        "entities": 0,
        "relations": 0,
    }
    assert "Local memory extractor unavailable" in result["error"]
    assert "ollama down" in result["error"]
    assert provider.requests == []


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


def test_forget_archives_fact_to_trash(manager):
    record = manager.vector.upsert("ephemeral fact")
    ref = f"semantic:{record['fact_id']}"
    assert manager.forget(ref) is True
    trash = manager.list_forgotten()
    assert [t["ref"] for t in trash] == [ref]
    assert trash[0]["kind"] == "fact"
    assert trash[0]["text"] == "ephemeral fact"
    assert trash[0]["forgotten_at"]


def test_restore_brings_back_a_forgotten_fact(manager):
    record = manager.vector.upsert("User loves ramen")
    ref = f"semantic:{record['fact_id']}"
    manager.forget(ref)
    assert manager.vector.get(record["fact_id"]) is None

    restored = manager.restore(ref)
    assert restored is not None
    assert restored["fact_id"] == record["fact_id"]  # exact id preserved
    # Back in the live store and recallable again.
    assert manager.vector.get(record["fact_id"]) is not None
    hits = manager.recall("what food do I love", k=5)
    assert any("ramen" in h["text"].lower() for h in hits)
    # And gone from the trash.
    assert manager.list_forgotten() == []


def test_forget_and_restore_graph_entity(manager):
    node = manager.graph.add_node("Marco", kind="person", props={"role": "friend"})
    ref = f"node:{node['node_id']}"
    assert manager.forget(ref) is True
    assert manager.graph.get_node(node["node_id"]) is None

    trash = manager.list_forgotten()
    assert [t["ref"] for t in trash] == [ref]
    assert trash[0]["kind"] == "person"
    assert trash[0]["text"] == "Marco"

    restored = manager.restore(ref)
    assert restored is not None
    assert restored["node_id"] == node["node_id"]
    back = manager.graph.get_node(node["node_id"])
    assert back is not None and back["label"] == "Marco"
    assert back["props"] == {"role": "friend"}
    assert manager.list_forgotten() == []


def test_list_forgotten_merges_and_orders_by_recency(manager):
    f = manager.vector.upsert("a fact")
    n = manager.graph.add_node("Ana", kind="person")
    manager.forget(f"semantic:{f['fact_id']}")
    manager.forget(f"node:{n['node_id']}")
    refs = [t["ref"] for t in manager.list_forgotten()]
    assert set(refs) == {f"semantic:{f['fact_id']}", f"node:{n['node_id']}"}


def test_restore_preserves_original_created_at(manager):
    """A restored fact keeps its original created_at, not 'now', so it does not
    jump to the top of recency-ordered views."""
    record = manager.vector.upsert("an old fact")
    original = manager.vector.get(record["fact_id"])["created_at"]
    manager.forget(f"semantic:{record['fact_id']}")
    manager.restore(f"semantic:{record['fact_id']}")
    assert manager.vector.get(record["fact_id"])["created_at"] == original


def test_restore_entity_preserves_updated_at(manager):
    node = manager.graph.add_node("Ada", kind="person")
    original = manager.graph.get_node(node["node_id"])["updated_at"]
    manager.forget(f"node:{node['node_id']}")
    manager.restore(f"node:{node['node_id']}")
    assert manager.graph.get_node(node["node_id"])["updated_at"] == original


def test_restore_unknown_ref_returns_none(manager):
    assert manager.restore("semantic:does-not-exist") is None
    assert manager.restore("node:does-not-exist") is None


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


def test_feedback_set_get_clear(manager):
    """Feedback CRUD round-trips through manager → sqlite → recall lookup."""
    manager.sqlite.save_goal("call mom")
    assert manager.sqlite.get_feedback_map() == {}

    manager.set_feedback("goal:call mom", "up")
    assert manager.sqlite.get_feedback_map() == {"goal:call mom": "up"}

    manager.set_feedback("goal:call mom", "down")  # last vote wins
    assert manager.sqlite.get_feedback_map() == {"goal:call mom": "down"}

    assert manager.clear_feedback("goal:call mom") is True
    assert manager.sqlite.get_feedback_map() == {}


def test_feedback_rejects_invalid_vote(manager):
    assert manager.set_feedback("goal:x", "maybe") is None
    assert manager.set_feedback("", "up") is None


def test_write_unknown_kind_returns_not_written(manager):
    result = manager.write({"kind": "spaceship", "fields": {}}, source="test")
    assert result["written"] is False
    assert result["ref"] is None


def test_write_fact_lands_in_vector_only(manager):
    result = manager.write(
        {"kind": "fact", "fields": {"text": "user enjoys ramen"}},
        source="manual",
    )
    assert result["written"] is True
    assert result["ref"].startswith("semantic:")
    assert result["stores"] == ["vector"]
    fact_id = result["ref"].removeprefix("semantic:")
    assert manager.vector.get(fact_id) is not None


def test_write_entity_creates_graph_node(manager):
    result = manager.write(
        {"kind": "entity", "fields": {"label": "Ana", "kind": "person"}},
        source="manual",
    )
    assert result["written"] is True
    assert result["ref"].startswith("node:")
    assert result["stores"] == ["graph"]
    assert manager.graph.get_node(result["node_id"]) is not None


def test_write_goal_writes_sqlite_and_paraphrases_to_vector(manager):
    """A skill writing a goal should also feed the recall ranker."""
    result = manager.write(
        {
            "kind": "goal",
            "fields": {
                "title": "learn Rust",
                "category": "career",
                "next_step": "install rustup",
            },
        },
        source="skill:daily:track_goal",
    )
    assert result["written"] is True
    assert result["ref"] == "goal:learn Rust"
    assert "sqlite" in result["stores"]
    assert "vector" in result["stores"]

    # The structured row exists.
    titles = [g["title"] for g in manager.sqlite.get_goals(limit=20)]
    assert "learn Rust" in titles

    # And recall finds the paraphrase even though the query never mentions
    # the structured field — this is the whole point of C.1.
    hits = manager.recall("rust", k=5)
    assert any(h["source"] == "vector" and "Rust" in h["text"] for h in hits)


def test_forget_goal_removes_vector_paraphrase(manager):
    result = manager.write(
        {
            "kind": "goal",
            "fields": {
                "title": "learn Rust",
                "category": "career",
                "next_step": "install rustup",
            },
        },
        source="manual",
    )
    ref = result["ref"]
    assert manager.vector.fact_ids_for_ref(ref)

    assert manager.forget(ref) is True

    assert manager.vector.fact_ids_for_ref(ref) == []
    facts = manager.vector.list_facts(limit=20)
    assert not any("rust" in fact["text"].lower() for fact in facts)


def test_update_goal_refreshes_vector_paraphrase(manager):
    result = manager.write(
        {
            "kind": "goal",
            "fields": {
                "title": "old name",
                "category": "career",
                "next_step": "install rustup",
            },
        },
        source="manual",
    )
    old_ref = result["ref"]

    updated = manager.update(old_ref, {"title": "new name", "next_step": "read"})

    assert updated is not None
    new_ref = "goal:new name"
    assert manager.vector.fact_ids_for_ref(old_ref) == []
    new_ids = manager.vector.fact_ids_for_ref(new_ref)
    assert new_ids
    facts = manager.vector.list_facts(limit=20)
    assert any(fact["metadata"].get("ref") == new_ref for fact in facts)
    assert not any("old name" in fact["text"].lower() for fact in facts)


def test_write_journal_attaches_id_in_ref(manager):
    result = manager.write(
        {"kind": "journal", "fields": {"entry": "today felt productive"}},
        source="skill:daily:save_journal_entry",
    )
    assert result["written"] is True
    assert result["ref"].startswith("journal:")
    entry_id = int(result["ref"].removeprefix("journal:"))
    assert entry_id > 0
    rows = manager.sqlite.get_journal(limit=5)
    assert any(r["id"] == entry_id for r in rows)


def test_write_calendar_uses_composite_ref(manager):
    result = manager.write(
        {
            "kind": "calendar",
            "fields": {"title": "dentist", "date": "2026-06-01", "time": "10:00"},
        },
        source="skill:calendar:save_calendar_item",
    )
    assert result["written"] is True
    assert result["ref"] == "calendar:dentist|2026-06-01|10:00"
    # And forget(ref) round-trips through the same scheme.
    assert manager.forget(result["ref"]) is True


def test_write_validates_required_fields(manager):
    """Each handler refuses to silently no-op on empty primary fields."""
    assert manager.write({"kind": "goal", "fields": {}}, source="x")["written"] is False
    assert manager.write({"kind": "journal", "fields": {"entry": ""}}, source="x")["written"] is False
    assert manager.write({"kind": "entity", "fields": {"label": ""}}, source="x")["written"] is False


def test_extract_now_uses_write_path(manager):
    """The extract refactor should still write everything; behavior unchanged."""
    def fake_llm(_prompt: str) -> str:
        return (
            '{"facts": ["User loves ramen"],'
            ' "entities": [{"name": "Ana", "kind": "person"}],'
            ' "relations": [{"src": "user", "dst": "Ana", "kind": "knows"}]}'
        )

    result = manager.extract(
        {"user": "I told Ana I love ramen.", "assistant": "Got it."},
        llm_call=fake_llm,
    )
    assert result["facts"] == 1
    assert result["entities"] == 1
    assert result["relations"] == 1
    # Recalling should now surface the extracted fact.
    hits = manager.recall("ramen", k=5)
    assert any("ramen" in h["text"].lower() for h in hits)


def test_feedback_down_demotes_recall(manager):
    """A thumbs-down on a vector hit should sort it after non-voted hits."""
    a = manager.vector.upsert("apples are crunchy", source="test")
    b = manager.vector.upsert("apples are tasty", source="test")

    # No feedback yet — both surface; baseline order depends on hash embed.
    baseline = manager.recall("apples", k=5)
    refs_baseline = [h["ref"] for h in baseline if h["source"] == "vector"]
    assert a["fact_id"] in refs_baseline
    assert b["fact_id"] in refs_baseline

    # Thumbs-down on `a` should push it after `b`.
    manager.set_feedback(f"semantic:{a['fact_id']}", "down")
    after = manager.recall("apples", k=5)
    vec_refs_after = [h["ref"] for h in after if h["source"] == "vector"]
    assert vec_refs_after.index(b["fact_id"]) < vec_refs_after.index(a["fact_id"])

    # The downvoted hit is annotated so the UI can mark it.
    voted = next(h for h in after if h["ref"] == a["fact_id"])
    assert voted.get("feedback") == "down"


def test_parse_json_block_strips_code_fence():
    raw = "```json\n{\"facts\": []}\n```"
    assert _parse_json_block(raw) == {"facts": []}


# ---------------------------------------------------------------------------
# Reversible forget — structured rows
# ---------------------------------------------------------------------------


def test_forget_goal_is_reversible(manager):
    result = manager.write(
        {"kind": "goal", "fields": {"title": "run a marathon", "category": "health", "next_step": "start training"}},
        source="manual",
    )
    ref = result["ref"]
    assert manager.forget(ref) is True
    assert all(g["title"].lower() != "run a marathon" for g in manager.sqlite.get_goals(limit=20))
    trash = manager.list_forgotten()
    trash_refs = [t["ref"] for t in trash]
    assert ref in trash_refs
    assert next(t for t in trash if t["ref"] == ref)["kind"] == "goal"
    assert "run a marathon" in next(t for t in trash if t["ref"] == ref)["text"].lower()

    restored = manager.restore(ref)
    assert restored is not None
    assert any(g["title"].lower() == "run a marathon" for g in manager.sqlite.get_goals(limit=20))
    assert not any(t["ref"] == ref for t in manager.list_forgotten())


def test_forget_open_loop_is_reversible(manager):
    result = manager.write(
        {"kind": "open_loop", "fields": {"topic": "call dentist", "next_step": "find number"}},
        source="manual",
    )
    ref = result["ref"]
    assert manager.forget(ref) is True
    assert all(
        loop["topic"].lower() != "call dentist"
        for loop in manager.sqlite.get_open_loops(status="", limit=20)
    )
    trash = manager.list_forgotten()
    assert any(t["ref"] == ref and t["kind"] == "open_loop" for t in trash)
    assert any("call dentist" in t["text"].lower() for t in trash if t["ref"] == ref)

    restored = manager.restore(ref)
    assert restored is not None
    assert any(
        loop["topic"].lower() == "call dentist"
        for loop in manager.sqlite.get_open_loops(status="", limit=20)
    )
    assert not any(t["ref"] == ref for t in manager.list_forgotten())


def test_forget_calendar_is_reversible(manager):
    result = manager.write(
        {
            "kind": "calendar",
            "fields": {"title": "eye checkup", "date": "2026-07-15", "time": "09:00"},
        },
        source="manual",
    )
    ref = result["ref"]
    assert manager.forget(ref) is True
    assert all(
        item["title"].lower() != "eye checkup"
        for item in manager.sqlite.get_calendar_items(limit=20)
    )
    trash = manager.list_forgotten()
    assert any(t["ref"] == ref and t["kind"] == "calendar" for t in trash)

    restored = manager.restore(ref)
    assert restored is not None
    assert any(
        item["title"].lower() == "eye checkup"
        for item in manager.sqlite.get_calendar_items(limit=20)
    )
    assert not any(t["ref"] == ref for t in manager.list_forgotten())


def test_forget_journal_is_reversible(manager):
    result = manager.write(
        {"kind": "journal", "fields": {"entry": "felt great today, made progress"}},
        source="manual",
    )
    ref = result["ref"]
    assert manager.forget(ref) is True
    assert not any(
        "great today" in e.get("entry", "").lower()
        for e in manager.sqlite.get_journal(limit=20)
    )
    trash = manager.list_forgotten()
    assert any(t["ref"] == ref and t["kind"] == "journal" for t in trash)

    restored = manager.restore(ref)
    assert restored is not None
    assert any(
        "great today" in e.get("entry", "").lower()
        for e in manager.sqlite.get_journal(limit=20)
    )
    assert not any(t["ref"] == ref for t in manager.list_forgotten())


def test_forget_body_metric_is_reversible(manager):
    # Write a metric on a PAST date so restore must land on that same day,
    # not today — guards the original-date-preservation fix.
    result = manager.write(
        {
            "kind": "body_metric",
            "fields": {"date": "2026-01-15", "weight_kg": 75.5, "sleep_hours": 7.5, "energy": 4},
        },
        source="manual",
    )
    ref = result["ref"]
    assert ref == "body_metric:2026-01-15"
    assert manager.forget(ref) is True
    assert manager.sqlite.get_body_metrics(days=400) == []
    trash = manager.list_forgotten()
    assert any(t["ref"] == ref and t["kind"] == "body_metric" for t in trash)

    restored = manager.restore(ref)
    assert restored is not None
    rows = manager.sqlite.get_body_metrics(days=400)
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-01-15"  # original day preserved, not today
    assert rows[0]["weight_kg"] == 75.5
    assert not any(t["ref"] == ref for t in manager.list_forgotten())


def test_forget_goal_excludes_from_recall_until_restored(manager):
    result = manager.write(
        {"kind": "goal", "fields": {"title": "climb Kilimanjaro", "next_step": "book guide"}},
        source="manual",
    )
    ref = result["ref"]
    hits_before = manager.recall("Kilimanjaro", k=5)
    assert any("kilimanjaro" in h["text"].lower() for h in hits_before)

    manager.forget(ref)
    hits_after = manager.recall("Kilimanjaro", k=5)
    assert not any("kilimanjaro" in h["text"].lower() for h in hits_after)

    manager.restore(ref)
    hits_restored = manager.recall("Kilimanjaro", k=5)
    assert any("kilimanjaro" in h["text"].lower() for h in hits_restored)


def test_list_forgotten_merges_all_stores(manager):
    f = manager.vector.upsert("a semantic fact")
    n = manager.graph.add_node("CityX", kind="place")
    goal_result = manager.write(
        {"kind": "goal", "fields": {"title": "learn piano"}}, source="manual"
    )
    manager.forget(f"semantic:{f['fact_id']}")
    manager.forget(f"node:{n['node_id']}")
    manager.forget(goal_result["ref"])

    trash = manager.list_forgotten()
    refs = {t["ref"] for t in trash}
    assert f"semantic:{f['fact_id']}" in refs
    assert f"node:{n['node_id']}" in refs
    assert goal_result["ref"] in refs


def test_purge_forgotten_empties_structured_trash(manager):
    result = manager.write(
        {"kind": "goal", "fields": {"title": "read more books"}}, source="manual"
    )
    manager.forget(result["ref"])
    assert any(t["ref"] == result["ref"] for t in manager.list_forgotten())

    count = manager.purge_forgotten()
    assert count >= 1
    assert manager.list_forgotten() == []
    assert manager.sqlite.list_forgotten_structured() == []


def test_restore_unknown_structured_ref_returns_none(manager):
    assert manager.restore("goal:does-not-exist") is None
    assert manager.restore("open_loop:does-not-exist") is None
    assert manager.restore("journal:99999") is None
