"""Integration tests for /memory routes.

Uses a deterministic hash embedder in place of Ollama so the tests never
need a running model. We patch ``june_brain.memory.vector._get_default_embedder``
so ``VectorStore`` instances created inside the route pick up the stub
automatically (ADR 0019: vectors live in sqlite-vec, embeddings via Ollama).
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app
from june_brain.memory import vector as vector_module


class _HashEmbedder:
    """Deterministic embedder — matches the brain test fixture (embed/embed_one)."""

    @staticmethod
    def _vec(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) * 2.0 - 1.0 for i in range(64)]

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        return self._vec(text)


@pytest.fixture
def memory_env(tmp_path):
    """Isolate memory to tmp_path and stub out the embedder."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)), patch.object(
        vector_module, "_get_default_embedder", return_value=_HashEmbedder()
    ):
        vector_module.reset_singletons()
        yield tmp_path
        vector_module.reset_singletons()


@pytest.fixture
def client(memory_env):
    app = create_app()
    return TestClient(app)


def _seed_mixed_search_memory(user_id: str = "alex") -> None:
    from june_brain.memory import KnowledgeGraph, Memory, VectorStore

    mem = Memory(user_id)
    mem.save_goal("cook ramen", category="food", next_step="buy noodles")
    mem.save_goal("learn rust", category="career", next_step="read the book")
    mem.save_open_loop("order ramen bowls", next_step="choose ceramic set")
    mem.save_open_loop("file taxes", next_step="collect receipts")
    mem.save_calendar_item(
        "Ramen dinner",
        "2026-07-01",
        details="Try the tonkotsu place",
    )
    mem.save_calendar_item(
        "Dentist",
        "2026-07-02",
        details="Routine cleaning",
    )
    mem.save_journal("Tried a spicy ramen broth today.")
    mem.save_journal("Finished a chapter of a Rust book.")
    mem.log_body_metrics(sleep_hours=7.5, notes="Ramen after the workout.")
    mem.save_message("user", "Ramen was excellent tonight.")
    mem.save_message("assistant", "I'll remember that.")

    vector = VectorStore(user_id, embedder=_HashEmbedder())
    vector.upsert("Alex prefers shoyu ramen", source="test", metadata={"kind": "food"})
    vector.upsert("Alex is learning rust", source="test", metadata={"kind": "career"})

    graph = KnowledgeGraph(user_id)
    graph.add_node("Noodle Lab", kind="place", props={"description": "Favorite ramen shop"})
    graph.add_node("Ana", kind="person", props={"description": "Sister"})


def test_memory_snapshot_empty_for_new_user(client):
    response = client.get("/memory/new_user")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "new_user"
    assert data["goals"] == []
    assert data["semantic_facts"] == []
    assert data["entities"] == []
    assert data["recent_messages"] == 0


def test_memory_stats_returns_zeroed_buckets_for_new_user(client):
    response = client.get("/memory/new_user/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "new_user"
    assert data["total"] == 0
    assert data["last_write"] == ""
    assert data["recent_messages"] == 0
    assert data["recent_facts"] == []
    kinds = {b["kind"]: b["count"] for b in data["buckets"]}
    assert kinds == {
        "goals": 0,
        "open_loops": 0,
        "calendar": 0,
        "journal": 0,
        "body_metrics": 0,
        "semantic_facts": 0,
        "entities": 0,
    }


def test_memory_stats_counts_writes_across_stores(memory_env, client):
    from june_brain.memory import KnowledgeGraph, Memory, VectorStore

    Memory("alex")
    vector = VectorStore("alex", embedder=_HashEmbedder())
    vector.upsert("I love ramen", source="test")
    vector.upsert("I went hiking on Sunday", source="test")

    graph = KnowledgeGraph("alex")
    graph.add_node("Ana", kind="person")

    response = client.get("/memory/alex/stats")
    assert response.status_code == 200
    data = response.json()
    by_kind = {b["kind"]: b["count"] for b in data["buckets"]}
    assert by_kind["semantic_facts"] == 2
    assert by_kind["entities"] == 1
    assert data["total"] >= 3
    # Two recent facts surface; entities don't go in recent_facts.
    assert len(data["recent_facts"]) == 2
    assert {f["body"] for f in data["recent_facts"]} == {
        "I love ramen",
        "I went hiking on Sunday",
    }


def test_memory_snapshot_includes_semantic_and_entities(memory_env, client):
    from june_brain.memory import KnowledgeGraph, Memory, VectorStore

    Memory("alex")  # create the SQLite schema
    vector = VectorStore("alex", embedder=_HashEmbedder())
    record = vector.upsert("I love ramen", source="test", metadata={"kind": "fact"})

    graph = KnowledgeGraph("alex")
    node = graph.add_node("Ana", kind="person", props={"relation": "sister"})

    response = client.get("/memory/alex")
    assert response.status_code == 200
    data = response.json()

    facts = data["semantic_facts"]
    assert len(facts) == 1
    assert facts[0]["body"] == "I love ramen"
    assert facts[0]["ref"] == f"semantic:{record['fact_id']}"

    entities = data["entities"]
    assert len(entities) == 1
    assert entities[0]["title"] == "Ana"
    assert entities[0]["ref"] == f"node:{node['node_id']}"
    assert entities[0]["kind"] == "entity:person"


def test_memory_snapshot_query_filters_across_stores(memory_env, client):
    _seed_mixed_search_memory()

    response = client.get("/memory/alex", params={"q": "ramen"})
    assert response.status_code == 200
    data = response.json()

    assert [goal["title"] for goal in data["goals"]] == ["cook ramen"]
    assert [loop["title"] for loop in data["open_loops"]] == ["order ramen bowls"]
    assert [item["title"] for item in data["calendar"]] == ["Ramen dinner"]
    assert [entry["body"] for entry in data["journal"]] == [
        "Tried a spicy ramen broth today."
    ]
    assert [metric["metadata"]["notes"] for metric in data["body_metrics"]] == [
        "Ramen after the workout."
    ]
    assert [fact["body"] for fact in data["semantic_facts"]] == [
        "Alex prefers shoyu ramen"
    ]
    assert [entity["title"] for entity in data["entities"]] == ["Noodle Lab"]
    assert data["recent_messages"] == 1


def test_memory_snapshot_without_query_is_unfiltered(memory_env, client):
    _seed_mixed_search_memory()

    response = client.get("/memory/alex")
    assert response.status_code == 200
    data = response.json()

    assert {goal["title"] for goal in data["goals"]} == {"cook ramen", "learn rust"}
    assert {loop["title"] for loop in data["open_loops"]} == {
        "order ramen bowls",
        "file taxes",
    }
    assert {item["title"] for item in data["calendar"]} == {"Ramen dinner", "Dentist"}
    assert {entry["body"] for entry in data["journal"]} == {
        "Tried a spicy ramen broth today.",
        "Finished a chapter of a Rust book.",
    }
    assert {fact["body"] for fact in data["semantic_facts"]} == {
        "Alex prefers shoyu ramen",
        "Alex is learning rust",
    }
    assert {entity["title"] for entity in data["entities"]} == {"Noodle Lab", "Ana"}
    assert data["recent_messages"] == 2


def test_delete_semantic_fact_removes_from_snapshot(memory_env, client):
    from june_brain.memory import Memory, VectorStore

    Memory("alex")
    vector = VectorStore("alex", embedder=_HashEmbedder())
    record = vector.upsert("temporary fact")

    ref = f"semantic:{record['fact_id']}"
    response = client.delete(f"/memory/alex/fact/{ref}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["removed"] is True
    assert payload["ref"] == ref

    snapshot = client.get("/memory/alex").json()
    assert snapshot["semantic_facts"] == []


def test_delete_graph_node_removes_from_snapshot(memory_env, client):
    from june_brain.memory import KnowledgeGraph, Memory

    Memory("alex")
    graph = KnowledgeGraph("alex")
    node = graph.add_node("Ana", kind="person")

    ref = f"node:{node['node_id']}"
    response = client.delete(f"/memory/alex/fact/{ref}")
    assert response.status_code == 200
    assert response.json()["removed"] is True

    snapshot = client.get("/memory/alex").json()
    assert snapshot["entities"] == []


def test_delete_unknown_ref_is_idempotent(client):
    response = client.delete("/memory/alex/fact/semantic:does-not-exist")
    assert response.status_code == 200
    assert response.json()["removed"] is False


def test_forgotten_fact_is_listed_and_restorable(memory_env, client):
    from june_brain.memory import Memory, VectorStore

    Memory("alex")
    vector = VectorStore("alex", embedder=_HashEmbedder())
    record = vector.upsert("temporary fact")
    ref = f"semantic:{record['fact_id']}"

    # Forget it — it leaves the snapshot but lands in the trash.
    client.delete(f"/memory/alex/fact/{ref}")
    assert client.get("/memory/alex").json()["semantic_facts"] == []

    forgotten = client.get("/memory/alex/forgotten").json()
    assert forgotten["count"] == 1
    assert forgotten["memories"][0]["ref"] == ref
    assert forgotten["memories"][0]["kind"] == "fact"
    assert forgotten["memories"][0]["text"] == "temporary fact"
    assert forgotten["memories"][0]["forgotten_at"]

    # Restore it — back in the snapshot, gone from the trash.
    restored = client.post("/memory/alex/forgotten/restore", json={"ref": ref})
    assert restored.status_code == 200
    assert restored.json()["restored"] is True

    snapshot = client.get("/memory/alex").json()
    assert any(f["ref"] == ref for f in snapshot["semantic_facts"])
    assert client.get("/memory/alex/forgotten").json()["count"] == 0


def test_forgotten_entity_is_listed_and_restorable(memory_env, client):
    from june_brain.memory import KnowledgeGraph, Memory

    Memory("alex")
    graph = KnowledgeGraph("alex")
    node = graph.add_node("Marco", kind="person")
    ref = f"node:{node['node_id']}"

    client.delete(f"/memory/alex/fact/{ref}")
    assert client.get("/memory/alex").json()["entities"] == []

    forgotten = client.get("/memory/alex/forgotten").json()
    assert any(m["ref"] == ref and m["text"] == "Marco" for m in forgotten["memories"])

    restored = client.post("/memory/alex/forgotten/restore", json={"ref": ref})
    assert restored.status_code == 200
    snapshot = client.get("/memory/alex").json()
    assert any(e["ref"] == ref for e in snapshot["entities"])


def test_restore_unknown_ref_returns_404(client):
    response = client.post(
        "/memory/alex/forgotten/restore", json={"ref": "semantic:does-not-exist"}
    )
    assert response.status_code == 404


def test_delete_goal_row_removes_from_snapshot(memory_env, client):
    from june_brain.memory import Memory

    mem = Memory("alex")
    mem.save_goal("learn rust", category="career", next_step="install rustup")

    response = client.delete("/memory/alex/fact/goal:learn rust")
    assert response.status_code == 200
    assert response.json()["removed"] is True

    snapshot = client.get("/memory/alex").json()
    assert all(g["title"].lower() != "learn rust" for g in snapshot["goals"])


def test_delete_goal_removes_semantic_paraphrase(memory_env, client):
    created = client.post(
        "/memory/alex/fact",
        json={
            "kind": "goal",
            "fields": {
                "title": "learn rust",
                "category": "career",
                "next_step": "install rustup",
            },
        },
    )
    assert created.status_code == 200
    ref = created.json()["ref"]

    snapshot = client.get("/memory/alex").json()
    assert any(
        fact["metadata"].get("ref") == ref
        for fact in snapshot["semantic_facts"]
    )

    response = client.delete(f"/memory/alex/fact/{ref}")
    assert response.status_code == 200
    assert response.json()["removed"] is True

    snapshot = client.get("/memory/alex").json()
    assert not any(
        "rust" in fact["body"].lower()
        for fact in snapshot["semantic_facts"]
    )


def test_patch_goal_in_place(memory_env, client):
    from june_brain.memory import Memory

    mem = Memory("alex")
    mem.save_goal("learn rust", category="career", next_step="install rustup")

    response = client.patch(
        "/memory/alex/fact/goal:learn rust",
        json={"fields": {"next_step": "read the book"}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["updated"] is True
    assert payload["ref"] == "goal:learn rust"  # PK unchanged
    assert payload["fact"]["body"] == "read the book"


def test_patch_goal_renames_returns_new_ref(memory_env, client):
    from june_brain.memory import Memory

    mem = Memory("alex")
    mem.save_goal("old name")

    response = client.patch(
        "/memory/alex/fact/goal:old name",
        json={"fields": {"title": "new name"}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["updated"] is True
    assert payload["ref"] == "goal:new name"
    titles = [g["title"] for g in client.get("/memory/alex").json()["goals"]]
    assert "new name" in titles
    assert "old name" not in titles


def test_patch_unknown_ref_returns_404(client):
    response = client.patch(
        "/memory/alex/fact/goal:nonexistent",
        json={"fields": {"next_step": "x"}},
    )
    assert response.status_code == 404


def test_patch_rejects_unsupported_ref_kind(client):
    response = client.patch(
        "/memory/alex/fact/semantic:abc",
        json={"fields": {"text": "x"}},
    )
    assert response.status_code == 400


def test_feedback_set_clear_round_trip(memory_env, client):
    from june_brain.memory import Memory

    Memory("alex").save_goal("call mom")

    up = client.post(
        "/memory/alex/feedback",
        json={"ref": "goal:call mom", "vote": "up"},
    )
    assert up.status_code == 200
    assert up.json()["vote"] == "up"

    down = client.post(
        "/memory/alex/feedback",
        json={"ref": "goal:call mom", "vote": "down"},
    )
    assert down.status_code == 200
    assert down.json()["vote"] == "down"

    cleared = client.post(
        "/memory/alex/feedback",
        json={"ref": "goal:call mom", "vote": "clear"},
    )
    assert cleared.status_code == 200
    assert cleared.json()["vote"] == ""


def test_feedback_validates_vote(client):
    bad = client.post(
        "/memory/alex/feedback",
        json={"ref": "goal:x", "vote": "maybe"},
    )
    assert bad.status_code == 400


def test_feedback_requires_ref(client):
    bad = client.post(
        "/memory/alex/feedback",
        json={"ref": "", "vote": "up"},
    )
    assert bad.status_code == 400
