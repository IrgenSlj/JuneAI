"""Integration tests for /memory routes.

Uses a deterministic hash embedder in place of sentence-transformers so
the tests never download the 90 MB model. We patch
``june_brain.memory.vector._get_embedding_function`` so ``VectorStore``
instances created inside the route pick up the stub automatically.
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest
from chromadb.api.types import EmbeddingFunction
from fastapi.testclient import TestClient

from june_api.app import create_app
from june_brain.memory import vector as vector_module


class _HashEmbedder(EmbeddingFunction):
    """Deterministic embedder — matches the brain test fixture."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input):
        if isinstance(input, str):
            texts = [input]
            single = True
        else:
            texts = list(input)
            single = False
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [(digest[i % len(digest)] / 255.0) * 2.0 - 1.0 for i in range(64)]
            vectors.append(vec)
        return vectors[0] if single else vectors

    @staticmethod
    def name():
        return "test-hash-embedder"

    @staticmethod
    def build_from_config(_config):
        return _HashEmbedder()

    def get_config(self):
        return {}


@pytest.fixture
def memory_env(tmp_path):
    """Isolate memory to tmp_path and stub out the embedder."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)), patch.object(
        vector_module, "_get_embedding_function", return_value=_HashEmbedder()
    ):
        vector_module.reset_singletons()
        yield tmp_path
        vector_module.reset_singletons()


@pytest.fixture
def client(memory_env):
    app = create_app()
    return TestClient(app)


def test_memory_snapshot_empty_for_new_user(client):
    response = client.get("/memory/new_user")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "new_user"
    assert data["goals"] == []
    assert data["semantic_facts"] == []
    assert data["entities"] == []
    assert data["recent_messages"] == 0


def test_memory_snapshot_includes_semantic_and_entities(memory_env, client):
    from june_brain.memory import KnowledgeGraph, Memory, VectorStore

    Memory("alex")  # create the SQLite schema
    vector = VectorStore("alex", embedding_function=_HashEmbedder())
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


def test_delete_semantic_fact_removes_from_snapshot(memory_env, client):
    from june_brain.memory import Memory, VectorStore

    Memory("alex")
    vector = VectorStore("alex", embedding_function=_HashEmbedder())
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


def test_delete_goal_row_removes_from_snapshot(memory_env, client):
    from june_brain.memory import Memory

    mem = Memory("alex")
    mem.save_goal("learn rust", category="career", next_step="install rustup")

    response = client.delete("/memory/alex/fact/goal:learn rust")
    assert response.status_code == 200
    assert response.json()["removed"] is True

    snapshot = client.get("/memory/alex").json()
    assert all(g["title"].lower() != "learn rust" for g in snapshot["goals"])


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
