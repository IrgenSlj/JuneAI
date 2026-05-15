"""Tests for Obsidian vault export."""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest
from chromadb.api.types import EmbeddingFunction
from fastapi.testclient import TestClient
from june_api.app import create_app
from june_brain.memory import vector as vector_module


class _HashEmbedder(EmbeddingFunction):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input):
        texts = [input] if isinstance(input, str) else list(input)
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append(
                [(digest[i % len(digest)] / 255.0) * 2.0 - 1.0 for i in range(64)]
            )
        return vectors[0] if isinstance(input, str) else vectors

    @staticmethod
    def name():
        return "test-hash-embedder"

    @staticmethod
    def build_from_config(_config):
        return _HashEmbedder()

    def get_config(self):
        return {}


@pytest.fixture
def client(tmp_path):
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)), patch.object(
        vector_module, "_get_embedding_function", return_value=_HashEmbedder()
    ):
        vector_module.reset_singletons()
        yield TestClient(create_app())
        vector_module.reset_singletons()


def test_obsidian_export_contains_memory_skills_and_canvas(client):
    created = client.post(
        "/memory/alex/fact",
        json={
            "kind": "goal",
            "fields": {"title": "Ship June", "next_step": "cut alpha release"},
        },
    )
    assert created.status_code == 200

    response = client.get("/obsidian/alex")
    assert response.status_code == 200
    payload = response.json()
    paths = {item["path"] for item in payload["files"]}

    assert payload["user_id"] == "alex"
    assert payload["count"] == len(payload["files"])
    assert "Dashboard.md" in paths
    assert "System Architecture.canvas" in paths
    assert "Memory/Goals/ship june.md" in paths
    assert "Skills/Index.md" in paths

    dashboard = next(item for item in payload["files"] if item["path"] == "Dashboard.md")
    assert "[[System Architecture.canvas|System Architecture]]" in dashboard["content"]
