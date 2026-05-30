"""Tests for GET /chat/history/{user_id}."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app
from june_brain.memory import Memory


@pytest.fixture
def chat_env(tmp_path):
    """Isolate Memory storage to tmp_path."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        yield tmp_path


@pytest.fixture
def client(chat_env):
    app = create_app()
    return TestClient(app)


def test_chat_history_returns_messages_in_order(client, chat_env):
    with patch("june_brain.memory.MEMORY_DIR", str(chat_env)):
        mem = Memory("testuser")
        mem.save_message("user", "hi")
        mem.save_message("assistant", "hello")

    res = client.get("/chat/history/testuser")
    assert res.status_code == 200
    data = res.json()
    messages = data["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hi"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "hello"


def test_chat_history_empty_for_new_user(client):
    res = client.get("/chat/history/brandnewuser")
    assert res.status_code == 200
    data = res.json()
    assert data["messages"] == []
