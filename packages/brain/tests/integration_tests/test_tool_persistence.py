"""Integration: June's memory tools persist across store instances.

Each test invokes a native tool directly with an injected ``state`` carrying the
user_id, then reads back through a *fresh* store object against the same db
file. A tool that reported success while writing nothing would pass a unit test
built on a shared in-process object and fail here — which is the point, because
a tool result the model relays to the user is a claim about durable state.

Rewritten for the four tools of ADR 0032; it previously covered the v1 domain
writers those replaced.
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

from june_brain.memory import MemoryManager
from june_brain.memory import vector as vector_module
from june_brain.tasks.models import TaskStatus
from june_brain.tasks.store import TasksStore
from june_brain.tools import JUNE_TOOLS


class _HashEmbedder:
    """Deterministic 64-dim embedder so these tests need no Ollama.

    Declared here rather than imported from unit_tests/test_vector_store.py:
    the two test directories are not one package, and a sys.path hop to share
    ten lines would cost more than it saves.
    """

    @staticmethod
    def _vec(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) * 2.0 - 1.0 for i in range(64)]

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        return self._vec(text)

_TOOLS = {t.name: t for t in JUNE_TOOLS}


def _invoke(name: str, args: dict, user_id: str):
    return _TOOLS[name].invoke({**args, "state": {"user_id": user_id}})


def test_remember_persists_and_is_recallable(tmp_path, monkeypatch):
    user_id = "remember_user"

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        vector_module.reset_singletons()
        monkeypatch.setattr(vector_module, "_default_embedder", _HashEmbedder())

        result = _invoke("remember", {"text": "The user's dog is called Otto."}, user_id)
        hits = MemoryManager(user_id).recall("The user's dog is called Otto.", k=5)
        vector_module.reset_singletons()

    assert "Remembered" in result
    assert any("Otto" in h.get("text", "") for h in hits)


def test_forget_persists_the_removal(tmp_path, monkeypatch):
    user_id = "forget_user"

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        vector_module.reset_singletons()
        monkeypatch.setattr(vector_module, "_default_embedder", _HashEmbedder())

        _invoke("remember", {"text": "The user is allergic to penicillin."}, user_id)
        result = _invoke("forget", {"description": "allergic to penicillin"}, user_id)
        hits = MemoryManager(user_id).recall("allergic to penicillin", k=5)
        vector_module.reset_singletons()

    assert "Forgotten" in result
    assert not any("penicillin" in h.get("text", "") for h in hits)


def test_update_promise_persists_the_status(tmp_path):
    user_id = "promise_user"

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        task = TasksStore(user_id=user_id).create(goal="Renew the passport")
        result = _invoke("update_promise", {"promise": task.id, "status": "completed"}, user_id)
        reread = TasksStore(user_id=user_id).get(task.id)

    assert "completed" in result
    assert reread is not None
    assert reread.status == TaskStatus.COMPLETED


def test_list_promises_reads_what_another_store_wrote(tmp_path):
    user_id = "list_user"

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        TasksStore(user_id=user_id).create(goal="Find a dentist")
        result = _invoke("list_promises", {}, user_id)

    assert "Find a dentist" in result


def test_a_tool_writes_only_to_its_own_user(tmp_path, monkeypatch):
    """The injected state is the whole isolation boundary between users."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        vector_module.reset_singletons()
        monkeypatch.setattr(vector_module, "_default_embedder", _HashEmbedder())

        _invoke("remember", {"text": "Alice's favourite colour is green."}, "alice")
        TasksStore(user_id="alice").create(goal="Alice's promise")

        bob_hits = MemoryManager("bob").recall("favourite colour", k=5)
        bob_promises = _invoke("list_promises", {}, "bob")
        vector_module.reset_singletons()

    assert not any("Alice" in h.get("text", "") for h in bob_hits)
    assert bob_promises == "No open promises."
