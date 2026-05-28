"""Tests for persisted memory path resolution."""

from __future__ import annotations

import json
from pathlib import Path

import june_brain.config
import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import june_brain.memory.sqlite as memory_sqlite
    from june_brain.activity import reset_for_tests
    from june_brain.memory import vector as vector_module

    monkeypatch.setattr(june_brain.config, "MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    vector_module.reset_singletons()
    reset_for_tests()
    yield tmp_path
    reset_for_tests()
    vector_module.reset_singletons()


def test_sqlite_db_path_uses_documented_memory_dir(isolated_data_dir: Path) -> None:
    from june_brain.datadir import memory_dir
    from june_brain.memory import Memory
    from june_brain.memory.sqlite import db_path

    expected = memory_dir() / "june.db"
    assert Path(db_path()) == expected

    memory = Memory("alice")
    memory.save_message("user", "hello")

    assert expected.exists()
    assert not (isolated_data_dir / "june.db").exists()


def test_vector_and_graph_paths_use_documented_memory_dir() -> None:
    from june_brain.datadir import memory_dir
    from june_brain.memory import graph as graph_module
    from june_brain.memory import vector as vector_module

    expected_db = memory_dir() / "june.db"
    assert Path(graph_module._db_path()) == expected_db
    assert Path(vector_module._db_path()) == expected_db
    assert vector_module._chroma_dir() == memory_dir() / "chroma"


def test_task_store_and_activity_log_use_documented_memory_db() -> None:
    from june_brain.activity import ActivityLog
    from june_brain.datadir import memory_dir
    from june_brain.tasks import TasksStore

    expected = memory_dir() / "june.db"
    store = TasksStore(user_id="alice")
    log = ActivityLog()

    assert Path(store._db_path) == expected
    assert Path(log._db_path) == expected


def test_import_and_export_use_documented_memory_db(tmp_path: Path) -> None:
    from june_brain.datadir import memory_dir
    from june_brain.memory.export import export_memory
    from june_brain.memory.import_ import import_memory

    expected = memory_dir() / "june.db"
    export_memory("alice")
    assert expected.exists()

    archive_path = tmp_path / "archive.json"
    archive_path.write_text(
        json.dumps(
            {
                "version": 1,
                "user_id": "source",
                "stores": {
                    "sqlite": {
                        "moods": [
                            {
                                "user_id": "source",
                                "mood": "steady",
                                "note": "",
                                "timestamp": "2026-05-28T00:00:00+00:00",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    counts = import_memory(archive_path, "alice")

    assert counts["moods"] == 1
    assert expected.exists()


def test_legacy_root_memory_store_is_used_when_canonical_store_is_missing(
    isolated_data_dir: Path,
) -> None:
    from june_brain.memory import vector as vector_module
    from june_brain.memory.sqlite import db_path

    legacy_db = isolated_data_dir / "june.db"
    legacy_db.touch()

    assert Path(db_path()) == legacy_db
    assert vector_module._chroma_dir() == isolated_data_dir / "chroma"


def test_documented_memory_dir_wins_when_both_legacy_and_canonical_exist(
    isolated_data_dir: Path,
) -> None:
    from june_brain.datadir import memory_dir
    from june_brain.memory.sqlite import db_path

    (isolated_data_dir / "june.db").touch()
    memory_dir().mkdir(parents=True)
    canonical_db = memory_dir() / "june.db"
    canonical_db.touch()

    assert Path(db_path()) == canonical_db
