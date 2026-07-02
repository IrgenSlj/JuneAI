"""Tests for corrupt-database startup recovery.

A non-technical user whose june.db is corrupt should get a self-healed fresh
database (with the corrupt file safely preserved) instead of a crash-loop.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import june_brain.memory as memory_pkg
import june_brain.memory.sqlite as memory_sqlite
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_memory_dir(tmp_path: Path) -> Path:
    """Return an isolated memory directory and patch the module to use it."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    return mem_dir


def _write_valid_db(db_file: Path) -> None:
    """Create a minimal but structurally valid SQLite database."""
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE IF NOT EXISTS canary (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO canary VALUES (1)")
    conn.commit()
    conn.close()


def _corrupt_db(db_file: Path) -> None:
    """Overwrite the first 64 bytes with garbage to destroy the SQLite header."""
    with open(db_file, "r+b") as f:
        f.write(b"\x00" * 64)


def _backup_count(mem_dir: Path) -> list[Path]:
    """Return all *.corrupt-* files under mem_dir."""
    return list(mem_dir.glob("june.db.corrupt-*"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Point the memory package and the thread-local connection pool at tmp_path."""
    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    # Also reset the module-level recovery flag between tests.
    monkeypatch.setattr(memory_sqlite, "_last_recovery_backup", None)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_corrupt_db_creates_backup_and_recovers(isolated):
    """A corrupt june.db is moved aside and a fresh working DB is created."""
    from june_brain.memory import Memory

    # Resolve the canonical memory directory the same way the module does.
    mem_dir = Path(memory_sqlite._current_memory_dir())
    mem_dir.mkdir(parents=True, exist_ok=True)
    db_file = mem_dir / "june.db"

    _write_valid_db(db_file)
    _corrupt_db(db_file)

    # Memory init should succeed despite the corruption.
    mem = Memory("test_user")

    # A backup file must exist.
    backups = _backup_count(mem_dir)
    assert len(backups) == 1, f"Expected 1 backup, found: {backups}"
    assert backups[0].name.startswith("june.db.corrupt-")

    # The fresh db file must exist and be valid.
    assert db_file.exists()

    # Basic operation must work on the fresh DB.
    mem.save_message("user", "hello after recovery")
    history = mem.load_chat()
    assert len(history) == 1
    assert history[0]["content"] == "hello after recovery"

    # The module flag is set.
    assert memory_sqlite._last_recovery_backup is not None
    assert "corrupt-" in memory_sqlite._last_recovery_backup


def test_healthy_db_untouched(isolated):
    """A valid june.db is not touched; no backup is created and data is preserved."""
    from june_brain.memory import Memory

    # First init: write real data.
    m1 = Memory("test_user")
    m1.save_message("user", "original data")

    # Reset the thread-local pool so the second init re-opens the file.
    memory_sqlite._local.conns = {}

    m2 = Memory("test_user")
    history = m2.load_chat()

    mem_dir = Path(memory_sqlite._current_memory_dir())
    backups = _backup_count(mem_dir)
    assert backups == [], f"Healthy DB should have no backups, found: {backups}"
    assert len(history) == 1
    assert history[0]["content"] == "original data"
    assert memory_sqlite._last_recovery_backup is None


def test_absent_db_first_run_no_backup(isolated):
    """A brand-new install (no june.db file) creates a fresh DB with no backup."""
    from june_brain.memory import Memory

    mem_dir = Path(memory_sqlite._current_memory_dir())
    db_file = mem_dir / "june.db"
    assert not db_file.exists(), "Precondition: db must not exist"

    mem = Memory("test_user")

    backups = _backup_count(mem_dir)
    assert backups == [], f"First-run should have no backup, found: {backups}"
    assert db_file.exists()

    # Sanity check: the fresh DB is usable.
    mem.save_message("assistant", "hello world")
    assert len(mem.load_chat()) == 1


def test_wal_shm_alongside_corrupt_db_do_not_block_recovery(isolated):
    """WAL/SHM sidecar files present alongside a corrupt DB do not block startup recovery.

    SQLite itself removes WAL/SHM files when probing a SQLITE_NOTADB file.  The
    important guarantee is that recovery succeeds (corrupt DB backed up, fresh DB
    created and functional) even when sidecars are present.  Sidecars deleted by
    SQLite during probe are replaced by fresh empty WAL/SHM files once the new DB
    is opened in WAL mode — that is correct and expected.
    """
    from june_brain.memory import Memory

    mem_dir = Path(memory_sqlite._current_memory_dir())
    mem_dir.mkdir(parents=True, exist_ok=True)
    db_file = mem_dir / "june.db"
    wal_file = mem_dir / "june.db-wal"
    shm_file = mem_dir / "june.db-shm"

    _write_valid_db(db_file)
    _corrupt_db(db_file)

    # Plant fake sidecar files alongside the corrupt DB.
    wal_file.write_bytes(b"fake wal data")
    shm_file.write_bytes(b"fake shm data")

    # Recovery must succeed despite sidecars being present.
    mem = Memory("test_user")

    # The corrupt DB must be backed up.
    db_backups = list(mem_dir.glob("june.db.corrupt-*"))
    assert len(db_backups) == 1, f"Expected 1 DB backup, found: {db_backups}"

    # The module flag must be set.
    assert memory_sqlite._last_recovery_backup is not None

    # Fresh DB must be fully functional.
    mem.save_message("user", "post-recovery with sidecars")
    assert len(mem.load_chat()) == 1

    # The WAL/SHM at original paths (if they exist now) belong to the fresh DB,
    # not the corrupt one — confirmed by their content being valid SQLite WAL format,
    # not the fake bytes we planted.  We verify this indirectly: the content of the
    # original wal_file (if recreated) must NOT be the fake bytes.
    if wal_file.exists():
        assert wal_file.read_bytes()[:8] != b"fake wal", (
            "WAL file still contains the fake corrupt content; original was not removed"
        )


def test_non_sqlite_file_treated_as_corrupt(isolated):
    """A file of random bytes at the db path triggers recovery, not a crash."""
    from june_brain.memory import Memory

    mem_dir = Path(memory_sqlite._current_memory_dir())
    mem_dir.mkdir(parents=True, exist_ok=True)
    db_file = mem_dir / "june.db"

    # Write non-SQLite content (e.g. stray JSON or binary).
    db_file.write_bytes(b"not a database file at all")

    mem = Memory("test_user")

    backups = _backup_count(mem_dir)
    assert len(backups) == 1
    assert not db_file.with_name(backups[0].name) == db_file

    # Must be usable after recovery.
    mem.save_goal("Run a marathon")
    goals = mem.get_goals()
    assert len(goals) == 1
    assert goals[0]["title"] == "Run a marathon"
