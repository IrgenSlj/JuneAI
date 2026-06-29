import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    """Point every brain test at a throwaway data dir.

    The shared june.db is now written from more seams than before — notably the
    Trust Ledger egress/action appends (ADR 0022) fire from the provider
    provenance path and the guard dispatch chokepoint. Any test exercising those
    would otherwise write to the user's real data dir. This autouse fixture makes
    the whole brain suite hermetic; per-file fixtures that patch MEMORY_DIR to
    their own tmp_path still override it harmlessly.

    Singletons that cache the resolved db_path (the activity log; the ledger
    handles) are reset around each test so they cannot pin to a stale path.
    """
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())

    def _reset_singletons() -> None:
        for module, fn in (
            ("june_brain.activity", "reset_for_tests"),
            ("june_brain.trust", "reset_for_tests"),
            ("june_brain.silence", "reset_for_tests"),
        ):
            try:
                mod = __import__(module, fromlist=[fn])
                getattr(mod, fn)()
            except Exception:  # noqa: BLE001 - reset is best-effort hygiene
                pass

    _reset_singletons()
    yield
    _reset_singletons()
