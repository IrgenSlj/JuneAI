"""License file loading and the cached process-wide entitlement singleton.

Uses the same lock-guarded singleton pattern as trust/ledger.py and
silence/store.py so fixtures that swap MEMORY_DIR get a fresh entitlement.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .keys import PUBLIC_KEYS
from .models import FREE, Entitlement
from .verify import entitlement_from_raw


def license_path() -> Path:
    """Resolve the license file path.

    Priority:
    1. ``JUNE_LICENSE_PATH`` env var (allows desktop shell and tests to
       point at a specific file).
    2. ``datadir.layout.license_path()`` (``<datadir>/config/license.json``).
    """
    override = os.environ.get("JUNE_LICENSE_PATH", "").strip()
    if override:
        return Path(override)
    from june_brain.datadir import layout as _layout  # noqa: PLC0415

    return _layout.license_path()


def _load_raw() -> Any:
    """Read and JSON-parse the license file; return None on any error."""
    try:
        path = license_path()
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - unreadable / malformed file -> None
        return None


# ---------------------------------------------------------------------------
# Process-wide cached singleton, keyed on the resolved license path.
# ---------------------------------------------------------------------------

_entitlement: Entitlement | None = None
_entitlement_path: str | None = None
_lock = threading.Lock()


def get_entitlement() -> Entitlement:
    """Return the cached Entitlement for the current license file.

    Re-loads when the resolved path changes (e.g. a test swapped MEMORY_DIR).
    Returns FREE on any failure — never raises.
    """
    global _entitlement, _entitlement_path

    try:
        current_path = str(license_path())
    except Exception:  # noqa: BLE001
        return FREE

    if _entitlement is None or _entitlement_path != current_path:
        with _lock:
            if _entitlement is None or _entitlement_path != current_path:
                raw = _load_raw()
                if raw is None:
                    _entitlement = FREE
                else:
                    _entitlement = entitlement_from_raw(raw, public_keys=PUBLIC_KEYS)
                _entitlement_path = current_path

    return _entitlement


def reset_for_tests() -> None:
    """Drop the cached entitlement so fixtures that swap paths get a fresh load."""
    global _entitlement, _entitlement_path
    with _lock:
        _entitlement = None
        _entitlement_path = None
