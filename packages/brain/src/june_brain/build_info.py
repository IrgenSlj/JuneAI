"""Build identity — the git SHA of the running code, for the System surface."""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path


@functools.lru_cache(maxsize=1)
def build_version() -> str:
    """Short identifier for the running build.

    Prefers ``JUNE_BUILD_SHA`` (set when packaging the desktop/mobile shells,
    where there is no git checkout). Falls back to the current git short SHA
    for local development, and ``"unknown"`` when neither is available. Cached
    so ``GET /system`` stays cheap.
    """
    env = os.getenv("JUNE_BUILD_SHA", "").strip()
    if env:
        return env
    try:
        root = Path(__file__).resolve().parents[4]
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            return sha
    except Exception:  # noqa: BLE001
        pass
    return "unknown"
