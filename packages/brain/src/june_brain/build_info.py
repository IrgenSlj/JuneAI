"""Build identity — which code is running, and which release it belongs to."""

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


@functools.lru_cache(maxsize=1)
def release_version() -> str:
    """The released version this build belongs to, e.g. ``"0.1.0"``.

    Distinct from :func:`build_version`, which is a git SHA. Conflating them
    breaks the update check in a way that only appears against the live
    endpoint: comparing a SHA to a release tag is never equal, so every user is
    told an update exists forever.

    Prefers ``JUNE_RELEASE_VERSION`` (set when packaging), then the installed
    package metadata, then ``"unknown"`` — and "unknown" is honoured as "do not
    claim anything about updates" rather than guessed at.
    """
    env = os.getenv("JUNE_RELEASE_VERSION", "").strip()
    if env:
        return normalize_version(env)
    try:
        from importlib.metadata import version

        return normalize_version(version("june-api"))
    except Exception:  # noqa: BLE001
        return "unknown"


def normalize_version(value: str) -> str:
    """Strip a leading ``v`` so ``v0.1.0`` and ``0.1.0`` compare equal."""
    text = (value or "").strip()
    return text[1:] if text[:1].lower() == "v" else text
