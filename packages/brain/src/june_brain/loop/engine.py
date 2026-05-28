"""Engine selection: reads JUNE_LOOP_ENGINE to pick which HarnessLoop to use."""

from __future__ import annotations

import os

from .interface import HarnessLoop


def active_engine_name() -> str:
    """Return the name of the active loop engine from env (default 'handwritten')."""
    return os.environ.get("JUNE_LOOP_ENGINE", "handwritten").strip() or "handwritten"


def get_loop(name: str | None = None) -> HarnessLoop:
    """Return the HarnessLoop for *name* (or the active engine if name is None).

    Raises ValueError for unknown names.
    """
    resolved = name if name is not None else active_engine_name()

    if resolved == "handwritten":
        from .handwritten import HandwrittenLoop

        return HandwrittenLoop()

    if resolved == "langgraph":
        from .langgraph_loop import LangGraphLoop

        return LangGraphLoop()

    raise ValueError(
        f"Unknown loop engine {resolved!r}. Known engines: 'handwritten', 'langgraph'."
    )
