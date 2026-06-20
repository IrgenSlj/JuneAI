"""Engine selection: returns the hand-written HarnessLoop."""

from __future__ import annotations

from .interface import HarnessLoop


def get_loop(name: str | None = None) -> HarnessLoop:
    """Return the HarnessLoop for *name* (defaults to the hand-written loop).

    Raises ValueError for any explicit name other than 'handwritten'.
    """
    if name is None or name == "handwritten":
        from .handwritten import HandwrittenLoop

        return HandwrittenLoop()

    raise ValueError(
        f"Unknown loop engine {name!r}. Known engines: 'handwritten'."
    )
