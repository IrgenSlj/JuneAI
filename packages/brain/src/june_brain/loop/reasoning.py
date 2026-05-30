"""Streaming reasoning splitter.

Separates model chain-of-thought enclosed in ``<think>...</think>`` or
``<thinking>...</thinking>`` from the final answer text.

Two interfaces are provided:

* :class:`ReasoningSplitter` — feed text deltas incrementally via
  :meth:`feed`; get back ``("reasoning", chunk)`` / ``("answer", chunk)``
  tagged segments; call :meth:`flush` at end-of-stream to drain any
  residual buffer.

* :func:`split_reasoning` — pure helper for the non-streaming path;
  accepts the full model text and returns ``(reasoning, answer)`` as a
  tuple of strings.
"""

from __future__ import annotations

import re

# Maximum tail we hold back waiting to decide if a partial ``<`` is the
# start of a tag.  ``<thinking>`` is 9 chars; we keep a bit of slack.
_MAX_TAIL = 12

# Compiled pattern for a complete think block in the full-text helper.
# Matches ``<think>...</think>`` and ``<thinking>...</thinking>``
# case-insensitively, non-greedy.
_FULL_PATTERN = re.compile(
    r"<think(?:ing)?>(.+?)</think(?:ing)?>",
    re.IGNORECASE | re.DOTALL,
)

# Tag open/close patterns for the streaming splitter state machine.
_OPEN_PATTERN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_CLOSE_PATTERN = re.compile(r"</think(?:ing)?>", re.IGNORECASE)


def split_reasoning(full_text: str) -> tuple[str, str]:
    """Split *full_text* into ``(reasoning, answer)``.

    All ``<think>...</think>`` / ``<thinking>...</thinking>`` blocks are
    extracted into *reasoning* (joined by newlines).  The rest of the text,
    with those blocks removed, is returned as *answer*.

    If no think-tags are present, returns ``("", full_text)``.
    Malformed or unclosed tags are tolerated — they do not raise; the
    unclosed content is left in *answer* unchanged.
    """
    reasoning_parts: list[str] = []

    def _replace(m: re.Match[str]) -> str:
        reasoning_parts.append(m.group(1).strip())
        return ""

    try:
        answer = _FULL_PATTERN.sub(_replace, full_text).strip()
    except Exception:
        return "", full_text

    reasoning = "\n".join(reasoning_parts)
    return reasoning, answer


class ReasoningSplitter:
    """Incremental splitter for a stream of text deltas.

    Usage::

        splitter = ReasoningSplitter()
        for delta in model_stream:
            for kind, text in splitter.feed(delta):
                # kind is "reasoning" or "answer"
                ...
        for kind, text in splitter.flush():
            ...

    Design notes:
    - Outside a think-tag: buffer a small tail (up to ``_MAX_TAIL`` chars)
      that might be the start of a tag.  When we see a complete open-tag,
      flush the safe prefix as ``"answer"``, enter reasoning mode.
    - Inside a think-tag: buffer until we find the matching close-tag.
      Emit reasoning segments as they arrive.
    - The tags themselves are NEVER emitted in either channel.
    - Malformed input (unclosed tags, garbage) must never crash.
    """

    def __init__(self) -> None:
        self._buf: str = ""
        self._in_think: bool = False

    def feed(self, delta: str) -> list[tuple[str, str]]:
        """Process *delta* and return a list of ``(kind, text)`` pairs."""
        if not delta:
            return []
        self._buf += delta
        out: list[tuple[str, str]] = []
        try:
            self._process(out)
        except Exception:
            # Safety net: emit whatever we have as answer and reset.
            if self._buf:
                out.append(("answer", self._buf))
                self._buf = ""
                self._in_think = False
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Drain any residual buffer at end-of-stream."""
        out: list[tuple[str, str]] = []
        if not self._buf:
            return out
        try:
            if self._in_think:
                # Unclosed think-tag at EOF — treat residual as reasoning.
                out.append(("reasoning", self._buf))
            else:
                out.append(("answer", self._buf))
        except Exception:
            out.append(("answer", self._buf))
        self._buf = ""
        self._in_think = False
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process(self, out: list[tuple[str, str]]) -> None:
        """State-machine loop; modifies *out* in place."""
        while self._buf:
            prev_len = len(self._buf)
            if self._in_think:
                self._process_inside_think(out)
            else:
                self._process_outside_think(out)
            # If neither branch consumed any buffer, we are waiting for more
            # data — stop iterating to avoid spinning.
            if len(self._buf) == prev_len:
                break

    def _process_outside_think(self, out: list[tuple[str, str]]) -> None:
        """Find the next open-tag; emit everything before it as answer."""
        m = _OPEN_PATTERN.search(self._buf)
        if m is None:
            # No open-tag found.  Hold back a tail in case it is partial.
            safe_len = max(0, len(self._buf) - _MAX_TAIL)
            if safe_len > 0:
                out.append(("answer", self._buf[:safe_len]))
                self._buf = self._buf[safe_len:]
            # else: buffer is small — wait for more data
            return

        # Found an open-tag.
        before = self._buf[: m.start()]
        if before:
            out.append(("answer", before))
        self._buf = self._buf[m.end():]
        self._in_think = True

    def _process_inside_think(self, out: list[tuple[str, str]]) -> None:
        """Find the close-tag; emit content as reasoning."""
        m = _CLOSE_PATTERN.search(self._buf)
        if m is None:
            # No close-tag yet — hold back a tail that might be a partial tag.
            safe_len = max(0, len(self._buf) - _MAX_TAIL)
            if safe_len > 0:
                out.append(("reasoning", self._buf[:safe_len]))
                self._buf = self._buf[safe_len:]
            return

        # Found the close-tag.
        inside = self._buf[: m.start()]
        if inside:
            out.append(("reasoning", inside))
        self._buf = self._buf[m.end():]
        self._in_think = False
