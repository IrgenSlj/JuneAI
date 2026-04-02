"""Shared card and shell helpers for the JuneAI UI."""

from __future__ import annotations

import html
from collections.abc import Sequence


def escape_html(value: object) -> str:
    """Escape a value for safe HTML rendering."""
    return html.escape("" if value is None else str(value))


def text_html(text: object) -> str:
    """Render text while preserving line breaks."""
    return f'<div style="white-space:pre-wrap;">{escape_html(text)}</div>'


def chips_html(items: Sequence[object]) -> str:
    """Render a row of compact chips."""
    chips = [escape_html(item) for item in items if item not in ("", None)]
    if not chips:
        return ""
    return '<div class="june-meta-row">' + "".join(
        f"<div class=\"june-chip\">{chip}</div>" for chip in chips
    ) + "</div>"


def collapse_text_html(text: object, *, threshold: int = 900, summary: str = "Show more") -> str:
    """Render text and collapse the tail when it is long."""
    raw = "" if text is None else str(text)
    if len(raw) <= threshold:
        return text_html(raw)
    preview = raw[: max(160, min(len(raw), threshold // 4))].rstrip()
    return (
        f'{text_html(preview)}'
        f'<details><summary>{escape_html(summary)}</summary>'
        f'{text_html(raw)}</details>'
    )


def card_html(
    title: object,
    body: object = "",
    *,
    eyebrow: object | None = None,
    meta: Sequence[object] = (),
    chips: Sequence[object] = (),
    footer: object | None = None,
    collapsed: bool = False,
    collapse_threshold: int = 900,
) -> str:
    """Render a reusable card using the existing June visual language."""
    parts = ['<div class="june-item">']
    if eyebrow not in ("", None):
        parts.append(f'<div class="june-label">{escape_html(eyebrow)}</div>')
    if title not in ("", None):
        parts.append(f'<div class="june-item-title">{escape_html(title)}</div>')
    body_text = "" if body is None else str(body)
    if body_text:
        parts.append(
            collapse_text_html(body_text, threshold=collapse_threshold)
            if collapsed
            else text_html(body_text)
        )
    for line in meta:
        if line not in ("", None):
            parts.append(f'<div class="june-item-meta">{escape_html(line)}</div>')
    chip_row = chips_html(chips)
    if chip_row:
        parts.append(chip_row)
    if footer not in ("", None):
        parts.append(f'<div class="june-item-meta">{escape_html(footer)}</div>')
    parts.append("</div>")
    return "".join(parts)


def empty_state_html(
    title: object,
    body: object,
    *,
    tips: Sequence[object] = (),
    actions: Sequence[object] = (),
    note: object | None = None,
    eyebrow: object | None = None,
) -> str:
    """Render a richer onboarding/empty-state card."""
    parts = ['<div class="june-item">']
    if eyebrow not in ("", None):
        parts.append(f'<div class="june-label">{escape_html(eyebrow)}</div>')
    parts.append(f'<div class="june-item-title">{escape_html(title)}</div>')
    parts.append(text_html(body))
    if tips:
        parts.append('<div class="june-item-meta" style="margin-top:0.5rem;">')
        parts.append("<ul>")
        parts.extend(f"<li>{escape_html(tip)}</li>" for tip in tips if tip not in ("", None))
        parts.append("</ul></div>")
    chip_row = chips_html(actions)
    if chip_row:
        parts.append(chip_row)
    if note not in ("", None):
        parts.append(f'<div class="june-item-meta" style="margin-top:0.5rem;">{escape_html(note)}</div>')
    parts.append("</div>")
    return "".join(parts)


def onboarding_card_html(
    title: object,
    body: object,
    *,
    tips: Sequence[object] = (),
    actions: Sequence[object] = (),
    note: object | None = None,
    eyebrow: object | None = None,
) -> str:
    """Alias for empty states when the card is explicitly onboarding-oriented."""
    return empty_state_html(
        title,
        body,
        tips=tips,
        actions=actions,
        note=note,
        eyebrow=eyebrow,
    )


def list_html(items: Sequence[tuple[object, object]], *, empty_text: str = "Nothing saved yet.") -> str:
    """Render a list of title/meta cards."""
    if not items:
        return f'<div class="june-item-meta">{escape_html(empty_text)}</div>'
    return '<div class="june-list">' + "".join(
        card_html(title, meta, meta=())
        for title, meta in items
    ) + "</div>"


def stat_card_html(label: object, value: object, *, meta: object | None = None) -> str:
    """Render a compact statistic card."""
    parts = ['<div class="june-stat-card">']
    parts.append(f'<div class="june-stat-label">{escape_html(label)}</div>')
    parts.append(f'<div class="june-stat-value">{escape_html(value)}</div>')
    if meta not in ("", None):
        parts.append(f'<div class="june-item-meta">{escape_html(meta)}</div>')
    parts.append("</div>")
    return "".join(parts)


def stat_grid_html(cards: Sequence[str]) -> str:
    """Render a grid of stat cards."""
    return '<div class="june-stat-grid">' + "".join(cards) + "</div>"
