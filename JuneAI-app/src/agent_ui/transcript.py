"""Transcript rendering helpers for the JuneAI UI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .shell import collapse_text_html, escape_html, text_html

COLLAPSE_VARIANTS = {
    "analysis",
    "debug",
    "json",
    "reasoning",
    "tool",
    "tool_call",
    "tool_result",
}

SUMMARY_VARIANTS = {
    "save_summary",
    "summary",
    "trust_summary",
}


@dataclass(frozen=True)
class TranscriptBlock:
    """Normalized assistant content block."""

    label: str | None
    text: str
    kind: str = "text"
    collapsed: bool = False
    compact: bool = False


def _save_summary_html(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    cards = "".join(
        '<div class="june-item">'
        f'<div class="june-item-meta">{escape_html(line)}</div>'
        '</div>'
        for line in lines
    )
    return '<div class="june-list">' + cards + "</div>"


def _strip_internal_thoughts(text: str) -> str:
    """Remove Gemma thought-channel markup from transcript text."""
    cleaned = re.sub(r"<\|channel>thought\s*.*?<channel\|>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def extract_text(content: Any) -> str:
    """Flatten message content into readable text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return _strip_internal_thoughts(content)
    if isinstance(content, list):
        flattened_parts = [extract_text(item) for item in content]
        return "\n".join(part for part in flattened_parts if part)
    if isinstance(content, dict):
        extracted_parts: list[str] = []
        for key in ("summary", "text", "body", "content", "details"):
            if key in content:
                text = extract_text(content[key])
                if text:
                    extracted_parts.append(text)
        items = content.get("items")
        if isinstance(items, list):
            bullet_lines = [f"• {extract_text(item)}" for item in items if extract_text(item)]
            if bullet_lines:
                extracted_parts.append("\n".join(bullet_lines))
        sections = content.get("sections")
        if isinstance(sections, list):
            extracted_parts.extend(extract_text(item) for item in sections if extract_text(item))
        if extracted_parts:
            return "\n".join(extracted_parts)
        return ""
    return str(content)


def _label_from_variant(variant: str) -> str:
    return variant.replace("_", " ").strip().title() or "Message"


def _normalize_block_label(label: Any, variant: str) -> str | None:
    if label in (None, ""):
        return None if variant in {"text", "message"} else _label_from_variant(variant)
    return str(label)


def _summary_item_text(item: Any) -> str:
    if isinstance(item, dict):
        title = extract_text(item.get("title") or item.get("label") or item.get("name"))
        body = extract_text(item.get("text") or item.get("body") or item.get("content") or item.get("details"))
        if title and body:
            return f"{title}: {body}"
        return title or body
    if isinstance(item, list) and len(item) >= 2:
        title = extract_text(item[0])
        body = extract_text(item[1])
        if title and body:
            return f"{title}: {body}"
        return title or body
    return extract_text(item)


def _blocks_from_mapping(payload: dict[str, Any], *, collapse_threshold: int) -> list[TranscriptBlock]:
    variant = str(payload.get("variant") or payload.get("type") or "text")
    label = payload.get("label") or payload.get("title")
    sections = payload.get("sections")
    if isinstance(sections, list) and sections:
        section_blocks: list[TranscriptBlock] = []
        for section in sections:
            section_blocks.extend(message_blocks(section, collapse_threshold=collapse_threshold))
        return section_blocks

    if variant in SUMMARY_VARIANTS:
        summary = extract_text(payload.get("summary") or payload.get("title"))
        details = extract_text(payload.get("details") or payload.get("note") or payload.get("body"))
        text = extract_text(payload.get("text") or payload.get("content"))
        items = payload.get("items")
        parts: list[str] = []
        if summary:
            parts.append(summary)
        if details and details != summary:
            parts.append(details)
        if isinstance(items, list) and items:
            bullet_lines: list[str] = []
            for item in items:
                item_text = _summary_item_text(item)
                if item_text:
                    bullet_lines.append(f"• {item_text}")
            if bullet_lines:
                parts.extend(bullet_lines)
        if not parts and text:
            parts.append(text)
        if not parts:
            parts.append(extract_text(payload))
        summary_text = "\n".join(part for part in parts if part)
        if summary_text:
            return [
                TranscriptBlock(
                    label=_normalize_block_label(label, variant),
                    text=summary_text,
                    kind=variant,
                    collapsed=len(summary_text) > collapse_threshold * 1.4,
                    compact=True,
                )
            ]
        return []

    summary = extract_text(payload.get("summary"))
    details = extract_text(payload.get("details"))
    text = extract_text(payload.get("text") or payload.get("body") or payload.get("content"))
    items = payload.get("items")

    blocks: list[TranscriptBlock] = []
    if summary and details:
        blocks.append(TranscriptBlock(label="Summary", text=summary, kind="summary"))
        blocks.append(
            TranscriptBlock(
                label=_normalize_block_label(label, variant),
                text=details,
                kind=variant,
                collapsed=True,
            )
        )
        return blocks

    if isinstance(items, list) and items:
        bullet_lines = [f"• {extract_text(item)}" for item in items if extract_text(item)]
        if bullet_lines:
            bullet_text = "\n".join(bullet_lines)
            blocks.append(
                TranscriptBlock(
                    label=_normalize_block_label(label, variant),
                    text=bullet_text,
                    kind=variant,
                    collapsed=len(bullet_text) > collapse_threshold or variant in COLLAPSE_VARIANTS,
                )
            )
            return blocks

    if text:
        blocks.append(
            TranscriptBlock(
                label=_normalize_block_label(label, variant),
                text=text,
                kind=variant,
                collapsed=len(text) > collapse_threshold or variant in COLLAPSE_VARIANTS,
            )
        )
        return blocks

    flattened = extract_text(payload)
    if flattened:
        blocks.append(
            TranscriptBlock(
                label=_normalize_block_label(label, variant),
                text=flattened,
                kind=variant,
                collapsed=len(flattened) > collapse_threshold or variant in COLLAPSE_VARIANTS,
            )
        )
    return blocks


def message_blocks(content: Any, *, collapse_threshold: int = 900) -> list[TranscriptBlock]:
    """Normalize arbitrary assistant content into renderable blocks."""
    if content in ("", None, []):
        return []
    if isinstance(content, list):
        blocks: list[TranscriptBlock] = []
        for item in content:
            blocks.extend(message_blocks(item, collapse_threshold=collapse_threshold))
        return blocks
    if isinstance(content, dict):
        if any(key in content for key in ("variant", "type", "summary", "details", "items", "sections", "text", "body", "content")):
            return _blocks_from_mapping(content, collapse_threshold=collapse_threshold)
        flattened = extract_text(content)
        if flattened:
            return [TranscriptBlock(label=None, text=flattened, collapsed=len(flattened) > collapse_threshold)]
        return []
    text = extract_text(content)
    if not text:
        return []
    return [TranscriptBlock(label=None, text=text, collapsed=len(text) > collapse_threshold)]


def _message_role_and_content(message: Any) -> tuple[str, Any, str]:
    if isinstance(message, HumanMessage):
        return "user", message.content, "You"
    if isinstance(message, AIMessage):
        return "assistant", message.content, "June"
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("speaker") or message.get("author") or "assistant").lower()
        content = message.get("content", message.get("text", message.get("body", "")))
        label = str(message.get("label") or message.get("name") or ("You" if role in {"user", "human"} else "June"))
        return role, content, label
    if isinstance(message, str):
        return "assistant", message, "June"
    return "assistant", extract_text(message), "June"


def _render_block(block: TranscriptBlock, *, collapse_threshold: int) -> str:
    label_html = (
        f'<div class="june-meta-row" style="margin-bottom:0.25rem;">'
        f'<div class="june-chip">{escape_html(block.label)}</div>'
        f'</div>'
        if block.label
        else ""
    )
    if block.compact:
        if block.kind == "save_summary":
            body_html = _save_summary_html(block.text)
        else:
            body_html = (
                collapse_text_html(block.text, threshold=collapse_threshold)
                if block.collapsed or len(block.text) > collapse_threshold
                else f'<div style="white-space:pre-wrap;font-size:13px;line-height:1.55;color:var(--j-text,#1A1815);">{escape_html(block.text)}</div>'
            )
        return (
            '<div class="june-summary-card" style="padding:0.65rem 0.78rem;border:1px solid rgba(15,95,74,0.13);'
            'border-radius:12px;background:rgba(15,95,74,0.035);">'
            + label_html
            + body_html
            + "</div>"
        )
    if block.kind == "save_summary":
        body_html = _save_summary_html(block.text)
    else:
        body_html = (
            collapse_text_html(block.text, threshold=collapse_threshold)
            if block.collapsed or len(block.text) > collapse_threshold
            else text_html(block.text)
        )
    return '<div style="display:grid;gap:0.35rem;">' + label_html + body_html + "</div>"


def render_save_summary_html(
    title: object,
    items: list[tuple[object, object]],
    *,
    note: object = "",
    collapse_threshold: int = 900,
) -> str:
    """Render a compact in-chat summary of what June saved."""
    payload = {
        "variant": "save_summary",
        "label": title,
        "summary": note,
        "items": [
            {"title": item_title, "text": item_copy}
            for item_title, item_copy in items
            if item_title not in ("", None) or item_copy not in ("", None)
        ],
    }
    return render_message_html({"role": "assistant", "content": payload}, collapse_threshold=collapse_threshold)


def render_message_html(message: Any, *, collapse_threshold: int = 900) -> str:
    """Render a single transcript message."""
    role, content, label = _message_role_and_content(message)
    if role in {"user", "human"}:
        text = extract_text(content)
        if not text:
            return ""
        return (
            '<div class="june-message june-message-user">'
            f'{text_html(text)}'
            '</div>'
        )

    blocks = message_blocks(content, collapse_threshold=collapse_threshold)
    if not blocks:
        return ""

    return (
        '<div class="june-message june-message-assistant">'
        '<div style="display:grid;gap:0.7rem;">'
        + "".join(_render_block(block, collapse_threshold=collapse_threshold) for block in blocks)
        + '</div>'
        + '</div>'
    )


def render_transcript_html(
    messages: list[Any],
    live_response: Any = "",
    *,
    extra_messages: list[Any] | None = None,
    collapse_threshold: int = 900,
) -> str:
    """Render a transcript and keep the latest message in view."""
    blocks = [render_message_html(message, collapse_threshold=collapse_threshold) for message in messages]
    if live_response:
        # Append a blinking cursor to signal the response is still streaming
        live_html = render_message_html({"role": "assistant", "content": live_response}, collapse_threshold=collapse_threshold)
        live_html = live_html.replace("</div>\n</div>", '<span class="june-stream-cursor"></span></div>\n</div>', 1)
        blocks.append(live_html)
    for message in extra_messages or []:
        blocks.append(render_message_html(message, collapse_threshold=collapse_threshold))
    blocks = [block for block in blocks if block]
    return (
        '<div class="june-transcript" id="june-transcript">'
        + "".join(blocks)
        + '<div id="june-transcript-end"></div>'
        + "</div>"
        + """<script>
        const scrollJuneTranscript = () => {
            const t = document.getElementById("june-transcript");
            const end = document.getElementById("june-transcript-end");
            if (t) t.scrollTop = t.scrollHeight;
            if (end) end.scrollIntoView({ block: "end", behavior: "auto" });
        };
        setTimeout(scrollJuneTranscript, 0);
        setTimeout(scrollJuneTranscript, 80);
        </script>"""
    )
