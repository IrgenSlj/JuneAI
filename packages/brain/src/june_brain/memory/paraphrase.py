"""Row -> prompt-text renderers for structured memories.

Each structured memory (goal, open loop, calendar item, journal entry,
body metric, mood) is stored as a SQLite row but also paraphrased into a
short natural-language sentence that is upserted to the vector store so the
structured row also feeds semantic recall. The graph hits get their own
``_format_node`` / ``_format_edge`` renderers.

These are pure functions of a row dict — no store access, no side effects —
extracted from ``manager.py`` so the write paths and recall paths share one
rendering surface (S3 decomposition).
"""

from __future__ import annotations

from typing import Any


def _paraphrase_goal(row: dict[str, Any]) -> str:
    title = str(row.get("title", "")).strip()
    if not title:
        return ""
    next_step = str(row.get("next_step", "")).strip()
    target = str(row.get("target_date", "")).strip()
    parts = [f"Goal: {title}."]
    if next_step:
        parts.append(f"Next step: {next_step}.")
    if target:
        parts.append(f"Target date: {target}.")
    return " ".join(parts)


def _paraphrase_open_loop(row: dict[str, Any]) -> str:
    topic = str(row.get("topic", "")).strip()
    if not topic:
        return ""
    next_step = str(row.get("next_step", "")).strip()
    due = str(row.get("due_date", "")).strip()
    parts = [f"Open loop: {topic}."]
    if next_step:
        parts.append(f"Next step: {next_step}.")
    if due:
        parts.append(f"Due {due}.")
    return " ".join(parts)


def _paraphrase_calendar(row: dict[str, Any]) -> str:
    title = str(row.get("title", "")).strip()
    if not title:
        return ""
    date = str(row.get("date", "")).strip()
    time = str(row.get("time", "")).strip()
    details = str(row.get("details", "")).strip()
    parts = [f"Calendar item: {title}."]
    if date and time:
        parts.append(f"On {date} at {time}.")
    elif date:
        parts.append(f"On {date}.")
    if details:
        parts.append(details if details.endswith(".") else f"{details}.")
    return " ".join(parts)


def _paraphrase_journal(row: dict[str, Any]) -> str:
    entry = str(row.get("entry", "")).strip()
    if not entry:
        return ""
    return f"Journal entry: {entry}"


def _paraphrase_body_metric(row: dict[str, Any]) -> str:
    date = str(row.get("date", "")).strip()
    weight = row.get("weight_kg") or 0
    sleep = row.get("sleep_hours") or 0
    energy = row.get("energy") or 0
    stress = row.get("stress") or 0
    parts = []
    if weight:
        parts.append(f"weight {weight}kg")
    if sleep:
        parts.append(f"slept {sleep}h")
    if energy:
        parts.append(f"energy {energy}/5")
    if stress:
        parts.append(f"stress {stress}/5")
    if not parts:
        return ""
    head = f"Body check on {date}" if date else "Body check"
    return f"{head}: {', '.join(parts)}."


def _paraphrase_mood(row: dict[str, Any]) -> str:
    mood = str(row.get("mood", "")).strip()
    if not mood:
        return ""
    note = str(row.get("note", "")).strip()
    return f"Mood: {mood}. {note}".strip() if note else f"Mood: {mood}."


def _format_node(node: dict[str, Any]) -> str:
    desc = node.get("props", {}).get("description", "")
    label = node.get("label", "")
    kind = node.get("kind", "entity")
    if desc:
        return f"{label} ({kind}) — {desc}"
    return f"{label} ({kind})"


def _format_edge(source_node: dict[str, Any], edge_hit: dict[str, Any]) -> str:
    other = edge_hit.get("node", {})
    edge = edge_hit.get("edge", {})
    kind = str(edge.get("kind", "related_to")).replace("_", " ")
    other_label = other.get("label", "")
    return f"{source_node.get('label', '')} {kind} {other_label}".strip()
