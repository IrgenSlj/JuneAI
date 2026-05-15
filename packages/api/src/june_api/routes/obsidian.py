"""Obsidian export routes."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from fastapi import APIRouter
from june_brain.config import resolve_runtime_config
from june_brain.skills.loader import list_status

from ..schemas import MemoryFact, ObsidianExportResponse, ObsidianFile
from .memory import get_memory

router = APIRouter(tags=["obsidian"])


def _slug(value: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyz0123456789 -_."
    safe = "".join(c for c in value.lower() if c in keep).strip()
    return (safe or "untitled")[:64]


def _frontmatter(**fields: object) -> str:
    lines = ["---"]
    for key, value in fields.items():
        if value is None or value == "":
            continue
        lines.append(f'{key}: "{str(value).replace(chr(34), chr(39))}"')
    lines.append("---")
    return "\n".join(lines)


def _note(path: str, content: str) -> ObsidianFile:
    return ObsidianFile(path=path, content=content.rstrip() + "\n")


def _fact_note(section: str, fact: MemoryFact) -> ObsidianFile:
    title = fact.title or fact.body[:80] or "Untitled"
    meta = fact.metadata or {}
    body = _frontmatter(
        june_type=fact.kind,
        june_ref=fact.ref,
        june_section=section,
        source=meta.get("source", ""),
        created_at=meta.get("created_at", meta.get("timestamp", "")),
        updated_at=meta.get("updated_at", ""),
    )
    body += f"\n# {title}\n\n"
    if fact.body and fact.body != title:
        body += f"{fact.body}\n\n"
    if meta:
        body += "## Fields\n\n"
        for key, value in sorted(meta.items()):
            if value is not None and str(value).strip():
                body += f"- **{key}:** {value}\n"
    if fact.ref:
        body += f"\n`{fact.ref}`\n"
    return _note(f"{section}/{_slug(title)}.md", body)


def _index(title: str, paths: list[str]) -> ObsidianFile:
    links = "\n".join(
        f"- [[{path.removesuffix('.md')}|{path.rsplit('/', 1)[-1].removesuffix('.md')}]]"
        for path in sorted(paths)
        if path.endswith(".md") and not path.endswith("/Index.md")
    )
    return _note(f"{title}/Index.md", f"# {title}\n\n{links or 'No notes yet.'}")


def _memory_canvas(snapshot_counts: dict[str, int], skills: list[dict[str, Any]]) -> str:
    nodes = [
        {"id": "sqlite", "type": "text", "text": f"SQLite structured\n{snapshot_counts['structured']} rows", "x": -520, "y": 0, "width": 240, "height": 120},
        {"id": "semantic", "type": "text", "text": f"Chroma semantic\n{snapshot_counts['semantic']} facts", "x": -160, "y": 0, "width": 240, "height": 120},
        {"id": "graph", "type": "text", "text": f"Knowledge graph\n{snapshot_counts['entities']} entities", "x": 200, "y": 0, "width": 240, "height": 120},
        {"id": "skills", "type": "text", "text": f"MCP skills\n{sum(1 for s in skills if s.get('enabled'))} enabled", "x": -160, "y": 220, "width": 240, "height": 120},
        {"id": "agent", "type": "text", "text": "LangGraph agent\nrecall -> generate -> extract", "x": -160, "y": -220, "width": 260, "height": 120},
    ]
    edges = [
        {"id": "agent-sqlite", "fromNode": "agent", "fromSide": "bottom", "toNode": "sqlite", "toSide": "top"},
        {"id": "agent-semantic", "fromNode": "agent", "fromSide": "bottom", "toNode": "semantic", "toSide": "top"},
        {"id": "agent-graph", "fromNode": "agent", "fromSide": "bottom", "toNode": "graph", "toSide": "top"},
        {"id": "skills-agent", "fromNode": "skills", "fromSide": "top", "toNode": "agent", "toSide": "bottom"},
    ]
    return json.dumps({"nodes": nodes, "edges": edges}, indent=2)


def build_obsidian_export(user_id: str) -> ObsidianExportResponse:
    """Build a vault-shaped Obsidian export from live June state."""
    snapshot = get_memory(user_id)
    skills = list_status()
    try:
        runtime = resolve_runtime_config()
        provider = f"{runtime.label} / {runtime.model} / {runtime.privacy_label}"
    except ValueError:
        provider = "unconfigured"

    groups: dict[str, list[MemoryFact]] = {
        "Memory/Facts": snapshot.semantic_facts,
        "Memory/Entities": snapshot.entities,
        "Memory/Goals": snapshot.goals,
        "Memory/Open Loops": snapshot.open_loops,
        "Memory/Calendar": snapshot.calendar,
        "Memory/Journal": snapshot.journal,
        "Memory/Body Metrics": snapshot.body_metrics,
    }

    files: list[ObsidianFile] = []
    files.append(
        _note(
            "Dashboard.md",
            "# June Memory\n\n"
            f"- **User:** {user_id}\n"
            f"- **Exported:** {date.today().isoformat()}\n"
            f"- **Provider:** {provider}\n"
            f"- **Recent messages:** {snapshot.recent_messages}\n\n"
            "## Maps\n\n"
            "- [[System Architecture.canvas|System Architecture]]\n"
            "- [[Memory/Index|Memory Index]]\n"
            "- [[Skills/Index|Skills Index]]\n",
        )
    )

    memory_index_paths: list[str] = []
    for section, facts in groups.items():
        section_paths: list[str] = []
        for fact in facts:
            note = _fact_note(section, fact)
            files.append(note)
            section_paths.append(note.path)
            memory_index_paths.append(note.path)
        files.append(_index(section, section_paths))
    files.append(_index("Memory", memory_index_paths))

    skill_paths: list[str] = []
    for skill in skills:
        key = str(skill.get("key", "skill"))
        tools = skill.get("tools", []) or []
        content = _frontmatter(
            june_type="skill",
            status=skill.get("status", ""),
            enabled=skill.get("enabled", False),
        )
        content += f"\n# {key}\n\n{skill.get('description', '')}\n\n"
        content += "## Tools\n\n"
        if tools:
            for tool in tools:
                enabled = "enabled" if tool.get("enabled", True) else "disabled"
                content += f"- **{tool.get('name', '')}** ({enabled}) — {tool.get('description', '')}\n"
        else:
            content += "No tools discovered yet.\n"
        if skill.get("error"):
            content += f"\n## Last Error\n\n```text\n{skill['error']}\n```\n"
        note = _note(f"Skills/{_slug(key)}.md", content)
        files.append(note)
        skill_paths.append(note.path)
    files.append(_index("Skills", skill_paths))

    counts = {
        "structured": len(snapshot.goals)
        + len(snapshot.open_loops)
        + len(snapshot.calendar)
        + len(snapshot.journal)
        + len(snapshot.body_metrics),
        "semantic": len(snapshot.semantic_facts),
        "entities": len(snapshot.entities),
    }
    files.append(_note("System Architecture.canvas", _memory_canvas(counts, skills)))
    files.append(
        _note(
            "System/Architecture.md",
            "# System Architecture\n\n"
            "June runs as a layered local-first stack: SvelteKit shells call FastAPI, "
            "FastAPI streams the LangGraph brain, and the brain recalls from SQLite, "
            "ChromaDB, and the knowledge graph before calling the active model provider.\n\n"
            "Open [[System Architecture.canvas]] for the visual map.\n",
        )
    )

    return ObsidianExportResponse(user_id=user_id, files=files, count=len(files))


@router.get("/obsidian/{user_id}", response_model=ObsidianExportResponse)
def export_obsidian(user_id: str) -> ObsidianExportResponse:
    """Return Markdown and Canvas files ready to write into an Obsidian vault."""
    return build_obsidian_export(user_id)
