"""Export June memory to an Obsidian vault as markdown notes.

Usage:
    python tools/export_obsidian.py [--vault PATH] [--user USER] [--api URL]

Defaults:
    --vault  ~/JuneMemory  (creates it if missing)
    --user   local
    --api    http://localhost:8000

Each memory type becomes a folder in the vault:
    Goals/          → individual goal notes
    Calendar/       → calendar event notes
    Journal/        → daily journal notes  
    Facts/          → semantic fact atomic notes
    Entities/       → entity notes with [[wikilinks]]
    Dashboard.md    → overview with links to everything
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen


def fetch_json(url: str) -> dict | list:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def sanitise(name: str) -> str:
    """Turn a string into a safe filename fragment."""
    keep = "abcdefghijklmnopqrstuvwxyz0123456789 -_."
    return "".join(c for c in name.lower() if c in keep).strip() or "untitled"


def frontmatter(**kw: str) -> str:
    lines = ["---"]
    for k, v in kw.items():
        if v:
            lines.append(f'{k}: "{v}"')
    lines.append("---")
    return "\n".join(lines)


def export_memory(api_url: str, user: str, vault: Path) -> int:
    memory = fetch_json(f"{api_url}/memory/{user}")

    goals = memory.get("goals", []) or []
    calendar = memory.get("calendar", []) or []
    journal = memory.get("journal", []) or []
    facts = memory.get("semantic_facts", []) or []
    entities = memory.get("entities", []) or []

    dirs = {
        "Goals": vault / "Goals",
        "Calendar": vault / "Calendar",
        "Journal": vault / "Journal",
        "Facts": vault / "Facts",
        "Entities": vault / "Entities",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    count = 0

    # --- Goals ---
    for g in goals:
        title = g.get("title", "Untitled")
        slug = sanitise(title)[:48]
        meta = g.get("metadata", {}) or {}
        md = frontmatter(
            type="goal",
            status=meta.get("status", "active"),
            category=meta.get("category", ""),
            target_date=meta.get("target_date", ""),
        )
        md += f"\n# {title}\n\n"
        if g.get("body"):
            md += f"{g['body']}\n\n"
        if meta.get("next_step"):
            md += f"**Next step:** {meta['next_step']}\n"
        if meta.get("target_date"):
            md += f"**Target:** {meta['target_date']}\n"
        (dirs["Goals"] / f"{slug}.md").write_text(md)
        count += 1

    # --- Calendar ---
    for c in calendar:
        title = c.get("title", "Untitled")
        slug = sanitise(title)[:48]
        meta = c.get("metadata", {}) or {}
        dt = meta.get("date", "") or ""
        tm = meta.get("time", "") or ""
        md = frontmatter(
            type="calendar",
            date=dt,
            time=tm,
            status=meta.get("status", "planned"),
        )
        md += f"\n# {title}\n\n"
        if dt:
            md += f"📅 {dt}"
            if tm:
                md += f" at {tm}"
            md += "\n\n"
        if c.get("body"):
            md += f"{c['body']}\n"
        if meta.get("details"):
            md += f"\n{meta['details']}\n"
        (dirs["Calendar"] / f"{slug}.md").write_text(md)
        count += 1

    # --- Journal ---
    for j in journal:
        ts = (j.get("metadata", {}) or {}).get("timestamp", "")
        day = ts[:10] if ts else str(date.today())
        body = j.get("body", "") or j.get("title", "") or ""
        md = frontmatter(type="journal", date=day)
        md += f"\n# {day}\n\n{body}\n"
        (dirs["Journal"] / f"{day}.md").write_text(md)
        count += 1

    # --- Facts ---
    for f in facts:
        title = f.get("title", "Untitled")
        slug = sanitise(title)[:48]
        meta = f.get("metadata", {}) or {}
        md = frontmatter(type="fact", source=meta.get("source", ""), kind=meta.get("kind", "fact"))
        md += f"\n# {title}\n\n"
        if f.get("body") and f["body"] != title:
            md += f"{f['body']}\n\n"
        ref = f.get("ref", "")
        if ref:
            md += f"`ref: {ref}`\n"
        (dirs["Facts"] / f"{slug}.md").write_text(md)
        count += 1

    # --- Entities ---
    for e in entities:
        title = e.get("title", "Untitled")
        slug = sanitise(title)[:48]
        meta = e.get("metadata", {}) or {}
        kind = meta.get("node_id", "").split(":")[1] if ":" in meta.get("node_id", "") else "entity"
        md = frontmatter(type=kind, node_id=meta.get("node_id", ""))
        md += f"\n# {title}\n\n"
        if e.get("body"):
            md += f"{e['body']}\n\n"
        if meta.get("node_id"):
            backlinks = _find_backlinks(title, entities)
            if backlinks:
                md += "### Links to this note\n"
                for b in backlinks:
                    bs = sanitise(b)[:48]
                    md += f"- [[{bs}]]\n"
        (dirs["Entities"] / f"{slug}.md").write_text(md)
        count += 1

    # --- Dashboard ---
    dash = "# June Memory\n\n"
    dash += f"**User:** {user}  \n"
    dash += f"**Exported:** {date.today()}\n\n"
    dash += "## Overview\n\n"
    dash += f"- **Goals:** {len(goals)} — [[Goals/Index|View all]]\n"
    dash += f"- **Calendar:** {len(calendar)} events — [[Calendar/Index|View all]]\n"
    dash += f"- **Journal:** {len(journal)} entries — [[Journal/Index|View all]]\n"
    dash += f"- **Facts:** {len(facts)} — [[Facts/Index|View all]]\n"
    dash += f"- **Entities:** {len(entities)} — [[Entities/Index|View all]]\n"
    (vault / "Dashboard.md").write_text(dash)

    # --- Index notes ---
    for section, path in dirs.items():
        files = sorted(path.glob("*.md"))
        index = f"# {section}\n\n"
        for f in files:
            if f.stem != "Index":
                index += f"- [[{section}/{f.stem}|{f.stem}]]\n"
        (path / "Index.md").write_text(index)

    return count


def _find_backlinks(title: str, all_entities: list) -> list[str]:
    """Find entity names whose body/text mentions *title*."""
    results: list[str] = []
    low = title.lower()
    for e in all_entities:
        candidate = (e.get("title", "") or "")
        if candidate.lower() == low:
            continue
        body = (e.get("body", "") or "").lower()
        if low in body:
            results.append(candidate)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export June memory to an Obsidian vault")
    parser.add_argument("--vault", default=str(Path.home() / "JuneMemory"), help="Obsidian vault path")
    parser.add_argument("--user", default="local", help="User profile to export")
    parser.add_argument("--api", default="http://localhost:8000", help="June API base URL")
    args = parser.parse_args()

    vault = Path(args.vault)
    vault.mkdir(parents=True, exist_ok=True)

    print(f"Exporting memory for user '{args.user}' from {args.api} ...")
    count = export_memory(args.api.rstrip("/"), args.user, vault)
    print(f"Done — {count} notes written to {vault}")
    print(f"Open this folder as an Obsidian vault to browse your memory.")


if __name__ == "__main__":
    main()
