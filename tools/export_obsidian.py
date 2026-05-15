"""Export June state to an Obsidian vault as markdown and canvas files.

Usage:
    python tools/export_obsidian.py [--vault PATH] [--user USER] [--api URL]

Defaults:
    --vault  ~/JuneMemory  (creates it if missing)
    --user   local
    --api    http://localhost:8000

The API returns a vault-shaped export that includes:
    Dashboard.md
    Memory/...
    Skills/...
    System Architecture.canvas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def fetch_json(url: str) -> dict | list:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def export_memory(api_url: str, user: str, vault: Path) -> int:
    export = fetch_json(f"{api_url}/obsidian/{user}")
    files = export.get("files", []) if isinstance(export, dict) else []
    for item in files:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe export path from API: {item['path']}")
        target = vault / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item["content"], encoding="utf-8")
    return len(files)


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
    print("Open this folder as an Obsidian vault to browse your memory.")


if __name__ == "__main__":
    main()
