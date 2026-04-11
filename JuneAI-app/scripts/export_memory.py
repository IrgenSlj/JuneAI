#!/usr/bin/env python3
"""Export a user's JuneAI memory to a portable JSON archive.

Usage:
    python scripts/export_memory.py              # exports 'admin' profile
    python scripts/export_memory.py --user alice # exports 'alice' profile
    python scripts/export_memory.py --out /tmp/june_export.json

The output file contains every memory surface as a JSON object, suitable
for backup, migration, or inspection.  No data is deleted from the database.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make sure the src package is importable when run from the repo root
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from agent.memory import Memory  # noqa: E402


def _serialise_messages(msgs: list) -> list[dict]:
    """Convert LangChain message objects to plain dicts."""
    out = []
    for m in msgs:
        role = getattr(m, "type", None) or m.__class__.__name__.lower().replace("message", "")
        content = getattr(m, "content", str(m))
        if isinstance(content, list):
            content = " ".join(
                (c.get("text", "") if isinstance(c, dict) else str(c)) for c in content
            )
        out.append({"role": role, "content": content})
    return out


def _export(user_id: str) -> dict:
    m = Memory(user_id)

    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "messages": _serialise_messages(m.load_chat_messages()),
        "goals": m.get_goals(status="", limit=500),
        "open_loops": m.get_open_loops(status="", limit=500),
        "calendar": m.get_calendar_items(status="", limit=500),
        "habits": m.get_habits(),
        "gym_plans": m.get_gym_plans(status="", limit=100),
        "food_programs": m.get_food_programs(status="", limit=100),
        "workout_sessions": m.get_workout_sessions(limit=500),
        "body_metrics": m.get_body_metrics(days=3650),
        "nutrition_recent": m.get_nutrition_recent(limit=500),
        "mood_history": m.get_mood_history(limit=500),
        "journal": m.get_journal(limit=500),
        "relationship_profiles": m.get_relationship_profiles(),
        "preferences": m.get_preferences(limit=1000),
        "favorites": m.get_favorites(limit=500),
        "progress_snapshot": m.get_progress_snapshot(),
        "chapter_completeness": m.get_chapter_completeness(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export JuneAI memory to JSON.")
    parser.add_argument("--user", default="admin", help="Profile name (default: admin)")
    parser.add_argument(
        "--out",
        default="",
        help="Output file path (default: june_export_<user>_<date>.json)",
    )
    args = parser.parse_args()

    data = _export(args.user)

    out_path = Path(
        args.out
        or f"june_export_{args.user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    total_items = sum(
        len(v) if isinstance(v, list) else 1
        for v in data.values()
        if v and not isinstance(v, str)
    )
    print(f"Exported {total_items} records for '{args.user}' → {out_path}")


if __name__ == "__main__":
    main()
