#!/usr/bin/env python3
"""Migrate June v1 user data to the v2 platform-appropriate location.

v1 stored everything under ``JuneAI-app/.june_memory/`` inside the repo.
v2 follows platform conventions and stores the database outside the
repo so updates never touch user data:

- macOS:   ~/Library/Application Support/June/june.db
- Linux:   ~/.local/share/June/june.db  (XDG_DATA_HOME if set)
- Windows: %APPDATA%/June/june.db

Behavior:
1. Copy (not move) the SQLite file + WAL/SHM siblings to the v2 path.
2. Verify the copy by opening it and counting chat_messages rows.
3. Mark the source as archived by renaming it to ``june.db.v1-archived``.

Run this once before wiping ``JuneAI-app/``. Idempotent: if the target
already exists, the script refuses to overwrite unless ``--force`` is
passed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path


def v1_memory_dir(repo_root: Path) -> Path:
    return repo_root / "JuneAI-app" / ".june_memory"


def v2_memory_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "June"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "June"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "June"


def verify_sqlite(db_path: Path) -> int:
    """Open the DB and return a row count — proves the copy is valid SQLite."""
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM chat_messages")
        (count,) = cursor.fetchone()
    return int(count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to the JuneAI repo root (default: parent of tools/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the v2 june.db if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without touching the filesystem.",
    )
    args = parser.parse_args()

    src_dir = v1_memory_dir(Path(args.repo_root))
    src_db = src_dir / "june.db"
    dst_dir = v2_memory_dir()
    dst_db = dst_dir / "june.db"

    if not src_db.exists():
        print(f"No v1 database at {src_db}. Nothing to migrate.")
        return 0

    print(f"v1 database: {src_db}")
    print(f"v2 database: {dst_db}")

    if dst_db.exists() and not args.force:
        print(f"Refusing to overwrite existing {dst_db}. Pass --force to replace it.")
        return 2

    if args.dry_run:
        print("[dry-run] Would create", dst_dir)
        print("[dry-run] Would copy june.db and any -shm/-wal siblings")
        print("[dry-run] Would verify and archive the v1 file")
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    for suffix in ("", "-shm", "-wal"):
        source = src_dir / f"june.db{suffix}"
        if source.exists():
            target = dst_dir / f"june.db{suffix}"
            shutil.copy2(source, target)
            print(f"Copied {source.name} -> {target}")

    count = verify_sqlite(dst_db)
    print(f"Verified: {count} chat messages in {dst_db}")

    archived = src_db.with_suffix(".db.v1-archived")
    src_db.rename(archived)
    print(f"Archived v1 source -> {archived}")

    print("Migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
