"""June data directory: portable, versioned, self-describing.

All components resolve file paths through layout.py. On startup call
load_or_init() to ensure the directory exists and the manifest is current.
"""

from .layout import (
    character_dir,
    character_file,
    config_dir,
    data_dir,
    ensure_layout,
    ledger_path,
    manifest_path,
    memory_dir,
    skills_dir,
    tasks_dir,
)
from .manifest import (
    SCHEMA_VERSION,
    Manifest,
    init_data_dir,
    load_or_init,
    read_manifest,
    write_manifest,
)

__all__ = [
    # layout
    "data_dir",
    "manifest_path",
    "memory_dir",
    "character_dir",
    "character_file",
    "skills_dir",
    "tasks_dir",
    "ledger_path",
    "config_dir",
    "ensure_layout",
    # manifest
    "Manifest",
    "SCHEMA_VERSION",
    "read_manifest",
    "write_manifest",
    "init_data_dir",
    "load_or_init",
]
