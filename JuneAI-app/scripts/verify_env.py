#!/usr/bin/env python3
"""Verify the JuneAI development environment."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _require_imports() -> None:
    """Import the packages JuneAI depends on at runtime."""
    import langchain  # noqa: F401
    import langchain_anthropic  # noqa: F401
    import langchain_openai  # noqa: F401
    import langgraph  # noqa: F401
    import streamlit  # noqa: F401


def main() -> int:
    """Verify the local environment and print a short summary."""
    _require_imports()

    from agent.config import resolve_runtime_config
    from agent.memory import Memory
    from agent.tools import DEFAULT_UI_STATE

    runtime = resolve_runtime_config()
    print(f"Interpreter: {sys.executable}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"streamlit: {importlib.metadata.version('streamlit')}")
    print(f"langgraph: {importlib.metadata.version('langgraph')}")
    print(f"langchain: {importlib.metadata.version('langchain')}")
    print(f"langchain-openai: {importlib.metadata.version('langchain-openai')}")
    print(f"langchain-anthropic: {importlib.metadata.version('langchain-anthropic')}")
    print(f"Runtime preset: {runtime.label} ({runtime.provider})")
    print(f"Default layout: {DEFAULT_UI_STATE['layout']}")

    with TemporaryDirectory() as temp_dir:
        with patch("agent.memory.MEMORY_DIR", temp_dir):
            memory = Memory("bootstrap_check")
            memory.save_message("user", "hello")
            history = memory.load_chat()
            if not history or history[0]["content"] != "hello":
                raise SystemExit("Memory round-trip failed.")

    print("Memory round-trip: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
