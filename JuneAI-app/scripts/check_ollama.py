#!/usr/bin/env python3
"""Check that Ollama is running and the resolved runtime model is available."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.config import resolve_runtime_config


def main() -> int:
    """Verify local model availability for the resolved runtime preset."""
    runtime = resolve_runtime_config()
    base_url = runtime.base_url or "http://localhost:11434/v1"
    ollama_url = base_url.replace("/v1", "")
    model = runtime.model

    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=3) as response:
            data = json.loads(response.read())
        names = [item["name"] for item in data.get("models", [])]
        if any(model in name for name in names):
            print(f"OK: Ollama running, model '{model}' found for preset '{runtime.preset_key}'.")
            return 0
        print(f"WARNING: Ollama running but model '{model}' not found for preset '{runtime.preset_key}'.")
        print(f"Available: {', '.join(names) or 'none'}")
        print(f"Fix: ollama pull {model}")
        return 1
    except Exception as exc:
        print(f"ERROR: Cannot reach Ollama at {ollama_url}: {exc}")
        print("Fix: ollama serve")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
