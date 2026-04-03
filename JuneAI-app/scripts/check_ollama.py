#!/usr/bin/env python3
"""Check that Ollama is running and the configured model is available."""
import os, sys, urllib.request, json
from pathlib import Path

base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
ollama_url = base_url.replace("/v1", "")
model = os.getenv("MODEL_NAME", "mistral")

try:
    with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=3) as r:
        data = json.loads(r.read())
    names = [m["name"] for m in data.get("models", [])]
    if any(model in n for n in names):
        print(f"OK: Ollama running, model '{model}' found.")
        sys.exit(0)
    else:
        print(f"WARNING: Ollama running but model '{model}' not found.")
        print(f"Available: {', '.join(names) or 'none'}")
        print(f"Fix: ollama pull {model}")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: Cannot reach Ollama at {ollama_url}: {e}")
    print("Fix: ollama serve")
    sys.exit(2)
