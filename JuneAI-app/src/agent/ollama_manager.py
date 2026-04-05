"""Ollama model lifecycle management for JuneAI.

Provides: availability checking, streaming pull with progress, and
a size estimate table so the UI can show the download cost up front.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Generator

# Approximate compressed download sizes in GB for known models.
MODEL_SIZE_GB: dict[str, float] = {
    "llama3.2:3b": 2.0,
    "llama3.2:1b": 1.3,
    "gemma3:4b": 2.5,
    "gemma3:12b": 7.0,
    "gemma3:27b": 17.0,
    "mistral:7b-instruct-v0.3": 4.1,
    "mistral-nemo": 7.1,
    "mistral": 3.8,
    "phi3:mini": 2.2,
}

_DEFAULT_TIMEOUT_S = 8


def _ollama_api_base(openai_compat_base_url: str) -> str:
    """Convert the OpenAI-compatible base URL to the Ollama native API base.

    e.g. "http://localhost:11434/v1" → "http://localhost:11434"
    """
    url = openai_compat_base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def list_local_models(base_url: str) -> list[str]:
    """Return the names of every model already pulled in this Ollama instance."""
    api_base = _ollama_api_base(base_url)
    try:
        with urllib.request.urlopen(f"{api_base}/api/tags", timeout=_DEFAULT_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def is_model_available(model_name: str, base_url: str) -> bool:
    """Return True when the model is already present locally.

    Matching is prefix-tolerant: "llama3.2:3b" matches "llama3.2:3b" stored
    as either "llama3.2:3b" or "llama3.2:3b-instruct-q4_K_M", etc.
    """
    available = list_local_models(base_url)
    target = model_name.lower().strip()
    for m in available:
        m_lower = m.lower().strip()
        if m_lower == target:
            return True
        # "gemma3:4b" matches "gemma3:4b-it-q4_K_M"
        if m_lower.startswith(target.split(":")[0] + ":") and target.split(":")[-1] in m_lower:
            return True
    return False


def model_size_label(model_name: str) -> str:
    """Return a human-readable download size string or an empty string if unknown."""
    gb = MODEL_SIZE_GB.get(model_name.lower().strip())
    if gb is None:
        return ""
    return f"~{gb:.1f} GB download"


def pull_model_stream(model_name: str, base_url: str) -> Generator[dict, None, None]:
    """Stream pull progress for a model.

    Yields dicts matching the Ollama pull API response schema:
      {"status": "pulling manifest"}
      {"status": "pulling ...", "total": int, "completed": int}
      {"status": "success"}
      {"status": "error", "error": str}   ← added on network/parse failure
    """
    api_base = _ollama_api_base(base_url)
    payload = json.dumps({"name": model_name, "stream": True}).encode()
    req = urllib.request.Request(
        f"{api_base}/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        yield {"status": "error", "error": str(exc)}
