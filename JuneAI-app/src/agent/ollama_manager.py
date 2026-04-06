"""Ollama model lifecycle management for JuneAI.

Provides: availability checking, CLI-based pull (non-blocking), and
a size estimate table so the UI can show the download cost up front.

Design rule: all model pulls go through the Ollama CLI subprocess, NOT the
REST API streaming endpoint. The REST pull endpoint blocks Python's main
thread (urllib socket reads hold the GIL), which prevents Streamlit from
re-rendering. The CLI subprocess is a separate OS process — Streamlit is
free to rerun and poll is_model_available() every few seconds.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from typing import Generator

# Approximate compressed download sizes in GB for known models.
MODEL_SIZE_GB: dict[str, float] = {
    "llama3.2:3b": 2.0,
    "llama3.2:1b": 1.3,
    "gemma4:e4b": 9.6,
    "gemma4:e2b": 7.2,
    "gemma4:26b": 18.0,
    "gemma4:31b": 20.0,
    "gemma3:4b": 3.3,
    "gemma3:12b": 8.1,
    "gemma3:27b": 17.0,
    "gemma3n:e4b": 7.5,
    "gemma3n:e2b": 5.6,
    "mistral:7b-instruct-v0.3": 4.1,
    "mistral-nemo": 7.1,
    "mistral": 3.8,
    "phi3:mini": 2.2,
}

_DEFAULT_TIMEOUT_S = 8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ollama_api_base(openai_compat_base_url: str) -> str:
    """Convert the OpenAI-compatible base URL to the Ollama native API base.

    e.g. "http://localhost:11434/v1" → "http://localhost:11434"
    """
    url = openai_compat_base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def _find_ollama_bin() -> str | None:
    """Return the path to the ollama CLI binary, or None if not on PATH."""
    return shutil.which("ollama") or shutil.which("ollama-darwin")


# ---------------------------------------------------------------------------
# Service health
# ---------------------------------------------------------------------------

def is_ollama_running(base_url: str) -> bool:
    """Return True when the Ollama service is reachable."""
    api_base = _ollama_api_base(base_url)
    try:
        with urllib.request.urlopen(f"{api_base}/api/tags", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model availability
# ---------------------------------------------------------------------------

def list_local_models(base_url: str) -> list[str]:
    """Return the names of every model already pulled in this Ollama instance.

    Tries the REST API first; falls back to the CLI so the function works
    even when the API port is temporarily unavailable.
    """
    # Primary: REST API
    api_base = _ollama_api_base(base_url)
    try:
        with urllib.request.urlopen(f"{api_base}/api/tags", timeout=_DEFAULT_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass

    # Fallback: CLI
    ollama_bin = _find_ollama_bin()
    if ollama_bin:
        try:
            result = subprocess.run(
                [ollama_bin, "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            models: list[str] = []
            for line in result.stdout.splitlines()[1:]:  # skip header row
                parts = line.split()
                if parts:
                    models.append(parts[0])
            return models
        except Exception:
            pass

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


# ---------------------------------------------------------------------------
# Async pull (CLI subprocess — non-blocking)
# ---------------------------------------------------------------------------

def start_pull(model_name: str) -> subprocess.Popen | None:
    """Launch `ollama pull <model>` as a background OS process.

    Returns the Popen handle so callers can check .poll() for completion,
    or None if the Ollama CLI binary is not found on PATH.

    The caller does NOT need to wait on this process. Poll
    is_model_available() every few seconds to detect completion.
    """
    ollama_bin = _find_ollama_bin()
    if not ollama_bin:
        return None
    return subprocess.Popen(
        [ollama_bin, "pull", model_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ollama_cli_available() -> bool:
    """Return True when the ollama CLI binary exists on PATH."""
    return _find_ollama_bin() is not None


# ---------------------------------------------------------------------------
# Size label
# ---------------------------------------------------------------------------

def model_size_label(model_name: str) -> str:
    """Return a human-readable download size string or an empty string if unknown."""
    gb = MODEL_SIZE_GB.get(model_name.lower().strip())
    if gb is None:
        return ""
    return f"~{gb:.1f} GB download"


# ---------------------------------------------------------------------------
# Legacy streaming pull (kept for CLI / non-Streamlit callers only)
# ---------------------------------------------------------------------------

def pull_model_stream(model_name: str, base_url: str) -> Generator[dict, None, None]:
    """Stream pull progress via the Ollama REST API.

    WARNING: this generator blocks the calling thread for the entire
    download duration (urllib socket reads hold the GIL). Do NOT call
    this from a Streamlit script. Use start_pull() + is_model_available()
    polling instead.
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
