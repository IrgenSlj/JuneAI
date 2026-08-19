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
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from collections.abc import Generator

from .failure import degrade_quietly

# Approximate compressed download sizes in GB for the Gemma 4 family.
MODEL_SIZE_GB: dict[str, float] = {
    "gemma4:e2b": 7.2,
    "gemma4:e4b": 9.6,
    "gemma4:26b": 18.0,
    "gemma4:31b": 20.0,
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
            return int(resp.status) == 200
    except Exception:
        return False


def warm_model(model: str, base_url: str, keep_alive: int = -1) -> bool:
    """Preload a model into memory and pin it resident.

    Sends an empty-prompt ``/api/generate`` so Ollama loads the weights without
    generating, with ``keep_alive`` so they stay resident — the first real
    request is then warm instead of paying a cold-start load. ``keep_alive=-1``
    keeps the model loaded until Ollama restarts. Returns True on success;
    callers treat warmup as best-effort.
    """
    api_base = _ollama_api_base(base_url)
    payload = json.dumps(
        {"model": model, "prompt": "", "stream": False, "keep_alive": keep_alive}
    ).encode()
    req = urllib.request.Request(
        f"{api_base}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return int(resp.status) == 200
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
        degrade_quietly("Ollama process probe")

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
            degrade_quietly("Ollama process probe")

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

def start_pull(model_name: str) -> subprocess.Popen[bytes] | None:
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


def start_pull_with_progress(model_name: str) -> tuple[subprocess.Popen[str] | None, str]:
    """Launch `ollama pull <model>` and track its progress via a temp file.

    Returns (process, progress_file_path).  A daemon thread reads the
    subprocess stdout (which contains lines like "45%") and writes the
    latest "pct|status" to progress_file so the Streamlit main thread can
    read it on each rerun without blocking.

    Returns (None, "") when the Ollama CLI is not on PATH.
    """
    ollama_bin = _find_ollama_bin()
    if not ollama_bin:
        return None, ""

    # Unique temp file per model so concurrent pulls don't clash
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    progress_file = os.path.join(tempfile.gettempdir(), f"juneai_pull_{safe_name}.txt")
    # Initialise so the reader has something to return immediately
    _write_progress(progress_file, 0, "Starting…")

    proc = subprocess.Popen(
        [ollama_bin, "pull", model_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,        # unbuffered so progress appears quickly
        encoding="utf-8",
        errors="replace",
    )

    def _reader(proc: subprocess.Popen[str], path: str) -> None:
        """Read subprocess output char-by-char; split on \\r and \\n."""
        pct = 0
        status = "Downloading"
        buf = ""
        try:
            assert proc.stdout is not None
            while True:
                char = proc.stdout.read(1)
                if not char:
                    break
                if char in ("\r", "\n"):
                    line = buf.strip()
                    buf = ""
                    if not line:
                        continue
                    # Strip ANSI codes then filter noise lines
                    clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
                    # Skip blank lines, URLs, and Ollama version-update warnings
                    if not clean:
                        continue
                    if re.match(r"https?://", clean):
                        continue
                    _cl = clean.lower()
                    if any(kw in _cl for kw in (
                        "please download", "latest version", "note:", "update available",
                        "out of date", "download ollama",
                    )):
                        continue
                    # Detect fatal version-mismatch error from Ollama CLI
                    if any(kw in _cl for kw in (
                        "requires a newer version",
                        "requires newer version",
                        "version of ollama",
                    )):
                        _write_progress(path, 0, "ERR:needs_update")
                        break
                    m = re.search(r"(\d+)%", clean)
                    if m:
                        pct = int(m.group(1))
                        status = "Downloading"
                    elif "success" in _cl:
                        pct = 100
                        status = "Done"
                    elif "error" in _cl:
                        status = f"ERR:{clean[:55]}"
                    elif clean:
                        # e.g. "pulling manifest", "verifying sha256 digest"
                        status = clean[:60]
                    _write_progress(path, pct, status)
                else:
                    buf += char
        except Exception:
            degrade_quietly("Ollama model listing")
        # Final write on process exit
        if pct == 100 or "done" in status.lower():
            _write_progress(path, 100, "Done")

    t = threading.Thread(target=_reader, args=(proc, progress_file), daemon=True)
    t.start()
    return proc, progress_file


def _write_progress(path: str, pct: int, status: str) -> None:
    try:
        with open(path, "w") as f:
            f.write(f"{pct}|{status}")
    except Exception:
        degrade_quietly("Ollama model listing")


def read_pull_progress(progress_file: str) -> tuple[int, str]:
    """Read the latest pull progress written by start_pull_with_progress.

    Returns (pct: 0-100, status_line: str).
    Returns (0, "Downloading") when the file doesn't exist yet.
    """
    if not progress_file or not os.path.exists(progress_file):
        return 0, "Downloading"
    try:
        raw = open(progress_file).read().strip()
        if "|" in raw:
            pct_str, status = raw.split("|", 1)
            return min(int(pct_str), 100), status
    except Exception:
        degrade_quietly("Ollama shutdown")
    return 0, "Downloading"


def cleanup_progress_file(progress_file: str) -> None:
    """Remove a progress temp file once the download is complete."""
    try:
        if progress_file and os.path.exists(progress_file):
            os.remove(progress_file)
    except Exception:
        degrade_quietly("Ollama shutdown")


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

def pull_model_stream(
    model_name: str,
    base_url: str,
) -> Generator[dict[str, object]]:
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
