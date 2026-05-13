"""Settings endpoints.

These are the ongoing counterpart to /setup. Same persistence layer (via
`june_brain.config_store`), same verification semantics (reuse POST /setup/apply
for writes), but a distinct read shape that tells the UI where the Gemini key
lives without ever returning the key itself.
"""

from __future__ import annotations

import os

from fastapi import APIRouter
from june_brain import graph as brain_graph
from june_brain.config import resolve_runtime_config
from june_brain.config_store import (
    forget_gemini_key,
    gemini_key_location,
    load_stored_config,
)
from june_brain.ollama_manager import is_model_available, is_ollama_running

from ..schemas import ForgetKeyResponse, SettingsView

router = APIRouter(prefix="/settings", tags=["settings"])

_STORAGE_LABELS = {
    "keyring": "System credential store",
    "file": "Local file (~/.june/config.json, mode 0600)",
    "none": "No key stored",
}


@router.get("", response_model=SettingsView)
def get_settings() -> SettingsView:
    """Return the current non-secret configuration plus reachability flags."""
    stored = load_stored_config()
    try:
        runtime = resolve_runtime_config()
    except ValueError:
        # Gemini with no key lands here — surface what we have so the UI can
        # still render the provider picker and the "missing key" state.
        location = gemini_key_location()
        return SettingsView(
            provider=stored.provider or "",
            model=stored.gemini_model or "",
            gemma_model=stored.gemma_model or "",
            gemini_model=stored.gemini_model or "",
            ollama_base_url=stored.ollama_base_url or "",
            ollama_reachable=False,
            ollama_has_model=False,
            api_key_present=False,
            key_storage=location,
            key_storage_label=_STORAGE_LABELS[location],
        )

    ollama_reachable = False
    ollama_has_model = False
    if runtime.preset_key == "gemma":
        ollama_reachable = is_ollama_running(runtime.base_url)
        if ollama_reachable:
            ollama_has_model = is_model_available(runtime.model, runtime.base_url)

    location = gemini_key_location()

    return SettingsView(
        provider=runtime.preset_key,
        model=runtime.model,
        gemma_model=stored.gemma_model or "",
        gemini_model=stored.gemini_model or "",
        ollama_base_url=stored.ollama_base_url or "",
        ollama_reachable=ollama_reachable,
        ollama_has_model=ollama_has_model,
        api_key_present=bool(runtime.api_key) and runtime.api_key != "ollama",
        key_storage=location,
        key_storage_label=_STORAGE_LABELS[location],
    )


@router.post("/forget-key", response_model=ForgetKeyResponse)
def forget_key() -> ForgetKeyResponse:
    """Delete the Gemini key from the OS credential store and the JSON fallback."""
    location = forget_gemini_key()
    os.environ.pop("GEMINI_API_KEY", None)
    brain_graph.invalidate_agent()
    return ForgetKeyResponse(cleared_from=location)
