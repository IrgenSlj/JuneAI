"""First-run setup endpoints.

These power the /setup screen in the web UI. The goal is to turn a fresh
install into a working configuration with one round-trip to a real model,
so the user never sees a broken state after completing the form.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from june_brain import graph as brain_graph
from june_brain.config import DEFAULT_RUNTIME_PRESET, resolve_runtime_config
from june_brain.config_store import (
    StoredConfig,
    is_configured,
    load_stored_config,
    save_stored_config,
)
from june_brain.models import build_chat_model
from june_brain.ollama_manager import is_model_available, is_ollama_running

from ..schemas import SetupApplyRequest, SetupApplyResponse, SetupStatus

router = APIRouter(prefix="/setup", tags=["setup"])


@router.get("/status", response_model=SetupStatus)
def get_setup_status() -> SetupStatus:
    """Report whether the active runtime is ready to serve chat requests."""
    stored = load_stored_config()
    try:
        runtime = resolve_runtime_config()
    except ValueError:
        return SetupStatus(
            is_configured=False,
            provider=stored.provider or "",
            model="",
        )

    ollama_reachable = False
    ollama_has_model = False
    if runtime.preset_key == "gemma":
        ollama_reachable = is_ollama_running(runtime.base_url)
        if ollama_reachable:
            ollama_has_model = is_model_available(runtime.model, runtime.base_url)

    api_key_present = bool(runtime.api_key) and runtime.api_key != "ollama"

    configured = is_configured(stored) and (
        (runtime.preset_key == "gemma" and ollama_reachable and ollama_has_model)
        or (runtime.preset_key == "gemini" and api_key_present)
    )

    return SetupStatus(
        is_configured=configured,
        provider=runtime.preset_key,
        model=runtime.model,
        ollama_reachable=ollama_reachable,
        ollama_has_model=ollama_has_model,
        api_key_present=api_key_present,
    )


@router.post("/apply", response_model=SetupApplyResponse)
def apply_setup(request: SetupApplyRequest) -> SetupApplyResponse:
    """Persist the user's provider choice and verify it with a live round-trip."""
    if request.provider == "gemini" and not (request.gemini_api_key or "").strip():
        raise HTTPException(
            status_code=400,
            detail="gemini_api_key is required when provider is 'gemini'.",
        )

    stored = load_stored_config()
    stored.provider = request.provider
    if request.gemma_model:
        stored.gemma_model = request.gemma_model
    if request.gemini_model:
        stored.gemini_model = request.gemini_model
    if request.gemini_api_key:
        stored.gemini_api_key = request.gemini_api_key.strip()

    save_stored_config(stored)
    _apply_to_env(stored)
    brain_graph.invalidate_agent()

    try:
        runtime = resolve_runtime_config(request.provider)
    except ValueError as exc:
        return SetupApplyResponse(
            ok=False,
            provider=request.provider,
            model="",
            verified=False,
            message=str(exc),
            hint="Double-check the Gemini API key and try again.",
        )

    if request.provider == "gemma":
        if not is_ollama_running(runtime.base_url):
            return SetupApplyResponse(
                ok=False,
                provider="gemma",
                model=runtime.model,
                verified=False,
                message="Ollama is not reachable.",
                hint="Install Ollama, then run `ollama serve` in a terminal.",
            )
        if not is_model_available(runtime.model, runtime.base_url):
            return SetupApplyResponse(
                ok=False,
                provider="gemma",
                model=runtime.model,
                verified=False,
                message=f"The '{runtime.model}' tag is not pulled yet.",
                hint=f"Run `ollama pull {runtime.model}` and try again.",
            )

    verified, message, hint = _verify_round_trip(runtime)
    if verified:
        brain_graph.reload_agent()
        if brain_graph.startup_error:
            return SetupApplyResponse(
                ok=False,
                provider=runtime.preset_key,
                model=runtime.model,
                verified=True,
                message=f"Provider verified, but June failed to reload: {brain_graph.startup_error}",
                hint="Check the runtime logs, then try applying the provider again.",
            )
    return SetupApplyResponse(
        ok=verified,
        provider=runtime.preset_key,
        model=runtime.model,
        verified=verified,
        message=message,
        hint=hint,
    )


def _apply_to_env(stored: StoredConfig) -> None:
    """Unlike the startup overlay, an explicit apply overrides existing env values."""
    for key, value in stored.to_env_patch().items():
        os.environ[key] = value
    os.environ.setdefault("MODEL_PROVIDER", stored.provider or DEFAULT_RUNTIME_PRESET)


def _verify_round_trip(runtime) -> tuple[bool, str, str]:
    """Ask the configured model to reply once. Any response at all counts as verified."""
    try:
        model = build_chat_model(runtime)
        reply = model.invoke("Reply with the single word OK.")
    except Exception as exc:  # noqa: BLE001 — the UI needs a message, not a stack trace
        return False, f"Verification failed: {exc}", _error_hint(runtime, exc)

    text = getattr(reply, "content", "") or ""
    if not str(text).strip():
        return False, "Provider returned an empty response.", "Try a different model tag."
    return True, "", ""


def _error_hint(runtime, exc: Exception) -> str:
    message = str(exc).lower()
    if runtime.preset_key == "gemini" and ("401" in message or "api key" in message or "permission" in message):
        return "The Gemini API key looks invalid. Generate a new one at https://aistudio.google.com."
    if runtime.preset_key == "gemma" and "connection" in message:
        return "Ollama isn't accepting connections. Is `ollama serve` running?"
    return "Check your network and credentials, then try again."
